# predict_renewables.py - v2, MULTI-SITE weather.
#
# Forecasts tomorrow's mean GB WIND and SOLAR generation (GW).
#
# Why multi-site: the wind fleet is in Scotland, the North Sea and offshore - not London.
# Using London weather to predict national wind was the single biggest error in v1.
# Measured on a 136-day holdout: multi-site cut wind MAE from 2.33 to 1.59 GW (-32%),
# and solar MAE from 0.46 to 0.31 GW (-32%).
#
# Self-grading: wind actuals backfill daily from Elexon FUELHH.
# Solar cannot self-grade - embedded solar is behind-the-meter and has no live feed,
# so actual_solar_gw is filled periodically from NESO settlement CSVs.
import os
import datetime as dt
import numpy as np
import pandas as pd
import requests
import lightgbm as lgb

LOG = "renewables_log.csv"

# where the wind fleet actually is
WIND_SITES = {
    "dogger":    (54.7,  2.0),    # Dogger Bank - the big offshore farms
    "n_sea_s":   (53.2,  1.7),    # southern North Sea (Hornsea, Triton Knoll)
    "scotland":  (57.0, -4.0),    # Scottish onshore
    "irish_sea": (53.8, -3.6),    # Irish Sea (Walney, Burbo)
    "moray":     (58.1, -2.8),    # Moray Firth offshore
}
# solar lives in the south - different geography entirely
SOLAR_SITES = {
    "london":   (51.51, -0.13),
    "bristol":  (51.45, -2.59),
    "norwich":  (52.63,  1.30),
    "midlands": (52.48, -1.90),
}

WIND_FEATS = ["ws_mean", "ws_max", "ws_min", "ws_std",
              "gust_mean", "gust_max", "gust_min", "gust_std",
              "t_mean", "month", "doy", "dow"]
SOLAR_FEATS = ["rad_mean", "rad_max", "rad_min", "rad_std",
               "t_mean", "month", "doy", "dow"]

COLS = ["date_made", "target_date", "pred_wind_gw", "actual_wind_gw", "wind_err",
        "pred_solar_gw", "actual_solar_gw", "status"]

today = pd.Timestamp.now(tz="Europe/London").normalize()
tom = today + pd.Timedelta(days=1)

log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=COLS)
for c in COLS:
    if c not in log.columns:
        log[c] = ""


def forecast_at(lat, lon, variables):
    """Tomorrow's daily forecast at one site."""
    u = ("https://api.open-meteo.com/v1/forecast"
         f"?latitude={lat}&longitude={lon}"
         f"&daily={','.join(variables)}&timezone=Europe%2FLondon&forecast_days=3")
    j = requests.get(u, timeout=60).json()["daily"]
    df = pd.DataFrame(j)
    df["time"] = pd.to_datetime(df["time"])
    return df.loc[df["time"].dt.date == tom.date()].iloc[0]


def spread(values, prefix):
    """mean / max / min / std across sites - the std tells the model whether the
    wind is blowing everywhere or only in one place."""
    a = np.array(values, dtype=float)
    return {
        f"{prefix}_mean": float(a.mean()),
        f"{prefix}_max": float(a.max()),
        f"{prefix}_min": float(a.min()),
        f"{prefix}_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
    }


# ---------- 1. BACKFILL wind actuals (Elexon FUELHH) ----------
try:
    no_actual = pd.to_numeric(log["actual_wind_gw"], errors="coerce").isna()
    has_pred = pd.to_numeric(log["pred_wind_gw"], errors="coerce").notna()
    past_due = pd.to_datetime(log["target_date"], errors="coerce").dt.date < today.date()
    due = log[no_actual & has_pred & past_due]

    print("rows needing a wind actual:", len(due))
    if len(due):
        lo = pd.to_datetime(due["target_date"]).min().date()
        hi = pd.to_datetime(due["target_date"]).max().date()
        u = ("https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH"
             f"?settlementDateFrom={lo}&settlementDateTo={hi}&format=json")
        data = requests.get(u, timeout=60).json().get("data", [])
        w = pd.DataFrame([x for x in data if x.get("fuelType") == "WIND"])
        if not w.empty:
            w["settlementDate"] = pd.to_datetime(w["settlementDate"]).dt.date
            w["generation"] = pd.to_numeric(w["generation"], errors="coerce")
            act = w.groupby("settlementDate")["generation"].mean() / 1000.0
            for i, r in due.iterrows():
                d = pd.to_datetime(r["target_date"]).date()
                if d in act.index:
                    a = round(float(act.loc[d]), 2)
                    log.at[i, "actual_wind_gw"] = a
                    log.at[i, "wind_err"] = round(float(r["pred_wind_gw"]) - a, 2)
                    print("  graded", d, "actual wind", a)
except Exception as e:
    print("wind backfill skipped:", type(e).__name__, e)


# ---------- 2. PREDICT tomorrow ----------
mask = log["target_date"].astype(str) == tom.date().isoformat()
already = bool(mask.any() and (log.loc[mask, "status"].astype(str) == "ok").any())
if mask.any() and not already:
    log = log[~mask].copy()   # drop a failed row so we can retry it
    print("retrying failed row for", tom.date())
if not already:
    row = {"date_made": today.date().isoformat(), "target_date": tom.date().isoformat(),
           "pred_wind_gw": "", "actual_wind_gw": "", "wind_err": "",
           "pred_solar_gw": "", "actual_solar_gw": "", "status": ""}
    try:
        cal = {"month": tom.month, "doy": tom.dayofyear, "dow": tom.dayofweek}

        # --- wind: fetch every fleet site ---
        ws, gusts, temps = [], [], []
        for name, (lat, lon) in WIND_SITES.items():
            r = forecast_at(lat, lon,
                            ["wind_speed_10m_max", "wind_gusts_10m_max", "temperature_2m_mean"])
            ws.append(float(r["wind_speed_10m_max"]))
            gusts.append(float(r["wind_gusts_10m_max"]))
            temps.append(float(r["temperature_2m_mean"]))
        wind_row = {**spread(ws, "ws"), **spread(gusts, "gust"),
                    "t_mean": float(np.mean(temps)), **cal}

        # --- solar: fetch every southern site ---
        rads, stemps = [], []
        for name, (lat, lon) in SOLAR_SITES.items():
            r = forecast_at(lat, lon, ["shortwave_radiation_sum", "temperature_2m_mean"])
            rads.append(float(r["shortwave_radiation_sum"]))
            stemps.append(float(r["temperature_2m_mean"]))
        solar_row = {**spread(rads, "rad"), "t_mean": float(np.mean(stemps)), **cal}

        wm = lgb.Booster(model_file="model_wind_multi.txt")
        sm = lgb.Booster(model_file="model_solar_multi.txt")
        pw = round(float(wm.predict(pd.DataFrame([wind_row])[WIND_FEATS])[0]), 2)
        ps = round(float(max(0.0, sm.predict(pd.DataFrame([solar_row])[SOLAR_FEATS])[0])), 2)

        row["pred_wind_gw"] = pw
        row["pred_solar_gw"] = ps
        row["status"] = "ok"
        print(f"predicted wind {pw} GW | solar {ps} GW for {tom.date()}")
        print(f"  (fleet wind speed: mean {wind_row['ws_mean']:.1f}, "
              f"spread {wind_row['ws_std']:.1f} km/h)")
    except Exception as e:
        row["status"] = "skipped: " + type(e).__name__
        print("prediction skipped:", e)
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
else:
    print("already have", tom.date())

log[COLS].to_csv(LOG, index=False)


# ---------- 3. running wind accuracy ----------
err = pd.to_numeric(log["wind_err"], errors="coerce").dropna()
if len(err):
    print("")
    print("--- live wind record:", len(err), "graded days ---")
    print("MAE:", round(err.abs().mean(), 2), "GW | bias:", round(err.mean(), 2))
    if len(err) < 20:
        print("(sample small - not yet meaningful)")
