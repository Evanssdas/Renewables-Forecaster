# predict_renewables.py - forecast tomorrow's mean WIND and SOLAR generation (GW).
# Wind actuals auto-backfill from Elexon FUELHH (live). Solar actuals need NESO data
# (embedded, not on the live feed), so solar actual is left blank for manual/periodic fill.
import os, datetime as dt
import numpy as np, pandas as pd, requests
import lightgbm as lgb

LAT, LON = 51.51, -0.13
LOG = "renewables_log.csv"
WIND_FEATS  = ["wind_max","gust_max","t_mean","month","doy","dow"]
SOLAR_FEATS = ["solar_rad","t_mean","month","doy","dow"]
COLS = ["date_made","target_date","pred_wind_gw","actual_wind_gw","wind_err",
        "pred_solar_gw","actual_solar_gw","status"]

today = pd.Timestamp.now(tz="Europe/London").normalize()
tom   = today + pd.Timedelta(days=1)

log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=COLS)

# ---------- 1. backfill WIND actuals (Elexon FUELHH) ----------
try:
    need = log[(log["actual_wind_gw"].isna() | (log["actual_wind_gw"].astype(str)=="")) &
               (pd.to_datetime(log["target_date"]).dt.date < today.date())]
    if len(need):
        lo=pd.to_datetime(need["target_date"]).min().date(); hi=pd.to_datetime(need["target_date"]).max().date()
        u=f"https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH?settlementDateFrom={lo}&settlementDateTo={hi}&format=json"
        data=requests.get(u,timeout=60).json().get("data",[])
        w=pd.DataFrame([x for x in data if x.get("fuelType")=="WIND"])
        if not w.empty:
            w["settlementDate"]=pd.to_datetime(w["settlementDate"]).dt.date
            w["generation"]=pd.to_numeric(w["generation"],errors="coerce")
            act=w.groupby("settlementDate")["generation"].mean()/1000.0
            for i,row in need.iterrows():
                d=pd.to_datetime(row["target_date"]).date()
                if d in act.index:
                    a=round(float(act.loc[d]),2)
                    log.at[i,"actual_wind_gw"]=a
                    log.at[i,"wind_err"]=round(float(row["pred_wind_gw"])-a,2)
        print("wind actuals backfilled where available")
except Exception as e:
    print("wind backfill skipped:", type(e).__name__)

# ---------- 2. predict tomorrow ----------
if not (log["target_date"].astype(str)==tom.date().isoformat()).any():
    try:
        wu=("https://api.open-meteo.com/v1/forecast"
            f"?latitude={LAT}&longitude={LON}"
            "&daily=wind_speed_10m_max,wind_gusts_10m_max,shortwave_radiation_sum,temperature_2m_mean"
            "&timezone=Europe%2FLondon&forecast_days=3")
        wdf=pd.DataFrame(requests.get(wu,timeout=60).json()["daily"]); wdf["time"]=pd.to_datetime(wdf["time"])
        r=wdf.loc[wdf["time"].dt.date==tom.date()].iloc[0]
        cal={"month":tom.month,"doy":tom.dayofyear,"dow":tom.dayofweek}
        wind_row={"wind_max":float(r["wind_speed_10m_max"]),"gust_max":float(r["wind_gusts_10m_max"]),
                  "t_mean":float(r["temperature_2m_mean"]), **cal}
        solar_row={"solar_rad":float(r["shortwave_radiation_sum"]),"t_mean":float(r["temperature_2m_mean"]), **cal}
        wm=lgb.Booster(model_file="model_wind.txt"); sm=lgb.Booster(model_file="model_solar.txt")
        pw=round(float(wm.predict(pd.DataFrame([wind_row])[WIND_FEATS])[0]),2)
        ps=round(float(max(0.0, sm.predict(pd.DataFrame([solar_row])[SOLAR_FEATS])[0])),2)  # solar >= 0
        new={"date_made":today.date().isoformat(),"target_date":tom.date().isoformat(),
             "pred_wind_gw":pw,"actual_wind_gw":"","wind_err":"",
             "pred_solar_gw":ps,"actual_solar_gw":"","status":"ok"}
    except Exception as e:
        new={"date_made":today.date().isoformat(),"target_date":tom.date().isoformat(),
             "pred_wind_gw":"","actual_wind_gw":"","wind_err":"",
             "pred_solar_gw":"","actual_solar_gw":"","status":f"skipped: {type(e).__name__}"}
    log=pd.concat([log, pd.DataFrame([new])], ignore_index=True)
    print("logged:", new)
else:
    print("already have", tom.date())

log[COLS].to_csv(LOG, index=False)
