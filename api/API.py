
from fastapi import FastAPI
from pydantic import BaseModel

from src.Run_MDmodel import run_demand_forecast

app = FastAPI(
    title="Forecasting Engine API",
    version="1.0.0"
)


class Forecast_Request(BaseModel):
    n_ticks: int = 7
    tick_size: int = 1
    start_date: str = "2011-09-08"

@app.get("/")
def root():
    return {"status": "Engine is running perfectly."}


@app.post('/generate_forecast')
async def generate_forecast(request : Forecast_Request):
    try:
        print(f"🚨 INCOMING REQUEST: Ticks: {request.n_ticks}, Size: {request.tick_size}, Date: {request.start_date}")
        forecast_df = run_demand_forecast(n_ticks = request.n_ticks,
                                          start_date = request.start_date,
                                          tick_size = request.tick_size)
        flat_df = forecast_df.reset_index()

        if 'forecast_date' in flat_df.columns:
            flat_df['forecast_date'] = flat_df['forecast_date'].astype(str)

        payload = flat_df.to_dict(orient = 'records')


        return {'status' : 'complete' , 'data' :payload}

    except Exception as e:
        return {"status": "error", "message": str(e)}





