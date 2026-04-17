import asyncio
from api.main import predict_price, _load_phase2_stack
from api.schemas import PredictionRequest

req = PredictionRequest(
  latitude=51.5,
  longitude=-0.1,
  property_type="detached",
  is_new="N",
  duration="F",
  year=2026,
  month=1,
  town_city="LONDON",
  district="CITY OF LONDON",
  county="GREATER LONDON",
  market_segment="mid",
  school_score=3.5,
  nearest_station_dist=1.0,
  interest_rate=4.5,
  d_med=450000.0,
  d_cnt=1000
)

async def run():
    _load_phase2_stack()
    try:
        from api.main import predict_price
        if asyncio.iscoroutinefunction(predict_price):
            res = await predict_price(req)
        else:
            res = predict_price(req)
        print(res)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run())
