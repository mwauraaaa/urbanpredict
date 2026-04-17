#!/usr/bin/env python
"""
Complete Pipeline Runner - UrbanPredict KE
Runs the full ML pipeline: Data → Features → Model → Report

Usage:
    python run_full_pipeline.py
"""

import json
from pathlib import Path
from datetime import datetime


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def print_step(step_num, text):
    """Print step header."""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {text}")
    print(f"{'='*70}\n")


def run_pipeline():
    """Run the complete UrbanPredict pipeline."""
    
    start_time = datetime.now()
    
    print_header("URBANPREDICT KE - COMPLETE ML PIPELINE")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure directories exist
    for dir_path in ["data/raw/properties", "data/processed", "models", "logs"]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Data Collection
    print_step(1, "DATA COLLECTION")
    
    # Check if data exists
    data_file = Path("data/raw/properties/nairobi_properties_latest.json")
    
    if data_file.exists():
        with open(data_file) as f:
            properties = json.load(f)
        print(f"✅ Using existing data: {len(properties)} properties")
    else:
        print("📥 Running data scraper...")
        from src.data_pipeline.real_scraper import RealPropertyScraper
        
        scraper = RealPropertyScraper()
        scraper.scrape_jiji_ke()
        scraper.save_listings()
        
        with open(data_file) as f:
            properties = json.load(f)
        print(f"✅ Scraped {len(properties)} properties")
    
    # Data summary
    prices = [p["price"] for p in properties]
    neighborhoods = set(p["neighborhood"] for p in properties)
    
    print(f"\n📊 Data Summary:")
    print(f"   • Properties: {len(properties)}")
    print(f"   • Neighborhoods: {len(neighborhoods)}")
    print(f"   • Price Range: KES {min(prices):,} - KES {max(prices):,}")
    print(f"   • Mean Price: KES {sum(prices)/len(prices):,.0f}")
    
    # STEP 2: Feature Engineering
    print_step(2, "FEATURE ENGINEERING")
    
    print("🔧 Running feature engineering...")
    from src.features.engineer_features import engineer_all_features
    
    input_file = "data/raw/properties/nairobi_properties_latest.json"
    output_file = "data/processed/nairobi_properties_engineered.json"
    
    engineered_data = engineer_all_features(input_file, output_file)
    
    # Feature summary
    print(f"\n📊 Feature Summary:")
    print(f"   • Total Features: {len(engineered_data[0].keys())}")
    
    # Count feature types
    location_features = sum(1 for k in engineered_data[0].keys() if "distance" in k.lower())
    amenity_features = sum(1 for k in engineered_data[0].keys() if "km" in k.lower() or "2km" in k.lower())
    score_features = sum(1 for k in engineered_data[0].keys() if "score" in k.lower())
    
    print(f"   • Location Features: {location_features}")
    print(f"   • Amenity Features: {amenity_features}")
    print(f"   • Derived Scores: {score_features}")
    
    # STEP 3: Model Training
    print_step(3, "MODEL TRAINING")
    
    print("🤖 Training ML model...")
    from src.models.train_numpy import main as train_model
    
    model, metrics = train_model()
    
    # Model performance summary
    test_r2 = metrics["test"]["r2"]
    test_rmse = metrics["test"]["rmse"]
    test_mae = metrics["test"]["mae"]
    
    print(f"\n📊 Model Performance:")
    print(f"   • Test R²: {test_r2:.4f} {'✅ EXCELLENT' if test_r2 >= 0.75 else '✅ GOOD' if test_r2 >= 0.60 else '⚠️  MODERATE'}")
    print(f"   • Test RMSE: KES {test_rmse:,.0f}")
    print(f"   • Test MAE: KES {test_mae:,.0f}")
    
    # STEP 4: Generate Report
    print_step(4, "GENERATING FINAL REPORT")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    report = f"""
{'='*70}
URBANPREDICT KE - PIPELINE EXECUTION REPORT
{'='*70}

EXECUTION SUMMARY
-----------------
Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}
Ended:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration.total_seconds():.1f} seconds ({duration.total_seconds()/60:.1f} minutes)

DATA COLLECTION
---------------
✅ Total Properties: {len(properties)}
✅ Neighborhoods: {len(neighborhoods)}
✅ Price Range: KES {min(prices):,} - KES {max(prices):,}
✅ Mean Price: KES {sum(prices)/len(prices):,.0f}
✅ Median Price: KES {sorted(prices)[len(prices)//2]:,}

FEATURE ENGINEERING
-------------------
✅ Total Features: {len(engineered_data[0].keys())}
✅ Location Features: {location_features}
✅ Amenity Features: {amenity_features}
✅ Derived Scores: {score_features}

MODEL PERFORMANCE
-----------------
✅ Algorithm: Random Forest Regressor
✅ Test R²: {test_r2:.4f}
✅ Test RMSE: KES {test_rmse:,.0f}
✅ Test MAE: KES {test_mae:,.0f}

OUTPUT FILES
------------
✅ Raw Data: data/raw/properties/nairobi_properties_latest.json
✅ Processed Data: data/processed/nairobi_properties_engineered.json
✅ Trained Model: models/price_prediction_model.json

NEXT STEPS
----------
1. Start API Server:
   uvicorn api.main:app --reload

2. Launch Dashboard:
   streamlit run dashboard/app.py

3. Test API Endpoints:
   http://localhost:8000/docs

4. View Dashboard:
   http://localhost:8501

{'='*70}
✅ PIPELINE COMPLETED SUCCESSFULLY!
{'='*70}
"""
    
    print(report)
    
    # Save report
    report_path = "pipeline_execution_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"\n💾 Report saved to {report_path}")
    
    # Final summary
    print_header("PIPELINE COMPLETE - SUMMARY")
    
    print(f"""
✅ DATA: {len(properties)} properties from {len(neighborhoods)} neighborhoods
✅ FEATURES: {len(engineered_data[0].keys())} features engineered
✅ MODEL: Trained with R² = {test_r2:.4f}
✅ FILES: All outputs saved to data/ and models/

🚀 YOUR PROPERTY PRICE PREDICTION SYSTEM IS READY!

   Start API:  uvicorn api.main:app --reload
   Dashboard:  streamlit run dashboard/app.py
""")
    
    return {
        "properties": len(properties),
        "features": len(engineered_data[0].keys()),
        "r2": test_r2,
        "rmse": test_rmse,
        "duration": duration.total_seconds(),
    }


if __name__ == "__main__":
    try:
        results = run_pipeline()
        print("\n✅ Pipeline completed successfully!")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
