"""
UrbanPredict KE - Main Pipeline Runner

This script runs the complete data pipeline:
1. Scrape property listings
2. Download geospatial data
3. Engineer features
4. Train model
5. Generate report

Usage:
    python main.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import logger
from src.utils.config import PROPERTIES_DIR, PROCESSED_DATA_DIR


def run_pipeline():
    """Run the complete UrbanPredict pipeline."""
    logger.info("=" * 60)
    logger.info("URBANPREDICT KE - PROPERTY PRICE PREDICTION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Scrape property data
    logger.info("\n[STEP 1/5] Scraping property listings...")
    try:
        from src.data_pipeline.scraper import PropertyScraper
        
        scraper = PropertyScraper()
        scraper.run(save=True)
        
        logger.info("✅ Property scraping complete")
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        return

    # Step 2: Download geospatial data
    logger.info("\n[STEP 2/5] Downloading geospatial data...")
    try:
        from src.data_pipeline.geodata import GeoDataDownloader
        
        downloader = GeoDataDownloader()
        downloader.run()
        
        logger.info("✅ Geospatial data download complete")
    except Exception as e:
        logger.error(f"❌ Geodata download failed: {e}")
        return

    # Step 3: Preprocess and engineer features
    logger.info("\n[STEP 3/5] Preprocessing and feature engineering...")
    try:
        from src.data_pipeline.preprocessing import preprocess_pipeline
        from src.features.location import engineer_location_features
        
        # Find latest property data
        property_files = list(PROPERTIES_DIR.glob("*.csv"))
        if not property_files:
            logger.error("❌ No property data found")
            return
        
        input_file = property_files[0]
        logger.info(f"Using property data: {input_file}")
        
        # Preprocess
        df = preprocess_pipeline(str(input_file))
        
        # Engineer features
        df = engineer_location_features(df)
        
        # Save processed data
        output_file = PROCESSED_DATA_DIR / "nairobi_properties_full.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Feature engineering complete ({len(df.columns)} features)")
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Train model
    logger.info("\n[STEP 4/5] Training prediction model...")
    try:
        from src.models.train import train_model
        
        model, metrics = train_model(
            df,
            target_col="price",
            model_type="xgboost",
            tune_params=True,
        )
        
        logger.info(f"✅ Model training complete")
        logger.info(f"   Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"   Test RMSE: KES {metrics['test_rmse']:,.0f}")
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Generate summary report
    logger.info("\n[STEP 5/5] Generating summary report...")
    try:
        import pandas as pd
        
        report = f"""
{'='*60}
URBANPREDICT KE - PIPELINE SUMMARY REPORT
{'='*60}

DATA COLLECTION:
  - Property listings scraped: {len(df)}
  - Neighborhoods covered: {df['neighborhood'].nunique()}
  - Price range: KES {df['price'].min():,} - KES {df['price'].max():,}

FEATURE ENGINEERING:
  - Total features: {len(df.columns)}
  - Location features: distance_to_cbd, amenity distances
  - Demographic features: population_density, night_lights

MODEL PERFORMANCE:
  - Algorithm: XGBoost Regressor
  - Test R²: {metrics['test_r2']:.4f}
  - Test RMSE: KES {metrics['test_rmse']:,.0f}
  - Test MAE: KES {metrics['test_mae']:,.0f}

OUTPUT FILES:
  - Raw data: data/raw/properties/
  - Processed data: data/processed/
  - Model: models/xgboost_price_model.joblib

NEXT STEPS:
  1. Start API: uvicorn api.main:app --reload
  2. Launch Dashboard: streamlit run dashboard/app.py
  3. Test predictions at: http://localhost:8000/docs

{'='*60}
Pipeline completed successfully!
{'='*60}
"""
        
        logger.info(report)
        
        # Save report
        report_path = project_root / "pipeline_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")

    logger.info("\n✅ PIPELINE COMPLETE!")


if __name__ == "__main__":
    run_pipeline()
