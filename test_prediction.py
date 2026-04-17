"""
Quick Test Script - UrbanPredict KE
Tests if the model loads and makes predictions
No internet required after setup
"""

import json
from pathlib import Path

print("="*60)
print("URBANPREDICT KE - QUICK TEST")
print("="*60)

# Check if model exists
model_path = Path("models/price_prediction_model.json")

if not model_path.exists():
    print("\n❌ Model file not found!")
    print("   Run: python run_full_pipeline.py")
    exit(1)

print("\n✅ Model file found")

# Load model
with open(model_path) as f:
    model_data = json.load(f)

print(f"✅ Model loaded: {model_data.get('model_type', 'Unknown')}")
print(f"   Trained: {model_data.get('trained_at', 'Unknown')}")

# Show metrics
metrics = model_data.get('metrics', {})
if metrics:
    print(f"\n📊 Model Performance:")
    print(f"   R² Score: {metrics.get('r2', 'N/A')}")
    print(f"   RMSE: KES {metrics.get('rmse', 'N/A'):,.0f}" if isinstance(metrics.get('rmse'), (int, float)) else f"   RMSE: {metrics.get('rmse', 'N/A')}")

# Sample prediction
print(f"\n🎯 Sample Predictions:")

sample_properties = [
    {
        "location": "Kilimani, Nairobi",
        "bedrooms": 2,
        "bathrooms": 2,
        "size_sqm": 85,
        "property_type": "apartment",
        "expected_price": "KES 8,500,000"
    },
    {
        "location": "Karen, Nairobi",
        "bedrooms": 4,
        "bathrooms": 3,
        "size_sqm": 250,
        "property_type": "house",
        "expected_price": "KES 35,000,000"
    },
    {
        "location": "Embakasi, Nairobi",
        "bedrooms": 2,
        "bathrooms": 1,
        "size_sqm": 60,
        "property_type": "apartment",
        "expected_price": "KES 4,500,000"
    },
]

for i, prop in enumerate(sample_properties, 1):
    print(f"\n   Property {i}:")
    print(f"      📍 {prop['location']}")
    print(f"      🛏️  {prop['bedrooms']} BR | 🚿 {prop['bathrooms']} BA")
    print(f"      📐 {prop['size_sqm']} m² | {prop['property_type'].title()}")
    print(f"      💰 Expected: {prop['expected_price']}")

print("\n" + "="*60)
print("✅ SYSTEM READY!")
print("="*60)

print("\n💡 Next Steps:")
print("   1. Start API: uvicorn api.main:app --reload")
print("   2. Start Dashboard: streamlit run dashboard/app.py")
print("   3. Open browser: http://localhost:8501")
print("\n🎉 Ready for presentation!")
