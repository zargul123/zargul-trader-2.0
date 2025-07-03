
from tensorflow.keras.models import load_model
from scripts.core.analysis_engine import AttentionLayer  # Import custom layer

try:
    model = load_model('trained_models/SOL-USD_model.keras', 
                      custom_objects={'AttentionLayer': AttentionLayer})
    print("✅ SOL Model Loaded Successfully!")
    model.summary()
except Exception as e:
    print(f"❌ Error: {str(e)}")
