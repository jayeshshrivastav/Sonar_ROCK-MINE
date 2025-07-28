import numpy as np
import pickle
import sys
import json

with open("model/sonar_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_from_input(input_string):
    try:
        input_data = np.array([float(i) for i in input_string.split(",")])
        if input_data.shape[0] != 60:
            raise ValueError("Expected 60 features.")
        input_scaled = scaler.transform([input_data])
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        result = {
            "prediction": "Mine" if pred == "M" else "Rock",
            "confidence": round(float(max(proba)), 4)
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        raw_input = sys.argv[1]
        print(predict_from_input(raw_input))
    else:
        print("Usage: python predict.py \"<comma,separated,60,features>\"")
