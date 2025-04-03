from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import RobertaTokenizer, RobertaModel
import torch
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500"])

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple GPU via Metal
elif torch.cuda.is_available():
    device = torch.device("cuda")  # For other systems (not your Mac)
else:
    device = torch.device("cpu")

# Load CodeBERT model
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
model.eval()

# Load your trained bug classifier
clf = joblib.load("./model/bug-classifier.pkl")

@app.route("/predict_bug", methods=["POST"])
def predict_bug():
    data = request.json
    code = data.get("code")

    if not code:
        return jsonify({"error": "No code provided"}), 400

    try:
        # Tokenize and move to device
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Get embedding
        embedding = outputs.last_hidden_state.mean(1).squeeze().cpu().numpy()

        # Predict
        prediction = clf.predict([embedding])[0]
        proba = clf.predict_proba([embedding])[0]

        return jsonify({
            "prediction": "buggy" if prediction == 1 else "clean",
            "confidence": float(np.max(proba))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
