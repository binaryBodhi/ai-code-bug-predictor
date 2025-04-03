# 🧠 AI Code Reviewer & Bug Predictor

A full-stack machine learning system that analyzes C code to predict potential bugs using transformer-based code embeddings and a custom-trained classifier.

---

## 🚀 Features

- 🪄 Predicts whether a C code snippet is **buggy** or **clean**
- 📈 Returns a **confidence score** for each prediction
- 🧠 Uses **CodeBERT** (a transformer model trained on source code) for embeddings
- 🛠 Trained on **CodeXGLUE** defect detection dataset
- ⚡ Real-time predictions via a **Flask API**
- 💻 User-friendly frontend with **Vanilla JavaScript** and **Tailwind CSS**
- 🧪 Unit-tested backend with `unittest`

---

## 🧱 Tech Stack

**Backend:** Python, Flask, scikit-learn, Transformers (CodeBERT), NumPy  
**Frontend:** HTML, CSS, Tailwind CSS, JavaScript  
**ML/Infra:** HuggingFace Transformers, joblib, CodeXGLUE  
**Tools:** Git, VS Code, Postman/Thunder Client

---

## 🧠 How It Works

1. User submits C code via the frontend
2. Backend uses CodeBERT to generate a 768-dimensional embedding
3. A trained `RandomForestClassifier` predicts if the code is buggy
4. JSON response returns:
   - `prediction`: `"buggy"` or `"clean"`
   - `confidence`: 0.0–1.0

---

## 📦 Setup Instructions

### 🐍 Backend

1. Clone this repo
2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```pip install -r requirements.txt```

5. Run the server:
```python app.py```

## 🌐 Frontend
Open index.html in browser (via Live Server or python3 -m http.server) 
Paste C code → click Predict Bug 
View results instantly

## 🧪 Testing
```python -m unittest tests.test-app```
