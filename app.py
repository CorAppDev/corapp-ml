from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

app = Flask(__name__)
model = None

def train_model():
    with open('data/training.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentences = []
    labels = []
    for intent in data['intents']:
        for example in intent['examples']:
            sentences.append(example.lower())
            labels.append(intent['tag'])
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)),
        ('clf', LinearSVC(max_iter=1000))
    ])
    pipeline.fit(sentences, labels)
    os.makedirs('model', exist_ok=True)
    joblib.dump(pipeline, 'model/intent_classifier.pkl')
    return pipeline

def load_model():
    global model
    if model is None:
        if not os.path.exists('model/intent_classifier.pkl'):
            model = train_model()
        else:
            model = joblib.load('model/intent_classifier.pkl')
    return model

def get_confidence(text):
    clf = load_model()
    text = text.lower().strip()
    intent = clf.predict([text])[0]
    decision = clf.decision_function([text])[0]
    exp_scores = np.exp(decision - np.max(decision))
    confidence = float(np.max(exp_scores) / exp_scores.sum())
    return intent, confidence

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Se requiere el campo message'}), 400
    message = data['message']
    intent, confidence = get_confidence(message)
    return jsonify({
        'message': message,
        'intent': intent,
        'confidence': round(confidence, 4),
        'use_fallback': confidence < 0.2
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)