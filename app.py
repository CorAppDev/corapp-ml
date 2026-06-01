from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists('model/intent_classifier.pkl'):
            import subprocess
            subprocess.run(['python', 'train.py'], check=True)
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
    use_fallback = confidence < 0.5
    return jsonify({
        'message': message,
        'intent': intent,
        'confidence': round(confidence, 4),
        'use_fallback': use_fallback
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)