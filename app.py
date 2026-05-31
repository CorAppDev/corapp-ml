from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo
model = joblib.load('model/intent_classifier.pkl')

CONFIDENCE_THRESHOLD = 0.5

def get_confidence(text):
    """Obtiene la intención y confianza del modelo"""
    text = text.lower().strip()
    
    # Predecir intención
    intent = model.predict([text])[0]
    
    # Obtener distancias de decisión (SVM no da probabilidades directas)
    decision = model.decision_function([text])[0]
    
    # Normalizar a confianza entre 0 y 1
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
    
    # Si la confianza es baja, indicar fallback a OpenAI
    use_fallback = confidence < CONFIDENCE_THRESHOLD
    
    return jsonify({
        'message': message,
        'intent': intent,
        'confidence': round(confidence, 4),
        'use_fallback': use_fallback
    })

@app.route('/health', methods=['GET'])
def health():
    return