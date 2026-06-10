from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import json
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

app = Flask(__name__)
model = None

# ============================================================
# ML — CLASIFICADOR DE INTENCIONES
# ============================================================

def train_model():
    with open('data/training.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentences, labels = [], []
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

# ============================================================
# EXTRACTOR DE DATOS DE ENTREGA
# ============================================================

VALID_LOCATIONS = {
    "bosa": "Bosa",
    "kennedy": "Kennedy",
    "puente aranda": "Puente aranda",
    "tunjuelito": "Tunjuelito",
    "antonio narino": "Antonio narino",
    "antonio nariño": "Antonio narino",
    "teusaquillo": "Teusaquillo",
    "barrios unidos": "Barrios unidos",
    "martires": "Martirez",
    "mártires": "Martirez",
    "martirez": "Martirez",
    "fontibon": "Fontibon",
    "fontibón": "Fontibon",
    "engativa": "Engativa",
    "engativá": "Engativa",
    "chapinero": "Chapinero",
    "usaquen": "Usaquen",
    "usaquén": "Usaquen",
    "soacha": "Soacha",
    "candelaria": "Candelaria",
    "suba": "Suba",
    "modelia": "Fontibon",
    "capellania": "Fontibon",
    "capellanía": "Fontibon",
    "patio bonito": "Kennedy",
    "ciudad montes": "Puente aranda",
    "tibabuyes": "Suba",
    "pinar": "Suba",
    "porvenir": "Bosa",
}

VALID_DAYS = {
    "lunes": "Lunes",
    "martes": "Martes",
    "miercoles": "Miercoles",
    "miércoles": "Miercoles",
    "jueves": "Jueves",
    "viernes": "Viernes",
    "sabado": "Sabado",
    "sábado": "Sabado",
}

ADDRESS_START = r'(?:calle|cll|cl|carrera|cra|cr|kra|avenida|av|transversal|transv|tranv|tv|diagonal|dg|autopista)'


def normalize_text(text):
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def extract_day(text):
    norm = normalize_text(text)
    for day_key, day_value in VALID_DAYS.items():
        if re.search(r'\b' + normalize_text(day_key) + r'\b', norm):
            return day_value
    return None


def extract_locality(text):
    norm = normalize_text(text)
    sorted_locs = sorted(VALID_LOCATIONS.items(), key=lambda x: len(x[0]), reverse=True)
    for loc_key, loc_value in sorted_locs:
        loc_norm = normalize_text(loc_key)
        if re.search(r'\b' + re.escape(loc_norm) + r'\b', norm):
            return loc_value
    return None


def extract_address_and_indications(text):
    lines = [l.strip() for l in text.replace(',', '\n').split('\n') if l.strip()]
    address_line = None
    indication_parts = []

    for line in lines:
        norm_line = normalize_text(line)
        if re.match(ADDRESS_START, norm_line, re.IGNORECASE):
            addr_match = re.match(
                r'(' + ADDRESS_START + r'\s*[\w\s.\-#bis]+?\d+[\w\s.\-#]*\d*)',
                line, re.IGNORECASE
            )
            if addr_match:
                address_line = addr_match.group(1).strip()
                rest = line[len(address_line):].strip().strip(',').strip()
                if rest:
                    indication_parts.append(rest)
            else:
                address_line = line
        elif re.search(r'\b(apto|apartamento|torre|bloque|interior|int|piso|local|conjunto|edificio|porteria|portería)\b', norm_line):
            indication_parts.append(line)
        elif re.search(r'\b(despues|después|dejar|llamar|tocar|frente|cerca|esquina|rejas|color)\b', norm_line):
            indication_parts.append(line)

    seen = set()
    clean_indications = []
    for part in indication_parts:
        norm_part = normalize_text(part)
        if norm_part not in seen and len(norm_part) > 2:
            seen.add(norm_part)
            clean_indications.append(part)

    indications = ', '.join(clean_indications[:3])
    return address_line, indications


def extract_name(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    day_keywords = '|'.join(normalize_text(d) for d in VALID_DAYS.keys())
    loc_keywords = '|'.join(normalize_text(l) for l in VALID_LOCATIONS.keys())

    for line in lines[:3]:
        norm = normalize_text(line)
        if (re.match(r'^[a-záéíóúñ\s]+$', norm) and
                len(line.split()) >= 2 and
                not re.search(r'\b(' + day_keywords + r')\b', norm) and
                not re.search(r'\b(' + loc_keywords + r')\b', norm)):
            return line.strip()
    return None


def extract_delivery_info(text, valid_locations=None, valid_days=None, valid_times=None):
    if not text or not text.strip():
        return {"error": True, "errorMessage": "Mensaje vacío", "info": None}

    address, indications = extract_address_and_indications(text)
    locality = extract_locality(text)
    day = extract_day(text)
    name = extract_name(text)

    if not address or not locality or not day:
        return {"error": True, "errorMessage": "No se pudo extraer la información de entrega", "info": None}

    return {
        "error": False,
        "errorMessage": None,
        "info": {
            "address": address,
            "indications": indications,
            "locationDelivery": locality,
            "dayDelivery": day,
            "timeDelivery": "morning",
            "latitude": 4.7110,
            "longitude": -74.0721,
            "userName": name,
        }
    }

# ============================================================
# ENDPOINTS
# ============================================================

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

@app.route('/extract-delivery', methods=['POST'])
def extract_delivery():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': True, 'errorMessage': 'Se requiere el campo text', 'info': None}), 400
    
    result = extract_delivery_info(
        data['text'],
        data.get('validLocations'),
        data.get('validDays'),
        data.get('validTimes'),
    )
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)