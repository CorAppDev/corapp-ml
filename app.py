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
    # Barrios → Localidad
    "modelia": "Fontibon",
    "capellania": "Fontibon",
    "capellanía": "Fontibon",
    "patio bonito": "Kennedy",
    "ciudad montes": "Puente aranda",
    "tibabuyes": "Suba",
    "tibabuyes universal": "Suba",
    "pinar": "Suba",
    "porvenir": "Bosa",
    "bosa piamonte": "Bosa",
    "bosa libertad": "Bosa",
    "bosa nueva": "Bosa",
    "alqueria": "Kennedy",
    "alquería": "Kennedy",
    "alqueria de la fragua": "Kennedy",
    "alquería de la fragua": "Kennedy",
    "prado veraniego": "Suba",
    "prado pinzon": "Suba",
    "prado pinzón": "Suba",
    "ciudad kennedy": "Kennedy",
    "cedro": "Engativa",
    "alamos": "Engativa",
    "álamos": "Engativa",
    "portales": "Engativa",
    "senderos del porvenir": "Bosa",
    "san agustin": "Kennedy",
    "san agustín": "Kennedy",
    "castellon de los condes": "Kennedy",
    "antiguo country": "Chapinero",
    "country": "Chapinero",
    "rosales": "Chapinero",
    "portal de rosales": "Chapinero",
    "gran estacion": "Teusaquillo",
    "gran estación": "Teusaquillo",
    "bahia solano": "Fontibon",
    "bahía solano": "Fontibon",
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

NOISE_PATTERNS = [
    r'[\w\.-]+@[\w\.-]+\.\w+',
    r'\b3\d{9}\b',
    r'\b\d{10}\b',
]

ADDRESS_START = r'(?:calle|cll|cl|carrera|cra|cr|kra|avenida|av|transversal|transv|tranv|tv|diagonal|dg|autopista|ak)'

INDICATION_KEYWORDS = [
    'apto', 'apartamento', 'apt', 'torre', 'bloque', 'interior', 'int',
    'piso', 'local', 'oficina', 'conjunto', 'edificio', 'etapa', 'unidad',
    'porteria', 'portería', 'dejar en', 'entregar en', 'llamar', 'timbrar',
    'rejas', 'reja', 'esquina', 'frente', 'cerca', 'al lado',
    'despues', 'después', 'casa azul', 'casa blanca', 'casa roja',
    'primer piso', 'segundo piso', 'tercer piso', 'si no estoy',
]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def clean_noise(text: str) -> str:
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, '', text)
    return text


def extract_labeled_fields(text: str) -> dict:
    """Extrae campos con etiquetas como Nombre:, Dirección:, Barrio:, etc."""
    fields = {}
    label_patterns = {
        'name': r'(?:nombre(?:\s+completo)?|contacto)\s*[:*]?\s*(.+)',
        'address': r'(?:direcci[oó]n(?:\s+completa)?|dir)\s*[:*]?\s*(.+)',
        'indications': r'(?:indicaciones?|detalles?|barrio|referencias?|informaci[oó]n\s+adicional)\s*[:*]?\s*(.+)',
        'locality': r'(?:localidad(?:\s+de\s+entrega)?)\s*[:*]?\s*(.+)',
        'day': r'(?:d[ií]a(?:\s+de\s+entrega)?|fecha(?:\s+de\s+entrega)?)\s*[:*]?\s*(.+)',
    }
    text_lower = text.lower()
    for field, pattern in label_patterns.items():
        match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip().split('\n')[0].strip()
            if value and value.lower() not in ('ninguna', 'ninguno', 'n/a', 'na', '-'):
                fields[field] = value
    return fields


def extract_day(text: str) -> str | None:
    norm = normalize_text(text)
    for day_key, day_value in VALID_DAYS.items():
        if re.search(r'\b' + normalize_text(day_key) + r'\b', norm):
            return day_value
    return None


def extract_locality(text: str) -> str | None:
    norm = normalize_text(text)
    sorted_locs = sorted(VALID_LOCATIONS.items(), key=lambda x: len(x[0]), reverse=True)
    for loc_key, loc_value in sorted_locs:
        loc_norm = normalize_text(loc_key)
        if re.search(r'\b' + re.escape(loc_norm) + r'\b', norm):
            return loc_value
    return None


def extract_address_line(text: str) -> tuple:
    """
    Extrae dirección e indicaciones.
    Maneja tanto texto multilínea como todo en una línea.
    """
    # Preparar líneas
    lines = []
    for part in text.replace(',', '\n').split('\n'):
        part = part.strip()
        if part:
            lines.append(part)
    
    address_line = None
    indication_parts = []

    for line in lines:
        norm_line = normalize_text(line)
        clean_line = clean_noise(line).strip()
        if not clean_line:
            continue

        # Si la línea empieza con tipo de vía
        if re.match(ADDRESS_START, norm_line, re.IGNORECASE):
            # Extraer solo la parte de la dirección
            addr_match = re.match(
                r'(' + ADDRESS_START + r'\s*[\w\s.\-#bis]+?\d+[\w\s.\-#]*\d*)',
                clean_line, re.IGNORECASE
            )
            if addr_match:
                address_line = addr_match.group(1).strip()
                rest = clean_line[len(address_line):].strip().strip(',').strip()
                if rest:
                    indication_parts.append(rest)
            else:
                address_line = clean_line
        # Si tiene keywords de indicación
        elif any(kw in norm_line for kw in INDICATION_KEYWORDS):
            clean = clean_noise(line).strip()
            if clean:
                indication_parts.append(clean)

    # Si no encontramos dirección con líneas, buscar en texto completo
    if not address_line:
        match = re.search(
            r'(' + ADDRESS_START + r'\s*[\w\s.\-#]+?\d+[\w\s.\-#]*\d*)',
            text, re.IGNORECASE
        )
        if match:
            address_line = match.group(1).strip()

    # Limpiar indicaciones
    seen = set()
    clean_indications = []
    for part in indication_parts:
        norm_part = normalize_text(part)
        if norm_part not in seen and len(norm_part) > 2:
            seen.add(norm_part)
            clean_indications.append(part)

    indications = ', '.join(clean_indications[:3])
    return address_line, indications


def extract_name(text: str) -> str | None:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    day_keywords = '|'.join(normalize_text(d) for d in VALID_DAYS.keys())
    loc_keywords = '|'.join(normalize_text(l) for l in VALID_LOCATIONS.keys())

    for line in lines[:4]:
        norm = normalize_text(line)
        clean = clean_noise(line).strip()
        if (re.match(r'^[a-záéíóúñ\s]+$', norm) and
                len(line.split()) >= 2 and
                not re.search(r'\b(' + day_keywords + r')\b', norm) and
                not re.search(r'\b(' + loc_keywords + r')\b', norm) and
                not re.search(r'@|\d', line)):
            return clean
    return None


def extract_delivery_info(text, valid_locations=None, valid_days=None, valid_times=None):
    if not text or not text.strip():
        return {"error": True, "errorMessage": "Mensaje vacío", "info": None}

    clean_text = clean_noise(text)
    labeled = extract_labeled_fields(clean_text)

    name = labeled.get('name') or extract_name(clean_text)

    address = None
    indications = labeled.get('indications', '')

    if labeled.get('address'):
        address = labeled['address']
        if not indications:
            _, ind = extract_address_line(clean_text)
            indications = ind
    else:
        address, ind = extract_address_line(clean_text)
        if ind and not indications:
            indications = ind

    locality = None
    if labeled.get('locality'):
        locality = extract_locality(labeled['locality'])
    if not locality:
        locality = extract_locality(clean_text)

    day = None
    if labeled.get('day'):
        day = extract_day(labeled['day'])
    if not day:
        day = extract_day(clean_text)

    # Errores específicos para mejor UX
    missing = []
    if not address:
        missing.append("dirección")
    if not locality:
        missing.append("localidad")
    if not day:
        missing.append("día de entrega")

    if missing:
        return {
            "error": True,
            "errorMessage": "No se pudo extraer la información de entrega",
            "info": None,
        }

    return {
        "error": False,
        "errorMessage": None,
        "info": {
            "address": address,
            "indications": indications or "",
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

# v2.1.0 — extractor mejorado con soporte para etiquetas, barrios y fechas largas
if __name__ == "__main__":
    app.run(debug=True, port=5000)