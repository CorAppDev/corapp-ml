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
# EXTRACTOR DE DATOS DE ENTREGA — v3.0
# ============================================================

VALID_LOCATIONS = {
    # Localidades oficiales
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
    # Barrios y sectores → Localidad
    "modelia": "Fontibon",
    "capellania": "Fontibon",
    "capellanía": "Fontibon",
    "capellania la camelia": "Fontibon",
    "patio bonito": "Kennedy",
    "ciudad montes": "Puente aranda",
    "tibabuyes": "Suba",
    "tibabuyes universal": "Suba",
    "pinar": "Suba",
    "pinar de suba": "Suba",
    "porvenir": "Bosa",
    "bosa piamonte": "Bosa",
    "bosa libertad": "Bosa",
    "bosa nueva": "Bosa",
    "bosa el porvenir": "Bosa",
    "senderos del porvenir": "Bosa",
    "alqueria": "Kennedy",
    "alquería": "Kennedy",
    "alqueria de la fragua": "Kennedy",
    "alquería de la fragua": "Kennedy",
    "prado veraniego": "Suba",
    "prado pinzon": "Suba",
    "prado pinzón": "Suba",
    "ciudad kennedy": "Kennedy",
    "cedro": "Engativa",
    "el cedro": "Engativa",
    "alamos": "Engativa",
    "álamos": "Engativa",
    "portales": "Engativa",
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
    "villa del prado": "Suba",
    "alhambra": "Suba",
    "cedritos": "Usaquen",
    "santa barbara": "Usaquen",
    "santa bárbara": "Usaquen",
    "mazuren": "Suba",
    "mazurén": "Suba",
    "niza": "Suba",
    "ciudad salitre": "Fontibon",
    "salitre": "Fontibon",
    "palermo": "Teusaquillo",
    "la soledad": "Teusaquillo",
    "gaitan": "Barrios unidos",
    "gaitán": "Barrios unidos",
    "alcazares": "Barrios unidos",
    "alcázares": "Barrios unidos",
    "siete de agosto": "Barrios unidos",
    "la floresta": "Engativa",
    "floresta": "Engativa",
    "quirigua": "Engativa",
    "quiriguá": "Engativa",
    "minuto de dios": "Engativa",
    "bachue": "Engativa",
    "bachuë": "Engativa",
    "tintal": "Kennedy",
    "americas": "Kennedy",
    "américas": "Kennedy",
    "timiza": "Kennedy",
    "bello horizonte": "Suba",
    "verbenal": "Usaquen",
    "toberin": "Usaquen",
    "toberín": "Usaquen",
    "country norte": "Usaquen",
    "santa cecilia": "Engativa",
    "villa luz": "Engativa",
    "la giralda": "Fontibon",
    "la esperanza": "Kennedy",
    "muzú": "Puente aranda",
    "muzu": "Puente aranda",
    "cundinamarca": None,  # No es Bogotá
    "zipaquira": None,
    "zipaquirá": None,
    "chia": None,
    "chía": None,
    "sopo": None,
    "sopó": None,
    "cajica": None,
    "cajicá": None,
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

# Días inválidos que los usuarios envían frecuentemente
INVALID_DAYS = {
    "domingo": "Domingo no está disponible para entregas. Los días disponibles son: Lunes, Martes, Miércoles, Jueves, Viernes y Sábado",
    "manana": "Por favor envía el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "mañana": "Por favor envía el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "hoy": "Por favor envía el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "pasado manana": "Por favor envía el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "pasado mañana": "Por favor envía el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
}

NOISE_PATTERNS = [
    r'[\w\.-]+@[\w\.-]+\.\w+',  # emails
    r'\b3\d{9}\b',               # teléfonos colombianos
    r'\b\d{10}\b',               # otros teléfonos
    r'\bpago\s+(?:contra\s+entrega|en\s+efectivo|nequi|bre-?b)\b',  # métodos de pago
    r'\btelefono\b[:*]?\s*\d*',  # etiqueta teléfono
    r'\bcorreo\b[:*]?\s*\S*',    # etiqueta correo
]

ADDRESS_START = r'(?:calle|cll|cl|carrera|cra|cr|kra|avenida|av|transversal|transv|tranv|tv|diagonal|dg|autopista|ak)'

INDICATION_KEYWORDS = [
    'apto', 'apartamento', 'apt', 'torre', 'bloque', 'interior', 'int',
    'piso', 'local', 'oficina', 'conjunto', 'edificio', 'etapa', 'unidad',
    'porteria', 'portería', 'dejar en', 'entregar en', 'llamar', 'timbrar',
    'rejas', 'reja', 'esquina', 'frente', 'cerca', 'al lado',
    'despues', 'después', 'casa azul', 'casa blanca', 'casa roja', 'casa verde',
    'primer piso', 'segundo piso', 'tercer piso', 'si no estoy',
    'peluqueria', 'peluquería', 'tienda', 'drogueria', 'droguería',
    'supermercado', 'parque', 'iglesia', 'colegio', 'hospital',
]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def clean_noise(text: str) -> str:
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text


def extract_labeled_fields(text: str) -> dict:
    """
    Extrae campos cuando el usuario usa etiquetas.
    Soporta: Nombre:, Dirección:, Barrio:, Localidad:, Día:, Día de entrega:
    También soporta formato con * como: *Dirección* calle 13...
    """
    fields = {}
    label_patterns = {
        'name': r'(?:\*?nombre(?:\s+completo)?\*?|contacto)\s*[:*]?\s*(.+)',
        'address': r'(?:\*?direcci[oó]n(?:\s+completa)?\*?|dir)\s*[:*]?\s*(.+)',
        'indications': r'(?:\*?indicaciones?\*?|\*?detalles?\*?|\*?barrio\*?|\*?referencias?\*?|informaci[oó]n\s+adicional)\s*[:*]?\s*(.+)',
        'locality': r'(?:\*?localidad(?:\s+de\s+entrega)?\*?)\s*[:*]?\s*(.+)',
        'day': r'(?:\*?d[ií]a(?:\s+de\s+entrega)?\*?|\*?fecha(?:\s+de\s+entrega)?\*?|entrega\s+el)\s*[:*]?\s*(.+)',
    }
    text_lower = text.lower()
    for field, pattern in label_patterns.items():
        match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip().split('\n')[0].strip()
            # Limpiar asteriscos y caracteres especiales
            value = re.sub(r'[*]', '', value).strip()
            if value and normalize_text(value) not in ('ninguna', 'ninguno', 'n/a', 'na', '-', 'no tengo', 'no hay'):
                fields[field] = value
    return fields


def extract_day(text: str):
    """
    Extrae el día de entrega.
    Retorna (day, error_message) donde error_message es None si es válido
    o un mensaje descriptivo si el día es inválido.
    """
    norm = normalize_text(text)

    # Buscar día válido
    for day_key, day_value in VALID_DAYS.items():
        if re.search(r'\b' + normalize_text(day_key) + r'\b', norm):
            return day_value, None

    # Buscar día inválido y dar mensaje específico
    for bad_day, error_msg in INVALID_DAYS.items():
        if re.search(r'\b' + normalize_text(bad_day) + r'\b', norm):
            return None, error_msg

    # Detectar fecha completa como "28 de mayo"
    date_pattern = r'\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b'
    if re.search(date_pattern, norm):
        return None, "Por favor envía solo el nombre del día (ej: Viernes), no la fecha completa"

    # Detectar "próximo lunes", "el lunes", etc.
    for day_key, day_value in VALID_DAYS.items():
        pattern = r'(?:proximo|próximo|el|este|para\s+el)\s+' + normalize_text(day_key)
        if re.search(pattern, norm):
            return day_value, None

    return None, None


def extract_locality(text: str):
    """
    Extrae la localidad.
    Retorna (locality, error_message).
    """
    norm = normalize_text(text)

    # Ordenar por longitud para evitar matches parciales
    sorted_locs = sorted(VALID_LOCATIONS.items(), key=lambda x: len(x[0]), reverse=True)
    for loc_key, loc_value in sorted_locs:
        loc_norm = normalize_text(loc_key)
        if re.search(r'\b' + re.escape(loc_norm) + r'\b', norm):
            if loc_value is None:
                return None, f"Lo sentimos, por el momento no hacemos entregas en esa zona. Las localidades disponibles son: Bosa, Kennedy, Puente Aranda, Tunjuelito, Antonio Nariño, Teusaquillo, Barrios Unidos, Mártires, Fontibón, Engativá, Chapinero, Usaquén, Soacha y Candelaria"
            return loc_value, None

    return None, None


def extract_address_and_indications(text: str) -> tuple:
    """Extrae dirección e indicaciones."""
    lines = [l.strip() for l in text.replace(',', '\n').split('\n') if l.strip()]

    address_line = None
    indication_parts = []

    for line in lines:
        norm_line = normalize_text(line)
        clean_line = clean_noise(line).strip()
        if not clean_line:
            continue

        if re.match(ADDRESS_START, norm_line, re.IGNORECASE):
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
        elif any(kw in norm_line for kw in INDICATION_KEYWORDS):
            clean = clean_noise(line).strip()
            if clean:
                indication_parts.append(clean)

    # Buscar en texto completo si no encontró por líneas
    if not address_line:
        match = re.search(
            r'(' + ADDRESS_START + r'\s*[\w\s.\-#]+?\d+[\w\s.\-#]*\d*)',
            text, re.IGNORECASE
        )
        if match:
            address_line = match.group(1).strip()

    seen = set()
    clean_indications = []
    for part in indication_parts:
        norm_part = normalize_text(part)
        if norm_part not in seen and len(norm_part) > 2:
            seen.add(norm_part)
            clean_indications.append(part)

    indications = ', '.join(clean_indications[:3])
    return address_line, indications


def extract_name(text: str):
    """Extrae el nombre del usuario."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    day_keywords = '|'.join(normalize_text(d) for d in VALID_DAYS.keys())
    loc_keywords = '|'.join(normalize_text(l) for l in VALID_LOCATIONS.keys() if l)

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
    """
    Extrae información de entrega con errores descriptivos por campo faltante.
    """
    if not text or not text.strip():
        return {
            "error": True,
            "errorMessage": "No recibí ningún mensaje. Por favor envía tus datos de entrega.",
            "info": None
        }

    # Detectar si el mensaje es muy corto o irrelevante
    word_count = len(text.strip().split())
    if word_count <= 2:
        return {
            "error": True,
            "errorMessage": "No se pudo extraer la información de entrega",
            "info": None
        }

    clean_text = clean_noise(text)
    labeled = extract_labeled_fields(clean_text)

    # Extraer nombre
    name = labeled.get('name') or extract_name(clean_text)

    # Extraer dirección
    address = None
    indications = labeled.get('indications', '')

    if labeled.get('address'):
        address = labeled['address']
        if not indications:
            _, ind = extract_address_and_indications(clean_text)
            indications = ind
    else:
        address, ind = extract_address_and_indications(clean_text)
        if ind and not indications:
            indications = ind

    # Extraer localidad
    locality = None
    locality_error = None
    if labeled.get('locality'):
        locality, locality_error = extract_locality(labeled['locality'])
    if not locality and not locality_error:
        locality, locality_error = extract_locality(clean_text)

    # Si la localidad es inválida (municipio fuera de Bogotá)
    if locality_error and "no hacemos entregas" in locality_error:
        return {
            "error": True,
            "errorMessage": locality_error,
            "info": None
        }

    # Extraer día
    day = None
    day_error = None
    if labeled.get('day'):
        day, day_error = extract_day(labeled['day'])
    if not day and not day_error:
        day, day_error = extract_day(clean_text)

    # Construir errores descriptivos por campo faltante
    missing = []

    if not address:
        missing.append("dirección completa (ej: Calle 13 #45-67)")

    if not locality:
        if locality_error:
            missing.append(f"localidad válida ({locality_error})")
        else:
            missing.append("localidad (ej: Kennedy, Suba, Chapinero)")

    if not day:
        if day_error:
            # Error específico de día inválido
            return {
                "error": True,
                "errorMessage": day_error,
                "info": None
            }
        else:
            missing.append("día de entrega (Lunes, Martes, Miércoles, Jueves, Viernes o Sábado)")

    if missing:
        if len(missing) == 1:
            error_msg = f"No encontré el/la {missing[0]}"
        elif len(missing) == 2:
            error_msg = f"No encontré: {missing[0]} y {missing[1]}"
        else:
            error_msg = "No se pudo extraer la información de entrega"

        return {
            "error": True,
            "errorMessage": error_msg,
            "info": None
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
# ANÁLISIS DE FALLOS — /analyze-failures
# ============================================================

KNOWN_LOCATIONS_SET = set(VALID_LOCATIONS.keys())

DAY_SUGGESTIONS_ANALYSIS = {
    "manana": "Usuario escribió 'mañana' en lugar del nombre del día",
    "hoy": "Usuario escribió 'hoy' en lugar del nombre del día",
    "domingo": "Domingo no es día de entrega disponible",
    "pasado manana": "Usuario escribió 'pasado mañana' en lugar del nombre del día",
}


def analyze_failures_logic(messages: list) -> list:
    suggestions = []
    seen = set()

    for text in messages:
        if not text or not text.strip():
            continue

        norm = normalize_text(text)
        clean_text = clean_noise(text)

        # Detectar barrios nuevos no reconocidos
        barrio_patterns = [
            r'\bbarrio\s+([a-záéíóúñ\s]{3,30})',
            r'\bsector\s+([a-záéíóúñ\s]{3,30})',
            r'\burbanizacion\s+([a-záéíóúñ\s]{3,30})',
        ]
        for pattern in barrio_patterns:
            matches = re.findall(pattern, norm)
            for match in matches:
                match_clean = match.strip()
                if match_clean and match_clean not in KNOWN_LOCATIONS_SET:
                    key = f"loc_{match_clean}"
                    if key not in seen:
                        seen.add(key)
                        suggestions.append({
                            "type": "new_location",
                            "value": match_clean,
                            "suggested_locality": None,
                            "original_message": text[:120],
                        })

        # Detectar días no válidos
        day, day_error = extract_day(text)
        if not day:
            for bad_day, suggestion in DAY_SUGGESTIONS_ANALYSIS.items():
                if re.search(r'\b' + bad_day + r'\b', norm):
                    key = f"day_{bad_day}"
                    if key not in seen:
                        seen.add(key)
                        suggestions.append({
                            "type": "day_format",
                            "value": bad_day,
                            "suggestion": suggestion,
                            "original_message": text[:120],
                        })

            date_pattern = r'\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b'
            date_match = re.search(date_pattern, norm)
            if date_match:
                key = f"date_{date_match.group(0)}"
                if key not in seen:
                    seen.add(key)
                    suggestions.append({
                        "type": "day_format",
                        "value": date_match.group(0),
                        "suggestion": "Usuario envió fecha completa en lugar del nombre del día",
                        "original_message": text[:120],
                    })

        # Detectar si tiene dirección y localidad pero no día
        has_address = bool(re.search(ADDRESS_START, norm))
        locality, _ = extract_locality(text)
        if has_address and locality and not day:
            key = f"missing_day_{text[:40]}"
            if key not in seen:
                seen.add(key)
                suggestions.append({
                    "type": "missing_day",
                    "value": text[:120],
                    "suggestion": "Usuario envió dirección y localidad pero olvidó el día de entrega",
                    "original_message": text[:120],
                })

    return suggestions


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


@app.route('/analyze-failures', methods=['POST'])
def analyze_failures():
    data = request.get_json()
    if not data or 'messages' not in data:
        return jsonify({'error': True, 'errorMessage': 'Se requiere el campo messages', 'suggestions': []}), 400

    messages = data['messages']
    if not isinstance(messages, list):
        return jsonify({'error': True, 'errorMessage': 'messages debe ser una lista', 'suggestions': []}), 400

    suggestions = analyze_failures_logic(messages)
    return jsonify({
        'error': False,
        'errorMessage': None,
        'analyzed': len(messages),
        'suggestions': suggestions,
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': '3.0.0'})


# v3.0.0 — extractor robusto con errores descriptivos por campo
if __name__ == '__main__':
    app.run(debug=True, port=5000)


# ============================================================
# RESPUESTAS PREDEFINIDAS POR INTENCIÓN — siempre responde
# ============================================================

INTENT_RESPONSES = {
    "saludo": [
        "¡Hola! 😊 Soy Vecinito de CorApp, tu asistente para comprar productos frescos directo de la central de abastos. ¿En qué te puedo ayudar hoy?",
        "¡Hola! 👋 Bienvenido a CorApp. Tenemos frutas y verduras frescas con domicilio gratis. ¿Qué necesitas hoy?",
        "¡Hola vecino! 😊 Estoy aquí para ayudarte con tu mercado. ¿Quieres ver el catálogo? https://wa.me/c/573124929496",
    ],
    "consulta_catalogo": [
        "¡Claro! Aquí tienes nuestro catálogo completo: https://wa.me/c/573124929496 😊 Elige lo que necesites y sigue los pasos para pedir.",
        "Puedes ver todos nuestros productos aquí: https://wa.me/c/573124929496 🛒 El proceso es fácil — agrega al carrito, ver carrito y realizar pedido.",
    ],
    "consulta_domicilio": [
        "¡Sí! El domicilio es completamente gratis 🚚 Llegamos a casi toda Bogotá. ¿En qué localidad estás?",
        "El envío es totalmente gratis 😊 Entregamos en Bosa, Kennedy, Suba, Chapinero, Engativá, Fontibón, Teusaquillo, Usaquén y más localidades de Bogotá.",
    ],
    "hora_pedido": [
        "Las entregas se realizan en horas de la mañana ☀️ entre 7:00 AM y 12:00 PM del día que elijas. Si haces el pedido hoy, llega mañana.",
        "Entregamos de 7:00 AM a 12:00 PM 🚚 Si pides hoy antes de las 7 PM, te llega mañana en la mañana.",
    ],
    "metodo_pago": [
        "Aceptamos efectivo, Nequi y Bre-b 💳 Todo se paga al momento de recibir tu pedido en la puerta.",
        "Puedes pagar en efectivo, por Nequi o Bre-b 😊 El pago es al recibir — sin anticipos.",
    ],
    "consulta_producto": [
        "Tenemos muchos productos frescos 🥦🍎 Revisa nuestro catálogo para ver todo disponible: https://wa.me/c/573124929496",
        "¡Puedo ayudarte! Dime el nombre del producto y te comparto el enlace directo 😊",
    ],
    "pedido_no_ha_llegado": [
        "Lamento mucho la espera 😔 Ya notifico al equipo para que revisen el estado de tu entrega. Pronto te contactarán.",
        "Entiendo tu preocupación, voy a informar al supervisor ahora mismo para que te den respuesta cuanto antes 🙏",
    ],
    "queja_servicio": [
        "Lamento mucho lo que pasó 😔 Tu comentario es muy importante para nosotros. Ya informo al supervisor para darte una solución.",
        "Entiendo tu molestia y me disculpo por los inconvenientes 🙏 Voy a escalar tu caso al equipo para que te contacten pronto.",
    ],
    "consulta_estado_pedido": [
        "Déjame revisar el estado de tu pedido 🔍 ¿Me confirmas el día en que lo programaste?",
        "Voy a informar al equipo para que te confirmen el estado de tu entrega. ¿Tienes algún dato adicional del pedido?",
    ],
    "fuera_de_tema": [
        "Por ahora solo manejo ventas de frutas y verduras 🥦 Si necesitas algo de nuestro catálogo, aquí estoy: https://wa.me/c/573124929496",
        "Eso está fuera de lo que puedo ayudarte, pero ya informo al equipo 😊 Para compras, aquí tienes el catálogo: https://wa.me/c/573124929496",
    ],
    "confirmar_pedido": [
        "Para confirmar tu pedido, por favor escribe la palabra *Confirmar* exactamente así 😊",
    ],
    "rechazar_pedido": [
        "Para cancelar tu pedido, por favor escribe la palabra *Rechazar* exactamente así 😊",
    ],
}

# Respuestas para cuando no se reconoce nada
FALLBACK_RESPONSES = [
    "¡Hola! 😊 Soy Vecinito de CorApp. ¿En qué te puedo ayudar hoy? Si quieres ver nuestros productos frescos, aquí está el catálogo: https://wa.me/c/573124929496",
    "Entiendo tu mensaje 😊 Para ayudarte mejor, ¿puedes decirme si necesitas hacer un pedido, consultar un producto o tienes alguna duda?",
    "¡Aquí estoy para ayudarte! 😊 Si quieres comprar frutas y verduras frescas, entra al catálogo: https://wa.me/c/573124929496",
    "Ya informé tu mensaje al equipo 😊 Mientras tanto, si quieres ver nuestros productos: https://wa.me/c/573124929496",
]

import random


def get_response_for_intent(intent: str, confidence: float) -> str | None:
    """Retorna una respuesta predefinida para la intención dada."""
    if confidence < 0.15:
        return random.choice(FALLBACK_RESPONSES)

    responses = INTENT_RESPONSES.get(intent)
    if responses:
        return random.choice(responses)

    return random.choice(FALLBACK_RESPONSES)


@app.route('/respond', methods=['POST'])
def respond():
    """
    Siempre retorna una respuesta — nunca deja al usuario sin contestar.
    """
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({
            'response': random.choice(FALLBACK_RESPONSES),
            'intent': 'unknown',
            'confidence': 0,
        })

    message = data['message']

    # Clasificar intención
    try:
        intent, confidence = get_confidence(message)
    except Exception:
        intent = 'unknown'
        confidence = 0.0

    response = get_response_for_intent(intent, confidence)

    return jsonify({
        'response': response,
        'intent': intent,
        'confidence': round(confidence, 4),
        'use_fallback': confidence < 0.2,
    })