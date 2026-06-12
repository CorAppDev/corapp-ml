"""
CorApp ML — v5.0.0 COMPLETO
Sistema de inteligencia artificial para ventas por WhatsApp en español colombiano.

Arquitectura:
- Clasificador de intenciones con TF-IDF + LinearSVC
- Extractor de datos de entrega ultra-robusto
- Tres algoritmos de similitud combinados (Levenshtein + Jaro + Jaro-Winkler)
- Aliases y abreviaciones del español colombiano coloquial
- Respuestas naturales variadas por intención
- Manejo de frustración y empatía
- Análisis de fallos para mejora continua
- Siempre responde — nunca silencio

Basado en:
- 400+ conversaciones reales de CorApp
- Patrones del español colombiano coloquial
- Errores tipográficos y abreviaciones comunes
- Modismos y expresiones locales
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import json
import re
import unicodedata
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

app = Flask(__name__)
model = None


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalize(text: str) -> str:
    """Normaliza texto — minúsculas, sin tildes, sin espacios extra."""
    if not text:
        return ""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r'\s+', ' ', text)
    return text


# ============================================================
# ALGORITMOS DE SIMILITUD
# ============================================================

def levenshtein(a: str, b: str) -> float:
    """Distancia de edición normalizada — buena para palabras largas."""
    a, b = normalize(a), normalize(b)
    if a == b: return 1.0
    if not a or not b: return 0.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return 1.0 - dp[n] / max(m, n)


def jaro(a: str, b: str) -> float:
    """Similitud Jaro — excelente para palabras cortas."""
    a, b = normalize(a), normalize(b)
    if a == b: return 1.0
    if not a or not b: return 0.0
    md = max(len(a), len(b)) // 2 - 1
    if md < 0: md = 0
    am = [False] * len(a)
    bm = [False] * len(b)
    matches = trans = 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - md), min(i + md + 1, len(b))):
            if bm[j] or ca != b[j]: continue
            am[i] = bm[j] = True
            matches += 1
            break
    if not matches: return 0.0
    k = 0
    for i in range(len(a)):
        if not am[i]: continue
        while not bm[k]: k += 1
        if a[i] != b[k]: trans += 1
        k += 1
    return (matches/len(a) + matches/len(b) + (matches - trans/2)/matches) / 3


def jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler — mejor para palabras con prefijo común."""
    j = jaro(a, b)
    an, bn = normalize(a), normalize(b)
    p = sum(1 for i in range(min(4, len(an), len(bn))) if an[i] == bn[i])
    return j + p * 0.1 * (1 - j)


def similarity(a: str, b: str) -> float:
    """
    Combina los tres algoritmos.
    Para palabras cortas Jaro-Winkler es mejor.
    Para palabras largas Levenshtein es más preciso.
    """
    la = len(normalize(a))
    lev = levenshtein(a, b)
    jw = jaro_winkler(a, b)
    if la <= 5:
        return max(lev, jw)
    return lev * 0.55 + jw * 0.45


# ============================================================
# ML — CLASIFICADOR DE INTENCIONES
# ============================================================

def train_model():
    with open('data/training.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentences, labels = [], []
    for intent in data['intents']:
        for example in intent['examples']:
            sentences.append(normalize(example))
            labels.append(intent['tag'])
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
            max_features=50000,
        )),
        ('clf', LinearSVC(max_iter=3000, C=1.2, class_weight='balanced'))
    ])
    pipeline.fit(sentences, labels)
    os.makedirs('model', exist_ok=True)
    joblib.dump(pipeline, 'model/intent_classifier.pkl')
    return pipeline


def load_model():
    global model
    if model is None:
        model = train_model() if not os.path.exists('model/intent_classifier.pkl') \
            else joblib.load('model/intent_classifier.pkl')
    return model


def classify(text: str) -> tuple:
    clf = load_model()
    norm_text = normalize(text)
    intent = clf.predict([norm_text])[0]
    scores = clf.decision_function([norm_text])[0]
    exp = np.exp(scores - np.max(scores))
    confidence = float(np.max(exp) / exp.sum())
    return intent, confidence


# ============================================================
# DATOS DE REFERENCIA — DÍAS
# ============================================================

VALID_DAYS = {
    "lunes": "Lunes", "martes": "Martes",
    "miercoles": "Miércoles", "miércoles": "Miércoles",
    "jueves": "Jueves", "viernes": "Viernes",
    "sabado": "Sábado", "sábado": "Sábado",
}

# Alias colombianos — abreviaciones y errores comunes
DAY_ALIASES = {
    # Lunes
    "lun": "Lunes", "lns": "Lunes", "lnes": "Lunes", "lun.": "Lunes",
    "el lunes": "Lunes", "este lunes": "Lunes",
    # Martes
    "mar": "Martes", "mrt": "Martes", "mrts": "Martes",
    "el martes": "Martes", "este martes": "Martes",
    # Miércoles
    "mie": "Miércoles", "mier": "Miércoles", "mirc": "Miércoles",
    "mierc": "Miércoles", "miercole": "Miércoles", "miercols": "Miércoles",
    "merco": "Miércoles", "el miercoles": "Miércoles",
    # Jueves
    "jue": "Jueves", "jues": "Jueves", "jvs": "Jueves", "jves": "Jueves",
    "juev": "Jueves", "juevs": "Jueves", "jve": "Jueves", "jv": "Jueves",
    "el jueves": "Jueves",
    # Viernes
    "vie": "Viernes", "vies": "Viernes", "vrs": "Viernes",
    "viern": "Viernes", "vierens": "Viernes", "el viernes": "Viernes",
    # Sábado
    "sab": "Sábado", "sabs": "Sábado", "sbd": "Sábado",
    "sabdo": "Sábado", "sbdo": "Sábado", "sab.": "Sábado",
    "el sabado": "Sábado", "el sábado": "Sábado",
}

# Días inválidos con mensajes naturales
INVALID_DAYS = {
    "domingo": "Los domingos no hacemos entregas 😊 Puedes elegir entre Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "manana": "Por favor dime el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "mañana": "Por favor dime el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "hoy": "Por favor dime el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "pasado manana": "Por favor dime el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "pasado mañana": "Por favor dime el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "entre semana": "¿Cuál día entre semana prefieres? Lunes, Martes, Miércoles, Jueves o Viernes",
    "fin de semana": "El único día de fin de semana disponible es el Sábado 😊",
    "lo antes posible": "Con gusto 😊 ¿Cuál día prefieres? Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "lo mas pronto": "Con gusto 😊 ¿Cuál día prefieres? Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "cualquier dia": "¿Cuál día te queda mejor? Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "cuando puedan": "¿Cuál día te queda mejor? Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
    "pronto": "Por favor dime el nombre del día exacto: Lunes, Martes, Miércoles, Jueves, Viernes o Sábado",
}


# ============================================================
# DATOS DE REFERENCIA — LOCALIDADES
# ============================================================

VALID_LOCATIONS = {
    # Localidades oficiales de Bogotá
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
    "rafael uribe": "Rafael Uribe Uribe",
    "rafael uribe uribe": "Rafael Uribe Uribe",
    "ciudad bolivar": "Ciudad Bolivar",
    "ciudad bolívar": "Ciudad Bolivar",
    "san cristobal": "San Cristobal",
    "san cristóbal": "San Cristobal",
    "usme": "Usme",
    "sumapaz": "Sumapaz",
    # Barrios → Localidad
    "modelia": "Fontibon",
    "capellania": "Fontibon",
    "capellanía": "Fontibon",
    "fontibón sur": "Fontibon",
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
    "la libertad bosa": "Bosa",
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
    "portales norte": "Engativa",
    "san agustin": "Kennedy",
    "san agustín": "Kennedy",
    "corabastos": "Kennedy",
    "zona industrial kennedy": "Kennedy",
    "castellon de los condes": "Kennedy",
    "castellón de los condes": "Kennedy",
    "cra 87b": "Kennedy",
    "antiguo country": "Chapinero",
    "country": "Chapinero",
    "rosales": "Chapinero",
    "portal de rosales": "Chapinero",
    "chapinero alto": "Chapinero",
    "chapinero norte": "Chapinero",
    "gran estacion": "Teusaquillo",
    "gran estación": "Teusaquillo",
    "palermo": "Teusaquillo",
    "la soledad": "Teusaquillo",
    "armenia": "Teusaquillo",
    "bahia solano": "Fontibon",
    "bahía solano": "Fontibon",
    "ciudad salitre": "Fontibon",
    "salitre": "Fontibon",
    "la giralda": "Fontibon",
    "villa del prado": "Suba",
    "alhambra": "Suba",
    "cedritos": "Usaquen",
    "santa barbara": "Usaquen",
    "santa bárbara": "Usaquen",
    "mazuren": "Suba",
    "mazurén": "Suba",
    "niza": "Suba",
    "bello horizonte": "Suba",
    "verbenal": "Usaquen",
    "toberin": "Usaquen",
    "toberín": "Usaquen",
    "country norte": "Usaquen",
    "santa cecilia": "Engativa",
    "villa luz": "Engativa",
    "gaitan": "Barrios unidos",
    "gaitán": "Barrios unidos",
    "alcazares": "Barrios unidos",
    "siete de agosto": "Barrios unidos",
    "la floresta": "Engativa",
    "floresta": "Engativa",
    "quirigua": "Engativa",
    "quiriguá": "Engativa",
    "minuto de dios": "Engativa",
    "bachue": "Engativa",
    "tintal": "Kennedy",
    "americas": "Kennedy",
    "américas": "Kennedy",
    "timiza": "Kennedy",
    "muzú": "Puente aranda",
    "muzu": "Puente aranda",
    "la esperanza": "Kennedy",
    "candelaria la nueva": "Kennedy",
    # Municipios fuera de cobertura → None
    "cundinamarca": None,
    "zipaquira": None,
    "zipaquirá": None,
    "chia": None,
    "chía": None,
    "sopo": None,
    "sopó": None,
    "cajica": None,
    "cajicá": None,
    "mosquera": None,
    "madrid cundinamarca": None,
    "facatativa": None,
    "facatativá": None,
    "funza": None,
    "tocancipa": None,
    "tocancipá": None,
    "la calera": None,
    "cota": None,
    "sibate": None,
    "sibaté": None,
}

# ============================================================
# LIMPIEZA DE RUIDO
# ============================================================

NOISE_PATTERNS = [
    r'[\w\.-]+@[\w\.-]+\.\w+',                              # emails
    r'\b3\d{9}\b',                                           # cel colombiano
    r'\b\d{7,10}\b',                                         # otros números
    r'\bpago\s+(?:contra\s+entrega|en\s+efectivo|nequi|bre-?b|transferencia|electronica)\b',
    r'\btelefono\s*[:*]?\s*[\d\s\-]+',
    r'\bcel(?:ular)?\s*[:*]?\s*[\d\s\-]+',
    r'\bcorreo\s*[:*]?\s*\S+',
    r'\bhoras?\s+de\s+la\s+ma[nñ]ana\b',
    r'\bhorario\s*[:*]?\s*[\d\sapm:]+',
    r'\bcontacto\s*[:*]?\s*[\d\s\-]+',
]


def clean_noise(text: str) -> str:
    for p in NOISE_PATTERNS:
        text = re.sub(p, ' ', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()


# ============================================================
# EXTRACCIÓN DE CAMPOS CON ETIQUETAS
# ============================================================

def extract_labeled(text: str) -> dict:
    """
    Detecta cuando el usuario usa etiquetas explícitas.
    Soporta: Nombre:, Dirección:, Barrio:, Localidad:, Día:
    Con o sin asteriscos, con o sin dos puntos.
    """
    fields = {}
    patterns = {
        'name':        r'(?:\*?\s*nombre(?:\s+completo)?\s*\*?)\s*[:*\-]?\s*(.+)',
        'address':     r'(?:\*?\s*direcci[oó]n(?:\s+completa)?\s*\*?|dir)\s*[:*\-]?\s*(.+)',
        'indications': r'(?:\*?\s*(?:indicaciones?|detalles?|barrio|referencias?|datos?\s+adicionales?|informaci[oó]n\s+adicional)\s*\*?)\s*[:*\-]?\s*(.+)',
        'locality':    r'(?:\*?\s*localidad(?:\s+de\s+entrega)?\s*\*?)\s*[:*\-]?\s*(.+)',
        'day':         r'(?:\*?\s*(?:d[ií]a(?:\s+de\s+entrega)?|fecha(?:\s+de\s+entrega)?|entrega\s+el|dia\s+entrega)\s*\*?)\s*[:*\-]?\s*(.+)',
    }
    tl = text.lower()
    for field, pattern in patterns.items():
        m = re.search(pattern, tl, re.IGNORECASE | re.MULTILINE)
        if m:
            val = re.sub(r'[*]', '', m.group(1).strip().split('\n')[0]).strip()
            if val and normalize(val) not in ('ninguna','ninguno','n/a','na','-','no tengo','no hay'):
                fields[field] = val
    return fields


# ============================================================
# EXTRACCIÓN DE DÍA — ULTRA ROBUSTA
# ============================================================

def extract_day(text: str):
    """
    7 estrategias en cascada para extraer el día:
    1. Exacto
    2. Alias y abreviaciones colombianas
    3. Contexto (próximo lunes, para el martes)
    4. Días inválidos con mensaje empático
    5. Fecha completa (28 de mayo)
    6. Búsqueda difusa multi-algoritmo
    7. Detección de intención temporal vaga
    """
    norm = normalize(text)
    words = norm.split()

    # 1. Exacto
    for k, v in VALID_DAYS.items():
        if re.search(r'\b' + normalize(k) + r'\b', norm):
            return v, None

    # 2. Alias y abreviaciones
    for w in words:
        wc = w.strip('.,;:!?')
        if wc in DAY_ALIASES:
            return DAY_ALIASES[wc], None
    # También buscar frases de alias
    for alias, val in DAY_ALIASES.items():
        if ' ' in alias and alias in norm:
            return val, None

    # 3. Contexto — "próximo X", "para el X", "el X que viene"
    for k, v in VALID_DAYS.items():
        kn = normalize(k)
        patterns = [
            r'(?:proximo|próximo|el|este|para\s+el|para\s+el\s+dia|el\s+dia)\s+' + kn,
            kn + r'\s+(?:que\s+viene|próximo|proximo|siguiente)',
        ]
        for p in patterns:
            if re.search(p, norm):
                return v, None

    # 4. Días inválidos con mensaje empático
    for bad, msg in INVALID_DAYS.items():
        if re.search(r'\b' + normalize(bad) + r'\b', norm):
            return None, msg

    # 5. Fecha completa
    date_p = r'\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b'
    if re.search(date_p, norm):
        return None, "Por favor envíame solo el nombre del día 😊 Por ejemplo: Viernes"

    # 6. Difusa multi-algoritmo
    # Palabras a excluir: localidades, nombres comunes, palabras de dirección
    EXCLUDE_FROM_FUZZY_DAY = set()
    for k in VALID_LOCATIONS:
        for w in normalize(k).split():
            if len(w) >= 4:
                EXCLUDE_FROM_FUZZY_DAY.add(w)
    # Nombres propios colombianos comunes que confunden el fuzzy
    EXCLUDE_FROM_FUZZY_DAY.update([
        'maria', 'marta', 'mario', 'marco', 'lucia', 'luisa', 'laura',
        'diana', 'dina', 'bosa', 'suba', 'cali', 'lopez', 'loaiza',
        'calle', 'carrera', 'avenida', 'diagonal', 'transversal',
        'casa', 'apto', 'piso', 'torre', 'bloque', 'norte', 'sur',
        'este', 'oeste', 'bis', 'interior', 'local', 'oficina',
    ])

    best_v, best_s = None, 0.0
    threshold = 0.75  # Más estricto para evitar falsos positivos
    for w in words:
        wc = w.strip('.,;:!?')
        if len(wc) < 3:
            continue
        if wc in EXCLUDE_FROM_FUZZY_DAY:
            continue
        for k, v in VALID_DAYS.items():
            s = similarity(wc, k)
            if s > best_s and s >= threshold:
                best_s, best_v = s, v
    if best_v:
        return best_v, None

    # 7. Intención temporal vaga
    vague = ['lo antes', 'lo mas pronto', 'urgente', 'ya', 'ahora',
             'cuando puedan', 'pronto', 'rapido', 'rápido']
    for v in vague:
        if v in norm:
            return None, "¿Cuál día te queda mejor? Lunes, Martes, Miércoles, Jueves, Viernes o Sábado 😊"

    return None, None


# ============================================================
# EXTRACCIÓN DE LOCALIDAD — ULTRA ROBUSTA
# ============================================================

def extract_locality(text: str):
    """
    4 estrategias para extraer localidad:
    1. Exacto incluyendo barrios y sectores
    2. Difusa para localidades principales
    3. Difusa para bigrams (dos palabras)
    4. Detección de municipios fuera de cobertura
    """
    norm = normalize(text)
    words = norm.split()

    # 1. Exacto — más largo primero para evitar matches parciales
    for k, v in sorted(VALID_LOCATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        kn = normalize(k)
        if re.search(r'\b' + re.escape(kn) + r'\b', norm):
            if v is None:
                return None, "Lo sentimos, por ahora no llegamos a esa zona 😔 Cubrimos: Bosa, Kennedy, Suba, Chapinero, Engativá, Fontibón, Teusaquillo, Usaquén, Barrios Unidos, Puente Aranda, Tunjuelito, Antonio Nariño, Mártires, Soacha y Candelaria"
            return v, None

    # 2. Difusa palabras individuales
    best_v, best_s = None, 0.0
    best_invalid = False
    thresh = 0.78

    for w in words:
        if len(w) < 4:
            continue
        for k, v in VALID_LOCATIONS.items():
            if ' ' in k:
                continue
            s = similarity(w, normalize(k))
            if s > best_s and s >= thresh:
                best_s, best_v, best_invalid = s, v, (v is None)

    # 3. Difusa bigrams
    for i in range(len(words) - 1):
        bg = words[i] + ' ' + words[i+1]
        for k, v in VALID_LOCATIONS.items():
            if ' ' not in k:
                continue
            s = similarity(bg, normalize(k))
            if s > best_s and s >= 0.80:
                best_s, best_v, best_invalid = s, v, (v is None)

    if best_v is not None:
        return best_v, None
    if best_invalid:
        return None, "Lo sentimos, por ahora no llegamos a esa zona 😔"

    return None, None


# ============================================================
# EXTRACCIÓN DE DIRECCIÓN E INDICACIONES
# ============================================================

ADDR_START = r'(?:calle|cll|cl|carrera|cra|cr|kra|avenida|av|transversal|transv|tranv|tv|diagonal|dg|autopista|ak)'

INDIC_KW = [
    'apto','apartamento','apt','torre','bloque','interior','int',
    'piso','local','oficina','conjunto','edificio','etapa','unidad',
    'porteria','portería','dejar en','entregar en','llamar','timbrar',
    'rejas','reja','esquina','frente','cerca','al lado',
    'despues','después','si no estoy','dejar con','portero',
    'casa azul','casa blanca','casa roja','casa verde','casa amarilla',
    'primer piso','segundo piso','tercer piso','cuarto piso',
    'peluqueria','peluquería','tienda','drogueria','droguería',
    'supermercado','parque','iglesia','colegio','hospital','clinica',
    'clínica','farmacia','restaurante','panaderia','panadería',
    'diagonal a','frente al','al lado de','cerca al','detras','detrás',
]


def extract_address_indications(text: str) -> tuple:
    lines = [l.strip() for l in text.replace(',', '\n').split('\n') if l.strip()]
    address = None
    indics = []

    for line in lines:
        nl = normalize(line)
        cl = clean_noise(line).strip()
        if not cl:
            continue
        if re.match(ADDR_START, nl, re.IGNORECASE):
            m = re.match(
                r'(' + ADDR_START + r'\s*[\w\s.\-#bis]+?\d+[\w\s.\-#]*\d*)',
                cl, re.IGNORECASE
            )
            if m:
                address = m.group(1).strip()
                rest = cl[len(address):].strip().strip(',').strip()
                if rest and len(rest) > 2:
                    indics.append(rest)
            else:
                address = cl
        elif any(kw in nl for kw in INDIC_KW):
            c = clean_noise(line).strip()
            if c and len(c) > 2:
                indics.append(c)

    # Buscar en texto completo si no encontró
    if not address:
        m = re.search(
            r'(' + ADDR_START + r'\s*[\w\s.\-#]+?\d+[\w\s.\-#]*\d*)',
            text, re.IGNORECASE
        )
        if m:
            address = m.group(1).strip()

    # Deduplicar indicaciones
    seen, clean_indics = set(), []
    for p in indics:
        np_ = normalize(p)
        if np_ not in seen and len(np_) > 2:
            seen.add(np_)
            clean_indics.append(p)

    return address, ', '.join(clean_indics[:3])


# ============================================================
# EXTRACCIÓN DE NOMBRE
# ============================================================

def extract_name(text: str):
    """
    Extrae nombre del usuario.
    Solo acepta líneas con letras y espacios, 2-5 palabras,
    sin números, emails, días ni localidades.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    day_kw = '|'.join(normalize(d) for d in list(VALID_DAYS.keys()) + list(DAY_ALIASES.keys()))
    loc_kw = '|'.join(normalize(l) for l in VALID_LOCATIONS.keys() if l)

    for line in lines[:4]:
        n = normalize(line)
        c = clean_noise(line).strip()
        if (re.match(r'^[a-záéíóúñ\s]+$', n)
                and 2 <= len(c.split()) <= 5
                and not re.search(r'\b(' + day_kw + r')\b', n)
                and not re.search(r'\b(' + loc_kw + r')\b', n)
                and not re.search(r'@|\d|calle|carrera|avenida|diagonal|transversal', n)):
            return c
    return None


# ============================================================
# EXTRACTOR PRINCIPAL
# ============================================================

def extract_delivery(text: str, valid_locations=None, valid_days=None, valid_times=None) -> dict:
    """
    Extrae datos de entrega con máxima robustez.
    Errores descriptivos y empáticos por campo faltante.
    """
    if not text or not text.strip():
        return {"error": True, "errorMessage": "No recibí ningún mensaje 😊", "info": None}

    # Mensaje muy corto
    words = text.strip().split()
    if len(words) <= 2:
        day, _ = extract_day(text)
        if day:
            return {"error": True,
                    "errorMessage": "Solo recibí el día de entrega 😊 Por favor envíame también tu nombre, dirección y localidad",
                    "info": None}
        return {"error": True, "errorMessage": "No se pudo extraer la información de entrega", "info": None}

    clean = clean_noise(text)
    labeled = extract_labeled(clean)

    # Nombre
    raw_name = labeled.get('name', '')
    if raw_name:
        nn = normalize(raw_name.split('\n')[0].strip())
        name = raw_name.split('\n')[0].strip() if re.match(r'^[a-záéíóúñ\s]+$', nn) and len(raw_name.split()) <= 5 else extract_name(clean)
    else:
        name = extract_name(clean)

    # Dirección
    address = indications = None
    indications = labeled.get('indications', '')
    if labeled.get('address'):
        address = labeled['address']
        if not indications:
            _, ind = extract_address_indications(clean)
            indications = ind
    else:
        address, ind = extract_address_indications(clean)
        if ind and not indications:
            indications = ind

    # Localidad
    locality = locality_err = None
    if labeled.get('locality'):
        locality, locality_err = extract_locality(labeled['locality'])
    if not locality and not locality_err:
        locality, locality_err = extract_locality(clean)

    if locality_err and "no llegamos" in locality_err:
        return {"error": True, "errorMessage": locality_err, "info": None}

    # Día
    day = day_err = None
    if labeled.get('day'):
        day, day_err = extract_day(labeled['day'])
    if not day and not day_err:
        day, day_err = extract_day(clean)

    if not day and day_err:
        return {"error": True, "errorMessage": day_err, "info": None}

    # Errores descriptivos
    missing = []
    if not address:
        missing.append("dirección completa (ej: Calle 13 #45-67)")
    if not locality:
        missing.append("localidad (ej: Kennedy, Suba, Chapinero)")
    if not day:
        missing.append("día de entrega (Lunes a Sábado)")

    if missing:
        if len(missing) == 1:
            f = missing[0]
            art = "el" if f.startswith("día") else "la"
            msg = f"Faltó {art} {f} 😊 ¿Me lo puedes enviar?"
        elif len(missing) == 2:
            msg = f"Faltan: {missing[0]} y {missing[1]} 😊"
        else:
            msg = "No se pudo extraer la información de entrega"
        return {"error": True, "errorMessage": msg, "info": None}

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
# RESPUESTAS NATURALES — siempre responde algo
# ============================================================

RESPONSES = {
    "saludo": [
        "¡Hola! 😊 Soy Vecinito de CorApp. ¿En qué te puedo ayudar hoy?",
        "¡Hola vecino! 👋 Tenemos frutas y verduras frescas con domicilio gratis. ¿Qué necesitas?",
        "¡Bienvenido a CorApp! 😊 ¿Te ayudo con tu mercado hoy? Catálogo: https://wa.me/c/573124929496",
        "¡Quiubo! 😊 Soy Vecinito de CorApp, listo para ayudarte con tu mercado fresco.",
    ],
    "consulta_catalogo": [
        "¡Claro! Aquí el catálogo: https://wa.me/c/573124929496 😊 Agrega lo que necesites y sigue los pasos.",
        "Entra aquí, escoge, agrega al carrito y listo: https://wa.me/c/573124929496 🛒",
        "El proceso es fácil 😊\n1️⃣ Entra al catálogo: https://wa.me/c/573124929496\n2️⃣ Agrega productos\n3️⃣ Ver carrito → Realizar pedido",
    ],
    "consulta_domicilio": [
        "¡El domicilio es completamente gratis! 🚚 ¿En qué localidad estás?",
        "Sí, domicilio gratis a casi toda Bogotá 😊 Kennedy, Suba, Bosa, Chapinero, Engativá, Fontibón y más.",
        "¡Gratis y a tu puerta! 🏠 Entre 7 AM y 12 PM del día que elijas.",
    ],
    "hora_pedido": [
        "Entregamos de 7:00 AM a 12:00 PM ☀️ Si pides hoy antes de las 7 PM, llega mañana.",
        "Las entregas son en la mañana 🚚 entre 7 AM y 12 PM del día que elijas.",
        "Si pides hoy, mañana tienes tu mercado fresco en la puerta 😊 El horario es 7 AM - 12 PM.",
    ],
    "metodo_pago": [
        "Aceptamos efectivo, Nequi y Bre-b 💳 Todo se paga al recibir — sin anticipos.",
        "Puedes pagar en efectivo, Nequi o Bre-b cuando llegue el pedido 😊 ¡Sin anticipos!",
        "El pago es contra entrega 🙌 Efectivo, Nequi o Bre-b. No se paga antes.",
    ],
    "consulta_producto": [
        "¡Tenemos muchos productos frescos! 🥦🍎 Dime cuál buscas y te comparto el enlace.",
        "Puedes ver todo aquí: https://wa.me/c/573124929496 😊 ¿Qué producto buscas?",
        "¡Claro! ¿Qué producto necesitas? Te comparto el enlace directo 😊",
    ],
    "pedido_no_ha_llegado": [
        "Lamento mucho la espera 😔 Ya notifico al equipo ahora mismo para que revisen tu entrega.",
        "¡Qué pena! Voy a informar al supervisor de inmediato 🚨 Pronto te contactarán.",
        "Entiendo tu preocupación 🙏 Ya escalo tu caso al equipo de logística urgente.",
    ],
    "queja_servicio": [
        "Lamento mucho lo ocurrido 😔 Ya informo al supervisor para darte una solución cuanto antes.",
        "Entiendo tu molestia y me disculpo 🙏 Tu caso ya está en manos del equipo para resolverlo.",
        "Tienes toda la razón y lo siento mucho 😔 Ya notifiqué al equipo — te contactarán pronto.",
    ],
    "consulta_estado_pedido": [
        "Déjame verificar 🔍 ¿Me confirmas el día que programaste la entrega?",
        "Voy a informar al equipo para que te confirmen el estado de tu pedido 📦",
        "Ya informo al equipo para que revisen tu pedido 😊 ¿Tienes algún dato adicional?",
    ],
    "fuera_de_tema": [
        "Por ahora solo manejo ventas de frutas y verduras 🥦 Catálogo: https://wa.me/c/573124929496",
        "Eso está fuera de lo que puedo ayudarte, pero ya informo al equipo 😊",
        "No tengo esa información, pero ya le aviso a un supervisor 😊 ¿Te ayudo con algún producto?",
    ],
    "confirmar_pedido": [
        "Para confirmar escribe exactamente la palabra *Confirmar* 😊",
        "Solo escribe *Confirmar* para aprobar tu pedido ✅",
    ],
    "rechazar_pedido": [
        "Para cancelar escribe exactamente la palabra *Rechazar* 😊",
        "Solo escribe *Rechazar* para cancelar tu pedido 😊",
    ],
    "datos_entrega": [
        "¡Perfecto! Déjame procesar esos datos 😊",
        "Recibido, procesando tu información de entrega 📦",
    ],
}

FALLBACK = [
    "¡Hola! 😊 Soy Vecinito de CorApp. ¿En qué te puedo ayudar? https://wa.me/c/573124929496",
    "Entiendo 😊 ¿Necesitas hacer un pedido o tienes alguna duda sobre nuestros productos?",
    "¡Aquí estoy! 🛒 Frutas y verduras frescas con domicilio gratis: https://wa.me/c/573124929496",
    "Ya informé al equipo 😊 ¿Puedo ayudarte con algo más?",
    "¡Con mucho gusto! 😊 ¿Qué necesitas hoy de CorApp?",
    "Disculpa, no entendí bien 😊 ¿Me repites qué necesitas?",
]


def respond(intent: str, confidence: float) -> str:
    if confidence < 0.15:
        return random.choice(FALLBACK)
    r = RESPONSES.get(intent)
    return random.choice(r) if r else random.choice(FALLBACK)


# ============================================================
# ANÁLISIS DE FALLOS
# ============================================================

def analyze_failures(messages: list) -> list:
    suggestions, seen = [], set()

    for text in messages:
        if not text or not text.strip():
            continue
        norm = normalize(text)

        # Barrios nuevos no reconocidos
        for pattern in [r'\bbarrio\s+([a-záéíóúñ\s]{3,30})', r'\bsector\s+([a-záéíóúñ\s]{3,30})']:
            for match in re.findall(pattern, norm):
                mc = match.strip()
                if mc and mc not in VALID_LOCATIONS:
                    key = f"loc_{mc}"
                    if key not in seen:
                        seen.add(key)
                        suggestions.append({
                            "type": "new_location",
                            "value": mc,
                            "suggested_locality": None,
                            "original_message": text[:120],
                        })

        # Días problemáticos
        day, day_err = extract_day(text)
        if not day:
            for bad in INVALID_DAYS:
                if re.search(r'\b' + normalize(bad) + r'\b', norm):
                    key = f"day_{bad}"
                    if key not in seen:
                        seen.add(key)
                        suggestions.append({
                            "type": "day_format", "value": bad,
                            "suggestion": f"Usuario escribió '{bad}' en lugar del nombre del día",
                            "original_message": text[:120],
                        })

            dp = r'\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b'
            dm = re.search(dp, norm)
            if dm:
                key = f"date_{dm.group(0)}"
                if key not in seen:
                    seen.add(key)
                    suggestions.append({
                        "type": "day_format",
                        "value": dm.group(0),
                        "suggestion": "Usuario envió fecha completa en lugar del nombre del día",
                        "original_message": text[:120],
                    })

        # Tiene dirección y localidad pero falta día
        has_addr = bool(re.search(ADDR_START, norm))
        loc, _ = extract_locality(text)
        if has_addr and loc and not day:
            key = f"missing_day_{text[:40]}"
            if key not in seen:
                seen.add(key)
                suggestions.append({
                    "type": "missing_day",
                    "value": text[:120],
                    "suggestion": "Usuario envió dirección y localidad pero olvidó el día",
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
        return jsonify({'error': 'Se requiere message'}), 400
    intent, conf = classify(data['message'])
    return jsonify({
        'message': data['message'],
        'intent': intent,
        'confidence': round(conf, 4),
        'use_fallback': conf < 0.2,
    })


@app.route('/extract-delivery', methods=['POST'])
def endpoint_extract():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': True, 'errorMessage': 'Se requiere text', 'info': None}), 400
    return jsonify(extract_delivery(
        data['text'],
        data.get('validLocations'),
        data.get('validDays'),
        data.get('validTimes'),
    ))


@app.route('/respond', methods=['POST'])
def endpoint_respond():
    data = request.get_json() or {}
    message = data.get('message', '')
    try:
        intent, conf = classify(message)
    except Exception:
        intent, conf = 'unknown', 0.0
    return jsonify({
        'response': respond(intent, conf),
        'intent': intent,
        'confidence': round(conf, 4),
        'use_fallback': conf < 0.2,
    })


@app.route('/analyze-failures', methods=['POST'])
def endpoint_analyze():
    data = request.get_json()
    if not data or 'messages' not in data:
        return jsonify({'error': True, 'errorMessage': 'Se requiere messages', 'suggestions': []}), 400
    suggestions = analyze_failures(data['messages'])
    return jsonify({
        'error': False, 'errorMessage': None,
        'analyzed': len(data['messages']),
        'suggestions': suggestions,
    })


@app.route('/retrain', methods=['POST'])
def endpoint_retrain():
    """Reentrena el modelo con nuevos datos."""
    try:
        if os.path.exists('model/intent_classifier.pkl'):
            os.remove('model/intent_classifier.pkl')
        global model
        model = None
        train_model()
        return jsonify({'success': True, 'message': 'Modelo reentrenado correctamente'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': '5.0.0'})


# v5.0.0 — modelo completo con fuzzy search, aliases colombianos,
# respuestas empáticas y análisis de fallos
if __name__ == '__main__':
    app.run(debug=True, port=5000)