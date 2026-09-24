import cv2
import numpy as np
import requests
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

# Cargar el diccionario ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

URL_BACKEND = "http://10.32.0.13:5000/api/actualizar_fisico"

# --- 🧠 MEMORIA DEL EDGE (OPTIMIZACIÓN) ---
estado_anterior = {
    "id_visto": None,
    "unidades": None
}
ultimo_tiempo_envio = 0
COOLDOWN = 2.0 # Segundos mínimos entre envíos para evitar falsas alarmas por "parpadeos" visuales

@app.route('/upload', methods=['POST'])
def process_image():
    global estado_anterior, ultimo_tiempo_envio

    if not request.data:
        return jsonify({"error": "Sin datos"}), 400

    np_arr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Imagen corrupta"}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detección de marcadores
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected = detector.detectMarkers(gray)

    tiempo_actual = time.time()
    conteo_actual = 0
    id_actual = None

    # 1. Determinar qué estamos viendo en este milisegundo
    if ids is not None and len(ids) > 0:
        conteo_actual = len(ids)
        id_actual = int(ids.flatten()[0])
    else:
        # LÓGICA DE STOCK AGOTADO: Si no ve marcadores, asumimos 0 unidades del ÚLTIMO producto que vio
        conteo_actual = 0
        id_actual = estado_anterior["id_visto"]

    # 2. Filtrado Inteligente (Edge Computing)
    # Solo procedemos si tenemos un ID válido y hubo un cambio real en la cantidad o en el producto
    hubo_cambio = (id_actual != estado_anterior["id_visto"]) or (conteo_actual != estado_anterior["unidades"])

    if id_actual is not None and hubo_cambio:
        # Verificamos que haya pasado el cooldown para confirmar que no es un fallo visual rápido
        if (tiempo_actual - ultimo_tiempo_envio) > COOLDOWN:
            print(f"\n🔄 CAMBIO DETECTADO: ArUco {id_actual} | Unidades en físico: {conteo_actual}")
            
            payload = {
                "zona_id": id_actual,
                "unidades_fisicas": conteo_actual
            }
            
            try:
                # Disparamos la alerta al servidor central (Node.js)
                respuesta = requests.post(URL_BACKEND, json=payload, timeout=5)
                print(f"📡 Backend sincronizado. Código HTTP: {respuesta.status_code}")
                
                # Actualizamos la memoria del Edge solo si el envío fue exitoso
                estado_anterior["id_visto"] = id_actual
                estado_anterior["unidades"] = conteo_actual
                ultimo_tiempo_envio = tiempo_actual
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Error al conectar con Node.js: {e}")
    else:
        # Silencio absoluto. Si no hay cambios, el Edge no consume red y la terminal queda limpia.
        pass

    return jsonify({"status": "procesado", "unidades_fisicas": conteo_actual}), 200

if __name__ == '__main__':
    print("🚀 Nodo Edge Vision Optimizado iniciado.")
    print("📡 Esperando imágenes... Solo enviaré tráfico a la red cuando el inventario cambie.")
    app.run(host='0.0.0.0', port=5000)
