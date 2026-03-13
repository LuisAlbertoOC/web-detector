from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from detector import PlantDiseaseDetector
import os
import uuid
from pathlib import Path
import logging
import datetime
from db import get_db_connection

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template("index.html")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = os.getenv("MODEL_VERSION", "plantdoc_300_epochs3")

detector = PlantDiseaseDetector(
    model_path=BASE_DIR,
    model_version=MODEL_VERSION
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No se proporcionó imagen'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'Nombre de archivo vacío'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': 'Tipo de archivo no permitido',
            'allowed_types': list(ALLOWED_EXTENSIONS)
        }), 400

    filepath = None
    
    try:
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4()}.{file_ext}"
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)
        logger.info(f"Imagen guardada temporalmente en: {filepath}")
        
        results = detector.detect(str(filepath))
        
        response = {
            'status': results.get('status', 'success'),
            'data': {
                'detections': results.get('detections', []),
                'image_info': {
                    'original_size': results.get('image_size'),
                    'processed_size': [640, 640]
                },
                'model_info': results.get('model_info', {})
            },
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error en /predict: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Error procesando imagen',
            'details': str(e)
        }), 500

    finally:
        if filepath and filepath.exists():
            filepath.unlink()


@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'status': 'success',
        'model': {
            'name': 'YOLOv7',
            'version': MODEL_VERSION,
            'classes': detector.class_names,
            'device': detector.device
        }
    })


@app.route('/disease-info', methods=['GET'])
def disease_info():
    class_name = request.args.get('class')
    
    if not class_name:
        return jsonify({'status': 'error', 'message': 'Parámetro class requerido'}), 400
    
    connection = None
    cursor = None
    
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'status': 'error', 'message': 'Error de conexión a la base de datos'}), 500
        
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM plantas WHERE clases = %s"
        cursor.execute(query, (class_name,))
        result = cursor.fetchone()
        
        if result:
            filename = os.path.basename(result['foto']) if result.get('foto') else None
            
            return jsonify({
                'status': 'success',
                'data': {
                    'clases': result['clases'],
                    'clases_e': result['clases_e'],
                    'descripcion': result['descripcion'],
                    'solucion': result['solucion'],
                    'foto': filename
                }
            })
        else:
            return jsonify({'status': 'error', 'message': 'Enfermedad no encontrada'}), 404
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error de base de datos: {str(e)}'}), 500
    
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
