import os
import logging
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import uuid
import threading
from inference_service import InferenceService

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Enable CORS for API access
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize inference service
inference_service = InferenceService()

# Store processing status
processing_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and file.filename and allowed_file(file.filename):
            # Generate unique filename
            file_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            file_extension = filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{file_id}.{file_extension}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            app.logger.info(f"File uploaded successfully: {unique_filename}")
            
            return jsonify({
                'file_id': file_id,
                'filename': unique_filename,
                'original_filename': filename,
                'message': 'File uploaded successfully'
            }), 200
        else:
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
            
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/process', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        
        if not file_id:
            return jsonify({'error': 'No file_id provided'}), 400
        
        # Check if file exists
        uploaded_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(file_id)]
        if not uploaded_files:
            return jsonify({'error': 'File not found'}), 404
        
        input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_files[0])
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}.obj")
        
        # Initialize processing status
        processing_status[file_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting 3D model generation...'
        }
        
        # Start processing in background thread
        def process_in_background():
            try:
                app.logger.info(f"Starting processing for file_id: {file_id}")
                processing_status[file_id]['message'] = 'Loading model...'
                processing_status[file_id]['progress'] = 25
                
                # Process the image
                success = inference_service.process_image(input_filepath, output_filepath)
                
                if success:
                    processing_status[file_id] = {
                        'status': 'completed',
                        'progress': 100,
                        'message': '3D model generated successfully!',
                        'output_file': f"{file_id}.obj"
                    }
                    app.logger.info(f"Processing completed for file_id: {file_id}")
                else:
                    processing_status[file_id] = {
                        'status': 'error',
                        'progress': 0,
                        'message': 'Failed to generate 3D model. Please try again.'
                    }
                    app.logger.error(f"Processing failed for file_id: {file_id}")
                    
            except Exception as e:
                app.logger.error(f"Processing error for file_id {file_id}: {str(e)}")
                processing_status[file_id] = {
                    'status': 'error',
                    'progress': 0,
                    'message': f'Processing failed: {str(e)}'
                }
        
        # Start background processing
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'file_id': file_id,
            'message': 'Processing started',
            'status': 'processing'
        }), 200
        
    except Exception as e:
        app.logger.error(f"Process error: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/status/<file_id>', methods=['GET'])
def get_status(file_id):
    try:
        status = processing_status.get(file_id, {
            'status': 'not_found',
            'progress': 0,
            'message': 'File not found or processing not started'
        })
        return jsonify(status), 200
    except Exception as e:
        app.logger.error(f"Status error: {str(e)}")
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500

@app.route('/api/download/<file_id>', methods=['GET'])
def download_file(file_id):
    try:
        filename = f"{file_id}.obj"
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        app.logger.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/api/model/<file_id>', methods=['GET'])
def get_model(file_id):
    try:
        filename = f"{file_id}.obj"
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='application/octet-stream')
        else:
            return jsonify({'error': 'Model file not found'}), 404
            
    except Exception as e:
        app.logger.error(f"Model retrieval error: {str(e)}")
        return jsonify({'error': f'Model retrieval failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    app.logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
