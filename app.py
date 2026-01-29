"""
Wan 2.0 Video Generation Web Application
Flask backend for text-to-video and image-to-video generation
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import logging

from model_handler import Wan2ModelHandler

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['OUTPUT_FOLDER'] = os.getenv('OUTPUT_FOLDER', 'outputs')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_UPLOAD_SIZE', 10485760))  # 10MB default

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Create directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)
Path(app.config['UPLOAD_FOLDER']).joinpath('.gitkeep').touch()
Path(app.config['OUTPUT_FOLDER']).joinpath('.gitkeep').touch()

# Initialize model handler
logger.info("Initializing Wan 2.0 model...")
model_handler = Wan2ModelHandler(
    model_name=os.getenv('MODEL_NAME', 'alibaba-pai/wan-2.0-5b'),
    device=os.getenv('DEVICE', None),
    use_fp16=os.getenv('USE_FP16', 'True').lower() == 'true'
)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('static', 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_handler.pipeline is not None,
        'device': model_handler.device
    })


@app.route('/api/text-to-video', methods=['POST'])
def text_to_video():
    """
    Generate video from text prompt
    
    Expected JSON:
    {
        "prompt": "A cat playing piano",
        "negative_prompt": "blurry, low quality",
        "duration": 5,
        "fps": 24,
        "resolution": 720
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400
        
        prompt = data['prompt']
        negative_prompt = data.get('negative_prompt', '')
        duration = int(data.get('duration', 5))
        fps = int(data.get('fps', 24))
        resolution = int(data.get('resolution', 720))
        
        # Calculate parameters
        num_frames = duration * fps
        height = resolution
        width = int(resolution * 16 / 9)  # 16:9 aspect ratio
        
        # Generate unique filename
        video_id = str(uuid.uuid4())
        output_filename = f"text2video_{video_id}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        logger.info(f"Generating text-to-video: {prompt}")
        
        # Generate video
        result_path = model_handler.generate_text_to_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            output_path=output_path
        )
        
        return jsonify({
            'success': True,
            'video_id': video_id,
            'filename': output_filename,
            'download_url': f'/api/download/{output_filename}'
        })
        
    except Exception as e:
        logger.error(f"Error in text-to-video: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/image-to-video', methods=['POST'])
def image_to_video():
    """
    Generate video from image
    
    Expected form data:
    - image: file
    - prompt: optional text prompt
    - duration: video duration in seconds
    - fps: frames per second
    - resolution: video height
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, webp'}), 400
        
        # Save uploaded image
        filename = secure_filename(file.filename)
        upload_id = str(uuid.uuid4())
        upload_filename = f"{upload_id}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
        file.save(upload_path)
        
        # Get parameters
        prompt = request.form.get('prompt', '')
        duration = int(request.form.get('duration', 5))
        fps = int(request.form.get('fps', 24))
        resolution = int(request.form.get('resolution', 720))
        
        # Calculate parameters
        num_frames = duration * fps
        height = resolution
        width = int(resolution * 16 / 9)
        
        # Generate unique filename for output
        video_id = str(uuid.uuid4())
        output_filename = f"img2video_{video_id}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        logger.info(f"Generating image-to-video from: {upload_filename}")
        
        # Generate video
        result_path = model_handler.generate_image_to_video(
            image_path=upload_path,
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            output_path=output_path
        )
        
        # Clean up uploaded image
        try:
            os.remove(upload_path)
        except:
            pass
        
        return jsonify({
            'success': True,
            'video_id': video_id,
            'filename': output_filename,
            'download_url': f'/api/download/{output_filename}'
        })
        
    except Exception as e:
        logger.error(f"Error in image-to-video: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_video(filename):
    """Download generated video"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            file_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview/<filename>', methods=['GET'])
def preview_video(filename):
    """Stream video for preview"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, mimetype='video/mp4')
        
    except Exception as e:
        logger.error(f"Error previewing file: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting Wan 2.0 Video Generation Server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
