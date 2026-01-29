# Wan 2.0 Video Generation Project

Generate stunning AI-powered videos from text prompts or images using the Wan 2.0 model by Alibaba.

![Wan 2.0](https://img.shields.io/badge/Wan-2.0-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=for-the-badge&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)

## ✨ Features

- 🎬 **Text-to-Video Generation** - Create videos from descriptive text prompts
- 🖼️ **Image-to-Video Animation** - Bring static images to life with AI
- 🌐 **Modern Web Interface** - Beautiful, responsive UI with glassmorphism design
- ⚡ **Real-time Processing** - Fast video generation with progress tracking
- 📱 **Mobile Responsive** - Works seamlessly on all devices
- 🎨 **Customizable Settings** - Control resolution, FPS, and duration

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd d:\Freelancing\ImagetoVideo
```

2. **Create a virtual environment:**
```bash
python -m venv venv
```

3. **Activate the virtual environment:**

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Set up environment variables:**
```bash
copy .env.example .env
```

Edit `.env` file to configure your settings (optional).

### Running the Application

1. **Start the Flask server:**
```bash
python app.py
```

2. **Open your browser and navigate to:**
```
http://localhost:5000
```

3. **Start creating videos!**

## 📖 Usage

### Text-to-Video

1. Select the **Text to Video** tab
2. Enter a descriptive prompt (e.g., "A majestic dragon flying over a medieval castle at sunset")
3. Optionally add a negative prompt to avoid unwanted elements
4. Configure settings (duration, FPS, resolution)
5. Click **Generate Video**
6. Wait for processing and download your video!

### Image-to-Video

1. Select the **Image to Video** tab
2. Upload an image (PNG, JPG, JPEG, or WEBP)
3. Optionally add a motion prompt (e.g., "gentle camera zoom")
4. Configure settings
5. Click **Generate Video**
6. Download your animated video!

## 🎨 Example Prompts

### Text-to-Video Examples:
- "A serene lake at sunrise with mist rising from the water, cinematic 4k"
- "Futuristic city with flying cars and neon lights, cyberpunk style"
- "Ocean waves crashing on a rocky shore, slow motion"
- "Northern lights dancing in the night sky over snowy mountains"

### Image-to-Video Motion Prompts:
- "Slow zoom in with gentle camera movement"
- "Pan from left to right smoothly"
- "Character walking forward naturally"
- "Leaves rustling in the wind"

## ⚙️ Configuration

Edit the `.env` file to customize:

```env
# Model Configuration
MODEL_NAME=alibaba-pai/wan-2.0-5b
DEVICE=cuda  # Options: cuda, cpu, mps
USE_FP16=True

# Generation Settings
DEFAULT_RESOLUTION=720
DEFAULT_FPS=24
DEFAULT_DURATION=5
MAX_VIDEO_LENGTH=10

# Server Settings
HOST=0.0.0.0
PORT=5000
```

## 🏗️ Project Structure

```
ImagetoVideo/
├── app.py                 # Flask backend server
├── model_handler.py       # Wan 2.0 model integration
├── requirements.txt       # Python dependencies
├── .env.example          # Environment configuration template
├── .gitignore            # Git ignore rules
├── static/               # Frontend files
│   ├── index.html        # Main web interface
│   ├── css/
│   │   └── style.css     # Styling with glassmorphism
│   └── js/
│       └── app.js        # Client-side JavaScript
├── uploads/              # Temporary image uploads
└── outputs/              # Generated videos
```

## 🔧 API Endpoints

### Health Check
```
GET /api/health
```

### Text-to-Video
```
POST /api/text-to-video
Content-Type: application/json

{
  "prompt": "Your prompt here",
  "negative_prompt": "Optional",
  "duration": 5,
  "fps": 24,
  "resolution": 720
}
```

### Image-to-Video
```
POST /api/image-to-video
Content-Type: multipart/form-data

image: <file>
prompt: "Optional motion description"
duration: 5
fps: 24
resolution: 720
```

### Download Video
```
GET /api/download/<filename>
```

## 🎯 System Requirements

### Minimum Requirements:
- CPU: 4+ cores
- RAM: 8GB
- Storage: 10GB free space
- GPU: Not required (CPU mode available)

### Recommended Requirements:
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 20GB+ free space
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)

## 🐛 Troubleshooting

### Model Loading Issues
If the model fails to load, the application will run in **mock mode** for demonstration purposes. To use the actual Wan 2.0 model:

1. Ensure you have sufficient GPU memory
2. Try setting `USE_FP16=True` in `.env` to reduce memory usage
3. Consider using a smaller model variant
4. Check your internet connection for model downloads

### CUDA Out of Memory
- Reduce resolution in settings
- Decrease video duration
- Set `USE_FP16=True`
- Close other GPU-intensive applications

### Slow Generation
- Use GPU instead of CPU (set `DEVICE=cuda`)
- Reduce resolution and FPS
- Enable half-precision (`USE_FP16=True`)

## 📝 Notes

- First run will download the Wan 2.0 model (~10GB), which may take time
- Video generation time depends on duration, resolution, and hardware
- Generated videos are saved in the `outputs/` directory
- Uploaded images are temporarily stored and automatically cleaned up

## 🌟 Technologies Used

- **Backend**: Flask, Python
- **AI/ML**: PyTorch, Hugging Face Diffusers, Wan 2.0
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Video Processing**: OpenCV, imageio

## 📄 License

This project is for educational and demonstration purposes. Please refer to the Wan 2.0 model license for commercial usage restrictions.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## 🔗 Resources

- [Wan 2.0 Model on Hugging Face](https://huggingface.co/alibaba-pai/wan-2.0-5b)
- [Wan AI Official Website](https://openwanai.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Made with ❤️ using Wan 2.0 AI**
