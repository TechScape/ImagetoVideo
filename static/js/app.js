// API Base URL
const API_BASE = window.location.origin;

// DOM Elements
const tabs = document.querySelectorAll('.tab');
const panels = document.querySelectorAll('.panel');
const textForm = document.getElementById('text-form');
const imageForm = document.getElementById('image-form');
const textPrompt = document.getElementById('text-prompt');
const textCharCount = document.getElementById('text-char-count');
const uploadArea = document.getElementById('upload-area');
const imageInput = document.getElementById('image-input');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const removeImageBtn = document.getElementById('remove-image');
const imageSubmitBtn = document.getElementById('image-submit');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingMessage = document.getElementById('loading-message');
const resultModal = document.getElementById('result-modal');
const modalClose = document.getElementById('modal-close');
const resultVideo = document.getElementById('result-video');
const videoSource = document.getElementById('video-source');
const downloadLink = document.getElementById('download-link');
const createAnotherBtn = document.getElementById('create-another');

// State
let selectedFile = null;

// Tab Switching
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const targetTab = tab.dataset.tab;
        
        // Update active tab
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        // Update active panel
        panels.forEach(panel => {
            panel.classList.remove('active');
            if (panel.id === targetTab) {
                panel.classList.add('active');
            }
        });
    });
});

// Character Counter
textPrompt.addEventListener('input', () => {
    const count = textPrompt.value.length;
    textCharCount.textContent = `${count} / 500`;
});

// Image Upload - Click
uploadArea.addEventListener('click', () => {
    imageInput.click();
});

// Image Upload - File Selection
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleImageFile(file);
    }
});

// Image Upload - Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImageFile(file);
    }
});

// Handle Image File
function handleImageFile(file) {
    // Validate file size (10MB)
    if (file.size > 10485760) {
        alert('File size must be less than 10MB');
        return;
    }
    
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (PNG, JPG, JPEG, WEBP)');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadArea.querySelector('.upload-content').style.display = 'none';
        previewContainer.style.display = 'block';
        imageSubmitBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Remove Image
removeImageBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedFile = null;
    imageInput.value = '';
    imagePreview.src = '';
    uploadArea.querySelector('.upload-content').style.display = 'block';
    previewContainer.style.display = 'none';
    imageSubmitBtn.disabled = true;
});

// Text to Video Form Submission
textForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        prompt: document.getElementById('text-prompt').value,
        negative_prompt: document.getElementById('text-negative').value,
        duration: parseInt(document.getElementById('text-duration').value),
        fps: parseInt(document.getElementById('text-fps').value),
        resolution: parseInt(document.getElementById('text-resolution').value)
    };
    
    await generateVideo('text-to-video', formData);
});

// Image to Video Form Submission
imageForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('prompt', document.getElementById('image-prompt').value);
    formData.append('duration', document.getElementById('image-duration').value);
    formData.append('fps', document.getElementById('image-fps').value);
    formData.append('resolution', document.getElementById('image-resolution').value);
    
    await generateVideo('image-to-video', formData);
});

// Generate Video
async function generateVideo(endpoint, data) {
    // Show loading overlay
    loadingOverlay.style.display = 'flex';
    loadingMessage.textContent = 'Initializing AI model...';
    
    try {
        const isFormData = data instanceof FormData;
        
        const response = await fetch(`${API_BASE}/api/${endpoint}`, {
            method: 'POST',
            headers: isFormData ? {} : { 'Content-Type': 'application/json' },
            body: isFormData ? data : JSON.stringify(data)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to generate video');
        }
        
        loadingMessage.textContent = 'Generating your video...';
        
        const result = await response.json();
        
        // Hide loading
        loadingOverlay.style.display = 'none';
        
        // Show result
        showResult(result);
        
    } catch (error) {
        console.error('Error:', error);
        loadingOverlay.style.display = 'none';
        alert(`Error: ${error.message}`);
    }
}

// Show Result Modal
function showResult(result) {
    const videoUrl = `${API_BASE}${result.download_url}`;
    
    videoSource.src = videoUrl;
    resultVideo.load();
    downloadLink.href = videoUrl;
    downloadLink.download = result.filename;
    
    resultModal.style.display = 'flex';
}

// Close Modal
modalClose.addEventListener('click', () => {
    resultModal.style.display = 'none';
    resultVideo.pause();
});

// Create Another Video
createAnotherBtn.addEventListener('click', () => {
    resultModal.style.display = 'none';
    resultVideo.pause();
    
    // Reset forms
    textForm.reset();
    imageForm.reset();
    textCharCount.textContent = '0 / 500';
    
    if (selectedFile) {
        removeImageBtn.click();
    }
});

// Close modal on outside click
resultModal.addEventListener('click', (e) => {
    if (e.target === resultModal) {
        modalClose.click();
    }
});

// Health Check on Load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const health = await response.json();
        console.log('Server health:', health);
    } catch (error) {
        console.error('Server health check failed:', error);
    }
});
