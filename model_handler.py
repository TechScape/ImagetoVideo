"""
Wan 2.0 Model Handler
Handles video generation using the Wan 2.0 AI model
"""

import os
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Wan2ModelHandler:
    """Handler for Wan 2.0 video generation model"""
    
    def __init__(self, model_name="alibaba-pai/wan-2.0-5b", device=None, use_fp16=True):
        """
        Initialize the Wan 2.0 model
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cuda/cpu/mps)
            use_fp16: Whether to use half precision
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing Wan 2.0 model on {self.device}")
        
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the Wan 2.0 model pipeline"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Check if using lightweight alternative
            if "text2video" in self.model_name.lower() or "modelscope" in self.model_name.lower():
                logger.info("Loading lightweight ModelScope Text2Video model...")
                from diffusers import DiffusionPipeline
                
                self.pipeline = DiffusionPipeline.from_pretrained(
                    "damo-vilab/text-to-video-ms-1.7b",
                    torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                    variant="fp16" if self.device == "cuda" else None
                )
            else:
                # Load the standard Wan pipeline
                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.use_fp16 and self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable optimizations
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                self.pipeline.enable_vae_slicing()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.warning("Falling back to mock mode for demonstration")
            self.pipeline = None
    
    def generate_text_to_video(
        self,
        prompt,
        negative_prompt="",
        num_frames=120,
        height=720,
        width=1280,
        fps=24,
        guidance_scale=7.5,
        num_inference_steps=50,
        output_path="output.mp4"
    ):
        """
        Generate video from text prompt
        
        Args:
            prompt: Text description of the video
            negative_prompt: What to avoid in generation
            num_frames: Number of frames to generate
            height: Video height
            width: Video width
            fps: Frames per second
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
            output_path: Where to save the video
            
        Returns:
            Path to generated video
        """
        logger.info(f"Generating text-to-video: {prompt}")
        
        try:
            if self.pipeline is None:
                return self._generate_mock_video(output_path, "text")
            
            # Generate video
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            
            # Save video
            self._save_video(output.frames[0], output_path, fps)
            
            logger.info(f"Video saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            return self._generate_mock_video(output_path, "text")
    
    def generate_image_to_video(
        self,
        image_path,
        prompt="",
        num_frames=120,
        height=720,
        width=1280,
        fps=24,
        guidance_scale=7.5,
        num_inference_steps=50,
        output_path="output.mp4"
    ):
        """
        Generate video from image
        
        Args:
            image_path: Path to input image
            prompt: Optional text prompt for guidance
            num_frames: Number of frames to generate
            height: Video height
            width: Video width
            fps: Frames per second
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
            output_path: Where to save the video
            
        Returns:
            Path to generated video
        """
        logger.info(f"Generating image-to-video from: {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image = image.resize((width, height))
            
            if self.pipeline is None:
                return self._generate_mock_video(output_path, "image", image)
            
            # Generate video
            output = self.pipeline(
                image=image,
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            
            # Save video
            self._save_video(output.frames[0], output_path, fps)
            
            logger.info(f"Video saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            image = Image.open(image_path).convert("RGB")
            return self._generate_mock_video(output_path, "image", image)
    
    def _save_video(self, frames, output_path, fps=24):
        """Save frames as video file"""
        import imageio
        
        # Convert frames to numpy arrays
        if isinstance(frames[0], Image.Image):
            frames = [np.array(frame) for frame in frames]
        
        # Save as video
        imageio.mimsave(output_path, frames, fps=fps)
    
    def _generate_mock_video(self, output_path, mode="text", image=None):
        """Generate a mock video for demonstration when model is not available"""
        import imageio
        
        logger.info("Generating mock video for demonstration")
        
        # Create simple animation
        frames = []
        width, height = 1280, 720
        num_frames = 48  # 2 seconds at 24fps
        
        for i in range(num_frames):
            if mode == "image" and image is not None:
                # Animate the image with a simple zoom effect
                frame = np.array(image.resize((width, height)))
                scale = 1.0 + (i / num_frames) * 0.1  # Slight zoom
                # Simple brightness variation
                brightness = 1.0 + 0.1 * np.sin(i / num_frames * np.pi)
                frame = np.clip(frame * brightness, 0, 255).astype(np.uint8)
            else:
                # Create gradient animation for text-to-video
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                color_shift = int((i / num_frames) * 255)
                frame[:, :, 0] = (128 + color_shift // 2) % 255  # Red channel
                frame[:, :, 1] = (64 + color_shift) % 255  # Green channel
                frame[:, :, 2] = (192 - color_shift // 2) % 255  # Blue channel
            
            frames.append(frame)
        
        # Save video
        imageio.mimsave(output_path, frames, fps=24)
        logger.info(f"Mock video saved to: {output_path}")
        
        return output_path
