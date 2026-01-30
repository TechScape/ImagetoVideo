"""
Replicate API Client for Video Generation
Handles text-to-video and image-to-video generation using Replicate's cloud API
"""

import os
import logging
import replicate
from pathlib import Path
import requests
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReplicateAPIClient:
    """Client for Replicate API video generation"""
    
    def __init__(self, api_token=None):
        """
        Initialize Replicate API client
        
        Args:
            api_token: Replicate API token (if None, reads from env)
        """
        self.api_token = api_token or os.getenv('REPLICATE_API_TOKEN')
        
        if self.api_token:
            os.environ['REPLICATE_API_TOKEN'] = self.api_token
            logger.info("Replicate API client initialized")
        else:
            logger.warning("No Replicate API token found - API mode disabled")
    
    def is_available(self):
        """Check if API is available"""
        return bool(self.api_token)
    
    def generate_text_to_video(
        self,
        prompt,
        negative_prompt="",
        num_frames=120,
        height=720,
        width=1280,
        fps=24,
        output_path="output.mp4"
    ):
        """
        Generate video from text using Replicate API
        
        Args:
            prompt: Text description
            negative_prompt: What to avoid
            num_frames: Number of frames
            height: Video height
            width: Video width
            fps: Frames per second
            output_path: Where to save
            
        Returns:
            Path to generated video
        """
        if not self.is_available():
            raise Exception("Replicate API token not configured")
        
        logger.info(f"Generating text-to-video via API: {prompt}")
        
        try:
            # Using Zeroscope v2 XL for text-to-video
            output = replicate.run(
                "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351",
                input={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_frames": min(num_frames, 120),  # API limit
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "num_inference_steps": 50,
                    "guidance_scale": 17.5
                }
            )
            
            # Download the video
            if output:
                video_url = output if isinstance(output, str) else output[0]
                self._download_video(video_url, output_path)
                logger.info(f"Video saved to: {output_path}")
                return output_path
            else:
                raise Exception("No output from API")
                
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            raise
    
    def generate_image_to_video(
        self,
        image_path,
        prompt="",
        num_frames=120,
        height=720,
        width=1280,
        fps=24,
        output_path="output.mp4"
    ):
        """
        Generate video from image using Replicate API
        
        Args:
            image_path: Path to input image
            prompt: Optional motion description
            num_frames: Number of frames
            height: Video height
            width: Video width
            fps: Frames per second
            output_path: Where to save
            
        Returns:
            Path to generated video
        """
        if not self.is_available():
            raise Exception("Replicate API token not configured")
        
        logger.info(f"Generating image-to-video via API from: {image_path}")
        
        try:
            # Open and prepare image
            with open(image_path, 'rb') as f:
                image_file = f
                
                # Using Stable Video Diffusion for image-to-video
                output = replicate.run(
                    "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
                    input={
                        "input_image": image_file,
                        "cond_aug": 0.02,
                        "decoding_t": 7,
                        "video_length": "14_frames_with_svd",  # or "25_frames_with_svd_xt"
                        "sizing_strategy": "maintain_aspect_ratio",
                        "motion_bucket_id": 127,
                        "frames_per_second": fps
                    }
                )
            
            # Download the video
            if output:
                video_url = output if isinstance(output, str) else output[0]
                self._download_video(video_url, output_path)
                logger.info(f"Video saved to: {output_path}")
                return output_path
            else:
                raise Exception("No output from API")
                
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            raise
    
    def _download_video(self, url, output_path):
        """Download video from URL to local path"""
        try:
            logger.info(f"Downloading video from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write video file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Video downloaded successfully")
            
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise
