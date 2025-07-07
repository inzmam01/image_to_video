import cv2
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict, Any


class VideoEffects:
    @staticmethod
    def zoom_effect(image: np.ndarray, duration: float, fps: int, 
                    zoom_type: str = "in", zoom_factor: float = 0.2) -> List[np.ndarray]:
        height, width = image.shape[:2]
        total_frames = int(duration * fps)
        frames = []
        
        for i in tqdm(range(total_frames), desc=f"Creating {zoom_type} zoom"):
            progress = i / total_frames
            if zoom_type == "in":
                scale = 1 + (zoom_factor * progress)
            else:
                scale = 1 + (zoom_factor * (1 - progress))
                
            M = cv2.getRotationMatrix2D((width/2, height/2), 0, scale)
            frame = cv2.warpAffine(image, M, (width, height), 
                                  borderMode=cv2.BORDER_REPLICATE)
            frames.append(frame)
            
        return frames

    @staticmethod
    def pan_effect(image: np.ndarray, duration: float, fps: int,
                  direction: str = "left", intensity: float = 0.3) -> List[np.ndarray]:
        height, width = image.shape[:2]
        total_frames = int(duration * fps)
        frames = []
        
        # Create a larger canvas to pan over
        canvas = cv2.copyMakeBorder(image, 
                                    int(height * intensity), 
                                    int(height * intensity), 
                                    int(width * intensity), 
                                    int(width * intensity), 
                                    cv2.BORDER_REPLICATE)
        
        canvas_height, canvas_width = canvas.shape[:2]
        
        for i in tqdm(range(total_frames), desc=f"Creating {direction} pan"):
            progress = i / total_frames
            
            if direction == "left":
                x_offset = int(progress * width * intensity * 2)
                y_offset = int(height * intensity)
            elif direction == "right":
                x_offset = int(canvas_width - width - progress * width * intensity * 2)
                y_offset = int(height * intensity)
            elif direction == "up":
                x_offset = int(width * intensity)
                y_offset = int(progress * height * intensity * 2)
            elif direction == "down":
                x_offset = int(width * intensity)
                y_offset = int(canvas_height - height - progress * height * intensity * 2)
            else:
                x_offset = int(width * intensity)
                y_offset = int(height * intensity)
                
            frame = canvas[y_offset:y_offset+height, x_offset:x_offset+width]
            frames.append(frame)
            
        return frames

    @staticmethod
    def rotate_effect(image: np.ndarray, duration: float, fps: int,
                     rotation_direction: str = "cw", rotations: float = 1) -> List[np.ndarray]:
        height, width = image.shape[:2]
        total_frames = int(duration * fps)
        frames = []
        
        for i in tqdm(range(total_frames), desc=f"Creating {rotation_direction} rotation"):
            progress = i / total_frames
            angle = progress * 360 * rotations
            if rotation_direction == "ccw":
                angle = -angle
                
            M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            frame = cv2.warpAffine(image, M, (width, height), 
                                  borderMode=cv2.BORDER_REPLICATE)
            frames.append(frame)
            
        return frames

    @staticmethod
    def fade_effect(image: np.ndarray, duration: float, fps: int,
                   fade_type: str = "in") -> List[np.ndarray]:
        total_frames = int(duration * fps)
        frames = []
        
        for i in tqdm(range(total_frames), desc=f"Creating fade {fade_type}"):
            progress = i / total_frames
            if fade_type == "in":
                alpha = progress
            else:
                alpha = 1 - progress
                
            frame = image.copy()
            frame = (frame * alpha).astype(np.uint8)
            frames.append(frame)
            
        return frames

    @staticmethod
    def color_shift_effect(image: np.ndarray, duration: float, fps: int,
                          shift_type: str = "warm") -> List[np.ndarray]:
        total_frames = int(duration * fps)
        frames = []
        
        for i in tqdm(range(total_frames), desc=f"Creating {shift_type} color shift"):
            progress = i / total_frames
            frame = image.copy().astype(np.float32)
            
            if shift_type == "warm":
                # Increase red and decrease blue
                frame[:, :, 2] *= (1 + progress * 0.5)  # Red
                frame[:, :, 0] *= (1 - progress * 0.3)  # Blue
            elif shift_type == "cool":
                # Increase blue and decrease red
                frame[:, :, 0] *= (1 + progress * 0.5)  # Blue
                frame[:, :, 2] *= (1 - progress * 0.3)  # Red
            elif shift_type == "vintage":
                # Sepia tone effect
                frame[:, :, 0] = frame[:, :, 0] * (0.4 + progress * 0.3)  # Blue
                frame[:, :, 1] = frame[:, :, 1] * (0.7 + progress * 0.1)  # Green
                frame[:, :, 2] = frame[:, :, 2] * (0.9 + progress * 0.5)  # Red
                
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frames.append(frame)
            
        return frames

    @staticmethod
    def apply_effect(image: np.ndarray, effect_config: Dict[str, Any], 
                    duration: float, fps: int) -> List[np.ndarray]:
        effect_type = effect_config["type"]
        params = effect_config.get("params", {})
        
        if effect_type == "zoom":
            return VideoEffects.zoom_effect(image, duration, fps, **params)
        elif effect_type == "pan":
            return VideoEffects.pan_effect(image, duration, fps, **params)
        elif effect_type == "rotate":
            return VideoEffects.rotate_effect(image, duration, fps, **params)
        elif effect_type == "fade":
            return VideoEffects.fade_effect(image, duration, fps, **params)
        elif effect_type == "color_shift":
            return VideoEffects.color_shift_effect(image, duration, fps, **params)
        else:
            # Default to static if effect not recognized
            return [image.copy()] * int(duration * fps)