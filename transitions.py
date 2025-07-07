import cv2
import random
import numpy as np
from tqdm import tqdm
from skimage.transform import resize


class VideoTransitions:
    @staticmethod
    def crossfade(img1: np.ndarray, img2: np.ndarray, 
                 duration: float, fps: int) -> list[np.ndarray]:
        """Crossfade between two images"""
        total_frames = int(duration * fps)
        frames = []
        
        # Ensure images are same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        for i in tqdm(range(total_frames), desc="Crossfade transition"):
            alpha = i / total_frames
            frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
            frames.append(frame)
            
        return frames

    @staticmethod
    def slide(img1: np.ndarray, img2: np.ndarray, 
             duration: float, fps: int, direction: str = "left") -> list[np.ndarray]:
        """Slide transition between two images"""
        height, width = img1.shape[:2]
        total_frames = int(duration * fps)
        frames = []
        
        # Ensure images are same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (width, height))
        
        for i in tqdm(range(total_frames), desc=f"Slide {direction} transition"):
            progress = i / total_frames
            offset = int(progress * width)
            
            frame = img1.copy()
            
            if direction == "left":
                frame[:, :width - offset] = img1[:, offset:]
                frame[:, width - offset:] = img2[:, :offset]
            elif direction == "right":
                frame[:, offset:] = img1[:, :width - offset]
                frame[:, :offset] = img2[:, width - offset:]
            elif direction == "up":
                offset = int(progress * height)
                frame[:height - offset, :] = img1[offset:, :]
                frame[height - offset:, :] = img2[:offset, :]
            elif direction == "down":
                offset = int(progress * height)
                frame[offset:, :] = img1[:height - offset, :]
                frame[:offset, :] = img2[height - offset:, :]
                
            frames.append(frame)
            
        return frames

    @staticmethod
    def zoom_transition(img1: np.ndarray, img2: np.ndarray, 
                       duration: float, fps: int, zoom_type: str = "in") -> list[np.ndarray]:
        """Zoom transition between two images"""
        height, width = img1.shape[:2]
        total_frames = int(duration * fps)
        frames = []
        
        # Ensure images are same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (width, height))
        
        for i in tqdm(range(total_frames), desc=f"Zoom {zoom_type} transition"):
            progress = i / total_frames
            
            if zoom_type == "in":
                # Zoom out img1, zoom in img2
                scale1 = 1.0 + progress * 0.5
                scale2 = 1.5 - progress * 0.5
                
                M1 = cv2.getRotationMatrix2D((width/2, height/2), 0, scale1)
                M2 = cv2.getRotationMatrix2D((width/2, height/2), 0, scale2)
                
                frame1 = cv2.warpAffine(img1, M1, (width, height), 
                                       borderMode=cv2.BORDER_REPLICATE)
                frame2 = cv2.warpAffine(img2, M2, (width, height), 
                                       borderMode=cv2.BORDER_REPLICATE)
                
                # Blend
                frame = cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
            else:
                # Zoom in img1, zoom out img2
                scale1 = 1.5 - progress * 0.5
                scale2 = 1.0 + progress * 0.5
                
                M1 = cv2.getRotationMatrix2D((width/2, height/2), 0, scale1)
                M2 = cv2.getRotationMatrix2D((width/2, height/2), 0, scale2)
                
                frame1 = cv2.warpAffine(img1, M1, (width, height), 
                                       borderMode=cv2.BORDER_REPLICATE)
                frame2 = cv2.warpAffine(img2, M2, (width, height), 
                                       borderMode=cv2.BORDER_REPLICATE)
                
                # Blend
                frame = cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
                
            frames.append(frame)
            
        return frames

    @staticmethod
    def warp_transition(img1: np.ndarray, img2: np.ndarray, 
                       duration: float, fps: int) -> list[np.ndarray]:
        """Creative warp transition between two images"""
        height, width = img1.shape[:2]
        total_frames = int(duration * fps)
        frames = []
        
        # Ensure images are same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (width, height))
        
        for i in tqdm(range(total_frames), desc="Warp transition"):
            progress = i / total_frames
            frame = np.zeros_like(img1)
            
            # Create a wave effect
            for y in range(height):
                offset = int(20 * np.sin(2 * np.pi * (y/50 + progress * 2)))
                x_offset = int(progress * width + offset)
                
                if 0 <= x_offset < width:
                    frame[y, :width - x_offset] = img1[y, x_offset:]
                    frame[y, width - x_offset:] = img2[y, :x_offset]
                elif x_offset < 0:
                    frame[y] = img1[y]
                else:
                    frame[y] = img2[y]
                    
            frames.append(frame)
            
        return frames

    @staticmethod
    def apply_transition(img1: np.ndarray, img2: np.ndarray, 
                        transition_config: dict[str, any], 
                        duration: float, fps: int) -> list[np.ndarray]:
        transition_type = transition_config["type"]
        params = transition_config.get("params", {})
        
        if transition_type == "crossfade":
            return VideoTransitions.crossfade(img1, img2, duration, fps, **params)
        elif transition_type == "slide":
            return VideoTransitions.slide(img1, img2, duration, fps, **params)
        elif transition_type == "zoom":
            return VideoTransitions.zoom_transition(img1, img2, duration, fps, **params)
        elif transition_type == "warp":
            return VideoTransitions.warp_transition(img1, img2, duration, fps, **params)
        else:
            # Default to crossfade
            return VideoTransitions.crossfade(img1, img2, duration, fps)
        
    @staticmethod
    def get_random_transition_params() -> dict:
        """Generate random parameters for transitions"""
        return {
            "slide": {
                "direction": random.choice(["left", "right", "up", "down"])
            },
            "zoom": {
                "zoom_type": random.choice(["in", "out"])
            }
        }

    @staticmethod
    def get_random_transition_config() -> dict:
        """Generate a random transition configuration"""
        transition_type = random.choice(["crossfade", "slide", "zoom", "warp"])
        return {
            "type": transition_type,
            "params": VideoTransitions.get_random_transition_params().get(transition_type, {})
        }

    @staticmethod
    def apply_random_transition(img1: np.ndarray, img2: np.ndarray, 
                               duration: float, fps: int) -> list[np.ndarray]:
        """Apply a randomly selected transition between two images"""
        config = VideoTransitions.get_random_transition_config()
        return VideoTransitions.apply_transition(img1, img2, config, duration, fps)
