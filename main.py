import cv2
import argparse
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from effects import VideoEffects
from transitions import VideoTransitions


def parse_args():
    parser = argparse.ArgumentParser(description="Convert image(s) to video with effects and transitions")
    parser.add_argument("input_path", nargs="+", help="Path to input image(s) or directory")
    parser.add_argument("output_path", help="Path to output video")
    parser.add_argument("--duration", type=float, default=5.0, 
                       help="Duration per image in seconds (for single image: total duration)")
    parser.add_argument("--fps", type=int, default=30, 
                       help="Frames per second")
    parser.add_argument("--transition_duration", type=float, default=1.0, 
                       help="Duration of transitions in seconds")
    parser.add_argument("--effect", type=str, default="zoom_in", 
                       help="Effect for images (zoom_in, zoom_out, pan_left, pan_right, rotate_cw, rotate_ccw, fade_in, fade_out, warm_shift, cool_shift, vintage_shift)")
    parser.add_argument("--transition", type=str, default="crossfade", 
                       help="Transition between images (crossfade, slide_left, slide_right, slide_up, slide_down, zoom_in, zoom_out, warp)")
    parser.add_argument("--resolution", type=str, default="1920x1080", 
                       help="Output video resolution (e.g., 1920x1080)")
    parser.add_argument("--loop", action="store_true", 
                       help="Loop back to first image at the end")
    return parser.parse_args()

def get_effect_config(effect_name: str) -> dict:
    """Get effect configuration based on effect name"""
    if effect_name == "zoom_in":
        return {"type": "zoom", "params": {"zoom_type": "in", "zoom_factor": 0.3}}
    elif effect_name == "zoom_out":
        return {"type": "zoom", "params": {"zoom_type": "out", "zoom_factor": 0.3}}
    elif effect_name == "pan_left":
        return {"type": "pan", "params": {"direction": "left", "intensity": 0.3}}
    elif effect_name == "pan_right":
        return {"type": "pan", "params": {"direction": "right", "intensity": 0.3}}
    elif effect_name == "pan_up":
        return {"type": "pan", "params": {"direction": "up", "intensity": 0.3}}
    elif effect_name == "pan_down":
        return {"type": "pan", "params": {"direction": "down", "intensity": 0.3}}
    elif effect_name == "rotate_cw":
        return {"type": "rotate", "params": {"rotation_direction": "cw", "rotations": 1}}
    elif effect_name == "rotate_ccw":
        return {"type": "rotate", "params": {"rotation_direction": "ccw", "rotations": 1}}
    elif effect_name == "fade_in":
        return {"type": "fade", "params": {"fade_type": "in"}}
    elif effect_name == "fade_out":
        return {"type": "fade", "params": {"fade_type": "out"}}
    elif effect_name == "warm_shift":
        return {"type": "color_shift", "params": {"shift_type": "warm"}}
    elif effect_name == "cool_shift":
        return {"type": "color_shift", "params": {"shift_type": "cool"}}
    elif effect_name == "vintage_shift":
        return {"type": "color_shift", "params": {"shift_type": "vintage"}}
    else:
        return {"type": "zoom", "params": {"zoom_type": "in", "zoom_factor": 0.3}}

def get_transition_config(transition_name: str) -> dict:
    """Get transition configuration based on transition name"""
    if transition_name == "crossfade":
        return {"type": "crossfade"}
    elif transition_name == "slide_left":
        return {"type": "slide", "params": {"direction": "left"}}
    elif transition_name == "slide_right":
        return {"type": "slide", "params": {"direction": "right"}}
    elif transition_name == "slide_up":
        return {"type": "slide", "params": {"direction": "up"}}
    elif transition_name == "slide_down":
        return {"type": "slide", "params": {"direction": "down"}}
    elif transition_name == "zoom_in":
        return {"type": "zoom", "params": {"zoom_type": "in"}}
    elif transition_name == "zoom_out":
        return {"type": "zoom", "params": {"zoom_type": "out"}}
    elif transition_name == "warp":
        return {"type": "warp"}
    else:
        return {"type": "crossfade"}

def load_images(input_paths: list) -> list:
    """Load images from input paths"""
    images = []
    for path in input_paths:
        if os.path.isdir(path):
            # Load all images in directory
            for file in sorted(os.listdir(path)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(path, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
        else:
            # Load single image
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
    
    if not images:
        raise ValueError("No valid images found in input paths")
    
    return images

def resize_images(images: list, resolution: tuple) -> list:
    """Resize images to target resolution"""
    width, height = resolution
    resized_images = []
    for img in images:
        h, w = img.shape[:2]
        
        # Calculate aspect ratio preserving resize
        target_aspect = width / height
        img_aspect = w / h
        
        if img_aspect > target_aspect:
            # Image is wider than target
            new_w = width
            new_h = int(width / img_aspect)
        else:
            # Image is taller than target
            new_h = height
            new_w = int(height * img_aspect)
            
        # Resize image
        resized = cv2.resize(img, (new_w, new_h))
        
        # Pad to target resolution
        pad_top = (height - new_h) // 2
        pad_bottom = height - new_h - pad_top
        pad_left = (width - new_w) // 2
        pad_right = width - new_w - pad_left
        
        padded = cv2.copyMakeBorder(resized, 
                                   pad_top, pad_bottom, 
                                   pad_left, pad_right, 
                                   cv2.BORDER_CONSTANT, 
                                   value=(0, 0, 0))
        resized_images.append(padded)
    
    return resized_images

def main():
    args = parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Load and resize images
    images = load_images(args.input_path)
    images = resize_images(images, (width, height))
    
    # Handle loop option
    if args.loop and len(images) > 1:
        images.append(images[0])
    
    # Get effect and transition configurations
    effect_config = get_effect_config(args.effect)
    transition_config = get_transition_config(args.transition)
    
    # Calculate durations
    num_images = len(images)
    if num_images == 1:
        # Single image - all time for effect
        image_duration = args.duration
        transition_duration = 0
    else:
        # Multiple images - split time
        total_duration = args.duration * num_images
        image_duration = (total_duration - (num_images - 1) * args.transition_duration) / num_images
        transition_duration = args.transition_duration
        if image_duration <= 0:
            raise ValueError("Total duration too short for the number of images and transitions")
    
    # Generate video frames
    all_frames = []
    
    for i in range(num_images):
        # Apply effect to current image
        img_frames = VideoEffects.apply_effect(
            images[i], effect_config, image_duration, args.fps
        )
        all_frames.extend(img_frames)
        
        # Apply transition to next image (if not last)
        if i < num_images - 1:
            # Get last frame of current image
            last_frame = all_frames[-1]
            
            # Get first frame of next image with effect applied
            next_img_frames = VideoEffects.apply_effect(
                images[i+1], effect_config, image_duration, args.fps
            )
            first_next_frame = next_img_frames[0]
            
            # Apply transition
            transition_frames = VideoTransitions.apply_transition(
                last_frame, first_next_frame, 
                transition_config, transition_duration, args.fps
            )
            all_frames.extend(transition_frames)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, args.fps, (width, height))
    
    # Write frames to video
    for frame in tqdm(all_frames, desc="Writing video"):
        out.write(frame)
    
    out.release()
    print(f"Video created successfully at {args.output_path}")
    print(f"Total duration: {len(all_frames)/args.fps:.2f} seconds")
    print(f"Total frames: {len(all_frames)}")

if __name__ == "__main__":
    main()