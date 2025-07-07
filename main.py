import cv2
import sys
import argparse
import numpy as np
import os
import random
from tqdm import tqdm
from pathlib import Path
from effects import VideoEffects
from transitions import VideoTransitions

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Image-to-Video Converter with Selective Randomization")
    parser.add_argument("input_path", nargs="+", help="Input image(s) or directory")
    parser.add_argument("output_path", help="Output video file path")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--resolution", type=str, default="1920x1080", 
                       help="Output resolution (width x height)")
    
    # Duration settings
    parser.add_argument("--img_duration", type=float, default=3.0, 
                       help="Duration per image segment (seconds)")
    parser.add_argument("--transition_duration", type=float, default=1.0, 
                       help="Transition duration (seconds)")
    
    # Effect and transition selection
    parser.add_argument("--effect", type=str, default="zoom_in", 
                       help="Effect type (zoom_in/out, pan_*, rotate_*, fade_*, color_*)")
    parser.add_argument("--transition", type=str, default="crossfade", 
                       help="Transition type (crossfade, slide_*, zoom_*, warp)")
    
    # Randomization options with selective choices
    parser.add_argument("--random_effects", nargs="*", default=None, metavar="EFFECT",
                       help="Apply random effects from specified list (e.g., zoom_in pan_left fade_in)")
    parser.add_argument("--random_transitions", nargs="*", default=None, metavar="TRANSITION",
                       help="Apply random transitions from specified list (e.g., crossfade slide_left warp)")
    parser.add_argument("--random_all", action="store_true", 
                       help="Enable full randomization for all effects and transitions")
    
    # Video options
    parser.add_argument("--loop", action="store_true", 
                       help="Loop back to first image at the end")
    parser.add_argument("--codec", type=str, default="mp4v", 
                       help="Video codec (mp4v, avc1, vp09)")
    
    return parser.parse_args()

def load_images(paths):
    """Load images from file paths or directories"""
    images = []
    for path in paths:
        if os.path.isdir(path):
            for file in sorted(os.listdir(path)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img = cv2.imread(os.path.join(path, file))
                    if img is not None:
                        images.append(img)
        else:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
                
    if not images:
        raise ValueError("No valid images found")
    return images

def resize_with_padding(image, target_width, target_height):
    """Resize image while maintaining aspect ratio with padding"""
    h, w = image.shape[:2]
    target_aspect = target_width / target_height
    image_aspect = w / h
    
    if image_aspect > target_aspect:
        # Fit to width
        new_w = target_width
        new_h = int(target_width / image_aspect)
    else:
        # Fit to height
        new_h = target_height
        new_w = int(target_height * image_aspect)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # Calculate padding
    pad_top = (target_height - new_h) // 2
    pad_bottom = target_height - new_h - pad_top
    pad_left = (target_width - new_w) // 2
    pad_right = target_width - new_w - pad_left
    
    # Apply padding
    padded = cv2.copyMakeBorder(
        resized, 
        pad_top, pad_bottom, 
        pad_left, pad_right, 
        cv2.BORDER_CONSTANT, 
        value=(0, 0, 0)
    )
    return padded

def get_effect_config(effect_name=None, random_effect=False, effect_pool=None):
    """Get effect configuration with selective randomization"""
    # Predefined effects
    all_effects = {
        
        # Zoom effects
        "zoom_in": {"type": "zoom", "params": {"zoom_type": "in", "zoom_factor": 0.3}},
        "zoom_out": {"type": "zoom", "params": {"zoom_type": "out", "zoom_factor": 0.3}},
        
        # Pan effects
        "pan_left": {"type": "pan", "params": {"direction": "left", "intensity": 0.3}},
        "pan_right": {"type": "pan", "params": {"direction": "right", "intensity": 0.3}},
        "pan_up": {"type": "pan", "params": {"direction": "up", "intensity": 0.3}},
        "pan_down": {"type": "pan", "params": {"direction": "down", "intensity": 0.3}},
        
        # Rotation effects
        "rotate_cw": {"type": "rotate", "params": {"rotation_direction": "cw", "rotations": 1}},
        "rotate_ccw": {"type": "rotate", "params": {"rotation_direction": "ccw", "rotations": 1}},
        
        # Fade effects
        "fade_in": {"type": "fade", "params": {"fade_type": "in"}},
        "fade_out": {"type": "fade", "params": {"fade_type": "out"}},
        
        # Color effects
        "warm_shift": {"type": "color_shift", "params": {"shift_type": "warm"}},
        "cool_shift": {"type": "color_shift", "params": {"shift_type": "cool"}},
        "vintage_shift": {"type": "color_shift", "params": {"shift_type": "vintage"}},

    
    }
    
    # Handle selective randomization
    if effect_pool:
        valid_effects = [e for e in effect_pool if e in all_effects]
        if valid_effects:
            return all_effects[random.choice(valid_effects)]
        # Fallback to default if no valid effects specified
        return all_effects["zoom_in"]
    
    # Handle full randomization
    if random_effect:
        return random.choice(list(all_effects.values()))
    
    # Handle fixed effect
    return all_effects.get(effect_name, all_effects["zoom_in"])

def get_transition_config(transition_name=None, random_transition=False, transition_pool=None):
    """Get transition configuration with selective randomization"""
    # Predefined transitions
    all_transitions = {
        "crossfade": {"type": "crossfade"},
        "slide_left": {"type": "slide", "params": {"direction": "left"}},
        "slide_right": {"type": "slide", "params": {"direction": "right"}},
        "slide_up": {"type": "slide", "params": {"direction": "up"}},
        "slide_down": {"type": "slide", "params": {"direction": "down"}},
        "zoom_in": {"type": "zoom", "params": {"zoom_type": "in"}},
        "zoom_out": {"type": "zoom", "params": {"zoom_type": "out"}},
        "warp": {"type": "warp"}
    }
    
    # Handle selective randomization
    if transition_pool:
        valid_transitions = [t for t in transition_pool if t in all_transitions]
        if valid_transitions:
            return all_transitions[random.choice(valid_transitions)]
        # Fallback to default if no valid transitions specified
        return all_transitions["crossfade"]
    
    # Handle full randomization
    if random_transition:
        return random.choice(list(all_transitions.values()))
    
    # Handle fixed transition
    return all_transitions.get(transition_name, all_transitions["crossfade"])

def main():
    args = parse_args()
    
    # Handle randomization flags
    if args.random_all:
        args.random_effects = []  # Empty list triggers full randomization
        args.random_transitions = []  # Empty list triggers full randomization
    
    # Load and process images
    images = load_images(args.input_path)
    width, height = map(int, args.resolution.split('x'))
    processed_images = [resize_with_padding(img, width, height) for img in images]
    
    # Handle looping
    if args.loop and len(processed_images) > 1:
        processed_images.append(processed_images[0])
    
    # Precompute effect configurations
    effect_configs = []
    for i in range(len(processed_images)):
        if args.random_effects is not None:
            # Selective randomization
            effect_configs.append(get_effect_config(effect_pool=args.random_effects))
        else:
            # Fixed effect
            effect_configs.append(get_effect_config(args.effect))
    
    # Precompute transition configurations
    transition_configs = []
    for i in range(len(processed_images) - 1):
        if args.random_transitions is not None:
            # Selective randomization
            transition_configs.append(get_transition_config(transition_pool=args.random_transitions))
        else:
            # Fixed transition
            transition_configs.append(get_transition_config(args.transition))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    video_writer = cv2.VideoWriter(args.output_path, fourcc, args.fps, (width, height))
    
    # Generate video frames
    total_frames = 0
    
    for i, img in enumerate(tqdm(processed_images, desc="Processing images")):
        # Apply effect to current image
        img_frames = VideoEffects.apply_effect(
            img, 
            effect_configs[i], 
            args.img_duration, 
            args.fps
        )
        
        # Write image frames to video
        for frame in img_frames:
            video_writer.write(frame)
        total_frames += len(img_frames)
        
        # Apply transition if not last image
        if i < len(processed_images) - 1:
            next_img = processed_images[i + 1]
            
            # Get first frame of next image with its effect
            next_first_frame = VideoEffects.apply_effect(
                next_img, 
                effect_configs[i + 1], 
                0.1,  # Minimal duration to get first frame
                args.fps
            )[0]
            
            # Generate transition frames
            trans_frames = VideoTransitions.apply_transition(
                img_frames[-1], 
                next_first_frame, 
                transition_configs[i], 
                args.transition_duration, 
                args.fps
            )
            
            # Write transition frames to video
            for frame in trans_frames:
                video_writer.write(frame)
            total_frames += len(trans_frames)
    
    # Finalize video
    video_writer.release()
    
    # Print summary
    total_duration = total_frames / args.fps
    print(f"\nVideo created: {args.output_path}")
    print(f"Resolution: {width}x{height} | FPS: {args.fps} | Codec: {args.codec}")
    print(f"Total frames: {total_frames} | Duration: {total_duration:.2f} seconds")
    
    # Effect summary
    if args.random_effects is not None:
        if args.random_effects:
            print("Effects: Full randomization")
        else:
            print(f"Effects: Randomized from {args.random_effects}")
    else:
        print(f"Effects: Fixed ({args.effect})")
    
    # Transition summary
    if args.random_transitions is not None:
        if args.random_transitions:
            print("Transitions: Full randomization")
        else:
            print(f"Transitions: Randomized from {args.random_transitions}")
    else:
        print(f"Transitions: Fixed ({args.transition})")

if __name__ == "__main__":
    main()
