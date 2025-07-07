# Image to Video Converter

This Python project converts images into videos with customizable effects and transitions.

## Features

- Convert single or multiple images to video
- Apply various effects to images:
  - Zoom in/out
  - Pan in four directions
  - Rotation (clockwise/counter-clockwise)
  - Fade in/out
  - Color shifts (warm, cool, vintage)
- Add transitions between images:
  - Crossfade
  - Slide in four directions
  - Zoom transitions
  - Creative warp effect
- Customize video parameters:
  - Duration per image
  - FPS
  - Transition duration
  - Output resolution
  - Loop back to first image

## Installation

1. Clone the repository:
```bash
git clone https://github.com/inzmam01/image-to-video.git
cd image-to-video

## How to Use

1. Install the dependencies:
```bash
pip install -r requirements.txt

2. Run the script with your images:

For Single image with zoom effect:

# bash

python main.py (location of image like /cd/user/downloads/image.1) output.mp4 --duration 5 --effect zoom_in

2. Multiple images with slide transition:

# bash

python main.py (img1.jpg img2.jpg img3.jpg or location of folder like /cd/user/downloads/images/) slideshow.mp4 --duration 3 --transition slide_left