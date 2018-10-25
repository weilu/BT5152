# Tutorial 8 Demo

- Facial Detection using OpenCV and builtin classifier
- Feature Extraction using pre-trained CNN
- Object Identification using pre-trained CNN with OpenCV

## Requirements

- A camera (if you are running the OpenCV code)
- Python 3.6 (because Tensorflow only support up to Python 3.6)
- Tensorflow 1.x
- OpenCV 3.4 (with the Haar Cascade models)
- Numpy

## Installation

### macOS

You can install OpenCV using `brew` by running `brew install opencv`.

### Windows

You can follow the [official guide](https://docs.opencv.org/3.4.3/d5/de5/tutorial_py_setup_in_windows.html).

## Models

### Haar Casade Models

You can download the models from [Github](https://github.com/opencv/opencv/tree/master/data/haarcascades) and place it in `haarcascades` directory.

Or, you can get it from your OpenCV installation by locating the installation directory.

If you install using `brew`, the installation path is at `/usr/local/Cellar/opencv/3.4.3/share/OpenCV/haarcascades/`.

## Keras

Run `pip install -r requirements.txt` to install the essential packages for this demo.
