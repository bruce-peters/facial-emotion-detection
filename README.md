# Facial Emotion Detection

Real-time facial emotion detection from a webcam, built with Python, OpenCV and TensorFlow/Keras. A face detector finds each face, and a CNN classifies it into one of seven emotions: angry, disgust, fear, happy, neutral, sad, surprise.

**Live web version:** https://facial-emotion-app.web.app/ (needs a webcam and a browser with WebGL / hardware acceleration turned on; see [Web app](#web-app))

## Run it

### Prerequisites

- Python 3.9-3.11 (TensorFlow 2.15 does not support 3.12+). Tested with Python 3.11.
- A webcam and a desktop session for `main.py` (it opens OpenCV windows).

### Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the webcam demo

```bash
python main.py
```

`main.py` loads the pretrained model `models/emotion_tl_1.h5`, detects faces with OpenCV's Haar cascade, and draws a box and an `EMOTION confidence` label on each face. A second window, "Face Data", shows the 48x48 image the model sees. Press **Esc** to quit.

`models/emotion_tl_2.h5` is a second trained model with the same architecture; change the path in `main.py` to try it.

## Train your own model

### Dataset

The training scripts expect a folder-per-class image dataset:

```
<dataset>/
  train/
    angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
  test/
    angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
```

- Images must be **48x48** pixels. `model_build.py` reads them as grayscale; `model_build_tlearning.py` reads them as 3-channel.
- Class labels come from the **alphabetical order** of the folder names, so the folders must sort in the order above to match the labels in `main.py`.
- This is the layout of the FER-2013 dataset on Kaggle (48x48 grayscale faces, seven classes). The dataset is not included in this repo.

The dataset location is hard-coded at the top of each script. Edit `trainPath` and `testPath` to point at your copy before running.

### Scripts

| Script | What it does | Output |
|---|---|---|
| `model_build_tlearning.py` | Transfer learning: MobileNetV2 (ImageNet weights, downloaded on first run) plus a small dense head, 48x48x3 input, data augmentation, early stopping. This is the architecture of the models in `models/`. | `emotion_tl.h5` |
| `model_build.py` | A CNN trained from scratch on 48x48 grayscale input, with data augmentation. | `emotion.h5` |
| `model_build_high_res.ipynb` | Notebook experiment: VGG16 transfer learning on images upscaled to 224x224. | `emotion_tl.h5` |
| `loading_data.py` | Loads half of each class at 224x224 and saves NumPy arrays to `data/` (create that folder first). | `data/*.npy` |
| `testing_models.py` | Evaluates every model in `models/` on the test set and prints loss and accuracy. | printed scores |

```bash
python model_build_tlearning.py
python testing_models.py
```

Training runs on the CPU if no GPU is found, but a full run (150 epochs for the transfer-learning script) is slow without a GPU. Each script saves the best checkpoint (lowest validation loss) in the current folder; move it into `models/` to use it with `main.py` or `testing_models.py`.

## Web app

The live site at https://facial-emotion-app.web.app/ runs the same idea in the browser: it detects faces with [`@vladmandic/face-api`](https://github.com/vladmandic/face-api) (TinyFaceDetector) and classifies each one with a TensorFlow.js copy of a Keras emotion model (48x48 input). Allow camera access when the browser asks.

The React source for the web app is **not** in this repository, so it cannot be built from here.

The site needs WebGL. With hardware acceleration turned off (or in a browser without GPU access) the models fail to load and the console shows `The highest priority backend 'wasm' has not yet been initialized`.

## Project structure

```
main.py                         webcam demo (inference)
model_build_tlearning.py        transfer-learning training script (MobileNetV2)
model_build.py                  CNN-from-scratch training script
model_build_high_res.ipynb      VGG16 / 224x224 experiment
loading_data.py                 dataset -> .npy preprocessing
testing_models.py               evaluate all models in models/
models/emotion_tl_1.h5          pretrained model used by main.py (~44 MB)
models/emotion_tl_2.h5          second pretrained model (~44 MB)
haarcascade_frontalface_default.xml  OpenCV face detector (main.py uses the copy bundled with OpenCV)
requirements.txt
```

## Notes

- The two `.h5` models (~44 MB each) are committed directly, not through Git LFS. Git history also still contains an old ~100 MB `shape_predictor_68_face_landmarks.dat` that was later deleted, so a full clone is about 160 MB. `git clone --depth 1` skips the history.
