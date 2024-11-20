## Guide
Full installation & run:

MACOS & linux
```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install mediapipe-silicon
$ brew install wget
$ !wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
$ cd project
$ python iris_tracking.py
```