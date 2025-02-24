# Face Scanner

This project is a face recognition system using OpenCV and Dlib. It captures images from a webcam, trains a model with the captured images, and recognizes users based on the trained model.

## Requirements

- Python 3.x
- OpenCV
- Dlib
- NumPy
- SQLite3

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Sammisam8888/face-scanner.git
   cd face-scanner
   ```

2. Create and activate the Python virtual environment:
   ```sh
   python3 -m venv env
   source ./env/bin/activate
   ```

3. Install dependencies on Linux:
   ```sh
   sudo apt-get update
   sudo apt-get install build-essential cmake
   sudo apt-get install libgtk-3-dev
   sudo apt-get install libboost-all-dev
   ```

4. Install the required Python packages:
   ```sh
   pip3 install -r requirements.txt
   ```

5. Download the required Dlib model files:
   - [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)

6. Extract the `.bz2` files and place the `.dat` files in the `models` directory.

## Usage

1. Run the script:
   ```sh
   python3 face-scanner.py
   ```

2. Follow the prompts to either train a new user or test face recognition.

### Training a New User

- Enter `train` when prompted for the operation.
- Enter the user's name.
- Look into the camera for 20 seconds to capture training images.

### Testing Face Recognition

- Enter `test` when prompted for the operation.
- Look into the camera to test face recognition.

## Notes

- Ensure that the `models` directory contains the required Dlib model files.
- Adjust the recognition threshold in the `recognize_user` function for better accuracy.
- Highest accuracy achieved: 99.4%
- Face comparison is done using the cosine formula.

