# DigitVision: Handwritten Digit Recognizer

A lightweight, full-stack web application that allows users to draw a digit (0-9) on a browser canvas. It uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to predict the drawn number in real-time.

![DigitVision Overview](https://via.placeholder.com/800x400.png?text=DigitVision+App)

## ✨ Features
* **Simple CNN Model**: Built with TensorFlow/Keras. It uses data augmentation (rotation, zoom, shifting) during training to handle the variations in real hand-drawn numbers.
* **Smart Image Preprocessing**: Canvas drawings are automatically cropped, resized, and **centered by their center of mass** (matching the original MNIST dataset generation process). This crucial step dramatically improves real-world accuracy.
* **Vanilla HTML/JS Frontend**: Clean, responsive layout using native web APIs. Fully functional on desktop and mobile touch devices without heavy frameworks like React.
* **Auto-Prediction**: Automatically queries the backend 500ms after the user stops drawing for a seamless experience.

## 🛠️ Tech Stack
* **Machine Learning**: Python, TensorFlow/Keras, NumPy, SciPy
* **Backend API**: FastAPI, Uvicorn, Pillow (for image processing)
* **Frontend**: Vanilla HTML5, CSS3, JavaScript

## 📂 Folder Structure
```text
digitvision/
├── train_model.py        # Script to train and save the CNN model
├── model/                # Directory containing the saved mnist_cnn.keras model
├── backend/
│   ├── main.py           # FastAPI server and endpoint routing
│   └── preprocess.py     # Image decoding, cropping, and center-of-mass centering
├── frontend/
│   ├── index.html        # Main web page
│   ├── style.css         # Responsive styling and progress bar animations
│   └── script.js         # Canvas drawing logic and API communication
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- *Note for Windows Users:* Ensure your C++ compiler (like GCC) is up to date (GCC >= 8.4 is required for certain `numpy` builds if using older Python environments).

### 1. Install Dependencies
Clone the repository and install the required Python packages:
```bash
git clone https://github.com/hardikkanfade/Handwritten-Classification.git
cd Handwritten-Classification
pip install -r requirements.txt
```

### 2. Train the Model
You must train the model before running the server so the `.keras` file is generated.
```bash
python train_model.py
```
This script downloads the MNIST dataset, trains the CNN for ~10 epochs, and saves it to `model/mnist_cnn.keras`.

### 3. Start the FastAPI Server
Launch the backend server using Uvicorn:
```bash
uvicorn backend.main:app --reload
```
This serves both the backend API (on `/predict`) and the frontend static files.

### 4. Use the App
Open your browser and navigate to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**. 
Draw on the black canvas, and you will see the prediction and probability distribution update automatically!

## 🔮 Future Improvements
* Add a Dockerfile for easier, environment-agnostic deployment.
* Add extra drawing tools (e.g., an eraser or brush hardness setting).
* Add support for recognizing letters using the EMNIST dataset.
