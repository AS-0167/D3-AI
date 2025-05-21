
# 🚗 D3-CNN: Driver Distraction Detector

**D3-CNN** is a simple Convolutional Neural Network (CNN) built **from scratch using NumPy** to detect driver distractions from images and videos. It’s an academic project designed to deeply understand how CNNs work internally — with **no deep learning frameworks like PyTorch or TensorFlow used**.

It is trained purely on **CPU**, and also features a **Streamlit-based frontend** for easy testing with image or video input.

---

## 📸 Sample Input vs Prediction

| Input Image | Prediction |
|-------------|------------|
| ![sample](https://drive.google.com/file/d/1sKFoxgn0NRPQ5AWjx9h4_eMI-yC_eAh4/view?usp=drive_link) | `Texting` |

---

## 🧠 Model Architecture

The CNN is simple and built from scratch using NumPy only:

```python
self.layers = [
    Conv2D(filters=32, kernel_size=3, in_channels=3, padding=1),
    ReLU(),
    MaxPool2D(),
    Conv2D(filters=64, kernel_size=3, in_channels=32, padding=1),
    ReLU(),
    MaxPool2D(),
    Flatten(),
    Dense(input_size=16*16*64, output_size=128),
    ReLU(),
    Dense(input_size=128, output_size=NUM_CLASSES),
    Softmax()
]
````

### 📊 Architecture Diagram

![CNN Architecture](architecture.png)

> *Note: Please include the architecture diagram as `architecture.png` in the project root directory.*

---

## 📂 Dataset

* A mix of **publicly available data** and **self-collected images**.
* Download the dataset from this Google Drive link:
  📎 [Download Dataset](https://drive.google.com/drive/folders/1PbSZ1RF0z9Mcdn93eCshyd-z3H1v_uS4?usp=drive_link)

---

## ✅ Model Performance

* Trained on CPU from scratch
* Test Accuracy: **\~75%**
* Lightweight model for quick experimentation and learning

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/AS-0167/D3-CNN.git
cd D3-CNN
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Running the Streamlit App

You can upload an image or a video and get distraction predictions.

```bash
streamlit run app.py
```

---

## 🏋️‍♂️ Training the Model (Optional)

To retrain the model from scratch using the dataset:

1. Ensure dependencies are installed:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the training notebook:

   ```bash
   jupyter notebook train-detect-distraction.ipynb
   ```

---

## 🎓 About the Project

This project was created as a hands-on academic experiment to:

* Learn how CNNs work internally
* Understand forward and backward propagation using NumPy
* Handle dataset collection and preprocessing
* Build a mini deep learning pipeline without relying on big frameworks

No libraries like TensorFlow or PyTorch were used.

---

## 👨‍💻 Author

**\[AS-0167]**
🧑‍🎓 Student, Enthusiastic Learner of Machine Learning & Computer Vision



---
