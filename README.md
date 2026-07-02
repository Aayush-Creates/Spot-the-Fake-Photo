# 📸 Spot the Fake Photo

A lightweight computer vision and machine learning solution to distinguish between a **real photograph** and a **photo of a screen (recaptured image)**.

This project was developed as part of the **SalesCode AI – Spot the Fake Photo** take-home assignment. Instead of using a deep learning model, it relies on handcrafted image features and a Gradient Boosting classifier to detect subtle artifacts introduced when a photograph is taken of a digital display.

---

## ✨ Features

* Detects whether an image is:

  * 📷 **Real Photo**
  * 🖥️ **Photo of a Screen (Recapture)**
* Classical computer vision feature extraction
* Gradient Boosting classifier
* 5-fold Stratified Cross Validation
* Fast inference suitable for on-device deployment
* Probability-based prediction (`0 → Real`, `1 → Screen`)
* Evaluation script with accuracy, confusion matrix, and latency measurement

---

## 🧠 Approach

Rather than training a deep neural network, this project extracts handcrafted features that capture the differences between real-world photographs and screen recaptures.

### Extracted Features

* **FFT Frequency Analysis**

  * Detects periodic pixel-grid patterns produced by displays.

* **Noise Statistics**

  * Measures sensor noise characteristics.

* **Local Binary Patterns (LBP)**

  * Captures texture differences.

* **Color Statistics**

  * RGB mean, standard deviation and skewness.

* **HSV Statistics**

  * Saturation and brightness characteristics.

* **Edge Sharpness**

  * Laplacian variance and edge distribution.

* **Vignetting**

  * Center-to-corner brightness ratio.

* **Glare Detection**

  * Percentage of saturated pixels.

* **Color Cast**

  * Red/Blue channel imbalance.

* **JPEG Block Artifacts**

  * Detects recompression traces.

These features are standardized and fed into a **Gradient Boosting Classifier**.

---

## 📊 Model Performance

### Cross Validation

| Metric        | Value             |
| ------------- | ----------------- |
| Mean Accuracy | **94.5% ± 3.4%**  |
| Mean F1 Score | **0.947 ± 0.033** |

### Fold Accuracy

* Fold 1 — **95.5%**
* Fold 2 — **100.0%**
* Fold 3 — **95.5%**
* Fold 4 — **90.9%**
* Fold 5 — **90.9%**

### Training Performance

The final model correctly classified all training samples after fitting on the complete dataset.

### Most Important Features

1. Vignette Ratio
2. FFT Standard Deviation
3. Color Cast
4. LBP Histogram
5. Blue Channel Mean
6. Value Mean (HSV)
7. Vignette Difference
8. Blue Channel Standard Deviation
9. Value Standard Deviation
10. Saturation Mean

---

## 📂 Project Structure

```text
Spot-the-Fake-Photo/
│
├── real/                  # Real photographs
├── screen/                # Screen recaptures
│
├── Train.py               # Train Gradient Boosting model
├── Evaluate.py            # Evaluate model
├── Predict.py             # Predict single image
│
├── model.pkl              # Saved trained model
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Aayush-Creates/Spot-the-Fake-Photo.git

cd Spot-the-Fake-Photo
```

Install dependencies:

```bash
pip install -r requirements.txt
```

or

```bash
pip install numpy scipy scikit-learn opencv-python scikit-image joblib
```

---

## 🚀 Training

Update the dataset path inside `Train.py` and run:

```bash
python Train.py
```

This will:

* Extract image features
* Perform 5-fold cross validation
* Train the final model
* Save `model.pkl`

---

## 📈 Evaluation

Run:

```bash
python Evaluate.py
```

The evaluation script reports:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Per-image predictions
* Average inference latency
* Estimated deployment cost

---

## 🔍 Prediction

Predict a single image:

```bash
python Predict.py image.jpg
```

Output:

```text
0.93
```

Where

* **0.0** → Real Photo
* **1.0** → Photo of a Screen

---

## 📱 Deployment

The current implementation is lightweight and suitable for deployment on edge devices because it uses handcrafted image features instead of a large neural network.

Future optimizations include:

* Feature selection
* Model quantization
* Mobile deployment
* Hybrid CNN + Computer Vision model

---

## 🔮 Future Improvements

* Larger and more diverse dataset
* MobileNet fine-tuning
* ResNet18 comparison
* Hybrid CNN + handcrafted features
* Hyperparameter optimization
* Better threshold calibration
* Test-Time Augmentation
* Real-time mobile application

---

## 🛠️ Technologies Used

* Python
* OpenCV
* NumPy
* SciPy
* scikit-image
* scikit-learn
* Joblib

---

## 📄 License

This project was developed for the **SalesCode AI Spot the Fake Photo Assignment** and is shared for educational and portfolio purposes.

---

## 👨‍💻 Author

**Aayush Gupta**

GitHub: https://github.com/Aayush-Creates
