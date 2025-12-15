
# 📝 **Pneumonia Detection from Chest X-Ray Images Using Deep Learning**

Prepared for **UMBC Data Science Master Degree Capstone**
Under the guidance of **Dr. Chaojie (Jay) Wang)**

**Author:** Raheem Shaik
**GitHub Repository:** https://github.com/Raheem-shaik7/UMBC-DATA606-Capstone
**LinkedIn:** [https://www.linkedin.com/in/raheem-shaik](https://www.linkedin.com/in/raheem-shaik)
**Project Presentation (PPT):** 

---

# **1. Background**

Pneumonia is a severe lung infection responsible for millions of hospitalizations worldwide. Early and accurate detection is crucial, especially in regions with limited access to radiologists. Traditionally, diagnosis relies on manual interpretation of chest X-rays, which is:

* Time-consuming
* Dependent on expert radiologists
* Prone to human error
* Not scalable in emergency or remote settings

Deep learning, especially **Convolutional Neural Networks (CNNs)**, has proven capable of analyzing medical images with high accuracy. This project explores whether a CNN can detect pneumonia using pediatric chest X-ray images and deliver predictions through a **user-friendly Streamlit app**.

---

# **2. Project Objective**

The goal of this project is to build a complete end-to-end system that can:

* Accept chest X-ray images as input
* Preprocess and normalize images
* Classify them as **Normal** or **Pneumonia**
* Output prediction + confidence score
* Deploy the model through an interactive web application

This system demonstrates how AI can support healthcare workflows by providing fast, reliable assistance to clinicians.

---

# **3. Dataset**

The dataset comes from the **Kaggle Chest X-Ray Pneumonia Dataset (5,863 images)**.

### **Folder Structure**

```
/train
   /NORMAL
   /PNEUMONIA
/test
   /NORMAL
   /PNEUMONIA
/val
   /NORMAL
   /PNEUMONIA
```

### **Final Dataset Summary**

| Class         | Count  |
| ------------- | ------ |
| **NORMAL**    | ~1,583 |
| **PNEUMONIA** | ~4,273 |

Pneumonia images dominate the dataset (~73%), creating class imbalance.

### **Data Columns**

* `image_id` — file name
* `label` — Normal / Pneumonia
* `pixel_data` — 224×224 normalized pixels

---

# **4. Image Processing and Preprocessing**

Each image undergoes the following steps:

1. **Loading & RGB conversion**
2. **Resize to 224×224 pixels**
3. **Normalize pixel values (0–1)**
4. **Batch processing through ImageDataGenerator**
5. **Data augmentation applied only to training images:**

   * Random rotation
   * Width & height shifts
   * Horizontal flip
   * Zoom

This improves model generalization and reduces overfitting.

---

# **5. Exploratory Data Analysis (EDA)**

Key observations:

* Pneumonia images tend to show visible opacities, highlighting infection in lung tissues.
* Normal images show clear lung fields with sharp contrast.
* Dataset is imbalanced → impacts evaluation metrics.
* Pixel intensity distributions differ between classes, supporting learnability.

EDA motivated careful selection of evaluation metrics beyond accuracy.

---

# **6. Model Development**

A **custom CNN** was implemented in TensorFlow/Keras.

### **CNN Architecture Overview**

* **Conv2D → MaxPool**
* **Conv2D → MaxPool**
* **Conv2D → MaxPool**
* **Flatten**
* **Dense Layer with ReLU**
* **Dropout for regularization**
* **Output Layer (Sigmoid) for binary classification**

### **Training Setup**

| Component    | Choice               |
| ------------ | -------------------- |
| Loss         | Binary Cross-Entropy |
| Optimizer    | Adam                 |
| Batch Size   | 32                   |
| Image Size   | 224×224              |
| Augmentation | Yes                  |
| Epochs       | 10                   |

Training was performed in **Google Colab GPU** environment for speed.

---

# **7. Train–Test Split**

Dataset already includes separate train/val/test splits.

To maintain consistency and fairness:

* Only test set images were used for final evaluation.
* No test data leakage occurred during training.

---

# **8. Class Imbalance Handling**

Instead of oversampling (which can distort medical datasets), this project uses:

* **Weighted evaluation metrics**
* **Balanced CNN training batches**
* **Careful threshold selection**

Accuracy alone is misleading due to imbalance.

---

# **9. Model Performance**

### **Classification Report (Test Set)**

| Metric        | NORMAL | PNEUMONIA |
| ------------- | ------ | --------- |
| **Precision** | 0.72   | 0.99      |
| **Recall**    | 0.99   | 0.87      |
| **F1-Score**  | 0.83   | 0.93      |

### **Overall Metrics**

* **Accuracy:** 0.90
* **Macro Avg F1:** 0.88
* **Weighted Avg F1:** 0.90

Interpretation:

* The model is extremely good at detecting **Normal** cases (Recall = 0.99)
* Slight drop in Pneumonia recall due to imbalance
* Performs well overall and suitable for clinical assistance

---

# **10. Model Evaluation Insights**

Key conclusions:

* Accuracy alone hides imbalance issues → F1, Precision, Recall are more meaningful.
* CNN generalizes well despite limited data variety.
* Pneumonia and Normal patterns are distinguishable enough for CNN learning.
* Pediatric dataset limits generalization to adults.

---

# **11. Saving & Packaging the Model**

The trained model is saved as:

```
final_cnn_model.h5
```

Benefits:

* Easy deployment in Streamlit
* Reproducible across systems
* Can be integrated into future API or mobile app

---

# **12. Streamlit Web Application**

A fully functional **web interface** was developed.

### **App Workflow**

1. User uploads a chest X-ray image
2. Image is preprocessed (resize + normalize)
3. CNN model predicts Probability of Pneumonia
4. User sees:

   * Prediction (Normal / Pneumonia)
   * Confidence score
   * X-ray preview

### **Design Choices**

* Large centered UI components for medical readability
* Warm color scheme for risk alerts
* Confidence score displayed with high precision

The app offers a real-world demonstration of AI-assisted radiology.

---

# **13. Results Interpretation**

This system is **not a diagnostic tool**, but:

* A screening assistant
* A support system for radiologists
* An educational demo of deep learning in healthcare

Model confidence scores help clinicians understand prediction certainty.

---

# **14. Limitations**

* Dataset includes only pediatric patients
* Training images vary in quality and acquisition equipment
* No localization (heatmaps, Grad-CAM disabled for stability)
* Model doesn’t differentiate between viral and bacterial pneumonia

Future versions can address these issues.

---

# **15. Future Work**

* Add **Grad-CAM heatmaps** for visual explainability
* Train on larger and more diverse datasets
* Use **transfer learning** (ResNet50, EfficientNet)
* Deploy as a cloud API (FastAPI or Flask)
* Build an Android/iOS mobile app
* Incorporate DICOM medical formats

---

# **16. Conclusion**

This project demonstrates that deep learning models — even lightweight CNNs — can successfully detect pneumonia from chest X-ray images. The combination of:

* Medical image preprocessing
* CNN classification
* Performance evaluation
* Streamlit deployment

creates a powerful foundation for real-world medical AI applications.

The system is fast, accessible, and effective — showing the potential of AI tools in assisting radiologists and improving patient care.

---

# **End of Report**
