# Pneumonia Detection from Chest X-Rays  
Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang  

**Author:** Raheem Shaik  
**GitHub:** [https://github.com/raheemshaik](https://github.com/raheemshaik)  
**LinkedIn:** [https://www.linkedin.com/in/raheem-shaik](https://www.linkedin.com/in/raheem-shaik)  

---

## 1. Background  

Pneumonia is a serious lung infection that causes many hospitalizations and deaths worldwide, especially among children and the elderly. Normally, doctors detect pneumonia by reading chest X-rays. However, this process can be slow, depends on expert radiologists, and sometimes leads to mistakes.  

This project aims to build a **deep learning model** to automatically classify chest X-rays as either **Normal** or **Pneumonia**. The model will use **Convolutional Neural Networks (CNNs)**, which are well suited for image recognition tasks.  

### Why It Matters  
- Helps doctors by providing quick and accurate results.  
- Reduces errors in diagnosis.  
- Useful in hospitals or countries with limited access to radiologists.  
- Shows how AI can solve important healthcare challenges.  

### Research Questions  
1. Can a CNN correctly classify chest X-rays into “Normal” or “Pneumonia”?  
2. Do pretrained models (like ResNet or VGG16) perform better than a simple CNN?  
3. Can we explain which parts of the X-ray influence the model’s decision?  

---

## 2. Data  

- **Source:** [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- **Size:** ~5,863 images (~1.5 GB).  
- **Structure:** Images are divided into `train`, `test`, and `val` folders with two classes: `NORMAL` and `PNEUMONIA`.  
- **Format:** JPEG chest X-ray images of pediatric patients.  

### Data Dictionary  

| Column Name | Type   | Description            | Example            |  
|-------------|--------|------------------------|--------------------|  
| image_id    | String | File name of the image | NORMAL-1001.jpeg   |  
| label       | Binary | Target variable        | 0 = Normal, 1 = Pneumonia |  
| pixel_data  | Image  | Chest X-ray pixels     | 224×224 grayscale  |  

**Target Variable:** `label` (0 = Normal, 1 = Pneumonia)  
**Predictors:** Pixel values from the chest X-ray images (with preprocessing and augmentation).  

---

## 3. Methodology  

The project will follow a standard machine learning workflow:  

1. **Data Preparation**  
   - Resize images to 224×224 pixels.  
   - Normalize pixel values (0–1 scale).  
   - Apply data augmentation (rotations, flips, zooms) to improve generalization.  

2. **Model Development**  
   - **Baseline Model:** A custom CNN with several convolutional and pooling layers.  
   - **Transfer Learning:** Fine-tuning pretrained networks such as ResNet50 or VGG16.  
   - Output layer: Sigmoid activation for binary classification.  

3. **Training Setup**  
   - Loss: Binary Cross-Entropy.  
   - Optimizer: Adam.  
   - Tools: Python, TensorFlow/Keras, NumPy, Matplotlib.  
   - Environment: Google Colab (GPU support).  

4. **Evaluation**  
   - Accuracy  
   - Precision, Recall, and F1-Score  
   - Confusion Matrix  
   - ROC curve and AUC score  

---
