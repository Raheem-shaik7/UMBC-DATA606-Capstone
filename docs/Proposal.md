Pneumonia Detection from Chest X-Rays

Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang
Author: Raheem Shaik
GitHub: https://github.com/raheemshaik

LinkedIn: https://www.linkedin.com/in/raheem-shaik

1. Background

Pneumonia is a serious lung infection that causes many hospitalizations and deaths every year, especially among children and older adults. Normally, doctors use chest X-rays to check for pneumonia. But this takes time, requires trained radiologists, and mistakes can happen.

This project will use deep learning to build a system that can look at chest X-ray images and decide whether a patient has pneumonia or not. The model will be a Convolutional Neural Network (CNN), which is a type of algorithm that works very well with images.

Why It Matters

Helps doctors by giving quick results.

Reduces human errors in reading X-rays.

Useful in countries or hospitals where expert radiologists are not available.

Shows how AI can make healthcare more effective.

Research Questions

Can a CNN correctly separate pneumonia X-rays from normal ones?

Do advanced models (like ResNet or VGG16) work better than a simple CNN?

Can we explain what part of the lung the model is using to make decisions?

2. Data

Source: Kaggle – Chest X-Ray Images (Pneumonia)

Size: About 5,800 images (~1.5 GB).

Structure: Images are already divided into train, test, and val folders, each with two groups: NORMAL and PNEUMONIA.

Format: JPEG X-ray images of children’s lungs.

Data Dictionary
Column Name	Type	Description	Example
image_id	String	File name of the image	NORMAL-1001.jpeg
label	Binary	Target: pneumonia or not	0 = Normal, 1 = Pneumonia
pixel_data	Image	Chest X-ray pixels	224×224 grayscale

Target Variable: label (0 = Normal, 1 = Pneumonia).
Predictors: Image pixel values.

3. Methodology

Steps for the project:

Prepare the Data

Resize images to the same size (224×224).

Scale pixel values to make training easier.

Use data augmentation (flips, rotations) to avoid overfitting.

Build the Model

Start with a simple CNN.

Try advanced models (ResNet, VGG16) to compare performance.

Use sigmoid activation in the last layer for binary output.

Training

Loss: Binary Cross-Entropy.

Optimizer: Adam.

Tools: Python, TensorFlow/Keras, NumPy, Matplotlib.

Environment: Google Colab with GPU.

Evaluation

Accuracy, Precision, Recall, F1-Score.

Confusion Matrix.

ROC curve and AUC score.
