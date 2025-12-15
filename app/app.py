import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("final_cnn_model.h5")
    return model

model = load_model()

# -----------------------
# Preprocess Function
# -----------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)
    return img_array

# -----------------------
# Grad-CAM Heatmap
# -----------------------
def generate_gradcam(img_array, model):
    last_conv_layer = model.get_layer(model.layers[-4].name)
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])

        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap

def overlay_heatmap(original, heatmap, intensity=0.5):
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result = cv2.addWeighted(original, 1 - intensity, heatmap_color, intensity, 0)
    return result

# -----------------------
# UI
# -----------------------
st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.write("Upload an X-ray image to detect Normal vs Pneumonia.")

uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]

    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = round(prediction if prediction > 0.5 else 1 - prediction, 2)

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence * 100}%**")

    # Grad-CAM Generation
    heatmap = generate_gradcam(img_array, model)
    original_np = np.array(image)
    gradcam_img = overlay_heatmap(original_np, heatmap)

    st.subheader("🔥 Grad-CAM Heatmap")
    st.image(gradcam_img, use_column_width=True)

    # Download prediction result
    st.download_button(
        label="📥 Download Prediction",
        data=f"Label: {label}\nConfidence: {confidence}",
        file_name="prediction.txt",
        mime="text/plain"
    )
