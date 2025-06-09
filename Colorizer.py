import streamlit as st
import numpy as np
import cv2
import os
import tempfile

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    proto = "colorization_deploy_v2.prototxt"
    model = "colorization_release_v2.caffemodel"
    points = "pts_in_hull.npy"

    net = cv2.dnn.readNetFromCaffe(proto, model)
    pts = np.load(points)
    pts = pts.transpose().reshape(2, 313, 1, 1)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net

net = load_model()

# ---------- Streamlit UI ----------
st.title("🎨 Vintage Photo Colorizer")
st.markdown("Upload an old grayscale photo and AI will colorize it!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        image_path = tmp_file.name

    # Read image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_orig = img.copy()

    h, w = img.shape[:2]

    # Convert to float and normalize
    img_input = img.astype("float32") / 255.0
    lab = cv2.cvtColor(img_input, cv2.COLOR_BGR2LAB)

    # Resize only L channel and subtract 50
    L = cv2.resize(lab[:, :, 0], (224, 224))
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (w, h))

    # Merge original L with predicted ab
    L_orig = lab[:, :, 0]
    colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

    st.image([img_rgb, colorized_rgb], caption=["Original", "Colorized"], width=350)
