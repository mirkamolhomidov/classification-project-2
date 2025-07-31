import streamlit as st
from fastai.vision.all import *
import pathlib
import io
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

st.title('Bu klassifikatsiya model')

model = load_learner('classify-mohirdev.pkl')

file = st.file_uploader('Rasm yuklash', type=['png', 'gif', 'jpg', 'jpeg'])
if file:
    st.image(io.BytesIO(img_bytes), caption='Yuklangan rasm', use_container_width=True)
    
    img_bytes = file.read()
    img = PILImage.create(io.BytesIO(img_bytes))

    pred, pred_id, probs = model.predict(img)

    st.success(f"Predict: {pred}")
    st.info(f"Probablity: {probs[pred_id]:.3f}")
