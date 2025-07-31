import streamlit as st
from fastai.vision.all import *
import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

st.title('Bu klassifikatsiya model')

model = load_learner('classify-mohirdev.pkl')

file = st.file_uploader('Rasm yuklash', type=['png', 'gif', 'jpg', 'jpeg'])
if file:
    st.image(file, caption='Yuklangan rasm', use_container_width=True)

    img = PILImage.create(file)

    pred, pred_id, probs = model.predict(img)

    st.success(f"Predict: {pred}")
    st.info(f"Probablity: {probs[pred_id]:.3f}")
