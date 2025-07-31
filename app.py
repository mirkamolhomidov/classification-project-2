import streamlit as st
from fastai.vision.all import *
import pathlib
import platform
import io

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title('Bu klassifikatsiya model')

model = load_learner('classify-mohirdev.pkl')

file = st.file_uploader('Rasm yuklash', type=['png', 'gif', 'jpg', 'jpeg'])
if file:
    st.image(file, caption='Yuklangan rasm', use_container_width=True)

    img_bytes = file.read()

    img = PILImage.create(io.BytesIO(img_bytes))

    pred, pred_id, probs = model.predict(img)

    st.success(f"✅ Bashorat: {pred}")
    st.info(f"📈 Ishonchlilik: {probs[pred_id]:.3f}")
