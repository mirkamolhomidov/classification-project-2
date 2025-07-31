import streamlit as st
from fastai.vision.all import load_learner, PILImage
import io

st.title("Rasm Klassifikatsiyasi")

uploaded_file = st.file_uploader("Rasmni yuklang", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    
    img = PILImage.create(io.BytesIO(img_bytes))
    
    st.image(img, caption="Yuklangan rasm", use_container_width=True)

    model = load_learner("classify-mohirdev.pkl")

    pred, pred_id, probs = model.predict(img)

    st.markdown(f"### ✅ Bashorat: **{pred}**")
    st.markdown(f"### 📊 Ishonchlilik: **{probs[pred_id]:.2f}**")
