import streamlit as st
from fastai.vision.all import load_learner, PILImage
import io
from PIL import Image
import torch

st.title("Rasm Klassifikatsiyasi")

# Model yuklash - caching bilan
@st.cache_resource
def load_model():
    try:
        return load_learner("classify-mohirdev.pkl")
    except Exception as e:
        st.error(f"Model yuklanmadi: {e}")
        return None

uploaded_file = st.file_uploader("Rasmni yuklang", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        # Faylni o'qish
        img_bytes = uploaded_file.read()
        
        # PILImage yaratish - turli usullar
        # Usul 1: PIL Image orqali
        pil_img = Image.open(io.BytesIO(img_bytes))
        
        # RGB formatga o'tkazish (agar RGBA bo'lsa)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # FastAI PILImage yaratish - alternativ usullar
        try:
            img = PILImage.create(pil_img)
        except:
            # Alternativ: bytes orqali yaratish
            img_bytes_io = io.BytesIO(img_bytes)
            img = PILImage.create(img_bytes_io)
            
        # Yoki faylni vaqtincha saqlash usuli
        # import tempfile
        # with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        #     tmp_file.write(img_bytes)
        #     img = PILImage.create(tmp_file.name)
        
        # Rasmni ko'rsatish
        st.image(pil_img, caption="Yuklangan rasm", use_container_width=True)
        
        # Model yuklash
        model = load_model()
        
        if model is not None:
            # Bashorat qilish
            with st.spinner('Bashorat qilinmoqda...'):
                pred, pred_id, probs = model.predict(img)
            
            # Natijalarni ko'rsatish
            st.markdown(f"### ✅ Bashorat: **{pred}**")
            st.markdown(f"### 📊 Ishonchlilik: **{probs[pred_id]:.2%}**")
            
            # Barcha ehtimolliklarni ko'rsatish
            st.markdown("### 📈 Barcha sinflar uchun ehtimolliklar:")
            for i, (class_name, prob) in enumerate(zip(model.dls.vocab, probs)):
                st.write(f"{class_name}: {prob:.2%}")
        
    except Exception as e:
        st.error(f"Xatolik yuz berdi: {str(e)}")
        st.write("Batafsil xatolik ma'lumoti:")
        st.exception(e)
