import streamlit as st
from fastai.vision.all import *
import os

# Model fayl yo'lini ko'rsatish
MODEL_PATH = "transport_model.pkl"

# Rasm yuklash interfeysi
st.title("Rasmni Tahlil Qilish Ilovasi")
uploaded_file = st.file_uploader("Rasm yuklang", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        # Yuklangan rasmni ochish
        img = PILImage.create(uploaded_file)
        st.image(img, caption="Yuklangan rasm", use_column_width=True)

        # Modelni yuklash
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model fayli topilmadi: {MODEL_PATH}")
        else:
            learner = load_learner(MODEL_PATH, cpu=True)
            pred, pred_idx, probs = learner.predict(img)

            # Natijalarni ko'rsatish
            st.write(f"Natija: **{pred}**")
            st.write(f"Ishonch: **{probs[pred_idx]:.2f}**")

    except Exception as e:
        st.error(f"Xatolik yuz berdi: {e}")
else:
    st.info("Iltimos, rasm yuklang.")

# import streamlit as st
# from fastai.vision.all import *
# import pathlib
# import PIL
# import plotly.express as px

# # Agar muhit Windows bo'lsa, pathlib bilan moslikni o'rnatish
# if pathlib.Path().anchor.startswith("\\"):
#     pathlib.PosixPath = pathlib.WindowsPath

# # Model yo'lini ko'rsatish
# MODEL_PATH = "transport_model.pkl"

# # Rasm yuklash
# st.title("Rasmni Tahlil Qilish Ilovasi")
# uploaded_file = st.file_uploader("Rasm yuklang", type=["jpg", "jpeg", "png", "webp"])

# if uploaded_file is not None:
#     try:
#         # Yuklangan rasmni o'qish
#         img = PILImage.create(uploaded_file)

#         # Yuklangan rasmni ko'rsatish
#         st.image(img, caption="Yuklangan rasm", use_column_width=True)

#         # Modelni yuklash
#         if not pathlib.Path(MODEL_PATH).exists():
#             st.error(f"Model fayli topilmadi: {MODEL_PATH}")
#         else:
#             learner = load_learner(MODEL_PATH)

#             # Model orqali bashorat qilish
#             pred, pred_idx, probs = learner.predict(img)
#             st.write(f"Natija: **{pred}**")
#             st.write(f"Ishonch: **{probs[pred_idx]:.2f}**")

#             # Natijalarni vizualizatsiya qilish
#             fig = px.bar(
#                 x=learner.dls.vocab,
#                 y=probs.numpy(),
#                 labels={"x": "Toifalar", "y": "Ishonch"},
#                 title="Bashorat qilingan ishonch darajalari",
#             )
#             st.plotly_chart(fig)

#     except Exception as e:
#         st.error(f"Xatolik yuz berdi: {e}")
# else:
#     st.info("Iltimos, rasm yuklang.")


# #streamlit run ./app.py
# import streamlit as st
# from fastai.vision.all import *
# import pathlib
# import plotly.express as px
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
# # Rasm yuklash
# uploaded_file = st.file_uploader("Rasm yuklang", type=["jpg", "jpeg", "png","webp"])

# if uploaded_file is not None:
#     # Yuklangan rasmni o'qish
#     img = PILImage.create(uploaded_file)
#     # Modelni yuklash
#     learner = load_learner("transport_model.pkl")  #model.pkl
#     # Debugging: Learner turini tekshirish
#     st.write(f"Learner turi: {type(learner)}")
#     # Rasmni aniqlash
#     try:
#         # Learner obyektining to'g'ri ekanligini tekshirish
#         if isinstance(learner, Learner):
#             pred, pred_idx, probs = learner.predict(img)    
#             # Natijani ko'rsatish
#             st.image(img, caption='Yuklangan rasm', use_column_width=True)
#             st.write(f"Bu rasm: {pred} (Ishonch: {probs[pred_idx]:.2f})")
#         else:
#             st.error("Learner obyektida xato. Model yuklash jarayonini tekshiring.")
#     except Exception as e:
#         st.error(f"Rasmni aniqlashda xato: {e}")























