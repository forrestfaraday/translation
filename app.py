import streamlit as st
from transformers import MarianMTModel, MarianTokenizer, pipeline

# Opus-MT modeli için tokenizer ve modeli yükleyin
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
pipe = pipeline("translation_en_to_es", tokenizer=tokenizer,model=model, device=-1)
# Streamlit uygulamasının başlığını ayarlayın
st.title("Opus-MT Çeviri")

# Çevrilecek metni girdi olarak alın
text = st.text_area("Metni girin", value='', height=200)

# Çeviriyi gerçekleştirin
if st.button("Çevir"):
    if text:
        translated_text = pipe(text)[0]['translation_text']
        st.write("Çeviri Sonucu:")
        st.write(translated_text)
    else:
        st.warning("Çevrilecek metni girin.")
