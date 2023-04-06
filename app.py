import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Opus-MT modeli için tokenizer ve modeli yükleyin
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")

# Streamlit uygulamasının başlığını ayarlayın
st.title("Opus-MT Çeviri")

# Çevrilecek metni girdi olarak alın
text = st.text_area("Metni girin", value='', height=200)

# Çeviriyi gerçekleştirin
if st.button("Çevir"):
    if text:
        inputs = tokenizer.encode(text, return_tensors="pt")
        translated = model.generate(inputs["input_ids"], max_length=128, num_return_sequences=1)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        st.write("Çeviri Sonucu:")
        st.write(translated_text)
    else:
        st.warning("Çevrilecek metni girin.")
