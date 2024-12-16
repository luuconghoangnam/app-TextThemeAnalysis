import streamlit as st
import numpy as np
import joblib  
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from class_lib import Bow_lib, Tfidf_lib
import re

# Initialize lemmatizer and stop words
stop_words = stopwords.words('english')
lemmer = WordNetLemmatizer()

# Load models and vectorizers
lda_model = joblib.load('lda_model.pkl')  
vectorizer_lda = joblib.load('vectorizer_lda.pkl') 
nmf_model = joblib.load('nmf_model.pkl')  
vectorizer_nmf = joblib.load('vectorizer_nmf.pkl') 

# Topic mappings
topic_lda = {
    0: 'education',
    1: 'business',
    2: 'sports',
    3: 'entertainment',
    4: 'technology'
}
topic_nmf = {
    0: 'sports',
    1: 'business',
    2: 'education',
    3: 'entertainment',
    4: 'technology'
}

# Streamlit interface
st.title('Ứng Dụng Phân Loại Chủ Đề Văn Bản')

# Add model selection
model_choice = st.radio(
    "Chọn mô hình phân loại:",
    ('LDA', 'NMF')
)

# Text input
user_input = st.text_area("Nhập văn bản cần trích chọn chủ đề:", "")

if st.button('Phân tích'):
    if user_input:
      
        
        # Preprocess exactly as during training
        # Remove special characters and convert to lowercase
        processed_text = re.sub(r'[^a-zA-Z\s]', '', user_input.lower())
        
        # Tokenize and lemmatize
        words = processed_text.split()
        lemmatized_words = [lemmer.lemmatize(word) for word in words if word not in stop_words]
        processed_text = ' '.join(lemmatized_words)
        
      
        if model_choice == 'LDA':
            vectorizer = vectorizer_lda
            model = lda_model
            topic_mapping = topic_lda
            st.subheader("Kết quả phân tích LDA:")
        else:
            vectorizer = vectorizer_nmf
            model = nmf_model
            topic_mapping = topic_nmf
            st.subheader("Kết quả phân tích NMF:")
        
      
        
        # Transform using vectorizer
        input_vector = vectorizer.transform([processed_text])
        
        
        prediction = model.transform(input_vector)[0]
        
        prediction = np.argmax(prediction)
        predicted_topic = topic_mapping.get(prediction)
        
        # Display results
        st.write(f"Chủ đề dự đoán: {predicted_topic}")
        
        # Display topic distribution
        st.write("Phân phối xác suất các chủ đề:")
        topic_probs = model.transform(input_vector)[0]
        for idx, prob in enumerate(topic_probs):
            st.write(f"{topic_mapping[idx]}: {prob:.4f}")
        
        # Visualize topic distribution
        import plotly.express as px
        
        fig = px.bar(
            x=[topic_mapping[i] for i in range(len(topic_probs))],
            y=topic_probs,
            title="Phân phối xác suất các chủ đề",
            labels={'x': 'Chủ đề', 'y': 'Xác suất'}
        )
        st.plotly_chart(fig)
        
    else:
        st.warning("Vui lòng nhập một văn bản để phân loại!")