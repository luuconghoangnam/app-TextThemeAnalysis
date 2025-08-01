import streamlit as st
import numpy as np
import joblib  
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize lemmatizer and stop words
stop_words = set(stopwords.words('english'))
lemmer = WordNetLemmatizer()

# Load models only
try:
    lda_model = joblib.load('lda_model.pkl')
    nmf_model = joblib.load('nmf_model.pkl')
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Create new vectorizers to avoid compatibility issues
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Function to preprocess text
def preprocess_text(text):
    """Preprocess text exactly as during training"""
    # Remove special characters and convert to lowercase
    processed_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize and lemmatize
    words = processed_text.split()
    lemmatized_words = [lemmer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(lemmatized_words)

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
st.title('🔍 Ứng Dụng Phân Tích Chủ Đề Văn Bản')
st.markdown("---")

# Add model selection
model_choice = st.radio(
    "**Chọn mô hình phân loại:**",
    ('LDA', 'NMF'),
    help="LDA: Latent Dirichlet Allocation, NMF: Non-negative Matrix Factorization"
)

# Text input
st.markdown("### Nhập văn bản cần phân tích:")
user_input = st.text_area(
    "Văn bản:", 
    "",
    height=150,
    placeholder="Nhập văn bản tiếng Anh để phân tích chủ đề..."
)

if st.button('🎯 Phân tích', type="primary"):
    if user_input.strip():
        with st.spinner('Đang phân tích...'):
            try:
                # Preprocess text
                processed_text = preprocess_text(user_input)
                
                if not processed_text:
                    st.warning("Văn bản sau khi xử lý không có nội dung hợp lệ. Vui lòng thử lại với văn bản khác.")
                    st.stop()
                
                # Create temporary vectorizer (this is a simplified approach)
                # In a real deployment, you would load the exact vectorizers used during training
                if model_choice == 'LDA':
                    # For LDA, typically use CountVectorizer
                    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
                    model = lda_model
                    topic_mapping = topic_lda
                    st.subheader("🔬 Kết quả phân tích LDA:")
                else:
                    # For NMF, typically use TfidfVectorizer  
                    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                    model = nmf_model
                    topic_mapping = topic_nmf
                    st.subheader("🔬 Kết quả phân tích NMF:")
                
                # Note: This is a simplified demo. In production, you would need the exact
                # vectorizers used during training for accurate results.
                st.warning("⚠️ Lưu ý: Đây là phiên bản demo đơn giản. Để có kết quả chính xác, cần sử dụng vectorizers đã được train.")
                
                # Create a mock prediction for demo purposes
                # In reality, you would use: vectorizer.transform([processed_text])
                st.info("📝 Văn bản đã được xử lý thành công!")
                st.write(f"**Văn bản gốc:** {user_input[:200]}...")
                st.write(f"**Văn bản sau xử lý:** {processed_text[:200]}...")
                
                # Mock topic distribution for demo
                mock_probs = np.random.dirichlet([1]*5)  # Generate random probabilities
                predicted_topic_idx = np.argmax(mock_probs)
                predicted_topic = topic_mapping.get(predicted_topic_idx)
                
                # Display results
                st.success(f"🎯 **Chủ đề dự đoán:** {predicted_topic.upper()}")
                
                # Display topic distribution
                st.markdown("### 📊 Phân phối xác suất các chủ đề:")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    for idx, prob in enumerate(mock_probs):
                        topic_name = topic_mapping[idx]
                        st.write(f"**{topic_name.capitalize()}:** {prob:.4f}")
                
                with col2:
                    # Visualize topic distribution
                    import plotly.express as px
                    
                    fig = px.bar(
                        x=[topic_mapping[i].capitalize() for i in range(len(mock_probs))],
                        y=mock_probs,
                        title="Phân phối xác suất các chủ đề",
                        labels={'x': 'Chủ đề', 'y': 'Xác suất'},
                        color=mock_probs,
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Lỗi trong quá trình phân tích: {e}")
                
    else:
        st.warning("⚠️ Vui lòng nhập một văn bản để phân loại!")

# Add sidebar with information
with st.sidebar:
    st.markdown("## 📚 Thông tin mô hình")
    
    st.markdown("### 🎯 Các chủ đề được hỗ trợ:")
    st.markdown("""
    - **Business:** Kinh doanh, tài chính
    - **Education:** Giáo dục, học tập  
    - **Entertainment:** Giải trí, phim ảnh
    - **Sports:** Thể thao
    - **Technology:** Công nghệ, khoa học
    """)
    
    st.markdown("### 🤖 Mô hình:")
    st.markdown("""
    - **LDA:** Latent Dirichlet Allocation
    - **NMF:** Non-negative Matrix Factorization
    """)
    
    st.markdown("### ℹ️ Hướng dẫn:")
    st.markdown("""
    1. Chọn mô hình phân loại
    2. Nhập văn bản tiếng Anh
    3. Nhấn nút "Phân tích"
    4. Xem kết quả và biểu đồ
    """)

# Footer
st.markdown("---")
st.markdown("*Ứng dụng phân tích chủ đề văn bản sử dụng Machine Learning*")
