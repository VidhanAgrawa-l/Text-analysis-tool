# import streamlit as st
# import pandas as pd
# import time
# import os
# import sys

# # Add parent directory to path to import utils
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from utils.preprocessing import preprocess_text
# from utils.model_loader import ModelLoader

# def load_models():
#     """Load all the NLP models for spam detection"""
#     return ModelLoader()

# def predict_spam_probability(text, model_type, model_loader):
#     """
#     Predict if a message is spam using the selected model
    
#     Args:
#         text (str): Input text
#         model_type (str): Type of model to use
#         model_loader (ModelLoader): Model loader instance
        
#     Returns:
#         tuple: (is_spam (bool), probability (float))
#     """
#     # Preprocess text based on model type
#     if model_type == "word2vec":
#         processed_text = preprocess_text(text, stem=False)  # Use lemmatization for Word2Vec
#     else:
#         processed_text = preprocess_text(text, stem=True)  # Use stemming for BOW/TF-IDF
    
#     # Make prediction
#     prediction, probability = model_loader.predict(processed_text, model_type)
    
#     return bool(prediction), probability

# def main():
#     """Main function for the spam detection page"""
#     st.title("📱 Spam Detection")
#     st.markdown("""
#     This tool analyzes text messages to determine if they are spam or legitimate (ham).
    
#     Enter a message below and select a model to analyze it:
#     """)
    
#     # Load models
#     try:
#         with st.spinner("Loading models..."):
#             model_loader = load_models()
#         st.success("Models loaded successfully!")
#     except Exception as e:
#         st.error(f"Error loading models: {e}")
#         st.stop()
    
#     # Model selection
#     model_type = st.selectbox(
#         "Select Model",
#         ["bow", "tfidf", "word2vec"],
#         format_func=lambda x: {
#             "bow": "Bag of Words", 
#             "tfidf": "TF-IDF", 
#             "word2vec": "Word2Vec"
#         }[x],
#         index=1  # Default to TF-IDF
#     )
    
#     # Description of selected model
#     model_descriptions = {
#         "bow": """
#             **Bag of Words (BOW)** represents text as the frequency of each word, ignoring grammar and word order.
#             It's simple but effective for many text classification tasks.
#             """,
#         "tfidf": """
#             **TF-IDF (Term Frequency-Inverse Document Frequency)** weights words by their importance in the document
#             compared to their frequency in the entire corpus. It often performs better than BOW for spam detection.
#             """,
#         "word2vec": """
#             **Word2Vec** represents words as vectors in a semantic space where similar words are closer together.
#             This model captures more semantic meaning than BOW or TF-IDF.
#             """
#     }
#     st.info(model_descriptions[model_type])
    
#     # Text input
#     st.subheader("Enter a message")
#     text_input = st.text_area("", height=150)
    
#     # Example messages
#     with st.expander("Show example messages"):
#         examples = {
#             "Spam Example 1": "URGENT! You have won a 1-week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
#             "Spam Example 2": "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
#             "Ham Example 1": "Hey, what time should we meet for dinner tonight?",
#             "Ham Example 2": "Just wanted to remind you about tomorrow's meeting at 10 AM. Please bring your laptop."
#         }
        
#         for name, example in examples.items():
#             if st.button(name):
#                 text_input = example
#                 st.session_state.text_input = example
#                 st.rerun()
    
#     # Store text input in session state to persist after button clicks
#     if 'text_input' in st.session_state and not text_input:
#         text_input = st.session_state.text_input
    
#     # Analysis button
#     if st.button("Analyze") or text_input:
#         if text_input:
#             st.session_state.text_input = text_input
            
#             with st.spinner("Analyzing..."):
#                 # Add a small delay to show the spinner (optional)
#                 time.sleep(0.5)
                
#                 # Make prediction
#                 is_spam, probability = predict_spam_probability(text_input, model_type, model_loader)
                
#                 # Display results
#                 st.subheader("Analysis Result")
                
#                 # Create columns for displaying result
#                 col1, col2 = st.columns([1, 3])
                
#                 # Display classification result
#                 with col1:
#                     if is_spam:
#                         st.error("SPAM")
#                     else:
#                         st.success("HAM (Not Spam)")
                
#                 # Display probability
#                 with col2:
#                     probability_pct = probability * 100
#                     if is_spam:
#                         st.markdown(f"**Confidence:** {probability_pct:.2f}% likely to be spam")
#                     else:
#                         st.markdown(f"**Confidence:** {probability_pct:.2f}% likely to be legitimate")
                
#                 # Show probability gauge
#                 st.progress(probability)
                
#                 # Show explanation
#                 st.subheader("What does this mean?")
#                 if is_spam:
#                     st.markdown("""
#                     This message has been classified as **spam**. Such messages typically:
#                     - Contain urgent or promotional language
#                     - May include offers or prizes
#                     - Often have unusual punctuation or capitalization
#                     - Frequently contain URLs or phone numbers
#                     """)
#                 else:
#                     st.markdown("""
#                     This message has been classified as **ham** (not spam). Such messages typically:
#                     - Have natural language patterns
#                     - Lack promotional content
#                     - Contain personal references or conversational elements
#                     """)
#         else:
#             st.warning("Please enter a message to analyze")

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import time
import os
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import preprocess_text
from utils.model_loader import ModelLoader

# Set page configuration
st.set_page_config(
    page_title="Spam Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-top: 0;
        padding-top: 0;
        margin-bottom: 2rem;
    }
    .stProgress > div > div {
        background-image: linear-gradient(to right, #10B981, #EF4444);
    }
    .result-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .spam-badge {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        text-align: center;
    }
    .ham-badge {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #6B7280;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #E5E7EB;
    }
    .model-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .analyze-button {
        text-align: center;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
    }
    .examples-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def load_models():
    """Load all the NLP models for spam detection"""
    return ModelLoader()

def predict_spam_probability(text, model_type, model_loader):
    """
    Predict if a message is spam using the selected model
    
    Args:
        text (str): Input text
        model_type (str): Type of model to use
        model_loader (ModelLoader): Model loader instance
        
    Returns:
        tuple: (is_spam (bool), probability (float))
    """
    # Preprocess text based on model type
    if model_type == "word2vec":
        processed_text = preprocess_text(text, stem=False)  # Use lemmatization for Word2Vec
    else:
        processed_text = preprocess_text(text, stem=True)  # Use stemming for BOW/TF-IDF
    
    # Make prediction
    prediction, probability = model_loader.predict(processed_text, model_type)
    
    return bool(prediction), probability

def main():
    """Main function for the spam detection page"""
    # Title with custom styling
    st.markdown('<h1 class="main-title">🛡️ Spam Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Analyze messages to identify spam or legitimate content</p>', unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 🔍 Model Configuration")
        
        # Load models with better styling
        try:
            with st.spinner("⏳ Loading models..."):
                model_loader = load_models()
            st.success("✅ Models loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            st.stop()
        
        # Model selection with better styling
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        model_type = st.selectbox(
            "Select Analysis Model",
            ["bow", "tfidf", "word2vec"],
            format_func=lambda x: {
                "bow": "🔤 Bag of Words", 
                "tfidf": "📊 TF-IDF", 
                "word2vec": "🔠 Word2Vec"
            }[x],
            index=1  # Default to TF-IDF
        )
        
        # Description of selected model
        model_descriptions = {
            "bow": """
                **Bag of Words (BOW)** represents text as the frequency of each word, ignoring grammar and word order.
                It's simple but effective for many text classification tasks.
                """,
            "tfidf": """
                **TF-IDF (Term Frequency-Inverse Document Frequency)** weights words by their importance in the document
                compared to their frequency in the entire corpus. It often performs better than BOW for spam detection.
                """,
            "word2vec": """
                **Word2Vec** represents words as vectors in a semantic space where similar words are closer together.
                This model captures more semantic meaning than BOW or TF-IDF.
                """
        }
        st.info(model_descriptions[model_type])
        st.markdown('</div>', unsafe_allow_html=True)


        
        
        # Example messages with better styling
        st.markdown("### 📝 Example Messages")
        with st.expander("Show example messages", expanded=True):
            examples = {
                "🔴 Spam Example 1": "URGENT! You have won a 1-week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
                "🔴 Spam Example 2": "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
                "🟢 Ham Example 1": "Hey, what time should we meet for dinner tonight?",
                "🟢 Ham Example 2": "Just wanted to remind you about tomorrow's meeting at 10 AM. Please bring your laptop."
            }
            
            st.markdown('<div class="examples-grid">', unsafe_allow_html=True)
            for name, example in examples.items():
                if st.button(name, key=f"btn_{name}"):

                    st.session_state.text_input = example
                    st.session_state.message_input = example



                    # st.session_state.text_input = example
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col1:
        # Text input
        st.markdown("### ✏️ Enter a message")
        text_input = st.text_area("", height=150, key="message_input")
        
        # Store text input in session state to persist after button clicks
        if 'text_input' in st.session_state and not text_input:
            text_input = st.session_state.text_input
        
        # Analysis button with better styling
        st.markdown('<div class="analyze-button">', unsafe_allow_html=True)
        analyze_button = st.button("🔍 Analyze Message", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if analyze_button or text_input:
            if text_input:
                st.session_state.text_input = text_input
                
                with st.spinner("🔄 Analyzing message..."):
                    # Add a small delay to show the spinner (optional)
                    time.sleep(0.5)
                    
                    # Make prediction
                    is_spam, probability = predict_spam_probability(text_input, model_type, model_loader)
                    
                    # Display results with better styling
                    st.markdown("### 📊 Analysis Result")
                    
                    # Result card with background color based on classification
                    result_bg_color = "#FEF2F2" if is_spam else "#ECFDF5"
                    st.markdown(f'<div class="result-card" style="background-color: {result_bg_color};">', unsafe_allow_html=True)
                    
                    # Create columns for displaying result
                    res_col1, res_col2 = st.columns([1, 3])
                    
                    # Display classification result with badge
                    with res_col1:
                        if is_spam:
                            st.markdown('<div class="spam-badge">SPAM</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="ham-badge">HAM</div>', unsafe_allow_html=True)
                    
                    # Display probability
                    with res_col2:
                        probability_pct = probability * 100
                        if is_spam:
                            st.markdown(f"**Confidence:** {probability_pct:.2f}% likely to be spam")
                        else:
                            st.markdown(f"**Confidence:** {100-probability_pct:.2f}% likely to be legitimate")
                    
                    # Show probability gauge
                    st.progress(probability)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show explanation with better styling
                    st.markdown("### 📋 What does this mean?")
                    if is_spam:
                        st.markdown("""
                        This message has been classified as **spam**. Such messages typically:
                        - 🚨 Contain urgent or promotional language
                        - 🎁 May include offers or prizes
                        - ❗ Often have unusual punctuation or capitalization
                        - 📞 Frequently contain URLs or phone numbers
                        """)
                    else:
                        st.markdown("""
                        This message has been classified as **ham** (not spam). Such messages typically:
                        - 💬 Have natural language patterns
                        - 🙂 Lack promotional content
                        - 👋 Contain personal references or conversational elements
                        - 📅 Often include personal context or scheduling information
                        """)
            else:
                st.warning("⚠️ Please enter a message to analyze")
    
    # Footer
    st.markdown('<div class="footer">Spam Detection Tool • Powered by Machine Learning</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()