import streamlit as st
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Main function to run the Streamlit application"""
    # Set up page configuration
    st.set_page_config(
        page_title="Text Analysis Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar configuration
    st.sidebar.image("https://img.icons8.com/fluency/96/text-analysis.png", width=80)
    st.sidebar.title("Navigation")
    st.sidebar.info("""
    Use the pages menu above to navigate to:
    - 📱 **Spam Detection**: Analyze SMS messages
    - 📚 **Kindle Analysis**: Analyze book reviews
    """)
    
    # Create a header with modern styling
    st.markdown("""
        <style>
            .header-container {
                display: flex;
                align-items: center;
                padding: 1rem;
                background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 10px;
                margin-bottom: 2rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .title-container {
                margin-left: 2rem;
            }
            .main-title {
                color: #1E3D59;
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            .subtitle {
                color: #666;
                font-size: 1.2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        # Using a local image instead of URL
        # First, create an 'assets' folder in your project directory and place the icon there
        try:
            st.image("assets/text-analysis-icon.png", width=150)
        except:
            # Fallback icon using Unicode character
            st.markdown("""
                <div style="font-size: 80px; text-align: center; color: #1E3D59;">
                    📊
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="title-container">
                <div class="main-title">TEXT ANALYSIS TOOL</div>
                <div class="subtitle">Advanced Text Analysis & Classification Tool</div>
            </div>
        """, unsafe_allow_html=True)

    # Add a decorative separator
    st.markdown("""
        <div style="height: 3px; background: linear-gradient(90deg, #1E3D59 0%, #ff6b6b 100%); margin: 20px 0;"></div>
    """, unsafe_allow_html=True)
    
    # Main content in tabs
    tab1, tab2, tab3 = st.tabs(["📋 Overview", "🔍 Features", "📊 Models"])
    
    with tab1:
        st.header("Welcome to the Text Analysis Tool!")
        st.markdown("""
        This intelligent text analysis tool uses advanced machine learning techniques to analyze and classify text content.
        Whether you're trying to identify spam messages or analyze sentiment in Kindle reviews, our tool provides 
        accurate and insightful analysis with easy-to-understand visualizations.
        
        ### Getting Started
        
        1. Select a feature from the sidebar menu
        2. Choose your preferred analysis model
        3. Enter text or select from examples
        4. Get instant analysis with detailed explanations
        """)
        
        # Add a sample visualization to make the page more engaging
        st.subheader("Text Analysis in Action")
        col1, col2 = st.columns(2)
        
        with col1:
            # Create a simple demo chart
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = ['Spam', 'Ham']
            values = [35, 65]
            colors = ['#FF9999', '#66B2FF']
            ax.bar(categories, values, color=colors)
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Sample Dataset Composition')
            for i, v in enumerate(values):
                ax.text(i, v + 1, f"{v}%", ha='center')
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            ### Applications
            
            Our text analysis tools can be used for:
            
            - 🛡️ **Filtering unwanted messages** in communication systems
            - 📊 **Understanding customer sentiment** from reviews
            - 📝 **Exploring text patterns** in different contexts
            - 🔍 **Identifying key features** in written content
            
            The modular design allows for easy expansion to additional text analysis tasks.
            """)
    
    with tab2:
        st.header("Available Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📱 Spam Detection
            
            **Available now**
            
            - Detect spam in SMS messages
            - Choose from three different models
            - Get detailed probability scores
            - See model explanations
            - Try with example messages or your own text
            
            *Access this feature from the sidebar navigation*
            """)
            
            # Create a simple visual example
            st.markdown("#### Sample Analysis")
            sample_text = "Congratulations! You've won a free gift card worth $500! Click here to claim your prize now!"
            st.code(sample_text)
            st.success("🚨 Detected as: SPAM (Confidence: 96%)")
        
        with col2:
            st.markdown("""
            ### 📚 Kindle Review Analysis
            
            **Available now**
            
            - Analyze sentiment in Kindle reviews
            - Determine if reviews are positive or negative
            - Choose from different analysis models
            - Get detailed probability breakdown
            - Try with example reviews or your own text
            
            *Access this feature from the sidebar navigation*
            """)
            
            # Create a simple visual example
            st.markdown("#### Sample Analysis")
            sample_text = "This Kindle Paperwhite is amazing! The battery lasts for weeks and the screen looks just like real paper. Best e-reader I've owned."
            st.code(sample_text)
            st.success("😊 Detected as: POSITIVE (Confidence: 92%)")
    
    with tab3:
        st.header("Understanding Our Models")
        
        st.markdown("""
        This tool uses several Natural Language Processing (NLP) models with different approaches to text analysis:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### Bag of Words (BOW)
            
            A simple representation that counts word occurrences in text.
            
            **Strengths:**
            - Simple and intuitive
            - Fast processing
            - Works well for basic classification
            
            **Best for:**
            - Short text messages
            - Well-defined categories
            """)
        
        with col2:
            st.markdown("""
            ### TF-IDF
            
            Term Frequency-Inverse Document Frequency weights words based on their importance in the text.
            
            **Strengths:**
            - Better at identifying important words
            - Reduces impact of common words
            - More nuanced than BOW
            
            **Best for:**
            - Longer text documents
            - Content with varied vocabulary
            """)
        
        with col3:
            st.markdown("""
            ### Word2Vec
            
            Uses neural networks to represent words as vectors in semantic space.
            
            **Strengths:**
            - Captures semantic relationships
            - Understands word context
            - Best for nuanced language
            
            **Best for:**
            - Complex text analysis
            - Understanding context
            - Finding semantic patterns
            """)
        
        # Add a comparison chart
        st.subheader("Model Performance Comparison")
        model_data = pd.DataFrame({
            'Model': ['Bag of Words', 'TF-IDF', 'Word2Vec'],
            'Accuracy': [0.89, 0.92, 0.95],
            'Processing Speed': [0.95, 0.85, 0.70],
            'Complexity': [0.30, 0.65, 0.95]
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(model_data['Model']))
        width = 0.25
        
        ax.bar(x - width, model_data['Accuracy'], width, label='Accuracy', color='#66B2FF')
        ax.bar(x, model_data['Processing Speed'], width, label='Speed', color='#99FF99')
        ax.bar(x + width, model_data['Complexity'], width, label='Complexity', color='#FFCC99')
        
        ax.set_xticks(x)
        ax.set_xticklabels(model_data['Model'])
        ax.set_ylabel('Score (0-1)')
        ax.set_title('Model Comparison')
        ax.legend()
        
        st.pyplot(fig)
    
    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    footer_col1, footer_col2 = st.columns([3, 1])
    
    with footer_col1:
        st.markdown("Advanced Text Analysis Tool | Powered by Machine Learning & VIDHAN AGRAWAL")
    
    with footer_col2:
        st.text("© 2025")

if __name__ == "__main__":
    main()