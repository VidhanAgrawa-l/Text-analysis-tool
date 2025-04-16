import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Import from utils
from utils.model_loader import load_kindle_model
from utils.preprocessing import preprocess_kindle_review

def run():
    # Set page configuration
    st.set_page_config(
        page_title="Kindle Review Analyzer",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Alternative approach using Streamlit's native controls
    if st.sidebar.button("Close Sidebar"):
        st.session_state.sidebar_state = "collapsed"
        st.rerun()
    
    # Custom CSS for theme compatibility
    st.markdown("""
    <style>
    /* Base styles that work in both light and dark themes */
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #FF9900;
        text-align: center;
        margin-bottom: 30px;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .subheader {
        font-size: 28px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 20px;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Universal card styling - works in both themes */
    .card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Light/dark mode adaptive styling for containers */
    [data-testid="stAppViewContainer"] .card {
        background-color: rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Sentiment results */
    .positive-result {
        background-color: rgba(0, 128, 0, 0.2);
        color: rgb(40, 167, 69);
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        border: 1px solid rgba(40, 167, 69, 0.3);
    }
    .negative-result {
        background-color: rgba(220, 53, 69, 0.2);
        color: rgb(220, 53, 69);
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        border: 1px solid rgba(220, 53, 69, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #FF9900;
        color: white;
        font-weight: bold;
        padding: 10px 25px;
        border: none;
        border-radius: 5px;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #E88A00;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Radio buttons */
    div.row-widget.stRadio > div {
        flex-direction: row;
        justify-content: center;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #FF9900;
    }
    
    /* Info boxes */
    .info-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 50px;
        opacity: 0.7;
        font-size: 14px;
    }
    
    /* Chart tweaks */
    [data-testid="stExpander"] {
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Specific tweaks for dark mode text */
    [data-testid="stMarkdownContainer"] ul li,
    [data-testid="stMarkdownContainer"] ol li {
        margin-bottom: 5px;
    }
    
    /* Sidebar sections */
    .sidebar-section {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Step headings */
    .step-heading {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        padding-bottom: 5px;
        border-bottom: 2px solid #FF9900;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with icon and title
    st.markdown('<div class="main-header">📚 Kindle Review Sentiment Analysis</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    This AI-powered tool analyzes Amazon Kindle reviews to determine if they express positive or negative sentiment.
    Try with your own review or select from example reviews below to see how the sentiment analysis works!
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create layout with columns for main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-heading">Step 1: Choose Your Model</div>', unsafe_allow_html=True)
        
        # Model selection with descriptions
        model_type = st.radio(
            "",
            ["bow", "tfidf"],
            horizontal=True,
            help="Select the model type for sentiment analysis"
        )
        
        # Model descriptions
        if model_type == "bow":
            st.info("**Bag of Words (BoW)** counts word occurrences in the review and is simpler but effective.")
        else:
            st.info("**TF-IDF** weighs words by their importance and often provides more nuanced results.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Load appropriate model
    try:
        model, vectorizer = load_kindle_model(model_type)
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_loaded = False
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-heading">Step 2: Enter or Select a Review</div>', unsafe_allow_html=True)
        
        # Example reviews
        example_reviews = [
            "This e-reader exceeded my expectations. The battery life is amazing and reading on it feels just like a real book.",
            "Disappointed with this Kindle model. The interface is confusing and it freezes regularly.",
            "Good value for money, but the screen resolution could be better.",
            "Absolutely love this! Best e-reader I've ever owned. The backlight is perfect for night reading.",
            "It's okay but not worth the premium price. Screen is nice but software is sluggish."
        ]
        
        # Select example or create custom review
        review_option = st.radio("", ["Enter custom review", "Select example review"])
        
        if review_option == "Enter custom review":
            user_review = st.text_area("Write your review here:", height=150, 
                                      placeholder="Example: The Kindle Paperwhite has changed how I read books...")
        else:
            example_idx = st.selectbox("Choose an example:", range(len(example_reviews)), 
                                     format_func=lambda i: example_reviews[i][:50] + "...")
            user_review = example_reviews[example_idx]
            st.text_area("Selected review:", user_review, height=150, disabled=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="step-heading">Step 3: Analyze Your Review</div>', unsafe_allow_html=True)
        analyze_button = st.button("📊 Analyze Sentiment")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Sidebar content moved to right column
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 18px; font-weight: bold;">About the Dataset</p>', unsafe_allow_html=True)
        st.markdown("""
        This model was trained on Amazon Kindle reviews with ratings from 1-5 stars:
        - 📈 **Positive reviews**: 4-5 stars (rating > 3)
        - 📉 **Negative reviews**: 1-2 stars (rating < 3)
        
        The AI learns language patterns associated with positive or negative sentiment.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 18px; font-weight: bold;">Tips for Better Results</p>', unsafe_allow_html=True)
        st.markdown("""
        - ✓ Be specific about features
        - ✓ Include emotional reactions
        - ✓ Mention comparisons to other products
        - ✓ Describe your usage experience
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add sample review distribution chart
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 18px; font-weight: bold;">Training Data Distribution</p>', unsafe_allow_html=True)
        
        # Create a chart that looks good in both light and dark mode
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # Set the figure background to transparent so it works in both themes
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        ratings = ['★', '★★', '★★★', '★★★★', '★★★★★']
        counts = [15, 20, 30, 55, 80]
        
        # Use colors that work in both themes
        colors = ['#ff9966', '#ff9966', '#ffcc99', '#66b3ff', '#66b3ff']
        bars = ax.bar(ratings, counts, color=colors)
        
        # Customize text colors to work in both themes
        ax.set_ylabel('Number of Reviews')
        ax.set_title('Review Rating Distribution')
        
        # Improve theme compatibility
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_alpha(0.3)
        ax.spines['left'].set_alpha(0.3)
        ax.tick_params(axis='x', colors='gray')
        ax.tick_params(axis='y', colors='gray')
        ax.yaxis.label.set_color('gray')
        ax.xaxis.label.set_color('gray')
        ax.title.set_color('gray')
        
        # Add text labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    str(counts[i]), ha='center', va='bottom', color='gray')
        
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis results section (will show after button click)
    if analyze_button and model_loaded and user_review:
        # Preprocess the review
        processed_review = preprocess_kindle_review(user_review)
        
        with st.spinner("AI is analyzing your review..."):
            # Add a slight delay for visual effect
            import time
            time.sleep(0.8)
            
            # Vectorize the review
            if model_type == "bow":
                review_vector = vectorizer.transform([processed_review]).toarray()
            elif model_type == "tfidf":
                review_vector = vectorizer.transform([processed_review]).toarray()
            
            # Make prediction
            prediction = model.predict(review_vector)[0]
            probability = model.predict_proba(review_vector)[0]
        
        # Display results section
        st.markdown('<div class="subheader">Analysis Results</div>', unsafe_allow_html=True)
        
        result_cols = st.columns([1, 1])
        
        with result_cols[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Display sentiment result
            if prediction == 1:
                st.markdown('<div class="positive-result">😊 POSITIVE REVIEW</div>', unsafe_allow_html=True)
                sentiment = "Positive"
                # Show confidence gauge for positive
                st.markdown(f"<p style='text-align: center; font-size: 18px;'>Confidence: {probability[1]:.1%}</p>", unsafe_allow_html=True)
                st.progress(float(probability[1]))
            else:
                st.markdown('<div class="negative-result">😔 NEGATIVE REVIEW</div>', unsafe_allow_html=True)
                sentiment = "Negative"
                # Show confidence gauge for negative
                st.markdown(f"<p style='text-align: center; font-size: 18px;'>Confidence: {probability[0]:.1%}</p>", unsafe_allow_html=True)
                st.progress(float(probability[0]))
            
            st.markdown(f"""
            <p style='margin-top: 20px;'>The AI model classified this as a <strong>{sentiment.upper()}</strong> review with:</p>
            <ul>
                <li>Positive probability: {probability[1]:.2%}</li>
                <li>Negative probability: {probability[0]:.2%}</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with result_cols[1]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Create sentiment probability chart with theme compatibility
            fig, ax = plt.subplots(figsize=(4, 3))
            
            # Set the figure background to transparent for theme compatibility
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            labels = ['Negative', 'Positive']
            bars = ax.bar(labels, [probability[0], probability[1]], color=['#ff9966', '#66b3ff'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            ax.set_title('Sentiment Probability')
            
            # Improve theme compatibility
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_alpha(0.3)
            ax.spines['left'].set_alpha(0.3)
            ax.tick_params(axis='x', colors='gray')
            ax.tick_params(axis='y', colors='gray')
            ax.yaxis.label.set_color('gray')
            ax.xaxis.label.set_color('gray')
            ax.title.set_color('gray')
            
            # Add text labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f"{[probability[0], probability[1]][i]:.2%}", 
                        ha='center', va='bottom', color='gray')
            
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Model explanation
        st.markdown('<div class="subheader">How the Analysis Works</div>', unsafe_allow_html=True)
        
        explanation_cols = st.columns([1, 1])
        
        with explanation_cols[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if model_type == "bow":
                st.markdown("""
                <p style='font-weight: bold; font-size: 18px;'>Bag of Words (BoW) Model</p>
                
                <p>This model works by:</p>
                <ol>
                    <li>Counting how many times each word appears in your review</li>
                    <li>Creating a numerical representation (vector) of your text</li>
                    <li>Comparing word patterns to those found in positive and negative reviews</li>
                    <li>Calculating the probability of the review being positive or negative</li>
                </ol>
                <p>This approach is simple but effective for sentiment analysis.</p>
                """, unsafe_allow_html=True)
            elif model_type == "tfidf":
                st.markdown("""
                <p style='font-weight: bold; font-size: 18px;'>TF-IDF Model</p>
                
                <p>This model works by:</p>
                <ol>
                    <li>Calculating Term Frequency (how often a word appears in your review)</li>
                    <li>Measuring Inverse Document Frequency (how unique the word is across all reviews)</li>
                    <li>Giving higher importance to distinctive words that strongly signal sentiment</li>
                    <li>Creating a weighted representation that better captures meaningful words</li>
                </ol>
                <p>This approach often provides more nuanced sentiment analysis.</p>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with explanation_cols[1]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Show preprocessed text
            st.markdown("<p style='font-weight: bold; font-size: 18px;'>Text Preprocessing</p>", unsafe_allow_html=True)
            
            st.markdown("""
            <p>Your review was preprocessed through these steps:</p>
            <ol>
                <li>Converting to lowercase</li>
                <li>Removing punctuation and special characters</li>
                <li>Removing URLs</li>
                <li>Removing stopwords (common words like 'the', 'and', etc.)</li>
                <li>Lemmatizing words (converting words to their base form)</li>
            </ol>
            """, unsafe_allow_html=True)
            
            with st.expander("See preprocessed text"):
                st.code(processed_review)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">Kindle Review Sentiment Analysis Tool • Created with Streamlit and Machine Learning</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    run()