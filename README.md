# Text Analysis Tool

A comprehensive text analysis application built with Streamlit that offers multiple NLP features.

## Features

### Current Features:
- **Spam Detection**: Analyze text messages to determine if they are spam or legitimate, using multiple NLP models:
  - Bag of Words (BOW)
  - TF-IDF
  - Word2Vec

### Coming Soon:
- **Kindle Analysis**: Analyze text from Kindle books and documents

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/text-analysis-tool.git
cd text-analysis-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

4. Train and save models (if needed):
```bash
python train_and_save_models.py
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Navigate to the displayed URL (typically http://localhost:8501)

3. Select the desired analysis tool from the sidebar and follow the instructions

## Project Structure

```
text_analysis_tool/
│
├── app.py                     # Main Streamlit application entry point
├── models/                    # Pre-trained models and vectorizers
├── pages/                     # Streamlit pages
├── utils/                     # Utility functions
├── data/                      # Data storage
├── train_and_save_models.py   # Training script
├── requirements.txt           # Project dependencies
└── README.md                  # Documentation
```

## Adding New Features

To add new analysis capabilities:
1. Create a new page in the `pages/` directory
2. Implement the necessary preprocessing and model functions in `utils/`
3. Update the main app description if needed

## License

[MIT License](LICENSE)