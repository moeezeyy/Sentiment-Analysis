# Multimodal Sentiment Analysis

This project performs sentiment analysis on social media posts by analyzing both images and associated text captions. It uses machine learning to classify posts into sentiment categories (very positive, positive, neutral, negative, very negative).

## Features
- Processes images by converting to grayscale and flattening pixel values
- Cleans and vectorizes text data using CountVectorizer
- Implements three classifiers (KNN, Decision Tree, Random Forest) for both image and text analysis
- Handles class imbalance using RandomOverSampler
- Provides accuracy and F1 score metrics for model evaluation

## Dataset
- Contains 6,992 social media posts with images and text captions
- Each post labeled with one of five sentiment categories

## Results
- Best image classifier: Random Forest (83.4% accuracy)
- Best text classifier: Random Forest (77.5% accuracy)
- Combined overall performance: ~59.8% accuracy

## Requirements
- Python 3.10+
- Libraries:
  - scikit-learn
  - pandas
  - numpy
  - nltk
  - Pillow
  - scikit-image
  - imbalanced-learn

## Installation
```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
