"""
Text preprocessing and feature engineering module.
Handles text cleaning, tokenization, vectorization, and feature extraction.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union
import logging

# Text processing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from textblob import TextBlob
from wordcloud import WordCloud

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Text preprocessing class for cleaning and normalizing text data.
    """
    
    def __init__(self, 
                 remove_html: bool = True,
                 remove_urls: bool = True,
                 remove_special_chars: bool = True,
                 convert_lowercase: bool = True,
                 remove_stopwords: bool = True,
                 apply_stemming: bool = False,
                 apply_lemmatization: bool = True,
                 min_word_length: int = 2):
        """
        Initialize text preprocessor with configuration options.
        
        Args:
            remove_html (bool): Remove HTML tags
            remove_urls (bool): Remove URLs
            remove_special_chars (bool): Remove special characters and punctuation
            convert_lowercase (bool): Convert text to lowercase
            remove_stopwords (bool): Remove stopwords
            apply_stemming (bool): Apply stemming
            apply_lemmatization (bool): Apply lemmatization
            min_word_length (int): Minimum word length to keep
        """
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_special_chars = remove_special_chars
        self.convert_lowercase = convert_lowercase
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming
        self.apply_lemmatization = apply_lemmatization
        self.min_word_length = min_word_length
        
        # Initialize NLTK components
        self._download_nltk_data()
        self.stemmer = PorterStemmer() if apply_stemming else None
        self.lemmatizer = WordNetLemmatizer() if apply_lemmatization else None
        
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = set()
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('chunkers/maxent_ne_chunker')
            nltk.data.find('corpora/words')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
    
    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning steps to a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        if self.remove_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Convert to lowercase
        if self.convert_lowercase:
            text = text.lower()
        
        # Remove special characters and digits
        if self.remove_special_chars:
            text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        words = word_tokenize(text)
        
        # Filter words
        words = [word for word in words if len(word) >= self.min_word_length]
        
        # Remove stopwords
        if self.remove_stopwords:
            words = [word for word in words if word not in self.stop_words]
        
        # Apply stemming or lemmatization
        if self.apply_stemming and self.stemmer:
            words = [self.stemmer.stem(word) for word in words]
        elif self.apply_lemmatization and self.lemmatizer:
            words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[str]: List of preprocessed texts
        """
        logger.info(f"Preprocessing {len(texts)} texts...")
        processed_texts = [self.clean_text(text) for text in texts]
        logger.info("Text preprocessing completed")
        return processed_texts


class FeatureExtractor:
    """
    Feature extraction class for converting text to numerical features.
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.svd_transformer = None
        
    def extract_tfidf_features(self, 
                             texts: List[str], 
                             max_features: int = 10000,
                             ngram_range: Tuple[int, int] = (1, 2),
                             min_df: int = 2,
                             max_df: float = 0.95) -> Tuple[np.ndarray, TfidfVectorizer]:
        """
        Extract TF-IDF features from texts.
        
        Args:
            texts (List[str]): List of texts
            max_features (int): Maximum number of features
            ngram_range (Tuple[int, int]): N-gram range
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency
            
        Returns:
            Tuple[np.ndarray, TfidfVectorizer]: Features and fitted vectorizer
        """
        logger.info(f"Extracting TF-IDF features with max_features={max_features}")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        
        logger.info(f"TF-IDF feature matrix shape: {tfidf_features.shape}")
        return tfidf_features.toarray(), self.tfidf_vectorizer
    
    def extract_count_features(self,
                             texts: List[str],
                             max_features: int = 5000,
                             ngram_range: Tuple[int, int] = (1, 2),
                             min_df: int = 2,
                             max_df: float = 0.95) -> Tuple[np.ndarray, CountVectorizer]:
        """
        Extract count-based features from texts.
        
        Args:
            texts (List[str]): List of texts
            max_features (int): Maximum number of features
            ngram_range (Tuple[int, int]): N-gram range
            min_df (int): Minimum document frequency
            max_df (float): Maximum document frequency
            
        Returns:
            Tuple[np.ndarray, CountVectorizer]: Features and fitted vectorizer
        """
        logger.info(f"Extracting count features with max_features={max_features}")
        
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        
        count_features = self.count_vectorizer.fit_transform(texts)
        
        logger.info(f"Count feature matrix shape: {count_features.shape}")
        return count_features.toarray(), self.count_vectorizer
    
    def apply_dimensionality_reduction(self,
                                     features: np.ndarray,
                                     n_components: int = 300) -> Tuple[np.ndarray, TruncatedSVD]:
        """
        Apply dimensionality reduction using SVD.
        
        Args:
            features (np.ndarray): Input features
            n_components (int): Number of components to keep
            
        Returns:
            Tuple[np.ndarray, TruncatedSVD]: Reduced features and fitted transformer
        """
        logger.info(f"Applying SVD dimensionality reduction to {n_components} components")
        
        self.svd_transformer = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_features = self.svd_transformer.fit_transform(features)
        
        explained_variance = self.svd_transformer.explained_variance_ratio_.sum()
        logger.info(f"SVD explained variance ratio: {explained_variance:.3f}")
        
        return reduced_features, self.svd_transformer


class SentimentFeatures:
    """
    Class for extracting sentiment-specific features.
    """
    
    @staticmethod
    def extract_sentiment_features(texts: List[str]) -> pd.DataFrame:
        """
        Extract sentiment-related features from texts.
        
        Args:
            texts (List[str]): List of texts
            
        Returns:
            pd.DataFrame: DataFrame with sentiment features
        """
        logger.info("Extracting sentiment features...")
        
        features = []
        
        for text in texts:
            # Basic text statistics
            text_length = len(text)
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.!?]+', text))
            avg_word_length = np.mean([len(word) for word in text.split()]) if text.split() else 0
            
            # Sentiment polarity and subjectivity using TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Punctuation and capitalization features
            exclamation_count = text.count('!')
            question_count = text.count('?')
            uppercase_count = sum(1 for c in text if c.isupper())
            
            # Emotional words (simple approach)
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 
                            'fantastic', 'awesome', 'brilliant', 'perfect', 'love']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disgusting',
                            'worst', 'hate', 'stupid', 'boring', 'disappointing']
            
            positive_word_count = sum(1 for word in positive_words if word in text.lower())
            negative_word_count = sum(1 for word in negative_words if word in text.lower())
            
            features.append({
                'text_length': text_length,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_word_length': avg_word_length,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'exclamation_count': exclamation_count,
                'question_count': question_count,
                'uppercase_ratio': uppercase_count / text_length if text_length > 0 else 0,
                'positive_word_count': positive_word_count,
                'negative_word_count': negative_word_count,
                'sentiment_word_ratio': (positive_word_count + negative_word_count) / word_count if word_count > 0 else 0
            })
        
        features_df = pd.DataFrame(features)
        logger.info(f"Extracted {features_df.shape[1]} sentiment features")
        
        return features_df


def create_word_cloud(texts: List[str], title: str = "Word Cloud", save_path: Optional[str] = None):
    """
    Create and display a word cloud from texts.
    
    Args:
        texts (List[str]): List of texts
        title (str): Title for the word cloud
        save_path (Optional[str]): Path to save the word cloud image
    """
    # Combine all texts
    combined_text = ' '.join(texts)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(combined_text)
    
    # Display
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()


def main():
    """
    Demonstrate feature extraction functionality.
    """
    # Sample texts for demonstration
    sample_texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Terrible film. Waste of time and money. Very disappointing.",
        "An okay movie. Nothing special but not bad either."
    ]
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess texts
    processed_texts = preprocessor.preprocess_texts(sample_texts)
    logger.info("Sample processed texts:")
    for i, text in enumerate(processed_texts):
        logger.info(f"{i+1}: {text}")
    
    # Extract features
    feature_extractor = FeatureExtractor()
    
    # TF-IDF features
    tfidf_features, tfidf_vectorizer = feature_extractor.extract_tfidf_features(
        processed_texts, max_features=100
    )
    logger.info(f"TF-IDF features shape: {tfidf_features.shape}")
    
    # Sentiment features
    sentiment_features = SentimentFeatures.extract_sentiment_features(sample_texts)
    logger.info(f"Sentiment features:\n{sentiment_features}")


if __name__ == "__main__":
    main()