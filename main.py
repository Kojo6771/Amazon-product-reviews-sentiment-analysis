#Library imports
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# NLTK setup for text processing
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


