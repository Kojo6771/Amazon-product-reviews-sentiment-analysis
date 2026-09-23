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

#importing dataset
data = pd.read_csv("Amazon-Product-Reviews-Sentiment-Analysis-in-Python-Dataset.csv")
data.head()



# Dropping rows with missing values
data.dropna(inplace=True)
data.info()

#1,2,3->negative(i.e 0)
data.loc[data['Sentiment']<=3, 'Sentiment'] = 0
#4,5->positive(i.e 1)
data.loc[data['Sentiment']>3, 'Sentiment'] = 1

# Cleaning the reviews by removing stopwords
stp_words=stopwords.words('english')
def clean_review(review):
    cleanreview="".join(word for word in review.split() if word not in stp_words)
    return cleanreview
data['Review']=data['Review'].apply(clean_review)
data.head()

data['Sentiment'].value_counts()

# Visualizing the distribution of sentiments
consolidated=''.join(word for word in data['Review'][data['Sentiment']==0].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()

# Visualizing the positive sentiment word cloud
consolidated_positive=''.join(word for word in data['Review'][data['Sentiment']==1].astype(str))
wordCloud_positive=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud_positive.generate(consolidated_positive),interpolation='bilinear')
plt.axis('off')
plt.show()