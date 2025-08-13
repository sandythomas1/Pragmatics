#step 1: setting up my enviroment 

# Import libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

#download NLTK data (only need to run once)
'''
nltk.download('punkt')
nltk.download('stopwords')
'''

data = pd.read_csv('datasets/train-balanced-sarcasm.csv')
df = pd.DataFrame(data)

#drop any rows with missing comments 
df.dropna(subset=['comment'], inplace=True)

#test to make sure data is loaded
print(df.head(5))

#Step 2: text Preprocessing

#get list of english stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    #cleans and tokenizes text

    #lowercase text
    text = text.lower()
    #tokenize text
    tokens = word_tokenize(text)
    #remove punctuation
    tokens_processed = [word for word in tokens if word.isalpha() and word not in stop_words]

    return tokens_processed

#apply preprocessing to the 'comment' column
df['processed_comment'] = df['comment'].apply(preprocess_text)

print("\n")
print("Data after preprocessing:")
print(df[['comment', 'processed_comment']].head(5))
