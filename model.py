#Beginning of baseline model

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

#Step 3: vectorize words
#train Word2Vec model

#smaller sample
df_sample = df.sample(n=25000, random_state=42)
model = Word2Vec(sentences=df_sample['processed_comment'], vector_size = 100, window = 5, min_count = 1, workers = 4)

print("\n Model Trained")

#test
print("Vector for word 'university':")
print(model.wv['university'])

#creating comment vectors: average the vectors of all the words in the comment 
def comment_to_vector(comment, model):
    #list of word vectors
    vectors = [model.wv[word] for word in comment if word in model.wv.key_to_index]
    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)

#creating comment vectors
df_sample['comment_vector'] = df_sample['processed_comment'].apply(lambda x: comment_to_vector(x, model))

print("\nComment Vectors:")
print(df_sample[['comment', 'comment_vector']].head(5))

#Step 4: Building and Training the Classifier

#sklearn data prep
x = np.stack(df_sample['comment_vector'].values)
y = df_sample['label'].values

#test/train split
#80% train, 20% test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\nData Split Complete")
print(f"\n Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

#train logistic regression model
clf = LogisticRegression(max_iter = 1000)
clf.fit(x_train, y_train)
print("\nClassifier Training Complete")

# Step 5: Evaluating the Classifier

# Make predictions
y_pred = clf.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nBaseline Model Results")
print(f"\nTest Accuracy: {accuracy:.4f}")

#End of baseline model


'''
Iterative Research Plan

Goal: See which features improve the baseline model

'''

#model 2: Baseline comment + parent comment
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#preprocess parrent comments
df.dropna(subset=['parent_comment'], inplace=True)
df['processed_parent_comment'] = df['parent_comment'].apply(preprocess_text)

#create vectors
df['comment_vector'] = df['processed_comment'].apply(lambda x: comment_to_vector(x, model))
df['parent_comment_vector'] = df['processed_parent_comment'].apply(lambda x: comment_to_vector(x, model))

#combine vectors
df['combined_vector'] = [np.concatenate((c_vec, p_vec)) for c_vec, p_vec in zip(df['comment_vector'], df['parent_comment_vector'])]

#train and 