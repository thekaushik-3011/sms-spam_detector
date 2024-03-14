import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


df = pd.read_csv('spam.csv', encoding='latin1')

"""# **Data Cleaning**"""

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)

le = LabelEncoder()

df['target'] = le.fit_transform(df['target'])

df = df.drop_duplicates(keep='first')

"""# **EDA**"""
df['num_char'] = df['text'].apply(len)

df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))

df['num_sent'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

"""# **Data Preprocessing**"""
ps = PorterStemmer()

def transform_text(text):
  text = text.lower() # Convert to lowercase
  text = nltk.word_tokenize(text) # Tokenization

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)  # Remove all special characters

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i)) # Stemming

  return " ".join(y)

transform_text(df['text'][0])

df['transformed_text'] = df['text'].apply(transform_text)

spam_corpus = []

for msg in df[df['target'] == 1]['transformed_text'].to_list():
  for word in msg.split():
    spam_corpus.append(word)

data = pd.DataFrame(Counter(spam_corpus).most_common(30))

ham_corpus = []

for msg in df[df['target'] == 0]['transformed_text'].to_list():
  for word in msg.split():
    ham_corpus.append(word)

data = pd.DataFrame(Counter(ham_corpus).most_common(30))


"""# **Model Building**"""

tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

mnb = MultinomialNB()

mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))