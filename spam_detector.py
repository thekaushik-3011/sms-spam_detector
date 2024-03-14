import numpy as np
import pandas as pd

df = pd.read_csv('spam.csv', encoding='latin1')
df.head()

df.shape

"""# **Data Cleaning**"""

df.info()

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.head()

df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)

df.sample(5)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['target'] = le.fit_transform(df['target'])
df.head()

df.isnull().sum()

df.duplicated().sum()

df = df.drop_duplicates(keep='first')

df.shape

"""# **EDA**"""

df.head()

df.target.value_counts()

import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['hame', 'spam'], autopct='%0.2f')
plt.show()

import nltk

nltk.download('punkt')

df['num_char'] = df['text'].apply(len)
df.head()

df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))

df['num_sent'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

df.head()

df[['num_char', 'num_words', 'num_sent']].describe()

df[df['target'] == 0][['num_char', 'num_words', 'num_sent']].describe() # For ham messages

df[df['target'] == 1][['num_char', 'num_words', 'num_sent']].describe()  # for spam message

import seaborn as sns

sns.histplot(df[df['target'] == 0]['num_char'])
sns.histplot(df[df['target'] == 1]['num_char'], color='red')

sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'], color='red')

sns.pairplot(df, hue='target')

sns.heatmap(df.corr(), annot=True)

"""# **Data Preprocessing**"""

nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords.words('english')

import string
string.punctuation

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')

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

df.head()

from wordcloud import WordCloud

wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))

plt.imshow(spam_wc)

ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.imshow(ham_wc)

df.head()

spam_corpus = []

for msg in df[df['target'] == 1]['transformed_text'].to_list():
  for word in msg.split():
    spam_corpus.append(word)

len(spam_corpus)

from collections import Counter
data = pd.DataFrame(Counter(spam_corpus).most_common(30))
sns.barplot(x=data[0], y=data[1])
plt.xticks(rotation='vertical')
plt.show()

ham_corpus = []

for msg in df[df['target'] == 0]['transformed_text'].to_list():
  for word in msg.split():
    ham_corpus.append(word)

len(ham_corpus)

from collections import Counter
data = pd.DataFrame(Counter(ham_corpus).most_common(30))
sns.barplot(x=data[0], y=data[1])
plt.xticks(rotation='vertical')
plt.show()

"""# **Model Building**"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

X = cv.fit_transform(df['transformed_text']).toarray()

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

X.shape

y = df['target'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))

bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))

X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))

bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))

# WE chose tfidf and mnb because in imbalanced data precision matters more than accuracy

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

svc = SVC(kernel='sigmoid', gamma=1.0)
knn = KNeighborsClassifier()
mnb = MultinomialNB()
dt = DecisionTreeClassifier(max_depth=5)
lr =LogisticRegression(solver='liblinear', penalty='l1')
rf = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC' : svc,
    'KNN' : knn,
    'MNB' : mnb,
    'DT' : dt,
    'LR' : lr,
    'RF' : rf,
    'ABC' : abc,
    'BC' : bc,
    'ETC' : etc,
    'GBDT' : gbdt,
    'XGB' : xgb
}

def train_classifier(clf, X_train, y_train, X_test,  y_test):
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)

  return accuracy, precision

accuracy_scores = []
precision_scores = []

for name, clf in clfs.items():
  current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)

  print("For", name)
  print("Accuracy: ", current_accuracy)
  print("Precision: ", current_precision)

  accuracy_scores.append(current_accuracy)
  precision_scores.append(current_precision)

performance_df = pd.DataFrame({'Algorithm' : clfs.keys(), 'Accuracy' : accuracy_scores, 'Precision' : precision_scores}).sort_values('Precision', ascending=False)

performance_df

performance_df1 = pd.melt(performance_df, id_vars="Algorithm")

sns.catplot(x = 'Algorithm', y = 'value',
            hue = 'variable', data=performance_df1, kind='bar', height=5)
plt.ylim(0.5, 1.0)
plt.xticks(rotation='vertical')
plt.show()

# Model improvement

temp_df = pd.DataFrame({'Algorithm' : clfs.keys(), 'Accuracy_max_ft_3000' : accuracy_scores, 'Precision_max_ft_3000' : precision_scores})

temp_df2 = pd.DataFrame({'Algorithm' : clfs.keys(), 'Accuracy_scaled' : accuracy_scores, 'Precision_scaled' : precision_scores})

new_df = performance_df.merge(temp_df, on='Algorithm')

new_df

new_df_scaled = new_df.merge(temp_df2, on='Algorithm')

new_df_scaled

# Voting Classifer
svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)], voting='soft')

voting.fit(X_train, y_train)

y_pred = voting.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))

"""**Voting has equal say of all estimators but stacking has variable weights according to final estimator.**

"""

# Applying stacking
estimators = [('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator = RandomForestClassifier()

from sklearn.ensemble import StackingClassifier

clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))

import pickle
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))

