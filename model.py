# Import required libraries
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, roc_curve

# Download necessary NLTK resources
nltk.download('punkt')  # for tokenization
nltk.download('omw-1.4')  # for WordNet
nltk.download('wordnet')  # for WordNet
nltk.download('stopwords')  # for stopwords
nltk.download('averaged_perceptron_tagger')  # for POS tagging

# Load the dataset
data = pd.read_csv("FinalBalancedDataset.csv")
data = data.drop("Unnamed: 0", axis=1)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def prepare_text(text):
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    # Clean the text
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    text = text.split()
    text = ' '.join(text)

    # Tokenize and lemmatize the text
    text = word_tokenize(text)
    text = pos_tag(text)
    lemma = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in text]
    lemma = ' '.join(lemma)
    return lemma

# Prepare the text data by applying text preprocessing
data['clean_tweets'] = data['tweet'].apply(prepare_text)

# Create the corpus
corpus = data['clean_tweets'].values.astype('U')

# Define and fit the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words=nltk_stopwords.words('english'))
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, data['Toxicity'], test_size=0.8, random_state=42, shuffle=True
)

# Train the Naive Bayes model
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)

# Predict probabilities and calculate ROC-AUC score
y_pred_proba = naive_bayes_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Test with sample text
test_text = "I love you"
test_tfidf = tfidf_vectorizer.transform([test_text])
predicted_proba = naive_bayes_model.predict_proba(test_tfidf)
predicted_label = naive_bayes_model.predict(test_tfidf)
print(predicted_proba)
print(predicted_label)

# Save the model and TF-IDF vectorizer
pickle.dump(naive_bayes_model, open("toxicity_model.pkt", "wb")) # Save the trained model
pickle.dump(tfidf_vectorizer, open("tf_idf.pkt", "wb")) # Save the TF-IDF vectorizer