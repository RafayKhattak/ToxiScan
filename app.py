# Import required libraries
import streamlit as st
import pickle

# Set Streamlit page configuration
st.set_page_config(page_title="ToxiScan", page_icon="💬", layout="wide")

# Display ToxiScan title and description
st.markdown(
    '<p style="display:inline-block;font-size:40px;font-weight:bold;">📄ToxiScan </p>'
    ' <p style="display:inline-block;font-size:16px;">Text analysis tool that leverages the power of Natural Language Toolkit (NLTK) and the Naive Bayes classifier to determine the presence of toxicity in textual data<br><br></p>',
    unsafe_allow_html=True
)

# Load TF-IDF vectorizer and Naive Bayes model
def load_tfidf():
    tfidf_vectorizer = pickle.load(open("tf_idf.pkt", "rb"))
    return tfidf_vectorizer

def load_model():
    model = pickle.load(open("toxicity_model.pkt", "rb"))
    return model

# Perform toxicity prediction on input text
def predict_toxicity(text):
    tfidf_vectorizer = load_tfidf()
    text_tfidf = tfidf_vectorizer.transform([text]).toarray()
    model = load_model()
    prediction = model.predict(text_tfidf)
    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    return class_name

# Display text input and analyze button
st.subheader("Input your text")
text_input = st.text_input("Enter your text:")

if text_input is not None:
    if st.button("Analyze"):
        result = predict_toxicity(text_input)
        st.subheader("Result:")
        st.info("The result is " + result + ".")

# Hide Streamlit header, footer, and menu
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""

# Apply CSS code to hide header, footer, and menu
st.markdown(hide_st_style, unsafe_allow_html=True)
