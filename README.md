# ToxiScan
ToxiScan is an advanced text analysis tool designed to detect toxicity in textual data. By leveraging the power of Natural Language Toolkit (NLTK), TfidfVectorizer, and the Naive Bayes classifier, ToxiScan provides accurate predictions on whether a given text is toxic or non-toxic. With its simple user interface built using Streamlit, ToxiScan makes toxicity analysis easily accessible to users.
![imgonline-com-ua-twotoone-vzmYnnxxlrjvC](https://user-images.githubusercontent.com/90026724/236762042-ec1b2801-3718-4825-8ab5-413c4e82039c.jpg)
## Key Features
- ### Toxicity Detection: ToxiScan uses the Naive Bayes classifier, trained on a diverse dataset of labeled toxic and non-toxic comments, to predict the presence of toxicity in a given text.
- ### Text Preprocessing: ToxiScan employs NLTK, a powerful natural language processing library, for comprehensive text preprocessing. It performs essential tasks such as tokenization, part-of-speech tagging, lemmatization, and stopword removal to ensure the input text is properly prepared for analysis.
- ### Feature Extraction: TfidfVectorizer is utilized to extract relevant features from the preprocessed text. This vectorization technique transforms text into numerical feature vectors, enabling the Naive Bayes classifier to make predictions.
- ### Accuracy Evaluation: To assess the performance of the classifier, ToxiScan employs metrics such as roc_auc_score and roc_curve, providing insights into the accuracy and efficiency of the toxicity detection model.
## Installation
To run ToxiScan on your local machine, follow these steps:
1. Clone the repository:
```
git clone https://github.com/<username>/<repository>.git
cd <repository>
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Launch the ToxiScan application:
```
streamlit run toxiscan.py
```
4. Access ToxiScan in your web browser:
```
http://localhost:8501
```
## Usage
1. Input Text: Enter the text you want to analyze for toxicity in the provided text input box.
2. Analyze: Click the "Analyze" button to trigger the toxicity prediction process.
3. Result: ToxiScan will display the prediction result, indicating whether the text is classified as toxic or non-toxic.
## Dependencies
ToxiScan utilizes the following libraries and resources:
- NLTK - Natural Language Toolkit for text preprocessing.
- Scikit-learn - Machine learning library for feature extraction and classification.
- Streamlit - Framework for building interactive web applications.



