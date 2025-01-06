from flask import Flask, request, jsonify
from joblib import load
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load model and vectorizer
model = load('model.joblib')
cv = load('cv.joblib')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    # Preprocess the text (similar to training)
    preprocessed_text = preprocess_text(data)  # Define a function for this.
    vectorized_text = cv.transform([preprocessed_text]).toarray()
    prediction = model.predict(vectorized_text)[0]
    
    return jsonify({'prediction': prediction})

def preprocess_text(text):
    # Example preprocessing steps (match training):
    import re
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    stop_words = set(ENGLISH_STOP_WORDS)
    
    # Clean and preprocess
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return ' '.join(ps.stem(word) for word in text if word not in stop_words)

if __name__ == '__main__':
    app.run(debug=True)
