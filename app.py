from flask import Flask, render_template, request
import joblib

# Load the saved model and vectorizer
model = joblib.load(open('model.joblib', 'rb'))
vectorizer = joblib.load(open('cv.joblib', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']
        transformed_tweet = vectorizer.transform([tweet])
        prediction = model.predict(transformed_tweet)[0]
        
        sentiment = "Offensive" if prediction == 1 else "Normal"
        
        return render_template('index.html', tweet=tweet, sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)

    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)