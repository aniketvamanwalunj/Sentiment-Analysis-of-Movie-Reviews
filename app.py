from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model, vectorizer, and label encoder
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data['review']
    review_vectorized = vectorizer.transform([review])
    prediction = model.predict(review_vectorized)
    sentiment = label_encoder.inverse_transform(prediction)[0]
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
