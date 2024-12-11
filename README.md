
# Sentiment Analysis of Movie Reviews

This project provides a web application for sentiment analysis of movie reviews. It uses machine learning to predict whether a given review is positive, negative, or neutral.

## Installation

1. Clone this repository to your local machine.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask app:

```bash
python app.py
```

4. Navigate to `http://127.0.0.1:5000/` in your browser to access the application.

## Files Included

- `app.py`: Main Flask application file that handles routing and sentiment prediction.
- `model.pkl`: Trained logistic regression model for sentiment classification.
- `vectorizer.pkl`: TF-IDF vectorizer used for transforming the review text.
- `label_encoder.pkl`: Label encoder for encoding/decoding sentiment labels.
- `index.html`: Frontend HTML template for the web interface.

## How It Works

1. The user inputs a movie review in the text box on the homepage.
2. The review is sent to the Flask backend for sentiment prediction.
3. The model returns the predicted sentiment (positive, negative, or neutral).
4. The result is displayed on the page with color-coded sentiment.

## Requirements

- Python 3.7 or above
- Flask
- scikit-learn
- pandas
- numpy

## License

This project is open-source and available under the MIT License.

## Contact Information
For any queries, suggestions, or feedback, feel free to reach out:

- Name: Aniket Walunj
- Email: aniketvamanwalunj@gmail.com
