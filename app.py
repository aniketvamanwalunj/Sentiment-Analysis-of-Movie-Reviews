import streamlit as st
import pickle

# Load the model, vectorizer, and label encoder
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Streamlit app
def main():
    # Set page title and layout
    st.set_page_config(page_title="Sentiment Analysis of Movie Reviews", layout="centered")
    
    # Title
    st.title("Sentiment Analysis of Movie Reviews")
    
    # Description
    st.markdown("""
    This application analyzes the sentiment of movie reviews. Enter your review below and get the sentiment result.
    """)

    # Input for review
    review_text = st.text_area("Enter your review:")

    # Analyze sentiment button
    if st.button("Analyze Sentiment"):
        if review_text:
            # Transform the review and predict sentiment
            review_vectorized = vectorizer.transform([review_text])
            prediction = model.predict(review_vectorized)
            sentiment = label_encoder.inverse_transform(prediction)[0]
            
            # Display sentiment result
            sentiment_color = "green" if sentiment == "positive" else "red" if sentiment == "negative" else "orange"
            st.markdown(f"<h4 style='color:{sentiment_color};'>Sentiment: <strong>{sentiment}</strong></h4>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a review to analyze.")

    # Footer
    st.markdown("""
    <footer style="text-align: center; padding: 10px; color: #495057;">
        <small>&copy; 10/08/2024 | Movie Review Sentiment | Designed By Aniket Walunj</small>
    </footer>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    main()
