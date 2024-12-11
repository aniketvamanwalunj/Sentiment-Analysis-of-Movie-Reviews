import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import pickle

# Load your dataset
data = pd.read_csv("IMDB Dataset.csv")  

# Check if the dataset is loaded correctly
print(data.head())

# Check for missing values
if data.isnull().sum().any():
    print("Missing values found. Consider handling them.")
    data = data.dropna()  # Or use another strategy like filling with a default value

# Split data into features and labels
X = data['review']  # Assuming your reviews are in the 'review' column
y = data['sentiment']  # Assuming your sentiments are in the 'sentiment' column

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to top 5000 features
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)

# Train the model using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)  # Convert predictions to binary labels
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Model Accuracy with Linear Regression: {accuracy * 100:.2f}%")

# Save the model, vectorizer, and label encoder
with open('linear_regression_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)

print("Linear Regression Model, vectorizer, and label encoder saved successfully.")
