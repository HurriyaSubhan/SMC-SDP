import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
@st.cache
def load_data():
    data = pd.read_csv("exams.csv")
    return data

# Clean data
def clean_data(data):
    le = LabelEncoder()
    data['gender'] = le.fit_transform(data['gender'])
    data['parental level of education'] = le.fit_transform(data['parental level of education'])
    data['lunch'] = le.fit_transform(data['lunch'])
    data['test preparation course'] = le.fit_transform(data['test preparation course'])
    data['race/ethnicity'] = le.fit_transform(data['race/ethnicity'])
    data["reading score"].fillna(data["reading score"].mean(), inplace=True)
    data["math score"].fillna(data["math score"].mean(), inplace=True)
    data["writing score"].fillna(data["writing score"].mean(), inplace=True)
    data["gender"].fillna(data["gender"].mode()[0], inplace=True)
    return data

# Train model
def train_model(data):
    X = data.drop(columns=['writing score'])
    y = data['writing score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    feature_names = X.columns.tolist()  # Store feature names
    return model, mse, r2, feature_names

# Main function
def main():
    st.title("Exam Score Prediction")
    st.write("This app predicts writing scores based on other exam scores and demographic data.")

    # Load data
    data_load_state = st.text("Loading data...")
    data = load_data()
    data_load_state.text("Data loaded successfully!")

    # Clean data
    data_clean_state = st.text("Cleaning data...")
    data = clean_data(data)
    data_clean_state.text("Data cleaned successfully!")

    # Train model
    model_train_state = st.text("Training model...")
    model, mse, r2, feature_names = train_model(data)
    model_train_state.text("Model trained successfully!")

    st.subheader("Model Evaluation")
    st.write("Mean Squared Error (MSE): ", mse)
    st.write("R-squared (R2): ", r2)

    st.subheader("Make Predictions")
    st.write("Enter exam scores and demographic information to make predictions.")
    # User input for features
    features = {}
    for feature in feature_names:
        if feature != 'writing score':
            features[feature] = st.selectbox(f"{feature.capitalize().replace('_', ' ')}", data[feature].unique())
    for feature in ['reading score', 'math score']:
        features[feature] = st.number_input(f"{feature.capitalize().replace('_', ' ')}", min_value=0, max_value=100, value=50)

    # Make prediction
    if st.button("Predict"):
        input_data = pd.DataFrame([features])
        prediction = model.predict(input_data)
        st.write("Predicted Writing Score: ", prediction[0])

if __name__ == "__main__":
    main()
