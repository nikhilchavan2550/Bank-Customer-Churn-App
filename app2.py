import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Set page title and icon
st.set_page_config(page_title="Bank Customer Churn Prediction", page_icon="🏦")

# Custom CSS for a polished look
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        color: #333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sidebar .sidebar-content {
        background: #004494;
        color: white;
    }
    .css-1aumxhk {
        background-color: #0056b3;
        color: white;
        border-radius: 10px;
        font-size: 16px;
    }
    .css-1aumxhk:hover {
        background-color: #004494;
    }
    .stButton>button {
        background-color: #0056b3;
        color: white;
        border-radius: 10px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #004494;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #004494;
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("Churn_Modelling.csv")
    df.drop(['CustomerId', 'RowNumber', 'Surname'], axis='columns', inplace=True)
    return df

@st.cache_data
def preprocess_data(df, is_user_data=False):
    df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})
    df = pd.get_dummies(data=df, columns=['Geography'])

    scale_var = ['Tenure', 'CreditScore', 'Age', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    scaler = MinMaxScaler()
    df[scale_var] = scaler.fit_transform(df[scale_var])

    # Ensure user data has the same columns as training data
    if is_user_data:
        all_columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                       'Geography_France', 'Geography_Germany', 'Geography_Spain']
        for col in all_columns:
            if col not in df.columns:
                df[col] = 0  # Add missing columns with default value 0

    return df

def train_model(X_train, y_train):
    inputs = Input(shape=(X_train.shape[1],))
    x = Dense(12, activation='relu')(inputs)
    x = Dense(6, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, verbose=0)

    return model

def main():
    st.sidebar.header("Navigation")
    st.sidebar.markdown("""
        - [Customer Details](#customer-details)
        - [Model Training](#model-training)
        - [Predict Customer Churn](#predict-customer-churn)
    """)

    st.title("🏦 Bank Customer Churn Prediction")
    st.image("https://imgs.search.brave.com/1HBZIQswmALMf5FAxyPVJdeQp3UFJJjKAxHyv-PZL6c/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5saWNkbi5jb20v/ZG1zL2ltYWdlL0Q0/RDEyQVFGd0l1cU1z/ZEIxSlEvYXJ0aWNs/ZS1jb3Zlcl9pbWFn/ZS1zaHJpbmtfNjAw/XzIwMDAvMC8xNjk4/ODAyMTIwNDI0P2U9/MjE0NzQ4MzY0NyZ2/PWJldGEmdD1mU1M0/NjJQTDBRTUJMbVk0/WUpPTmlwNmJyTXg5/c2Y0YS1RMWZ2N1k4/dEJv", use_column_width=True)  # Add a relevant banner image
    st.markdown("""
    ### Welcome to the Bank Customer Churn Prediction App!
    Use this app to predict the likelihood of a customer leaving the bank. It's an ideal choice for bankers looking to retain their valuable customers.
    """)

    # Load and preprocess data
    df = load_data()
    df_processed = preprocess_data(df)

    # Split features and target
    X = df_processed.drop('Exited', axis='columns')
    y = df_processed['Exited']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    # Customer Details Section
    st.subheader("Customer Details")
    
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
    geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.number_input("Age", min_value=18, max_value=100)
    tenure = st.number_input("Tenure", min_value=0, max_value=10)
    balance = st.number_input("Balance")
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4)
    has_cr_card = st.selectbox("Has Credit Card", [0, 1])
    is_active_member = st.selectbox("Is Active Member", [0, 1])
    estimated_salary = st.number_input("Estimated Salary")

    # Create a dataframe with user input
    user_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Preprocess user input
    user_data_processed = preprocess_data(user_data, is_user_data=True)

    # Model Training Section
    st.subheader("Model Training")
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            model = train_model(X_train, y_train)
            model.save("churn_model.h5")
            joblib.dump(X_train.columns, "model_columns.pkl")
        st.success("Model trained successfully!")

        # Model evaluation
        st.subheader("Model Evaluation")
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        st.pyplot(fig)

        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy score: {accuracy * 100:.2f}%")
    
    # Prediction Section
    st.subheader("Predict Customer Churn")
    
    if st.button("Predict"):
        model = load_model("churn_model.h5")
        model_columns = joblib.load("model_columns.pkl")
        user_data_processed = user_data_processed[model_columns]  # Ensure the columns are in the same order
        prediction = model.predict(user_data_processed)
        churn_prob = prediction[0][0]
        st.write(f"Probability of customer churn: {churn_prob:.2%}")
        if churn_prob > 0.5:
            st.warning("This customer is likely to churn.")
        else:
            st.success("This customer is likely to stay.")

if __name__ == "__main__":
    main()
