import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Load the data
@st.cache
def load_data():
    df = pd.read_csv('german_credit_data.csv')
    return df

df = load_data()

# Preprocessing function
def preprocess_data():
    df['Risk'] = df.apply(
        lambda row: 'High Risk' if (
            row['Credit amount'] > df['Credit amount'].median() or
            row['Job'] == 0
        ) else 'Low Risk', 
        axis=1
    )
    df['Risk'] = df['Risk'].map({'High Risk': 1, 'Low Risk': 0})

    le = LabelEncoder()
    df['Saving accounts'] = le.fit_transform(df['Saving accounts'])
    df['Checking account'] = le.fit_transform(df['Checking account'])
    df['Purpose'] = le.fit_transform(df['Purpose'])
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Housing'] = le.fit_transform(df['Housing'])

    X = df.drop(columns=['Risk'])
    y = df['Risk']
    return X, y

X, y = preprocess_data()

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Prediction function
def predict_credit_risk(input_data):
    le = LabelEncoder()
    # Encoding the input data based on the model's encoding
    input_data['Saving accounts'] = le.fit_transform([input_data['Saving accounts']])[0]
    input_data['Checking account'] = le.fit_transform([input_data['Checking account']])[0]
    input_data['Purpose'] = le.fit_transform([input_data['Purpose']])[0]
    input_data['Sex'] = le.fit_transform([input_data['Sex']])[0]
    input_data['Housing'] = le.fit_transform([input_data['Housing']])[0]

    # Scaling the input data
    input_scaled = scaler.transform([list(input_data.values())])
    prediction = rf_model.predict(input_scaled)

    # Return prediction
    if prediction == 1:
        return "Good Credit Risk"
    else:
        return "Bad Credit Risk"

# Streamlit UI
st.title("Credit Risk Prediction App")

st.sidebar.header("User Input")

# Input fields for the user
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.sidebar.selectbox("Sex", options=["male", "female"])
job = st.sidebar.number_input("Job", min_value=0, max_value=4, value=1)
housing = st.sidebar.selectbox("Housing", options=["own", "free", "rent"])
saving_accounts = st.sidebar.selectbox("Saving Accounts", options=["little", "moderate", "rich", "quite rich"])
checking_account = st.sidebar.selectbox("Checking Account", options=["little", "moderate", "rich"])
credit_amount = st.sidebar.number_input("Credit Amount", min_value=100, max_value=100000, value=5000)
duration = st.sidebar.number_input("Duration", min_value=1, max_value=72, value=12)
purpose = st.sidebar.selectbox("Purpose", options=["radio/TV", "education", "new car", "used car", "business", "car"])

# Create a dictionary with the user input
input_data = {
    'Age': age,
    'Sex': sex,
    'Job': job,
    'Housing': housing,
    'Saving accounts': saving_accounts,
    'Checking account': checking_account,
    'Credit amount': credit_amount,
    'Duration': duration,
    'Purpose': purpose
}

# Predict button
if st.sidebar.button("Predict Credit Risk"):
    result = predict_credit_risk(input_data)
    st.write(f"Prediction: {result}")

