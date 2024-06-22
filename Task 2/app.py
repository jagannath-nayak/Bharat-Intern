import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Create the web app
st.title('Titanic Survival Prediction')

# Create input fields for the features
st.header('Passenger Information')

pclass = st.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
sex = st.selectbox('Sex (0 = Male, 1 = Female)', [0, 1])
age = st.slider('Age', 0, 100, 30)
sibsp = st.slider('Number of Siblings/Spouses Aboard', 0, 10, 0)
parch = st.slider('Number of Parents/Children Aboard', 0, 10, 0)
fare = st.slider('Fare', 0.0, 500.0, 30.0)
embarked = st.selectbox('Port of Embarkation (0 = Cherbourg, 1 = Queenstown, 2 = Southampton)', [0, 1, 2])

# Create a DataFrame from the input data
data = {
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
}

input_data = pd.DataFrame(data)

# Predict button
if st.button('Predict'):
    # Predict the survival
    prediction = model.predict(input_data)

    # Display the prediction
    if prediction[0] == 1:
        st.success('The passenger is likely to survive.')
    else:
        st.error('The passenger is not likely to survive.')




