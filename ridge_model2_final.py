import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('ridge_model2.pkl')

# Streamlit app
st.title('Predict Expected GPA ')


# Collect user input
Age = st.number_input('Age', min_value=15.0, max_value=18.0, step=1.0)
Gender = st.selectbox('Gender', [1,0])
Ethnicity = st.selectbox('Ethnicity', [0,1,2,3])
ParentalEducation = st.selectbox('ParentalEducation', [0,1,2,3,4])
StudyTimeWeekly = st.number_input('StudyTimeWeekly', min_value=0.0, max_value=168.0, step=1.0)
Absences = st.number_input('Absences', min_value=0.0, max_value=100.0, step=1.0)
Tutoring = st.selectbox('Tutoring', [1,0])
ParentalSupport = st.selectbox('ParentalSupport', [0,1,2,3,4])
Extracurricular = st.selectbox('Extracurricular', [1,0])
Sports = st.selectbox('Sports', [1,0])
Music = st.selectbox('Music', [1,0])
Volunteering = st.selectbox('Volunteering', [1,0])


# create dataframe 
input_data= pd.DataFrame({
		'Age':[Age],
		'Gender': [Gender],
		'Ethnicity': [Ethnicity],
		'ParentalEducation':[ParentalEducation],
		'StudyTimeWeekly':[StudyTimeWeekly],
		'Absences': [Absences],
		'Tutoring': [Tutoring],
		'ParentalSupport':[ParentalSupport],
		'Extracurricular':[Extracurricular],
		'Sports':[Sports],
		'Music':[Music],
		'Volunteering':[Volunteering]

})


# Predict button
if st.button('Predict'):
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.write(f'Prediction: {prediction[0]}')

if __name__ == '__main__':
    st.run()
        