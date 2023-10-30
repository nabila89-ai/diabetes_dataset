import streamlit as st
import pickle

#load model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

#coding streamlit
st.title("Data Mining Prediksi Diabetes dengan Algoritma SVM")

#input text pada tiap atribut
Pregnancies = st.text_input('Input nilai Pregnancies')
Glucose = st.text_input('Input nilai Glucose')
BloodPressure = st.text_input('Input nilai BloodPressure')
SkinThickness = st.text_input('Input nilai SkinThickness')
Insulin = st.text_input('Input nilai Insulin')
bmi = st.text_input('Input nilai BMI')
DiabetesPedigreeFunction = st.text_input('Input nilai DiabetesPedigreeFunction')
Age = st.text_input('Input nilai Age')

#code untuk prediksi
diagnosis = ''
print(diagnosis)
#membuat tombol prediksi
if st.button('Tes Prediksi Diabetes') :
    prediction = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,bmi,
                                          DiabetesPedigreeFunction,Age]])

    if(prediction[0] == 1) :
        diagnosis = 'Pasien TERKENA DIABETES'
    else :
        diagnosis = 'Pasien TIDAK TERKENA DIABETES'
        
    st.success(diagnosis)



