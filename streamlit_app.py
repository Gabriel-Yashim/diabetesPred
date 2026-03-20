import streamlit as st
import pandas as pd
import joblib

# Load saved model and features
model = joblib.load('XGBoost_Diabetes_model.pkl')

def preprocess_input(data):
    # Convert categorical inputs to numerical values
    gender_map = {'Female': 0, 'Male': 1}

    gender = data['Gender']
    if isinstance(gender, list):
        gender_value = gender[0]
    else:
        gender_value = gender
    data['Gender'] = gender_map.get(gender_value, 1)

    df = pd.DataFrame(data)
    return df


def main():
    st.title("Diabetes Risk Predictor")
    st.subheader("By SIMBIYAT OYIZA IBRAHIM")
    st.markdown("Matric No: 22L1CS0398")
    
    st.markdown("""
    ### Project Description
    This AI-powered tool analyzes clinical laboratory data to predict the likelihood of diabetes. 
    The data was sourced from **Medical City Hospital** and **Al-Kindy Teaching Hospital** in Baghdad.
    """)

   

    with st.sidebar:
        st.header("Model Info")
        st.info("Trained on 1,000 clinical records using a Random Forest Classifier.")
        st.write("**Key Features used:** HbA1c, BMI, Creatinine, and Lipid Profile.")

    if st.checkbox("Show Dataset Details"):
        st.markdown("""# Diabetes Prediction Dataset

**Description:**

Welcome to the **Diabetes Prediction Dataset**, a valuable resource for researchers, data scientists, and medical professionals interested in the field of diabetes risk assessment and prediction. This dataset contains a diverse range of health-related attributes, meticulously collected to aid in the development of predictive models for identifying individuals at risk of diabetes. By sharing this dataset, we aim to foster collaboration and innovation within the data science community, leading to improved early diagnosis and personalized treatment strategies for diabetes. This dataset is a clinical collection of 1,000 patient records sourced from Medical City Hospital and Al-Kindy Teaching Hospital. It bridges demographic data with intensive metabolic markers.

**Column groups:**

- **Demographic:** Age, Gender
- **Renal Markers:** Creatinine ratio (Cr), Urea
- **Lipid Profile:** Cholesterol (Chol), LDL, VLDL, Triglycerides (TG), HDL
- **Glucose Control:** HBA1C
- **Body Composition:** Body Mass Index (BMI)
- **Target Class:** Diabetic (Y), Non-Diabetic (N), Pre-Diabetic (P))""")


    # Create form for user input
    with st.form(key='input_form'):
        age = st.number_input("👤 Age", min_value=0, value=0)
        bmi = st.number_input("⚖️ BMI", min_value=0.0, value=0.0)
        urea = st.number_input("🧪 Urea", min_value=0.0, value=0.0)
        Cr = st.number_input("🛡️ Creatinine ratio", min_value=0, value=0)
        tg = st.number_input("🍔 Triglycerides", min_value=0.0, value=0.0)
        hba1c = st.number_input("🩸 HbA1c", min_value=0.0, value=0.0)
        cholesterol = st.number_input("📉 Total Cholesterol", min_value=0.0, value=0.0)
        hdl = st.number_input("📊 HDL", min_value=0.0, value=0.0)
        ldl = st.number_input("📉 LDL", min_value=0.0, value=0.0)
        vldl = st.number_input("📊 VLDL", min_value=0.0, value=0.0)
        gender = st.selectbox("🚻 Gender", ['Male', 'Female'])

        submit_button = st.form_submit_button(label='Predict')
        
    if submit_button:
        # Gather input data
        input_data = {
            'Gender': [gender],
            'AGE': [age],
            'Urea': [urea],
            'Cr': [Cr],
            'HbA1c': [hba1c],
            'Chol': [cholesterol],
            'TG': [tg],
            'HDL': [hdl],
            'LDL': [ldl],
            'VLDL': [vldl],
            'BMI': [bmi]
        }

        # Preprocess input data
        input_df = preprocess_input(input_data)

        prediction = model.predict(input_df)
        
        if prediction[0] == 0:
            predicted_label = 'Non-Diabetic'
        elif prediction[0] == 1:
            predicted_label = 'Pre-Diabetic'
        elif prediction[0] == 2:
            predicted_label = 'Diabetic'
        else:
            predicted_label = 'Unknown'

        st.success(f"The result shows that you are: {predicted_label}")
    else:
        st.info("Please fill in the form and click 'Predict'.")
    
if __name__ == '__main__':
    main()