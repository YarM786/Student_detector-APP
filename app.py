import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------------------------
# Create Sample Dataset
# ---------------------------

data = {
    "Study_Hours": [2,4,6,8,10,3,5,7,9,1],
    "GPA": [2.5,3.0,3.2,3.8,4.0,2.8,3.4,3.6,3.9,2.2],
    "City_Tier": [1,2,1,3,3,2,1,3,2,1],  # 1=Small, 2=Medium, 3=Big
    "Course_Type": [1,1,2,2,3,1,2,3,3,1],  # 1=Arts, 2=Science, 3=Engineering
    "Tuition_Price": [20000,30000,35000,50000,70000,25000,40000,60000,65000,15000]
}

df = pd.DataFrame(data)

# ---------------------------
# Train Model
# ---------------------------

X = df.drop("Tuition_Price", axis=1)
y = df["Tuition_Price"]

model = LinearRegression()
model.fit(X, y)

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Student Price Prediction", page_icon="🎓")

st.title("🎓 Student Tuition Price Prediction System")
st.write("Enter student details to predict tuition price.")

# User Inputs
study_hours = st.slider("Study Hours per Day", 1, 12, 5)
gpa = st.slider("GPA", 2.0, 4.0, 3.0)
city_tier = st.selectbox("City Tier", 
                         options=[1,2,3],
                         format_func=lambda x: 
                         "Small City" if x==1 else 
                         "Medium City" if x==2 else 
                         "Big City")

course_type = st.selectbox("Course Type",
                           options=[1,2,3],
                           format_func=lambda x:
                           "Arts" if x==1 else
                           "Science" if x==2 else
                           "Engineering")

# Prediction Button
if st.button("Predict Tuition Price"):
    
    input_data = np.array([[study_hours, gpa, city_tier, course_type]])
    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Tuition Price: Rs {int(prediction):,}")

st.markdown("---")
st.write("Built with ❤️ using Streamlit & Machine Learning")
