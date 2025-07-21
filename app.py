import gradio as gr
import pandas as pd
import joblib

# Load the model
model = joblib.load("salary_predictor_model.pkl")

workclass_map = {'Private': 3, 'Self-emp-not-inc': 4, 'Local-gov': 1, 'State-gov': 5}
occupation_map = {'Tech-support': 12, 'Craft-repair': 2, 'Other-service': 9, 'Sales': 10}
relationship_map = {'Husband': 0, 'Not-in-family': 1, 'Own-child': 2, 'Unmarried': 4}

def predict_salary(age, education_num, hours_per_week, capital_gain, capital_loss,
                   workclass, occupation, relationship):
    workclass_val = workclass_map.get(workclass, 0)
    occupation_val = occupation_map.get(occupation, 0)
    relationship_val = relationship_map.get(relationship, 0)

    input_data = pd.DataFrame([[
        age, workclass_val, 0, 0, education_num, 0,
        occupation_val, relationship_val, 0, 0,
        capital_gain, capital_loss, hours_per_week, 0
    ]], columns=[
        'age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status',
        'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country'
    ])

    prediction = model.predict(input_data)[0]
    return ">50K" if prediction == 1 else "<=50K"

interface = gr.Interface(
    fn=predict_salary,
    inputs=[
        gr.Slider(18, 80, value=30, label="Age"),
        gr.Slider(1, 16, value=10, label="Education Level"),
        gr.Slider(1, 100, value=40, label="Hours per Week"),
        gr.Number(label="Capital Gain", value=0),
        gr.Number(label="Capital Loss", value=0),
        gr.Dropdown(["Private", "Self-emp-not-inc", "Local-gov", "State-gov"], label="Workclass"),
        gr.Dropdown(["Tech-support", "Craft-repair", "Other-service", "Sales"], label="Occupation"),
        gr.Dropdown(["Husband", "Not-in-family", "Own-child", "Unmarried"], label="Relationship"),
    ],
    outputs="text",
    title="💼 Employee Salary Predictor",
    description="Predict whether an employee earns >50K or <=50K annually based on job and demographic features."
)

interface.launch()
