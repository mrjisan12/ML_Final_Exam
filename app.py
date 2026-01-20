import gradio as gr
import numpy as np
import pickle


# 1. Load the Model
with open("final_exam_model.pkl", "rb") as f:
    model = pickle.load(f)
    

def predict_diabetes(preg, glucose, bp, skin, insulin, bmi, dpf, age):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(data)[0]
    return "Diabetic" if prediction == 1 else "Non-Diabetic"

inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age"),
    ]


app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="text",
    title="Diabetes Prediction System",
    description="Enter patient details to predict diabetes"
)

app.launch(share=True)
