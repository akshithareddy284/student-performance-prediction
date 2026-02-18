import pickle
import pandas as pd

# Load model
model = pickle.load(open("student_model.pkl", "rb"))

# Example manual input
input_data = pd.DataFrame({
    "G2": [9],
    "G1": [7],
    "absences": [5],
    "failures": [5],
    "studytime": [2]
})

prediction = model.predict(input_data)[0]
predicted_marks = round(prediction, 2)

if predicted_marks >= 10:
    result = "Pass"
else:
    result = "Fail"

print("Predicted Marks:", predicted_marks)
print("Final Result:", result)
