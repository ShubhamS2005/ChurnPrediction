from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and encoder
with open('churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data from the form
    age = int(request.form['age'])
    job = request.form['job']
    marital = request.form['marital']
    education = request.form['education']
    default = request.form['default']
    balance = float(request.form['balance'])
    housing = request.form['housing']
    loan = request.form['loan']
    contact = request.form['contact']
    day = int(request.form['day'])
    month = request.form['month']
    duration = int(request.form['duration'])
    campaign = int(request.form['campaign'])
    pdays = int(request.form['pdays'])
    previous = int(request.form['previous'])
    poutcome = request.form['poutcome']

    new_data = pd.DataFrame(
        [[age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome]],
        columns=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]
    )

    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
    new_data_encoded = pd.DataFrame(encoder.transform(new_data[categorical_columns]), columns=encoder.get_feature_names_out())

    numerical_columns = ["age", "balance", "day", "month", "duration", "campaign", "pdays", "previous"]
    new_data_final = pd.concat([new_data[numerical_columns].reset_index(drop=True), new_data_encoded], axis=1)

    new_data_final = new_data_final.reindex(columns=model.feature_names_in_, fill_value=0)

    churn_prediction = model.predict(new_data_final)[0]

    if churn_prediction == 1:
        result = "Customer has churned!"
        result_class = "churned"
    else:
        result = "Customer has not churned!"
        result_class = "not-churned"

    # Return results to the web page
    return render_template('home.html', result=result, result_class=result_class)

if __name__ == "__main__":
    app.run(debug=True)
