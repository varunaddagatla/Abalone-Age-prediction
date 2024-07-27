from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    input_features = [float(x) for x in request.form.values()]

    # Convert the input features to a numpy array#
    input_features = np.array(input_features).reshape(1,-1)

    # Predict the age using the loaded model
    prediction = model.predict(input_features)[0]

    # Prepare a response with the predicted age
    return render_template('index.html', prediction_text=f'Predicted Age: {prediction}')


if __name__ == '__main__':
    app.run(port=5002, debug=True)




















