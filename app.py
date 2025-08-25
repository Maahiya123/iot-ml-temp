from flask import Flask, request
from main import genrateAI
import pickle
import numpy as np

genrateAI()
ai = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "AI Model Server is running"

@app.route('/predict', methods=['GET'])
def predict():
    temp = request.args.get('temp')
    temp = float(temp)
    data = np.array([[temp]])
    result = ai.predict(data)
    result = result[0]
    return (result)

if (__name__ == "__main__"):
    app.run(host='0.0.0.0',port=5000,debug=True)
