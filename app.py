from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load pre-trained model (make sure model.pkl exists!)
ai = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "AI Model Server is running"

@app.route('/predict', methods=['GET'])
def predict():
    try:
        temp = request.args.get('temp')
        if temp is None:
            return jsonify({"error": "Missing 'temp' parameter"}), 400

        temp = float(temp)
        data = np.array([[temp]])
        result = ai.predict(data)[0]
        return jsonify({"prediction": str(result)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
