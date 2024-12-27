from flask import Flask, request, jsonify
from model import Model

app = Flask(__name__)

# Load the trained model
model = Model()
model.load_model("models/bangalore_home_price_model.pickle")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']
        prediction = model.predict(features)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
