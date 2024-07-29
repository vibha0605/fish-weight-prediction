from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('fish_weight_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    length = float(request.form['length'])
    height = float(request.form['height'])
    width = float(request.form['width'])
    prediction = model.predict([[length, height, width]])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
