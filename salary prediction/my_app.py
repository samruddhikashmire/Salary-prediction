import numpy as np 
from flask import Flask, request, jsonify, render_template
from flask_bootstrap import Bootstrap
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb')) #python object to byte stream

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
	int_features = [int(x) for x in request.form.values()]
	final_features = [np.array(int_features)]
	prediction = model.predict(final_features)
	output = np.round(prediction[0], 2)
	return render_template('index.html', prediction_text='Employee Salary Should Be $ {}'.format(output))


@app.route('/predict_api')
def predict_api():
	data = request.get_json(force=True)
	prediction = model.predict([np.array(list(data.values()))])
	output = prediction[0]
	return jsonify(output)

if __name__ == "__main__":
	app.run(debug=True)

