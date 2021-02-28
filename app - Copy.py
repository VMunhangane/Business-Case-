import os
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load

PROJECT_ROOT = "C:/Users/putos/OneDrive/Documents/Segundo Semestre/Flask apps/BC0_example/data"

app = Flask(__name__)  # Initialize the flask App
model = load(os.path.join(PROJECT_ROOT,  'best_decision_tree.joblib'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = list(request.form.values())[0].split(',')
    final_features = [np.array(int_features)]
    prediction = model.predict_proba(final_features)

    output = np.argmax(prediction[0])

    if output == 0:
        return render_template('index.html', prediction_text='This current customer belongs to cluster: {} muito bom'.format(output))
    elif output ==1:
        return render_template('index.html', prediction_text='This current customer belongs to cluster: {}bnmxbcvnbncbv'.format(output))
    elif output == 2:
        return render_template('index.html', prediction_text='This current customer belongs to cluster: {} dnfmdsbfbjkdsfbjkdsbfkjdbbfkbdsk'.format(output))
    else:
        return render_template('index.html', prediction_text='This current customer belongs to cluster: {}bnbbsnbdnhdhs'.format(output))

if __name__ == "__main__":
    app.run(debug=True)