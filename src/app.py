from flask import Flask, request, render_template
import logging
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load pre-trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_der')

with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        weight = float(request.form['energy'])

        MEANS = [121.77931253236497,
                    41.97259965127788,
                    82.19542642422176,
                    63.264746914268116,
                    2.4290802781911025,
                    19.315934929239564,
                    167.51519021742698,
                    1.4276174479047175,
                    0.05432433778849873]
        
        MEANS.append(weight)

        MEANS = np.array(MEANS).reshape(1, -1)

        columns = ['der_mass_mmc', 'der_mass_transverse_met_lep', 'der_mass_vis',
       'der_pt_h', 'der_deltar_tau_lep', 'der_pt_tot', 'der_sum_pt',
       'der_pt_ratio_lep_tau', 'der_met_phi_centrality', 'weight']

        df_predict = pd.DataFrame(MEANS, columns=columns)

        predicted_label = model.predict(MEANS)
    
    else:
        predicted_label = None
    
    if predicted_label == 1:
        predicted_label = 'signal'
    elif predicted_label == 0:
        predicted_label = 'background'
    else:
        predicted_label = None

    return render_template('prediction.html', result=predicted_label)


if __name__ == '__main__':
    app.run(debug=True)