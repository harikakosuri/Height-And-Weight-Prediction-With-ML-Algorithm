import numpy as np
from flask import Flask, jsonify, render_template, request
import pickle

app = Flask(__name__)
fe_wt_lo=pickle.load(open('female_wt_pred.pkl','rb'))
fe_ht_lo=pickle.load(open('female_ht_pred.pkl','rb'))
ma_wt_lo=pickle.load(open('male_wt_pred.pkl','rb'))
ma_ht_lo=pickle.load(open('male_ht_pred.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])    
def predict():
    '''
    For rendering results on HTML GUI
    '''
    fe_wt_data  = [int(x) for x in request.form.values()]
    final_fe_wt = [np.array(fe_wt_data)]
    prediction_fe_wt=fe_wt_lo.predict(final_fe_wt)

    output1 = round(prediction_fe_wt[0], 2)
    height1=fe_wt_data
    less1=round(output1 - 4)
    high1=round(output1 + 4)
    return render_template('index.html',output1=output1,height1=height1,less1=less1,high1=high1)


#Female Height Prediction
@app.route('/predict_fe_ht',methods=['POST'])    
def predict_fe_ht():
    '''
    For rendering results on HTML GUI
    '''
    fe_ht_data  = [int(x) for x in request.form.values()]
    final_fe_ht = [np.array(fe_ht_data)]
    prediction_fe_ht=fe_ht_lo.predict(final_fe_ht)

    output2 = round(prediction_fe_ht[0], 2)
    height2=fe_ht_data
    less2=round(output2 - 4)
    high2=round(output2 + 4)
    return render_template('index.html',output2=output2,height2=height2,less2=less2,high2=high2)

#Male weight Predition

@app.route('/predict_ma_wt',methods=['POST'])    
def predict_ma_wt():
    '''
    For rendering results on HTML GUI
    '''
    ma_wt_data  = [int(x) for x in request.form.values()]
    final_ma_wt = [np.array(ma_wt_data)]
    prediction_ma_wt=ma_wt_lo.predict(final_ma_wt)

    output3 = round(prediction_ma_wt[0], 2)
    height3=ma_wt_data
    less3=round(output3 - 4)
    high3=round(output3 + 4)
    return render_template('index.html',output3=output3,height3=height3,less3=less3,high3=high3)

#Male Height Prediction
@app.route('/predict_ma_ht',methods=['POST'])    
def predict_ma_ht():
    '''
    For rendering results on HTML GUI
    '''
    ma_ht_data  = [int(x) for x in request.form.values()]
    final_ma_ht = [np.array(ma_ht_data)]
    prediction_ma_ht=ma_ht_lo.predict(final_ma_ht)

    output4 = round(prediction_ma_ht[0], 2)
    height4=ma_ht_data
    less4=round(output4 - 4)
    high4=round(output4 + 4)
    return render_template('index.html',output4=output4,height4=height4,less4=less4,high4=high4)


@app.route('/predict_api',methods=['POST'])
def predict_api1():
    '''
    For direct API calls trought request
    '''
    data1 = request.get_json(force=True)
    prediction1 = fe_wt_lo.predict([np.array(list(data1.values()))])

    output1 = prediction1[0]
    return jsonify('Your Approximate Weight is {}'.format(output1))

@app.route('/predict_api2',methods=['POST'])
def predict_api2():
    '''
    For direct API calls trought request
    '''
    data2 = request.get_json(force=True)
    prediction2 = fe_ht_lo.predict([np.array(list(data2.values()))])

    output2 = prediction2[0]
    return jsonify('Your Approximate Height is {}'.format(output2))

@app.route('/predict_api3',methods=['POST'])
def predict_api3():
    '''
    For direct API calls trought request
    '''
    data3 = request.get_json(force=True)
    prediction3 = ma_wt_lo.predict([np.array(list(data3.values()))])

    output3 = prediction3[0]
    return jsonify('Your Approximate Weight is {}'.format(output3))    


@app.route('/predict_api4',methods=['POST'])
def predict_api4():
    '''
    For direct API calls trought request
    '''
    data4 = request.get_json(force=True)
    prediction4 = ma_ht_lo.predict([np.array(list(data4.values()))])

    output4 = prediction4[0]
    return jsonify('Your Approximate Weight is {}'.format(output4))
if __name__ == "__main__":
  app.run(debug=True)