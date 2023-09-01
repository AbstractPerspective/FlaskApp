from flask import Flask ,render_template, request
import pandas as pd
import math
from datetime import datetime
import joblib
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

app = Flask(__name__)
  

@app.route('/',methods=['GET'])
def launch_the_webpage():
    return render_template('homepage.html')
        
@app.route('/predict',methods=['GET'])
def predict():

    os = request.args.get('os')
    browser = request.args.get('browser')
    locale = request.args.get('locale')
    location = request.args.get('location')
    gender = request.args.get('gender')    
    lenta = request.args.get('lenta.ru') 
    toptal = request.args.get('toptal.com') 
    vk = request.args.get('vk.com') 
    slack = request.args.get('slack.com') 
    mail= request.args.get('mail.google.com') 
    youtube = request.args.get('youtube.com') 
    year = request.args.get('year')
    month = request.args.get('month')
    day = request.args.get('day')
    n_sites = request.args.get('n_sites')
    time_conv =  request.args.get('time')
    time_spent = request.args.get('time_spent')

    X = pd.DataFrame()
    X['browser'] = [browser]
    X['os'] = [os]
    X['locale'] = [locale]
    X['gender'] = gender
    X['location'] = location
    X['n_sites'] = int(n_sites)
    X['time_spent'] = 0 if int(n_sites) == 0 else int(time_spent) / int(n_sites)
    X['lenta.ru'] = 0 if lenta == None else 1
    X['toptal.com'] = 0 if toptal == None else 1
    X['vk.com'] = 0 if vk == None else 1
    X['slack.com'] = 0 if slack == None else 1
    X['mail.google.com'] = 0 if mail == None else 1
    X['youtube.com'] = 0 if youtube == None else 1
    X['year'] = int(year)
    X['month'] = int(month)
    X['day'] = int(day)

    # Generate the time feature. First the time is transformed from hours and minutes, in minutes. 
    # Then is transformed from distribution 0..1439 in distribution 0..2pi .

    time_conv = datetime.strptime(time_conv,'%H:%M:%S').time()
    X['time_conv'] = math.sin((time_conv.hour*60+time_conv.minute)/1439*2*math.pi)

    columns = ['browser', 'os', 'locale', 'location', 'gender']

    xgbest = joblib.load('xgbest.json')
    ordinal_encoder = joblib.load('ordinal_encoder.pkl')

    X[columns] = ordinal_encoder.transform(X[columns])

    if xgbest.predict(X):
        return f"<html><body> <p> There is a catch! With the probability: {xgbest.predict_proba(X)[0][1]} </p></body></html>"
    else:
        return f"<html><body> <p> That's not user Zero. With the probability: {xgbest.predict_proba(X)[0][0]} </p></body></html>"

if __name__ == '__main__':
    app.run(debug=True)