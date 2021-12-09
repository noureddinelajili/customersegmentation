from flask import Flask, render_template, request,redirect
import pickle
import numpy as np


model = pickle.load(open('modell.pkl','rb'))

app = Flask(__name__,template_folder='templates')


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict',methods = ['POST'])
def home():
  
    data1 = request.form['Annual Income (k$)']
    data2 = request.form['Spending Score (1-100)']
    arr = np.array([[data1,data2]])
    pred = model.predict((arr))
    a=int(pred)
    if float(a) == 0:
            prediction="0\nwe see that people have average income and an average spending score, these people will not be the prime targets of the shops or mall"
    elif float(a) == 1:
            prediction='1 , we see that people have high income but low spending scores,These can be the prime targets of the mall, as they have the potential to spend money.'
    elif float(a) == 2:
            prediction='2 , income more than 70 and annual income and spending score are equal '
    elif float(a) == 3:
            prediction='3 , we can see that people have low income but higher spending scores, these are those people who for some reason love to buy products more often even though they have a low income. '
    elif float(a) == 4:
            prediction='4 , we can see people have low annual income and low spending scores, this is quite reasonable as people having low salaries prefer to buy less'
    return render_template('after.html',data=prediction)

    

if (__name__ == "__main__"):
    app.run(debug=True)
