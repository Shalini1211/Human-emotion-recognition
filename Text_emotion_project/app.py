from flask import Flask,render_template,request
import numpy as np

import pandas as pd
import numpy as np
import seaborn as sns
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline


app = Flask(__name__,template_folder='template')

df = pd.read_csv("data\emotion_dataset.csv")

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('text.html')

@app.route('/predict',methods=['POST'])
def predict():
    dir(nfx)
    df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
    df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
    Xfeatures = df['Clean_Text']
    ylabels = df['Emotion']
    x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=42)
    pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
    pipe_lr.fit(x_train,y_train)
    text_input = request.form['text-input']
    prediction = pipe_lr.predict([text_input])
    return render_template('text.html',pred='Emotion is {}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)