from flask import Flask,render_template,request
import numpy as np
import pickle
import soundfile
import librosa


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

app = Flask(__name__,template_folder='template')

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('audio.html')

@app.route('/predict',methods=['POST'])
def predict():
    filename = 'model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    input = request.files['audio-input']
    feature=extract_feature(input, mfcc=True, chroma=True, mel=True)
    feature=feature.reshape(1,-1)
    prediction=loaded_model.predict(feature)
    return render_template('audio.html',pred='Predicted emotion is {}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(port=5001, debug=True)