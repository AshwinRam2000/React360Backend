from flask import Flask, request, url_for, Response
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.models import load_model
import cv2
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import simplejson as json
app = Flask(__name__)



# def predict(image):
#     print(image)
#     image = face_recognition.load_image_file(image)
#     face_locations = face_recognition.face_locations(image)
#     top, right, bottom, left = face_locations[0]
#     face_image = image[top:bottom, left:right]
#     face_image = cv2.resize(face_image, (48,48))
#     face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
#     face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
#     emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
#     model = load_model("model_v6_23.hdf5")
#     predicted_class = np.argmax(model.predict(face_image))
#     label_map = dict((v,k) for k,v in emotion_dict.items()) 
#     predicted_label = label_map[predicted_class]


#     print(predicted_label)
#     return predicted_label


def sentiment_scores(sentence): 
    
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
  
    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
      
    print("Overall sentiment dictionary is : ", sentiment_dict) 
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 
  
    print("Sentence Overall Rated As", end = " ") 
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
    negative=sentiment_dict['neg']*100
    Positive=sentiment_dict['pos']*100
    # decide sentiment as positive, negative and neutral 
    if sentiment_dict['compound'] >= 0.05 : 
        print("Positive") 
        return "Positive",negative,Positive
  
    elif sentiment_dict['compound'] <= - 0.05 : 
        print("Negative") 
        return "Negative",negative,Positive
    
  
     
  
  
    



@app.route('/<name>', methods=['GET', 'POST'])


def hello_name(name):
    print(name)
    sen=" "
    score=" "
    neg=" "
    pos=" "
    status=" "
    if request.method=="POST":
        if name=="pred":
                sen = request.form.get('sentence')
                print("++++++++++++++++",sen)
                score,neg,pos=sentiment_scores(sen)
        # if name=="pic":
        #         file=request.files['image']
        #         print(file)
                
        
        #         status=predict(file)
        

        

        data={
                'status':status,
                'score': score,
                'positive':pos,
                'negative':neg
        }
        js=json.dumps(data)
        res = Response(js, status=200, mimetype='application/json')
        print(res)
        return res


if __name__ == '__main__':
    app.run()