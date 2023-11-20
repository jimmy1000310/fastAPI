from typing import Optional
from fastapi import FastAPI
import tensorflow.compat.v1 as tf
from pydantic import BaseModel
import numpy as np


MODEL = tf.keras.models.load_model('model.h5')

app = FastAPI()

class UserInput(BaseModel):
    Speal_Leng : float
    Speal_Width : float
    Petal_Leng : float
    Petal_Width : float

@app.get('/')
async def index():
    return {"Message": "鳶尾花分類器"}

@app.post('/predict')
async def predict(Data: UserInput):

    data=Data.dict()
    Speal_Leng=data['Speal_Leng']
    Speal_Width=data['Speal_Width']
    Petal_Leng=data['Petal_Leng']
    Petal_Width=data['Petal_Width']

    prediction = MODEL.predict([[Speal_Leng,Speal_Width,Petal_Leng,Petal_Width]])
    prediction1=np.argmax(prediction,axis=-1)
    # print(prediction1)
    if prediction1[0]==0:
        return {"prediction": "Iris-setosa"}
    elif prediction1[0]==1:
        return {"prediction": "Iris-versicolor"}
    elif prediction1[0]==2:
        return {"prediction": "Iris-virginica"}