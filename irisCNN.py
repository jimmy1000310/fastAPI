import numpy as np 
import pandas as pd
from tensorflow.keras.datasets import mnist
# from keras.utils import np_utils  tf1
from tensorflow.keras.utils import to_categorical  
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D ,Input,Reshape
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers ,regularizers ,applications
import glob
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import psycopg2
conn = psycopg2.connect(database = 'postgres' ,user = 'postgres',password='123456',port='5432')
curs=conn.cursor()

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config) 

X=[]

Y=[]

n=0


# #建立花資料庫
# sql="""CREATE TABLE "iris" ("Speal Leng" real,"Speal Width" real,"Petel Leng" real,"Petal Width" real,"Species" varchar(80));"""   
# curs.execute(sql)
# # 將txt檔案轉成資料庫
# sql="""COPY iris FROM 'D:\python/fastAPI/iris.txt' (DELIMITER(','));"""
# curs.execute(sql)
sql = 'select * from iris;'
curs.execute(sql) 
data=curs.fetchall() 
print(type(data[0]))
a=np.array(data)
print(len(a))
for i in range(0,len(a)):
    if a[i][4]=='Iris-setosa':
        Y.append(0)
    elif a[i][4]=='Iris-versicolor':
        Y.append(1)
    elif a[i][4]=='Iris-virginica':
        Y.append(2)
    else:
        print("ERROR")
for i in range(0,len(a)):
    X_tmp=[]
    for j in range(0,len(a[i])-1):
        X_tmp.append(float(a[i][j]))
    X.append(X_tmp)
        

print(Y)
print(X)
Y=np.array(Y)
X=np.array(X)
print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)
print("X_train.shape{},Y_train.shape{}".format(X_train.shape,Y_train.shape))
print("X_test.shape{},Y_test.shape{}".format(X_test.shape,Y_test.shape))
print("X_train.shape={}".format(X_train.shape))
print("X_test.shape={}".format(X_test.shape))
yyy=np.copy(Y_test)

Y_train=to_categorical(Y_train,3)
Y_test_categories=Y_test
Y_test=to_categorical(Y_test,3)

print("X_train.shape{},Y_train.shape{}".format(X_train.shape,Y_train.shape))
print("X_test.shape{},Y_test.shape{}".format(X_test.shape,Y_test.shape))
print("X_train.shape={}".format(X_train.shape))
print("X_test.shape={}".format(X_test.shape))



input2=Input(shape=4)
model2=Dense(64, activation='relu')(input2)
# model2=Reshape(8,8,1)(model2)
# model2=Conv2D(filters=32, kernel_size=(3,3), padding='valid', activation='relu')(model2)
# model2=MaxPooling2D(pool_size=(2,2))(model2)
# model2=Flatten()(model2)
model2=Dense(50,activation='relu')(model2)
model2=Dropout(0.3)(model2)
model2 = Dense(8,activation='relu')(model2)
model2=Dropout(0.3)(model2)
output_layer = Dense(3,activation='softmax')(model2)

model= Model(inputs=input2, outputs=output_layer)
model.summary()
# classifier loss, Adam optimizer, classifier accuracy

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])
checkpointer=ModelCheckpoint('model/model_{epoch:03d}.h5',save_weights_only=False)

# train the model with input images and labels

history=model.fit(X_train, #X1_train,
          Y_train,
          validation_data=(X_test, Y_test),  #X1_test,
          epochs=100,
          batch_size=20,callbacks=[checkpointer])    
model.save('modelcon.h5') 
# model accuracy on test dataset
scores = model.evaluate(X_test,Y_test)
print(scores)
prediction1=model.predict(X_test)  #X1_test,
prediction1=np.argmax(prediction1,axis=1)
print(yyy)
print(prediction1)
prediction = prediction1.reshape(prediction1.shape[0],1)
# print(prediction)
print(Y_test.shape)
# Y_test_categories = Y_test_categories.reshape(Y_test_categories.shape[0],1)
# pd.crosstab(Y_test_categories,prediction,rownames=['real'],colnames=['predict'])
# Y_test = Y_test.reshape(Y_test.shape[0],1)
# print(Y_test.shape)
# confusion_mat = pd.crosstab(Y_test,prediction,rownames=['label'],colnames=['predict'])
# print(confusion_mat)

print('測試損失度:', scores[0]) 
print('測試準確率:', scores[1])

loss = history.history['loss']
acc = history.history['categorical_accuracy']
val_loss=history.history['val_loss']
val_acc=history.history['val_categorical_accuracy']
plt.plot(loss,label='loss')
plt.plot(val_loss,label='val_loss')
plt.plot(acc,label='acc')
plt.plot(val_acc,label='val_acc')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim(0,2)
plt.legend(['train','valid','acc','val_acc'],loc='upper left')
plt.savefig('./loss.png')