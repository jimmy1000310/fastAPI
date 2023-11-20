import numpy as np 
import keras
from tensorflow.keras.utils import to_categorical 
import tensorflow.compat.v1 as tf
import matplotlib
matplotlib.use('agg')
import psycopg2
conn = psycopg2.connect(database = 'postgres' ,user = 'postgres',password='123456',port='5432')
curs=conn.cursor()


tf.compat.v1.disable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.4
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
sess=tf.Session()

textH=[]
textM=[]
textL=[]
X=[]
Y=[]
n=0
#建立花資料庫
sql="""CREATE TABLE "iris" ("Speal Leng" real,"Speal Width" real,"Petel Leng" real,"Petal Width" real,"Species" varchar(80));"""   
curs.execute(sql)
# 將txt檔案轉成資料庫
sql="""COPY iris FROM 'D:\python/fastAPI/testdata.txt' (DELIMITER(','));"""
curs.execute(sql)
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

model = tf.keras.models.load_model('model.h5')  #135  150 149  218
y_test=np.array(Y)
X_test=np.array(X)
print("Y_test={}".format(y_test))
yyy=np.copy(y_test)

y_test_categories=y_test
y_test=to_categorical(y_test,3)
scores = model.evaluate(X_test,y_test) #X1_test,
print("X_test.shape={}".format(X_test.shape))
prediction1=model.predict(X_test) #X1_test,
# prediction = prediction.reshape(prediction.shape[0],1)
prediction1=np.argmax(prediction1,axis=1)
print("Y_test.shape={}".format(prediction1))

# confusion=tf.math.confusion_matrix([1,2,3],[1,2,3])
confusion=tf.math.confusion_matrix(yyy,prediction1)
con=confusion.eval(session=sess)
print(con)
print('  pred \ real | Iris-setosa | Iris-versicolor | Iris-virginica ')
print('    Iris-setosa     |  {:2.0f}  |  {:2.0f}  |  {:2.0f} '.format(con[0][0],con[1][0],con[2][0]))
print('    Iris-versicolor      |  {:2.0f}  |  {:2.0f}  |  {:2.0f} '.format(con[0][1],con[1][1],con[2][1]))
print('    Iris-virginica     |  {:2.0f}  |  {:2.0f}  |  {:2.0f} '.format(con[0][2],con[1][2],con[2][2]))
print('all={}'.format((con[0][0]+con[1][1]+con[2][2])/sum(sum(con))))
print('Iris-setosa={}'.format(con[0][0]/(con[0][0]+con[0][1]+con[0][2])))
print('Iris-versicolor={}'.format(con[1][1]/(con[1][0]+con[1][1]+con[1][2])))
print('Iris-virginica={}'.format(con[2][2]/(con[2][0]+con[2][1]+con[2][2])))
print('Iris-setosa={}'.format(con[0][0]/(con[0][0]+con[1][0]+con[2][0])))
print('Iris-versicolor={}'.format(con[1][1]/(con[0][1]+con[1][1]+con[2][1])))
print('Iris-virginica={}'.format(con[2][2]/(con[0][2]+con[1][2]+con[2][2])))
