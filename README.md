# my_new
general AI projects
import tensorflow as tf
from tensorflow.python.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

def read_file():
    data = pd.read_csv("/users/arunkarthik/Downloads/voice.csv")
    df_x = data.drop(['label'],axis=1)
    df_y = data['label']
    X = np.array(df_x)
    l = LabelEncoder()
    Y_ = l.fit_transform(df_y)
    Y = np.array(Y_).reshape(-1,1)
    return X,Y

x,y= read_file()
x,y = shuffle(x,y)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=23,test_size=0.2)

features = x_train
labels = y_train

input = tf.keras.Input((20,))

x = layers.Dense(100,activation='relu')(input)
x = layers.Dense(100,activation='relu')(x)
prediction = layers.Dense(2,activation='softmax')(x)


model  = tf.keras.Model(inputs = input,outputs = prediction)


model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics= ['accuracy'])


model.fit(features,labels,epochs=1000,batch_size=30)

model.evaluate(features,labels)
res = model.predict(x_test)
print(res)
