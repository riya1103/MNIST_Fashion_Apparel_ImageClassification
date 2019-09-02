import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm

train = pd.read_csv('apparel_train.csv')

train_image=[]
for i in tqdm(range(train.shape[0])):
    img=image.load_img('apparel_train/'+train['id'][i].astype(str)+ '.png',target_size=(28,28,1),
                       grayscale=True)
    img=image.img_to_array(img)
    img=img/255
    train_image.append(img)


X=np.array(train_image)

y=train['label']
y=to_categorical(y)

test=pd.read_csv('apparel_test.csv')
test_image=[]


for j in tqdm(range(test.shape[0])):
    img1=image.load_img('apparel_test/'+test['id'][j].astype(str)+'.png',target_size=(28,28,1),
                       grayscale=True)
    img1=image.img_to_array(img1)
    img1=img1/255
    test_image.append(img1)
test_X= np.array(test_image)


classifier=Sequential()
classifier.add(Conv2D(56,3,3,activation='relu',input_shape=(28,28,1)))
classifier.add(Conv2D(80,2,2,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=10,activation='softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

classifier.fit(X_train,y_train,epochs=10,,validation_data=(X_test,y_test))























































