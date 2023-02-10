from astroNN.datasets import galaxy10
from astroNN.datasets.galaxy10 import galaxy10cls_lookup
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from tqdm import tqdm



images, labels = galaxy10.load_data()
labels = labels.astype(np.float32)
labels = to_categorical(labels)
images = images.astype(np.float32)
images = images/255
     
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.15)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=(69,69,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))



history = model.compile(
loss='categorical_crossentropy',
optimizer='Adam',
metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
     
hist_df = pd.DataFrame(history.history)

hist_csv_file = "history.csv"
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)#

prediction = model.predict(X_test)

np.save("test_predictions.npy",prediction)
np.save("test_data.npy",X_test)
np.save('test_labels.npy',y_test)