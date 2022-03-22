from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import numpy as np
import cv2
import os

dataset = 'Dataset'
test = 0.2
validation = 0.2
imageSize = (32, 32, 3)

batch = 50
epoch = 20
epochperstep = 2000

varCount = 0
arrImages = []
classNo = []
arrlist = os.listdir(dataset)
print("Total No. Of Classes Detected", len(arrlist))
noOfClasses = len(arrlist)
print("Importing Classes")

for x in range(0, noOfClasses):
    myPicList = os.listdir(dataset + "/" + str(varCount))
    for y in myPicList:
        curImg = cv2.imread(dataset + "/" + str(varCount) + "/" + y)
        curImg = cv2.resize(curImg, (imageSize[0], imageSize[1]))
        arrImages.append(curImg)
        classNo.append(varCount)
    print(x, end=" ")
    varCount += 1
print(" ")
print("Total Images in Image List = ", len(arrImages))
print("Total IDS in classNo List = ", len(classNo))

arrImages = np.array(arrImages)
classNo = np.array(classNo)
print(arrImages.shape)

# Spliting the data
x_train, x_test, y_train, y_test = train_test_split(arrImages, classNo, test_size=test)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

numOfSamples = []
for x in range(0, noOfClasses):
    numOfSamples.append(len(np.where(y_train == x)[0]))
print(numOfSamples)

plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClasses), numOfSamples)
plt.title("No of Images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


x_train = np.array(arrlist(map(preprocessing, x_train)))
x_test = np.array(arrlist(map(preprocessing, x_test)))
x_validation = np.array(arrlist(map(preprocessing, x_validation)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(x_train)
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)


def mymodel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 50

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageSize[0],
                                                               imageSize[1], 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return  model

model = mymodel()
print(model.summary())


history = model.fit_generator(dataGen.flow(x_train, y_train,
                                           batch_size=batch),
                              steps_per_epoch = stepsPerEpoch,
                              epochs = epoch,
                              validation_data = (x_validation, y_validation),
                              shuffle =1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(x_test, y_test, verbose=0)
print('Test Score =', score[0])
print('Test Accuracy =', score[1])

model.save("20epochModel.h5")
pickle_out = open("20epochModel.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()