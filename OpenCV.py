import numpy as np
import cv2
import pickle

frameWidth = 640
frameHeight = 480
brightness = 150
threshold = 0.65
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
cap.set(cv2.CAP_PROP_FPS, 30)

pickle_in = open("20epochModel.p", "rb")
model = pickle.load(pickle_in)


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def getClassName(classNo):
    if classNo == 0:
        return 'Tagalog Translation : a'
    elif classNo == 1:
        return 'Tagalog Translation : ba'
    elif classNo == 2:
        return 'Tagalog Translation : da'
    elif classNo == 3:
        return 'Tagalog Translation : ei'
    elif classNo == 4:
        return 'Tagalog Translation : ga'
    elif classNo == 5:
        return 'Tagalog Translation : ha'
    elif classNo == 6:
        return 'Tagalog Translation : ka'
    elif classNo == 7:
        return 'Tagalog Translation : la'
    elif classNo == 8:
        return 'Tagalog Translation : ma'
    elif classNo == 9:
        return 'Tagalog Translation : na'
    elif classNo == 10:
        return 'Tagalog Translation : nga'
    elif classNo == 11:
        return 'Tagalog Translation : ou'
    elif classNo == 12:
        return 'Tagalog Translation : pa'
    elif classNo == 13:
        return 'Tagalog Translation : sa'
    elif classNo == 14:
        return 'Tagalog Translation : ta'
    elif classNo == 15:
        return 'Tagalog Translation : wa'
    elif classNo == 16:
        return 'Tagalog Translation : ya'


while True:
    success, imgOrignal = cap.read()
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CS THESIS", (250, 20), font, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "Baybayin Script Translation to Tagalog Using LeNet CNN With Real-time Recognition on OpenCV ", (10, 50), font, 0.40, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "CLASS: ", (50, 450), font, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (400, 450), font, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
        print(getClassName(classIndex))
        cv2.putText(imgOrignal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 450), font, 0.50,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (520, 450), font, 0.50, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow("Baybayin Recognition", imgOrignal)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break