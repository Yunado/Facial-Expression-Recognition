from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
import matplotlib.pyplot as plt

# face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load('haarcascades/haarcascade_frontalface_default.xml')
# eye
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eye_cascade.load('haarcascades/haarcascade_eye.xml')
# mouth
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
mouth_cascade.load('haarcascades/haarcascade_mcs_mouth.xml')
# nose
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
nose_cascade.load('haarcascades/haarcascade_mcs_nose.xml')


def hog(image_face):
    if image_face.shape[0] == 0 or image_face.shape[1] == 0:
        return 0
    winSize = (32, 32)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    numBins = 9
    hog_descriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, numBins)
    h = hog_descriptor.compute(image_face)
    return sum(h)


def face_detect(image_face):
    img = cv2.imread(image_face)

    # face
    faces = face_cascade.detectMultiScale(img, 1.2, 3)
    total_hog = 0
    for (x, y, w, h) in faces:
        roi_gray = img[y:y + h, x:x + w]
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # eye
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 3)
        hog_eye = 0
        for (ex, ey, ew, eh) in eyes:
            hog_eye += hog(img[ex: ey, ex + ew: ey + eh])
            cv2.rectangle(roi_gray, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # mouth
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 5)
        hog_mouse = 0
        for (mx, my, mw, mh) in mouth:
            hog_mouse += hog(img[mx: my, mx + mw: my + mh])
            cv2.rectangle(roi_gray, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

        # nose
        nose = nose_cascade.detectMultiScale(roi_gray, 1.2, 5)
        hog_nose = 0
        for (nx, ny, nw, nh) in nose:
            hog_nose += hog(img[nx: ny, nx + nw: ny + nh])
            cv2.rectangle(roi_gray, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2)

        total_hog += hog_eye + hog_mouse + hog_nose

    # plt.imshow(img)
    # plt.show()
    print(float(total_hog))
    return float(total_hog)


def load_data_name_label():
    """Loads the name and the corresponding label from the csv file."""
    with open("500_picts_satz.csv") as f:
        lines = [line.split() for line in f]  # create a list of lists

    lines.pop(0)
    data_arr = np.char.split(lines, sep=',').flatten()
    return data_arr


def load_data():
    """Loads the gray scale image data and then split the entire database randomly
    into 75% train, 15% validation, 15% test"""
    data_label = load_data_name_label()
    data = []
    labels = []
    for i in data_label:
        img = cv2.imread("og_500/"+str(i[1]))
        data.append(hog(img))
        if i[2] == 'fear':
            labels.append(0)
        if i[2] == 'happiness':
            labels.append(1)
        if i[2] == 'neutral':
            labels.append(2)
        if i[2] == 'anger':
            labels.append(3)
        if i[2] == 'sad':
            labels.append(4)

    data = np.array(data)
    labels = np.array(labels)

    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.15, train_size=0.7,
                                                        shuffle=True, stratify=labels)
    X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test, test_size=0.5, train_size=0.5,
                                                              shuffle=True, stratify=Y_test)

    return X_train, Y_train, X_validate, Y_validate, X_test, Y_test


def select_knn_model():
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test = load_data()

    best_k = -1
    best_score = 0.0
    train_error = []
    validation_error = []

    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)
        train_score = knn.score(X_train, Y_train)
        train_error.append(train_score)
        validation_score = knn.score(X_validate, Y_validate)
        validation_error.append(validation_score)
        if validation_score > best_score:
            best_score = validation_score
            best_k = k

    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X_train, Y_train)
    best_k_accuracy = knn_best.score(X_test, Y_test)
    print("The best validation score is " + str(best_score) + " with k = " + str(best_k) +
          ", and its accuracy on the test data is " + str(best_k_accuracy) + ".")

    x_axis = list(range(1, 21))

    plt.figure("KNN accuracy")
    plt.plot(x_axis, train_error, '-o')
    plt.plot(x_axis, validation_error, '-o')
    plt.xlabel("K number of nearest neighbours")
    plt.ylabel("training/validation accuracy")
    plt.legend(["training accuracy", "validation accuracy"])
    plt.show()


if __name__ == "__main__":
    select_knn_model()
