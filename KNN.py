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
    return h


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
            hog_eye += hog(img[ex: ex + ew, ey: ey + eh])
            cv2.rectangle(roi_gray, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # mouth
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 5)
        hog_mouse = 0
        for (mx, my, mw, mh) in mouth:
            hog_mouse += hog(img[mx: mx + mw, my: my + mh])
            cv2.rectangle(roi_gray, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

        # nose
        nose = nose_cascade.detectMultiScale(roi_gray, 1.2, 5)
        hog_nose = 0
        for (nx, ny, nw, nh) in nose:
            hog_nose += hog(img[nx: nx + nw, ny: ny + nh])
            cv2.rectangle(roi_gray, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2)

        total_hog += hog_eye + hog_mouse + hog_nose

    # plt.imshow(img)
    # plt.show()

    return float(total_hog)


def hog_face_diff(image_face, default_hog):
    img_face = cv2.imread(image_face)
    image_face_hog = hog(img_face[50:300, 50:300])
    hog_diff_euclidean_norm = np.linalg.norm(default_hog - image_face_hog)
    return hog_diff_euclidean_norm


def load_data_label():
    """Loads the data and the corresponding label from the csv file."""
    with open("500_picts_satz.csv") as f:
        lines = [line.split() for line in f]  # create a list of lists

    lines.pop(0)
    data_arr = np.char.split(lines, sep=',').flatten()
    return data_arr


def load_data_label_3_label():
    """Loads the data and the corresponding label from the csv file. Only keep label happiness, neutral, and anger"""
    with open("500_picts_satz.csv") as f:
        lines = [line.split() for line in f]  # create a list of lists

    lines.pop(0)
    data_arr = np.char.split(lines, sep=',').flatten()
    filtered_arr = []

    for i in data_arr:
        if i[2] == "happiness" or i[2] == "neutral" or i[2] == "anger":
            filtered_arr.append(i)

    filtered_arr = np.array(filtered_arr)
    return filtered_arr


def load_data(choice):
    """Loads the gray scale image data and then split the entire database randomly
    into 75% train, 15% validation, 15% test"""
    Default_Face = cv2.imread('og_500/Spencer_Abraham_0003.jpg')
    Default_Face_Hog = hog(Default_Face[50:300, 50:300])
    if choice == 0:
        data_label = load_data_label()
    else:  # filtered data
        data_label = load_data_label_3_label()
    data = []
    labels = []
    for i in data_label:
        data.append([hog_face_diff("og_500/" + str(i[1]), Default_Face_Hog)])
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


def select_knn_model(choice):
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test = load_data(choice)

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
    if choice == 0:
        plt.title("KNN accuracy with 5 labels")
    else:   # choice == 1
        plt.title("KNN accuracy with 3 labels: happiness, neutral, anger.")
    plt.show()


if __name__ == "__main__":
    select_knn_model(0)
    select_knn_model(1)
    # img = cv2.imread("og_500/Sourav_Ganguly_0003.jpg")
    # #face
    # cv2.rectangle(img, (50, 50), (300, 300), (255, 0, 255), 2)
    # #eye
    # cv2.rectangle(img, (75, 125), (150, 170), (255, 0, 255), 2)
    # cv2.rectangle(img, (200, 125), (275, 170), (255, 0, 255), 2)
    # #nose
    # cv2.rectangle(img, (200, 125), (150, 225), (255, 0, 255), 2)
    # #mouth
    # cv2.rectangle(img, (125, 250), (225, 280), (255, 0, 255), 2)
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(img[50:300, 50:300])
    # plt.show()
