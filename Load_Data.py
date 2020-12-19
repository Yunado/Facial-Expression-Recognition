from sklearn.model_selection import train_test_split
from imutils import face_utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import dlib


def facial_landmark(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    rects = detector(image, 0)

    landmarks = None
    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
        landmarks = shape

    return landmarks


def gradientMagnitude(image_face):
    """Compute the magnitude of gradients of an image."""
    g_x = cv2.Sobel(image_face, cv2.CV_64F, 1, 0, ksize=3)
    g_y = cv2.Sobel(image_face, cv2.CV_64F, 0, 1, ksize=3)
    g = np.sqrt(np.square(g_x) + np.square(g_y))
    return g


def edge_detect(image_face):
    edge = cv2.Canny(image_face, 100, 100)
    return edge


def hog(image_face):
    winSize = (32, 32)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    numBins = 9
    hog_descriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, numBins)
    h = hog_descriptor.compute(image_face)
    return h


def diff(img_g, default):
    diff_euclidean_norm = np.linalg.norm(default - img_g)
    return diff_euclidean_norm


def load_data_label():
    """Loads the data and the corresponding label from the csv file."""
    with open("image_labels.csv") as f:
        lines = [line.split() for line in f]  # create a list of lists

    lines.pop(0)
    data_arr = np.char.split(lines, sep=',').flatten()
    filtered_arr = []

    for i in data_arr:
        if i[2] == "happiness" or i[2] == "neutral" or i[2] == "anger" or i[2] == "sadness":
            filtered_arr.append(i)

    filtered_arr = np.array(filtered_arr)

    return filtered_arr


def load_data(choice_input_method):
    """Loads the gray scale image data and then split the entire database randomly
    into 75% train, 15% validation, 15% test"""
    Default_Face = cv2.imread('og/Spencer_Abraham_0003.jpg')
    Default_Face = Default_Face[50:300, 50:300]
    Default_Face_g = gradientMagnitude(Default_Face)
    Default_Face_edge = edge_detect(Default_Face)
    Default_Face_Hog = hog(Default_Face)
    Default_Facial_Feature = facial_landmark(Default_Face)

    data_label = load_data_label()

    data = []
    labels = []

    label_count = [0, 0, 0, 0, 0]
    for i in data_label:
        print("at data og/" + str(i[1]))

        img_face = cv2.imread("og/" + str(i[1]))

        if choice_input_method == 0:  # GrayScale
            data.append([diff(img_face[50:300, 50:300], Default_Face)])
            if i[2] == 'sadness':
                img_face_g = gradientMagnitude(cv2.flip(img_face, 1)[50:300, 50:300])
                data.append([diff(img_face_g, Default_Face_g)])
        if choice_input_method == 1:  # Gradient
            img_face_g = gradientMagnitude(img_face[50:300, 50:300])
            data.append([diff(img_face_g, Default_Face_g)])
            if i[2] == 'sadness':
                img_face_g = gradientMagnitude(cv2.flip(img_face, 1)[50:300, 50:300])
                data.append([diff(img_face_g, Default_Face_g)])
        elif choice_input_method == 2:  # Edge
            image_edge = edge_detect(img_face[50:300, 50:300])
            data.append([diff(image_edge, Default_Face_edge)])
            if i[2] == 'sadness':
                image_edge = edge_detect(cv2.flip(img_face, 1)[50:300, 50:300])
                data.append([diff(image_edge, Default_Face_edge)])
        elif choice_input_method == 3:  # HoG
            image_HoG = hog(img_face[50:300, 50:300])
            data.append([diff(image_HoG, Default_Face_Hog)])
            if i[2] == 'sadness':
                image_HoG = hog(cv2.flip(img_face, 1)[50:300, 50:300])
                data.append([diff(image_HoG, Default_Face_Hog)])
        else:   # Facial LandMark
            facial_feature = facial_landmark(img_face)
            if facial_feature is None:
                continue
            data.append([diff(facial_feature, Default_Facial_Feature)])
            if i[2] == 'sadness':
                facial_feature = facial_landmark(cv2.flip(img_face, 1))
                label_count[4] += 1
                labels.append(4)
                if facial_feature is None:
                    continue
                data.append([diff(facial_feature, Default_Facial_Feature)])
                label_count[4] += 1
                labels.append(4)
                continue
        if i[2] == 'fear':
            label_count[0] += 1
            labels.append(0)
        if i[2] == 'happiness':
            label_count[1] += 1
            labels.append(1)
        if i[2] == 'neutral':
            label_count[2] += 1
            labels.append(2)
        if i[2] == 'anger':
            label_count[3] += 1
            labels.append(3)
        if i[2] == 'sadness':
            label_count[4] += 1
            labels.append(4)

            label_count[4] += 1
            labels.append(4)

    print("label counts: happiness: " + str(label_count[1]) + ";\nneutral: " + str(label_count[2]) +
          ";\nanger: " + str(label_count[3]) + ";\nsadness: " + str(label_count[4]))

    data = np.array(data)
    labels = np.array(labels)

    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.15, train_size=0.7,
                                                        shuffle=True, stratify=labels)
    X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test, test_size=0.5, train_size=0.5,
                                                              shuffle=True, stratify=Y_test)

    return X_train, Y_train, X_validate, Y_validate, X_test, Y_test


if __name__ == "__main__":
    img = cv2.imread("og/Spencer_Abraham_0003.jpg")
    plt.imshow(img)
    plt.show()

    plt.imshow(gradientMagnitude(img).astype(np.uint8), cmap='gray')
    plt.show()

    plt.imshow(edge_detect(img), cmap='gray')
    plt.show()

    feature = facial_landmark(img)
    for (x, y) in feature:
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    plt.imshow(img)
    plt.show()
