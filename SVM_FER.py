import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Load_Data import *


def svm_model(chocie):
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test = load_data(chocie)

    best_score = 0.0
    best_parameter = [0, 0]

    for i in range(0, 6):
        for j in range(0, 8):
            svm_facial = cv2.ml.SVM_create()
            svm_facial.setType(cv2.ml.SVM_C_SVC)
            svm_facial.setKernel(cv2.ml.SVM_LINEAR)
            svm_facial.setGamma(10**(i))
            svm_facial.setC(10**(-j))
            svm_facial.train(X_train, cv2.ml.ROW_SAMPLE, Y_train)

            predict = []
            for i in range(len(X_validate)):
                predict.append(svm_facial.predict(X_validate[i]))
            predict = np.asarray(predict, 'float64')
            correct = 0
            for j in range(len(predict)):
                if int(Y_validate[j]) == int(predict[j][1]):
                    correct = correct + 1
            if correct / len(X_validate) >= best_score:
                best_score = correct / len(X_validate)
                best_parameter = [10**(i), 10**(-j)]

    svm_facial = cv2.ml.SVM_create()
    svm_facial.setType(cv2.ml.SVM_C_SVC)
    svm_facial.setKernel(cv2.ml.SVM_LINEAR)
    svm_facial.setGamma(best_parameter[0])
    svm_facial.setC(best_parameter[1])
    svm_facial.train(X_train, cv2.ml.ROW_SAMPLE, Y_train)

    predict = []
    for i in range(len(X_test)):
        predict.append(svm_facial.predict(X_test[i]))
    predict = np.asarray(predict, 'float64')
    print(predict)
    correct = 0
    for j in range(len(predict)):
        if int(Y_test[j]) == int(predict[j][1]):
            correct = correct + 1
    print('accuracy = ' + str(correct/len(X_test)))

    return svm_facial


if __name__ == "__main__":
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test = load_data(0)

    svm_facial = cv2.ml.SVM_create()
    svm_facial.setType(cv2.ml.SVM_C_SVC)
    svm_facial.setKernel(cv2.ml.SVM_COEF)
    svm_facial.setGamma(10000)
    svm_facial.setC(0.01)
    svm_facial.train(X_train, cv2.ml.ROW_SAMPLE, Y_train)

    predict = []
    for i in range(len(X_test)):
        predict.append(svm_facial.predict(X_test[i]))
    predict = np.asarray(predict, 'float64')

    for i in predict:
        print(i)

    correct = 0
    for j in range(len(predict)):
        if int(Y_test[j]) == int(predict[j][1]):
            correct = correct + 1
    print('accuracy for 5 kinds of expressions: ')
    print(correct/len(X_test))

    X_train, Y_train, X_validate, Y_validate, X_test, Y_test = load_data(1)

    svm_facial = cv2.ml.SVM_create()
    svm_facial.setType(cv2.ml.SVM_C_SVC)
    svm_facial.setKernel(cv2.ml.SVM_COEF)
    svm_facial.setGamma(10000)
    svm_facial.setC(0.01)
    svm_facial.train(X_train, cv2.ml.ROW_SAMPLE, Y_train)

    predict = []
    for i in range(len(X_test)):
        predict.append(svm_facial.predict(X_test[i]))
    predict = np.asarray(predict, 'float64')
    correct = 0
    for j in range(len(predict)):
        if int(Y_test[j]) == int(predict[j][1]):
            correct = correct + 1
    print('accuracy for 3 kinds of expressions: ')
    print(correct / len(X_test))


