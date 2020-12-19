from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from Load_Data import *


def svm_model(method_type):
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test = load_data(method_type)

    # defining parameter range
    parameters = {
        "estimator__C": [1, 10, 100, 1000],
        'estimator__gamma': [10, 1, 0.01, 0.001],
        "estimator__kernel": ["rbf"]}

    model_to_set = OneVsRestClassifier(SVC())

    grid = GridSearchCV(model_to_set, parameters, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, Y_train)

    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(X_test)

    if method_type == 0:
        print("SVM classification report. Gradient")
    elif method_type == 1:
        print("SVM classification report. Edge")
    elif method_type == 2:
        print("SVM classification report. HoG")
    elif method_type == 3:
        print("SVM classification report. Facial Landmarks")

    # print classification report
    print(classification_report(Y_test, grid_predictions))


if __name__ == "__main__":
    # svm_model(0)
    svm_model(1)
    # svm_model(2)
    # svm_model(3)
