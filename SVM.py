from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from Load_Data import *


def svm_model(label_size, method_type):
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test = load_data(label_size, method_type)

    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, Y_train)

    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(X_test)

    print(grid_predictions)

    # print classification report
    print(classification_report(Y_test, grid_predictions))


if __name__ == "__main__":
    svm_model(0, 0)
