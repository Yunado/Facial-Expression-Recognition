from Load_Data import *
from sklearn.neighbors import KNeighborsClassifier


def select_knn_model(label_size, method_type):
    X_train, Y_train, X_validate, Y_validate, X_test, Y_test = load_data(label_size, method_type)

    best_k = -1
    best_score = 0.0
    train_error = []
    validation_error = []
    k_range = range(1, 51)

    for k in k_range:
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

    x_axis = list(k_range)

    plt.figure("KNN accuracy")
    plt.plot(x_axis, train_error, '-o')
    plt.plot(x_axis, validation_error, '-o')
    plt.xlabel("K number of nearest neighbours")
    plt.ylabel("training/validation accuracy")
    plt.legend(["training accuracy", "validation accuracy"])
    if label_size == 0 and method_type == 0:
        plt.title("KNN accuracy with 5 labels. HoG")
    elif label_size == 0 and method_type == 1:
        plt.title("KNN accuracy with 5 labels. Facial Landmarks")
    elif label_size == 1 and method_type == 0:
        plt.title("KNN accuracy with 3 labels: happiness, neutral, anger. HoG.")
    else:   # label_size == 1 method type == 1
        plt.title("KNN accuracy with 3 labels: happiness, neutral, anger. Facial Landmarks.")
    plt.show()


if __name__ == "__main__":
    select_knn_model(0, 0)
    # select_knn_model(0, 1)
    # select_knn_model(1, 0)
    # select_knn_model(1, 1)

