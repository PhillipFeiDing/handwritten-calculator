from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
import pickle


train_X_data_path = "train_data/X_train_reduced.json"
train_X_original_data_path = "train_data/X_train.json"
train_y_data_path = "train_data/y_train.json"
test_X_data_path = "test_data/X_test_reduced.json"
test_X_original_data_path = "test_data/X_test.json"
test_y_data_path = "test_data/y_test.json"
model_save_path = "classifier_model/clf.pkl"


poly_degree = 2
penalty = "l2" # l1 = LASSO, l2 = Ridge
C = 0.2 # regularization smaller = stronger
multi_class = "ovr"
solver = "newton-cg"


def PolynomialLogisticRegression(solver=solver, degree=poly_degree, C=C, penalty=penalty, multi_class=multi_class):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression(solver=solver, C=C, penalty=penalty, n_jobs=-1, multi_class=multi_class))
    ])


def train_classifier(X_train, y_train):
    clf = PolynomialLogisticRegression()

    clf.fit(X_train[:, :], y_train[:])

    return clf


def load_data(X_data_path, y_data_path):
    X = np.array(json.load(open(X_data_path)))
    y = np.array(json.load(open(y_data_path)))

    np.random.seed(666)
    perm = np.random.permutation(len(y))
    X = np.array([X[perm[i], :] for i in perm])
    y = np.array([y[perm[i]] for i in perm])

    return X, y


def save_clf(clf, path):
    clf_str = pickle.dumps(clf)
    with open(path, "wb") as fh:
        fh.write(clf_str)


def load_clf(path):
    fh = open(path, "rb")
    clf = pickle.load(fh)
    return clf


if __name__ == "__main__":
    X_train, y_train = load_data(train_X_data_path, train_y_data_path)

    clf = train_classifier(X_train, y_train)

    X_test, y_test = load_data(test_X_data_path, test_y_data_path)
    X_test_original, y_test = load_data(test_X_original_data_path, test_y_data_path)
    y_predict = clf.predict(X_test)

    count = 0
    for i in range(len(y_predict)):
        if y_predict[i] != y_test[i]:
            if count < 50:
                # plot_feature(X_test_original[i, :])
                pass
            elif count == 50:
                print("more false detections are omitted. (> 50)")
            count = count + 1

    print("accuracy score on train: %.6f" % clf.score(X_train, y_train))
    print("accuracy score on test: %.6f" % (np.sum(y_test == y_predict) / len(y_test)))
    print("%d errors found in %d samples" % (count, len(y_test)))

    save_clf(clf, model_save_path)


