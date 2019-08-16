from generate_X_y_files import train_X_data_path, train_y_data_path, test_X_data_path, test_y_data_path
from generate_X_y_files import write_X_y_file
from sklearn.decomposition import PCA
import json
import numpy as np
import pickle

pca_save_path = "pca_model/pca.pkl"
explain_ratio = 0.95


def display_component(X):
    pca = PCA(n_components=X.shape[1])
    pca.fit(X_train)
    ratios = pca.explained_variance_ratio_
    cum_ratios = []
    cum_ratio = 0
    print("component\t ratio\t cumulative_ratio")
    for i in range(0, len(ratios)):
        ratio = ratios[i]
        cum_ratio += ratio
        cum_ratios.append(cum_ratio)
        print("%d\t\t\t %.3f\t %.3f" % (i + 1, ratio, cum_ratio))

    import matplotlib.pyplot as plt
    x = [i for i in range(0, len(ratios))]
    y = cum_ratios
    plt.scatter(x, y, color="red", marker="o")
    plt.plot(x, y, color="blue")
    plt.show()


def transform_train_test(X_train, X_test, target_feature=None, explain_ratio=0.95):
    pca = pca_instance(X_train, target_feature, explain_ratio)
    return pca.transform(X_train), pca.transform(X_test), pca


def save_pca(pca, path):
    pca_str = pickle.dumps(pca)
    with open(path, "wb") as fh:
        fh.write(pca_str)


def load_pca(path):
    fh = open(path, "rb")
    pca = pickle.load(fh)
    return pca


def pca_instance(X_train, target_feature=None, explain_ratio=0.95):
    assert not target_feature or target_feature <= X_train.shape[1], \
        "target_feature count must be no greater than original."

    if not target_feature:
        pca = PCA(explain_ratio)
    else:
        pca = PCA(target_feature)

    pca.fit(X_train)
    return pca


if __name__ == "__main__":
    X_train = np.array(json.load(open(train_X_data_path)))
    X_test = np.array(json.load(open(test_X_data_path)))
    y_train = np.array(json.load(open(train_y_data_path)))
    y_test = np.array(json.load(open(test_y_data_path)))
    # display_component(X_train)

    X_train_reduced, X_test_reduced, pca = transform_train_test(X_train, X_test, explain_ratio=explain_ratio)
    write_X_y_file(X_train_reduced, y_train, "train_data/X_train_reduced.json", "train_data/y_train.json")
    write_X_y_file(X_test_reduced, y_test, "test_data/X_test_reduced.json", "test_data/y_test.json")
    save_pca(pca, pca_save_path)

    print("pca found %d critical features." % len(pca.components_))

    print("done")

