import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD
import argparse
import numpy as np
from utils.LogisticRegression import LinearLogisticRegression, RegularLogisticRegression
from utils.RDA import RegularizedDiscriminantAnalysis
from utils.DeepLearn import DeepLearningModel
from utils.SVM import MultiClassSVM

def SVD(x, d):
    svd = TruncatedSVD(n_components=d)
    newX = svd.fit_transform(x)
    return newX

def svd_dimension_reduction(data, information_retained=0.8):
    U, S, VT = np.linalg.svd(data, full_matrices=False)
    total_variance = np.sum(S**2)
    cumulative_variance = np.cumsum(S**2)
    # print(cumulative_variance[499] / total_variance)
    # print(total_variance, cumulative_variance)
    k = np.where(cumulative_variance >= total_variance * information_retained)[0][0] + 1
    print(k)
    reduced_data = np.dot(U[:, :k], np.diag(S[:k]))
    return reduced_data

parser = argparse.ArgumentParser()

if __name__ == '__main__':
    # parser.add_argument('--classify', type=str)
    # parser.add_argument('--selection', type=str)
    with open('./train_feature.pkl', 'rb') as f:
        raw_train_data = pickle.load(f).toarray()
    train_label = np.load('train_labels.npy')
    with open('./test_feature.pkl', 'rb') as f:
        raw_test_data = pickle.load(f).toarray()
    train_data = SVD(raw_train_data, d=733)
    test_data = SVD(raw_test_data, d=733)
    print(train_data.shape)
    # train_data = svd_dimension_reduction(raw_train_data, 0.5)

    accuracies = []
    AICs = []
    BICs = []
    """ K-fold cross-validation """
    k = 10
    X_folds = np.array_split(train_data, k)
    y_folds = np.array_split(train_label, k)
    for i in range(k):
        X_train = np.concatenate([X_folds[j] for j in range(k) if j != i])
        y_train = np.concatenate([y_folds[j] for j in range(k) if j != i])
        X_test = X_folds[i]
        y_test = y_folds[i]
        
        # model = RegularLogisticRegression(reg_lambda=0)
        # model = RegularizedDiscriminantAnalysis(alpha=0.995, gamma=0.995)
        # model = DeepLearningModel(hidden=[256, 256], lr=0.001)
        model = MultiClassSVM()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # aic = model.compute_aic(X_test, y_test)
        # bic = model.compute_bic(X_test, y_test)

        accuracy = np.sum([pred == y_test]) / np.shape(X_test)[0]
        accuracies.append(accuracy)
        # AICs.append(aic)
        # BICs.append(bic)
        print(f"Fold {i+1}, Accuracy: {accuracy:.3f}")
    print(f"Average Cross-Validation Accuracy: {np.mean(accuracies):.3f}")
    # print(f"Cross-Validation: AIC={np.mean(AICs)}, BIC={np.mean(BICs)}")

    """ Bootstrap """
    # for i in range(10):
    #     train_idx = np.random.choice(len(train_data), size=len(train_data), replace=True)
    #     test_idx = [idx for idx in range(len(train_data)) if idx not in train_idx]

    #     X_train, y_train = train_data[train_idx], train_label[train_idx]
    #     X_test, y_test = train_data[test_idx], train_label[test_idx]

    #     # model = RegularLogisticRegression(reg_lambda=0)
    #     model = RegularizedDiscriminantAnalysis(alpha=0.97, gamma=0.97)
    #     model.fit(X_train, y_train)
    #     pred = model.predict(X_test)
    #     aic = model.compute_aic(X_test, y_test)
    #     bic = model.compute_bic(X_test, y_test)

    #     accuracy = np.sum([pred == y_test]) / np.shape(X_test)[0]
    #     accuracies.append(accuracy)
    #     AICs.append(aic)
    #     BICs.append(bic)
    #     print(f"Iteration {i+1}, Accuracy: {accuracy:.3f}")
    # print(f"Bootstrap Accuracy: mean={np.mean(accuracies):.3f}, std={np.std(accuracies):.3f}")
    # print(f"Bootstrap: AIC={np.mean(AICs)}, BIC={np.mean(BICs)}")
