import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD
import argparse
import numpy as np
from utils.LogisticRegression import LinearLogisticRegression, RegularLogisticRegression
from utils.RDA import RegularizedDiscriminantAnalysis
from utils.DeepLearn import DeepLearningModel
from utils.SVM import MultiClassSVM
import csv
import copy
import argparse

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
    parser.add_argument('--algo', type=str, default='LR')
    parser.add_argument('--val', type=str, default='CV')
    args = parser.parse_args()
    with open('./train_feature.pkl', 'rb') as f:
        raw_train_data = pickle.load(f).toarray()
    train_label = np.load('train_labels.npy')
    with open('./test_feature.pkl', 'rb') as f:
        raw_test_data = pickle.load(f).toarray()
    all_data = np.concatenate([raw_train_data, raw_test_data], axis=0)
    reduced_all_data = SVD(all_data, d=733)
    train_data = reduced_all_data[:raw_train_data.shape[0],:]
    test_data = reduced_all_data[raw_train_data.shape[0]:,:]
    # train_data = SVD(raw_train_data, d=733)
    # test_data = SVD(raw_test_data, d=733)
    # train_data = svd_dimension_reduction(raw_train_data, 0.5)

    accuracies = []
    AICs = []
    BICs = []
    """ K-fold cross-validation """
    if (args.val=='CV'):
        k = 10
        X_folds = np.array_split(train_data, k)
        y_folds = np.array_split(train_label, k)
        for i in range(k):
            X_train = np.concatenate([X_folds[j] for j in range(k) if j != i])
            y_train = np.concatenate([y_folds[j] for j in range(k) if j != i])
            X_test = X_folds[i]
            y_test = y_folds[i]
            
            if (args.algo == 'LR'):
                model = RegularLogisticRegression(reg_lambda=1)
            elif (args.algo == 'RDA'):
                model = RegularizedDiscriminantAnalysis(alpha=0.99, gamma=0.99)
            elif (args.algo == 'DL'):
                model = DeepLearningModel(hidden=[512, 512], lr=0.001)
            elif (args.algo == 'SVM'):
                model = MultiClassSVM(C=2)
            if (args.algo=='SVM'):
                model.fit(X_train[:2500], y_train[:2500])
            else:
                model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if (args.algo!='SVM'):
                aic = model.compute_aic(X_test, y_test)
                bic = model.compute_bic(X_test, y_test)
                AICs.append(aic)
                BICs.append(bic)

            accuracy = np.sum([pred == y_test]) / np.shape(X_test)[0]
            accuracies.append(accuracy)
            if (accuracy == max(accuracies)): final_model = copy.deepcopy(model)

            print(f"Fold {i+1}, Accuracy: {accuracy:.3f}")
        print(f"Average Cross-Validation Accuracy: {np.mean(accuracies):.3f}")
        print(f"Cross-Validation: AIC={np.mean(AICs)}, BIC={np.mean(BICs)}")
    else:
        """ Bootstrap """
        for i in range(3):
            if (args.algo=='SVM'):
                train_idx = np.random.choice(len(train_data), size=2500, replace=True)
            else:
                train_idx = np.random.choice(len(train_data), size=len(train_data), replace=True)
            test_idx = [idx for idx in range(len(train_data)) if idx not in train_idx]

            X_train, y_train = train_data[train_idx], train_label[train_idx]
            X_test, y_test = train_data[test_idx], train_label[test_idx]

            if (args.algo == 'LR'):
                model = RegularLogisticRegression(reg_lambda=1)
            elif (args.algo == 'RDA'):
                model = RegularizedDiscriminantAnalysis(alpha=0.98, gamma=0.98)
            elif (args.algo == 'DL'):
                model = DeepLearningModel(hidden=[256, 256], lr=0.001)
            elif (args.algo == 'SVM'):
                model = MultiClassSVM(C=2)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if (args.algo!='SVM'):
                aic = model.compute_aic(X_test, y_test)
                bic = model.compute_bic(X_test, y_test)
                AICs.append(aic)
                BICs.append(bic)

            accuracy = np.sum([pred == y_test]) / np.shape(X_test)[0]
            accuracies.append(accuracy)
            if (accuracy == max(accuracies)): final_model = copy.deepcopy(model)
            print(f"Iteration {i+1}, Accuracy: {accuracy:.3f}")
        print(f"Bootstrap Accuracy: mean={np.mean(accuracies):.3f}, std={np.std(accuracies):.3f}")
        print(f"Bootstrap: AIC={np.mean(AICs)}, BIC={np.mean(BICs)}")

    """output the test result"""
    test_pred = final_model.predict(test_data)
    with open("output.csv", "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        name=['ID','label']
        csv_writer.writerow(name)
        for i, pred in enumerate(test_pred):
            csv_writer.writerow([int(i), int(pred)])
        f.close()