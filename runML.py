# Deby Katz - Aug 2017

import argparse
import os
import math
import random
import statistics
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import spectral_clustering, AffinityPropagation, KMeans, Birch, MeanShift, estimate_bandwidth

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

rng = np.random.RandomState(42)
random.seed(42)

# Arguments
# balance_training
# balance_testing
# supervised or unsupervised
# directory from which to gather data
# classification algorithm
# classification parameters
# reduce_signals
# one_class
# normalization ?
################################################################################
#
# balance
#
################################################################################
def balance(X_np, y_np, by_duplicating=True):
    X = X_np.tolist()
    y = y_np.tolist()

    # Find the number of positives and negatives
    pos = y.count(1)
    neg = y.count(0)
    assert(pos+neg == len(y))
    assert(len(y) == len(X))
    if pos == 0 or neg == 0:
        return [X_np, y_np]

    # While the number of examples is different by more than one
    while (abs(pos-neg) > 1):
        # Find the class that contains fewer examples
        minority = 1 if pos < neg else 0

        # choose a minority example to duplicate, pseudorandomly
        found = False
        while not found:
            rand_index = random.randrange(0, len(y))
            if y[rand_index] == minority:
                y.append(y[rand_index])
                X.append(X[rand_index])
                found = True
        pos = y.count(1)
        neg = y.count(0)

    return [np.array(X), np.array(y)]

################################################################################
#
# get_name
#
################################################################################
def get_name(data_dir):
    # todo: make this more general
    possibilities = (data_dir.strip()).split(os.sep)
    try:
        if possibilities[-1] == '':
            return possibilities[-3]
        else:
            return possibilities[-2]
    except:
        return possibilities[0]
################################################################################
#
# data_one_file
#
################################################################################
def data_one_file(filename):
    file_obj = open(filename, 'r')
    lines = list(file_obj)
    file_obj.close()
    data = [ x.strip().split() for x in lines ]
    return data
################################################################################
#
# get_data
#
################################################################################
def get_data(data_dir, file_suffix, file_contains, pass_word="nominal",
             fail_word="attitude_bug", max_samples=0, ignore_index=False):
    print("pass_word: %s, fail_word: %s" % (pass_word, fail_word))
    assert(os.path.isdir(data_dir)), data_dir
    data_files = [ os.path.join(data_dir, x) for x in os.listdir(data_dir) ]
    print("len(data_files): %d" % len(data_files))
    index_files = [ x for x in data_files if x.endswith("index.txt") ]
    if file_suffix:
        data_files = [ x for x in data_files if x.endswith(file_suffix) ]
    print("len(data_files): %d" % len(data_files))
    if file_contains:
        print("file_contains: %s" % file_contains)
        print("data_files[0]: %s" % data_files[0])
        data_files = [ x for x in data_files if file_contains in x ]
    print("len(data_files): %d" % len(data_files))
    #data_files.sort()

    if len(index_files) == 1 and not ignore_index:
        print("Working with an index file")
        index = data_one_file(index_files[0])
        #print(index)

        X = []
        used_files = []
        y = []
        index.sort()
        for item in index:
            #print(item[0])
            #data_file = [ x for x in data_files if (item[0] + "_") in x]
            #assert(len(data_file) <= 1), ("Too many data files for each index "
            #                              + "entry. Try adding --file_suffix\n"
            #                              + "%s; %s" % (data_file, item[0]))
            data_file = []
            for one_file in data_files:
                if (item[0] + "_") in one_file:
                    data_file = [one_file]
                    data_files.remove(one_file)
                    break
            if len(data_file) > 0:
                one_file_data = data_one_file(data_file[0])
                data_explanations = [x[0] for x in one_file_data]
                data_values = [x[1] for x in one_file_data]
                X.append(data_values)
                used_files.append(item[0])
                y.append(1 if item[1] == 'fail' else 0)
                #data_files.remove(data_file[0])

    else:
        assert(len(index_files) == 0 or ignore_index), ("Wrong number of index files: %d" %
                                    len(index_files))
        print("Working without an index file")
        X = []
        used_files = []
        y = []
        print("len(data_files): %d" % len(data_files))
        data_files = [ x for x in data_files if 
                       pass_word in x or fail_word in x ]
        for data_file in data_files:
            assert(pass_word in data_file or fail_word in data_file), data_file
            # assert(not (pass_word in data_file and fail_word in data_file)), data_file
            if len(data_file) > 0:
                one_file_data = data_one_file(data_file)
            #print(one_file_data)
                data_explanations = [x[0] for x in one_file_data]
                data_values = [x[1] for x in one_file_data]
                X.append(data_values)
                used_files.append(data_file)
                y.append(0 if pass_word in os.path.basename(data_file) else 1)
    assert(len(X) == len(y))
    assert(len(X) == len(used_files))

    if (max_samples < len(X)) and (max_samples != 0):
        zipped = list(zip(X, y, used_files))
        print(type(zipped))
        choices = random.sample(zipped, max_samples)
        X = [ x[0] for x in choices]
        y = [ x[1] for x in choices]
        used_files = [ x[2] for x in choices]
        assert(len(X) == len(y))
    assert(len(X) == len(used_files))
    for i in range(len(X)):
        assert(len(data_explanations) == len(X[i]))
    print("Number of passing data points: %d\nNumber of failing data points: %d"
          % (y.count(0), y.count(1)))
    return [X, y, used_files, data_explanations]


################################################################################
#
# compute_pos_neg
#
################################################################################
def compute_pos_neg(y_test, predictions):
    zipped = zip(y_test, predictions)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for gt, pred in zipped:
        if gt == 1:
            if pred == 1:
                TP += 1
            elif pred == 0:
                FN += 1
            else:
                assert False
        elif gt == 0:
            if pred == 1:
                FP += 1
            elif pred == 0:
                TN += 1
            else:
                assert False, ("gt: %s, pred: %s" % (gt, pred))
        else:
            assert False

    print("TP: %s, TN: %s, FP: %s, FN: %s" % (TP, TN, FP, FN))
    return [TP, TN, FP, FN]
################################################################################
#
# run_supervised
#
################################################################################
def run_supervised(data, balance_training, balance_testing):
    X, y, used_files, data_explanations = data

    X = np.array(X)
    y = np.array(y)

    if balance_training and balance_testing:
        X, y = balance(X, y)

    accumulated_metrics = {'acc': [], 'prec': [], 'rec': [], 'f': []}

    # k-fold if enough samples
    if len(X) >= 100:
        K = 10
    else:
        K = math.floor(len(X)/10)
    print("using %s folds" % K)
    kf = KFold(n_splits=K, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if balance_training:
            assert(len(X_train) == len(y_train))
            print("train length before balancing: %s" % len(X_train))
            X_train, y_train = balance(X_train, y_train)
            assert(len(X_train) == len(y_train))
            print("train length after balancing: %s" % len(X_train))
        if balance_testing:
            assert(len(X_test) == len(y_test))
            print("test length before balancing: %s" % len(X_test))
            X_test, y_test = balance(X_test, y_test)
            assert(len(X_test) == len(y_test))
            print("test length after balancing: %s" % len(X_test))


        clf = DecisionTreeClassifier(random_state=42)
        #print("shape of X_train: %s" % str(X_train.shape))
        #print("shape of y_train: %s" % str(y_train.shape))
        clf = clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        #print(predictions)
        accuracy = accuracy_score(y_test, predictions, normalize=True)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f_score = f1_score(y_test, predictions)
        print("%.2f & %.2f & %.2f & %.2f" %
              (accuracy, precision, recall, f_score))
        accumulated_metrics['acc'].append(float(accuracy))
        accumulated_metrics['prec'].append(float(precision))
        accumulated_metrics['rec'].append(float(recall))
        accumulated_metrics['f'].append(float(f_score))
        assert(len(y_test) == len(predictions))
        #print("y_test")
        #print(y_test)
        #print("predictions")
        #print(predictions)
        compute_pos_neg(y_test, predictions)

    mean_acc = statistics.mean(accumulated_metrics['acc'])
    mean_prec = statistics.mean(accumulated_metrics['prec'])
    mean_rec = statistics.mean(accumulated_metrics['rec'])
    mean_f = statistics.mean(accumulated_metrics['f'])


    return [mean_acc, mean_prec, mean_rec, mean_f]


################################################################################
#
# accuracy_metrics_from_predictions
#
################################################################################
def accuracy_metrics_from_predictions(y, predictions):
    #print("predictions")
    #print(predictions.tolist())
    predictions_list = predictions.tolist()
    predictions_list = [ x if x ==  1 else 0 for x in predictions_list]
    reverse_list = [ 0 if x == 1 else 1 for x in predictions_list]
    #for prediction in predictions_list:
    #    if prediction == 1:
    #        reverse_list.append(0)
    #    elif prediction == 0:
    #        reverse_list.append(1)
    reverse_predictions = np.array(reverse_list)
    #print("reverse predictions")
    #print(reverse_predictions.tolist())

    accuracy = accuracy_score(y, predictions, normalize=True)
    accuracy_reverse = accuracy_score(y, reverse_predictions, normalize=True)
    if accuracy_reverse > accuracy:
        predictions = reverse_predictions
        accuracy = accuracy_reverse
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f_score = f1_score(y, predictions)
    compute_pos_neg(y, predictions)

    print("%.2f & %.2f & %.2f & %.2f" %
          (accuracy, precision, recall, f_score))
    return [accuracy, precision, recall, f_score]

################################################################################
#
# run_unsupervised
#
################################################################################
def run_unsupervised(data):
    X, y, used_files, data_explanations = data

    X = np.array(X,dtype=int)
    print(X.shape)
    y = np.array(y)

    outliers_fraction = 0.25
    n_samples = len(X)


    classifiers = {
        "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
                                         kernel="rbf", gamma=0.1,
                                         random_state=42),
        "Robust covariance": EllipticEnvelope(contamination=outliers_fraction,
                                              random_state=42),
        "Isolation Forest": IsolationForest(max_samples=n_samples,
                                            contamination=outliers_fraction,
                                            random_state=42),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=35,
            contamination=outliers_fraction)
        #"K-means": KMeans(n_clusters=2),
        #"MeanShift": MeanShift(bandwidth=estimate_bandwidth(X))
        #"Birch": Birch(branching_factor=50, n_clusters = 2, threshold=0.25,
 #                      compute_labels=True)
        }

    for i, (clf_name, clf) in enumerate(classifiers.items()):
        if clf_name == "Local Outlier Factor":
            y_pred = clf.fit_predict(X)
            scores_pred = clf.negative_outlier_factor_
        else:
            clf.fit(X)
            scores_pred = clf.decision_function(X)
            y_pred = clf.predict(X)
        threshold = stats.scoreatpercentile(scores_pred,
                                            100 * outliers_fraction)
        print("\n\nUnsupervised %s" % clf_name)
        accuracy_metrics_from_predictions(y, y_pred)



    # print("\n\nUnsupervised Spectral Clustering")
    # predictions = spectral_clustering(X, n_clusters=2)
    #print("\n\nUnsupervised Affininty Propagation")
    #af = AffinityPropagation(preference=-50).fit(X)
    #predictions = af.labels_
    #print(predictions)
    #accuracy = accuracy_score(y, predictions, normalize=True)
    #precision = precision_score(y, predictions)
    #recall = recall_score(y, predictions)
    #f_score = f1_score(y, predictions)
    #print("%.4f & %.4f & %.4f & %.4f" %
    #      (accuracy, precision, recall, f_score))


    #print("\n\nUnsupervised K-means")
    #predictions = KMeans(n_clusters=2).fit_predict(X)
    #accuracy_metrics_from_predictions(y, predictions)

    #print("\n\nUnsupervised MeanShift")
    #bandwidth = estimate_bandwidth(X)
    #ms = MeanShift(bandwidth=bandwidth)
    #ms.fit(X)
    #labels = ms.labels_
    #print(labels)


    #print("\n\nUnsupervised Birch")
    #brc = Birch(branching_factor=50, n_clusters=2, threshold=0.5,
    #            compute_labels=True)
    #predictions = brc.fit_predict(X)
    #accuracy_metrics_from_predictions(y, predictions)

################################################################################
#
# run_novelty_detection
#
################################################################################
def run_novelty_detection(data):
    # To use one_class as novelty detection, we train only on nominal
    # data.  We then use mixed data to test.

    X, y, used_files, data_explanations = data

    X = np.array(X,dtype=int)
    print(X.shape)
    y = np.array(y)

    outliers_fraction = 0.25
    n_samples = len(X)

    clf = svm.OneClassSVM(random_state=42, kernel='sigmoid')
    accumulated_metrics = {'acc': [], 'prec': [], 'rec': [], 'f': []}

    # 10-fold if enough samples, smaller K if not
    if len(X) >= 100:
        K = 10
    else:
        K = math.floor(len(X)/10)
    print("using %s folds" % K)
    kf = KFold(n_splits=K, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Take only the positive training samples (which have y == 0)
        assert(len(X_train) == len(y_train))
        zipped = zip(X_train, y_train)
        pos_zip = [ x for x in zipped if x[1] == 0 ]
        print("len(pos_zip): %s" % len(pos_zip))
        X_train = np.array([x[0] for x in pos_zip])
        y_train = np.array([x[1] for x in pos_zip])

        clf.fit(X_train)
        y_pred = clf.predict(X_test)
        print(y_pred.tolist())
        if -1 in y_pred:
            y_pred = np.array([ 0 if x == -1 else 1 for x in y_pred])
        assert(len(y_test) == len(y_pred))
        acc, prec, rec, f = accuracy_metrics_from_predictions(y_test, y_pred)
        accumulated_metrics['acc'].append(float(acc))
        accumulated_metrics['prec'].append(float(prec))
        accumulated_metrics['rec'].append(float(rec))
        accumulated_metrics['f'].append(float(f))
        compute_pos_neg(y_test, y_pred)
        mean_acc = statistics.mean(accumulated_metrics['acc'])
    mean_prec = statistics.mean(accumulated_metrics['prec'])
    mean_rec = statistics.mean(accumulated_metrics['rec'])
    mean_f = statistics.mean(accumulated_metrics['f'])


    return [mean_acc, mean_prec, mean_rec, mean_f]



################################################################################
#
# parse_arguments
#
################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--balance_training', action='store_true',
                        default=False)
    parser.add_argument('--balance_testing', action='store_true',
                        default=False)
    parser.add_argument('--supervised', action='store_true', default=False)
    parser.add_argument('--unsupervised', action='store_true', default=False)
    parser.add_argument('--novelty', action='store_true', default=False)
    parser.add_argument('--data_dir', type=str, required=True, action='append')
    parser.add_argument('--file_suffix', type=str)
    parser.add_argument('--file_contains', type=str)
    parser.add_argument('--pass_word', type=str, default="nominal")
    parser.add_argument('--fail_word', type=str, default="attitude_bug")
    parser.add_argument('--limit_samples_num', type=int, default=0)
    parser.add_argument('--ignore_index', action='store_true', default=False)
    parser.add_argument('--places', type=int, default=2)

    args = parser.parse_args()
    return args

################################################################################
#
# main
#
################################################################################
if __name__ == '__main__':
    args = parse_arguments()

    supervised_results = dict()

    for data_dir in args.data_dir:
        data = get_data(data_dir, args.file_suffix, args.file_contains,
                        pass_word=args.pass_word,
                        fail_word=args.fail_word,
                        max_samples=args.limit_samples_num,
                        ignore_index = args.ignore_index)
        # print(data[0])
        # print(data[1])
        # print(data[2])
        # print(data[3])

        prog_name = get_name(data_dir)

        if args.supervised:
            supervised_results[prog_name] = \
                run_supervised(data, args.balance_training,
                               args.balance_testing)
        if args.unsupervised:
            run_unsupervised(data)
        if args.novelty:
            run_novelty_detection(data)

    if supervised_results:
        print("\t& Mean Acc. & Mean Prec. & Mean Rec. & Mean F Score")
        for prog_name, results in supervised_results.items():
            if args.limit_samples_num:
                identifier = args.limit_samples_num
                print("%s & %.2f & %.2f & %.2f & %.2f & %.1f \\\\" %
                      (identifier, results[0], results[1],
                       results[2], results[3], (99.9*float(identifier)/60.0)))
            else:
                identifier = prog_name
                if args.places == 4:
                    print("%s & %.4f & %.4f & %.4f & %.4f" %
                          (identifier, results[0], results[1],
                           results[2], results[3]))
                    pass
                else:
                    print("%s & %.2f & %.2f & %.2f & %.2f" %
                          (identifier, results[0], results[1],
                           results[2], results[3]))
