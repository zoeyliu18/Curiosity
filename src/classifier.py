import os
import argparse
import numpy as np
import itertools

np.random.seed(42)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score


def create_model(X, y):
    model = LinearSVC(verbose=True, dual=False,
                      max_iter=10000)
    model.fit(X, y)

    return model

def feature_ranking(X, y, feats):
    model = LinearSVC(dual=False, verbose=True,
                      max_iter=10000)
    model.fit(X, y)
    coefs = model.coef_.ravel()

    dict_feats = {}
    for c in range(len(coefs)):
        dict_feats[feats[c]] = abs(coefs[c])

    sorted_dict = sorted(dict_feats.items(), key=lambda x: x[1], reverse=True)

    return sorted_dict

def preprocess_ranking(data, dict_feats):
    X = []
    y = []

    with open(data, 'r') as f:
        next(f)
        for line in f:
            elems = line.rstrip('\n').split('\t')
            diff = np.array(dict_feats[elems[0]]) - np.array(dict_feats[elems[1]])
            vector = dict_feats[elems[0]] + dict_feats[elems[1]] + diff.tolist()
            X.append(vector)
            y.append(int(elems[2]))

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X = np.array([np.array(xi) for xi in X])
    y = np.array([np.array(yi) for yi in y])

    return X, y

def preprocess_data(data, train, test, dict_feats):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    with open(data, 'r') as f:
        next(f)
        for line in f:
            elems = line.rstrip('\n').split('\t')
            stud = elems[0].split('.')[0]
            if stud in train:
                diff = np.array(dict_feats[elems[0]]) - np.array(dict_feats[elems[1]])
                vector = dict_feats[elems[0]] + dict_feats[elems[1]] + diff.tolist()
                X_train.append(vector)
                y_train.append(int(elems[2]))
            else:
                diff = np.array(dict_feats[elems[0]]) - np.array(dict_feats[elems[1]])
                vector = dict_feats[elems[0]] + dict_feats[elems[1]] + diff.tolist()
                X_test.append(vector)
                y_test.append(int(elems[2]))

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.array([np.array(xi) for xi in X_train])
    X_test = np.array([np.array(xi) for xi in X_test])
    y_train = np.array([np.array(yi) for yi in y_train])
    y_test = np.array([np.array(yi) for yi in y_test])

    return X_train, y_train, X_test, y_test

def classifier(data, dict_feats, feats):
    list_stud = []

    count_samples = 0
    with open(data, 'r') as f:
        next(f)
        for line in f:
            elems = line.rstrip('\n').split('\t')
            stud = elems[0].split('.')[0]
            list_stud.append(stud)
            count_samples+=1
            
    X, y = preprocess_ranking(data, dict_feats)

    ranking_feats = feature_ranking(X, y, feats)
    
    list_stud = list(np.unique(list_stud))

    accs = []
    skf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in skf.split(list_stud):
        train_essays = []
        test_essays = []
        for index in train_index:
            train_essays.append(list_stud[index])
        for index in test_index:
            test_essays.append(list_stud[index])

        X_train, y_train, X_test, y_test = preprocess_data(data, train_essays,
                                                           test_essays, dict_feats)

        model = create_model(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accs.append(accuracy)

    mean_acc = np.mean(accs)*100
    
    return mean_acc, ranking_feats, count_samples
    
def parse_arg():
    parser = argparse.ArgumentParser(description='Written competence ordering classifier')
    parser.add_argument('-d', '--data_dir', type=str,
                        help='Dir with pairs of essays datasets')
    parser.add_argument('-m', '--monit', type=str,
                        help='File with linguistic features extracted from the corpus')
    parser.add_argument('-o', '--output', type=str,
                        help='Output dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arg()
    data_dir = args.data_dir
    monit = args.monit
    out_dir = args.output
    
    dict_feats = {}
    feats = []
    with open(monit, 'r') as f:
        first_line = f.readline()
        elems = first_line.rstrip('\n').split('\t')
        for el in elems[1:]:
            feats.append(el + '_1')
        for el in elems[1:]:
            feats.append(el + '_2')
        for el in elems[1:]:
            feats.append(el + '_diff')
    with open(monit, 'r') as f:
        next(f)
        for line in f:
            elems = line.rstrip('\n').split('\t')
            dict_feats[elems[0]] = [float(i) for i in elems[1:]]

    datasets = os.listdir(data_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out = open(out_dir + '/results.txt', 'w')
    for f in datasets:
        rank = open(out_dir + '/ranking_' + f + '.txt', 'w')
        rank.write('Feature\tWeight\n')
        data = data_dir + '/' + f
        accuracy, ranking_feats, samples = classifier(data, dict_feats, feats)
        print()
        print('Num samples: ' + str(samples))
        print('Accuracy: ' + str(accuracy) + ' ' + str(f))
        out.write('Num samples: ' + str(samples)
                  + '\nAccuracy: ' + str(accuracy) + ' ' + str(f) + '\n\n')

        for el in ranking_feats:
            rank.write(el[0] + '\t' + str(el[1]) + '\n')
        rank.close()
    out.close()
