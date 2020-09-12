from json import load
from random import shuffle
from math import floor
from sklearn.neural_network import MLPClassifier
from joblib import dump

with open('run/data.json') as file:
    array = load(file)
shuffle(array)
in_rows = [x['in'] for x in array]
out_rows = [x['out'] for x in array]

split = floor(len(in_rows) * 0.60)

in_train_rows = in_rows[:split]
out_train_rows = out_rows[:split]

clf = MLPClassifier(hidden_layer_sizes=(64, 8), max_iter=5000, tol=-1, verbose=True)
clf.fit(in_train_rows, out_train_rows)

in_test_rows = in_rows[split:]
out_test_rows = out_rows[split:]

score = clf.score(in_test_rows, out_test_rows)
print(f'Test score is {score}')

dump(clf, 'run/model.jlb')
