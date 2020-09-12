from csv import reader
from common import get_image_array
from math import floor
from sklearn.neural_network import MLPClassifier
from joblib import dump

in_rows = []
out_rows = []

with open('players.csv') as csvfile:
    csv = reader(csvfile)
    for row in csv:
        name, uuid, gender = row

        array = get_image_array(uuid)

        in_rows.append(array)
        out_rows.append(gender)

split = floor(len(in_rows) * 0.80)

in_train_rows = in_rows[:split]
out_train_rows = out_rows[:split]

clf = MLPClassifier(hidden_layer_sizes=(20, 20), solver='adam', alpha=0.0002, verbose=True)
clf.fit(in_train_rows, out_train_rows)

in_test_rows = in_rows[split:]
out_test_rows = out_rows[split:]

score = clf.score(in_test_rows, out_test_rows)
print(f'Test score is {score}')

dump(clf, 'model.jlb')
