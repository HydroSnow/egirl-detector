from csv import reader
from requests import get
from PIL import Image
from io import BytesIO
from numpy import concatenate
from sklearn.neural_network import MLPClassifier
from joblib import dump

in_rows = []
out_rows = []

with open('players.csv') as csvfile:
    csv = reader(csvfile)
    for row in csv:
        name, uuid, gender = row

        response = get(f'https://crafatar.com/avatars/{uuid}.png?size=8')
        buffer = BytesIO(response.content)
        image = Image.open(buffer)
        data = image.getdata()
        array = concatenate(data, axis=0)

        in_rows.append(array)
        out_rows.append(gender)

clf = MLPClassifier(hidden_layer_sizes=(20, 20), solver='adam', alpha=0.0002, verbose=True)
clf.fit(in_rows, out_rows)
dump(clf, 'model.jlb')
