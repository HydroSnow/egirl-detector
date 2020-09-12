from json import dump
from csv import reader
from common import get_image_array

rows = []

with open('run/players.csv') as csvfile:
    csv = reader(csvfile)
    for row in csv:
        name, uuid, gender = row
        print(f'{name} ({uuid})')
        array = get_image_array(uuid)
        obj = { 'in': array, 'out': gender }
        rows.append(obj)

with open('run/data.json', 'w') as outfile:
    dump(rows, outfile)
