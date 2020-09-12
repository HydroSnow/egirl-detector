from sklearn.neural_network import MLPClassifier
from joblib import load
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from json import loads, dumps
from common import get_image_array

clf = load('run/model.jlb')

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # chargement des données de la requête
            content_length = int(self.headers['Content-Length'])
            data = loads(self.rfile.read(content_length))
            # prédiction
            prediction = clf.predict([ data ]).tolist()
            # envoi des headers
            self.send_response(200)
            self.end_headers()
            # envoi du contenu
            json_data = dumps(prediction)
            bytes_data = bytes(json_data, 'utf8')
            self.wfile.write(bytes_data)
        except Exception as e:
            # s'il y a une erreur, renvoyer 500
            print(e)
            self.send_response(500)
            self.end_headers()

# démarrage de l'api
httpd = ThreadingHTTPServer(('localhost', 64001), Handler)
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
httpd.server_close()
