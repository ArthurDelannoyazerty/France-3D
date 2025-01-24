import json
import geopandas as gpd

from shapely.geometry import shape
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)

gpkg_folder = 'data/gpkg/'
GEOJSON_PRETTY_PRINT = True

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_geometry')
def handle_geometry(geojson):
    try:
        output_file = f'{int(datetime.now().timestamp())}--{request.sid}.geojson'
        output_filepath = gpkg_folder + output_file
        with open(output_filepath, 'w') as f:
            if GEOJSON_PRETTY_PRINT:
                f.write(json.dumps(json.loads(geojson), indent=1))
            else:
                f.write(geojson)
        print(f"GeoPackage saved: {output_file}")
        socketio.emit('geojson_received')
    except Exception as e:
        print(f"Error processing geometry: {e}")

if __name__ == '__main__':
    socketio.run(app, debug=True)
