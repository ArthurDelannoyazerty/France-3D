import json
import geopandas as gpd

from shapely.geometry import shape
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)

gpkg_folder = 'data/gpkg/'

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_geometry')
def handle_geometry(data):
    try:
        # Convert GeoJSON to GeoDataFrame
        features = json.loads(data)['features']
        geometries = [shape(feature['geometry']) for feature in features]
        geo_df = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")

        # Save as a GeoPackage
        output_file = f'{int(datetime.now().timestamp())}--{request.sid}--gpkg_zone.gpkg'
        output_filepath = gpkg_folder + output_file
        geo_df.to_file(output_filepath, driver="GPKG")
        print(f"GeoPackage saved: {output_file}")
    except Exception as e:
        print(f"Error processing geometry: {e}")

if __name__ == '__main__':
    socketio.run(app, debug=True)
