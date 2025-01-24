import json
import geopandas as gpd

from shapely.geometry import shape
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from datetime import datetime
from tiles import geodataframe_from_leaflet_to_ign, geodataframe_from_ign_to_leaflet

app = Flask(__name__)
socketio = SocketIO(app)

gpkg_folder = 'data/geojson/order/'
GEOJSON_PRETTY_PRINT = True


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_geometry')
def handle_geometry(geojson):
    try:
        # Save geojson as crs_leaflet
        output_file = f'{int(datetime.now().timestamp())}--{request.sid}.geojson'
        output_filepath = gpkg_folder + output_file
        with open(output_filepath, 'w') as f:
            if GEOJSON_PRETTY_PRINT:
                f.write(json.dumps(json.loads(geojson), indent=1))
            else:
                f.write(geojson)
        print(f"GeoPackage saved: {output_file}")

        # Load, change crs and save again
        gdf:gpd.GeoDataFrame = gpd.read_file(output_filepath, engine="pyogrio")
        gdf = geodataframe_from_leaflet_to_ign(gdf)
        gdf.to_file(output_filepath, driver='GeoJSON')

        socketio.emit('geojson_received')
    except Exception as e:
        print(f"Error processing geometry: {e}")


@socketio.on("request_available_tiles")
def send_available_tiles():
    """
    Send the GeoJSON data to the client when requested.
    """
    gdf = gpd.read_file('data/geojson/all_tiles/all_tiles.geojson')
    gdf = gdf.set_crs('EPSG:2154', allow_override=True)
    gdf = geodataframe_from_ign_to_leaflet(gdf)
    geojson_data = gdf.to_json()

    print(f'Sending geojson all tiles to {request.sid}')
    socketio.emit("receive_geojson", geojson_data)
    print(f'Finished sending all tiles to {request.sid}')

if __name__ == '__main__':
    socketio.run(app, debug=True)
