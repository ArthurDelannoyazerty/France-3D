import json
import geopandas as gpd
from pathlib import Path

from shapely.geometry import shape
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from datetime import datetime
from tiles import geodataframe_from_leaflet_to_ign, geodataframe_from_ign_to_leaflet

app = Flask(__name__)
socketio = SocketIO(app)

orders_folder = 'data/orders/'


@app.route('/')
def index():
    return render_template('index2.html')

@socketio.on('send_geometry')
def handle_geometry(geojson_str):
    try:
        # Transform the string of geojson into a GeoDataFrame ovject
        geojson_dict = json.loads(geojson_str)
        geometry = geojson_dict['zone']["features"][0]["geometry"]
        geojson_data = [{"geometry": shape(geometry)}]
        gdf = gpd.GeoDataFrame.from_dict(geojson_data)

        # Transform the coordinate system
        gdf = geodataframe_from_leaflet_to_ign(gdf)
        
        # Create paths
        order_folder = orders_folder + str(int(datetime.now().timestamp())) + '--' + str(request.sid) + '/'
        output_filename = 'zone.geojson'
        output_filepath = order_folder + output_filename
        
        # Create folder for the current order (after the transformation operation because not created if an error occurs)
        Path(order_folder).mkdir(parents=True, exist_ok=True)

        # Save the geodataframe to file
        gdf.to_file(output_filepath, driver='GeoJSON')
        socketio.emit('geojson_received')
    except Exception as e:
        print(f"Error processing geometry: {e}")


@socketio.on("request_available_tiles")
def send_available_tiles():
    """
    Send the GeoJSON data to the client when requested.
    """
    print('Loading geojson file')
    gdf = gpd.read_file('data/data_grille/all_tiles_available_merged.geojson')
    
    print('Transformation crs')
    gdf = gdf.set_crs('EPSG:2154', allow_override=True)
    gdf = geodataframe_from_ign_to_leaflet(gdf)

    print('Transformation Geojson to json')
    geojson_data = gdf.to_json()

    print(f'Sending geojson all tiles to {request.sid}')
    socketio.emit("receive_geojson", geojson_data)
    print(f'Finished sending all tiles to {request.sid}')

if __name__ == '__main__':
    socketio.run(app, debug=True)
