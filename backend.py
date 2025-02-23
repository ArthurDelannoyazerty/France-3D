import logging
import json
import os
import geopandas as gpd
import laspy
import numpy as np
import osmnx as ox

from pathlib import Path
from shapely.geometry import shape
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from datetime import datetime

from utils.logger import setup_logging
from tiles import (
    geodataframe_from_leaflet_to_ign, 
    geodataframe_from_ign_to_leaflet,
    init_folders,
    download_ign_available_tiles,
    merge_all_geojson_features,
    get_intersecting_tiles_from_order,
    download_tiles_from_gdf,
    decimate_array,
    filter_points_by_polygon,
    numpy_to_ply,
    meshlib_terrain_point_cloud_to_surface_mesh,
    add_base_to_surface_mesh
)


app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('send_geometry')
def handle_geometry(geojson_str):
    try:
        # Transform the string of geojson into a GeoDataFrame ovject
        order_dict = json.loads(geojson_str)
        geometry = order_dict['geojson']["features"][0]["geometry"]
        geojson_data = [{"geometry": shape(geometry)}]
        gdf = gpd.GeoDataFrame.from_dict(geojson_data)

        # Transform the coordinate system
        gdf = geodataframe_from_leaflet_to_ign(gdf)
        gdf_json = json.loads(gdf.to_json())

        # Reasssemble the order with ign CRS
        order_dict['geojson'] = gdf_json
        
        # Create paths
        orders_folder   = Path('data/orders/')
        order_id        = Path(str(int(datetime.now().timestamp())) + '--' + str(request.sid))
        output_filename = Path('order.json')
        order_folder   = orders_folder / order_id 
        order_filepath = order_folder / output_filename

        # Create folder for the current order (after the transformation operation because not created if an error occurs)
        Path(order_folder).mkdir(parents=True, exist_ok=True)
        with open(order_filepath, 'w') as f:
            f.write(json.dumps(order_dict))

        socketio.emit('alert', f'The order {order_id} have been received.')
        process_order(order_folder)
    except Exception as e:
        socketio.emit('user_info_update_unsafe', f'<p style="color:red;">An error happenned during the order process.</p>')
        logger.exception(f'An error happened during the process of the order {order_id}')


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


@socketio.on('send_city_search')
def location_string_to_geojson(location_string:str):
    try:
        geojson = ox.geocode_to_gdf(location_string).to_json()
        socketio.emit('add_geojson', geojson)
    except:
        socketio.emit('user_info_update_unsafe', f'<p style="color:red;">No result for the search : {location_string}</p>')


# PROCESS ------------------------------------------------------------------------------
def process_order(order_folderpath:Path):
    socketio.emit('user_info_update', f'Initialize the order.')
    FORCE_LAZ_TO_PLY = True
    Z_OFFSET = 10

    order_filepath = order_folderpath / 'order.json'
    with open(order_filepath, 'r') as f:
        dict_order = json.load(f)

    choosen_point_class        = dict_order['point_class']
    percentage_point_to_remove = dict_order['points_to_remove']
    smoothing_iteration        = dict_order['smoothing']

    laz_folderpath = Path('data/raw_point_cloud/')
    order_zone_filepath   = order_folderpath / 'order.json'
    order_intersects      = order_folderpath / 'tiles_intersect.geojson'
    ply_filepath          = order_folderpath / 'point_cloud.ply'
    surface_mesh_filepath = order_folderpath / 'surface_mesh.stl'
    final_mesh_filepath   = order_folderpath / 'final_mesh.stl'

    # Do the intersection if not already done
    if not os.path.isfile(order_intersects):
        socketio.emit('user_info_update', f'Calculate which product to download from IGN.')
        gdf_intersect = get_intersecting_tiles_from_order(order_zone_filepath , filepath_all_tiles_geojson)
        gdf_intersect.to_file(order_intersects)
    
    # Download the non downloaded tiles
    socketio.emit('user_info_update', f'Download products from IGN (May take time. Check console for downloads status).')
    gdf_intersect = gpd.read_file(order_intersects)
    download_tiles_from_gdf(gdf_intersect, laz_folderpath)
    
    # point cloud .laz to .ply + point decimation + point filtering by user zone selection
    if not os.path.isfile(ply_filepath) or FORCE_LAZ_TO_PLY:
        socketio.emit('user_info_update', f'Create the custom cloud point from IGN products (May take time. Check console for status).')
        polygon = gpd.read_file(order_zone_filepath).iloc[0].geometry
        list_intersecting_tiles_filename = list(gdf_intersect['name'])
        merged_xyz = list()
        for tile_filename in list_intersecting_tiles_filename:
            tile_filepath = laz_folderpath / tile_filename

            # Laz -> numpy   +   Remove unwanted points
            las = laspy.read(tile_filepath)
            choosen_xyz = np.empty((0,3))
            for point_class in choosen_point_class:
                las_points = las.points[las.classification == point_class]
                x = las_points.x.array
                y = las_points.y.array
                z = las_points.z.array
                xyz = np.vstack([x,y,z]).T * las_points.scales   # Merged the coordinates into a (n,3) array and rescale them
                choosen_xyz = np.vstack([choosen_xyz, xyz])

            # Decimation of class-filtered points of the current tile
            xyz = decimate_array(choosen_xyz, percentage_point_to_remove)
            filtered_array = filter_points_by_polygon(xyz, polygon)
            merged_xyz.append(filtered_array)                           # Merge the processed tile with the other tiles
        # Save all the needed points
        merged_xyz = np.vstack(merged_xyz)
        merged_xyz = merged_xyz - merged_xyz[0]     # Center the point cloud on (0,0,0)
        numpy_to_ply(merged_xyz, ply_filepath)

    # Point cloud to mesh
    socketio.emit('user_info_update', f'Create the mesh from the cloud point (May take time. Check console for status).')
    meshlib_terrain_point_cloud_to_surface_mesh(ply_filepath, surface_mesh_filepath, smoothing=smoothing_iteration)
    add_base_to_surface_mesh(surface_mesh_filepath, final_mesh_filepath, Z_OFFSET)

    full_final_mesh_path = Path(os.path.dirname(os.path.abspath(__file__))) / final_mesh_filepath
    socketio.emit('user_info_update_unsafe', f'Your file is ready at : <a href="file:///{full_final_mesh_path}" target="_blank">{full_final_mesh_path}</a>.')


if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger(__name__)

    # Init folder tree if not existing
    init_folders()

    ox.utils.settings.cache_folder = Path('data/cache')

    # Download and merge all tiles availables
    FORCE_DOWNLOAD_ALL_TILES_AVAILABLE = False
    filepath_all_tiles_geojson         = Path('data/data_grille/all_tiles_available.geojson')
    filepath_all_tiles_geojson_merged  = Path('data/data_grille/all_tiles_available_merged.geojson')
    download_ign_available_tiles(filepath_all_tiles_geojson, FORCE_DOWNLOAD_ALL_TILES_AVAILABLE)
    merge_all_geojson_features(filepath_all_tiles_geojson, filepath_all_tiles_geojson_merged, FORCE_DOWNLOAD_ALL_TILES_AVAILABLE)

    socketio.run(app, debug=True)
