import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import unary_union
import json

crs_leaflet = 'EPSG:4326'
crs_ign =     'EPSG:2154'


def geodataframe_from_leaflet_to_ign(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.set_crs(crs_leaflet)
    gdf_transformed  = gdf.to_crs(crs_ign)
    return gdf_transformed

def geodataframe_from_ign_to_leaflet(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.set_crs(crs_ign)
    gdf_transformed  = gdf.to_crs(crs_leaflet)
    return gdf_transformed



if __name__=="__main__":
    tiles_filepath = 'data/data_grille/TA_diff_pkk_lidarhd_classe.shp'

    tiles_df:gpd.GeoDataFrame = gpd.read_file(tiles_filepath, engine="pyogrio")
    tiles_df.set_crs(crs_ign)

    merged_polygone = unary_union(list(tiles_df['geometry']))
    geojson = mapping(merged_polygone)

    with open('data/geojson/all_tiles/all_tiles.geojson', 'w') as f:
        f.write(json.dumps(geojson))
