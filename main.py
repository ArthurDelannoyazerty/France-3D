import geopandas as gpd

aoi_file = "data/gpkg/1737724257--exqaglB_2i2BlcyiAAAB--gpkg_zone.gpkg"

aoi_df = gpd.read_file(aoi_file, engine="pyogrio")
print(aoi_df.iloc[0])