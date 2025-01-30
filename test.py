import laspy
import numpy as np
from stl import mesh
from PIL import Image
import numpy as np

def create_dsm(input_laz, output_tiff, resolution=1.0):
    """
    Create a DSM from a LAZ file
    input_laz: path to input LAZ file
    output_tiff: path to output GeoTIFF file
    resolution: grid cell size in meters (default 1.0m)
    """
    # Read the LAZ file
    print('rea laz file')
    las = laspy.read(input_laz)
    
    # Extract X, Y, Z coordinates
    x = las.x
    y = las.y
    z = las.z

    # Define grid resolution
    grid_resolution = .3  # Adjust as needed

    # Create grid coordinates
    xi = np.arange(x.min(), x.max(), grid_resolution)
    yi = np.arange(y.min(), y.max(), grid_resolution)
    xi, yi = np.meshgrid(xi, yi)

    from scipy.interpolate import griddata

    # Interpolate using nearest method
    zi = griddata((x, y), z, (xi, yi), method='nearest')

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.imshow(zi, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis')
    plt.colorbar(label='Elevation (m)')
    plt.title('Digital Surface Model')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    import rasterio
    from rasterio.transform import from_origin

    # Define transform
    transform = from_origin(x.min(), y.max(), grid_resolution, grid_resolution)

    # Save as GeoTIFF
    with rasterio.open(
        output_tiff,
        'w',
        driver='GTiff',
        height=zi.shape[0],
        width=zi.shape[1],
        count=1,
        dtype=zi.dtype,
        crs='EPSG:2154',
        transform=transform,
    ) as dst:
        dst.write(zi, 1)


def tiff_to_stl():
    # Open the TIFF file
    with Image.open('data/orders/1737817398--5XgZhio-_bDeS4AqAAAB/raster.tif') as img:
        # Convert the image to grayscale (if not already)
        img = img.convert('L')
        # Convert the image to a NumPy array
        elevation_data = np.array(img)

    # Get the dimensions of the elevation data
    nrows, ncols = elevation_data.shape

    # Create a grid of x, y coordinates
    x = np.arange(ncols)
    y = np.arange(nrows)
    x, y = np.meshgrid(x, y)

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    z = elevation_data.flatten()

    # Scale the z values if necessary
    z_scale = 1.0  # Adjust this value to scale the elevation
    z = z * z_scale

    # Create vertices
    vertices = np.vstack((x, y, z)).T

    # Create faces
    faces = []
    for i in range(nrows - 1):
        for j in range(ncols - 1):
            # Define the indices of the vertices of the square
            idx = i * ncols + j
            faces.append([idx, idx + 1, idx + ncols])
            faces.append([idx + 1, idx + ncols + 1, idx + ncols])
    faces = np.array(faces)

    def write_stl(filename, vertices, faces):
        with open(filename, 'w') as file:
            file.write('solid elevation_model\n')
            for face in faces:
                v1, v2, v3 = vertices[face]
                # Compute the normal vector (not normalized)
                normal = np.cross(v2 - v1, v3 - v1)
                file.write(f'  facet normal {normal[0]} {normal[1]} {normal[2]}\n')
                file.write('    outer loop\n')
                for vertex in [v1, v2, v3]:
                    file.write(f'      vertex {vertex[0]} {vertex[1]} {vertex[2]}\n')
                file.write('    endloop\n')
                file.write('  endfacet\n')
            file.write('endsolid elevation_model\n')

    # Write to STL
    write_stl('output_model.stl', vertices, faces)


# Usage example
if __name__ == "__main__":
    # Create basic DSM
    # grid = create_dsm('data/orders/1737817398--5XgZhio-_bDeS4AqAAAB/filtered_point_cloud.laz', 'data/orders/1737817398--5XgZhio-_bDeS4AqAAAB/raster.tif', resolution=0.3)
    
    tiff_to_stl()