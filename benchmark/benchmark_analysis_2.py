import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Parse JSON
benchmark_filepath = 'data/benchmark/benchmark_results--174059526704.json'
with open(benchmark_filepath, 'r') as f:
    data = json.load(f)

# Prepare the data for plotting
records = []
for key, values in data.items():
    smoothing = values['smoothing']
    percentage_point_to_remove = values['percentage_point_to_remove']
    point_class_title = values['point_class_title']
    duration_filtering_decimating = values['duration_filtering_decimating']
    total_all_choosen_points = values['total_all_choosen_points']
    duration_surface_to_mesh = values['duration_surface_to_mesh']
    surface_mesh_triangles = values['surface_mesh_triangles']
    duration_add_base_mesh = values['duration_add_base_mesh']
    final_mesh_triangles = values['final_mesh_triangles']
    point_cloud_file_size = values['point_cloud_file_size']
    surface_mesh_file_size = values['surface_mesh_file_size']
    final_mesh_file_size = values['final_mesh_file_size']
    
    records.append({
        'Smoothing': smoothing,
        'Percentage Point to Remove': percentage_point_to_remove,
        'Point Class Title': point_class_title,
        'Duration Filtering Decimating': duration_filtering_decimating,
        'Total All Chosen Points': total_all_choosen_points,
        'Duration Surface to Mesh': duration_surface_to_mesh,
        'Surface Mesh Triangles': surface_mesh_triangles,
        'Duration Add Base Mesh': duration_add_base_mesh,
        'Final Mesh Triangles': final_mesh_triangles,
        'Point Cloud File Size': point_cloud_file_size,
        'Surface Mesh File Size': surface_mesh_file_size,
        'Final Mesh File Size': final_mesh_file_size
    })

# Create a DataFrame
df = pd.DataFrame(records)


for x_axis in ['Smoothing', 'Percentage Point to Remove', 'Point Class Title']:

    # Plotting
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))

    # Plot 1: Duration Filtering Decimating
    ax1 = axes[0, 0]
    sns.barplot(data=df, x=x_axis, y='Duration Filtering Decimating', hue='Point Class Title', ax=ax1)
    ax1.set_title('Duration Filtering Decimating')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Plot 2: Total All Chosen Points
    ax2 = axes[0, 1]
    sns.barplot(data=df, x=x_axis, y='Total All Chosen Points', hue='Point Class Title', ax=ax2)
    ax2.set_title('Total All Chosen Points')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    # Plot 3: Duration Surface to Mesh
    ax3 = axes[0, 2]
    sns.barplot(data=df, x=x_axis, y='Duration Surface to Mesh', hue='Point Class Title', ax=ax3)
    ax3.set_title('Duration Surface to Mesh')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')

    # Plot 4: Surface Mesh Triangles
    ax4 = axes[1, 0]
    sns.barplot(data=df, x=x_axis, y='Surface Mesh Triangles', hue='Point Class Title', ax=ax4)
    ax4.set_title('Surface Mesh Triangles')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')

    # Plot 5: Duration Add Base Mesh
    ax5 = axes[1, 1]
    sns.barplot(data=df, x=x_axis, y='Duration Add Base Mesh', hue='Point Class Title', ax=ax5)
    ax5.set_title('Duration Add Base Mesh')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')

    # Plot 6: Final Mesh Triangles
    ax6 = axes[1, 2]
    sns.barplot(data=df, x=x_axis, y='Final Mesh Triangles', hue='Point Class Title', ax=ax6)
    ax6.set_title('Final Mesh Triangles')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')

    # Plot 7: Point Cloud File Size
    ax7 = axes[2, 0]
    sns.barplot(data=df, x=x_axis, y='Point Cloud File Size', hue='Point Class Title', ax=ax7)
    ax7.set_title('Point Cloud File Size')
    ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45, ha='right')

    # Plot 8: Surface Mesh File Size
    ax8 = axes[2, 1]
    sns.barplot(data=df, x=x_axis, y='Surface Mesh File Size', hue='Point Class Title', ax=ax8)
    ax8.set_title('Surface Mesh File Size')
    ax8.set_xticklabels(ax8.get_xticklabels(), rotation=45, ha='right')

    # Plot 9: Final Mesh File Size
    ax9 = axes[2, 2]
    sns.barplot(data=df, x=x_axis, y='Final Mesh File Size', hue='Point Class Title', ax=ax9)
    ax9.set_title('Final Mesh File Size')
    ax9.set_xticklabels(ax9.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'benchmark/x--{x_axis}')
    plt.show()