import json
import pandas as pd
import matplotlib.pyplot as plt


# Parse JSON
benchmark_filepath = 'data/benchmark/benchmark_results--174059526704.json'
with open(benchmark_filepath, 'r') as f:
    data = json.load(f)

# Convert to DataFrame
rows = []
for key, values in data.items():
    rows.append(values)
    
df = pd.DataFrame(rows)

print(df.head())
print(df.dtypes)


df[['smoothing', 'duration_surface_to_mesh']].groupby('smoothing').mean().plot(kind='line', title='Duration meshing over Smoothing')
plt.savefig('benchmark/Duration-over-smoothing.png')
plt.show()

a = df[['percentage_point_to_remove', 'final_mesh_file_size']].groupby('percentage_point_to_remove').mean()
a['percentage_point_to_remove'] = a.index
a.plot(kind='line', y="final_mesh_file_size", x='percentage_point_to_remove', title='Percentage to remove over Final Mesh size')
plt.savefig('benchmark/Percentage-to-remove-over-final-mesh-size.png')
plt.show()

a = df[['smoothing', 'final_mesh_file_size']].groupby('smoothing').mean()
a['smoothing'] = a.index
a.plot(kind='line', y="final_mesh_file_size", x='smoothing', title='Smoothing over Final Mesh size')
plt.savefig('benchmark/Smoothing-over-final-mesh-size.png')
plt.show()

a = df[['point_class_title', 'final_mesh_file_size']].groupby('point_class_title').mean()
a['point_class_title'] = ['Ground', 'Ground + Buildings', 'Ground + Vegetation + Buildings', 'All classes']
print(a.head())
a.plot(kind='line', y="final_mesh_file_size", x='point_class_title', title='Points classes over Final Mesh size')
plt.xticks(rotation=7)
plt.savefig('benchmark/Point-class-over-final-mesh-size.png')
plt.show()

a = df[['final_mesh_triangles', 'final_mesh_file_size']].groupby('final_mesh_triangles').mean()
a['final_mesh_triangles'] = a.index
a.plot(kind='line', y="final_mesh_file_size", x='final_mesh_triangles', title='Number of triangles over Final Mesh size')
plt.savefig('benchmark/Number-of-triangles-over-final-mesh-size.png')
plt.show()


a = df[['point_class_title', 'total_all_choosen_points']].groupby('point_class_title').mean()
a['point_class_title'] = a.index
a.plot(kind='bar', y="total_all_choosen_points", x='point_class_title', title='Number of points over point class')
plt.xticks(rotation=7)
plt.savefig('benchmark/Number-of-points-over-point-class.png')
plt.show()