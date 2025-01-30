```cmd
uv venv --python 3.12
.venv\Scripts\activate
uv pip compile --no-annotate --no-header -o requirements.txt requirements.in
uv pip install -r requirements.txt
```


For the selection, the transformation point cloud to mesh will create a mesh that if full / equivalent to convex hull, (So a round shape is ok, but an 'empty' portion of the selection will be filled)