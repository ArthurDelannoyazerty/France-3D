```cmd
uv venv --python 3.12
.venv\Scripts\activate
uv pip compile --no-annotate --no-header -o requirements.txt requirements.in
uv pip install -r requirements.txt
```