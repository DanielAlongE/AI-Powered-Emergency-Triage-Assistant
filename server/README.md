
## Install Poetry
This will make poetry available globally
```
# macOS / Linux / WSL
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

## Install Project
```
cd  server
poetry install
```


## Run Server Project

```
cd  server
poetry run uvicorn app.main:app --reload
```

## Add New Dependency

```
poetry add {new-dependency}
```

## Remove Dependency

```
poetry remove {new-dependency}
```

## Run a python script

```
poetry run python my_script.py

# or 

python -m my_package.module1
```