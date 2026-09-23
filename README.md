# Why pyproject.toml? 
To be able to use the packages(such as datafunctions which was not visible to the models folder etc)

# Project Template

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── src                         <- Source code for this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    │    
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations 
    │
    └── services                <- Service classes to connect with external platforms, tools, or APIs
        └── __init__.py 
```


fl_vip/
├── fl_vip/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── partitioner.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   └── cnn.py
│   ├── federated/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── server.py
│   │   └── trainer.py
│   └── viz/
│       ├── __init__.py
│       └── visualizer.py
├── main.py
├── requirements.txt
└── .env (optional, for local overrides)





# Data Project Template

<a target="_blank" href="https://datalumina.com/">
    <img src="https://img.shields.io/badge/Datalumina-Project%20Template-2856f7" alt="Datalumina Project" />
</a>

## Cookiecutter Data Science
This project template is a simplified version of the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org) template, created to suit the needs of Datalumina and made available as a GitHub template.

## Adjusting .gitignore

Ensure you adjust the `.gitignore` file according to your project needs. For example, since this is a template, the `/data/` folder is commented out and data will not be exlucded from source control:

```plaintext
# exclude data from source control by default
# /data/
```

Typically, you want to exclude this folder if it contains either sensitive data that you do not want to add to version control or large files.

## Duplicating the .env File
To set up your environment variables, you need to duplicate the `.env.example` file and rename it to `.env`. You can do this manually or using the following terminal command:

```bash
cp .env.example .env # Linux, macOS, Git Bash, WSL
copy .env.example .env # Windows Command Prompt
```

This command creates a copy of `.env.example` and names it `.env`, allowing you to configure your environment variables specific to your setup.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── src                         <- Source code for this project
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    │    
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    ├── plots.py                <- Code to create visualizations 
    │
    └── services                <- Service classes to connect with external platforms, tools, or APIs
        └── __init__.py 
```

# Instructions for toml
## Project Setup: Making `src` modules importable

This project uses a `src/` layout, where reusable code lives in packages like `src/datafunctions/` and `src/models/`. To import these anywhere in the project (notebooks, scripts, other modules) — e.g.:

```python
from datafunctions.dataset import KneeOsteoDataset
```

you must install the project in **editable mode** once per environment.

### 1. Requirements

- Every package folder under `src/` (e.g. `datafunctions/`, `models/`, `federated/`) must contain an `__init__.py` file, even if empty.
- `src/` itself must also contain an `__init__.py`.
- The project root must contain a `pyproject.toml` with the following, in addition to your existing project metadata:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

### 2. Install the project in editable mode

Run this once per environment (local machine, server, each new virtual environment):

```bash
cd <project_root>
pip install -e .
```

If this project uses `uv` instead of plain pip:

```bash
uv sync
```

### 3. Verify the install worked

```bash
python -c "from datafunctions.dataset import KneeOsteoDataset; print('Import successful')"
```

Run this from **any directory**, not just the project root, to confirm the install (not your working directory) is what makes the import work.

### Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'datafunctions'` | Package not installed in this environment | Run `pip install -e .` (or `uv sync`) in the active environment |
| `ImportError: attempted relative import with no known parent package` | A file inside a package (e.g. `__init__.py`) was run directly as a script | Don't run `__init__.py` or internal package files directly — always run scripts from outside the package, or use `python -m package.module` |
| Editable install fails silently or partially | Missing `__init__.py` in one of the `src/` subfolders | Run `find src -name "__init__.py"` and confirm every package folder has one |
| Works locally but not on server | Editable installs are per-environment | Re-run `pip install -e .` (or `uv sync`) in the server's own virtual environment |

# Instructions on how to run on a server

Commands to set the venv and run main
FIRST TIME
cd /nobackup/proj/disk/asaxlab/personal/tsampika/knee_osteo
deactivate 2>/dev/null
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
pip install torch torchvision

python main.py

When done
deactive 

SECOND TIME
cd /nobackup/proj/disk/asaxlab/personal/tsampika/knee_osteo
source .venv/bin/activate
python main.py

If you're syncing this project back and forth with rsync, make sure .venv/ is in your exclude list virtual environments are environment-specific and shouldn't be copied between machines. Each machine should build its own via pip install -e ..