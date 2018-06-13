atnlp
==============================

Natural language processing investigations

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

Quick start
-----------

- **Intall conda**: [click here](https://conda.io/docs/user-guide/install/index.html) (select python 3)

- **Setup conda (shell only)**, ensue you setup conda each time you enter a clean shell. 
Setup alias will be something like `setupAnaconda`, which is added to your `.bashrc` (`.bash_profile` on mac) on install.  

<!---
- **Configure conda-forge**: 
```bash
conda config --add channels conda-forge anaconda
```
--->

- **Create project environment and install dependencies**, from project dir call:
```bash
conda env create -f linux_env.yml
```
<!--- conda create --name atnlp --file requirements.txt --->


- **Activate environment (shell only)**, each time you enter clean shell call:
```bash
conda activate atnlp
```

<!---
- **Update packages**: 
```bash
conda update --all
```

```bash
pip install google-compute-engine
```
--->

- **Install atnlp**: 
```bash
python setup.py develop
```

- **Start analysing**, open one of the notebooks and give it a go:
```bash
cd notebooks
jupyter notebook
```


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
