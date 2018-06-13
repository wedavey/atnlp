Topic labelling
===============


Workflow
--------

- convert input corpus into raw text format
- build/train topic labelling models
- evaluate and compare model performance


Reuters dataset
---------------
As a simple example we'll use the Reuters corpus (available through the nltk library).
The corpus consists of ~10k news articles (70/30 train/test split) with ~90 different topic labels.
Each document can be attributed multiple topic labels and the topics have a highly non-uniform
distribution.

Go to a new work area and issue::

    reuters_to_txt.py

This will create the following files:

    - data_train.txt
    - data_test.txt
    - labels_train.txt
    - labels_test.txt

The data files contain the raw text from the news articles (one document per line).
The labels files contain the topic labels in a one-hot format (an MxN boolean matrix where M is the number of
documents, N is the number of topics, indicating which labels are attributed to each document).

This is the standard format used by the library. If you convert other corpora into this format,
then you will easily be able to apply the atnlp topic modelling tools to them.


Model building
--------------

atnlp comes with a number of preconfigured topic labelling models.
The models take in raw text and produce labels in a one-hot format.
The text parsing and modelling is typically split into a few steps
(called a pipeline). Blueprints for some basic pipelines are
provided at :ref:`models`. User-defined pipelines can be utilised
by placing the pipeline definition (python file) in your local
working directory.

Each of the algorithms in the pipelines typically have a number
of hyperparameters, which the user will typically want to tune.
Pipelines are configured using yaml. Some basic configurations
for the atnlp pipelines can be found at :ref:`configs`. The config
first defines the blueprint for the pipeline. Pipelines
already in the ``share/models`` path can be referenced by
filename without ``.py`` extension, while for user-defined
models an absolute/relative path including ``.py`` extension must
be given.

User-defined configs can also be made, and should be placed
in the working directory.


Train model
-----------
Let's train the default svm model on the training dataset::

    train.py data_train.txt labels_train.txt -m svm

The model is saved to ``svm.pkl``, which can then be used to provide predictions.

To train a user-defined model, provide full path to your model config (including ``.yml`` extension).


Evaluate model
--------------
A quick evaluation of the topic modelling performance can be obtained via::

    evaluate.py data_test.txt labels_test.txt svm.pkl

Note: multiple models can be passed for comparisons

This will generate and html report.

For more detailed investigations we suggest using a jupyter notebook. You may want to take advantage of
the functionality provided in :mod:`atnlp.eval`.


Model predictions
-----------------
Model predictions in txt format can be obtained using::

    predict.py data_test.txt svm.pkl

