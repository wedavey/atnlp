Quickstart
==========

Dependencies
------------

- conda (Anaconda / Miniconda)
- python >= 3.5 (installed through conda)
- python data science libraries (installed through conda)


Install Anaconda
----------------
`Anaconda <https://www.anaconda.com/>`_ is used to manage python dependencies in a virtual environment.

Install **python 3** version of **Anaconda** (or miniconda): https://www.anaconda.com/download/

Update to the latest version::

    conda update conda -c conda-forge


Clone atnlp
-----------
Clone atnlp from the repo::

    git clone git@bitbucket.org:wedavey/atnlp.git

.. todo:: update this to github


Create environment
------------------
Create a conda environment called ``atnlp`` where we will install the atnlp package and all its dependencies.

From commandline::

    conda env create -f atnlp/environment.yml -n atnlp

If you prefer, Anaconda also provides a GUI for managing environments.

Activate the environment via::

    conda activate atnlp

.. note:: on older versions of conda you may need to use ``source activate atnlp``

Make sure you do this anytime you want to use the atnlp package or manage any dependencies.

To deactivate the environment issue ``conda deactivate`` (``source deactivate`` on older conda versions).


Install atnlp and dependencies
------------------------------
Install (development mode)::

    pip install -e ./atnlp

Install (production mode)::

    pip install ./atnlp

