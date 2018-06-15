Quickstart
==========

Prerequisites
-------------

- `conda <https://conda.io/>`_ (`Anaconda <https://www.anaconda.com/>`_
  / `Miniconda <https://conda.io/miniconda.html>`_), used to manage python dependencies in a virtual environment
- `python <https://www.python.org/>`_ >= 3.5 (installed through conda)
- datascience libraries: matplotlib, numpy, sklearn, etc... (installed through conda)

Install conda
~~~~~~~~~~~~~
`Install Anaconda or Miniconda <https://conda.io/docs/user-guide/install/index.html>`_.
Select the python 3 version. We recommend Anaconda for development and Miniconda for production.

Update to the latest version::

    conda update conda -c conda-forge


Install atnlp (production)
--------------------------
Follow these instructions if you want to deploy atnlp for production (ie without developing).

Download the atnlp conda production environment configuration::

    wget https://raw.githubusercontent.com/wedavey/atnlp/master/envprod.yml

Create `atnlp` environment (including dependencies)::

    conda env create -f envprod.yml -n atnlp

If you prefer, Anaconda also provides a GUI for managing environments.

Activate the environment::

    conda activate atnlp

.. note:: on older versions of conda you may need to use ``source activate atnlp``

Make sure you activate this environment anytime you want to use the atnlp package or manage dependencies.

To deactivate the environment issue ``conda deactivate`` (``source deactivate`` on older conda versions).


Install atnlp (development)
---------------------------
Follow these instructions if you want to develop the atnlp package.

Create a fork of `wedavey/atnlp <https://github.com/wedavey/atnlp>`_ (button at top right).

Clone your fork::

    git clone git@github.com:<your-user-name>/atnlp.git

.. note:: make sure to replace `<your-user-name>` with your github user name!

Create `atnlp-dev` environment (including dependencies)::

    conda env create -f atnlp/envdev.yml -n atnlp-dev

If you prefer, Anaconda also provides a GUI for managing environments.

Activate the environment::

    conda activate atnlp-dev

.. note:: on older versions of conda you may need to use ``source activate atnlp-dev``

Make sure you do this anytime you want to use the atnlp package or manage any dependencies.

To deactivate the environment issue ``conda deactivate`` (``source deactivate`` on older conda versions).

Install atnlp::

    cd atnlp; python setup.py develop

Now start developing! When you're happy with the changes on your fork and want to merge into the main repo, make a pull request.