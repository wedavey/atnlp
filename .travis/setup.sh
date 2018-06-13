#!/bin/bash

#setup travis-ci configuration basing one the being-built branch

if [[ $TRAVIS_BRANCH == 'master' ]] ; then
    export DEPLOY_HTML_DIR=docs
elif [[ $TRAVIS_BRANCH == 'develop' ]] ; then
    export DEPLOY_HTML_DIR=docs/develop
elif [[ $TRAVIS_BRANCH =~ ^v[0-9.]+$ ]]; then
    export DEPLOY_HTML_DIR=docs/${TRAVIS_BRANCH:1}
else
    export DEPLOY_HTML_DIR=docs/$TRAVIS_BRANCH
fi

# build conda
sudo apt-get update
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda env create -q -f environment.yml -n test-environment
source activate test-environment
python setup.py install