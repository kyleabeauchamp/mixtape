sudo apt-get install -qq -y g++ gfortran valgrind csh
sudo apt-get install -qq -y g++-multilib gcc-multilib
wget http://repo.continuum.io/miniconda/Miniconda-3.5.5-Linux-x86_64.sh
bash Miniconda-3.5.5-Linux-x86_64.sh -b
PIP_ARGS="-U"

export PATH=$HOME/miniconda/bin:$PATH

conda update --yes conda
conda create --yes -n ${python} --file devtools/ci/requirements-conda-${python}.txt
conda config --add channels http://conda.binstar.org/omnia
source activate $python
conda install --yes mdtraj scikit-learn sphinx openmm ipython-notebook matplotlib
conda list -e
$HOME/miniconda/envs/${python}/bin/pip install $PIP_ARGS -r devtools/ci/requirements-${python}.txt

# Install msmbuilder
wget https://github.com/SimTk/msmbuilder/archive/master.zip
unzip master.zip
cd msmbuilder-master/
python setup.py install
cd ..
