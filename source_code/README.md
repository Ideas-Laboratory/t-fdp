Here implements a python simulation solver, which followed d3-force simulation framework, for simulating forces in [t-FDP](#) model. 
# Setup

The code is tested under ubuntu 20.04.
#### Requires: Anaconda3, python3.8, gcc

#### cmd for conda install:
```
conda install -c conda-forge cupy cudatoolkit=11.2
pip install scikit-learn pyfftw numba_kdtree torch torchvision pandas dask[dataframe]
pip install numpy==1.20.3 numba==0.54.1
```


#### Setup for BH method.
```
cd bh_tforce
python setup.py build
python setup.py install
```

#### Run the example:

```
mkdir layouts
python example.py
```
