# Setup

The code is tested under ubuntu 20.04.
#### Requires: Anaconda3, python3.8, gcc

#### cmd for conda install:
```
conda install -c conda-forge cupy cudatoolkit=10.2  ## cupy version=8.6.0
pip install scikit-learn pyfftw numba_kdtree pytorch torchvision pandas dask[dataframe]
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
python example.py
```