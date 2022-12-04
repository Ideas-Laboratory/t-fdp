# Setup

Install the environment by anaconda:

```
conda create -n tfdp python=3.8 cupy numba numpy scikit-learn torch
conda activate tfdp
```

Setup for BH method.

```
cd bh_tforce
python setup.py build
python setup.py install
```

Run the example:

```
python example.py
```
