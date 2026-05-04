# TSNN_release
 
Official implementation of *TSNN: A Non-parametric and Interpretable Framework for Traffic Time Series Forecasting*.

## Install Dependencies

>pip install numpy scipy

>pip install torch --index-url https://download.pytorch.org/whl/cu124

## Download Datasets

You can download the dataset from BasicTS v0.5.8 ([link](https://github.com/GestaltCogTeam/BasicTS/blob/v0.5.8/tutorial/getting_started.md)). *Note that the format of datasets has been changed since BasicTS v1.0.* 

Because of poor coding habits, `build_dataset.py` currently saves the constructed data to disk in \[*total length $\times$ steps*\]. Please ensure you have enough space when processing large datasets.

## Acknowledgement

The `basicts/` is copied from [BasicTS](https://github.com/GestaltCogTeam/BasicTS) for the metrics and dataloaders. We thank their contribution to the time series community. 