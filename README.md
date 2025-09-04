# MLP-Mmax-Predictor
MLP Mmax Predictor
Introduction:
Accurate prediction of the maximum magnitude of hydraulic fracturing–induced seismicity is crucial for risk management. We develop a framework combining a dual-parameter linear regression model and a multilayer perceptron (MLP). The regression model offers interpretability and efficiency, while the MLP captures nonlinear interactions among injection parameters and fault properties. Together, they balance predictive accuracy and mechanistic insight across sedimentary, igneous, and mixed lithologies.

'train_mlp.py': MLP model for predicting maximum magnitude of hydraulic fracturing–induced seismicity. The program loads Excel datasets, cleans outliers, standardizes data, tunes hyperparameters, trains the model, and outputs predictions with evaluation metrics and plots. Input Data: Excel-formatted dataset with features in all columns except the last, which represents the target maximum magnitude (Mmax). 'processed_data': Processes raw geoscientific data by adding Gaussian noise to selected columns and applying Gaussian filtering, then outputs the processed dataset (processed_data.xlsx).

Any questions, please contact: BZ23010012@s.upc.edu.cn
