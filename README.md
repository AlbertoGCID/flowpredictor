# LSTM Rainfall Prediction

```
└── 📁src
    └── AI_algorithms.py
    └── AI_results.py
    └── __init__.py
    └── data_load.py
    └── main.py
    └── split_delay_time.py
    └── test.py
    └── 📁models
        └── inputLSTM.keras
        └── outputLSTM.keras
    └── 📁scaler
        └── inflow_train_scaler.pkl
        └── outflow_train_scaler.pkl
        └── x_train_scaler.pkl

```

## Overview

This project provides a RainfallDataset class and utility functions for time series prediction using Long Short-Term Memory (LSTM) networks. The main function `main()` serves as a script for training and prediction. The script is designed to be run from the command line, accepting various arguments for configuration.

## Prerequisites

Make sure you have the following installed:

- Python 3.10.9
- Required libraries: pandas, numpy, scikit-learn, tensorflow (or keras), prettytable, hydroeval

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage
### Main Function - main()

The main() function is the entry point for the training and prediction script. It accepts command-line arguments to configure the training and prediction process.

```python
python main.py --train_folder datasets/train_folder/ --predict_folder datasets/predict_folder/ --input_width 7 --offset 3 --inflow_name input --outflow_name output --target input --model_path models/inflowLSTM.keras --x_scaler_path scaler/x_train_scaler.pkl --inflow_scaler_path scaler/inflow_train_scaler.pkl --outflow_scaler_path scaler/outflow_train_scaler.pkl
```

### Command-Line Arguments:

- --train_folder: Path to the training dataset.
- --predict_folder: Path to the prediction dataset.
- --input_width: Width of the input window.
- --offset: Offset between input and output.
- --inflow_name: Name of the input column for inflow.
- --outflow_name: Name of the output column for outflow.
- --target: Name of the target column.
- --predict_only: Run prediction only (flag).
- --model_path: Path to the trained model (optional for prediction).
- --x_scaler_path: Path to the x_scaler (optional for prediction).
- --inflow_scaler_path: Path to the inflow_scaler (optional for prediction).
- --outflow_scaler_path: Path to the outflow_scaler (optional for prediction).

## Example Usage

### Training

```python
python main.py --train_folder datasets/train_folder/ --input_width 7 --offset 3 --inflow_name input --outflow_name output --target input
```

### Prediction only

```python
python main.py --predict_folder datasets/predict_folder/ --input_width 7 --offset 3 --inflow_name input --outflow_name output --target input --predict_only --model_path models/inflowLSTM.keras --x_scaler_path scaler/x_train_scaler.pkl --inflow_scaler_path scaler/inflow_train_scaler.pkl --outflow_scaler_path scaler/outflow_train_scaler.pkl
```

## License
This project is licensed under the GNU General Public License, version 3 - see the LICENSE file for details.