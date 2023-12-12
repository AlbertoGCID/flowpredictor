import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from split_delay_time import sliding_window
from typing import Dict,Union,Any
import hydroeval


def lstm_evaluate(lstm_model:Sequential,x_test_norm:np.ndarray,x_test:Union[pd.DataFrame,np.ndarray],
                  y_test:Union[pd.DataFrame,np.ndarray],scaler:MinMaxScaler,pred_config:Dict)->None:
    """
    Evaluate the performance of an LSTM model on a given test dataset.

    Parameters:
    - lstm_model (Sequential): Trained LSTM model.
    - x_test_norm (np.ndarray): Normalized input data for the test dataset.
    - x_test (Union[pd.DataFrame, np.ndarray]): Input data for the test dataset.
    - y_test (Union[pd.DataFrame, np.ndarray]): Target variable values for the test dataset.
    - scaler (MinMaxScaler): Scaler used for normalization.
    - pred_config (Dict): Prediction configuration parameters.

    Returns:
    None
    """
    y_pred = lstm_model.predict(x_test_norm)
    y_pred_denorm = scaler.inverse_transform(y_pred)
    temporal,y_test = sliding_window(data_x=x_test,data_y=y_test,input_width=pred_config['input_width'],
                                     label_width=pred_config['label_width'],offset=pred_config['offset'])
    mse_LSTM = mean_squared_error(y_test, y_pred_denorm)
    correlacion = np.corrcoef(y_pred_denorm,y_test,rowvar=False)[0][1]
    r_cuadrado_LSTM = correlacion ** 2
    ns_LSTM = hydroeval.nse(y_test,y_pred_denorm)[0]
    diff_modelo_test = np.abs(y_pred_denorm-y_test)     
    std_LSTM = np.std(diff_modelo_test)
    print('\n\nResults of evaluate LSTM in eval dataset\n')
    print(f'NS test: {ns_LSTM:.3f}')
    print(f'MSE test: {mse_LSTM}')
    print(f'R2 test: {r_cuadrado_LSTM}')
    print(f'std test: {std_LSTM}\n')


if __name__ == "__main__":
  print("Todas las librerías son cargadas correctamente")