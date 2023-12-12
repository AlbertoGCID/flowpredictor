import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,BatchNormalization,Dropout
from typing import Dict, List




def lstm_create(param_config: dict) -> Sequential:
    """
    Creates an LSTM model.

    Parameters:
        param_config (dict): Configuration parameters for the LSTM model.
            num_layers (int): Number of LSTM layers in the model.
            units (int): Number of units/neurons in each LSTM layer.
            batchnorm (int): Whether to use Batch Normalization (1 for True, 0 for False).
            dropout (int): Whether to use Dropout (1 for True, 0 for False).
            dropout_rate (float): Dropout rate if dropout is enabled.
            seed (Optional[int]): Random seed for reproducibility.
            label_width (int): Number of output units in the final Dense layer.
            act (Optional[str]): Activation function for the final Dense layer ('None', 'sigmoid', or 'linear').

    Returns:
        tf.keras.models.Sequential: LSTM model.
    
    Example of use:
    param_config = {
    'num_layers': 2,
    'units': 32,
    'batchnorm': 1,
    'dropout': 1,
    'dropout_rate': 0.5,
    'seed': 42,
    'label_width': 1,
    'act': 'sigmoid'
    }
    model = lstm_create(param_config)
    """
    lstm_model = Sequential()

    if param_config['num_layers'] > 1:
        for i in range(param_config['num_layers']):
            lstm_model.add(LSTM(param_config['units'], return_sequences=True))
            
            if param_config['dropout'] == 1:
                lstm_model.add(Dropout(param_config['dropout_rate'], seed=param_config['seed']))
            if param_config['batchnorm'] == 1:
                lstm_model.add(BatchNormalization(axis=-1, center=True, scale=True))

        lstm_model.add(LSTM(units=param_config['units'], return_sequences=False))

        if param_config['dropout'] == 1:
            lstm_model.add(Dropout(param_config['dropout_rate'], seed=param_config['seed']))
        if param_config['batchnorm'] == 1:
            lstm_model.add(BatchNormalization(axis=-1, center=True, scale=True))
    else:
        lstm_model.add(LSTM(units=param_config['units'], return_sequences=False))

        if param_config['dropout'] == 1:
            lstm_model.add(Dropout(param_config['dropout_rate'], seed=param_config['seed']))
        if param_config['batchnorm'] == 1:
            lstm_model.add(BatchNormalization(axis=-1, center=True, scale=True))

    if param_config['act'] == 'sigmoid':
        lstm_model.add(Dense(param_config['label_width'], activation=tf.keras.activations.sigmoid))
    else:
        lstm_model.add(Dense(param_config['label_width'], activation=tf.keras.activations.linear))

    return lstm_model


def save_model(model:Sequential, folder:str='models', model_name:str='LSTM_model') -> None:
    """
    Save a Keras Sequential model to a specified folder with a given name.

    Args:
        model (Sequential): The Keras Sequential model to be saved.
        folder (str, optional): Path to the directory where the model will be saved. Default is 'models'.
        model_name (str, optional): Name of the saved model file (without extension). Default is 'LSTM_model'.

    Returns:
        None
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    model_path = os.path.join(folder, f"{model_name}.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")


def lstm_train(x_train_lstm: np.ndarray, y_train_lstm: np.ndarray, x_val_lstm:np.ndarray, y_val_lstm: np.ndarray, 
               param_config: Dict,pred_config: Dict) -> Sequential:
    """
    Trains an LSTM model.

    Parameters:
        x_train_lstm (np.ndarray): Training data.
        y_train_lstm (np.ndarray): Training labels.
        x_val_lstm (np.ndarray): Validation data.
        y_val_lstm (np.ndarray): Validation labels.
        param_config (Dict): Configuration parameters for the LSTM model.

    Returns:
        tf.keras.models.Sequential: Trained LSTM model.
    """

    lstm_model = lstm_create(param_config=param_config)
    lstm_model.compile(loss='mae', optimizer=tf.optimizers.Adam(learning_rate=0.01), metrics=['mae'])
    lstm_model.fit(x_train_lstm,  y_train_lstm, epochs=param_config['epochs'], batch_size=param_config['batch_size'], validation_data=(x_val_lstm, y_val_lstm))
    save_model(lstm_model,model_name=pred_config['output']+"LSTM")
    return lstm_model


if __name__ == "__main__":
    print("All libraries are loaded correctly")
