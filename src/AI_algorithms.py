import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback,EarlyStopping,ModelCheckpoint
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


def save_model(model, folder='models', model_name='LSTM_model'):
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





































































def entrenar_red(modelo: tf.keras.models.Model,
                 x_train: tf.Tensor,
                 y_train: tf.Tensor,
                 x_val: tf.Tensor,
                 y_val: tf.Tensor,
                 x_test: tf.Tensor,
                 y_test: tf.Tensor) -> tuple[float, float]:
    """
    Trains, validates, and evaluates a neural network model.

    Parameters:
        modelo (tf.keras.models.Model): Neural network model to be trained.
        x_train (tf.Tensor): Training input data.
        y_train (tf.Tensor): Training target data.
        x_val (tf.Tensor): Validation input data.
        y_val (tf.Tensor): Validation target data.
        x_test (tf.Tensor): Test input data.
        y_test (tf.Tensor): Test target data.

    Returns:
        tuple[float, float]: Mean squared error on the last epoch of training, evaluation metric on the test set.
    """
    modelo_entrenado = modelo.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=80, epochs=1000)
    evaluacion = modelo.evaluate(x_test, y_test, batch_size=80)

    return modelo_entrenado.history['mean_squared_error'][-1], evaluacion[1]

def train_model(x_train: tf.Tensor, 
                y_train: tf.Tensor, 
                x_val: tf.Tensor, 
                y_val: tf.Tensor, 
                x_test: tf.Tensor, 
                y_test: tf.Tensor,
                num_cap: int,
                num_neu: int,
                func_act: str,
                regul: float,
                optim: float = 0.0001,
                fn_perdida: tf.keras.losses.Loss = tf.keras.losses.MeanSquaredError(),
                metrica: tf.keras.metrics.Metric = tf.keras.metrics.MeanSquaredError(),
                epochs: int = 500,
                batch: int = 25,
                clipvalue: float = None,
                clipnorm: float = None) -> tuple[tf.keras.models.Model, float, float]:
    """
    Trains a neural network model and evaluates it on validation and test sets.

    Parameters:
        x_train (tf.Tensor): Training input data.
        y_train (tf.Tensor): Training target data.
        x_val (tf.Tensor): Validation input data.
        y_val (tf.Tensor): Validation target data.
        x_test (tf.Tensor): Test input data.
        y_test (tf.Tensor): Test target data.
        num_cap (int): Number of hidden layers in the neural network.
        num_neu (int): Number of neurons in each hidden layer.
        func_act (str): Activation function for hidden layers.
        regul (float): L2 regularization strength.
        optim (float): Learning rate for the optimizer. Default is 0.0001.
        fn_perdida (tf.keras.losses.Loss): Loss function for model training. Default is MeanSquaredError.
        metrica (tf.keras.metrics.Metric): Evaluation metric for model performance. Default is MeanSquaredError.
        epochs (int): Number of training epochs. Default is 500.
        batch (int): Batch size for training. Default is 25.
        clipvalue (float): Clip value for gradient clipping. Default is None.
        clipnorm (float): Clip norm for gradient clipping. Default is None.

    Returns:
        tuple[tf.keras.models.Model, float, float]: Trained model, mean squared error on the last epoch of training, and evaluation metric on the test set.
    """
    red_base = crear_red(x_train, num_cap, num_neu, func_act, regul, optim, fn_perdida=fn_perdida, metrica=metrica, clip_value=clipvalue, clip_norm=clipnorm)
    es = EarlyStopping(monitor='val_loss', patience=5)
    modelo_entrenado = red_base.fit(x_train, y_train,
                                    validation_data=(x_val, y_val),
                                    batch_size=batch,
                                    epochs=epochs,
                                    callbacks=[es], verbose=False)
    evaluacion = red_base.evaluate(x_test, y_test, batch_size=batch)
    mse_entrenamiento = modelo_entrenado.history['mean_squared_error'][-1]
    mse_test = evaluacion[1]

    return red_base, mse_entrenamiento, mse_test


def entrenar_modelo(model: tf.keras.models.Model,
                    train_data: tf.Tensor,
                    train_label: tf.Tensor,
                    val_data: tf.Tensor,
                    val_label: tf.Tensor,
                    epochs: int,
                    batch_size: int,
                    num_columns: int,
                    patience: int = 7,
                    callback_plot: List[Callback] = []) -> tf.keras.callbacks.History:
    """
    Trains a neural network model with optional callbacks.

    Parameters:
        model (tf.keras.models.Model): Neural network model to be trained.
        train_data (tf.Tensor): Training input data.
        train_label (tf.Tensor): Training target data.
        val_data (tf.Tensor): Validation input data.
        val_label (tf.Tensor): Validation target data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        num_columns (int): Number of columns in the input data.
        patience (int): Patience for early stopping. Default is 7.
        callback_plot (List[Callback]): List of additional callbacks. Default is an empty list.

    Returns:
        tf.keras.callbacks.History: History object containing training metrics.
    """
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(filepath='mejor_modelo_lstm.h5',
                                          monitor='val_loss',
                                          save_best_only=True,
                                          mode='min')
    callbacks.append(checkpoint_callback)

    # Additional callbacks
    callbacks.extend(callback_plot)

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate=0.005),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    
    history = model.fit(train_data, train_label, epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(val_data, val_label),
                        callbacks=callbacks)

    return history


def crearencoder(entradas: np.ndarray, dimension: int) -> tf.keras.models.Model:
    """
    Creates a convolutional autoencoder encoder.

    Parameters:
        entradas (np.ndarray): Input data.
        dimension (int): Dimension of the encoded representation.

    Returns:
        tf.keras.models.Model: Convolutional autoencoder encoder.
    """
    filtro1, filtro2 = 8, 12
    core1, core2 = 3, 4

    # Assuming entradas is a list of matrices and extracting the shape from the first matrix
    tam_matriz = np.squeeze(np.asarray(entradas[0]))
    num_entradas = (int(tam_matriz.shape[0]), int(tam_matriz.shape[1]), 1)

    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.layers.InputLayer(input_shape=num_entradas))
    encoder.add(tf.keras.layers.Conv2D(filters=filtro1, kernel_size=(core1, core1), activation=tf.keras.activations.relu, padding="same"))
    encoder.add(tf.keras.layers.Conv2D(filters=filtro2, kernel_size=(core2, core2), activation=tf.keras.activations.relu, padding="same"))
    encoder.add(tf.keras.layers.Flatten())
    encoder.add(tf.keras.layers.Dense(units=dimension, activation='linear'))
    encoder.build()

    return encoder

def creardecoder(dim_entrada: tuple[int, int, int]) -> tf.keras.models.Model:
    """
    Creates a decoder for a convolutional autoencoder.

    Parameters:
        dim_entrada (tuple): Shape of the input data.

    Returns:
        tf.keras.models.Model: Convolutional autoencoder decoder.
    """
    filtro1, filtro2 = 8, 12
    core1, core2 = 3, 4

    input_shape = dim_entrada[1:]  # Assuming dim_entrada is a tuple with shape (batch_size, height, width, channels)

    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    decoder.add(tf.keras.layers.Dense(units=1008, activation='relu'))
    decoder.add(tf.keras.layers.Reshape(target_shape=(7, 12, 12)))  # Adjust the target_shape based on your specific requirements
    decoder.add(tf.keras.layers.Conv2D(filters=filtro1, kernel_size=(core1, core1), activation=tf.keras.activations.relu, padding="same"))
    decoder.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(core1, core1), activation='relu', padding="same"))
    decoder.build()

    return decoder

def buildautoencoder(entradas: tf.Tensor, dimension: int) -> tuple[tf.keras.models.Model, tf.keras.models.Model]:
    """
    Builds a convolutional autoencoder model.

    Parameters:
        entradas (tf.Tensor): Input data.
        dimension (int): Dimension of the encoded representation.

    Returns:
        tuple[tf.keras.models.Model, tf.keras.models.Model]: Convolutional autoencoder and its decoder.
    """
    fn_perdida = tf.keras.losses.MeanSquaredError()

    encoder = crearencoder(entradas, dimension)
    decoder = creardecoder((1, dimension))

    autoencoder = tf.keras.Sequential([encoder, decoder])
    autoencoder.build((None, 7, 12, 1))
    autoencoder.compile(loss=fn_perdida, optimizer='Adam', metrics="BinaryAccuracy")

    return autoencoder, decoder

if __name__ == "__main__":
    print("All libraries are loaded correctly")
