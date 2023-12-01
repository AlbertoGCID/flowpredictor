import time
import io
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
from src.core.utils.AI_results import rank_five, sensitivity_analysis,metricasconfusion
from src.core.utils.AI_algorithms import crearmodelo, PlotLearning, entrenar_modelo
from src.core.utils.graphics import Graficas
from src.core.utils.telebot_api import enviar_mensaje_con_espera
from src.core.utils.preprocess.split_delay_time import procesado_de_tiempo, split_data,añadir_retardo_lista,sliding_window
from src.core.utils.preprocess.normalize import normalizar_datos
from typing import Union, Dict,List, Tuple,Any,Optional
import hydroeval
import telebot
bot = telebot.TeleBot("6114856166:AAGYqKXk1qSoupZZ9thLQOjT5QevfdL4aMA", parse_mode=None) # You can set parse_mode by default. HTML or MARKDOWN
idchatconbot = -807792928

def devolverregla(dato: pd.DataFrame, margen: float) -> str:
    """
    Determines the correctness of certain conditions based on values in the input DataFrame.

    Args:
        dato (pd.DataFrame): The input DataFrame containing specific columns.
        margen (float): The margin used for evaluating conditions.

    Returns:
        str: A string indicating the correctness of the conditions:
             - 'correcto1', 'correcto2', 'correcto3', or 'correcto4' if conditions are met.
             - 'error1', 'error2', 'error3', or 'error4' if conditions are not met.
    """
    Ze = dato["Ze"].values[0]
    Zme = dato["Zme"].values[0]
    Qe = dato["Qe"].values[0]
    Qtmax = dato["Qtmax"].values[0]
    Qaliv = dato["Qaliv"].values[0]
    Qfondo = dato["Qfondo"].values[0]
    Qsalida = dato["Qsalida"].values[0]
    Zmin = dato["Zmin"].values[0]
    Qeco = dato["Qeco"].values[0]
    Qtch = dato["Qtch"].values[0]

    if Ze > Zme:
        if Qe > Qtmax + Qaliv:
            target_min = Qaliv + Qtmax + Qfondo - margen
            target_max = Qaliv + Qtmax + Qfondo + margen
            if target_min < Qsalida < target_max:
                return "correcto2"
            else:
                return "error2"
        else:
            target_min = Qaliv + Qtmax - margen
            target_max = Qaliv + Qtmax + margen
            if target_min < Qsalida < target_max:
                return "correcto1"
            else:
                return "error1"
    elif Qe > Qtmax and Ze > Zmin:
        target_min = Qtmax - margen
        target_max = Qtmax + margen
        if target_min < Qsalida < target_max:
            return "correcto4"
        else:
            return "error4"
    else:
        maximo = max(Qeco, Qtch)
        target_min = maximo - margen
        target_max = maximo + margen
        if target_min < Qsalida < target_max:
            return "correcto3"
        else:
            return "error3"

        
def comprobardataframe(df: pd.DataFrame, margen: float) -> pd.DataFrame:
    """
    Applies the devolverregla function to each row in the DataFrame and adds the results as a new column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        margen (float): The margin used in devolverregla function.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column "prueba" containing the results.
    """
    lista = []
    for i in range(len(df)): 
        lista.append(devolverregla(df.iloc[[i]], margen))
    df["prueba"] = np.array(lista)
    return df

def conteoresultados(df: pd.DataFrame, 
                     name: str, 
                     datosprueba: pd.DataFrame) -> dict:
    """
    Counts the occurrences of different result categories in the 'prueba' column of the DataFrame.

    Args:
        df (pd.DataFrame): The original DataFrame.
        name (str): The name of the margin used in the analysis.
        datosprueba (pd.DataFrame): The DataFrame containing the 'prueba' column.

    Returns:
        dict: A dictionary containing counts for different result categories.
    """
    dic = dict()
    dic["margen"] = name
    for i in range(4):
        dic[f"cumple{str(i+1)}"] = datosprueba.prueba.str.contains(r'correcto{str(i+1)}').sum()
        dic[f"error1{str(i+1)}"] = datosprueba.prueba.str.contains(r'error{str(i+1)}').sum()
    dic["correcto"] = datosprueba.prueba.str.contains(r'correcto').sum()
    dic["incorrecto"] = datosprueba.prueba.str.contains(r'error').sum()
    return dic

def process_data(i: Tuple) -> Dict:
    """
    Process data using two functions: comprobardataframe and conteoresultados.

    Args:
        i (tuple): A tuple containing two elements.
            - Element 0 (str): The name for the margin used in the analysis.
            - Element 1 (pd.DataFrame): The DataFrame to be processed.

    Returns:
        dict: A dictionary containing counts for different result categories from conteoresultados.
    """
    result_dataframe = comprobardataframe(i[1], i[0])
    result_counts = conteoresultados(result_dataframe, i[0], i[1])
    return result_counts

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocesar_datos(datos: pd.DataFrame, 
                      col_a_eliminar: List = [], 
                      coltemporal: str = str) -> tuple:
    """
    Preprocesses a pandas dataset by removing specified columns, normalizing, and splitting into training, validation, and test sets.

    Args:
        datos (pd.DataFrame): The input DataFrame.
        col_a_eliminar (list, optional): List of column names to be removed. Defaults to an empty list.
        coltemporal (str): The column name representing temporal information.

    Returns:
        tuple: A tuple containing the following elements:
            - x_train (pd.DataFrame): Training feature data.
            - y_train (pd.DataFrame): Training target data.
            - x_val (pd.DataFrame): Validation feature data.
            - y_val (pd.DataFrame): Validation target data.
            - x_test (pd.DataFrame): Test feature data.
            - y_test (pd.DataFrame): Test target data.
            - datos_comp (pd.DataFrame): Complete feature data after normalization.
            - y_datos_comp (pd.DataFrame): Complete target data after normalization.
    """
    x_data = datos.copy()
    columnas_existentes = x_data.columns.tolist()
    columnas_a_eliminar_existen = [col for col in col_a_eliminar if col in columnas_existentes]
    if columnas_a_eliminar_existen:
        x_data.drop(columns=columnas_a_eliminar_existen, inplace=True, axis=1)
    x_data = procesado_de_tiempo(x_data, coltemporal)  # Assuming procesado_de_tiempo is defined elsewhere
    x_train, x_val, x_test = split_data(x_data, 0.1, 0.2)  # Assuming split_data is defined elsewhere

    normalizado = StandardScaler()  # Apply mean-std normalization
    normalizado2 = MinMaxScaler()  # Apply min-max normalization
    normalizado3 = StandardScaler()  # Apply mean-std normalization
    normalizado4 = MinMaxScaler()  # Apply min-max normalization

    normalizado.fit(x_train)
    x_train = normalizado.transform(x_train)
    x_val = normalizado.transform(x_val)
    x_test = normalizado.transform(x_test)

    normalizado3.fit(x_data)
    datos_comp = normalizado3.transform(x_data)

    normalizado2.fit(x_train)
    x_train = normalizado2.transform(x_train)
    x_val = normalizado2.transform(x_val)
    x_test = normalizado2.transform(x_test)

    normalizado4.fit(datos_comp)
    datos_comp = normalizado4.transform(datos_comp)

    y_train = x_train[:, 0]
    y_val = x_val[:, 0]
    y_test = x_test[:, 0]
    y_datos_comp = datos_comp[:, 0]

    x_train = pd.DataFrame(x_train, columns=x_data.columns)
    x_val = pd.DataFrame(x_val, columns=x_data.columns)
    x_test = pd.DataFrame(x_test, columns=x_data.columns)
    datos_comp = pd.DataFrame(datos_comp, columns=x_data.columns)

    y_train = pd.DataFrame(y_train, columns=[x_data.columns[0]])
    y_val = pd.DataFrame(y_val, columns=[x_data.columns[0]])
    y_test = pd.DataFrame(y_test, columns=[x_data.columns[0]])
    y_datos_comp = pd.DataFrame(y_datos_comp, columns=[x_data.columns[0]])

    return x_train, y_train, x_val, y_val, x_test, y_test, datos_comp, y_datos_comp


def preprocesar_datos2(datos: pd.DataFrame, 
                       col_a_eliminar: List = [], 
                       coltemporal: str = str,
                       colsalida: str = None, 
                       tiempo: bool = True) -> Union[Tuple,pd.DataFrame]:
    """
    Preprocesses a pandas dataset by removing specified columns, normalizing, and optionally processing temporal information.

    Args:
        datos (pd.DataFrame): The input DataFrame.
        col_a_eliminar (list, optional): List of column names to be removed. Defaults to an empty list.
        coltemporal (str): The column name representing temporal information.
        colsalida (str, optional): The column name for the target variable. Defaults to None.
        tiempo (bool, optional): Whether to process temporal information. Defaults to True.

    Returns:
        tuple: A tuple containing the following elements:
            - x_ds (pd.DataFrame): The preprocessed feature data.
            - y_ds (pd.DataFrame, optional): The preprocessed target data if colsalida is provided.
    """
    x_data = datos.copy()

    # Remove non-numeric columns
    x_data.drop(col_a_eliminar, inplace=True, axis=1)

    # Process temporal information (transform dates into sines and cosines)
    if tiempo:
        x_data = procesado_de_tiempo(x_data, coltemporal)
    else:
        eliminado = x_data.pop(coltemporal)

    # Normalize all columns
    x_ds = pd.DataFrame(x_data, columns=x_data.columns)

    if colsalida is not None:
        y_dataset = x_data.loc[:, colsalida]
        y_ds = pd.DataFrame(y_dataset, columns=[colsalida])
        return x_ds, y_ds
    else:
        return x_ds




def preparado_dataset(datos: pd.DataFrame, 
                      lista_retardos: List[int] = [], 
                      retardo: int = 0,
                      col_a_eliminar: List[str] = [], 
                      coltemporal: str = str,
                      colsalida: str = None, 
                      tiempo: bool = True, 
                      mezclar: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares a pandas dataset by adding temporal context, applying delays, and optionally shuffling rows.

    Args:
        datos (pd.DataFrame): The input DataFrame.
        lista_retardos (List[int], optional): List of delays to apply. Defaults to an empty list.
        retardo (int, optional): A single delay value to apply. Defaults to 0.
        col_a_eliminar (List[str], optional): List of column names to be removed. Defaults to an empty list.
        coltemporal (str): The column name representing temporal information.
        colsalida (str, optional): The column name for the target variable. Defaults to None.
        tiempo (bool, optional): Whether to add temporal context. Defaults to True.
        mezclar (bool, optional): Whether to shuffle rows. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the preprocessed feature data (x) and the target data (y).
    """
    dataset = datos.copy()
    print("\nPreparando_dataset")

    # Add temporal context
    if tiempo:
        print("Añadiendo contexto temporal")
        dataset_tiempo = procesado_de_tiempo(dataset, coltemporal)
    else:
        print("Sin contexto temporal")
        eliminado = dataset.pop(coltemporal)
        dataset_tiempo = dataset

    # Apply delays
    if len(lista_retardos) > 0:
        print("Añadiendo retardos")
        dataset_retardo = añadir_retardo_lista(dataset_tiempo, datos, lista_retardos, retardo)
    else:
        print("Sin retardos")
        dataset_retardo = dataset_tiempo.copy()

    # Shuffle rows if specified
    if mezclar:
        print("Mezclando")
        dataset_mezcla = dataset_retardo.sample(frac=1).reset_index(drop=True)
    else:
        print("Sin mezclar filas")
        dataset_mezcla = pd.DataFrame(dataset_retardo, columns=dataset_retardo.columns.tolist())

    # Extract target variable and feature data
    if colsalida in dataset_mezcla.columns:
        y = pd.DataFrame(dataset_mezcla[colsalida], columns=[colsalida])
        x = dataset.drop(colsalida, axis=1)
    else:
        raise KeyError(f'{colsalida} no se encuentra en el dataset: {dataset_mezcla.columns.to_list()}')

    # Remove unwanted columns
    print("Eliminando columnas no deseadas\n")
    x = x.drop(columns=col_a_eliminar)
    x.fillna(0, inplace=True)
    y.fillna(0, inplace=True)

    return x, y



def comprobar_resultados(model: Union[Sequential, tf.keras.Model], 
                         x_train: pd.DataFrame,
                         y_train: pd.DataFrame, 
                         ds: pd.DataFrame, 
                         index: Union[pd.DataFrame, int],
                         rango: int, 
                         normalizado: Any, 
                         etiqueta: str = "Qe",
                         num_salida: int = 0, 
                         imprimir: bool = False, 
                         threshold: Optional[float] = None) -> Tuple[List[Tuple[float, float]], List[int], np.ndarray]:
    """
    Checks the results of a model by making predictions on the training set and evaluating performance metrics.

    Args:
        model (Union[Sequential, tf.keras.Model]): The trained TensorFlow or Keras model.
        x_train (pd.DataFrame): The input features for the training set.
        y_train (pd.DataFrame): The target values for the training set.
        ds (pd.DataFrame): The original dataset.
        index (Union[pd.DataFrame, int]): Index information for mapping predictions to the original dataset.
        rango (int): The range parameter.
        normalizado (Union[Normalizer, StandardScaler]): The normalization method used for denormalizing predictions.
        etiqueta (str, optional): The target variable label. Defaults to "Qe".
        num_salida (int, optional): The index of the output variable to consider. Defaults to 0.
        imprimir (bool, optional): Whether to print predictions. Defaults to False.
        threshold (float, optional): Threshold for binary classification metrics. Defaults to None.

    Returns:
        Tuple[List[Tuple[float, float]], List[int], np.ndarray]: A tuple containing:
            - A list of tuples containing predicted and actual values.
            - A list of indices representing the top 5 important features.
            - An array of feature names.
    """
    predicciones = []
    listax = []
    listay = []

    for i in range(len(x_train)):
        # Convert train to numpy
        x_train_np = np.array(x_train.iloc[i, :]).reshape(1, -1)
        # Convert from numpy to tensor
        x_train_tensor = tf.convert_to_tensor(x_train_np, dtype=tf.float32)
        # Predict the output
        if isinstance(model, Sequential):
            salida = model.call(x_train_tensor)
        else:
            salida = model.predict(x_train_tensor)
        salida_desn = normalizado.denormalize(salida, 2)  # min_max denormalize
        salida_fin = normalizado.denormalize(salida_desn, 1)  # mean_std denormalize
        # Locate the column number of the target variable
        index2 = ds.columns.get_loc(etiqueta)
        if isinstance(index, pd.DataFrame):
            ind = index.iloc[i + rango][0]
        elif isinstance(index, int):
            ind = index + i
        predicciones.append((salida_fin.numpy()[0][num_salida], ds.iloc[ind, index2]))
        listax.append(salida_fin.numpy()[0][num_salida])
        listay.append(ds.iloc[ind, index2])
        if imprimir:
            print(f'Valor real para la fila {ind} es {ds.iloc[ind, index2]:.2f} m3/s, valor predicho por la red {salida_fin.numpy()[0][num_salida]:.2f}')

    x = np.array(listax)
    y = np.array(listay)

    # Calculate the correlation coefficient using numpy's corrcoef function
    corr_coef = np.corrcoef(x, y)[0][1]
    # Square the correlation coefficient to get the r² correlation
    r_squared = corr_coef ** 2

    print(f"\nEl coeficiente de correlación es {corr_coef} y su r^2 es {r_squared}\n")

    # Compute binary classification metrics
    if threshold is not None:
        metricasconfusion(x, y, threshold)

    # Compute feature importance
    features = sensitivity_analysis(x_train, y_train, model)
    mejores_5 = rank_five(features)
    nombres = x_train.columns.values
    for j, item in enumerate(mejores_5):
        print(f"Con el ranking {j+1} de importancia tenemos la variable {nombres[item]} {features[item]:.3f}%\n")

    return predicciones, mejores_5, nombres

def comprobar_resultados2(model: Union[SequentialFeatureSelector, Sequential], 
                          x_train: pd.DataFrame, 
                          y_train: pd.DataFrame,
                          x_test: pd.DataFrame, 
                          y_test: pd.DataFrame, 
                          index: pd.DataFrame, 
                          normalizado_train: Any,
                          normalizado_test: Any, 
                          etiqueta: str = "Qe", 
                          num_salida: int = 0, 
                          imprimir: bool = False,
                          threshold: Optional[float] = None) -> Union[Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[int], np.ndarray], Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
    """
    Checks the results of a model on training and testing sets, making predictions and evaluating performance metrics.

    Args:
        model (Union[SequentialFeatureSelector, Sequential]): The trained TensorFlow or Keras model.
        x_train (pd.DataFrame): The input features for the training set.
        y_train (pd.DataFrame): The target values for the training set.
        x_test (pd.DataFrame): The input features for the testing set.
        y_test (pd.DataFrame): The target values for the testing set.
        index (pd.DataFrame): Index information for mapping predictions to the original dataset.
        normalizado_train (Normalizer): The normalization method used for denormalizing training set predictions.
        normalizado_test (Normalizer): The normalization method used for denormalizing testing set predictions.
        etiqueta (str, optional): The target variable label. Defaults to "Qe".
        num_salida (int, optional): The index of the output variable to consider. Defaults to 0.
        imprimir (bool, optional): Whether to print predictions. Defaults to False.
        threshold (float, optional): Threshold for binary classification metrics. Defaults to None.

    Returns:
        Union[Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[int], np.ndarray], Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
            - A tuple containing:
                - A list of tuples containing predicted and actual values for the training set.
                - A list of tuples containing predicted and actual values for the testing set.
                - A list of indices representing the top 5 important features.
                - An array of feature names.
            - A tuple containing:
                - A list of tuples containing predicted and actual values for the training set.
                - A list of tuples containing predicted and actual values for the testing set.
    """
    predicciones_train = []
    predicciones_test = []
    listax_train = []
    listay_train = []
    listax_test = []
    listay_test = []

    if y_train.shape == (y_train.shape[0],):
        # Reshape the Series to (num_samples, 1)
        y_train = y_train.values.reshape(-1, 1)
    if y_test.shape == (y_test.shape[0],):
        # Reshape the Series to (num_samples, 1)
        y_test = y_test.values.reshape(-1, 1)

    salida_real_desn_train = normalizado_train.inverse_transform(y_train)
    salida_real_desn_test = normalizado_test.inverse_transform(y_test)

    variables = x_train.columns.tolist()

    for i in range(len(x_train)):
        # Covert row "i" to numpy and reshape
        x_train_np = np.array(x_train.iloc[i, :]).reshape(1, -1)
        # Convert row "i" from numpy to tensor
        x_train_tensor = tf.convert_to_tensor(x_train_np, dtype=tf.float32)

        # Predict output for row "i"
        if isinstance(model, SequentialFeatureSelector):
            salida = model.call(x_train_tensor)
        else:
            x_train_tensor = pd.DataFrame(x_train_tensor)
            x_train_tensor.columns = variables
            salida = model.predict(x_train_tensor)
            # Reshape min-max normalized prediction for row "i"
            salida = salida.reshape(1, -1)

        salida_fin = normalizado_train.inverse_transform(salida) 
        predicciones_train.append((salida_fin[0][0], salida_real_desn_train[i, 0]))
        listax_train.append(salida_fin[0][0])
        listay_train.append(salida_real_desn_train[i, 0])

        if imprimir:
            print(f'Valor real para la fila {index.iloc[i, 0]} es {salida_real_desn_train[i, 0]:.2f} m3/s, valor predicho por la red {salida_fin[0][0]:.2f}')

    for i in range(len(x_test)):
        # Covert row "i" to numpy and reshape
        x_test_np = np.array(x_test.iloc[i, :]).reshape(1, -1)
        # Convert row "i" from numpy to tensor
        x_test_tensor = tf.convert_to_tensor(x_test_np, dtype=tf.float32)

        # Predict output for row "i"
        if isinstance(model, SequentialFeatureSelector):
            salida = model.call(x_test_tensor)
        else:
            x_test_tensor = pd.DataFrame(x_test_tensor)
            x_test_tensor.columns = variables
            salida = model.predict(x_test_tensor)
            # Reshape min-max normalized prediction for row "i"
            salida = salida.reshape(1, -1)

        salida_fin = normalizado_test.inverse_transform(salida) 
        predicciones_test.append((salida_fin[0][0], salida_real_desn_test[i, 0]))
        listax_test.append(salida_fin[0][0])
        listay_test.append(salida_real_desn_test[i, 0])

        if imprimir:
            print(f'Valor real para la fila {index.iloc[i, 0]} es {salida_real_desn_test[i, 0]:.2f} m3/s, valor predicho por la red {salida_fin[0][0]:.2f}')

    x_train_array = np.array(listax_train)
    y_train_array = np.array(listay_train)

    x_test_array = np.array(listax_test)
    y_test_array = np.array(listay_test)

    # Calculate the correlation coefficient using numpy's corrcoef function
    corr_coef_train = np.corrcoef(x_train_array, y_train_array)[0][1]
    r_squared_train = corr_coef_train ** 2

    corr_coef_test = np.corrcoef(x_test_array, y_test_array)[0][1]
    r_squared_test = corr_coef_test ** 2

    print(f"\nEl coeficiente de correlación para el conjunto de entrenamiento es {corr_coef_train} y su r^2 es {r_squared_train}\n")
    print(f"El coeficiente de correlación para el conjunto de prueba es {corr_coef_test} y su r^2 es {r_squared_test}\n")

    # Compute binary classification metrics
    if threshold is not None:
        metricasconfusion(x_test_array, y_test_array, threshold)

    # Compute feature importance
    if isinstance(model, SequentialFeatureSelector):
        features = sensitivity_analysis(x_test, y_test, model)
        mejores_5 = rank_five(features)
        nombres = x_train.columns.values
        for j, item in enumerate(mejores_5):
            print(f"Con el ranking {j+1} de importancia tenemos la variable {nombres[item]} {features[item]:.3f}%\n")
        Graficas.plot_feature_importance(features, x_test)
        return predicciones_train, predicciones_test, mejores_5, nombres
    else:
        return predicciones_train, predicciones_test
    
def pred_a_x_dias(ds_lstm_tr: pd.DataFrame,
                  ds_lstm_va: pd.DataFrame,
                  ds_lstm_te: pd.DataFrame,
                  gap: int,
                  exclusiones: List[str],
                  linea: bool = False,
                  objetivo: str = "Ze",
                  tipo: str = "mean",
                  input_width: int = 15,
                  Tiempo: bool = False,
                  ultimo_envio: float = time.time()) -> Tuple[tf.keras.Model, float, float, float, float, float, float, float, float, float, float, float, float]:
    """
    Perform prediction for a given number of days using LSTM model.

    Parameters:
    - ds_lstm_tr: Training dataset (DataFrame).
    - ds_lstm_va: Validation dataset (DataFrame).
    - ds_lstm_te: Test dataset (DataFrame).
    - gap: Number of days for prediction.
    - exclusiones: Excluded columns.
    - linea: Flag to indicate if a line is to be plotted.
    - objetivo: Target variable.
    - tipo: Type of normalization ("mean" or "other").
    - input_width: Number of days for input sequence.
    - Tiempo: Flag to include time in preprocessing.
    - ultimo_envio: Timestamp of the last message sent.

    Returns:
    - lstm_model: Trained LSTM model.
    - loss_test: Mean squared error on the test set.
    - loss_train: Mean squared error on the training set.
    - ns_test: Nash-Sutcliffe efficiency on the test set.
    - ns_train: Nash-Sutcliffe efficiency on the training set.
    - r_cuadrado_test: R-squared on the test set.
    - r_cuadrado_train: R-squared on the training set.
    - std_naive_te: Standard deviation of naive model on the test set.
    - std_naive_tr: Standard deviation of naive model on the training set.
    - std_diff_modelo_te: Standard deviation of the difference between model prediction and true values on the test set.
    - std_diff_modelo_tr: Standard deviation of the difference between model prediction and true values on the training set.
    - ultimo_envio: Updated timestamp of the last message sent.
    """


    #lstm_model_prueba = modelo

    naive_train = pd.DataFrame()
    naive_val = pd.DataFrame()
    naive_test = pd.DataFrame()
    naive_train[objetivo+'+'+ str(gap)] = ds_lstm_tr[objetivo].shift(-gap,fill_value=0)
    # calculo el modelo naive con (Ze - Ze+1)
    naive_train['diff']= np.abs(ds_lstm_tr[objetivo] - naive_train[objetivo+'+'+ str(gap)])
    # como la última fila no tiene Ze+1 y toma valor 0 la última diferencia daría Ze
    # así que le asigno 0
    naive_train.loc[naive_train['diff'] > 200] = 0 


    naive_val[objetivo+'+'+ str(gap)] = ds_lstm_va[objetivo].shift(-gap,fill_value=0)
    naive_val['diff'] = np.abs(ds_lstm_va[objetivo] - naive_val[objetivo+'+'+ str(gap)])
    naive_val.loc[naive_val['diff'] > 200, 'diff'] = 0 

    fill = ds_lstm_te[objetivo]
    fill = fill.iloc[-1]
    naive_test[objetivo+'+'+ str(gap)] = ds_lstm_te[objetivo].shift(-gap,fill_value=fill)
    naive_test['diff'] = np.abs(ds_lstm_te[objetivo] - naive_test[objetivo+'+'+ str(gap)])
    naive_test.loc[naive_test['diff'] > 200, 'diff'] = 0 ####### MSE #########:
    #print("forma de los modelos:")
    #print(y_test.shape,model_result_desnorm.shape)



    # calcular esto con train
    std_naive_te = np.std(naive_test['diff'])
    media_naive_te = np.mean(naive_test['diff'])

    std_naive_tr = np.std(naive_train['diff'])
    media_naive_tr = np.mean(naive_train['diff'])
    #print(f'El tamaño de test antes de preprocesar es {ds_lstm_te.shape}')
    # me quedo con el índice donde está la columna que me interesa

    num_columns_df1 = ds_lstm_tr.shape[1]
    num_columns_df2 = ds_lstm_va.shape[1]
    num_columns_df3 = ds_lstm_te.shape[1]

    assert num_columns_df1 == num_columns_df2 == num_columns_df3, "Número de columnas diferente"

    datos_train_dn_lstm,y_datos_train_dn_lstm = preprocesar_datos2(ds_lstm_tr ,exclusiones,"Fecha",objetivo,tiempo=Tiempo)
    datos_val_dn_lstm,y_datos_val_dn_lstm = preprocesar_datos2(ds_lstm_va,exclusiones,"Fecha",objetivo,tiempo=Tiempo)
    datos_test_dn_lstm,y_datos_test_dn_lstm = preprocesar_datos2(ds_lstm_te,exclusiones,"Fecha",objetivo,tiempo=Tiempo)
    
    indice_col_obj = naive_test.columns.get_loc(objetivo+'+'+ str(gap))
    #tengo norm1 y norm2, puedo coger el valor 252

    #finalmente los datos normalizados
    ds_lstm_train,norm1_dt_lstm = normalizar_datos(datos_train_dn_lstm,datos_train_dn_lstm,tipo)
    y_datos_train_lstm,norm1_ydt_lstm = normalizar_datos(y_datos_train_dn_lstm,y_datos_train_dn_lstm,tipo)

    ds_lstm_val,norm1_dv_lstm = normalizar_datos(datos_train_dn_lstm,datos_val_dn_lstm,tipo)
    y_datos_val2_lstm,norm1_ydv2v = normalizar_datos(y_datos_train_dn_lstm,y_datos_val_dn_lstm,tipo)

    ds_lstm_test,norm1_dtest_lstm = normalizar_datos(datos_train_dn_lstm,datos_test_dn_lstm,tipo)
    y_datos_test2_lstm,norm1_ydtest2_lstm= normalizar_datos(y_datos_train_dn_lstm,y_datos_test_dn_lstm,tipo)

    label_width = 1
    target_labels = objetivo 
    x_train, y_train = sliding_window(ds_lstm_train, y_datos_train_lstm, input_width, label_width, gap)
    x_val, y_val = sliding_window(ds_lstm_val, y_datos_val2_lstm, input_width, label_width, gap)
    x_test, y_test = sliding_window(ds_lstm_test, y_datos_test2_lstm, input_width, label_width, gap)
    num_columns = len(ds_lstm_train.columns)
    MAX_EPOCHS = 10
    batch_size = 32
    nc = 1
    unid = 24
    lstm_model= crearmodelo(num_capas=nc,unidades=unid,batchnorm=0,dropout=0,dropout_rate=0.2,seed=123,label_width=label_width,act='None')

    plot_fit = PlotLearning(num_capas=nc, unidades=unid, batch=batch_size, epochs=MAX_EPOCHS)
    history_prueba = entrenar_modelo(lstm_model, x_train, y_train, x_val, y_val, MAX_EPOCHS, batch_size,num_columns,20,[plot_fit])
    lstm_model = tf.keras.models.load_model('mejor_modelo_lstm.h5')
    model_result = lstm_model.predict(x_test)
    model_result_train = lstm_model.predict(x_train)
    # desnormalización min-max (dato+min * max-min)
    model_result_desnorm = norm1_ydtest2_lstm.inverse_transform(model_result) 
    model_result_desnorm_train = norm1_ydt_lstm.inverse_transform(model_result_train) 

    x_test_copiado = x_test.copy()
    x_train_copiado = x_train.copy()
    
    x_test_redimensionado = x_test_copiado[:,-1,:]
    x_train_redimensionado = x_train_copiado[:,-1,:]
    #print(f'El shape de x_test_redimensionado es {x_test_redimensionado.shape}')
    etiquetas_x_test = norm1_dt_lstm.inverse_transform(x_test_redimensionado)
    etiquetas_x_train = norm1_dt_lstm.inverse_transform(x_train_redimensionado)

    # parte de análisis de sensibilidad con el modelo
    # evaluate utiliza la función de pérdida definida en el modelo lstm para calcular el loss entre
    # las predicciones y la salida real
    baseline_score = lstm_model.evaluate(x_train, y_train, verbose=0)[0]

    if tipo == "mean":
        baseline = (baseline_score + norm1_ydtest2_lstm.mean_) * (np.sqrt(norm1_ydtest2_lstm.var_))
    else:
        baseline = (baseline_score + norm1_ydtest2_lstm.data_min_) * (norm1_ydtest2_lstm.data_max_ - norm1_ydtest2_lstm.data_min_) 

    num_features = x_train.shape[2]
    # Initialize an array to store the importance scores
    feature_importance_scores = np.zeros(num_features)

    
    # Loop through each feature
    for i in range(num_features):
        X_removed = x_train.copy()
        X_removed[:,:,i] = 0
        
        # Evaluate devuelve el loss entre y_pred e y_true

        sc2 = lstm_model.evaluate(X_removed, y_train, verbose=0)[0]
        if tipo == "mean":
            score = (sc2 + norm1_ydtest2_lstm.mean_) * (np.sqrt(norm1_ydtest2_lstm.var_))
        else:
            score = (sc2 + norm1_ydtest2_lstm.data_min_) * (norm1_ydtest2_lstm.data_max_ - norm1_ydtest2_lstm.data_min_) 
        
        #print(f'loss desnormalizado una variable = 0: {score} loss desnormalizado todas variables: {baseline} , \nloss normalizado una variable = 0: {sc2} loss normalizado todas variables: {baseline_score} \n\n')
        # Calculate the decrease in performance
        feature_importance_scores[i] = ((score - baseline) / baseline) * 100

        # Reset the weights of the model
        lstm_model.reset_states()
    
    
    mejores_5 = rank_five(feature_importance_scores)
    nombres = datos_train_dn_lstm.columns.tolist()
    for j, item in enumerate(mejores_5):
        mensaje_rank = f"Con el ranking {j+1} de importancia tenemos la variable {nombres[item]} {feature_importance_scores[item]:.3f}%\n"
        print(mensaje_rank)
        #bot.send_message(idchatconbot, mensaje_rank)
    ploteado = Graficas(ultimo_envio=ultimo_envio)
    ultimo_envio = ploteado.plot_feature_importance(feature_importance_scores=feature_importance_scores, dataset= datos_train_dn_lstm,ultimo_envio=ultimo_envio)

    y_test_denorm = norm1_ydtest2_lstm.inverse_transform(y_test.reshape(-1,1)) 
    y_train_denorm = norm1_ydt_lstm.inverse_transform(y_train.reshape(-1,1)) 
    mod_naive = naive_test['diff']


    # calcular esto con train
    diff_modelo_test = np.abs(model_result_desnorm.reshape(-1)-y_test_denorm.reshape(-1))     
    std_diff_modelo_te = np.std(diff_modelo_test)
    media_diff_modelo_te = np.mean(diff_modelo_test)

    diff_modelo_train = np.abs(model_result_desnorm_train.reshape(-1)-y_train_denorm.reshape(-1))
    std_diff_modelo_tr = np.std(diff_modelo_train)
    media_diff_modelo_tr = np.mean(diff_modelo_train)

    ########## METRICAS ######################

    ####### MSE #########:
    #print("forma de los modelos:")
    #print(y_test.shape,model_result_desnorm.shape)


    y_t = y_test.reshape(y_test.shape[0], 1)


    y_tr = y_train.reshape(y_train.shape[0],1)
    #print(y_t.shape,model_result_desnorm.shape)


    y_t = y_t[(input_width+gap):-1]
    y_tr = y_tr[(input_width+gap):-1]

    model_result_desnorm_trimmed = model_result_desnorm[(input_width+gap):-1]
    model_result_desnorm_train_trimmed = model_result_desnorm_train[(input_width+gap):-1]


    loss_test = mean_squared_error(y_t, model_result_desnorm_trimmed)

    loss_train = mean_squared_error(y_tr,model_result_desnorm_train_trimmed)

    ######### R2 ##########

    y_test_denorm_trimmed = y_test_denorm[(input_width+gap):-1]
    y_train_denorm_trimmed = y_train_denorm[(input_width+gap):-1]

    correlacion_test = np.corrcoef(model_result_desnorm_trimmed.reshape(-1),y_test_denorm_trimmed.reshape(-1),rowvar=False)[0][1]
    r_cuadrado_test = correlacion_test ** 2
    mensaje_corr_test = f"\nEl coeficiente de correlación en test es {(correlacion_test*100):.2f}% y su r^2 es {(r_cuadrado_test*100):.2f}% \n\n"
    print(mensaje_corr_test)

    correlacion_train = np.corrcoef(model_result_desnorm_train_trimmed.reshape(-1),y_train_denorm_trimmed.reshape(-1),rowvar=False)[0][1]
    r_cuadrado_train = correlacion_train ** 2
    mensaje_corr_train = f"\nEl coeficiente de correlación en train es {(correlacion_train*100):.2f}% y su r^2 es {(r_cuadrado_train*100):.2f}% \n\n"
    print(mensaje_corr_train)

    ######### NS #########

    ns_test = hydroeval.nse(y_test_denorm_trimmed.reshape(-1),model_result_desnorm_trimmed.reshape(-1))
    mensaje_ns_test= f"\nEl coeficiente de eficiencia Nash-Sutcliffe (NS) en test es {(ns_test):.2f} \n"
    print(mensaje_ns_test)
    ultimo_envio = enviar_mensaje_con_espera(idchatconbot,mensaje_ns_test,ultimo_envio)

    ns_train = hydroeval.nse(y_train_denorm_trimmed.reshape(-1),model_result_desnorm_train_trimmed.reshape(-1))
    mensaje_ns_train= f"\nEl coeficiente de eficiencia Nash-Sutcliffe (NS) en train es {(ns_train):.2f} \n"
    print(mensaje_ns_train)
    ultimo_envio = enviar_mensaje_con_espera(idchatconbot,mensaje_ns_train,ultimo_envio)
    



    #bot.send_message(idchatconbot, mensaje_corr)
    print('######################## TEST ###################')
    mensaje_media = f'ref:La media de la variación entre un día y el siguiente (real) es {media_naive_te} y su desv. tip. es {std_naive_te}'
    print(mensaje_media)
    #bot.send_message(idchatconbot,mensaje_media)
    mensaje_desv = f'La media de la variación entre la predicción y el valor real es {media_diff_modelo_te} y su desv. tip. es {std_diff_modelo_te}'
    print(mensaje_desv)
    #bot.send_message(idchatconbot,mensaje_desv)


    ultimo_envio = ploteado.plot_seaborn(mod_naive,media_naive_te,std_naive_te,diff_modelo_test,
             media_diff_modelo_te,std_diff_modelo_te,ultimo_envio)
    ultimo_envio = ploteado.plot_boxplot(mod_naive, diff_modelo_test,ultimo_envio)

    ############ test #################

    plt.figure(figsize=(12, 12))
    plt.plot(range(len(model_result_desnorm)), model_result_desnorm, label='y_pred')
    plt.plot(range(len(y_test_denorm.reshape(-1))), y_test_denorm.reshape(-1), label='y_test')
    plt.plot(range(len(etiquetas_x_test[:,indice_col_obj])),etiquetas_x_test[:,indice_col_obj],
              label='Naive')
    #if linea:
    #    plt.axhline(y=252, color='r', linestyle='--', label='Nivel de riesgo presa')
    plt.xlabel('Daily samples')
    plt.ylabel("Dam Inflow")
    plt.title(f'Prediction and actual value for a {str(gap)}-day interval using {str(input_width)} days of information in TEST')
    plt.legend()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Enviar la gráfica como imagen al chat específico
    # Reemplaza "ID_DEL_CHAT" con el ID del chat al que deseas enviar la gráfica
    presente = time.time()
    if presente - ultimo_envio < 3:
        time.sleep(3 - (presente-ultimo_envio))
    bot.send_photo(idchatconbot,buffer)
    ultimo_envio = time.time()
    # Cerrar el buffer y liberar recursos
    buffer.close()
    plt.show()

    diff_y_pred = model_result_desnorm.ravel() - y_test_denorm.reshape(-1)
    diff_naive = etiquetas_x_test[:, indice_col_obj] - y_test_denorm.reshape(-1)

    # Crear una nueva figura
    plt.figure(figsize=(12, 12))

    # Agregar las líneas de las diferencias a la gráfica
    plt.plot(range(len(diff_y_pred)), diff_y_pred, label='LSTM error')
    plt.plot(range(len(diff_naive)), diff_naive, label='Naive error')
    plt.xlabel('Index')
    plt.ylabel('Error')
    plt.title(f'Naive error vs model error')
    plt.legend()

    # Crear y mostrar la gráfica
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Enviar la gráfica como imagen al chat específico
    # Reemplaza "ID_DEL_CHAT" con el ID del chat al que deseas enviar la gráfica
    presente = time.time()
    if presente - ultimo_envio < 3:
        time.sleep(3 - (presente-ultimo_envio))
    bot.send_photo(idchatconbot,buffer)
    ultimo_envio = time.time()

    # Cerrar el buffer y liberar recursos
    buffer.close()
    plt.show()

    ############## train #####################

    print('######################## TRAIN ###################')

    mensaje_media_tr = f'ref:La media de la variación entre un día y el siguiente (real) es {media_naive_tr} y su desv. tip. es {std_naive_tr}'
    print(mensaje_media_tr)
    #bot.send_message(idchatconbot,mensaje_media_tr)
    mensaje_desv_tr = f'La media de la variación entre la predicción y el valor real es {media_diff_modelo_tr} y su desv. tip. es {std_diff_modelo_tr}'
    print(mensaje_desv_tr)
    #bot.send_message(idchatconbot,mensaje_desv_tr)

    plt.figure(figsize=(12, 12))
    plt.plot(range(len(model_result_desnorm_train)), model_result_desnorm_train, label='y_pred')
    plt.plot(range(len(y_train_denorm.reshape(-1))), y_train_denorm.reshape(-1), label='y_train')
    plt.plot(range(len(etiquetas_x_train[:,indice_col_obj])),etiquetas_x_train[:,indice_col_obj],
              label='Naive')
    if linea:
        plt.axhline(y=252, color='r', linestyle='--', label='Nivel de riesgo presa')
    plt.xlabel('Indice')
    plt.ylabel(target_labels)
    plt.title(f'Predicción y valor real para un intervalo de {str(gap)} dias usando {str(input_width)} días de información en TRAIN')
    plt.legend()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Enviar la gráfica como imagen al chat específico
    # Reemplaza "ID_DEL_CHAT" con el ID del chat al que deseas enviar la gráfica
    presente = time.time()
    if presente - ultimo_envio < 3:
        time.sleep(3 - (presente-ultimo_envio))
    bot.send_photo(idchatconbot,buffer)
    ultimo_envio = time.time()
    
    # Cerrar el buffer y liberar recursos
    buffer.close()
    plt.show()


    diff_y_pred_train = model_result_desnorm_train.ravel() - y_train_denorm.reshape(-1)
    diff_naive_train = etiquetas_x_train[:, indice_col_obj] - y_train_denorm.reshape(-1)



    # Crear una nueva figura
    plt.figure(figsize=(12, 12))

    # Agregar las líneas de las diferencias a la gráfica
    plt.plot(range(len(diff_y_pred_train)), diff_y_pred_train, label='Error modelo')
    plt.plot(range(len(diff_naive_train)), diff_naive_train, label='Error naive')
    plt.xlabel('Indice')
    plt.ylabel('Error')
    plt.title(f'Error naive vs Error modelo')
    plt.legend()

    # Crear y mostrar la gráfica
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Enviar la gráfica como imagen al chat específico
    # Reemplaza "ID_DEL_CHAT" con el ID del chat al que deseas enviar la gráfica
    presente = time.time()
    if presente - ultimo_envio < 3:
        time.sleep(3 - (presente-ultimo_envio))
    bot.send_photo(idchatconbot,buffer)
    ultimo_envio = time.time()

    # Cerrar el buffer y liberar recursos
    buffer.close()
    plt.show()


    return lstm_model,loss_test,loss_train,ns_test,ns_train,r_cuadrado_test,r_cuadrado_train,std_naive_te,std_naive_tr, std_diff_modelo_te ,std_diff_modelo_tr, ultimo_envio


def pruebamodelos(model: Any,
                  model_name: str, 
                  entrenamiento: Union[pd.DataFrame,pd.Series, np.ndarray], 
                  validacion: Union[pd.DataFrame,pd.Series, np.ndarray], 
                  test: Union[pd.DataFrame,pd.Series, np.ndarray], 
                  gap: int, 
                  retardo: int, 
                  unique_years: List[str], 
                  exclusiones: List[str], 
                  ultimo_envio: Any, iter):
    """
    Perform model training and evaluation.

    Parameters:
    - model: Model object (type depends on the model used).
    - model_name: Name of the model.
    - entrenamiento: Training dataset (DataFrame).
    - validacion: Validation dataset (DataFrame).
    - test: Test dataset (DataFrame).
    - gap: Time gap for shifting values.
    - retardo: Number of days for sliding window.
    - unique_years: List of unique years.
    - exclusiones: List of excluded columns.
    - ultimo_envio: Timestamp of the last message sent.
    - iter: Iteration number.

    Returns:
    - Tuple containing loss_test, loss_train, ns_test, ns_train, r2_test, r2_train, std_diff_test, std_diff_train, ultimo_envio.
    """
    input_width = retardo
    train,y_train= preprocesar_datos2(entrenamiento ,exclusiones,"Fecha",'Qe')
    val, y_val = preprocesar_datos2(validacion,exclusiones,"Fecha",'Qe')
    test,y_test = preprocesar_datos2(test,exclusiones,"Fecha",'Qe')

    last_row_value = y_train['Qe'].iloc[-1]
    y_train['Qe'] = y_train['Qe'].shift(-gap, fill_value=last_row_value)

    last_row_value = y_val['Qe'].iloc[-1]
    y_val['Qe'] = y_val['Qe'].shift(-gap, fill_value=last_row_value)

    last_row_value = y_test['Qe'].iloc[-1]
    # Desplazar las filas en gap y rellenar los huecos con el valor de la última fila
    y_test['Qe'] = y_test['Qe'].shift(-gap, fill_value=last_row_value)
    
    lista_columnas_retardar = train.columns.tolist()
            
    train = añadir_retardo_lista(train,train,lista_columnas_retardar,retardo)
    val = añadir_retardo_lista(val,val,lista_columnas_retardar,retardo)
    test = añadir_retardo_lista(test,test,lista_columnas_retardar,retardo)

    # NORMALIZADO
    tipo = "mean"
    train_norm,train_param = normalizar_datos(train,train,tipo)
    y_train_norm,y_train_param = normalizar_datos(y_train,y_train,tipo)

    val_norm,val_param = normalizar_datos(train,val,tipo)
    y_val_norm,y_val_param = normalizar_datos(y_train,y_val,tipo)

    test_norm,test_param = normalizar_datos(train,test,tipo)
    y_test_norm,y_test_param = normalizar_datos(y_train,y_test,tipo)

    # entreno el modelo
    y_train_norm = y_train_norm.values.ravel()
    model.fit(train_norm, y_train_norm)

    # Realizar predicciones en el conjunto de prueba
    predictions_test = model.predict(test_norm)
    predictions_test = predictions_test.reshape(-1, 1)
    predictions_train = model.predict(train_norm)
    predictions_train = predictions_train.reshape(-1,1)

    # desnormalizamos
    pred_denorm_test = y_test_param.inverse_transform(predictions_test) 
    pred_denorm_train = y_train_param.inverse_transform(predictions_train) 

    # Calcular la métrica de pérdida (en este caso, MSE)
    y_test_denorm = y_test.to_numpy().astype(float)
    y_test_denorm = y_test_denorm.reshape(y_test.shape[0],1)
    
    y_train_denorm = y_train.to_numpy().astype(float)
    y_train_denorm = y_train_denorm.reshape(y_train.shape[0],1)

    y_test_denorm_trimmed = y_test_denorm[(input_width+gap):]
    y_train_denorm_trimmed = y_train_denorm[(input_width+gap):]

    pred_denorm_test_trimmed = pred_denorm_test[(input_width+gap):]
    pred_denorm_train_trimmed = pred_denorm_train[(input_width+gap):]
    
    print("############# 1 ############",y_test_denorm_trimmed.shape,y_train_denorm_trimmed.shape,pred_denorm_test_trimmed.shape,
          pred_denorm_train_trimmed.shape)
    loss_test = mean_squared_error(y_test_denorm_trimmed, pred_denorm_test_trimmed)
    loss_train = mean_squared_error(y_train_denorm_trimmed, pred_denorm_train_trimmed)
    print("############# 2 ############")
    diff_test = y_test_denorm -  pred_denorm_test
    std_diff_test = np.std(diff_test)
    print("############# 3 ############")
    diff_train = y_train_denorm -  pred_denorm_train
    std_diff_train = np.std(diff_train)
    print("############# 4 ############")
    # Calculo de R2 
    pred_denorm_test = pred_denorm_test.reshape(-1, 1)
    pred_denorm_train= pred_denorm_train.reshape(-1, 1)
    print("############# 5 ############")
    # Calcula la correlación entre las dos matrices
    correlacion_test = np.corrcoef(y_test_denorm,pred_denorm_test,rowvar=False)[0][1]
    r2_test = correlacion_test ** 2
    correlacion_train = np.corrcoef(y_train_denorm,pred_denorm_train,rowvar=False)[0][1]
    r2_train = correlacion_train ** 2

    
    # Calculo del coeficiente de eficiencia Nash-Sutcliffe (NS)
    ns_test = hydroeval.nse(y_test_denorm_trimmed,pred_denorm_test_trimmed)
    
    ns_train = hydroeval.nse(y_train_denorm_trimmed,pred_denorm_train_trimmed)
    mensaje_ns_test= f"\nEl coeficiente de eficiencia Nash-Sutcliffe (NS) en test es {float(ns_test):.2f} \n"
    print(mensaje_ns_test)
    mensaje_ns_train= f"\nEl coeficiente de eficiencia Nash-Sutcliffe (NS) en train es {float(ns_train):.2f} \n"
    print(mensaje_ns_train)

    ultimo_envio = enviar_mensaje_con_espera(idchatconbot,mensaje_ns_test,ultimo_envio)
    ultimo_envio = enviar_mensaje_con_espera(idchatconbot,mensaje_ns_train,ultimo_envio)

    mensaje_iteracion = f"Iteración número {iter+1} de {len(unique_years)} para el modelo {model_name}, loss del modelo en test: {loss_test}, loss del modelo en train: {loss_train}, ns_test: {ns_test} y ns_train: {ns_train}"
    print(mensaje_iteracion)
    ultimo_envio = enviar_mensaje_con_espera(idchatconbot, mensaje_iteracion,ultimo_envio)
    return loss_test,loss_train,ns_test,ns_train,r2_test,r2_train,std_diff_test,std_diff_train,ultimo_envio

if __name__ == "__main__":
    print("Todas las librerías son cargadas correctamente")