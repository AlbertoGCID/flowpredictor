import numpy as np
import pandas as pd
from typing import Union,Tuple,List
from sklearn.model_selection import train_test_split



def split_data(data, test_ratio: float, val_ratio: float) -> Tuple:
    """
    Function to split the dataset into training, validation, and test sets.

    Parameters:
    - data: The input dataset to be split.
    - test_ratio (float): The ratio of the dataset to be used for testing.
    - val_ratio (float): The ratio of the dataset to be used for validation.

    Returns:
    Tuple: A tuple containing three sets - training set, validation set, and test set.
    """
    x_train_data, x_test = train_test_split(data, test_size=test_ratio)
    x_train, x_val = train_test_split(x_train_data, test_size=val_ratio)

    return x_train, x_val, x_test

def split_dataset(ds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[int]]:
    """
    Function to split a dataset into training, validation, and test sets.

    Parameters:
    - ds (pd.DataFrame): The input dataset to be split.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[int]]: A tuple containing the training set,
    validation set, test set, and a list of starting indices for each set.
    """
    n = len(ds)
    
    # Define the sizes of the train, validation, and test sets
    train_size = int(0.6 * n)
    val_size = int(0.2 * n)
    
    # Set the starting index for each set
    train_start = 0
    val_start = train_start + train_size
    test_start = val_start + val_size
    
    # Split the DataFrame into train, validation, and test sets using iloc
    x_train_df = ds.iloc[train_start:val_start, :]
    x_val_df = ds.iloc[val_start:test_start, :]
    x_test_df = ds.iloc[test_start:, :]
    
    return x_train_df, x_val_df, x_test_df, [train_start, val_start, test_start]

def split_train_test_by_date(dataset: pd.DataFrame, date: Union[str, pd.Timestamp]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to split a dataset into training and testing sets based on a specified date.

    Parameters:
    - dataset (pd.DataFrame): The input dataset to be split.
    - date (Union[str, pd.Timestamp]): The date used for splitting the dataset.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training set and testing set.
    """
    if dataset.empty:
        raise ValueError('Dataset is empty')
    
    try:
        if not isinstance(dataset['Fecha'], pd.Timestamp):
            dataset['Fecha'] = pd.to_datetime(dataset['Fecha'])
    except ValueError:
        print('Error: Fecha column cannot be converted to datetime format')
        return None, None
    
    fecha_fin1 = pd.to_datetime(str(date) + '-01-01 00:00:00')
    fecha_inicio2 = pd.to_datetime(str(date) + '-01-02 00:00:00')
    
    fecha_train = (dataset['Fecha'] <= fecha_fin1)
    fecha_test = (dataset['Fecha'] >= fecha_inicio2) 
    
    dataset_train = dataset.loc[fecha_train]
    dataset_test = dataset.loc[fecha_test]
    
    return dataset_train, dataset_test


def train_val_by_column_balance(dataset: pd.DataFrame, column: str, threshold: Union[int, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to split a dataset into training and validation sets by balancing a specified column.

    Parameters:
    - dataset (pd.DataFrame): The input dataset to be split.
    - column (str): The column used for balancing the dataset.
    - threshold (Union[int, float]): The threshold value for balancing the dataset based on the specified column.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training set and validation set.
    """
    seed = 123
    
    if column not in dataset.columns:
        raise ValueError('Column not found in the dataset')
    
    # Create training and validation sets by selecting the size based on the number of positives in a specific column
    # Using a threshold to balance the column in both datasets
    pos_samples = dataset[dataset[column] > threshold]
    neg_samples = dataset[dataset[column] <= threshold].sample(n=len(pos_samples), random_state=seed)
    
    desired_sample_size_train = int(len(pos_samples) * 0.8)
    
    neg_train = neg_samples.sample(n=desired_sample_size_train, random_state=seed)
    pos_train = pos_samples.sample(n=desired_sample_size_train, random_state=seed)
    
    neg_val = neg_samples.drop(neg_train.index)
    pos_val = pos_samples.drop(pos_train.index)
    
    # Concatenate the balanced samples for each subset
    df_train = pd.concat([neg_train, pos_train])
    df_val = pd.concat([neg_val, pos_val])
    
    return df_train, df_val


def procesado_de_tiempo(datos:pd.DataFrame,nombrecolumnatemporal:str)->pd.DataFrame:
    """Función que captura la columna temporal y devuelve variables periódicas en función de la fecha"""
    date_timefin = pd.to_datetime(datos.pop(nombrecolumnatemporal), format='%Y-%m-%d')
    timestamp_s = date_timefin.map(pd.Timestamp.timestamp) # esto lo pasa a segundos
    dia = 24*3600 # calculo los segundos
    año = dia*365
    semana = dia*7
    mes = dia*31
    trimestre = mes*3
    semestre = trimestre *2
    ts_sem = (timestamp_s / semana) * 2 * np.pi 
    ts_mes = (timestamp_s / mes) * 2 * np.pi 
    ts_tri = (timestamp_s / trimestre) * 2 * np.pi 
    ts_semes = (timestamp_s / semestre) * 2 * np.pi 
    ts_año = (timestamp_s / año) * 2 * np.pi
    datos['sem_sin'] = np.sin(ts_sem)
    datos['mes_sin'] = np.sin(ts_mes)
    datos['trim_sin'] = np.sin(ts_tri)
    datos['semes_sin'] = np.sin(ts_semes)
    datos['año_sin'] = np.sin(ts_año)
    return datos

def sliding_window(data:pd.DataFrame, labels:str, input_width:int, label_width:int=1, offset:int=1)->pd.DataFrame:
    x = []
    y = []

    for i in range(len(data)):
      if i+input_width + offset + label_width >= len(data) and i+input_width+ offset <= len(data):
        _x = data[i:i+input_width]
        _y = labels[i + input_width + offset :]
      elif i+input_width+ offset >= len (data):
        pass
      else:
        _x = data[i:i+input_width]
        _y = labels[i + input_width + offset : i + input_width + offset +label_width]
        x.append(_x)
        y.append(_y)

    x, y = np.array(x),np.array(y)
    # quitar esto
    if len(x.shape) == 2:
        x = x[:,:,np.newaxis]
        

    if len(y.shape) == 2:
        y = y[:,:,np.newaxis]
    
    return x, y

def sliding_window2(data:pd.DataFrame, labels:str, input_width:int, label_width:int=1, offset:int=1)->pd.DataFrame:
    x = []
    y = []

    for i in range(len(data)):
      if i+input_width + offset + label_width >= len(data) and i+input_width+ offset <= len(data):
        _x = data[i:i+input_width]
        _y = labels[i + input_width + offset :]
      elif i+input_width+ offset >= len (data):
        pass
      else:
        _x = data[i:i+input_width]
        _y = labels[i + input_width + offset : i + input_width + offset +label_width]
        x.append(_x)
        y.append(_y)

    x, y = np.array(x),np.array(y)

    if len(x.shape) == 2:
        x = x[:,:,np.newaxis]
        

    if len(y.shape) == 2:
        y = y[:,:,np.newaxis]
    
    return x, y


def extraer_x_y(df:pd.DataFrame,col_salida:str,mezclar:bool=False)-> pd.DataFrame:
    dataset = df.copy()
    if mezclar:
        dataset = dataset.sample(frac=1).reset_index(drop=True)
    if col_salida in dataset.columns:
        y = dataset[col_salida]
        x = dataset.drop(col_salida, axis=1)
        return (x, y)
    else:
        raise KeyError(f'{col_salida} no se encuentra en el dataset')
    
def añadir_retardo_lista2(ds_nuevo:pd.DataFrame,ds_fuente:pd.DataFrame,columnas:list,retardo:int=0)->pd.DataFrame:
    ds_ret_nuevo = ds_nuevo.copy()
    ds_ret_fuente = ds_fuente.copy()
    if retardo <= 0:
        raise ValueError('retardo must be a positive integer')
    for columna in columnas:
        for i in range(1,retardo):
            try:
                ds_ret_nuevo[columna+"menos"+str(i)] = ds_ret_fuente[columna].shift(i)
            except Exception as e:
                print(f"Error while shifting column {columna}: {e}")
    ds_ret_nuevo = ds_ret_nuevo.fillna(0)
    return ds_ret_nuevo


def añadir_retardo_lista(ds_nuevo:pd.DataFrame,ds_fuente:pd.DataFrame,columnas:list,retardo:int=0)->pd.DataFrame:
    ds_ret_nuevo = ds_nuevo.copy()
    ds_ret_fuente = ds_fuente.copy()
    if retardo <= 0:
        raise ValueError('retardo must be a positive integer')

    # Crear una lista para almacenar las columnas retardadas
    columnas_retardadas = []

    for columna in columnas:
        # Crear las columnas retardadas y agregarlas a la lista
        for i in range(1, retardo):
            try:
                nueva_columna = ds_ret_fuente[columna].shift(i)
                nueva_columna.name = f"{columna}menos{i}"
                columnas_retardadas.append(nueva_columna)
            except Exception as e:
                print(f"Error while shifting column {columna}: {e}")

    # Concatenar las columnas retardadas al DataFrame nuevo
    ds_ret_nuevo = pd.concat([ds_ret_nuevo] + columnas_retardadas, axis=1)
    ds_ret_nuevo = ds_ret_nuevo.fillna(0)
    return ds_ret_nuevo

def añadir_retardo(ds_nuevo:pd.DataFrame,ds_fuente:pd.DataFrame,columna:str,retardo:int=0)-> pd.DataFrame:
    if retardo <= 0:
        raise ValueError('retardo must be a positive integer')
    for i in range(1,retardo):
        try:
            ds_nuevo[columna+"menos"+str(i)] = ds_fuente[columna].shift(i)
        except Exception as e:
            print(f"Error while shifting column {columna}: {e}")
    ds_nuevo = ds_nuevo.fillna(0)
    return ds_nuevo

if __name__ == "__main__":
  print("Todas las librerías son cargadas correctamente")