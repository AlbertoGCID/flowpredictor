import pandas as pd
import os
import numpy as np
from typing import Tuple




def transform_csv(path:str,output_name)->None:
    columns_name = ["year","month","day",output_name]
    dataset = pd.read_csv(path, sep="   ")
    dataset.columns = columns_name
    # Asegúrate de que las columnas sean de tipo string
    dataset['year'] =  dataset['year'].astype(int).astype(str)
    dataset['month'] = dataset['month'].astype(int).astype(str)
    dataset['day'] = dataset['day'].astype(int).astype(str)
    # Combina las columnas en una nueva columna 'fecha'
    dataset['date'] = dataset['year'] + '-' + dataset['month'] + '-' + dataset['day']
    columns = ['date'] + [col for col in dataset.columns if col != 'date']
    dataset = dataset[columns]
    # Elimina las columnas originales
    dataset = dataset.drop(columns=['year', 'month', 'day'])
    filename = output_name +'.csv'
    save_path = os.path.join('datasets', filename)
    dataset.to_csv(save_path, index=False)
    print(dataset.head(50))


def sliding_window(data_x:pd.DataFrame, data_y:pd.DataFrame, input_width:int=5, label_width:int=1, offset:int=1)->Tuple[np.ndarray,np.ndarray]:
    # Verifica y convierte data_x a DataFrame si no lo es
    if not isinstance(data_x, pd.DataFrame):
        data_x = pd.DataFrame(data_x)

    # Verifica y convierte data_y a DataFrame si no lo es
    if not isinstance(data_y, pd.DataFrame):
        data_y = pd.DataFrame(data_y)

    x = []
    y = []

    for i in range(len(data_x)):
        if i + input_width + offset + label_width > len(data_x):
            pass
        else:
            _x = data_x.iloc[i:i + input_width, :]  # Utiliza iloc para seleccionar por índices numéricos
            _y = data_y.iloc[i + input_width + offset:i + input_width + offset + label_width, :]
            x.append(_x.values)  # Convierte a valores de numpy para mantener la consistencia
            y.append(_y.values)

    x, y = np.array(x), np.array(y)
    return x, y



if __name__ == "__main__":
    # Ejemplo de uso de la clase RainfallDataset

    path_dataset = 'datasets'
    pd1_path = os.path.join('datasets','Precipitacion_Diaria_3dias')
    output_name = 'pred3day'

    #transform_csv(pd1_path,output_name)
    data_x = pd.DataFrame({'col1': range(1, 21), 'col2': range(21, 41)})  # Ejemplo de 20 filas y 2 columnas
    data_y = pd.DataFrame({'label_col': range(1, 21)})  # Ejemplo de datos de etiqueta

    # Crear una instancia de TuClase (reemplaza 'TuClase' con el nombre de tu clase)
    # y luego llama a la función sliding_window
    # con los parámetros apropiados
    # (asegúrate de que input_width, label_width y offset sean adecuados para tu caso de uso)
    x, y = sliding_window(data_x, data_y, input_width=7, label_width=1, offset=1)

    # Imprimir los resultados para verificar
    print(len(x))
    print(len(y))
    for i in range(len(x)):
        print(f"x: {x[i]}, y: {y[i]}")