import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalizar_datos(dataset_fit = pd.DataFrame,dataset_transform= pd.DataFrame,tipo="min"):
    '''Normalize the given dataset using MinMaxScaler.

    Args:
        dataset_fit (pandas.DataFrame): The dataset to fit the scaler.
        dataset_transform (pandas.DataFrame): The dataset to transform using the fitted scaler.

    Returns:
        pandas.DataFrame: The transformed dataset.
        MinMaxScaler: The fitted scaler.
    '''
    if not isinstance(dataset_fit, pd.DataFrame):
        raise TypeError('dataset_fit must be a pandas DataFrame')
    if not isinstance(dataset_transform, pd.DataFrame):
        raise TypeError('dataset_transform must be a pandas DataFrame')
    try:
        if tipo == "mean":
            print("Aplicando normalización por media y desviación típica")
            normalizado = StandardScaler()
        else:
            print("Aplicando normalización por máximo y mínimo")
            normalizado = MinMaxScaler()
        normalizado.fit(dataset_fit)
        ds_transformed = normalizado.transform(dataset_transform)
        ds2_transformed_df = pd.DataFrame(ds_transformed,columns=dataset_transform.columns)
        return ds2_transformed_df,normalizado
    except (TypeError, ValueError) as e:
        print(f'Error: {e}')










if __name__ == "__main__":
  print("Todas las librerías son cargadas correctamente")