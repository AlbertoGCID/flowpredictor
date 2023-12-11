import os
from data_load import RainfallDataset,PredictionDataset,load_keras_model,load_scaler
from typing import Union
from AI_algorithms import lstm_train
from AI_results import lstm_evaluate
from split_delay_time import delay_offset_add
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib

tf.get_logger().setLevel('ERROR')


def main_train(pred_config:dict()=None,param_config: dict()=None)-> None:
    print(f'\n\n{"*"*50} Loading Dataset {"*"*50}\n\n')
    DamRegisters = RainfallDataset()
    DamRegisters.load_folder(data_path=pred_config['folder_path'],temporal_column='date')
    DamRegisters.set_inflow(inflow_column=pred_config['inflow_name'])
    DamRegisters.set_outflow(outflow_column=pred_config['outflow_name'])
    #print(DamRegisters)
    if pred_config['output'] == "outflow":
        sets = delay_offset_add(dataset=DamRegisters,pred_config=pred_config)
        scaler = DamRegisters.outflow_scaler
        y_test = DamRegisters.test_outflow
    else:
        sets = delay_offset_add(dataset=DamRegisters,pred_config=pred_config)
        scaler = DamRegisters.inflow_scaler
        y_test = DamRegisters.test_inflow
    x_test = DamRegisters.test_data
    print(f'\n\n{"*"*50} Training LSTM {"*"*50}\n\n')
    filas_del_medio = DamRegisters.data.iloc[450: 499]

    # Guardar el subconjunto en un archivo CSV
    filas_del_medio.to_csv('x_prueba.csv', index=False)
    #print(sets)
    lstm_model = lstm_train(x_train_lstm = sets.x_train, y_train_lstm= sets.y_train, 
                            x_val_lstm= sets.x_val, y_val_lstm= sets.y_val, 
                            param_config= param_config,pred_config=pred_config) 
    lstm_evaluate(lstm_model=lstm_model,x_test_norm=sets.x_test,x_test=x_test,y_test=y_test,scaler=scaler,pred_config=pred_config)

def preprocess(x_data):
    x = x_data

def main_pred(model:Sequential=None,x_data_path:str=None,temporal_column_name:str=None,
              x_scaler:MinMaxScaler= None,y_scaler:MinMaxScaler= None,input_width:int=7)->None:
    dataset = PredictionDataset()
    dataset.load_data(data_path=x_data_path, temporal_column=temporal_column_name)
    dataset.data = dataset.data.iloc[-input_width:]
    print("€"*500)
    print(dataset.data.tail(10))
    if x_scaler is None:
        scaler_filename = 'scaler/x_train_scaler.pkl'
        x_scaler = joblib.load(scaler_filename)
    x = x_scaler.transform(dataset.data.drop(columns=['date']))
    y_pred = model.predict(np.expand_dims(x, axis=0))
    if y_scaler is None:
        scaler_filename = 'scaler/inflow_train_scaler.pkl'
        y_scaler = joblib.load(scaler_filename)
    y_pred_denorm=y_scaler.inverse_transform(y_pred)
    print(f"\n\nInflow prediction for indicated data is {y_pred_denorm[0][0]:.3f} m3/s\n\n")


def main_pred_folder(pred_folder_config:dict()=None)->None:
    dataset = PredictionDataset()
    dataset.load_folder(data_path=pred_folder_config['folder_path'],temporal_column='date')
    dataset.set_inflow(inflow_column=pred_folder_config['inflow_name'])
    dataset.set_outflow(outflow_column=pred_folder_config['outflow_name'])
    print(f'\n\n{"*"*50} Predicting {"*"*50}\n\n')
    print(dataset.data.tail(10))
    if pred_folder_config['x_scaler'] is None:
        scaler_filename = 'scaler/x_train_scaler.pkl'
        x_scaler = joblib.load(scaler_filename)
    else:
        x_scaler = pred_folder_config['x_scaler']
    x = x_scaler.transform(dataset.data.drop(columns=['date']))
    model = pred_folder_config['model'] # añadir aquí la parte de si no se encuentra que lo descargue
    y_pred = model.predict(np.expand_dims(x[-pred_folder_config['input_width']:, :], axis=0))
    if pred_folder_config['output'] == 'inflow':
        if pred_folder_config['inflow_scaler'] is None:
            scaler_filename = 'scaler/inflow_train_scaler.pkl'
            y_scaler = joblib.load(scaler_filename)
        else:
            y_scaler = pred_folder_config['inflow_scaler']
    else:
        if pred_folder_config['outflow_scaler'] is None:
            scaler_filename = 'scaler/outflow_train_scaler.pkl'
            y_scaler = joblib.load(scaler_filename)
        else:
            y_scaler = pred_folder_config['outflow_scaler']
    y_pred_denorm=y_scaler.inverse_transform(y_pred)
    print(f"\n\nInflow prediction for indicated data is {y_pred_denorm[0][0]:.3f} m3/s\n\n")



if __name__ == '__main__':

 
    param_config = {
                'num_layers': 1,
                'units': 24,
                'batchnorm': 1,
                'dropout': 1,
                'dropout_rate': 0.5,
                'seed': 42,
                'label_width': 1,
                'act': 'linear',
                'batch_size': 32,
                'epochs' :500}
    
    pred_config ={'folder_path': 'datasets/predict_folder/',
                'inflow_name':'CaudalEntrante_m3_s',
                'outflow_name':'CaudalAliviado_m3_s',
                'output':'inflow',
                'input_width':7,
                'label_width':1,
                'offset':0}
    
    main_train(pred_config=pred_config,param_config= param_config)

    input_width = 7
    model_path = f"models/{pred_config['output']}LSTM.keras"
    model = load_keras_model(model_path)
    x_data_path = "datasets/predict_folder/" # x_prueba.csv
    temporal_column_name='date'
    x_scaler = load_scaler(scaler_path="scaler/x_train_scaler.pkl")
    inflow_scaler = load_scaler(scaler_path="scaler/inflow_train_scaler.pkl")
    outflow_scaler = load_scaler(scaler_path="scaler/outflow_train_scaler.pkl")
    #main_pred(model=model,x_data_path=x_data_path,temporal_column_name=temporal_column_name,x_scaler=x_scaler,y_scaler=outflow_scaler,input_width=input_width)

    pred_folder_config = {'folder_path': 'datasets/predict_folder/',
                'model' : model,
                'inflow_name':'CaudalEntrante_m3_s',
                'outflow_name':'CaudalAliviado_m3_s',
                'output':'inflow',
                'x_scaler' : x_scaler,
                'inflow_scaler' : inflow_scaler,
                'outflow_scaler' : outflow_scaler,
                'input_width':7,
                'label_width':1,
                'offset':0}

    main_pred_folder(pred_folder_config=pred_folder_config)



### TERMINADO EL DATASET, HABRÍA QUE ELIMINAR EL CAUDAL ALIVIADERO Y TAL, CREAR UNO CON LAS SALIDAS ÚNICAMENTE PARA QUITARLAS DEL MODELO DE ENTRADA Y ASÍ SIMULAR EL CORRECTO