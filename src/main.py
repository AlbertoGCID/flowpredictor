from data_load import RainfallDataset,PredictionDataset,load_keras_model,load_scaler
from AI_algorithms import lstm_train
from AI_results import lstm_evaluate
from split_delay_time import delay_offset_add
import tensorflow as tf
import numpy as np
import joblib
from datetime import timedelta

tf.get_logger().setLevel('ERROR')


def main_train(pred_config:dict()=None,param_config: dict()=None)-> None:
    print(f'\n\n{"*"*50} Loading Dataset {"*"*50}\n\n')
    DamRegisters = RainfallDataset()
    DamRegisters.load_folder(data_path=pred_config['folder_path'],temporal_column='date')
    DamRegisters.set_inflow(inflow_column=pred_config['inflow_name'])
    DamRegisters.set_outflow(outflow_column=pred_config['outflow_name'])
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
    lstm_model = lstm_train(x_train_lstm = sets.x_train, y_train_lstm= sets.y_train, 
                            x_val_lstm= sets.x_val, y_val_lstm= sets.y_val, 
                            param_config= param_config,pred_config=pred_config) 
    lstm_evaluate(lstm_model=lstm_model,x_test_norm=sets.x_test,x_test=x_test,y_test=y_test,scaler=scaler,pred_config=pred_config)

def preprocess(x_data):
    x = x_data

def main_pred_folder(pred_folder_config:dict()=None)->None:
    dataset = PredictionDataset()
    dataset.load_folder(data_path=pred_folder_config['folder_path'],temporal_column='date')
    dataset.set_inflow(inflow_column=pred_folder_config['inflow_name'])
    dataset.set_outflow(outflow_column=pred_folder_config['outflow_name'])
    print(f'\n\n{"*"*50} Predicting for this values {"*"*50}\n\n')
    print(f'Lengh prediction dataset is {len(dataset.data)} days')
    print(dataset.data.tail(int(pred_folder_config['input_width'])))
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
    last_date = dataset.data['date'].max()
    days_to_add = pred_folder_config['offset'] + 1


    # Calcular la nueva fecha sumando los días
    new_date_dt = last_date + timedelta(days=days_to_add)

    # Formatear la nueva fecha en el formato deseado (d-m-y)
    new_date_formatted = new_date_dt.strftime('%d-%m-%Y')

    print(f"\n\nInflow prediction for {new_date_formatted} is {y_pred_denorm[0][0]:.3f} m3/s\n\n")



if __name__ == '__main__':

    train_data = "datasets/train_folder/" 
    predict_data = "datasets/predict_folder/" 
    input_width = 7
    offset = 3
    inflow_name = 'input'
    outflow_name = 'output'
    target = 'output'

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
                'epochs' :10}
    
    pred_config ={'folder_path': train_data,
                'inflow_name':inflow_name,
                'outflow_name':outflow_name,
                'output':target,
                'input_width':input_width,
                'label_width':1,
                'offset':offset}
    
    main_train(pred_config=pred_config,param_config= param_config)

    
    model_path = f"models/{pred_config['output']}LSTM.keras"
    model = load_keras_model(model_path)
    x_scaler = load_scaler(scaler_path="scaler/x_train_scaler.pkl")
    inflow_scaler = load_scaler(scaler_path="scaler/inflow_train_scaler.pkl")
    outflow_scaler = load_scaler(scaler_path="scaler/outflow_train_scaler.pkl")
    pred_folder_config = {'folder_path': predict_data,
                'model' : model,
                'inflow_name': inflow_name,
                'outflow_name':outflow_name,
                'output':target,
                'x_scaler' : x_scaler,
                'inflow_scaler' : inflow_scaler,
                'outflow_scaler' : outflow_scaler,
                'input_width':input_width,
                'label_width':1,
                'offset':offset}

    main_pred_folder(pred_folder_config=pred_folder_config)



