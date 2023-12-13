from data_load import RainfallDataset,PredictionDataset,load_keras_model,load_scaler
from AI_algorithms import lstm_train
from AI_results import lstm_evaluate
from split_delay_time import delay_offset_add
import tensorflow as tf
import numpy as np
import joblib
from datetime import timedelta
import argparse

tf.get_logger().setLevel('ERROR')


def train(pred_config:dict()=None)-> None:
    """
    Train an LSTM model using the specified prediction configuration.

    Parameters:
    - pred_config (dict): A dictionary containing configuration parameters for the prediction.
        - 'folder_path' (str): Path to the folder containing training data.
        - 'inflow_name' (str): Name of the inflow column in the dataset.
        - 'outflow_name' (str): Name of the outflow column in the dataset.
        - 'output' (str): Target variable for prediction ('inflow' or 'outflow').
        - 'input_width' (int): Number of context days.
        - 'label_width' (int): Number of days to be predicted.
        - 'offset' (int): Number of gap between last input day and prediction.

    Returns:
    None
    """
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

def pred_folder(pred_folder_config:dict()=None)->None:
    """
    Perform predictions using a trained model on a specified folder of input data.

    Parameters:
    - pred_folder_config (dict): A dictionary containing configuration parameters for the prediction.
        - 'folder_path' (str): Path to the folder containing prediction data.
        - 'inflow_name' (str): Name of the inflow column in the dataset.
        - 'outflow_name' (str): Name of the outflow column in the dataset.
        - 'output' (str): Target variable for prediction ('inflow' or 'outflow').
        - 'model' (Keras model): Trained model for prediction.
        - 'x_scaler' (Scaler object): Scaler for input data normalization.
        - 'inflow_scaler' (Scaler object): Scaler for inflow column normalization.
        - 'outflow_scaler' (Scaler object): Scaler for outflow column normalization.
        - 'input_width' (int): Number of context days.
        - 'label_width' (int): Number of days to be predicted.
        - 'offset' (int): Number of gap between last input day and prediction.

    Returns:
    None
    """
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
    new_date_dt = last_date + timedelta(days=days_to_add)
    new_date_formatted = new_date_dt.strftime('%d-%m-%Y')

    print(f"\n\nInflow prediction for {new_date_formatted} is {y_pred_denorm[0][0]:.3f} m3/s\n\n")



def main():
    """
    Main function for training and prediction script.

    Parses command-line arguments, including paths to datasets, model, and scalers.
    Executes training or prediction based on the provided arguments.

    Args:
        --train_folder (str): Path to the training dataset.
        --predict_folder (str): Path to the prediction dataset.
        --input_width (int): Width of the input window.
        --offset (int): Offset between input and output.
        --inflow_name (str): Name of the input column for inflow.
        --outflow_name (str): Name of the output column for outflow.
        --target (str): Name of the target column.
        --predict_only (bool): Run prediction only.
        --model_path (str): Path to the trained model (optional for prediction).
        --x_scaler_path (str): Path to the x_scaler (optional for prediction).
        --inflow_scaler_path (str): Path to the inflow_scaler (optional for prediction).
        --outflow_scaler_path (str): Path to the outflow_scaler (optional for prediction).

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Training and Prediction Script')
    parser.add_argument('--train_folder', default='datasets/train_folder/', help='Path to the training dataset')
    parser.add_argument('--predict_folder', default='datasets/predict_folder/', help='Path to the prediction dataset')
    parser.add_argument('--input_width', type=int, default=7, help='Width of the input window')
    parser.add_argument('--offset', type=int, default=3, help='Offset between input and output')
    parser.add_argument('--inflow_name', default='input', help='Name of the input column')
    parser.add_argument('--outflow_name', default='output', help='Name of the output column')
    parser.add_argument('--target', default='input', help='Name of the target column')
    parser.add_argument('--predict_only', action='store_true', help='Run prediction only')
    parser.add_argument('--model_path', default='models/inflowLSTM.keras', help='Path to the trained model (optional for prediction)')
    parser.add_argument('--x_scaler_path', default='scaler/x_train_scaler.pkl', help='Path to the x_scaler (optional for prediction)')
    parser.add_argument('--inflow_scaler_path', default='scaler/inflow_train_scaler.pkl', help='Path to the inflow_scaler (optional for prediction)')
    parser.add_argument('--outflow_scaler_path', default='scaler/outflow_train_scaler.pkl', help='Path to the outflow_scaler (optional for prediction)')

    args = parser.parse_args()

    if not args.predict_only:
        pred_config = {
            'folder_path': args.train_folder,
            'inflow_name': args.inflow_name,
            'outflow_name': args.outflow_name,
            'output': args.target,
            'input_width': args.input_width,
            'label_width': 1,
            'offset': args.offset
        }
        print(f'\n******************* Training with following parameters **************************\n')
        print(pred_config)
        train(pred_config=pred_config)
        if args.target == 'input':
            model = model = load_keras_model(model_path='models/inflowLSTM.keras')
        else:
            model = model = load_keras_model(model_path='models/outflowLSTM.keras')
        x_scaler = load_scaler(scaler_path='scaler/x_train_scaler.pkl')
        inflow_scaler = load_scaler(scaler_path='scaler/inflow_train_scaler.pkl')
        outflow_scaler = load_scaler(scaler_path='scaler/outflow_train_scaler.pkl')
        
    else:
        model = load_keras_model(model_path=args.model_path)
        x_scaler = load_scaler(scaler_path=args.x_scaler_path)
        inflow_scaler = load_scaler(scaler_path=args.inflow_scaler_path)
        outflow_scaler = load_scaler(scaler_path=args.outflow_scaler_path)

    pred_folder_config = {
        'folder_path': args.predict_folder,
        'model': model,
        'inflow_name': args.inflow_name,
        'outflow_name': args.outflow_name,
        'output': args.target,
        'x_scaler': x_scaler,
        'inflow_scaler': inflow_scaler,
        'outflow_scaler': outflow_scaler,
        'input_width': args.input_width,
        'label_width': 1,
        'offset': args.offset
    }
    print(f'\n******************* Predicting with following parameters **************************\n')
    print(pred_folder_config)
    pred_folder(pred_folder_config=pred_folder_config)

if __name__ == '__main__':

    main()



