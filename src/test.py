import pytest
from main import train, pred_folder
from data_load import load_keras_model,load_scaler

# Fixture para proporcionar datos de prueba
@pytest.fixture
def pred_config():
    return {
        'folder_path': '../datasets/train_folder/',
        'inflow_name': 'input',
        'outflow_name': 'output',
        'output': 'input',
        'input_width': 7,
        'label_width': 1,
        'offset': 3
    }

@pytest.fixture
def pred_folder_config():
    return {
        'folder_path': '../datasets/predict_folder/',
        'model': load_keras_model(model_path='models/inputLSTM.keras'),  # Puedes proporcionar un modelo entrenado para las pruebas si es necesario
        'inflow_name': 'input',
        'outflow_name': 'output',
        'output': 'input',
        'x_scaler': load_scaler(scaler_path='scaler/x_train_scaler.pkl'),  # Puedes proporcionar un escalador para las pruebas si es necesario
        'inflow_scaler': load_scaler(scaler_path='scaler/inflow_train_scaler.pkl'),
        'outflow_scaler': load_scaler(scaler_path='scaler/outflow_train_scaler.pkl'),
        'input_width': 7,
        'label_width': 1,
        'offset': 3
    }

def test_train(pred_config):
    # Ejecutar la función de entrenamiento y verificar si no hay excepciones
    train(pred_config=pred_config)

def test_pred_folder(pred_folder_config):
    # Ejecutar la función de predicción y verificar si no hay excepciones
    pred_folder(pred_folder_config=pred_folder_config)
