import numpy as np
import pandas as pd
from typing import Union,Tuple,List
from data_load import RainfallDataset
from dataclasses import dataclass


@dataclass
class DataSets:
    """
    Data class to store datasets for training, validation, and testing.

    Attributes:
        x_train (Union[pd.DataFrame, np.ndarray]): Features for training data.
        y_train (Union[pd.DataFrame, np.ndarray]): Labels for training data.
        x_val (Union[pd.DataFrame, np.ndarray]): Features for validation data.
        y_val (Union[pd.DataFrame, np.ndarray]): Labels for validation data.
        x_test (Union[pd.DataFrame, np.ndarray]): Features for testing data.
        y_test (Union[pd.DataFrame, np.ndarray]): Labels for testing data.

    Methods:
        __str__(): Returns a formatted string containing information about the datasets.

    Private Methods:
        _get_info_str(element): Helper method to get information string for a dataset element.

    """
    x_train: Union[pd.DataFrame,np.ndarray] = None
    y_train: Union[pd.DataFrame,np.ndarray] = None
    x_val: Union[pd.DataFrame,np.ndarray] = None
    y_val:Union[pd.DataFrame,np.ndarray] = None
    x_test: Union[pd.DataFrame,np.ndarray] = None
    y_test: Union[pd.DataFrame,np.ndarray] = None
    def __str__(self):
        info_str = ""
        info_str += "x_train:\n"
        info_str += self._get_info_str(self.x_train)
        info_str += "\n\n"

        info_str += "y_train:\n"
        info_str += self._get_info_str(self.y_train)
        info_str += "\n\n"

        info_str += "x_val:\n"
        info_str += self._get_info_str(self.x_val)
        info_str += "\n\n"

        info_str += "y_val:\n"
        info_str += self._get_info_str(self.y_val)
        info_str += "\n\n"

        info_str += "x_test:\n"
        info_str += self._get_info_str(self.x_test)
        info_str += "\n\n"

        info_str += "y_test:\n"
        info_str += self._get_info_str(self.y_test)
        info_str += "\n\n"

        return info_str

    def _get_info_str(self, element):
        if element is None:
            return "None"

        info_str = ""
        if isinstance(element, pd.DataFrame):
            info_str += f"Type: pd.DataFrame\n"
            info_str += f"Shape: {element.shape}\n"
            info_str += f"Columns: {', '.join(element.columns)}\n"
            info_str += f"Sample:\n{element.tail(2)}\n"
        elif isinstance(element, np.ndarray):
            info_str += f"Type: np.ndarray\n"
            info_str += f"Shape: {element.shape}\n"
            info_str += f"Sample:\n{element[:2]}\n"
        return info_str


def sliding_window(data_x:pd.DataFrame, data_y:pd.DataFrame, input_width:int=5, label_width:int=1, offset:int=1)->Tuple[np.ndarray,np.ndarray]:
    """
    Generate sliding window sequences from input features (data_x) and corresponding labels (data_y).

    Args:
        data_x (Union[pd.DataFrame, np.ndarray]): Input features.
        data_y (Union[pd.DataFrame, np.ndarray]): Corresponding labels.
        input_width (int, optional): Width of the input window. Default is 5.
        label_width (int, optional): Width of the output labels. Default is 1.
        offset (int, optional): Offset between input and output. Default is 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the input sequences (x) and corresponding labels (y).
    """
    if not isinstance(data_x, pd.DataFrame):
        data_x = pd.DataFrame(data_x)
    if not isinstance(data_y, pd.DataFrame):
        data_y = pd.DataFrame(data_y)

    x = []
    y = []

    for i in range(len(data_x)):
        if i + input_width + offset + label_width > len(data_x):
            pass
        else:
            _x = data_x.iloc[i:i + input_width, :].drop(columns=['date']) 
            _y = data_y.iloc[i + input_width + offset:i + input_width + offset + label_width, :].drop(columns=['date'])  
            
            x.append(_x.values.astype(np.float32))
            y.append(_y.values.astype(np.float32))
    x, y = np.array(x), np.array(y)
    if y.ndim > 2:
        y = np.squeeze(y, axis=2)
    return x, y

def delay_offset_add(dataset:RainfallDataset,pred_config:dict) -> DataSets:
    """
    Generate sliding window sequences for input and output based on the specified RainfallDataset and prediction configuration.

    Args:
        dataset (RainfallDataset): RainfallDataset instance containing normalized training, validation, and testing data.
        pred_config (dict): Dictionary containing prediction configuration parameters.

    Returns:
        DataSets: DataSets instance containing x_train, y_train, x_val, y_val, x_test, and y_test.
    """
    data = DataSets()
    if pred_config['output'] == 'outflow':
        data.x_train,data.y_train = sliding_window(data_x=dataset.train_data_norm,
                                                                 data_y=dataset.train_outflow_norm,
                                                                 input_width=pred_config['input_width'],
                                                                 label_width=pred_config['label_width'],
                                                                 offset=pred_config['offset'])
        data.x_val,data.y_val = sliding_window(data_x=dataset.val_data_norm,
                                                                 data_y=dataset.val_outflow_norm,
                                                                 input_width=pred_config['input_width'],
                                                                 label_width=pred_config['label_width'],
                                                                 offset=pred_config['offset'])
        data.x_test,data.y_test = sliding_window(data_x=dataset.test_data_norm,
                                                                 data_y=dataset.test_outflow_norm,
                                                                 input_width=pred_config['input_width'],
                                                                 label_width=pred_config['label_width'],
                                                                 offset=pred_config['offset'])
    else:
        data.x_train,data.y_train = sliding_window(data_x=dataset.train_data_norm,
                                                                 data_y=dataset.train_inflow_norm,
                                                                 input_width=pred_config['input_width'],
                                                                 label_width=pred_config['label_width'],
                                                                 offset=pred_config['offset'])
        data.x_val,data.y_val = sliding_window(data_x=dataset.val_data_norm,
                                                                 data_y=dataset.val_inflow_norm,
                                                                 input_width=pred_config['input_width'],
                                                                 label_width=pred_config['label_width'],
                                                                 offset=pred_config['offset'])
        data.x_test,data.y_test = sliding_window(data_x=dataset.test_data_norm,
                                                                 data_y=dataset.test_inflow_norm,
                                                                 input_width=pred_config['input_width'],
                                                                 label_width=pred_config['label_width'],
                                                                 offset=pred_config['offset'])
    return data



if __name__ == "__main__":
  print("Todas las librerías son cargadas correctamente")