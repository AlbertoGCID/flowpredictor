import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from split_delay_time import sliding_window
from typing import Dict,Union,Any
import hydroeval


def lstm_evaluate(lstm_model:Sequential,x_test_norm:np.ndarray,x_test:Union[pd.DataFrame,np.ndarray],
                  y_test:Union[pd.DataFrame,np.ndarray],scaler:MinMaxScaler,pred_config:Dict)->None:
  
    y_pred = lstm_model.predict(x_test_norm)
    y_pred_denorm = scaler.inverse_transform(y_pred)
    temporal,y_test = sliding_window(data_x=x_test,data_y=y_test,input_width=pred_config['input_width'],
                                     label_width=pred_config['label_width'],offset=pred_config['offset'])
    mse_LSTM = mean_squared_error(y_test, y_pred_denorm)
    correlacion = np.corrcoef(y_pred_denorm,y_test,rowvar=False)[0][1]
    r_cuadrado_LSTM = correlacion ** 2
    ns_LSTM = hydroeval.nse(y_test,y_pred_denorm)[0]
    diff_modelo_test = np.abs(y_pred_denorm-y_test)     
    std_LSTM = np.std(diff_modelo_test)
    print('\n\nResults of evaluate LSTM in eval dataset\n')
    print(f'NS test: {ns_LSTM:.3f}')
    print(f'MSE test: {mse_LSTM}')
    print(f'R2 test: {r_cuadrado_LSTM}')
    print(f'std test: {std_LSTM}\n')
    #plot_ = Graficas(ultimo_envio=time.time())
    #plot_.plot2feats(feat1=y_test,feat2=y_pred_denorm, pred_config=pred_config)

def rank_five(numpyarray: np.ndarray) -> np.ndarray:
    """
    Returns the indices of the top 5 elements in the input numpy array.

    Parameters:
        numpyarray (np.ndarray): Input NumPy array.

    Returns:
        np.ndarray: Indices of the top 5 elements.
    """
    if not isinstance(numpyarray, np.ndarray):
        raise TypeError('Input must be a numpy array')

    if len(numpyarray) < 5:
        return np.argsort(-numpyarray)

    return np.argsort(-numpyarray)[:5]

import numpy as np

def sensitivity_analysis(X: Union[pd.DataFrame, np.ndarray], 
                         y: Union[pd.Series, np.ndarray], 
                         model: Any) -> np.ndarray:
    """
    Performs sensitivity analysis for a Keras model by iteratively replacing each feature with zeros 
    and measuring the decrease in performance.

    Args:
        X (pandas DataFrame): The feature matrix.
        y (pandas Series): The target variable.
        model (Keras model): The trained Keras model to analyze.

    Returns:
        feature_importance_scores (numpy array): The feature importance scores.
    """
    # Get the baseline score
    baseline_loss = model.evaluate(X, y, verbose=0)[0]

    # Get the number of features
    num_features = X.shape[1]

    # Initialize an array to store the importance scores
    feature_importance_scores = np.zeros(num_features)

    # Loop through each feature
    for i in range(num_features):
        # Make a copy of the data with the i-th feature replaced with zeros
        X_removed = X.copy()
        X_removed[X_removed.columns[i]] = 0

        # Evaluate the loss of the model
        score = model.evaluate(X_removed, y, verbose=0)[0]

        # Calculate the relative importance of the feature
        feature_importance_scores[i] = ((score - baseline_loss) / baseline_loss) * 100

        # Reset the internal state of the model
        model.reset_states()

    return feature_importance_scores


def metricasconfusion(x: np.ndarray, y: np.ndarray, umbral: float) -> None:
    """Calculate binary classification metrics.

    Args:
        x (array-like): Predicted values.
        y (array-like): True values.
        umbral (float): Threshold value for binary classification.

    Returns:
        None
    """
    x = np.array(x)
    y = np.array(y)
    if len(x) != len(y):
        raise ValueError('x and y must be of the same length')
    binary_y = (y >= umbral).astype(int)
    binary_pred = (x >= umbral).astype(int)
    accuracy = accuracy_score(binary_y, binary_pred)
    if np.sum(binary_pred) == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = precision_score(binary_y, binary_pred)
        recall = recall_score(binary_y, binary_pred)
        f1 = f1_score(binary_y, binary_pred)
    print(f"Clasificación binaria con un umbral de {umbral}:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 score: {f1:.3f}")
    results = {
        'umbral': umbral,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    return results


if __name__ == "__main__":
  print("Todas las librerías son cargadas correctamente")