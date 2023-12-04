import matplotlib.pyplot as plt
import seaborn as sns
import telebot
import io
import os
import time
from src.core.utils.telebot_api import conexion_verify
bot = telebot.TeleBot("6114856166:AAGYqKXk1qSoupZZ9thLQOjT5QevfdL4aMA", parse_mode=None) 
idchatconbot = -807792928
import matplotlib.pyplot as plt
import numpy as np
import io
import time
import seaborn as sns
from typing import List, Tuple, Union,Any,Dict
import pandas as pd



class Graficas:
    def __init__(self, ultimo_envio: float):
        """
        Initializes the Graficas class.

        Parameters:
        - ultimo_envio (float): A timestamp representing the last time a plot was sent.
        """
        self.ultimo_envio = ultimo_envio

    def plot2feats(self,feat1:np.ndarray,feat2:np.ndarray,pred_config:Dict)-> None:
        x_indices = np.arange(len(feat1))
        # Crear la figura y los ejes
        fig, ax = plt.subplots(figsize=(20, 10))
        # Graficar los valores reales (y_test)
        ax.plot(x_indices, feat1, label='Real Value',  color='red',linestyle='-')

        # Graficar las predicciones del modelo LSTM (y_pred_lstm)
        ax.plot(x_indices, feat2 , label='LSTM prediction', color='green', linestyle='--')

        # Etiquetas de los ejes y título
        ax.set_xlabel('Day')
        ax.set_ylabel('Outflow value')
        ax.set_title(f"{pred_config['output']} Flow in test with {pred_config['offset']} day gap and {pred_config['input_width']} day info comparison")

        # Leyenda
        ax.legend()
        output_folder = 'evaluate_results'
        os.makedirs(output_folder, exist_ok=True)

        # Guardar la figura en la carpeta 'evaluate_results'
        plt.savefig(os.path.join(output_folder, "last_graphic.png"))
        # Mostrar la gráfica
        plt.show()

    def plot_dataframe(self, df: pd.DataFrame, x_column: str, y_column: str, ultimo_envio: float) -> float:
        """
        Plot the values of the y_column of the DataFrame against the values of the x_column.

        Parameters:
        - df (pd.DataFrame): DataFrame to plot.
        - x_column (str): Column name for x-axis values.
        - y_column (str): Column name for y-axis values.

        Returns:
        - float: The updated timestamp.
        """
        plt.plot(df[x_column], df[y_column])
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f"{y_column} vs {x_column}")
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        presente = time.time()
        if presente - ultimo_envio < 3:
            time.sleep(3 - (presente-ultimo_envio))
        if conexion_verify():
            bot.send_photo(idchatconbot,buffer)
            self.ultimo_envio = time.time()
        buffer.close()
        plt.show()
        return self.ultimo_envio
        
    def plot_histogram(self, df: pd.DataFrame, column: str, ultimo_envio: float) -> float:
        """
        Plot a histogram of the values in the specified column of the DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame to plot.
        - column (str): Column name for histogram values.

        Returns:
        - float: The updated timestamp.
        """
        plt.hist(df[column], bins=10)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {column}")
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        presente = time.time()
        if presente - ultimo_envio < 3:
            time.sleep(3 - (presente-ultimo_envio))
        if conexion_verify():
            bot.send_photo(idchatconbot,buffer)
            self.ultimo_envio = time.time()
        buffer.close()
        plt.show()
        return self.ultimo_envio

    def plot_3D(self, df: pd.DataFrame, x_column: str, y_column: str, z_column: str, ultimo_envio: float) -> float:
        """
        Plot a 3D graph of the values in the specified columns of the DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame to plot.
        - x_column (str): Column name for x-axis values.
        - y_column (str): Column name for y-axis values.
        - z_column (str): Column name for z-axis values.

        Returns:
        - float: The updated timestamp.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(df[x_column], df[y_column], df[z_column])
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)  

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        presente = time.time()
        if presente - ultimo_envio < 3:
            time.sleep(3 - (presente-ultimo_envio))
        if conexion_verify():
            bot.send_photo(idchatconbot,buffer)
            self.ultimo_envio = time.time()
        buffer.close()
        return self.ultimo_envio

    def plot_lines(self,data,ultimo_envio):
        y1 = [t[0] for t in data]
        y2 = [t[1] for t in data]
        x = range(len(data))

        plt.plot(x, y1, label='Predicciones')
        plt.plot(x, y2, label='Valores reales')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        presente = time.time()
        if presente - ultimo_envio < 3:
            time.sleep(3 - (presente-ultimo_envio))
        if conexion_verify():
            bot.send_photo(idchatconbot,buffer)
            self.ultimo_envio = time.time()
        buffer.close()
        plt.show()
        return self.ultimo_envio

    def plot_feature_importance(self, feature_importance_scores: np.ndarray, dataset: pd.DataFrame, ultimo_envio: float) -> float:
        """
        Plot the feature importance scores for a given dataset.

        Parameters:
        - feature_importance_scores (np.ndarray): Feature importance scores.
        - dataset (pd.DataFrame): Dataset containing the features.

        Returns:
        - float: The updated timestamp.
        """
        # Get the feature names from the dataset
        feature_names = dataset.columns.values

        # Sort the feature importance scores in descending order
        sorted_idx = feature_importance_scores.argsort()[::-1]
        sorted_scores = feature_importance_scores[sorted_idx]

        # Plot the feature importance scores in a bar chart
        plt.figure(figsize=(15, 8))
        plt.bar(range(len(sorted_scores)), sorted_scores)
        plt.xticks(range(len(sorted_scores)), feature_names[sorted_idx], rotation=270)
        plt.title("Sensitivity Analysis")
        plt.xlabel("Variables")
        plt.ylabel("Importance Score")
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        presente = time.time()
        if presente - ultimo_envio < 3:
            time.sleep(3 - (presente-ultimo_envio))
        if conexion_verify():
            bot.send_photo(idchatconbot,buffer)
            self.ultimo_envio = time.time()
        buffer.close()
        plt.show()
        return self.ultimo_envio



    def plot_boxplot(self, dist_naive: np.ndarray, dist_lstm: np.ndarray, ultimo_envio: float) -> float:
        """
        Plot a boxplot comparing the prediction error between a Naive model and an LSTM model.

        Parameters:
        - dist_naive (np.ndarray): Distribution of the prediction error for the Naive model.
        - dist_lstm (np.ndarray): Distribution of the prediction error for the LSTM model.

        Returns:
        - float: The updated timestamp.
        """
        data = [dist_naive, dist_lstm]
        labels = ['Modelo Naive', 'Modelo LSTM']

        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data, notch=True, palette='Set2')
        plt.xticks(ticks=[0, 1], labels=labels)
        plt.xlabel('Modelos')
        plt.ylabel('Error de Predicción')
        plt.title('Comparación de Error de Predicción entre Modelos')
        plt.grid(True, axis='y')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        presente = time.time()
        if presente - ultimo_envio < 3:
            time.sleep(3 - (presente-ultimo_envio))
        if conexion_verify():
            bot.send_photo(idchatconbot,buffer)
            self.ultimo_envio = time.time()
        buffer.close()
        plt.show()
        return self.ultimo_envio

    def plot_violinplot(self, dist_naive: np.ndarray, dist_lstm: np.ndarray, ultimo_envio: float) -> float:
        """
        Plot a violin plot comparing the prediction error between a Naive model and an LSTM model.

        Parameters:
        - dist_naive (np.ndarray): Distribution of the prediction error for the Naive model.
        - dist_lstm (np.ndarray): Distribution of the prediction error for the LSTM model.

        Returns:
        - float: The updated timestamp.
        """
        data = [dist_naive, dist_lstm]
        labels = ['Modelo Naive', 'Modelo LSTM']

        plt.figure(figsize=(8, 6))
        sns.violinplot(data=data, inner='quartile', palette='Pastel1')
        plt.xticks(ticks=[0, 1], labels=labels)
        plt.xlabel('Modelos')
        plt.ylabel('Error de Predicción')
        plt.title('Comparación de Error de Predicción entre Modelos')
        plt.grid(True, axis='y')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        presente = time.time()
        if presente - ultimo_envio < 3:
            time.sleep(3 - (presente-ultimo_envio))
        if conexion_verify():
            bot.send_photo(idchatconbot,buffer)
            self.ultimo_envio = time.time()
        buffer.close()
        plt.show()
        return self.ultimo_envio

    def plot_seaborn(self, dist1: np.ndarray, mean1: float, std1: float, dist2: np.ndarray, mean2: float, std2: float, ultimo_envio: float) -> float:
        """
        Plot a Seaborn histogram comparing two distributions with mean and standard deviation.

        Parameters:
        - dist1 (np.ndarray): Distribution 1.
        - mean1 (float): Mean of distribution 1.
        - std1 (float): Standard deviation of distribution 1.
        - dist2 (np.ndarray): Distribution 2.
        - mean2 (float): Mean of distribution 2.
        - std2 (float): Standard deviation of distribution 2.

        Returns:
        - float: The updated timestamp.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(dist1, bins=30, kde=True, color='blue', label='Distribución Naive')
        sns.histplot(dist2, bins=30, kde=True, color='orange', label='Distribución Modelo')
        plt.axvline(mean1, color='r', linestyle='dashed', linewidth=2, label='Media Naive')
        plt.axvline(mean2, color='g', linestyle='dashed', linewidth=2, label='Media Modelo')
        plt.axvspan(mean1 - std1, mean1 + std1, facecolor='r', alpha=0.3, label='Desv. típica Naive')
        plt.axvspan(mean2 - std2, mean2 + std2, facecolor='g', alpha=0.3, label='Desv. típica Modelo')
        plt.xlabel('Valores')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.title('Histograma de dos distribuciones con Media y Desv. típica')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        presente = time.time()
        if presente - ultimo_envio < 3:
            time.sleep(3 - (presente-ultimo_envio))
        if conexion_verify():
            bot.send_photo(idchatconbot,buffer)
            self.ultimo_envio = time.time()
        buffer.close()
        plt.show()
        return self.ultimo_envio
        
    def plot_matplot(self, dist1: np.ndarray, mean1: float, std1: float, dist2: np.ndarray, mean2: float, std2: float, ultimo_envio: float) -> float:
        """
        Plot a Matplotlib histogram comparing two distributions with mean and standard deviation.

        Parameters:
        - dist1 (np.ndarray): Distribution 1.
        - mean1 (float): Mean of distribution 1.
        - std1 (float): Standard deviation of distribution 1.
        - dist2 (np.ndarray): Distribution 2.
        - mean2 (float): Mean of distribution 2.
        - std2 (float): Standard deviation of distribution 2.

        Returns:
        - float: The updated timestamp.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(dist1, bins=30, alpha=0.5, label='Distribución Naive')
        plt.hist(dist2, bins=30, alpha=0.5, label='Distribución Modelo')
        plt.axvline(mean1, color='r', linestyle='dashed', linewidth=2, label='Media Naive')
        plt.axvline(mean2, color='g', linestyle='dashed', linewidth=2, label='Media Modelo')
        plt.axvspan(mean1 - std1, mean1 + std1, facecolor='r', alpha=0.3, label='Desv. típica Naive')
        plt.axvspan(mean2 - std2, mean2 + std2, facecolor='g', alpha=0.3, label='Desv. típica Modelo')
        plt.xlabel('Valores')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.title('Histograma de dos distribuciones con Media y Desv. típica')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        presente = time.time()
        if presente - ultimo_envio < 3:
            time.sleep(3 - (presente-ultimo_envio))
        if conexion_verify():
            bot.send_photo(idchatconbot,buffer)
            self.ultimo_envio = time.time()
        buffer.close()
        plt.show()
        return self.ultimo_envio



if __name__ == "__main__":
    print("Todas las librerías son cargadas correctamente")