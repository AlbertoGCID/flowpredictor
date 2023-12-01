import requests
import time
import telebot
from decouple import config
from matplotlib import pyplot as plt
from typing import Optional
from io import BytesIO



def conexion_verify() -> bool:
    """
    Verify internet connection by attempting to make a request to a known server (e.g., Google).

    Returns:
    - bool: True if the connection is successful, False otherwise.
    """
    try:
        response = requests.get("http://www.google.com", timeout=5)
        return response.status_code // 100 == 2
    except requests.ConnectionError:
        return False

def enviar_mensaje_con_espera(mensaje: str, idchatconbot: Optional[int] = None, bot_token: Optional[str] = None, ultimo_envio: float = 0, tiempo_espera: int = 3) -> float:
    """
    Send a message with a waiting period between consecutive messages.

    Args:
    - idchatconbot (int): The ID of the chat with the bot.
    - mensaje (str): The message to be sent.
    - ultimo_envio (float): The timestamp of the last message sent (default is 0).
    - tiempo_espera (int): The waiting time between messages in seconds (default is 3).

    Returns:
    - float: The updated timestamp of the last message sent.
    """
    # Create a TeleBot instance
    if bot_token is None:
        bot_token = config('TELEGRAM_BOT_TOKEN')

    bot = telebot.TeleBot(bot_token, parse_mode=None) 
    
    # Hard-coded ID for demonstration; you may want to use the provided idchatconbot parameter
    if idchatconbot is None:
        idchatconbot = config('idchatconbot')

    # Get the current time
    tiempo_actual = time.time()

    # Check for internet connection
    if not conexion_verify():
        print("No hay conexión a internet.")
        return time.time()

    # Check if less than tiempo_espera has passed since the last message
    transcurrido = tiempo_actual - ultimo_envio
    if transcurrido < tiempo_espera:
        time.sleep(tiempo_espera - transcurrido)

    # Send the message
    bot.send_message(idchatconbot, mensaje)

    # Update the timestamp of the last message sent
    return time.time()

def enviar_tabla_o_grafica(chat_id: Optional[int] = None, bot_token: Optional[str] = None, dataframe=None, ultimo_envio: float = 0, tiempo_espera: int = 3) -> float:
    """
    Send a table or chart to a Telegram chat.

    Args:
    - chat_id (Optional[int]): The ID of the chat to send the table or chart.
    - bot_token (Optional[str]): The Telegram bot token.
    - dataframe: The DataFrame containing the data.
    - ultimo_envio (float): The timestamp of the last message sent (default is 0).
    - tiempo_espera (int): The waiting time between messages in seconds (default is 3).

    Returns:
    - float: The updated timestamp of the last message sent.
    """
    # Check if bot_token is provided, otherwise read from environment variable
    if bot_token is None:
        bot_token = config('TELEGRAM_BOT_TOKEN')

    # Create a TeleBot instance
    bot = telebot.TeleBot(bot_token, parse_mode=None)

    # Check if chat_id is provided, otherwise read from environment variable
    if chat_id is None:
        chat_id = config('idchatconbot')

    tiempo_actual = time.time()

    if not conexion_verify():
        print("No hay conexión a internet.")
        return tiempo_actual

    # Check if less than tiempo_espera has passed since the last message
    transcurrido = tiempo_actual - ultimo_envio
    if transcurrido < tiempo_espera:
        time.sleep(tiempo_espera - transcurrido)

    # Convert the DataFrame to a table chart (optional)
    primeras_filas = dataframe.head(5)

    # Get the name of each column
    columnas = primeras_filas.columns.tolist()

    # Create a table chart
    plt.figure(figsize=(10, 6))
    tabla = plt.table(cellText=primeras_filas.values, colLabels=columnas, cellLoc='center', loc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 1.2)

    # Save the table chart to a buffer
    buffer = BytesIO()
    plt.axis('off')
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.5)
    buffer.seek(0)

    bot.send_photo(chat_id=chat_id, photo=buffer)
    buffer.close()

    return time.time()

if __name__ == "__main__":
    print("Todas las librerías son cargadas correctamente")