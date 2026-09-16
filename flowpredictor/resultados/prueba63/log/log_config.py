"""Global logger: console plus one log file per level under ``tests<N>/logs*/``."""
import logging
import sys
from pathlib import Path
from types import TracebackType
from typing import Optional, Type, Union

# Custom EXCEPTION logging level
EXCEPTION_LEVEL = 45  # between ERROR (40) and CRITICAL (50)
logging.addLevelName(EXCEPTION_LEVEL, "EXCEPTION")
DEFAULT_LOG_LEVEL = logging.DEBUG

# Icon prefix per log level
level_icons = {
    "DEBUG": "🔎🔎🔎🔎  ",
    "INFO": "🖨️🖨️🖨️ℹ️  ",
    "WARNING": "⚠️⚠️⚠️⚠️     ",
    "ERROR": "❌💀🔥🚨    ",
    "CRITICAL": "💀💀💀💀  ",
    "EXCEPTION": "💥💥💥💥  "
}

def get_log_directory(numero_prueba: Union[int, str]) -> Path:
    """Create and return a fresh log directory (``logs``, ``logs2``, ...) for an experiment.

    Args:
        numero_prueba (Union[int, str]): Experiment number.

    Returns:
        Path: The new log directory (relative to the cwd).
    """
    base_dir = Path(f"flowpredictor/resultados/prueba{numero_prueba}/tests{numero_prueba}")
    log_dir = base_dir / "logs"
    count = 1
    while log_dir.exists():
        count += 1
        log_dir = base_dir / f"logs{count}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def get_next_log_filename(LOG_DIR: Path, log_level: str) -> Path:
    """Log file path of a level inside a log directory."""
    return LOG_DIR / f"log_{log_level.lower()}.log"

class EmojiFormatter(logging.Formatter):
    """Formatter that prefixes each message with the icon of its level."""

    def format(self, record: logging.LogRecord) -> str:
        icon = level_icons.get(record.levelname, "")
        if not record.msg.startswith(icon):
            record.msg = f"{icon} {record.msg}"
        return super().format(record)

# --- Placeholder initialization of the global logger ---
_logger = logging.getLogger("MiLogger")
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())

def init_logger(log_level: int = DEFAULT_LOG_LEVEL, numero_prueba: Optional[Union[int, str]] = None) -> logging.Logger:
    """Reconfigure the global logger (console + per-level files) and install an excepthook.

    Call it once from the entry point, when ``numero_prueba`` is known.

    Args:
        log_level (int): Logger level.
        numero_prueba (Optional[Union[int, str]]): Experiment number.

    Returns:
        logging.Logger: The global logger.

    Raises:
        ValueError: If ``numero_prueba`` is not given.
    """
    global _logger
    if numero_prueba is None:
        raise ValueError("'numero_prueba' must be specified to initialize the logger.")

    # Log directory of this experiment
    LOG_DIR = get_log_directory(numero_prueba)

    # Remove previous handlers (including the NullHandler)
    for handler in _logger.handlers[:]:
        _logger.removeHandler(handler)

    # Logger level
    _logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_format = "%(levelname)s: %(message)s (%(filename)s:%(lineno)d in %(funcName)s)"
    console_handler.setFormatter(EmojiFormatter(console_format))
    _logger.addHandler(console_handler)

    # One file handler per log level
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "EXCEPTION"]:
        log_file = get_next_log_filename(LOG_DIR, level)
        file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
        file_format = "%(asctime)s - %(levelname)s: %(message)s (%(filename)s:%(lineno)d in %(funcName)s)"
        file_handler.setFormatter(EmojiFormatter(file_format))
        # Handler level
        file_handler.setLevel(EXCEPTION_LEVEL if level == "EXCEPTION" else getattr(logging, level))
        _logger.addHandler(file_handler)

    # Quieter third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    def handle_exception(exc_type: Type[BaseException], exc_value: BaseException,
                         exc_traceback: Optional[TracebackType]) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        _logger.log(EXCEPTION_LEVEL, "Unhandled exception",
                    exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

    return _logger

def get_logger() -> logging.Logger:
    """Return the global logger."""
    return _logger
