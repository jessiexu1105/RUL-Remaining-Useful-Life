import logging
from pathlib import Path
from sklearn.externals import joblib
import sys

logger = logging.getLogger(__name__)

def save_model(model, filename, path: Path) -> None:
    """Saves the trained random forest model to the specified file path.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): The trained random forest model to save.
        save_path (Path): A Path object representing the file path where the model should be saved.
    """
    # Create the directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)
    try:
        # Set the full file path
        file_path = path / filename
        # Save the model
        joblib.dump(model, file_path)
        logger.info("Model saved to %s", path)
    except FileNotFoundError:
        logger.error("Could not find file %s to save model.", file_path)
        sys.exit(1)
    except IsADirectoryError:
        logger.error("Cannot save model to a directory. Please provide a valid file path.")
        sys.exit(1)

def save_figure(fig, filename, path:Path)->None:
    """
    Save a matplotlib figure to a specified path.

    Args:
        figure (matplotlib.figure.Figure): The figure to save.
        filename (str): The filename for the saved figure.
        path (str): The directory path to save the figure to.
    """
    # Create the directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)
    try:
        # Set the full file path
        file_path = path / filename
        fig.savefig(file_path)
        logger.info("Figure saved to %s", file_path)
    except FileNotFoundError:
        logger.error("Could not find file %s to save figure.", file_path)
        sys.exit(1)
    except IsADirectoryError:
        logger.error("Cannot save figure to a directory. Please provide a valid file path.")
        sys.exit(1)