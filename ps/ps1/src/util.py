import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from collections.abc import Callable

def add_intercept(x: np.ndarray) -> np.ndarray:
    """Add intercept to matrix x.
    
    Args:
        x: 2D NumPy array.

    Returns:
        Augmented matrix (1 | x).
    """
    return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)

def load_dataset(csv_path: str, label_col: str = 'y', add_intercept: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from a csv file.

    Args:
        csv_path: Path to CSV file containing dataset.
        label_col: Name of column to use as labels(should be 'y' or 'l')
        add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: NumPy array of x-values (inputs).
        ys: NumPy array of y-values (labels).
    """

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Validate label_col argument
    allowed_label_col = ['y', 'l']
    if label_col not in allowed_label_col:
        raise ValueError("Invalue label_col: {} (expected {})".format(label_col, allowed_label_col))
    
    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        add_intercept_fn(inputs)

    return inputs, labels

def derivative(f: Callable[[float], float], x: float) -> float:
    return (f(x+1e-5)-f(x))/(1e-5)