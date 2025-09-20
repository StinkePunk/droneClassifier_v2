import numpy as np
from scipy.signal import correlate

def find_best_offset(input_signal, output_signal):
    """
    Find the best offset between input and output signals using cross-correlation.

    Parameters:
    input_signal (array): The input signal.
    output_signal (array): The output signal.

    Returns:
    int: The best offset.
    """
    correlation = correlate(output_signal, input_signal, mode='full')
    lag = np.argmax(np.abs(correlation)) - len(input_signal) + 1
    return lag
