import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut=0.1, highcut=3.0, fs=30.0, order=4):
    """
    Apply a band-pass filter to door trace data.
    
    Parameters:
    -----------
    data : dict or array-like
        Dictionary containing door traces or a single array of trace data.
    lowcut : float, optional
        Low cutoff frequency in Hz, default is 0.1 Hz.
    highcut : float, optional
        High cutoff frequency in Hz, default is 3.0 Hz.
    fs : float, optional
        Sampling frequency in Hz, default is 30.0 Hz (assuming 30 fps video).
    order : int, optional
        Order of the filter, default is 4.
    
    Returns:
    --------
    dict or array-like
        Filtered data in the same format as input.
    """
    # Calculate Nyquist frequency
    nyq = 0.5 * fs
    
    # Normalize cutoff frequencies
    low = lowcut / nyq
    high = highcut / nyq
    
    # Design the filter
    b, a = butter(order, [low, high], btype='band')
    
    if isinstance(data, dict):
        # If input is a dictionary, apply filter to each trace
        filtered_data = {}
        for key, trace in data.items():
            # Apply the filter using filtfilt which applies the filter twice,
            # once forward and once backward to avoid phase shifts
            filtered_data[key] = filtfilt(b, a, trace)
        return filtered_data
    else:
        # If input is an array, apply filter directly
        return filtfilt(b, a, data)

def process_door_traces(door_traces, lowcut=0.1, highcut=3.0, fs=30.0, order=4, n_std = 2.5):
    """
    Process door traces by applying a band-pass filter.
    
    Parameters:
    -----------
    door_traces : dict
        Dictionary containing door traces, where keys are door names and values are lists of motion values.
    lowcut : float, optional
        Low cutoff frequency in Hz, default is 0.1 Hz.
    highcut : float, optional
        High cutoff frequency in Hz, default is 3.0 Hz.
    fs : float, optional
        Sampling frequency in Hz, default is 30.0 Hz (assuming 30 fps video).
    order : int, optional
        Order of the filter, default is 4.
    n_std : float, optional
        Strength of Signal Detector, default motion detection is signal greater than 2.5 standard deviations from the mean. 
    
    Returns:
    --------
    dict
        Dictionary with processed door traces.
    """
    # First, normalize by subtracting the mean
    normalized_traces = {}
    for key in door_traces.keys():
        trace_array = np.array(door_traces[key])
        normalized_traces[key] = trace_array - np.mean(trace_array)
    
    # Apply the band-pass filter to the normalized traces
    filtered_traces = bandpass_filter(normalized_traces, lowcut, highcut, fs, order)

    # Calculate thresholds based on statistics for each door
    thresholds = {}

    for key in filtered_traces.keys():
        # Calculate mean and standard deviation of the trace
        mean = np.mean(filtered_traces[key])
        std = np.std(filtered_traces[key])
        #Set threshold as mean plus n standard deviations
        # Can adjust n to be more or less strict based on sensitivity needs
        thresholds[key] = mean + (n_std * std)
    
    for key in filtered_traces.keys():
        for i in range(len(filtered_traces[key])):
            if filtered_traces[key][i] > thresholds[key]:
                filtered_traces[key][i] = 1;
            else:
                filtered_traces[key][i] = 0;
    
    return filtered_traces
