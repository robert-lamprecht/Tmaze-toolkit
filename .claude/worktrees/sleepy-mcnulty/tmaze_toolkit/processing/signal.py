import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from tmaze_toolkit.visualization.plotDoorTraces import plotDoorTraces
from tmaze_toolkit.processing.extractTrialTimes import pad_movement

def gaussian_filter(data, sigma=5):
    for key in data.keys():
        data[key] = gaussian_filter1d(data[key], sigma)
    return data

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

def process_door_traces(door_traces, lowcut=0.1, highcut=3.0, fs=30.0, order=4, n_std = 2.5, plot=False):
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

    # Apply a Gaussian filter to the filtered traces
    filtered_traces = gaussian_filter(filtered_traces)

    # Calculate thresholds based on statistics for each door
    thresholds = {}

    for key in filtered_traces.keys():
        mean = np.mean(filtered_traces[key])
        std = np.std(filtered_traces[key])
        # Calculate mean and standard deviation of the trace
        if key == 'door5' or key == 'door6':
            n_std = .75 # This value can be adjusted to change sensitivity
        else:
            n_std = 2.5 # This value can be adjusted to change sensitivity
        #Set threshold as mean plus n standard deviations
        # Can adjust n to be more or less strict based on sensitivity needs
        
        thresholds[key] = mean + (n_std * std)

    if plot:
        fig, axs = plt.subplots(6, 1, figsize=(15, 10), sharex=True)
        fig.suptitle('Thresholds over Door Traces')

        for i, (door, trace) in enumerate(filtered_traces.items()):
            axs[i].plot(trace)
            axs[i].axhline(thresholds[door], color='r', linestyle='--', label='Threshold')
            axs[i].set_title(f'{door} Trace with Threshold')
            axs[i].set_ylabel('Motion')
            axs[i].legend()

        axs[-1].set_xlabel('Frame Number')  
    
    for key in filtered_traces.keys():
        for i in range(len(filtered_traces[key])):
            if filtered_traces[key][i] > thresholds[key]:
                filtered_traces[key][i] = 1;
            else:
                filtered_traces[key][i] = 0;
    
    

    return filtered_traces
