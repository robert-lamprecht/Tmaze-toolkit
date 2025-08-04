import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    # Apply the band-pass filter to the normalized traces (except doors 5 and 6)
    filtered_traces = {}
    for key in normalized_traces.keys():
        if key == 'door5' or key == 'door6':
            # Skip filtering for doors 5 and 6
            filtered_traces[key] = normalized_traces[key]
        else:
            # Apply band-pass filter to other doors
            filtered_traces[key] = bandpass_filter(normalized_traces[key], lowcut, highcut, fs, order)

    # Apply a Gaussian filter to the filtered traces (except doors 5 and 6)
    for key in filtered_traces.keys():
        if key != 'door5' and key != 'door6':
            # Apply Gaussian filter only to doors that aren't 5 or 6
            filtered_traces[key] = gaussian_filter1d(filtered_traces[key], 5)

    # Calculate thresholds based on statistics for each door
    thresholds = {}

    for key in filtered_traces.keys():
        mean = np.mean(filtered_traces[key])
        std = np.std(filtered_traces[key])
        # Calculate mean and standard deviation of the trace
        if key == 'door5' or key == 'door6':
            n_std = 1.5 # This value can be adjusted to change sensitivity
        else:
            n_std = 2.5 # This value can be adjusted to change sensitivity
        #Set threshold as mean plus n standard deviations
        # Can adjust n to be more or less strict based on sensitivity needs
        
        thresholds[key] = mean + (n_std * std)

    if plot:
       
        
        numPlots = len(filtered_traces.keys())
        fig = make_subplots(
            rows=numPlots, 
            cols=1,
            subplot_titles=[f'{door} Trace with Threshold' for door in filtered_traces.keys()],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Plot each trace with its threshold
        for i, (door, trace) in enumerate(filtered_traces.items(), 1):
            # Plot the trace
            fig.add_trace(
                go.Scatter(
                    y=trace,
                    mode='lines',
                    name=f'{door} Trace',
                    line=dict(width=1)
                ),
                row=i, 
                col=1
            )
            
            # Plot the threshold line
            fig.add_trace(
                go.Scatter(
                    y=[thresholds[door]] * len(trace),
                    mode='lines',
                    name=f'{door} Threshold',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=i, 
                col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Door Traces with Thresholds',
            height=200 * numPlots,
            width=1200,
            showlegend=False,
            hovermode='x unified'
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Frame Number", row=numPlots, col=1)
        
        # Update y-axis labels
        for i in range(1, numPlots + 1):
            fig.update_yaxes(title_text="Motion", row=i, col=1)
        
        fig.show()

    for key in filtered_traces.keys():
        for i in range(len(filtered_traces[key])):
            if filtered_traces[key][i] > thresholds[key]:
                filtered_traces[key][i] = 1;
            else:
                filtered_traces[key][i] = 0;
    
    

    return filtered_traces
