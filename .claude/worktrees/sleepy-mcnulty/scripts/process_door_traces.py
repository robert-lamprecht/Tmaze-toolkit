import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tmaze_toolkit.processing.signal import process_door_traces

def main():
    # Set up argument parser for command line options
    parser = argparse.ArgumentParser(description="Process door motion traces with band-pass filter")
    parser.add_argument("--lowcut", type=float, default=0.1, help="Low cutoff frequency in Hz")
    parser.add_argument("--highcut", type=float, default=3.0, help="High cutoff frequency in Hz")
    parser.add_argument("--fs", type=float, default=30.0, help="Sampling frequency in Hz (video fps)")
    parser.add_argument("--order", type=int, default=4, help="Order of the filter")
    parser.add_argument("--plot", action="store_true", help="Plot the original and filtered traces")
    parser.add_argument("--output", "-o", help="Output directory for results", default=".")
    
    # Parse arguments from the command line
    args = parser.parse_args()
    
    # Use tkinter to open a file dialog for selecting the door traces file
    Tk().withdraw()  # Hide the root window
    traces_path = askopenfilename(
        initialdir='.', 
        title='Select door traces file',
        filetypes=[('Pickle files', '*.pkl')]
    )
    
    if not traces_path:
        print("No file selected. Exiting.")
        return
        
    print(f"Processing {traces_path}...")
    
    # Load the door traces
    try:
        with open(traces_path, 'rb') as f:
            door_traces = pickle.load(f)
    except Exception as e:
        print(f"Error loading door traces: {e}")
        return
        
    # Process the door traces with the band-pass filter
    filtered_traces = process_door_traces(
        door_traces, 
        lowcut=args.lowcut, 
        highcut=args.highcut, 
        fs=args.fs, 
        order=args.order
    )
    
    # Generate output file path
    base_name = os.path.basename(traces_path).split('.')[0]
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}_filtered.pkl")
    
    # Save the filtered traces
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(filtered_traces, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved filtered traces to {output_path}")
    except Exception as e:
        print(f"Error saving filtered traces: {e}")
    
    # Optionally plot the results
    if args.plot:
        fig, axs = plt.subplots(len(door_traces), 2, figsize=(15, 10), sharex=True)
        fig.suptitle(f'Door Traces: Band-pass Filter ({args.lowcut}-{args.highcut} Hz)')
        
        for i, (door, trace) in enumerate(door_traces.items()):
            # Plot original data
            axs[i, 0].plot(trace)
            axs[i, 0].set_title(f'{door} - Original')
            axs[i, 0].set_ylabel('Motion')
            
            # Plot filtered data
            axs[i, 1].plot(filtered_traces[door])
            axs[i, 1].set_title(f'{door} - Filtered')
        
        axs[-1, 0].set_xlabel('Frame Number')
        axs[-1, 1].set_xlabel('Frame Number')
        
        plt.tight_layout()
        plt.show()
        
        # Save the plot
        plot_path = os.path.join(output_dir, f"{base_name}_filter_comparison.png")
        fig.savefig(plot_path)
        print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
