import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from tmaze_toolkit.processing.signal import process_door_traces
from tmaze_toolkit.data.openFunctions import openDoorTracesPkl

def plot_door_traces_with_trial_times(dat, trials_df):
    """
    Plot thresholded door traces with trial start and end times as vertical lines
    
    Parameters:
    -----------
    dat : dict
        Door traces data loaded from pkl file
    trials_df : pandas.DataFrame
        DataFrame containing trial information with 'start', 'stop', and 'trial_number' columns
    """
    # Process door traces to get thresholded data
    thresholded_dat = process_door_traces(dat, plot=False)
    
    # Create subplots for each door trace
    num_doors = len(thresholded_dat.keys())
    fig = make_subplots(
        rows=num_doors, 
        cols=1,
        subplot_titles=[f'{door} Trace with Trial Times' for door in thresholded_dat.keys()],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # Plot each door trace
    for i, (door, trace) in enumerate(thresholded_dat.items(), 1):
        # Plot the thresholded trace
        fig.add_trace(
            go.Scatter(
                y=trace,
                mode='lines',
                name=f'{door} Trace',
                line=dict(width=1, color='blue')
            ),
            row=i, 
            col=1
        )
        
        # Add vertical lines for trial starts (green)
        for _, trial in trials_df.iterrows():
            if not pd.isna(trial['start']):
                fig.add_vline(
                    x=trial['start'], 
                    line_dash="solid", 
                    line_color="green",
                    line_width=2,
                    annotation_text=f"Trial {trial['trial_number']} Start",
                    annotation_position="top right"
                )
        
        # Add vertical lines for trial ends (red)
        for _, trial in trials_df.iterrows():
            if not pd.isna(trial['stop']):
                fig.add_vline(
                    x=trial['stop'], 
                    line_dash="solid", 
                    line_color="red",
                    line_width=2,
                    annotation_text=f"Trial {trial['trial_number']} End",
                    annotation_position="bottom right"
                )
    
    # Update layout
    fig.update_layout(
        title='Door Traces with Trial Start/End Times',
        height=200 * num_doors,
        width=1200,
        showlegend=False,
        hovermode='x unified'
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Frame Number", row=num_doors, col=1)
    
    # Update y-axis labels
    for i in range(1, num_doors + 1):
        fig.update_yaxes(title_text="Motion (Thresholded)", row=i, col=1)
    
    fig.show()
    return fig

def plot_door_traces_with_trial_times_simplified(dat, trials_df):
    """
    Simplified version that plots all door traces in one subplot with trial times
    """
    # Process door traces to get thresholded data
    thresholded_dat = process_door_traces(dat, plot=False)
    
    # Create single plot
    fig = go.Figure()
    
    # Plot each door trace
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for i, (door, trace) in enumerate(thresholded_dat.items()):
        fig.add_trace(
            go.Scatter(
                y=trace,
                mode='lines',
                name=f'{door}',
                line=dict(width=1, color=colors[i % len(colors)])
            )
        )
    
    # Add vertical lines for trial starts (green)
    for _, trial in trials_df.iterrows():
        if not pd.isna(trial['start']):
            fig.add_vline(
                x=trial['start'], 
                line_dash="solid", 
                line_color="green",
                line_width=2,
                annotation_text=f"Trial {trial['trial_number']} Start",
                annotation_position="top right"
            )
    
    # Add vertical lines for trial ends (red)
    for _, trial in trials_df.iterrows():
        if not pd.isna(trial['stop']):
            fig.add_vline(
                x=trial['stop'], 
                line_dash="solid", 
                line_color="red",
                line_width=2,
                annotation_text=f"Trial {trial['trial_number']} End",
                annotation_position="bottom right"
            )
    
    # Update layout
    fig.update_layout(
        title='All Door Traces with Trial Start/End Times',
        height=600,
        width=1200,
        showlegend=True,
        hovermode='x unified',
        xaxis_title="Frame Number",
        yaxis_title="Motion (Thresholded)"
    )
    
    fig.show()
    return fig

if __name__ == "__main__":
    # Example usage
    pkl_file = r"N:\TMAZE\MasterMouseFolder\ATO1\ClippedVideos\ATO1_2025-03-22T18_28_44_doorTraces.pkl"
    dat = openDoorTracesPkl(pkl_file)
    
    # You would need to load your trials_df here
    # trials_df = your_trials_dataframe
    
    # Uncomment the line below when you have trials_df loaded
    # plot_door_traces_with_trial_times(dat, trials_df)
    # plot_door_traces_with_trial_times_simplified(dat, trials_df) 