# Code to add to your test.ipynb notebook after cell 6 (the thresholded door traces)

# Import the plotting function
from scripts.plot_door_traces_with_trials import plot_door_traces_with_trial_times, plot_door_traces_with_trial_times_simplified

# Assuming you have:
# - dat: your door traces data (already loaded)
# - trials_df: your trials dataframe with 'start', 'stop', and 'trial_number' columns

# Option 1: Plot each door trace in separate subplots with trial times
fig1 = plot_door_traces_with_trial_times(dat, trials_df)

# Option 2: Plot all door traces in one plot with trial times (simplified view)
fig2 = plot_door_traces_with_trial_times_simplified(dat, trials_df)

# If you want to customize the plot further, you can modify the functions or create your own:

def custom_plot_door_traces_with_trials(dat, trials_df):
    """
    Custom function to plot door traces with trial times
    """
    # Process door traces to get thresholded data
    thresholded_dat = process_door_traces(dat, plot=False)
    
    # Create plot
    fig = go.Figure()
    
    # Plot each door trace with different colors
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
    
    # Add vertical lines for trial starts (green dashed)
    for _, trial in trials_df.iterrows():
        if not pd.isna(trial['start']):
            fig.add_vline(
                x=trial['start'], 
                line_dash="dash", 
                line_color="green",
                line_width=3,
                annotation_text=f"Trial {trial['trial_number']}",
                annotation_position="top right"
            )
    
    # Add vertical lines for trial ends (red dashed)
    for _, trial in trials_df.iterrows():
        if not pd.isna(trial['stop']):
            fig.add_vline(
                x=trial['stop'], 
                line_dash="dash", 
                line_color="red",
                line_width=3,
                annotation_text=f"End",
                annotation_position="bottom right"
            )
    
    # Update layout
    fig.update_layout(
        title='Door Traces with Trial Start/End Times',
        height=600,
        width=1200,
        showlegend=True,
        hovermode='x unified',
        xaxis_title="Frame Number",
        yaxis_title="Motion (Thresholded)"
    )
    
    fig.show()
    return fig

# Use the custom function
fig3 = custom_plot_door_traces_with_trials(dat, trials_df) 