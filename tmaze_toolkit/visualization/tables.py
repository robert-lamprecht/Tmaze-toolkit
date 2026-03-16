import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.offline import init_notebook_mode
import pandas as pd

def create_interactive_table(df, title="DataFrame View", height=600, width=1200, 
                           header_color='paleturquoise', cell_color='lavender',
                           header_font_size=12, cell_font_size=10):
    """
    Create an interactive scrollable table from a pandas DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to display
    title : str, optional
        Title for the table (default: "DataFrame View")
    height : int, optional
        Height of the table in pixels (default: 600)
    width : int, optional
        Width of the table in pixels (default: 1200)
    header_color : str, optional
        Background color for header cells (default: 'paleturquoise')
    cell_color : str, optional
        Background color for data cells (default: 'lavender')
    header_font_size : int, optional
        Font size for header text (default: 12)
    cell_font_size : int, optional
        Font size for cell text (default: 10)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive table figure
    """
    # Initialize plotly for Jupyter
    init_notebook_mode(connected=True)
    
    # Create interactive table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color=header_color,
            align='left',
            font=dict(size=header_font_size)
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color=cell_color,
            align='left',
            font=dict(size=cell_font_size)
        )
    )])
    
    fig.update_layout(
        title=title,
        height=height,
        width=width
    )
    
    return fig

def show_interactive_table(df, **kwargs):
    """
    Display an interactive scrollable table from a pandas DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to display
    **kwargs : 
        Additional arguments passed to create_interactive_table()
    
    Returns:
    --------
    None
        Displays the table directly
    """
    fig = create_interactive_table(df, **kwargs)
    fig.show()

def create_window_length_table(trials_df, include_original_order=True):
    """
    Create a table sorted by window length (shortest to longest) from a trials DataFrame.
    
    Parameters:
    -----------
    trials_df : pandas.DataFrame
        DataFrame containing trial data with window_start_time_seconds and window_end_time_seconds columns
    include_original_order : bool, optional
        Whether to include the original trial number as a separate column (default: True)
    
    Returns:
    --------
    pandas.DataFrame
        New DataFrame sorted by window length (shortest to longest)
    """
    # Calculate window length in seconds
    window_length = trials_df['window_end_time_seconds'] - trials_df['window_start_time_seconds']
    
    # Create new DataFrame with relevant columns
    if include_original_order:
        window_df = pd.DataFrame()
        window_df['original_trial_number'] = trials_df['trial_number']
    else:
        window_df = pd.DataFrame()
    
    window_df['window_start_time_seconds'] = trials_df['window_start_time_seconds']
    window_df['window_end_time_seconds'] = trials_df['window_end_time_seconds']
    window_df['window_length_seconds'] = window_length
    window_df['valid'] = trials_df['valid']
    
    # Sort by window length (shortest to longest)
    window_df = window_df.sort_values('window_length_seconds')
    
    # Reset index to show new order
    window_df = window_df.reset_index(drop=True)
    
    return window_df

def show_window_length_table(trials_df, include_original_order=True, **kwargs):
    """
    Display an interactive table sorted by window length (shortest to longest).
    
    Parameters:
    -----------
    trials_df : pandas.DataFrame
        DataFrame containing trial data with window_start_time_seconds and window_end_time_seconds columns
    include_original_order : bool, optional
        Whether to include the original trial number as a separate column (default: True)
    **kwargs : 
        Additional arguments passed to create_interactive_table()
    
    Returns:
    --------
    None
        Displays the table directly
    """
    window_df = create_window_length_table(trials_df, include_original_order)
    show_interactive_table(window_df, title="Trials Sorted by Window Length", **kwargs)