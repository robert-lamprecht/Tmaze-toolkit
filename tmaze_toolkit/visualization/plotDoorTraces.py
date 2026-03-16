import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plotDoorTraces(data):
    # Create subplots for each trace
    numPlots = len(data.keys())
    
    # Create subplot layout
    fig = make_subplots(
        rows=numPlots, 
        cols=1,
        subplot_titles=[f'{door} Trace' for door in data.keys()],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # Plot each trace
    for i, (door, trace) in enumerate(data.items(), 1):
        fig.add_trace(
            go.Scatter(
                y=trace,
                mode='lines',
                name=door,
                line=dict(width=1)
            ),
            row=i, 
            col=1
        )
    
    # Update layout for better appearance
    fig.update_layout(
        title='Door Traces',
        height=200 * numPlots,  # Adjust height based on number of plots
        width=1200,
        showlegend=False,
        hovermode='x unified'
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Frame Number", row=numPlots, col=1)
    
    # Update y-axis labels
    for i in range(1, numPlots + 1):
        fig.update_yaxes(title_text="Motion", row=i, col=1)
    
    # Show the plot
    fig.show()
