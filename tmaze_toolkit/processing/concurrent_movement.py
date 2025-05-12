import numpy as np

def find_concurrent_movement(trace1, trace2):
    """
    Find periods where two traces have concurrent movement (both are 1).
    
    Args:
        trace1 (array-like): First binary trace (0s and 1s)
        trace2 (array-like): Second binary trace (0s and 1s)
        
    Returns:
        tuple: Two lists containing the start and end indices of concurrent movement
    """
    if len(trace1) != len(trace2):
        raise ValueError("Traces must be the same length")
        
    concurrent_starts = []
    concurrent_ends = []
    in_concurrent = False
    
    for i in range(len(trace1)):
        # Check if both traces are 1 at this point
        is_concurrent = trace1[i] == 1 and trace2[i] == 1
        
        # Start of concurrent movement
        if not in_concurrent and is_concurrent:
            concurrent_starts.append(i)
            in_concurrent = True
            
        # End of concurrent movement
        elif in_concurrent and not is_concurrent:
            concurrent_ends.append(i)
            in_concurrent = False
    
    # Handle case where traces end during concurrent movement
    if in_concurrent:
        concurrent_ends.append(len(trace1))
        
    return concurrent_starts, concurrent_ends

def plot_concurrent_periods(trace1, trace2, concurrent_starts, concurrent_ends):
    """
    Plot the traces and highlight periods of concurrent movement.
    
    Args:
        trace1 (array-like): First binary trace
        trace2 (array-like): Second binary trace
        concurrent_starts (list): Start indices of concurrent movement
        concurrent_ends (list): End indices of concurrent movement
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(trace1, label='Trace 1', alpha=0.7)
    plt.plot(trace2, label='Trace 2', alpha=0.7)
    
    # Highlight concurrent periods
    for start, end in zip(concurrent_starts, concurrent_ends):
        plt.axvspan(start, end, color='gray', alpha=0.3)
    
    plt.legend()
    plt.title('Traces with Concurrent Movement Periods Highlighted')
    plt.xlabel('Frame')
    plt.ylabel('State')
    plt.grid(True)
    plt.show()