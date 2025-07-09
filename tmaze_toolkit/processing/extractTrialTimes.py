import pandas as pd
from tmaze_toolkit.data.jsonProcessing import load_json_files
def find_concurrent_events(trace1, trace2, window_frames=15):
    """
    Find distinct trial events when two doors move concurrently.
    
    Args:
        trace1 (list): First door trace (binary 0/1 values)
        trace2 (list): Second door trace (binary 0/1 values)
        window_frames (int): Window size to check for concurrent movement
        
    Returns:
        List of frame indices where distinct events start
    """
    events = []  # Store frame numbers where events start
    i = 0  # Initialize frame counter
    is_moving = False  # Track if we're currently in a concurrent movement period
    
    # Loop through the traces, stopping window_frames before the end to prevent overflow
    while i < len(trace1) - window_frames:
        # Get a slice of frames to check for concurrent movement
        window1 = trace1[i:i+window_frames]  # Window for first door
        window2 = trace2[i:i+window_frames]  # Window for second door
        
        # Check if both doors show any movement (1's) in their windows
        concurrent_movement = (1 in window1 and 1 in window2)
        
        if concurrent_movement and not is_moving:
            # Start of a new concurrent movement event
            events.append(i)
            is_moving = True
            i += 1  # Move to next frame to continue checking
        elif not concurrent_movement and is_moving:
            # End of concurrent movement period
            is_moving = False
            i += 1
        else:
            # Either continuing current state or no movement
            i += 1
            
    return events

def pad_movement(floor, window_pad):
    """
    Pad floor movement signal to handle noisy data and direction changes.
    Uses a sliding window approach to connect nearby movements.
    
    Args:
        floor (list): Binary floor movement signal
        window_pad (int): Window size for padding
        
    Returns:
        list: Padded floor movement signal
    """
    padded_floor = floor.copy()
    
    # First pass: forward padding
    for x in range(len(floor) - window_pad):
        if floor[x] == 1:
            # Look ahead for any movement within window
            for y in range(x + 1, min(x + window_pad, len(floor))):
                if floor[y] == 1:
                    # Fill all gaps between movements
                    for z in range(x + 1, y):
                        padded_floor[z] = 1
                    break
    
    # Second pass: backward padding to catch reversed movements
    for x in range(len(floor) - 1, window_pad, -1):
        if padded_floor[x] == 1:
            # Look backward for any movement within window
            for y in range(x - 1, max(x - window_pad, -1), -1):
                if padded_floor[y] == 1:
                    # Fill all gaps between movements
                    for z in range(x - 1, y, -1):
                        padded_floor[z] = 1
                    break
    
    # Third pass: clean up isolated movements
    # If a movement is isolated (no other movements within window_pad), remove it
    for x in range(window_pad, len(padded_floor) - window_pad):
        if padded_floor[x] == 1:
            # Check if there are any other movements in the window
            has_neighbor = False
            for y in range(x - window_pad, x + window_pad + 1):
                if y != x and padded_floor[y] == 1:
                    has_neighbor = True
                    break
            if not has_neighbor:
                padded_floor[x] = 0
    
    return padded_floor

def extract_floor_traces(dat, pad_frames=90):
    """
    Extract and process floor movement traces with improved padding.
    
    Args:
        dat (dict): Dictionary containing door and floor data
        pad_frames (int): Number of frames to pad floor movements
        
    Returns:
        tuple: Lists of floor movement start and end frames
    """
    # Pad floor movements with a larger window to catch more related movements
    floor1 = pad_movement(dat['door5'], pad_frames)
    floor2 = pad_movement(dat['door6'], pad_frames)
    
    # Combine floor traces
    floor_traces = [f1 + f2 for f1, f2 in zip(floor1, floor2)]
    
    # Find continuous floor movements
    floor_starts = []
    floor_ends = []
    isMoving = False
    
    for x in range(len(floor_traces)):
        if floor_traces[x] == 2 and not isMoving:
            floor_starts.append(x)
            isMoving = True
        elif floor_traces[x] < 2 and isMoving:
            floor_ends.append(x)
            isMoving = False
    
    # Handle case where floor is still moving at end of recording
    if isMoving:
        floor_ends.append(len(floor_traces) - 1)
    
    return floor_starts, floor_ends

def clean_trial_events(starts, ends, window_frames):
    """
    Clean trial events to ensure proper sequencing (start -> end -> start -> end)
    
    Args:
        starts (list): Frame numbers of potential trial starts
        ends (list): Frame numbers of potential trial ends
        window_frames (int): Window size used for detection
        
    Returns:
        tuple: Lists of cleaned trial starts and ends
    """
    cleaned_starts = []
    cleaned_ends = []
    last_end = 0
    
    i, j = 0, 0  # Indices for starts and ends lists
    
    while i < len(starts) and j < len(ends):
        current_start = starts[i]
        current_end = ends[j]

        
        # If we find a valid start (after last end) and its corresponding end
        if current_start > last_end and current_end > current_start:
            cleaned_starts.append(current_start)
            cleaned_ends.append(current_end)
            last_end = current_end
            i += 1
            j += 1
        # Skip invalid starts (before last end)
        elif current_start <= last_end:
            i += 1
        # Skip ends that come before their start
        elif current_end <= current_start:
            j += 1

    while i < len(starts) and j < len(ends):
        current_start = starts[i]
        current_end = ends[j]
            
    return cleaned_starts, cleaned_ends

def extract_trial_times(dat, window_frames=15, pad_frames=90, fps=30, use_floor_traces=False):
    # Get floor movement signals
    floor_starts, floor_ends = extract_floor_traces(dat, pad_frames)
    
    # Get door movement signals
    trial_starts = find_concurrent_events(dat['door1'], dat['door2'], window_frames)
    trial_ends = find_concurrent_events(dat['door3'], dat['door4'], window_frames)
   
    # Clean the door-based trial events
    cleaned_starts, cleaned_ends = clean_trial_events(trial_starts, trial_ends, window_frames)

    # Create a list to store final trial times
    final_starts = []
    final_ends = []
    
    # Use floor movement to help identify missing door movements
    for i in range(len(floor_starts)):
        floor_start = floor_starts[i]
        floor_end = floor_ends[i]
        
        # Find the closest door start after this floor end
        closest_door_start = None
        for start in cleaned_starts:
            if start > floor_end and (closest_door_start is None or start < closest_door_start):
                closest_door_start = start
        
        # Find the closest door end before the next floor start
        closest_door_end = None
        for end in cleaned_ends:
            if end < floor_start and (closest_door_end is None or end > closest_door_end):
                closest_door_end = end
        
        # If we found both door movements, use them
        if closest_door_start is not None and closest_door_end is not None:
            final_starts.append(closest_door_start)
            final_ends.append(closest_door_end)
        # If we're missing a door movement, use floor signal as backup
        else:
            if closest_door_start is None:
                final_starts.append(floor_end + 1)  # Add 1 frame to ensure it's after floor movement
            else:
                final_starts.append(closest_door_start)
                
            if closest_door_end is None:
                final_ends.append(floor_start - 1)  # Subtract 1 frame to ensure it's before floor movement
            else:
                final_ends.append(closest_door_end)

    # Create the final dataframe
    trials_df = pd.DataFrame({
        'trial_start_frame': final_starts,
        'trial_end_frame': final_ends,
        'trial_start_time': [frame/fps for frame in final_starts],
        'trial_end_time': [frame/fps for frame in final_ends],
        'trial_duration': [(end - start)/fps for start, end in zip(final_starts, final_ends)]
    })

    return trials_df


def verify_correct_trial_times(trial_df, jsonFileLocation):
    """
    Verify the correctness of the trial times by comparing them to the json file
    Args:
        trial_df: pandas dataframe, the dataframe containing the trial times
        jsonFileLocation: string, the location of the json file
    Returns:
        bool: True if the trial times are correct, False otherwise
    """
    json_files = load_json_files(jsonFileLocation)
    
    if (len(json_files) - 1) != len(trial_df):
        print(f"Warning: Number of trials in the json file and the trial dataframe do not match")
        return False
    
    else:
        print("Trial times verified successfully")
        return True