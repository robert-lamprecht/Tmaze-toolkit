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
    
    # Loop through the traces, stopping window_frames before the end to prevent overflow
    while i < len(trace1) - window_frames:
        # Get a slice of frames to check for concurrent movement
        window1 = trace1[i:i+window_frames]  # Window for first door
        window2 = trace2[i:i+window_frames]  # Window for second door
        
        # Check if both doors show any movement (1's) in their windows
        if 1 in window1 and 1 in window2:
            # If both doors moved, record the start frame of this window
            events.append(i)
            # Skip ahead by the window size to avoid detecting the same event multiple times
            i += window_frames
        else:
            # If no concurrent movement, check the next frame
            i += 1
            
    return events

def pad_movement(floor, window_pad):
    for x in range(len(floor) - window_pad):
        if floor[x] == 1:
            # Search for the next 1 in the next window_pad frames
            for y in range(x + 1, x + window_pad):
                if floor[y] == 1:
                    # If a 1 is found, set all values in between to 1
                    for z in range(x + 1, y):
                        floor[z] = 1
                    break
    return floor

def extract_floor_traces(dat, pad_frames=90):
    floor1 = pad_movement(dat['door5'], pad_frames)
    floor2 = pad_movement(dat['door6'], pad_frames)
    floor_traces = []

    # Add floor1 [x] to floor2 [x]
    for x in range(len(floor1)):
        floor_traces.append(floor1[x] + floor2[x])
    
    floor_starts = []
    floor_ends = []
    isMoving = False
    for x in range(len(floor_traces)):
        if floor_traces[x] == 2:
            if isMoving == False:
                floor_starts.append(x)
                isMoving = True
        if floor_traces[x] < 2:
            if isMoving == True:
                floor_ends.append(x)
                isMoving = False
    
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
    floor_starts, floor_ends = extract_floor_traces(dat, pad_frames)
    trial_starts = find_concurrent_events(dat['door1'], dat['door2'], window_frames)
    trial_ends = find_concurrent_events(dat['door3'], dat['door4'], window_frames)
   
    cleaned_starts, cleaned_ends = clean_trial_events(trial_starts, trial_ends, window_frames)

    print(f"Floor Starts: {len(floor_starts)}")
    print(f"Floor Ends: {len(floor_ends)}")

    print(f"Trial Starts: {len(cleaned_starts)}")
    print(f"Trial Ends: {len(cleaned_ends)}")

    if use_floor_traces:
        cleaned_starts, cleaned_ends = clean_trial_events(floor_starts, floor_ends, window_frames)
    

    floors_trials_df = pd.DataFrame({
        'trial_start_frame': floor_starts,
        'trial_end_frame': floor_ends,
        'trial_start_time': [frame/fps for frame in floor_starts],
        'trial_end_time': [frame/fps for frame in floor_ends],
        'trial_duration': [(end - start)/fps for start, end in zip(floor_starts, floor_ends)]
    })

    trials_df = pd.DataFrame({
        'trial_start_frame': cleaned_starts,     # Frame numbers where trials begin
        'trial_end_frame': cleaned_ends,         # Frame numbers where trials end
        'trial_start_time': [frame/fps for frame in cleaned_starts],  # Convert frames to seconds
        'trial_end_time': [frame/fps for frame in cleaned_ends]       # Convert frames to seconds
    })
    for i in range(len(trials_df)):
        if abs(trials_df['trial_end_frame'][i] - floors_trials_df['trial_start_frame'][i]) > 60:
            print(f"Trial {i} likely has a missed detection in the doors 1 and 2")

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