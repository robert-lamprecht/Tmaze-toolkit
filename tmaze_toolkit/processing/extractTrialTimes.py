import pandas as pd

def find_concurrent_movement(trace1, trace2, window_frames = 15):
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

def find_floor_movements(data):
    """
    Find start and end frames when floor doors ('door5', 'door6') move concurrently.

    Args:
        data (dict): Dictionary containing door movement traces, including 'door5' and 'door6'.

    Returns:
        tuple: A tuple containing two lists:
               - start_frames: List of frame indices where concurrent movement starts.
               - end_frames: List of frame indices where concurrent movement ends.
    """
    trace5 = data.get('door5', [])
    trace6 = data.get('door6', [])
    
    start_frames = []
    end_frames = []
    in_concurrent_movement = False
    n_frames = len(trace5)

    for i in range(n_frames):
        currently_moving_together = (trace5[i] == 1 and trace6[i] == 1)

        if currently_moving_together and not in_concurrent_movement:
            # Start of concurrent movement
            start_frames.append(i)
            in_concurrent_movement = True
        elif not currently_moving_together and in_concurrent_movement:
            # End of concurrent movement
            end_frames.append(i) # Records the first frame they are NOT moving together
            in_concurrent_movement = False

    # Handle case where movement continues until the very last frame
    if in_concurrent_movement:
        end_frames.append(n_frames) 

    return start_frames, end_frames

def find_trial_times(data, fps=30, window_frames=15):
    """
    Determines trial start/end frames and times using primary and fail-safe signals,
    returning a DataFrame. Ensures strict alternation of starts and ends.

    Primary Start: Concurrent movement of Door 1 & 2.
    Fail-safe Start: Floor (Door 5 & 6) stops moving.
    Primary End: Concurrent movement of Door 3 & 4, only after Floor starts moving again.
    Fail-safe End: Floor (Door 5 & 6) starts moving (after the trial start).

    Constraints:
    - Trials must alternate: Start, End, Start, End...
    - An End event must occur after its corresponding Start event.
    - A Primary End (Door 3&4) must occur after the Floor starts moving following the Start.

    Args:
        data (dict): Dictionary containing door movement traces 
                     (expects 'door1', 'door2', 'door3', 'door4', 'door5', 'door6').
        fps (int or float): Frames per second of the video recording (default 30).
        window_frames (int): Window size (frames) for detecting concurrent door movement (default 15).

    Returns:
        pd.DataFrame: DataFrame with columns: 'trial_start_frame', 'trial_end_frame',
                      'trial_start_time', 'trial_end_time'. Returns an empty
                      DataFrame if essential data is missing or no valid trials are found.
    """
    # --- 1. Check for necessary data --- 
    required_doors = ['door1', 'door2', 'door3', 'door4', 'door5', 'door6']
    # Check if door key is missing OR if the associated list/array is empty
    missing_doors = [d for d in required_doors if d not in data or len(data[d]) == 0]
    if missing_doors:
        print(f"Warning: Missing or empty data for doors: {missing_doors}. Cannot determine trial times.")
        return pd.DataFrame(columns=['trial_start_frame', 'trial_end_frame', 'trial_start_time', 'trial_end_time'])

    # --- 2. Calculate all potential event markers --- 
    d12_starts = find_concurrent_movement(data['door1'], data['door2'], window_frames)
    f_starts, f_stops = find_floor_movements(data) # floor_starts, floor_stops
    d34_starts = find_concurrent_movement(data['door3'], data['door4'], window_frames)

    # --- 3. Iterative Matching (State Machine) --- 
    final_starts = []
    final_ends = []

    i12, ifs_stop, i34, ifs_start = 0, 0, 0, 0
    n12, nfs_stop, n34, nfs_start = len(d12_starts), len(f_stops), len(d34_starts), len(f_starts)

    state = 'seeking_start'
    current_start = -1
    last_found_end = -1

    while True:
        if state == 'seeking_start':
            # Find next available primary start (d12) after last end
            while i12 < n12 and d12_starts[i12] <= last_found_end:
                i12 += 1
            next_d12 = d12_starts[i12] if i12 < n12 else float('inf')

            # Find next available fail-safe start (f_stop) after last end
            while ifs_stop < nfs_stop and f_stops[ifs_stop] <= last_found_end:
                ifs_stop += 1
            next_f_stop = f_stops[ifs_stop] if ifs_stop < nfs_stop else float('inf')

            # If no more potential starts, break
            if next_d12 == float('inf') and next_f_stop == float('inf'):
                break

            # Choose the earliest start (prefer primary d12 if simultaneous)
            if next_d12 <= next_f_stop:
                current_start = next_d12
                i12 += 1
            else:
                current_start = next_f_stop
                ifs_stop += 1
            
            state = 'seeking_end'

        elif state == 'seeking_end':
            # Find the first floor movement *starting* after the current trial start
            temp_ifs_start = ifs_start # Use temp index to find requirement without consuming
            while temp_ifs_start < nfs_start and f_starts[temp_ifs_start] <= current_start:
                temp_ifs_start += 1
            
            required_f_start = f_starts[temp_ifs_start] if temp_ifs_start < nfs_start else float('inf')

            # If no floor start happens after trial start, this start is invalid, seek next start
            if required_f_start == float('inf'):
                state = 'seeking_start' 
                continue # Go back to find the next start

            # Find the next primary end (d34) occurring at or after the required floor start
            temp_i34 = i34 # Use temp index
            while temp_i34 < n34 and d34_starts[temp_i34] < required_f_start:
                 temp_i34 += 1
            next_d34 = d34_starts[temp_i34] if temp_i34 < n34 else float('inf')

            # Decide on the end event
            current_end = -1
            if next_d34 != float('inf'): # Primary end (d34) is valid and available
                current_end = next_d34
                # Consume d34 event only if it was chosen
                i34 = temp_i34 + 1 
                # We also need to advance the floor start pointer past the chosen end
                while ifs_start < nfs_start and f_starts[ifs_start] <= current_end:
                    ifs_start += 1
            else: # Use fail-safe end (the required floor start)
                current_end = required_f_start
                # Consume the required floor start event if it was chosen as the end
                ifs_start = temp_ifs_start + 1 
                # We still need to advance the d34 pointer past this end
                while i34 < n34 and d34_starts[i34] <= current_end:
                   i34 += 1
            
            # Record the valid trial
            final_starts.append(current_start)
            final_ends.append(current_end)
            last_found_end = current_end
            state = 'seeking_start'

    # --- 4. Create DataFrame --- 
    if not final_starts: # No valid trials found
        return pd.DataFrame(columns=['trial_start_frame', 'trial_end_frame', 'trial_start_time', 'trial_end_time'])
        
    trials_df = pd.DataFrame({
        'trial_start_frame': final_starts,
        'trial_end_frame': final_ends
    })

    # Calculate times
    trials_df['trial_start_time'] = trials_df['trial_start_frame'] / fps
    trials_df['trial_end_time'] = trials_df['trial_end_frame'] / fps

    return trials_df

    
