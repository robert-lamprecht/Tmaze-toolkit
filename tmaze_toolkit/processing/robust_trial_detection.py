import pandas as pd
import numpy as np

def find_trial_events(trace1, trace2, window_frames=15):
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
    
    return events  # FIXED: Added missing return statement

def find_floor_trial_windows(floor_trace1, floor_trace2, window_frames):
    """
    Find trial windows using floor traces (doors 5+6).
    When doors 5+6 stop moving = trial start window
    When doors 5+6 start moving = trial end window
    
    Args:
        floor_trace1 (list): First floor door trace (binary 0/1 values)
        floor_trace2 (list): Second floor door trace (binary 0/1 values)
        window_frames (int): Window size to check for concurrent movement
        
    Returns:
        List of tuples (start_frame, end_frame) for each trial window
    """
    # Combine floor traces - we want to detect when BOTH stop moving (trial start)
    # and when BOTH start moving again (trial end)
    combined_floor = np.logical_and(floor_trace1, floor_trace2)
    
    trial_windows = []
    i = 0
    
    # Skip initial movement - find first time both doors stop moving
    while i < len(combined_floor) - window_frames:
        window = combined_floor[i:i+window_frames]
        if not np.any(window):  # Found first time both doors stop
            break
        i += 1
    
    # Skip the initial no-movement period (setup time)
    # Look for the first time doors start moving again after initial stop
    while i < len(combined_floor) - window_frames:
        window = combined_floor[i:i+window_frames]
        if np.any(window):  # Found first movement after initial stop
            break
        i += 1
    
    # Now look for actual trial windows
    while i < len(combined_floor) - window_frames:
        window = combined_floor[i:i+window_frames]
        
        # Check if both doors stopped moving (all 0's in window)
        if not np.any(window):  # All values are 0 (no movement)
            # This is a potential trial start window
            start_frame = i
            
            # Look for when they start moving again (any 1's)
            j = i + window_frames
            while j < len(combined_floor) - window_frames:
                next_window = combined_floor[j:j+window_frames]
                if np.any(next_window):  # Found movement again
                    end_frame = j
                    trial_windows.append((start_frame, end_frame))
                    i = j + window_frames
                    break
                j += 1
            else:
                # If we reach the end without finding movement, this is the final trial
                # Use the end of the trace as the trial end
                end_frame = len(combined_floor)
                trial_windows.append((start_frame, end_frame))
                break
        else:
            i += 1
    
    return trial_windows

def validate_trials_in_window(trial_starts, trial_ends, window_start, window_end, tolerance=30):
    """
    Validate that there's exactly one start and one end event within a trial window.
    
    Args:
        trial_starts (list): All potential trial start frames
        trial_ends (list): All potential trial end frames
        window_start (int): Start frame of trial window
        window_end (int): End frame of trial window
        tolerance (int): Tolerance frames for window boundaries
        
    Returns:
        tuple: (valid_start_frame, valid_end_frame) or (None, None) if invalid
    """
    # Find starts and ends within the window (with tolerance)
    valid_starts = [s for s in trial_starts if window_start - tolerance <= s <= window_end + tolerance]
    valid_ends = [e for e in trial_ends if window_start - tolerance <= e <= window_end + tolerance]
    
    # Check if we have exactly one start and one end
    if len(valid_starts) == 1 and len(valid_ends) == 1:
        start_frame = valid_starts[0]
        end_frame = valid_ends[0]
        
        # Ensure start comes before end
        if start_frame < end_frame:
            return start_frame, end_frame
    
    return None, None

def validate_door_movements_in_windows(frames_df, door_data, window_frames=15):
    """
    Validate door movements within trial windows defined by frames_df.
    
    Args:
        frames_df (DataFrame): DataFrame with columns 'trial_number', 'trial_time', 'end_time'
        door_data (dict): Dictionary containing door traces with keys 'door1', 'door2', 'door3', 'door4'
        window_frames (int): Window size for concurrent movement detection
        
    Returns:
        DataFrame: DataFrame with columns trial_number, trial_time, end_time, start, stop, duration, valid
    """
    results = []
    
    for _, row in frames_df.iterrows():
        trial_number = row['trial_number']
        trial_time = row['trial_time']
        end_time = row['end_time']
        
        # FIXED: Added bounds checking for slice indices
        if trial_time >= len(door_data['door1']) or end_time > len(door_data['door1']):
            print(f"Warning: Trial {trial_number} has invalid time bounds")
            continue
            
        if trial_time >= end_time:
            print(f"Warning: Trial {trial_number} has trial_time >= end_time")
            continue
      
        # Find door movements within the trial window
        # Look for doors 1+2 moving together (start)
        start_events = find_trial_events(
            door_data['door1'][trial_time:end_time], 
            door_data['door2'][trial_time:end_time], 
            window_frames
        )
        
        # Look for doors 3+4 moving together (stop)
        stop_events = find_trial_events(
            door_data['door3'][trial_time:end_time], 
            door_data['door4'][trial_time:end_time], 
            window_frames
        )
        
        # FIXED: Corrected the logic for validation
        if len(start_events) == 0 or len(stop_events) == 0:
            valid = "no"
            start = start_events[0] if start_events else None
            stop = stop_events[0] if stop_events else None
            duration = None
        else:
            start = start_events[0]
            stop = stop_events[0]
            duration = stop - start if start is not None and stop is not None else None
            valid = "yes"
            
        results.append({
            'trial_number': trial_number,
            'trial_time': trial_time,
            'end_time': end_time,
            'start': start,
            'stop': stop,
            'duration': duration,
            'valid': valid
        })
    
    return pd.DataFrame(results)

def robust_trial_detection(door_data, fps=30, window_frames=15, tolerance=30):
    """
    Robust trial detection using doors 5+6 for validation and doors 1-4 for precise timing.
    
    Args:
        door_data (dict): Dictionary containing door traces
        fps (int): Frames per second
        window_frames (int): Window size for concurrent movement detection
        tolerance (int): Tolerance frames for window boundaries
        
    Returns:
        tuple: (trials_df, diagnostics)
    """
    # Detect trial starts and ends using doors 1-4 (primary signal)
    trial_starts = find_trial_events(door_data['door1'], door_data['door2'], window_frames)
    trial_ends = find_trial_events(door_data['door3'], door_data['door4'], window_frames)
    
    # Detect trial windows using doors 5-6 (validation signal)
    trial_windows = find_floor_trial_windows(door_data['door5'], door_data['door6'], window_frames)
    
    # Create DataFrame with one row per trial window
    trials_data = []
    
    for trial_num, (window_start, window_end) in enumerate(trial_windows, 1):
        valid_start, valid_end = validate_trials_in_window(
            trial_starts, trial_ends, window_start, window_end, tolerance
        )
        
        if valid_start is not None and valid_end is not None:
            # Valid trial found
            trials_data.append({
                'trial_number': trial_num,
                'start_frame': valid_start,
                'stop_frame': valid_end,
                'start_time_seconds': valid_start / fps,
                'stop_time_seconds': valid_end / fps,
                'window_start': window_start,
                'window_end': window_end,
                'window_start_time_seconds': window_start / fps,
                'window_end_time_seconds': window_end / fps,
                'valid': True
            })
        else:
            # Invalid trial - no valid start/stop found in window
            trials_data.append({
                'trial_number': trial_num,
                'start_frame': np.nan,
                'stop_frame': np.nan,
                'start_time_seconds': np.nan,
                'stop_time_seconds': np.nan,
                'window_start': window_start,
                'window_end': window_end,
                'window_start_time_seconds': window_start / fps,
                'window_end_time_seconds': window_end / fps,
                'valid': False
            })
    
    # Create DataFrame
    trials_df = pd.DataFrame(trials_data)
    
    # Create diagnostics
    total_windows = len(trial_windows)
    valid_trials = len(trials_df[trials_df['valid'] == True])
    invalid_trials = len(trials_df[trials_df['valid'] == False])
    
    diagnostics = {
        'total_trial_starts_detected': len(trial_starts),
        'total_trial_ends_detected': len(trial_ends),
        'total_trial_windows': total_windows,
        'valid_trials': valid_trials,
        'invalid_trials': invalid_trials,
        'success_rate': valid_trials / total_windows if total_windows > 0 else 0
    }
    
    return trials_df, diagnostics

# Example usage:
if __name__ == "__main__":
    # Assuming you have door_data with keys 'door1', 'door2', 'door3', 'door4', 'door5', 'door6'
    # door_data = {'door1': [...], 'door2': [...], ...}
    
    # trials_df, diagnostics = robust_trial_detection(door_data)
    # print("Diagnostics:", diagnostics)
    # print("Trials DataFrame:")
    # print(trials_df[['trial_number', 'start_frame', 'stop_frame', 'start_time_seconds', 'stop_time_seconds', 'valid']])
    pass 