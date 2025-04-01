"""
extraction.py - Door Motion Analysis from Video Files

This module contains functions for extracting motion traces from doors in experimental videos.
It provides interactive UI components for selecting video frames and door positions,
along with efficient algorithms for detecting and quantifying movement.

The main workflow is:
1. Select video files
2. Select which frames to analyze
3. Mark the door positions
4. Process the videos to extract motion traces
"""

import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy for numerical operations on arrays
import os  # Operating system utilities (file paths, etc.)
import pickle  # For saving/loading Python objects to files
from tqdm import tqdm  # Progress bar for long-running operations
from tkinter import Tk  # Basic GUI toolkit
from tkinter.filedialog import askopenfilenames  # File selection dialog


# Default door locations - can be overridden by the user during selection
# These are [x, y] coordinates in the video frame
DEFAULT_DOOR_COORDS = {
    'door1': [14, 36],   # Floor Left door position
    'door2': [13, 125],  # Floor Right door position
    'door3': [217, 183], # End Zone Top door position
    'door4': [223, 16],  # End Zone Bottom door position
}

# Global variables used for GUI interaction
# These need to be global because they're used in callback functions
BLUE = [255, 0, 0]  # BGR format (not RGB) - Blue color for selected doors
RED = [0, 0, 255]   # BGR format - Red color for initial door positions


def select_video_files(initial_dir='.', file_types=None):
    """
    Opens a file dialog window to let the user select one or more video files.
    
    This is the first step in the workflow - selecting which videos to analyze.
    
    Args:
        initial_dir (str): Directory to start the file browser in
        file_types (list): Types of files to show in the browser
        
    Returns:
        tuple: List of selected video file paths
    """
    # Default to common video file types if none specified
    if file_types is None:
        file_types = [('Video Files', '*.mp4 *.avi *.mov')]
        
    # Create a hidden Tkinter root window for the file dialog
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Make sure dialog appears on top
    
    # Display the file selection dialog
    video_files = askopenfilenames(
        initialdir=initial_dir,
        filetypes=file_types,
        title="Select Video Files"
    )
    
    # Clean up the Tkinter window
    root.destroy()
    return video_files


def select_frame_range(video_path):
    """
    Interactive GUI for selecting which frames of the video to analyze.
    
    This is step 2 in the workflow - defining the time segment to analyze.
    This lets you skip irrelevant parts at the beginning or end of videos.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        tuple: Selected frame range as (start_frame, end_frame)
    """
    # Open the video
    vid = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv2.CAP_PROP_FPS)  # Frames per second
    
    # Create a window for the frame selection interface
    cv2.namedWindow('Select Frame Range')
    
    # Dummy function for trackbar callback (required by OpenCV)
    def nothing(x):
        pass
    
    # Create trackbars (sliders) for frame selection
    cv2.createTrackbar('Start Frame', 'Select Frame Range', 0, total_frames-1, nothing)
    cv2.createTrackbar('End Frame', 'Select Frame Range', total_frames-1, total_frames-1, nothing)
    cv2.createTrackbar('Current Frame', 'Select Frame Range', 0, total_frames-1, nothing)
    
    # Initialize playback state
    playing = False  # Whether video is currently playing
    current_frame = 0  # Current frame being displayed
    
    # Main loop for frame range selection
    while True:
        # Get current positions of all trackbars
        start_frame = cv2.getTrackbarPos('Start Frame', 'Select Frame Range')
        end_frame = cv2.getTrackbarPos('End Frame', 'Select Frame Range')
        
        # Ensure end frame comes after start frame
        if end_frame <= start_frame:
            end_frame = start_frame + 1
            cv2.setTrackbarPos('End Frame', 'Select Frame Range', end_frame)
        
        # Update current frame position based on playback state
        if playing:
            # If playing, automatically advance frame
            current_frame += 1
            # Loop back to start frame if we reach the end
            if current_frame >= end_frame:
                current_frame = start_frame
            # Update the current frame trackbar
            cv2.setTrackbarPos('Current Frame', 'Select Frame Range', current_frame)
        else:
            # If not playing, use trackbar position
            current_frame = cv2.getTrackbarPos('Current Frame', 'Select Frame Range')
            
        # Display the current frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = vid.read()
        if ret:
            # Calculate time values for display
            total_duration = total_frames / fps  # Total video duration in seconds
            selected_duration = (end_frame - start_frame) / fps  # Selected segment duration
            current_time = current_frame / fps  # Current position in seconds
            
            # Format time values as MM:SS for display
            def format_time(seconds):
                minutes = int(seconds // 60)
                seconds = int(seconds % 60)
                return f"{minutes:02d}:{seconds:02d}"
            
            # Prepare text information to display
            info_text = [
                f'Start: Frame {start_frame} ({format_time(start_frame/fps)})',
                f'Current: Frame {current_frame} ({format_time(current_time)})',
                f'End: Frame {end_frame} ({format_time(end_frame/fps)})',
                f'Selected Duration: {format_time(selected_duration)}',
                f'Total Duration: {format_time(total_duration)}',
                '',
                'Controls:',
                'SPACE - Play/Pause',
                'ENTER - Confirm Selection',
                'ESC - Reset Selection'
            ]
            
            # Add each line of information text to the frame
            y_position = 30
            for text in info_text:
                cv2.putText(frame, text, (10, y_position), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_position += 25
            
            # Draw a progress bar at the bottom of the frame
            frame_height, frame_width = frame.shape[:2]
            bar_y = frame_height - 50  # Position from top
            
            # Calculate progress bar width based on current position
            progress_ratio = (current_frame - start_frame) / max(1, end_frame - start_frame)
            bar_width = int(progress_ratio * frame_width)
            
            # Draw background bar (red)
            cv2.rectangle(frame, (0, bar_y), (frame_width, bar_y + 20), (0, 0, 255), -1)
            # Draw progress bar (green)
            cv2.rectangle(frame, (0, bar_y), (bar_width, bar_y + 20), (0, 255, 0), -1)
            
            # Show the frame with overlays
            cv2.imshow('Select Frame Range', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(int(1000/fps)) & 0xFF  # Wait based on video framerate
        
        if key == 13:  # Enter key - confirm selection
            break
        elif key == 27:  # Esc key - reset selection to full video
            cv2.setTrackbarPos('Start Frame', 'Select Frame Range', 0)
            cv2.setTrackbarPos('End Frame', 'Select Frame Range', total_frames-1)
        elif key == 32:  # Space key - toggle play/pause
            playing = not playing
    
    # Clean up
    cv2.destroyAllWindows()
    vid.release()
    
    return (start_frame, end_frame)


def mouse_callback_doors(event, x, y, flags, params):
    """
    Mouse callback function for selecting door positions.
    
    This is used as a callback for the door selection interface.
    It handles drawing and updating the door position rectangles.
    
    Args:
        event: Mouse event type (OpenCV constant)
        x, y: Mouse coordinates
        flags: Additional flags provided by OpenCV
        params: Additional parameters (not used)
    """
    # These globals are used to communicate with the outer function
    global move_rectangle, BLUE, fg, bg, bgCopy, final_coords, door, rows, cols
    
    # When user presses mouse button, start rectangle movement
    if event == cv2.EVENT_LBUTTONDOWN:
        move_rectangle = True

    # When user moves mouse while button is pressed
    elif event == cv2.EVENT_MOUSEMOVE:
        try:
            if move_rectangle:
                # Reset background image to original state
                bg = bgCopy.copy()
                # Draw rectangle centered at current mouse position
                cv2.rectangle(bg, 
                             (x-int(0.5*cols), y-int(0.5*rows)),  # Top-left corner
                             (x+int(0.5*cols), y+int(0.5*rows)),  # Bottom-right corner
                             BLUE, -1)  # Blue fill (-1 means filled)
        except UnboundLocalError:
            # Handle case where variables aren't initialized yet
            pass

    # When user releases mouse button
    elif event == cv2.EVENT_LBUTTONUP:
        move_rectangle = False
        # Draw final rectangle position
        cv2.rectangle(bg, 
                     (x-int(0.5*cols), y-int(0.5*rows)),
                     (x+int(0.5*cols), y+int(0.5*rows)),
                     BLUE, -1)
        # Store the selected coordinates for this door
        final_coords[door] = [x, y]


def select_door_coords(video_path, initial_coords=DEFAULT_DOOR_COORDS):
    """
    Interactive GUI for selecting door positions in the video.
    
    This is step 3 in the workflow - marking exactly where the doors are
    located in the frame. This lets the algorithm focus on just these areas.
    
    Args:
        video_path (str): Path to the video file
        initial_coords (dict): Initial door coordinates as a starting point
        
    Returns:
        dict: Selected door coordinates {door_name: [x, y]}
    """
    # These globals will be used by the mouse callback function
    global fg, bg, bgCopy, move_rectangle, BLUE, RED, final_coords, door, rows, cols
    
    print(f'Selecting door coordinates for: {video_path}')
    
    # Open the video and jump to the middle frame
    vid = cv2.VideoCapture(video_path)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.set(cv2.CAP_PROP_POS_FRAMES, int(length/2))
    ret, frame = vid.read()
    
    # Human-readable names for the doors (used in the UI)
    display_names = {
        'door1': 'Floor Left',
        'door2': 'Floor Right',
        'door3': 'Endzone Top',
        'door4': 'Endzone Bottom'
    }
    
    # Set up variables for the rectangle drawing
    fg = frame[:14, :14]  # Small sample of the frame for door size (15x15 pixels)
    bg = frame.copy()     # Working copy of the frame
    bgCopy = bg.copy()    # Original frame that we'll reset to
    move_rectangle = False  # Tracks if we're currently moving a rectangle
    final_coords = {}     # Will store the final door coordinates
    
    # Helper function to draw the initial door positions
    def draw_initial_coord(k, display_img):
        """Draw a red rectangle at the initial position of a door"""
        cv2.rectangle(display_img,
                     (initial_coords[k][0]-int(0.5*cols), initial_coords[k][1]-int(0.5*rows)),
                     (initial_coords[k][0]+int(0.5*cols), initial_coords[k][1]+int(0.5*rows)),
                     RED, -1)  # Red fill
    
    # Get the dimensions of our door marker
    rows, cols = fg.shape[:2]
    
    # Process each door one by one
    for door in initial_coords.keys():
        # Create window with instruction for this door
        window_name = f'Select {display_names[door]} and press ENTER'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback_doors)
        
        # Loop until user confirms door position with Enter
        while True:
            # Create fresh copy of frame for display
            display_img = bg.copy()
            # Draw the initial position in red
            draw_initial_coord(door, display_img)
            # Show the image
            cv2.imshow(window_name, display_img)
            # Wait for key press
            k = cv2.waitKey(1)
            # Check if Enter was pressed
            if k == 13 & 0xFF:  # 13 is ASCII for Enter
                break
                
        # Close the window after selection
        cv2.destroyAllWindows()

    # For any doors that weren't manually selected, use the default position
    for door in initial_coords:
        if door not in final_coords:
            final_coords[door] = initial_coords[door]  # Use default
            print(f"Using default position for {display_names[door]}: {final_coords[door]}")
        else:
            print(f"Selected position for {display_names[door]}: {final_coords[door]}")

    # Clean up
    vid.release()
    return final_coords


def extract_door_traces(video_path, door_coords, frame_range, output_dir=None, 
                        window_size=15, batch_processing=True):
    """
    Extract motion traces for doors in the video.
    
    This is step 4 in the workflow - the actual analysis of door movement.
    It detects motion at each door position by comparing consecutive frames
    and measuring pixel differences.
    
    Args:
        video_path (str): Path to the video file
        door_coords (dict): Door coordinates {door_name: [x, y]}
        frame_range (tuple): Frame range (start_frame, end_frame)
        output_dir (str, optional): Output directory for results
        window_size (int, optional): Size of door detection window
        batch_processing (bool, optional): Whether to process frames in batches
        
    Returns:
        dict: Door traces {door_name: [motion_values]}
    """
    print(f'Extracting door traces from: {video_path}')
    
    # Open the video
    vid = cv2.VideoCapture(video_path)
    
    # Extract frame range boundaries
    start_frame, end_frame = frame_range
    num_frames = end_frame - start_frame  # Total frames to process
    
    # Pre-compute door regions to avoid repeated calculations
    # For each door, we'll only analyze a small window around its position
    door_regions = {}
    half_window = window_size // 2  # Half the window size (for +/- from center)
    
    for door_key, coords in door_coords.items():
        # Create slice objects for efficient numpy array indexing
        # Format is (y_slice, x_slice) because images are indexed as [row, col]
        door_regions[door_key] = (
            slice(coords[1] - half_window, coords[1] + half_window),  # y1:y2
            slice(coords[0] - half_window, coords[0] + half_window)   # x1:x2
        )
    
    # Initialize output arrays - one for each door
    # We'll store the motion value for each frame
    door_traces = {door_key: np.zeros(num_frames, dtype=np.float32) 
                  for door_key in door_coords.keys()}
    
    # Choose processing method based on parameter
    if batch_processing:
        # ===== BATCH PROCESSING MODE =====
        # This mode loads multiple frames at once, which is faster but uses more memory
        
        batch_size = 1000  # Number of frames to process at once
        
        # Process each batch with a progress bar
        for batch_start in tqdm(range(start_frame, end_frame, batch_size), 
                               desc="Processing batches"):
            # Calculate end of current batch (handle last batch correctly)
            batch_end = min(batch_start + batch_size, end_frame)
            
            # Load a batch of frames
            frames = []
            vid.set(cv2.CAP_PROP_POS_FRAMES, batch_start)  # Jump to start frame
            
            for _ in range(batch_end - batch_start):
                ret, frame = vid.read()
                if not ret:  # End of video
                    break
                # Store the full frame
                frames.append(frame)
            
            # Skip processing if we couldn't read any frames
            if not frames:
                break
                
            # Convert list of frames to a numpy array for faster processing
            frames = np.array(frames)
            
            # We need at least 2 frames to detect motion
            if len(frames) < 2:
                continue
                
            # Process each frame in the batch
            prev_frame = frames[0]
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            
            # For each frame after the first one:
            for i, frame in enumerate(frames[1:], start=1):
                # Calculate actual frame index in our output arrays
                frame_idx = batch_start - start_frame + i
                if frame_idx >= num_frames:
                    break  # Safety check
                    
                # Convert current frame to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Process each door region
                for door_key, region in door_regions.items():
                    # Only process the small region around the door
                    # This is much faster than processing the whole frame
                    
                    # Calculate absolute difference between frames in door region
                    door_diff = cv2.absdiff(
                        prev_gray[region],
                        gray[region]
                    )
                    
                    # Threshold the difference to get binary image (movement = white)
                    # This converts any pixel change > 7 to value 255 (white)
                    _, door_thresh = cv2.threshold(door_diff, 7, 255, cv2.THRESH_BINARY)
                    
                    # Sum the white pixels to get total motion
                    # More white pixels = more movement
                    door_traces[door_key][frame_idx] = np.sum(door_thresh)
                
                # Current frame becomes previous frame for next iteration
                prev_gray = gray
                
                # Check for user interrupt
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    else:
        # ===== SINGLE FRAME PROCESSING MODE =====
        # This mode processes one frame at a time
        # It's slower but uses less memory
        
        # Jump to start frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, prev_frame = vid.read()
        if not ret:
            raise ValueError("Could not read first frame")
            
        # Convert first frame to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Process frames with progress bar
        for frame_idx in tqdm(range(1, num_frames), desc="Processing frames"):
            # Read the next frame
            ret, frame = vid.read()
            if not ret:
                break
                
            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process each door region
            for door_key, region in door_regions.items():
                # Calculate absolute difference in door region
                door_diff = cv2.absdiff(
                    prev_gray[region],
                    gray[region]
                )
                
                # Threshold to binary (movement = white)
                _, door_thresh = cv2.threshold(door_diff, 7, 255, cv2.THRESH_BINARY)
                
                # Sum white pixels for motion value
                door_traces[door_key][frame_idx] = np.sum(door_thresh)
            
            # Current frame becomes previous frame
            prev_gray = gray
            
            # Check for user interrupt
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Clean up resources
    vid.release()
    cv2.destroyAllWindows()
    
    # Save results to file
    if output_dir is not None:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Generate output filename based on video name
        output_path = os.path.join(output_dir, os.path.basename(video_path).split('.')[0] + '_doorTraces.pkl')
    else:
        # Save in same directory as video
        output_path = video_path.split('.')[0] + '_doorTraces.pkl'
        
    # Save using pickle format
    with open(output_path, 'wb') as f:
        pickle.dump(door_traces, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Door traces saved to: {output_path}")
    return door_traces


def process_video(video_path, output_dir=None, initial_coords=DEFAULT_DOOR_COORDS):
    """
    Process a single video file through the complete workflow.
    
    This is a convenience function that runs all steps in sequence:
    1. Frame range selection
    2. Door position selection
    3. Motion trace extraction
    
    Args:
        video_path (str): Path to the video file
        output_dir (str, optional): Output directory for results
        initial_coords (dict, optional): Initial door coordinates
        
    Returns:
        dict: Door traces {door_name: [motion_values]}
    """
    print(f"Processing video: {video_path}")
    
    # Step 1: Select frame range
    print("Select frame range...")
    frame_range = select_frame_range(video_path)
    
    # Step 2: Select door coordinates
    print("Select door positions...")
    door_coords = select_door_coords(video_path, initial_coords)
    
    # Step 3: Extract door traces
    print("Extracting door traces...")
    door_traces = extract_door_traces(
        video_path, 
        door_coords,
        frame_range,
        output_dir
    )
    
    return door_traces


def batch_process_videos(video_paths, output_dir=None, initial_coords=DEFAULT_DOOR_COORDS):
    """
    Process multiple video files in sequence.
    
    This function runs the complete workflow on each video file:
    1. Checks if the video has already been processed
    2. If not, runs the full processing pipeline
    3. Returns all results combined
    
    Args:
        video_paths (list): List of paths to video files
        output_dir (str, optional): Output directory for results
        initial_coords (dict, optional): Initial door coordinates
        
    Returns:
        dict: Dictionary mapping video paths to door traces
    """
    results = {}
    
    for video_path in video_paths:
        # Determine output filename for this video
        if output_dir:
            output_path = os.path.join(output_dir, os.path.basename(video_path).split('.')[0] + '_doorTraces.pkl')
        else:
            output_path = video_path.split('.')[0] + '_doorTraces.pkl'
        
        # Check if already processed
        if os.path.exists(output_path):
            print(f"Door traces already exist for {video_path}. Skipping...")
            
            # Load existing traces
            with open(output_path, 'rb') as f:
                results[video_path] = pickle.load(f)
        else:
            # Process video
            results[video_path] = process_video(video_path, output_dir, initial_coords)
    
    return results
