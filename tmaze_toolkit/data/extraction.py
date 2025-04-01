"""
extraction.py - Door Motion Analysis from Video Files

This module contains functions for extracting motion traces from doors in experimental videos.
It provides interactive UI components for selecting door positions,
along with efficient algorithms for detecting and quantifying movement.

The main workflow is:
1. Select video files
2. Mark the door positions
3. Process the videos to extract motion traces
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
    
    This is step 2 in the workflow - marking exactly where the doors are
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


def extract_door_traces(video_path, door_coords, output_dir=None, 
                        window_size=15, batch_processing=True):
    """
    Extract motion traces for doors in the video.
    
    This is step 3 in the workflow - the actual analysis of door movement.
    It detects motion at each door position by comparing consecutive frames
    and measuring pixel differences.
    
    Args:
        video_path (str): Path to the video file
        door_coords (dict): Door coordinates {door_name: [x, y]}
        output_dir (str, optional): Output directory for results
        window_size (int, optional): Size of door detection window
        batch_processing (bool, optional): Whether to process frames in batches
        
    Returns:
        dict: Door traces {door_name: [motion_values]}
    """
    print(f'Extracting door traces from: {video_path}')
    
    # Open the video
    vid = cv2.VideoCapture(video_path)
    
    # Get total number of frames
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Pre-compute door regions to avoid repeated calculations
    door_regions = {}
    half_window = window_size // 2
    
    for door_key, coords in door_coords.items():
        door_regions[door_key] = (
            slice(coords[1] - half_window, coords[1] + half_window),
            slice(coords[0] - half_window, coords[0] + half_window)
        )
    
    # Initialize output arrays
    door_traces = {door_key: np.zeros(total_frames, dtype=np.float32) 
                  for door_key in door_coords.keys()}
    
    if batch_processing:
        # Batch processing mode
        batch_size = 1000
        
        for batch_start in tqdm(range(0, total_frames, batch_size), 
                               desc="Processing batches"):
            batch_end = min(batch_start + batch_size, total_frames)
            
            frames = []
            vid.set(cv2.CAP_PROP_POS_FRAMES, batch_start)
            
            for _ in range(batch_end - batch_start):
                ret, frame = vid.read()
                if not ret:
                    break
                frames.append(frame)
            
            if not frames:
                break
                
            frames = np.array(frames)
            
            if len(frames) < 2:
                continue
                
            prev_frame = frames[0]
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            for i, frame in enumerate(frames[1:], start=1):
                frame_idx = batch_start + i
                if frame_idx >= total_frames:
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                for door_key, region in door_regions.items():
                    door_diff = cv2.absdiff(
                        prev_gray[region],
                        gray[region]
                    )
                    
                    _, door_thresh = cv2.threshold(door_diff, 7, 255, cv2.THRESH_BINARY)
                    door_traces[door_key][frame_idx] = np.sum(door_thresh)
                
                prev_gray = gray
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    else:
        # Single frame processing mode
        ret, prev_frame = vid.read()
        if not ret:
            raise ValueError("Could not read first frame")
            
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        for frame_idx in tqdm(range(1, total_frames), desc="Processing frames"):
            ret, frame = vid.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            for door_key, region in door_regions.items():
                door_diff = cv2.absdiff(
                    prev_gray[region],
                    gray[region]
                )
                
                _, door_thresh = cv2.threshold(door_diff, 7, 255, cv2.THRESH_BINARY)
                door_traces[door_key][frame_idx] = np.sum(door_thresh)
            
            prev_gray = gray
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Clean up
    vid.release()
    cv2.destroyAllWindows()
    
    # Save results
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(video_path).split('.')[0] + '_doorTraces.pkl')
    else:
        output_path = video_path.split('.')[0] + '_doorTraces.pkl'
        
    with open(output_path, 'wb') as f:
        pickle.dump(door_traces, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Door traces saved to: {output_path}")
    return door_traces


def process_video(video_path, output_dir=None, initial_coords=DEFAULT_DOOR_COORDS):
    """
    Process a single video file through the complete workflow.
    
    This is a convenience function that runs all steps in sequence:
    1. Door position selection
    2. Motion trace extraction
    
    Args:
        video_path (str): Path to the video file
        output_dir (str, optional): Output directory for results
        initial_coords (dict, optional): Initial door coordinates
        
    Returns:
        dict: Door traces {door_name: [motion_values]}
    """
    print(f"Processing video: {video_path}")
    
    # Step 1: Select door coordinates
    print("Select door positions...")
    door_coords = select_door_coords(video_path, initial_coords)
    
    # Step 2: Extract door traces
    print("Extracting door traces...")
    door_traces = extract_door_traces(
        video_path, 
        door_coords,
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
        if output_dir:
            output_path = os.path.join(output_dir, os.path.basename(video_path).split('.')[0] + '_doorTraces.pkl')
        else:
            output_path = video_path.split('.')[0] + '_doorTraces.pkl'
        
        if os.path.exists(output_path):
            print(f"Door traces already exist for {video_path}. Skipping...")
            with open(output_path, 'rb') as f:
                results[video_path] = pickle.load(f)
        else:
            results[video_path] = process_video(video_path, output_dir, initial_coords)
    
    return results
