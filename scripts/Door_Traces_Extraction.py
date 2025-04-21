import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilenames


# user selects video files
Tk().withdraw()
videoFiles = askopenfilenames(initialdir=r'N:\TMAZE',defaultextension='.mp4',filetypes=[('Video Files', '*.mp4 *.avi'), ('All Files', '*.*')])

initial_coords = {} ### default locations for door centers
initial_coords['door1'] = [14, 36] # Floor Left
initial_coords['door2'] = [13, 125] # Floor Right
initial_coords['door3'] = [217, 183] # End Zone Top
initial_coords['door4'] = [223, 16]  # End Zone Bottom
# Adding Floor doors (Door 5 and 6) based on toolkit structure
initial_coords['door5'] = [50, 98] # Placeholder, adjust as needed
initial_coords['door6'] = [200, 98] # Placeholder, adjust as needed

# Declare global variables before assigning values
# Note: Using globals is generally discouraged; consider passing variables as arguments
# or using classes if the script complexity increases.
global BLUE, RED

# Assign values to the global variables
BLUE = [255, 0, 0]
RED = [0, 0, 255]


def mouse(event,x,y,flags,params): ## defining the callback for the door selection
    global move_rectangle, BLUE, fg, bg, bgCopy, final_coords, door, rows, cols
    #draw rectangle where x,y is rectangle center
    if event == cv2.EVENT_LBUTTONDOWN:
        move_rectangle = True

    elif event == cv2.EVENT_MOUSEMOVE:
        try:
            if move_rectangle:
                bg = bgCopy.copy() #!! your image is reinitialized with initial one
                cv2.rectangle(bg,(x-int(0.5*cols),y-int(0.5*rows)),
                (x+int(0.5*cols),y+int(0.5*rows)),BLUE, -1)
        except UnboundLocalError:
            pass

    elif event == cv2.EVENT_LBUTTONUP:
        move_rectangle = False
        cv2.rectangle(bg,(x-int(0.5*cols),y-int(0.5*rows)),
        (x+int(0.5*cols),y+int(0.5*rows)),BLUE, -1)
        final_coords[door] = [x,y]

def selectDoorCoords(video, initial_coords=initial_coords):
    print('On video {}'.format(video))
    vid = cv2.VideoCapture(video)
    if not vid.isOpened():
        print(f"Error: Could not open video {video} for coordinate selection.")
        return None

    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # Ensure we don't go past the end if the video is short
    middle_frame_pos = min(int(length/2), length - 1) if length > 0 else 0
    vid.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_pos)
    ret, frame = vid.read()
    if not ret:
        print(f"Error: Could not read frame from {video}. Skipping coordinate selection.")
        vid.release()
        return None # Return None if frame reading fails

    global fg, bg, bgCopy, move_rectangle, BLUE, RED, final_coords, door, rows, cols

    # Display names mapping (only for GUI display)
    # Include Door 5 and 6 in display names
    display_names = {
        'door1': 'Floor Left (1)',
        'door2': 'Floor Right (2)',
        'door3': 'Endzone Top (3)',
        'door4': 'Endzone Bottom (4)',
        'door5': 'Floor Door 1 (5)',
        'door6': 'Floor Door 2 (6)'
    }

    fg = frame[:14,:14] ## door size is about 15x15 pixels
    bg = frame ## middle frame of video
    bgCopy = bg.copy()
    move_rectangle = False
    final_coords = {} # Reset final_coords for each video

    # Function to draw the initial coordinate rectangle
    # Ensure it handles cases where the key might be missing from initial_coords
    def draw_initial_coord(k, display_img):
        if k in initial_coords:
             # Make sure coords are integers
             center_x = int(initial_coords[k][0])
             center_y = int(initial_coords[k][1])
             half_cols = int(0.5*cols)
             half_rows = int(0.5*rows)
             cv2.rectangle(display_img,
                         (center_x - half_cols, center_y - half_rows),
                         (center_x + half_cols, center_y + half_rows),
                         RED, -1)
        else:
            print(f"Warning: Initial coordinates for {k} not found.")

    # Check if fg was successfully read
    if fg.shape[0] == 0 or fg.shape[1] == 0:
         print(f"Error: Could not determine door size (fg shape is {fg.shape}). Using default 14x14.")
         rows, cols = 14, 14
    else:
         rows, cols = fg.shape[:2]

    # Iterate through the keys present in initial_coords
    for door_key in initial_coords.keys():
        # Assign door_key to the global `door` variable for the mouse callback
        door = door_key
        window_name = f'Draw {display_names.get(door_key, door_key)} and press ENTER' # Use .get for safety
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse)
        while True:
            # Create a fresh copy for drawing each loop iteration
            display_img = bg.copy()
            # Draw the *initial* position in red
            draw_initial_coord(door_key, display_img)
            # If user has already clicked for this door, draw the *new* position in blue
            if door_key in final_coords:
                 center_x = int(final_coords[door_key][0])
                 center_y = int(final_coords[door_key][1])
                 half_cols = int(0.5*cols)
                 half_rows = int(0.5*rows)
                 cv2.rectangle(display_img,
                         (center_x - half_cols, center_y - half_rows),
                         (center_x + half_cols, center_y + half_rows),
                         BLUE, -1)

            cv2.imshow(window_name, display_img)
            k = cv2.waitKey(1)
            if k == 13: # Enter key (no need for & 0xFF typically)
                break
            elif k == 27: # ESC key to cancel and use initial coord
                if door_key in final_coords: # Remove if user had clicked before ESC
                    del final_coords[door_key]
                break
        cv2.destroyAllWindows()

    # Finalize coordinates, ensuring all initial keys are present
    final_door_coords_for_video = {}
    print("--- Final Selected Coordinates ---")
    for door_key in initial_coords:
        # Use initial coordinate if not set by user (or user pressed ESC)
        final_door_coords_for_video[door_key] = final_coords.get(door_key, initial_coords[door_key])
        print(f"{display_names.get(door_key, door_key)}: {final_door_coords_for_video[door_key]}")

    vid.release() # Release video capture object
    return final_door_coords_for_video

# --- Removed select_frame_range function ---
# --- Removed prompt_for_crop function --- 

def extractDoorTraces(video, door_coords_for_video):
    print(f'\nExtracting traces for: {os.path.basename(video)}')
    vid = cv2.VideoCapture(video)
    if not vid.isOpened():
        print(f"Error: Could not open video {video}")
        return None

    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if length <= 1:
        print(f"Error: Video {video} has insufficient frames ({length}). Cannot calculate differences.")
        vid.release()
        return None

    num_frames_to_process = length - 1 # Process differences between N frames (N-1 differences)

    dc = door_coords_for_video # Use the coordinates passed for this specific video

    # Define number of doors based on the provided coordinates
    num_doors = len(dc)
    door_keys = list(dc.keys()) # Get the keys like ['door1', 'door2', ...]

    # Pre-compute door regions based on the actual keys present
    door_regions = {}
    region_half_size = 7 # Defines the 14x14 region
    for doorkey in door_keys:
        center_x, center_y = dc[doorkey]
        # Ensure coordinates are integers
        center_x = int(center_x)
        center_y = int(center_y)
        door_regions[doorkey] = (
            slice(max(0, center_y - region_half_size), center_y + region_half_size),
            slice(max(0, center_x - region_half_size), center_x + region_half_size)
        )

    # Pre-allocate arrays for better memory efficiency
    doorTraces = {key: np.zeros(num_frames_to_process, dtype=np.float32) for key in door_keys}

    # Read the first frame
    ret, prev_frame = vid.read()
    if not ret:
        print(f"Error: Could not read the first frame of {video}.")
        vid.release()
        return None
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Process frames sequentially
    for frame_idx in tqdm(range(num_frames_to_process), desc=f"Processing {os.path.basename(video)}"):
        ret, frame = vid.read()
        if not ret:
            print(f"\nWarning: Could not read frame {frame_idx + 2} (index {frame_idx+1}) from {video}. Stopping processing.")
            # Trim the doorTraces arrays to the number of frames successfully processed
            for key in door_keys:
                doorTraces[key] = doorTraces[key][:frame_idx]
            break # Stop processing this video

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Process each door region
        for doorkey in door_keys:
            region = door_regions[doorkey]

            # Ensure slices are valid before applying them
            prev_region_gray = prev_gray[region]
            current_region_gray = gray[region]

            # Check if regions are valid (e.g., coordinates not out of bounds)
            if prev_region_gray.size == 0 or current_region_gray.size == 0:
                # print(f"\nWarning: Invalid region extracted for {doorkey} at frame {frame_idx + 1}. Assigning 0 difference.")
                doorTraces[doorkey][frame_idx] = 0 # Assign 0 if region is invalid
                continue # Skip calculation for this door on this frame

            # Only process the small door region
            door_diff = cv2.absdiff(prev_region_gray, current_region_gray)
            _, door_thresh = cv2.threshold(door_diff, 7, 255, cv2.THRESH_BINARY)
            doorTraces[doorkey][frame_idx] = np.sum(door_thresh)

        prev_gray = gray # Update previous frame for the next iteration

        # Allow quitting with 'q' (optional, can be removed if not needed)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print("\nProcessing interrupted by user.")
        #     # Trim arrays if interrupted
        #     for key in door_keys:
        #         doorTraces[key] = doorTraces[key][:frame_idx+1]
        #     break

    vid.release()
    cv2.destroyAllWindows() # Close any OpenCV windows that might be open

    # --- Save results --- 
    # Construct output filename based on input video name
    base_name = os.path.splitext(os.path.basename(video))[0]
    output_dir = os.path.dirname(video) # Save in the same directory as the video
    output_file = os.path.join(output_dir, f'{base_name}_doorTraces.pkl')

    try:
        with open(output_file, 'wb') as f:
            pickle.dump(doorTraces, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved door traces to {output_file}")
    except Exception as e:
        print(f"Error saving door traces to {output_file}: {e}")
        return None # Indicate failure if saving fails

    return doorTraces

# Removed multiprocessing function for simplicity, can be added back

door_coords = {} # Dictionary to store coordinates for each video file

# --- Main loop to get coordinates for each video --- 
for file in videoFiles:
    print("-"*40)
    print(f"Step 1: Select Coordinates for: {os.path.basename(file)}")
    
    # Get coordinates for this specific video
    coords = selectDoorCoords(file, initial_coords=initial_coords)
    if coords:
        door_coords[file] = coords
    else:
        print(f"Skipping coordinate selection and extraction for {file}.")
        # Optionally remove the file from videoFiles list if selection fails
        # videoFiles.remove(file) # Be careful modifying list while iterating
        
    print("-"*40)

# --- Main loop to extract traces --- 
if __name__ == '__main__':
    print("\n" + "="*40)
    print("Step 2: Starting Door Trace Extraction")
    print("="*40)
    processed_files = 0
    failed_files = 0
    skipped_files = 0
    
    # Use a copy of the keys if modifying dict during iteration, or iterate over items
    videos_to_process = list(door_coords.keys())

    for file in videos_to_process:
        print(f"\nAttempting extraction for: {os.path.basename(file)}")
        
        # Check if coordinates were successfully obtained (redundant if we handle failure above)
        if file not in door_coords:
            print("--> Skipping (no coordinates found).") # Should not happen if handled earlier
            skipped_files += 1
            continue
            
        # Construct expected output filename
        base_name = os.path.splitext(os.path.basename(file))[0]
        output_dir = os.path.dirname(file)
        output_file = os.path.join(output_dir, f'{base_name}_doorTraces.pkl')
        
        # Check if traces already exist
        if os.path.exists(output_file):
            print(f"--> Skipping (traces file already exists: {output_file})")
            skipped_files += 1
        else:
            # Pass the specific coordinates for this file
            result = extractDoorTraces(file, door_coords_for_video=door_coords[file])
            if result is not None:
                processed_files += 1
            else:
                print(f"--> Extraction failed for {os.path.basename(file)}.")
                failed_files += 1
                
    print("\n" + "="*40)
    print("Extraction Summary")
    print("="*40)
    print(f"Successfully processed: {processed_files}")
    print(f"Failed:                 {failed_files}")
    print(f"Skipped (no coords/exists): {skipped_files}")
    print(f"Total videos selected initially: {len(videoFiles)}") 