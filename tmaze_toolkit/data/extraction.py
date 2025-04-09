import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm
from tkinter import Tk
from multiprocessing import Pool, cpu_count

# Define initial coordinates for door centers
initial_coords = {
    'door1': [14, 36],
    'door2': [13, 125],
    'door3': [217, 183],
    'door4': [223, 16],
    'floor1': [50, 90],
    'floor2': [200, 90]
}

# Define colors for drawing rectangles
BLUE = [255, 0, 0]
RED = [0, 0, 255]

# Mouse callback function for selecting door coordinates
def mouse(event, x, y, flags, params):
    global move_rectangle, BLUE, fg, bg, bgCopy, final_coords, door, rows, cols
    # Start moving the rectangle on left button down
    if event == cv2.EVENT_LBUTTONDOWN:
        move_rectangle = True
    # Move the rectangle with the mouse
    elif event == cv2.EVENT_MOUSEMOVE:
        try:
            if move_rectangle:
                bg = bgCopy.copy()
                cv2.rectangle(bg, (x - int(0.5 * cols), y - int(0.5 * rows)),
                              (x + int(0.5 * cols), y + int(0.5 * rows)), BLUE, -1)
        except UnboundLocalError:
            pass
    # Stop moving the rectangle on left button up
    elif event == cv2.EVENT_LBUTTONUP:
        move_rectangle = False
        cv2.rectangle(bg, (x - int(0.5 * cols), y - int(0.5 * rows)),
                      (x + int(0.5 * cols), y + int(0.5 * rows)), BLUE, -1)
        final_coords[door] = [x, y]

# Function to select door coordinates from a video
def selectDoorCoords(video, initial_coords=initial_coords):
    print('On video {}'.format(video))
    vid = cv2.VideoCapture(video)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.set(cv2.CAP_PROP_POS_FRAMES, int(length / 2))
    ret, frame = vid.read()

    global fg, bg, bgCopy, move_rectangle, BLUE, RED, final_coords, door, rows, cols
    fg = frame[:14, :14]
    bg = frame
    bgCopy = bg.copy()
    move_rectangle = False
    final_coords = {}

    # Function to draw the initial coordinate rectangle
    def draw_initial_coord(k, bg):
        cv2.rectangle(bg, (initial_coords[k][0] - int(0.5 * cols), initial_coords[k][1] - int(0.5 * rows)),
                      (initial_coords[k][0] + int(0.5 * cols), initial_coords[k][1] + int(0.5 * rows)), RED, -1)

    rows, cols = fg.shape[:2]

    # Loop through each door to set coordinates
    for door in initial_coords.keys():
        cv2.namedWindow('draw {} and press ENTER'.format(door))
        cv2.setMouseCallback('draw {} and press ENTER'.format(door), mouse)
        while True:
            cv2.imshow('draw {} and press ENTER'.format(door), bg)
            draw_initial_coord(door, bg)
            k = cv2.waitKey(1)
            if k == 13 & 0xFF:
                break
        cv2.destroyAllWindows()

    # Finalize coordinates
    for door in initial_coords:
        try:
            print(door, final_coords[door])
        except KeyError:
            final_coords[door] = initial_coords[door]
            print(door, final_coords[door])

    return final_coords

# Function to process a batch of frames
def process_frame_batch(batch):
    results = []
    for frame1, frame2, dc in batch:
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # Compute the absolute difference between frames
        diff = cv2.absdiff(gray1, gray2)
        # Apply a binary threshold to the difference
        _, thresh = cv2.threshold(diff, 7, 255, cv2.THRESH_BINARY)

        doorImages = {}
        # Extract door regions from the thresholded image
        for door in range(4):
            doorkey = 'door{}'.format(door + 1)
            doorImages[doorkey] = thresh[dc[doorkey][1] - 7: dc[doorkey][1] + 7, dc[doorkey][0] - 7: dc[doorkey][0] + 7]

        # Compute the sum of pixel values in door regions
        doorTraces = {doorkey: np.sum(doorImages[doorkey]) for doorkey in doorImages}
        results.append(doorTraces)
    return results

# Main function to extract door traces from a video
def extractDoorTraces(video, door_coords, ret = False):
    #Optional Set ret to true to return the original door traces array
    print('On video {}'.format(video))
    vid = cv2.VideoCapture(video)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = vid.read()
    dc = door_coords[video]

    frames = []
    batch_size = 100  # Set batch size for processing
    batch = []
    
    # Read frames in batches
    for frameCounter in tqdm(range(length - 1)):
        ret, frame2 = vid.read()
        if not ret:
            break
        batch.append((frame, frame2, dc))
        frame = frame2
        if len(batch) >= batch_size:
            frames.append(batch)
            batch = []

    # Add remaining frames to the batch
    if batch:
        frames.append(batch)

    # Use multiprocessing to process frames in parallel
    with Pool(cpu_count()) as pool:
        all_results = pool.map(process_frame_batch, frames)
        pool.close()
        pool.join()  # Ensure the pool is properly closed and joined

    # Initialize door traces dictionary
    doorTraces = {f'door{door+1}': [] for door in range(4)}
    # Aggregate results from all batches
    for results in all_results:
        for result in results:
            for key in result:
                doorTraces[key].append(result[key])

    # Release video capture object and close any OpenCV windows
    vid.release()
    cv2.destroyAllWindows()
    
    
    # Save the door traces to a pickle file
    output_file = video.split('.')[0] + '_doorTraces.pkl'
    try:
        print("Starting save operation...")
        file_size_estimate = sum(len(v) for v in doorTraces.values()) * 8 / (1024*1024)  # Rough size in MB
        print(f"Estimated file size: ~{file_size_estimate:.2f} MB")
        start_time = time.time()
        with open(output_file, 'wb') as f:
            pickle.dump(doorTraces, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Save completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Failed to save door traces: {e}")

    if ret == True: 
        return doorTraces
