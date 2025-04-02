import numpy as np
import cv2
import os
import pickle
from tqdm import tqdm
from tkinter import Tk     # from tkinter import Tk for Python 3.x

initial_coords = {} ### default locations for door centers
initial_coords['door1'] = [14, 36] # top left
initial_coords['door2'] = [13, 125] # bottom left
initial_coords['door3'] = [217, 183] # bottom right
initial_coords['door4'] = [223, 16]  # top right
initial_coords['floor1'] = [50, 90] # Floor 1
initial_coords['floor2'] = [200, 90] # Floor 2

global BLUE, RED

BLUE = [255,0,0]
RED = [0,0,255]


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
    vid = cv2.VideoCapture(video) ##  video
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.set(cv2.CAP_PROP_POS_FRAMES, int(length/2)) ## set video to middle frame
    ret, frame = vid.read() ## read middle frame of video

    global fg, bg, bgCopy, move_rectangle, BLUE, RED, final_coords, door, rows, cols
    fg = frame[:14,:14] ## door size is about 15x15 pixels
    bg = frame ## middle frame of video
    bgCopy = bg.copy()
    move_rectangle = False
    final_coords = {}
    def draw_initial_coord(k, bg):
        cv2.rectangle(bg,(initial_coords[k][0]-int(0.5*cols),initial_coords[k][1]-int(0.5*rows)),
        (initial_coords[k][0]+int(0.5*cols),initial_coords[k][1]+int(0.5*rows)),RED, -1)
    rows, cols = fg.shape[:2]


        
    for door in initial_coords.keys():
        cv2.namedWindow('draw {} and press ENTER'.format(door))
        cv2.setMouseCallback('draw {} and press ENTER'.format(door), mouse)
        while True:
            cv2.imshow('draw {} and press ENTER'.format(door), bg)
            draw_initial_coord(door, bg)
            k = cv2.waitKey(1)
            if k == 13 & 0xFF: ## enter key
                break
        cv2.destroyAllWindows()

    for door in initial_coords:
        try:
            print(door,final_coords[door])
        except KeyError:
            final_coords[door] = initial_coords[door] ## if the door was not moved, use the initial coordinates
            print(door,final_coords[door])

    return final_coords


def extractDoorTraces(video, door_coords):
    print('On video {}'.format(video))
    vid = cv2.VideoCapture(video) ##  video
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = vid.read()  ## read the first frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dc = door_coords[video]
    
    frameCounter = 0

    doorTraces = {}
    for door in range(4):
        doorkey = 'door{}'.format(door+1)
        doorTraces[doorkey] = []

    # Convert the first frame to grayscale
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for frameCounter in tqdm(range(length-1)):
        # Read the next frame
        ret, frame2 = vid.read()
        if not ret:
            break

        # Convert the current frame to grayscale
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
        # Compute the absolute difference between the current frame and the previous frame
        diff = cv2.absdiff(gray1, gray2) 

        # Apply a threshold to get the binary image
        _, thresh = cv2.threshold(diff, 7, 255, cv2.THRESH_BINARY)
        
        
        doorImages = {}
        for door in range(4):
            doorkey = 'door{}'.format(door+1)
            doorImages[doorkey] = thresh[dc[doorkey][1]-7:dc[doorkey][1]+7, dc[doorkey][0]-7:dc[doorkey][0]+7]  ## PARAMETERIZE IN FUTURE
        # cv2.imshow('door1',doorImages['door1'])
        # Optional: Apply some morphological operations to reduce noise
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Highlight the motion areas in the original frame
        #motion_areas = cv2.bitwise_and(frame2, frame2, mask=thresh)

        # # Display the original frame and the thresholded difference
        # cv2.imshow('Original', frame2)
        # cv2.imshow('Motion Detection', motion_areas)
        
        
        
        for door in range(4):
            doorkey = 'door{}'.format(door+1)
            doorTraces[doorkey].append(np.sum(doorImages[doorkey]))
    

        # Update the previous frame
        gray1 = gray2
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    vid.release()
    cv2.destroyAllWindows()
    
    
    ### save doorTraces
    with open(video.split('.')[0]+'_doorTraces.pkl', 'wb') as f:
        pickle.dump(doorTraces, f, protocol=pickle.HIGHEST_PROTOCOL)
