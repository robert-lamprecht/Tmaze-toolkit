import cv2
import numpy as np
import os

def select_corners(videoFile):
    # Open video and grab middle frame
    vid = cv2.VideoCapture(videoFile)
    if not vid.isOpened():
        print(f"Error: Could not open video {videoFile} for corner selection.")
        return None
    
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_pos = min(int(length/2), length - 1) if length > 0 else 0
    vid.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_pos)
    ret, frame = vid.read()
    if not ret:
        print(f"Error: Could not read frame from {videoFile}.")
        vid.release()
        return None
    
    # Create window for corner selection
    window_name = "Select 4 corners (click in order, press ENTER when done)"
    cv2.namedWindow(window_name)
    
    # Store coordinates and current corner
    corners = {}
    current_corner = 1
    
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_corner, corners, frame, display_img
        
        if event == cv2.EVENT_LBUTTONDOWN and current_corner <= 4:
            # Add corner to the dictionary
            corners[f"corner{current_corner}"] = (x, y)
            print(f"Corner {current_corner} selected at ({x}, {y})")
            current_corner += 1
            
            # Update display image
            display_img = frame.copy()
            for i in range(1, min(current_corner, 5)):
                cx, cy = corners[f"corner{i}"]
                cv2.circle(display_img, (cx, cy), 5, (0, 0, 255), -1)  # Red circle
                cv2.putText(display_img, f"{i}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2)
    
    # Set mouse callback
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Make a copy for displaying
    display_img = frame.copy()
    
    # Main loop
    while True:
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(1) & 0xFF
        
        # Press Enter to finish when all 4 corners are selected
        if (key == 13 and current_corner > 4) or key == 27:  # Enter or Esc
            break
    
    cv2.destroyAllWindows()
    vid.release()
    
    # Check if all 4 corners were selected
    if current_corner <= 4:
        print(f"Warning: Only {current_corner-1} corners were selected. Need 4 corners.")
    
    return corners
    
def drawLine(point1, point2):
    """
    write the equation of the line between two points
    """
    x1, y1 = point1
    x2, y2 = point2
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return [slope, intercept]

def calculateIntersection(line1, line2):
    """
    Calculate the intersection of two lines
    """
    slope1, intercept1 = line1
    slope2, intercept2 = line2
    x = (intercept2 - intercept1) / (slope1 - slope2)   
    y = slope1 * x + intercept1
    return [x, y]

def calculate_length(point1, point2):
    """
    Calculate the length of the line between two points
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def normalize_points(x, y, corners):
    """
    Normalize the points to the corners of the rectangle
    Input:
        x: x coordinates of the points
        y: y coordinates of the points
        corners: dictionary of the corners
    Output:
        x: normalized x coordinates
        y: normalized y coordinates
    """
    input_point = (x, y)

    point1 = corners['corner1']
    point2 = corners['corner2']
    point3 = corners['corner3']
    point4 = corners['corner4']


    #Draw a line between points 1 and 3
    topline = drawLine(point1, point3)
    
    bottomline = drawLine(point2, point4)
    
    rightline = drawLine(point1, point2)
   
    leftline = drawLine(point3, point4)
   
    # Calculate intersection between topline and bottomline
    intersectionY = calculateIntersection(topline, bottomline)
  
    intersectionX = calculateIntersection(rightline, leftline)
    

    # Draw a line between intersectionY and input_point
    yLine = drawLine(intersectionY, input_point)
   
    # Calculate intersection between intersectionX and input_point
    xLine = drawLine(intersectionX, input_point)
    

    # find intersection between yLine and rightline
    intersectionYwithRightline = calculateIntersection(yLine, rightline)
    

    # find intersection between xLine and topline
    intersectionXwithTopline = calculateIntersection(xLine, topline)


    toplineLength = calculate_length(point1, point3)
    bottomlineLength = calculate_length(point2, point4)
    rightlineLength = calculate_length(point1, point2)
    leftlineLength = calculate_length(point3, point4)

    # Find the ratio between intersectionXwithTopline and entire length of topline
    ratioX = intersectionXwithTopline[0] / toplineLength
    ratioY = intersectionYwithRightline[1] / rightlineLength


    return ratioX, ratioY

def input_numpy_array(x, y, corners):
    """
    Input:
        x: x coordinates of the points
        y: y coordinates of the points
        corners: dictionary of the corners
    """
    x_new = np.copy(x)
    y_new = np.copy(y)
    for i in range(len(x)):
        x_new[i], y_new[i] = normalize_points(x[i], y[i], corners)
    return x_new, y_new

def normalize_trajectory(outDict, videoFile):
    # Get corners from video
    corners = select_corners(videoFile)
    if not corners:
        print("Failed to get corners, cannot normalize")
        return
    
    # Get the DLC scorer name 
    scorer = outDict[0]['trajectory'].columns.get_level_values('scorer')[0]
     # Get all body parts
    body_parts = outDict[0]['trajectory'].columns.get_level_values('bodyparts').unique()
    # For each trial in the dictionary
    for trial_id in range(0, len(outDict) - 1):
        # Create a copy for the optimized trajectory
        outDict[trial_id]['trajectoryOptomized'] = outDict[trial_id]['trajectory'].copy()
        # Process each body part
        for body_part in body_parts:
            x_original = np.copy(outDict[trial_id]['trajectory'][scorer, body_part, 'x'].values)
            y_original = np.copy(outDict[trial_id]['trajectory'][scorer, body_part, 'y'].values)
            #x, y = input_numpy_array(x, y, corners)
           
            x_transformed, y_transformed = input_numpy_array(x_original, y_original, corners)

            """
            # Update with transformed coordinates
            outDict[trial_id]['trajectoryOptomized'].loc[:, (scorer, body_part, 'x')] = x  # Replace with transformed x
            outDict[trial_id]['trajectoryOptomized'].loc[:, (scorer, body_part, 'y')] = y  # Replace with transformed y
            """

            outDict[trial_id]['trajectoryOptomized'][(scorer, body_part, 'x')] = x_transformed  # Replace with transformed x
            outDict[trial_id]['trajectoryOptomized'][(scorer, body_part, 'y')] = y_transformed  # Replace with transformed y
    return outDict