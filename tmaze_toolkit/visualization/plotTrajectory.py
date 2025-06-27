import matplotlib.pyplot as plt
import numpy as np
import cv2
from tmaze_toolkit.processing.normalize import select_corners

def plot_trajectory(jsonFile, videoFile, lick = 'left(V1)', bodyPart = 'UpperSpine', confidence_threshold = 0.01):
    "plot the trajectory of the animal in the video"
    video = cv2.VideoCapture(videoFile)
    ret, frame = video.read()
    outDict = jsonFile
    
    for k in outDict:
        if type(k) == int:
            if outDict[k]['lick'] == lick:
                plt.figure()
                plt.imshow(frame,cmap='gray')
                filt_x = np.copy(outDict[k]['trajectory'].droplevel(0,axis=1)[bodyPart]['x'])
                filt_x[outDict[k]['trajectory'].droplevel(0,axis=1)[bodyPart]['likelihood'] < confidence_threshold] = np.nan
                filt_y = np.copy(outDict[k]['trajectory'].droplevel(0,axis=1)[bodyPart]['y'])
                filt_y[outDict[k]['trajectory'].droplevel(0,axis=1)[bodyPart]['likelihood'] < confidence_threshold] = np.nan
                
                """"
                filt_x1 = np.copy(outDict[k]['trajectory'].droplevel(0,axis=1)['TailBase']['x'])
                filt_x1[outDict[k]['trajectory'].droplevel(0,axis=1)['TailBase']['likelihood'] < confidence_threshold] = np.nan
                filt_y1 = np.copy(outDict[k]['trajectory'].droplevel(0,axis=1)['TailBase']['y'])
                filt_y1[outDict[k]['trajectory'].droplevel(0,axis=1)['TailBase']['likelihood'] < confidence_threshold] = np.nan
              
                """
              
              
                plt.plot(filt_x,filt_y, color='b')
                plt.title('Trial {}'.format(outDict[k]['trial_number']))
                plt.show()
                plt.close()

def plot_normalized_trajectory(jsonFile, videoFile, corners=None, lick='left(V1)', bodyPart='UpperSpine', confidence_threshold=0.01):
    """Plot the normalized trajectory of the animal in the video"""
    video = cv2.VideoCapture(videoFile)
    ret, frame = video.read()
    outDict = jsonFile
    
    # If corners aren't provided, ask user to select them
    if corners is None:
        corners = select_corners(videoFile)
        if not corners:
            print("Failed to get corners, cannot plot normalized trajectory")
            return
    
    # Create a figure with original frame
    plt.figure(figsize=(12, 6))
    
    for k in outDict:
        if isinstance(k, int):
            if outDict[k]['lick'] == lick and 'trajectoryOptomized' in outDict[k]:
                # Create two subplots - original and normalized
                plt.subplot(1, 2, 1)
                plt.imshow(frame, cmap='gray')
                plt.title(f'Original - Trial {outDict[k]["trial_number"]}')
                
                # Plot original trajectory for comparison
                filt_x = np.copy(outDict[k]['trajectory'].droplevel(0,axis=1)[bodyPart]['x'])
                filt_x[outDict[k]['trajectory'].droplevel(0,axis=1)[bodyPart]['likelihood'] < confidence_threshold] = np.nan
                filt_y = np.copy(outDict[k]['trajectory'].droplevel(0,axis=1)[bodyPart]['y'])
                filt_y[outDict[k]['trajectory'].droplevel(0,axis=1)[bodyPart]['likelihood'] < confidence_threshold] = np.nan
                
                
                plt.plot(filt_x, filt_y, 'r-', label=bodyPart)
                plt.legend()
                
                # Draw corner points on original
                for i, (corner_name, (x, y)) in enumerate(corners.items()):
                    plt.plot(x, y, 'go', markersize=10)
                    plt.text(x+5, y+5, f"{i+1}", color='g', fontsize=12)
                
                # Plot normalized trajectory
                plt.subplot(1, 2, 2)
                # Create a blank image for normalized space (0-1 coordinates)
                normalized_img = np.ones((400, 400, 3), dtype=np.uint8) * 200
                plt.imshow(normalized_img, extent=[0, 1, 1, 0])  # Note y-axis is flipped to match image coords
                plt.title(f'Normalized - Trial {outDict[k]["trial_number"]}')
                
                # Get normalized trajectory data
                scorer = outDict[k]['trajectoryOptomized'].columns.get_level_values('scorer')[0]
                
                # Get normalized UpperSpine data
                norm_x = np.copy(outDict[k]['trajectoryOptomized'][scorer, bodyPart, 'x'])
                norm_x[outDict[k]['trajectoryOptomized'][scorer, bodyPart, 'likelihood'] < confidence_threshold] = np.nan
                norm_y = np.copy(outDict[k]['trajectoryOptomized'][scorer, bodyPart, 'y'])
                norm_y[outDict[k]['trajectoryOptomized'][scorer, bodyPart, 'likelihood'] < confidence_threshold] = np.nan
                
                
                plt.plot(norm_x, norm_y, 'r-', label=bodyPart)
                
                # Draw a unit square to represent normalized space
                plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'g-', linewidth=2)
                plt.xlim(0, 1)
                plt.ylim(1, 0)  # Flip y-axis to match image coordinates
                plt.legend()
                
                plt.tight_layout()
                plt.show()
                plt.close()

def plot_trajectory_all_trials(jsonFile, videoFile, lick='left(V1)', bodyPart='UpperSpine', confidence_threshold=0.01):
    """Plot all trajectories on a single plot with a colorbar showing trial progression"""
    video = cv2.VideoCapture(videoFile)
    ret, frame = video.read()
    outDict = jsonFile  
    
    # Create a figure with subplots to accommodate colorbar
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame, cmap='gray')
    
    # Get all valid trials
    valid_trials = [k for k in outDict if isinstance(k, int) and 'trajectory' in outDict[k] and outDict[k]['lick'] == lick]
    trial_numbers = [outDict[k]['trial_number'] for k in valid_trials]
    
    # Create a colormap
    cmap = plt.cm.rainbow
    norm = plt.Normalize(min(trial_numbers), max(trial_numbers))
    
    # Plot each trajectory with color based on trial number
    for k in valid_trials:
        # Filter points by confidence
        filt_x = np.copy(outDict[k]['trajectory'].droplevel(0,axis=1)[bodyPart]['x'])
        filt_y = np.copy(outDict[k]['trajectory'].droplevel(0,axis=1)[bodyPart]['y'])
        mask = outDict[k]['trajectory'].droplevel(0,axis=1)[bodyPart]['likelihood'] < confidence_threshold
        filt_x[mask] = np.nan
        filt_y[mask] = np.nan
        
        # Get color based on trial number
        color = cmap(norm(outDict[k]['trial_number']))
        
        # Plot without adding to legend
        ax.plot(filt_x, filt_y, '-', color=color, linewidth=2)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', 
                       label='Trial Number', pad=0.01)
    
    # Set title and adjust layout
    ax.set_title(f'All {lick} Trials - {bodyPart}')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_multiple_bodyparts(jsonFile, videoFile, lick='left(V1)', bodyParts=['UpperSpine', 'TailBase'], confidence_threshold=0.01):
    """Plot multiple body parts for individual trials"""
    video = cv2.VideoCapture(videoFile)
    ret, frame = video.read()
    outDict = jsonFile
    
    # Use different colors for different body parts
    colors = plt.cm.tab10(np.linspace(0, 1, len(bodyParts)))
    
    for k in outDict:
        if isinstance(k, int):
            if outDict[k]['lick'] == lick:
                plt.figure(figsize=(10, 8))
                plt.imshow(frame, cmap='gray')
                plt.title(f'Trial {outDict[k]["trial_number"]} - Multiple Body Parts')
                
                # Plot each body part with a different color
                for i, bodyPart in enumerate(bodyParts):
                    # Skip if body part not found
                    if bodyPart not in outDict[k]['trajectory'].droplevel(0,axis=1).columns.levels[0]:
                        print(f"Warning: {bodyPart} not found in trial {k}")
                        continue
                        
                    # Filter points by confidence
                    filt_x = np.copy(outDict[k]['trajectory'].droplevel(0,axis=1)[bodyPart]['x'])
                    filt_y = np.copy(outDict[k]['trajectory'].droplevel(0,axis=1)[bodyPart]['y'])
                    mask = outDict[k]['trajectory'].droplevel(0,axis=1)[bodyPart]['likelihood'] < confidence_threshold
                    filt_x[mask] = np.nan
                    filt_y[mask] = np.nan
                    
                    # Plot with the current color
                    plt.plot(filt_x, filt_y, '-', color=colors[i], label=bodyPart)
                
                plt.legend(loc='best')
                plt.tight_layout()
                plt.show()
                plt.close() 
