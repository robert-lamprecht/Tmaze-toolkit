import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_trajectory(jsonFile, videoFile, lick='left(V1)'):
    """
    Plot the trajectory of the animal in the video using X and Y data in jsonFile
    
    Parameters:
    -----------
    jsonFile : dict
        Dictionary with trial data where each trial (numeric key) contains 'X' and 'Y' Series
    videoFile : str
        Path to the video file
    lick : str
        Filter trials by lick type (e.g., 'left(V1)', 'right(V2)')
    """
    video = cv2.VideoCapture(videoFile)
    ret, frame = video.read()
    
    if not ret:
        print(f"Error: Could not read video file: {videoFile}")
        return
    
    outDict = jsonFile
    
    for k in outDict:
<<<<<<< HEAD
        if isinstance(k, int):  # Only process numeric trial keys
            if 'X' in outDict[k] and 'Y' in outDict[k] and outDict[k]['lick'] == lick:
                plt.figure(figsize=(10, 8))
=======
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
>>>>>>> b0be84f7cc7ed33273fb938588706bf1f2a2e924
                plt.imshow(frame, cmap='gray')
                
                # Get X and Y data
                x_data = outDict[k]['X'].values
                y_data = outDict[k]['Y'].values
                
                # Plot trajectory
                plt.plot(x_data, y_data, 'b-', linewidth=2)
                plt.plot(x_data[0], y_data[0], 'go', markersize=10, label='Start')  # Start point
                plt.plot(x_data[-1], y_data[-1], 'ro', markersize=10, label='End')  # End point
                
                plt.title(f'Trial {outDict[k]["trial_number"]} - {lick}')
                plt.legend()
                plt.tight_layout()
                plt.show()
                plt.close()
    
    video.release()


def plot_trajectory_all_trials(jsonFile, videoFile, lick='left(V1)'):
    """
    Plot all trajectories on a single plot with a colorbar showing trial progression
    
    Parameters:
    -----------
    jsonFile : dict
        Dictionary with trial data where each trial (numeric key) contains 'X' and 'Y' Series
    videoFile : str
        Path to the video file
    lick : str
        Filter trials by lick type (e.g., 'left(V1)', 'right(V2)')
    """
    video = cv2.VideoCapture(videoFile)
    ret, frame = video.read()
    
    if not ret:
        print(f"Error: Could not read video file: {videoFile}")
        return
    
    outDict = jsonFile  
    
    # Create a figure with subplots to accommodate colorbar
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(frame, cmap='gray')
    
    # Get all valid trials
    valid_trials = [k for k in outDict 
                   if isinstance(k, int) 
                   and 'X' in outDict[k] 
                   and 'Y' in outDict[k] 
                   and outDict[k]['lick'] == lick]
    
    if not valid_trials:
        print(f"No valid trials found for lick type: {lick}")
        video.release()
        return
    
    trial_numbers = [outDict[k]['trial_number'] for k in valid_trials]
    
    # Create a colormap
    cmap = plt.cm.rainbow
    norm = plt.Normalize(min(trial_numbers), max(trial_numbers))
    
    # Plot each trajectory with color based on trial number
    for k in valid_trials:
        x_data = outDict[k]['X'].values
        y_data = outDict[k]['Y'].values
        
        # Get color based on trial number
        color = cmap(norm(outDict[k]['trial_number']))
        
        # Plot trajectory
        ax.plot(x_data, y_data, '-', color=color, linewidth=2, alpha=0.7)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', 
                       label='Trial Number', pad=0.01)
    
    # Set title and adjust layout
    ax.set_title(f'All {lick} Trials - Trajectory Overlay')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    video.release()


def plot_trajectory_by_decision(jsonFile, videoFile):
    """
    Plot trajectories separated by correct/incorrect decisions
    
    Parameters:
    -----------
    jsonFile : dict
        Dictionary with trial data where each trial (numeric key) contains 'X' and 'Y' Series
    videoFile : str
        Path to the video file
    """
    video = cv2.VideoCapture(videoFile)
    ret, frame = video.read()
    
    if not ret:
        print(f"Error: Could not read video file: {videoFile}")
        return
    
    outDict = jsonFile
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot correct trials
    ax1.imshow(frame, cmap='gray')
    ax1.set_title('Correct Decisions')
    
    # Plot incorrect trials
    ax2.imshow(frame, cmap='gray')
    ax2.set_title('Incorrect Decisions')
    
    correct_count = 0
    incorrect_count = 0
    
    for k in outDict:
        if isinstance(k, int) and 'X' in outDict[k] and 'Y' in outDict[k]:
            x_data = outDict[k]['X'].values
            y_data = outDict[k]['Y'].values
            
            if outDict[k]['decision'] == 'correct':
                ax1.plot(x_data, y_data, 'g-', linewidth=1.5, alpha=0.6)
                correct_count += 1
            else:
                ax2.plot(x_data, y_data, 'r-', linewidth=1.5, alpha=0.6)
                incorrect_count += 1
    
    ax1.text(10, 30, f'n = {correct_count}', fontsize=14, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(10, 30, f'n = {incorrect_count}', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    video.release()


def plot_trajectory_summary(jsonFile, videoFile):
    """
    Create a summary plot with multiple views of trajectory data
    
    Parameters:
    -----------
    jsonFile : dict
        Dictionary with trial data where each trial (numeric key) contains 'X' and 'Y' Series
    videoFile : str
        Path to the video file
    """
    video = cv2.VideoCapture(videoFile)
    ret, frame = video.read()
    
    if not ret:
        print(f"Error: Could not read video file: {videoFile}")
        return
    
    outDict = jsonFile
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. All trajectories overlay
    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(frame, cmap='gray')
    ax1.set_title('All Trajectories')
    
    # 2. Correct vs Incorrect
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(frame, cmap='gray')
    ax2.set_title('Correct (Green)')
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(frame, cmap='gray')
    ax3.set_title('Incorrect (Red)')
    
    # 4. Statistics
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    correct_count = 0
    incorrect_count = 0
    left_count = 0
    right_count = 0
    
    for k in outDict:
        if isinstance(k, int) and 'X' in outDict[k] and 'Y' in outDict[k]:
            x_data = outDict[k]['X'].values
            y_data = outDict[k]['Y'].values
            
            # Count statistics
            if outDict[k]['decision'] == 'correct':
                correct_count += 1
                color = 'g'
                ax2.plot(x_data, y_data, 'g-', linewidth=1, alpha=0.6)
            else:
                incorrect_count += 1
                color = 'r'
                ax3.plot(x_data, y_data, 'r-', linewidth=1, alpha=0.6)
            
            if 'left' in outDict[k]['lick'].lower():
                left_count += 1
            else:
                right_count += 1
            
            # Plot on overlay
            ax1.plot(x_data, y_data, color=color, linewidth=1, alpha=0.4)
    
    # Display statistics
    total = correct_count + incorrect_count
    accuracy = (correct_count / total * 100) if total > 0 else 0
    
    stats_text = f"""
    Trial Statistics
    ================
    Total Trials: {total}
    Correct: {correct_count}
    Incorrect: {incorrect_count}
    Accuracy: {accuracy:.1f}%
    
    Lick Distribution
    ================
    Left: {left_count}
    Right: {right_count}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Trajectory Analysis Summary', fontsize=16, fontweight='bold')
    plt.show()
    plt.close()
    
    video.release()