import pandas as pd
import glob
import pickle
import os

import os
import pickle

def save_outDict(outDict, jsonFileLocation):
    """
    Save the outDict to a pickle file at the location specified by jsonFileLocation
    
    Parameters:
    -----------
    outDict : dict
        Dictionary containing trial data with X and Y trajectories
    jsonFileLocation : str
        Path to the original JSON file (used to determine save location and filename)
    """
    dir_path = os.path.dirname(jsonFileLocation)
    filename = os.path.basename(jsonFileLocation)  # More reliable than split
    
    # Parse filename components
    parts = filename.split('_')
    if len(parts) < 3:
        print(f"Warning: Unexpected filename format: {filename}")
        animal_id = "UNKNOWN"
        date_id = "UNKNOWN"
    else:
        animal_id = parts[1]  # Second item is the animal id
        date_id = parts[2].split('.')[0]  # Third item, remove extension
        
        # Remove trailing * if present
        if date_id.endswith('*'):
            date_id = date_id[:-1]
    
    # Create pickle filename
    pkl_filename = f"{animal_id}_{date_id}_withTrajectory.pkl"
    pkl_file_path = os.path.join(dir_path, pkl_filename)
    
    # Save dictionary
    try:
        with open(pkl_file_path, 'wb') as f:
            pickle.dump(outDict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Successfully saved outDict to {pkl_file_path}")
        return pkl_file_path
    except Exception as e:
        print(f"Error saving outDict: {e}")
        return None


def load_outDict(pkl_file_path):
    """
    Load a previously saved outDict from a pickle file
    
    Parameters:
    -----------
    pkl_file_path : str
        Path to the pickle file
        
    Returns:
    --------
    dict : The loaded outDict
    """
    try:
        with open(pkl_file_path, 'rb') as f:
            outDict = pickle.load(f)
        print(f"Successfully loaded outDict from {pkl_file_path}")
        return outDict
    except Exception as e:
        print(f"Error loading outDict: {e}")
        return None

def add_trajectories(json_dict, dlc_dict, eventTimes):
    """
    Add the trajectories to the json dictionary
    """
    trial_starts = eventTimes['trial_start_frame'].tolist()
    trial_ends = eventTimes['trial_end_frame'].tolist()

    for i in range(len(json_dict) - 1):
        json_dict[i]['trajectory'] = dlc_dict[trial_starts[i]:trial_ends[i]]
    return json_dict

def load_json_files(jsonFileLocation):
    """
    Load the json files related to the video
    jsonFileLocation: string, the location of the json files
    A star (*) can be used to select multiple files for the same video
    Returns:
        outDict: dictionary, the dictionary containing the json files
    """
    outDict = {}
    outDict['originalFiles'] = []
    
    #load the json files related to the video

    json_files = glob.glob(jsonFileLocation)
    json_files.sort()
    print(f"Found {len(json_files)} json files")

    filename = jsonFileLocation.split('\\')[-1]
    animal_id = filename.split('_')[1] # Split by _ and the second item is the animal id
    print(f"Animal ID: {animal_id}")

    n = 0

    for file in json_files:
        print('Working on file {}'.format(file))
        outDict['originalFiles']
        outDict['originalFiles'].append(file)
        json_dict = pd.read_json(file)
        for i in range(len(json_dict[animal_id])):
            trialID = 'trial{}'.format(i+1) #Refine the id
            outDict[n] = json_dict[animal_id][trialID]
            n += 1

    return outDict





