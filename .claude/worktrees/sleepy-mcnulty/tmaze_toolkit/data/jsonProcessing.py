import pandas as pd
import glob
import pickle
import os

def save_outDict(outDict, jsonFileLocation):
    "save the outDict to a json file at the location specified by jsonFileLocation"
    dir_path = os.path.dirname(jsonFileLocation)
    filename = jsonFileLocation.split('\\')[-1] #Last item in the path is the filename
    animal_id = filename.split('_')[1] # Split by _ and the second item is the animal id

    # Filename already esablished
    date_id = filename.split('_')[2]
    date_id = date_id.split('.')[0] # remove the .json ext
    if date_id[-1]=='*':
        date_id = date_id[0:-1] # remove the * if there is one. If it's a singular json file theres no need to remove a star.

    # Example file = BDY5_20240701_data.pkl
    pkl_filename = f"{animal_id}_{date_id}_data.pkl"

    #Create
    pkl_file_path = os.path.join(dir_path, pkl_filename)
    ## save new dictionary to data folder
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(outDict, f, protocol=pickle.HIGHEST_PROTOCOL)

    outDict[0]


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





