import numpy as np
import pandas as pd

def find_trial_length(events):
    trial_lengths = []
    trial_numbers = []
    n = 1
    for i in range(len(events)):
        trial_numbers.append(n)
        trial_lengths.append(events.iloc[i]['trial_end_frame'] - events.iloc[i]['trial_start_frame'])
        n += 1

    trial_duration = pd.DataFrame({'trial_number': trial_numbers, 'trial_duration': trial_lengths})
    return trial_duration

def save_trial_duration(trial_duration, filename):
    trial_duration.to_csv(filename, index=False)

def load_trial_duration(filename):
    return pd.read_csv(filename)

def find_percent_correct(outDict):
    correct_trials = 0
    for i in range(len(outDict) -1):
        if outDict[i]['decision'] == 'correct':
            correct_trials += 1
    return correct_trials / len(outDict)

