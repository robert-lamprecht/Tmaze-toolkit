import pickle

def openDoorTracesPkl(filepath):
    # Open the .pkl file and load its contents
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data