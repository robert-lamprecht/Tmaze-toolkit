import pandas as pd
def load_deeplabcut_files(deeplabcut_file_location):
    """
    Load the deeplabcut files
    input:
        deeplabcut_file_location: string, the location of the deeplabcut file
    output:
        dlc_dict: dictionary, the dictionary containing the deeplabcut data
    """
    dlc_dict = pd.read_hdf(deeplabcut_file_location)
    return dlc_dict
