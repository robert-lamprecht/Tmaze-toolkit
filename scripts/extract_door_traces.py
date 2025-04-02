import argparse
from tmaze_toolkit.data.extraction import selectDoorCoords, extractDoorTraces, initial_coords
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import os

def main():
    parser = argparse.ArgumentParser(description="Extract door motion traces from videos")
    parser.add_argument("--output", "-o", help="Output directory for results", default="data/processed")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Select videos using tkinter directly
    Tk().withdraw()
    video_paths = askopenfilenames(
        initialdir='.', 
        filetypes=[
            ('MP4 Files', '*.mp4'),
            ('AVI Files', '*.avi'),
            ('All Files', '*.*')
        ]
    )
    
    # If files were selected, process them
    if video_paths:
        # Get door coordinates for each video
        door_coords = {}
        for file in video_paths:
            door_coords[file] = selectDoorCoords(file, initial_coords=initial_coords)
            
        # Process each video
        for file in video_paths:
            if os.path.exists(file.split('.')[0]+'_doorTraces.pkl'):
                print('Door traces already extracted for {}'.format(file))
                print('Skipping...')
            else:
                extractDoorTraces(file, door_coords=door_coords)
    else:
        print("No videos selected. Exiting.")

if __name__ == "__main__":
    main()