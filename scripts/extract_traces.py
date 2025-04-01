import argparse
from tmaze_toolkit.data.extraction import select_video_files, batch_process_videos

def main():
    parser = argparse.ArgumentParser(description="Extract door motion traces from videos")
    parser.add_argument("--output", "-o", help="Output directory for results", default="data/processed")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Select videos and process them
    video_paths = select_video_files()
    
    # If files were selected, process them
    if video_paths:
        batch_process_videos(video_paths, output_dir=args.output)
    else:
        print("No videos selected. Exiting.")

if __name__ == "__main__":
    main()