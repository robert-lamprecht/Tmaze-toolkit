import cv2
import pandas as pd

def visualize_video_with_stamps(video_path, stamps_df, output_path=None):
    """
    Visualize the video with stamps overlayed on the video

    Args:
        video_path (str): Path to the video file
        stamps_df (pd.DataFrame): DataFrame containing stamps information with columns
                                  'start', 'stop', and 'trial_number'.
        output_path (str): Path to save the output video, by default the video_path with '_stamped' added to the end of the filename
    """ 
    if output_path is None:
        output_path = video_path.rsplit('.', 1)[0] + '_stamped.mp4'

    # Read the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create video writer for {output_path}")
        cap.release()
        return

    # Filter out rows with None values and prepare stamp data for quick lookup
    valid_starts = stamps_df.dropna(subset=['start'])
    valid_ends = stamps_df.dropna(subset=['stop'])
    
    # Create sets and dictionaries for quick lookup
    start_frames = set(valid_starts['start'].astype(int))
    end_frames = set(valid_ends['stop'].astype(int))
    
    # Create dictionaries to map frame number to trial number
    start_frame_to_trial = pd.Series(valid_starts['trial_number'], index=valid_starts['start']).to_dict()
    end_frame_to_trial = pd.Series(valid_ends['trial_number'], index=valid_ends['stop']).to_dict()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color_start = (0, 255, 0)  # Green
    color_end = (0, 0, 255)    # Red
    thickness = 2
    position = (50, 50)  # Top-left corner

    # Process all frames from input video
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Draw stamps if the current frame matches a start or end frame
        text_to_draw = None
        color_to_use = None

        if frame_count in start_frames:
            trial_number = start_frame_to_trial[frame_count]
            text_to_draw = f"Trial {trial_number} Start"
            color_to_use = color_start
        elif frame_count in end_frames:
            trial_number = end_frame_to_trial[frame_count]
            text_to_draw = f"Trial {trial_number} End"
            color_to_use = color_end

        if text_to_draw:
            cv2.putText(frame, text_to_draw, position, font, font_scale, color_to_use, thickness, cv2.LINE_AA)

        out.write(frame)
        frame_count += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Stamped video saved to {output_path}")
    
    # Print summary of what was processed
    total_starts = len(valid_starts)
    total_ends = len(valid_ends)
    print(f"Processed {total_starts} trial starts and {total_ends} trial ends")



