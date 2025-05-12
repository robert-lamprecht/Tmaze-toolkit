def pad_movement(floor, window_pad):
    """
    Pad movement events in a binary array by connecting nearby events within a window.
    
    Args:
        floor (list/array): Binary array where 1 indicates movement
        window_pad (int): Number of frames to look for nearby events
        
    Returns:
        list: Padded binary array with connected events
    """
    floor = floor.copy()  # Work on a copy to avoid modifying the original
    n = len(floor)
    
    # First pass: Forward padding
    for x in range(n - window_pad):
        if floor[x] == 1:
            # Look ahead for the next 1 within the window
            for y in range(x + 1, min(x + window_pad + 1, n)):
                if floor[y] == 1:
                    # Fill in all values between x and y
                    floor[x:y+1] = 1
                    break
    
    # Second pass: Backward padding
    for x in range(n - 1, window_pad - 1, -1):
        if floor[x] == 1:
            # Look back for the previous 1 within the window
            for y in range(x - 1, max(x - window_pad - 1, -1), -1):
                if floor[y] == 1:
                    # Fill in all values between y and x
                    floor[y:x+1] = 1
                    break
    
    return floor