def crop_pedestrian_area(frame):

    x1, y1 = 80, 50
    x2, y2 = 300, 600

    frame_crop_pedestrian = frame[y1:y2, x1:x2]

    return frame_crop_pedestrian