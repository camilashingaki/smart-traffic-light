def crop_traffic_area(frame):

    x1, y1 = 80, 50
    x2, y2 = 1200, 650

    frame_crop_traffic = frame[y1:y2, x1:x2]

    return frame_crop_traffic