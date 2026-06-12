def crop_pedestrian_area(frame):

    x1, y1 = 70, 95 #esquerda, topo 
    x2, y2 = 220, 400 #direita, baixo

    frame_crop_pedestrian = frame[y1:y2, x1:x2]

    return frame_crop_pedestrian
