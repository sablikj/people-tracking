import cv2
import math
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

show_gt = False

minIOU = 0.3
thr = 50 # Used for background subtraction
minArea = 175 # Minimal area to be considered a component
maxDistance = 25 #Centroid:25  # Maximal distance between frames of each person
allViews = False
kernel = np.ones((5,5),np.uint8) # Used for orphological operations

# Initialize variables for tracking
tracks = []  # List of tracks, each track is a dict containing the ID, bounding box, and descriptor of a pedestrian
next_id = 1  # ID to assign to the next detected pedestrian
max_lost_frames = 5  # Maximum number of frames a track can be lost before it is removed


id = 1 # ID of detected person
frame = 1 # Number of current frame
ground_truth = pd.read_csv('./gt.txt', sep=',', names=["Frame", "ID", "bbLeft", "bbTop", "Width", "Height", "Confidence", "x", "y", "z"])
cap = cv2.VideoCapture("./dataset/frame_%04d.jpg")

# Frame, ID, bbLeft, bbTop, Width, Height, Confidence,x,y,z
data = []

############################################################################################
def getBackground(cap, n=25):
    """
    Takes 'n' image samples to create background image using median.
    
    Args:
        cap (cv2.VideoCapture): A VideoCapture object containing sequence of frames.
    
    Returns:
        np.array: Created background image.
    """
    # Randomly select 25 frames
    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=n)
    
    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, f = cap.read()
        frames.append(f)
    
    # Calculate the median of the images -> background
    bg = np.median(frames, axis=0).astype(dtype=np.uint8)   

    # Set cap back to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    return bg

def subtractBackground(img, bg):
    """
    Subtracts image background from current frame to obtain foreground.
    
    Args:
        img (np.array): Current frame.
        bg (np.array): Computed background image.
    
    Returns:
        np.array: A foreground of the image.
    """
    # Subtracting current frame from background
    diff = (np.abs(bg[:, :, 0].astype(np.float64) - img[:, :, 0].astype(np.float64)) > thr) | \
                (np.abs(bg[:, :, 1].astype(np.float64) - img[:, :, 1].astype(np.float64)) > thr) | \
                (np.abs(bg[:, :, 2].astype(np.float64) - img[:, :, 2].astype(np.float64)) > thr)
    
    # Converting boolean array to integer array
    diff = (diff * 255).astype(np.uint8)

    return diff

def getCentroid(box):
    x, y, w, h = box
    return (int(x+(w/2)), int(y+(h/2)))

def find_matching_row(data, obj):
    for i, row in enumerate(data):
        if row[2:6] == obj[2:6]:
            return i
    return -1  # return -1 if no matching row is found

def predictPosition(prev, curr, roi):
    """
    Uses OpticalFlow to predict position of each pixel of ROI in next frame.

    Args:
        prev (np.array): First 8-bit single-channel input image.
        img (np.array): Next second input image of the same size and the same type as prev.
        roi (tuple): Bounding box coordinates (x,y,w,h).

    Returns:
        tuple: Updated bounding box coordinates (x,y,w,h).
    """
    x,y,w,h = roi
    prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
    # Apply an optical flow algorithm to calculate the motion vectors
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 2, 15, 2, 5, 1.1, 0)

    # Track the ROI using optical flow
    new_x = x + int(round(flow[y:y+h, x:x+w, 0].mean()))
    new_y = y + int(round(flow[y:y+h, x:x+w, 1].mean()))

    return new_x, new_y, w, h

def compute_iou(bbox1, bbox2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Args:
        bbox1 (tuple): A tuple (x1, y1, x2, y2) representing the coordinates of the first bounding box.
        bbox2 (tuple): A tuple (x1, y1, x2, y2) representing the coordinates of the second bounding box.

    Returns:
        float: The IoU value, ranging from 0 (no overlap) to 1 (complete overlap).
    """
    x1_bbox1, y1_bbox1, w_bbox1, h_bbox1 = bbox1
    x1_bbox2, y1_bbox2, w_bbox2, h_bbox2 = bbox2

    x2_bbox1 = x1_bbox1 + w_bbox1
    x2_bbox2 = x1_bbox2 + w_bbox2
    y2_bbox1 = y1_bbox1 + h_bbox1
    y2_bbox2 = y1_bbox2 + h_bbox2

    # Calculate the intersection coordinates
    x1_intersection = max(x1_bbox1, x1_bbox2)
    y1_intersection = max(y1_bbox1, y1_bbox2)
    x2_intersection = min(x2_bbox1, x2_bbox2)
    y2_intersection = min(y2_bbox1, y2_bbox2)

    # Calculate the intersection area
    intersection_width = max(0, x2_intersection - x1_intersection)
    intersection_height = max(0, y2_intersection - y1_intersection)
    intersection_area = intersection_width * intersection_height

    # Calculate the area of each bounding box
    bbox1_area = (x2_bbox1 - x1_bbox1) * (y2_bbox1 - y1_bbox1)
    bbox2_area = (x2_bbox2 - x1_bbox2) * (y2_bbox2 - y1_bbox2)

    # Calculate the union area
    union_area = bbox1_area + bbox2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

def update_track(track, x, y, w, h, descriptor, centroid):
    """
    Update the state of a track based on a matched pedestrian in the current frame.

    Args:
        track (dict): A dictionary representing the track to be updated.
        x (int): The x-coordinate of the top-left corner of the bounding box of the matched pedestrian.
        y (int): The y-coordinate of the top-left corner of the bounding box of the matched pedestrian.
        w (int): The width of the bounding box of the matched pedestrian.
        h (int): The height of the bounding box of the matched pedestrian.
        descriptor (ndarray): An array representing the descriptor of the matched pedestrian.
        centroid (tuple): A tuple representing the centroid of the matched pedestrian in the format (x, y).

    Returns:
        None
    """
    # Update the bounding box, descriptor, and centroid of the track
    track['bbox'] = (x, y, w, h)
    track['descriptor'] = descriptor
    track['centroid'] = centroid
    
    # Compute the velocity of the track
    prev_cx, prev_cy = track['centroid']
    vx = centroid[0] - prev_cx
    vy = centroid[1] - prev_cy
    track['velocity'] = (vx, vy)
    
    # Reset the lost frames counter
    track['lost_frames'] = 0

def compute_distance(descriptor, track_desc, pos, track_bbox):
    """
    Compute the distance between a pedestrian in the current frame and a track.

    Parameters:
        descriptor (ndarray): An array representing the descriptor of the pedestrian in the current frame.
        track_desc (ndarray): An array representing the descriptor of the track.
        pos (tuple): A tuple representing the position of the pedestrian in the current frame in the format (x, y, w, h).
        track_bbox (tuple): A tuple representing the bounding box of the track in the format (x, y, w, h).

    Returns:
        float: The distance between the pedestrian and the track.
    """
    # Euclidean distance between the descriptor of the pedestrian and the descriptor of the track
    desc_dist = np.linalg.norm(track_desc-descriptor)
    print(desc_dist)
    # Euclidean distance between the position of the pedestrian and the position of the track
    c1 = getCentroid(pos)
    c2 = getCentroid(track_bbox)
    pos_dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    
    # Total distance with added weights
    dist = desc_dist + pos_dist

    return pos_dist    


############################################################################################
bg = getBackground(cap)
prev_frame = None
while cap.isOpened():  
    ret, img = cap.read()
    img_track = img.copy()
    if not ret:
        break
    
    # 2) OBJECT DETECTION
    
    # Getting foreground objects -> pedestrians
    img_diff = subtractBackground(img, bg)

    if allViews:
        cv2.imshow("1. Subtracted image", img_diff)
    
    # Using closing to fill the holes in objs    
    img_cls = cv2.morphologyEx(img_diff, cv2.MORPH_CLOSE, kernel)
    if allViews:
        cv2.imshow('2. Applied closing', img_cls)

    # Using erosion to remove noise
    img_opn = cv2.erode(img_cls, kernel)
    img_opn = cv2.dilate(img_opn, kernel)
    if allViews:
        cv2.imshow("3. Applied erd+dil", img_opn)

    # Running CCA alg - using connectivity 4
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_opn, connectivity=8) 

    # Filter labels and their stats with minArea param
    filtered_labels = np.where(np.isin(labels, np.where(stats[:, cv2.CC_STAT_AREA] >= minArea)[0]), labels, 0)
    filtered_label_stats = np.where(stats[:, cv2.CC_STAT_AREA] >= minArea)[0]
    
    img_detect = img.copy()
    # Initialize an empty list to store the detections
    detections = []

    # Iterating in detected pedestrians
    for i, lab in enumerate(filtered_label_stats):
        if lab == 0:  # Skip the background
            continue
        x = stats[lab, cv2.CC_STAT_LEFT]
        y = stats[lab, cv2.CC_STAT_TOP]
        w = stats[lab, cv2.CC_STAT_WIDTH]
        h = stats[lab, cv2.CC_STAT_HEIGHT]

        centroid = (int(x+(w/2)), int(y+(h/2)))       

        # Frame, ID, bbLeft, bbTop, Width, Height, Confidence, x, y, z
        data.append([frame, id, x, y, w, h, 0, 0, 0, 0])                 

        # Compute the descriptor of the pedestrian
        descriptor = cv2.calcHist([img[x:x+w, y:y+h]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        # Normalize histogram
        descriptor = cv2.normalize(descriptor, descriptor).flatten()

        # Add the pedestrian information to the detections list
        detections.append({'bbox': (x, y, w, h), 'descriptor': descriptor, 'centroid': centroid})
        
        # If it's not the first frame, associate tracks and detections using the Hungarian algorithm
        if frame > 1:
            cost_matrix = np.zeros((len(tracks), len(detections)))

            for t, track in enumerate(tracks):
                for d, det in enumerate(detections):
                    distance = compute_distance(det['descriptor'], track['descriptor'], det['bbox'], track['bbox'])
                    cost_matrix[t, d] = distance

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < maxDistance:
                    update_track(tracks[r], detections[c]['bbox'][0], detections[c]['bbox'][1], detections[c]['bbox'][2], detections[c]['bbox'][3], detections[c]['descriptor'], detections[c]['centroid'])
                else:
                    # Add a new track for the current pedestrian
                    track = {'id': next_id,
                            'bbox': (detections[c]['bbox'][0], detections[c]['bbox'][1], detections[c]['bbox'][2], detections[c]['bbox'][3]),
                            'descriptor': detections[c]['descriptor'],
                            'centroid': detections[c]['centroid'],
                            'velocity': (0, 0),
                            'lost_frames': 0}
                    tracks.append(track)
                    next_id += 1
        else:
            # Initialize tracks with the detected pedestrians in the first frame
            for det in detections:
                track = {'id': next_id,
                        'bbox': det['bbox'],
                        'descriptor': det['descriptor'],
                        'centroid': det['centroid'],
                        'velocity': (0, 0),
                        'lost_frames': 0}
                tracks.append(track)
                next_id += 1

    # Update the lost tracks
    if(frame > 1):
        for track in tracks:
            if track['lost_frames'] > max_lost_frames:
                tracks.remove(track)
            else:
                track['lost_frames'] += 1

        # Show the current frame with the tracks
        for track in tracks:
            x, y, w, h = track['bbox']
            cv2.rectangle(img_track, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(img_track, f"{track['id']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Tracked image', img_track)
        
         
    #####################################
    frame += 1
    prev_frame = img.copy()
    # Wait for a key press to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()