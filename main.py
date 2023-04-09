import cv2
import math
import pandas as pd
import numpy as np

tracking = True
show_gt = False

thr = 50 # Used for background subtraction
minArea = 175 # Minimal area to be considered a component
maxDistance = 30 # Maximal distance between frames of each person | Also mean width of bbbox in these frames
allViews = True
kernel = np.ones((5,5),np.uint8) # Used for orphological operations

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
    
    Parameters:
        cap (cv2.VideoCapture): A VideoCapture object containing sequence of frames.
    
    Returns:
        bg (np.array): Created background image.
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
    
    Parameters:
        img (np.array): Current frame.
        bg (np.array): Computed background image.
    
    Returns:
        diff (np.array): A foreground of the image.
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

############################################################################################
bg = getBackground(cap)

while cap.isOpened():    
    ret, img = cap.read()
    if not ret:
        break
    
    # For saving data from current frame
    current_frame = []

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
    
    #label_image = (filtered_labels * (255 / num_labels)).astype(np.uint8)
    #colored_label_image = cv2.applyColorMap(label_image, cv2.COLORMAP_HOT)
    #if allViews:
        #cv2.imshow('Filtered labels', colored_label_image)

    img_detect = img.copy()
    
    for i, lab in enumerate(filtered_label_stats):
        if lab == 0:  # Skip the background
            continue
        x = stats[lab, cv2.CC_STAT_LEFT]
        y = stats[lab, cv2.CC_STAT_TOP]
        w = stats[lab, cv2.CC_STAT_WIDTH]
        h = stats[lab, cv2.CC_STAT_HEIGHT]

        centroid = (int(x+(w/2)), int(y+(h/2)))       

        # Saving centroids of each label 
        # Frame, ID, bbLeft, bbTop, Width, Height, Confidence,x,y,z
        data.append([frame, id, x, y, w, h, 0, 0, 0, 0])                 
                
        cv2.rectangle(img_detect, (x,y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(img_detect, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.circle(img_detect, centroid, 3, (0,255,0), -1)            
    

    # 3) OBJECT TRACKING - Gaussian distance
    # Comparing with data from previous frame   
    if frame > 1 & tracking:
        img_track = img.copy()
        
        prev_frame = [sublist for sublist in data if sublist[0] == frame-1] 
        current_frame = [sublist for sublist in data if sublist[0] == frame] 
        
        # Extract the centroids from previous frame
        obj_prev = [row[2:6] for row in prev_frame]
        
        # Calculate the Euclidean distance
        for obj_curr in current_frame: 
            x = obj_curr[2]
            y = obj_curr[3]
            w = obj_curr[4]
            h = obj_curr[5]  
            print(f"Width: {w} Height: {h}")

            c_curr = getCentroid([x,y,w,h])
            distances = []
            for obj_prev in prev_frame:  
                c_prev = getCentroid(obj_prev[2:6]) 
                distance = math.sqrt((c_prev[0]-c_curr[0])**2 + (c_prev[1]-c_curr[1])**2)
                distances.append(distance)

            # Find the index of the centroids that have the smallest distance
            closest_centroid_index = np.argmin(distances)
            
            print(distances[closest_centroid_index])

            # Large distance -> new object
            if distances[closest_centroid_index] < maxDistance:
                # Retrieve the closest centroid from the array
                closest = prev_frame[closest_centroid_index]
                # Get row index from dataset by bbox
                index = find_matching_row(data, closest)
                
                
                # Update the id in the dataset
                data[index][1] = obj_curr[1]
            else:
                # Assign a new ID
                obj_curr[1] = id 
                id+=1        
            
            cv2.rectangle(img_track, (x,y), (x+w, y+h), (0, 0, 255), 1)
            cv2.putText(img_track, str(obj_curr[1]), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.imshow('Tracked objects', img_track)

        # Get all GTs for current frame
        gt = ground_truth[ground_truth["Frame"] == frame]    

        # 1) PLOTTING GROUND TRUTH
        if show_gt:
            for i, bbox in gt.iterrows():    
                cv2.rectangle(img_detect, (int(bbox[2]), int(bbox[3])), (int(bbox[2]+bbox[4]), int(bbox[3]+bbox[5])), (255,0,0), 1)
                cv2.putText(img_detect, str(int(bbox[1])), (int(bbox[2]+10), int(bbox[3]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)  

        
        cv2.imshow('Detected objects', img_detect)   
    #####################################
    frame += 1
    # Wait for a key press to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()