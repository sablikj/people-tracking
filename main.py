import pandas as pd
import numpy as np
import cv2

thr = 50
minArea = 150
allViews = False
kernel = np.ones((5,5),np.uint8)

frame = 1
ground_truth = pd.read_csv('./gt.txt', sep=',', names=["Frame", "Identity", "bbLeft", "bbTop", "Width", "Height", "Confidence", "x", "y", "z"])
cap = cv2.VideoCapture("./dataset/frame_%04d.jpg")

# BACKGROUND ESTIMATION
# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
 
# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, f = cap.read()
    frames.append(f)
 
# Calculate the median of the images -> background
img_bg = np.median(frames, axis=0).astype(dtype=np.uint8)   
if allViews:
    cv2.imshow('Background', img_bg)

# Set cap back to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while cap.isOpened():    
    ret, img = cap.read()
    if not ret:
        break

    # Get all GTs for current frame
    gt = ground_truth[ground_truth["Frame"] == frame]    

    # 1) PLOTTING GROUND TRUTH
    img_gt = img.copy()
    for i, bbox in gt.iterrows():    
        cv2.rectangle(img_gt, (int(bbox[2]), int(bbox[3])), (int(bbox[2]+bbox[4]), int(bbox[3]+bbox[5])), (255,0,0), 1)
        cv2.putText(img_gt, str(int(bbox[1])), (int(bbox[2]), int(bbox[3]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    cv2.imshow('Ground truths', img_gt)

    # 2) OBJECT DETECTION
    # Subtracting current frame from background
    img_diff = (np.abs(img_bg[:, :, 0].astype(np.float64) - img[:, :, 0].astype(np.float64)) > thr) | \
                (np.abs(img_bg[:, :, 1].astype(np.float64) - img[:, :, 1].astype(np.float64)) > thr) | \
                (np.abs(img_bg[:, :, 2].astype(np.float64) - img[:, :, 2].astype(np.float64)) > thr)
    
    # Converting boolean array to integer array
    img_diff = (img_diff * 255).astype(np.uint8)

    if allViews:
        cv2.imshow("Subtracted image", img_diff)
    
    # Using closing to fill the holes in objs    
    img_cls = cv2.morphologyEx(img_diff, cv2.MORPH_CLOSE, kernel)
    if allViews:
        cv2.imshow('Applied closing', img_cls)

    # Using opening to remove noise
    img_opn = cv2.morphologyEx(img_cls, cv2.MORPH_OPEN, kernel)
    if allViews:
        cv2.imshow("Applied opening", img_opn)

    # Running CCA alg - using connectivity 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_opn, connectivity=4) 

    # Filter labels and their stats with minArea param
    filtered_labels = np.where(np.isin(labels, np.where(stats[:, cv2.CC_STAT_AREA] >= minArea)[0]), labels, 0)
    filtered_label_stats = np.where(stats[:, cv2.CC_STAT_AREA] >= minArea)[0]

    label_image = (filtered_labels * (255 / num_labels)).astype(np.uint8)
    colored_label_image = cv2.applyColorMap(label_image, cv2.COLORMAP_HOT)
    if allViews:
        cv2.imshow('Filtered labels', colored_label_image)

    # Creating bounding boxes from labels stats
    img_detect = img.copy()
    for i, lab in enumerate(filtered_label_stats):
        if lab == 0:  # Skip the background
            continue
        x = stats[lab, cv2.CC_STAT_LEFT]
        y = stats[lab, cv2.CC_STAT_TOP]
        w = stats[lab, cv2.CC_STAT_WIDTH]
        h = stats[lab, cv2.CC_STAT_HEIGHT]        
        cv2.rectangle(img_detect, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(img_detect, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow('Detected objects', img_detect)       

    # 3) OBJECT TRACKING 
    

    #####################################
    frame += 1
    # Wait for a key press to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()