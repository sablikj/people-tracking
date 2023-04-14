import cv2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture

show_gt = True

thr = 50 # Used for background subtraction
minArea = 175 # Minimal area to be considered a component
maxDistance = 25 #Centroid:25  # Maximal distance between frames of each person
max_lost_frames = 7  # Maximum number of frames a track can be lost before it is removed
path_len = 15 # length of displayed trajectory

next_id = 1  # ID to assign to the next detected pedestrian
frame = 1 # Number of current frame

# Define colormap and normalize colors based on number of pedestrians
colormap = plt.cm.get_cmap('inferno')
kernel = np.ones((5,5),np.uint8) # Used for orphological operations

iou_values = [] # List of IOU values for the plot
tracks = []  # List of tracks, each track is a dict containing the ID, bounding box, and descriptor of a pedestrian
ground_truth = pd.read_csv('./gt.txt', sep=',', names=["Frame", "ID", "bbLeft", "bbTop", "Width", "Height", "Confidence", "x", "y", "z"])
cap = cv2.VideoCapture("./dataset/frame_%04d.jpg")

width = int(cap.get(3))
height = int(cap.get(4))

# Initialize an empty heatmap
heatmap = np.zeros((height, width)) # 3 - Frame width, 4 - Frame height

# Initialize the plot for IoU
plt.ion()


# Create figure and subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
fig.canvas.manager.set_window_title('Pedestrian tracking project | Group 11') 
# Get the current figure manager and set the window to fullscreen
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.tight_layout()

# Tracked image
axes[0, 0].set_title('Pedestrian tracking')
axes[0, 0].imshow(np.zeros((height, width)))
axes[0, 0].set_axis_off()

# Heatmap
axes[1, 0].imshow(heatmap)
axes[1, 0].set_title('Occupancy map')

# IoU plot
axes[1, 1].plot(heatmap)
axes[1, 1].set_title('Success plot')

# EA analysis
axes[0, 1].imshow(np.zeros((height, width)))
axes[0, 1].set_title('Analysis of pedestrian trajectories')
axes[0, 1].set_axis_off()


# Evaluation
iou_thr = 0.4
total_detections = 0
false_negatives = 0
false_positives = 0
true_positives = 0
ground_truth_count = 0 # len(ground_truth) - instead    , increasing dynamically

# Gaussian mixture model for EM
n_components = 5  # The number of Gaussian components to be used in the GMM
gmm = GaussianMixture(n_components=n_components, covariance_type='full')
trajectories = []
em_update_interval = 10  # Update the EM algorithm every 10 frames
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'gray']

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


def compute_iou(bbox1, bbox2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Args:
        bbox1 (tuple): A tuple (x1, y1, w, h) representing the coordinates of the first bounding box.
        bbox2 (tuple): A tuple (x1, y1, w, h) representing the coordinates of the second bounding box.

    Returns:
        float: The IoU value, ranging from 0 (no overlap) to 1 (complete overlap).
    """

    # Check for perfect match
    if bbox1 == bbox2:
        return 1.0

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


def update_track(track, bbox, path_point):
    """
    Update the state of a track based on a matched pedestrian in the current frame.

    Args:
        track (dict): A dictionary representing the track to be updated.
        bbox (tuple): Coordinates of the bounding box.
        path_point (tuple): A tuple representing the path point of the matched pedestrian in the format (x, y).

    Returns:
        None
    """
    # Update the bounding box, descriptor, and centroid of the track
    track['bbox'] = bbox
    
    if(len(track['path']) > path_len):
        track['path'].pop(0)

    track['path'].append(path_point)
        
    # Reset the lost frames counter
    track['lost_frames'] = 0


def compute_distance(descriptor, track_desc,c1, track_bbox):
    """
    Compute the distance between a pedestrian in the current frame and a track.
    Parameters:
        descriptor (ndarray): An array representing the descriptor of the pedestrian in the current frame.
        track_desc (ndarray): An array representing the descriptor of the track.
        c1 (tuple): A tuple representing the position of the pedestrians centroid in the current frame in the format (x, y).
        track_bbox (tuple): A tuple representing the bounding box of the track in the format (x, y, w, h).
    Returns:
        float: The distance between the pedestrian and the track.
    """
    
    c2 = getCentroid(track_bbox)
    pos_dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    hist_dist = np.linalg.norm(track_desc-descriptor)
    return pos_dist * hist_dist  


def get_ground_truth_bboxes(frame, ground_truth):
    """
    Get the ground truth bounding boxes for the given frame number.

    Args:
        frame (int): The frame number.
        ground_truth (pd.DataFrame): Ground truth data.

    Returns:
        List[tuple]: A list of tuples containing ground truth bounding boxes in the format (x, y, w, h).
    """
    gt_bboxes = ground_truth[ground_truth['Frame'] == frame][['bbLeft', 'bbTop', 'Width', 'Height']].values
    return [tuple(x) for x in gt_bboxes]

def extract_trajectories(tracks):
    trajectories = []
    for track in tracks:
        trajectory = np.array(track['path'])
        trajectories.append(trajectory)
    return trajectories


def draw_ellipse(position, covariance, ax, edge_color, face_color, lw=2, alpha=0.5):
    """Draw an ellipse with a given position, covariance, and color."""
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    ellipse = Ellipse(position, width, height, angle=angle, edgecolor=edge_color, facecolor=face_color, lw=lw, alpha=alpha)
    ax.add_patch(ellipse)

def em_algorithm(data, n_components):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(data)
    labels = gmm.predict(data)
    return labels

############################################################################################
bg = getBackground(cap)

while cap.isOpened():  
    ret, img = cap.read()    
    if not ret:
        break
    img_track = img.copy()
    
    # Check if user closed the window
    if not plt.fignum_exists(fig.number):
        # User closed the window, break the loop
        break

    # Initialize an empty list to store IoUs
    iou_frame = []
    
    # 2) OBJECT DETECTION
    # Getting foreground objects -> pedestrians
    img_diff = subtractBackground(img, bg)
    
    # Using closing to fill the holes in objs    
    img_cls = cv2.morphologyEx(img_diff, cv2.MORPH_CLOSE, kernel)

    # Using erosion to remove noise
    img_opn = cv2.erode(img_cls, kernel)
    img_opn = cv2.dilate(img_opn, kernel)   

    # Running CCA alg - using connectivity 4
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_opn, connectivity=8) 

    # Filter labels and their stats with minArea param
    filtered_labels = np.where(np.isin(labels, np.where(stats[:, cv2.CC_STAT_AREA] >= minArea)[0]), labels, 0)
    filtered_label_stats = np.where(stats[:, cv2.CC_STAT_AREA] >= minArea)[0]
    
    # Initialize an empty list to store the detections in current frame
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
        path_point = (int(x+(w/2)), int(y+h))  

        # Updating heatmap and preventing index to be out of bounds
        heatmap[min(path_point[1], heatmap.shape[0] - 1), min(path_point[0], heatmap.shape[1] - 1)] += 5
        #heatmap[path_point[1],path_point[0]] += 5  

        # Remove bg from pedestrian bbox with mask from CCA
        ped = cv2.bitwise_and(img[y:y+h, x:x+w], img[y:y+h, x:x+w], mask=img_opn[y:y+h, x:x+w])
        
        # Compute the descriptor of the pedestrian
                                    # used channels, no mask, 8 bins, 0-255 val range
        descriptor = cv2.calcHist(ped, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        # Normalize the histogram
        descriptor = cv2.normalize(descriptor, descriptor).flatten()

        if(frame > 1):
            roi = x,y,w,h

            # Get ground truth bounding boxes for the current frame
            gt_bboxes = get_ground_truth_bboxes(frame, ground_truth)                    

            # Computing IOU
            # Calculate IoU between the detected bounding box and all ground truth bounding boxes
            iou = [compute_iou(roi, gt_bbox) for gt_bbox in gt_bboxes]            

            # Find the maximum IoU value - Closest bbox
            max_iou = max(iou)
            # Append the maximum IoU value to the list of IoUs
            iou_frame.append(max_iou)
            
            # Initialize variables for tracking
            matched_track = None  # The track that matches the current pedestrian
            min_distance = float('inf')  # The minimum distance between the current pedestrian and any track
            # Try to match the current pedestrian with a track
            for track in tracks:                      
                # Compute the distance between the current pedestrian and the track
                distance = compute_distance(descriptor, track['descriptor'],centroid, track['bbox'])
                                
                # Update the closest track                
                if distance < min_distance:
                    matched_track = track
                    min_distance = distance                  
                    
            if matched_track is not None and min_distance < maxDistance:    

                # Update the matched track with the current pedestrian
                update_track(matched_track, roi, path_point)                
            else:
                # Add a new track for the current pedestrian
                track = {'id': next_id,
                        'bbox': (x, y, w, h),
                        'descriptor': descriptor,
                        'path': [path_point],
                        'lost_frames': 0}
                tracks.append(track)
                next_id += 1
        else:
            # Initialize tracks with the detected pedestrians in the first frame
            track = {'id': next_id,
                    'bbox': (x,y,w,h),
                    'descriptor': descriptor,
                    'path': [path_point],
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
        
        # Show the current frame with the tracks, update heatmap
        for track in tracks:            
            id = track['id']
            x, y, w, h = track['bbox']         

            cv2.rectangle(img_track, (x, y), (x+w, y+h), (0, 255, 0), 1)
            for c in track['path']:
                cv2.circle(img_track, (c[0], c[1]), 2, (0,0,255), -1)

            if len(track['path']) >= path_len:
                trajectory = np.array(track['path'])[-path_len:]
                trajectories.append(trajectory)              


        # Increase number of gt detections
        ground_truth_count += len(gt_bboxes)

        # Check for false positives or false negatives
        for iou in iou_frame:
            # Correct detection - True positive
            if iou >= iou_thr:
                true_positives += 1

            # Wrong detection - False positive
            else:
                false_positives += 1                

            # Increase number of detections
            total_detections += 1

        # Calculate the average IoU for the current frame
        avg_iou = round(np.mean(iou_frame),2) 

        # Append mean value for the plot
        iou_values.append(avg_iou)            
        
        cv2.putText(img_track, f"Frame: {frame}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img_track, f"IoU: {avg_iou}", (5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img_track, f"FP: {false_positives} | {round(false_positives / total_detections * 100, 2)} %", (5,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img_track, f"FN: {ground_truth_count - true_positives} | {round((ground_truth_count - true_positives) / total_detections * 100, 2)} %", (5,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Displaying ground truth
        if show_gt:
            for i, bbox in ground_truth[ground_truth['Frame'] == frame].iterrows():    
                cv2.rectangle(img_track, (int(bbox[2]), int(bbox[3])), (int(bbox[2]+bbox[4]), int(bbox[3]+bbox[5])), (255,0,0), 1)
                cv2.putText(img_track, str(int(bbox[1])), (int(bbox[2]+10), int(bbox[3]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        
        axes[0,0].images[0].set_data(cv2.cvtColor(img_track, cv2.COLOR_BGR2RGB))

    # Display heatmap
    smoothed_heatmap = gaussian_filter(heatmap, sigma=6)
    showHeatmap = (colormap(smoothed_heatmap) * 2**32).astype(np.uint32)[:,:,:3]
    
    axes[1, 0].images[0].set_data(colormap(smoothed_heatmap))

    # Update the IoU plot
    axes[1,1].clear()
    axes[1,1].set_ylim(0,1)

    # Select every 5 element in iou_values list
    subset_iou_values = iou_values[::5]

    # Get the x-axis values for the subset data points
    x_values = list(range(0, len(iou_values), 5))

    # Plot the subset values with the corresponding x-axis values
    axes[1,1].plot(x_values, subset_iou_values, label='IoU')

    axes[1,1].legend()
    axes[1,1].set_xlabel('Frame')
    axes[1,1].set_ylabel('IoU')
    axes[1,1].set_title('Success plot')

    # EM
    # Add the following lines before the end of the main loop:
    if frame % em_update_interval == 0 and len(trajectories) > 0:
        trajectories_data = np.vstack(trajectories)
        labels = em_algorithm(trajectories_data, n_components)

        axes[0, 1].clear()
        axes[0, 1].imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Analysis of pedestrian trajectories')
        
        for i, label in enumerate(np.unique(labels)):
            traj_data = trajectories_data[labels == label]
            axes[0, 1].scatter(traj_data[:, 0], traj_data[:, 1], s=5, color=colormap(i / len(np.unique(labels))), label=f"Group {i + 1}")

        axes[0, 1].set_xlim(0, width)
        axes[0, 1].set_ylim(height, 0)
        axes[0, 1].legend()
        axes[0, 1].set_axis_off()


    update_interval = 3
    # Update the IoU plot only if the current frame is a multiple of update_interval
    if frame % update_interval == 0:        
        plt.draw()
        plt.pause(0.001)
         
    #####################################
    frame += 1
    # Wait for a key press to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.waitKey()
cap.release()

# Close the figure window
plt.close(fig)