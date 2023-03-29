import pandas as pd
import cv2

thr = 20
frame = 1
ground_truth = pd.read_csv('./gt.txt', sep=',', names=["Frame", "Identity", "bbLeft", "bbTop", "Width", "Height", "Confidence", "x", "y", "z"])

cap = cv2.VideoCapture("./dataset/frame_%04d.jpg")

while cap.isOpened():    
        # extract image number from file name
        _, img = cap.read()
        cv2.imshow('Original image', img)

        # get all GTs for the frame
        gt = ground_truth[ground_truth["Frame"] == frame]    

        # 1) Plotting ground truth
        img_gt = img
        for i, bbox in gt.iterrows():    
            cv2.rectangle(img_gt, (int(bbox[2]), int(bbox[3])), (int(bbox[2]+bbox[4]), int(bbox[3]+bbox[5])), (255,0,0), 1)

        cv2.imshow('Ground truths', img_gt)

        # Grayscale
        imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Grayscale', imgg)

        # Denoising
        img_blur = cv2.GaussianBlur(imgg,(13,13),0)      
        
       
        # Thresholding
        _,img_thr = cv2.threshold(img_blur,thr,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #fgMask = backSub.apply(img_blur)
        #cv2.imshow('Thresholded image', img_thr)
        
        
        frame += 1
        # Wait for a key press to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()