import cv2
import numpy as np

def align_image(img, template):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img, None)
    kp2, des2 = orb.detectAndCompute(template, None)
    
    if des1 is None or des2 is None:
        return img
        
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Keep top 15% matches
    num_good = int(len(matches) * 0.15)
    matches = matches[:num_good]
    
    if len(matches) < 10:
        return img
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is not None:
        h, w = template.shape[:2]
        aligned = cv2.warpPerspective(img, H, (w, h))
        return aligned
    return img

def extract_score_cells(aligned_img):
    gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Dilate slightly to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thresh.shape[1]//30, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, thresh.shape[0]//30))
    
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
    
    grid = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0)
    _, grid = cv2.threshold(grid, 128, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        # Filter out very small/large boxes to find table cells
        if 5000 < area < 100000 and w > 20 and h > 20:
            bounding_boxes.append((x, y, w, h))
            
    # Sort boxes top-to-bottom, left-to-right roughly
    bounding_boxes.sort(key=lambda b: (b[1] // 20, b[0]))
    
    score_crops = []
    for (x, y, w, h) in bounding_boxes:
        # Assuming score column is the right-most. If x is high (e.g., right side of the page)
        # We can just extract all cells and let VLM/OCR process them, but to save time,
        # we try to just return the rightmost cells.
        if x > aligned_img.shape[1] * 0.6: 
            crop = aligned_img[y:y+h, x:x+w]
            score_crops.append(crop)
            
    return score_crops
