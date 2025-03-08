import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def calibrate_ycrcb(image_path):
    img = cv2.imread(image_path)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    def update(x):
        cr_min = cv2.getTrackbarPos('Cr Min', 'Calibration')
        cr_max = cv2.getTrackbarPos('Cr Max', 'Calibration')
        cb_min = cv2.getTrackbarPos('Cb Min', 'Calibration')
        cb_max = cv2.getTrackbarPos('Cb Max', 'Calibration')
        
        mask = cv2.inRange(ycrcb[:,:,1], cr_min, cr_max) & \
               cv2.inRange(ycrcb[:,:,2], cb_min, cb_max)
        cv2.imshow('Calibration', mask)
    
    cv2.namedWindow('Calibration')
    cv2.createTrackbar('Cr Min', 'Calibration', 135, 255, update)
    cv2.createTrackbar('Cr Max', 'Calibration', 160, 255, update)
    cv2.createTrackbar('Cb Min', 'Calibration', 85, 255, update)
    cv2.createTrackbar('Cb Max', 'Calibration', 110, 255, update)
    update(0)
    cv2.waitKey(0)

def calibrate_texture(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def update(x):
        radius = cv2.getTrackbarPos('Radius', 'Texture') + 1
        n_points = 8 * radius
        threshold = cv2.getTrackbarPos('Threshold', 'Texture')
        
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        mask = (lbp > threshold).astype(np.uint8) * 255
        cv2.imshow('Texture', mask)
    
    cv2.namedWindow('Texture')
    cv2.createTrackbar('Radius', 'Texture', 3, 10, update)
    cv2.createTrackbar('Threshold', 'Texture', 150, 255, update)
    update(0)
    cv2.waitKey(0)

def highlight_mold(image_path):
    # Load and preprocess
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return
    
    # Step 1: Downsample for processing
    scale_factor = 0.5  # Adjust based on your image size
    small_img = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor)
    
    # Step 2: Improved bread segmentation (using LAB color space)
    lab = cv2.cvtColor(small_img, cv2.COLOR_BGR2LAB)
    _, a, b = cv2.split(lab)
    
    # Create bread mask (adjust these values)
    bread_mask = cv2.inRange(a, 120, 140) & cv2.inRange(b, 120, 145)
    bread_mask = cv2.morphologyEx(bread_mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    
    # Step 3: Texture analysis using Local Binary Patterns
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    radius = 3  # Pixels to consider
    n_points = 8 * radius  # Number of circularly symmetric neighbours
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Normalize LBP and create texture mask
    lbp = (lbp / lbp.max() * 255).astype(np.uint8)
    texture_mask = cv2.inRange(lbp, 150, 255)  # High texture areas
    
    # Step 4: Color analysis in YCrCb space (better for mold detection)
    ycrcb = cv2.cvtColor(small_img, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:,:,1]
    cb = ycrcb[:,:,2]
    
    # Mold tends to have high Cr and low Cb values
    color_mask = cv2.inRange(cr, 135, 160) & cv2.inRange(cb, 85, 110)
    
    # Step 5: Combine evidence
    combined_mask = cv2.bitwise_and(texture_mask, color_mask)
    combined_mask = cv2.bitwise_and(combined_mask, bread_mask)
    
    # Post-processing
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    # Step 6: Upscale results to original size
    final_mask = cv2.resize(combined_mask, (img.shape[1], img.shape[0]), 
                   interpolation=cv2.INTER_NEAREST)
    
    # Visualization
    output = img.copy()
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 100/scale_factor:  # Scale area threshold
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x,y), (x+w,y+h), (0,0,255), 2)
    
    # Show intermediate steps
    cv2.imshow('Bread Mask', cv2.resize(bread_mask, (img.shape[1], img.shape[0])))
    cv2.imshow('Texture Analysis', cv2.resize(lbp, (img.shape[1], img.shape[0])))
    cv2.imshow('Color Mask', cv2.resize(color_mask, (img.shape[1], img.shape[0])))
    cv2.imshow('Final Detection', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate_ycrcb("bread/bread1.jpg")