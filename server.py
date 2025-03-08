import panel as pn
import cv2
import numpy as np
import json
import os, io
from flask import Flask, request, jsonify

app = Flask(__name__)
THRESHOLD_FILE = "threshold.json"
UPLOAD_FOLDER = "static/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pn.extension()

def load_threshold():
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r") as f:
            return json.load(f)
    return {"h_min": 30, "h_max": 90, "s_min": 50, "s_max": 255, "v_min": 50, "v_max": 255}

def save_threshold(threshold):
    with open(THRESHOLD_FILE, "w") as f:
        json.dump(threshold, f)

def detect_mold(image_path, threshold):
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([threshold['h_min'], threshold['s_min'], threshold['v_min']])
    upper_bound = np.array([threshold['h_max'], threshold['s_max'], threshold['v_max']])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        if (area >= threshold["area_min"] and
            perimeter >= threshold["perimeter_min"] and
            circularity >= threshold["circularity_min"]):
            filtered_contours.append(contour)
    
    # Draw bounding boxes
    result = image.copy()
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    mask_path = os.path.join(UPLOAD_FOLDER, "mask.jpg")
    result_path = os.path.join(UPLOAD_FOLDER, "result.jpg")

    cv2.imwrite(mask_path, mask)
    cv2.imwrite(result_path, result)

    return mask_path, result_path

threshold = load_threshold()
image_path = None

sliders = {
    "h_min": pn.widgets.IntSlider(name="Hue Min", start=0, end=180, value=threshold["h_min"]),
    "h_max": pn.widgets.IntSlider(name="Hue Max", start=0, end=180, value=threshold["h_max"]),
    "s_min": pn.widgets.IntSlider(name="Saturation Min", start=0, end=255, value=threshold["s_min"]),
    "s_max": pn.widgets.IntSlider(name="Saturation Max", start=0, end=255, value=threshold["s_max"]),
    "v_min": pn.widgets.IntSlider(name="Value Min", start=0, end=255, value=threshold["v_min"]),
    "v_max": pn.widgets.IntSlider(name="Value Max", start=0, end=255, value=threshold["v_max"]),
    "circularity_min": pn.widgets.FloatSlider(name="Circularity Min", start=0, end=1, value=0.50),
    "area_min": pn.widgets.IntSlider(name="Min Area", start=100, end=5000, value=500),
    "perimeter_min": pn.widgets.IntSlider(name="Min Perimeter", start=10, end=1000, value=50)
}

mask_pane = pn.pane.Image(None, width=400)
result_pane = pn.pane.Image(None, width=400)

def update_images(event=None):
    global image_path
    if not image_path:
        return
    new_threshold = {k: v.value for k, v in sliders.items()}
    save_threshold(new_threshold)
    mask_img, result_img = detect_mold(image_path, new_threshold)

    if mask_img and result_img:
        with open(mask_img, "rb") as f:
            mask_pane.object = f.read()
        with open(result_img, "rb") as f:
            result_pane.object = f.read()
            
for slider in sliders.values():
    slider.param.watch(update_images, "value")

def upload_image(event):
    global image_path
    file_obj = file_input.value  # No need for [0], value itself is bytes
    if file_obj:
        image_path = os.path.join(UPLOAD_FOLDER, "uploaded.jpg")
        with open(image_path, "wb") as f:  # Ensure writing as binary
            f.write(file_obj)  # No need to access index
        update_images()

file_input = pn.widgets.FileInput()
file_input.param.watch(upload_image, "value")

controls = pn.Column("### Upload Image", file_input, "### Adjust Thresholds", *sliders.values())

mask_pane = pn.pane.Image(None, height=400, sizing_mode="scale_both")
result_pane = pn.pane.Image(None, height=400, sizing_mode="scale_both")

image_display = pn.Column(
    pn.pane.Markdown("### Masked Image"), mask_pane,
    pn.pane.Markdown("### Result Image"), result_pane
)

layout = pn.Row(controls, pn.Card(image_display, title="Image Output"))

pn.serve(layout, title="Mold Detection Panel", show=True)