import cv2
import numpy as np
import os

clicked = False
r = g = b = h = s = v = 0
xpos = ypos = 0

image_path = 'colorpic.jpg'

if not os.path.exists(image_path):
    print(f"\n[ERROR] '{image_path}' not found in this folder!")
    exit()

img = cv2.imread(image_path)
img_copy = img.copy()

def draw_function(event, x, y, flags, param):
    global b, g, r, h, s, v, xpos, ypos, clicked
    
    # CHANGED HERE: Now triggers on a SINGLE left-click for better reliability
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        xpos = x
        ypos = y
        
        b, g, r = img[y, x]
        b, g, r = int(b), int(g), int(r)
        
        pixel_bgr = np.uint8([[[b, g, r]]])
        pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = pixel_hsv[0][0]
        h, s, v = int(h), int(s), int(v)

cv2.namedWindow('Color Detection Project')
cv2.setMouseCallback('Color Detection Project', draw_function)

print("\n" + "="*60)
print("             COLOR DETECTION SYSTEM IS NOW LIVE")
print("="*60)
print("👉 Action:   SINGLE-CLICK ANYWHERE on the image window to detect colors.")
print("📸 Save:     Press the 's' key to export a project screenshot.")
print("❌ Exit:     Press the 'q' key to shut down the application safely.")
print("="*60 + "\n")

while True:
    cv2.imshow('Color Detection Project', img)
    
    if clicked:
        img = img_copy.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 45), (0, 0, 0), -1)
        text_display = f"RGB: ({r}, {g}, {b}) | HSV: ({h}, {s}, {v})"
        cv2.putText(img, text_display, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        clicked = False

    key = cv2.waitKey(20) & 0xFF
    if key == ord('s'):
        screenshot_name = "color_detection_screenshot.png"
        cv2.imwrite(screenshot_name, img)
        print(f"[ARTIFACT] Saved screenshot to folder: {screenshot_name}")
        
    if key == ord('q'):
        break

cv2.destroyAllWindows()