import cv2
import requests
import time
import sys

# 1. Configuration Settings
PC_IP_ADDRESS = "192.168.1.50"  # <-- Change this to match your home PC IP!
SERVER_URL = f"http://{PC_IP_ADDRESS}:5000/predict"

# 2. Open standard input stream instead of a direct hardware node index
# This intercepts a clean, system-pre-decoded MJPEG video pipe
cap = cv2.VideoCapture("-") 

if not cap.isOpened():
    print("❌ ERROR: Could not open the video pipe stream.")
    sys.exit()

print("Raspberry Pi streaming client started. Press 'q' to quit.")

while cap.isOpened():
    start_time = time.time()
    
    # This reads pre-formatted, clean frames out of the system pipe safely!
    success, frame = cap.read()
    
    if not success or frame is None:
        print("Waiting for camera frame stream...")
        time.sleep(0.05)
        continue

    # 3. Compress the valid frame to JPEG bytes to send it over Wi-Fi
    _, img_encoded = cv2.imencode('.jpg', frame)
    
    try:
        # 4. Send the image over the network to your home PC's GPU
        response = requests.post(SERVER_URL, files={"image": img_encoded.tobytes()}, timeout=2)
        
        if response.status_code == 200:
            data = response.json()

            # 5. Loop through predictions and paint boxes
            for pred in data["predictions"]:
                x1, y1, x2, y2 = pred["box"]
                label = f"{pred['class']} {pred['confidence']:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    except Exception as e:
        cv2.putText(frame, "Connecting to PC Server...", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Frame rate calculation display
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"Network FPS: {fps:.1f}", (20, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Raspberry Pi Client Window", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
