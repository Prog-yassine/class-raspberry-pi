import cv2
import requests
import time

# 1. Configuration Settings
# Replace with your home PC's actual local IPv4 address
PC_IP_ADDRESS = "192.168.11.102"  
SERVER_URL = f"http://{PC_IP_ADDRESS}:3000/predict"

# 2. Initialize Camera with V4L2 Driver Backend
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# 🛑 CRITICAL FIX: Force the camera hardware driver to handle MJPEG compression.
# This prevents OpenCV from crashing while decoding raw bayer sensor streams.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Set lightweight streaming dimensions for efficient network transfer
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Raspberry Pi streaming client started. Press 'q' to quit.")

while cap.isOpened():
    start_time = time.time()
    
    # Grab the frame from your camera
    success, frame = cap.read()
    
    # Safety Check: If the hardware skips a frame, don't crash, just retry
    if not success or frame is None:
        print("Waiting for camera frame...")
        time.sleep(0.1)
        continue

    # 3. Compress the valid frame to JPEG bytes to send it over Wi-Fi
    _, img_encoded = cv2.imencode('.jpg', frame)
    
    try:
        # 4. Send the image over the network to your home PC's GPU
        response = requests.post(SERVER_URL, files={"image": img_encoded.tobytes()}, timeout=2)
        
        if response.status_code == 200:
            data = response.json()

            # 5. Loop through the returned predictions and paint bounding boxes
            for pred in data["predictions"]:
                x1, y1, x2, y2 = pred["box"]
                label = f"{pred['class']} {pred['confidence']:.2f}"
                
                # Draw box and class title string
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    except Exception as e:
        # Display feedback on screen if Wi-Fi drops or PC server is offline
        cv2.putText(frame, "Connecting to PC Server...", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Calculate and display the actual network pipeline FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"Network FPS: {fps:.1f}", (20, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 6. Render the live desktop frame
    cv2.imshow("Raspberry Pi Client Window", frame)
    
    # Press 'q' to close down gracefully
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
