import cv2
import requests
import time

# 1. Replace with your home PC's actual local IP address
PC_IP_ADDRESS = "192.168.11.102"  # <-- Change this to your PC's local IP
SERVER_URL = f"http://{PC_IP_ADDRESS}:3000/predict"

cap = cv2.VideoCapture(0)

print("Raspberry Pi streaming client started. Press 'q' to quit.")

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break

    # 2. Compress the frame to JPEG to send it over Wi-Fi quickly
    _, img_encoded = cv2.imencode('.jpg', frame)
    
    try:
        # 3. POST the image to your PC AI server
        response = requests.post(SERVER_URL, files={"image": img_encoded.tobytes()}, timeout=2)
        
        if response.status_code == 200:
            data = response.json()

            # 4. Draw the bounding boxes calculated by your PC onto the Pi's screen
            for pred in data["predictions"]:
                x1, y1, x2, y2 = pred["box"]
                label = f"{pred['class']} {pred['confidence']:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    except Exception as e:
        cv2.putText(frame, "Connecting to PC Server...", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display your framerate calculation
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"Network FPS: {fps:.1f}", (20, 440), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Raspberry Pi - Stream to PC GPU", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
