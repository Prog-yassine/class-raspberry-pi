import cv2
import requests
import time
import sys

# Change this to your PC's actual local IP address
PC_IP_ADDRESS = "192.168.1.50"  
SERVER_URL = f"http://{PC_IP_ADDRESS}:5000/predict"

print("Step 1: Attempting to open the camera...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

print("Step 2: Configuring camera settings...")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ ERROR: Could not open the camera interface at all.")
    sys.exit()
else:
    print("✅ Camera interface opened successfully!")

print("Step 3: Entering streaming loop. Press 'q' to quit.")
while cap.isOpened():
    start_time = time.time()
    
    # 1. Grab Frame
    success, frame = cap.read()
    if not success or frame is None:
        print("⚠️ Camera warning: Frame is empty or failing to read. Retrying...")
        time.sleep(0.5)
        continue
    
    # 2. Compress Frame
    success_encode, img_encoded = cv2.imencode('.jpg', frame)
    if not success_encode:
        print("❌ ERROR: Failed to compress frame to JPEG.")
        continue

    # 3. Send to PC Server
    try:
        print(f"Sending frame to server at {SERVER_URL}...", end="\r")
        response = requests.post(SERVER_URL, files={"image": img_encoded.tobytes()}, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            # Draw boxes
            for pred in data["predictions"]:
                x1, y1, x2, y2 = pred["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, pred['class'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f"\n❌ Server returned an error status code: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("\n❌ Network Error: Request timed out. The PC server is taking too long or isn't responding.")
    except requests.exceptions.ConnectionError:
        print("\n❌ Network Error: Could not reach the PC server. Check the IP address or your Wi-Fi network.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

    cv2.imshow("Debug Window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
