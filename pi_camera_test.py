import cv2
import requests
import time
import sys
import numpy as np

# 1. Configuration Settings
PC_IP_ADDRESS = "192.168.11.102"  # <-- Make sure this matches your home PC's IP!
SERVER_URL = f"http://{PC_IP_ADDRESS}:3000/predict"

print("Raspberry Pi streaming client started. Reading from pipe stream...")

# We read raw MJPEG chunks from the system pipe buffer
# JPEG images always start with the bytes b'\xff\xd8' and end with b'\xff\xd9'
buffer = b""

try:
    while True:
        start_time = time.time()
        
        # Read a chunk of bytes from the system camera pipe
        chunk = sys.stdin.buffer.read(4096)
        if not chunk:
            break
        buffer += chunk

        # Find the start and end of a complete JPEG frame in the stream
        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')

        if a != -1 and b != -1 and b > a:
            jpg_bytes = buffer[a:b+2]
            buffer = buffer[b+2:]  # Clear out processed bytes from the buffer

            # 2. Decode the raw bytes into a visual OpenCV frame matrix safely
            np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # 3. Send the image over the network to your home PC's GPU
            try:
                response = requests.post(SERVER_URL, files={"image": jpg_bytes}, timeout=2)
                
                if response.status_code == 200:
                    data = response.json()

                    # 4. Paint bounding boxes onto the frame
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

            # 5. Render the frame
            cv2.imshow("Raspberry Pi Client Window", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nStopping stream...")

cv2.destroyAllWindows()
