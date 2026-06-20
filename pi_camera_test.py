import sys
import numpy as np
import cv2
import requests
import threading
import queue
import time

# CONFIGURATION: Set this to your Mac Server's actual local Wi-Fi IP address!
MAC_SERVER_IP = "192.168.11.102"
SERVER_URL = f"http://{MAC_SERVER_IP}:3000/predict"

# A thread-safe queue holding only the absolute freshest frame to send
# maxsize=1 ensures we never accumulate a backlog of old, lagging frames
frame_queue = queue.Queue(maxsize=1)

def network_sender_thread():
    """Background worker that handles sending frames without blocking the camera read loop."""
    print("🚀 Network sender thread initialized.")
    while True:
        # Get the latest frame from the queue (blocks until a frame is available)
        jpg_bytes = frame_queue.get()
        
        try:
            # Send the frame over the network
            requests.post(SERVER_URL, files={"image": jpg_bytes}, timeout=0.5)
        except Exception:
            # Pass silently if network hiccups happen
            pass
        
        # Tell the queue the task is done
        frame_queue.task_done()

# Start the background sender thread
sender_worker = threading.Thread(target=network_sender_thread, daemon=True)
sender_worker.start()

print("Raspberry Pi pipeline active. Transmitting frames smoothly to server...")
buffer = b""

try:
    while True:
        # Read incoming data blocks from the system camera pipe stream
        chunk = sys.stdin.buffer.read(4096)
        if not chunk:
            break
        buffer += chunk

        # Find full JPEG segments
        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')

        if a != -1 and b != -1 and b > a:
            jpg_bytes = buffer[a:b+2]
            buffer = buffer[b+2:]  # Advance buffer state

            # If the queue already has an unsent frame sitting in it,
            # clear it out immediately so we don't build up a lag backlog.
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                    frame_queue.task_done()
                except queue.Empty:
                    pass

            # Push the absolute newest frame to the background worker thread
            frame_queue.put(jpg_bytes)

except KeyboardInterrupt:
    print("\nStopping Pi hardware client stream...")
