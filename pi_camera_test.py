import sys
import numpy as np
import cv2
import requests
import time

# 🛑 CONFIGURATION: Set this to your Mac Server's actual local Wi-Fi IP address!
MAC_SERVER_IP = "192.168.11.102"
SERVER_URL = f"http://{MAC_SERVER_IP}:3000/predict"

print("Raspberry Pi pipeline active. Transmitting frames to server...")

buffer = b""

try:
    while True:
        # Read incoming data blocks from the system camera pipe stream
        chunk = sys.stdin.buffer.read(4096)
        if not chunk:
            break
        buffer += chunk

        # Track full JPEG segments
        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')

        if a != -1 and b != -1 and b > a:
            jpg_bytes = buffer[a:b+2]
            buffer = buffer[b+2:]  # Advance buffer state

            try:
                # Fast asynchronous upload directly to your Mac server
                requests.post(SERVER_URL, files={"image": jpg_bytes}, timeout=1)
            except Exception:
                # Silently ignore frame drops if network latency spikes
                pass

except KeyboardInterrupt:
    print("\nStopping Pi hardware client stream...")
