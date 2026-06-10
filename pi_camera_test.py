import cv2
from ultralytics import YOLO
import time

def main():
    # Load your pure Python-exported NCNN folder
    # Make sure 'best_ncnn_model' is in the same folder as this script
    model = YOLO("best_ncnn_model") 

    # Open the Pi camera or USB webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("YOLOv8 NCNN running on Raspberry Pi! Press 'q' to quit.")

    while cap.isOpened():
        start_time = time.time()
        success, frame = cap.read()
        if not success:
            break

        # Run inference using the exact 320x320 size we exported
        results = model.predict(source=frame, imgsz=320, conf=0.30, verbose=False)

        for r in results:
            annotated_frame = r.plot()

        # Calculate FPS to see your hardware performance gains
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Raspberry Pi 4 - YOLOv8 NCNN", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()