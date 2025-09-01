from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
from pathlib import Path

# origens possíveis: image, screenshot, URL, video, YouTube, Streams -> ESP32 / Intelbras / Cameras On-Line
# mais informações em https://docs.ultralytics.com/modes/predict/#inference-sources


testIPs = [
    'rtsp://admin:123456@177.215.103.130:554/stream1',
    'rtsp://admin:123456@177.215.103.130:554/live',
    'rtsp://admin:123456@177.215.103.130:554/h264',
    'rtsp://admin:123456@177.215.103.130:554/ch1/main',
    'rtsp://admin:123456@177.215.103.130:554/Streaming/Channels/101',
]

stream_found = False
for url in testIPs:
    print(f"Testando stream: {url}")
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        print(f"✅ Conectado com sucesso em: {url}")
        stream_found = True
        break
    else:
        print(f"❌ Falha ao conectar em: {url}")
        cap.release()

if not stream_found:
    print("⚠️ Nenhuma das URLs RTSP funcionou. Verifique IP, porta, usuário/senha ou firewall.")
    exit(1)

# Usa modelo da Yolo
# Model	    size    mAPval  Speed       Speed       params  FLOPs
#           (pixels) 50-95  CPU ONNX A100 TensorRT   (M)     (B)
#                           (ms)        (ms)
# YOLOv8n	640	    37.3	80.4	    0.99	    3.2	    8.7
# YOLOv8s	640	    44.9	128.4	    1.20	    11.2	28.6
# YOLOv8m	640	    50.2	234.7	    1.83	    25.9	78.9
# YOLOv8l	640	    52.9	375.2	    2.39	    43.7	165.2
# YOLOv8x	640	    53.9	479.1	    3.53	    68.2	257.8

model = YOLO("yolov8n.pt")

track_history = defaultdict(lambda: [])
seguir = True
deixar_rastro = True

while True:
    success, img = cap.read()

    if success:
        if seguir:
            results = model.track(img, persist=True)
        else:
            results = model(img)

        # Process results list
        for result in results:
            # Visualize the results on the frame
            img = result.plot()

            if seguir and deixar_rastro:
                try:
                    # Get the boxes and track IDs
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    # Plot the tracks
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y center point
                        if len(track) > 30:  # retain 90 tracks for 90 frames
                            track.pop(0)

                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
                except:
                    pass

        cv2.imshow("Tela", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("desligando")