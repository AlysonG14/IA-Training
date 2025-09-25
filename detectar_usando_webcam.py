from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
import time

# Stream RTSP (mude para o IP/câmera desejado)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir o stream")
    exit()

# Carrega modelo leve da YOLO
model = YOLO("yolov8n.pt")

track_history = defaultdict(lambda: [])
seguir = False
deixar_rastro = False

while True:
    success, img = cap.read()
    if not success:
        print("Falha ao capturar frame")
        break

    # Rastreia ou apenas detecta
    try:
        if seguir:
            results = model.track(img, persist=True)
        else:
            results = model(img)

        for result in results:
            img = result.plot()

            if seguir and deixar_rastro:
                try:
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)

                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
                except Exception as e:
                    print(f"Erro ao processar rastro: {e}")

        cv2.imshow("Tela", img)

    except Exception as e:
        print(f"Erro ao processar frame: {e}")

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Desligando")
