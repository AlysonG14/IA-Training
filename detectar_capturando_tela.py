from ultralytics import YOLO
import cv2
from collections import defaultdict
from windowcapture import WindowCapture
import numpy as np
import ctypes

# Detecta resolução da tela
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

# Configuração do WindowCapture
offset_x = 400
offset_y = 300
wincap = WindowCapture(size=(1024, 768), origin=(offset_x, offset_y))

# Carrega o modelo YOLO
model = YOLO("yolov8n.pt")
# model = YOLO("runs/detect/train4/weights/best.pt")  # Modelo customizado

track_history = defaultdict(lambda: [])
seguir = True
deixar_rastro = True

# Configura janela em tela cheia
cv2.namedWindow("Tela", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Tela", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    img = wincap.get_screenshot()

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
            except:
                pass

    # Redimensiona imagem para a tela inteira
    img = cv2.resize(img, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)

    # Exibe a imagem
    cv2.imshow("Tela", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
print("desligando")
