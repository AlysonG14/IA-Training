from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
import time

# Carrega o modelo YOLO
try:
    model = YOLO("runs/detect/train8/weights/best.pt")
    print("✅ Modelo YOLO carregado com sucesso")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    exit()

# Inicializa a webcam (fallback)
cap = cv2.VideoCapture(0)  # 0 = webcam padrão
if not cap.isOpened():   
    print("❌ Webcam não encontrada! Criando fallback...")
    # Fallback para imagem estática se webcam falhar
    webcam_available = False
else:
    webcam_available = True
    print("✅ Webcam inicializada com sucesso")

# Variáveis de estado
track_history = defaultdict(lambda: [])
seguir = True
deixar_rastro = True

# Configura a janela
cv2.namedWindow("Tela", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tela", 640, 480)

print("\n🔄 Iniciando detecção pela WEBCAM...")
print("Controles:")
print("q - Sair | s - Toggle rastreamento | r - Toggle rastro | c - Limpar rastros")

# Variáveis para FPS
frame_count = 0
start_time = time.time()
fps = 0.0

while True:
    # CAPTURA DA WEBCAM
    if webcam_available:
        ret, img = cap.read()
        if not ret:
            print("⚠️ Erro na webcam, usando fallback...")
            webcam_available = False
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Webcam falhou", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Fallback: imagem de teste com movimento
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Adiciona algum movimento simulado para teste
        center_x = int(320 + 100 * np.sin(frame_count * 0.1))
        center_y = int(240 + 80 * np.cos(frame_count * 0.1))
        cv2.circle(img, (center_x, center_y), 30, (0, 255, 0), -1)
        cv2.putText(img, "Modo Simulacao - Use Webcam", (50, 440), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # CONVERTE CORES (BGR para RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # PROCESSAMENTO YOLO
    try:
        if seguir:
            results = model.track(
                source=img_rgb, 
                persist=True, 
                verbose=False, 
                conf=0.5,
                tracker="bytetrack.yaml"
            )
        else:
            results = model(
                source=img_rgb, 
                verbose=False, 
                conf=0.5
            )
    except Exception as e:
        print(f"❌ Erro no YOLO: {e}")
        results = []

    # PROCESSAMENTO DOS RESULTADOS
    annotated_frame = img.copy()  # Mantém BGR para exibição
    
    if results:
        for result in results:
            # Plota as detecções (result.plot() retorna imagem BGR)
            annotated_frame = result.plot()
            
            # RASTREAMENTO
            if (seguir and deixar_rastro and 
                result.boxes is not None and 
                result.boxes.id is not None):
                
                try:
                    boxes = result.boxes.xywh.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        
                        # Limita o histórico
                        if len(track) > 30:
                            track.pop(0)
                        
                        # Desenha o rastro
                        if len(track) > 1:
                            points = np.array(track, dtype=np.int32)
                            cv2.polylines(
                                annotated_frame, 
                                [points], 
                                isClosed=False, 
                                color=(230, 0, 0), 
                                thickness=3
                            )
                except Exception as e:
                    # print(f"⚠️ Erro no rastreamento: {e}")
                    continue

    # CÁLCULO DE FPS
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = current_time
        print(f"📊 FPS: {fps:.1f}")

    # PREPARAÇÃO PARA EXIBIÇÃO
    display_frame = cv2.resize(annotated_frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    
    # ADICIONA INFORMAÇÕES NA TELA
    status_text = f"Track: {seguir} | Rastro: {deixar_rastro} | FPS: {fps:.1f}"
    cv2.putText(
        display_frame, 
        status_text, 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        (0, 255, 0), 
        2
    )
    
    # EXIBE A IMAGEM
    cv2.imshow("Tela", display_frame)

    # CONTROLES DE TECLADO
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        seguir = not seguir
        estado = "LIGADO" if seguir else "DESLIGADO"
        print(f"🔁 Rastreamento: {estado}")
    elif key == ord('r'):
        deixar_rastro = not deixar_rastro
        estado = "LIGADO" if deixar_rastro else "DESLIGADO"
        print(f"🎯 Rastro: {estado}")
    elif key == ord('c'):
        track_history.clear()
        print("🧹 Rastros limpos")
    elif key == ord('w'):
        # Tenta alternar para webcam se disponível
        if not webcam_available and cap.isOpened():
            webcam_available = True
            print("🔄 Alternando para webcam")

# LIMPEZA FINAL
if webcam_available:
    cap.release()
cv2.destroyAllWindows()
print("🛑 Programa finalizado com sucesso!")