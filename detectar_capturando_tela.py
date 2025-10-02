import cv2
import pandas as pd
from ultralytics import solutions
import time

cap = cv2.VideoCapture(r"C:/Users/dsadm/Desktop/IA para Reconhecimento Facial/videos/Pessoas.mp4")
assert cap.isOpened(), "Error reading video file"

# Configurações da região
region_points = [(20, 400), (1080, 400)]  # linha para contagem

# Configuração do video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Inicializar o contador de objetos
counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model="yolov8n.pt",  # ou "yolo11n.pt" se disponível
    classes=[0],  # classe 0 = 'person' no COCO
    tracker="bytetrack.yaml",
)

# Lista para armazenar os dados
data = []
start_time = time.time()

# Processar o vídeo
while cap.isOpened():
    success, im0 = cap.read()
    
    if not success:
        print("Video processing complete.")
        break

    # Processar frame com o contador
    results = counter(im0)
    
    # Extrair informações de contagem
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    current_time = current_frame / fps if fps > 0 else 0
    
    # Obter contagem de pessoas (já filtrada pela classe 0)
    person_count = counter.in_count  # pessoas entrando
    # person_count = counter.out_counts  # pessoas saindo
    # person_count = counter.counting_dict.get(0, 0)  # total atual
    
    print(f"Frame {current_frame}: {person_count} pessoas")
    
    # Armazenar dados
    data.append([current_frame, current_time, person_count])
    
    # Escrever frame processado
    if hasattr(results, 'plot') and results.plot() is not None:
        video_writer.write(results.plot())
    else:
        video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Criar DataFrame e exportar
df = pd.DataFrame(data, columns=["Frame", "Tempo(s)", "Quantidade_Pessoas"])

# Adicionar estatísticas básicas
print("\n--- Estatísticas da Contagem ---")
print(f"Total de frames processados: {len(df)}")
print(f"Média de pessoas por frame: {df['Quantidade_Pessoas'].mean():.2f}")
print(f"Máximo de pessoas em um frame: {df['Quantidade_Pessoas'].max()}")

# Exportar para Excel
try:
    df.to_excel("contagem_de_pessoas.xlsx", index=False)
    print("\nDados exportados para 'contagem_de_pessoas.xlsx'")
except Exception as e:
    print(f"Erro ao exportar para Excel: {e}")
    # Exportar como CSV alternativo
    df.to_csv("contagem_de_pessoas.csv", index=False)
    print("Dados exportados para 'contagem_de_pessoas.csv'")