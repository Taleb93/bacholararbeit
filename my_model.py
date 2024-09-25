import cv2
import requests
from roboflow import Roboflow

# Initialisiere die Verbindung zu Roboflow mit deinem API-Schlüssel
rf = Roboflow(api_key="hWhXSGZFtQtPcSQD2B6b")
project = rf.workspace().project("self-driving-bike")
model = project.version("1").model

# Lade das Video
cap = cv2.VideoCapture('trafic.mp4')

# Führe die Vorhersage auf dem Video durch
job_id, signed_url, expire_time = model.predict_video(
    "trafic.mp4",  # Aktualisiere den Pfad zur Videodatei
    fps=5,
    prediction_type="batch-video",
)

# Warten, bis die Vorhersagen für das Video abgeschlossen sind
results = model.poll_until_video_results(job_id)

# Lade die Vorhersageergebnisse
for frame_idx, result in enumerate(results):
    ret, frame = cap.read()
    if not ret:
        break  # Wenn es keine Frames mehr gibt, brich die Schleife ab
    
    # Zeichne die Bounding Boxes und Labels auf die Frames
    for prediction in result['predictions']:
        x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        label = prediction['class']

        # Berechne die Eckpunkte des Bounding Boxes
        start_point = (int(x - w / 2), int(y - h / 2))
        end_point = (int(x + w / 2), int(y + h / 2))

        # Zeichne das Rechteck (Bounding Box) auf dem Frame
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

        # Zeichne das Label auf dem Frame
        cv2.putText(frame, label, (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Zeige den Frame an (optional)
    cv2.imshow('Frame', frame)

    # Warte für eine kurze Zeit (press 'q' to quit)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Beende das Video und alle Fenster
cap.release()
cv2.destroyAllWindows()

