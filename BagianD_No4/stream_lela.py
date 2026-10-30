import cv2
import time
import threading
from flask import Flask, Response, render_template_string, request
from ultralytics import YOLO

app = Flask(__name__)


# Gunakan hand_yolov8n.pt
model = YOLO("hand_yolov8n.pt")

# Parameter adaptif untuk jaringan tidak stabil (kelancaran > kualitas)
stream_config = {
    "scale": 1.0,        # 1.0 = 640x480 (Normal), 0.5 = 320x240 (Jaringan Buruk)
    "jpeg_quality": 70   # 70 = Normal, 30 = Kompresi Tinggi (Hemat Bandwidth)
}

# Variabel berbagi antar-thread (Zero-Queue / Buffer 1 Frame)
latest_raw_frame = None
latest_boxes = []
inference_ms = 0.0
lock = threading.Lock()
running = True

def camera_capture_thread():
    """Thread 1: Khusus membaca webcam secepat mungkin tanpa menunggu AI"""
    global latest_raw_frame, running
    cap = cv2.VideoCapture(0)
    # Set buffer kamera ke 1 agar tidak ada tumpukan frame lama (mencegah delay)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while running:
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_raw_frame = frame
        else:
            time.sleep(0.01)
    cap.release()

def yolo_inference_thread():
    """Thread 2: Menjalankan deteksi YOLO di latar belakang secara asinkron"""
    global latest_raw_frame, latest_boxes, inference_ms, running
    while running:
        frame_to_process = None
        with lock:
            if latest_raw_frame is not None:
                frame_to_process = latest_raw_frame.copy()

        if frame_to_process is not None:
            t_start = time.time()
            # Inferensi dengan ukuran 320 agar ringan di CPU
            results = model(frame_to_process, imgsz=320, conf=0.5, verbose=False)
            
            temp_boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label_name = model.names[cls_id]
                    temp_boxes.append((x1, y1, x2, y2, conf, label_name))
            
            with lock:
                latest_boxes = temp_boxes
                inference_ms = (time.time() - t_start) * 1000.0
        else:
            time.sleep(0.02)

def generate_frames():
    """Generator pengiriman frame ke klien dengan metrik FPS dan Latensi"""
    prev_time = time.time()
    fps = 0.0

    while True:
        t_frame_start = time.time()
        with lock:
            if latest_raw_frame is None:
                continue
            frame = latest_raw_frame.copy()
            boxes = list(latest_boxes)
            ai_lat = inference_ms

        # 1. Gambar kotak deteksi tangan dari hasil thread AI terakhir
        for (x1, y1, x2, y2, conf, name) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 2. Adaptasi resolusi jika jaringan memburuk
        scale = stream_config["scale"]
        if scale < 1.0:
            new_w = int(frame.shape[1] * scale)
            new_h = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 3. Hitung FPS dan Latensi Encoding
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            fps = (fps * 0.9) + ((1.0 / dt) * 0.1)

        proc_lat_ms = (time.time() - t_frame_start) * 1000.0

        # 4. Tampilkan Metrik di Pojok Kiri Atas Video
        info_text1 = f"Stream FPS: {fps:.1f} | Frame Lat: {proc_lat_ms:.1f} ms"
        info_text2 = f"AI Lat: {ai_lat:.1f} ms | Q: {stream_config['jpeg_quality']}%"
        cv2.putText(frame, info_text1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, info_text2, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 5. Kompresi JPEG sesuai kondisi jaringan
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), stream_config["jpeg_quality"]]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    html = """
    <html>
    <head>
        <title>UAV Live Stream - LELA</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background:#121212; color:#fff; font-family:sans-serif; text-align:center; padding:10px; }
            img { width:100%; max-width:640px; border:2px solid #00e676; border-radius:8px; }
            .btn { padding:10px 16px; margin:6px; border:none; border-radius:5px; font-weight:bold; cursor:pointer; }
            .normal { background:#00e676; color:#000; }
            .poor { background:#ff5252; color:#fff; }
        </style>
    </head>
    <body>
        <h2>UAV Live Stream + YOLOv8n Hand Detection</h2>
        <img src="/video_feed"><br><br>
        <p>Mode Adaptasi Jaringan (Utamakan Kelancaran):</p>
        <a href="/set_mode?mode=normal"><button class="btn normal">Jaringan Stabil (640p, Q:70)</button></a>
        <a href="/set_mode?mode=poor"><button class="btn poor">Jaringan Buruk / Low Bandwidth (320p, Q:30)</button></a>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/set_mode')
def set_mode():
    mode = request.args.get('mode', 'normal')
    if mode == 'poor':
        stream_config["scale"] = 0.5
        stream_config["jpeg_quality"] = 30
    else:
        stream_config["scale"] = 1.0
        stream_config["jpeg_quality"] = 70
    return index()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t1 = threading.Thread(target=camera_capture_thread, daemon=True)
    t2 = threading.Thread(target=yolo_inference_thread, daemon=True)
    t1.start()
    t2.start()
    print("Server Streaming menyala! Buka http://<IP_LAPTOP_ANDA>:5000 di Browser HP.")
    app.run(host='0.0.0.0', port=5000, threaded=True)