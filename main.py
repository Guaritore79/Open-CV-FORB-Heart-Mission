import cv2
import os
import serial
import serial.tools.list_ports
import time
import platform
import numpy as np

# ================= 1. DETEKSI OS & MODEL =================
os_name = platform.system()

if os_name == 'Linux':
    MODEL_PATH = "modelv4.eim"
    try:
        from edge_impulse_linux.image import ImageImpulseRunner
    except Exception as e:
        print(f"Gagal memuat library. Error aslinya: {e}") 
        exit()
        
elif os_name == 'Windows':
    MODEL_PATH = "modelv4.lite" 
    try:
        import tensorflow as tf
    except ImportError:
        print("Library tensorflow belum terinstall! Jalankan: pip install tensorflow")
        exit()
else:
    print("Sistem Operasi tidak didukung!")
    exit()

# ================= 2. MENU PORT MANUAL =================
def pilih_port_manual():
    print(f"=== SETUP KONEKSI LENGAN ROBOT ({os_name}) ===")
    print("Mencari port yang tersedia...")
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("Tidak ada perangkat USB/Serial yang terdeteksi.")
        return None
        
    print("\nDaftar Port yang tersedia:")
    for i, port in enumerate(ports):
        print(f"[{i + 1}] {port.device} - {port.description}")
        
    while True:
        try:
            pilihan = int(input("\nMasukkan nomor port STM32 (misal: 1): "))
            if 1 <= pilihan <= len(ports):
                return ports[pilihan - 1].device
            else:
                print("Nomor di luar daftar. Silakan coba lagi.")
        except ValueError:
            print("Input tidak valid! Harap masukkan angka.")

PORT = pilih_port_manual()
BAUDRATE = 9600

if not PORT:
    print("Program dihentikan karena tidak ada port yang dipilih.")
    exit()

try:
    stm32 = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"\n=> Sukses terhubung ke {PORT}")
except Exception as e:
    print(f"\n=> Gagal membuka port: {e}")
    exit()

# ================= 3. INISIALISASI AI & KAMERA =================
runner = None
interpreter = None
input_details = None
output_details = None

dir_path = os.path.dirname(os.path.realpath(__file__))
model_file = os.path.join(dir_path, MODEL_PATH)

if os_name == 'Linux':
    runner = ImageImpulseRunner(model_file)
    model_info = runner.init()
    
elif os_name == 'Windows':
    print(f"Memuat model {MODEL_PATH} untuk Windows...")
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Ambil resolusi input dari model
    input_shape = input_details[0]['shape']
    model_height = input_shape[1]
    model_width = input_shape[2]

cap = cv2.VideoCapture(0)
start_time = 0
is_timing = False
stm_state = 0

print("\nMemulai deteksi... Tekan 'q' pada jendela kamera untuk keluar.")

try:
    while True:
        # --- A. Baca Data Handshaking STM32 ---
        if stm32.in_waiting > 0:
            try:
                incoming_data = stm32.readline().decode('utf-8').strip()
                if incoming_data.isdigit():
                    stm_state = int(incoming_data)
                    print(f"STM32 State: {stm_state}")
            except Exception as e:
                pass
            
        # --- B. Baca Frame Kamera ---
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_height, frame_width, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mouth_open_detected = False
        
        # --- C. Inferensi Berdasarkan OS ---
        if os_name == 'Linux':
            features, cropped = runner.get_features_from_image(img_rgb)
            res = runner.classify(features)
            
            if "bounding_boxes" in res["result"]:
                for bb in res["result"]["bounding_boxes"]:
                    if bb['label'] == 'MOP' and bb['value'] > 0.8:
                        mouth_open_detected = True
                        cv2.rectangle(frame, (bb['x'], bb['y']), 
                                      (bb['x']+bb['width'], bb['y']+bb['height']), 
                                      (0, 255, 0), 2)
                        cv2.putText(frame, f"MOP ({int(bb['value']*100)}%)", (bb['x'], bb['y']-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        elif os_name == 'Windows':
            # 1. Resize gambar sesuai permintaan model TFLite (96x96)
            img_resized = cv2.resize(img_rgb, (model_width, model_height))
            input_data = np.expand_dims(img_resized, axis=0)
            
            # 2. Cek tipe data yang diminta oleh model
            input_dtype = input_details[0]['dtype']

            # 3. Konversi format gambar (UINT8 dari kamera) agar cocok dengan model
            if input_dtype == np.int8:
                input_data = (input_data.astype(np.float32) - 128.0).astype(np.int8)
            elif input_dtype == np.float32:
                input_data = input_data.astype(np.float32) / 255.0

            # 4. Masukkan data ke model dan jalankan inferensi
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Hasil dari model FOMO berupa grid
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            
            # Grid model FOMO (contoh: 12x12). Kita memetakan skor ke ukuran frame asli
            grid_h, grid_w, num_classes = output_data.shape
            
            # Edge Impulse FOMO memunculkan probabilitas objek per sel grid
            for y in range(grid_h):
                for x in range(grid_w):
                    # Ambil nilai probabilitas MOP
                    confidence = output_data[y, x, 1] if num_classes > 1 else output_data[y, x, 0]
                    
                    # TFLite int8 biasanya memiliki rentang -128 s.d 127 atau 0-255
                    if output_details[0]['dtype'] == np.uint8 or output_details[0]['dtype'] == np.int8:
                         # Sesuaikan konversi skor ke persen
                         if output_details[0]['dtype'] == np.int8:
                             score = (confidence + 128.0) / 255.0
                         else:
                             score = confidence / 255.0
                    else:
                         score = confidence
                         
                    if score > 0.8: # Threshold 80%
                        mouth_open_detected = True
                        
                        # Hitung kordinat tengah kotak
                        cx = int((x + 0.5) / grid_w * frame_width)
                        cy = int((y + 0.5) / grid_h * frame_height)
                        
                        # Gambar kotak (karena FOMO memprediksi pusat objek, kita gambar kotak statis)
                        box_size = 50 
                        cv2.rectangle(frame, (cx - box_size, cy - box_size), 
                                      (cx + box_size, cy + box_size), 
                                      (0, 255, 0), 2)
                        cv2.putText(frame, f"MOP ({int(score*100)}%)", (cx - box_size, cy - box_size - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # --- D. Logika Timer 2 Detik ---
        if mouth_open_detected and stm_state == 0:
            if not is_timing:
                start_time = time.time()
                is_timing = True
            elif time.time() - start_time >= 2:
                print("Perintah valid! Mengirim '1' ke STM32")
                stm32.write(b'1')
                is_timing = False # Reset timer
        else:
            is_timing = False     # Reset timer jika mulut tertutup sebelum 2 detik
        
        # --- E. Tampilkan di Layar ---
        cv2.imshow(f'Robot Arm Vision (OS: {os_name})', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if runner and os_name == 'Linux':
        runner.stop()
    if 'cap' in locals():
        cap.release()
    if 'stm32' in locals() and stm32.is_open:
        stm32.close()
    cv2.destroyAllWindows()