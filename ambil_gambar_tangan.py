import cv2
import os

angka = input("Masukkan label angka (0-9): ")
folder = f"dataset/{angka}"
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Ambil Gambar Tangan', frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite(f"{folder}/{angka}_{count}.jpg", frame)
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
