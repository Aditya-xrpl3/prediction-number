import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = []
y = []

for label in os.listdir('dataset'):
    folder = os.path.join('dataset', label)
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        X.append(img.flatten())
        y.append(int(label))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Akurasi:', accuracy_score(y_test, y_pred))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (64, 64)).flatten().reshape(1, -1)
    pred = model.predict(img)[0]
    cv2.putText(frame, str(pred), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
    cv2.imshow('Prediksi Angka Tangan', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
