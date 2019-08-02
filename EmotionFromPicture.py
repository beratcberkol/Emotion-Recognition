from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import sys
import os
import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("Duygu Analizi")

img_counter = 0

## Opencv kullanılarak kamera açılır,
## Klavyeden "space" tuşuna basıldığında duygu analizi için kullanılacak olan fotoğraf çekilir.
## Klavyeden "esc" tuşuna basıldığında kamera kapanır.

while True:
    ret, frame = cam.read()
    cv2.imshow("Duygu Analizi", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "emotion.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        break

cam.release()

cv2.destroyAllWindows()

img_path = "emotion.png"

## Çekilen fotoğraf için kullanılmak üzere verilen frame'de insan yüzü tespit eden haarcascade classifier'ı "frontalface_default" tanımlanır.
## Tespit edilen insan yüzünden yüz ifadelerini baz alarak duygu analizi yapan daha önce eğitime sokulmuş "_mini_XCEPTION.106.0.65" modeli tanımlanır.
## Duygu analizi sonucunda tespit edilecek 7 muhtemel yüz ifadesi tanımlanır.

frontal_face = cv2.CascadeClassifier("haarcascade_files/haarcascade_frontalface_default.xml")
emotion_model = load_model("models/_mini_XCEPTION.106-0.65.hdf5", compile=True)
EMOTIONS = ["Kizgin", "Igrenmis", "Korkmuş", "Mutlu", "Uzgun", "Sasirmis", "Notr"]

# Opencv yardımıyla yolu belirtilen fotoğraf okunur ve uygun ölçülere dönüştürülür.


orig_frame = cv2.imread(img_path)
frame = cv2.imread(img_path, 0)
fFaces = frontal_face.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)

## Fotoğrafta bulunan bütün yüzler(roi) tek tek istenilen formata dönüştürülüp daha önce tanımlanan "emotion_model" kullanılanarak yüz ifadesinden duygu tahmini(preds) yapılır.
## Tekrar Opencv yardımıyla bu tahminler yüzler kare içine alınarak yazılır.

if len(fFaces):
    fFaces = sorted(fFaces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = fFaces
    roi = frame[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    preds = emotion_model.predict(roi)[0]
    emotion_probability = np.max(preds)
    emotion = EMOTIONS[preds.argmax()]
    cv2.putText(orig_frame, emotion, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

cv2.imshow('test_face', orig_frame)
cv2.imwrite('test_output/' + img_path.split('/')[-1], orig_frame)
if (cv2.waitKey(99999999) & 0xFF == ord('q')):
    sys.exit("Tesekkurler")
cv2.destroyAllWindows()

## Duygu analizinde kullanılmak üzere kamera üzerinden çekilen resim silinir.
os.remove("emotion.png")
