from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
import time

# mallit------------------------------------------------------------------------------

model = YOLO("yolov8n.pt")  # YOLOv8:san oma esiharjoitettu malli

#model = YOLO("yolov8m.pt")  # YOLOv8:san oma esiharjoitettu malli

#model = YOLO("runs/detect/train6_yksi_kierros/weights/best.pt") # nanomalli, joka on kesken ja yksi kerta käyty ainesitoa läpi. Epoch-arvo on 100

#model = YOLO("runs/detect/train6/weights/best.pt") # nanomalli, joka on kesken ja 10 kertaa käyty aineisto läpi. Epoch-arvo on 100

#model = YOLO("runs/detect/train7/weights/best.pt") # keskikokoinen malli, joka on valmis ja 2 kertaa käyty aineisto läpi. Epoch-arvo on 2

#model = YOLO("runs/detect/train8/weights/best.pt") # nanomalli, joka on valmis ja 2 kertaa käyty aineisto läpi. Epoch-arvo on 2



#lähteet------------------------------------------------------------------------------

#ESP-EYE:n kuvavirran linkki
#source = 'rtsp://192.168.0.12:8554/mjpeg/1'


#Kaksi koekuvaa linnuista, joilla voi kokeilla mallin toimivuutta
#source = "bird-2698953_1280.jpg"
#source = "istockphoto-172899256-1024x1024.jpg"
source = "bus.jpg"


# Tulokset, näytäarvo (show) näyttää lähteen samalla, kun siitä tutkitaan kohteita, muuten kuvaa tai kuvavirtaa ei näytetä
# Viive on kuvia varten, jotta tuloksen ehtii nähdä, kun näytäarvo on tosi
results = model.predict(source, show = True)
time.sleep(6)

#YoloV8 tulostukset kehotteeseen
print(results)