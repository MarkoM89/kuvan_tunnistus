from ultralytics import YOLO
import time

# mallit------------------------------------------------------------------------------

#model = YOLO("yolov8n.pt")  # YOLOv8:n oma esiharjoitettu malli

model = YOLO("yolov8m.pt")  # YOLOv8:n oma esiharjoitettu malli

#model = YOLO("runs/detect/train6_yksi_kierros/weights/best.pt") # nanomalli, joka on kesken ja yksi kerta käyty aineisto läpi. Epoch-arvo on 100

#model = YOLO("runs/detect/train6/weights/best.pt") # nanomalli, joka on kesken ja 10 kertaa käyty aineisto läpi. Epoch-arvo on 100

#model = YOLO("runs/detect/train7/weights/best.pt") # keskikokoinen malli, joka on valmis ja 2 kertaa käyty aineisto läpi. Epoch-arvo on 2

#model = YOLO("runs/detect/train8/weights/best.pt") # nanomalli, joka on valmis ja 2 kertaa käyty aineisto läpi. Epoch-arvo on 2



#lähteet------------------------------------------------------------------------------

#ESP-EYE:n kuvavirran linkki tai verkkokamera, kun lähde on 0
#source = 'rtsp://192.168.0.12:8554/mjpeg/1'
source = "0"


#Koekuva linja-autosta ja ihmisistä, jolla voi kokeilla YOLOv8:n omia malleja
#source = "bus.jpg"

#Kaksi koekuvaa linnuista, joilla voi kokeilla mallin toimivuutta
#source = "bird-2698953_1280.jpg"
#source = "chipping-sparrow-7989209_1280.jpg"



# Tulokset, näytäarvo (show) näyttää lähteen samalla, kun siitä tutkitaan kohteita, muuten kuvaa tai kuvavirtaa ei näytetä
# Viive on kuvia varten, jotta tuloksen ehtii nähdä, kun näytäarvo on tosi
results = model.predict(source, show = True)
#time.sleep(5)

#YOLOv8 tulostukset kehotteeseen
print(results)