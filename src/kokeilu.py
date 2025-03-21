from ultralytics import YOLO


# mallit------------------------------------------------------------------------------

model = YOLO("yolov8n.pt")  # YOLOv8:n oma esiharjoitettu malli


#lähteet------------------------------------------------------------------------------

source = "0"


# Tulokset, näytäarvo (show) näyttää lähteen samalla, kun siitä tutkitaan kohteita, muuten kuvaa tai kuvavirtaa ei näytetä
results = model.predict(source, show = True)


#YOLOv8 tulostukset kehotteeseen
print(results)
