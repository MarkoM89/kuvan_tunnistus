from ultralytics import YOLO

# Lataa malli
model = YOLO("yolov8n.pt")  # lataa esiharjoitettu malli (suositeltu harjoittamiseen)

#model = YOLO("yolov8m.pt")  # lataa esiharjoitettu malli (suositeltu harjoittamiseen)


# Harjoita malli
results = model.train(data="C:\\Users\Marko\Desktop\metropolia koulu\Opinnäytetyö\Bird species.v3i.yolov8\data.yaml", epochs=2) #epoch-perusarvo on 100