from ultralytics import YOLO

# Lataa malli
#model = YOLO("yolov8n.pt")  # lataa esiopetettu malli (suositeltu opetusvaiheeseen)

model = YOLO("yolov8m.pt")  # lataa esiopetettu malli (suositeltu opetusvaiheeseen)


# Aineisto
results = model.train(data="C:\\Users\Marko\Desktop\metropolia koulu\Opinnäytetyö\Bird species.v3i.yolov8\data.yaml", epochs=100) #epoch-perusarvo on 100