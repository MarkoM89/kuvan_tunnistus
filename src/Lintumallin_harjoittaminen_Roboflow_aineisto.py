from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

#model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

#model = YOLO() # train from scratch

# Train the model
results = model.train(data="C:\\Users\Marko\Desktop\metropolia koulu\Opinnäytetyö\Bird species.v3i.yolov8\data.yaml", epochs=2) #Default epoch value is 100