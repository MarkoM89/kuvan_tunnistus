from ultralytics import YOLO


# load a partially trained model
model = YOLO(r"C:\Users\Marko\Desktop\metropolia koulu\Opinnäytetyö\ESP-EYE\kuvan_tunnistus\runs\detect\train6\weights\last.pt")  # YOLOv8 nano model, epoch 100

# Resume training
results = model.train(resume=True)