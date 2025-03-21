from ultralytics import YOLO


# lataa osittain opetettu malli
model = YOLO("C:\\Users\Marko\Desktop\metropolia koulu\Opinnäytetyö\ESP-EYE\kuvan_tunnistus\runs\detect\train6\weights\last.pt")


# Jatka opetusta
results = model.train(resume=True)