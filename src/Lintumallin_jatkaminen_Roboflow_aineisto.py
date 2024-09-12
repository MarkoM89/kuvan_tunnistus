from ultralytics import YOLO


# lataa osittain harjoitettu malli
model = YOLO(r"C:\Users\Marko\Desktop\metropolia koulu\Opinnäytetyö\ESP-EYE\kuvan_tunnistus\runs\detect\train6\weights\last.pt")


# Jatka harjoittamista
results = model.train(resume=True)