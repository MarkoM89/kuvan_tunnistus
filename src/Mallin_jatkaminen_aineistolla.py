from ultralytics import YOLO


# lataa osittain opetettu malli
model = YOLO("C:\\Users\kayttaja\projekti\runs\detect\train6\weights\last.pt")


# Jatka opetusta
results = model.train(resume=True)