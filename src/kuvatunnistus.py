from ultralytics import YOLO
import time
import cv2






def malli():
    #mallit------------------------------------------------------------------------------
    #Yolov8:an mukana tulevat mallit, jolla voi eri kohteita tunnistaa yleistasolla------------------------------------------------------------------------------

    toiminto = int(input("Valitse malli,\n1: YOLOv8:n nanomalli\n"
            + "2: YOLOv8:n keskikokoinen malli \n"
            + "3: Lintumalli, joka on harjoitettu nanomallilla\n"
            + "4: Lintumalli, joka on harjoitettu keskikokoisella mallilla\n"))

    if toiminto == 1:
        model = YOLO("yolov8_mallit/yolov8n.pt")  # YOLOv8:n oma esiharjoitettu malli

    elif toiminto == 2:
        model = YOLO("yolov8_mallit/yolov8m.pt")  # YOLOv8:n oma esiharjoitettu malli

    elif toiminto == 3:
        #lintumallit lintulajien tunnistukseen------------------------------------------------------------------------------

        model = YOLO("runs/detect/train6/weights/best.pt") # nanomalli, joka on kesken ja 10 kertaa käyty aineisto läpi. Epoch-arvo on 100

    elif toiminto == 4:

        model = YOLO("runs/detect/train7/weights/best.pt") # keskikokoinen malli, joka on valmis ja 2 kertaa käyty aineisto läpi. Epoch-arvo on 2


    return model



def lahde():
    #lähteet------------------------------------------------------------------------------

    toiminto = int(input("Valitse lähde,\n1: RTSP-linkki\n"
         + "2: Oletusverkkokamera\n"
         + "3: Koekuva linja-autosta\n"
         + "4: Koekuva sinitöyhtönärhestä\n"
         + "5: Koekuva kenttäsirkkulista\n"))

    if toiminto == 1:
        #ESP-EYE:n kuvavirran linkki tai verkkokamera, kun lähde on 0
        source = 'rtsp://192.168.0.12:8554/mjpeg/1'

    elif toiminto == 2:    
        source = "0"

    elif toiminto == 3:
        #Koekuva linja-autosta ja ihmisistä, jolla voi kokeilla YOLOv8:n omia malleja
        source = "koekuvat/bus.jpg"

    elif toiminto == 4:
        #Kaksi koekuvaa linnuista, joilla voi kokeilla lintumallien toimivuutta
        source = "koekuvat/bird-2698953_1280.jpg"

    elif toiminto == 5:    
        source = "koekuvat/chipping-sparrow-7989209_1280.jpg"

    return source



def main():
        
    #muuttujat
    toiminto = int(-1)



    while toiminto != 0:

        toiminto = int(input("1: Alusta ohjelma (valitse lähde ja malli)\n"
            + "2: Vaihda malli \n"
            + "3: Vaihda lähde\n"
            + "4: Aloita tunnistus\n"
            + "0: Sammuta ohjelma\n"))


        if toiminto == 1:
            model = malli()
            source = lahde()

        if toiminto == 2:
            model = malli()

        if toiminto == 3:
            source = lahde()

        if toiminto == 4:


            #Tulostukset ja viiveet------------------------------------------------------------------------------

            # Tulokset, näytäarvo (show) näyttää lähteen samalla, kun siitä tutkitaan kohteita, muuten kuvaa tai kuvavirtaa ei näytetä
            # Luotu ikkuna suljetaan painamalla jotain näppäimistön painiketta käyttämällä cv2-kirjastoa
            results = model.predict(source, show = True)


            #YOLOv8 tulostukset kehotteeseen
            print(results)       


            cv2.waitKey(0)
            cv2.destroyAllWindows()
            

        if toiminto == 0:
            print("Ohjelma sammutetaan")

        else:
            print("Ohjelma toimii luvuilla 1-4 sekä 0")





if __name__ == '__main__':
    main()