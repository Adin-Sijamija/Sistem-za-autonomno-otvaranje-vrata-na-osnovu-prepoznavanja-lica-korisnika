from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
from datetime import datetime
import cv2
import os
import serial
import playsound
from collections import deque

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--Detektor", required=True,
                help="Put do detektora")
ap.add_argument("-m", "--embedding-model", required=True,
                help="put do embedding-modela")
ap.add_argument("-r", "--Prepoznavac", required=True,
                help="Put do prepoznovaca")
ap.add_argument("-l", "--Labela", required=True,
                help="Put do enkodiranja")
args = vars(ap.parse_args())

# očitavanje detektora
protoPath = os.path.sep.join([args["Detektor"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["Detektor"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# očitavanje  face embedding modela
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
        
# očitavanje SVM modela za prepoznavanje lica i imena korisnika
recognizer = pickle.loads(open(args["Prepoznavac"], "rb").read())
le = pickle.loads(open(args["Labela"], "rb").read())

# aktiviraj kameru
vs = cv2.VideoCapture(0)
#pokreni arduino komunikaciju
arduino = serial.Serial('COM3', 9600)
time.sleep(2.0)

DeckLica=deque([],maxlen=50)

UspjesnaIdetifikacija=False
KorisnickoIme=""
NepoznatiKorisnik=False

#BGR Inicializacija tupel boja okvira lica
Crvena = (0, 0, 255)
Plave = (255, 0, 0)
Zelena = (0, 255, 0)
BojaOkvira = Crvena

#Funkcija za brojanje slučaja imena zadnje prepoznatog korisnika
def ProcenatAutorizacije(ime):
    rezultat=DeckLica.count(ime)
    #U slučaju da nepoznata osoba ili Vjerovatno korisnik vrati 0
    if(ime=="Nepoznata Osoba"):
        return 0
        pass

    if("Vjerovatno" in ime):
        return 0
        pass
    #ako je korektan korisnik vrati broj slučaja imena u deck-u *2 da predstavi procenat autorizacije
    return rezultat*2

def BrojJednakihAutorizacija(ime):
#funkcija vraća uspješnost autorizacije deque-ovih 50 polja
#ako imamo svih 50 polja ispunjenim istim vrijednostima prihvaćamo to kao uspješnu autorizaciju i vraćamo "1"
#ako nemamo 50 istih vrijednosti vraćamo 0 kao status quo
#ako imamo 50 istih vrijednosti sa vjrijednošću "Nepoznata Osoba" prekidamo autentifikaciju i snimamo video kako bi
#kasnije vidjeli ko je to bio

    #ukoliko nemamo 50 provjera u deque ignorišemo provjeru 
    if(len(DeckLica)<50):
        return 0
    
    rezultat=DeckLica.count(ime)
    if(rezultat==50):
        if(ime=="Nepoznata Osoba"):
            return -1
            pass

        if("Vjerovatno" in ime):
            return 0
            pass

        return 1
        pass

    return 0



# dok se snima video
while True:
    # uzmi sliku iz videa
    ret, frame = vs.read()
    #dohvati visinu i širinu za kasniju upotrebu
    (h, w) = frame.shape[:2]

    if(UspjesnaIdetifikacija):
        print("",flush=True)
        #Proizvodi zvuk za otvaranje vrata
        playsound.playsound("Door-buzzer.mp3",False)
        #pošalji arduinu inkodirano ime korisnika i ">" znak koji predstavlja kraj poruke
        Poruka=KorisnickoIme+">"
        arduino.write(str.encode(Poruka)) 
        print("[INFO] Komanda poslana")
        time.sleep(1.0) #čekaj odgovor arduina
        while True:
            print("[INFO] Čekanje signala zaključavanja...")
            ReadData=arduino.readline().decode("ascii") #očitaj odgovor od arduina
            time.sleep(1.0)
            if ReadData.strip():#U slučaju da odgovor nije prazan očitaj odgovor
                print(ReadData)
                break
                pass
            print("[INFO] Nastavak rada ...")
        #očisti deck ,UspjesnaIdetifikacija i KorisnickoIme
        DeckLica.clear()
        UspjesnaIdetifikacija=False
        KorisnickoIme=""

    #slučaj u kojem smo u 50 frameova zaredom imali samo "Nepoznate korisnike"
    #Spašavamo idućih 10 sekundi snimka u obliku video filma
    #Čije ime sadrzi datum i vrijeme snimka
    if(NepoznatiKorisnik):
        #broj Frame prestavlja broj spašenih frame-ova unutar video snimka
        #kada postigne određen broj snimanje se zaustavlja
        FrameId=0
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y %H-%M-%S")#dohvati datum i vrijeme 
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G') 
        VideoNaziv="NepoznateOsobe\VideoSnimak {}.avi".format(date_time) #kreiraj video zapisivač
        video_writer = cv2.VideoWriter(VideoNaziv, fourcc, 25, (int(w),int(h)))
         #snimanje video snimka sadrži 30 FPS-a u sekundi tako da 300 frame-ova jeste 10 sekundi
        while FrameId<300:
            status,snimak=vs.read()
            video_writer.write(snimak)
            FrameId=FrameId+1#povećaj FrameId brojač
            cv2.waitKey(1)#prikaži snimak
            cv2.imshow("Nepoznata osoba snimak", snimak)
            pass
        #Oslobodi resurse snimanja video snimka i nastavi sa radom
        cv2.destroyWindow("Nepoznata osoba snimak")
        video_writer.release()
        NepoznatiKorisnik=False
        DeckLica.clear()
        pass

    # generiši blob
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # prepoznaj lica u blobu
    detector.setInput(imageBlob)
    detections = detector.forward()


    for i in range(0, detections.shape[2]):
        # dohvati vjerovatnoću da je lice
        vjerovatnoca = detections[0, 0, i, 2]

        # ukloni nisku vjerovatnoću
        if vjerovatnoca > 0.7:
            # dohvati lice
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            #provjeri da li je lice dovoljno veliko
            if fW < 20 or fH < 20:
                continue
            #pretvori lice u blob
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            #Klasificiraj blob lica tj. odredi ko je
            preds = recognizer.predict_proba(vec)[0]
            #Dohvati index lica sa najvećom vjerovatnoćom
            j = np.argmax(preds)
            proba = preds[j] #dohvati procent sigurnosti
            ime = le.classes_[j]#koristeći index lica dohvati ime na tom indexu unutar labela pickle-a

            #Označi okvir oko 
            if(proba < 0.63):
                ime = "Nepoznata Osoba"
                BojaOkvira=tuple(Crvena)
                
            if(proba > 0.65 and proba < 0.71):
                ime = "Vjerovatno "+ime
                BojaOkvira=tuple(Plave)

            if(proba>0.72):
                BojaOkvira=tuple(Zelena)

            print("Broj imena je {}".format(len(DeckLica)))
            # Uokviri lice i napiši vjerovatnoću da je to korisnik
            text = "{}: {:.2f}%".format(ime, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          BojaOkvira, 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, BojaOkvira)
            #Prikaži procent Autorizacije korisnika u zadnjim frame-ovima
            cv2.putText(frame,"Autorizacija: {}%".format(ProcenatAutorizacije(ime)),(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.45,color=Zelena )

            DeckLica.append(ime)
            print(DeckLica) 
            rezultat=BrojJednakihAutorizacija(ime)

            if(rezultat==-1):
                NepoznatiKorisnik=True
                pass

            if(rezultat==1):
                KorisnickoIme=ime
                UspjesnaIdetifikacija=True
                pass

    #Prikaži video snimak sa uokvirenim licem i procentom identifikacije 
    cv2.imshow("Kamera snimak", frame)
    key = cv2.waitKey(1) & 0xFF
    # Obustavi rad na klik dugmeta "q"
    if key == ord("q"):
        break

#Oslobodi resurse 
cv2.destroyAllWindows()
vs.release()


