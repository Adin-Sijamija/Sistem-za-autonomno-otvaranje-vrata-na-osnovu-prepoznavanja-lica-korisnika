import cv2 #openCV python biblioteka
import time #biblioteka za zaustavljanje rada programa određeno vrijeme
import os #biblioteka za 
import pathlib #putokazna biblioteka
import argparse #argument biblioteka
import numpy #biblioteka dvodienzionalnih nizova
import string #string biblioteka
import copy #biblioteka za kopiranje
import sys #biblioteka za rad sa operatinim sistemom

#Kreiranje argumenta za ime korisnika i foldera
arguments=argparse.ArgumentParser()
arguments.add_argument("-n","--Ime",required=True,help="Ime korisnika za kojeg spašavamo slike.")
arguments.add_argument("-c","--Kamera",required=False,type=int,default=0,
help="Broj kamere koja će biti korištena 0 (interna) 1(eksterna)")
arguments.add_argument("-b","--Broj",required=False,type=int,default=10,help="Broj slika koji ćemo uzeti za korisnika")
args=vars(arguments.parse_args())

print("[INFO] pokretanje programa")
print("[INFO] očitavanje foldera...")

#Kreiranje direktorija za korisnikove slike
dirName=args["Ime"] #Ime korisnika dobijeno iz argparser-a
PathToImages=pathlib.Path(__file__).parent.absolute().joinpath("podaci").joinpath("slike")
print("Put do slika je: {0}".format(PathToImages))
NewDireLink=PathToImages.joinpath(dirName)
print("Kreiranje foldera {0}".format(NewDireLink))

try:
    os.mkdir(NewDireLink)
    print("Direktori {0} uspješno kreiran".format(dirName))
except FileExistsError as identifier:
    print("[GREŠKA]Folder sa datim imenom već postoji.") 
    #u slučaju postojanja korisnika sa tim imenom aplikacija se prekida
    print("[GREŠKA]Odaberite drugo ime za folder!")
    pass

#aktivacija kamere
print("[INFO] pokretanje kamere...")
webcam = cv2.VideoCapture(args["Kamera"])
time.sleep(2)
print("[INFO] kamera aktivirana!")

#Očitaj kaskadu za prepoznavanje lica
KaskadaLica=cv2.CascadeClassifier("Haar Cascades/haarcascade_frontalface_default.xml")

#"id" prestavlja redni broj slike
#"Max" predstavlja maximum broja slika
Max=args["Broj"]+1
id=1
while id<Max:
    try:
        check, frame = webcam.read()
       # Check provjerava je li stream aktivan, frame je jedan frame iz video snimka kamere
        SlikaKopija=copy.copy(frame) #Kkopiranje čistog frame za kopiranje
        #Kopija frama bez boje  za operaciju har kaskade
        SlikaBezBoje=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        Lica=KaskadaLica.detectMultiScale(image=SlikaBezBoje, scaleFactor=1.1, minNeighbors=4)
        for (x,y,w,h) in Lica:
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0),thickness= 2)

        cv2.putText(frame, "Slika {0} od {1}".format(id,Max),(20,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        cv2.imshow("Klikni slovo S kako bi spasio sliku", frame)

        key = cv2.waitKey(1)
        if(key==ord("s")):#klikom na dugme "s" spašava se trenutna slika koja prestavlja video
            print("Spašavanje slike broj:{0}".format(id))
            print("Spašena slika:{0}{1}.jpg".format(dirName,id))
            ImagePath=NewDireLink.as_posix()
            ImagePath=ImagePath+"/"+dirName+str(id)+".jpg"
            cv2.imwrite(filename=ImagePath,img=SlikaKopija)
            id=id+1 #snimanje se ponavlja dok korisnik nema željeni broj slika

    except(KeyboardInterrupt):#oslobodi uređaje u slučaju prekida
        break
            
print("[INFO] Gašenje kamere.")
webcam.release()
print("[INFO] Kamera ugašena.")
print("[INFO] Program završio radnju.")
cv2.destroyAllWindows()

