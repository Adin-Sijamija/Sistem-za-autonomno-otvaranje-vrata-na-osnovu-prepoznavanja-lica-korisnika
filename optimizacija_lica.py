from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os
import pathlib
import time

#pronalazak foldera slika
SlikeFolder=str(pathlib.Path(__file__).parent.absolute().joinpath("podaci").joinpath("slike"))
FajlStruktura=tuple(os.walk(SlikeFolder))
Folderi=FajlStruktura[0][1] # dohvatanje imena svih foldera


# inicializacija dlib detectora
# i facealigner objekta
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredLeftEye=(0.6,0.6) ,desiredFaceWidth=256)
#petlja za svaki folder
for folder in Folderi:
	print("Procesiranje foldera {0}".format(folder))
	FolderLokacija=str(pathlib.Path(__file__).parent.absolute().joinpath("podaci").joinpath("slike").joinpath(folder))
	BrojSlika=len(os.listdir(FolderLokacija))#provjeriti ako korisnik ima slika 
	
	if BrojSlika>0:
		print("Kreiranje korisničkog foldera")
		KorisnikFolderLokacija=str(pathlib.Path(__file__).parent.absolute().joinpath("podaci").joinpath("korisnici").joinpath(folder))
		os.mkdir(KorisnikFolderLokacija) #kreirati  folder za procesirane slike
		print("Korisnički folder korisnika {0} kreiran".format(folder))
	#petlja za svaku sliku unutar foldera
	for slike in os.listdir(FolderLokacija):
		print("Procesiranje slike: {0}".format(slike))
		PutDoSlike=str(pathlib.Path(__file__).parent.absolute().joinpath("podaci").joinpath("slike").joinpath(folder).joinpath(slike))

		image = cv2.imread(PutDoSlike)#očitavanje slike
		image = imutils.resize(image, width=800)#formatiranje slika da sve imaju i ste dimenzije
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#siva slika za detekciju

		#preporznavanje lica i dobivanje praugaonika lica na slici
		rects = detector(gray, 2) 
		if len(rects)==0: #u slučaju da na slici nema lica 
			print("[UPOZORENJE Slika {0} nema vidljivih lica, slika se preskaće".format(slike))
			height, width ,channels=image.shape #dohvatanje visine i širine slike 
			#obavijestiti korisnika o kojoj slici se tiče
			cv2.putText(image,"[Greska] Nema vidljivih lica",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
			#crtanje "x" simbola preko cijele slike
			cv2.line(image,(0,0),(width,height),(0,0,255))
			cv2.line(image,(width,0),(0,height),(0,0,255))
			#prikaz pogrešne slike
			cv2.imshow("Greška",image)
			cv2.waitKey(30)
			#čekanje da korisnik primjeti sliku
			time.sleep(1.2)
			cv2.destroyAllWindows()# uništavanje prozora slike i nastavak rada petlje
			continue

		if len(rects)>1:#u slučaju da ima više lica 
			print("[UPOZORENJE Slika {0} ima vise od 1 lica, slika se preskaće".format(slike))
			height, width ,channels=image.shape #dohvatanje visine i širine slike
			#obavijestiti korisnika o kojoj slici se tiče
			cv2.putText(image,"[Greska]Slika ima vise od jednog lica",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
			#crtanje "x" simbola preko cijele slike
			cv2.line(image,(0,0),(width,height),(0,0,255))
			cv2.line(image,(width,0),(0,height),(0,0,255))
			#prikaz pogrešne slike
			cv2.imshow("Greška",image)
			cv2.waitKey(30)
			#čekanje da korisnik primjeti sliku
			time.sleep(1.2)
			cv2.destroyAllWindows()# uništavanje prozora slike i nastavak rada petlje
			continue
		#ako slika ima samo jedno lice prikazat je
		cv2.imshow("Orginal slika", image)
	
		for rect in rects:	
			#pretvaranje pravougaonika lica u visini širinu i početne tačke x,y
			(x, y, w, h) = rect_to_bb(rect)
			#smanjivanje pronađenog lica na 256 piksela širine
			faceOrig = imutils.resize(image[y:y + h, x:x + w],width=256)
			faceAligned = fa.align(image, gray, rect) #rotiranje lica

			LokacijaSlike=str(pathlib.Path(__file__).parent.absolute().joinpath("podaci").joinpath("korisnici").joinpath(folder).joinpath(slike))

			height,witdh,channell = faceAligned.shape
			CentarSlike=(w/2,h/2)
			#preokret lica za 180 stepeni
			Rotirana=imutils.rotate(faceAligned,180)
			cv2.imwrite(LokacijaSlike, Rotirana)
			cv2.imshow("Lice Neispravljeno", faceOrig)
			cv2.imshow("Lice Ispravljeno", Rotirana)
			cv2.waitKey(30)

			time.sleep(1.2)
			cv2.destroyAllWindows()
			pass

print("[INFO] Proces uspješno završen!")