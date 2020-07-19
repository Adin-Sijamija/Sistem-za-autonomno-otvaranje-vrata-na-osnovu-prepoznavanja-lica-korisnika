from imutils import paths
import numpy as np 
import argparse
import imutils
import cv2
import pickle
import os
import copy

# kreiranje argumenata
ap = argparse.ArgumentParser()
ap.add_argument("-P", "--Podaci", required=True,
	help="put to baze slika")
ap.add_argument("-E", "--Embeding", required=True,
	help="put do mjesta spašavanja embeddinga")
ap.add_argument("-D", "--Detektor", required=True,
	help="put do OpenCV's deep learning face detektora")
ap.add_argument("-M", "--Embeding-Model", required=True,
	help="put do OpenCV's deep learning face embedding modela")
ap.add_argument("-V", "--Vjerovatnoca", type=float, default=0.5,
	help="minimalna vjerovatnoća da imamo kompletno lice")
ap.add_argument("-S","--Spasi-Slike",type=int,default=0,required=False,
help="Da li želimo spasiti slike lica kako bi provjerili šta naš model od slike enkodira")
args = vars(ap.parse_args())

# očitavanje detektora
protoPath = os.path.sep.join([args["Detektor"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["Detektor"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# očitavanje našeg embeded face modela
embedder = cv2.dnn.readNetFromTorch(args["Embeding_Model"])


# kreacije niza koji će držati vrijednosti naših enkodiranja 
# kao i niz koji će čuvati imena korisnika
knownEmbeddings = []
knownNames = []
total = 0

# Dohvati sve naše slike korisnika
imagePaths = list(paths.list_images(args["Podaci"]))

# procesirajs svaku sliku
for (i, imagePath) in enumerate(imagePaths):
	ImeKorisnika = imagePath.split(os.path.sep)[-2]
	NazivSlike = imagePath.split(os.path.sep)[-1]
	# dohvati ime korisnika iz linka do slike
	print("[INFO] Procesiranje slike korisnika {} slika {} -- {} od {}".format(ImeKorisnika,NazivSlike,i + 1,
		len(imagePaths)))
	

	# očitaj sliku i namjesti rezoluciju na visinu od 600 pixela
	# I dohvati širini i visinu slike
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]


	if(args["Spasi_Slike"]==1):
		SlikaKopija=copy.copy(image)
	# generiši Blob od slike 
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# koristi open cv deep learning face detector
	# da pronađeš sva lica u slici 
	detector.setInput(imageBlob)
	detections = detector.forward()

	
	if len(detections) > 0:
		#dohvatanje vrijednosti iz numpy niza
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# provjeri da je pronađeno lice ima veliku šansu tačnosti da je to ustvari pravo lice 
		if confidence > args["Vjerovatnoca"]:
			# pronađi visinu i širinu lica
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# izvedi lice iz ostatka slike 
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# provjeri da je lice dovoljno veliko za embedding 
			if fW < 20 or fH < 20:
				continue
			#Ako spašavamo sliku označi gdje smo našli lice
			if(args["Spasi_Slike"]==1):
				cv2.rectangle(SlikaKopija, (startX, startY), (endX,endY), (255, 0, 0), 2)

			#generiši blob lica i pošalji u face embedding model da bi dobili 128-d tačaka lica
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			#dodaj ime osobe kao i embedding lica u listu podataka za pickle
			knownNames.append(ImeKorisnika)
			knownEmbeddings.append(vec.flatten()) #izravnati vrijednosti od namjmanej ka največoj
			total += 1
	#Ako spašavamo lice snimi sliku
	if(args["Spasi_Slike"]==1):
		cv2.imwrite("encoding_rezultati/PROCESIRANA-SLIKA-{}.jpg".format(NazivSlike),SlikaKopija)

# spasi podatke na disk
print("[INFO] spašavanje {} enkodiranja...".format(total))
data = {"encodings": knownEmbeddings, "names": knownNames}
f = open(args["Embeding"], "wb") #otvori steam za pisanje
f.write(pickle.dumps(data)) #kreiraj pickle svih lica i imena
f.close()#zatvori stream
print("enkodiranja uspješno spašena")