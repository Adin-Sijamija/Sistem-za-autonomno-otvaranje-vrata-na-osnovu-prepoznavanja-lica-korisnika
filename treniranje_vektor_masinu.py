from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


print("[INFO] očitavanje podataka ...")
#očitaj naš embedding lica 
data = pickle.loads(open("output/encodings.pickle", "rb").read())

#očitaj naš embedding imena 
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[INFO] treniranje modela ...")
#Kreacija objekta koji drži vektor mašinu	
recognizer = SVC(C=1.0, kernel="linear", probability=True)
#treniranje modela na našim podacima 
recognizer.fit(data["encodings"], labels) #treniranje modela

#spašavanje recognizer pickla koji drži model prepoznavanja
f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()
#spašavanje imena korisnika
f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
print("[INFO] treniranje modela uspješno završeno ")
