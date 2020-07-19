import serial
import time


arduino = serial.Serial('COM3', 9600)
time.sleep(5)
KorisnickoIme="Adin"

Poruka = KorisnickoIme+">"
arduino.write(str.encode(Poruka))
print("[INFO] Komanda poslana")
time.sleep(1.0)  # čekaj odgovor arduina
while True:
    print("[INFO] Čekanje signala zaključavanja...")
    ReadData = arduino.readline().decode("ascii")  # očitaj odgovor od arduina
    time.sleep(1.0)
    if ReadData.strip():  # U slučaju da odgovor nije prazan očitaj odgovor
        print(ReadData)
        break
        pass
    print("[INFO] Nastavak rada ...")
    # očisti deck ,UspjesnaIdetifikacija i KorisnickoIme
