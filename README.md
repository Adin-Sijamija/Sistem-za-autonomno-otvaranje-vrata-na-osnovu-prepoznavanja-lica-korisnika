# Sistem-za-autonomno-otvaranje-vrata-na-osnovu-prepoznavanja-lica-korisnika
[AI, deep learning, machine learning, facial recognition, arduino]


[Bosnian]
Završni rad nakon prvog cisklusa studija na fakultetu FIT Mostar.
Obuhvata polja umjetne inteligencije i računarskog vida u cilju prepoznavanja lica korisnika koji sačinjavaju bazu podataka korisnika

Korišten python programski jezik.

Navedena sistem se dijeli na računarsku aplikaciju i na arduino komponente instalirane na model vrata.

Unutar navedene računarske aplikacije provodimo sljedeće procese:
1. Dodavanje slika korisnika i našu bazu podataka
2. Ekstraktovanje lica korisnika iz slika koje smo spasili u bazi podataka --- U tom procesu ih također ispravljamo lice na jednu horizontalnu ravnu tj. da su oči na istoj razini kao i odbacivanje neadekvatnih slika ( bez lica/2 ili više lica )
3. Enkodiranje lica korisnika-- pretvaranje lica korisnika u 128 numericki niz podataka.
4.Treniranje vektor mašine- korištenje prije dobivenih enkodiranka kako bi trenirali novu umjetnu inteligenciju koja može porediti nova enkodiranjima sa spašenim enkodiranjim naših korisnika i izraziti koliko je AI suguran da nova enkidiranja pripadaju jednom od naših korisnika
5. Prepoznavanje korisnika u video snimku- Koristimo umjetnu inteligenciju kako bi prepoznali lice u video snimku. Zatim izvodimo okvir lica koje enkodiramo i šaljemo našoj umjetnoj inteligenciji prepoznavanja. Svako prepoznavanje držimo u deque listi koja spašava rezultate zadnih 50 frameova. Ako smo sigurni da je u pitanju lice korisnika otvarama vrata slanjem signala arduinu ili ako mislimo da je nepoznata osoba počinjemo proces snimanja video snimka za kasniji pregled.

Više informacija o radu sistema možete naći unutar word dokumentacije.

[English] Todo...
