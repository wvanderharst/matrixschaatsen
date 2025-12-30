README 

Mijn model is een reconstructie van het model dat gebruikt wordt door de KNSB. Zover ik weet is dit model niet volledig bekend, dus heb ik mij laten leiden door de uitingen in https://pure.rug.nl/ws/files/170958955/81438422_6497325_ORMS_April_2021_Speed_Skating.pdf
Advies is om eerst deze publicatie te lezen.

Dit verstaat het normaliseren door AV van de top 5 te gebruiken.
Dit verstaat het gebruiken van een lognormal distributie.
Dit verstaat het gebruiken van internationale wedstrijden van het afgelopen jaar.

Opmerkingen 
Je zou de verdeling ook nog tijdafhankelijk kunnen maken, echter is daar erg weinig data voor.
Je zou de zwaarte van observaties meer kunnen laten meetellen voor recentere tijden, lijkt mij een goede verbetering, echter niet gedaan.

Voor sommige afstanden heb ik besloten het OKT wel mee te nemen. Zodat deze update wel in de data zit, eigenlijk is dit foutief omdat de benchmark hier verandert. Echter anders is voor sommige mensen, zoals Femke kok of Stijn geen enkel datapunt op de 1500 of 5000/10000. Echter kan je de code ook runnen zonder deze data. Het verscheelt niet veel

Rijders die te weinig datapunten hebben voor een goede schatting geef ik een gemmidelde standaard deviatie. 
Ik heb geen onderzoek gedaan naar correlatie tussen opvolgende tijden. Dit is dus geen onderdeel

Omdat een lognormal een onder vereisten heeft van 0 maar door de differencing methode er schaatsters zijn met - getallen wordt een transformatie gedaan zodat alle getallen positief zijn. omdat 0 ook out of bounds is heb ik vervolgens nog 0.1 toegevoegd.

Het model geeft vergelijkbare maar niet exact dezelfde matrix als de echte matrix. Aantal simulaties is niet hoog genoeg om te zeggen dat convergance is bereikt, heb ik verder ook niet getest.

Massstart is gek omdat het natuurlijk niet op tijd is, echter heb ik hier geen uitzondering gemaakt. Neem met een korrel zout.

Aanmerkingen

Het standariseren tegen de wereldtop maakt dat alleen internationale wedstrijden meegenomen kunnen worden. Door bijvoorbeeld baanrecord te gebruiken zou je dit anders kunnen doen. Mogelijk met nog een adjustment of de wereldtop er rijdt.

Een schaatser in vorm op 1 afstand heeft geen impact op een andere afstand. Dit zou een aanmerkelijke verbetering zijn.

Code is niet nogmaals nagekeken, nog data nagekeken aangezien dit het werk van 1 dag was.

Ik heb meerdere selectiemethode, waaronder huidig, puur statisitisch,  okt met aanwijsplekken vooraf, en huidig maar we sturen de beste rijder niet(om effect te meten). 

 
