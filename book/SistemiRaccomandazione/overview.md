# Sistemi di Raccomandazione

Il 2 ottobre 2006 Netflix (che allora campava spedendo DVD per posta) pubblica
un annuncio senza precedenti: un milione di dollari a chiunque riesca a
migliorare del 10% il suo sistema di raccomandazione, Cinematch. Del 10% *che
cosa*, conviene dirlo subito, perché la gara si gioca tutta lì: dell'errore con
cui il sistema prevede le stelle che un utente darà a un film. Il metro è il
**RMSE**, la radice dell'errore quadratico medio incontrata nel capitolo di
Machine Learning, sezione *Metriche*: si legge in stelle, e dice di quanto la
previsione sbaglia in media. Cinematch stava a $0{,}9525$ stelle; l'assegno
andava a chi scendeva sotto $0{,}8572$. Per
partecipare basta scaricare un dataset che all'epoca sembra sterminato: poco
più di 100 milioni di voti, da una a cinque stelle, dati da circa 480.000
utenti anonimi a 17.770 film. La gara diventa un caso mondiale: migliaia di
squadre, forum incandescenti, ricercatori universitari e ingegneri che di
notte inseguono decimali. E dura molto più del previsto: quasi tre anni. Solo
il 21 settembre 2009 Netflix consegna l'assegno al team **BellKor's Pragmatic
Chaos**, che ha superato Cinematch del 10,06%: battendo sul filo di lana i
rivali di *The Ensemble*, fermi su un punteggio equivalente ma con una
consegna arrivata venti minuti più tardi.

L'ironia arriva dopo, e vale come lezione per tutto il capitolo. Quella
soluzione da un milione di dollari **non fu mai adottata per intero**: era un
mosaico di oltre cento modelli combinati, troppo costoso da portare in
produzione (cioè da far girare davvero, tutti i giorni, per i clienti veri) per
il guadagno che prometteva; e nel frattempo il business stava
migrando dai DVD allo streaming, dove prevedere il voto in stelle conta meno
di prevedere che cosa guarderai stasera. In produzione finirono però due
ingredienti emersi durante la gara, e uno dei due, la **fattorizzazione di
matrici** {cite}`koren2009matrix`, è il protagonista di questo capitolo.
Fattorizzare vuol dire scomporre in fattori, come si fa da sempre con i numeri
($12 = 3 \times 4$): qui si scompone una tabella, e i fattori sono due tabelle
strette al posto di una larghissima.

Una parola sul nome, prima di partire, perché in italiano «raccomandazione»
significa due cose e una delle due è la spintarella. Qui vale l'altra, quella
del consiglio: raccomandare è consigliare, e un sistema di raccomandazione è
una macchina che consiglia. Il senso brutto però non è del tutto fuori luogo, e
il capitolo se ne accorge alla fine, quando la domanda diventa se una macchina
che decide cosa vedi ti stia servendo o spingendo. Conviene leggere tutto il
resto con quella domanda in mano.

## Non «qual è il film più bello», ma «quale piacerà a te»

Un motore di ricerca risponde a una domanda che fai tu. Un sistema di
raccomandazione risponde a una domanda che non hai fatto: *tra queste
centomila cose, quali vale la pena mostrarti?* Quelle cose, nel gergo del
settore, si chiamano **oggetti** (in inglese *item*): film, canzoni, prodotti,
articoli, video, a seconda del catalogo; qui useremo spesso «film» perché
l'esempio è quello, ma il discorso non cambia. La differenza cruciale è che
non esiste una risposta valida per tutti.

`````{tab} Elementare

Chiedere «qual è il film più bello?» è come chiedere «qual è il piatto più
buono?»: una classifica unica (la media dei voti di tutti) accontenta la
maggioranza e non entusiasma nessuno. Un buon libraio non ti indica il libro
più venduto: ti guarda, ricorda cosa hai comprato l'ultima volta, e ti mette
in mano un titolo che *a te* probabilmente piacerà. Un sistema di
raccomandazione prova a fare il libraio su scala industriale: milioni di
clienti, milioni di scaffali, un consiglio diverso per ciascuno.

`````

`````{tab} Superiore

Formalmente, dati un insieme di utenti $\mathcal{U}$ e un insieme di oggetti
(*item*) $\mathcal{I}$ (film, prodotti, brani, articoli) un sistema di
raccomandazione stima una funzione di utilità
$f:\mathcal{U}\times\mathcal{I} \to \mathbb{R}$ che assegna a ogni coppia
(utente, oggetto) un punteggio di affinità, e per ogni utente restituisce gli
oggetti con punteggio massimo. Non è una classifica globale ma una famiglia di
classifiche **personalizzate**: la stessa $f$, valutata su utenti diversi,
produce ordinamenti diversi. Il problema di apprendimento consiste nello
stimare $f$ dalle interazioni passate, che coprono una frazione minuscola di
$\mathcal{U}\times\mathcal{I}$.

`````

## Il carburante: feedback esplicito e implicito

Da dove impara, il libraio automatico? Da due tipi di segnale molto diversi.
Entrambi si chiamano, quando si parla di tutti e due insieme, **interazioni**:
ogni volta che una persona incontra un oggetto e lascia una traccia (un voto,
un click, un acquisto, dieci minuti di visione, un brano saltato), quella
traccia è un'interazione, ed è tutto ciò che il sistema ha in mano.

`````{tab} Elementare

Il **feedback esplicito** è quando dichiari il tuo giudizio: le cinque stelle
su un film, il pollice in su, la recensione. È chiaro ma raro: quasi nessuno
vota. Pensa a te stesso: quante cose hai guardato o comprato quest'anno, e
quante ne hai *recensite*?

Il **feedback implicito** è tutto ciò che fai senza pensare di stare
giudicando: i click, gli acquisti, i minuti di visione, i brani saltati dopo
dieci secondi. È abbondante (ogni gesto ne produce) e per certi versi più
sincero delle dichiarazioni: puoi *dire* che ami i documentari, ma la
cronologia rivela le serie poliziesche. Ha però un difetto: è ambiguo. Un
click non è una promozione (magari il film ti ha deluso), e un film ignorato
non è una bocciatura: forse non l'hai mai visto passare. Per abbondanza,
l'implicito domina i sistemi reali; per ambiguità, richiede più cautela.

`````

`````{tab} Superiore

Il feedback **esplicito** produce una matrice di voti $\mathbf{R}$ con entrate
$r_{ui}$ su scala ordinale (ad esempio $1$–$5$): segnale ad alta qualità ma
estremamente scarso, e per di più **non mancante a caso**
{cite}`marlin2009collaborative`; gli utenti votano soprattutto ciò che hanno
scelto di consumare, quindi le celle osservate sono un campione distorto. Il
feedback **implicito** produce eventi unari o conteggi (click, acquisti, tempo
di visione): copertura enormemente maggiore, ma niente segnale negativo
esplicito. L'assenza di interazione confonde due casi indistinguibili («non gli
piace» e «non l'ha mai visto») e questo cambia la formulazione del problema: non
più regressione sul voto, ma *ranking* da osservazioni positive e
non-osservazioni, come vedremo con BPR {cite}`rendle2009bpr`. Nei sistemi
industriali l'implicito domina per volume (ordini di grandezza di differenza) e
perché misura il comportamento effettivo, non quello dichiarato.

`````

## Un problema di machine learning anomalo

Visto da lontano sembra il solito apprendimento supervisionato: dati storici
in ingresso, una predizione in uscita. Visto da vicino, tre cose lo rendono
un animale a parte.

La prima è la **sparsità**. La materia prima è una tabella con un utente per
riga e un film per colonna, e i voti nelle celle, come in
{numref}`fig-matrice-utenti-film`. Il punto è quante celle sono vuote: nel
dataset del Netflix Prize i 100 milioni di voti sembrano tanti, ma la tabella
completa avrebbe $480.000 \times 17.770 \approx 8{,}5$ miliardi di celle, ne
era piena appena l'1,2%. Nei cataloghi industriali di oggi, con milioni di
oggetti, si supera facilmente il 99,9% di vuoto. Raccomandare significa
riempire quei buchi in modo sensato.

```{figure} ../figures/matrice-utenti-film.svg
:name: fig-matrice-utenti-film
:alt: Griglia di sei utenti per otto film in cui poche celle contengono un voto da uno a cinque e tutte le altre un punto interrogativo; una cella evidenziata in terracotta indica il voto da prevedere.
:width: 90%

La matrice utenti × film: poche celle di cui conosciamo il voto (le cornici
distinguono i voti alti, 4 e 5, dai bassi), un oceano di punti interrogativi.
Prevedere il valore di una cella vuota (qui, il voto di Anna al film D) è
l'intero problema.
```

La seconda anomalia è che **non esiste una «risposta giusta» osservabile**.
Un classificatore di cifre può essere confrontato con l'etichetta vera; qui
la domanda riguarda un fatto che non è avvenuto: *se* ti avessimo mostrato
quel film, ti sarebbe piaciuto? (una domanda del genere si dice
**controfattuale**). Per la stragrande maggioranza delle coppie utente–film
questa risposta non verrà mai osservata, e la valutazione deve arrangiarsi con
approssimazioni: nascondere una parte delle interazioni note e verificare se
il modello le recupera. *Quale* parte si nasconde non è un dettaglio di
procedura: è la decisione che sposta i risultati più di quasi ogni scelta di
modello, e la riprenderemo quando parleremo di come si misura una classifica.

La terza è la più insidiosa: **il sistema influenza i dati che raccoglie**.

`````{tab} Elementare

Immagina un cameriere che consiglia sempre gli stessi tre piatti. Dopo un
mese, i piatti più ordinati del ristorante saranno... quei tre. Se il
ristoratore guardasse le ordinazioni per capire cosa piace ai clienti,
concluderebbe che i tre piatti sono i favoriti, ma è una profezia che si
autoavvera: i clienti hanno scelto nel menù che il cameriere ha proposto. I
sistemi di raccomandazione vivono in questo cerchio: ciò che mostri determina
ciò che viene cliccato, e ciò che viene cliccato determina ciò che mostrerai.

`````

`````{tab} Superiore

I dati di addestramento non sono campionati dalla distribuzione «vera» delle
preferenze, ma filtrati dalla **politica di esposizione** del sistema stesso:
osserviamo interazioni solo sugli oggetti che il modello precedente ha deciso
di mostrare. È un *feedback loop*: il modello al tempo $t$ genera i dati con
cui si addestra il modello al tempo $t+1$, e i bias si amplificano invece di
mediarsi. È un caso particolarmente severo del *dataset shift* che abbiamo
incontrato nella sezione *Quando i dati cambiano* del capitolo di Machine
Learning {cite}`quinonero2009dataset`, con l'aggravante che qui lo shift non è
un incidente esterno, ma è prodotto dal sistema stesso. Le contromisure
(esplorazione controllata, correzioni per propensità) esistono, ma nessuna è
gratis: esplorare significa mostrare a qualche utente qualcosa che il modello
non avrebbe scelto.

`````

## Come è organizzato il capitolo

Due sezioni, dall'idea classica a quella neurale.

La prima muove da un'idea che usiamo tutti i giorni senza chiamarla così:
**chiedere all'amico giusto**. Vedremo come si rende calcolabile (il metodo si
chiama *filtraggio collaborativo*), perché la versione ingenua si arena sul
vuoto della tabella, e come se ne esce riassumendo persone e film in poche
schede di numeri: è l'idea che ha vinto il Netflix Prize, e la scriveremo in
PyTorch in una ventina di righe.

La seconda porta le reti neurali dentro il problema, e la risposta non è quella
che ci si aspetta. Proveremo a sostituire il confronto fra le schede con una
rete (e vedremo che non conviene); ridisegneremo la tabella come un disegno di
pallini e linee, dove consigliare vuol dire indovinare le linee che mancano;
cambieremo obiettivo, perché quando non ci sono voti non si prevede un numero,
si mette in ordine una vetrina, e allora serve anche un altro modo di misurare
se la vetrina è buona. Chiuderemo con il funzionamento vero della macchina che
ti consiglia i video, e con la domanda che le sta sotto: quando un consiglio
smette di essere un consiglio.

```{admonition} Da ricordare
:class: important
- Consigliare non è classificare: **non esiste una risposta valida per tutti**,
  e lo stesso sistema deve produrre una classifica diversa per ogni persona.
- Il carburante sono le **interazioni**: o dichiarate (le stelle, il pollice in
  su: chiare ma rarissime) o lasciate senza pensarci (click, acquisti, minuti
  di visione: abbondanti ma ambigue, perché un titolo ignorato non è una
  bocciatura). Nei sistemi veri dominano le seconde.
- Il dato di partenza è una **tabella quasi tutta vuota**: nel Netflix Prize
  era piena all'1,2%, nei cataloghi di oggi si va sotto lo 0,1%. Consigliare
  vuol dire riempire quei buchi in modo sensato.
- **Non c'è una risposta giusta da guardare**: nessuno saprà mai se ti sarebbe
  piaciuto il film che non hai visto, e per valutare si nasconde una parte di
  ciò che si sa, per poi vedere se il modello la ritrova.
- **Il sistema si fabbrica da solo i dati con cui impara**: mostra, e ciò che
  mostra è ciò che verrà cliccato. È il cameriere che consiglia sempre gli
  stessi tre piatti, ed è il problema che questo capitolo si porta dietro fino
  all'ultima riga.
```
