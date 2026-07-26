# Sistemi di Raccomandazione

Il 2 ottobre 2006 Netflix — che allora campava spedendo DVD per posta — pubblica
un annuncio senza precedenti: un milione di dollari a chiunque riesca a
migliorare del 10% il suo sistema di raccomandazione, Cinematch. Per
partecipare basta scaricare un dataset che all'epoca sembra sterminato: poco
più di 100 milioni di voti, da una a cinque stelle, dati da circa 480.000
utenti anonimi a 17.770 film. La gara diventa un caso mondiale: migliaia di
squadre, forum incandescenti, ricercatori universitari e ingegneri che di
notte inseguono decimali. E dura molto più del previsto: quasi tre anni. Solo
il 21 settembre 2009 Netflix consegna l'assegno al team **BellKor's Pragmatic
Chaos**, che ha superato Cinematch del 10,06% — battendo sul filo di lana i
rivali di *The Ensemble*, fermi su un punteggio equivalente ma con una
consegna arrivata venti minuti più tardi.

L'ironia arriva dopo, e vale come lezione per tutto il capitolo. Quella
soluzione da un milione di dollari **non fu mai adottata per intero**: era un
mosaico di oltre cento modelli combinati, troppo costoso da portare in
produzione per il guadagno che prometteva; e nel frattempo il business stava
migrando dai DVD allo streaming, dove prevedere il voto in stelle conta meno
di prevedere che cosa guarderai stasera. In produzione finirono però due
ingredienti emersi durante la gara — e uno dei due, la **fattorizzazione di
matrici** {cite}`koren2009matrix`, è il protagonista di questo capitolo.

## Non «qual è il film più bello», ma «quale piacerà a te»

Un motore di ricerca risponde a una domanda che fai tu. Un sistema di
raccomandazione risponde a una domanda che non hai fatto: *tra queste
centomila cose, quali vale la pena mostrarti?* La differenza cruciale è che
non esiste una risposta valida per tutti.

`````{tab} Elementare

Chiedere «qual è il film più bello?» è come chiedere «qual è il piatto più
buono?»: una classifica unica — la media dei voti di tutti — accontenta la
maggioranza e non entusiasma nessuno. Un buon libraio non ti indica il libro
più venduto: ti guarda, ricorda cosa hai comprato l'ultima volta, e ti mette
in mano un titolo che *a te* probabilmente piacerà. Un sistema di
raccomandazione prova a fare il libraio su scala industriale: milioni di
clienti, milioni di scaffali, un consiglio diverso per ciascuno.

`````

`````{tab} Superiore

Formalmente, dati un insieme di utenti $\mathcal{U}$ e un insieme di oggetti
(*item*) $\mathcal{I}$ — film, prodotti, brani, articoli — un sistema di
raccomandazione stima una funzione di utilità $f:\mathcal{U}\times\mathcal{I}
\to \mathbb{R}$ che assegna a ogni coppia (utente, oggetto) un punteggio di
affinità, e per ogni utente restituisce gli oggetti con punteggio massimo. Non
è una classifica globale ma una famiglia di classifiche **personalizzate**:
la stessa $f$, valutata su utenti diversi, produce ordinamenti diversi. Il
problema di apprendimento consiste nello stimare $f$ dalle interazioni
passate, che coprono una frazione minuscola di $\mathcal{U}\times\mathcal{I}$.

`````

## Il carburante: feedback esplicito e implicito

Da dove impara, il libraio automatico? Da due tipi di segnale molto diversi.

`````{tab} Elementare

Il **feedback esplicito** è quando dichiari il tuo giudizio: le cinque stelle
su un film, il pollice in su, la recensione. È chiaro ma raro: quasi nessuno
vota. Pensa a te stesso: quante cose hai guardato o comprato quest'anno, e
quante ne hai *recensite*?

Il **feedback implicito** è tutto ciò che fai senza pensare di stare
giudicando: i click, gli acquisti, i minuti di visione, i brani saltati dopo
dieci secondi. È abbondante — ogni gesto ne produce — e per certi versi più
sincero delle dichiarazioni: puoi *dire* che ami i documentari, ma la
cronologia rivela le serie poliziesche. Ha però un difetto: è ambiguo. Un
click non è una promozione (magari il film ti ha deluso), e un film ignorato
non è una bocciatura: forse non l'hai mai visto passare. Per abbondanza,
l'implicito domina i sistemi reali; per ambiguità, richiede più cautela.

`````

`````{tab} Superiore

Il feedback **esplicito** produce una matrice di voti $R$ con entrate
$r_{ui}$ su scala ordinale (ad esempio $1$–$5$): segnale ad alta qualità ma
estremamente scarso, e per di più **non mancante a caso** — gli utenti votano
soprattutto ciò che hanno scelto di consumare, quindi le celle osservate sono
un campione distorto. Il feedback **implicito** produce eventi unari o conteggi
(click, acquisti, tempo di visione): copertura enormemente maggiore, ma niente
segnale negativo esplicito. L'assenza di interazione confonde due casi
indistinguibili — «non gli piace» e «non l'ha mai visto» — e questo cambia la
formulazione del problema: non più regressione sul voto, ma *ranking* da
osservazioni positive e non-osservazioni, come vedremo con BPR
{cite}`rendle2009bpr`. Nei sistemi industriali l'implicito domina per volume
(ordini di grandezza di differenza) e perché misura il comportamento
effettivo, non quello dichiarato.

`````

## Un problema di machine learning anomalo

Visto da lontano sembra il solito apprendimento supervisionato: dati storici
in ingresso, una predizione in uscita. Visto da vicino, tre cose lo rendono
un animale a parte.

La prima è la **sparsità**. La materia prima è una tabella con un utente per
riga e un film per colonna, e i voti nelle celle, come in
{numref}`fig-matrice-utenti-film`. Il punto è quante celle sono vuote: nel
dataset del Netflix Prize i 100 milioni di voti sembrano tanti, ma la tabella
completa avrebbe $480.000 \times 17.770 \approx 8{,}5$ miliardi di celle — ne
era piena appena l'1,2%. Nei cataloghi industriali di oggi, con milioni di
oggetti, si supera facilmente il 99,9% di vuoto. Raccomandare significa
riempire quei buchi in modo sensato.

```{figure} ../figures/matrice-utenti-film.svg
:name: fig-matrice-utenti-film
:alt: Griglia di sei utenti per otto film in cui poche celle contengono un voto da uno a cinque e tutte le altre un punto interrogativo; una cella evidenziata in terracotta indica il voto da prevedere.
:width: 90%

La matrice utenti × film: poche celle osservate, un oceano di punti
interrogativi. Prevedere il valore di una cella vuota — qui, il voto di Anna
al film D — è l'intero problema.
```

La seconda anomalia è che **non esiste una «risposta giusta» osservabile**.
Un classificatore di cifre può essere confrontato con l'etichetta vera; qui
la domanda è controfattuale: *se* ti avessimo mostrato quel film, ti sarebbe
piaciuto? Per la stragrande maggioranza delle coppie utente–film questa
risposta non verrà mai osservata, e la valutazione deve arrangiarsi con
approssimazioni: nascondere una parte delle interazioni note e verificare se
il modello le recupera.

La terza è la più insidiosa: **il sistema influenza i dati che raccoglie**.

`````{tab} Elementare

Immagina un cameriere che consiglia sempre gli stessi tre piatti. Dopo un
mese, i piatti più ordinati del ristorante saranno... quei tre. Se il
ristoratore guardasse le ordinazioni per capire cosa piace ai clienti,
concluderebbe che i tre piatti sono i favoriti — ma è una profezia che si
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
Learning {cite}`quinonero2009dataset` — con l'aggravante che qui lo shift
non è un incidente esterno, ma è prodotto dal sistema stesso. Le contromisure
(esplorazione controllata, correzioni per propensità) esistono, ma nessuna è
gratis: esplorare significa mostrare a qualche utente qualcosa che il modello
non avrebbe scelto.

`````

## Come è organizzato il capitolo

Due sezioni, dall'idea classica a quella neurale. Nella prima incontreremo il
**filtraggio collaborativo**: prima nella versione «a vicini» — chi ha amato
i film che ami tu, ha amato anche… — poi nella versione che ha vinto il
Netflix Prize, la fattorizzazione di matrici con i suoi fattori latenti, che
implementeremo in PyTorch con `nn.Embedding`. Nella seconda porteremo le reti
neurali dentro il problema: il Neural Collaborative Filtering, il passaggio
dal prevedere voti all'**imparare a ordinare** (la loss BPR e le metriche di
ranking), un cenno alla raccomandazione sequenziale e alle architetture
industriali a due stadi, e una riflessione finale su cosa succede quando il
suggerimento diventa pilotaggio.
