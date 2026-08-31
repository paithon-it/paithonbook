# Sistemi di Raccomandazione

Il 2 ottobre 2006 Netflix, che allora campava spedendo DVD per posta, mette in
palio un milione di dollari. Li prende chi riesce a migliorare del 10% il suo
sistema di raccomandazione, Cinematch.

Del 10% *che cosa*, conviene dirlo subito, perché la gara si gioca tutta lì.
Cinematch prevede quante stelle un utente darà a un film, e ogni previsione
sbaglia di un po’. Il metro della gara è quanto sbaglia, e si misura in stelle.
Non è però la semplice media degli errori. Ogni errore viene prima elevato al
quadrato, così una cantonata da tre stelle pesa nove volte un errore da una
stella ($3^2 = 9$ contro $1^2 = 1$); poi si fa la media di quei quadrati; e
solo alla fine se ne prende la radice, che non annulla i quadrati, perché in
mezzo c'è la media, e serve a riportare il risultato sulla scala delle stelle.
Con due sole previsioni, sbagliate di 1 e di 3 stelle, il conto fa
$(1 + 9) / 2 = 5$ e poi $\sqrt{5} \approx 2{,}24$: più della media semplice,
che sarebbe 2, ed è proprio quello che si voleva. Questo metro ha un nome che
ritroverai ovunque, **RMSE**, e la {doc}`sezione sulle metriche </MachineLearning/metriche>` lo definisce
per
esteso. Misurato così, l'errore di Cinematch valeva $0{,}9525$ stelle: togliere
il 10% vuol dire scendere sotto $0{,}8572$, ed è quella la soglia dell'assegno.

Per partecipare basta scaricare un dataset che all'epoca sembra sterminato:
poco più di 100 milioni di voti, da una a cinque stelle, dati da circa 480.000
utenti anonimi a 17.770 film. La gara diventa un caso mondiale: migliaia di
squadre, forum incandescenti, ricercatori universitari e ingegneri che di
notte inseguono decimali. E dura molto più del previsto, quasi tre anni. Solo
il 21 settembre 2009 Netflix consegna l'assegno al team **BellKor's Pragmatic
Chaos**, che chiude a $0{,}8567$ stelle: il 10,06% meglio di Cinematch, cioè
la soglia superata per un soffio. E sul filo di lana anche in gara: i rivali di
*The Ensemble* erano arrivati allo stesso punteggio, ma avevano consegnato
venti minuti più tardi.

L'ironia arriva dopo, e vale come lezione per tutto il capitolo. Quella
soluzione da un milione di dollari **non fu mai adottata per intero**. Era un
mosaico di oltre cento modelli combinati, e portarlo in produzione (cioè farlo
girare davvero, tutti i giorni, per i clienti veri) costava più di quanto
promettesse di rendere. Nel frattempo il business stava migrando dai DVD allo
streaming, dove prevedere il voto in stelle conta meno di prevedere che cosa
guarderai stasera. Due ingredienti emersi durante la gara, invece, in
produzione ci finirono davvero. Uno è una rete che impara a riconoscere le
combinazioni di gusti che ricorrono fra gli spettatori, e si chiama macchina di
Boltzmann ristretta: il {doc}`capitolo sui modelli a energia </ModelliEnergia/overview>` la racconta per intero.
L'altro è la **fattorizzazione di matrici**, che è la protagonista di questo
capitolo {cite}`koren2009matrix`.
Fattorizzare vuol dire scomporre in fattori, come si fa da sempre con i numeri
($12 = 3 \times 4$): qui si scompone una tabella, e i fattori sono due tabelle
strette al posto di una larghissima.

Una parola sul nome, prima di partire, perché in italiano «raccomandazione»
significa due cose e una delle due è la spintarella. Qui vale l'altra, quella
del consiglio: raccomandare è consigliare, e un sistema di raccomandazione è
una macchina che consiglia. Il senso brutto, però, non è del tutto fuori luogo.
Una macchina che decide cosa vedi ti sta servendo, o ti sta spingendo? Le
ultime pagine del capitolo affrontano la domanda, e conviene leggere tutto il
resto tenendola in mano.

## Non «qual è il film più bello», ma «quale piacerà a te»

Un motore di ricerca risponde a una domanda che fai tu. Un sistema di
raccomandazione risponde a una domanda che non hai fatto: *tra queste
centomila cose, quali conviene mostrarti?* Quelle cose sono film, canzoni,
prodotti, articoli, video, a seconda del catalogo. Nel gergo del settore si
chiamano tutte **oggetti** (in inglese *item*), e qui useremo spesso «film»,
perché l'esempio è quello. La differenza cruciale è che non esiste una
risposta valida per tutti.

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

## Il carburante: quello che diciamo e quello che facciamo

Da dove impara, una macchina che consiglia? Da quello che le persone lasciano
dietro di sé. Ogni volta che qualcuno incontra un film e lascia una traccia (un
voto, un click, un acquisto, dieci minuti di visione, un brano saltato), quella
traccia si può registrare, ed è tutto ciò che il sistema ha in mano. Le tracce
si chiamano **interazioni**, e sono di due specie molto diverse: quelle che
diciamo apposta e quelle che ci scappano mentre facciamo altro. In inglese si
chiamano *feedback* **esplicito** e **implicito**, e la distinzione conta più
di quanto sembri.

`````{tab} Elementare

Il **feedback esplicito** è quando dichiari il tuo giudizio: le cinque stelle
su un film, il pollice in su, la recensione. È chiaro ma raro: quante cose hai
guardato o comprato quest'anno, e quante ne hai *recensite*? E quelle poche
stelle arrivano quasi tutte dallo stesso posto: un film lo voti dopo averlo
scelto, e l'avevi scelto perché pensavi ti sarebbe piaciuto. Chi legge quei
voti legge il parere di gente che partiva ben disposta.

Il **feedback implicito** è tutto ciò che fai senza pensare di stare
giudicando: i click, gli acquisti, i minuti di visione, i brani saltati dopo
dieci secondi. È abbondante (ogni gesto ne produce) e per certi versi più
sincero delle dichiarazioni: puoi *dire* che ami i documentari, ma la
cronologia rivela le serie poliziesche. Ha però un difetto: è ambiguo. Un
click non è una promozione (magari il film ti ha deluso), e un film ignorato
non è una bocciatura: forse non l'hai mai visto passare. Nei sistemi veri è
l'implicito a farla da padrone, perché ce n'è tantissimo; ma proprio perché è
ambiguo va maneggiato con più cautela. E cambia la domanda: senza stelle da
prevedere, resta da decidere quale titolo mettere in cima alla vetrina.

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

La prima si vede guardando la materia prima, che è una tabella: una riga per
ogni persona iscritta (nel gergo del settore, un **utente**), una colonna per
ogni film, i voti nelle celle, come in
{numref}`fig-matrice-utenti-film`. Il punto è quante celle sono **vuote**. Nel
Netflix Prize i 100 milioni di voti sembrano tanti, ma la tabella completa
avrebbe $480.000 \times 17.770 \approx 8{,}5$ miliardi di celle: cento milioni
su otto miliardi e mezzo fa l'1,2%, ed è quanto ne era piena. Nei cataloghi
industriali di oggi, con milioni di oggetti, si scende facilmente sotto lo
0,1% di celle piene. Quel vuoto ha un nome, **sparsità**, e raccomandare
significa riempirlo in modo sensato.

```{figure} ../figures/matrice-utenti-film.svg
:name: fig-matrice-utenti-film
:alt: Griglia di sei utenti per otto film in cui poche celle contengono un voto da uno a cinque e tutte le altre un punto interrogativo; una cella evidenziata in terracotta indica il voto da prevedere.
:width: 90%

La tabella dei voti: poche celle di cui conosciamo il voto (le cornici
distinguono i voti alti, 4 e 5, dai bassi), e in quattro celle su cinque un
punto interrogativo. Prevedere il valore di una cella vuota (qui, il voto di
Anna al film D) è l'intero problema.
```

La seconda anomalia è che **non esiste una «risposta giusta» da guardare**. Un
classificatore di cifre si può confrontare con l'etichetta vera, perché
quell'immagine o è un 7 o non lo è, e qualcuno lo sa. Qui invece la domanda
riguarda un fatto che non è avvenuto: *se* ti avessimo mostrato quel film, ti
sarebbe piaciuto? Una domanda così si dice **controfattuale**, e per la
stragrande maggioranza delle coppie utente-film non avrà mai una risposta
osservata. Per giudicare un modello bisogna allora arrangiarsi con un ripiego:
si nasconde una parte delle interazioni che si conoscono, e si guarda se il
modello le ritrova. *Quale* parte si nasconde sposta i risultati più di quasi
ogni scelta di modello, e ci torneremo quando parleremo di come si misura una
classifica.

La terza è la più insidiosa: **il sistema influenza i dati che raccoglie**.

`````{tab} Elementare

Un cameriere consiglia sempre gli stessi tre piatti. Dopo un mese, i piatti più
ordinati del ristorante saranno... quei tre. Se il ristoratore guardasse le
ordinazioni per capire cosa piace ai clienti, concluderebbe che i tre piatti
sono i favoriti, ma è una profezia che si autoavvera: i clienti hanno scelto
dentro il menù che il cameriere ha proposto, e le ordinazioni del mese gli
danno una ragione in più per riproporli. I sistemi di raccomandazione vivono in
questo cerchio: ciò che mostri determina ciò che viene cliccato, e ciò che
viene cliccato determina ciò che mostrerai.

Uscirne si può, ma ogni strada ha un prezzo. Per sapere com'è il quarto piatto
bisogna portarlo a qualcuno che non l'ha chiesto, e quel qualcuno cenerà peggio
perché il locale aveva bisogno di saperlo. Oppure si aggiustano i conti a
tavolino, contando di più le ordinazioni dei piatti che il cameriere proponeva
di rado: al tavolo non cambia niente, ma bisogna sapere quante volte ciascun
piatto è stato proposto, e quel registro nessuno lo tiene da sé.

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

## Da chi somiglia a chi, fino alle reti

Due sezioni, dall'idea classica a quella neurale.

La prima muove da un'idea che usiamo tutti i giorni senza chiamarla così:
**chiedere all'amico giusto**. Vedremo come si rende calcolabile, e il metodo
si chiama *filtraggio collaborativo*. Vedremo poi perché la versione ingenua si
arena sul vuoto della tabella, e come se ne esce: riassumendo ogni persona e
ogni film in una scheda di pochi numeri. È l'idea che ha vinto il Netflix
Prize, e la scriveremo in PyTorch in una ventina di righe.

La seconda porta le reti neurali dentro il problema. La domanda è ovvia (una
rete farà meglio del confronto fra due schede?) e la risposta non è quella che
ci si aspetta. Proveremo a sostituire il confronto fra le schede con una
rete, e vedremo che non conviene. Ridisegneremo poi la tabella come un disegno
di pallini e linee, dove consigliare vuol dire indovinare le linee che mancano.
Cambieremo infine obiettivo: quando non ci sono voti non si prevede un numero,
si mette in ordine una vetrina, e per una vetrina serve anche un altro modo di
misurare se è buona. Chiuderemo con il funzionamento vero della macchina che ti
consiglia i video, e con la domanda che le sta sotto: quando un consiglio
smette di essere un consiglio.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Consigliare non è riconoscere: **non esiste una risposta valida per tutti**,
  e lo stesso sistema deve mettere le cose in un ordine diverso per ogni
  persona.
- Il carburante sono le **interazioni**: o dichiarate (le stelle, il pollice in
  su: chiare ma rarissime) o lasciate senza pensarci (click, acquisti, minuti
  di visione: abbondanti ma ambigue, perché un titolo ignorato non è una
  bocciatura). Nei sistemi veri dominano le seconde.
- Il dato di partenza è una **tabella quasi tutta vuota**: nel Netflix Prize
  era piena all'1,2%, nei cataloghi di oggi si scende sotto lo 0,1% di celle
  piene. Consigliare vuol dire riempire quei buchi in modo sensato.
- **Non c'è una risposta giusta da guardare**: nessuno saprà mai se ti sarebbe
  piaciuto il film che non hai visto, e per valutare si nasconde una parte di
  ciò che si sa, per poi vedere se il modello la ritrova.
- **Il sistema si fabbrica da solo i dati con cui impara**: mostra quello che
  ha scelto lui, e ciò che mostra è ciò che verrà cliccato. È il cameriere che
  consiglia sempre gli stessi tre piatti, ed è il problema che questo capitolo
  si porta dietro fino all'ultima riga. Uscirne si può, ma non gratis: per
  sapere com'è il quarto piatto bisogna portarlo a qualcuno che non l'ha
  chiesto.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il problema è prevedere le preferenze mancanti nella matrice di interazione
  $\mathbf{R}$, utenti per oggetti, di cui si osserva una frazione minima: nel
  Netflix Prize l'1,2% delle celle, nei cataloghi industriali di oggi meno
  dello 0,1%. Quel vuoto ha un nome, **sparsità**, ed è il vincolo che detta
  quasi tutte le scelte che seguono.
- Il segnale è di due specie. **Esplicito**: voti su una scala, cioè un target
  continuo o ordinale su una matrice incompleta, e raro. **Implicito**: click,
  acquisti, minuti di visione, cioè dati binari o di conteggio, abbondanti ma
  senza negativi certi, perché un oggetto mai mostrato non è un oggetto
  rifiutato.
- La verità di riferimento è **controfattuale** e per la gran parte delle
  coppie non esisterà mai. Si valuta per ripiego: si nascondono interazioni
  note e si guarda se il modello le ritrova. *Quale* parte si nasconde sposta
  i risultati più di quasi ogni scelta di modello.
- I dati non vengono dalla distribuzione vera delle preferenze ma dalla
  **politica di esposizione** del sistema che li ha raccolti: è un *feedback
  loop* che amplifica i bias invece di mediarli, cioè un caso severo di
  *dataset shift*, con l'aggravante di essere prodotto dal sistema stesso. Le
  contromisure (esplorazione controllata, correzioni per propensità) esistono
  e nessuna è gratis.
```

`````

