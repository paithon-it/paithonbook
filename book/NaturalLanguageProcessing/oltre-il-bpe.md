# Oltre il BPE: WordPiece, SentencePiece e i byte

## WordPiece: non il più frequente, il meno casuale

BPE ha un difetto che si vede bene nell'[esempio svolto](tokenizzatori.md)
con le cinque parole. Ha fuso per
prima la coppia `ss` perché la `s` è una lettera comunissima e le doppie
italiane sono ovunque: la coppia è frequente soprattutto perché i suoi pezzi
lo sono. Ma "frequente" e "significativo" non sono la stessa cosa. In quel
corpus la `a` compare *solo* dopo la `b`: `ba` è una coppia rara in assoluto
(8 occorrenze contro 25), ma è una coppia che non capita mai per caso.

Da qui il criterio alternativo di **WordPiece**, introdotto da Mike Schuster e
Kaisuke Nakajima nel 2012 per la ricerca vocale in giapponese e coreano
{cite}`schuster2012japanese` e diventato noto anni dopo come il tokenizzatore
di BERT {cite}`devlin2019bert`. La struttura dell'algoritmo è identica a
quella di BPE, si parte dai caratteri e si fonde una coppia per volta fino a
riempire il vocabolario. Cambia solo *quale* coppia si sceglie.

`````{tab} Elementare

In un giornale, «di» seguito da «un» capita in continuazione e non vuol dire
niente: capita perché entrambe sono parole comunissime. «Acqua» seguito da
«minerale» è molto più raro in assoluto, eppure ogni volta che leggete
«minerale» prima c'era «acqua». La seconda coppia dice qualcosa; la prima è
rumore di fondo.

WordPiece fa esattamente questa distinzione. Invece di chiedersi «quante volte
questi due pezzi si trovano attaccati?», si chiede: «si trovano attaccati più
di quanto capiterebbe per caso?». Il conto è semplice: si prende quante volte
la coppia compare e la si divide per quanto sono comuni i due pezzi presi
singolarmente. Un pezzo che è dappertutto viene penalizzato, e le sue coppie
devono essere davvero frequenti per vincere.

Nel nostro corpus di cinque parole, con questo criterio, la prima fusione non è
più `ss` ma `ba`, e il conto si può rifare a mano con due divisioni. Nei 146
caratteri del corpus la `s` compare 50 volte e la coppia `ss` 25: il punteggio è
25 diviso (50 per 50), cioè 0,01. La `b` compare 11 volte, la `a` 8 e la coppia
`ba` 8: il punteggio è 8 diviso (11 per 8), cioè circa 0,09, nove volte tanto.
La `s` è talmente diffusa che le sue coppie non stupiscono nessuno, mentre la
`a`, che si presenta sempre e solo dopo la `b`, è una compagnia troppo fedele
per essere una coincidenza. BPE premia ciò che ricorre, WordPiece premia ciò
che sta insieme per una ragione.

Questa divisione, però, non guarda quanto lavoro farà la fusione. Tornate al
giornale e mettiamo che «acqua» compaia 40 volte e «minerale» 12, sempre
preceduto da «acqua»: il punteggio della coppia è 12 diviso (40 per 12), cioè
0,025. Adesso prendete due parole che in tutto il giornale compaiono due volte
ciascuna, sempre una di seguito all'altra, come il nome e il cognome di un tale
che il giornale nomina in due righe e mai più: il punteggio è 2 diviso (2 per
2), cioè 0,5, venti volte tanto. Vincono loro, e incollarle accorcia il giornale
in due punti invece che in dodici. Il criterio misura quanto una coppia è
sorprendente, non quante volte tornerà utile, e giudica come se dovesse servire
una volta sola. Nel corpus delle cinque parole si vede già in piccolo: `et`, che
compare 5 volte, passa davanti a `ro`, che ne compare 14.

`````

`````{tab} Superiore

Il criterio del lavoro originale è: a ogni passo si aggiunge al vocabolario
l'unità ottenuta combinandone due esistenti che **aumenta di più la
verosimiglianza dei dati** sotto il modello di linguaggio. Nella forma in cui
l'algoritmo si è poi diffuso quel modello è un **unigramma**, cioè un modello
che assegna a una segmentazione $x = (x_1, \dots, x_\ell)$ la probabilità
$P(x) = \prod_i p(x_i)$ con $p$ stimata per frequenza relativa. Sotto questo
modello, il guadagno esatto di log-verosimiglianza di una fusione cresce
(circa) come $\mathrm{freq}(ab)$ volte il logaritmo di quanto la coppia è più
frequente del previsto: pesa cioè anche *quante volte* la fusione si applica.
Il criterio adottato in pratica lascia cadere quel peso e valuta il guadagno
*per occorrenza*; un'euristica ispirata alla verosimiglianza, più che una sua
conseguenza:

$$
(a^\star, b^\star) \;=\;
\arg\max_{(a,b)} \ \frac{\mathrm{freq}(ab)}{\mathrm{freq}(a)\cdot\mathrm{freq}(b)},
$$

dove $\mathrm{freq}(a)$ è il numero di occorrenze del simbolo $a$ nel corpus
segmentato allo stato corrente e $\mathrm{freq}(ab)$ quello della coppia
adiacente. È una trasformazione monotona della **informazione mutua puntuale**
(PMI): passando dalle frequenze assolute a quelle relative, il rapporto diventa
$\frac{p(ab)}{p(a)\,p(b)}$ a meno di un fattore che dipende solo dalla taglia
del corpus, quindi costante entro un singolo passo e ininfluente
sull’$\arg\max$. Quel rapporto è l'esponenziale della PMI fra $a$ e $b$, che
per definizione è il suo logaritmo,
$\mathrm{PMI}(a,b) = \log \frac{p(ab)}{p(a)\,p(b)}$: il rapporto vale
$1$ (e la PMI zero) quando i due simboli si incontrano esattamente come
farebbero per caso, più di $1$ quando si attirano.

Sul corpus giocattolo dell'esempio svolto, al primo passo (frequenze dei simboli:
`s` 50, `o` 44, `t` 14, `r` 14, `b` 11, `a` 8, `e` 5, su 146 caratteri):

| coppia | $\mathrm{freq}(ab)$ | $\mathrm{freq}(a)$ | $\mathrm{freq}(b)$ | punteggio |
|---|---|---|---|---|
| `b a` | 8 | 11 | 8 | **0,0909** |
| `e t` | 5 | 5 | 14 | 0,0714 |
| `t t` | 7 | 14 | 14 | 0,0357 |
| `r o` | 14 | 14 | 44 | 0,0227 |
| `s s` | 25 | 50 | 50 | 0,0100 |

La coppia più frequente in assoluto, `s s`, scende all'ottavo posto su dodici;
vince `b a`, che ha un terzo delle occorrenze ma due componenti rari. Il
denominatore è esattamente il correttivo: penalizza le coppie che devono la
loro frequenza alla diffusione dei pezzi.

Sul piano pratico, nell'implementazione resa popolare da BERT i pezzi **non
iniziali** portano il prefisso `##`, così che `bassetto` possa uscire come
`bass`, `##etto` e la ricomposizione sia priva di ambiguità (`##etto` non è lo
stesso token di `etto` a inizio parola: la distinzione fra prefissi e suffissi
è codificata nel vocabolario, non ricostruita a valle; il lavoro del 2012
otteneva lo stesso effetto con un marcatore di spazio attaccato alle unità).
BERT usa un vocabolario di circa $30\,000$ token così costruiti. In fase di
codifica, inoltre, quella implementazione non replica una lista di fusioni ma
applica una scansione **greedy del prefisso più lungo** presente nel
vocabolario: differenza sottile che può produrre segmentazioni diverse da
quelle che il replay di BPE darebbe a parità di vocabolario.

`````

## SentencePiece e i byte: togliere gli ultimi presupposti

BPE e WordPiece, così come li abbiamo descritti, danno per scontata una cosa:
che il testo arrivi **già diviso in parole**. Entrambi partono da un
dizionario parola $\to$ frequenza, e quel dizionario qualcuno lo deve
costruire, tagliando il testo sugli spazi e sulla punteggiatura. È un
presupposto innocuo in italiano e in inglese, e falso in giapponese, in cinese
e in thailandese, dove gli spazi fra le parole semplicemente non ci sono. In
quelle lingue serve un segmentatore addestrato a parte, che diventa un pezzo
in più da mantenere e una fonte in più di errori.

**SentencePiece**, presentato da Taku Kudo e John Richardson nel 2018
{cite}`kudo2018sentencepiece`, toglie il presupposto nel modo più diretto
possibile: non tratta il testo come una lista di parole, ma come un **flusso
grezzo di caratteri** in cui lo spazio è un carattere come gli altri.

`````{tab} Elementare

In una collana non ci sono vuoti, e «il gatto nero» ci entra così: al posto di
ogni spazio va una perlina segnaposto, `▁` (una barretta bassa, non un trattino
di sottolineatura), attaccata alla parola che comincia. Viene `▁il▁gatto▁nero`,
una fila ininterrotta su cui BPE gira come prima. In giapponese, dove gli spazi
non ci sono, non cambia niente: la fila era già così.

Per riavere la frase basta riaccostare i pezzi e rimettere uno spazio dove c'è
la barretta, senza regole su dove ci va e dove no (prima di *gatto* sì, prima
della virgola no, dopo l'apostrofo nemmeno). La frase torna com'era e non quasi
com'era, apostrofi e punteggiatura compresi: uguale a come la si era uniformata
prima di infilare, se quel passaggio si è fatto.

I pezzi con cui si compone la collana si possono anche scegliere al rovescio di
BPE. Nel cassetto ce ne sono migliaia di pronti, spezzoni già infilati, tutti i
frammenti che nel testo ricorrono un po', e non si chiude: allora se ne mette
via uno alla volta, si rifanno le collane della giornata senza di lui, e quello
di cui non si sente la mancanza si butta. Poi si rifà il giro, finché il
cassetto si chiude. Quanto serva ciascuno si scopre solo usandolo: si infila, si
segna quali sono serviti, si corregge la stima e si ricomincia, finché non
cambia più niente.

Davanti a una parola nuova, con il cassetto ripulito non si ripetono gli
incollaggi nell'ordine in cui sono stati scoperti: si cerca il modo migliore di
coprirla con gli spezzoni rimasti. Ogni modo ha il suo punteggio, quindi si può
prendere apposta il secondo migliore, e il modello vede la stessa parola
composta in modi diversi senza affezionarsi a uno solo.

Resta un buco. Gli spezzoni sono fatti delle lettere già viste, e le lettere del
mondo sono oltre centomila: nessun corpus le contiene tutte. Un ideogramma raro,
un simbolo matematico, un'emoji uscita ieri, e in mano resta un `<UNK>`, lo
stesso buco intravisto con `rossellini`.

Guardate una perlina più da vicino: anche quella che sembra una lettera è
l'incastro di perline più piccole, e queste hanno un nome, **byte**. Ne esistono
256 tipi, tante quante le file di otto caselle da zero o uno con cui un computer
scrive qualunque cosa: non 256 nei testi che si sono visti, 256 e basta, e non
le ha scelte nessuno. Un ideogramma ne occupa tre, un'emoji quattro, ma sempre
di quelle. Il cassetto di partenza copre tutto per costruzione, e «sconosciuto»
esce dal dizionario.

C'è un prezzo, e si paga sulla lunghezza della collana. Una lettera accentata
sono due perline, un ideogramma tre, un'emoji quattro; se quelle sequenze non
ricorrono abbastanza da meritarsi uno spezzone loro, vanno infilate una alla
volta, e un solo ideogramma giapponese può costare più pezzi di un'intera parola
inglese. Il conto arriva alle lingue che nel mucchio di testo di partenza
c'erano poco.

`````

`````{tab} Superiore

SentencePiece è insieme un formato e una libreria, e va guardato su due piani:
come tratta il testo in ingresso, e con quale algoritmo costruisce il
vocabolario.

Il primo piano è la **normalizzazione e la reversibilità**. L'input è trattato
come una sequenza Unicode, passata per una normalizzazione (NFKC nella
configurazione predefinita), in cui lo spazio è rimpiazzato dal meta-simbolo
`▁` (U+2581, LOWER ONE EIGHTH BLOCK) prefisso al segmento che segue. La
decodifica è la concatenazione dei token seguita dalla sostituzione inversa, e
vale l'identità

$$
\mathrm{decode}\bigl(\mathrm{encode}(\mathrm{norm}(s))\bigr)
= \mathrm{norm}(s),
$$

dove $s$ è il testo grezzo e $\mathrm{norm}$ la normalizzazione scelta. È la
proprietà che gli autori chiamano tokenizzazione *lossless*, e che nelle
pipeline basate su tokenizzatori dipendenti dalla lingua non è garantita,
perché la detokenizzazione è lì una collezione di regole ad hoc.

Il secondo piano è l'algoritmo di costruzione del vocabolario, dove
SentencePiece offre BPE e in alternativa il modello **unigram**
{cite}`kudo2018subword`, che procede al contrario: si parte da un vocabolario
candidato ampio $V$ e lo si **pota**. A $V$ fissato, il modello è lo stesso
unigramma già incontrato con WordPiece, cioè assegna a una segmentazione
$x = (x_1,\dots,x_\ell)$ di una stringa la probabilità $P(x) = \prod_i p(x_i)$,
ma qui la verosimiglianza di una stringa $s$ è la somma su tutte le
segmentazioni possibili, $P(s) = \sum_{x \in S(s)} P(x)$. Le $p(x_i)$ si
stimano con
l'algoritmo EM (lo stesso delle misture gaussiane, nel capitolo sul Machine
Learning: qui la variabile che renderebbe facile la stima e non si osserva non
è l'identità della componente, è la segmentazione);
poi, per ogni token, si calcola quanto la verosimiglianza
totale calerebbe rimuovendolo, e si elimina la frazione di token meno utili.
Si itera fino alla taglia voluta. La segmentazione di una stringa nuova è
quella di massima probabilità, trovata con Viterbi in tempo lineare nella
lunghezza. Due proprietà distinguono unigram da BPE: è un modello
**probabilistico**, quindi sa dire *quanto* una segmentazione è buona e
campionarne di alternative (la *subword regularization*, che addestrando su
segmentazioni diverse della stessa frase fa da regolarizzatore), e non dipende
da un ordine di fusioni.

Una terza mossa, che SentencePiece non ha inventato ma che si combina con le
prime due, è il **BPE a livello di byte**, adottato da GPT-2
{cite}`radford2019language`: si applica BPE non ai caratteri Unicode (che sono
oltre centomila, un alfabeto di base già proibitivo) ma alla codifica UTF-8 del
testo. L'alfabeto di base ha allora esattamente $|\Sigma| = 256$ elementi, e il
tasso di `<UNK>` è **zero per costruzione**, su qualunque input: testo, codice
sorgente, dati binari mascherati. Il prezzo è che un carattere fuori ASCII
occupa da 2 a 4 byte, e se le sue sequenze non sono abbastanza frequenti da
meritare una fusione, un singolo ideogramma può costare più token di una parola
inglese intera. La
copertura universale non è gratis: si paga in lunghezza di sequenza, sempre
per le lingue meno rappresentate nel corpus di addestramento del
tokenizzatore.

`````

## Quattro conseguenze che incontrerete davvero

Fin qui la meccanica. Ma la ragione per cui conviene conoscerla è che il
tokenizzatore, che sembra un dettaglio della preparazione dei dati, produce
quattro effetti visibili a chiunque usi un modello di linguaggio, e nessuno
dei quattro è una curiosità: sono tutti conseguenze dirette dell'algoritmo
appena descritto.

**Primo: i numeri si spezzano in modo irregolare, e l'aritmetica ne soffre.**
Le fusioni si scelgono per frequenza, e le cifre non fanno eccezione. Le
sequenze numeriche comuni sul web (gli anni recenti, i numeri tondi, `100`,
`000`, le cifre singole) si guadagnano un token tutto loro; quelle rare no.

Il guaio si vede con due numeri quasi uguali. Su un corpus in cui `2024` è
frequentissimo e `2025` meno, il primo può uscire come *un token solo* e il
secondo come *due*, `20` e `25`. Due numeri della stessa lunghezza, tagliati
in modo diverso, e la cifra `2` che nel primo caso sta dentro un pezzo unico e
nel secondo apre il pezzo `20`. Adesso pensate a come si fa una somma in
colonna a scuola: si incolonnano le unità sotto le unità, le decine sotto le
decine. Chiedere a un modello di sommare due numeri lunghi significa
chiedergli di incolonnare cifre che nella sua rappresentazione non sono
incolonnate affatto, perché è stato tagliato tutto secondo la frequenza e non
secondo il posto che ogni cifra occupa. Non spiega da solo tutti gli errori di
calcolo dei modelli, ma è un contributo strutturale e riconosciuto: tanto che
vari tokenizzatori recenti forzano la segmentazione delle cifre, una per una o
a gruppi fissi di tre, proprio per restituire al modello una griglia regolare.

**Secondo: l'italiano costa più token dell'inglese, a parità di significato.**

```{figure} ../figures/italiano-costa-piu-token.svg
:name: fig-italiano-token
:alt: "La stessa frase scritta in inglese e in italiano, una sopra l'altra, con i confini fra un token e l'altro marcati sopra ciascuna. In inglese, «The cat is on the table», le sei parole restano sei token interi. In italiano, «Il gatto è sopra il tavolo», due parole si spezzano in due pezzi ciascuna, «gatto» in «g» e «atto» e «tavolo» in «tav» e «olo», e i token diventano otto."
:width: 96%

Stessa frase, due conti diversi. Le parole italiane si frammentano perché il
vocabolario è stato costruito su un corpus in prevalenza inglese, e i posti se
li sono presi le sottostringhe inglesi.
```

Come mostra {numref}`fig-italiano-token`, il costo non è metaforico. Nasce da
due cause indipendenti che tirano nella stessa direzione.

La prima è la composizione del corpus. Se il testo su cui il tokenizzatore è
stato addestrato è in prevalenza inglese, le fusioni che «pagano» sono quelle
inglesi, e i posti nel vocabolario finiscono lì. Alle parole italiane restano i
pezzi avanzati, presi in prestito da altre parole, e si frammentano.

La seconda è la nostra grammatica. Le lingue che declinano e coniugano tutto
moltiplicano le forme: *gatto*, *gatta*, *gatti*, *gatte*, *gattino*,
*gattini* sono sei parole distinte, ciascuna delle quali va imparata per conto
suo, e ciascuna singolarmente più rara dell'inglese *cat*, che sta al posto di
quasi tutte. Più rara vuol dire meno probabile che si meriti un token suo.

E in che valuta si paga? In tre.

- **In denaro.** I servizi che danno accesso a un modello di linguaggio (un
  programma che, dato un testo, scommette su come continua: è il tema di una
  sezione più avanti) fanno pagare un tanto a token. La stessa richiesta
  scritta in italiano costa dunque più della stessa richiesta in inglese, e di
  quanto dipende dal tokenizzatore: nella frase della figura, spezzata dal tokenizzatore di GPT-2, sono dodici
  token contro sei, cioè il doppio (con tokenizzatori più recenti il rapporto
  scende, ma il verso non cambia mai).
- **In posti occupati.** Un modello può tenere davanti agli occhi solo una
  certa quantità di testo per volta, misurata anch'essa in token: è la sua
  **finestra di contesto**. Un documento che in inglese ci sta, in italiano può
  non starci.
- **In lavoro.** E qui il conto non è proporzionale, per via del costo
  quadratico dell'attenzione di cui si diceva all'inizio: se una lingua consuma
  il 50 per cento di token in più, l'attenzione su quel testo costa
  $1{,}5^2 = 2{,}25$ volte tanto.

È una disparità che non nasce da una scelta contro l'italiano, ma dalla
composizione di un corpus, e che si corregge solo addestrando il tokenizzatore
su dati più bilanciati.

**Terzo: uno spazio in più o in meno cambia i token.** Nei tokenizzatori
moderni lo spazio non è un separatore invisibile, è attaccato al token che lo
segue: `▁gatto` e `gatto` sono due voci diverse del vocabolario, con due
posizioni diverse sulla mappa e due storie diverse alle spalle. La prima è
comunissima, perché quasi sempre *gatto* è preceduto da uno spazio; la seconda
è rara, perché ricorre solo dove *gatto* attacca senza spazio davanti, cioè
quasi mai.

Ne segue una cosa che sorprende chiunque non l'abbia mai sentita. Il testo che
si scrive a un modello per farlo lavorare (una domanda, un'istruzione, l'inizio
di un documento da completare) si chiama **prompt**. Se il vostro prompt
finisce con uno spazio, quello spazio l'avete già speso voi, e al modello
tocca continuare con un token *senza* barretta iniziale, cioè con la variante
rara, quella su cui ha molta meno esperienza. Un solo carattere invisibile in
coda alla richiesta, e la risposta può peggiorare senza che si capisca il
perché. Non è fragilità del modello: è che gli avete dato in ingresso una
sequenza diversa da quella che credevate.

**Quarto: il vocabolario si fissa prima dell'addestramento e non si cambia
dopo.** Questa è la conseguenza più vincolante, e riguarda com'è fatto il
modello per dentro. In ingresso c'è una tabella con **una riga per ogni token
del vocabolario**: la riga contiene i numeri con cui quel pezzo di parola viene
rappresentato, ed è quella che nel resto del libro si chiama *matrice di
embedding*. In uscita ce n'è una seconda che fa il lavoro opposto, e infatti è
girata di novanta gradi: ha **una colonna per ogni token**, e serve a dare a
ciascun pezzo un punteggio per decidere quale scrivere. Una per entrare, una
per uscire. Aggiungere un token al vocabolario vuol dire allora aggiungere
una riga e una colonna vuote a due tabelle che l'addestramento ha già
riempito: numeri che nessuno ha mai regolato, in mezzo a numeri regolati per
mesi. E cambiare la segmentazione di un token esistente è peggio, perché
tutto ciò che il modello ha imparato su quella riga si riferisce ormai a
un'altra cosa. Il tokenizzatore è quindi parte del modello quanto i suoi pesi, si
distribuisce insieme a essi, e se il dominio d'uso non era rappresentato nel
corpus su cui è stato costruito (una lingua minore, la notazione chimica, un
linguaggio di programmazione poco diffuso) quel testo resterà frammentato per
tutta la vita del modello. Si può porre rimedio solo riaddestrando, o almeno
estendendo il vocabolario e riadattando gli embedding nuovi: entrambe
operazioni costose, che è il motivo per cui la scelta del tokenizzatore va
fatta all'inizio e con calma.

## Un'idea che vale oltre il testo

Conviene chiudere allargando lo sguardo. Tutto quello che avete letto serve a
una cosa sola: costruire un **alfabeto discreto** su cui possa lavorare un
modello che scrive un pezzo per volta, guardando quelli che ha già scritto (è
ciò che si intende con *autoregressivo*). Discreto vuol dire fatto di pezzi
separati e contabili, come le lettere di un alfabeto e non come le sfumature
di un colore: un insieme finito di simboli in cui qualunque testo in ingresso
si possa scrivere e da cui qualunque testo in uscita si possa ricomporre. Il
testo quell'alfabeto ce l'aveva già mezzo pronto (i caratteri) e il lavoro è
stato scegliere i raggruppamenti giusti.

Altri segnali quell'alfabeto non ce l'hanno affatto. L'audio è un'onda
continua, e per darlo in pasto allo stesso tipo di modello bisogna prima
inventarsi dei simboli. Il modo è più semplice di quanto sembri: ci si prepara
un catalogo fisso di frammenti sonori campione, diciamo mille, e poi si scorre
la registrazione un pezzetto alla volta, si cerca nel catalogo il campione che
somiglia di più a quello che si ha davanti, e al posto del suono si scrive il
suo numero di catalogo. L'onda diventa così una fila di numeri fra mille, cioè
un testo in un alfabeto di mille lettere. Questo mestiere ha un nome che
incontrerete nel capitolo sull'audio, ed è il **codec neurale**; il pezzo che
sceglie il campione più vicino si chiama *quantizzatore vettoriale*. Il
problema è lo stesso di questa sezione, la soluzione è diversa perché diversa è
la materia prima.

E la domanda che resta aperta, in entrambi i casi, è se il testo e il suono
debbano davvero passare per dei simboli, o se un giorno i modelli lavoreranno
direttamente sui byte grezzi. Per ora la risposta è economica più che teorica:
i simboli accorciano le sequenze, e la lunghezza delle sequenze è ciò che si
paga.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- **WordPiece** cambia una cosa sola rispetto a BPE: non incolla la coppia più
  frequente, ma quella che sta insieme più di quanto ci si aspetterebbe per
  caso («acqua minerale» contro «di un»).
- **SentencePiece** tratta il testo come una collana ininterrotta di simboli,
  con lo spazio scritto come `▁`: non c'è bisogno di tagliarlo prima in parole
  (indispensabile per cinese e giapponese, che gli spazi non li usano) e il
  testo si ricompone identico. Sotto i caratteri ci sono i **byte**, che sono
  256 comunque vada: partendo da lì non resta fuori più niente, mai.
- Le conseguenze si toccano con mano: **i numeri** vengono spezzati secondo la
  frequenza e non secondo il posto delle cifre, e i conti ne soffrono;
  **l'italiano costa più token dell'inglese**, cioè più soldi e più posti
  occupati nella finestra di contesto; **uno spazio di troppo** in fondo a una
  richiesta cambia davvero la domanda; e il vocabolario, una volta scelto,
  **non si cambia più**, perché fa parte del modello quanto i numeri che ha
  imparato.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- **WordPiece** {cite}`schuster2012japanese` ha la stessa struttura di BPE ma
  sceglie la coppia che massimizza
  $\mathrm{freq}(ab)/(\mathrm{freq}(a)\,\mathrm{freq}(b))$: non ciò che ricorre,
  ma ciò che ricorre più di quanto ci si aspetterebbe dal caso.
- **SentencePiece** {cite}`kudo2018sentencepiece` tratta il testo come flusso
  grezzo con lo spazio marcato da `▁`: nessun bisogno di pre-segmentare in
  parole (il che lo rende usabile per cinese e giapponese, dove gli spazi fra
  le parole non esistono) e detokenizzazione esatta. A livello di **carattere**
  l'assenza di `<UNK>` dipende da quali caratteri stavano nel corpus; il
  **livello dei byte** la rende una proprietà per costruzione, perché i byte
  sono 256 e basta.
- Le conseguenze si vedono a valle: **numeri** segmentati in modo irregolare
  (e aritmetica fragile), **lingue non inglesi** che consumano più token e più
  contesto, **spazi** che cambiano la sequenza in ingresso, e un vocabolario
  **congelato** prima dell'addestramento, perché è parte del modello quanto i
  suoi pesi.
```
`````
