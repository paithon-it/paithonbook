# Rappresentare il testo: dai token agli embedding

Un calcolatore non sa cosa sia una parola: sa solo manipolare numeri. Perciò,
prima ancora di costruire un traduttore automatico o un assistente
conversazionale, dobbiamo rispondere a una domanda che sembra banale e non lo
è: come si trasforma una frase come *"Il gatto nero salta sul muro"* in
qualcosa che una rete neurale possa moltiplicare, sommare, confrontare? Tutta
l'elaborazione del linguaggio naturale (*Natural Language Processing*, NLP) è,
in fondo, una lunga ricerca di **rappresentazioni del testo** sempre più
ricche. Partiamo dal gradino più basso e saliamo.

## Spezzare il testo: la tokenizzazione

Il primo passo è sempre lo stesso: tagliare il flusso di caratteri in unità
discrete, i **token**. Solo dopo potremo assegnare numeri a queste unità.

`````{tab} Elementare

Tokenizzare vuol dire affettare la frase. La ricetta più intuitiva è
"spezza a ogni spazio": *"Il gatto nero salta sul muro"* diventa la lista
`["Il", "gatto", "nero", "salta", "sul", "muro"]`. Sei parole, sei token.

L'elenco di tutti i pezzi che il sistema conosce si chiama **vocabolario**: è la
scatola dei mattoncini disponibili, e niente che non ci sia dentro può essere
rappresentato.

Riempire la scatola di parole intere non funziona. Le parole di una lingua non
finiscono mai: la scatola diventa enorme e resta comunque incompleta, e la prima
parola che manca lascia il sistema muto.

I posti nella scatola si contano allora in partenza, qualche decina di migliaia,
e si riempiono così. All'inizio ci si mette una lettera per posto: con le sole
lettere si scrive qualunque parola, anche se ci vogliono molti pezzi. Poi si
guarda un mucchio di testo e si cerca la coppia di pezzi vicini che ricorre più
spesso. Se `i` e `z` capitano attaccati un'infinità di volte, tanto vale
incollarli e tenere `iz` come pezzo unico: costa un posto solo e ne fa
risparmiare uno in ogni parola che lo contiene. La mossa si ripete finché i
posti finiscono. (Su quale coppia incollare i sistemi si dividono: c'è chi
prende la più frequente e chi quella che rende il testo più facile da indovinare
pezzo per pezzo.)

Alla fine nella scatola convivono pezzi di ogni taglia. Le parole comunissime,
*il* e *sul*, ci stanno per intero; *tokenizzazione* è rara e non ci sta, quindi
si scrive come `token` + `izzazione`, due pezzi visti tante volte altrove. E una
parola mai incontrata, un cognome o una sigla, si scrive lo stesso, al peggio
lettera per lettera: la scatola non resta mai senza pezzi.

`````

`````{tab} Superiore

Formalmente definiamo un **vocabolario** $V$, l'insieme dei token noti. La
tokenizzazione è una funzione che mappa una stringa nella sequenza dei suoi
token $\in V$. La segmentazione a spazi bianchi soffre di due problemi: un
vocabolario enorme e le parole fuori dizionario (*out-of-vocabulary*).

I sistemi moderni usano perciò tokenizzatori **sottoparola** (*subword*). Il
*Byte Pair Encoding* {cite}`sennrich2016neural` parte dai singoli caratteri e
fonde iterativamente la coppia di simboli più frequente; *WordPiece*
{cite}`schuster2012japanese` adotta una strategia analoga, ma fonde la coppia
che
massimizza la verosimiglianza del corpus anziché la semplice frequenza. In
entrambi i casi il processo si arresta quando il vocabolario raggiunge una
taglia fissata (tipicamente $30\,000$–$100\,000$ token). Ogni stringa, anche
mai vista, resta rappresentabile; nel caso limite si scende fino al singolo
byte.

`````

## Ogni parola un interruttore: il one-hot encoding

Abbiamo i token, e adesso bisogna dar loro dei numeri. L'idea più diretta è
numerarli in fila: *gatto* è 1, *cane* è 2, *mercoledì* è 3. Occupa pochissimo
spazio e non funziona, e la {numref}`fig-one-hot` mostra perché su un esempio
più semplice del testo: tre città.

```{figure} ../figures/one-hot-label-encoding.svg
:name: fig-one-hot
:alt: "La stessa colonna di dati, quattro righe e tre città (Milano, Roma, Napoli, Roma), codificata in due modi. Con l'ordinal encoding diventa una sola colonna di interi, 0, 2, 1, 2, che però introduce un ordine e delle distanze fra le città che non ne hanno. Con il one-hot diventa una colonna per città, riempita di zeri e con un solo uno per riga: tutte le città restano equidistanti."
:width: 96%

Due codifiche, due significati impliciti. Numerare in fila (è la codifica che
si chiama *ordinal*) dice che Milano viene prima di Napoli e che Roma dista da
Milano il doppio di Napoli: cose che nessuno intendeva dire. Una casella per
città, tutte a zero tranne una, non dice niente di tutto questo, ed è il suo
pregio.
```

Con le parole vale identico. Se *gatto* è 1 e *mercoledì* è 3, un programma
che vede numeri leggerà lì dentro un ordine e delle distanze, perché è quello
che i programmi fanno con i numeri: *mercoledì* risulterebbe «il triplo» di
*gatto*, e sarebbe una relazione che nessuno ha mai voluto affermare. Meglio
allora una **casella per ogni parola**, tutte a zero tranne quella giusta: è la
codifica che si chiama **one-hot**, «uno solo acceso». Lo spreco si vede a
occhio (con un vocabolario da decine di migliaia di voci servono decine di
migliaia di caselle per scrivere una parola sola) e lo si accetta lo stesso,
perché almeno non mente.

`````{tab} Elementare

Una lunghissima pulsantiera, con un interruttore per ogni parola del
vocabolario. Per rappresentare *gatto* accendi il suo interruttore e lasci
spenti tutti gli altri.

Scrivi ora $1$ per «acceso» e $0$ per «spento», e leggi la pulsantiera da
sinistra a destra: quello che ottieni è una lunga fila di numeri,
`0 0 1 0 0 ... 0`. Una fila di numeri presa nel suo ordine si chiama
**vettore**, e dentro quella parola non c'è niente di più misterioso di così.
Quello di *gatto* è lungo quanto il vocabolario ed è tutto zeri tranne un
singolo $1$.

Funziona, ma è uno spreco e, soprattutto, è cieco al significato. Per questa
codifica *gatto* e *felino* sono lontani esattamente quanto *gatto* e
*mercoledì*: ogni parola è un'isola, nessuna somiglianza è possibile.

`````

`````{tab} Superiore

La $i$-esima parola diventa il vettore della base canonica
$\mathbf{e}_i \in \mathbb{R}^{|V|}$: tutte componenti nulle tranne un $1$ in
posizione $i$. Due parole distinte $i \neq j$ danno vettori **ortogonali**,

$$
\mathbf{e}_i^\top \mathbf{e}_j = 0 ,
\qquad
\lVert \mathbf{e}_i - \mathbf{e}_j \rVert_2 = \sqrt{2} ,
$$

cioè prodotto scalare nullo e distanza identica per *qualsiasi* coppia. La
geometria non porta alcuna informazione semantica. In più la dimensione
$|V|$ è dell'ordine di $10^5$–$10^6$: i vettori sono enormi e sparsissimi.

`````

## Contare le parole: bag-of-words e TF-IDF

Per rappresentare un intero documento, e non una sola parola, si conta: quante
volte compare ciascuna parola del vocabolario, e amen all'ordine in cui
comparivano. Si chiama **sacchetto di parole** (*bag-of-words*), e il nome dice
tutto: come rovesciare il testo dentro un sacchetto e scuoterlo.

```{figure} ../figures/bag-of-words-tf-idf.svg
:name: fig-bag-of-words
:alt: "Un breve documento di testo viene trasformato in un vettore: ogni posizione del vettore corrisponde a una parola del vocabolario, e il valore è il numero di volte che quella parola compare nel documento. Le parole assenti lasciano zeri, e l'ordine originale del testo non è più ricostruibile."
:width: 92%

Il documento diventa un vettore di conteggi. Il testo di partenza non si può
più ricostruire: dell'ordine delle parole non resta traccia, e «il cane morde
l'uomo» dà lo stesso vettore di «l'uomo morde il cane».
```

L'esempio in coda a {numref}`fig-bag-of-words` è la misura esatta di cosa si
butta via. Per molti compiti non è grave (per capire se una recensione è
positiva, le parole contano più del loro ordine) ma è bene sapere che la
perdita c'è, ed è definitiva: nessun programma messo dopo questo passaggio
potrà recuperare l'ordine, perché a quel punto non è più scritto da nessuna
parte.

`````{tab} Elementare

Il documento diventa un conteggio: quante volte compare ciascuna parola. È la
pulsantiera di prima, con gli interruttori sostituiti da manopole: non più
«c'è / non c'è», ma «c'è, e tante volte». Resta **sparsa**, che è il modo
tecnico di dire che quasi tutte le caselle stanno a zero: un documento usa
qualche centinaio di parole diverse, e le caselle disponibili sono decine di
migliaia.

C'è però un problema: articoli e preposizioni come *il* o *di* compaiono
ovunque, e proprio per questo dicono poco su *cosa* parla il testo. Parole rare
come *retina* o *sinapsi* sono molto più rivelatrici.

Il peso **TF-IDF** corregge lo squilibrio moltiplicando fra loro due numeri, che
sono poi le due metà della sigla. La frequenza nel documento (*term frequency*)
è quante volte la parola compare nel testo che stiamo rappresentando: più ci
compare, più conta. La rarità nella raccolta (*inverse document frequency*, la
frequenza documentale rovesciata) guarda in quanti documenti la parola compare,
e premia quelle che ne occupano pochi.

La rarità si calcola così: si divide il numero totale di documenti per il numero
di quelli in cui la parola compare, e del risultato si prende il logaritmo, che
è solo un modo di schiacciare i numeri grandi perché non prendano il
sopravvento. Su una raccolta di mille documenti: *il* compare in tutti e mille,
mille diviso mille fa 1, e il logaritmo di 1 è zero. La rarità di *il* vale
zero, e zero per qualunque cosa fa zero: *il* sparisce, che è esattamente quello
che volevamo. *Sinapsi* compare in due documenti, mille diviso due fa
cinquecento, e il logaritmo di cinquecento è circa 6,2. Sopravvive, e pesa.

Il prodotto dei due numeri gonfia dunque le parole rare e informative e sgonfia
quelle comuni a tutti: stessa pulsantiera, manopole tarate meglio. Una cosa però
non cambia: le caselle restano mute fra loro, come gli interruttori di prima. Un
testo che parla di *gatti* e uno che parla di *felini* non hanno una sola
casella in comune, e per il conteggio non si somigliano per niente.

`````

`````{tab} Superiore

Il *bag-of-words* rappresenta un documento $d$ con il vettore dei conteggi
$\in \mathbb{R}^{|V|}$. Il peso **TF-IDF** (*Term Frequency-Inverse Document
Frequency*) di un termine $t$ è

$$
\text{tfidf}(t, d) = \text{tf}(t, d)\cdot \log\frac{N}{\text{df}(t)} ,
$$

dove $\text{tf}(t,d)$ è la frequenza di $t$ in $d$, $N$ il numero totale di
documenti, $\text{df}(t)$ il numero di documenti che contengono $t$, e il
logaritmo è quello **naturale** (la base cambia solo un fattore di scala comune
a tutti i termini). Il fattore logaritmico penalizza i termini onnipresenti
(df alto) fino ad azzerare chi compare ovunque: se $\text{df}(t) = N$ allora
$\log(N/N) = 0$. Questa è la forma da manuale, e le librerie ne calcolano
varianti lisciate. Restano due limiti strutturali: i vettori sono ancora sparsi
e $|V|$-dimensionali, e nessuna relazione lega parole diverse tra loro.

`````

Con `scikit-learn`, la cassetta degli attrezzi che in Python raccoglie i metodi
classici di machine learning, tutto questo è una manciata di righe. La riga che
conta è `fit_transform`: legge i due testi, si costruisce da sé l'elenco delle
parole che ci trova (il vocabolario) e restituisce una tabella con una riga per
documento e una colonna per parola.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["il gatto nero salta sul muro",
          "il cane dorme sul divano"]

vec = TfidfVectorizer()
X = vec.fit_transform(corpus)   # matrice sparsa documenti x vocabolario
print(vec.get_feature_names_out())  # il vocabolario appreso
print(X.toarray())                  # pesi TF-IDF per ciascun documento
```

I numeri che compaiono a schermo non sono però quelli del TF-IDF «da
manuale»: la libreria ne usa una variante, per ragioni pratiche. Il verso resta
quello, le parole rare pesano più di quelle comuni; i numeri no, e su una
raccolta piccola la differenza si vede.

`````{tab} Elementare

La variante serve a due cose. La prima è non trovarsi mai a dividere per zero.
Nel conto di poco fa si divide il numero dei documenti per quello dei documenti
che contengono la parola, e se quel secondo numero fosse zero (una parola che
c'è nel vocabolario ma in nessun testo) la divisione non si potrebbe fare:
la libreria aggiunge $1$ sopra e sotto e il problema sparisce. La seconda è
mettere sulla stessa scala documenti di lunghezza diversa, così che un testo
lungo non risulti più «pesante» solo perché contiene più parole; a conti fatti
ogni documento viene riportato alla stessa misura complessiva, e a contare sono
le proporzioni fra le sue parole, non quante ne ha.

C'è un effetto collaterale, ed è meglio conoscerlo perché altrimenti i numeri a
schermo sorprendono. Con la ricetta da manuale una parola presente in *tutti* i
documenti prendeva zero e spariva; con la variante della libreria le resta
addosso un peso, e su una raccolta piccola quel peso è tutt'altro che poco. Nei
due testi dell'esempio *il* esce con $0{,}318$ e *gatto* con $0{,}447$:
l'articolo che sta in tutti e due i testi pesa circa sette decimi della parola
che compare in uno solo. La distanza fra i due si allarga man mano che la
raccolta cresce, perché la parola che sta dappertutto resta ferma dov'è mentre
quella rara sale.

`````

`````{tab} Superiore

`scikit-learn` non applica la formula qui sopra alla lettera: usa un idf
*lisciato*, $\ln\frac{1+N}{1+\text{df}(t)} + 1$, e normalizza poi in $L^2$ il
vettore di ogni documento. Il «$+1$» finale ha una conseguenza da tenere a
mente: un termine presente in tutti i documenti non si annulla, come vorrebbe
$\log(N/\text{df})$, ma conserva idf pari a $1$. Nel corpus giocattolo qui
sopra ($N = 2$) non è affatto un residuo trascurabile: nel primo documento
*il* esce con peso $0{,}318$ contro lo $0{,}447$ di *gatto*, cioè circa il
71% del peso di un termine che compare in un solo documento. Il divario si
apre solo al crescere del corpus, perché l'idf del termine onnipresente resta
fisso a $1$ mentre quello del termine raro cresce come
$\ln\frac{1+N}{2} + 1$.

`````

## Vettori densi: i word embedding

Il salto concettuale arriva nel 2013. L'idea guida è vecchia, il linguista
John Firth nel 1957 la riassunse così: *"You shall know a word by the company
it keeps"*, conoscerai una parola dalla compagnia che frequenta. Parole che
appaiono in contesti simili hanno significati simili. Se lo facciamo dire ai
numeri, otteniamo gli **word embedding**: la parola inglese vuol dire
«immersione», e l'immagine è quella di ogni parola calata dentro uno spazio,
in un punto suo.

Come si fa in pratica lo mostra la {numref}`fig-finestra-contesto`. Si prende
una **finestra**, cioè un ritaglio di poche parole che scorre lungo il testo,
e a ogni posizione si guarda la parola al centro e quelle che le stanno
intorno. La coppia «parola centrale, sue vicine» è un esempio; il programma ne
raccoglie miliardi e aggiusta i numeri di ogni parola finché quelle che
capitano in mezzo alle stesse compagnie si ritrovano vicine.

```{figure} ../figures/word2vec-parola-dal-contesto.svg
:name: fig-finestra-contesto
:alt: "Una finestra tratteggiata ritaglia cinque parole di una frase: al centro, evidenziata, la parola «sul»; ai lati, due parole per parte, il suo contesto. Sotto, una freccia con la scritta «addestramento su miliardi di finestre» porta al vettore denso della parola, una fila di numeri con segno."
:width: 94%

La frase di Firth resa procedura. Nessuno dice al modello cosa significhi una
parola: gli si fa vedere in quali compagnie compare, milioni di volte, e il
vettore è il riassunto di quelle compagnie.
```

Quella procedura non usa nessun dizionario, nessun elenco di significati,
nessuna persona che spieghi qualcosa. Serve solo del testo qualsiasi, e questa
è la ragione del suo successo. Il testo qualsiasi è
gratis e infinito; un archivio di testi con le spiegazioni scritte a mano da
esperti (in gergo si dice che è **annotato**) costa mesi di lavoro di persone
vere ed è sempre piccolo. Il mucchio di testi su cui un programma si addestra
ha un nome che da qui in poi ricorre di continuo, ed è **corpus** (al plurale,
alla latina, *corpora*).

`````{tab} Elementare

Invece di migliaia di zeri con un solo uno, ogni parola diventa una corta
lista di poche centinaia di numeri, tutti "pieni". Questi numeri non li
scegliamo a mano: li impara un modello leggendo montagne di testo e notando
quali parole si accompagnano.

Il risultato è una **mappa del significato**. Su questa mappa *gatto* e
*felino* finiscono vicini, *gatto* e *cane* poco più lontani, *gatto* e
*mercoledì* agli antipodi. La vicinanza geometrica diventa vicinanza di
senso.

Quella vicinanza ha un modo standard di misurarsi, e conviene impararne il
nome adesso perché lo si incontra ovunque: la **similarità del coseno**.

Prima però serve un anello che finora è rimasto implicito. Una fila di due
numeri, `(3, 4)`, si può leggere come un punto su un foglio a quadretti: tre
caselle a destra, quattro in su. E un punto lo si può raggiungere solo in un
modo, con una freccia che parte dall'origine e arriva lì. Quindi fila di
numeri, punto e freccia sono la stessa cosa vista in tre modi. Con trecento
numeri il foglio a quadretti non basta più, ma le parole «punto» e «freccia»
continuano a valere, ed è per questo che di due parole si dice che «puntano»
da qualche parte.

La similarità del coseno guarda proprio le direzioni delle due frecce, e
ignora quanto sono lunghe. Non è una distanza in metri: è un numero fra $-1$ e
$+1$. Vale $+1$ quando le due frecce puntano esattamente dalla stessa parte,
$0$ quando sono perpendicolari, cioè non hanno niente da spartire, $-1$ quando
puntano in versi opposti. Un «coseno $0{,}88$» si legge «si somigliano molto»;
un «coseno $-0{,}26$» si legge «non c'entrano niente l'uno con l'altra».
(Nella pratica il caso $-1$ fra due parole quasi non si vede: i valori negativi
che si incontrano davvero sono piccoli, e vogliono dire «estranei», non
«contrari».)

I due programmi che hanno reso comuni questi vettori portano nomi che si
incontrano ovunque. **word2vec**, del 2013, è quello che lavora con la finestra
scorrevole. Chiedergli a ogni finestra «quale delle centomila parole del
vocabolario stava lì accanto?» costerebbe però una fortuna, e allora gli si
chiede molto meno: accanto alla vicina vera si pescano a caso cinque o dieci
parole qualsiasi dal testo, e il modello deve solo separare quella che c'era
davvero dalle intruse. Una domanda piccola, ripetuta miliardi di volte.

**GloVe**, del 2014, arriva allo stesso risultato per un'altra strada: conta una
volta per tutte quante volte ogni parola compare vicino a ogni altra, e poi
cerca i numeri che spiegano quei conteggi. Due estremi vanno tenuti a bada. Le
coppie che non si sono mai viste non hanno niente da insegnare, e restano fuori
dal conto invece di pesare come una somiglianza mancata. Le coppie che
contengono una parola come *di* si contano a milioni, e da sole coprirebbero la
voce a tutte le altre: oltre un certo tetto il loro peso smette di crescere, per
quanto salga il conteggio.

`````

`````{tab} Superiore

Un embedding è una funzione che associa a ogni token un vettore **denso**
$\mathbf{v} \in \mathbb{R}^{d}$ con $d$ piccolo (tipicamente $100$–$300$),
appreso dai dati. **word2vec** {cite}`mikolov2013efficient` addestra una rete
poco profonda a predire il contesto data la parola (*skip-gram*) o viceversa
(*CBOW*). Nello skip-gram l'obiettivo è massimizzare, su ogni coppia
parola-vicina $(w, c)$ del corpus, la probabilità
$p(c \mid w) = \mathrm{softmax}(\mathbf{u}_c^\top \mathbf{v}_w)$, dove
$\mathbf{v}$ e $\mathbf{u}$ sono i vettori della parola come centro e come
contesto. Quella softmax corre però su tutto il vocabolario, e in pratica la
si sostituisce con il **negative sampling** {cite}`mikolov2013distributed`:
per ogni coppia vera si pescano $k$ parole a caso dal corpus e si addestra un
classificatore binario a distinguere la vicina vera dalle intruse,
massimizzando

$$
\log \sigma(\mathbf{u}_c^\top \mathbf{v}_w)
+ \sum_{i=1}^{k}
\log \sigma(-\mathbf{u}_{c_i}^\top \mathbf{v}_w),
$$

con $\sigma$ la sigmoide e $c_1, \dots, c_k$ le intruse pescate. **GloVe**
{cite}`pennington2014glove` fattorizza invece la matrice
globale di co-occorrenza $\mathbf{X}$, minimizzando

$$
\sum_{i,j} f(X_{ij})\,
\big(\mathbf{v}_i^\top \tilde{\mathbf{v}}_j + b_i + \tilde{b}_j
- \log X_{ij}\big)^2,
$$

dove $X_{ij}$ conta quante volte la parola $j$ compare vicino alla parola $i$
e la funzione di pesatura $f$ cresce fino a un tetto e poi resta piatta: così
le coppie mai viste non contano niente ($f(0)=0$) e quelle frequentissime non
dominano il conto. Da $\mathbb{R}^{|V|}$ sparso si passa a
$\mathbb{R}^{d}$ denso: meno dimensioni, ma cariche di struttura semantica.

`````

Quella procedura ha un nome, e conviene impararlo qui perché torna per tutto il
libro. Nessuno ha preparato gli esercizi su cui word2vec si addestra: la parola
al centro e le sue vicine stavano già nel testo, e a separarle per farne una
domanda e una risposta siamo stati noi. Un compito costruito così si chiama
**auto-supervisionato**. La sezione «Imparare senza etichette», nel capitolo
sulla visione artificiale, l'ha già fatto sulle immagini: si copre un pezzo di
foto e si chiede di indovinarlo. Il capitolo che porta quel nome lo racconta
per esteso.

## Sotto la parola: fastText

word2vec e GloVe hanno però un punto cieco: assegnano un vettore a ogni parola
*intera*. Ne seguono due guai.

Il primo: se una parola nel corpus di addestramento non compare mai, per lei
non c'è nessun vettore, e il programma resta muto. Il secondo riguarda le
lingue come la nostra, che declinano e coniugano tutto. *Gatto*, *gatta*,
*gatti*, *gattino* sono quattro parole distinte da imparare quattro volte, e
ciascuna singolarmente più rara dell'inglese *cat*, che sta al posto di tutte.
**fastText** {cite}`bojanowski2017enriching` risolve i due guai con una mossa
sola: scendere sotto il livello della parola.

`````{tab} Elementare

La mossa è scomporre ogni parola in mattoncini di poche lettere che si
sovrappongono. Prendiamo *gatto* e scriviamoci accanto due segnacci, uno
all'inizio e uno alla fine, per ricordarci dove la parola comincia e dove
finisce: `<gatto>`. Ora la si affetta a gruppi di tre caratteri, spostandosi di
uno per volta: `<ga`, `gat`, `att`, `tto`, `to>`. (Tre non è sacro: nella
pratica si tengono insieme le fette da tre fino a sei lettere, così da prendere
sia le sillabe sia le desinenze intere.)

Ogni mattoncino ha il proprio vettore, e il vettore di *gatto* è semplicemente
la **somma** dei vettori dei suoi mattoncini. Sommare due file di numeri vuol
dire sommarle casella per casella: `(3, 4)` più `(1, 2)` fa `(4, 6)`, e basta.
Fra i mattoncini di *gatto* c'è anche `<gatto>` per intero, che è un pezzo come
gli altri e per una parola comune è il più informativo di tutti.

I vantaggi sono due, e concreti. Primo: *gatto*, *gatta* e *gattino*
condividono i pezzi `gat` e `att`, quindi i loro vettori nascono già simili
(prezioso in una lingua di desinenze come la nostra). Secondo: di una parola mai
vista prima manca il mattoncino della parola intera, che nessuno ha avuto
occasione di imparare, ma le fette da tre, quattro, cinque lettere ci sono
tutte, e sommando quelle un vettore esce lo stesso. Nessuna parola resta più
senza rappresentazione.

`````

`````{tab} Superiore

fastText estende lo *skip-gram* di word2vec rappresentando ogni parola $w$
con l'insieme $\mathcal{G}_w$ dei suoi **n-grammi di caratteri** (tipicamente
$3 \le n \le 6$, con i delimitatori `<` e `>` a marcare i confini), più la
parola stessa. Per $n = 3$, *gatto* produce `<ga`, `gat`, `att`, `tto`,
`to>`. Il vettore della parola è la somma dei vettori dei suoi n-grammi:

$$
\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g ,
$$

dove $\mathbf{z}_g \in \mathbb{R}^{d}$ è il vettore appreso per l'n-gramma
$g$. Due conseguenze pratiche: le parole **out-of-vocabulary** restano
rappresentabili sommando i soli n-grammi, e nelle lingue morfologicamente
ricche le forme flesse di una stessa radice condividono parametri; Bojanowski
e colleghi misurano i guadagni maggiori proprio su lingue molto flessive come
ceco e tedesco {cite}`bojanowski2017enriching`.

`````

## L'aritmetica del significato: re − uomo + donna ≈ regina

Quando le parole diventano vettori densi, succede qualcosa di sorprendente:
le relazioni di significato prendono la forma di **direzioni** nello spazio, e
si possono sommare e sottrarre come frecce.

```{figure} ../figures/embedding-analogia.svg
:name: fig-embedding-analogia
:alt: Quattro punti (uomo, re, donna, regina) in un piano; le frecce che li collegano formano un parallelogramma, con la freccia "regalità" e la freccia "femminile" ripetute su entrambi i lati.
:width: 85%

I quattro embedding formano un parallelogramma: la stessa freccia
"regalità" separa *uomo* da *re* e *donna* da *regina*; la stessa freccia
"femminile" separa *uomo* da *donna* e *re* da *regina*. Il disegno è
idealizzato: nello spazio vero le due frecce non sono identiche e il
parallelogramma si chiude solo per approssimazione, come si legge qui sotto.
```

`````{tab} Elementare

Guarda la {numref}`fig-embedding-analogia`. La freccia che va da *uomo* a *re*
significa più o meno "diventare regale". Se prendi quella stessa freccia e la
applichi a *donna*, dove atterri? Molto vicino a *regina*.

Quella freccia si scrive con una sottrazione, ed è l'unico passaggio da
digerire. Provalo su due numeri soli, che si disegnano sul quaderno. Se *uomo*
sta in `(1, 1)` e *re* sta in `(3, 4)`, la freccia che porta da *uomo* a *re* è
«due a destra e tre in su»: e infatti `(3, 4)` meno `(1, 1)`, fatto casella per
casella, dà `(2, 3)`. La regola sta tutta qui: si sottrae il punto di partenza
dal punto di arrivo, e quel che resta è la freccia che va dall'uno all'altro.
Applicarla a *donna* vuol dire sommargliela: se *donna* sta in `(1, 5)`, atterro
in `(3, 8)`, e se lì vicino c'è *regina* il gioco è fatto. In formula:
*re − uomo + donna ≈ regina*. I quattro punti disegnano un parallelogramma, e
questo è il segno che il modello ha catturato da solo il concetto di "regalità"
e quello di "genere", senza che nessuno glieli abbia mai spiegati.

C'è un'avvertenza che quasi nessuno racconta, e che invece è la parte più
istruttiva. Se il conto lo si fa davvero, e poi si cerca la parola più vicina
al punto di arrivo, la vincitrice non è *regina*: è *re*. La freccia del genere
è una spintarella, debole rispetto alla distanza che separa una parola
dall'altra: sposta il punto quel tanto che basta a portare *regina* al secondo
posto, non abbastanza da farle superare *re*. Tutti i programmi che fanno
queste analogie **tolgono dalla gara le tre parole della domanda**, e solo così
la risposta che esce è quella famosa. L'analogia geometrica esiste davvero,
insomma, ma è più tenue di come la si disegna, e il parallelogramma della
figura è un'idealizzazione.

E le frecce portano con sé anche quello che non vorremmo. I numeri vengono da
testi scritti da persone, e le associazioni che stavano in quei testi ci
finiscono dentro tali e quali. Fatta la stessa domanda con *medico* al posto di
*re*, la risposta che il programma dà per *donna* è *infermiera*. Il conto non
ha sbagliato niente: ha restituito quello che nei testi c'era.

`````

`````{tab} Superiore

L'osservazione, resa celebre da Mikolov et al. (2013), è che le analogie sono
approssimativamente lineari nello spazio degli embedding:

$$
\mathbf{v}_{\text{re}} - \mathbf{v}_{\text{uomo}} + \mathbf{v}_{\text{donna}}
\approx \mathbf{v}_{\text{regina}} .
$$

Operativamente si calcola il vettore a sinistra e si cerca la parola il cui
embedding gli è più vicino, misurando la prossimità con la **similarità del
coseno** già incontrata in *Algebra lineare*:

$$
\cos\theta = \frac{\mathbf{a}^\top \mathbf{b}}
{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert} \in [-1, 1] .
$$

La ricerca però si fa **escludendo dai candidati le tre parole della domanda**,
e quel vincolo non è un dettaglio di implementazione: senza di esso il primo
vicino è quasi sempre *re* stesso. Su GloVe da $100$ dimensioni addestrato su
6 miliardi di token (i vettori distribuiti come `glove-wiki-gigaword-100`, su
cui i conti che seguono si rifanno in tre righe), il coseno con *king* vale
$0{,}855$ contro lo $0{,}783$ di *queen*, e la stessa cosa succede a
`man : doctor :: woman : ?` ($0{,}866$ per *doctor*, $0{,}776$ per *nurse*) e a
`good : better :: bad : ?` ($0{,}886$ per *bad*, $0{,}839$ per *worse*).
L'enunciato onesto non è dunque l’$\approx$ della formula, ma

$$
\arg\max_{w \,\notin\, \{\text{re},\,\text{uomo},\,\text{donna}\}}
\cos\bigl(\mathbf{v}_w,\ \mathbf{v}_{\text{re}} - \mathbf{v}_{\text{uomo}}
+ \mathbf{v}_{\text{donna}}\bigr) = \text{regina} .
$$

La lettura corretta è che l'analogia lineare è una **direzione debole
sovrapposta a una posizione forte**: la geometria sposta il punto abbastanza da
mettere *regina* al secondo posto, non abbastanza da farle superare *re*. Chi
ha discusso a fondo la questione è Nissim, van Noord e van der Goot
{cite}`nissim2020fair`, che mostrano quanto di ciò che si legge sulle analogie,
comprese quelle usate come prova di *bias*, dipenda da quella scelta di
implementazione, mai scritta nelle equazioni.

Non è magia e non è perfetta, quindi, anche al netto dell'esclusione: molte
analogie falliscono, e questi vettori ereditano i **pregiudizi** dei testi su
cui sono addestrati (per esempio associazioni di genere a certi mestieri). Ne
parleremo, ma il messaggio resta: il significato, ridotto a geometria, si
lascia misurare con un prodotto scalare.

`````

## Dalla parola alla frase: gli embedding di frase

Tutto quello che abbiamo costruito finora dà un vettore per **parola**. Ma
quasi tutto ciò che si vuole fare davvero riguarda testi interi: trovare i
documenti che rispondono a una domanda, accorgersi che due segnalazioni
arrivate allo sportello sono lo stesso reclamo, raggruppare le recensioni per
tema, pescare da un archivio i tre paragrafi giusti da mettere sotto gli occhi
di un chatbot prima che risponda. Serve un vettore per **frase**, e ottenerlo
non è altrettanto ovvio.

`````{tab} Elementare

Diecimila moduli di reclamo in uno scatolone, e bisogna trovare i due che
dicono la stessa cosa.

L'impiegata riassume ogni modulo con una fila di numeri e confronta le file. La
media dei numeri già pronti di ogni parola, come primo tentativo, regge
sorprendentemente bene, e si rompe in due punti. Mescolate, le parole perdono
l'ordine, e «il cane morde l'uomo» e «l'uomo morde il cane» diventano lo
stesso riassunto. E le parole piccole pesano quanto le altre, così fra «il
film mi è piaciuto» e «il film non mi è piaciuto» il «non» annega.

Si fa aiutare da un lettore che l'ordine lo tiene, della famiglia dei
Transformer. Si chiama **BERT**, è del 2018, e ha passato miliardi di frasi su
due esercizi: indovinare le parole che gli avevano cancellato, e dire se due
frasi stessero davvero una dopo l'altra. Quell'allenamento si riusa per compiti
diversissimi senza rifare tutto. Ma nessuno gli ha mai chiesto quanto due
moduli vogliano dire la stessa cosa, e infatti lo giudica male: quello che si
vuole da uno spazio bisogna insegnarglielo.

Gliela insegna tre moduli per volta: uno di riferimento, l'**ancora**, uno che
dice la stessa cosa, uno che parla d'altro. Chiede solo che l'ancora finisca
più vicina al primo che al secondo, e non di un soffio, ma di uno scarto minimo
deciso in partenza, il margine; dove vadano i tre non lo dice. Le terne che
rispettano già il margine non hanno più niente da insegnare, e le mette da
parte. Su milioni di terne lo spazio si riordina da sé, e «vicino» diventa
«stesso argomento».

Il lettore dev'essere lo stesso per tutti e tre i moduli, e la rete si dice
**siamese** per questo: tre riassuntori diversi darebbero numeri non
confrontabili. E le
terne non si preparano una per una. Sul tavolo vanno mille coppie «reclamo e
suo gemello», e per ognuna i moduli delle altre novecentonovantanove fanno da
lontani: quasi mille, senza cercarne uno.

Lo scatolone, adesso: quasi cinquanta milioni di coppie (ognuna con ognuna,
$10\,000 \times 9\,999$ diviso $2$). Passare a BERT le due frasi attaccate,
così che le legga come un testo solo, dà il giudizio migliore, ma vale per
quella coppia sola. Cinquanta milioni di letture sono circa 65 ore,
cronometrate nel 2019 su una GPU da laboratorio dagli autori di
**Sentence-BERT**. Riassumere una volta sola le diecimila frasi costa cinque
secondi, e i cinquanta milioni di confronti fra file già pronte un centesimo di
secondo. È la differenza fra un'idea e un prodotto.

Quello che l'impiegata ha insegnato, però, è «questi due si somigliano», non
«questo risponde a quello». «Chi ha scritto la Divina Commedia?» e «Dante
Alighieri la compose fra il 1304 e il 1321» non hanno una parola in comune.
Allo sportello delle domande servono allora due lettori distinti, uno per le
domande e uno per i testi, allenati insieme perché una domanda e la sua
risposta finiscano vicine. Cambia il compito, cambiano gli esempi, cambia lo
spazio.

`````

`````{tab} Superiore

La *media dei vettori di parola* è una baseline seria (con pesatura
inversa alla frequenza regge il confronto con molti metodi neurali), ma è
**invariante alla permutazione**, quindi cieca alla sintassi, e diluisce le
parole di funzione che ne rovesciano il senso.

Con un encoder contestuale il problema si sposta ma non sparisce. Prendere il
vettore del token `[CLS]` di un BERT pre-addestrato, o la media dei suoi token,
dà rappresentazioni **peggiori** della media di GloVe su compiti di similarità
semantica: `[CLS]` è ottimizzato per il *next sentence prediction* e per
essere rifinito, non per vivere in uno spazio metrico. La lezione è generale:
**la geometria di uno spazio latente riflette l'obiettivo con cui è stato
addestrato**, e la similarità del coseno non è una proprietà che si ottiene
per caso.

**Sentence-BERT** {cite}`reimers2019sentence` risolve il problema con una
struttura **siamese**: lo stesso encoder $f_\theta$ (pesi condivisi, non due
reti gemelle) applicato a ciascun ingresso, un *pooling* sui token (la media
funziona meglio del `[CLS]`) e un obiettivo che agisce sulle **distanze**.

Le funzioni obiettivo di questa famiglia, il *metric learning*, sono tre e
conviene distinguerle.

La **contrastive loss** lavora su coppie: avvicina le simili, allontana le
dissimili fino a un margine $m$, e oltre quel margine smette di spingere
(altrimenti spenderebbe capacità a separare cose già separate).

La **triplet loss** lavora su terne $(\mathbf{a}, \mathbf{p}, \mathbf{n})$, gli
embedding di ancora, positivo e negativo, e chiede una disuguaglianza
**relativa**:

$$
\mathcal{L} = \max\big(0,\; m + \mathrm{dist}(\mathbf{a}, \mathbf{p})
- \mathrm{dist}(\mathbf{a}, \mathbf{n})\big),
$$

dove $\mathrm{dist}(\cdot,\cdot)$ è una distanza fra due embedding (di solito
l'euclidea, o $1-\cos$) e $m > 0$ è il margine di poco fa: cioè «il positivo deve stare più
vicino del negativo, e di almeno $m$». È più robusta della contrastive perché
non impone distanze assolute, che sarebbero arbitrarie, ma solo un ordinamento.

La **multiple negatives ranking loss** (o InfoNCE, la stessa forma già
incontrata per SimCLR nella sezione *Imparare senza etichette* del capitolo
sulla visione artificiale, e che tornerà per CLIP in *Allineare due spazi*, nel
capitolo su visione e linguaggio) usa come negativi tutti gli altri elementi
del batch:

$$
\mathcal{L} = -\log \frac{\exp\big(\mathrm{sim}(\mathbf{a}, \mathbf{p})/\tau\big)}
{\sum_{j=1}^{B} \exp\big(\mathrm{sim}(\mathbf{a}, \mathbf{p}_j)/\tau\big)} ,
$$

dove $\mathbf{a}$ è l'ancora e $\mathbf{p}$ il suo positivo,
$\mathbf{p}_1, \dots, \mathbf{p}_B$ sono i positivi di tutti gli esempi del
batch di taglia $B$ (il proprio, che sta anche al numeratore, più i $B-1$
altrui, che fanno da negativi), $\mathrm{sim}$ è la similarità del coseno di
poco sopra e $\tau > 0$ è la **temperatura**, che decide quanto il denominatore
sia dominato dai candidati più vicini. È una cross-entropia su un problema a
$B$ vie in cui la classe giusta è «il proprio positivo».

È oggi la scelta prevalente, perché un batch da 1024 fornisce 1023 negativi
gratis a ogni esempio, ed è precisamente la ricetta degli *in-batch negatives*
che la sezione *Retrieval e RAG*, nel capitolo sui Transformer, attribuisce a
DPR {cite}`karpukhin2020dense`.

Due avvertenze pratiche che separano un modello che funziona da uno che no.
La prima è la **scelta dei negativi**: quelli presi a caso diventano presto
banali (due testi su argomenti scorrelati sono facilissimi da separare, e la
loss va a zero senza aver insegnato niente), per cui si passa ai *hard
negatives*, cercati apposta fra i quasi-simili, tipicamente scavandoli con un
modello precedente. La seconda è il rischio opposto, il **collasso**: nulla
vieta alla rete di mandare tutte le frasi di un argomento esattamente nello
stesso punto, che azzera la loss e distrugge ogni distinzione fine.
Temperatura, margine e regolarizzazione servono a governare quel compromesso.

Infine una nota che chiude il cerchio con la sezione su RAG: la similarità del
coseno misura **«si somigliano»**, non **«questo risponde a quella»**. Una
domanda e la sua risposta spesso non si somigliano affatto, e infatti si
addestrano due torri distinte, $E_q$ ed $E_p$, con positivi che sono coppie
domanda-passaggio e non coppie di parafrasi. Cambia il compito, cambiano i
positivi, cambia lo spazio.

`````

Che l'addestramento *riorganizzi lo spazio* si può guardare da vicino con un
esperimentino, e senza scaricare nessuno dei modelli veri, che pesano gigabyte.
L'idea: fabbrichiamo centoventotto finte frasi, divise in quattro argomenti da
trentadue, e diamo a ciascuna una fila di sedici numeri **tirati a caso**. Uno
spazio così non sa niente degli argomenti: due frasi dello stesso tema sono
lontane quanto due di temi diversi, ed è esattamente la situazione di una rete
a cui la somiglianza non è mai stata insegnata. Poi si applica la regola delle
terne descritta qui sopra (ancora, simile, diverso, e l'ordine da rispettare) e
si guarda che cosa succede.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# Uno spazio di partenza che NON separa gli argomenti: quattro argomenti,
# trentadue "frasi" ciascuno, coordinate casuali. È il caso di un encoder
# che non è mai stato addestrato alla somiglianza.
N_ARG, PER_ARG, D = 4, 32, 16
argomento = torch.arange(N_ARG).repeat_interleave(PER_ARG)
grezzi = torch.randn(N_ARG * PER_ARG, D)
indici = [torch.nonzero(argomento == k).flatten() for k in range(N_ARG)]

def coseni(V):
    """Coseno medio dentro l'argomento e fra argomenti diversi."""
    V = F.normalize(V, dim=-1)
    S = V @ V.T
    stesso = argomento[:, None] == argomento[None, :]
    stesso.fill_diagonal_(False)
    return S[stesso].mean().item(), S[~stesso].mean().item()

def sorteggia(n):
    """Una tripletta per riga: ancora, un simile, un diverso."""
    a = torch.randint(0, len(grezzi), (n,))
    ka = argomento[a]
    p = torch.stack([indici[k][torch.randint(0, PER_ARG, (1,))][0] for k in ka])
    kn = (ka + torch.randint(1, N_ARG, (n,))) % N_ARG      # un argomento diverso
    neg = torch.stack([indici[k][torch.randint(0, PER_ARG, (1,))][0] for k in kn])
    return a, p, neg

# La "torre": UNA sola rete, applicata a tutti e tre gli ingressi.
# I pesi condivisi sono ciò che rende la rete siamese.
torre = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, D))
ott = torch.optim.Adam(torre.parameters(), lr=1e-2)
MARGINE = 0.3

for _ in range(400):
    a, p, neg = sorteggia(128)
    A, P, N = (F.normalize(torre(grezzi[i]), dim=-1) for i in (a, p, neg))
    # l'ancora deve stare più vicina al positivo che al negativo, e non di
    # poco: almeno di un margine. Chi già rispetta il margine non contribuisce.
    perdita = F.relu(MARGINE - (A * P).sum(-1) + (A * N).sum(-1)).mean()
    ott.zero_grad(); perdita.backward(); ott.step()

for etichetta, V in [("prima", grezzi), ("dopo ", torre(grezzi).detach())]:
    dentro, fuori = coseni(V)
    print(f"{etichetta}: coseno dentro l'argomento {dentro:+.3f}, "
          f"fra argomenti diversi {fuori:+.3f}, distacco {dentro - fuori:+.3f}")
```

Il programma stampa due numeri, prima e dopo: il coseno medio fra due frasi
**dello stesso argomento** e quello fra due frasi **di argomenti diversi**. Se
lo spazio sa il fatto suo, il primo deve essere molto più alto del secondo, e
la loro differenza è il distacco.

All'inizio il distacco è $-0{,}004$, cioè zero: dentro e fuori si somigliano
allo stesso modo, come dovevamo aspettarci da numeri tirati a caso. Dopo
quattrocento passi il coseno fra frasi dello stesso argomento è salito a
$+0{,}876$ e quello fra argomenti diversi è sceso a $-0{,}261$. Nessuno ha
detto alla rete quali fossero i quattro argomenti né dove metterli: le sono
state date solo delle terne e un ordine da rispettare, e la geometria si è
riorganizzata da sé.

Conviene guardare anche il rovescio della medaglia. Un coseno di $0{,}876$ fra
frasi dello stesso argomento è tantissimo: vuol dire che quelle trentadue
frasi si sono quasi ammucchiate in un punto solo. Va benissimo se il compito è
separare quattro temi, e malissimo se il compito è distinguere due sfumature
*dentro* lo stesso tema, perché lì dentro non c'è più spazio. Il rimedio, in
un modello vero, è scegliere meglio la terza frase di ogni terna, quella che
deve stare lontana: se la si pesca a caso è quasi sempre di un altro pianeta,
la regola è già rispettata e la rete non impara niente. Si vanno allora a
cercare apposta le frasi *quasi* uguali all'ancora e però diverse, che in
gergo si chiamano **negativi difficili**, e sono quelle che insegnano
qualcosa. Ecco perché un modello di embedding si sceglie guardando il compito
che si ha davvero, e non una classifica generica.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- **Tokenizzare** vuol dire affettare il testo in pezzi. I sistemi moderni
  usano pezzi più piccoli della parola, così anche una parola mai vista si
  ricostruisce dai suoi mattoncini.
- Un **vettore** è una fila di numeri, e niente di più. Il modo più ingenuo di
  darne uno a una parola è la pulsantiera con un interruttore acceso e tutti
  gli altri spenti: funziona, ma per lei *gatto* e *felino* sono lontani
  esattamente quanto *gatto* e *mercoledì*.
- Contare quante volte compare ogni parola di un documento (il **sacchetto di
  parole**) butta via l'ordine per sempre; il peso **TF-IDF** aggiusta i conti
  gonfiando le parole rare e sgonfiando quelle che stanno dappertutto.
- Gli **embedding** danno a ogni parola poche centinaia di numeri, imparati
  leggendo montagne di testo: una mappa del significato, in cui la vicinanza si
  misura con la **similarità del coseno**, un numero fra $-1$ e $+1$.
- *Re meno uomo più donna* atterra vicino a *regina*, ma la risposta esce solo
  se dalla ricerca si tolgono le tre parole della domanda: la freccia del
  significato esiste, ed è più debole di come la si disegna.
- **fastText** spezza le parole in mattoncini di poche lettere e ne somma i
  vettori: un vettore ce l'ha anche una parola mai vista, e *gatto*, *gatta* e
  *gattino* nascono già simili fra loro.
- Per una **frase** intera, la media dei vettori delle sue parole è un punto di
  partenza onesto ma cieco all'ordine; e un BERT preso così com'è non fa
  meglio, perché nessuno gliel'aveva chiesto. Se vuoi che uno spazio abbia una
  certa proprietà, quella proprietà devi addestrarla: una sola rete usata tre
  volte, e triplette di frasi da avvicinare e da allontanare.
- Il coseno dice «si somigliano», non «questo risponde a quella»: chi cerca
  risposte addestra due reti separate, una per le domande e una per i testi.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- **Tokenizzare** spezza il testo in unità; i sistemi moderni usano token
  *sottoparola* per gestire qualunque parola.
- **One-hot** e **bag-of-words / TF-IDF** danno vettori enormi, sparsi e senza
  nozione di somiglianza tra parole diverse.
- Gli **word embedding** (word2vec, GloVe) sono densi e a bassa dimensione: la
  **vicinanza geometrica riflette la vicinanza di significato**, misurata con
  la **similarità del coseno**.
- L’**analogia lineare** ($\mathbf{v}_{\text{re}} - \mathbf{v}_{\text{uomo}} +
  \mathbf{v}_{\text{donna}}$) restituisce *regina* solo perché i tre termini di
  ingresso sono esclusi dai candidati: senza quel vincolo, mai scritto nelle
  equazioni, vince *re* {cite}`nissim2020fair`.
- **fastText** somma i vettori degli *n-grammi di caratteri*: dà un vettore
  anche alle parole mai viste e sfrutta la morfologia; un aiuto concreto per
  lingue flessive come l'italiano.
- Per un vettore di **frase** la media dei vettori di parola è una baseline
  onesta ma cieca all'ordine; e un BERT preso così com'è dà embedding di frase
  mediocri, perché **è stato addestrato ad altro**. La similarità va
  addestrata: reti **siamesi** (un solo encoder a pesi condivisi) e obiettivi
  sulle distanze (contrastive, **triplet**, in-batch negatives).
- La scelta dei **negativi** decide il risultato: quelli casuali diventano
  presto banali, quelli difficili insegnano; e il rischio opposto è il
  **collasso** di tutto un argomento in un punto.
- Attenzione a cosa si misura: il coseno dice «si somigliano», non «questo
  risponde a quella». È il motivo per cui il retrieval usa **due torri**,
  domande da una parte e passaggi dall'altra.
```
`````
