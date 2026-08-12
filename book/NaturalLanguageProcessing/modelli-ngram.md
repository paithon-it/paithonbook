# Scommettere sulla prossima parola: i modelli n-gram

San Pietroburgo, inverno 1913. Andrej Andreevič Markov, cinquantaseienne,
matematico dell'Accademia Imperiale delle Scienze, ha davanti a sé l'*Evgenij
Onegin* di Puškin (il romanzo in versi che ogni russo conosce a memoria) e una
matita. Ricopia le prime 20.000 lettere, tutto il primo capitolo e parte del
secondo, eliminando spazi e punteggiatura; le dispone in duecento tabelle da
dieci righe per dieci colonne; poi conta, a mano, vocali e consonanti
{cite}`markov1913essai`. Il primo numero è banale: il 43 per cento delle
lettere sono vocali. Il secondo no: dopo una consonante, una vocale arriva
circa due volte su tre (0,663); dopo un'altra vocale, appena una volta su otto
(0,128). Le lettere di Puškin non sono estrazioni del lotto, indipendenti
l'una dall'altra: ogni lettera *ricorda* quella che la precede.

Il movente era una polemica tutta matematica: il collega moscovita Pavel
Nekrasov sosteneva che la legge dei grandi numeri valesse solo per eventi
indipendenti. Markov, polemista di rara costanza, costruì per smentirlo la
teoria delle sequenze di eventi dipendenti (oggi le chiamiamo **catene di
Markov**, e le ritroveremo nel capitolo sul Reinforcement Learning) e andò a
cercare la dipendenza dentro un capolavoro della letteratura. Presentò i
conteggi all'Accademia il 23 gennaio 1913.

Trentacinque anni dopo, Claude Shannon (nell'articolo del 1948 che abbiamo già
incontrato nel capitolo sui richiami di matematica
{cite}`shannon1948mathematical`) rovescia il gioco: non contare per *capire*
un testo, ma contare per *generarne* uno. Le sue «approssimazioni in serie»
all'inglese si costruiscono con un metodo artigianale quanto quello di Markov:
apri un libro a caso, leggi finché non incontri l'ultima lettera che hai
scritto, annoti quella che la segue; riapri il libro da un'altra parte e
ripeti. Salendo di ordine (dalle lettere alle coppie di parole) esce roba come
*«THE HEAD AND IN FRONTAL ATTACK ON AN ENGLISH WRITER THAT THE CHARACTER OF
THIS POINT…»*: ogni gruppetto di due o tre parole fila liscio, l'insieme non
significa nulla. Tre anni dopo, con un gioco di predizione fatto in casa,
Shannon stimerà quanto l'inglese sia prevedibile lettera per lettera
{cite}`shannon1951prediction`: circa un bit a lettera, come abbiamo visto
parlando di compressione.

L'idea che unisce il conta-lettere di Pietroburgo e l'ingegnere dei Bell Labs
è la tesi di questa sezione: **il linguaggio si può modellare come una catena
di scommesse**. Nella sezione precedente il testo era un sacchetto di parole,
e per il classificatore «Il gatto nero salta sul muro» e «Il muro nero salta
sul gatto» erano gemelli indistinguibili. Qui l'ordine torna protagonista:
costruiamo la macchina più semplice che scommette sulla parola successiva, il
modello **n-gram**, che per mezzo secolo è stato il cuore del NLP statistico.

## La lingua come catena di scommesse

Un **modello di linguaggio** (*language model*) è un sistema che assegna una
probabilità a una frase, o, che è lo stesso, scommette su quale parola verrà
dopo, date quelle già lette. Sembra un esercizio astratto, ma è dappertutto: è
la barra dei suggerimenti della tastiera del telefono, è il correttore che
preferisce «buona serata» a «buona serrata», ed è ciò che permette a un
riconoscitore vocale di scegliere la trascrizione che «suona» più italiana.

`````{tab} Elementare

Facciamo il gioco del completamento: «Il gatto nero salta sul…». Tu una
risposta ce l'hai già: «muro», forse «tetto», di sicuro non «marmellata». Come
fai? Hai letto e ascoltato milioni di frasi, e *sai cosa viene di solito dopo
cosa*.

Un modello n-gram fa la stessa cosa, ma con un quaderno. Immagina di leggere
tonnellate di testo con la matita in mano: ogni volta che incontri la parola
«gatto», vai alla pagina intestata «gatto» e annoti la parola che la segue. A
fine lettura, la pagina è la scommessa bell'e pronta: se dopo «gatto» hai
visto 3 parole in tutto (2 volte «nero», 1 volta «bianco»), allora scommetti
«nero» con fiducia 2 su 3 e «bianco» con fiducia 1 su 3.

C'è un patto nascosto, però. A rigore, per scommettere bene dovresti tener
conto di *tutta* la frase letta fin lì; ma quasi nessuna frase intera compare
due volte, nemmeno in una biblioteca, e non avresti mai conteggi. Il patto (lo
stesso di Markov con le sue vocali) è fingere che conti solo l'ultima parola
(modello a coppie, il **bigramma**) o le ultime due (a terne, il
**trigramma**). È una semplificazione dichiarata, e funziona molto meglio di
quanto meriti.

`````

`````{tab} Superiore

Un modello di linguaggio stima la probabilità congiunta di una sequenza di
parole $w_1, \dots, w_m$. La **regola della catena** la scompone, senza
alcuna approssimazione, in un prodotto di probabilità condizionate:

$$
P(w_1, \dots, w_m) = \prod_{t=1}^{m} P(w_t \mid w_1, \dots, w_{t-1}),
$$

dove $w_t$ è la parola in posizione $t$ e ogni fattore è la scommessa sulla
parola successiva dato il prefisso. Il problema è la stima: i prefissi
lunghi sono quasi tutti unici e i loro conteggi valgono zero o uno.
L'**assunzione di Markov** tronca la storia alle ultime $n-1$ parole:

$$
P(w_t \mid w_1, \dots, w_{t-1}) \;\approx\; P(w_t \mid w_{t-n+1}, \dots, w_{t-1}).
$$

Con $n=2$ si ha il **bigramma**, $P(w_t \mid w_{t-1})$; con $n=3$ il
**trigramma**. La stima è quella di **massima verosimiglianza** (MLE), già
incontrata nel capitolo sui richiami di matematica: semplici frequenze
relative,

$$
P(w_t \mid w_{t-1}) = \frac{C(w_{t-1}\, w_t)}{C(w_{t-1})},
$$

dove $C(\cdot)$ conta le occorrenze nel corpus di addestramento: quante volte
la coppia $w_{t-1} w_t$ appare, diviso quante volte appare $w_{t-1}$. Si
dimostra che queste frazioni massimizzano la verosimiglianza del corpus. Due
accorgimenti pratici: si incorniciano le frasi con simboli di inizio e fine,
`<s>` e `</s>`, così anche la prima parola e la chiusura sono scommesse come
le altre; e i prodotti di molte probabilità piccole si calcolano come somme di
logaritmi, per evitare l'*underflow* (lo stesso trucco di Naive Bayes).

`````

## Tre frasi e un quaderno di conteggi

Basta un corpus giocattolo per vedere tutta la macchina in funzione. Il
nostro sarà di tre frasi:

1. «il gatto nero salta sul muro»
2. «il gatto bianco dorme sul divano»
3. «il cane guarda il gatto nero»

Contiamo le coppie di parole adiacenti (aggiungendo a ogni frase i segnali
di inizio `<s>` e fine `</s>`). I conti che seguono si scrivono in una
notazione che vale la pena decifrare una volta per tutte, perché ricorre in
tutto il libro: $P(\text{gatto} \mid \text{il})$ si legge «la probabilità di
*gatto*, **sapendo che** prima c'era *il*». La barretta verticale vuol dire
«dato che», e separa la cosa su cui si scommette (a sinistra) da quello che si
sa già (a destra). Tutto qui: è una frazione con un nome, e la frazione è
proprio la pagina del quaderno, «quante volte questa parola ha seguito
quell'altra, diviso quante volte quell'altra è comparsa».

- ogni frase comincia con «il»: $P(\text{il} \mid \langle s \rangle) = 3/3 = 1$;
- «il» compare 4 volte, seguito 3 volte da «gatto» e 1 da «cane»:
  $P(\text{gatto} \mid \text{il}) = 3/4 = 0{,}75$ e
  $P(\text{cane} \mid \text{il}) = 0{,}25$;
- «gatto» compare 3 volte, seguito 2 volte da «nero» e 1 da «bianco»:
  $P(\text{nero} \mid \text{gatto}) = 2/3$;
- «nero» compare 2 volte: una seguita da «salta», una a fine frase:
  $P(\text{salta} \mid \text{nero}) = 0{,}5$;
- «sul» compare 2 volte, seguito una volta da «muro» e una da «divano»:
  $P(\text{muro} \mid \text{sul}) = 0{,}5$.

La probabilità dell'intera frase-simbolo del libro è il prodotto delle
scommesse lungo la catena:

$$
\begin{align*}
P(\text{il gatto nero salta sul muro})
&= 1 \cdot 0{,}75 \cdot \tfrac{2}{3} \cdot 0{,}5 \cdot 1 \cdot 0{,}5 \cdot 1 \\
&= 0{,}125,
\end{align*}
$$

dove i sette fattori sono, nell'ordine: «il» a inizio frase, «gatto» dopo
«il», «nero» dopo «gatto», «salta» dopo «nero», «sul» dopo «salta» (unica
continuazione vista: probabilità 1), «muro» dopo «sul», e la chiusura di frase
dopo «muro». Un ottavo di probabilità: alta, per un corpus così piccolo; la
frase è perfettamente «in stile».

## Gli zeri, o la maledizione delle coppie mai viste

Ora proviamo con «il cane nero salta sul divano». Frase italiana
ineccepibile, eppure il modello la boccia senza appello: la coppia «cane
nero» non compare mai nel corpus, quindi $P(\text{nero} \mid \text{cane}) = 0$,
e uno zero nel prodotto azzera tutto. Il modello confonde «mai visto» con
«impossibile».

Non è un difetto del nostro corpus giocattolo: è la regola. Jurafsky e Martin
{cite}`jurafsky2026speech` fanno i conti sull'opera omnia di Shakespeare: circa
884.000 parole con un
vocabolario di 29.000 tipi; le coppie *possibili* sono più di 800 milioni,
quelle effettivamente osservate circa 300.000 (il 99,96 per cento dei bigrammi
possibili non compare mai). E Shakespeare è un corpus generoso. Qualunque
testo nuovo conterrà coppie legittime che il modello non ha mai visto: la
**sparsità** è la condizione normale del linguaggio, non l'eccezione. I rimedi
si chiamano **smoothing** (lisciamento): togliere un po' di probabilità alle
coppie viste per regalarne un po' a quelle mai viste.

`````{tab} Elementare

Il rimedio più semplice l'abbiamo già incontrato nel filtro antispam: la
«regola del +1» di Laplace (eccolo, il vecchio amico promesso). Si regala un
conteggio a ogni coppia possibile, così nessuna resta a zero. Nel nostro
corpus il vocabolario ha 12 simboli (11 parole più il segnale di fine frase),
e i due numeri della frazione cambiano tutti e due. Prendiamo «cane nero», mai
vista. Sopra, al posto dello 0, va l'unico conteggio regalato: 1. Sotto vanno
le volte in cui «cane» è comparso (una sola) più i 12 regali che abbiamo appena
distribuito: $1 + 12 = 13$. La probabilità passa quindi da 0 a $1/13$, circa
0,08: piccola, ma viva.

Il regalo però lo pagano i ricchi, e lo pagano caro. «Gatto» dopo «il» valeva
$3/4 = 0{,}75$; adesso sopra c'è $3 + 1$ e sotto ci sono le 4 volte in cui «il»
è comparso più i 12 regali, cioè $(3+1)/(4+12) = 4/16 = 0{,}25$. Il piatto si
divide fra tutte le 12 continuazioni possibili, dieci delle quali sono coppie
inventate, mai viste davvero. Con un vocabolario vero, di decine di migliaia di
parole, il +1 diventa una patrimoniale che ridistribuisce quasi tutto ai
fantasmi.

L'idea migliore è un'altra: quando la coppia non s'è mai vista, invece di
inventare, **ripiega** su un giudice meno esigente. Non sai nulla di «nero
dopo cane»? Chiedi quanto è comune «nero» da solo. O meglio ancora:
**mescola** sempre i due giudizi (un po' di coppia, un po' di parola singola)
come chi chiede consiglio a due amici, uno pignolo e smemorato, uno vago che
ricorda tutto, e pesa le risposte.

`````

`````{tab} Superiore

Lo **smoothing add-1 di Laplace** (identico a quello visto per Naive Bayes)
somma 1 a ogni conteggio:

$$
P_{+1}(w_t \mid w_{t-1}) = \frac{C(w_{t-1}\, w_t) + 1}{C(w_{t-1}) + |V|},
$$

dove $|V|$ è la dimensione del vocabolario, che compare al denominatore perché
il +1 va garantito a tutte le $|V|$ continuazioni possibili. Sul corpus
giocattolo ($|V| = 12$):
$P_{+1}(\text{nero} \mid \text{cane}) = (0+1)/(1+12) \approx 0{,}077$, ma
$P_{+1}(\text{gatto} \mid \text{il}) = (3+1)/(4+12) = 0{,}25$, contro lo
$0{,}75$ della stima MLE. Add-1 sposta *troppa* massa verso l'inosservato:
tanto che si usa semmai la variante add-$k$ con $k \ll 1$. Le alternative
serie sono due, spesso combinate:

- **interpolazione**: mescolare sempre gli ordini,

  $$
  \hat{P}(w_t \mid w_{t-1}) = \lambda_1\, P(w_t) + \lambda_2\, P(w_t \mid w_{t-1}),
  \qquad \lambda_1 + \lambda_2 = 1,
  $$

  dove i pesi $\lambda_i$ non si fissano a mano ma si ottimizzano su un
  insieme di validazione (e con i trigrammi si aggiunge un terzo termine);

- **backoff**: usare l'ordine alto quando il suo conteggio è positivo e
  *ripiegare* sull'ordine inferiore altrimenti, con un fattore di sconto che
  tenga i conti in regola (la probabilità totale deve restare 1).

Il confronto sistematico tra queste famiglie è lo studio empirico di Chen e
Goodman {cite}`chen1999empirical`, per anni la bussola di chi costruiva
modelli n-gram.

`````

C'è però una trappola sottile nel ripiegare sulla parola singola, e la si
racconta con un esempio diventato canonico. In un corpus americano la parola
*Francisco* è frequente (si parla spesso di San Francisco) ma compare
praticamente solo dopo *San*. Se il bigramma non sa che pesci pigliare e tu
ripieghi sulla frequenza pura, finirai per scommettere *Francisco* in posti
dove non può stare: *«I can't see without my reading ___»* completato con
*Francisco* invece che con *glasses*. L'italiano ha gli stessi fantasmi:
«soppiatto» non è una parola rara, ma vive quasi soltanto nella coppia fissa
«di soppiatto». La frequenza di una parola non dice quanto è *versatile*.

`````{tab} Elementare

Pensa a due conoscenti. Il primo compare in mille foto, ma sempre alla
stessa festa: inseparabile dal padrone di casa, mai visto altrove. Il
secondo compare in cento foto, ma di cento feste diverse. Chi è più
probabile incontrare a una festa *nuova*? Il secondo, ovviamente: il primo è
frequente, ma non va da nessuna parte senza il suo amico.

L'idea di Kneser e Ney è tutta qui: quando devi ripiegare sulla parola
singola, non chiederti «quante volte l'ho vista?» ma «**in quanti contesti
diversi** l'ho vista?». «Francisco» e «soppiatto» hanno conteggi alti ma un
solo contesto: come riempitivi di buchi nuovi valgono poco. Una parola vista
dopo cento parole diverse, invece, è una buona scommessa quasi ovunque.

`````

`````{tab} Superiore

Lo smoothing di **Kneser–Ney** {cite}`kneser1995improved` sostituisce, nel
termine di ripiego, la frequenza unigramma con la **probabilità di
continuazione**:

$$
P_{\text{cont}}(w) =
\frac{\bigl|\{\, w' : C(w'\, w) > 0 \,\}\bigr|}
     {\bigl|\{\, (u, v) : C(u\, v) > 0 \,\}\bigr|},
$$

dove il numeratore conta i *contesti distinti* in cui $w$ è comparsa (quante
parole diverse l'hanno preceduta almeno una volta) e il denominatore è il
numero di bigrammi distinti del corpus, che normalizza. Per «Francisco» il
numeratore vale quasi 1, per quanto alta sia la sua frequenza. La forma
interpolata del modello bigramma è

$$
P_{KN}(w_t \mid w_{t-1}) =
\frac{\max\bigl(C(w_{t-1}\, w_t) - d,\; 0\bigr)}{C(w_{t-1})}
+ \lambda(w_{t-1})\, P_{\text{cont}}(w_t),
$$

dove $d$ è uno **sconto assoluto** (tipicamente intorno a $0{,}75$) sottratto
a ogni conteggio positivo, e $\lambda(w_{t-1})$ è il coefficiente che
raccoglie esattamente la massa scontata e la ridistribuisce secondo
$P_{\text{cont}}$, così che le probabilità sommino a 1. La variante
«modificata» di Chen e Goodman (tre sconti diversi a seconda che il conteggio
valga 1, 2 o di più) risultò la vincitrice sistematica del loro studio
{cite}`chen1999empirical` ed è rimasta lo standard de facto dei modelli n-gram
fino all'era neurale.

`````

## La pagella della scommettitrice: la perplessità

Come si misura se un modello di linguaggio scommette bene? Con la
**perplessità**, e l'immagine da tenere è quella del dado: la perplessità dice
con quante facce è il dado su cui il modello sta tirando a ogni scommessa.
Perplessità 2, e il modello esita fra due parole soltanto; perplessità 100, e
ne ha davanti cento tutte ugualmente plausibili. Più il numero è basso, meglio
scommette. Nel capitolo sui richiami di matematica l'avevamo definita nella
sezione sulla teoria dell'informazione, promettendo di riprenderla numeri alla
mano: eccoci.

`````{tab} Elementare

Il conto si fa in tre mosse, e sulla nostra frase si può seguire tutto a mano.

**Prima mossa: moltiplicare le scommesse.** Sono quelle appena calcolate,
$1 \times 0{,}75 \times \frac{2}{3} \times 0{,}5 \times 1 \times 0{,}5 \times
1 = 0{,}125$, cioè un ottavo. Sono sette fattori, uno per ogni scommessa: le
sei parole della frase più la chiusura, che è una scommessa anche lei (il
modello deve decidere che la frase finisce lì). L'apertura invece non si conta,
perché il segnale di inizio non lo indovina nessuno: c'è e basta.

**Seconda mossa: capovolgere.** Uno diviso un ottavo fa 8. Questo è quanto il
modello ha «esitato» sull'intera frase, ma dipende da quanto la frase è lunga:
una frase doppia esita di più anche se il modello è identico, quindi così non
si possono confrontare due frasi diverse.

**Terza mossa: riportare alla singola scommessa.** Le scommesse erano sette:
cerchiamo allora il numero che, moltiplicato per sé stesso sette volte, dà 8.
È la radice settima di 8, parente stretta della radice quadrata che conoscete,
e vale circa 1,35. Ecco la perplessità: un dado da 1,35 facce, cioè quasi
nessun dubbio.

`````

`````{tab} Superiore

Su una sequenza di $N$ token la perplessità è

$$
\mathrm{PP} = P(w_1, \dots, w_N)^{-1/N},
$$

cioè l'inverso della probabilità del testo, riportato «per token» dalla media
geometrica: è la stessa quantità $2^H$ della teoria dell'informazione, con $H$
la cross-entropia media per token, solo riscritta.

Un avvertimento sul conto di $N$, perché è il punto esatto in cui si sbaglia.
Se le frasi si incorniciano con `<s>` e `</s>`, come qui, allora anche `</s>` è
una scommessa e va contato fra gli $N$; `<s>` invece no, perché non lo si
predice mai. Sulla frase di sei parole qui sotto gli $N$ sono quindi **sette**,
non sei, ed è la convenzione di Jurafsky e Martin {cite}`jurafsky2026speech`
oltre che quella del codice a fine sezione. La ragione di escludere `<s>` è che
al marcatore di fine frase segue il marcatore di inizio con probabilità quasi
1: contare quella transizione fittizia gonfierebbe artificialmente il
punteggio.

`````

```{figure} ../figures/perplexity-sorpresa-modello.svg
:name: fig-perplessita-frase
:alt: "Una frase scritta token per token, e sopra ciascun token la probabilità che il modello gli aveva assegnato: barre alte sulle parole prevedibili, barre basse su quelle sorprendenti. In fondo le singole sorprese si combinano nella perplessità dell'intera frase."
:width: 96%

La perplessità nasce parola per parola. Il numero finale è la media
geometrica delle sorprese, e nessuna singola parola lo determina da sola.
```

{numref}`fig-perplessita-frase` mostra anche il limite della misura. Le parole
rare, che sono quelle su cui si gioca la qualità di un testo, sono anche le
più rare: entrano nella media poche volte, e un modello può avere una buona
perplessità complessiva inciampando proprio lì.

Lo stesso conto, scritto in una riga. Il bigramma MLE dava alla frase «il
gatto nero salta sul muro» probabilità $0{,}125$ in 7 scommesse (sei parole
più la chiusura di frase):

$$
\mathrm{PP} = 0{,}125^{-1/7} = 8^{1/7} \approx 1{,}35.
$$

Un dado da 1,35 facce: quasi nessun dubbio. Troppo bello per essere onesto, e
infatti stiamo barando: abbiamo valutato il modello *sulla frase con cui
l'abbiamo addestrato*, che è la prima del corpus. La perplessità va sempre
misurata su un testo di test, mai visto in addestramento; e lì, senza
smoothing, basta un bigramma nuovo per mandarla a infinito.

Rifacciamo dunque il confronto come si deve, con il modello lisciato alla
Laplace e su tre frasi che il modello non ha mai letto. Il codice a fine
sezione troverà perplessità intorno a **5,5** su «il gatto nero salta sul
divano» (frase nuova, ma tutta fatta di coppie già viste: proprio ciò che si
intende con «in stile»), **7,0** su «il cane nero salta sul divano», che
contiene una coppia mai vista, e oltre **14** sulle stesse parole rimescolate a
caso. L'ordine delle parole, finalmente, conta, ed è il confronto fra la
seconda e la terza a dirlo con più forza: stesse identiche parole, perplessità
doppia.

Per le grandezze reali, nell'esperimento classico riportato da Jurafsky e
Martin {cite}`jurafsky2026speech` su 38 milioni di parole del *Wall Street
Journal*, la perplessità scende da circa 960 con l'**unigramma** (il modello
che scommette guardando solo quanto una parola è comune, senza nemmeno
l'ultima parola letta: è il gradino sotto il bigramma) a 170 col bigramma e a
circa 110 col trigramma. I modelli neurali che incontreremo faranno molto
meglio, ma sulla stessa pagella.

## La passeggiata del bigramma

Un modello che assegna probabilità sa anche **generare**: si parte dal segnale
di inizio frase, si tira il dado (truccato secondo i conteggi) per scegliere
la prima parola, poi si riparte da quella e si tira ancora, fino al segnale di
fine. È esattamente il gioco dei libri sfogliati a caso di Shannon,
automatizzato. Sul nostro corpus di tre frasi la passeggiata produce cose come
«il cane guarda il cane guarda il gatto nero»: ogni passo è impeccabile (tutte
coppie viste nel corpus) ma la frase gira in tondo. Su corpora veri l'effetto
è identico, solo più elegante: come nelle approssimazioni di Shannon, il testo
*suona giusto da vicino ed è sconnesso da lontano*.

`````{tab} Elementare

La nostra scommettitrice ha una memoria da pesce rosso: quando sceglie la
parola nuova, ricorda solo l'ultima scritta (o le ultime due). È come
attraversare una città chiedendo indicazioni a un passante diverso a ogni
incrocio, senza mai dire da dove sei partito: ogni singolo consiglio è
ragionevole, il percorso complessivo non porta da nessuna parte.

Allungare la memoria sembrerebbe facile: invece di coppie, terne; invece di
terne, quaterne… Ma il quaderno esplode: per ogni parola in più da ricordare,
le pagine si moltiplicano per tutto il vocabolario, e quasi tutte resterebbero
bianche (combinazioni mai viste nemmeno in una biblioteca). E c'è un difetto
più sottile: per il quaderno «gatto» e «micio» sono estranei totali. Aver
letto mille volte «il gatto dorme» non lo aiuta di un grammo a scommettere su
«il micio dorme». Servirebbe un modello che capisca che parole *simili*
meritano scommesse *simili*, ed è proprio quello che sanno fare le reti
neurali della prossima sezione.

`````

`````{tab} Superiore

I limiti del modello n-gram sono strutturali, non di taratura:

- **Crescita esponenziale dei parametri.** Gli n-gram possibili sono
  $|V|^{\,n}$ (i contesti sono $|V|^{\,n-1}$, ciascuno con $|V|$
  continuazioni): con un vocabolario di 50.000 parole, i bigrammi possibili
  sono $2{,}5 \times 10^9$ e i trigrammi $1{,}25 \times 10^{14}$. Oltre
  $n = 4$ o $5$, nessun corpus basta: la sparsità vince su qualunque
  smoothing.
- **Nessuna generalizzazione tra parole simili.** Gli n-gram vivono nello
  spazio dei simboli discreti: $C(\text{il gatto dorme})$ non trasferisce
  nulla a $P(\text{dorme} \mid \text{il micio})$, perché «gatto» e «micio»
  sono ID distinti senza geometria. Gli **embedding** della sezione su come
  rappresentare il testo risolvono esattamente questo, collocando le parole
  in $\mathbb{R}^d$ dove la similarità è misurabile.

La sintesi delle due cure è il modello di linguaggio neurale proposto da
Yoshua Bengio e colleghi nel 2003 (embedding più rete feed-forward su una
finestra fissa) e soprattutto le **RNN** della prossima sezione, il cui stato
nascosto riassume in un vettore di dimensione fissa *tutto* il prefisso, non
le ultime $n-1$ parole. La scommessa resta identica, $P(w_t \mid w_{<t})$,
addestrata con la stessa cross-entropia e valutata con la stessa perplessità:
cambia solo quanta memoria porta con sé lo scommettitore. Portata all'estremo
(con i Transformer) questa identica scommessa diventerà GPT
{cite}`brown2020language`.

`````

## Un bigramma in trenta righe di Python

Tutto ciò che serve è contare. Il codice che segue costruisce il bigramma sul
corpus di tre frasi, con le probabilità MLE e Laplace, la generazione per
campionamento e la perplessità, e gira così com'è, senza librerie esterne
oltre alla libreria standard:

```python
import math
import random
from collections import Counter, defaultdict

corpus = [
    "il gatto nero salta sul muro",
    "il gatto bianco dorme sul divano",
    "il cane guarda il gatto nero",
]

INIZIO, FINE = "<s>", "</s>"

# 1. Conteggi: conta[w1][w2] = quante volte w2 segue w1
conta = defaultdict(Counter)
for frase in corpus:
    parole = [INIZIO] + frase.split() + [FINE]
    for w1, w2 in zip(parole, parole[1:]):
        conta[w1][w2] += 1

vocabolario = {w for frase in corpus for w in frase.split()} | {FINE}
V = len(vocabolario)                      # 12: 11 parole + </s>

# 2. Probabilita': massima verosimiglianza e Laplace
def p_mle(w1, w2):
    tot = sum(conta[w1].values())
    return conta[w1][w2] / tot if tot else 0.0

def p_laplace(w1, w2):
    return (conta[w1][w2] + 1) / (sum(conta[w1].values()) + V)

print(p_mle("il", "gatto"))       # 0.75
print(p_mle("cane", "nero"))      # 0.0 -> lo zero che azzera tutto
print(p_laplace("cane", "nero"))  # 0.0769... -> piccola ma viva

# 3. Generazione: una passeggiata di scommesse da <s> a </s>
def genera(seme):
    rng = random.Random(seme)
    parola, frase = INIZIO, []
    while len(frase) < 20:
        seguiti = conta[parola]
        parola = rng.choices(list(seguiti), weights=seguiti.values())[0]
        if parola == FINE:
            break
        frase.append(parola)
    return " ".join(frase)

for seme in range(3):
    print(genera(seme))
# il cane guarda il gatto nero
# il cane guarda il gatto nero
# il cane guarda il cane guarda il gatto nero

# 4. Perplessita' di una frase secondo il modello lisciato
def perplessita(frase):
    parole = [INIZIO] + frase.split() + [FINE]
    log2p = sum(math.log2(p_laplace(w1, w2))
                for w1, w2 in zip(parole, parole[1:]))
    return 2 ** (-log2p / (len(parole) - 1))

# nessuna delle tre e' nel corpus di addestramento: si valuta su testo nuovo
print(perplessita("il gatto nero salta sul divano"))  # ~5.5  tutte coppie viste
print(perplessita("il cane nero salta sul divano"))   # ~7.0  una coppia mai vista
print(perplessita("muro sul salta nero gatto il"))    # ~14.2 stesse parole, altro ordine
```

Vale la pena soffermarsi sulle uscite. La generazione con il seme 2 inciampa
nell'anello «il cane guarda il cane guarda…»: a ogni passo il bigramma vede
solo l'ultima parola, e da «guarda» si torna legittimamente a «il». E le tre
perplessità raccontano la storia giusta, tutte e tre su frasi che il modello
non ha mai letto: bassa per la frase in stile, più alta per quella con il
bigramma mai visto, più che doppia per le stesse parole rimescolate. Con la
matita al posto di Python, sono i conti che Markov fece nel 1913.

## Gli n-gram non sono morti

Sarebbe facile chiudere con «poi arrivarono le reti neurali e gli n-gram
finirono in soffitta». Non è andata così, e l'onestà storica impone di dirlo.
Contare è imbattibilmente *economico*: l'addestramento è una sola passata sul
corpus, la scommessa un accesso a una tabella (niente GPU, niente gradienti).
Nel 2006 Google distribuì i conteggi fino ai 5-grammi estratti da circa mille
miliardi di parole di web: modelli giganteschi costruiti, in fondo, con la
matita di Markov. Per anni la barra dei suggerimenti delle tastiere dei
telefoni è stata proprio questo (un n-gram con smoothing, piccolo e veloce
abbastanza da girare sul dispositivo) e solo di recente le reti neurali
compatte l'hanno affiancata o sostituita. E nel riconoscimento vocale, come
vedremo nel capitolo sullo Speech Recognition, un modello di linguaggio si
fonde ancora col modello acustico per scegliere fra trascrizioni identiche
all'orecchio («l'ago» o «lago») e per anni quel correttore silenzioso è stato
un n-gram alla Kneser–Ney. Quando serve una probabilità *subito*, su hardware
qualunque, contare resta un'ottima idea.

Ma il soffitto degli n-gram è quello che abbiamo toccato con mano: memoria
corta per costruzione, e nessuna nozione del fatto che «gatto» e «micio» si
somiglino. La prossima sezione riparte esattamente da qui: la stessa
scommessa sulla parola successiva, affidata però a una rete che porta con
sé, parola dopo parola, un riassunto dell'intera frase.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un **modello di linguaggio** scommette su quale parola viene dopo, e la
  probabilità di una frase intera è il prodotto di tutte quelle scommesse in
  fila. L'idea nasce con Markov, che nel 1913 conta a matita le lettere
  dell'*Onegin*, e con le «approssimazioni» di Shannon del 1948, costruite
  sfogliando libri a caso.
- Il patto degli **n-gram**: fingere che conti solo l'ultima parola letta
  (le coppie, il bigramma) o le ultime due (le terne, il trigramma), perché
  una frase intera non si ripete quasi mai e di lei non si potrebbe contare
  niente. Le scommesse stanno in un quaderno, una pagina per parola, con
  sopra le parole che l'hanno seguita e quante volte.
- Una coppia mai vista vale zero, e uno zero azzera l'intera frase: il modello
  confonde «mai visto» con «impossibile». Il rimedio più semplice è la
  **regola del $+1$** (un conteggio regalato a tutti), ma con un vocabolario
  vero diventa una patrimoniale che consegna quasi tutto ai fantasmi. Meglio
  **mescolare** il giudizio della coppia con quello della parola singola, o
  **ripiegare** sulla seconda quando la prima manca; e meglio ancora, con
  **Kneser–Ney**, chiedersi non quante volte una parola è comparsa ma in
  quanti posti diversi («Francisco» è frequentissimo ma non va da nessuna
  parte senza «San»).
- La **perplessità** è la pagella: il numero di facce del dado con cui il
  modello esita a ogni scommessa, e più è basso meglio scommette. Va misurata
  su testo mai letto in addestramento, altrimenti si sta barando.
- Un n-gram sa anche **generare**, tirando il suo dado truccato una parola
  alla volta: il risultato suona giusto da vicino e non porta da nessuna
  parte da lontano, perché la memoria è da pesce rosso. Allungarla riempie il
  quaderno di pagine bianche, e comunque per il quaderno «gatto» e «micio»
  restano due estranei: sono le due ragioni che portano alle reti della
  prossima sezione.
- Gli n-gram **non sono morti**: le tastiere che suggeriscono la parola, la
  spalla del riconoscimento vocale, un metro di paragone velocissimo che gira
  su qualunque computer, senza GPU.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un **modello di linguaggio** assegna una probabilità a una frase
  scomponendola, con la regola della catena, in scommesse sulla parola
  successiva; l'idea nasce con i conteggi a mano di Markov sull'*Onegin*
  (1913) e con le «approssimazioni» di Shannon (1948).
- L'**assunzione di Markov** tronca la storia alle ultime $n-1$ parole:
  bigrammi, trigrammi. La stima **MLE** è un rapporto di conteggi:
  $C(w_{t-1} w_t) / C(w_{t-1})$.
- I conteggi zero confondono «mai visto» con «impossibile»: servono gli
  **smoothing**. Il +1 di Laplace è semplice ma sposta troppa massa;
  interpolazione e backoff mescolano gli ordini; **Kneser–Ney** ripiega sui
  *contesti distinti* («Francisco» è frequente ma vive solo dopo «San») ed è
  rimasto lo standard fino all'era neurale.
- La **perplessità** (il $2^H$ della teoria dell'informazione) è la pagella:
  va misurata su testo di test, mai su quello di addestramento.
- Un n-gram **genera** testo campionando scommessa dopo scommessa: giusto da
  vicino, sconnesso da lontano. I limiti sono strutturali: contesti
  $|V|^{\,n-1}$ e nessuna generalizzazione tra parole simili; le ragioni che
  portano alle RNN della prossima sezione.
- Gli n-gram **non sono morti**: tastiere predittive, fusione col modello
  acustico nell'ASR, baseline velocissime senza GPU.
```
`````
