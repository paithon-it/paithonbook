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

Non sempre basta. La parola *tokenizzazione* è rara: conviene spezzarla in
pezzi più piccoli e frequenti, come `token` + `izzazione`, che il modello ha
già visto tante volte altrove. Così anche una parola mai incontrata prima si
ricostruisce dai suoi mattoncini.

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

Abbiamo i token. Il modo più ingenuo di dar loro un numero è la codifica
**one-hot**.

```{figure} ../figures/one-hot-label-encoding.svg
:name: fig-one-hot
:alt: "La stessa colonna di categorie codificata in due modi. Con l'ordinal encoding diventa una sola colonna di interi, che però introduce un ordine e delle distanze fra categorie che non ne hanno. Con il one-hot diventa una colonna per categoria, riempita di zeri e con un solo uno: tutte le categorie restano equidistanti."
:width: 96%

Due codifiche, due significati impliciti. Gli interi dicono che una categoria
viene prima di un'altra e dista il doppio da una terza; il one-hot non dice
niente di tutto questo, ed è il suo pregio.
```

Il confronto di {numref}`fig-one-hot` spiega perché si accetti lo spreco. Una
colonna per parola è un'enormità con un vocabolario da decine di migliaia di
voci; ma la codifica compatta introdurrebbe un ordine inventato, e un modello
che vede numeri cerca sempre relazioni fra numeri, comprese quelle che nessuno
intendeva metterci.

`````{tab} Elementare

Immagina una lunghissima pulsantiera con un interruttore per ogni parola del
vocabolario. Per rappresentare *gatto* accendi solo il suo interruttore e
lasci spenti tutti gli altri: un vettore lunghissimo, tutto zeri tranne un
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

Per rappresentare un intero documento (non una sola parola) sommiamo i token
in un **sacchetto di parole** (*bag-of-words*): buttiamo via l'ordine e
teniamo solo le frequenze.

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
perdita c'è, ed è totale: nessun modello a valle potrà recuperarla.

`````{tab} Elementare

Il documento diventa un conteggio: quante volte compare ciascuna parola. Ma
c'è un problema: articoli e preposizioni come *il* o *di* compaiono ovunque,
e proprio per questo dicono poco su *cosa* parla il testo. Parole rare come
*retina* o *sinapsi* sono molto più rivelatrici.

Il peso **TF-IDF** corregge lo squilibrio: gonfia le parole rare e
informative, sgonfia quelle comuni a tutti i documenti. È ancora una
pulsantiera sparsa, ma con volumi tarati meglio.

`````

`````{tab} Superiore

Il *bag-of-words* rappresenta un documento $d$ con il vettore dei conteggi
$\in \mathbb{R}^{|V|}$. Il peso **TF-IDF** (*Term Frequency – Inverse
Document Frequency*) di un termine $t$ è

$$
\text{tfidf}(t, d) = \text{tf}(t, d)\cdot \log\frac{N}{\text{df}(t)} ,
$$

dove $\text{tf}(t,d)$ è la frequenza di $t$ in $d$, $N$ il numero totale di
documenti e $\text{df}(t)$ il numero di documenti che contengono $t$. Il
fattore logaritmico penalizza i termini onnipresenti (df alto). Restano due
limiti strutturali: i vettori sono ancora sparsi e $|V|$-dimensionali, e
nessuna relazione lega parole diverse tra loro.

`````

In `scikit-learn` tutto questo è una manciata di righe:

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
manuale»: la libreria ne usa una variante, per ragioni pratiche. La sostanza
non cambia (le parole rare pesano più di quelle comuni), i decimali sì.

`````{tab} Elementare

La variante serve a due cose. La prima è non trovarsi mai a dividere per zero
quando una parola compare in pochissimi documenti. La seconda è mettere sulla
stessa scala documenti di lunghezza diversa, così che un testo lungo non
risulti più «pesante» solo perché contiene più parole. Il risultato è che
anche le parole presenti in tutti i documenti si portano dietro un po' di
peso, invece di sparire del tutto: poco male, perché quello che conta è che le
parole rare ne abbiano di più.

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
numeri, otteniamo gli **word embedding**.

```{figure} ../figures/word2vec-parola-dal-contesto.svg
:name: fig-finestra-contesto
:alt: "Una finestra scorre lungo una frase, evidenziando una parola centrale e le parole che la circondano a destra e a sinistra. Dall'insieme dei contesti raccolti scorrendo tutto il testo si ricava il vettore della parola centrale; parole che ricorrono in contesti simili finiscono con vettori simili."
:width: 94%

La frase di Firth resa procedura. Nessuno dice al modello cosa significhi una
parola: gli si fa vedere in quali compagnie compare, migliaia di volte, e il
vettore è il riassunto di quelle compagnie.
```

Vale la pena notare cosa {numref}`fig-finestra-contesto` non usa: nessun
dizionario, nessuna annotazione, nessuna persona che spieghi il significato.
Serve solo del testo, e questo è il motivo per cui gli embedding si poterono
addestrare su corpora enormi quando le risorse annotate erano scarse.

`````{tab} Elementare

Invece di migliaia di zeri con un solo uno, ogni parola diventa una corta
lista di poche centinaia di numeri, tutti "pieni". Questi numeri non li
scegliamo a mano: li impara un modello leggendo montagne di testo e notando
quali parole si accompagnano.

Il risultato è una **mappa del significato**. Su questa mappa *gatto* e
*felino* finiscono vicini, *gatto* e *cane* poco più lontani, *gatto* e
*mercoledì* agli antipodi. La vicinanza geometrica diventa vicinanza di
senso.

`````

`````{tab} Superiore

Un embedding è una funzione che associa a ogni token un vettore **denso**
$\mathbf{v} \in \mathbb{R}^{d}$ con $d$ piccolo (tipicamente $100$–$300$),
appreso dai dati. **word2vec** {cite}`mikolov2013efficient` addestra una rete
poco profonda a predire il contesto data la parola (*skip-gram*) o viceversa
(*CBOW*); **GloVe** {cite}`pennington2014glove` fattorizza invece la matrice
globale di co-occorrenza. Da $\mathbb{R}^{|V|}$ sparso si passa a
$\mathbb{R}^{d}$ denso: meno dimensioni, ma cariche di struttura semantica.

`````

## Sotto la parola: fastText

word2vec e GloVe hanno però un punto cieco: assegnano un vettore a ogni parola
*intera*. Se una parola non compare mai nel corpus di addestramento, per lei
non c'è nessun vettore; e una lingua ricca di flessioni come l'italiano
(*gatto*, *gatta*, *gatti*, *gattino*), moltiplica le forme da imparare una
per una. **fastText** {cite}`bojanowski2017enriching` risolve entrambi i
problemi con una mossa sola: scendere sotto il livello della parola.

`````{tab} Elementare

Immagina di scomporre ogni parola in mattoncini di poche lettere che si
sovrappongono. Con mattoncini da tre lettere, *gatto* (scritto con i suoi
confini, `<gatto>`), diventa `<ga`, `gat`, `att`, `tto`, `to>`. Ogni
mattoncino ha il proprio vettore, e il vettore di *gatto* è semplicemente la
**somma** dei vettori dei suoi mattoncini.

I vantaggi sono due, e concreti. Primo: *gatto*, *gatta* e *gattino*
condividono i pezzi `gat` e `att`, quindi i loro vettori nascono già simili
(prezioso in una lingua di desinenze come la nostra). Secondo: anche una
parola mai vista prima ha comunque i suoi mattoncini, e quindi un vettore:
nessuna parola resta più senza rappresentazione.

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
:alt: Quattro punti (uomo, re, donna, regina) in un piano; le frecce che li collegano formano un parallelogramma, con l'offset "regalità" e l'offset "femminile" ripetuti su entrambi i lati.
:width: 85%

I quattro embedding formano un parallelogramma: la stessa freccia
"regalità" separa *uomo* da *re* e *donna* da *regina*; la stessa freccia
"femminile" separa *uomo* da *donna* e *re* da *regina*.
```

`````{tab} Elementare

Guarda la {numref}`fig-embedding-analogia`. La freccia che va da *uomo* a *re*
significa più o meno "diventare regale". Se prendi quella stessa freccia e la
applichi a *donna*, dove atterri? Molto vicino a *regina*. In formula:
*re − uomo + donna ≈ regina*. I quattro punti disegnano un parallelogramma, e
questo è il segno che il modello ha catturato da solo il concetto di "regalità"
e quello di "genere", senza che nessuno glieli abbia mai spiegati.

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

Non è magia e non è perfetta: molte analogie falliscono, e questi vettori
ereditano i **pregiudizi** dei testi su cui sono addestrati (per esempio
associazioni di genere a certi mestieri). Ne parleremo, ma il messaggio resta:
il significato, ridotto a geometria, si lascia misurare con un prodotto
scalare.

`````

## Dalla parola alla frase: gli embedding di frase

Tutto quello che abbiamo costruito finora dà un vettore per **parola**. Ma
quasi tutto ciò che si vuole fare davvero riguarda testi interi: trovare i
documenti che rispondono a una domanda, accorgersi che due segnalazioni sono
la stessa, raggruppare le recensioni per tema, dare a un modello linguistico i
passaggi giusti da leggere. Serve un vettore per **frase**, e ottenerlo non è
altrettanto ovvio.

`````{tab} Elementare

La prima idea che viene in mente è anche la più ragionevole: prendere i vettori
delle parole della frase e farne la media. Funziona sorprendentemente bene come
punto di partenza, e ha due difetti che si vedono subito.

Il primo è che **l'ordine sparisce**. «Il cane morde l'uomo» e «l'uomo morde il
cane» contengono le stesse parole, quindi hanno la stessa media, quindi per la
macchina sono la stessa frase. Il secondo è che le parole piccole pesano quanto
le altre, e certe parole piccole ribaltano tutto: «il film mi è piaciuto» e «il
film non mi è piaciuto» differiscono per un «non» che nella media si perde.

Con i Transformer il problema sembra risolto, perché quei modelli l'ordine lo
tengono. Ma c'è una sorpresa: **un BERT preso così com'è dà vettori di frase
mediocri**. Il motivo è semplice e vale la pena farci caso, perché è una
lezione generale: BERT è stato addestrato a indovinare parole mancanti, non a
mettere vicine due frasi che vogliono dire la stessa cosa. Nessuno gli ha mai
chiesto di farlo, e infatti non lo fa bene. Se vuoi che uno spazio abbia una
certa proprietà, quella proprietà devi addestrarla.

Da qui l'idea, che è vecchia e bellissima: invece di insegnare al modello
*che cosa* è una frase, gli si insegna **quali frasi vanno vicine**. Gli si
mostrano triplette, un'ancora, una frase che vuol dire la stessa cosa e una che
vuol dire altro, e gli si chiede una cosa sola: fai in modo che la prima
distanza sia minore della seconda. Ripetuto su milioni di triplette, lo spazio
si riorganizza da solo, e alla fine «vicino» significa «di argomento simile».

La rete che fa questo lavoro si chiama **siamese** perché è una sola rete usata
più volte: la stessa identica, con gli stessi pesi, applicata all'ancora, alla
frase simile e a quella diversa. È essenziale che sia la stessa, altrimenti i
vettori finirebbero in spazi diversi e confrontarli non vorrebbe dire niente.

Il guadagno pratico è enorme, e gli autori di Sentence-BERT lo misurano:
trovare la coppia più simile fra diecimila frasi richiede circa **65 ore** se
per ogni coppia si deve far girare un BERT sulle due frasi insieme, e circa
**cinque secondi** se ogni frase ha già il suo vettore e basta confrontare
numeri. È la differenza fra un'idea e un prodotto.

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

La **triplet loss** lavora su terne $(a, p, n)$, ancora, positivo, negativo, e
chiede una disuguaglianza **relativa**:

$$
\mathcal{L} = \max\big(0,\; m + d(a, p) - d(a, n)\big),
$$

cioè «il positivo deve stare più vicino del negativo, e di almeno $m$». È più
robusta della contrastive perché non impone distanze assolute, che sarebbero
arbitrarie, ma solo un ordinamento.

La **multiple negatives ranking loss** (o InfoNCE, la stessa forma incontrata
in *Imparare senza etichette* per SimCLR e in *Allineare due spazi* per CLIP)
usa come negativi tutti gli altri elementi del batch:

$$
\mathcal{L} = -\log \frac{\exp(\mathrm{sim}(a, p)/\tau)}
{\sum_{j} \exp(\mathrm{sim}(a, p_j)/\tau)} .
$$

È oggi la scelta prevalente, perché un batch da 1024 fornisce 1023 negativi
gratis a ogni esempio, ed è precisamente la ricetta degli *in-batch negatives*
che il capitolo su RAG attribuisce a DPR {cite}`karpukhin2020dense`.

Due avvertenze pratiche che separano un modello che funziona da uno che no.
La prima è la **scelta dei negativi**: quelli presi a caso diventano presto
banali (due testi su argomenti scorrelati sono facilissimi da separare, e la
loss va a zero senza aver insegnato niente), per cui si passa ai *hard
negatives*, cercati apposta fra i quasi-simili, tipicamente scavandoli con un
modello precedente. La seconda è il rischio opposto, il **collasso**: nulla
vieta alla rete di mandare tutte le frasi di un argomento esattamente nello
stesso punto, che azzera la loss e distrugge ogni distinzione fine.
Temperatura, margine e regolarizzazione servono a governare quel compromesso.

Infine una nota che chiude il cerchio con il capitolo su RAG: la similarità del
coseno misura **«si somigliano»**, non **«questo risponde a quella»**. Una
domanda e la sua risposta spesso non si somigliano affatto, e infatti si
addestrano due torri distinte, $E_q$ ed $E_p$, con positivi che sono coppie
domanda-passaggio e non coppie di parafrasi. Cambia il compito, cambiano i
positivi, cambia lo spazio.

`````

Che l'addestramento *riorganizzi lo spazio* si può guardare da vicino su
vettori finti, senza scaricare nessun modello: partiamo da coordinate casuali,
in cui gli argomenti non sono affatto separati, e lasciamo lavorare una triplet
loss.

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

All'inizio il distacco è $-0{,}004$: dentro e fuori si somigliano allo stesso
modo, cioè lo spazio non sa niente degli argomenti, ed è esattamente la
situazione di un encoder mai addestrato alla somiglianza. Dopo quattrocento
passi il coseno interno sale a $+0{,}876$ e quello esterno scende a $-0{,}261$.
Nessuno ha detto alla rete quali fossero gli argomenti né dove metterli: le
sono state date solo delle terne e una disuguaglianza da rispettare, e la
geometria si è riorganizzata da sé.

Vale la pena guardare anche il rovescio: $0{,}876$ **dentro** l'argomento è
tanto, e siamo vicini al caso in cui le frasi di uno stesso tema collassano
tutte nello stesso punto. Andrebbe benissimo per separare quattro temi, e
malissimo per distinguere due sfumature dentro lo stesso tema. È il compromesso
che in un modello vero si governa con negativi difficili e temperatura, e la
ragione per cui un modello di embedding va scelto guardando il compito, non la
classifica.

```{admonition} Da ricordare
:class: important
- **Tokenizzare** spezza il testo in unità; i sistemi moderni usano token
  *sottoparola* per gestire qualunque parola.
- **One-hot** e **bag-of-words / TF-IDF** danno vettori enormi, sparsi e senza
  nozione di somiglianza tra parole diverse.
- Gli **word embedding** (word2vec, GloVe) sono densi e a bassa dimensione: la
  **vicinanza geometrica riflette la vicinanza di significato**, misurata con
  la **similarità del coseno**.
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
