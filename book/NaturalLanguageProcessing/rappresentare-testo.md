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

## Vettori densi: i word embedding

Il salto concettuale arriva nel 2013. L'idea guida è vecchia, il linguista
John Firth nel 1957 la riassunse così: *"You shall know a word by the company
it keeps"*, conoscerai una parola dalla compagnia che frequenta. Parole che
appaiono in contesti simili hanno significati simili. Se lo facciamo dire ai
numeri, otteniamo gli **word embedding**.

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
ricche le forme flesse di una stessa radice condividono parametri (Bojanowski
et al). misurano i guadagni maggiori proprio su lingue molto flessive come
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
```
