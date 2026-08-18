# Quando non c'è una risposta giusta: valutare un raggruppamento

Che Plutone non sia un pianeta lo ha deciso un'alzata di mano. Successe
all'assemblea generale dell'Unione Astronomica Internazionale, a Praga
nell'estate del 2006: la risoluzione fissava tre criteri (orbitare attorno al
Sole, essere abbastanza massiccio da essersi fatto tondo da sé, e aver ripulito
la propria orbita dagli altri corpi) e Plutone cadeva sul terzo.

La cosa da notare non è il verdetto: è che ci sia voluta una **votazione**. Le
misure erano note a tutti e nessuno le contestava; a mancare era il criterio con
cui raggruppare gli oggetti del sistema solare in famiglie, perché di criteri
ragionevoli ce n'era più d'uno e portavano a risposte diverse. Tenendo i tre
criteri, i pianeti sono otto; togliendo il terzo diventano tredici o più.
Nessuna delle due tassonomie è sbagliata: sono due modi di tagliare la stessa
collezione, e a scegliere è stata una comunità, non un dato.

La sezione precedente ha costruito quattro modi di raggruppare dei punti e li ha
lasciati lì, ciascuno con i suoi gruppi. Questa affronta la domanda che viene
subito dopo, ed è più difficile di quanto sembri: **come si fa a sapere se un
raggruppamento è buono?** In un problema supervisionato bastava contare gli
errori. Qui non c'è niente da contare, perché la risposta giusta non esiste da
nessuna parte: non è stata persa, non c'è proprio.

## Due domande diverse, due famiglie di indici

`````{tab} Elementare

Immagina una biblioteca da riordinare. Un bibliotecario dispone i libri per
genere, un altro per epoca, un terzo per lingua. Tre scaffalature diverse, tre
raggruppamenti, e nessuno dei tre ha sbagliato: hanno risposto a domande diverse.

Ora immagina di dover giudicare il loro lavoro, e distingui bene i due casi in
cui ti puoi trovare.

**Primo caso: qualcuno ha già la risposta.** Sai che i libri andavano divisi per
genere, e vuoi sapere quanto la scaffalatura che hai davanti somiglia a quella
giusta. Qui c'è qualcosa da confrontare, e gli indici che servono si chiamano
**esterni** perché guardano fuori dal raggruppamento, verso una verità nota.
Succede più spesso di quanto si creda, e non perché qualcuno bari: capita ogni
volta che si prova un algoritmo su dati di cui l'etichetta esiste, tenendola
nascosta durante il lavoro e tirandola fuori solo per il voto.

Di indici esterni ne girano due, e conviene sapere come si chiamano perché si
trovano scritti così dappertutto, sigla e basta. L'**ARI** (*Adjusted Rand
Index*, indice di Rand aggiustato) guarda i libri **a due a due**: per ogni
coppia si chiede se le due scaffalature li hanno messi insieme, o separati, o se
sono in disaccordo, e tiene il conto. L'**NMI** fa la stessa domanda dal versante
dell'informazione: sapendo su che ripiano sta un libro nella prima scaffalatura,
quanto si è già indovinato di dove sta nella seconda. Tutti e due valgono $1$ per
due scaffalature identiche e $0$ per due che non c'entrano niente, e nessuno dei
due si lascia ingannare dai nomi dei ripiani, perché il «ripiano 1» di uno e il
«ripiano 1» dell'altro non hanno niente a che vedere.

**Secondo caso, quello vero: nessuno ha la risposta.** Sono clienti, o cellule,
o documenti, e a raggrupparli si è per la prima volta. Qui non si può misurare
la somiglianza con niente, e tutto quello che si può fare è guardare la
scaffalatura in sé: i libri sullo stesso ripiano si somigliano fra loro? I
ripiani sono ben distinti l'uno dall'altro? Gli indici che fanno questo si
chiamano **interni**, e il più noto è la **silhouette**, già incontrata nella
sezione precedente: per ogni punto chiede se sta più vicino ai compagni del suo
gruppo che agli estranei del gruppo accanto, e poi fa la media su tutti.

C'è però una trappola, e conviene saperla prima di aver preso una decisione
sbagliata: un indice interno non misura se il raggruppamento è **giusto**.
Misura se è **compatto e ben separato**, che è un'altra cosa. Se i gruppi veri
hanno forme strane, un indice interno preferisce la risposta sbagliata, e lo fa
con convinzione. Il conto qui sotto lo mostra sul caso più semplice possibile.

`````

`````{tab} Superiore

Gli indici **interni** valutano una partizione usando solo $\mathbf{X}$ e le
etichette assegnate. Tutti quantificano una qualche forma di rapporto fra
coesione e separazione. La **silhouette** media è

$$
\bar{s} = \frac{1}{m}\sum_{i=1}^{m}
\frac{b_i - a_i}{\max(a_i,\, b_i)} ,
$$

dove $a_i$ è la distanza media del punto $i$ dagli altri punti del **suo**
gruppo (la coesione) e $b_i$ la distanza media dai punti del gruppo diverso più
vicino (la separazione): vale $+1$ per un punto molto meglio collocato dove sta,
$0$ sul confine, negativo per un punto che starebbe meglio altrove
{cite}`rousseeuw1987silhouettes`. Gli altri due indici usati di frequente sono
il **Calinski–Harabasz** {cite}`calinski1974dendrite`, rapporto fra la
dispersione fra i gruppi e quella dentro i gruppi, e il **Davies–Bouldin**
{cite}`davies1979cluster`, media della peggior somiglianza fra coppie di gruppi.

La cosa da tenere presente è che tutti e tre presuppongono una nozione di
«buono» che è **geometrica e centrata**: sono massimizzati da gruppi compatti,
convessi e ben spaziati. Su geometrie non convesse non misurano la qualità della
partizione, misurano quanto la partizione somiglia a quella che produrrebbe
$k$-means. Sono, in altre parole, indici allineati con l'ipotesi di un
particolare algoritmo, e usarli per scegliere fra algoritmi con ipotesi diverse
è un errore di metodo.

Gli indici **esterni** confrontano la partizione ottenuta $C$ con una nota $T$.
Il capostipite è l'indice di **Rand** {cite}`rand1971objective`: sulle
$\binom{m}{2}$ coppie di punti, la
frazione su cui le due partizioni sono d'accordo (stessa coppia insieme in
entrambe, o separata in entrambe),

$$
\mathrm{RI} = \frac{u + s}{\binom{m}{2}},
$$

con $u$ le coppie **unite** in entrambe e $s$ quelle **separate** in entrambe
(le lettere $a$ e $b$ sono già occupate dalla silhouette qui sopra). Ha un
difetto grave: **non vale zero sul caso nullo**, e la linea di base non è
nemmeno una costante: dipende da quanti gruppi hanno le due partizioni. Due
etichettature casuali concordano infatti su tutte le coppie che entrambe
separano, e più i gruppi sono fini più coppie separano, quindi $\mathrm{RI}$
sale verso $1$ per puro conteggio. Il rimedio è
l’**Adjusted Rand Index** di Hubert e Arabie {cite}`hubert1985comparing`, che
sottrae il valore atteso sotto un modello di permutazione casuale e normalizza:

$$
\mathrm{ARI} = \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}
                    {\max(\mathrm{RI}) - \mathbb{E}[\mathrm{RI}]},
$$

dove il massimo è quello raggiungibile **tenendo fisse** le taglie dei gruppi
delle due partizioni, e non $1$. L'indice vale $1$ per l'accordo perfetto, $0$
in media sul caso casuale, e può essere negativo per un accordo peggiore del
caso. L'alternativa dal versante
informazionale è l’**NMI**, informazione mutua fra le due partizioni normalizzata
dalle rispettive entropie, che ha lo stesso spirito e una diversa sensibilità al
numero di gruppi (tende a premiare le partizioni fini, e ha a sua volta una
versione aggiustata, l'AMI).

Tutti e due sono **invarianti alla permutazione delle etichette**, che è
indispensabile: il «gruppo 0» di un algoritmo e il «gruppo 0» di un altro non
hanno niente in comune, e a contare sono solo le coppie di punti messe insieme o
separate.

`````

## Il caso in cui l'indice interno boccia la risposta giusta

Il banco di prova sono le due lune della sezione precedente, dove già sappiamo
chi ha ragione: $k$-means taglia dritto e sbaglia, DBSCAN segue la densità e
azzecca. Vediamo cosa ne dicono gli indici.

```python
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             rand_score, silhouette_score)

X, vero = make_moons(n_samples=600, noise=0.06, random_state=0)
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
db = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)   # qui nessun punto va a rumore

print(f"{'':12}{'silhouette':>12}{'Rand':>8}{'ARI':>8}{'NMI':>8}")
for nome, e in (("k-means", km), ("DBSCAN", db)):
    print(f"{nome:12}{silhouette_score(X, e):12.3f}{rand_score(vero, e):8.3f}"
          f"{adjusted_rand_score(vero, e):8.3f}{normalized_mutual_info_score(vero, e):8.3f}")

# quanto vale il "niente"? due etichettature tirate a caso, confrontate fra loro
r = np.random.default_rng(0)
print("\ndue etichettature a caso, quanto si somigliano:")
for k in (2, 5, 20):
    grezzi, aggiustati = [], []
    for _ in range(200):
        a, b = r.integers(0, k, 600), r.integers(0, k, 600)
        grezzi.append(rand_score(a, b))
        aggiustati.append(adjusted_rand_score(a, b))
    print(f"  con {k:2d} gruppi ciascuna:  Rand {np.mean(grezzi):.3f}"
          f"   ARI {np.mean(aggiustati):+.4f}")
```

```text
              silhouette    Rand     ARI     NMI
k-means            0.486   0.628   0.255   0.194
DBSCAN             0.331   1.000   1.000   1.000

due etichettature a caso, quanto si somigliano:
  con  2 gruppi ciascuna:  Rand 0.500   ARI +0.0003
  con  5 gruppi ciascuna:  Rand 0.680   ARI -0.0003
  con 20 gruppi ciascuna:  Rand 0.905   ARI -0.0001
```

La prima colonna dice il contrario delle ultime due, ed è il punto della
sezione. Secondo ARI e NMI, che sanno qual è la risposta giusta, DBSCAN ha
ricostruito le due lune **alla perfezione**: valgono $1{,}000$, cioè la sua
partizione e quella vera sono la stessa. La **silhouette** dà a
DBSCAN $0{,}331$ e a $k$-means $0{,}486$: giudicando con lei si sceglierebbe il
metodo che ha sbagliato, e lo si sceglierebbe con un margine confortevole.

Non è un difetto della silhouette, è la sua definizione presa sul serio. Lei
chiede «ogni punto è più vicino ai suoi compagni che agli estranei?», e in una
luna intrecciata con l'altra la risposta è no: la punta di una luna ha vicini
dell'altra a un centimetro e compagni all'altro capo della curva. La silhouette
non sta misurando la partizione sbagliata, sta misurando una cosa diversa da
quella che ci interessa. **Un indice interno è una domanda geometrica, e va usato
solo quando è la domanda che ci si sta ponendo.**

Le tre righe in fondo riguardano l'altro indice, e sono la ragione per cui il
Rand grezzo non va usato. Sono due etichettature **tirate a caso**, cioè due
raggruppamenti che non contengono nessuna informazione, messi a confronto fra
loro: con due gruppi per parte il Rand dà $0{,}500$, con cinque $0{,}680$, con
venti $0{,}905$. Su una scala che arriva a $1$, il puro caso prende $0{,}9$
purché i gruppi siano abbastanza fini, e chi legge quel numero pensa di aver
quasi indovinato. La ragione è aritmetica: due partizioni fini separano quasi
tutte le coppie, e il Rand conta come «accordo» anche l'aver separato.

L'ARI sulle stesse tre righe vale $0{,}0000$, $-0{,}0003$, $-0{,}0001$: non si
muove. È esattamente lo stesso inganno dell'accuratezza su classi sbilanciate, e
la correzione è la stessa idea: sottrarre quello che si otterrebbe per caso, che
è ciò che l'aggettivo *aggiustato* significa.

## Senza risposta giusta: chiedere se il raggruppamento tiene

Resta il caso vero, quello in cui l'etichetta non c'è e un indice interno non
basta. C'è una terza via, e non misura la qualità: misura la **riproducibilità**.

`````{tab} Elementare

L'idea è quella del bibliotecario messo alla prova due volte. Dagli metà dei
libri, presi a caso, e fagli fare gli scaffali. Poi dagli un'altra metà, presa a
caso a parte, e faglieli rifare. Se il criterio che sta usando è davvero nei
libri, i due lavori diranno la stessa cosa sui libri capitati in tutte e due le
mani. Se invece si sta inventando le categorie, i due lavori saranno diversi.

Questo si può fare senza sapere niente della risposta giusta, e serve soprattutto
a decidere **quanti gruppi** cercare: il numero di gruppi buono è quello che
regge alla prova, mentre uno sbagliato produce partizioni che cambiano ogni
volta che si cambiano i dati. Il ricampionamento è quello del bootstrap della
sezione apposita, usato qui per una domanda diversa.

Attenzione a una trappola, che si vedrà nella tabella qui sotto: dividere in
**pochissimi** gruppi regge quasi sempre, anche quando è la risposta sbagliata,
perché un taglio grossolano viene uguale comunque. Il modo di accorgersene è
guardare non solo quanto le prove vanno d'accordo **in media**, ma anche se
vanno d'accordo **sempre**. La prova serve dunque a **scartare** i numeri che
non tengono, non a incoronare il più stabile.

`````

`````{tab} Superiore

La **stabilità** come criterio di selezione formalizza questo: si estraggono due
sottocampioni $S_1, S_2 \subset \mathbf{X}$, si adatta l'algoritmo a $k$ gruppi su
ciascuno, si predicono le etichette sui punti $S_1 \cap S_2$ e si misura
l'accordo fra le due assegnazioni con un indice esterno (l'ARI, appunto, perché
l'accordo fra due partizioni è esattamente ciò che misura). Ripetuto e mediato,
dà $\mathrm{stab}(k)$, e si sceglie il $k$ che la massimizza.

Il criterio ha una nota da conoscere prima di usarlo, ed è visibile nella
tabella qui sotto: i valori **piccoli** di $k$ sono stabili quasi per
costruzione, perché una bipartizione grossolana di dati ben separati esce sempre
uguale. La stabilità va quindi letta come un vincolo (scarta i $k$ instabili),
non come una funzione da massimizzare alla cieca.

`````

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

r = np.random.default_rng(0)
# quattro gruppi ben separati: la risposta giusta, che l'algoritmo non sa, e' 4
X = np.vstack([r.normal(c, 0.55, (200, 2)) for c in ([0, 0], [5, 0], [0, 5], [5, 5])])

def stabilita(X, k, prove=25, seme=0):
    """Due meta' dei dati prese a caso, due raggruppamenti: quanto sono d'accordo?"""
    rr = np.random.default_rng(seme)
    accordi = []
    for _ in range(prove):
        a = rr.permutation(len(X))[:len(X)//2]
        b = rr.permutation(len(X))[:len(X)//2]
        comuni = np.intersect1d(a, b)          # i punti capitati in tutt'e due
        ea = KMeans(k, n_init=10, random_state=0).fit(X[a]).predict(X[comuni])
        eb = KMeans(k, n_init=10, random_state=1).fit(X[b]).predict(X[comuni])
        accordi.append(adjusted_rand_score(ea, eb))
    return np.mean(accordi), np.std(accordi)   # la media da sola nasconde troppo

print(f"{'k':>3}{'silhouette':>12}{'stabilita':>12}{'(fra le prove)':>16}")
for k in range(2, 9):
    e = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    media, disp = stabilita(X, k)
    print(f"{k:3d}{silhouette_score(X, e):12.3f}{media:12.3f}"
          f"{'+/- ' + format(disp, '.3f'):>16}")
```

```text
  k  silhouette   stabilita  (fra le prove)
  2       0.508       0.960       +/- 0.197
  3       0.606       0.655       +/- 0.238
  4       0.795       1.000       +/- 0.000
  5       0.676       0.897       +/- 0.075
  6       0.550       0.777       +/- 0.083
  7       0.440       0.699       +/- 0.086
  8       0.318       0.662       +/- 0.099
```

Su dati che hanno davvero quattro gruppi ben separati, i due criteri concordano e
indicano $k = 4$: la silhouette con $0{,}795$, il suo massimo, e la stabilità con
$1{,}000$, che è il valore pieno e vuol dire che due metà indipendenti dei dati
hanno prodotto **esattamente** la stessa partizione sui punti in comune. Quando
la struttura c'è e ha la forma che $k$-means si aspetta, misurarla è facile e
ogni strumento la trova.

Vale la pena guardare anche la riga $k = 2$, che è la trappola annunciata:
stabilità $0{,}960$, quasi quanto quella del $k$ giusto. È qui che serve
l'ultima colonna, ed è la ragione per cui c'è. A $k = 4$ la stabilità è
$1{,}000$ con dispersione **zero**: venticinque prove su venticinque hanno dato
lo stesso identico risultato. A $k = 2$ la stessa media di $0{,}960$ arriva da
prove che ballano di $\pm 0{,}197$, cioè non è affatto un valore ripetibile: è
quasi sempre un accordo pieno e ogni tanto un disaccordo totale.

Il perché sta nella geometria. Quattro mucchi ai vertici di un quadrato si
possono tagliare in due in **due** modi che costano esattamente uguale, in
orizzontale o in verticale; quasi ogni volta le due metà dei dati scelgono lo
stesso, e ogni tanto no. La stabilità di un $k$ sbagliato può quindi essere
altissima per pura simmetria, e a smascherarla non è la media ma la sua
dispersione.

Ecco perché la stabilità serve a **scartare** i valori che non tengono (qui il
$3$, con $0{,}655 \pm 0{,}238$: un raggruppamento in tre parti di quattro mucchi
simmetrici deve decidere quali due unire, e ogni volta decide diversamente) e
non a scegliere il massimo assoluto senza guardare altro.

## Una parentesi sul nome: «non supervisionato»

Prima di tirare le somme conviene sistemare una faccenda di vocabolario, perché
il libro chiama questi metodi «non supervisionati» e altrove spiega perché quel
nome non andrebbe usato. La contraddizione è solo apparente, e scioglierla
serve a capire di che cosa parliamo.

Yann LeCun ha rinunciato pubblicamente all'espressione «apprendimento non
supervisionato», e la ragione, scritta con Ishan Misra nel 2021, è che quel nome
è mal definito e fuorviante {cite}`lecun2021darkmatter`: suggerisce che
l'apprendimento non usi supervisione affatto, mentre nei metodi che a lui
interessano (prevedere una parte del dato dal resto: la parola coperta in una
frase, il pezzo mancante di un'immagine) un segnale di correzione c'è eccome, ed
è molto più ricco di quello di un'etichetta. Quei metodi si chiamano
**auto-supervisionati**, e il libro dedica loro un capitolo intero.

Ma i metodi di questa sezione e della precedente non sono quelli. Quando
$k$-means sposta un centroide o la PCA cerca la direzione di massima varianza,
non c'è nessun bersaglio da indovinare, nessuna previsione confrontata con una
risposta: c'è una funzione obiettivo che descrive **la forma dei dati** e la si
ottimizza. Qui la supervisione manca davvero, e il nome tradizionale non inganna
nessuno.

La regola che il libro segue è dunque questa: «non supervisionato» per i metodi
che descrivono i dati senza prevederne nessuna parte (raggruppamento, riduzione
della dimensionalità, stima di densità), e **mai** per l'auto-supervisione, dove
il bersaglio c'è e se lo fabbrica il metodo stesso. È quel secondo uso ad aver
spinto LeCun a cambiare parola, ed è l'unico che qui si evita.

## Perché non esiste un indice giusto

Chiusa la parentesi, resta la domanda che tutta la sezione ha rimandato: fra
tutti questi voti, qual è quello buono? La risposta è che la domanda non ne ha
una, e non per pigrizia della ricerca: c'è un teorema che lo dice.

Nel 2002 Jon Kleinberg dimostra che tre proprietà che a chiunque sembrerebbero
minime per una funzione di raggruppamento **non possono valere tutte e tre
insieme** {cite}`kleinberg2002impossibility`. Le tre sono:

- **invarianza di scala**: misurando le distanze in centimetri o in pollici, i
  gruppi devono venire gli stessi;
- **ricchezza**: cambiando le distanze, l'algoritmo deve poter produrre
  *qualunque* suddivisione dei punti, e non solo un sottoinsieme privilegiato;
- **coerenza**: se si stringono i gruppi trovati e si allontanano l'uno
  dall'altro, la risposta non deve cambiare. Avendo reso più evidente la
  partizione che si era già scelta, non può essere questo a far cambiare idea.

Nessuna funzione le soddisfa tutte e tre: è il teorema, e la dimostrazione non
passa da nessun algoritmo particolare (viene da un fatto generale su quali
famiglie di suddivisioni una funzione così può produrre). Quello che Kleinberg
mostra **sugli algoritmi** è la metà complementare, ed è la più istruttiva: che
a cadere è **una** proprietà sola, e che si può scegliere quale.

Gliene basta una famiglia, il raggruppamento per **legame singolo**, che parte
con ogni punto per conto suo e fonde ogni volta i due gruppi più vicini fra
loro. Cambiando soltanto la regola con cui si smette di fondere si ottengono tre
metodi, ciascuno dei quali soddisfa due proprietà su tre:

- fermarsi quando i gruppi sono $k$ rinuncia alla **ricchezza**, perché le
  suddivisioni con un numero diverso di gruppi non sono più raggiungibili;
- fermarsi a una distanza fissa rinuncia all'**invarianza di scala**, perché
  quella distanza è in centimetri e cambiando unità cambia tutto;
- fermarsi a una frazione della distanza massima rinuncia alla **coerenza**.

Il primo caso vale per chiunque fissi il numero di gruppi in anticipo,
$k$-means compreso. E allora i metodi noti non sono approssimazioni imperfette
di un ideale che un giorno qualcuno troverà: sono i rami di una biforcazione
obbligatoria, e ciascuno dichiara, con la sua regola d'arresto, a che cosa ha
rinunciato.

È lo stesso Jon Kleinberg che il capitolo sull'AI responsabile incontra per il
teorema di impossibilità sull’**equità**, dove tre criteri ragionevoli di
imparzialità non possono valere insieme se non nei casi degeneri. Due
impossibilità distinte, stessa forma dell'argomento e stesso autore, a quindici
anni di distanza; e in tutti e due i casi la conseguenza pratica è che la scelta
va **dichiarata** invece che cercata, perché nessun dato la farà al posto nostro.

Che è, poi, la storia di Plutone: alla fine si vota.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un raggruppamento **non ha una risposta giusta**: raggruppare i libri per
  genere o per epoca sono due domande diverse e due risposte entrambe valide. È
  la storia di Plutone: a decidere è stata una votazione, non una misura.
- Gli indici **interni** (la silhouette) guardano solo la scaffalatura: i gruppi
  sono compatti? ben separati? Gli indici **esterni** (ARI, NMI) confrontano con
  una risposta nota, quando c'è.
- **Un indice interno può bocciare la risposta giusta.** Sulle due lune, DBSCAN
  ricostruisce i gruppi veri alla perfezione (ARI $1{,}000$) e la silhouette
  preferisce $k$-means, che ha sbagliato ($0{,}486$ contro $0{,}331$). La
  silhouette non misura «giusto», misura «tondo e ben distanziato».
- L'indice di **Rand** grezzo non parte da zero: etichette tirate a caso ne
  prendono $0{,}5$. La versione **aggiustata** (ARI) toglie quello che si
  prenderebbe per caso, e sul caso vale $0$.
- Senza risposta giusta si può chiedere se il raggruppamento **tiene**: rifallo
  su due metà dei dati e guarda se dicono la stessa cosa. Serve soprattutto a
  scartare i numeri di gruppi che non reggono.
- Un metodo di raggruppamento perfetto **non esiste**, e non è colpa di nessuno:
  tre proprietà minime e ragionevoli non possono valere tutte e tre insieme
  (Kleinberg, 2002). La scelta va dichiarata, perché nessun dato la farà al
  posto nostro.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Indici **interni** (silhouette, Calinski–Harabasz, Davies–Bouldin): usano solo
  $\mathbf{X}$ e le etichette assegnate, e premiano coesione e separazione. Sono
  allineati a un'ipotesi geometrica **convessa**: su geometrie non convesse
  misurano la somiglianza con la partizione di $k$-means, non la qualità.
  Misurato: sulle due lune la silhouette dà $0{,}486$ a $k$-means e $0{,}331$ a
  DBSCAN, mentre ARI e NMI di DBSCAN valgono $1{,}000$.
- Indici **esterni** (ARI, NMI/AMI): confrontano con una partizione nota, sono
  invarianti alla permutazione delle etichette e ragionano sulle **coppie** di
  punti. L'indice di Rand grezzo non è corretto per il caso: misurato,
  $\mathrm{RI} = 0{,}500$ su etichette casuali contro $\mathrm{ARI} = 0{,}0000$.
- **Stabilità**: due sottocampioni, due adattamenti a $k$ gruppi, accordo
  misurato con l'ARI sull'intersezione. Criterio applicabile senza etichette; da
  usare per **scartare** i $k$ instabili, perché i $k$ piccoli sono stabili quasi
  per costruzione ($0{,}960$ a $k=2$ contro $1{,}000$ a $k=4$ nel conto qui
  sopra).
- **Teorema di impossibilità di Kleinberg** {cite}`kleinberg2002impossibility`:
  nessuna funzione di clustering soddisfa insieme invarianza di scala, ricchezza
  e coerenza. Gli algoritmi noti sono i rami della rinuncia, non approssimazioni
  di un ideale.
- **Nome**: «non supervisionato» è corretto per raggruppamento, riduzione di
  dimensionalità e stima di densità, dove nessuna parte del dato viene prevista.
  Non lo è per l’**auto-supervisione**, dove il bersaglio esiste e se lo
  costruisce il metodo, ed è quello il caso su cui verte l'obiezione di LeCun e
  Misra {cite}`lecun2021darkmatter`.
```

`````

Con questa sezione il capitolo ha detto tutto quello che sa dire su dati senza
etichette, e ha finito con una richiesta invece che con una risposta: dichiarare
il criterio, perché i dati non lo contengono. È un'abitudine che tornerà utile
subito, perché la sezione seguente osserva cosa succede quando anche il criterio
sta fermo e a muoversi sono i dati.
