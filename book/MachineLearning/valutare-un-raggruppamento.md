# Quando non c'è una risposta giusta: valutare un raggruppamento

Che Plutone non sia un pianeta lo ha deciso un'alzata di mano. Successe
all'assemblea generale dell'Unione Astronomica Internazionale, a Praga,
nell'estate del 2006. La risoluzione fissava tre criteri (orbitare attorno al
Sole, essere abbastanza massiccio da essersi fatto tondo da sé, e aver ripulito
la propria orbita dagli altri corpi), e Plutone cadeva sul terzo.

Ci è voluta una votazione, anche se le misure erano note a tutti e nessuno le
contestava. Mancava il criterio con cui raggruppare gli oggetti del sistema
solare in famiglie, perché di criteri ragionevoli ce n'era più d'uno e
portavano a risposte diverse. Tenendo i tre criteri, ad esempio, i pianeti sono
otto; togliendo il terzo diventano tredici o più. Nessuna delle due tassonomie
è sbagliata: sono due modi di tagliare la stessa collezione, e a scegliere è
stata una comunità, non un dato.

I quattro metodi di raggruppamento di {doc}`Riduzione e clustering
<riduzione-clustering>` ($k$-means, DBSCAN, clustering gerarchico, misture
gaussiane) lasciano aperta la domanda più difficile: come si valuta un
raggruppamento? In un problema supervisionato si confronta la predizione con
l'etichetta e si contano gli errori. Nel clustering, di solito, [la risposta
giusta non esiste da nessuna parte: non è stata persa, non esiste]{.pt-matita}.
Anche quando un'etichetta c'è (nei dati di prova, o tenuta da parte per il voto
finale), non è comunque l'unico raggruppamento sensato, perché dipende dal
criterio con cui si decide che due punti si somigliano.

## Due domande diverse, due famiglie di indici

Gli indici con cui si giudica un raggruppamento si dividono in due famiglie, a
seconda delle informazioni che hanno a disposizione. Gli **indici esterni**
confrontano la partizione trovata con una partizione di riferimento, quando ce
n'è una; gli **indici interni** usano soltanto i dati e la partizione, e
misurano quanto i gruppi sono compatti e separati fra loro.

`````{tab} Elementare

Una biblioteca da riordinare, e tre bibliotecari: uno mette i libri per genere,
uno per epoca, uno per lingua. Ne escono tre scaffalature, e nessuno ha
sbagliato, perché hanno risposto a domande diverse. Tocca a te dare un voto, e
dipende tutto da una cosa: se qualcuno, in quella stanza, ha già la risposta.

A volte ce l'ha. Il direttore sa che i libri andavano divisi per genere, e
mentre i bibliotecari lavorano tiene l'elenco giusto in tasca; lo tira fuori
solo per il voto, e confronta. Gli indici che fanno questo confronto guardano
fuori dagli scaffali, verso una verità nota, e per questo si chiamano esterni.

Il più vecchio guarda i libri due a due. Peschi il Calvino e l'Ovidio e chiedi
a tutte e due le scaffalature la stessa cosa: insieme o separati? Se rispondono
uguale, quella coppia è un accordo, e la frazione di coppie in accordo è
l'indice di **Rand**.

Ora prova a barare. Due bibliotecari che tirano i libri sui ripiani senza
leggerne i titoli vanno d'accordo su moltissime coppie. Se i ripiani sono
tanti, infatti, due libri presi a caso finiscono quasi sempre su ripiani
diversi in tutte e due le scaffalature, e per il Rand separare è già un
accordo. Di qui la correzione: si toglie l'accordo che due lavori tirati a caso
si prenderebbero comunque, e si guarda quanto del resto il bibliotecario ha
guadagnato davvero. Se il caso arriva già a $0{,}9$ di accordo e lui a
$0{,}95$, ha guadagnato metà di quello che restava:
$(0{,}95-0{,}9)/(1-0{,}9) = 0{,}5$. È l’**ARI** (*Adjusted Rand Index*, indice
di Rand aggiustato), che dà $1$ a due scaffalature identiche e, in media, zero
a due lavori tirati a caso.

L’**NMI** (*Normalized Mutual Information*, informazione mutua normalizzata) fa
un'altra domanda: sapendo dove sta il Calvino nella prima scaffalatura, quanto
hai già indovinato della seconda? Anche l'NMI dà $1$ a due scaffalature
identiche, ma sul lavoro tirato a caso con tanti ripiani piccoli resta un po’
sopra lo zero, tanto che ne gira una versione corretta allo stesso modo, l'AMI.

Nessuno di questi indici si fa ingannare dai cartellini: il «ripiano 1» di
un bibliotecario e quello di un altro non hanno niente a che vedere, e conta
soltanto quali libri stanno insieme.

Il caso vero è l'altro: hai appena ereditato la biblioteca, e la risposta non ce
l'ha nessuno. Resta solo lo scaffale da guardare: i libri di uno stesso ripiano
si somigliano fra loro? I ripiani si distinguono l'uno dall'altro? Un indice che
si accontenta di questo si dice interno, e il più noto è la silhouette, già
incontrata in {doc}`Riduzione e clustering <riduzione-clustering>`.

Per calcolarla prendi un libro e misuri la sua distanza media dai compagni di
ripiano, cioè quanto è diverso da loro, e quella dagli estranei del ripiano più
vicino; poi fai la differenza fra la seconda e la prima, e la dividi per la più
grande delle due. Ad esempio, un libro che dista in media $2$ dai suoi e $6$
dagli estranei prende $(6-2)/6 \approx 0{,}67$; uno che dista $5$ dai suoi e
$4$ dagli estranei prende $(4-5)/5 = -0{,}2$, segno che starebbe meglio di là.
La media su tutti i libri è la silhouette della scaffalatura. Con un ripiano
solo, però, di estranei non ce ne sono e il conto non si fa: la silhouette non
saprà mai consigliarti di lasciare tutti i libri insieme.

L'altro difetto è che [la silhouette premia sempre lo stesso tipo di scaffale,
il mucchio tondo e ben distanziato]{.pt-matita}. Una saga lunga è invece una
fila: ogni volume somiglia al successivo, il primo e l'ultimo per niente, e in
un punto la fila sfiora quella di un'altra saga. In quel punto il volume è più
vicino, in media, agli estranei della fila accanto che alla sua stessa saga,
che si allunga lontano. La silhouette lo dà quindi per mal collocato, e con lui
boccia una scaffalatura che aveva ragione.

Fra due bibliotecari che fanno tutti e due mucchi tondi la silhouette resta un
buon arbitro. Usata per scegliere fra criteri diversi, invece, premia
quello che fa scaffali tondi come piacciono a lei, e lo fa con convinzione.

`````

`````{tab} Superiore

Gli indici interni valutano una partizione usando solo $\mathbf{X}$ e le
etichette assegnate, e tutti quantificano una qualche forma di rapporto fra
coesione e separazione. Consideriamo per prima la silhouette media,

$$
\bar{s} = \frac{1}{m}\sum_{i=1}^{m}
\frac{b_i - a_i}{\max(a_i,\, b_i)} ,
$$

dove $a_i$ è la distanza media del punto $i$ dagli altri punti del suo gruppo
(la coesione) e $b_i$ la distanza media dai punti del gruppo diverso più vicino
(la separazione). La silhouette del singolo punto si avvicina a $+1$ se il
punto è molto meglio collocato dove sta (vale $+1$ solo con $a_i = 0$), $0$ se
sta sul confine, meno di zero se starebbe meglio altrove
{cite}`rousseeuw1987silhouettes`.

Gli altri due indici usati di frequente si calcolano dai centroidi, con un
costo lineare in $m$ invece del quadratico della silhouette, che richiede tutte
le distanze fra coppie. Il primo, il **Calinski–Harabasz**
{cite}`calinski1974dendrite`, è

$$
\mathrm{CH} = \frac{\operatorname{tr}(\mathbf{B})/(k-1)}{\operatorname{tr}(\mathbf{W})/(m-k)},
$$

dove $\operatorname{tr}(\mathbf{W}) = \sum_j\sum_{i\in
C_j}\lVert\mathbf{x}_i-\boldsymbol{\mu}_j\rVert^2$ è la dispersione dentro i
gruppi, cioè l'inerzia di $k$-means, e $\operatorname{tr}(\mathbf{B}) =
\sum_j|C_j|\,\lVert\boldsymbol{\mu}_j-\boldsymbol{\mu}\rVert^2$ quella fra i
gruppi, con $\boldsymbol{\mu}_j$ il centroide del gruppo $C_j$ e
$\boldsymbol{\mu}$ la media di tutti i punti. Si noti che senza i due fattori
di normalizzazione il rapporto crescerebbe sempre con $k$. Il secondo, il
**Davies–Bouldin** {cite}`davies1979cluster`, è

$$
\mathrm{DB} = \frac{1}{k}\sum_{j=1}^{k}\max_{l\neq j}\frac{\delta_j+\delta_l}{\lVert\boldsymbol{\mu}_j-\boldsymbol{\mu}_l\rVert},
$$

con $\delta_j$ la distanza media dei punti del gruppo $j$ dal suo centroide.
Per ogni gruppo si prende il vicino che gli somiglia di più, poi si fa la
media.

Tutti e tre presuppongono una nozione geometrica di «buono», cioè premiano i
gruppi compatti, convessi e ben spaziati. La silhouette e il Calinski–Harabasz
lo fanno crescendo, mentre il Davies–Bouldin, il solo dei tre da minimizzare,
lo fa scendendo. Nessuno dei tre è definito per $k = 1$, quindi nessuno può
dire che non ci sono gruppi; a quella domanda risponde la *gap statistic*
{cite}`tibshirani2001estimating`, che confronta l'inerzia con quella di dati
senza struttura.

Su geometrie non convesse, però, compattezza e separazione non coincidono con
la qualità della partizione, e [gli indici favoriscono i gruppi compatti e
convessi, cioè quelli che $k$-means tende a produrre]{.pt-matita}. In altri
termini, sono indici allineati con l'ipotesi di un particolare algoritmo. Usati
per scegliere fra algoritmi con ipotesi diverse, tendono a premiare quello che
la condivide, mentre fra partizioni che la condividono ($k$-means a $k$
diversi, Ward contro $k$-means) sono il metro giusto.

Per le forme non convesse esistono indici interni costruiti su un'altra
ipotesi, quella dei gruppi come regioni dense. Il più usato è il DBCV
{cite}`moulavi2014dbcv`, che per ogni gruppo confronta la regione meno densa al
suo interno con la più densa che lo separa dagli altri; l'indice assume valori
fra $-1$ e $1$ e va massimizzato. Neanche lui è neutro: è il metro giusto per
chi raggruppa per densità, come la silhouette lo è per $k$-means.

Gli indici esterni, invece, confrontano la partizione ottenuta $C$ con una nota
$T$. Il capostipite è l'indice di Rand {cite}`rand1971objective`, definito
come la frazione delle $\binom{m}{2}$ coppie di punti su cui le due partizioni
sono d'accordo (stessa coppia insieme in entrambe, o separata in entrambe),

$$
\mathrm{RI} = \frac{n_{\text{ins}} + n_{\text{sep}}}{\binom{m}{2}},
$$

con $n_{\text{ins}}$ le coppie tenute insieme da entrambe e $n_{\text{sep}}$
quelle separate da entrambe (le lettere $a$, $b$ e $s$ sono già occupate
dalla silhouette).

L'indice di Rand ha un difetto grave: non vale zero sul caso nullo, e la
linea di base non è nemmeno una costante, perché dipende da quanti gruppi hanno
le due partizioni. Due etichettature casuali, infatti, concordano su tutte le
coppie che entrambe separano, e più i gruppi sono fini, più coppie separano;
quindi $\mathrm{RI}$ sale verso $1$ per puro conteggio. Il rimedio è
l’Adjusted Rand Index di Hubert e Arabie {cite}`hubert1985comparing`, che
sottrae il valore atteso sotto un modello di permutazione casuale e normalizza:

$$
\mathrm{ARI} = \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}
                    {\max(\mathrm{RI}) - \mathbb{E}[\mathrm{RI}]},
$$

dove $\max(\mathrm{RI})$ vale $1$. Questo valore non è il massimo
raggiungibile, ma un limite superiore, che con taglie diverse nelle due
partizioni non si tocca. Sulla tabella di contingenza $n_{ij}$ (i punti che
stanno nel gruppo $i$ di $C$ e nel gruppo $j$ di $T$), con $u_i$ e $v_j$ i
totali di riga e di colonna, l'indice è

$$
\mathrm{ARI} = \frac{\sum_{ij}\binom{n_{ij}}{2} - t}
{\tfrac12\bigl[\sum_i\binom{u_i}{2} + \sum_j\binom{v_j}{2}\bigr] - t},
\qquad
t = \frac{\sum_i\binom{u_i}{2}\,\sum_j\binom{v_j}{2}}{\binom{m}{2}},
$$

dove $t$ è il numero atteso di coppie tenute insieme da entrambe sotto il
modello di permutazione, e il primo termine del denominatore è quell’$1$,
contato in coppie. L'indice vale $1$ per l'accordo perfetto, $0$ in media sul
caso nullo, e può essere negativo per un accordo peggiore del caso.

L'alternativa che viene dalla teoria dell'informazione è l’NMI,

$$
\mathrm{NMI}(C,T) = \frac{I(C;T)}{\tfrac12\big[H(C)+H(T)\big]},
\qquad
I(C;T) = \sum_{ij}\frac{n_{ij}}{m}\log\frac{m\,n_{ij}}{u_i\,v_j},
$$

con $H$ l'entropia delle frequenze dei gruppi. La media aritmetica al
denominatore è la scelta di scikit-learn; altre versioni usano la geometrica o
il massimo, per cui i valori non sono confrontabili fra versioni diverse. L'NMI
vale $1$ sull'accordo perfetto ma non $0$ sul caso, e cresce con il numero di
gruppi; la versione corretta per il caso, l'AMI, sottrae il valore atteso sotto
il modello di permutazione, come l'ARI {cite}`vinh2010information`.

Osserviamo infine che Rand, ARI e NMI sono invarianti alla permutazione delle
etichette, una proprietà indispensabile, perché il «gruppo 0» di un algoritmo e
il «gruppo 0» di un altro non hanno niente in comune. Rand e ARI ci arrivano
contando le coppie di punti tenute insieme o separate; NMI e AMI lo fanno
misurando quanta informazione una partizione dà sull'altra.

`````

## Il caso in cui l'indice interno boccia la risposta giusta

Il banco di prova sono le due lune di {doc}`Riduzione e clustering
<riduzione-clustering>`, dove si sa già chi ha ragione: $k$-means le taglia con
un confine rettilineo e sbaglia, mentre DBSCAN segue la densità e ricostruisce
le due forme. Accanto alla silhouette media stampiamo anche la quota di punti
con silhouette negativa.

```python
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             rand_score, silhouette_samples, silhouette_score)

X, vero = make_moons(n_samples=600, noise=0.06, random_state=0)
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
db = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)   # qui nessun punto va a rumore

print(f"{'':12}{'silhouette':>12}{'s<0':>8}{'Rand':>8}{'ARI':>8}{'NMI':>8}")
for nome, e in (("k-means", km), ("DBSCAN", db)):
    negativi = (silhouette_samples(X, e) < 0).mean()   # quota con s < 0
    print(f"{nome:12}{silhouette_score(X, e):12.3f}{negativi:8.3f}"
          f"{rand_score(vero, e):8.3f}{adjusted_rand_score(vero, e):8.3f}"
          f"{normalized_mutual_info_score(vero, e):8.3f}")

# quanto vale il "niente"? due etichettature tirate a caso, confrontate fra loro
r = np.random.default_rng(0)
print("\ndue etichettature a caso, quanto si somigliano:")
for k in (2, 5, 20):
    grezzi, aggiustati, nmi = [], [], []
    for _ in range(200):
        a, b = r.integers(0, k, 600), r.integers(0, k, 600)
        grezzi.append(rand_score(a, b))
        aggiustati.append(adjusted_rand_score(a, b))
        nmi.append(normalized_mutual_info_score(a, b))
    print(f"  con {k:2d} gruppi ciascuna:  Rand {np.mean(grezzi):.3f}"
          f"   ARI {np.mean(aggiustati):+.4f}   NMI {np.mean(nmi):.3f}")
```

```text
              silhouette     s<0    Rand     ARI     NMI
k-means            0.486   0.003   0.628   0.255   0.194
DBSCAN             0.331   0.162   1.000   1.000   1.000

due etichettature a caso, quanto si somigliano:
  con  2 gruppi ciascuna:  Rand 0.500   ARI +0.0003   NMI 0.001
  con  5 gruppi ciascuna:  Rand 0.680   ARI -0.0003   NMI 0.008
  con 20 gruppi ciascuna:  Rand 0.905   ARI -0.0001   NMI 0.119
```

L'indice interno dice il contrario di quelli esterni. Secondo ARI e NMI, che
sanno qual è la risposta giusta, DBSCAN ha ricostruito le due lune alla
perfezione: valgono $1{,}000$, e la sua partizione coincide con quella vera. La
silhouette dà a DBSCAN $0{,}331$ e a $k$-means $0{,}486$; giudicando con lei,
quindi, si sceglierebbe il metodo che ha sbagliato, e lo si sceglierebbe con un
margine confortevole.

La silhouette, infatti, risponde a un'altra domanda. Per i punti vicini alla
punta di una luna, la luna accanto è più vicina, in media, del resto della
propria, che si allunga fino all'altro capo della curva. Per quei punti $b_i$
(la distanza media dalla luna vicina) è minore di $a_i$ (quella dal resto della
propria), e quindi $s_i < 0$ anche se l'assegnazione è giusta. Con la
partizione di DBSCAN, che è quella vera, succede al $16\%$ dei punti (la quota
di `s<0`), contro lo $0{,}3\%$ di $k$-means.

In generale, [un indice interno esprime una domanda geometrica di compattezza e
separazione, e va usato solo quando questa è la domanda che ci si
pone]{.pt-matita}. (Qui DBSCAN non manda nessun punto nel rumore; quando lo fa,
scikit-learn tratta l'etichetta $-1$ come un gruppo qualsiasi, e per l'ARI
tutti i punti di rumore formano un gruppo unico.)

Il confronto fra etichettature casuali riguarda il Rand, e mostra perché nella
sua forma grezza non va usato. Sono due etichettature tirate a caso, due
raggruppamenti privi di informazione, messi a confronto fra loro. Con due
gruppi per parte il Rand dà $0{,}500$, con cinque $0{,}680$, con venti
$0{,}905$. Su una scala che arriva a $1$, quindi, il puro caso prende $0{,}9$
purché i gruppi siano tanti e piccoli, e chi legge quel numero pensa di aver
quasi indovinato. La ragione è aritmetica: due partizioni fini separano quasi
tutte le coppie, e il Rand conta come «accordo» anche l'aver separato.

L'ARI sulle stesse etichettature vale $+0{,}0003$, $-0{,}0003$, $-0{,}0001$,
cioè non si muove. Il Rand cade nello stesso inganno dell'accuratezza su classi
sbilanciate, perché qui le coppie separate da tutte e due le partizioni fanno la
parte della classe dominante. La correzione è la stessa del kappa di Cohen,
discusso in {doc}`Valutare un modello <metriche>` nella forma pesata: sottrarre
l'accordo che si otterrebbe per caso e dividere per quanto ne resta da
guadagnare; da qui l'aggettivo *aggiustato*.

L'NMI sta in mezzo: $0{,}001$ e $0{,}008$ con due e cinque gruppi, ma $0{,}119$
con venti, e per questo ne esiste una versione aggiustata, l'AMI.

## Senza risposta giusta: chiedere se il raggruppamento tiene

Resta il caso vero, quello in cui l'etichetta non c'è e un indice interno non
basta. C'è una terza via, che non misura la qualità ma la **riproducibilità**.

`````{tab} Elementare

L'idea è quella del bibliotecario messo alla prova due volte. Dagli metà dei
libri, presi a caso, digli quanti ripiani usare e fagli fare gli scaffali. Poi
dagli un'altra metà, pescata a parte, e faglieli rifare con lo stesso numero di
ripiani. Una parte dei libri gli è capitata in mano tutte e due le volte, e il
confronto si fa su quelli.

Se il criterio che sta usando è nei libri, le due volte li ha sistemati allo
stesso modo. Se invece si sta inventando le categorie, i due lavori saranno
diversi. L'ARI, l'indice a coppie fatto apposta per confrontare due
scaffalature, dice quanto vanno d'accordo.

Una prova sola dice poco, perché due metà possono andare d'accordo per fortuna.
Per questo si rifà da capo con altre metà, molte volte, e si guarda com'è
andata nell'insieme.

Tutto questo si può fare senza sapere niente della risposta giusta, e serve
soprattutto a decidere quanti gruppi cercare, cioè quanti ripiani chiedere: il
numero buono dovrebbe superare la prova, e uno sbagliato produrre scaffalature
che cambiano ogni volta che cambiano i libri.

Un numero sbagliato, però, può superare la prova lo stesso. Se i libri sono di
quattro generi e chiedi due ripiani, ad esempio, il bibliotecario deve unire i
generi a due a due; se un modo di unirli è nettamente il più comodo, lo sceglie
ogni volta, e le due prove vanno d'accordo anche se due ripiani sono troppo
pochi.

A volte ci si accorge del numero sbagliato guardando non solo quanto le prove
vanno d'accordo in media, ma anche se vanno d'accordo tutte le volte: quando
due modi di unire i generi sono comodi uguale, ogni tanto le due metà ne
scelgono due diversi. Se il modo più comodo è uno solo, le prove vanno
d'accordo sempre, e la prova non se ne accorge. Serve dunque a scartare i
numeri che non tengono, non a incoronare il più stabile.

`````

`````{tab} Superiore

La **stabilità** come criterio di selezione formalizza questa idea: si
estraggono due sottocampioni $S_1, S_2 \subset \mathbf{X}$, si adatta
l'algoritmo a $k$ gruppi su ciascuno, si predicono le etichette sui punti
$S_1 \cap S_2$ e si misura l'accordo fra le due assegnazioni con un indice
esterno (l'ARI, appunto, perché l'accordo fra due partizioni è ciò che misura).
Mediando i risultati su estrazioni multiple si ottiene $\mathrm{stab}(k)$, e si
sceglie il $k$ che la massimizza.

Il criterio ha tuttavia un limite che è un teorema. Per un $k$-means che trovi
il minimo globale dell'inerzia, quando quel minimo è unico, la stabilità tende
a $1$ al crescere dei dati qualunque sia $k$, giusto o sbagliato, e scende
sotto $1$ solo quando i minimi equivalenti sono più d'uno, per una simmetria
dei dati {cite}`bendavid2006sober`. La stabilità va quindi letta come un
vincolo (scarta i $k$ instabili), non come una funzione da massimizzare alla
cieca.

Si noti che il teorema riguarda la misura di instabilità così com'è e un
algoritmo che trova l'ottimo globale. Riscalandola nel modo giusto rispetto al
numero di esempi, Shamir e Tishby mostrano che la stabilità di $k$-means non
perde potere discriminante nemmeno per campioni molto grandi
{cite}`shamir2008model`.

`````

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

r = np.random.default_rng(0)
# quattro gruppi ben separati: la risposta giusta, che l'algoritmo non sa, è 4
X = np.vstack([r.normal(c, 0.55, (200, 2)) for c in ([0, 0], [5, 0], [0, 5], [5, 5])])

def stabilita(X, k, prove=25, seme=0):
    """Due metà dei dati a caso, due raggruppamenti: quanto vanno d'accordo?"""
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

print(f"{'k':>3}{'silhouette':>12}{'stabilità':>12}{'(fra le prove)':>16}")
for k in range(2, 9):
    e = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    media, disp = stabilita(X, k)
    print(f"{k:3d}{silhouette_score(X, e):12.3f}{media:12.3f}"
          f"{'+/- ' + format(disp, '.3f'):>16}")

# la riga k = 2 dipende dal campione: altre cinque estrazioni
print("\nk = 2 su altre cinque estrazioni degli stessi quattro gruppi:")
for estrazione in range(1, 6):
    r = np.random.default_rng(estrazione)
    Y = np.vstack([r.normal(c, 0.55, (200, 2))
                   for c in ([0, 0], [5, 0], [0, 5], [5, 5])])
    media, disp = stabilita(Y, 2, prove=15)
    print(f"  estrazione {estrazione}:  {media:.3f} +/- {disp:.3f}")
```

```text
  k  silhouette   stabilità  (fra le prove)
  2       0.508       0.960       +/- 0.197
  3       0.606       0.655       +/- 0.238
  4       0.795       1.000       +/- 0.000
  5       0.676       0.897       +/- 0.075
  6       0.550       0.777       +/- 0.083
  7       0.440       0.699       +/- 0.086
  8       0.318       0.662       +/- 0.099

k = 2 su altre cinque estrazioni degli stessi quattro gruppi:
  estrazione 1:  0.466 +/- 0.500
  estrazione 2:  0.933 +/- 0.251
  estrazione 3:  0.332 +/- 0.472
  estrazione 4:  0.466 +/- 0.500
  estrazione 5:  0.332 +/- 0.472
```

Su dati con quattro gruppi ben separati, i due criteri concordano e indicano
$k = 4$: la silhouette con $0{,}795$, il suo massimo, e la stabilità con
$1{,}000$, il valore pieno, segno che due metà indipendenti dei dati hanno
prodotto la stessa partizione sui punti in comune. Quando la struttura c'è e ha
la forma che $k$-means si aspetta, misurarla è facile e ogni strumento la
trova.

Il caso $k = 2$ è una trappola: stabilità $0{,}960$, quasi quanto quella del
$k$ giusto. La dispersione fra le prove serve a smascherarla. A $k = 4$ la
stabilità è $1{,}000$ con dispersione zero, perché venticinque prove su
venticinque hanno dato lo stesso identico risultato. A $k = 2$ la stessa media
di $0{,}960$ arriva da prove che variano di $\pm 0{,}197$: quella media
nasconde due esiti opposti, quasi sempre un accordo pieno e ogni tanto un
disaccordo totale.

Il perché sta nella geometria. Quattro nuvole ai vertici di un quadrato si
possono tagliare in due in due modi con la stessa inerzia, in orizzontale o in
verticale. Con questo campione uno dei due tagli ha un vantaggio minimo, e le
due metà, che ne ereditano in parte la preferenza, di solito scelgono lo
stesso.

Ma il vantaggio dipende dal campione. Sulle altre cinque estrazioni degli
stessi quattro gruppi, in quattro casi su cinque la media di $k = 2$
scende sotto $0{,}5$ con dispersione vicina a $0{,}5$, cioè due metà che
scelgono un taglio o l'altro come lanciando una moneta. Il segnale è nella
dispersione, più che nella media.

La simmetria fa oscillare il $k = 2$, e in questo caso salva l'analisi. Basta
però allungare il quadrato in altezza, facendone un rettangolo, perché il
taglio che separa le due nuvole in basso dalle due in alto diventi l'unico
ottimo: allora il $k = 2$ sbagliato esce stabile quanto il $k = 4$ giusto,
e né la media né la dispersione se ne accorgono. [La stabilità dice se il
raggruppamento migliore è unico, e sulla sua correttezza tace]{.pt-matita}.

La stabilità serve quindi a scartare i valori instabili (qui il $3$, con
$0{,}655 \pm 0{,}238$: un raggruppamento in tre parti di quattro nuvole
simmetriche deve decidere quali due unire, e ogni volta decide diversamente) e
non a scegliere il massimo assoluto senza guardare altro.

## Una parentesi sul nome: «non supervisionato»

Questi metodi si chiamano «non supervisionati», ma c'è chi sostiene che il nome
non andrebbe usato. Capirne il motivo aiuta a distinguerli dai metodi che
costruiscono un compito da imparare.

Yann LeCun, fra i pionieri delle reti neurali convoluzionali, ha rinunciato
pubblicamente all'espressione «apprendimento non supervisionato», che in un
testo scritto con Ishan Misra nel 2021 giudica mal definita e fuorviante
{cite}`lecun2021darkmatter`. Il nome, infatti, suggerisce che l'apprendimento
non usi supervisione affatto, mentre nei metodi che a lui interessano
(prevedere una parte del dato dal resto: la parola coperta in una frase, il
pezzo mancante di un'immagine) un segnale di correzione c'è eccome, e molto più
ricco di quello di un'etichetta. Quei metodi si chiamano auto-supervisionati, e
hanno il loro capitolo in {doc}`Auto-supervisione </AutoSupervisione/overview>`.

I metodi di raggruppamento e di riduzione della dimensionalità sono un'altra
cosa. Quando $k$-means sposta un centroide o la PCA cerca la direzione di
massima varianza, l'obiettivo è una funzione del solo dato (l'inerzia, l'errore
di ricostruzione), non un compito di previsione fra parti diverse dell'input.
Il confine non è netto: la PCA si può leggere come una ricostruzione del dato
dal suo codice compresso, e $k$-means ricostruisce ogni punto con il suo
centroide. A tracciare il confine è il compito: l'auto-supervisione fabbrica
un bersaglio predicendo una parte nascosta dell'input da una visibile.

Ne segue che [«non supervisionato» si tiene per i metodi che descrivono la
struttura dei dati]{.pt-matita} (raggruppamento, riduzione della
dimensionalità, stima di densità), e non per quelli che si costruiscono un
compito di previsione. LeCun e Misra rinunciano al nome per tutti; il confine
fra descrivere e prevedere è una convenzione, e non è la loro.

## Perché nessun algoritmo di raggruppamento è neutro

Resta una domanda: fra tutti questi voti, qual è quello buono? Un teorema
c'entra, ma parla degli algoritmi e non degli indici. Non esclude che
esista un buon indice: Ackerman e Ben-David hanno mostrato che le stesse
richieste, riscritte per le misure di qualità di un raggruppamento, non
si contraddicono {cite}`ackerman2008measures`. [Esclude che un algoritmo di
raggruppamento possa essere neutro: ognuno incorpora una scelta, e la scelta va
dichiarata]{.pt-matita}.

Nel 2002 Jon Kleinberg dimostra che tre proprietà che a chiunque sembrerebbero
minime per una funzione di raggruppamento non possono valere tutte e tre
insieme {cite}`kleinberg2002impossibility`. Le tre sono:

- **invarianza di scala**: misurando le distanze in centimetri o in pollici, i
  gruppi devono venire gli stessi;
- **ricchezza**: cambiando le distanze, l'algoritmo deve poter produrre
  *qualunque* suddivisione dei punti (anche tutti in un gruppo solo, o ciascuno
  separato dagli altri), e non solo un sottoinsieme privilegiato;
- **coerenza**: se si stringono i gruppi trovati e si allontanano l'uno
  dall'altro, la risposta non deve cambiare. Rendere più evidente la
  partizione già scelta non è una buona ragione per cambiarla.

Nessuna funzione le soddisfa tutte e tre: è questo il teorema, e la
dimostrazione non passa da nessun algoritmo particolare (viene da un fatto
generale su quali famiglie di suddivisioni una funzione così può produrre).
Sugli algoritmi, poi, Kleinberg mostra la metà complementare, la più
istruttiva: con il legame singolo cade una proprietà sola, e si può scegliere
quale.

Per mostrarlo gli basta una famiglia, il raggruppamento per legame singolo, che
parte con ogni punto in un gruppo a sé e fonde ogni volta i due gruppi più
vicini fra loro. Cambiando soltanto la regola con cui si smette di fondere si
ottengono tre metodi, ciascuno dei quali soddisfa due proprietà su tre:

- fermarsi quando i gruppi sono $k$ rinuncia alla ricchezza, perché le
  suddivisioni con un numero diverso di gruppi non sono più raggiungibili;
- fermarsi a una distanza fissa rinuncia all’invarianza di scala, perché
  quella distanza è in centimetri e cambiando unità cambia tutto;
- fermarsi a una frazione della distanza massima rinuncia alla coerenza, perché
  stringere i gruppi e allontanarli fa crescere la distanza massima, e con lei
  la soglia a cui ci si ferma, che può arrivare a fondere gruppi prima separati.

Il primo caso vale per chiunque fissi il numero di gruppi in anticipo, e quindi
anche per $k$-means, che però perde di più. Kleinberg mostra infatti che, per
$k \ge 2$ e un numero di punti abbastanza grande, nessun metodo che scelga $k$
centroidi (fra i punti) e assegni ogni punto al più vicino, $k$-means incluso,
soddisfa la coerenza: stringendo i gruppi e allontanandoli l'uno dall'altro,
l'ottimo può cambiare. Dei tre requisiti $k$-means conserva soltanto
l'invarianza di scala.

Perdere una proprietà sola è una caratteristica del legame singolo
con le tre regole d'arresto, non di tutti i metodi. In definitiva, i metodi
noti non sono approssimazioni imperfette di un ideale che un giorno qualcuno
troverà: ciascuno dichiara, con la sua regola d'arresto o la sua funzione
obiettivo, a che cosa ha rinunciato.

È lo stesso Jon Kleinberg del teorema di impossibilità sull’equità, che
tratterà il {doc}`capitolo sull'AI responsabile
</AIResponsabile/equita-e-bias>`: tre criteri ragionevoli di
imparzialità non possono valere insieme se non in due casi particolari, la
previsione perfetta e gruppi con la stessa frequenza di base. Sono due
impossibilità distinte, con la stessa forma dell'argomento e lo stesso autore,
a una quindicina d’anni di distanza; e in tutti e due i casi la conseguenza
pratica è che la scelta va dichiarata invece che cercata, perché nessun dato la
farà al posto nostro.

Che è, poi, la storia di Plutone: alla fine si vota.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un raggruppamento spesso non ha una risposta giusta sola: raggruppare i libri
  per genere o per epoca sono due domande diverse e due risposte entrambe
  valide. È la storia di Plutone: a decidere è stata una votazione, non una
  misura.
- Gli indici interni (la silhouette) guardano solo la scaffalatura: i gruppi
  sono compatti? ben separati? Gli indici esterni (ARI, NMI) confrontano con
  una risposta nota, quando c'è.
- Un indice interno può bocciare la risposta giusta. Sulle due lune, DBSCAN
  ricostruisce i gruppi veri alla perfezione (ARI $1{,}000$) e la silhouette
  preferisce $k$-means, che ha sbagliato ($0{,}486$ contro $0{,}331$). La
  silhouette non misura «giusto», misura «tondo e ben distanziato»: resta un
  buon arbitro fra raggruppamenti che fanno tutti mucchi tondi, non fra chi li
  fa e chi no.
- L'indice di Rand grezzo non parte da zero: due etichettature tirate a caso in
  due gruppi ne prendono $0{,}5$, e più i gruppi sono tanti e piccoli più sale
  (con venti gruppi, $0{,}9$). La versione aggiustata (ARI) toglie l'accordo
  che si avrebbe per caso e misura quanto di ciò che restava è stato
  guadagnato davvero: vale $1$ su due scaffalature identiche e, in media, $0$
  sul caso.
- Senza risposta giusta si può chiedere se il raggruppamento tiene: rifallo
  su due metà dei dati e guarda se dicono la stessa cosa. Serve a scartare i
  numeri di gruppi che non sono stabili, non a scegliere il più stabile,
  perché anche un numero sbagliato può superare la prova.
- Un metodo di raggruppamento perfetto non esiste, e non è colpa di nessuno:
  tre proprietà minime e ragionevoli non possono valere tutte e tre insieme
  (Kleinberg, 2002). La scelta va dichiarata, perché nessun dato la farà al
  posto nostro.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Indici interni: usano solo $\mathbf{X}$ e le etichette assegnate, e premiano
  coesione e separazione, salendo la silhouette e il Calinski–Harabasz,
  scendendo il Davies–Bouldin, che dei tre è il solo da minimizzare. Sono
  allineati a un'ipotesi geometrica convessa: fra partizioni che la condividono
  ($k$-means a $k$ diversi) sono il metro giusto, ma su geometrie non convesse
  misurano compattezza e separazione, che lì non coincidono con la qualità, e
  favoriscono le partizioni che $k$-means tende a produrre; per le forme non
  convesse c'è il DBCV, costruito sulla densità. Sulle due lune la silhouette
  dà $0{,}486$ a $k$-means e $0{,}331$ a DBSCAN, mentre ARI e NMI di DBSCAN
  valgono $1{,}000$.
- Indici esterni: confrontano con una partizione nota e sono invarianti alla
  permutazione delle etichette; Rand e ARI contano le coppie di punti, NMI e AMI
  misurano l'informazione mutua fra le due partizioni. L'indice di Rand grezzo
  non è corretto per il caso: su etichette casuali dà $\mathrm{RI} = 0{,}500$
  con due gruppi e $0{,}905$ con venti, contro $\mathrm{ARI} \approx 0$; anche
  l'NMI sale con i gruppi ($0{,}119$ con venti), e la sua versione corretta è
  l'AMI.
- Stabilità: due sottocampioni, due adattamenti a $k$ gruppi, accordo
  misurato con l'ARI sull'intersezione. Criterio applicabile senza etichette; da
  usare per scartare i $k$ instabili e non per scegliere il più stabile: con un
  minimo unico dell'inerzia ogni $k$, giusto o sbagliato, tende a stabilità
  $1$ al crescere del campione, e sotto $1$ la porta solo una simmetria dei
  dati ($0{,}960$ a $k=2$ contro $1{,}000$ a $k=4$ sulle quattro nuvole
  disposte in quadrato).
- Teorema di impossibilità di Kleinberg {cite}`kleinberg2002impossibility`:
  nessuna funzione di clustering soddisfa insieme invarianza di scala, ricchezza
  e coerenza. Il legame singolo, con tre regole d'arresto diverse, rinuncia a
  una proprietà per volta; $k$-means ne perde due. Nessun algoritmo è neutro; il
  teorema non dice niente sugli indici, per i quali le stesse richieste sono
  compatibili {cite}`ackerman2008measures`.
- Nome: «non supervisionato» si tiene per raggruppamento, riduzione di
  dimensionalità e stima di densità, che descrivono la struttura dei dati. Non
  lo è per l’auto-supervisione, dove il bersaglio esiste e se lo costruisce il
  metodo. LeCun e Misra {cite}`lecun2021darkmatter` rinunciano al nome per tutti
  i metodi, perché lo giudicano mal definito; il confine fra descrivere e
  prevedere è una convenzione.
```

`````

Su dati senza etichette resta una richiesta invece di una risposta: dichiarare
il criterio, perché i dati non lo contengono. Un criterio dichiarato, però,
vale per i dati su cui lo si è fissato, e {doc}`Quando i dati cambiano
<dati-che-cambiano>` guarda che cosa succede quando a cambiare sono i dati.
