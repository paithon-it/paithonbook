# Il filtraggio collaborativo

C'è un algoritmo di raccomandazione che usiamo da sempre, e non richiede
computer: chiedere all'amico giusto. Non a un amico qualunque: a quello con
cui, film dopo film, ci siamo sempre trovati d'accordo. Se ha amato gli stessi
film che ho amato io, e ne ha visto uno che io non conosco, il suo entusiasmo
vale una previsione. Il **filtraggio collaborativo** è questa idea resa
calcolabile: prevedere i gusti di una persona usando i giudizi delle persone
che le somigliano. Il dettaglio sorprendente è ciò che *ignora*: il sistema
non sa nulla dei film, né trama, né genere, né regista. Vede solo la tabella
dei voti, e gli basta.

Il nome è del 1992, e nasce a Xerox PARC con Tapestry, un sistema che
setacciava posta elettronica e newsgroup {cite}`goldberg1992using`. Per
decidere cosa far passare guardava le reazioni che altri lettori avevano
lasciato sui documenti, e quelle reazioni bisognava scriverle a mano. Due anni
dopo GroupLens rese automatico il passaggio successivo, cioè trovare da solo i
lettori con i gusti più vicini ai tuoi {cite}`resnick1994grouplens`. Le due
parole dicono esattamente cosa succede.
**Collaborativo** perché ognuno, mettendo un voto, senza saperlo aiuta degli
sconosciuti che gli somigliano: nessuno collabora di proposito, eppure il
lavoro è collettivo. **Filtraggio** perché di fronte a un catalogo enorme il
mestiere è lasciar passare e non produrre: di centomila titoli te ne arrivano
dieci, e il sistema è il setaccio.

## La saggezza dei vicini

La versione più diretta dell'idea si chiama filtraggio collaborativo *a
vicini* (*neighborhood-based*), dove i «vicini» non sono quelli di casa ma i
gemelli di gusto: chi ha votato come te. E ha due versioni speculari, secondo
da dove si comincia: dagli utenti simili, o dagli oggetti simili.

```{figure} ../figures/recommender-collaborative-filtering.svg
:name: fig-matrice-voti
:alt: "Griglia di cinque utenti per cinque film, con i voti da uno a cinque nelle celle riempite e sei celle lasciate vuote. La cella di Carla su Notting Hill è evidenziata in terracotta: è il voto da prevedere. Un tratteggio marca una riga (Bruno, che ha votato in modo simile a Carla) e un altro marca una colonna (Love Actually, il film votato in modo più simile a Notting Hill): sono le due strade da cui si ricava la stima."
:width: 92%

Il compito, in una griglia: quanto piacerà *Notting Hill* a Carla? Prevedere
una cella vuota significa guardare la riga di chi ha votato in modo simile sui
film che entrambi hanno visto (nel disegno, la riga di Bruno), oppure la
colonna dei film votati in modo simile dalle stesse persone (la colonna di
*Love Actually*). Le due strade hanno un nome inglese ciascuna, *user-based* e
*item-based*, e qui, seguendo la riga e la colonna che il disegno segna,
portano alla stessa previsione, due stelle: il voto di Bruno a *Notting Hill*
per la prima, quello di Carla a *Love Actually* per la seconda.
```

La {numref}`fig-matrice-voti` indica un vicino, non il vincitore di una
classifica, e conviene dire perché. Applicando alla lettera il modo standard di
misurare la somiglianza, davanti a Bruno finiscono in due: prima Dario, poi
Anna. E Dario
con Carla ha in comune **un film solo**, il che, come vedremo fra poco, basta
a farlo sembrare un gemello perfetto. Il difetto si vede già qui, su una
griglia di venticinque caselle.

Il disegno però mente su una cosa, ed è la più importante: lì le celle piene
sono la maggioranza. In un catalogo vero ognuno ha visto una frazione minuscola
dei titoli, quindi due persone qualsiasi hanno pochissimi film in comune su cui
misurare la somiglianza. Quel vuoto è la difficoltà vera del mestiere, e da lì
nasce il bisogno di riassumere ogni persona e ogni film in una scheda di pochi
numeri, che è la mossa della fattorizzazione.

`````{tab} Elementare

**Da utente a utente.** Per consigliare *Notting Hill* a Carla cerco i suoi
"gemelli di gusto": le persone che le hanno dato voti simili sui film che
entrambi hanno visto. I film che uno dei due non ha visto restano fuori dal
conto: una casella vuota dice «non l'ho visto», non «non mi è piaciuto».

Di gemelli ce n'è più d'uno e non li ascolto tutti: prendo i pochi più
somiglianti fra quelli che il film l'hanno visto, e faccio la media dei loro
voti, pesata in modo che i gemelli quasi perfetti contino più dei sosia
approssimativi. Bruno somiglia a Carla con un peso di $0{,}9$ e ha dato 2,
Elena le somiglia molto
meno, peso $0{,}2$, e ha dato 4. La previsione non è la media dei due voti, che
sarebbe 3: è
$(0{,}9 \cdot 2 + 0{,}2 \cdot 4) / (0{,}9 + 0{,}2) \approx 2{,}4$ (per la
precisione $2{,}36$), cioè quasi il voto di Bruno.

Restano due ritocchi, e senza di quelli il conto sbaglia in modo prevedibile.
Il peso di un vicino si sconta in base a quanti film ha in comune con Carla:
Dario, che ne ha uno solo, non può contare quanto chi ne ha cinquanta, e sotto
una manciata conviene rispondere che non si sa. E i voti si rimettono sullo
stesso metro prima di
mediarli, perché c'è chi dà 5 a tutto e chi non supera mai il 3: un 2 da chi di
media dà 4 è una stroncatura, un 3 da chi di media dà 2 è un elogio. Nella
media entrano questi scarti, e alla fine si sommano alla media di Carla.

**Da oggetto a oggetto.** Si può ribaltare il punto di vista: invece di
cercare utenti simili, cerco *film* simili; dove "simili" non significa stesso
genere, ma "votati in modo simile dalle stesse persone". È il celebre «chi ha
comprato questo ha comprato anche...» di Amazon: per stimare quanto ti piacerà
un film, guardo i voti che *tu* hai dato ai film che gli somigliano. Questa
variante ha un pregio pratico: i gusti delle persone cambiano, le somiglianze
tra film sono più stabili e si possono calcolare in anticipo, una volta per
tutte.

`````

`````{tab} Superiore

Ogni utente $u$ è rappresentato dalla riga $\mathbf{r}_u$ della matrice dei
voti, un vettore con una componente per film (quasi tutte mancanti). La
somiglianza fra due utenti è la **similarità del coseno** incontrata nella
{doc}`sezione di algebra lineare </Matematica/algebra-lineare>`, ristretta
all'insieme $\mathcal{I}_{uv}$ dei film votati da entrambi:

$$
\mathrm{sim}(u,v) \;=\;
\frac{\sum_{i \in \mathcal{I}_{uv}} r_{ui}\, r_{vi}}
{\sqrt{\sum_{i \in \mathcal{I}_{uv}} r_{ui}^2}\;
 \sqrt{\sum_{i \in \mathcal{I}_{uv}} r_{vi}^2}} .
$$

Restringere a $\mathcal{I}_{uv}$ non è pignoleria. Sui vettori interi il coseno
si può calcolare solo dopo aver deciso cosa mettere nelle componenti mancanti,
e la scelta corrente per il coseno «pieno», imputare zero, mette il non visto
sotto al peggiore dei voti possibili, che su una scala da 1 a 5 parte da 1: è
una decisione di modellazione, non una necessità.

Il voto previsto per l'utente $u$ sul film $i$ è la media dei voti dei vicini,
pesata per la somiglianza:

$$
\hat{r}_{ui} \;=\; \frac{\sum_{v \in \mathcal{N}_i(u)} \mathrm{sim}(u,v)\; r_{vi}}
{\sum_{v \in \mathcal{N}_i(u)} \lvert \mathrm{sim}(u,v)\rvert} ,
$$

dove $\mathcal{N}_i(u)$ è il vicinato di $u$, cioè i pochi utenti (tipicamente
qualche decina) più simili a $u$ fra quelli che hanno votato $i$, e $r_{vi}$ è
il voto del vicino $v$. Il vicinato non lo chiamiamo $k$, come farebbe la
tradizione dei $k$ vicini più prossimi: quella lettera serve qui al numero di
fattori latenti, che è tutt'altro conteggio.

C'è un guasto in agguato in questa formula, e non è quello che si direbbe. Con
$|\mathcal{I}_{uv}| = 0$ la similarità non è definita e i due utenti
semplicemente non si vedono, cioè un falso negativo, sgradevole ma
riconoscibile.
Con $|\mathcal{I}_{uv}| = 1$ la formula restituisce $\mathrm{sim}(u,v) = 1$
**sempre**, qualunque siano i due voti: anche se uno ha dato 1 e l'altro 5, il
numeratore e il denominatore coincidono. Il metodo fabbrica cioè un gemello
perfetto, con peso massimo nella media, a partire da nessuna evidenza; e
centrare i voti sposta il guasto di un passo invece di chiuderlo, perché su un
film solo la correlazione di Pearson non è nemmeno definita, e su due vale
$\pm 1$ ogni volta che lo è.
Nel regime di sparsità descritto poco fa le coppie con uno o due film in comune
sono la maggioranza delle coppie non vuote, quindi questo è il caso tipico, non
il caso limite. Il correttivo standard è lo **smorzamento per numerosità**
(*shrinkage*, o *significance weighting*): si moltiplica la similarità per
$\frac{|\mathcal{I}_{uv}|}{|\mathcal{I}_{uv}| + \beta}$, con $\beta$ da tarare
sui dati (in letteratura si va da qualche decina al centinaio), così
una somiglianza vista su due film pesa una frazione di una vista su cinquanta;
in alternativa si impone una soglia minima su $|\mathcal{I}_{uv}|$ e sotto
quella soglia si dichiara di non sapere.

Questa forma media voti grezzi, e i voti grezzi non sono confrontabili da
persona a persona: c'è chi dà 5 a tutto e chi non supera mai il 3. Sottrarre a
ciascuno la propria media è il correttivo standard, e si applica in **due
punti indipendenti**, che conviene non confondere. Nella *predizione* si media
lo scarto di ogni vicino dalla propria media, e il risultato si riporta sulla
scala di $u$:

$$
\hat{r}_{ui} \;=\; \bar{r}_u \;+\;
\frac{\sum_{v \in \mathcal{N}_i(u)} \mathrm{sim}(u,v)\,\big(r_{vi} - \bar{r}_v\big)}
{\sum_{v \in \mathcal{N}_i(u)} \lvert \mathrm{sim}(u,v)\rvert} .
$$

Nella *similarità*, invece, centrare cambia la metrica, e le cambia il nome:
se la media sottratta è calcolata sui soli film di $\mathcal{I}_{uv}$ si ottiene
esattamente la correlazione di Pearson; se è la media di *tutti* i voti
dell'utente si ottiene il coseno centrato (*mean-centered cosine*), variante
vicina ma distinta. Le due centrature sono ortogonali: si possono adottare
entrambe, una sola, o nessuna. Attenzione a una collisione di nomi che costa
un pomeriggio: gran parte della letteratura di settore, fin da GroupLens,
chiama «correlazione di Pearson» anche la seconda variante, quindi lo stesso
nome copre due formule diverse a seconda di chi lo scrive.

La variante **item-based** applica le stesse formule alle *colonne* della
matrice: similarità fra film, previsione come media dei voti di $u$ sui film
simili a $i$ {cite}`sarwar2001item`. In produzione è spesso preferita, ed è la
scelta con cui Amazon ha fatto girare il proprio motore
{cite}`linden2003amazon`: le similarità fra oggetti
sono più stabili nel tempo e precalcolabili, e il costo per singola
raccomandazione crolla, perché a richiesta resta solo da guardare i pochi film
già votati dall'utente.

`````

I vicini funzionano, e per anni hanno fatto girare i primi sistemi
commerciali. Ma pagano la sparsità, e la pagano due volte. Due persone con
gusti gemelli che per caso non hanno votato *nessun* film in comune risultano
perfette estranee: il metodo non le vede. E due persone che hanno visto un film
solo in comune risultano gemelle perfette, qualunque voto gli abbiano dato. La
ragione sta in che cosa guarda il conto della somiglianza: non mette a
confronto i due voti uno con l'altro, guarda in che direzione punta la fila dei
voti di ciascuno e ignora quanto è lunga. Con un film solo in comune ogni fila
si riduce a un numero, e due numeri positivi puntano per forza dalla stessa
parte: non c'è nessuna direzione da confrontare, e il conto risponde
«identiche» comunque, tanto a chi ha dato 1 quanto a chi ha dato 5. Il metodo,
qui, vede una somiglianza che non c'è. La radice è la stessa, cioè che il
confronto passa dai film in comune, e in una tabella quasi vuota i film in
comune sono pochissimi. Serve un modo per confrontare due persone che non passi
da lì.

## Fattori latenti: la matrice compressa

La mossa vincente del Netflix Prize, la gara raccontata nella prima pagina del
capitolo, fu cambiare il modo di guardare i dati. Invece di confrontare fra
loro le righe e le colonne della tabella, si parte da un'ipotesi: dietro quella
tabella gigantesca c'è una struttura piccola, pochi tratti di fondo che bastano
a spiegare i gusti. È la **fattorizzazione di matrici** (*matrix
factorization*) {cite}`koren2009matrix`, e «fattorizzare» è la stessa parola
della prima pagina del capitolo: scomporre in pezzi, qui una tabella al posto
di un numero. Il disegno è in {numref}`fig-matrix-factorization`.

```{figure} ../figures/matrix-factorization.svg
:name: fig-matrix-factorization
:alt: La grande matrice sparsa dei voti R è approssimata dal prodotto di due matrici strette, P con una riga per utente e Q trasposta con una colonna per film, entrambe con k fattori latenti.
:width: 95%

Tre lettere danno il nome alle tabelle: R è quella dei voti, enorme e quasi
tutta vuota; P raccoglie una scheda di pochi numeri per ogni utente (nel
disegno, il suo «profilo»); Q fa lo stesso per ogni film. Il voto previsto è il
confronto voce per voce fra due schede, una riga di P e una colonna di Q. Il
segno in mezzo è un «circa» e non un uguale, perché due tabelle strette non
possono riprodurre esattamente la grande, ed è lo scopo, perché costringere il
modello a dire tanto con pochi numeri è ciò che lo obbliga a cercare i tratti
che contano invece di ricopiare i voti. Il disegno si ferma
poi al confronto: nel modello completo si sommano anche due correzioni, quanto
quella persona vota alto in generale e quanto quel film è apprezzato in
generale. La piccola «T» accanto alla Q dice solo che quella tabella è girata
su un fianco, così che le schede dei film diventino colonne e le due tabelle
combacino.
```

`````{tab} Elementare

Ogni film si può descrivere con poche "manopole": quanto è commedia e quanto
dramma, quanto è mainstream e quanto di nicchia, quanto punta sull'azione. E
ogni persona con le *stesse* manopole. La previsione diventa un confronto fra
le due schede, voce per voce. Anna ha «commedia $0{,}9$, azione $0{,}1$», un
film ha «commedia $0{,}8$, azione $0{,}2$»: l'affinità è
$0{,}9 \cdot 0{,}8 + 0{,}1 \cdot 0{,}2 = 0{,}74$. Con un film d'azione puro
(«commedia $0{,}1$, azione $0{,}9$») verrebbe
$0{,}9 \cdot 0{,}1 + 0{,}1 \cdot 0{,}9 = 0{,}18$, e il primo è più di quattro
volte il secondo.

L'affinità però non è ancora un voto in stelle. Ci si arriva dal voto medio del
sito, corretto due volte: di quanto quella persona vota alto o basso rispetto a
tutti, e di quanto quel film è apprezzato rispetto a tutti. Se sul sito si
danno in media $3{,}4$ stelle, Anna sta mezza stella sotto e il film quattro
decimi sopra, la previsione è $3{,}4 - 0{,}5 + 0{,}4 + 0{,}74 = 4{,}04$. Tolto
di mezzo il facile, alle manopole resta l'incontro fra quella persona e quel
film.

E qui questa strada batte quella dei gemelli di gusto: la scheda c'è anche per
due persone senza un film in comune, e la tabella larga diecimila colonne
diventa una scheda lunga venti.

Il colpo di scena è che le manopole non le sceglie nessuno. Nessun esperto
etichetta i film: l'algoritmo riceve solo la tabella dei voti e cerca da sé i
numeri da mettere nelle schede, in modo che i voti già dati tornino. E non sono
numeri fra $0$ e $1$ come nell'esempio: vengono anche negativi, e una
«commedia» negativa dice che quella persona la commedia la evita. I tratti che
ne escono, che a guardarli dopo somigliano spesso a «commedia/dramma» o
«mainstream/nicchia», sono per questo detti **fattori latenti**: nascosti nei
dati, mai dichiarati da nessuno.

E le caselle vuote? Riempirle di zeri sarebbe un disastro: su una scala che
parte da 1, uno zero direbbe «peggio del peggio», mentre una casella vuota dice
«non lo so». L'algoritmo infatti non le guarda: cerca le manopole che fanno
tornare i voti *che ci sono*, e sulle vuote dice il numero che ne viene fuori.

Questo finché la gente vota. Dove nessuno mette stelle si sa soltanto che cosa
uno ha aperto, e quante volte, e la casella vuota cambia mestiere: smette di
dire «non lo so» e diventa l'unica cosa che somigli a un no, perché tutto il
resto è un sì.
Allora nessuna resta fuori dal conto, e accanto a ciascuna si scrive quanto ci
si crede: chi ha rivisto una serie dieci volte è un sì solido, chi non l'ha mai
aperta un no debolissimo, perché magari nessuno gliel'ha proposta.

`````

`````{tab} Superiore

A ogni utente $u$ si associa un vettore $\mathbf{p}_u \in \mathbb{R}^k$ e a ogni
film $i$ un vettore $\mathbf{q}_i \in \mathbb{R}^k$, con $k$ dell'ordine delle
decine, contro le decine di migliaia di colonne della matrice originale. Il voto
previsto è

$$
\hat{r}_{ui} \;=\; \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i ,
$$

dove $\mu$ è la media globale dei voti, $b_u$ il bias dell'utente (quanto
vota sopra o sotto la media), $b_i$ il bias del film (quanto è votato sopra o
sotto la media), e il prodotto scalare $\mathbf{p}_u^\top \mathbf{q}_i$ cattura
l'interazione personale tra i gusti di $u$ e i tratti di $i$. I parametri si
stimano minimizzando l'errore quadratico **sui soli voti osservati**
$\mathcal{K}$, con regolarizzazione $L_2$:

$$
\mathcal{L} \;=\; \sum_{(u,i)\in\mathcal{K}} \Big[
\big(r_{ui} - \hat{r}_{ui}\big)^2
\;+\; \lambda \big(\lVert \mathbf{p}_u\rVert^2 + \lVert \mathbf{q}_i\rVert^2 + b_u^2 + b_i^2\big)
\Big] ,
$$

dove $\lambda$ governa il compromesso tra aderenza ai voti noti e semplicità
dei fattori. Il vincolo «solo celle osservate» è ciò che distingue questo
problema dalla SVD classica dell'algebra lineare, che richiederebbe la
matrice completa: nella fattorizzazione **per feedback esplicito** i buchi non
sono zeri, sono incognite, e restano fuori dalla somma.

Attenzione a non promuovere questa frase a proprietà generale della
raccomandazione, perché sull'implicito il metodo canonico fa l'opposto. Hu,
Koren e Volinsky osservano che concentrarsi sul solo feedback raccolto
lascerebbe in mano *soltanto* esempi positivi, e che il segnale negativo, tale
e quale, sta proprio nelle celle mancanti {cite}`hu2008collaborative`. Il loro modello introduce allora due quantità distinte, e per non far
collidere le lettere chiamiamo $\pi_{ui}$ la prima: una **preferenza**
$\pi_{ui} = \mathbb{1}[n_{ui} > 0]$, che vale $1$ se un'interazione c'è stata,
e una **confidenza** $c_{ui} = 1 + \alpha\, n_{ui}$, che dice quanto crediamo
a quella preferenza (chi ha guardato una serie dieci volte è un caso più solido
di chi l'ha aperta una sera). Qui $n_{ui}$ non è un voto ma il conteggio delle
interazioni, che è tutto ciò che il feedback implicito lascia. Si minimizza

$$
\sum_{u,i} c_{ui}\big(\pi_{ui} - \mathbf{p}_u^\top \mathbf{q}_i\big)^2
+ \lambda \Big( \sum_u \lVert \mathbf{p}_u \rVert^2 + \sum_i \lVert \mathbf{q}_i \rVert^2 \Big),
$$

dove la somma corre su **tutte** le celle, osservate e no. È un cambio di
regime, non una variante: i termini diventano miliardi, la discesa stocastica
sulle triple non è più praticabile per questa loss, e i minimi quadrati
alternati (ALS) smettono di essere un'alternativa di gusto, grazie
a una precomputazione che riporta il costo per utente al numero delle sue
interazioni invece che al numero degli oggetti del catalogo. Il metodo si
chiama iALS, ha quasi vent'anni ed è tutt'altro che un cimelio: ritarato con
cura regge il confronto con quasi tutto ciò che è venuto dopo
{cite}`rendle2022revisiting`.

Sul come si ottimizza, nel caso esplicito, resta la scelta fra discesa del
gradiente stocastica sulle triple $(u, i, r_{ui})$ e ALS, che risolve in forma
chiusa alternando $\mathbf{P}$ e $\mathbf{Q}$ (fissato uno dei due, l'altro è
una regressione ridge, e ha soluzione esatta). Il criterio non è di gusto: SGD è
più semplice e più veloce sul dato sparso esplicito, ALS si parallelizza meglio
e diventa obbligato quando ogni cella conta, come appunto sull'implicito. In
nessuno dei due casi c'è la garanzia di arrivare a un minimo globale: il
problema è convesso in $\mathbf{P}$ e in $\mathbf{Q}$ *separatamente* (che è
precisamente ciò che rende sensato alternare) ma non nei due insieme, e dove si
finisce dipende anche da dove si è partiti.

`````

I fattori latenti non hanno un nome perché nessuno gliel'ha dato: sono le
coordinate che l'ottimizzazione ha trovato comode, non etichette. È il motivo
per cui una raccomandazione fattorizzata non si sa raccontare, e per cui le
spiegazioni che si leggono davvero («perché hai visto X») vengono quasi sempre
dal lato oggetto-oggetto della pagina precedente.

## Il modello in PyTorch

La palestra classica per questi modelli è **MovieLens**
{cite}`harper2015movielens`, una raccolta di voti veri messa insieme dal sito
omonimo del gruppo GroupLens dell'Università del Minnesota, attivo dal 1997. La
versione storica, MovieLens 100K, contiene 100.000 voti da 1 a 5 dati da 943
utenti a 1.682 film, con almeno 20 voti per utente: un fratello minore del
dataset Netflix, da vent'anni banco di prova standard del settore.

Qui però i voti ce li inventiamo noi, così il codice gira all'istante senza
scaricare niente. Hanno la stessa forma di quelli veri, cioè un elenco di
terzetti (utente, film, voto), e per passare a MovieLens basterebbe leggere il
suo file invece di generarli. Restano due differenze, e più avanti servono a
spiegare i risultati. I nostri voti li calcola una formula, e vengono numeri
con la virgola; quelli di MovieLens li hanno dati delle persone, in stelle
intere. E la nostra tabella la faremo piena al 10% circa, cioè più fitta
perfino di MovieLens 100K: là i 100.000 voti stanno in una tabella di
$943 \times 1.682$ celle, poco meno di un milione e seicentomila, e la
riempiono al 6,3%. Fra i banchi di prova del settore, MovieLens 100K è già uno
dei meno vuoti. Su un esempio piccolo come il nostro, con la sparsità vera non
resterebbe abbastanza da cui imparare in trenta secondi.

Il modello è la traduzione letterale dell'idea appena vista. Un
`nn.Embedding` è una tabella con una riga di numeri per ogni utente (o per
ogni film): la sua "scheda di manopole". Due tabelle tengono i fattori
latenti, due i bias (la tendenza di ciascuno a votare, o a essere votato,
sopra o sotto la media), e il confronto tra le schede è il prodotto scalare
nel `forward`: moltiplicazione voce per voce, poi somma.

```python
import torch
from torch import nn

class FattorizzazioneMatrici(nn.Module):
    def __init__(self, n_utenti, n_film, k=32):
        super().__init__()
        self.P = nn.Embedding(n_utenti, k)    # fattori latenti degli utenti
        self.Q = nn.Embedding(n_film, k)      # fattori latenti dei film
        self.b_u = nn.Embedding(n_utenti, 1)  # bias di utente
        self.b_i = nn.Embedding(n_film, 1)    # bias di film
        self.mu = nn.Parameter(torch.tensor(3.0))  # media globale
        nn.init.normal_(self.P.weight, std=0.05)   # si parte quasi dalla media
        nn.init.normal_(self.Q.weight, std=0.05)
        nn.init.zeros_(self.b_u.weight)
        nn.init.zeros_(self.b_i.weight)

    def forward(self, u, i):
        interazione = (self.P(u) * self.Q(i)).sum(dim=1)  # prodotto scalare
        return (self.mu + self.b_u(u).squeeze(1)
                + self.b_i(i).squeeze(1) + interazione)
```

I nostri voti finti nascono da fattori "veri" nascosti, che il modello non
vede: vede solo i terzetti, come vedrebbe i voti di MovieLens.

```python
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
n_utenti, n_film, k_vero = 300, 200, 4

P_vero = torch.randn(n_utenti, k_vero)   # gusti "veri", nascosti
Q_vero = torch.randn(n_film, k_vero)     # tratti "veri", nascosti

n_voti = 6_000        # 6.000 voti su 60.000 celle: matrice piena al 10% circa
u = torch.randint(0, n_utenti, (n_voti,))
i = torch.randint(0, n_film, (n_voti,))
affinita = (P_vero[u] * Q_vero[i]).sum(1)
voti = (3 + 1.2 * affinita / affinita.std()).clamp(1, 5)  # scala 1-5

# 80% per imparare, 20% messo da parte: su questi il modello non si addestra,
# e sono gli unici su cui il suo errore vorra' dire qualcosa
perm = torch.randperm(n_voti)
tr, te = perm[:4_800], perm[4_800:]
loader = DataLoader(TensorDataset(u[tr], i[tr], voti[tr]),
                    batch_size=256, shuffle=True)

celle = {*zip(u.tolist(), i.tolist())}
viste = {*zip(u[tr].tolist(), i[tr].tolist())}
ripetute = sum(c in viste for c in zip(u[te].tolist(), i[te].tolist()))
print(f"celle distinte: {len(celle)} su {n_voti} voti")
print(f"voti tenuti da parte su celle gia' viste: {ripetute} su {len(te)}")
```

```text
celle distinte: 5723 su 6000 voti
voti tenuti da parte su celle gia' viste: 91 su 1200
```

I voti si pescano a caso, quindi la stessa coppia (utente, film) può uscire due
volte, e in effetti succede. Sono pochi e non spostano le conclusioni, ma quei
91 voti meritano un nome, perché è lo stesso che la sezione sulle metriche
darà a un difetto di mezza letteratura: sono una **fuga di informazione**. Su
quelle celle il modello non deve indovinare niente, gli basta ricordare, e
l'errore che leggeremo fra poco è di quel tanto più basso del vero.

L'addestramento è un normale ciclo PyTorch. A ogni giro completo sui voti, e un
giro si chiama **epoca**, il modello prevede, si misura di quanto ha sbagliato
e l'ottimizzatore ritocca le schede. La misura è l'errore quadratico medio, la
**MSE** incontrata nella
{doc}`sezione sulle metriche </MachineLearning/metriche>`. È lo stesso metro
del Netflix Prize, meno l'ultimo passaggio: il RMSE della prima pagina del
capitolo è la radice quadrata della MSE che vedremo stampata. Una MSE di
$0{,}42$ vale quindi un errore di circa $0{,}65$ stelle, perché
$\sqrt{0{,}42} \approx 0{,}65$.

C'è poi un **freno**, che a ogni passo tira verso lo zero i numeri delle schede
e impedisce che crescano a dismisura pur di far tornare i voti già noti. Nel
codice è il `weight_decay` dell'ottimizzatore: più lo si stringe, più il
modello è costretto a spiegare i voti con schede modeste. Su quel freno
conviene essere precisi, perché la ricetta scritta sui libri dice una cosa e la
riga che gira ne fa una leggermente diversa.

`````{tab} Elementare

Il freno, come lo descrivono i libri, dovrebbe tirare verso lo zero soltanto i
numeri delle schede, cioè quelli che il modello si sta inventando. La riga che
gira davvero fa una cosa un po’ diversa, e non in un modo solo. Stringe verso
lo zero tutti i numeri del modello, a ogni passo, senza guardare chi sono, e
senza risparmiare le schede che in quel momento non sta nemmeno usando. Fra
questi numeri c'è anche quello che tiene la media dei voti di tutto il sito
(nel codice si chiama `mu`), e quello non andrebbe frenato affatto: non sta
inventando niente, sta constatando un fatto, e tirarlo verso lo zero vuol dire
spingerlo a dire il falso. E la forza con cui tira non è nemmeno quella scritta
nella ricetta, perché passa per lo stesso meccanismo con cui l'ottimizzatore
decide la lunghezza dei propri passi, e per strada viene riscalata.

Su un esempio piccolo come questo la differenza non si vede nei risultati. Fra
la ricetta scritta sui libri e le righe che girano davvero c'è però quasi
sempre un piccolo scarto, e chi scrive il codice è l'unico che può
accorgersene.

`````

`````{tab} Superiore

Tre scostamenti dalla formula, piccoli ma reali. Il `weight_decay` penalizza a
ogni passo *tutti* i fattori, non i soli $\mathbf{p}_u, \mathbf{q}_i$ che
compaiono nel batch. Tocca anche la media globale $\mu$, che il regolarizzatore
della loss non include, e la attira verso zero invece che verso la media dei
voti. E in Adam la penalità entra nel gradiente, dove viene riscalata dai
momenti adattivi: non coincide quindi con un termine $L_2$ sommato alla loss,
che è l'osservazione da cui nasce AdamW {cite}`loshchilov2019decoupled`. Per gli
scopi di questo esempio la differenza è irrilevante; chi la volesse annullare
esclude $\mu$ dalla penalità con i *param group* e passa a `torch.optim.AdamW`.

`````

```python
modello = FattorizzazioneMatrici(n_utenti, n_film, k=8)
ottim = torch.optim.Adam(modello.parameters(), lr=0.01, weight_decay=1e-4)
criterio = nn.MSELoss()

# il metro di paragone: prevedere per tutti la media dei voti di addestramento
banale = criterio(voti[tr].mean().expand_as(voti[te]), voti[te]).item()

for epoca in range(30):
    errore_tot = 0.0
    for batch_u, batch_i, batch_r in loader:
        pred = modello(batch_u, batch_i)
        loss = criterio(pred, batch_r)      # MSE sui soli voti osservati
        ottim.zero_grad()
        loss.backward()
        ottim.step()
        errore_tot += loss.item() * len(batch_r)
    if (epoca + 1) % 10 == 0:
        with torch.no_grad():               # sui voti tenuti da parte
            fuori = criterio(modello(u[te], i[te]), voti[te]).item()
        print(f"epoca {epoca + 1:2d} · MSE visti {errore_tot / len(tr):.3f}"
              f" · MSE tenuti da parte {fuori:.3f} · banale {banale:.3f}")
```

```text
epoca 10 · MSE visti 0.162 · MSE tenuti da parte 0.784 · banale 0.997
epoca 20 · MSE visti 0.043 · MSE tenuti da parte 0.549 · banale 0.997
epoca 30 · MSE visti 0.019 · MSE tenuti da parte 0.418 · banale 0.997
```

Il numero che salta all'occhio è il primo: sui voti già visti l'errore crolla a
$0{,}019$, cioè praticamente a zero. Da solo non dimostra niente. Contiamo
quanti numeri il modello ha da regolare: una scheda per ciascuno dei 300 utenti
e una per ciascuno dei 200 film, otto voci l'una (l'otto è la `k=8` del codice,
e sono un po’ più delle quattro manopole con cui i voti sono stati davvero
fabbricati, perché nella vita vera quel numero non lo si conosce e conviene
tenersi larghi), fanno $500 \times 8 = 4.000$; più una correzione per ogni
utente e per ogni film, altre $500$; più la media globale, e siamo a $4.501$. I
voti su cui si allena sono $4.800$: quasi un numero libero per voto, e con
tanta libertà impararseli a memoria è alla sua portata. È esattamente la
situazione in cui un errore basso sui voti già visti è la cosa che ci si
aspetta di vedere anche da un modello che non ha capito niente.

Il numero che conta è il secondo. Sui voti tenuti da parte, che il modello non
ha mai visto, l'errore si ferma a $0{,}42$: più che accettabile contro lo
$0{,}997$ della previsione banale «a tutti il voto medio», ma **ventidue volte
peggio** di quello che si legge sui voti già visti. Le due cose insieme sono la
misura onesta. Il modello ha imparato qualcosa di vero, perché $0{,}42$ è meno
della metà di $0{,}997$; anche se, essendo errori al quadrato, in stelle il
vantaggio si assottiglia, $0{,}65$ contro $1{,}00$. E insieme ha memorizzato
parecchio. Chi valuta un sistema di raccomandazione guardando il solo errore
sui voti già visti si sta raccontando una favola, ed è la ragione per cui il
20% dei voti viene messo da parte prima ancora di cominciare.

Una nota sul freno, già che i numeri ci sono. A `weight_decay=1e-4`, cioè
$0{,}0001$, non sta frenando quasi nulla, e lo si vede: se frenasse, l'errore
sui voti visti non arriverebbe a $0{,}019$. Portandolo a `1e-2`, cento volte
tanto, e lasciando tutto il resto com'è, il freno si sente eccome, e il modello
smette del tutto di personalizzare: rifacendo girare lo stesso programma
escono $0{,}987$ sui voti visti e $1{,}015$ su quelli tenuti da parte. Cioè
appena peggio dello $0{,}997$ della previsione banale, il che vuol dire che
tutto quell'addestramento non è servito a niente. Fra i due estremi c'è una
taratura buona, e trovarla è mestiere: non è un valore che si copia da un
libro.

Su MovieLens il procedimento è lo stesso, con due avvertenze. La prima è che i
nostri voti finti nascono da un conto, e un conto è ripetibile: rifatto due
volte dà sempre lo stesso voto. I voti veri no. La stessa persona, rivotando lo
stesso film a distanza di mesi, non dà sempre le stesse stelle, e
quell'oscillazione nessun modello può prevederla: su dati veri esiste quindi
una soglia sotto la quale l'errore non scende, per quanto si affini il modello.
Nel nostro esempio quella fonte di errore non c'è, e i numeri ne risentono. La
seconda avvertenza è quella già annunciata: la tabella vera è più vuota della
nostra, e i dataset più grandi stanno molto più in basso. Non aspettatevi
questi numeri.

## Dove il collaborativo si ferma

Due debolezze strutturali meritano di essere guardate in faccia, perché
nessun raffinamento del modello le elimina davvero.

`````{tab} Elementare

**La partenza a freddo.** Il nuovo iscritto è un perfetto sconosciuto: il
libraio che consiglia in base agli acquisti passati, con chi non ha mai
comprato nulla, è muto. Lo stesso vale per un film appena uscito: finché
nessuno lo vota non somiglia a niente, e nessun sistema collaborativo può
consigliarlo. Servirebbe sapere che film è, genere, attori, trama, ed è
l'unica cosa che il collaborativo non guarda. È il
problema della **partenza a freddo** (in inglese *cold start*, ed è il nome con
cui lo troverete scritto quasi ovunque), e spiega perché le piattaforme ti
tempestano di domande all'iscrizione («scegli tre titoli che ti piacciono»):
stanno comprando a poco prezzo le prime celle della tua riga.

Finché quelle celle non ci sono, nella scheda di chi è appena arrivato non c'è
niente che lo riguardi, e dal confronto non esce niente di personale: resta
soltanto quanto quel film piace in generale, cioè una classifica identica per
chiunque. Tanto vale sceglierla apposta, e mostrare i titoli che piacciono a
tutti. Con un'avvertenza, che è la stessa di prima: un film con dodici voti
entusiasti sembra amatissimo per la stessa ragione per cui due persone con un
film solo in comune sembrano gemelle, quindi quella classifica va fatta
contando anche quante persone hanno votato.

**La dittatura della popolarità.** I film con moltissimi voti entrano nei conti
di tutti, vengono consigliati spesso, e così raccolgono altri voti: i
ricchi diventano più ricchi. Il capolavoro di nicchia con dodici voti
entusiasti resta invisibile: proprio il titolo che il tuo amico cinefilo, lui
sì, ti avrebbe messo in mano. Rimediare si può, forzando la lista a fare posto
ai titoli poco visti, e si paga: qualche consiglio azzeccato in meno fra quelli
che avresti guardato comunque. Quanto pagarne lo decide chi progetta.

E la popolarità ha un rovescio: consigliare a tutti i titoli più visti, senza
sapere niente di nessuno, è un avversario che parecchi sistemi sofisticati non
riescono a battere. Chi ne presenta uno nuovo e non lo mette accanto a quella
lista salta la domanda più semplice.

`````

`````{tab} Superiore

**Partenza a freddo.** L'embedding di un utente o di un item senza interazioni
non compare in nessun termine della somma, quindi niente lo determina: con
l'SGD sulle sole triple resta all'inizializzazione, con ALS o con un
decadimento dei pesi finisce a zero. In nessuno dei due casi il modello
collaborativo puro ha un canale per informarlo. Le mitigazioni escono
dal paradigma: modelli *content-based* o ibridi che inizializzano l'embedding
dai metadati (genere, cast, descrizione, per gli item) o da questionari e dati
demografici (per gli utenti), oppure strategie di esplorazione che raccolgono
interazioni mirate nei primi giorni di vita. Nell'attesa che una di queste
faccia effetto, il ripiego standard è la classifica dei titoli più popolari, ed
è meno rozzo di quanto suoni. Su un utente di cui non si sa nulla il termine
$\mathbf{p}_u^\top \mathbf{q}_i$ è rumore attorno allo zero, o esattamente
zero se l'embedding ci è finito, e ciò che resta in piedi del modello è
$\mu + b_i$: un ordinamento per gradimento medio del
titolo, uguale per tutti. Il sistema una classifica non personalizzata la sta
già servendo, quindi tanto vale sceglierla apposta e sceglierla robusta,
perché $b_i$ stimato su una manciata di voti è esposto allo stesso guasto
della similarità su due film in comune.

**Bias di popolarità.** La distribuzione delle interazioni è a coda lunga, e
l'obiettivo di minimizzare l'errore medio concentra la capacità del modello
sulla testa della distribuzione, dove stanno quasi tutti i termini della
somma. Il feedback loop visto nella panoramica fa il resto: più esposizione,
più interazioni, più esposizione. Le contromisure (ripesare le coppie per
propensità inversa, penalizzare la popolarità nel punteggio, imporre quote di
diversità nella lista finale) comprano equità nella coda pagando qualche punto
di accuratezza in testa. È un compromesso da scegliere, non un difetto da
correggere una volta per tutte.

La popolarità però non è solo una patologia. Raccomandare i titoli più
popolari, senza alcuna personalizzazione e con zero parametri appresi, è anche
una **baseline difficile da battere**: è la prima riga che un revisore serio
cerca in fondo a una tabella di confronto, e la ragione per cui la cerca è che
molti metodi pubblicati non la battono. Quando lo fanno, il margine dice quanto
vale davvero la personalizzazione; senza quella riga, non lo dice niente.

`````

Nessuno dei due limiti si chiude restando dentro la sola tabella dei voti, ed è
il motivo per cui la sezione seguente va a cercare altrove: prima una rete
neurale al posto del confronto fra le schede, e non basterà; poi un modo
diverso di guardare lo stesso dato, che sulla partenza a freddo qualcosa dà.
Prima però, il riepilogo.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il filtraggio collaborativo è «chiedere all'amico giusto» reso calcolabile:
  prevede i gusti di una persona dai giudizi di chi le somiglia (oppure dai
  voti che lei stessa ha dato a film votati in modo simile), guardando solo la
  tabella dei voti: né trama, né genere, né regista.
- La **fattorizzazione** riassume ogni persona e ogni film in una scheda di
  poche manopole, e prevede il voto confrontando le due schede voce per voce,
  corretto da quanto quella persona vota alto in generale e da quanto quel film
  è apprezzato in generale. Le manopole non le sceglie nessuno: le trova
  l'algoritmo dai soli voti già dati (le celle vuote sono incognite, non zeri),
  con un freno che dovrebbe impedirgli di imparare quei voti a memoria, e che
  va tarato: qui è così largo che non stringe quasi nulla, e a memoria ne
  impara parecchi. Il «solo i voti già dati» però vale finché la gente vota:
  dove si sa soltanto che cosa uno ha aperto, il vuoto smette di essere
  un'incognita e diventa l'unico segnale negativo, con accanto quanto ci si
  crede.
- In PyTorch sono due tabelle di schede e un confronto voce per voce: poche
  righe, la stessa idea che ha vinto il Netflix Prize.
- L'errore va guardato **sui voti messi da parte**, non su quelli con cui il
  modello si è addestrato: qui fa $0{,}42$ sui voti messi da parte e $0{,}019$
  su quelli di addestramento, e solo lo $0{,}42$ dice se ha imparato o se ha
  imparato a memoria.
- Due limiti restano: di chi è appena arrivato non si sa nulla, e il libraio
  che consiglia in base agli acquisti passati è muto (**partenza a freddo**,
  e nell'attesa la cosa migliore da fare è mostrare i titoli che piacciono a
  tutti); i titoli già molto votati si consigliano da soli, e il capolavoro di
  nicchia con dodici voti entusiasti resta invisibile (**dittatura della
  popolarità**). La popolarità però fa due mestieri: è la patologia, ed è anche
  l'avversario che parecchi sistemi sofisticati non riescono a battere.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il filtraggio collaborativo prevede i gusti di un utente dai giudizi degli
  utenti (o degli oggetti) simili, usando solo la matrice dei voti: nessuna
  informazione sui contenuti.
- La **fattorizzazione di matrici** comprime la matrice in fattori latenti:
  $\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$, con loss
  MSE regolarizzata **sui soli voti osservati**. Il «solo osservati» vale per
  il feedback esplicito: sull'implicito il metodo canonico (iALS) somma su
  tutte le celle e le distingue con un peso di confidenza.
- In PyTorch il modello è due `nn.Embedding` e un prodotto scalare: poche
  righe, la stessa idea che ha vinto il Netflix Prize. Con quasi un parametro
  per voto, però, l'MSE di addestramento non è una misura: qui $0{,}019$ sui
  voti visti contro $0{,}42$ su quelli tenuti da parte.
- Limiti strutturali: **partenza a freddo** (*cold start*: senza interazioni
  il termine personalizzato è rumore e resta $\mu + b_i$, cioè una classifica
  per gradimento medio uguale per tutti; tanto vale sceglierla apposta, e
  sceglierla robusta) e **bias di popolarità** (la coda lunga resta
  invisibile). La popolarità è insieme la patologia e la baseline che molti
  metodi pubblicati non battono.
```

`````
