# Il filtraggio collaborativo

C'è un algoritmo di raccomandazione che usiamo da sempre, e non richiede
computer: chiedere all'amico giusto. Non a un amico qualunque: a quello con
cui, film dopo film, ci siamo sempre trovati d'accordo. Se ha amato gli stessi
film che ho amato io, e ne ha visto uno che io non conosco, il suo entusiasmo
vale una previsione. Il **filtraggio collaborativo** è questa idea resa
calcolabile: prevedere i gusti di una persona usando i giudizi delle persone
che le somigliano. Il dettaglio sorprendente è ciò che *ignora*: il sistema
non sa nulla dei film, né trama, né genere, né regista. Vede solo la matrice
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
mestiere non è produrre, è lasciar passare: di centomila titoli te ne arrivano
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
*item-based*, e qui portano alla stessa previsione, due stelle: il voto di
Bruno a *Notting Hill* per la prima, quello di Carla a *Love Actually* per la
seconda.
```

Il disegno indica un vicino, non il vincitore di una classifica, e vale la pena
dire perché. Applicando alla lettera la formula che vedremo fra poco, il più
somigliante a Carla non è Bruno: è Dario, che con lei ha in comune **un film
solo**, e su un film solo l'accordo è perfetto per costruzione. È lo stesso
inganno che questa pagina smonta poche righe più sotto, e capita già qui, su
una griglia di venticinque caselle.

Il disegno però mente su una cosa, ed è la più importante: lì le celle piene
sono la maggioranza. In un catalogo vero ogni utente ha visto una frazione
minuscola dei titoli, quindi due utenti qualsiasi hanno pochissimi film in
comune su cui misurare la somiglianza. Quel vuoto è la difficoltà vera del
mestiere, e da lì nasce il bisogno di riassumere persone e film in poche schede
di numeri, che è il resto di questa pagina.

`````{tab} Elementare

**Da utente a utente.** Per consigliare un film a Carla, cerco i suoi
"gemelli di gusto": le persone che hanno dato voti simili ai suoi sugli
stessi film. Se Bruno e Carla concordano su tutto ciò che entrambi hanno
visto, e Bruno ha dato 4 stelle a un film che Carla non ha ancora visto,
prevedo che anche Carla gli darà circa 4. Con più vicini, faccio una media
pesata: i gemelli quasi perfetti pesano di più dei sosia approssimativi.

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
somiglianza fra due utenti è la **similarità del coseno** incontrata nel
capitolo sui richiami di matematica, sezione *Algebra lineare*, ristretta
all'insieme $\mathcal{I}_{uv}$ dei film votati da entrambi:

$$
\mathrm{sim}(u,v) \;=\;
\frac{\sum_{i \in \mathcal{I}_{uv}} r_{ui}\, r_{vi}}
{\sqrt{\sum_{i \in \mathcal{I}_{uv}} r_{ui}^2}\;
 \sqrt{\sum_{i \in \mathcal{I}_{uv}} r_{vi}^2}} .
$$

Restringere a $\mathcal{I}_{uv}$ non è pignoleria. Sui vettori interi il coseno
si può calcolare solo dopo aver deciso cosa mettere nelle componenti mancanti,
e la scelta corrente per il coseno «pieno», imputare zero, afferma che il non
visto vale quanto il detestato: è una decisione di modellazione, non una
necessità.

Il voto previsto per l'utente $u$ sul film $i$ è la media dei voti dei vicini,
pesata per la somiglianza:

$$
\hat{r}_{ui} \;=\; \frac{\sum_{v \in \mathcal{N}_i(u)} \mathrm{sim}(u,v)\; r_{vi}}
{\sum_{v \in \mathcal{N}_i(u)} \lvert \mathrm{sim}(u,v)\rvert} ,
$$

dove $\mathcal{N}_i(u)$ è il vicinato di $u$, cioè i pochi utenti (tipicamente
qualche decina) più simili a $u$ fra quelli che hanno votato $i$, e $r_{vi}$ è
il voto del vicino $v$. La lettera $k$ la teniamo libera: da qui alla fine del
capitolo indica il numero di fattori latenti, che è tutt'altro conteggio.

C'è un guasto in agguato in questa formula, e non è quello che si direbbe. Con
$|\mathcal{I}_{uv}| = 0$ la similarità non è definita e i due utenti
semplicemente non si vedono: è un falso negativo, sgradevole ma riconoscibile.
Con $|\mathcal{I}_{uv}| = 1$ la formula restituisce $\mathrm{sim}(u,v) = 1$
**sempre**, qualunque siano i due voti: anche se uno ha dato 1 e l'altro 5, il
numeratore e il denominatore coincidono. Il metodo fabbrica cioè un gemello
perfetto, con peso massimo nella media, a partire da nessuna evidenza; e
centrando i voti la cosa si sposta appena, perché la correlazione di Pearson su
due soli item vale $\pm 1$ sempre.
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
solo in comune risultano gemelle perfette, qualunque voto gli abbiano dato: con
un film solo non c'è un andamento da confrontare, la ricetta che misura
l'accordo non ha su cosa lavorare e risponde «identiche» comunque, anche a chi
ha dato 1 e a chi ha dato 5. Il metodo, qui, vede troppo. La radice è la
stessa, cioè che il confronto passa dai film
in comune, e in una tabella quasi vuota i film in comune sono pochissimi. Serve
un modo per confrontare due persone che non passi da lì.

## Fattori latenti: la matrice compressa

La mossa vincente del Netflix Prize fu cambiare il modo di guardare i dati.
Invece di confrontare fra loro le righe e le colonne della tabella, si parte da
un'ipotesi: dietro quella tabella gigantesca c'è una struttura piccola, pochi
tratti di fondo che bastano a spiegare i gusti. È la **fattorizzazione di
matrici** (*matrix factorization*) {cite}`koren2009matrix`, illustrata in
{numref}`fig-matrix-factorization`.

```{figure} ../figures/matrix-factorization.svg
:name: fig-matrix-factorization
:alt: La grande matrice sparsa dei voti R è approssimata dal prodotto di due matrici strette, P con una riga per utente e Q trasposta con una colonna per film, entrambe con k fattori latenti.
:width: 95%

La matrice dei voti, enorme e quasi vuota, viene approssimata dal prodotto di
due matrici strette: una scheda di pochi numeri per ogni utente (nel disegno,
il suo «profilo») e una per ogni film, e il voto previsto è il confronto voce
per voce fra le due schede. Il segno in mezzo è un «circa» e non un uguale
perché due tabelle strette non possono riprodurre esattamente la grande, ed è
tutto il punto. Il disegno inoltre si ferma al prodotto: nel modello completo
si sommano anche due correzioni, quanto quella
persona vota alto in generale e quanto quel film è apprezzato in generale. La
Q ribaltata, «Qᵀ», è la stessa tabella dei film girata su un fianco, così che
le schede diventino colonne e le due tabelle combacino.
```

`````{tab} Elementare

Immagina di descrivere ogni film con poche "manopole": quanto è commedia e
quanto dramma, quanto è mainstream e quanto di nicchia, quanto punta
sull'azione. E di descrivere ogni persona con le *stesse* manopole: quanto le
piace la commedia, quanto cerca la nicchia, e così via. La previsione diventa
un confronto tra le due schede. Se Anna ha «commedia $0{,}9$, azione $0{,}1$»
e un film ha «commedia $0{,}8$, azione $0{,}2$», l'affinità si calcola voce
per voce: $0{,}9 \cdot 0{,}8 + 0{,}1 \cdot 0{,}2 = 0{,}74$. Con un film
d'azione puro («commedia $0{,}1$, azione $0{,}9$»), verrebbe
$0{,}9 \cdot 0{,}1 + 0{,}1 \cdot 0{,}9 = 0{,}18$. Il primo numero è quattro
volte il secondo, ed è così che si leggono: uno accanto all'altro, perché il
punteggio da solo non vuol dire niente. (Qui i numeri stanno fra $0$ e $1$ per
rendere l'esempio leggibile, ma non è una regola: nel programma di poco più
avanti le manopole vengono anche negative, e una manopola negativa vuol dire
semplicemente «di questo, il contrario».)

Ed è qui che si capisce **perché conviene**, cioè perché questa strada batte
quella dei gemelli di gusto. Con i vicini, per confrontare due persone bisogna
trovare i film che hanno visto entrambe, e in una tabella quasi vuota sono
pochi o nessuno. Con le schede il confronto non passa più di lì: due persone si
confrontano guardando due liste corte di numeri, dieci o venti voci, che
esistono sempre, anche se quelle due persone non hanno un solo film in comune.
La tabella larga diecimila colonne diventa una scheda lunga venti, e il vuoto
smette di essere un ostacolo.

Il colpo di scena è che le manopole **non le sceglie nessuno**. Non c'è un
esperto che etichetta i film: l'algoritmo riceve solo la tabella dei voti e
cerca da solo i numeri da mettere nelle schede, in modo che i voti già noti
tornino. I tratti che emergono (a posteriori, ispezionandoli, somigliano
spesso a "commedia/dramma" o "mainstream/nicchia") sono per questo detti
**fattori latenti**: nascosti nei dati, mai dichiarati da nessuno.

Due conseguenze di questo, che vale la pena portarsi via.

La prima riguarda le caselle vuote della tabella. Non sono zeri, e la
differenza è tutta: uno zero direbbe «questo film lo detesta», mentre una
casella vuota dice «non lo so», che è un'altra cosa. L'algoritmo infatti non
le guarda affatto: cerca le manopole che fanno tornare i voti *che ci sono*, e
sulle altre celle si limita poi a rispondere.

La seconda riguarda te. Se le manopole non le ha scelte nessuno e non hanno un
nome, allora quando l'app ti mette davanti un titolo **non è in grado di dirti
perché**: la ragione vera è un mucchietto di numeri senza nome che somigliano
ai tuoi. Non te lo nasconde, proprio non ce l'ha. Le spiegazioni che leggi
(«perché hai guardato X») sono ricostruzioni fatte dopo, plausibili e
qualche volta giuste, ma non sono il motivo per cui quel titolo è arrivato lì.

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
e quale, sta proprio nelle celle mancanti {cite}`hu2008collaborative`. Il loro
modello introduce allora due quantità distinte: una **preferenza**
$p_{ui} = \mathbb{1}[r_{ui} > 0]$, che vale $1$ se un'interazione c'è stata, e
una **confidenza** $c_{ui} = 1 + \alpha r_{ui}$, che dice quanto crediamo a
quella preferenza (chi ha guardato una serie dieci volte è un caso più solido
di chi l'ha aperta una sera). Si minimizza

$$
\sum_{u,i} c_{ui}\big(p_{ui} - \mathbf{p}_u^\top \mathbf{q}_i\big)^2
+ \lambda \Big( \sum_u \lVert \mathbf{p}_u \rVert^2 + \sum_i \lVert \mathbf{q}_i \rVert^2 \Big),
$$

dove la somma corre su **tutte** le celle, osservate e no. È un cambio di
regime, non una variante: i termini diventano miliardi, la discesa stocastica
sulle triple non è più praticabile, e i minimi quadrati alternati (ALS)
smettono di essere un'alternativa di gusto per diventare l'unica strada, grazie
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
nessuno dei due casi si arriva a un minimo globale: il problema è convesso in
$\mathbf{P}$ e in $\mathbf{Q}$ *separatamente* (che è precisamente ciò che rende
sensato alternare) ma non nei due insieme, e dove si finisce dipende anche da
dove si è partiti. È lo stesso motivo per cui, poco più avanti, un utente senza
interazioni resterà fermo alla sua inizializzazione casuale.

`````

## Il modello in PyTorch

La palestra classica per questi modelli è **MovieLens**
{cite}`harper2015movielens`: una famiglia di dataset di voti raccolti
dall'omonimo sito del gruppo GroupLens dell'Università del Minnesota, attivo
dal 1997. La versione storica, MovieLens 100K, contiene 100.000 voti da 1 a 5
dati da 943 utenti a 1.682 film, con almeno 20 voti per utente: un fratello
minore del dataset Netflix, da vent'anni banco di prova standard del settore.
Per un esempio eseguibile all'istante generiamo invece voti sintetici con la
stessa struttura, triple (utente, film, voto), così il codice gira senza
scaricare nulla; per usare MovieLens basterebbe sostituire la generazione con
la lettura del file dei voti. Due differenze da tenere a mente, perché più
avanti spiegano dei numeri. I nostri voti finti sono numeri con la virgola,
quelli di MovieLens sono stelle intere. E la nostra tabella la faremo piena al
10% circa: più fitta perfino di MovieLens 100K, che con 100.000 voti su
$943 \times 1.682$ celle sta al 6,3%, ed è un banco di prova fra i meno
sparsi. Su
un esempio così piccolo, con la sparsità vera non resterebbe abbastanza
segnale per imparare qualcosa in trenta secondi.

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

I dati sintetici nascono da fattori "veri" nascosti, che il modello non vede:
vede solo le triple, come vedrebbe i voti di MovieLens.

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
```

I voti si pescano a caso, quindi la stessa coppia (utente, film) può uscire due
volte, e in effetti succede: le celle distinte sono 5.723 invece di 6.000. Non
è un problema per l'esempio (sono due osservazioni concordi della stessa cosa),
ma è il tipo di dettaglio che su dati veri va guardato.

L'addestramento è un normale ciclo PyTorch. A ogni giro completo sui voti, e un
giro si chiama **epoca**, il modello prevede, si misura di quanto ha sbagliato
e l'ottimizzatore ritocca le schede. La misura è l'errore quadratico medio, la
**MSE** incontrata nel capitolo di Machine Learning, che è poi il metro del
Netflix Prize a meno di una radice quadrata: il RMSE dell'annuncio è la radice
della MSE che vedremo stampata, quindi una MSE di $0{,}42$ vale un errore di
circa $0{,}65$ stelle.

C'è poi un **freno**, e serve a impedire che i numeri delle schede crescano a
dismisura pur di far tornare i voti già noti: nel codice è il `weight_decay`
dell'ottimizzatore, e più è stretto, più il modello è costretto a spiegare i
voti con schede modeste. Su quel freno conviene essere precisi, perché la
ricetta scritta sulla carta dice una cosa e la riga che gira ne fa una
leggermente diversa.

`````{tab} Elementare

Sulla carta il freno tira verso lo zero i numeri delle schede, cioè quelli che
il modello si sta inventando, e solo quelli. Nel programma invece stringe verso
lo zero **tutti** i numeri del modello, a ogni passo, senza guardare chi sono.
Fra questi c'è anche il numero
che dice «su questo sito, in generale, si vota alto»: quello non andrebbe
frenato affatto, perché non sta inventando niente, sta constatando un fatto, e
tirarlo verso lo zero vuol dire spingerlo a dire il falso. Su un esempio
piccolo come questo la differenza non si vede nei risultati. Vale però la pena
saperlo: fra la ricetta scritta sulla carta e le righe che girano davvero c'è
quasi sempre un piccolo scarto, e chi scrive il codice è l'unico che può
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

```
epoca 10 · MSE visti 0.162 · MSE tenuti da parte 0.784 · banale 0.997
epoca 20 · MSE visti 0.043 · MSE tenuti da parte 0.549 · banale 0.997
epoca 30 · MSE visti 0.019 · MSE tenuti da parte 0.418 · banale 0.997
```

Vale la pena leggere questi tre numeri con calma, perché dicono tre cose
diverse e la più interessante è la meno spettacolare.

Sui voti già visti l'errore crolla a $0{,}019$, cioè praticamente a zero, e da
solo quel numero non dimostra niente: il modello ha $4.501$ numeri da regolare
(le $500$ schede da otto voci l'una, più una correzione per ogni utente e per
ogni film, più la media globale) per $4.800$ voti di addestramento, quasi uno
per voto, e con tanta libertà
impararseli a memoria è alla sua portata. È esattamente la situazione in cui
un errore basso sui dati di casa è la cosa che ci si aspetta di vedere anche da
un modello che non ha capito niente.

Il numero che conta è il secondo. Sui voti tenuti da parte, che il modello non
ha mai visto, l'errore si ferma a $0{,}42$: più che accettabile contro lo
$0{,}997$ della previsione banale «a tutti il voto medio», ma **ventidue volte
peggio** di quello che si legge sui voti di casa. Le due cose insieme sono la
misura onesta: il modello ha imparato qualcosa di vero (l'errore quadratico è
meno della metà di quello di chi non sa niente, che in stelle vuol dire
$0{,}65$ contro $1{,}00$) e insieme ha
memorizzato parecchio. Un capitolo che insegna a valutare i sistemi di
raccomandazione non può permettersi di guardare solo il primo numero, ed è la
ragione per cui il codice qui sopra mette da parte il 20% delle triple prima
ancora di cominciare.

Una nota sul freno, già che i numeri ci sono. A `weight_decay=1e-4`, cioè
$0{,}0001$, non sta frenando quasi nulla, e lo si vede: se frenasse, l'errore
sui voti visti non arriverebbe a $0{,}019$. Portandolo a `1e-2`, cento volte
tanto, e lasciando tutto il resto com'è, il freno si sente eccome, e il modello
smette del tutto di personalizzare: $0{,}987$ sui voti visti e $1{,}015$ su
quelli tenuti da parte, cioè la previsione banale. Fra i due estremi c'è una
taratura buona, e trovarla è mestiere: non è un valore che si copia da un
libro.

Su MovieLens la ricetta è la stessa, con due avvertenze. I voti veri sono
stelle intere, e un giudizio arrotondato all'intero si porta dietro un errore
che nessun modello potrà mai togliere: c'è quindi una soglia sotto la quale
non si scende, e qui non c'è. E la tabella vera è più vuota di questa (il 6,3%
contro il 10%, e i dataset più grandi stanno molto più in basso), quindi non
aspettatevi questi numeri.

## Dove il collaborativo si ferma

Due debolezze strutturali meritano di essere guardate in faccia, perché
nessun raffinamento del modello le elimina davvero.

`````{tab} Elementare

**La partenza a freddo.** Il nuovo iscritto è un perfetto sconosciuto: il
libraio che consiglia in base agli acquisti passati, con chi non ha mai
comprato nulla, è muto. Lo stesso vale per un film appena uscito: finché
nessuno lo vota, non somiglia a niente e nessun sistema collaborativo può
consigliarlo. È il problema della **partenza a freddo** (in inglese *cold
start*, ed è il nome con cui lo troverete scritto quasi ovunque), e spiega
perché le piattaforme ti tempestano di domande all'iscrizione («scegli tre
titoli che ti piacciono»): stanno comprando a poco prezzo le prime celle della
tua riga. Finché quelle celle non ci sono, la cosa più sensata da fare è anche
la più banale: mostrarti i titoli che piacciono a tutti. Non è una resa, è la
miglior risposta possibile a chi non si conosce.

**La dittatura della popolarità.** I film con moltissimi voti compaiono tra i
vicini di tutti, vengono consigliati spesso, e così raccolgono altri voti: i
ricchi diventano più ricchi. Il capolavoro di nicchia con dodici voti
entusiasti resta invisibile: proprio il titolo che il tuo amico cinefilo, lui
sì, ti avrebbe messo in mano.

`````

`````{tab} Superiore

**Partenza a freddo.** Un utente o item senza interazioni ha embedding fermo
all'inizializzazione casuale: nessun gradiente lo ha mai toccato, e il modello
collaborativo puro non ha alcun canale per informarlo. Le mitigazioni escono
dal paradigma: modelli *content-based* o ibridi che inizializzano l'embedding
dai metadati (genere, cast, descrizione, per gli item) o da questionari e dati
demografici (per gli utenti), oppure strategie di esplorazione che raccolgono
interazioni mirate nei primi giorni di vita. Nell'attesa che una di queste
faccia effetto, il ripiego standard è la classifica dei titoli più popolari, ed
è meglio di quanto suoni: su un utente di cui non si sa nulla, una
fattorizzazione con embedding all'inizializzazione produce un ordinamento
casuale, e la popolarità la batte largamente.

**Bias di popolarità.** La distribuzione delle interazioni è a coda lunga, e
l'obiettivo di minimizzare l'errore medio concentra la capacità del modello
sulla testa della distribuzione, dove stanno quasi tutti i termini della
somma. Il feedback loop visto nella panoramica fa il resto: più esposizione,
più interazioni, più esposizione. Le contromisure (ripesare le coppie per
propensità inversa, penalizzare la popolarità nel punteggio, imporre quote di
diversità nella lista finale) comprano equità nella coda pagando qualche punto
di accuratezza in testa. È un compromesso da scegliere, non un difetto da
correggere una volta per tutte.

E conviene tenere presente il rovescio, perché in questo capitolo la popolarità
compare quasi solo come patologia. Raccomandare i titoli più popolari, senza
alcuna personalizzazione e con zero parametri appresi, è anche una **baseline
difficile da battere**: è la prima riga che un revisore serio cerca in fondo a
una tabella di confronto, e la ragione per cui la cerca è che molti metodi
pubblicati non la battono. Quando lo fanno, il margine dice quanto vale
davvero la personalizzazione; senza quella riga, non lo dice niente.

`````

Nessuno dei due limiti si chiude restando dentro la sola tabella dei voti, ed è
il motivo per cui la sezione seguente va a cercare altrove: prima una rete al
posto del confronto fra le schede, poi un modo diverso di guardare lo stesso
dato. Prima però, il riepilogo.

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
  va tarato: qui è lasco, e a memoria ne impara parecchi.
- In PyTorch sono due tabelle di schede e un confronto voce per voce: poche
  righe, la stessa idea che ha vinto il Netflix Prize.
- L'errore va guardato **sui voti messi da parte**, non su quelli con cui il
  modello si è addestrato: sui secondi qui fa $0{,}019$, sui primi $0{,}42$, e
  solo il secondo numero dice se ha imparato o se ha imparato a memoria.
- Due limiti restano: di chi è appena arrivato non si sa nulla, e il libraio
  che consiglia in base agli acquisti passati è muto (**partenza a freddo**,
  e nell'attesa la cosa migliore da fare è mostrare i titoli che piacciono a
  tutti); i titoli già molto votati si consigliano da soli, e il capolavoro di
  nicchia con dodici voti entusiasti resta invisibile (**dittatura della
  popolarità**).
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
- Limiti strutturali: **partenza a freddo** (*cold start*: nessuna
  interazione, nessun consiglio;
  ripiego standard, la popolarità) e **bias di popolarità** (la coda lunga
  resta invisibile). La popolarità è insieme la patologia e la baseline che
  molti metodi pubblicati non battono.
```

`````
