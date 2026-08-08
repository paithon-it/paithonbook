# Il filtraggio collaborativo

C'è un algoritmo di raccomandazione che usiamo da sempre, e non richiede
computer: chiedere all'amico giusto. Non a un amico qualunque: a quello con
cui, film dopo film, ci siamo sempre trovati d'accordo. Se ha amato gli stessi
film che ho amato io, e ne ha visto uno che io non conosco, il suo entusiasmo
vale una previsione. Il **filtraggio collaborativo** è questa idea resa
calcolabile: prevedere i gusti di una persona usando i giudizi delle persone
che le somigliano. Il dettaglio sorprendente è ciò che *ignora*: il sistema
non sa nulla dei film, né trama, né genere, né regista. Vede solo la matrice
dei voti, e le basta.

## La saggezza dei vicini

La versione più diretta dell'idea si chiama filtraggio collaborativo *a
vicini* (*neighborhood-based*), e viene in due simmetrie: partire dagli
utenti simili, o dagli oggetti simili.

```{figure} ../figures/recommender-collaborative-filtering.svg
:name: fig-matrice-voti
:alt: "Matrice con gli utenti sulle righe e i film sulle colonne, e nelle celle i voti già espressi. Una cella è evidenziata come incognita: il voto che l'utente darebbe a un film che non ha visto. Le righe degli utenti con gusti simili sono marcate, e sono quelle da cui si ricava la stima."
:width: 92%

Il compito, in una griglia. Quasi tutte le celle sono vuote, e prevederne una
significa guardare le righe di chi ha votato in modo simile per i film che
entrambi hanno visto.
```

Il vuoto di {numref}`fig-matrice-voti` è la difficoltà vera, non un dettaglio
di rappresentazione. In un catalogo reale ogni utente ha visto una frazione
minuscola dei titoli, quindi due utenti qualsiasi hanno pochissimi film in
comune su cui misurare la somiglianza: è da qui che nasce il bisogno dei
fattori latenti della prossima sezione.

`````{tab} Elementare

**Da utente a utente.** Per consigliare un film ad Anna, cerco i suoi
"gemelli di gusto": le persone che hanno dato voti simili ai suoi sugli
stessi film. Se Carla e Anna concordano su tutto ciò che entrambe hanno
visto, e Carla ha dato 4 stelle a un film che Anna non ha ancora visto,
prevedo che anche Anna gli darà circa 4. Con più vicini, faccio una media
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

Ogni utente $u$ è rappresentato dalla riga $R_u$ della matrice dei voti, un
vettore con una componente per film (quasi tutte mancanti). La somiglianza
tra due utenti è la **similarità del coseno** incontrata nel capitolo di
richiami, sezione *Algebra lineare*, ristretta all'insieme $I_{uv}$ dei film
votati da entrambi (sui vettori interi, pieni di componenti mancanti, non
sarebbe nemmeno definita):

$$
\mathrm{sim}(u,v) \;=\;
\frac{\sum_{i \in I_{uv}} r_{ui}\, r_{vi}}
{\sqrt{\sum_{i \in I_{uv}} r_{ui}^2}\;\sqrt{\sum_{i \in I_{uv}} r_{vi}^2}} .
$$

Il voto previsto per l'utente $u$ sul film $i$ è la media dei voti dei vicini,
pesata per la somiglianza:

$$
\hat{r}_{ui} \;=\; \frac{\sum_{v \in N_i(u)} \mathrm{sim}(u,v)\; r_{vi}}
{\sum_{v \in N_i(u)} \lvert \mathrm{sim}(u,v)\rvert} ,
$$

dove $N_i(u)$ è l'insieme dei $k$ utenti più simili a $u$ tra quelli che
hanno votato $i$, e $r_{vi}$ è il voto del vicino $v$.

Questa forma media voti grezzi, e i voti grezzi non sono confrontabili da
persona a persona: c'è chi dà 5 a tutto e chi non supera mai il 3. Sottrarre a
ciascuno la propria media è il correttivo standard, e si applica in **due
punti indipendenti**, che conviene non confondere. Nella *predizione* si media
lo scarto di ogni vicino dalla propria media, e il risultato si riporta sulla
scala di $u$:

$$
\hat{r}_{ui} \;=\; \bar{r}_u \;+\;
\frac{\sum_{v \in N_i(u)} \mathrm{sim}(u,v)\,\big(r_{vi} - \bar{r}_v\big)}
{\sum_{v \in N_i(u)} \lvert \mathrm{sim}(u,v)\rvert} .
$$

Nella *similarità*, invece, centrare cambia la metrica, e le cambia il nome:
se la media sottratta è calcolata sui soli film di $I_{uv}$ si ottiene
esattamente la correlazione di Pearson; se è la media di *tutti* i voti
dell'utente si ottiene il coseno centrato (*mean-centered cosine*), variante
vicina ma distinta. Le due centrature sono ortogonali: si possono adottare
entrambe, una sola, o nessuna.

La variante **item-based** applica le stesse formule alle *colonne* della
matrice: similarità tra film, previsione come media dei voti di $u$ sui film
simili a $i$. In produzione è spesso preferita: le similarità tra oggetti
sono più stabili nel tempo e precalcolabili, e il costo per singola
raccomandazione crolla.

`````

I vicini funzionano, e per anni hanno fatto girare i primi sistemi
commerciali. Ma pagano la sparsità: due utenti con gusti gemelli che per caso
non hanno votato *nessun* film in comune risultano perfetti estranei. Serve
un modo per vedere la somiglianza anche dove i voti non si sovrappongono.

## Fattori latenti: la matrice compressa

La mossa vincente del Netflix Prize fu cambiare rappresentazione. Invece di
confrontare direttamente righe e colonne, si assume che dietro la gigantesca
matrice dei voti ci sia una struttura piccola: pochi tratti fondamentali che
spiegano i gusti. È la **fattorizzazione di matrici** (*matrix
factorization*) {cite}`koren2009matrix`, illustrata in
{numref}`fig-matrix-factorization`.

```{figure} ../figures/matrix-factorization.svg
:name: fig-matrix-factorization
:alt: La grande matrice sparsa dei voti R è approssimata dal prodotto di due matrici strette, P con una riga per utente e Q trasposta con una colonna per film, entrambe con k fattori latenti.
:width: 95%

La matrice dei voti, enorme e quasi vuota, viene approssimata dal prodotto di
due matrici strette: il voto previsto per una coppia (utente, film) è il
prodotto scalare tra la riga dell'utente e la colonna del film.
```

`````{tab} Elementare

Immagina di descrivere ogni film con poche "manopole": quanto è commedia e
quanto dramma, quanto è mainstream e quanto di nicchia, quanto punta
sull'azione. E di descrivere ogni persona con le *stesse* manopole: quanto le
piace la commedia, quanto cerca la nicchia, e così via. La previsione diventa
un confronto tra le due schede. Se Anna ha «commedia $0{,}9$, azione $0{,}1$»
e un film ha «commedia $0{,}8$, azione $0{,}2$», l'affinità si calcola voce
per voce: $0{,}9 \cdot 0{,}8 + 0{,}1 \cdot 0{,}2 = 0{,}74$ (alta). Con un film
d'azione puro («commedia $0{,}1$, azione $0{,}9$»), verrebbe
$0{,}9 \cdot 0{,}1 + 0{,}1 \cdot 0{,}9 = 0{,}18$: bassa.

Il colpo di scena è che le manopole **non le sceglie nessuno**. Non c'è un
esperto che etichetta i film: l'algoritmo riceve solo la tabella dei voti e
cerca da solo i numeri da mettere nelle schede, in modo che i voti già noti
tornino. I tratti che emergono (a posteriori, ispezionandoli, somigliano
spesso a "commedia/dramma" o "mainstream/nicchia") sono per questo detti
**fattori latenti**: nascosti nei dati, mai dichiarati da nessuno.

`````

`````{tab} Superiore

A ogni utente $u$ si associa un vettore $P_u \in \mathbb{R}^k$ e a ogni film
$i$ un vettore $Q_i \in \mathbb{R}^k$, con $k$ dell'ordine delle decine,
contro le decine di migliaia di colonne della matrice originale. Il voto
previsto è

$$
\hat{r}_{ui} \;=\; \mu + b_u + b_i + P_u^\top Q_i ,
$$

dove $\mu$ è la media globale dei voti, $b_u$ il bias dell'utente (quanto
vota sopra o sotto la media), $b_i$ il bias del film (quanto è votato sopra o
sotto la media), e il prodotto scalare $P_u^\top Q_i$ cattura l'interazione
personale tra i gusti di $u$ e i tratti di $i$. I parametri si stimano
minimizzando l'errore quadratico **sui soli voti osservati** $\mathcal{K}$,
con regolarizzazione $L_2$:

$$
\mathcal{L} \;=\; \sum_{(u,i)\in\mathcal{K}} \Big[
\big(r_{ui} - \hat{r}_{ui}\big)^2
\;+\; \lambda \big(\lVert P_u\rVert^2 + \lVert Q_i\rVert^2 + b_u^2 + b_i^2\big)
\Big] ,
$$

dove $\lambda$ governa il compromesso tra aderenza ai voti noti e semplicità
dei fattori. Il vincolo «solo celle osservate» è ciò che distingue il
problema dalla SVD classica dell'algebra lineare, che richiederebbe la
matrice completa: qui i buchi non sono zeri, sono incognite. Si ottimizza con
discesa del gradiente stocastica sulle triple $(u, i, r_{ui})$ o con minimi
quadrati alternati (ALS), che risolve in forma chiusa alternando $P$ e $Q$.

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
la lettura del file dei voti.

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

loader = DataLoader(TensorDataset(u, i, voti), batch_size=256, shuffle=True)
```

L'addestramento è un normale ciclo PyTorch: a ogni giro completo sui voti (un
giro si chiama **epoca**) il modello prevede, l'errore quadratico medio (la
MSE incontrata nel capitolo di Machine Learning) misura di quanto sbaglia, e
l'ottimizzatore ritocca le schede. Il freno che impedisce ai numeri delle
schede di crescere a dismisura pur di imparare i voti a memoria (nella scheda
Superiore qui sopra è il termine $\lambda$ della loss) è affidato al
`weight_decay` dell'ottimizzatore.

Su quel freno conviene essere precisi, perché il codice non fa esattamente
quello che promette la riga qui sopra.

`````{tab} Elementare

Il freno del codice e il freno di cui parlavamo non sono la stessa identica
cosa. Quello del codice stringe a ogni passo *tutti* i numeri del modello,
compresi quelli che andrebbero lasciati liberi di andare dove vogliono, e
quanto stringe dipende anche da come sta lavorando in quel momento la
procedura che aggiorna i numeri. Su un esempio piccolo come questo la
differenza non si vede nei risultati. Vale però la pena saperlo: fra la
formula scritta su carta e le righe che girano davvero c'è quasi sempre un
piccolo scarto, e chi scrive il codice è l'unico che può accorgersene.

`````

`````{tab} Superiore

Tre scostamenti dalla formula, piccoli ma reali. Il `weight_decay` penalizza a
ogni passo *tutti* i fattori, non i soli $P_u, Q_i$ che compaiono nel batch.
Tocca anche la media globale $\mu$, che il regolarizzatore della loss non
include, e la attira verso zero invece che verso la media dei voti. E in Adam
la penalità entra nel gradiente, dove viene riscalata dai momenti adattivi:
non coincide quindi con un termine $L_2$ sommato alla loss, che è
l'osservazione da cui nasce AdamW {cite}`loshchilov2019decoupled`. Per gli
scopi di questo esempio la differenza è irrilevante; chi la volesse annullare
esclude $\mu$ dalla penalità con i *param group* e passa a `torch.optim.AdamW`.

`````

```python
modello = FattorizzazioneMatrici(n_utenti, n_film, k=8)
ottim = torch.optim.Adam(modello.parameters(), lr=0.01, weight_decay=1e-4)
criterio = nn.MSELoss()

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
        print(f"epoca {epoca + 1:2d} · MSE {errore_tot / n_voti:.3f}")
```

In poche epoche l'errore scende molto sotto quello della previsione banale
"voto medio per tutti": gli embedding hanno ricostruito, dai soli voti, una
versione compressa dei fattori nascosti. Su MovieLens la ricetta è identica;
cambiano solo gli indici e la pazienza richiesta.

## Dove il collaborativo si ferma

Due debolezze strutturali meritano di essere guardate in faccia, perché
nessun raffinamento del modello le elimina davvero.

`````{tab} Elementare

**La partenza a freddo.** Il nuovo iscritto è un perfetto sconosciuto: il
libraio che consiglia in base agli acquisti passati, con chi non ha mai
comprato nulla, è muto. Lo stesso vale per un film appena uscito: finché
nessuno lo vota, non somiglia a niente e nessun sistema collaborativo può
consigliarlo. È il problema del *cold start*, e spiega perché le piattaforme
ti tempestano di domande all'iscrizione («scegli tre titoli che ti
piacciono»): stanno comprando a poco prezzo le prime celle della tua riga.

**La dittatura della popolarità.** I film con moltissimi voti compaiono tra i
vicini di tutti, vengono consigliati spesso, e così raccolgono altri voti: i
ricchi diventano più ricchi. Il capolavoro di nicchia con dodici voti
entusiasti resta invisibile: proprio il titolo che il tuo amico cinefilo, lui
sì, ti avrebbe messo in mano.

`````

`````{tab} Superiore

**Cold start.** Un utente o item senza interazioni ha embedding fermo
all'inizializzazione casuale: nessun gradiente lo ha mai toccato, e il modello
collaborativo puro non ha alcun canale per informarlo. Le mitigazioni escono
dal paradigma: modelli *content-based* o ibridi che inizializzano l'embedding
dai metadati (genere, cast, descrizione, per gli item) o da questionari e dati
demografici (per gli utenti), oppure strategie di esplorazione che raccolgono
interazioni mirate nei primi giorni di vita.

**Bias di popolarità.** La distribuzione delle interazioni è a coda lunga, e
l'obiettivo di minimizzare l'errore medio concentra la capacità del modello
sulla testa della distribuzione, dove stanno quasi tutti i termini della
somma. Il feedback loop visto nella panoramica fa il resto: più esposizione,
più interazioni, più esposizione. Le contromisure (ripesare le coppie per
propensità inversa, penalizzare la popolarità nel punteggio, imporre quote di
diversità nella lista finale) comprano equità nella coda pagando qualche punto
di accuratezza in testa. È un compromesso da scegliere, non un difetto da
correggere una volta per tutte.

`````

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
  con un freno che gli impedisce di imparare quei voti a memoria.
- In PyTorch sono due tabelle di schede e un confronto voce per voce: poche
  righe, la stessa idea che ha vinto il Netflix Prize.
- Due limiti restano: di chi è appena arrivato non si sa nulla, e il libraio
  che consiglia in base agli acquisti passati è muto (**partenza a freddo**);
  i titoli già molto votati si consigliano da soli, e il capolavoro di nicchia
  con dodici voti entusiasti resta invisibile (**dittatura della popolarità**).
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il filtraggio collaborativo prevede i gusti di un utente dai giudizi degli
  utenti (o degli oggetti) simili, usando solo la matrice dei voti: nessuna
  informazione sui contenuti.
- La **fattorizzazione di matrici** comprime la matrice in fattori latenti:
  $\hat{r}_{ui} = \mu + b_u + b_i + P_u^\top Q_i$, con loss
  MSE regolarizzata **sui soli voti osservati**.
- In PyTorch il modello è due `nn.Embedding` e un prodotto scalare: poche
  righe, la stessa idea che ha vinto il Netflix Prize.
- Limiti strutturali: **cold start** (nessuna interazione, nessun consiglio)
  e **bias di popolarità** (la coda lunga resta invisibile).
```

`````
