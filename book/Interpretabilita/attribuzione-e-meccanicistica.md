# Dentro le reti profonde: attribuzione e interpretabilità meccanicistica

C'è un esperimento diventato un piccolo classico della cattiva coscienza del
machine learning. Un classificatore addestrato a distinguere lupi da husky
sembrava funzionare benissimo, finché qualcuno non andò a guardare *dove*
posava lo sguardo: non sul muso, non sulle orecchie, ma sullo **sfondo**. Gli
husky del dataset erano quasi sempre fotografati sulla neve, i lupi no. La rete
non aveva imparato a riconoscere un lupo: aveva imparato a riconoscere la neve.
Funzionava per la ragione sbagliata, e nessuno se ne sarebbe accorto senza
aprire la scatola.

Nella sezione precedente abbiamo visto modelli **interpretabili per
costruzione**: una regressione lineare ci consegna un coefficiente per ogni
variabile, e quel numero *è* la spiegazione (leggibile a occhio nudo, come nel
capitolo sul Machine Learning classico). Una rete profonda no. Ha milioni di
parametri intrecciati, e nessuno di essi, preso da solo, dice qualcosa di
sensato. Dobbiamo cambiare domanda. Non «quanto pesa questa variabile in
generale?», ma «quanto ha contribuito *questo* ingresso a *questa*
decisione?». La risposta, sorprendentemente, è uno strumento che già
conosciamo bene: il **gradiente**.

## Saliency maps: il gradiente come mappa di importanza

Il gradiente lo abbiamo usato per *addestrare*: la derivata della loss rispetto
ai pesi ci dice come correggere il modello. Ma possiamo puntarlo altrove.
Invece di derivare rispetto ai pesi, deriviamo il punteggio della classe
predetta rispetto ai **pixel di ingresso**. Il risultato è una *saliency map*,
proposta da Simonyan, Vedaldi e Zisserman nel 2014 {cite}`simonyan2014deep`.

`````{tab} Elementare

Immagina di avere una foto e di volerti chiedere: quali pixel, se li toccassi
appena, cambierebbero di più il verdetto della rete? Se sposti di un nulla il
pixel del muso e la fiducia in «cane» crolla, quel pixel è *importante*. Se
tocchi un pixel dello sfondo e non succede niente, quello non conta. La saliency
map è esattamente questa: una mappa in bianco e nero, delle stesse dimensioni
della foto, che si accende dove un piccolo ritocco farebbe la differenza più
grande.

È come cercare i punti fragili di un castello di carte: dai un colpetto qua e
là e guardi cosa fa tremare tutta la struttura. Il difetto è che questi
colpetti, misurati un pixel alla volta, sono rumorosi: la mappa risulta piena di
puntini sparsi, un brusio granuloso da cui la forma dell'oggetto si intravede
appena.

`````

`````{tab} Superiore

Sia $S_c(X)$ il punteggio (il logit, prima della softmax) che la rete assegna
alla classe $c$ per l'immagine $X$. La saliency map è il modulo del gradiente
del punteggio rispetto all'ingresso:

$$
M = \left| \frac{\partial S_c}{\partial X} \right|,
$$

calcolato con una singola *backpropagation* fino allo strato di input anziché
fermarsi ai pesi. L'idea è una **linearizzazione locale**: nell'intorno di $X$,
$S_c(X + \delta) \approx S_c(X) + \big(\partial S_c/\partial X\big)^\top \delta$,
quindi le componenti del gradiente di modulo maggiore individuano i pixel la cui
piccola variazione altera di più il punteggio. Per un'immagine a colori si
prende in genere il massimo del modulo sui tre canali RGB.

Il limite è duplice. Primo, il gradiente è **locale**: coglie la pendenza solo
nel punto $X$, e le reti profonde sono tutt'altro che lineari. Secondo, è
**rumoroso**, perché la superficie $S_c$ ha derivate che oscillano rapidamente.
Le mappe risultano granulose, e le tecniche successive nascono quasi tutte per
domare questo rumore.

`````

## Grad-CAM: dove guarda la rete

La saliency lavora sui pixel e paga in rumore. Un'alternativa più stabile
rinuncia alla risoluzione fine e chiede una cosa più grossolana ma più
robusta: in quale *regione* dell'immagine la rete ha trovato le prove della sua
decisione? È l'idea di **Grad-CAM** (*Gradient-weighted Class Activation
Mapping*), di Selvaraju e colleghi nel 2017 {cite}`selvaraju2017grad`.

`````{tab} Elementare

Come abbiamo visto nel capitolo sulla Visione Artificiale, una rete
convoluzionale, salendo di strato in strato, smette di ragionare per pixel e
comincia a ragionare per **motivi**: l'ultimo strato convoluzionale non vede più
puntini, ma zone che «assomigliano a un muso», «assomigliano a una ruota». Ognuna
di queste mappe di attivazione è come un faretto puntato su una parte
dell'immagine.

Grad-CAM chiede al gradiente quali faretti contano per la classe che ci
interessa, e poi li accende in proporzione. Se stiamo spiegando la risposta
«cane», i faretti sul muso e sulle orecchie pesano tanto, quelli sull'erba
pesano zero. Sovrapposti alla foto, danno una macchia calda (una *heatmap*)
che dice, letteralmente, *dove* la rete ha guardato per dire «cane». È
grossolana (la risoluzione è quella dell'ultimo strato, non dei pixel), ma è
pulita e onesta: nel caso dell'husky, la macchia calda finirebbe proprio sulla
neve, smascherando l'inganno.

`````

`````{tab} Superiore

Sia $A^k \in \mathbb{R}^{u \times v}$ la $k$-esima *feature map* dell'ultimo
strato convoluzionale e $y^c$ il punteggio della classe $c$. Grad-CAM procede in
due mosse. Prima calcola un peso per ogni mappa, mediando spazialmente il
gradiente della classe rispetto a quella mappa:

$$
\alpha_k^c = \frac{1}{Z} \sum_{i}\sum_{j}
   \frac{\partial y^c}{\partial A^k_{ij}},
$$

dove $Z = u\,v$ è il numero di posizioni (un *global average pooling* del
gradiente). Poi combina le mappe pesate e tiene solo il contributo positivo:

$$
L^c_{\text{Grad-CAM}} = \mathrm{ReLU}\!\left( \sum_k \alpha_k^c\, A^k \right).
$$

Il peso $\alpha_k^c$ misura quanto la mappa $k$ conta per la classe $c$; la
$\mathrm{ReLU}$ scarta le regioni che *abbassano* il punteggio, tenendo solo
quelle che lo sostengono. La heatmap $L^c$ ha la bassa risoluzione dello strato
convoluzionale ($7\times 7$ in una ResNet su input $224\times 224$) e va
sovracampionata alle dimensioni dell'immagine per la sovrapposizione. A
differenza della saliency, non risale ai pixel: guadagna in robustezza al rumore
ciò che perde in dettaglio spaziale, e localizza in modo affidabile l'oggetto
che ha guidato la decisione.

`````

## Integrated Gradients: gli assiomi e il cammino dalla baseline

Sia la saliency sia Grad-CAM misurano il gradiente in **un solo punto**, e qui
si nasconde un problema. Se la rete è già «sicura» (il neurone è saturo, come
la parte piatta di una sigmoide), il gradiente locale è quasi zero, anche se
quell'ingresso è la ragione stessa della decisione. Sundararajan, Taly e Yan,
nel 2017, hanno affrontato la questione partendo non da un'euristica ma da due
**assiomi**: proprietà che una buona spiegazione *deve* soddisfare
{cite}`sundararajan2017axiomatic`.

Il primo è la **sensibilità**: se cambiando una variabile la predizione
cambia, quella variabile deve ricevere attribuzione non nulla. Il secondo è
l'**invarianza all'implementazione**: due reti che calcolano la stessa
funzione matematica, con architetture diverse, devono ricevere le stesse
attribuzioni (la spiegazione riguarda *cosa* la rete calcola, non *come* lo
scrive in codice). Il gradiente locale, da solo, viola la sensibilità proprio
nei casi di saturazione.

`````{tab} Elementare

Invece di misurare la pendenza solo nel punto di arrivo, immagina di partire
da un'immagine «neutra» (di solito tutta nera, la *baseline*) e di arrivare
piano piano all'immagine vera, mescolandole in tante tappe: 10% vera e 90%
nera, poi 20 e 80, e così via fino al 100%. A ogni tappa misuri la pendenza, e
alla fine fai la media. Così, anche se all'arrivo la rete è satura e non
reagisce più, hai comunque registrato la sua reazione lungo tutta la salita,
quando reagiva eccome.

Questo metodo ha una proprietà bellissima da controllare: se sommi le
attribuzioni di tutti i pixel, ottieni esattamente *quanto* la rete è passata
dalla fiducia sull'immagine nera a quella sull'immagine vera. Niente si perde e
niente si inventa: il conto torna sempre. È come dividere il conto di una cena
tra i commensali in modo che la somma delle quote faccia, al centesimo, il
totale sullo scontrino.

`````

`````{tab} Superiore

Sia $x'$ la baseline (per un'immagine, tipicamente il nero, $x' = 0$) e $x$
l'ingresso da spiegare. Gli *Integrated Gradients* integrano il gradiente lungo
il segmento rettilineo da $x'$ a $x$:

$$
\mathrm{IG}_i(x) = (x_i - x'_i)\,
   \int_0^1 \frac{\partial f\big(x' + \alpha\,(x - x')\big)}{\partial x_i}\,
   \mathrm{d}\alpha,
$$

dove $f$ è l'uscita della rete per la classe d'interesse, $\alpha \in [0,1]$
parametrizza il cammino e $\mathrm{IG}_i$ è l'attribuzione della $i$-esima
componente d'ingresso. In pratica l'integrale si approssima con una somma di
Riemann su $m$ passi. Integrando lungo il cammino, il metodo cattura anche i
gradienti *prima* della saturazione, dove il segnale è vivo, risolvendo la
cecità del gradiente locale.

La proprietà che ne fa uno strumento affidabile è la **completezza**: le
attribuzioni si sommano esattamente alla differenza di uscita tra input e
baseline,

$$
\sum_i \mathrm{IG}_i(x) = f(x) - f(x').
$$

La completezza implica la sensibilità e conferisce alle attribuzioni un
significato preciso: ciascuna è la *quota* di quel salto di punteggio imputabile
a quella componente. È l'assioma che verificheremo numericamente più avanti.

`````

## L'attenzione è una spiegazione?

C'è una tentazione naturale, per chi lavora con i Transformer del capitolo
dedicato: i pesi di **attenzione** {cite}`vaswani2017attention` sono già lì,
belli normalizzati, e sembrano dire su quali parole il modello si è
concentrato. Perché non usarli come spiegazione, gratis?

La comunità ci ha discusso a lungo. Nel 2019 Jain e Wallace, con un articolo
dal titolo programmatico *«Attention is not Explanation»*, hanno mostrato che
spesso si possono costruire distribuzioni di attenzione **molto diverse** che
portano alla **stessa** predizione: se più configurazioni dei pesi danno lo
stesso verdetto, nessuna di esse può essere *la* spiegazione. Altri (Wiegreffe
e Pinter, sempre nel 2019, con la replica *«Attention is not not
Explanation»*) hanno ribattuto che dipende da cosa si pretende: sotto vincoli
più stretti l'attenzione conserva un valore esplicativo. La morale operativa è
di **cautela**: i pesi di attenzione sono un indizio suggestivo, non una
prova; una heatmap di attenzione va letta come una traccia, non come una
confessione.

Un approccio complementare, più controllato, è il **probing**. L'idea: se una
rappresentazione interna «sa» qualcosa (poniamo, la parte del discorso di una
parola), allora un classificatore *lineare* addestrato su quella
rappresentazione dovrebbe saperlo prevedere. Si congela la rete, si estraggono
le attivazioni di uno strato e ci si allena sopra una semplice regressione
logistica per una proprietà a scelta. Se il probe riesce, l'informazione è
presente e linearmente accessibile in quello strato; se fallisce, non lo è. È
un modo economico per mappare *dove*, nella pila di strati, emergono le varie
proprietà, con l'avvertenza, discussa da Alain e Bengio e da altri, che un
probe troppo potente rischia di *imparare* lui la proprietà invece di
limitarsi a leggerla.

## Interpretabilità meccanicistica: fare reverse-engineering dei circuiti

Attribuzione e probing dicono *cosa* pesa e *dove* sta l'informazione, ma non
*come* la rete la calcola. La frontiera (giovane, ambiziosa, ancora molto
aperta) punta più in alto: **fare reverse-engineering** dei calcoli interni,
come si smonta un circuito elettronico per capire cosa fa ciascun componente.
È l'**interpretabilità meccanicistica**.

```{figure} ../figures/toy-models-superposition.svg
:name: fig-superposizione
:alt: "Due piani a due dimensioni. Nel primo, con feature dense, i due assi interni ospitano due sole feature, ortogonali fra loro, una per direzione. Nel secondo, con feature sparse, gli stessi due assi ospitano cinque feature disposte a raggiera: non sono ortogonali, si sovrappongono, ma poiché raramente sono attive insieme il modello riesce comunque a distinguerle."
:width: 92%

La sovrapposizione. Due dimensioni possono rappresentare più di due cose, a
patto che quelle cose si accendano di rado e quasi mai insieme.
```

{numref}`fig-superposizione` spiega perché smontare una rete sia difficile
oltre il previsto. La speranza naturale è che ogni neurone corrisponda a un
concetto; se invece i concetti sono più delle dimensioni e ci convivono a
raggiera, il singolo neurone risponde a un miscuglio di cose senza rapporto
fra loro, ed è esattamente ciò che si osserva guardando dentro i modelli.

```{figure} ../figures/interpretabilita-scatola-nera.svg
:name: fig-sparse-autoencoder
:alt: "A sinistra uno strato di attivazioni rappresentato come un fascio di direzioni aggrovigliate, in cui ogni neurone mescola più concetti. Una freccia attraversa uno sparse autoencoder, che proietta le stesse attivazioni in uno spazio molto più ampio ma con pochissime unità attive per volta. A destra le feature risultanti, ciascuna corrispondente a un concetto leggibile."
:width: 96%

La mossa che scioglie il groviglio. Si passa a uno spazio più largo di quello
di partenza, imponendo che pochissime unità siano accese insieme: la
sovrapposizione si srotola in feature che si possono leggere una per una.
```

La direzione di {numref}`fig-sparse-autoencoder` sembra paradossale (per
capire meglio si aumentano le dimensioni) e invece è la conseguenza diretta
della figura precedente. Se il problema è che troppe cose stanno in troppo
poco spazio, la cura è dare più spazio, e imporre con la sparsità che ciascuna
si prenda la propria direzione invece di dividerla con altre.

`````{tab} Elementare

Finora abbiamo trattato la rete come una scatola su cui bussare da fuori: le
mostri un ingresso, guardi l'uscita, misuri le reazioni. L'interpretabilità
meccanicistica apre la scatola e prova a leggere il circuito dentro.
L'obiettivo è ricostruire i **circuiti**: piccoli gruppi di neuroni collegati
che, insieme, svolgono un compito riconoscibile (un rilevatore di curve, un
pezzo che tiene il conto delle parentesi aperte in un testo).

C'è però un ostacolo curioso, chiamato **sovrapposizione**: la rete ha meno
neuroni dei concetti che deve rappresentare, e allora fa come chi ha poche
scatole e troppa roba; mette più concetti nella stessa scatola, e un singolo
neurone finisce per accendersi per cose scollegate (un po' per i gatti, un po'
per le automobili, un po' per il colore verde). Una tecnica recente, gli
*sparse autoencoder*, prova a «ri-sistemare gli scatoloni»: espande le
attivazioni in uno spazio molto più grande in cui, si spera, ogni casella
torni a rappresentare **una cosa sola** e leggibile. È un campo giovane:
promettente, ma ancora lontano dal capire una rete grande per intero.

`````

`````{tab} Superiore

Il programma dei **circuiti** è stato articolato da Olah e colleghi su
*Distill* nel 2020 {cite}`olah2020zoom`: studiare una rete come un oggetto
scientifico, individuando *feature* (direzioni nello spazio delle attivazioni
che codificano un concetto) e i *circuiti* che le collegano; sottografi di
neuroni e pesi che implementano un calcolo interpretabile, come i rilevatori
di curve nelle prime reti di visione.

L'ostacolo teorico è la **sovrapposizione** (*superposition*): una rete con
$n$ neuroni può rappresentare molte più di $n$ feature sfruttando direzioni
quasi ortogonali in $\mathbb{R}^n$, purché ciascuna feature sia rara. La
conseguenza pratica è la **polisemanticità**: un singolo neurone risponde a
stimoli non correlati, e diventa illeggibile. Bricken e colleghi, in *Towards
Monosemanticity* (Anthropic, 2023), affrontano il problema con uno **sparse
autoencoder** {cite}`bricken2023monosemanticity`: le attivazioni di uno strato
vengono ricodificate in un dizionario **sovracompleto** (molte più unità dei
neuroni originali) sotto un vincolo di **sparsità**, che spinge poche unità
attive per esempio. Le feature così estratte risultano in larga parte
**monosemantiche** (ciascuna corrisponde a un concetto singolo e nominabile) e
molto più interpretabili dei neuroni grezzi.

Il campo è nascente e va preso con l'onestà che si deve alle frontiere: i
risultati sono su modelli piccoli o su strati singoli, e nessuno ha ancora
«letto» un modello di grande scala per intero. La posta in gioco, però, è
alta, ne parliamo qui sotto.

`````

Perché tutto questo conta, e non è solo un esercizio di curiosità? Per la
**sicurezza**. Un modello linguistico di grandi dimensioni può apprendere
comportamenti che non vogliamo (inganni, scorciatoie, bias) senza che nulla,
dall'esterno, li tradisca. Poter leggere i circuiti interni significherebbe
accorgersene *prima* che si manifestino: è il ponte, che riprenderemo nel
capitolo sull'AI responsabile, tra l'interpretabilità come curiosità
scientifica e l'interpretabilità come strumento di controllo.

## Integrated Gradients coi numeri: un esempio eseguibile

Vale più di mille formule vedere la completezza tornare al centesimo. Prendiamo
una funzione giocattolo di due variabili che **satura**, per riprodurre proprio
il caso in cui il gradiente locale mente: $f(x) = \tanh(w^\top x)$ con
$w = (2, -1)$. Nel punto $x = (2, 1)$ si ha $w^\top x = 3$, e $\tanh(3) \approx
0{,}995$: siamo nella parte piatta, dove la derivata è quasi nulla. Un solo
gradiente direbbe «qui non conta niente»; gli Integrated Gradients, integrando
dal nero, recuperano l'intero contributo.

```python
import numpy as np

# funzione giocattolo che satura: f(x) = tanh(w . x)
w = np.array([2.0, -1.0])

def f(x):
    return np.tanh(w @ x)

def grad_f(x):
    z = w @ x
    return (1.0 - np.tanh(z) ** 2) * w   # regola della catena

x = np.array([2.0, 1.0])     # input da spiegare
baseline = np.zeros(2)        # baseline neutra (lo "zero")

# gradiente grezzo nel solo punto x: saturo, quasi nullo -> saliency cieca
print("gradiente in x :", np.round(grad_f(x), 4))       # [ 0.0197 -0.0099]

# Integrated Gradients: media dei gradienti lungo il cammino baseline -> x
m = 200
alphas = (np.arange(1, m + 1) - 0.5) / m   # punti medi delle m tappe
grad_medio = np.zeros(2)
for a in alphas:
    grad_medio += grad_f(baseline + a * (x - baseline))
grad_medio /= m
ig = (x - baseline) * grad_medio
print("attribuzioni IG:", np.round(ig, 4))              # [ 1.3267 -0.3317]

# assioma di completezza: la somma delle attribuzioni = f(x) - f(baseline)
print("somma IG       :", round(ig.sum(), 4))           # 0.9951
print("f(x) - f(base) :", round(f(x) - f(baseline), 4)) # 0.9951
```

Il gradiente nel punto è $(0{,}0197,\, -0{,}0099)$: minuscolo, come previsto per
un neurone saturo. Ma le attribuzioni integrate valgono $(1{,}327,\, -0{,}332)$
e la loro **somma**, $0{,}9951$, coincide al quarto decimale con
$f(x) - f(\text{baseline}) = 0{,}9951$: la completezza è verificata. Notate anche
il segno: la prima variabile ($w_1 = 2 > 0$) spinge il punteggio in alto, la
seconda ($w_2 = -1 < 0$) lo tira giù, esattamente come ci si aspetta.

## Uno sketch di Grad-CAM in PyTorch

Su una rete vera, Grad-CAM si costruisce agganciando due *hook* all'ultimo
strato convoluzionale: uno cattura le attivazioni in avanti, l'altro i gradienti
all'indietro. Ecco lo scheletro su una ResNet-18 di `torchvision`, con l'API
reale.

```python
import torch
import torch.nn.functional as F
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
target = model.layer4[-1].conv2          # ultimo strato convoluzionale

att, grad = {}, {}
target.register_forward_hook(lambda m, i, o: att.__setitem__("v", o.detach()))
target.register_full_backward_hook(
    lambda m, gi, go: grad.__setitem__("v", go[0].detach())
)

x = torch.randn(1, 3, 224, 224)          # immagine gia pre-processata
logit = model(x)                          # (1, 1000)
classe = logit.argmax(dim=1)              # classe predetta
model.zero_grad()
logit[0, classe].backward()               # gradiente della sola classe scelta

A = att["v"]                              # attivazioni  (1, C, h, w)
dY = grad["v"]                            # gradienti    (1, C, h, w)
alpha = dY.mean(dim=(2, 3), keepdim=True)  # peso per canale (global avg pool)
heatmap = F.relu((alpha * A).sum(dim=1))   # (1, h, w), solo contributi positivi
heatmap = heatmap / heatmap.max()          # normalizzata in [0, 1]
# heatmap va poi sovracampionata a 224x224 e sovrapposta all'immagine
```

Il cuore è tutto nelle ultime tre righe: `alpha` è il peso $\alpha_k^c$ (la
media spaziale del gradiente), la somma pesata delle mappe seguita dalla
`relu` è $L^c_{\text{Grad-CAM}}$, e l'ultima riga la porta in $[0,1]$ per
visualizzarla. Su un'immagine di cane la macchia calda cadrebbe sul muso; su
un husky del dataset ingannevole, sulla neve, ed è precisamente questo che
volevamo poter vedere.

```{admonition} Da ricordare
:class: important
- Le reti profonde non hanno coefficienti leggibili: l'**attribuzione** usa il
  **gradiente dell'uscita rispetto all'ingresso** per stimare quanto ogni parte
  dell'input ha pesato su una singola decisione.
- Le **saliency maps** {cite}`simonyan2014deep` sono il gradiente sui pixel:
  informative ma rumorose e locali. **Grad-CAM** {cite}`selvaraju2017grad` pesa
  le mappe dell'ultimo strato convoluzionale coi gradienti della classe e
  localizza in modo robusto *dove* guarda la CNN.
- Gli **Integrated Gradients** {cite}`sundararajan2017axiomatic` integrano il
  gradiente lungo il cammino dalla baseline all'input: fondati su assiomi
  (sensibilità, invarianza all'implementazione), risolvono la saturazione e
  soddisfano la **completezza**, $\sum_i \mathrm{IG}_i = f(x) - f(x')$.
- I **pesi di attenzione** non sono di per sé una spiegazione affidabile
  (dibattito *«Attention is not Explanation»*, 2019); il **probing** con
  classificatori lineari mappa dove sta l'informazione negli strati interni.
- L'**interpretabilità meccanicistica** (circuiti {cite}`olah2020zoom` e
  feature monosemantiche via sparse autoencoder
  {cite}`bricken2023monosemanticity`) punta a fare reverse-engineering dei
  calcoli interni. Campo giovane, ma centrale per la sicurezza degli LLM.
```
