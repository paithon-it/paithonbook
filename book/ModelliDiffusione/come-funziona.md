# Rumore e ritorno: come funziona la diffusione

Facciamo un gioco. Ti mostro una fotografia su cui ho steso un velo di
disturbo — un pulviscolo di puntini casuali — e ti faccio una domanda sola:
*quale disturbo ho aggiunto?* Non ti chiedo di ridipingere la foto, né di
dirmi cosa rappresenta: solo di indicare il pulviscolo. Sembra una richiesta
modesta, ma contiene tutto: se sai rispondere, sai anche restaurare — basta
togliere ciò che hai indicato. Adesso ripetiamo il gioco per mille livelli di
rovina, dalla grana appena percettibile alla neve televisiva totale. Chi
impara a rispondere a *ogni* livello possiede una scala che scende dal rumore
puro fino a un'immagine; e siccome ogni schermata di neve fresca è diversa
dall'altra, in fondo alla scala troverà ogni volta un'immagine diversa. Nuova.

Il capitolo si è aperto con la promessa di smontare questo giocattolo pezzo
per pezzo; questa sezione la mantiene: l'andata, il ritorno, uno sguardo
sotto il cofano, la scorciatoia di DDIM, la rete che fa il lavoro e infine
tutto il meccanismo in miniatura, funzionante, in PyTorch. La
{numref}`fig-diffusione-processo` è la mappa del viaggio.

```{figure} ../figures/diffusione-processo.svg
:name: fig-diffusione-processo
:alt: Due file di riquadri. In alto il processo in avanti, in teal, un paesaggio stilizzato che di riquadro in riquadro si copre di puntini fino a diventare rumore puro, con frecce verso destra. In basso il processo inverso, in terracotta, la rete epsilon-theta che a ogni passo toglie rumore, con frecce verso sinistra, fino a recuperare il paesaggio.
:width: 100%

I due processi della diffusione: l'andata (in alto) corrompe $x_0$ fino al
rumore puro $x_T$ con una regola fissa; il ritorno (in basso) risale la
catena con la rete $\epsilon_\theta$, un velo di rumore alla volta.
```

## L'andata: rovinare con metodo

Il processo in avanti non si impara: è una ricetta fissa, la esegue un
generatore di numeri casuali. Ma non è una distruzione qualsiasi — è una
distruzione *dosata*, e i dosaggi sono scelti con cura, perché è su di essi
che il ritorno farà affidamento. L'ingrediente è sempre lo stesso: rumore
estratto dalla campana di Gauss, la distribuzione normale incontrata nei
richiami di statistica.

`````{tab} Elementare

Un'immagine in bianco e nero è una griglia di numeri: 0 è nero, 1 è bianco.
Seguiamo un solo pixel, un grigio chiaro che vale 0,8. A ogni passo la
ricetta prevede due gesti:

1. **attenua** il valore, moltiplicandolo per un numero appena sotto 1 — nel
   passo che prendiamo a esempio, 0,99: il pixel scende a
   $0{,}8 \times 0{,}99 = 0{,}792$;
2. **aggiungi** un piccolo numero estratto a caso dalla campana di Gauss,
   riscalato di un fattore piccolo, qui 0,14. Se l'estrazione dà $-0{,}7$, il
   contributo è $-0{,}7 \times 0{,}14 \approx -0{,}10$, e il pixel finisce a
   circa $0{,}69$. Con un'estrazione diversa, poniamo $+0{,}3$, sarebbe finito
   a circa $0{,}83$.

Perché anche l'attenuazione, e non solo il rumore? Pensa a una tazza di
caffè sempre piena: a ogni giro togli un cucchiaino di caffè e ne versi uno
di latte. Il livello nella tazza non cambia mai — i numeri non esplodono —
ma il contenuto vira, giro dopo giro, dal caffè al latte. Qui il caffè è
l'immagine e il latte è il rumore: dopo mille giri, nella tazza c'è solo
latte. I conti lo confermano: moltiplicare per 0,99 mille volte lascia del
valore iniziale appena $0{,}99^{1000} \approx 0{,}00004$ — del nostro 0,8 non
resta traccia, il pixel è ormai un'estrazione pura dalla campana. Ed è
successo a *tutti* i pixel insieme: la foto è diventata neve televisiva che
non ricorda nulla di ciò che era.

`````

`````{tab} Superiore

Il processo diretto è la catena di Markov già dichiarata nel capitolo:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\
\beta_t I\right),
$$

dove $x_t$ è il dato al passo $t$, $\beta_t \in (0,1)$ è la varianza del
rumore iniettato al passo $t$ e $I$ è la matrice identità. La successione
$\beta_1, \dots, \beta_T$ è lo **schedule**: in DDPM è lineare, da
$\beta_1 = 10^{-4}$ a $\beta_T = 0{,}02$ su $T = 1000$ passi — veli
sottilissimi all'inizio, più decisi verso la fine. Nessun parametro
appreso: $q$ è fissata una volta per tutte.

Definendo $\alpha_t = 1-\beta_t$ e $\bar{\alpha}_t = \prod_{s=1}^{t}
\alpha_s$, la catena ammette una forma chiusa che salta direttamente da
$x_0$ a qualunque $x_t$:

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar{\alpha}_t}\,x_0,\
(1-\bar{\alpha}_t)\, I\right)
\quad\Longleftrightarrow\quad
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,
$$

con $\epsilon \sim \mathcal{N}(0, I)$; qui $\bar{\alpha}_t$ è la frazione di
segnale originale sopravvissuta al passo $t$ e $1-\bar{\alpha}_t$ la varianza
del rumore accumulato. La forma chiusa esiste perché la somma di gaussiane
indipendenti è ancora gaussiana (richiami di statistica): componendo $t$
passi, le varianze si sommano e i coefficienti si moltiplicano. E i dosaggi
sono calibrati perché la scala resti stabile: se $x_0$ ha varianza unitaria,
$\mathrm{Var}(x_t) = \bar{\alpha}_t + (1-\bar{\alpha}_t) = 1$ a ogni passo —
il processo è *variance-preserving*. Con lo schedule di DDPM,
$\bar{\alpha}_T \approx 4 \cdot 10^{-5}$: al passo finale
$q(x_T \mid x_0) \approx \mathcal{N}(0, I)$ per qualunque $x_0$, cioè $x_T$
è rumore gaussiano puro, indipendente dal dato di partenza.

`````

## Il ritorno: indovinare il disturbo

Ora entra in scena l'unica cosa che si impara: una rete neurale,
$\epsilon_\theta(x_t, t)$, che riceve il dato rumoroso $x_t$ e il numero del
passo $t$, e deve rispondere alla domanda del nostro gioco: *quale rumore è
stato aggiunto?* Non «com'era la foto pulita» — proprio il rumore. È una
scelta meno ovvia di quanto sembri, ed è uno dei motivi per cui DDPM
{cite}`ho2020denoising` funziona così bene.

`````{tab} Elementare

L'addestramento è un mazzo di carte per il ripasso, con le soluzioni sul
retro. Si prepara una carta così: pesca una foto vera dall'archivio, pesca
un livello di rovina a caso (poniamo il passo 700 su 1000), genera il
pulviscolo di disturbo e mescola i tre ingredienti con la ricetta
dell'andata. Sul fronte della carta: la foto rovinata e il numero 700. Sul
retro: il pulviscolo esatto che è stato usato — lo conosciamo alla
perfezione, perché l'abbiamo fabbricato noi un istante fa. La rete guarda il
fronte, propone la sua risposta, e il voto è la distanza tra il disturbo
indicato e quello vero: differenze piccole, voto alto. Milioni di carte
dopo, la rete ha imparato a rispondere a ogni livello di rovina.

Ma perché chiedere il disturbo e non direttamente la foto pulita? Prova a
metterti nei panni della rete al passo 900, davanti a una schermata quasi
tutta neve: «dimmi la foto originale» è una richiesta da veggente — dovrebbe
inventare di sana pianta dettagli che nella neve non ci sono più. «Dimmi il
disturbo» è invece un compito dello stesso formato a ogni livello: il
pulviscolo ha sempre lo stesso aspetto statistico, media zero e la stessa
ampiezza tipica, al passo 10 come al passo 990. È come interrogare uno
studente sempre con domande dello stesso tipo, invece che con quesiti la cui
difficoltà cambia in modo selvaggio. E le due risposte si equivalgono: chi
conosce il disturbo e la ricetta con cui è stato mescolato ricava la foto
con una sottrazione.

`````

`````{tab} Superiore

Il processo inverso è modellato da una gaussiana con media appresa:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\
\mu_\theta(x_t, t),\ \sigma_t^2 I\right),
$$

dove $\theta$ sono i parametri della rete e $\sigma_t^2$ è una varianza
fissata (in DDPM, $\sigma_t^2 = \beta_t$). La scelta di modellare il ritorno
con una gaussiana è legittimata dal risultato citato nel capitolo: per passi
$\beta_t$ piccoli, il vero inverso $q(x_{t-1} \mid x_t)$ è
approssimativamente gaussiano. Il contributo chiave di Ho, Jain e Abbeel
{cite}`ho2020denoising` è la **riparametrizzazione** della media: invece di
far predire alla rete $\mu_\theta$ direttamente (o la ricostruzione
$\hat{x}_0$), la si scrive in funzione del rumore stimato,

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t -
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right),
$$

dove $\epsilon_\theta(x_t, t)$ è la stima del rumore $\epsilon$ usato per
produrre $x_t$ dalla forma chiusa. Con questa scelta l'addestramento si
riduce a un errore quadratico medio:

$$
\mathcal{L}_{\text{semplice}}(\theta) = \mathbb{E}_{x_0,\, \epsilon,\, t}
\left[\, \big\lVert \epsilon - \epsilon_\theta\!\big(
\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\ t
\big) \big\rVert^2 \,\right],
$$

dove $x_0$ è un dato del training set, $t$ è uniforme su $\{1, \dots, T\}$
e $\epsilon \sim \mathcal{N}(0, I)$: si campiona una tripla, si costruisce
$x_t$ in un colpo solo con la forma chiusa (senza percorrere la catena), e
si confrontano rumore vero e rumore predetto. Si noti che predire
$\epsilon$, predire $\hat{x}_0$ o predire $\mu$ sono formulazioni legate da
relazioni affini — dato $x_t$, l'una si ricava dall'altra — ma non
equivalenti come problemi di regressione: il bersaglio $\epsilon$ ha
distribuzione $\mathcal{N}(0, I)$ *a ogni* $t$, quindi scala costante e ben
condizionata, e nelle ablazioni di DDPM produce campioni nettamente
migliori.

`````

## Generare: il viaggio da $x_T$ a $x_0$

Finito l'addestramento, il generatore è pronto. Non serve nessuna foto di
partenza: si estrae rumore puro e si percorre la scala all'indietro, un
gradino alla volta, interrogando la rete a ogni passo.

```{figure} ../figures/diffusione-denoising.gif
:name: fig-diffusione-denoising
:alt: "Animazione: un quadrato di rumore casuale in scala di grigi si trasforma progressivamente, in cinque passi etichettati da t=1000 a t=0, nella cifra 3 disegnata in pixel art."
:width: 70%

Il processo inverso su una cifra: a ogni passo la rete stima il disturbo e lo
toglie *in parte*. Nessuno ha disegnato il 3 — è emerso dal rumore.
```

La {numref}`fig-diffusione-denoising` comprime in cinque passi ciò che nel
DDPM originale ne richiede mille, ma il gesto è quello: non si costruisce
l'immagine, si **scava** togliendo rumore.

`````{tab} Elementare

La procedura è un rituale in tre mosse, ripetuto mille volte. Si parte da
una schermata di neve fresca, mai vista prima. Poi, dal passo 1.000 al passo
1: mostra alla rete la schermata e il numero del passo; fatti dire il
disturbo; toglilo — ma solo *in parte*, un velo — e aggiungi un pizzico
piccolissimo di neve nuova. All'ultimo passo, niente pizzico: solo pulizia.

Quel pizzico di neve fresca sembra un controsenso: perché sporcare ciò che
si sta pulendo? Perché nei primi passi la rete sta tirando a indovinare —
nella neve quasi pura non c'è ancora niente da vedere — e fidarsi ciecamente
della sua prima proposta congelerebbe una direzione presa alla cieca. Il
piccolo scossone tiene la partita aperta finché l'immagine non si decide da
sola; ed è il motivo per cui il risultato cambia a ogni esecuzione: neve
iniziale diversa, scossoni diversi, immagine diversa. Il costo però si vede:
mille interrogazioni della rete per *ogni* immagine — il conto salato di cui
parlava il capitolo.

`````

`````{tab} Superiore

Il campionamento *ancestrale* di DDPM percorre la catena inversa da
$t = T$ a $t = 1$: si parte da $x_T \sim \mathcal{N}(0, I)$ e si itera

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t -
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right)
+ \sigma_t z,
\qquad z \sim \mathcal{N}(0, I),
$$

dove il primo termine è la media appresa $\mu_\theta(x_t, t)$ e
$\sigma_t z$ è rumore fresco con $\sigma_t = \sqrt{\beta_t}$; al passo
finale $t = 1$ si pone $z = 0$ e si restituisce la media. Il termine
stocastico non è un vezzo: il processo inverso è esso stesso una catena di
distribuzioni, non una funzione deterministica, e campionare da
$p_\theta(x_{t-1} \mid x_t)$ richiede sia la media sia il rumore di varianza
$\sigma_t^2$. Il prezzo computazionale è esplicito: $T$ valutazioni
complete della rete per ogni campione — con $T = 1000$, tre ordini di
grandezza più di una GAN, che genera in una singola passata in avanti.

`````

## Sotto il cofano: l'evidenza e lo score

Due domande sono rimaste in sospeso. La prima: quella loss così semplice è
un colpo di fortuna o discende da un principio? La seconda, più profonda:
che cosa sta *davvero* imparando la rete, oltre a vincere al nostro gioco?

`````{tab} Elementare

Immagina una mappa sterminata in cui ogni punto è una possibile immagine —
ogni combinazione di pixel, anche le più assurde. Le immagini sensate (gatti,
muri, volti) occupano poche colline; la neve televisiva riempie la pianura
infinita tutt'attorno. La rete, allenandosi a indicare il disturbo, sta
imparando senza saperlo una **bussola**: in ogni punto della mappa, la
freccia che indica la salita — la direzione in cui ritoccare l'immagine per
renderla un po' più credibile. Generare, allora, è partire da un punto a
caso della pianura e seguire la bussola a piccoli passi, con qualche
scossone per non incastrarsi nei fossi. Per anni due scuole hanno lavorato
in parallelo — chi diceva «insegniamo a togliere il rumore» e chi diceva
«insegniamo la freccia della salita» — finché non si è capito che stavano
costruendo lo stesso oggetto con due linguaggi diversi.

`````

`````{tab} Superiore

**Da dove viene la loss.** Come per ogni modello a variabili latenti, la
log-verosimiglianza $\log p_\theta(x_0)$ (l'*evidenza*) non è calcolabile
direttamente, ma ammette un limite inferiore variazionale (ELBO) che si
decompone in una somma di divergenze KL, una per ogni passo della catena —
ciascuna confronta il vero inverso $q(x_{t-1} \mid x_t, x_0)$, gaussiano e
noto in forma chiusa, con quello appreso $p_\theta(x_{t-1} \mid x_t)$. Tra
gaussiane, ogni KL si riduce a una distanza quadratica tra medie; con la
riparametrizzazione di $\mu_\theta$ vista sopra, ogni termine diventa
$\lVert \epsilon - \epsilon_\theta \rVert^2$ moltiplicato per un peso
dipendente da $t$. La $\mathcal{L}_{\text{semplice}}$ è questo obiettivo con
i pesi posti a 1: non più un bound esatto, ma una sua versione ripesata che
nella pratica produce campioni migliori {cite}`ho2020denoising`.

**Cosa impara la rete.** Dalla forma chiusa dell'andata segue
$\nabla_{x_t} \log q(x_t \mid x_0) = -\epsilon / \sqrt{1-\bar{\alpha}_t}$:
a meno di un fattore di scala, predire il rumore equivale a stimare

$$
\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar{\alpha}_t}\;
\nabla_{x_t} \log q(x_t),
$$

dove $\nabla_{x_t} \log q(x_t)$ — il gradiente della log-densità dei dati
rumorosi, detto **score** — è esattamente la «freccia della salita» verso le
regioni più probabili. È la prospettiva *score-based* sviluppata da Yang
Song e Stefano Ermon, che Song e colleghi portano a compimento nel 2021
{cite}`song2021score`: l'andata è la discretizzazione di un'equazione
differenziale stocastica (SDE) che diffonde i dati nel rumore, e la SDE
inversa — che genera — dipende dai dati soltanto attraverso lo score. DDPM
e i modelli score-based si rivelano così due discretizzazioni dello stesso
processo continuo: una sola teoria, due dialetti.

`````

## Accelerare il ritorno: DDIM

Mille passi per un'immagine sono tanti, e la prima scorciatoia importante
arriva già nel 2021: i *Denoising Diffusion Implicit Models* (DDIM) di
Jiaming Song, Chenlin Meng e Stefano Ermon {cite}`song2021denoising`. La
promessa è notevole: **lo stesso identico modello già addestrato**, nessun
riaddestramento, campioni di qualità paragonabile in 20–50 passi invece di
1.000 — nel paper, da 10 a 50 volte più veloce in tempo di calcolo reale.

`````{tab} Elementare

Il restauratore alle prime armi solleva mille velature sottili, con mano
tremante e piccole correzioni casuali a ogni passaggio. Quello esperto ha
capito una cosa: il percorso di pulitura è sempre lo stesso film, e chi lo
conosce non ha bisogno di guardarlo fotogramma per fotogramma — può saltare
alle scene chiave. DDIM è il restauratore esperto: scende la stessa scala,
ma fermandosi solo a venti o cinquanta gradini scelti, con mano ferma e
**senza scossoni**. Niente scossoni significa anche un regalo inatteso:
il procedimento diventa ripetibile. Dalla stessa schermata di neve iniziale
esce sempre, esattamente, la stessa immagine — la neve di partenza diventa
una specie di codice dell'immagine, e sfumare da un codice a un altro fa
sfumare un'immagine nell'altra.

`````

`````{tab} Superiore

L'osservazione chiave è che $\mathcal{L}_{\text{semplice}}$ dipende dal
processo in avanti solo attraverso le marginali $q(x_t \mid x_0)$ — mai
attraverso la struttura congiunta della catena. Esiste allora un'intera
famiglia di processi **non markoviani** con le *stesse* marginali, per i
quali la rete già addestrata è altrettanto valida; Song, Meng ed Ermon la
parametrizzano con un grado di stocasticità $\eta$: per $\eta = 1$ si
recupera il campionatore di DDPM, per $\eta = 0$ il passo inverso diventa
**deterministico** — dato $x_T$, l'uscita $x_0$ è una funzione, non un
campione. E poiché la generazione non deve più simulare fedelmente una
catena markoviana passo-passo, può percorrere una sottosequenza
$\tau_1 < \dots < \tau_S$ di $\{1, \dots, T\}$ con $S \ll T$: nel paper,
$S = 50$ produce campioni vicini a quelli dei 1.000 passi di DDPM, mentre
sotto $S \approx 20$ la qualità inizia a degradare in modo visibile — un
compromesso regolabile tra costo e fedeltà. La mappa deterministica
rumore→immagine rende inoltre significative le interpolazioni in $x_T$ e la
ricostruzione (quasi) esatta di un'immagine dal suo rumore. Non è un caso
che tutto ciò ricordi la vista continua della sezione precedente: il
campionatore DDIM con $\eta = 0$ è, in effetti, una discretizzazione
dell'ODE del flusso di probabilità associata alla SDE di
{cite}`song2021score`.

`````

## Chi è $\epsilon_\theta$? Una vecchia conoscenza

Fin qui la rete è stata una scatola con due ingressi ($x_t$ e $t$) e
un'uscita della stessa forma dell'immagine. Ma quale architettura? Guardiamo
i requisiti: entra un'immagine, esce una «immagine» — la mappa del rumore
stimato, pixel per pixel, alla stessa risoluzione. È un compito
*image-to-image*, e il capitolo sulla visione artificiale ci ha già dato lo
strumento su misura: la **U-Net** di Ronneberger, Fischer e Brox
{cite}`ronneberger2015u`, nata nel 2015 per segmentare immagini biomediche.
La stessa rete che là colorava ogni pixel con la sua classe, qui gli
attribuisce la sua quota di rumore: un encoder che comprime, un decoder che
riespande, e le *skip connections* che traghettano i dettagli fini dalla
discesa alla risalita — preziose, perché il rumore è per sua natura un
dettaglio ad alta frequenza.

Resta da iniettare il tempo. Il passo $t$ entra nella rete come **embedding
sinusoidale** — lo stesso trucco degli encoding di posizione dei Transformer
{cite}`vaswani2017attention`, con $t$ al posto della posizione nella frase —
trasformato da un piccolo MLP e sommato alle feature di *ogni* blocco della
U-Net. Così una sola rete serve tutti i mille livelli di rumore: $t$ le dice
a quale punto della scala sta lavorando, se sgrossare forme globali (rumore
alto) o rifinire texture (rumore basso). La U-Net di DDPM aggiunge infine
blocchi di self-attention alle risoluzioni più basse, dove i pixel sono
pochi e guardarli tutti insieme costa poco. Teniamo a mente questa
composizione — convoluzioni più attenzione più embedding del tempo — perché
è un equilibrio provvisorio: più avanti nel capitolo vedremo l'architettura
DiT sostituire l'intera U-Net con un Transformer che lavora su patch, come
già accaduto nella visione con i ViT {cite}`dosovitskiy2021image`.

## La diffusione in miniatura: una spirale di punti

Tutto il meccanismo — schedule, forma chiusa, loss, campionamento — sta in
poche decine di righe di PyTorch, a patto di scegliere dati abbastanza
piccoli da vederci dentro. Useremo punti del piano disposti a spirale: ogni
«dato» è una coppia di coordinate, e la diffusione li disperderà in una
nuvola gaussiana per poi imparare a ridisporli. Gli ingredienti sono
*esattamente* quelli delle immagini; cambia solo la taglia.

Prima i dati e la ricetta dell'andata:

```python
import numpy as np
import torch
from torch import nn

torch.manual_seed(0)
rng = np.random.default_rng(0)

# --- Dati: 2000 punti disposti a spirale, coordinate in [-1, 1] ---
n = 2000
angolo = 3.0 * np.pi * np.sqrt(rng.uniform(size=n))    # angolo lungo la spirale
raggio = angolo / (3.0 * np.pi)                        # il raggio cresce con l'angolo
spirale = np.stack([raggio * np.cos(angolo),
                    raggio * np.sin(angolo)], axis=1)  # shape (2000, 2)
spirale += 0.02 * rng.standard_normal(spirale.shape)   # leggero spessore del tratto
x0 = torch.tensor(spirale, dtype=torch.float32)        # (2000, 2)

# --- Schedule del rumore: lo stesso di DDPM ---
T = 1000
beta = torch.linspace(1e-4, 0.02, T)       # beta_t, shape (T,)
alpha = 1.0 - beta                         # alpha_t
alpha_bar = torch.cumprod(alpha, dim=0)    # alpha_t barrato, shape (T,)

def rumorizza(x0, t, eps):
    """Forma chiusa dell'andata: x_t dato x_0, per t interi in [0, T-1]."""
    ab = alpha_bar[t].unsqueeze(1)                     # (B, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps    # (B, 2)
```

Poi la rete. Al posto della U-Net (i dati non sono immagini) basta un MLP;
il passo $t$ entra come embedding sinusoidale concatenato alle coordinate:

```python
def embedding_tempo(t, dim=16):
    """Embedding sinusoidale del passo t: da (B,) a (B, dim)."""
    freq = torch.exp(torch.arange(dim // 2) * (-np.log(10000.0) / (dim // 2)))
    ang = t.float().unsqueeze(1) * freq.unsqueeze(0)   # (B, dim/2)
    return torch.cat([ang.sin(), ang.cos()], dim=1)    # (B, dim)

class PredittoreRumore(nn.Module):
    """La rete epsilon_theta(x_t, t): un MLP al posto della U-Net."""
    def __init__(self, dim_t=16, dim_h=128):
        super().__init__()
        self.dim_t = dim_t
        self.rete = nn.Sequential(
            nn.Linear(2 + dim_t, dim_h), nn.SiLU(),
            nn.Linear(dim_h, dim_h), nn.SiLU(),
            nn.Linear(dim_h, 2),                       # stima del rumore 2D
        )

    def forward(self, x, t):
        emb = embedding_tempo(t, self.dim_t)           # (B, dim_t)
        return self.rete(torch.cat([x, emb], dim=1))   # (B, 2)
```

Il ciclo di addestramento è la loss semplice di DDPM, riga per riga: pesca
un minibatch, un livello di rumore a caso per ciascun punto, il rumore
«vero», e confronta:

```python
modello = PredittoreRumore()
ottimizzatore = torch.optim.Adam(modello.parameters(), lr=2e-3)

for passo in range(4000):
    idx = torch.randint(0, n, (256,))         # minibatch di 256 punti
    batch = x0[idx]                           # (256, 2)
    t = torch.randint(0, T, (256,))           # un livello di rumore per esempio
    eps = torch.randn_like(batch)             # il rumore "vero" (la soluzione)
    x_t = rumorizza(batch, t, eps)            # (256, 2)
    predetto = modello(x_t, t)                # (256, 2), rumore stimato
    loss = ((eps - predetto) ** 2).mean()     # MSE: la loss semplice di DDPM
    ottimizzatore.zero_grad()
    loss.backward()
    ottimizzatore.step()
    if passo % 1000 == 0:
        print(f"passo {passo:4d}  loss {loss.item():.3f}")
```

Infine il campionamento ancestrale: neve gaussiana in ingresso, mille
interrogazioni della rete, spirale in uscita:

```python
@torch.no_grad()
def campiona(n_campioni=1000):
    """Percorre la catena inversa da x_T (rumore puro) a x_0."""
    x = torch.randn(n_campioni, 2)                        # x_T ~ N(0, I)
    for t in reversed(range(T)):
        t_batch = torch.full((n_campioni,), t)            # (B,), tutti uguali a t
        eps_pred = modello(x, t_batch)                    # rumore stimato
        coeff = beta[t] / (1.0 - alpha_bar[t]).sqrt()
        media = (x - coeff * eps_pred) / alpha[t].sqrt()  # mu_theta(x_t, t)
        if t > 0:
            x = media + beta[t].sqrt() * torch.randn_like(x)  # sigma_t * z
        else:
            x = media                    # ultimo passo: niente rumore fresco
    return x                             # (n_campioni, 2)

nuovi = campiona()
print(nuovi.shape)   # torch.Size([1000, 2]): punti nuovi, disposti a spirale
```

Disegnando `nuovi` con un grafico a dispersione si vede la spirale
riemergere dalla nuvola gaussiana — punti *nuovi*, non copie del training
set. Due esperimenti valgono la pena: interrompere il campionamento a metà
strada, per vedere la forma «mezza decisa»; e passare da questi punti alle
immagini, dove l'unica modifica sostanziale è sostituire l'MLP con una U-Net
e le coppie di coordinate con griglie di pixel — schedule, loss e cicli
restano identici, carattere per carattere.

```{admonition} Da ricordare
:class: important
- L'**andata** è fissa: $q(x_t \mid x_{t-1}) =
  \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I)$ con schedule
  $\beta_t$; la forma chiusa $x_t = \sqrt{\bar{\alpha}_t}\,x_0 +
  \sqrt{1-\bar{\alpha}_t}\,\epsilon$ salta da $x_0$ a qualunque passo, e
  $x_T$ è rumore gaussiano puro.
- La rete $\epsilon_\theta(x_t, t)$ impara a predire **il rumore**, non
  l'immagine: bersaglio a scala costante per ogni $t$, loss MSE
  $\mathbb{E}\lVert\epsilon - \epsilon_\theta\rVert^2$
  {cite}`ho2020denoising` — una regressione, stabile come un problema
  supervisionato.
- Generare = partire da $x_T \sim \mathcal{N}(0,I)$ e risalire la catena in
  $T$ passi: a ogni passo si toglie il rumore stimato e si aggiunge un
  pizzico di rumore fresco (tranne all'ultimo).
- Sotto il cofano: la loss è un **ELBO ripesato**, e predire il rumore
  equivale a stimare lo **score** $\nabla_x \log q(x_t)$ — DDPM e modelli
  score-based sono due discretizzazioni della stessa SDE
  {cite}`song2021score`.
- **DDIM** {cite}`song2021denoising`: stesso modello, campionamento
  deterministico su 20–50 passi — possibile perché la loss dipende solo
  dalle marginali $q(x_t \mid x_0)$, non dalla catena markoviana.
- $\epsilon_\theta$ è una **U-Net** {cite}`ronneberger2015u` — la stessa
  della segmentazione — con il passo $t$ iniettato come embedding
  sinusoidale; nella sezione su DiT verrà sostituita da un Transformer.
```
