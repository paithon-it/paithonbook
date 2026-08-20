# Reinforcement learning basato su modello

La prima volta che ti siedi a un tavolo davanti a un gioco da tavolo nuovo,
dopo un paio di mani hai già smesso di muovere a caso. Non perché tu abbia
giocato migliaia di partite: ne bastano due o tre perché la testa cominci a
fare da sola una cosa preziosa (*provare le mosse prima di farle*). «Se scarto
questa carta lui pesca e chiude… allora no.» La mossa cattiva muore
nell'immaginazione, senza costarti la partita. È questa la differenza, ancora
oggi imbarazzante, fra un essere umano e un agente come il DQN di tre sezioni
fa: a noi bastano pochi minuti per capire *Breakout* (il gioco dei mattoncini da
abbattere con una pallina), all'agente servono decine di milioni di fotogrammi.
La parola tecnica per questa distanza è **sample efficiency**, l'efficienza nei
campioni (quanta esperienza serve per imparare) ed è il problema che questa
sezione affronta di petto.

Tutti gli algoritmi visti finora (Q-learning, DQN, i metodi a gradiente di
policy) condividono una scelta implicita: imparano *provando per davvero*.
Provano un'azione nel mondo, guardano cosa succede, aggiustano. Non si
costruiscono mai una copia del gioco da consultare in privato. Sono metodi
**model-free**, «senza modello». Qui cambiamo strategia: costruiamo prima un
modello dell'ambiente e poi lo usiamo per *pianificare*, cioè per provare le
mosse nella testa, come al tavolo da gioco.

## Provare per davvero o provare nella testa

```{figure} ../figures/model-based-loop.svg
:name: fig-model-based-loop
:alt: Anello a quattro blocchi. L'esperienza reale raccolta dall'ambiente addestra un modello appreso della dinamica e della ricompensa; dal modello si srotolano traiettorie immaginate; queste aggiornano policy e valore senza toccare l'ambiente; la policy agisce di nuovo nel mondo. La freccia dalle traiettorie immaginate verso policy e valore è arancione, ed è l'unica del giro a non toccare il mondo vero.
:width: 90%

Il giro dell'agente che si costruisce un simulatore. Le mosse fatte davvero
servono ad addestrare il simulatore; lì dentro se ne immaginano tante altre, e
sono quelle a migliorare la strategia. La freccia arancione è il pezzo
immaginato: l'unico che non costa un solo passo nel mondo vero.
```

Il disegno mostra due giri, non uno. Il primo tocca il mondo vero: si agisce, si
guarda cosa succede, si usa quel poco per aggiustare il simulatore. Il secondo
(quello con la freccia arancione) vive solo nella testa dell'agente. Tutto il
gioco sta nel far girare molto il secondo pagando poco il primo.

`````{tab} Elementare

Immagina due allievi che imparano a guidare. Il primo impara solo
schiantandosi: prova una manovra sull'auto vera, se va male paga il danno, e
solo così capisce che non andava fatta. Il secondo, dopo qualche giro, si è
costruito in testa un piccolo simulatore della macchina («se sterzo così a
questa velocità, il posteriore scappa») e le manovre pericolose le prova lì
dentro, gratis. Il primo è un agente **model-free**: impara solo dall'urto
reale. Il secondo è **model-based**: prima impara *come funziona* il mondo,
poi usa quella conoscenza per provare le mosse nell'immaginazione, e nel mondo
vero ci va già preparato.

Il vantaggio è ovvio: ogni giro reale del secondo allievo vale molto di più,
perché da esso spreme decine di prove immaginate. Il rischio, altrettanto
ovvio, è di fidarsi di un simulatore sbagliato: se nella sua testa le curve
sono più dolci che in strada, si allena a guidare un'auto che non esiste.

`````

`````{tab} Superiore

Un ambiente di RL è un processo decisionale di Markov con dinamica
$p(s' \mid s, a)$ (la probabilità di finire nello stato $s'$ partendo da $s$ e
compiendo $a$) e una funzione di ricompensa $r(s, a)$. Un metodo
**model-free** apprende direttamente la policy $\pi_\theta(a \mid s)$ o i
valori $Q(s, a)$, $V(s)$ dall'interazione, senza mai stimare $p$ e $r$. Un
metodo **model-based** fa l'opposto: apprende un modello
$\hat p_\psi(s' \mid s, a)$ e $\hat r_\psi(s, a)$ dalle transizioni osservate,
poi lo usa per **pianificare** (cercare, tra le traiettorie *immaginate*,
quelle a ritorno più alto) o per generare esperienza sintetica su cui allenare
policy e valore {cite}`sutton2018reinforcement`.

Il guadagno atteso è la **sample efficiency**: una transizione reale, digerita
nel modello, ne genera molte simulate. Il prezzo ha un nome preciso, **model
bias**: il modello $\hat p_\psi$ non è la dinamica vera, e l'errore di
predizione si propaga lungo l'orizzonte. Peggio, una policy ottimizzata *dentro*
il modello impara a sfruttarne i difetti (*model exploitation*), incassando
ritorni immaginari che l'ambiente reale non paga. L'intera storia di questa
sezione è il racconto di come si è negoziato questo compromesso: quanta fiducia
concedere al modello.

`````

## Dyna: intrecciare il vero e l'immaginato

L'idea non è nuova. Nel 1990 Richard Sutton (lo stesso del libro di
riferimento su cui poggia mezzo capitolo {cite}`sutton2018reinforcement`)
presenta **Dyna** {cite}`sutton1990integrated`, ripresa l'anno dopo in una
versione più diffusa {cite}`sutton1991dyna`. È un'architettura tanto semplice
quanto lungimirante: mentre l'agente gioca, impara *contemporaneamente* due cose
(una policy, come sempre, e un modellino del mondo) e usa il modellino per
«ripassare» esperienze mai vissute davvero.

`````{tab} Elementare

Pensa a uno studente che, dopo aver fatto un esercizio di matematica in
classe, la sera lo rifà a mente altre venti volte, variando i numeri. Non ha
bisogno di tornare a scuola: gli basta ricordare *come funzionano* i conti (la
sua regola appresa) per generarsi esercizi nuovi e allenarsi su quelli. Dyna
fa esattamente questo. Ogni volta che l'agente compie una mossa vera, impara
due cose: aggiusta le sue valutazioni in base a com'è andata, *e* si annota la
transizione («da qui, con questa mossa, sono finito lì e ho preso questa
ricompensa»). Poi, prima della mossa successiva, si concede qualche «ripasso»:
pesca a caso alcune transizioni già annotate e riaggiusta le valutazioni anche
su quelle, come se le stesse rivivendo. Una mossa vera, tanti ripassi
immaginati: la ricompensa si propaga all'indietro molto più in fretta.

`````

`````{tab} Superiore

Dyna-Q intreccia *apprendimento diretto* e *planning* attorno allo stesso
$Q(s,a)$. Dopo ogni passo reale $(s, a, r, s')$ esegue due tipi di
aggiornamento. Il primo, dall'esperienza vera:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \big[\, r + \gamma \max_{a'} Q(s', a') -
Q(s, a) \,\big],
$$

e memorizza la transizione nel modello,
$\text{modello}(s,a) \leftarrow (r, s')$. Poi ripete $n$ volte un passo di
**planning**: campiona una coppia $(s, a)$ già osservata, ne recupera
$(r, s')$ dal modello e applica lo *stesso* aggiornamento di sopra; solo che
l'esperienza è simulata, non vissuta. Qui $\alpha$ è il passo di apprendimento
e $\gamma$ il fattore di sconto. Con $n$ grande, ogni interazione reale
scatena molti aggiornamenti immaginati, e la propagazione dei valori accelera
drasticamente. È il seme di tutto il model-based moderno: separare
l'esperienza (costosa, reale) dagli aggiornamenti (economici, ripetibili nel
modello).

`````

Conviene vedere Dyna al lavoro su un ambiente minuscolo. Il nome per esteso è
**Dyna-Q**, con la $Q$ del Q-learning appiccicata in coda, perché i giudizi
che va ad aggiornare sono proprio quelli: quanto vale ciascuna mossa in
ciascuna casella. Il corridoio ha sei caselle: si parte a sinistra,
l'obiettivo è la casella più a destra, e la ricompensa arriva solo entrando
nell'obiettivo.

Prima però va scelto **che cosa guardare**. La
tentazione è guardare la strategia appresa e verificare che dica «vai sempre a
destra»: solo che il corridoio è così facile che il Q-learning con la tabella,
senza un solo ripasso, impara la stessa identica strategia. Sarebbe una misura che
non misura. Ciò che il ripasso cambia davvero è la quantità che il testo ha
appena promesso, cioè **quanto in fretta la ricompensa si propaga all'indietro**
fino allo stato di partenza. Perciò il codice qui sotto esegue lo stesso ciclo
due volte, con e senza ripassi, e a ogni giro stampa quanto vale, per l'agente,
trovarsi nella casella di partenza: zero vuol dire «di qui non ho ancora
imparato che si guadagna qualcosa», e più il numero sale più la buona notizia
è arrivata fin laggiù. Ogni giro è un **episodio**, cioè una partita dall'inizio
alla fine, e di partite se ne giocano trenta, ripetendo tutto su dieci semi.

```python
import numpy as np

# Ambiente: corridoio di 6 stati; l'obiettivo e' lo stato 5 (assorbente).
# Azioni: 0 = sinistra, 1 = destra. Ricompensa +1 solo entrando nell'obiettivo.
n_stati, n_azioni, goal = 6, 2, 5

def passo(s, a):
    s2 = min(s + 1, goal) if a == 1 else max(s - 1, 0)
    return s2, (1.0 if s2 == goal else 0.0), (s2 == goal)

def dyna(n_plan, seme, episodi=30):
    """Dyna-Q. Con n_plan=0 e' Q-learning tabellare puro: nessun ripasso."""
    rng = np.random.default_rng(seme)
    Q = np.zeros((n_stati, n_azioni))
    modello = {}                       # (s, a) -> (r, s2): la dinamica APPRESA
    alpha, gamma, eps = 0.1, 0.95, 0.1
    valore_partenza = []               # quanto vale lo stato 0, episodio per episodio
    for _ in range(episodi):
        s = 0
        for _ in range(100):
            if rng.random() < eps:
                a = int(rng.integers(n_azioni))
            else:                      # argmax con i pareggi rotti a caso
                a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
            s2, r, fine = passo(s, a)
            # 1) aggiornamento dall'esperienza REALE
            Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
            modello[(s, a)] = (r, s2)  # memorizza la transizione osservata
            # 2) n passi di PLANNING su transizioni gia' viste (esperienza immaginata)
            viste = list(modello.keys())
            for _ in range(n_plan):
                sp, ap = viste[rng.integers(len(viste))]
                rp, s2p = modello[(sp, ap)]
                Q[sp, ap] += alpha * (rp + gamma * Q[s2p].max() - Q[sp, ap])
            s = s2
            if fine:
                break
        valore_partenza.append(Q[0].max())
    return np.argmax(Q, axis=1), np.array(valore_partenza)

SEMI = range(10)
for n_plan in (0, 20):
    esiti = [dyna(n_plan, s) for s in SEMI]
    v = np.array([e[1] for e in esiti])          # (semi, episodi)
    print(f"n_plan={n_plan:2d} | policy appresa (0=sx, 1=dx): "
          f"{esiti[0][0][:goal].tolist()}")
    print(f"           valore dello stato di partenza, mediana su {len(SEMI)} semi: "
          f"dopo 3 episodi {np.median(v[:, 2]):.3f}, "
          f"dopo 10 {np.median(v[:, 9]):.3f}, alla fine {np.median(v[:, -1]):.3f}")
```

La strategia appresa, come previsto, è la stessa nei due casi:
`[1, 1, 1, 1, 1]`, vai sempre a destra. Il valore della casella di partenza no. Senza ripassi, dopo tre
episodi vale ancora $0{,}000$ (la notizia della ricompensa non è arrivata fin
laggiù) e dopo trenta si ferma a $0{,}156$. Con venti ripassi per ogni mossa
vera, dopo tre episodi vale già $0{,}200$, dopo dieci $0{,}808$ e alla
trentesima $0{,}815$.

Quel $0{,}815$ è la risposta esatta, e la risposta esatta si calcola a mano.
Prima però va detto perché un premio lontano conta meno di uno vicino: non è una
legge di natura, è una scelta di chi programma, e si fa per due motivi. Un
agente che dà lo stesso peso a un guadagno fra tre mosse e a uno fra tremila non
ha nessun motivo di sbrigarsi; e su una partita che potrebbe non finire mai, la
somma di tutti i premi futuri sarebbe infinita per chiunque, il che renderebbe
ogni strategia buona quanto le altre. Nel codice
ogni passo di attesa sconta il premio del $5\%$, cioè lo moltiplica per
$0{,}95$. Chi
si trova sulla casella accanto all'obiettivo incassa $1$ alla mossa dopo, e per
lui quel premio vale $1$; chi sta una casella più indietro deve aspettare una
mossa in più e per lui vale $0{,}95$; due caselle indietro, $0{,}95\times0{,}95$.
Dalla partenza al traguardo ci sono cinque mosse, ma l'ultima incassa il premio
subito e non aspetta niente: le attese vere sono quattro, quindi
$0{,}95^{4} \approx 0{,}815$, ed è lì che il valore deve arrivare.

È tutto il guadagno del model-based in due numeri. Dopo trenta episodi per parte
il ripasso ha portato la casella di partenza a $0{,}815$, cioè esattamente dove
doveva arrivare, mentre senza ripassi si è fermata a $0{,}156$: **cinque volte
più in basso**, e ancora lontanissima dal bersaglio. Se avessimo guardato solo
la policy non avremmo visto niente, e avremmo attribuito ai ripassi un merito
che in questo ambiente non hanno.

## Il tallone d'Achille: l'errore che si accumula

C'è un motivo se Dyna, nell'esempio, «immagina» transizioni di *un solo passo*
già osservate, e non intere partite inventate di sana pianta. È il problema
strutturale di ogni approccio model-based: più il sogno si allunga, più
l'errore del modello **si compone**, cioè non si somma soltanto, si moltiplica su
se stesso. Una predizione appena imprecisa a un passo diventa una predizione
mediocre a cinque passi e un'assurdità a venti. (Una di quelle partite
immaginate, in gergo, si chiama *rollout*.)

`````{tab} Elementare

È il gioco del telefono senza fili. Il primo bambino sussurra la frase giusta;
ognuno la ripete con un piccolo errore, e dopo dieci passaggi la frase è
irriconoscibile. Un modello del mondo fa lo stesso quando gli chiedi di
immaginare lontano: la prima previsione è quasi giusta, ma la seconda parte da
quella «quasi», la terza dal «quasi del quasi», e l'errore si gonfia a ogni
passo.

*Quanto* si gonfi, però, dipende dal mondo che stai immaginando, ed è la parte
che sfugge. Ci sono sistemi che si rimettono a posto da soli, come una biglia in
fondo a una scodella: lì uno scarto piccolo resta piccolo per sempre, e si può
sognare a lungo senza troppi danni. E ci sono sistemi instabili, come la biglia
in equilibrio sulla scodella rovesciata, dove ogni passaggio ingrandisce lo
scarto invece di smorzarlo: bastano pochi passi e il sogno non ha più niente a
che vedere con la realtà.

Morale: le previsioni utili sono quelle a breve. La cura è disarmante nella sua
semplicità, e nel 2019 trova la formulazione che farà scuola, un algoritmo che
si chiama **MBPO** («ottimizzare la strategia basandosi su un modello»): invece
di far partire i sogni dall'inizio della partita e
tirarli avanti a lungo, si parte da una situazione *vera*, appena visitata, e si
immagina solo pochi passi. Sogni corti, ancorati alla realtà, e l'errore non fa
in tempo ad accumularsi.

`````

`````{tab} Superiore

Il conto si fa su una dinamica **deterministica**, che è il caso in cui «lo
scarto» è una distanza fra due stati e non fra due distribuzioni (nel caso
stocastico la stessa idea regge, ma va riscritta in distanza di Wasserstein, e la
costante non è più la stessa). Se il modello sbaglia di una quantità $\epsilon$
a ogni passo, ogni passo successivo parte da uno stato già sbagliato, e quanto
quell'errore si gonfi dipende da quanto la dinamica **amplifica le
perturbazioni**. Con una dinamica $L$-Lipschitz, cioè che moltiplica al più per
$L$ la distanza fra due stati vicini, lo scarto dopo $k$ passi è maggiorato da

$$
\epsilon \sum_{i=0}^{k-1} L^{\,i} ,
$$

e i tre regimi sono diversissimi fra loro. Per $L>1$ la somma esplode
**esponenzialmente** in $k$, ed è questo il caso che rende il compounding error
un problema: un sistema instabile, o caotico, è per definizione uno dove $L>1$.
Per $L=1$ esattamente si ha la crescita **lineare**, $k\epsilon$, che è il caso
limite e non il caso tipico. E per $L<1$, cioè quando la dinamica è
contrattiva e gli scostamenti si riassorbono da sé, lo scarto è **limitato** da
$\epsilon/(1-L)$ e smette proprio di crescere: con $\epsilon = 0{,}01$, a
cinquanta passi e $L=0{,}5$ lo scarto resta $0{,}020$, contro i $0{,}500$ che
darebbe la lettura lineare, ed è già fermo lì dal ventesimo passo. È il
**compounding error**, e impone un compromesso: rollout lunghi danno più segnale
di allenamento ma sempre meno affidabile, e quanto meno affidabile non lo decide
l'orizzonte da solo, lo decide il sistema.

**MBPO** (*Model-Based Policy Optimization*, Janner et al., 2019
{cite}`janner2019trust`) risolve il compromesso con un'idea nel titolo del
lavoro: *When to Trust Your Model*, «quando fidarsi del modello». Invece di
srotolare lunghe traiettorie dallo stato iniziale, MBPO esegue **rollout
brevi** (spesso di uno o pochi passi) che *si diramano da stati reali*
campionati dal buffer di esperienza: il modello (in pratica un *ensemble* di
reti probabilistiche, che rappresenta anche la propria incertezza) viene
interrogato solo dove è più affidabile, vicino a stati davvero visitati. Le
transizioni sintetiche così generate alimentano un algoritmo model-free
off-policy (SAC). L'analisi degli autori lega esplicitamente il divario di
ritorno all'errore del modello *e* alla lunghezza del rollout, giustificando
formalmente la scelta di tenerlo corto: si sfrutta il modello dove aiuta, lo
si evita dove mente.

`````

## MuZero: pianificare senza conoscere le regole

Fin qui abbiamo dato per scontata una cosa: che l'agente, per costruirsi il suo
simulatore, sappia sempre com'è fatto il mondo in cui si trova. Ma in Go, negli
scacchi, in un videogioco Atari, quello che riceve sono pietre su una griglia o
puntini colorati su uno schermo, e le regole che li fanno muovere possono
essergli ignote, o essere troppo complicate da scrivere a mano. Nel 2020 un
gruppo di DeepMind guidato da Julian Schrittwieser presenta **MuZero**
{cite}`schrittwieser2020mastering`, che fa un passo che sembra un gioco di
prestigio: pianifica in profondità *senza conoscere le regole del gioco*.

`````{tab} Elementare

MuZero è l'erede di AlphaZero, l'algoritmo che nella sezione sui gradienti di
policy abbiamo visto padroneggiare Go, scacchi e shogi partendo dalle sole
regole. Ma ad AlphaZero le regole erano *date*: sapeva con esattezza, per ogni
mossa, quale posizione ne sarebbe seguita. MuZero no: se le costruisce da solo
guardando le partite, e (dettaglio cruciale) non si fa un modello che
ridisegna la scacchiera pezzo per pezzo, ma solo un modello «da stratega».
Immagina un maestro che ragiona per sensazioni: non visualizza ogni pedone
dopo dieci mosse, ma tiene in testa quel tanto che gli serve per rispondere a
tre domande («questa mossa mi avvicina alla vittoria? chi è in vantaggio? che
ricompensa arriva ora?»). MuZero impara questo riassunto astratto, il minimo
per pianificare, e butta via il resto; poi, lì dentro, esplora a fondo le
linee più promettenti prima di decidere.

`````

`````{tab} Superiore

MuZero apprende un **modello latente** fatto di tre funzioni, addestrate insieme
end-to-end:

$$
s^0 = h_\psi(o_{\le t}), \qquad
(s^{k+1}, \hat r^{k+1}) = g_\psi(s^k, a^k), \qquad
(\hat p^k, \hat v^k) = f_\psi(s^k).
$$

La *rappresentazione* $h_\psi$ codifica le osservazioni passate $o_{\le t}$ in
uno stato latente iniziale $s^0$; la *dinamica* $g_\psi$, dato lo stato
latente $s^k$ e un'azione ipotetica $a^k$, predice il latente successivo
$s^{k+1}$ e la ricompensa $\hat r^{k+1}$; la *predizione* $f_\psi$ ne ricava
una policy $\hat p^k$ e un valore $\hat v^k$. Punto decisivo: $s^k$ **non** è
addestrato a ricostruire l'osservazione. Non c'è alcuna pressione a
rappresentare i pixel; il latente deve solo contenere ciò che serve a predire
*policy, valore e ricompensa*: le tre quantità utili alla pianificazione. Su
questo modello latente MuZero esegue una **ricerca ad albero Monte Carlo**
(MCTS), la stessa idea di AlphaGo {cite}`silver2016mastering` e del suo
successore AlphaZero {cite}`silver2018general`, ma
srotolata dentro il modello appreso anziché su un simulatore dato. Il
risultato: prestazioni pari ad AlphaZero su Go, scacchi e shogi *senza*
riceverne le regole, e la stessa ricetta che regge sui giochi Atari, dove un
modello scritto a mano non esiste affatto. La differenza con AlphaZero è tutta
qui: AlphaZero pianifica su un modello *fornito*, MuZero su un modello
*appreso*.

`````

## Dreamer: allenare la policy nel sogno

Le strade viste finora sono due. Dyna immagina un passo alla volta e con quello
aggiusta i propri giudizi; MuZero, al momento di decidere, si ferma ed esplora
un albero di continuazioni. Ce n'è una terza, che il {doc}`capitolo sui World Model </WorldModels/overview>`
racconta per esteso e che qui serve solo a completare il quadro: costruirsi un
simulatore interno dell'ambiente (un **world model**) e allenare la strategia
*interamente lì dentro*, senza mai fermarsi a pianificare.

Perché questa terza via funzioni così bene c'è una ragione precisa, e sta nel
fatto che anche il simulatore è una rete neurale. Una rete sa correggersi
all'indietro: si parte da com'è andata a finire e si risale, un pezzo alla
volta, fino ai numeri interni che hanno prodotto quel risultato. Ora, se il
simulatore è una rete e il pilota pure, allora la catena all'indietro non si
ferma alla fine della partita immaginata: risale lungo tutta la partita, mossa
dopo mossa, fino alla prima. Il pilota impara quindi non solo *che* la manovra è
finita male, ma anche *quale* dettaglio della manovra andava cambiato.

Il simulatore, poi, non ridisegna il mondo puntino per puntino: ne tiene solo il
riassunto che serve a decidere, e in gergo quel riassunto si chiama **latente**.

`````{tab} Elementare

È il pilota che, la sera prima della gara, ripassa il circuito a occhi chiusi,
curva per curva. Non consuma benzina, non rischia incidenti: la pista ce l'ha
in testa, e lì dentro può girare quante volte vuole. Gli algoritmi della
famiglia **Dreamer** fanno questo: si costruiscono un modello del gioco e poi
addestrano il pilota *solo dentro il sogno*, riportandolo nel mondo vero già
allenato. La linea di ricerca nasce dai «mondi in miniatura» di Ha e
Schmidhuber {cite}`ha2018world` (l'agente che imparava a schivare palle di
fuoco esercitandosi nel proprio sogno) e arriva a **DreamerV3** di Danijar
Hafner e colleghi
{cite}`hafner2023mastering`, che con la *stessa* configurazione, senza
ritocchi, padroneggia oltre 150 compiti diversi (robot simulati, giochi Atari,
navigazione 3D) e riesce persino a raccogliere i diamanti in *Minecraft*
partendo da zero, senza che nessuno gli mostri come.

`````

`````{tab} Superiore

Dreamer apprende un modello ricorrente dello stato nello spazio latente e vi
addestra un attore-critico (i metodi visti nella sezione sui gradienti di
policy) per retropropagazione lungo **rollout immaginati** a orizzonte breve
(una quindicina di passi) proprio per contenere il compounding error.
DreamerV3 {cite}`hafner2023mastering` aggiunge normalizzazioni robuste di
osservazioni, ricompense e ritorni che rendono lo stesso set di iperparametri
valido su domini radicalmente diversi: è la dimostrazione che un agente
model-based può essere *generalista*. La parentela con Dyna è diretta (attore
e critico crescono su esperienza sintetica generata da un modello appreso) ma
il modello qui è una rete profonda che vive in uno spazio latente, non una
tabella di transizioni. Per la ricetta completa (encoder, modello ricorrente,
il «sogno» come rollout latente), si rimanda al capitolo sui World Model, che
tratta anche la proposta di LeCun e le architetture JEPA, la frontiera di
questa linea di ricerca.

`````

Un modello della dinamica appreso, come quelli che alimentano i metodi appena
visti, in PyTorch ha una forma semplice: da stato e azione predice lo stato
successivo e la ricompensa. La traiettoria «immaginata» è poi la sua
applicazione ripetuta, tenuta volutamente corta.

```python
import torch
from torch import nn

class ModelloDinamica(nn.Module):
    """Modello appreso: da (stato, azione) predice stato successivo e ricompensa."""
    def __init__(self, dim_s, dim_a, dim_h=200):
        super().__init__()
        self.corpo = nn.Sequential(
            nn.Linear(dim_s + dim_a, dim_h), nn.SiLU(),
            nn.Linear(dim_h, dim_h), nn.SiLU(),
        )
        self.testa_stato = nn.Linear(dim_h, dim_s)    # variazione dello stato
        self.testa_ricompensa = nn.Linear(dim_h, 1)   # ricompensa predetta

    def forward(self, s, a):                          # s: (B, dim_s), a: (B, dim_a)
        h = self.corpo(torch.cat([s, a], dim=-1))
        s_succ = s + self.testa_stato(h)              # residuo: predice il cambiamento
        r = self.testa_ricompensa(h).squeeze(-1)      # (B,)
        return s_succ, r

# Rollout BREVE immaginato (stile MBPO): parte da stati reali, pochi passi.
modello = ModelloDinamica(dim_s=4, dim_a=1)
policy = nn.Sequential(nn.Linear(4, 1), nn.Tanh())    # strategia giocattolo
s = torch.randn(32, 4)                                # 32 situazioni VERE gia' vissute
for _ in range(3):                                    # orizzonte corto: 3 passi
    a = policy(s)                                     # (32, 1)
    s, r = modello(s, a)                              # transizioni SINTETICHE
```

## Onestà: quando il sogno inganna

Resta il punto dolente, che nessun risultato spettacolare cancella. Una policy
addestrata dentro un modello finisce, prima o poi, per **sfruttarne i
difetti**. Se il modello sbaglia in modo sistematico (sopravvaluta una
ricompensa, dimentica un ostacolo), l'ottimizzazione trova con precisione
chirurgica proprio quelle crepe: emergono policy che incassano ritorni
immaginari altissimi e falliscono nel mondo vero. È il *model exploitation*, e
nel {doc}`capitolo sui World Model </WorldModels/overview>` se ne vede l'esempio da manuale: l'agente che,
dentro il proprio sogno di *Doom*, scopre movimenti per cui i mostri «non
sparano mai».

Il compromesso è strutturale, e si può leggere come una manopola. Girata da una
parte c'è la fiducia: sogni lunghi, tutta la strategia allenata
nell'immaginazione, pochissima esperienza vera spesa, e in cambio si eredita in
pieno l'errore sistematico del simulatore. Girata dall'altra c'è la prudenza:
sogni corti che partono da situazioni davvero visitate, e più simulatori
addestrati in parallelo per vedere dove vanno d'accordo e dove no. Il
disaccordo, di per sé, non dice quale abbia ragione; dice che i dati visti non
bastavano a stabilirlo, e quindi che di quel pezzo di mondo nessuno sa
abbastanza. È il modo più semplice di sapere dove non fidarsi. Si è più
robusti, e si torna a spendere esperienza vera. I metodi model-free, per
contro, non hanno nessun modello da sfruttare e restano competitivi quando i
campioni costano poco. Non c'è un vincitore assoluto: c'è quella manopola, e
sapere dove metterla è oggi materia di ricerca aperta.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un agente **model-free** impara solo schiantandosi per davvero; uno
  **model-based** si costruisce prima in testa un simulatore del mondo e le
  manovre le prova lì dentro, gratis. Guadagna in esperienza risparmiata,
  rischia di allenarsi su un'auto che non esiste.
- **Dyna** (Sutton, 1990) è il capostipite: dopo ogni mossa vera l'agente
  aggiusta le sue valutazioni e si annota che cosa è successo, poi si concede
  qualche "ripasso" pescando a caso fra le transizioni annotate. Una mossa
  vera, tanti ripassi immaginati, e la ricompensa si propaga all'indietro
  molto più in fretta: nel corridoio dell'esempio, a parità di partite giocate,
  la casella di partenza finisce a un valore cinque volte più alto, che è poi
  quello giusto.
- Immaginare lontano è il gioco del telefono senza fili: ogni previsione parte
  da una precedente già un po’ sbagliata e l'errore si gonfia a ogni passaggio.
  *Quanto* si gonfi dipende dal sistema: dove gli scarti si riassorbono da sé
  l'errore resta piccolo per sempre, dove il sistema li ingigantisce esplode in
  pochi passi. La cura è tenere i sogni **corti** e farli partire da situazioni
  davvero visitate, così l'errore non fa in tempo ad accumularsi.
- **MuZero** (2020) le regole del gioco se le costruisce da solo guardando le
  partite, e non si fa un modello che ridisegna la scacchiera pezzo per pezzo:
  tiene solo il riassunto che serve a rispondere a "chi è in vantaggio, che
  ricompensa arriva ora, quale mossa conviene". Lì dentro esplora a fondo le
  linee promettenti prima di decidere. È l'erede di AlphaZero, che invece le
  regole le riceveva già scritte.
- **Dreamer** allena il pilota interamente dentro il sogno, e DreamerV3 se la
  cava con la stessa configurazione su oltre 150 compiti diversi. Il limite di
  fondo non sparisce: una strategia vale quanto il mondo immaginario in cui è
  cresciuta, e prima o poi ne trova e ne sfrutta le crepe.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- **Model-free** impara provando per davvero; **model-based** apprende prima la
  dinamica $\hat p_\psi(s' \mid s, a)$ e la ricompensa, poi *pianifica* dentro
  il modello. Il premio è la **sample efficiency**; il pericolo è il **model
  bias**.
- **Dyna** (Sutton, 1990) è il capostipite: intreccia aggiornamenti da
  esperienza reale e da esperienza «immaginata» campionata dal modello appreso.
- L’**errore si accumula** lungo il rollout (compounding error), maggiorato da
  $\epsilon\sum_{i<k}L^{\,i}$: esponenziale in $k$ appena la dinamica amplifica
  le perturbazioni ($L>1$), lineare solo nel caso limite $L=1$, limitato se è
  contrattiva. **MBPO** (Janner et al., 2019) lo aggira con rollout *brevi*
  diramati da stati reali, usando il modello solo dove è affidabile.
- **MuZero** (Schrittwieser et al., 2020) apprende un modello *latente*
  (dinamica, ricompensa, valore) e pianifica con MCTS *senza conoscere le
  regole* del gioco: è l'erede di AlphaZero, che il modello lo riceveva già
  fatto.
- **Dreamer** (Hafner et al.) allena la policy interamente nell'immaginazione
  latente; DreamerV3 è generalista su oltre 150 compiti. Il limite di fondo
  resta uno: la policy è buona quanto il modello in cui è cresciuta.
```
`````
