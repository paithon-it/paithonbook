# Esplorazione e ricompensa: curiosità, sparsità, reward hacking

Fra i 49 giochi Atari su cui DeepMind mise alla prova il DQN nel 2015, ce n'è
uno dove l'agente che aveva imparato a giocare a decine di titoli superando i
campioni umani colleziona un punteggio desolante: **zero**
{cite}`mnih2015human`. Il gioco è *Montezuma's Revenge*, un platform del 1984:
un esploratore in un tempio azteco deve scendere una scala, saltare una fune,
scansare un teschio rotolante e raccogliere una chiave prima di ricevere il suo
primo punto. Decine di mosse esatte, in sequenza, per un solo segnale di
"bene". Un agente che sceglie mosse a caso non arriverà mai in fondo a quella
catena: cadrà, morirà, e non vedrà mai una ricompensa da cui imparare. Il muro
contro cui il DQN sbatté non è la percezione né la memoria: è
l'**esplorazione** in un mondo dove le ricompense sono **sparse**.

## Il problema delle ricompense sparse

Nella maggior parte dei giochi Atari qualcosa di buono o cattivo capita a ogni
secondo, e l'agente ha un flusso costante di segnali da cui correggersi. Ma
quando la ricompensa arriva solo dopo lunghe sequenze di azioni giuste (la
chiave, la porta, il livello), il segnale diventa un ago in un pagliaio.
Finché l'agente non inciampa *per caso* in quella prima ricompensa, non ha
nulla che gli dica in che direzione andare. E la probabilità di inciamparci
per caso, con l'esplorazione casuale, crolla esponenzialmente con la lunghezza
della catena.

Nel capitolo sul reinforcement learning abbiamo introdotto il dilemma
esplorazione–sfruttamento e la strategia $\varepsilon$-greedy: agire quasi
sempre secondo la stima migliore, ma ogni tanto scegliere a caso. Qui vediamo
perché quella ricetta, in ambienti come Montezuma, non basta, e cosa si è
inventato per andare oltre.

## Perché $\varepsilon$-greedy non basta

`````{tab} Elementare

Immagina di esplorare una città enorme lanciando una monetina a ogni incrocio
per decidere dove girare. Andrai avanti e indietro nello stesso quartiere per
ore: il caso non ha memoria, non sa quali strade hai già battuto e quali no.
Per raggiungere un vicolo lontano dieci svolte precise, la probabilità di
azzeccarle tutte tirando a sorte è minuscola.

L'esplorazione casuale è così: agita le mani nel buio nei dintorni di dove sei
già. Quello che servirebbe è un'esplorazione **diretta**: una spinta a puntare
verso i posti che non hai *ancora* visto, invece di rimescolare a caso quelli
di sempre.

`````

`````{tab} Superiore

Con $\varepsilon$-greedy le azioni esplorative sono scelte in modo *uniforme e
indipendente* dallo stato: la perturbazione è locale e non correlata nel tempo.
Per raggiungere uno stato-obiettivo che dista $n$ azioni "insolite" dalla
regione già visitata, la probabilità di percorrere l'intera sequenza per puro
caso scala come $(\varepsilon/|A|)^{\,n}$ e decade esponenzialmente in $n$.
Questo è **dithering**: rumore attorno alla policy corrente, non ricerca
strutturata.

L'esplorazione *diretta* (o *deep exploration*) tiene invece conto di ciò che
l'agente ha già visto e orienta deliberatamente il comportamento verso le
regioni poco note dello spazio degli stati. Il modo più naturale per ottenerla
è modificare non *come* si sceglie, ma *cosa* si ottiene: aggiungere alla
ricompensa dell'ambiente un **bonus** che premia la novità.

`````

## Bonus di novità: premiare ciò che si visita di rado

L'idea più intuitiva per un'esplorazione diretta ha una lunga storia nel RL
tabellare {cite}`sutton2018reinforcement`: se uno stato è stato visitato poche
volte, l'agente ne sa poco, quindi vale la pena andarci. Si aggiunge alla
ricompensa un termine che decresce col numero di visite.

`````{tab} Elementare

È il principio del turista curioso: dài a te stesso un piccolo premio ogni
volta che metti piede in un posto nuovo, un premio che si spegne man mano che
quel posto diventa familiare. All'inizio un quartiere ti frutta un bel bonus;
dopo esserci passato cento volte, quasi niente. Il risultato è che vieni
naturalmente spinto verso l'ignoto, senza bisogno di lanciare monetine.

`````

`````{tab} Superiore

Si sostituisce alla ricompensa dell'ambiente $r_t$ una ricompensa aumentata

$$
r_t^{+} = r_t + \beta\, \frac{1}{\sqrt{N(s_t)}} ,
$$

dove $N(s_t)$ è il numero di volte in cui lo stato $s_t$ è stato visitato e
$\beta>0$ dosa il peso della curiosità. Il bonus è alto sugli stati rari,
tende a zero su quelli battuti: l'agente è incentivato a raggiungere le zone
poco esplorate.

Il limite è evidente in spazi grandi o continui: con osservazioni ad alta
dimensione (i pixel di uno schermo) ogni stato è, letteralmente, unico, e
$N(s_t)$ vale sempre $1$. Il conteggio esatto non ha senso. La soluzione sono
gli **pseudo-conteggi**: si stima una densità $\rho(s)$ sugli stati visitati e
se ne ricava un conteggio *effettivo* $\hat N(s)$ coerente con quanto la
densità è "sorpresa" di rivedere $s$. È l'approccio *count-based* esteso agli
spazi grandi (Bellemare e colleghi, 2016), che diede i primi progressi
sostanziali proprio su Montezuma's Revenge.

`````

Il bonus di novità si calcola facilmente quando gli stati sono pochi e
distinti. Il frammento seguente mostra come, a parità di formula, gli stati più
rari ricevano la spinta esplorativa più forte:

```python
import numpy as np

# Conteggi di visita di 6 stati in un piccolo ambiente tabellare
visite = np.array([120, 40, 5, 0, 200, 1])

# Bonus di novità count-based: più raro lo stato, più alto il bonus.
beta = 0.5
bonus = beta / np.sqrt(visite + 1)   # +1 evita la divisione per zero

for s, (n, b) in enumerate(zip(visite, bonus)):
    print(f"stato {s}: visite={n:3d}  bonus={b:.3f}")
```

Lo stato mai visitato riceve il bonus massimo ($0{,}500$), quello battuto
duecento volte quasi nulla ($0{,}035$): la ricompensa aumentata inclina
l'agente verso l'ignoto senza toccare la regola di scelta.

## Curiosità intrinseca: la sorpresa come ricompensa

Contare le visite, anche per approssimazione, resta difficile. Un'idea più
elegante ribalta la domanda: invece di chiederci *quante volte* abbiamo visto
uno stato, chiediamoci *quanto ci sorprende*. Uno stato sorprendente (uno di
cui non sappiamo prevedere le conseguenze) è, per definizione, uno da cui
abbiamo ancora molto da imparare. La sorpresa diventa così una **ricompensa
intrinseca**, generata dall'agente stesso, che affianca quella estrinseca
dell'ambiente.

`````{tab} Elementare

Pensa a un bambino che gioca. Nessuno gli dà punti: eppure esplora
instancabilmente, attratto da ciò che non riesce a prevedere. Spinge un
bicchiere oltre il bordo del tavolo perché non sa ancora cosa succederà; una
volta imparato che cade e si rompe, quel gesto smette di interessarlo e ne
cerca un altro. La **curiosità** è proprio questo: un premio interno per la
sorpresa. L'agente costruisce dentro di sé un modello di "cosa succederà se
faccio questo"; quando il modello sbaglia la previsione, quello scarto vale
come una piccola ricompensa. Impara facendo, spinto dal desiderio di ridurre
lo stupore.

`````

`````{tab} Superiore

Nel modulo di curiosità intrinseca **ICM** (*Intrinsic Curiosity Module*,
Pathak e colleghi, 2017 {cite}`pathak2017curiosity`) la ricompensa intrinseca
è l'**errore di predizione** di un modello di dinamica. La chiave è che la
previsione non avviene sui pixel grezzi ma in uno **spazio di feature**
$\phi(s)$ appreso: si allena una rete a codificare lo stato in modo che
catturi solo ciò che l'agente *può controllare*, ignorando il rumore
irrilevante dell'ambiente. Un modello *forward* prevede la feature del prossimo
stato $\hat\phi(s_{t+1})$ da $\phi(s_t)$ e dall'azione $a_t$; la ricompensa
intrinseca è

$$
r_t^{\text{int}} = \frac{\eta}{2}\,\big\lVert \hat\phi(s_{t+1}) - \phi(s_{t+1}) \big\rVert^2 ,
$$

l'errore di predizione, con $\eta>0$ un fattore di scala. Alta sugli stati la
cui dinamica il modello non ha ancora imparato, la ricompensa si spegne man
mano che il modello migliora: la curiosità è auto-esauribile.

Una variante più semplice e sorprendentemente efficace è **RND** (*Random
Network Distillation*, Burda e colleghi, 2019 {cite}`burda2019exploration`).
Si fissa una rete *target* $f$ dai pesi **casuali e mai addestrati**, e si
allena una rete *predictor* $\hat f$ a imitarne l'output sugli stati
visitati. La ricompensa intrinseca è la distanza fra le due:

$$
r_t^{\text{int}} = \big\lVert \hat f(s_t) - f(s_t) \big\rVert^2 .
$$

Sugli stati già visti molte volte il predictor ha imparato a riprodurre la
target e l'errore è basso; su uno stato mai incontrato non ha idea di cosa
produrrà la rete casuale, e l'errore (cioè la novità) è alto. RND fu il primo
metodo a superare la prestazione media umana su Montezuma's Revenge senza
ricorrere a dimostrazioni umane né allo stato interno dell'emulatore: il
punteggio zero del DQN era già stato scalfito dagli pseudo-conteggi, ma ora,
tre anni dopo, anche la media umana era superata.

`````

Il cuore di RND si scrive in poche righe di PyTorch. Due reti identiche
d'architettura; una è congelata, l'altra impara a imitarla, e lo scarto è la
misura di novità:

```python
import torch
import torch.nn as nn

# RND: due reti con la stessa architettura.
# target: pesi casuali FISSI, mai addestrati; predictor: impara a imitarla.
def crea_rete(dim_stato, dim_feature=64):
    return nn.Sequential(
        nn.Linear(dim_stato, 128), nn.ReLU(),
        nn.Linear(128, dim_feature),
    )

target = crea_rete(dim_stato=16)
predictor = crea_rete(dim_stato=16)

# La rete target è congelata: non le passa mai gradiente.
for p in target.parameters():
    p.requires_grad_(False)

optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)

def ricompensa_intrinseca(stati):
    with torch.no_grad():
        obiettivo = target(stati)            # output della rete casuale fissa
    previsione = predictor(stati)
    # errore per-stato = novità: alto sugli stati poco visti
    return (previsione - obiettivo).pow(2).mean(dim=1)

stati = torch.randn(8, 16)
r_int = ricompensa_intrinseca(stati)         # bonus di novità del batch
# Il predictor si allena a ridurre l'errore sugli stati che l'agente visita:
# così quegli stati, in futuro, saranno meno "sorprendenti".
loss = r_int.mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Reward shaping: guidare senza barare

C'è un'alternativa più diretta al problema della sparsità: se le ricompense
sono troppo rare, perché non aggiungerne noi qualcuna intermedia, per guidare
l'agente passo dopo passo verso l'obiettivo? Questo si chiama **reward
shaping**, modellare la ricompensa. È potente, ma nasconde una trappola.

`````{tab} Elementare

Vuoi insegnare al robot a uscire dal labirinto e, per aiutarlo, gli dài un
premietto ogni volta che si avvicina all'uscita. Sembra ragionevole. Ma se non
stai attento, il robot può scoprire che gli conviene oscillare avanti e
indietro davanti a una porta, incassando premietti all'infinito, senza mai
uscire davvero. Gli hai insegnato a *inseguire il premietto*, non a uscire.

Per fortuna esiste un modo sicuro di dare questi aiuti. L'idea, dovuta a Andrew
Ng e colleghi nel 1999, è dare l'aiuto come *differenza di quota*: assegna a
ogni posto un "livello di vicinanza all'uscita", e premia l'agente solo per la
differenza di livello fra dove arriva e dove era. Fatta così, la somma dei
premietti lungo un giro chiuso è sempre zero: girare in tondo non frutta nulla,
e la strategia migliore resta esattamente quella di prima, solo più facile da
trovare.

`````

`````{tab} Superiore

Il rischio del reward shaping ingenuo è cambiare la policy ottima: un termine
aggiuntivo mal scelto può rendere conveniente un comportamento che l'obiettivo
originale non premia. Ng, Harada e Russell (1999) hanno dimostrato che esiste
una forma di shaping **garantita** a preservare l'ordine delle policy: il
**potential-based reward shaping**. Si sceglie una funzione potenziale
$\Phi(s)$ sugli stati e si aggiunge alla ricompensa il termine

$$
F(s, a, s') = \gamma\, \Phi(s') - \Phi(s) ,
$$

con $\gamma$ il fattore di sconto. Il risultato chiave è che la policy ottima
dell'MDP modellato coincide con quella dell'MDP originale, *per qualunque*
$\Phi$: la garanzia deriva da un argomento telescopico. Nella somma scontata
dei termini $F$ ogni potenziale intermedio compare una volta col segno più e
una col segno meno, e di una traiettoria di $T$ passi sopravvivono i soli due
termini di bordo, $-\Phi(s_0) + \gamma^{T}\Phi(s_T)$. Il secondo svanisce nei
due casi che interessano: a orizzonte infinito con $\gamma<1$ e $\Phi$
limitata, perché $\gamma^{T}\Phi(s_T)\to 0$; nei task episodici con la
convenzione (quella di Ng, Harada e Russell) $\Phi(s)=0$ sugli stati
terminali. Resta allora il solo $-\Phi(s_0)$: un
contributo che dipende dallo stato di partenza e non dal percorso, identico
quindi per tutte le policy. Nel caso non scontato ($\gamma=1$) l'argomento ha
il corollario intuitivo della quota: un ciclo chiuso frutta esattamente zero;
con $\gamma<1$ i cicli non sono più esattamente nulli, ma l'invarianza resta,
perché a garantirla è il telescopio. Lo shaping accelera
l'apprendimento rendendo il segnale più denso, senza spostare l'obiettivo
(si veda {cite}`sutton2018reinforcement`).

`````

## Reward hacking: la lettera contro l'intento

Il pericolo intravisto con il reward shaping è in realtà molto più generale, e
ha un nome: **reward hacking** (o *specification gaming*). L'agente ottimizza
esattamente la ricompensa che gli abbiamo scritto, e proprio per questo trova
scorciatoie che massimizzano quel numero tradendo del tutto ciò che
intendevamo.

```{figure} ../figures/reward-hacking.svg
:name: fig-reward-hacking
:alt: "Grafico con l'intensità dell'ottimizzazione in ascissa. Due curve partono insieme e salgono: la metrica surrogata, quella che stiamo effettivamente massimizzando, e l'obiettivo vero. Oltre un punto segnato come punto di Goodhart le due divergono: la surrogata continua a salire, mentre l'obiettivo vero comincia a scendere."
:width: 92%

Le due curve si separano, e il guaio è che se ne vede una sola. Chi guarda il
numero che ottimizza vede miglioramenti fino alla fine, anche molto dopo il
punto in cui le cose hanno cominciato a peggiorare.
```

Il punto di divergenza in {numref}`fig-reward-hacking` porta il nome di
Charles Goodhart, l'economista a cui si attribuisce la formulazione «quando
una misura diventa un obiettivo, cessa di essere una buona misura». Vale per
gli agenti come per le organizzazioni, e per la stessa ragione: la misura era
un buon indicatore *finché nessuno ci puntava contro tutto lo sforzo*.

L'esempio diventato manifesto è di OpenAI (2016): in *CoastRunners*, un gioco di
gare di barche, l'agente doveva completare un percorso il più in fretta
possibile. La ricompensa, però, era stata legata ai punti raccolti lungo il
tragitto, non all'arrivo. L'agente scoprì che in una laguna un gruppo di bonus
ricompariva a ciclo continuo: imparò a girare in tondo là dentro, andando a
sbattere e prendendo fuoco, incassando in media il **20% di punti in più** dei
giocatori umani senza mai finire la gara. Aveva "vinto" secondo la lettera
della ricompensa, perdendo secondo ogni ragionevole intento.

`````{tab} Elementare

È la stessa cosa che succede quando si paga un idraulico a numero di tubi
sostituiti: qualcuno inizierà a sostituire tubi che andavano benissimo. Il
metro con cui misuri diventa l'obiettivo, e l'obiettivo vero (l'impianto che
funziona) passa in secondo piano. Gli economisti la chiamano **legge di
Goodhart**: *quando una misura diventa un bersaglio, smette di essere una
buona misura*.

Con gli agenti è identico, e più insidioso, perché un ottimizzatore
instancabile cercherà *ogni* scorciatoia possibile. Il problema non è che
l'agente disobbedisce: è che obbedisce troppo bene, alla lettera sbagliata.
Scrivere una ricompensa che dica davvero ciò che vogliamo (e non una sua
approssimazione sfruttabile) è molto più difficile di quanto sembri.

`````

`````{tab} Superiore

Il reward hacking è la manifestazione, nel RL, della **legge di Goodhart**:
ogni ricompensa $r$ è una *proxy* misurabile dell'obiettivo reale, e un agente
che massimizza $\mathbb{E}[\sum_t \gamma^t r_t]$ spingerà la proxy fino a dove
essa diverge dall'intento. Più l'ottimizzazione è potente, più è probabile che
la soluzione ottima secondo $r$ cada in una regione dove proxy e intento non
coincidono più.

Le difese sono un ambito di ricerca attivo e nessuna è risolutiva: vincoli e
penalità esplicite, apprendimento della ricompensa dalle preferenze umane
(*reward modeling*, RLHF), verifica di robustezza rispetto a piccole modifiche
della specifica. Il nodo di fondo (specificare compiutamente ciò che vogliamo
tramite una funzione scalare) è il **problema dell'allineamento**, che
affronteremo nel capitolo sull'AI responsabile. Il reward hacking è il punto
in cui l'ottimizzazione tecnica incontra una domanda che tecnica non è del
tutto: siamo sicuri di aver chiesto la cosa giusta?

`````

Curiosità e reward hacking sono, in un certo senso, le due facce della stessa
libertà. Diamo all'agente margine per esplorare oltre ciò che gli indichiamo,
e scopre strategie che non avevamo immaginato: la mossa geniale, ma anche la
scorciatoia sleale. Progettare l'esplorazione e progettare la ricompensa è,
alla fine, lo stesso mestiere: decidere con cura cosa spingiamo davvero
l'agente a cercare.

```{admonition} Da ricordare
:class: important
- Con **ricompense sparse** (l'emblema è *Montezuma's Revenge*, dove il DQN
  segnava zero) l'esplorazione casuale di $\varepsilon$-greedy fallisce: serve
  esplorazione **diretta**, non rumore locale.
- I **bonus di novità** *count-based* premiano gli stati poco visitati
  ($\propto 1/\sqrt{N(s)}$); negli spazi grandi si usano **pseudo-conteggi**.
- La **curiosità intrinseca** trasforma la *sorpresa* in ricompensa: **ICM**
  usa l'errore di predizione della dinamica in uno spazio di feature, **RND**
  l'errore nel predire una rete casuale fissa, e risolse Montezuma.
- Il **reward shaping** densifica il segnale; solo la forma *potential-based*
  $F=\gamma\Phi(s')-\Phi(s)$ (Ng, Harada, Russell, 1999) preserva la policy
  ottima.
- Il **reward hacking** (legge di Goodhart) è l'agente che ottimizza la
  *lettera* della ricompensa, non l'intento, come la barca di *CoastRunners*.
  È il ponte verso il problema dell'**allineamento**.
```
