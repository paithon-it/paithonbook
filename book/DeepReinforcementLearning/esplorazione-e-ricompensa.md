# Esplorazione e ricompensa: curiosità, ricompense rade, reward hacking

Fra i 49 giochi Atari su cui DeepMind mise alla prova il DQN nel 2015, ce n'è
uno dove l'agente che su decine di altri titoli reggeva il confronto con un
collaudatore umano professionista colleziona un punteggio desolante: **zero**
{cite}`mnih2015human`. Il gioco è *Montezuma's Revenge*, un platform del 1984:
un esploratore in un tempio azteco deve scendere una scala, saltare una fune,
scansare un teschio rotolante e raccogliere una chiave prima di ricevere il suo
primo punto. Decine di mosse esatte, in sequenza, per un solo segnale di
"bene". Un agente che sceglie mosse a caso non arriverà mai in fondo a quella
catena: cadrà, morirà, e non vedrà mai una ricompensa da cui imparare. Il muro
contro cui il DQN sbatté non è la percezione né la memoria: è
l'**esplorazione** in un mondo dove le ricompense sono **rade**, o come si dice
di solito **sparse**: capitano una volta ogni tanto e in mezzo non c'è niente.

## Il problema delle ricompense sparse

Nella maggior parte dei giochi Atari qualcosa di buono o cattivo capita a ogni
secondo, e l'agente ha un flusso costante di segnali da cui correggersi. Ma
quando la ricompensa arriva solo dopo lunghe sequenze di azioni giuste (la
chiave, la porta, il livello), il segnale diventa un ago in un pagliaio.
Finché l'agente non inciampa *per caso* in quella prima ricompensa, non ha
nulla che gli dica in che direzione andare. E la probabilità di inciamparci per
caso crolla a precipizio con la lunghezza della catena: se a ogni passo ci sono
otto mosse possibili e una sola è quella giusta, azzeccarne dieci di fila per
puro caso capita una volta su un miliardo.

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
caso scala come $(\varepsilon/|\mathcal{A}|)^{\,n}$ e decade esponenzialmente
in $n$. Questo è **dithering**: rumore attorno alla policy corrente, non
ricerca strutturata.

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
r_t^{+} = r_t + \frac{\beta}{\sqrt{N(s_t)}} ,
$$

dove $N(s_t)$ è il numero di volte in cui lo stato $s_t$ è stato visitato e
$\beta>0$ dosa il peso della curiosità. Il bonus è alto sugli stati rari,
tende a zero su quelli battuti: l'agente è incentivato a raggiungere le zone
poco esplorate. Sullo stato mai visitato, $N=0$, la formula scritta così
diverge, e nella teoria è voluto: uno stato mai visto va visitato, e basta. In
un programma quell'infinito va smorzato, e si scrive
$\beta/\sqrt{N(s_t)+1}$, che è la forma implementata nel codice qui sotto.

Il limite è evidente in spazi grandi o continui: con osservazioni ad alta
dimensione (i pixel di uno schermo) ogni stato è, letteralmente, unico, e
$N(s_t)$ vale sempre $1$. Il conteggio esatto non ha senso. La soluzione sono
gli **pseudo-conteggi**: si stima una densità $\rho(s)$ sugli stati visitati e
se ne ricava un conteggio *effettivo* $\hat N(s)$ coerente con quanto la
densità è "sorpresa" di rivedere $s$. È l'approccio *count-based* esteso agli
spazi grandi {cite}`bellemare2016unifying`, che diede i primi progressi
sostanziali proprio su Montezuma's Revenge.

`````

Il bonus di novità si calcola facilmente quando gli stati sono pochi e
distinti. Il frammento seguente applica la stessa identica regola a sei stati
visitati un numero di volte molto diverso, e mostra come il premio vada quasi
tutto ai più rari:

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

I due numeri si controllano a mano, ed è un conto da fare. Il premio pieno vale
$0{,}5$ (è il `beta` scelto nel codice) e si divide per la radice quadrata di
quante volte quello stato è stato visto, più uno. Lo stato mai visitato prende
quindi $0{,}5$ diviso $1$, cioè il bonus massimo, $0{,}500$; quello battuto
duecento volte prende $0{,}5$ diviso la radice di $201$, che vale poco più di
quattordici, cioè $0{,}035$: quattordici volte meno. La ricompensa aumentata
inclina l'agente verso l'ignoto senza toccare la regola con cui sceglie.

## Curiosità intrinseca: la sorpresa come ricompensa

Contare le visite, anche per approssimazione, resta difficile. Un'idea più
elegante ribalta la domanda: invece di chiederci *quante volte* abbiamo visto
uno stato, chiediamoci *quanto ci sorprende*. Uno stato sorprendente (uno di
cui non sappiamo prevedere le conseguenze) è, per definizione, uno da cui
abbiamo ancora molto da imparare. La sorpresa diventa così una **ricompensa
intrinseca**, generata dall'agente stesso, che affianca quella estrinseca
dell'ambiente.

Di questa idea esistono due realizzazioni classiche, e vale la pena distinguerle
perché la sorpresa la misurano in due modi diversi. La prima si chiama **ICM**
(*Intrinsic Curiosity Module*, modulo di curiosità intrinseca): l'agente si
costruisce un modello di «che cosa succederà se faccio questo», e ogni volta
che sbaglia la previsione incassa un premietto. La seconda si chiama **RND**
(*Random Network Distillation*): due reti fatte allo stesso modo, una congelata
con pesi tirati a caso e mai toccati, l'altra che si allena a imitarla; dove
l'imitazione riesce male, l'agente non è ancora passato abbastanza, e proprio
quello scarto è la misura di novità.

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
$\phi(s)$ appreso, che cattura solo ciò che l'agente *può controllare* e ignora
il rumore irrilevante dell'ambiente. Come si ottenga una proprietà del genere è
metà del lavoro, e vale la pena dirlo: $\phi$ non si addestra da sé, si addestra
con un modello di dinamica **inversa**, una rete che da $\phi(s_t)$ e
$\phi(s_{t+1})$ deve indovinare l'azione $a_t$ che ha portato dall'uno all'altro.
Per riuscirci $\phi$ è costretta a conservare tutto ciò che le azioni
influenzano, e non ha ragione di conservare il resto: una foglia che si muove
per il vento non aiuta a indovinare quale tasto è stato premuto, e quindi esce
dalla rappresentazione. Su quello spazio, un modello *forward* prevede la
feature del prossimo stato $\hat\phi(s_{t+1})$ da $\phi(s_t)$ e dall'azione
$a_t$; la ricompensa
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
tre anni dopo, anche la media umana era superata. Non di più, ed è bene dirlo
con le parole del lavoro stesso, che sull'esito è prudente: l'agente
«occasionalmente completa il primo livello». Occasionalmente, e il primo: dopo
quello il gioco va avanti. Superare il punteggio umano medio e risolvere un
gioco sono due affermazioni diverse, e vale la pena tenerle separate.

`````

Il cuore di RND si scrive in poche righe di PyTorch:

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
    # errore per-stato = novità: alto sugli stati poco visti. La media al posto
    # della somma della formula cambia solo la scala, che poi si ritara comunque
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
ogni posto un "livello di vicinanza all'uscita" e premia l'agente solo per la
differenza di livello fra dove arriva e dove era. Sei alla quota $3$ e arrivi
alla $5$: prendi $+2$. Torni indietro: prendi $-2$. Avanti e indietro fa zero,
quindi oscillare davanti alla porta non frutta niente, e la strategia migliore
resta esattamente quella di prima, solo più facile da trovare.

(Il conto torna così tondo se i premi lontani valgono quanto quelli vicini. Se
il futuro conta un po' meno del presente, come succede quasi sempre, il giro
chiuso non fa esattamente zero. La garanzia regge lo stesso, per una ragione un
po' più sottile: sommando tutti i premietti di un percorso, alla fine sopravvive
solo la quota del punto di partenza, che è la stessa per qualunque strategia e
quindi non ne favorisce nessuna.)

`````

`````{tab} Superiore

Il rischio del reward shaping ingenuo è cambiare la policy ottima: un termine
aggiuntivo mal scelto può rendere conveniente un comportamento che l'obiettivo
originale non premia. Ng, Harada e Russell {cite}`ng1999policy` hanno
dimostrato che esiste una forma di shaping **garantita** a preservare l'ordine
delle policy: il
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

Due curve che si separano: quella che continua a salire è il numero che stiamo
massimizzando, quella che a un certo punto scende è ciò che volevamo davvero. Il
guaio è che se ne vede una sola. Chi guarda il numero che ottimizza vede
miglioramenti fino alla fine, anche molto dopo il punto in cui le cose hanno
cominciato a peggiorare.
```

Il punto di divergenza in {numref}`fig-reward-hacking` porta il nome di
Charles Goodhart, l'economista che nel 1975, parlando di politica monetaria,
osservò come una regolarità statistica tenda a rompersi non appena la si usa
per governare qualcosa. La formulazione che tutti citano («quando una misura
diventa un obiettivo, cessa di essere una buona misura») non è però sua: è
dell'antropologa **Marilyn Strathern**, che la scrisse nel 1997 a proposito
della valutazione delle università britanniche
{cite}`strathern1997improving`, ed è quella che ha portato la legge fuori
dall'economia. Vale per gli agenti come per le organizzazioni, e per la stessa
ragione: la misura era un buon indicatore *finché nessuno ci puntava contro
tutto lo sforzo*.

L'esempio diventato manifesto è di OpenAI {cite}`clark2016faulty`: in
*CoastRunners*, un gioco di
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
funziona) passa in secondo piano.

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

Con questo il capitolo si chiude, e quella frase vale anche per tutto ciò che lo
precede. Dal DQN in avanti ogni sezione ha dato all'agente un pezzo di libertà
in più, e subito dopo ha dovuto inventarsi come contenerla: la memoria delle
esperienze e la copia congelata perché i valori non esplodessero, il guinzaglio
di PPO perché non esplodesse la strategia, i sogni corti perché non esplodesse
l'immaginazione, il recinto attorno all'archivio perché non esplodessero le
stime su ciò che nessuno ha mai provato, l'esperto richiamato a etichettare
perché l'allievo non finisse nel fosso. Il reward hacking è la stessa storia
raccontata all'ultimo livello, quello dell'obiettivo. Con una differenza: lì il
contenimento non è più un accorgimento tecnico, è una domanda su che cosa
vogliamo davvero. Il capitolo sull'AI responsabile comincia da qui.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Quando la ricompensa arriva **di rado** (l'emblema è *Montezuma's Revenge*,
  dove il DQN segnava zero), tirare a caso ogni tanto non basta: è come
  esplorare una città lanciando una monetina a ogni incrocio, si gira per ore
  nello stesso quartiere. Serve una spinta che punti **deliberatamente** verso
  quello che non si è ancora visto.
- Il modo più semplice è il **premio alla novità**, il principio del turista
  curioso: un piccolo premio ogni volta che metti piede in un posto nuovo, che
  si spegne man mano che quel posto diventa familiare. Negli spazi enormi, dove
  ogni schermata è unica e nessun posto si ripete mai, il conteggio non si può
  fare e lo si stima.
- L'idea più elegante è la **curiosità**: il premio non va a ciò che è raro, va
  a ciò che **sorprende**, come il bambino che spinge il bicchiere oltre il
  bordo del tavolo finché non ha imparato cosa succede. L'agente si costruisce
  una previsione di come andrà a finire, e ogni volta che sbaglia la previsione
  incassa. Si spegne da sé: quando ha imparato, non c'è più sorpresa.
- Aggiungere premietti intermedi per guidare l'agente (**reward shaping**)
  funziona, ma solo se sono dati come *differenza di quota*: altrimenti il robot
  scopre che gli conviene oscillare davanti alla porta incassando premietti,
  senza mai uscire.
- Il pericolo grosso ha un nome, **reward hacking**: l'idraulico pagato a tubi
  sostituiti che comincia a sostituire tubi sani, la barca di *CoastRunners* che
  gira in tondo prendendo fuoco. Il problema non è che l'agente disobbedisce, è
  che obbedisce troppo bene, alla lettera sbagliata. Ed è il ponte verso il
  capitolo sull'AI responsabile.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Con **ricompense sparse** (l'emblema è *Montezuma's Revenge*, dove il DQN
  segnava zero) l'esplorazione casuale di $\varepsilon$-greedy fallisce: la
  probabilità di azzeccare $n$ azioni insolite di fila scala come
  $(\varepsilon/|\mathcal{A}|)^n$. Serve esplorazione **diretta**, non rumore
  locale.
- I **bonus di novità** *count-based* premiano gli stati poco visitati
  ($\propto 1/\sqrt{N(s)}$, con un $+1$ a smorzare l'infinito); negli spazi
  grandi il conteggio esatto non ha senso e si usano **pseudo-conteggi**
  derivati da una densità.
- La **curiosità intrinseca** trasforma la *sorpresa* in ricompensa: **ICM**
  usa l'errore di predizione della dinamica in uno spazio di feature appreso,
  **RND** l'errore nel predire una rete casuale fissa. RND fu il primo a
  superare il punteggio umano medio su Montezuma senza dimostrazioni né accesso
  allo stato dell'emulatore; il gioco non lo «risolse», e il paper stesso dice
  che il primo livello lo completa solo occasionalmente.
- Il **reward shaping** densifica il segnale; solo la forma *potential-based*
  $F=\gamma\Phi(s')-\Phi(s)$ (Ng, Harada, Russell, 1999) preserva la policy
  ottima, per un argomento telescopico valido *per qualunque* $\Phi$.
- Il **reward hacking** è l'agente che ottimizza la *lettera* della ricompensa,
  non l'intento, come la barca di *CoastRunners*. La legge che porta il nome di
  Goodhart, nella formulazione che tutti citano, è in realtà di Marilyn
  Strathern (1997). È il ponte verso il problema dell'**allineamento**.
```
`````
