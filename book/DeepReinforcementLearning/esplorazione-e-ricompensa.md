# Esplorazione e ricompensa: curiosità, ricompense rade, reward hacking

DeepMind mise alla prova il DQN su 49 giochi Atari, nel 2015, e su decine di
essi l'agente resse il confronto con un collaudatore umano professionista. Su
uno solo collezionò un punteggio desolante: **zero** {cite}`mnih2015human`.

Il gioco è *Montezuma's Revenge*, un platform del 1984. Un esploratore in un
tempio azteco deve scendere una scala, saltare una fune, scansare un teschio
rotolante e raccogliere una chiave prima di ricevere il suo primo punto: decine
di mosse esatte, in sequenza, per un solo segnale di «bene». Un agente che
sceglie mosse a caso non arriverà mai in fondo a quella catena: cadrà, morirà, e
non vedrà mai una ricompensa da cui imparare.

Notare che cosa *non* è andato storto. Il DQN quello schermo lo vedeva
benissimo, e i suoi conti li faceva come sugli altri quarantotto giochi. Il muro
è un altro, ed è il tema di questa sezione: come si va a cercare qualcosa in un
mondo dove le ricompense sono **rade**, o come si dice di solito **sparse**,
cioè capitano una volta ogni tanto e in mezzo non c'è niente. Si chiama
**esplorazione**.

## Il problema delle ricompense sparse

Nella maggior parte dei giochi Atari qualcosa di buono o cattivo capita a ogni
secondo, e l'agente ha un flusso costante di segnali da cui correggersi. Ma
quando la ricompensa arriva solo dopo lunghe sequenze di azioni giuste (la
chiave, la porta, il livello), il segnale diventa un ago in un pagliaio.
Finché l'agente non inciampa *per caso* in quella prima ricompensa, non ha
nulla che gli dica in che direzione andare. E il conto di quanto sia improbabile
inciamparci si fa a mente: se a ogni passo ci sono otto mosse possibili e una
sola è quella giusta, azzeccarne dieci di fila vuol dire $8^{10}$, cioè una
volta su un miliardo abbondante.

Nel {doc}`capitolo sul reinforcement learning </ReinforcementLearning/overview>` abbiamo introdotto il dilemma
esplorazione–sfruttamento e la strategia detta
$\varepsilon$-greedy: agire quasi sempre secondo la stima migliore, ma ogni
tanto, con una piccola probabilità, scegliere a caso. Quella probabilità è la
lettera greca $\varepsilon$ («epsilon»), che in matematica indica per tradizione
una quantità piccola; e *greedy*, «goloso», è il resto del tempo, quando
l'agente prende senza esitare la mossa che ha il voto più alto. Qui vediamo
perché quella ricetta, in ambienti come Montezuma, non basta, e cosa si è
inventato per andare oltre.

## Perché $\varepsilon$-greedy non basta

`````{tab} Elementare

Immagina di esplorare una città enorme tirando un dado a ogni incrocio per
decidere dove girare. Andrai avanti e indietro nello stesso quartiere per ore:
il caso non ha memoria, non sa quali strade hai già battuto e quali no. Per
raggiungere un vicolo che sta a dieci svolte precise da qui, la probabilità di
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

L'idea più intuitiva è anche la più vecchia, e nasce ai tempi in cui i giudizi
stavano in una tabella {cite}`sutton2018reinforcement`. Se in una certa
situazione l'agente si è trovato poche volte, di quella situazione sa poco, e
allora conviene andarci. Basta tenere il conto di quante volte ci è passato e
aggiungere alla ricompensa un premietto che scende man mano che quel conto
sale.

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
quattordici, cioè $0{,}035$: quattordici volte meno.

La radice quadrata, in quel conto, non è un capriccio: serve a far scendere il
premio in fretta all'inizio e piano dopo. Fra zero visite e tre il premio si
dimezza esatto, da $0{,}500$ a $0{,}250$, e la differenza si sente; fra cento e
centotré scende da $0{,}0498$ a $0{,}0490$, cioè non cambia niente. Ed è giusto
così: la centesima visita a un posto insegna molto meno della prima.

Notare che cosa non è cambiato: la regola con cui l'agente sceglie è quella di
sempre, prendere la mossa col voto più alto. È solo il voto ad avere adesso un
pezzo in più, e l'agente si dirige verso l'ignoto credendo di dirigersi verso il
guadagno. Non gli si è insegnata la curiosità: gliel'hanno pagata.

Questo però funziona finché le situazioni si possono contare. Davanti allo
schermo di un videogioco no: ogni schermata è unica, basta che un puntino si
sposti e non l'hai mai vista prima, quindi il conto delle visite vale sempre
uno e il premietto viene identico dappertutto, il che è come non darlo. La via
d'uscita è smettere
di contare e cominciare a **stimare**: si valuta quanto una schermata assomigli
a quelle già viste, e da quella somiglianza si ricava un conteggio finto, uno
*pseudo-conteggio*, che si usa al posto di quello vero. È così che, nel 2016,
sono arrivati i primi progressi veri proprio su Montezuma's Revenge.

## Curiosità intrinseca: la sorpresa come ricompensa

Stimare un conteggio, però, resta un mestiere delicato. Un'idea più elegante
ribalta la domanda: invece di chiederci *quante volte* abbiamo visto una
situazione, chiediamoci *quanto ci sorprende*. Se non sappiamo prevedere che
cosa succederà, vuol dire che di quel pezzo di mondo non abbiamo ancora capito
il funzionamento, e sono proprio i posti da cui c'è da imparare. La sorpresa
diventa così una ricompensa che l'agente si dà da sé, e che si somma a quella
che gli dà il gioco.

Di questa idea esistono due realizzazioni classiche, e la differenza sta tutta
nel come misurano la sorpresa. La prima si chiama **ICM** (*Intrinsic Curiosity
Module*, modulo di curiosità intrinseca): l'agente si costruisce una previsione
di «che cosa succederà se faccio questo», e ogni volta che la sbaglia incassa un
premietto.

La seconda si chiama **RND** (*Random Network Distillation*) e sembra un gioco
di prestigio, quindi conviene smontarla. Si prendono due reti fatte allo stesso
modo. La prima ha i numeri interni tirati a caso e non si tocca più: data una
schermata sputa fuori un altro numero, che non significa niente ed è però
sempre lo stesso per la stessa schermata. La seconda rete ha un compito solo,
indovinare che numero dirà la prima, e si allena a farlo sulle schermate per cui
l'agente passa davvero. Ecco allora perché lo scarto fra le due misura la
novità: dove l'agente è passato tante volte la seconda rete ha avuto tempo di
imparare la risposta a memoria e sbaglia poco; su una schermata nuova non ha mai
studiato quella risposta, e sbaglia parecchio.

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
è l’**errore di predizione** di un modello di dinamica. La chiave è che la
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
r_t^{\text{int}} = \frac{\eta}{2}\,\big\lVert \hat\phi(s_{t+1}) -
\phi(s_{t+1}) \big\rVert^2 ,
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
    # errore per ogni stato = novità: alto sugli stati poco visti.
    # (media invece di somma: cambia solo la scala, e la scala si ritara a parte)
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

## Un modo diverso di guardare la curiosità

Prima di cambiare argomento conviene fermarsi su una cosa che, messa così,
sembra ovvia e non lo è. In tutto quello che abbiamo visto finora la curiosità
è un **premio in più**: c'era una ricompensa, ci siamo accorti che non
bastava, e gliene abbiamo affiancata un'altra, fabbricata da noi. È una toppa
che funziona benissimo, ma resta una toppa, e la frase di prima («non gli si è
insegnata la curiosità: gliel'hanno pagata») lo dice meglio di qualunque
commento.

Esiste una lettura opposta, e chiarisce parecchio anche a chi non intende
seguirla. Nel quadro dell’**inferenza attiva**, che nelle neuroscienze teoriche
descrive percezione e azione come un unico problema {cite}`parr2022active`,
l'agente non massimizza una ricompensa: minimizza una grandezza (l’**energia
libera attesa**) che tiene insieme due cose, quanto un'azione lo avvicina a ciò
che preferisce e quanto gli farebbe **guadagnare informazione**. All'inferenza
attiva il {doc}`capitolo sui world model </WorldModels/overview>` dedica una sezione, e il capitolo
sull'auto-supervisione se ne serve per rispondere a un'obiezione sul rinforzo:
qui ci interessa solo il riflesso che getta su questa pagina.

`````{tab} Elementare

La differenza che conta è questa: lì il valore di sapere non è un premio
aggiunto, **c'era già dall'inizio**, accanto al valore di ottenere. E allora non
c'è nessun dosaggio da regolare fra il curiosare e l'incassare, perché tutti e
due sono pezzi della stessa quantità.

Vista da lì, la storia di questa sezione si legge al contrario. Non è che
abbiamo aggiunto la curiosità a un agente che non ce l'aveva: è che, partendo da
una ricompensa che dice solo «quanto ti è andata bene» e mai «quanto hai
imparato», eravamo **obbligati** a rimetterla dentro a mano. Ogni coefficiente
che in questa sezione dosa il peso del bonus è il prezzo di quella scelta
iniziale.

`````

`````{tab} Superiore

Formalmente il legame è più stretto di un'analogia. Gli autori mostrano che
diversi schemi noti si riottengono **togliendo pezzi** alla loro grandezza:
annullate le preferenze dell'agente, l'energia libera attesa **cambiata di
segno** «è variamente nota come sorpresa bayesiana attesa (nel contesto
dell'esplorazione attentiva) o **motivazione intrinseca** (nel contesto
dell'apprendimento autonomo)» {cite}`parr2022active`, che è esattamente la
famiglia di metodi di questa sezione. Il segno non è un dettaglio: quella
grandezza si **massimizza**, come si massimizza il bonus di novità qui sopra,
mentre l'energia libera attesa da cui viene si minimizza.

Il rapporto fra le due letture non è quindi di concorrenza: la ricompensa
intrinseca è il caso particolare che si ottiene spegnendo il termine
pragmatico. E c'è un motivo per cui la cosa riguarda proprio questa pagina: in
un ambiente a ricompense rade il termine pragmatico, per la gran parte della
traiettoria, vale lo stesso per tutte le mosse, e allora smette di ordinarle.
Quel che resta a decidere è il termine epistemico, cioè la curiosità. È una
conseguenza della decomposizione, non una misura: chi la volesse usare come tale
dovrebbe verificarla.

Resta la differenza che conta per chi implementa: questo quadro nasce come
teoria del comportamento biologico, e i sistemi che oggi arrivano più lontano in
*Montezuma's Revenge* sono quelli di questa sezione, non quelli dell'inferenza
attiva. Serve a capire da dove viene la toppa, non a sostituirla.

`````

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
uscire davvero. Gli hai insegnato a *inseguire il premietto*, non a uscire: hai
cambiato il gioco senza accorgertene, ed è questo il rischio da cui ci si vuole
difendere.

Per fortuna esiste un modo di dare questi aiuti che quel rischio non ce l'ha,
mai, comunque lo si usi. L'idea, dovuta ad Andrew Ng e colleghi nel 1999, è
dare l'aiuto come *differenza di quota*. A ogni posto del labirinto si attacca
un numero, la sua «quota», e si premia il robot soltanto per la differenza fra
la quota di dove arriva e la quota di dove era. Sei alla quota $3$ e arrivi alla
$5$: prendi $+2$. Torni indietro: prendi $-2$. Avanti e indietro fa zero, quindi
oscillare davanti alla porta non frutta più niente.

Verrebbe da obiettare: se so già dare a ogni posto un numero che dice quanto è
vicino all'uscita, il labirinto non l'ho già risolto? Ed è qui la finezza:
**la quota non deve essere giusta**. Può essere una stima grossolana, per
esempio la distanza in linea d'aria dall'uscita, che ignora i muri e ogni tanto
manda dalla parte sbagliata. Se la stima è buona il robot impara molto più in
fretta, perché a ogni passo ha un segnale invece del buio; se è pessima non
aiuta, e può anche fargli perdere tempo. Ma in nessuno dei due casi gli fa
imparare la cosa sbagliata: la strategia migliore resta quella di prima. È
tutto lì il valore della garanzia.

(Un'ultima precisazione, per chi ha fatto il conto. Il «vado e torno fa zero»
funziona così tondo solo se un premio incassato più tardi vale quanto uno
incassato subito. Di solito non è così: come si è visto nel corridoio di Dyna,
i premi lontani si scontano, cioè valgono un po’ meno. Il conto giusto, allora,
non è la somma nuda dei premietti: è quella somma con ciascun premietto già
moltiplicato per quanto vale a quella distanza. E in *quella* somma i pezzi si
cancellano di nuovo a due a due, perché ogni quota ci entra con lo stesso peso
in positivo e in negativo. Alla fine sopravvive soltanto la quota della casella
da cui si è partiti, che è la stessa per qualunque strategia: un vantaggio
uguale per tutti non favorisce nessuno.)

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
ha un nome: **reward hacking**, cioè «scassinare la ricompensa» (si dice anche
*specification gaming*, «giocare sulle regole scritte»). L'agente ottimizza
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

Il punto di divergenza in {numref}`fig-reward-hacking` ha una formulazione
celebre: «quando una misura diventa un obiettivo, cessa di essere una buona
misura». Vale per gli agenti come per le organizzazioni, e per la stessa
ragione: la misura era un buon indicatore *finché nessuno ci puntava contro
tutto lo sforzo*.

La legge porta il nome di Charles Goodhart, un economista britannico che nel
1975 la osservò a proposito della moneta: le banche centrali usavano certi
indicatori per capire come andasse l'economia, e quegli indicatori smisero di
funzionare non appena si cominciò a governarli. Ma quella frase così memorabile
non è sua: è dell'antropologa **Marilyn Strathern**, che la scrisse nel 1997
studiando come si valutano le università britanniche
{cite}`strathern1997improving`, ed è la formulazione che ha portato la legge
fuori dall'economia.

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
in più, e subito dopo ha dovuto inventarsi come contenerla.

L'elenco, riletto tutto insieme, è impressionante. Nella sezione su DQN, la
memoria delle esperienze e la copia congelata, perché i voti non esplodessero.
Nei gradienti di policy, il guinzaglio di PPO, perché non esplodesse la
strategia. Nel RL basato su modello, i sogni corti, perché non esplodesse
l'immaginazione. Nell'offline RL, il recinto attorno all'archivio, perché non
esplodessero le stime su ciò che nessuno ha mai provato. Nell'imitazione,
l'esperto richiamato a etichettare, perché l'allievo non finisse nel fosso.

Il reward hacking è la stessa storia raccontata all'ultimo livello, quello
dell'obiettivo, e con una differenza: qui il contenimento non è più un
accorgimento tecnico, è una domanda su che cosa vogliamo davvero. Il capitolo
sull'AI responsabile comincia da qui.

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
  funziona, ma può cambiargli l'obiettivo sotto il naso: il robot scopre che gli
  conviene oscillare davanti alla porta incassando premietti, senza mai uscire.
  C'è però un modo di darli che quel rischio non ce l'ha mai, e sono le
  *differenze di quota*.
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
  Strathern (1997). È il ponte verso il problema dell’**allineamento**.
```
`````

Quel ponte, però, è lungo: all'AI responsabile si arriva fra molti capitoli,
quasi in fondo al libro. La pagina dopo riparte da un'altra storia, la
traduzione automatica del 1954, e non è una deviazione. Qui la ricompensa la
scriveva un programma, e si è visto che cosa succede quando la scrive male;
quando l'allineamento tornerà, a scriverla sarà una persona che legge due frasi
e dice quale preferisce. Prima bisogna sapere che cos'è una frase, per una
macchina.
