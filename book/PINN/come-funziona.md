# La legge dentro la loss

Dentro PyTorch c'è un registratore. Nel capitolo che gli è dedicato lo abbiamo
raccontato così: annota i calcoli mentre li fai e poi li riavvolge
all'indietro, e a ogni addestramento di questo libro ha risposto sempre alla
stessa domanda, *se ritocco questo peso, quanto cambia l'errore?* Si chiama
**autograd**, e finora ha fatto un mestiere solo: milioni di volte la stessa
domanda, per addestrare classificatori, traduttori, generatori.

Questa sezione comincia con un colpo di scena: la domanda si può cambiare.
Quel registratore non sa che cosa siano «i pesi», e non gliene importa. Sa
rispondere a «se muovo *questo*, di quanto cambia *quello*» su qualunque
coppia di numeri compaia nei suoi conti. Possiamo allora puntarlo altrove.

Prendiamo una rete che riceve un istante di tempo e restituisce un numero, per
esempio la posizione di un oggetto in quell'istante: disegnata su un foglio, è
una curva. Ora chiediamo al registratore, invece del solito «di quanto cambia
l'errore se ritocco questo peso», quest'altro: «di quanto cambia il numero in
uscita se sposto di pochissimo l'istante che ti ho dato?». La risposta è la
**pendenza** della curva in quel punto, cioè quanto in fretta sta salendo o
scendendo proprio lì. Se la risposta è 2, vuol dire che nell'intorno di
quell'istante la curva sale di due quadretti ogni quadretto che si va a
destra; se è $-0{,}5$, scende di mezzo quadretto; se è 0, lì è in piano. E
rifacendo la stessa domanda sulla pendenza si ottiene la **curvatura**, quanto
in fretta la pendenza stessa sta cambiando, cioè quanto la curva piega.

Le due risposte arrivano precise. Non sono stime ricavate confrontando due
istanti vicini, che è il modo in cui una pendenza si misura di solito: sono le
pendenze vere, sbagliate solo di quel pochissimo che sbaglia un calcolatore
per il fatto di lavorare con un numero finito di cifre. In matematica pendenza
e curvatura si chiamano **derivata prima** e **derivata seconda**, e si
scrivono con gli apici: se la curva della rete è $u_\theta(t)$ (dove $t$ è
l'istante e $\theta$, «theta», sta per tutti i pesi della rete messi insieme),
la pendenza è $u_\theta'(t)$ e la curvatura $u_\theta''(t)$.

La rete allora smette di essere soltanto una scatola addestrabile e diventa
qualcosa di più: una curva **liscia** (che vuol dire una cosa precisa, senza
spigoli, senza punti in cui cambia direzione di colpo) di cui si sa dire, in
ogni punto, quanto è alta, quanto sale e quanto piega. Ed è esattamente ciò
che serve per chiederle di rispettare un'equazione differenziale, che di
quelle tre cose parla e non d'altro.

Tutto il metodo delle PINN sta in questa mossa, chiedere le derivate rispetto
all'ingresso invece che ai pesi. Vediamola all'opera.

## Tre vincoli e una penalità

L'idea si capisce meglio raccontandola come un compito in classe.

`````{tab} Elementare

Immagina uno studente alle prese con un compito insolito: disegnare, su un
foglio a quadretti, la curva di una molla che oscilla, cioè di quanto il corpo
appeso è spostato dalla sua posizione di riposo, istante per istante. Il tempo
scorre verso destra; sopra la riga di mezzo il corpo è più in alto del riposo,
sotto è più in basso, e la riga di mezzo è il riposo. Nessuna tabella di
valori da copiare. Solo tre vincoli:

1. la curva deve **partire dal punto giusto** (la molla è stata tirata fino
   a una certa altezza);
2. deve partire **in piano**, cioè con pendenza zero (il corpo è stato
   lasciato andare da fermo, e se all'inizio non si muove la curva all'inizio
   non sale né scende);
3. in **ogni punto del foglio** deve rispettare la regola della molla: quanto
   la curva piega in quel punto dev'essere coerente con quanto è alta e con
   quanto sta scendendo nello stesso punto («coerente» vuol dire che c'è una
   formula che lega le tre cose, e fra poche pagine la scriveremo con i numeri
   veri).

Il professore corregge in modo semplice e spietato: controlla la partenza, poi
punta il dito su una manciata di istanti e lì verifica la regola; ogni
violazione costa punti. Quegli istanti li ha sorteggiati una volta sola,
all'inizio, e da lì in poi controlla sempre quelli, il che sembra un dettaglio
da bidello e sarà invece la chiave di tutta la sezione. Lo studente ritocca la
curva e
riconsegna, ancora e ancora, finché i punti persi non si riducono a
briciole. Si noti la stranezza: *nessuno dei due conosce la soluzione*. Il
professore sa solo verificare la regola. Eppure alla fine la curva giusta
salta fuori, perché tra tutte le curve possibili quella vera è l'unica che
parte così *e* rispetta la regola dappertutto. Una PINN è esattamente questo
studente: la curva è la rete, i punti persi sono la loss, e gli istanti su cui
il professore punta il dito si chiamano **punti di collocazione**.

Su quel «dappertutto» conviene tenere un dito, perché è la parola su cui si
gioca tutto il resto della sezione. La curva vera rispetta la regola in ogni
singolo punto del foglio; il professore, invece, la controlla in una manciata
di punti soltanto. Sono due cose diverse, e fra poche pagine ci costeranno
care.

`````

`````{tab} Superiore

Sia $u_\theta : [0, T] \to \mathbb{R}$ una rete neurale con parametri
$\theta$, candidata a risolvere un'equazione differenziale che scriviamo in
forma compatta $\mathcal{N}[u](t) = 0$, con condizioni iniziali
$u(0) = u_0$ e $u'(0) = v_0$. Il **residuo** della candidata è
$r_\theta(t) = \mathcal{N}[u_\theta](t)$: vale zero esattamente dove la rete
rispetta l'equazione. La loss da minimizzare è

$$
\mathcal{L}(\theta) =
\underbrace{\frac{1}{N_c} \sum_{j=1}^{N_c} r_\theta(t_j)^2}_{\text{fisica}}
\;+\;
\lambda_0 \underbrace{\Big[ \big(u_\theta(0) - u_0\big)^2
+ \big(u_\theta'(0) - v_0\big)^2 \Big]}_{\text{condizioni iniziali}},
$$

dove i $t_j$ sono gli $N_c$ **punti di collocazione** (istanti sparsi nel
dominio, casuali o equispaziati, nei quali esigiamo il rispetto
dell'equazione) e $\lambda_0 > 0$ bilancia i due termini. Lo chiamiamo
$\lambda_0$, e non $\lambda$, perché non è lo stesso peso della loss vista in
apertura di capitolo: là $\lambda$ moltiplicava il termine di fisica, qui il
peso sta sulle condizioni iniziali. Dove metterlo è convenzione; ciò che
conta è il rapporto fra i termini. Per una PDE su un
dominio spaziale si aggiunge un termine identico per le **condizioni al
contorno**, con punti campionati sul bordo; e se esistono misure $(t_i, u_i)$
si aggiunge il termine dati
$\frac{1}{N_d}\sum_{i=1}^{N_d} \big(u_\theta(t_i) - u_i\big)^2$, come nella
loss vista in apertura di capitolo. Tutte le derivate che compaiono in
$r_\theta$ (per noi $u_\theta'$ e $u_\theta''$) le fornisce la
differenziazione automatica rispetto all'ingresso $t$: esatte, senza rapporti
incrementali né passo di discretizzazione.

Un dettaglio che sembra pedante e non lo è: il termine di fisica, da solo,
ha un minimo banale, perché la funzione $u \equiv 0$ risolve l'equazione
omogenea con residuo nullo ovunque. Il termine sulle condizioni iniziali
serve dunque a **selezionare**, dentro la famiglia delle soluzioni
dell'equazione (per una lineare del secondo ordine come quella della prossima
sezione, uno spazio a due dimensioni), proprio la nostra: senza di esso nulla
distingue la traiettoria che parte da $u(0)=1$ da quella che se ne sta ferma
a zero.
Attenzione però a non promettere troppo: nemmeno con quel termine il
minimizzatore della loss è unico in senso stretto. Con $N_c$ punti di
collocazione *finiti*, infinite funzioni annullano il residuo **in quei
punti** e rispettano le due condizioni iniziali; la soluzione vera è l'unico
minimo globale se ci si restringe alle soluzioni dell'equazione, ovvero nel
limite in cui il residuo è controllato su tutto il dominio e non solo sul
campione. Nel mezzo dovrebbe pensarci la regolarità della rete, che a rigore
non le impedisce affatto di oscillare fra un punto di collocazione e il
successivo: è un argomento asintotico, vale infittendo i punti, e a $N_c$
finito garantisce assai meno di quanto sembri. Ecco perché i punti vanno
abbastanza fitti rispetto alle scale della soluzione, e perché fra poche
pagine vedremo una rete addestrata così infilare un picco di residuo proprio
nel buco fra due punti di controllo.

Perché allora in pratica si sceglie $\lambda_0$ ben maggiore di 1? Non per
selezionare il minimo, ma per raggiungerlo: i due termini non pesano allo
stesso modo sulla discesa, e il rischio è che i pesi si muovano quasi solo
nella direzione dettata dalla fisica, trascurando l'unico ancoraggio che c'è.
Le ragioni per cui succede sono due, e conviene tenerle distinte perché non
agiscono sempre insieme.

La prima è l’**ampiezza dei gradienti**. Il residuo si ottiene applicando alla
rete degli operatori differenziali, e i gradienti che tornano indietro da quel
ramo *possono* essere di ordini di grandezza più grandi di quelli del termine
sulle condizioni iniziali. Succede sulle PDE con operatori di ordine alto e
condizioni campionate su una superficie, ed è lo squilibrio che Wang, Teng e
Perdikaris documentano, correggendolo con pesi ristimati durante
l'addestramento {cite}`wang2021understanding`. Sul problema di questa sezione,
però, il divario misurato è molto più modesto. All'inizializzazione, cioè nel
momento in cui $\lambda_0$ va scelto, il rapporto fra le ampiezze medie dei
gradienti dei due rami (la media di $|\partial \mathcal{L} / \partial \theta|$
su tutti i pesi, sui semi da 0 a 19) ha mediana $2{,}4$, e in quattro semi su
venti è addirittura rovesciato. Su una ODE del secondo ordine con due scalari
imposti al tempo zero, «ordini di grandezza» sarebbe una parola grossa.

La seconda, ed è quella che qui morde davvero, è la **copertura del dominio**:
le condizioni iniziali riguardano un istante soltanto, mentre il residuo tira
sull'intera curva in tutti i punti di collocazione e finisce per dettare quasi
da solo come cambiare i pesi. Non è un conteggio nel senso di prima (entrambi
i termini restano medie, e infittire i punti di collocazione non sposta la
bilancia), è una questione di dove i due termini guardano.

In tutti e due i casi il rimedio è lo stesso, dare voce al termine debole; e
in tutti e due i casi il valore giusto va scelto a mano, provando. È il primo
dei limiti che la prossima sezione mette in fila.

Attenzione però a non dare per scontato l'effetto: quando fra poche pagine
misureremo che cosa succede davvero a $\lambda_0 = 1$, la rete rispetterà le
condizioni iniziali lo stesso, e sbaglierà per un'altra strada. Il peso, su
questo problema, non compra l'ancoraggio: compra il percorso che ci arriva.

`````

```{figure} ../figures/pinn-schema.svg
:name: fig-pinn-schema
:alt: "Schema del metodo PINN, da sinistra a destra. Le coordinate entrano in una rete neurale, che restituisce la curva candidata. Da lì partono due rami: in alto quello della fisica, dove si calcolano le pendenze della curva e si misura di quanto viola la regola; in basso quello delle condizioni di partenza, di quelle sui bordi e delle eventuali misure. I due rami si sommano in un punteggio unico, dal quale una freccia tratteggiata torna indietro fino ai pesi della rete."
:width: 100%

L'anatomia di una PINN, da sinistra a destra: le coordinate entrano nella
rete, che risponde con la curva candidata; da lì partono due controlli, quello
della regola fisica (in alto) e quello delle condizioni di partenza, dei bordi
e delle eventuali misure (in basso); i due si sommano in un punteggio unico, e
la correzione torna indietro fino ai pesi della rete.
```

In {numref}`fig-pinn-schema` c'è il metodo per intero, e le poche scritte in
formula dicono cose che ormai abbiamo in mano: $\mathcal{L}$, la «L» decorata,
è il punteggio, cioè la loss; $\theta$ sono i pesi; «autograd» è il
registratore di cui parlavamo in apertura. L'unico simbolo nuovo è
$\partial$, quella «d» arrotondata: è il modo di scrivere una pendenza quando
la grandezza cambia per più di un motivo insieme. La sbarra di ferro della
pagina precedente, scaldata a un capo, è il caso tipico: la sua temperatura
cambia sia da un punto all'altro della sbarra sia da un istante al successivo,
e $\partial u / \partial t$ vuol dire «quanto cambia col tempo, tenendo fermo
il punto in cui guardo».

Lo scarto fra i due membri dell'equazione, quello che il ramo in alto calcola,
si chiama **residuo**: è lo stesso oggetto che nel racconto del compito in
classe erano i punti persi per una violazione della regola. Un'avvertenza che
tornerà utile: nel resto della sezione «residuo» indicherà quasi sempre il
*punteggio* che se ne ricava, cioè la media dei residui elevati al quadrato
sui punti di controllo. Il quadrato serve a due cose, a contare uguale una
violazione in su e una in giù, e a far pesare di più quelle grosse; e la
conseguenza da tenere a mente è che quel punteggio cresce con il **quadrato**
della violazione, quindi un punteggio cento volte più alto vuol dire una
violazione dieci volte più grossa.

Lo schema è disegnato nel caso generale, quello di un'equazione con una
coordinata di spazio e una di tempo; nel resto della sezione lavoreremo sul
caso più semplice, con il solo tempo in ingresso. Vale la pena fissare il ramo
in alto, quello giallo-bruno (color ocra): quelle derivate *rispetto
all'input* non compaiono in nessun'altra architettura di questo libro. È il
pezzo nuovo, ed è tutto qui.

## Una molla come banco di prova

Ci serve un problema abbastanza semplice da avere una soluzione esatta con cui
dare i voti alla rete, e abbastanza ricco da non essere un giocattolo. Il
classico dei classici: l’**oscillatore armonico smorzato**, cioè un corpo
appeso a una molla, con un po’ d'attrito che spegne piano piano le
oscillazioni. La legge di Newton per questo sistema è

$$
m\,u''(t) + c\,u'(t) + k\,u(t) = 0,
\qquad u(0) = 1, \quad u'(0) = 0,
$$

dove $u(t)$ è lo spostamento dalla posizione di riposo, $u'$ e $u''$ sono la
pendenza e la curvatura di poco fa, $m$ è la massa, $c$ il coefficiente di
smorzamento (l'attrito) e $k$ la rigidezza della molla. Quel «$= 0$» chiede
una cosa sola: in ogni istante i tre pezzi, sommati, devono dare zero. È la
regola che la curva deve rispettare punto per punto. La riga a destra,
$u(0)=1$ e $u'(0)=0$, dice invece da dove si parte: al tempo zero il corpo è
spostato di 1 e viene lasciato andare da fermo. Di 1 che cosa non importa,
perché non l'abbiamo mai fissato: centimetri, metri, quello che si preferisce.
Tutti gli scarti di questa sezione andranno letti nella stessa unità, come
frazioni di quello spostamento iniziale. Scegliamo poi numeri concreti per la
molla: $m = 1$, $c = 0{,}4$, $k = 4$.

`````{tab} Elementare

Prima di leggere l'equazione, saldiamo i due vocabolari, perché sono lo stesso
vocabolario. La curva sul foglio *è* il movimento del corpo appeso: la sua
pendenza è la **velocità** (quanto in fretta il corpo si sposta) e la sua
curvatura è l’**accelerazione** (quanto in fretta cambia quella velocità). Un
nome viene dal disegno, l'altro dalla fisica, e l'oggetto è lo stesso. È il
motivo per cui una regola sul moto di un corpo si può far rispettare a una
linea tracciata su un foglio.

Adesso l'equazione si legge come una regola di buon senso. Portiamo a destra
tutto tranne il primo pezzo; la massa vale 1, quindi non si vede, e resta:
*accelerazione* $= -4 \times$ *posizione* $- 0{,}4 \times$ *velocità*. Due
forze, cioè. La molla richiama sempre verso il centro, tanto più forte quanto
più sei lontano, ed è il fattore 4; l'attrito frena sempre, tanto più quanto
più vai veloce, ed è il fattore 0,4. I due segni meno dicono che tutte e due
lavorano contro il movimento. Facciamo il conto a mano sull'istante iniziale:
posizione 1, velocità 0, quindi accelerazione
$= -4 \cdot 1 - 0{,}4 \cdot 0 = -4$, e il corpo parte richiamato con decisione
verso il centro.

Il film completo lo conosce chiunque abbia giocato con una molla: il corpo
oscilla su e giù, e ogni oscillazione è un po’ più bassa della precedente,
perché l'attrito ruba energia a ogni passaggio. Con i nostri tre numeri
un'oscillazione completa dura **3,16 secondi**, e dopo 10 secondi l'ampiezza
(l'altezza del rimbalzo, misurata dalla posizione di riposo) è scesa al
**13,5%** di quella di partenza.

Nessuno dei due valori è stato misurato in laboratorio: escono dai tre numeri
della molla, e a grandi linee si capisce anche da dove. La durata di
un'oscillazione la decide la rigidezza rispetto alla massa: qui il rapporto è
4, la sua radice è 2, e da lì viene un giro completo ogni $2\pi/2 \approx
3{,}14$ secondi, che l'attrito rallenta appena fino a 3,16. Il calo, invece,
lo decide l'attrito da solo, e vale un fattore fisso per ogni secondo che
passa: dieci secondi di quel fattore fanno il 13,5% che resta. Questa è la
curva che lo
studente del compito in classe deve disegnare, e che la nostra rete dovrà
imparare senza vederne neppure un punto, tranne la partenza.

`````

`````{tab} Superiore

È un'equazione lineare del secondo ordine a coefficienti costanti: si
risolve con l'equazione caratteristica $m s^2 + c s + k = 0$,
ovvero $s^2 + 0{,}4\,s + 4 = 0$ (la lettera $s$, e non $r$, che in questo
capitolo è già il residuo). Il discriminante è negativo
($0{,}16 - 16 < 0$): radici complesse coniugate
$s = -\gamma \pm i\,\omega_d$, con

$$
\gamma = \frac{c}{2m} = 0{,}2,
\qquad
\omega_d = \sqrt{\frac{k}{m} - \gamma^2}
= \sqrt{4 - 0{,}04} = \sqrt{3{,}96} \approx 1{,}98997,
$$

dove $\gamma$ è il tasso di decadimento e $\omega_d$ la **pulsazione
smorzata**, appena più lenta della pulsazione naturale
$\omega_0 = \sqrt{k/m} = 2$: il fattore di smorzamento vale
$\zeta = \gamma/\omega_0 = 0{,}1$, smorzamento debole. La soluzione generale è
$e^{-\gamma t}(A\cos\omega_d t + B\sin\omega_d t)$; imponendo $u(0)=1$ si
ottiene $A = 1$, imponendo $u'(0)=0$ si ottiene
$B = \gamma/\omega_d \approx 0{,}1005$. Quindi

$$
u(t) = e^{-0{,}2\,t}\left( \cos(\omega_d\,t)
+ \frac{0{,}2}{\omega_d}\,\sin(\omega_d\,t) \right),
\qquad \omega_d = \sqrt{3{,}96}.
$$

Verifica dei conti: $u(0) = 1 \cdot (1 + 0) = 1$; derivando,
$u'(0) = -\gamma \cdot 1 + \omega_d \cdot \gamma/\omega_d = -0{,}2 + 0{,}2
= 0$. Tornano entrambe. Il periodo è $T = 2\pi/\omega_d \approx 3{,}16$ e
l'inviluppo $e^{-0{,}2 t}$ vale $e^{-2} \approx 0{,}135$ per $t = 10$: in
tre oscillazioni abbondanti l'ampiezza cala al 13,5% di quella iniziale.
Questa formula sarà la pagella con cui giudicheremo la PINN.

`````

## La curva dev'essere liscia: perché tanh e non ReLU

Prima di scrivere la rete c'è una scelta da fare, e in ogni altro capitolo del
libro sarebbe stata automatica. Una rete non è fatta solo di somme: fra uno
strato e l'altro ogni numero passa attraverso una funzioncina che lo piega, la
**funzione di attivazione**, ed è lei a decidere che forma possono avere le
curve che la rete sa disegnare. Dal capitolo sulle reti neurali in poi abbiamo
usato quasi sempre la stessa, la ReLU, che è la scelta giusta praticamente
ovunque. Qui è squalificata in partenza, e il motivo è istruttivo.

`````{tab} Elementare

**La curva dev'essere liscia**, e la ReLU non sa disegnare curve lisce. La
ReLU è fatta di due tratti dritti attaccati in un angolo, e una rete di sole
ReLU produce curve fatte così: segmenti dritti incollati uno dopo l'altro,
come una spezzata. Una spezzata però non ha curvatura da nessuna parte,
perché un tratto dritto non piega, e negli angoli, dove piegherebbe, la
curvatura non si riesce nemmeno a calcolare. Ma la regola della molla parla
proprio di curvatura, che ne è anzi il termine principale: con una curva a
spezzata il professore non vedrebbe più il pezzo più importante della regola,
e qualunque disegno gli sembrerebbe corretto. Serve dunque una funzioncina che
pieghi dolcemente dappertutto, senza angoli. Quella che si usa è una S
sdraiata e centrata nello zero: viene su da sinistra dove è quasi piatta, si
impenna passando per il centro e torna a spianarsi a destra, e in nessun punto
ha uno spigolo. Si chiama **tanh** e nel programma della prossima pagina
compare come `nn.Tanh()`. Nel resto del deep learning la ReLU l'aveva mandata
in pensione; qui si prende la rivincita.

`````

`````{tab} Superiore

La ReLU è fatta di due semirette: una rete di sole ReLU calcola una funzione
*lineare a tratti*, la cui derivata prima è a gradini e la cui **derivata
seconda è zero quasi ovunque** ("quasi" perché nei punti di piega non esiste
affatto, e sono un insieme di misura nulla). Ma nel nostro residuo compare
$u''$: per una rete ReLU sarebbe identicamente nullo, e il termine principale
dell'equazione diventerebbe invisibile alla loss. Vale la pena provarlo, e il
modo in cui fallisce merita una riga: chiedendo a autograd la derivata seconda
di una rete ReLU non si ottiene un errore né un `None`, si ottengono zeri, e
insieme a essi è nullo anche il gradiente di $(u'')^2$ rispetto a *tutti* i
pesi. Il termine c'è, costa il suo tempo di calcolo e non muove nulla: un
guasto perfettamente silenzioso. La `tanh`, al contrario, è liscia (derivabile
infinite volte, con derivate continue a ogni ordine) e infatti è la scelta
standard delle PINN; funzionano anche il seno e la softplus (una versione
arrotondata della ReLU), perché il requisito, qui, è la regolarità. Il che non
vuol dire che siano intercambiabili: le attivazioni periodiche cambiano quali
frequenze la rete impara in fretta, ed è un effetto di cui la prossima sezione
si serve come rimedio.

`````

## La PINN, riga per riga

Ecco il codice, completo ed eseguibile. È forse il più "scientifico" del
libro, quindi ce lo guadagniamo a pezzi: prima la preparazione, poi il cuore
(le derivate rispetto all'input) infine il confronto con la soluzione esatta.

```python
import numpy as np
import torch
from torch import nn

torch.manual_seed(42)

# Parametri fisici della molla: massa, smorzamento, rigidezza
m, c, k = 1.0, 0.4, 4.0

# La candidata soluzione: un MLP che da t produce u(t)
rete = nn.Sequential(
    nn.Linear(1, 32), nn.Tanh(),
    nn.Linear(32, 32), nn.Tanh(),
    nn.Linear(32, 32), nn.Tanh(),
    nn.Linear(32, 1),
)

# Punti di collocazione: 200 istanti a caso in [0, 10]
t_c = 10.0 * torch.rand(200, 1)     # shape (200, 1)
t_c.requires_grad_(True)            # derivate RISPETTO ALL'INPUT

# L'istante iniziale, dove imporremo u(0)=1 e u'(0)=0
t_0 = torch.zeros(1, 1, requires_grad=True)

ottimizzatore = torch.optim.Adam(rete.parameters(), lr=1e-3)
```

Due righe meritano una sosta. `t_c.requires_grad_(True)` accende il
registratore di autograd **sull'input**, non su un peso: è l'inversione di
prospettiva da cui siamo partiti. E la rete è minuscola, tre strati nascosti
da 32 neuroni, perché la funzione da rappresentare è una curva liscia in una
dimensione, non ImageNet. È la pila di strati densi che dal capitolo sulle
reti neurali chiamiamo **MLP** (*multi-layer perceptron*, percettrone
multistrato), la più semplice delle architetture di questo libro. E `shape`,
nei commenti, è la forma della tabella di numeri: qui 200 righe per una
colonna.

Il cuore del metodo sono due chiamate a `torch.autograd.grad`, con due
argomenti che non avevamo mai usato. Il ciclo che le contiene ripete
trentamila volte lo stesso giro di correzione, e ciascuno di quei giri si
chiama **epoca**:

```python
for epoca in range(30_000):
    ottimizzatore.zero_grad()

    # 1) fisica: residuo m*u'' + c*u' + k*u sui punti di collocazione
    u = rete(t_c)                                        # shape (200, 1)
    u_t = torch.autograd.grad(u, t_c, torch.ones_like(u),
                              create_graph=True)[0]      # u'(t)
    u_tt = torch.autograd.grad(u_t, t_c, torch.ones_like(u_t),
                               create_graph=True)[0]     # u''(t)
    residuo = m * u_tt + c * u_t + k * u
    loss_fisica = (residuo ** 2).mean()

    # 2) condizioni iniziali: u(0) = 1 e u'(0) = 0
    u_0 = rete(t_0)
    u_t0 = torch.autograd.grad(u_0, t_0, torch.ones_like(u_0),
                               create_graph=True)[0]
    loss_iniziale = (u_0 - 1.0).pow(2).mean() + u_t0.pow(2).mean()

    # 3) loss totale, con piu' peso all'unico ancoraggio che abbiamo
    loss = loss_fisica + 100.0 * loss_iniziale
    loss.backward()
    ottimizzatore.step()

    if epoca % 5_000 == 0:
        print(f"epoca {epoca:6d} | loss {loss.item():.2e}")
```

Il primo argomento nuovo è `torch.ones_like(u)`: `u` è una colonna di 200
valori, uno per punto di collocazione, e il vettore di uni dice ad autograd
«dammi la derivata di ciascuno», tutte le 200 in un colpo solo[^vjp]. Poiché
ogni $u_j$ dipende soltanto dal suo $t_j$, non c'è alcuna
mescolanza: nella colonna `u_t` la riga $j$ è esattamente $u_\theta'(t_j)$.

[^vjp]: Per la precisione, quello che autograd calcola nativamente è un
    prodotto vettore–jacobiana, $\mathbf{J}^\top \mathbf{v}$, dove
    $\mathbf{J}$ è la tabella di tutte le derivate di tutte le uscite rispetto
    a tutti gli ingressi (e la trasposta serve perché il risultato esce con la
    forma dell'ingresso, non con quella dell'uscita): qui $\mathbf{v}$ è il
    vettore di uni e $\mathbf{J}$ è diagonale, perché ogni uscita dipende da
    un solo ingresso, quindi il prodotto restituisce esattamente la colonna
    delle derivate.

Il secondo è `create_graph=True`, e senza non funzionerebbe niente: chiede ad
autograd di *registrare anche il calcolo della derivata*, così che la derivata
resti a sua volta derivabile. Ci serve due volte. Primo, per derivare di
nuovo: `u_tt` è la derivata di `u_t`, quindi il grafo di `u_t` deve esistere.
Secondo, più sottile: `u_t` e `u_tt` finiscono *dentro la loss*, e quando
chiamiamo `loss.backward()` il gradiente deve poter attraversare anche il
calcolo delle derivate per arrivare fino ai pesi. È una derivata di una
derivata (il registratore che registra sé stesso) ed è il motivo per cui ogni
epoca di una PINN costa più di un'epoca di regressione ordinaria.

Notare infine quel $100$ che moltiplica la `loss_iniziale`: è la cosa meno
innocente del programma. Il termine di fisica, da solo, non ha una risposta
sola: qualunque moto di *quella* molla lo soddisfa, compresa una curva piatta
ferma sullo zero per sempre (un corpo fermo al centro, senza nessuno che lo
sposti, resta fermo, e la regola dice proprio questo). A distinguere la nostra
traiettoria da tutte le altre ci sono soltanto le due condizioni di partenza,
che però da sole tirano poco, perché riguardano un istante mentre l'altro
termine tira sull'intera curva. Moltiplicarle per 100 serve a dare loro voce.
Un moltiplicatore messo lì per bilanciare due termini di una loss si chiama
**peso**, come i pesi della rete e come il corpo appeso alla molla: la parola
è la stessa e le tre cose non c'entrano niente l'una con l'altra, quindi
conviene tenerle separate a mente.

Che la faccenda sia seria si tocca con mano abbassando quel moltiplicatore a
1. In una prova fatta così, e senza cambiare nient'altro, l'addestramento è
arrivato a un residuo di $2 \cdot 10^{-5}$ sui suoi duecento punti, cioè due
centomillesimi; con il moltiplicatore a 100, sulla stessa misura e sullo
stesso seme, il residuo si ferma a $8 \cdot 10^{-3}$, otto millesimi,
**quattrocento volte più alto**. Verrebbe da dire che con 1 è andata meglio.

Invece è andata molto peggio, e si vede solo andando a confrontare la curva
con la risposta vera, che qui per fortuna conosciamo. Fra le due si apre uno
scarto di $0{,}73$: cioè, nel punto in cui va peggio, la curva sbaglia di
quasi tutto lo spostamento da cui il corpo era partito. Con il moltiplicatore
a 100 lo scarto è $0{,}15$, come vedremo fra poche righe: quasi cinque volte
meno.

Il tranello però non sta dove verrebbe da cercarlo. La partenza quella rete la
rispetta lo stesso, $u_\theta(0) = 0{,}999$; quello che ha fatto è azzerare il
residuo **dove veniva controllata** e lasciarlo correre altrove. Se glielo si
va a misurare in cinquecento istanti che non aveva mai visto, stesi fitti e in
fila lungo tutto l'intervallo (una **griglia**, come quella dei metodi
classici, ma qui usata solo per controllare, non per calcolare), il suo
residuo vale $2{,}8$: più di centomila volte quello dei suoi duecento punti.
Il $100$, insomma, non serve a tenere la curva
attaccata alla partenza, perché lì ci resta comunque: serve a rendere meno
conveniente quella scorciatoia. E la rende meno conveniente, **non la vieta**:
fra poche pagine ne vedremo la prova.

Trentamila epoche dopo, ecco il verdetto. A guidare l'addestramento è Adam
{cite}`kingma2015adam`, l'ottimizzatore che dal capitolo sul deep learning è
la nostra scelta di partenza, e a fare da pagella è la formula esatta della
molla, quella ricavata poco fa, calcolata qui in NumPy:

```python
# La soluzione analitica, per dare i voti alla rete
gamma = c / (2 * m)                        # 0.2
omega_d = np.sqrt(k / m - gamma ** 2)      # sqrt(3.96) ~ 1.98997

t_test = np.linspace(0.0, 10.0, 500)
u_esatta = np.exp(-gamma * t_test) * (
    np.cos(omega_d * t_test) + (gamma / omega_d) * np.sin(omega_d * t_test)
)

def diagnosi(rete, t_controllo):
    """Tre misure che conviene tenere separate: il residuo DOVE la rete e'
    stata controllata, il residuo su una griglia fitta che non ha mai visto,
    e l'errore vero contro la formula esatta."""
    def residuo_su(t):
        u = rete(t)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u),
                                  create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t))[0]
        return ((m * u_tt + c * u_t + k * u) ** 2).mean().item()

    t_griglia = torch.tensor(t_test, dtype=torch.float32).reshape(-1, 1)
    t_griglia.requires_grad_(True)
    with torch.no_grad():
        errore = np.abs(rete(t_griglia).squeeze().numpy() - u_esatta)
    return residuo_su(t_controllo), residuo_su(t_griglia), errore


res_punti, res_griglia, errore = diagnosi(rete, t_c)
print(f"residuo sui 200 punti di collocazione: {res_punti:.2e}")
print(f"residuo su una griglia fitta         : {res_griglia:.2e}")
print(f"errore massimo                       : {errore.max():.3f}")
print(f"  sui primi 5 secondi                : {errore[t_test <= 5.0].max():.3f}")
print(f"  sugli ultimi 5 secondi             : {errore[t_test > 5.0].max():.3f}")

# Il numero che il testo commenta e' una promessa: tanto vale verificarla qui.
assert errore.max() < 0.45, (
    f"errore massimo {errore.max():.3f}: la rete non sta ricostruendo "
    "l'oscillazione, e' collassata sulla soluzione banale. Succede: si veda "
    "il seguito della sezione."
)
```

Lanciando il programma, su CPU e in qualche minuto, escono queste righe:

```text
epoca      0 | loss 1.03e+02
epoca   5000 | loss 7.31e-02
epoca  10000 | loss 4.80e-02
epoca  15000 | loss 3.78e-02
epoca  20000 | loss 3.91e-02
epoca  25000 | loss 1.35e-02
residuo sui 200 punti di collocazione: 7.77e-03
residuo su una griglia fitta         : 3.12e-02
errore massimo                       : 0.154
  sui primi 5 secondi                : 0.070
  sugli ultimi 5 secondi             : 0.154
```

Prima di leggerle, una nota sul modo in cui il computer scrive i numeri
piccoli, perché tornerà per tutta la sezione. Intanto la virgola: il computer
la scrive con un punto, quindi `7.77` va letto «sette virgola settantasette».
Poi la `e`, che sta per «per dieci alla»: il numero che la segue dice di
quanti posti spostare la virgola, a sinistra se ha il meno davanti, a destra
se ha il più. Così `7.77e-03` vuol dire $7{,}77 \cdot 10^{-3}$, cioè 0,00777,
sette millesimi scarsi; e `1.23e+03` vuol dire 1230.

Le stampe si fermano a 25 000 perché arrivano ogni cinquemila epoche e
l'ultima cade lì; l'addestramento prosegue fino a 30 000, e le cinque righe
in fondo sono misurate alla fine.

La loss parte dall'ordine del centinaio e scende di circa quattro ordini di
grandezza, cioè si divide per diecimila. Quel centinaio, però, è quasi tutto
il termine sulla partenza: all'inizio la rete parte da un punto qualsiasi
invece che da 1, e quello sbaglio, moltiplicato per 100, fa da solo il grosso
del numero. La fisica, all'inizio, contribuisce appena 0,26. Conviene poi
leggere le cinque righe finali con attenzione, perché dicono due cose diverse,
ed è raro che un esempio da manuale sia così onesto.

La prima è che il metodo funziona. La rete non ha mai visto un solo valore
della soluzione, solo la partenza e la legge, e ne esce una curva che oscilla
con il periodo giusto e si smorza con il ritmo giusto. Sui primi cinque
secondi lo scarto dalla formula esatta non arriva a otto centesimi
dell'ampiezza iniziale. Per una curva ricostruita da una regola e da due
numeri, è molto.

La seconda è che quella curva **non è accurata quanto il residuo lascerebbe
credere**. Sui 200 punti in cui la regola è stata controllata il residuo vale
$8 \cdot 10^{-3}$, cioè la molla risulta obbedita quasi alla lettera; ma lo
scarto massimo dalla soluzione vera è $0{,}154$, il 15% dello spostamento di
partenza, e le due curve messe una sull'altra si distinguono benissimo.

E c'è un modo più severo di leggere quel 15%, che il testo non deve
nascondere. Quello scarto non è sparso: sta quasi tutto nella **coda**, dopo
il quinto secondo, dove la rete comincia ad appiattirsi mentre la molla vera
sta ancora oscillando. Ma nella coda l'oscillazione vera si è ormai ridotta
parecchio, e al decimo secondo è scesa al 13,5% dello spostamento di partenza.
Uno scarto di 0,154 in un tratto dove la molla vera si muove ormai di così
poco vuol dire che lì, di fatto, la rete l'oscillazione non la sta più
seguendo. Sui primi cinque secondi è brava; nella coda ha smesso.

Si noti infine il terzo numero, quello che quasi nessun articolo su questi
metodi si prende la briga di stampare: sulla griglia fitta di istanti che la
rete non ha mai visto il residuo è $3 \cdot 10^{-2}$, quattro volte più alto
che nei punti controllati. La rete va un po’ meglio dove la si guarda che dove
non la si guarda. Qui è uno scarto modesto, e fra poche righe vedremo quanto
può diventare grande. Un ultimo dettaglio da non lasciarsi sfuggire nel crollo
della loss: quasi tutta quella caduta è il termine sulle condizioni iniziali,
che si esaurisce entro le prime mille epoche; il termine di fisica, quello che
dovrebbe fare il lavoro, in tutto si divide soltanto per una trentina, da 0,26
a 0,0078.

```{figure} ../figures/pinn-residuo.svg
:name: fig-pinn-residuo
:alt: "Due pannelli sovrapposti, con il tempo in ascissa. In alto la curva della rete, che nel corso dell'addestramento passa da quasi piatta a sovrapposta all'oscillazione smorzata della soluzione esatta, con lo scarto massimo che cala da 1,009 a 0,154. In basso le barre del residuo in sedici dei duecento punti di collocazione: all'inizio crescono da sinistra a destra e dalla metà in poi superano il tratteggio orizzontale che segna la loro media; alla fine sono briciole, e la media dei quadrati è passata da 2,6 per dieci alla meno uno a 7,8 per dieci alla meno tre."
:width: 100%

Lo stesso addestramento guardato dai due lati che contano: mentre la curva
della rete si accosta alla soluzione esatta (in alto), il residuo nei punti di
collocazione si abbassa fino a diventare una briciola (in basso). Le
istantanee sono epoche vere dell'addestramento di questa pagina, con il
seme 42.
```

La {numref}`fig-pinn-residuo` mette le due misure una sopra l'altra e le fa
correre dall'inizializzazione alla trentamillesima epoca. Il numero in basso a
destra è il termine di fisica, e lì si vede la trentina di poco fa: parte da
$2{,}6 \cdot 10^{-1}$, ventisei centesimi, e arriva a $7{,}8 \cdot 10^{-3}$,
otto millesimi scarsi. Nel pannello di sopra si vede invece dove resta lo
scarto: le due curve stanno appiccicate per metà intervallo e si staccano
nella coda.

Quello scarto fra i due numeri, il residuo piccolo e l'errore grande, non è un
dettaglio di rifinitura. Vale la pena vedere fin dove arriva.

## Lo stesso codice, un altro seme

C'è una riga del programma che non abbiamo commentato: `torch.manual_seed(42)`,
in cima al primo blocco. Una rete comincia sempre con i pesi sorteggiati a
caso, perché partendo tutti uguali i neuroni resterebbero uguali per sempre, e
il sorteggio dipende da un numero di partenza che si chiama **seme**: dando lo
stesso seme si ottiene lo stesso sorteggio, e quindi lo stesso risultato.
Fissarlo, per tutto il resto del libro, è una cortesia al lettore, che così
rifacendo il conto ritrova i nostri numeri. Qui è molto di più: cambiando quel
42, cambia la conclusione.

Rimettiamo l'addestramento di prima dentro una funzione, così da poterlo
rilanciare cambiando soltanto quel numero. È lo stesso codice riga per riga,
con in più la stampa periodica delle due misure che ci interessano.

```python
def addestra(seme, epoche=30_000):
    """Come l'addestramento di sopra: cambia solo il punto di partenza."""
    torch.manual_seed(seme)
    rete = nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )
    t_c = 10.0 * torch.rand(200, 1)
    t_c.requires_grad_(True)
    t_0 = torch.zeros(1, 1, requires_grad=True)
    ottimizzatore = torch.optim.Adam(rete.parameters(), lr=1e-3)

    for epoca in range(epoche):
        ottimizzatore.zero_grad()
        u = rete(t_c)
        u_t = torch.autograd.grad(u, t_c, torch.ones_like(u),
                                  create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_c, torch.ones_like(u_t),
                                   create_graph=True)[0]
        loss_fisica = ((m * u_tt + c * u_t + k * u) ** 2).mean()

        u_0 = rete(t_0)
        u_t0 = torch.autograd.grad(u_0, t_0, torch.ones_like(u_0),
                                   create_graph=True)[0]
        loss_iniziale = (u_0 - 1.0).pow(2).mean() + u_t0.pow(2).mean()

        (loss_fisica + 100.0 * loss_iniziale).backward()
        ottimizzatore.step()

        if epoca % 2_500 == 0:      # residuo e errore vero, fianco a fianco
            errore_ora = diagnosi(rete, t_c)[2].max()
            print(f"epoca {epoca:6d} | residuo {loss_fisica.item():.2e}"
                  f" | errore vero {errore_ora:.3f}")

    return rete, t_c


rete_7, t_c7 = addestra(seme=7)
res_punti_7, res_griglia_7, errore_7 = diagnosi(rete_7, t_c7)

print(f"\n{'':<24}{'seme 42':>10}{'seme 7':>12}")
print(f"{'residuo sui suoi punti':<24}{res_punti:>10.2e}{res_punti_7:>12.2e}")
print(f"{'residuo sulla griglia':<24}{res_griglia:>10.2e}{res_griglia_7:>12.2e}")
print(f"{'errore vero':<24}{errore.max():>10.3f}{errore_7.max():>12.3f}")

# La lezione della pagina, resa verificabile. Tre affermazioni distinte:
assert res_punti_7 < res_punti, (
    "il seme 7 non ha piu' il residuo piu' basso dei due: la tabella qui "
    "sotto va rifatta con i numeri di questa esecuzione."
)
assert errore_7.max() > errore.max(), (
    "il seme 7 non collassa piu' su questa versione di PyTorch: serve un "
    "altro seme che collassi (se ne trovano provando)."
)
assert res_griglia_7 > res_griglia, (
    "il residuo fuori dai punti di collocazione non e' piu' quello alto: "
    "il paragrafo sui due residui va rifatto."
)
```

Ecco che cosa stampa la corsa con il **seme 7**, sulla macchina su cui è stato
scritto questo capitolo. Nella colonna «errore vero» c'è la distanza massima
fra la curva della rete e la formula esatta, sempre in frazioni dello
spostamento di partenza, che vale 1: 0,879 vuol dire che in qualche istante la
rete sbaglia di quasi tutto lo spostamento da cui il corpo era partito.

| epoca | residuo | errore vero |
|---:|---:|---:|
| 0 | $1{,}44 \cdot 10^{-1}$ | 0,879 |
| 2 500 | $6{,}10 \cdot 10^{-2}$ | 0,497 |
| 5 000 | $5{,}62 \cdot 10^{-2}$ | 0,467 |
| 7 500 | $5{,}41 \cdot 10^{-2}$ | 0,453 |
| 10 000 | $5{,}22 \cdot 10^{-2}$ | 0,444 |
| 12 500 | $4{,}74 \cdot 10^{-2}$ | 0,440 |
| 15 000 | $3{,}67 \cdot 10^{-2}$ | 0,431 |
| 17 500 | $3{,}35 \cdot 10^{-2}$ | 0,432 |
| 20 000 | $8{,}57 \cdot 10^{-3}$ | 0,499 |
| 22 500 | $3{,}24 \cdot 10^{-4}$ | 0,709 |
| 25 000 | $9{,}03 \cdot 10^{-5}$ | 0,720 |
| 27 500 | $5{,}46 \cdot 10^{-5}$ | 0,723 |

Si guardi la seconda metà della tabella, perché è il punto di tutta la
sezione. Fra le 17 500 e le 27 500 epoche il residuo si divide per seicento,
da $3{,}3 \cdot 10^{-2}$ a $5 \cdot 10^{-5}$: chiunque guardasse soltanto la
loss direbbe che proprio lì l'addestramento ha fatto un salto di qualità.
Nello stesso tratto l'errore vero **peggiora**, da 0,43 a 0,72, cioè da mezzo
spostamento di partenza a quasi tutto. Le due colonne, che dovrebbero
raccontare la stessa storia, vanno in direzioni opposte.

Che cosa sia successo si vede disegnando la curva che ne esce. La partenza la
rispetta in pieno ($u(0) = 0{,}9993$, praticamente perfetta); poi scende, e
scendendo sprofonda una volta sola fin sotto lo zero; e da lì in avanti si
appiattisce e non si muove più. Dal secondo 2 in poi, per otto secondi buoni,
resta a meno di $0{,}013$ dallo zero, mentre la molla vera in quel tratto
compie ancora due oscillazioni e mezza. Lo zero lo attraversa una volta sola,
contro le sei della soluzione vera, una a ogni mezza oscillazione: contare gli
attraversamenti è un modo rapido di vedere se una curva sta ancora oscillando
o se ha rinunciato. Questa ha rinunciato: si è arresa alla riga dritta sullo
zero, quella che rispetta la regola senza dire niente, e che d'ora in poi
chiameremo la **soluzione banale**.

Ma il confronto stampato alla fine dice qualcosa di più, ed è la ragione per
cui vale la pena misurare il residuo in due posti invece che in uno:

```text
                           seme 42      seme 7
residuo sui suoi punti    7.77e-03    4.77e-04
residuo sulla griglia     3.12e-02    1.23e+03
errore vero                  0.154       0.720
```

Sui duecento istanti in cui è stata controllata, la rete del seme 7 prende un
punteggio di fisica **sedici volte più basso** di quella del seme 42. Sulla
griglia fitta di istanti che non ha mai visto, il suo punteggio è
**quarantamila volte più alto**. E fra i suoi due punteggi, quello dove è
stata guardata e quello dove non lo è stata, corre un fattore di due milioni e
mezzo: stessa rete, stesso momento, due giudizi opposti. (Questi sono rapporti
fra *punteggi*, e un punteggio è una media di quadrati: per risalire a quanto
la regola è violata davvero bisogna prendere la radice. Due milioni e mezzo ha
per radice circa milleseicento, il che vuol dire che la violazione vera è
milleseicento volte più grossa. Resta un abisso.)

Una nota, per chi ha confrontato i numeri: nella tabella a 27 500 epoche il
residuo del seme 7 era $5{,}5 \cdot 10^{-5}$, qui alla fine è
$4{,}8 \cdot 10^{-4}$, quasi nove volte più alto. Non è un refuso: una discesa
del gradiente non scende sempre, e negli ultimi duemilacinquecento giri quel
termine è risalito. Non cambia niente di quello che segue.

Non è dunque soltanto che il residuo basso non garantisce la soluzione giusta:
è che quel residuo basso è stato ottenuto **proprio e soltanto nei punti che
si stanno guardando**. La rete ha imparato a essere impeccabile all'esame e
sregolata fuori.

`````{tab} Elementare

È la scorciatoia di cui parlavamo prima, quella che il moltiplicatore 100
doveva rendere sconveniente. Uno studente che consegna un foglio con una riga
dritta sullo zero non sta violando la regola della molla: un corpo fermo al
centro, senza nessuno che lo sposti, resta fermo, e la regola dice esattamente
questo. Il professore, che la soluzione non la conosce e sa solo verificare la
regola, non ha nulla da eccepire. L'unica cosa che distingue quel foglio da
quello giusto è la partenza, ed è un punto solo contro una curva intera.

E c'è il trucco in più, quello che spiega i due punteggi così diversi.
Ricordi il «dappertutto» di qualche pagina fa, quello su cui avevamo chiesto
di tenere un dito? Ecco il conto. Il professore non controlla dappertutto:
controlla duecento istanti, sempre gli stessi. Lo studente lo ha capito, e ha
imparato a stare in riga *esattamente lì*.

E adesso guardiamo che cosa fa **in mezzo**, perché c'è da restare a bocca
aperta. Due dei suoi punti di controllo cadono a 1,205 e a 1,415 secondi: fra
loro corrono due decimi di secondo, un buco quattro volte più largo del
solito, perché quei punti sono stati sorteggiati e il caso li ha lasciati
radi lì. In quel buco la curva sprofonda fino a $-0{,}57$ e risale, tutto
dentro quei due decimi di secondo. Non è un sussulto impercettibile: è un
tuffo profondo più di mezzo foglio, che comincia e finisce fra due controlli.
Proprio perché è così stretto, lì la curva **piega** in modo mostruoso, e la
regola della molla parla soprattutto di quanto la curva piega. Se il
professore ci mettesse il dito, quel compito verrebbe stracciato.

Il professore lì non ci mette il dito. Sul suo registro il compito è quasi
perfetto; il disegno è sbagliato due volte, perché è piatto dove dovrebbe
oscillare e perché fa un tuffo dove nessuno guarda. Non è furbizia di
nessuno: è che l'unico modo che ha di prendere voti è stare in riga dove si
guarda, e nulla in quel punteggio gli chiede di comportarsi anche altrove.

`````

`````{tab} Superiore

Due meccanismi distinti si sommano qui, e conviene separarli.

Il primo è la **degenerazione del termine di fisica**, già annotata nella
scheda di apertura: $u \equiv 0$ risolve esattamente l'equazione omogenea,
quindi il minimo del solo residuo è degenere e la soluzione banale ne fa
parte. Il termine sulle condizioni iniziali dovrebbe selezionare la nostra fra
le infinite soluzioni, ma agisce su un singolo istante, e $\lambda_0 = 100$
**rende quella scorciatoia meno attraente, non la vieta**: la rete infatti la
paga per intero, $u_\theta(0) = 0{,}9993$, e si tiene il residuo quasi nullo
nei punti in cui viene interrogata. È il fenomeno che la prossima sezione
chiamerà mancanza di **ordine causale**: la loss somma residui su punti sparsi
nel dominio e nulla obbliga la rete a propagare in avanti nel tempo
l'informazione della partenza.

Il secondo è il **campionamento finito**, ed è il posto in cui si tocca con
mano l'avvertenza della scheda d'apertura: fra un punto di collocazione e il
successivo la regolarità della rete non le impedisce affatto di oscillare.
Qui infatti oscilla: misurato su una griglia da
200 000 istanti, il residuo di questa rete tocca un massimo di $6{,}6 \cdot
10^2$ a $t = 1{,}263$, con un picco largo $0{,}02$ secondi a metà altezza. I
due punti di collocazione che se lo trovano in mezzo stanno a $1{,}205$ e a
$1{,}415$: fra loro corrono $0{,}21$ secondi, quattro volte il passo medio del
campione. **Il picco sta nel buco fra due punti di controllo, ed è largo un
decimo di quel buco.** E non è un'increspatura invisibile nella soluzione: su
quello stesso intervallo $u_\theta$ passa da $-0{,}006$ a $-0{,}565$ e
ritorna, un'escursione di $0{,}56$ interamente contenuta fra due istanti in
cui la rete non viene interrogata. Fuori di lì la rete è davvero piatta: per
$t > 2$ resta entro $1{,}2 \cdot 10^{-2}$ dallo zero. Una rete `tanh` con tre
strati da 32 neuroni ha abbastanza capacità per infilare una guglia dove non
viene interrogata, e duecento punti su dieci secondi non bastano a
impedirglielo. La regolarità
della rete è un argomento asintotico, non una garanzia a $N_c$ finito: dice
che infittendo i punti la cosa si chiude, non che sia già chiusa.

`````

Il seme 7 non è nemmeno un caso isolato. Rilanciamo lo stesso programma sei
volte, cambiando ogni volta soltanto il seme. Di ciascuna corsa prendiamo il
residuo sui suoi duecento punti, quello della prima riga del confronto qui
sopra, e mettiamo le sei in fila **dal punteggio migliore al peggiore**:

| seme | residuo finale | errore massimo | attraversamenti dello zero |
|---:|---:|---:|---:|
| 7 | $4{,}8 \cdot 10^{-4}$ | 0,720 | 1 |
| 3 | $1{,}7 \cdot 10^{-3}$ | 0,629 | 4 |
| 42 | $7{,}8 \cdot 10^{-3}$ | **0,154** | 6 |
| 1 | $1{,}7 \cdot 10^{-2}$ | 0,212 | 5 |
| 0 | $2{,}7 \cdot 10^{-2}$ | 0,259 | 5 |
| 2 | $3{,}1 \cdot 10^{-2}$ | 0,289 | 5 |

Le due corse con il residuo **più basso in assoluto** sono le due sbagliate, e
sono anche quelle che restano più lontane dai sei attraversamenti dello zero
della soluzione vera: uno e quattro, contro i cinque o sei delle altre. Solo
il seme 42 li fa tutti. Fra le altre quattro il residuo torna a essere una
guida sensata (chi ce
l'ha più basso sbaglia meno), il che rende la trappola ancora più insidiosa:
la loss è informativa finché la rete sta risolvendo il problema giusto, e
smette di esserlo esattamente quando serve, cioè quando ha smesso di
risolverlo. Due corse su sei, su un'equazione che si risolve a mano in mezza
pagina.

Due conseguenze pratiche, e sono le più utili di tutta la sezione. La prima:
in una PINN **la loss non è una pagella**. In un problema di apprendimento
ordinario un punteggio che scende è una buona notizia; qui il residuo può
scendere allontanandosi dalla risposta, e nella loss che si sta minimizzando
non c'è niente che lo segnali. Confrontare con la soluzione vera, come abbiamo
fatto qui, nei casi veri non si può: se quella soluzione ce l'avessimo, non
staremmo usando una PINN. La seconda: un risultato ottenuto con un solo seme
non è un risultato. Il modo minimo di lavorare seriamente con questi metodi è
rilanciare con qualche seme diverso e guardare quanto le risposte si somigliano
fra loro, che è l'unica cosa che si può fare quando la risposta giusta non si
conosce.

E una terza, che costa pochissimo e che i due punteggi affiancati suggeriscono
da sé: **il residuo va sempre misurato anche dove la rete non è stata
addestrata**, su una griglia fitta o su punti estratti di nuovo. È lo stesso
motivo per cui, nei capitoli sull'apprendimento, un modello si giudica su dati
tenuti da parte e mai visti in addestramento: un punteggio calcolato dove il
modello si è allenato misura anche quanto bene ha imparato a compiacere quel
campione. C'è anche un rimedio, ed è il più diffuso: **ricampionare** i punti
di collocazione a ogni epoca, cioè sorteggiarne di nuovi ogni volta, così che
non esista un esame su cui prepararsi. Il programma di questa pagina non lo fa
apposta, perché è proprio tenendo fermi i duecento punti che la scorciatoia si
vede a occhio nudo.

Niente di tutto questo smentisce il metodo, e non è il caso di esagerare in
senso opposto: quattro corse su sei ricostruiscono un'oscillazione smorzata
riconoscibile, partendo da un punto e da una regola, il che resta notevole. Ma
«di solito funziona» non è una garanzia, e su un problema da manuale, con
soluzione nota, tre parametri e una sola variabile, a separare la corsa buona
da quella fallita è stato il seme del generatore casuale. È il motivo per cui
la prossima sezione mette in fila i limiti prima delle applicazioni, e non
dopo.

## Non unire i puntini, non calcolarli: una terza via

Fermiamoci a guardare che cosa è successo, perché è facile passarci sopra.

`````{tab} Elementare

Confrontiamo con i due mestieri che già conosciamo. La **regressione** dei
capitoli sul machine learning è unire i puntini: senza puntini non parte
nemmeno, e per disegnare questa curva le sarebbero servite decine di misure
sparse su tutti i 10 secondi. La nostra rete ha ricevuto **zero misure**: un
punto di partenza, una regola, fine; la fisica ha fatto il lavoro dei dati. Il
**solutore classico** visto in apertura di capitolo, il conto a passettini del
caffè, la curva la sa calcolare; ma la calcola su una griglia di istanti, e i
valori in mezzo li ricostruisce interpolando fra quelli che ha in mano (i
solutori maturi lo fanno bene, non tirando una retta fra due puntini). La PINN
invece restituisce una **funzione**: chiedile il valore a $3{,}7$ secondi, o
in qualunque altro punto, e risponde, perché la soluzione ormai abita dentro
la rete.

Onestà d'obbligo, perché qui è facile vendere fumo. Su questo problemino il
conto a passettini vince su tutta la linea: qualche centesimo di secondo
contro i minuti dell'addestramento, e uno scarto dalla formula esatta di
$5 \cdot 10^{-11}$, cioè cinque centomiliardesimi, contro il nostro
$0{,}15$[^tolleranze]. E se la curva
la vuoi oltre i dieci secondi, a lui basta continuare ad avanzare, mentre la
rete fuori dal tratto su cui è stata addestrata si inventa quello che le pare
e va riaddestrata da capo. Il vantaggio della PINN è altrove, e comincia dove
dati e legge vanno mescolati, come vedremo tra un attimo.

[^tolleranze]: Quei cinque centomiliardesimi non sono un numero di targa: a un
    solutore si dice quanta precisione si vuole, e vale la pena dichiarare
    quanta gliene abbiamo chiesta. Il conto è fatto con `solve_ivp` di SciPy
    stringendo **entrambe** le tolleranze, la relativa (`rtol=1e-10`) e
    l'assoluta (`atol=1e-12`); lasciando l'assoluta al valore di default lo
    scarto sale a $1{,}7 \cdot 10^{-6}$, più di quattro ordini di grandezza
    più largo, e resta comunque quasi centomila volte più piccolo del nostro. La
    tolleranza assoluta è quella che conta nella coda, dove la soluzione è
    ormai piccola: taciuta, il confronto cambia di ordini di grandezza.

`````

`````{tab} Superiore

Rispetto alla **regressione pura**: minimizzare solo il termine dati richiede
$N_d$ grande e non promette nulla tra un campione e l'altro, mentre qui il
residuo vincola $u_\theta$ su tutto il dominio e bastano le condizioni
iniziali (il termine di fisica agisce come una regolarizzazione infinitamente
informata). Rispetto a un **integratore classico** (Eulero, Runge–Kutta):
quello discretizza il tempo con passo $h$, propaga sequenzialmente e offre
garanzie di convergenza con errore $O(h^p)$ (sotto le ipotesi del caso:
stabilità dello schema e soluzione abbastanza regolare); la PINN sostituisce
la propagazione con un'ottimizzazione globale non convessa: nessuna garanzia
formale, costo superiore di ordini di grandezza su un problema standard come
questo, ma soluzione *mesh-free* e continua, valutabile e **derivabile** in
qualunque punto. Su quest'ultimo vantaggio conviene non calcare troppo: un
integratore a passo adattivo offre da decenni l’*output denso*, cioè
un'interpolante dello stesso ordine del metodo, richiamabile in qualunque
istante. Quello che resta davvero alla rete è la derivabilità, e il fatto che
la stessa impalcatura non cambia passando a più variabili o a un problema
inverso.

Quanto alla storia, un'onestà dovuta: l'idea non nasce nel 2019, e nemmeno
nel 1998. La loss delle PINN, esattamente com'è scritta qui (un MLP che
approssima la soluzione, il residuo minimizzato ai punti di collocazione e le
condizioni imposte **come penalità nella stessa funzione obiettivo**), è di
Dissanayake e Phan-Thien nel **1994** {cite}`dissanayake1994neural`, e reti
neurali messe a risolvere equazioni differenziali compaiono già in Lee e Kang
nel 1990, con un impianto però diverso: lì la rete non rappresenta la
soluzione come funzione delle coordinate, minimizza l'errore di uno schema
alle differenze finite già discretizzato {cite}`lee1990neural`. Isaac Lagaris,
Aristidis Likas e Dimitrios
Fotiadis, nel 1998, pubblicano la variante che di solito viene citata come
capostipite {cite}`lagaris1998artificial`, ed è utile distinguerla perché non
è la stessa cosa: Lagaris costruisce la soluzione di prova in modo che
condizioni iniziali e al contorno siano soddisfatte *esattamente*, per
costruzione, e resta da minimizzare il solo residuo. Sul nostro problema
basterebbe cercare la soluzione nella forma
$\hat{u}(t) = 1 + t^2\,u_\theta(t)$, che dà $\hat{u}(0)=1$ e
$\hat{u}'(0) = 2t\,u_\theta + t^2 u_\theta' \big|_{t=0} = 0$ qualunque cosa
faccia la rete: niente $\lambda_0$ da scegliere, e la soluzione banale non è
più raggiungibile. È il vincolo imposto
*a priori*; la PINN, come la formulazione del 1994, lo impone invece come
penalità, una scelta che tiene il metodo generale (una forma così va
riscritta per ogni geometria e ogni tipo di condizione) ma che, come abbiamo
appena visto con il seme sfortunato e come vedremo nella prossima sezione, ha
un costo. Ma nel 1994 le derivate della rete andavano ricavate con
formule scritte a mano, caso per caso, e l'ottimizzazione girava su CPU
dell'epoca: l'idea restò di nicchia per vent'anni. Quando Maziar Raissi, Paris
Perdikaris e George Karniadakis la rilanciano nel 2019
{cite}`raissi2019physics` (il nome «physics-informed neural networks» lo
avevano già usato nei due preprint del 2017 da cui quel lavoro nasce), la
differenza non è concettuale ma
infrastrutturale: autograd generale e maturo {cite}`paszke2019pytorch`; le due
chiamate a `torch.autograd.grad` di poco fa, e GPU per l'addestramento. A
volte, nella ricerca, l'idea giusta deve solo aspettare i suoi attrezzi.

`````

## Il problema inverso, in tre righe di codice

Chiudiamo con la variazione promessa in apertura di capitolo: quella che, più
di ogni altra, giustifica l'esistenza delle PINN. Finora abbiamo fatto il
percorso in un verso: legge nota, e da lì la curva. Adesso lo percorriamo
all'incontrario, curva osservata e da lì un pezzo di legge, ed è per questo
che si chiama **problema inverso**.

Ecco la situazione. La molla è dentro una scatola chiusa e la sua rigidezza
$k$ non la sappiamo; in compenso un sensore ci passa 25 misure della
posizione, equispaziate lungo i dieci secondi e sporche, come è giusto che
siano, di un rumore casuale di ampiezza tipica $0{,}05$, cioè un ventesimo
dello spostamento iniziale. (Il sensore non esiste: le 25 misure le abbiamo
fabbricate noi, prendendo la formula esatta con $k = 4$ e aggiungendoci il
rumore. È l'unico modo di sapere, alla fine, se la stima era giusta.) Nel
codice cambia pochissimo, tre cose in tutto:

```{code-block} python
:class: pt-non-eseguibile

# k non lo conosciamo piu': diventa un parametro da apprendere
k_appreso = nn.Parameter(torch.tensor(1.0))   # partenza volutamente sbagliata

ottimizzatore = torch.optim.Adam(
    list(rete.parameters()) + [k_appreso], lr=1e-3
)

# nel ciclo di addestramento: il residuo usa il k appreso...
residuo = m * u_tt + c * u_t + k_appreso * u
# ...e accanto alla fisica c'e' il termine dati sulle misure rumorose
loss_dati = ((rete(t_oss) - u_oss) ** 2).mean()
loss = loss_fisica + 100.0 * loss_dati
```

dove `t_oss` e `u_oss` sono le colonne di numeri con gli istanti e le misure.
E la `loss_iniziale`? Non serve più: l'ancoraggio che prima spettava alle
condizioni di partenza ora lo danno le 25 misure, e il loro termine ne prende
il posto, moltiplicatore compreso.

Il meccanismo non è cambiato di una virgola, ed è questa la cosa notevole.
Fino a un attimo fa l'addestramento girava una manopola per ogni peso della
rete; adesso gliene abbiamo messa davanti una in più, la rigidezza, e lui non
si accorge nemmeno che è diversa dalle altre: la gira insieme a tutte,
misurando come sempre di quanto ciascuna farebbe scendere il punteggio.
Curva e legge si aggiustano nello stesso movimento, finché non vanno
d'accordo con le osservazioni.

Partendo dal valore volutamente sbagliato $k = 1$, la stima si stacca subito,
sale e si assesta vicino al valore vero $k = 4$, quello con cui avevamo
fabbricato le misure: nella prova di questa pagina (seme 42, le 25 misure
sporcate come si è detto, trentamila epoche) si ferma a $3{,}92$, e per
arrivarci ricostruisce l'intera traiettoria. La rigidezza della molla, che
nessuno ha misurato, esce come sottoprodotto dello stesso addestramento. Un
seme diverso la porta a $3{,}75$, che è un buon promemoria di quello che
abbiamo appena finito di dire: una corsa sola non è una misura.

E c'è un dettaglio che vale la pena raccogliere, dopo la brutta figura di
poco fa. Qui la traiettoria ricostruita è **più accurata** di quella che
avevamo ottenuto conoscendo la legge per intero, pur essendo il problema più
difficile dei due: lo scarto massimo dalla curva vera è circa $0{,}07$, e
circa $0{,}08$ rilanciando con il seme 7, quello sfortunato, contro lo
$0{,}15$ di prima. Il
motivo è tutto nella disposizione degli ancoraggi: prima la rete aveva un solo
punto fermo, l'istante zero, e più si andava avanti nel tempo più era libera
di inventare; qui ha venticinque misure sparse su tutto l'intervallo, che la
tengono per mano fino in fondo. Sono rumorose e sono poche, ma sono
*dappertutto*, ed è quello che conta.

Viene però il sospetto legittimo: con venticinque punti stesi su tutta la
curva, non basterebbe unirli? Vale la pena rispondere con un numero invece che
con una frase. Facendo passare una curva morbida per quei venticinque punti, e
basta, lo scarto massimo dalla soluzione vera si ferma attorno a $0{,}12$
(mediana su duemila sorteggi del rumore, con lo scarto tipico $0{,}05$ di
sopra). Non è un caso che quel numero somigli al rumore delle misure: chi si
limita a unire i puntini ne ricopia anche gli errori, e più preciso dei
puntini che ha non può diventare.

La PINN invece si ferma attorno a $0{,}07$, cioè **sotto** l'errore delle
misure che le sono state date. A permetterglielo è la legge: fra tutte le
curve che passano vicino a quei venticinque punti, tiene solo quelle che una
molla potrebbe davvero percorrere, e le altre le scarta, rumore compreso. Non
è un dettaglio di rifinitura: è tutto il capitolo in un numero.

Sembra poco: tre modifiche. È moltissimo: è il medico legale che risale all'ora
del decesso, il geofisico che deduce la struttura del sottosuolo dalle onde
sismiche, l'ingegnere che stima l'usura di un componente dai sensori. È la
famiglia di problemi in cui le PINN danno il meglio, e la prossima sezione è
dedicata a loro; con l'avvertenza, che là svilupperemo, che «il meglio delle
PINN» non vuol dire «meglio di tutti».

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il colpo di scena: lo stesso meccanismo che finora diceva di quanto
  ritoccare ciascun peso sa dire anche **quanto la curva della rete sale o
  scende**, e quanto in fretta cambia quella pendenza, in qualunque istante e
  senza approssimazioni. La rete smette di essere una scatola e diventa una
  curva liscia, alla quale si può chiedere di rispettare una regola.
- Il punteggio da abbassare somma due voci. La prima sono le violazioni della
  regola negli istanti di controllo scelti a caso, i **punti di
  collocazione**; la seconda sono gli scarti sulla partenza, cioè il punto da
  cui si parte e la pendenza con cui si parte (se il problema è esteso anche
  nello spazio, come la sbarra che si scalda, in questa voce entra anche
  quello che succede ai bordi). Alla partenza si dà più peso, perché quando
  altro non c'è è l'unico ancoraggio. Se invece ci sono misure sparse lungo
  tutto l'intervallo, sono loro l'ancoraggio, e prendono il posto della
  partenza (è quello che succede nel problema inverso qui sopra).
- **La curva dev'essere liscia**: se è fatta di segmenti dritti incollati
  uno dopo l'altro, come quelli che escono dalla ReLU, non ha curvatura da
  nessuna parte, e il professore non vedrebbe più il pezzo più importante
  della regola. Per questo qui si torna alla vecchia S centrata nello zero,
  che invece piega dolcemente dappertutto.
- **Il registratore deve annotare anche i propri conti.** Per sapere quanto
  la curva sale in un istante gli si chiede di riavvolgere i calcoli; ma
  quella pendenza serve poi altre due volte, per ricavarne la curvatura e per
  far arrivare la correzione fino ai pesi. Quindi gli si dice, con l'opzione
  `create_graph=True`, di tenere memoria anche del conto della pendenza. È il
  motivo per cui un giro di addestramento di una PINN costa più di uno
  normale.
- Sulla molla con attrito (massa 1, attrito 0,4, rigidezza 4) la rete
  ricostruisce l'oscillazione senza aver mai visto un solo valore della
  soluzione oltre la partenza: un'oscillazione ogni 3,16 secondi e ampiezza
  scesa al 13,5% dopo 10 secondi. Non però con la precisione che la loss
  lascerebbe credere: lo scarto dalla curva vera arriva a 0,154, cioè al 15%
  dello spostamento di partenza, e si concentra tutto nella seconda metà,
  lontano dall'unico ancoraggio: là la rete l'oscillazione, di fatto, ha
  smesso di seguirla.
- **Un punteggio basso non vuol dire risposta giusta**, ed è la lezione da
  portarsi via. Su sei ripartenze dello stesso identico programma, le due che
  ottengono il punteggio migliore sono le due che sbagliano di più. La
  partenza la rispettano; quello che fanno è stare in riga **dove il
  professore guarda** e lasciarsi andare in mezzo, fino a spegnere del tutto
  l'oscillazione nel caso peggiore. Rinunciare a oscillare è un modo di
  rispettare la regola come un altro: con una regola sola e nessuna misura fra
  un controllo e l'altro, «rispettare la regola» non basta a inchiodare la
  risposta.
- Non è unire i puntini (nel problema diretto di puntini non ce n'è nessuno)
  e non è un conto fatto a passettini su una griglia di istanti: quello che
  resta alla fine è una **curva intera**, che risponde a qualunque istante le
  si chieda, anche a 3,7 secondi, anche in mezzo a due punti qualsiasi, senza
  tabelle da interpolare.
- **Problema inverso**: se un pezzo della regola manca (quanto è rigida la
  molla), diventa una manopola in più che l'addestramento gira insieme alla
  curva, bastano poche misure rumorose. È la mossa che rende uniche le PINN.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il colpo di scena tecnico: le stesse derivate automatiche usate finora sui
  pesi, calcolate **rispetto all'input**, rendono la rete $u_\theta$ una
  funzione derivabile su cui si può imporre un'equazione differenziale.
- La loss di una PINN somma: media dei **residui** sui punti di
  collocazione (la fisica) + scarti su **condizioni iniziali/al contorno**
  (+ eventuali dati), con pesi che proteggono i pochi ancoraggi.
- **`tanh`, non ReLU**: la ReLU ha derivata seconda nulla quasi ovunque e
  renderebbe cieco il residuo; servono attivazioni lisce.
- **`create_graph=True`** è la chiave pratica: mantiene derivabile la
  derivata, per poter calcolare $u''$ e per far passare `backward()`
  attraverso il residuo.
- Sull'oscillatore smorzato ($m=1$, $c=0{,}4$, $k=4$) la PINN si avvicina alla
  soluzione analitica $u(t)=e^{-0{,}2t}(\cos\omega_d t +
  0{,}1005\,\sin\omega_d t)$, $\omega_d=\sqrt{3{,}96}$, senza aver visto un
  solo dato oltre le condizioni iniziali; ma con uno scarto massimo di
  $\approx 0{,}15$, non di $10^{-3}$, concentrato nella coda dell'intervallo.
- **Residuo piccolo non implica soluzione corretta.** Su sei semi, le due
  corse con $\mathcal{L}_{\text{fisica}}$ più bassa ($4{,}8\cdot10^{-4}$ e
  $1{,}7\cdot10^{-3}$) sono quelle con l'errore più grande (0,72 e 0,63):
  $u \equiv 0$ annulla il residuo dell'equazione omogenea, e $\lambda_0=100$
  rende quella scorciatoia meno attraente ma non la vieta. Corollario
  operativo: **più semi**, e mai fidarsi della sola loss.
- L'idea è del 1994 {cite}`dissanayake1994neural` (vincolo *soft*, penalità
  nella loss), con antecedenti al 1990 {cite}`lee1990neural`; Lagaris 1998
  {cite}`lagaris1998artificial` è la variante a vincolo *hard*. L'esplosione
  del 2019 {cite}`raissi2019physics` arriva quando autograd e GPU la rendono
  praticabile.
- **Problema inverso**: basta promuovere un coefficiente a `nn.Parameter`
  per stimarlo da poche misure rumorose, insieme alla soluzione. È la mossa
  che rende uniche le PINN.
```

`````
