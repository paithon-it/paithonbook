# La legge dentro la loss

Per tutto il libro, autograd ha fatto un solo mestiere. Nel capitolo su
PyTorch l'abbiamo raccontato come un registratore che annota i calcoli e li
riavvolge all'indietro; da allora, ogni chiamata a `backward()` ha risposto
sempre alla stessa domanda: *se ritocco questo peso, quanto cambia l'errore?*
Derivate della loss **rispetto ai pesi**, milioni di volte, per addestrare
classificatori, traduttori, generatori.

Questa sezione comincia con un colpo di scena: la domanda si può cambiare.
Quel registratore non sa che cosa siano "i pesi", e non gliene importa: sa
rispondere alla domanda «se muovo *questo*, di quanto cambia *quello*» su
qualunque coppia di numeri che compaia nei suoi conti. Possiamo allora
puntarlo altrove.

Prendiamo una rete che riceve un istante di tempo e restituisce un numero, per
esempio la posizione di un oggetto in quell'istante: disegnata su un foglio, è
una curva. Ora chiediamo al registratore, invece del solito «di quanto cambia
l'errore se ritocco questo peso», quest'altro: «di quanto cambia il numero in
uscita se sposto di pochissimo l'istante che ti ho dato?». La risposta è la
**pendenza** della curva in quel punto, cioè quanto in fretta sta salendo o
scendendo proprio lì. E rifacendo la stessa domanda sulla pendenza si ottiene
la **curvatura**, quanto in fretta la pendenza stessa sta cambiando, cioè
quanto la curva piega.

In notazione: data una rete $u_\theta(t)$, otteniamo $u_\theta'(t)$ (la
pendenza) e $u_\theta''(t)$ (la curvatura) in qualunque punto, esatte a meno
della precisione di macchina, cioè con il solo errore che resta a un
calcolatore che lavora con un numero finito di cifre. Non sono stime, e non
sono ricavate confrontando due punti vicini: sono le derivate vere. La rete
smette di essere soltanto una scatola addestrabile: diventa una **funzione
derivabile**, un oggetto matematico a pieno titolo. E a una funzione
derivabile si può chiedere di rispettare un'equazione differenziale, cioè una
regola su come le cose cambiano.

Tutto il metodo delle PINN sta in questa mossa, chiedere le derivate rispetto
all'ingresso invece che ai pesi. Vediamola all'opera.

## Tre vincoli e una penalità

L'idea si capisce meglio raccontandola come un compito in classe.

`````{tab} Elementare

Immagina uno studente alle prese con un compito insolito: disegnare, su un
foglio a quadretti, la curva di una molla che oscilla (la posizione del peso,
istante per istante). Nessuna tabella di valori da copiare. Solo tre vincoli:

1. la curva deve **partire dal punto giusto** (la molla è stata tirata fino
   a una certa altezza);
2. deve partire **con la pendenza giusta** (il peso è stato lasciato andare
   da fermo: velocità zero, quindi la curva comincia in piano);
3. in **ogni punto del foglio** deve rispettare la regola della molla: la
   curvatura in quel punto dev'essere coerente con la posizione e la
   velocità in quello stesso punto («coerente» vuol dire che c'è una formula
   che lega le tre cose, e fra poche pagine la scriveremo con i numeri veri).

Il professore corregge in modo semplice e spietato: controlla la partenza,
poi apre il foglio in una manciata di punti scelti a caso e verifica la
regola; ogni violazione costa punti. Lo studente ritocca la curva e
riconsegna, ancora e ancora, finché i punti persi non si riducono a
briciole. Si noti la stranezza: *nessuno dei due conosce la soluzione*. Il
professore sa solo verificare la regola. Eppure alla fine la curva giusta
salta fuori, perché tra tutte le curve possibili quella vera è l'unica che
parte così *e* rispetta la regola dappertutto. Una PINN è esattamente questo
studente: la curva è la rete, i punti persi sono la loss, e i punti di
controllo aperti a caso si chiamano **punti di collocazione**.

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

La prima è l'**ampiezza dei gradienti**. Il residuo si ottiene applicando alla
rete degli operatori differenziali, e i gradienti che tornano indietro da quel
ramo *possono* avere ampiezze di ordini di grandezza superiori a quelli del
termine sulle condizioni iniziali: succede sulle PDE con operatori di ordine
alto e condizioni campionate su una superficie, ed è lo squilibrio che Wang,
Teng e Perdikaris documentano e correggono con pesi ristimati durante
l'addestramento {cite}`wang2021understanding`. Sul problema di questa sezione,
però, il divario misurato è molto più modesto: alla inizializzazione, cioè nel
momento in cui $\lambda_0$ va scelto, il rapporto fra le ampiezze medie dei
gradienti dei due rami ha mediana $2{,}4$ su venti semi, e in quattro semi su
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

`````

```{figure} ../figures/pinn-schema.svg
:name: fig-pinn-schema
:alt: "Schema del metodo PINN, da sinistra a destra. Le coordinate entrano in una rete neurale, che restituisce la curva candidata. Da lì partono due rami: in alto quello della fisica, dove si calcolano le pendenze della curva e si misura di quanto viola la regola; in basso quello delle misure e delle condizioni di partenza. I due rami si sommano in un punteggio unico, dal quale la correzione torna indietro fino ai pesi della rete."
:width: 100%

L'anatomia di una PINN, da sinistra a destra: le coordinate entrano nella
rete, che risponde con la curva candidata; da lì partono due controlli, quello
della regola fisica (in alto) e quello delle misure e delle condizioni di
partenza (in basso); i due si sommano in un punteggio unico, e la correzione
torna indietro fino ai pesi della rete.
```

In {numref}`fig-pinn-schema` c'è il metodo per intero, e le poche scritte in
formula dicono cose che ormai abbiamo in mano. Il simbolo $\partial$, quella
«d» arrotondata, è il modo di scrivere una pendenza quando le variabili in
gioco sono più di una; la «L» decorata, $\mathcal{L}$, è il punteggio, cioè la
loss; $\theta$, «theta», sta per l'insieme dei pesi della rete; e «autograd»
non è un nome proprio, è l'abbreviazione con cui in PyTorch si chiama il
registratore delle derivate di cui parlavamo in apertura. Lo scarto fra i due
membri dell'equazione, quello che il ramo in alto calcola, si chiama
**residuo**: è lo stesso oggetto che nel racconto del compito in classe erano
i punti persi per una violazione della regola.

Lo schema è disegnato nel caso generale, quello di un'equazione con una
coordinata di spazio e una di tempo; nel resto della sezione lavoreremo sul
caso più semplice, con il solo tempo in ingresso. Vale la pena fissare il ramo
in alto, quello giallo-bruno (color ocra): quelle derivate *rispetto
all'input* non compaiono in nessun'altra architettura di questo libro. È il
pezzo nuovo, ed è tutto qui.

## Una molla come banco di prova

Ci serve un problema abbastanza semplice da avere una soluzione esatta con cui
dare i voti alla rete, e abbastanza ricco da non essere un giocattolo. Il
classico dei classici: l'**oscillatore armonico smorzato** (un peso appeso a
una molla, con un po' d'attrito che spegne piano piano le oscillazioni). La
legge di Newton per questo sistema è

$$
m\,u''(t) + c\,u'(t) + k\,u(t) = 0,
\qquad u(0) = 1, \quad u'(0) = 0,
$$

dove $u(t)$ è lo spostamento dalla posizione di riposo, $m$ la massa, $c$
il coefficiente di smorzamento (l'attrito) e $k$ la rigidezza della molla.
Scegliamo numeri concreti: $m = 1$, $c = 0{,}4$, $k = 4$. Le condizioni
iniziali dicono che al tempo zero il peso è spostato di un'unità e viene
lasciato andare da fermo.

`````{tab} Elementare

Leggiamo l'equazione come una regola di buon senso, portando tutto a destra:
*accelerazione* $= -4 \times$ *posizione* $- 0{,}4 \times$ *velocità*. Due
forze, cioè: la molla richiama sempre verso il centro, tanto più forte quanto
più sei lontano (il fattore 4); l'attrito frena sempre, tanto più quanto più
vai veloce (il fattore 0,4). Facciamo il conto a mano sull'istante iniziale:
posizione 1, velocità 0, quindi accelerazione
$= -4 \cdot 1 - 0{,}4 \cdot 0 = -4$ (il peso parte richiamato con decisione
verso il centro).

E qui i due nomi che abbiamo usato finora si saldano, perché la curva sul
foglio *è* il movimento del peso: la sua pendenza è la **velocità** (quanto in
fretta il peso si sposta) e la sua curvatura è l'**accelerazione** (quanto in
fretta cambia quella velocità). Un nome viene dal disegno, l'altro dalla
fisica, ma l'oggetto è lo stesso, ed è il motivo per cui una regola sul moto
di un peso si può far rispettare a una linea tracciata su un foglio.

Il film completo lo conosce chiunque abbia giocato con una molla: il peso
oscilla su e giù, circa una volta ogni 3,2 secondi con i nostri numeri, e ogni
oscillazione è un po' più bassa della precedente, perché l'attrito ruba
energia a ogni passaggio. Dopo 10 secondi l'ampiezza (l'altezza del rimbalzo,
misurata dal centro) è scesa a circa un settimo di quella iniziale. I due
numeri non piovono dal cielo: si ricavano dai valori scelti sopra, con il
conto svolto nell'altra scheda, che dà 3,16 secondi per un'oscillazione e
un'ampiezza finale pari al 13,5% di quella di partenza. Questa è la curva che
lo studente del compito in classe deve disegnare, e che la nostra rete dovrà
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
tre oscillazioni abbondanti l'ampiezza cala a circa il 14%. Questa formula
sarà la pagella con cui giudicheremo la PINN.

`````

## Perché tanh, e non ReLU

Prima di scrivere la rete, una scelta che in ogni altro capitolo sarebbe stata
automatica: la funzione di attivazione. Nel capitolo sulle reti neurali
abbiamo incoronato la ReLU regina degli strati nascosti: veloce, niente
saturazione, gradienti che non svaniscono. Qui però la ReLU è squalificata in
partenza, e il motivo è istruttivo.

`````{tab} Elementare

**La curva dev'essere liscia**, e la ReLU non sa disegnare curve lisce. La
ReLU è fatta di due tratti dritti attaccati in un angolo, e una rete di sole
ReLU produce curve fatte così: segmenti dritti incollati uno dopo l'altro,
come una spezzata. Una spezzata però non ha curvatura da nessuna parte,
perché un tratto dritto non piega, e negli angoli, dove piegherebbe, la
curvatura non si riesce nemmeno a calcolare. Ma la regola della molla parla
proprio di curvatura, che ne è anzi il termine principale: con una curva a
spezzata il professore non vedrebbe più il pezzo più importante della regola,
e qualunque disegno gli sembrerebbe corretto. Serve dunque una curva che
pieghi dolcemente dappertutto, ed è quello che fa la vecchia S centrata nello
zero: pensionata dalla ReLU nel deep learning "normale", qui si prende la
rivincita.

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
standard delle PINN; funzionano bene anche il seno e la softplus (una versione
arrotondata della ReLU), perché l'unica cosa che conta davvero è la
regolarità.

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
multistrato), la più semplice delle architetture di questo libro; e `shape`,
nei commenti, è semplicemente la forma della tabella di numeri, qui 200 righe
per una colonna.

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
    prodotto vettore–jacobiana $\mathbf{v}^\top \mathbf{J}$, dove
    $\mathbf{J}$ è la tabella di tutte le derivate di tutte le uscite rispetto
    a tutti gli ingressi: qui $\mathbf{v}$ è il vettore di uni e
    $\mathbf{J}$ è diagonale (ogni uscita dipende da un solo ingresso),
    quindi il prodotto restituisce esattamente la colonna delle derivate.

Il secondo è `create_graph=True`, e senza non funzionerebbe niente: chiede ad
autograd di *registrare anche il calcolo della derivata*, così che la derivata
resti a sua volta derivabile. Ci serve due volte. Primo, per derivare di
nuovo: `u_tt` è la derivata di `u_t`, quindi il grafo di `u_t` deve esistere.
Secondo, più sottile: `u_t` e `u_tt` finiscono *dentro la loss*, e quando
chiamiamo `loss.backward()` il gradiente deve poter attraversare anche il
calcolo delle derivate per arrivare fino ai pesi. È una derivata di una
derivata (il registratore che registra sé stesso) ed è il motivo per cui ogni
epoca di una PINN costa più di un'epoca di regressione ordinaria.

Notare infine il peso $100$ davanti alla `loss_iniziale`, che è la cosa meno
innocente del programma. Il termine di fisica, da solo, non ha una risposta
sola: qualunque moto di *quella* molla lo soddisfa, compresa una curva piatta
ferma sullo zero per sempre (un peso fermo al centro, senza nessuno che lo
sposti, resta fermo, e la regola dice proprio questo). A distinguere la nostra
traiettoria da tutte le altre ci sono soltanto le due condizioni di partenza,
che però da sole tirano poco, perché riguardano un istante mentre l'altro
termine tira sull'intera curva. Moltiplicarle per 100 serve a dare loro voce.

Che la faccenda sia seria si tocca con mano abbassando quel peso a 1. In una
prova fatta così, e senza cambiare nient'altro, l'addestramento è arrivato a
un residuo di $2 \cdot 10^{-5}$ (con il peso a 100, sulla stessa misura e
sullo stesso seme, il residuo si ferma a $8 \cdot 10^{-3}$: **quasi
quattrocento volte più alto**) e a uno scarto dalla nostra soluzione di
$0{,}73$. Non aveva sbagliato a risolvere l'equazione: l'aveva
risolta benissimo, scegliendo però un'altra delle sue infinite traiettorie.
Il che dice anche quanto vale il rimedio, e conviene dirlo subito: il peso
$100$ rende quella scorciatoia meno attraente, **non la vieta**. Fra poche
pagine ne vedremo la prova. Trentamila epoche di Adam
{cite}`kingma2015adam` dopo, ecco il verdetto, confrontando con la soluzione
analitica calcolata in NumPy:

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

Lanciando il programma, la loss parte dall'ordine del centinaio (all'inizio la
rete viola allegramente sia la fisica sia le condizioni iniziali) e scende di
oltre quattro ordini di grandezza; su CPU il tutto richiede qualche minuto.
Conviene però leggere le quattro righe finali con attenzione, perché dicono
due cose diverse, ed è raro che un esempio da manuale sia così onesto.

La prima è che il metodo funziona. La rete non ha mai visto un solo valore
della soluzione, solo la partenza e la legge, e ne esce una curva che oscilla
con il periodo giusto e si smorza con il ritmo giusto. Sui primi cinque
secondi lo scarto dalla formula esatta resta sotto le sette centesime parti
dell'ampiezza iniziale. Per una curva ricostruita da una regola e da due
numeri, è molto.

La seconda è che quella curva **non è accurata quanto il residuo lascerebbe
credere**. Sui 200 punti in cui la regola è stata controllata il residuo vale
$8 \cdot 10^{-3}$, cioè la molla risulta obbedita quasi alla lettera; ma lo
scarto massimo dalla soluzione vera è di circa $0{,}15$, un settimo
dell'ampiezza di partenza, e le due curve messe una sull'altra si distinguono
benissimo. Si noti anche il terzo numero, quello che quasi nessuno stampa: su
una griglia fitta di istanti che la rete non ha mai visto il residuo è
$3 \cdot 10^{-2}$, quattro volte più alto che nei punti controllati. La rete
va un po' meglio dove la si guarda che dove non la si guarda. Qui è uno
scarto modesto, e fra poche righe vedremo quanto può diventare grande. Non è
nemmeno un errore sparso: sta quasi tutto nella **coda**, dopo il quinto
secondo, dove la rete comincia ad appiattirsi mentre la molla vera sta ancora
oscillando, con ampiezza ormai piccola ma non nulla. E c'è un dettaglio da
non lasciarsi sfuggire nel crollo della loss: quasi tutta quella caduta è il
termine sulle condizioni iniziali, che si esaurisce entro le prime mille
epoche; il termine di fisica, quello che dovrebbe fare il lavoro, scende in
tutto di un ordine e mezzo.

Quello scarto fra i due numeri, il residuo piccolo e l'errore grande, non è un
dettaglio di rifinitura. Vale la pena vedere fin dove arriva.

## Lo stesso codice, un altro seme

C'è una riga del programma che non abbiamo commentato: `torch.manual_seed(42)`,
in cima al primo blocco. Fissa il punto di partenza dei pesi, che altrimenti
sarebbero sorteggiati a caso ogni volta. Serve a rendere il risultato
riproducibile, e per tutto il resto del libro è una cortesia al lettore.
Qui è molto di più: cambiandola, cambia la conclusione.

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

Ecco che cosa stampa, sulla macchina su cui è stato scritto questo capitolo:

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
sezione. Fra le 17 500 e le 27 500 epoche il residuo scende di **quasi tre
ordini di grandezza**, da $3{,}3 \cdot 10^{-2}$ a $5 \cdot 10^{-5}$: chiunque
guardasse soltanto la loss direbbe che proprio lì l'addestramento ha fatto un
salto di qualità. Nello stesso tratto l'errore vero **peggiora**, da 0,43 a
0,72. Le due colonne, che dovrebbero raccontare la stessa storia, vanno in
direzioni opposte.

Che cosa è successo lo si capisce guardando la curva: dopo il primo mezzo
periodo la rete scende a zero e ci resta, piatta, per tutti i dieci secondi.
Attraversa lo zero una volta sola, contro le sei della soluzione vera. La
condizione di partenza la paga per intero ($u(0) = 1{,}0006$, praticamente
perfetta) e per tutto il resto dell'intervallo si arrende alla soluzione
banale.

Ma il confronto stampato alla fine dice qualcosa di più, ed è la ragione per
cui vale la pena misurare il residuo in due posti invece che in uno:

```
                           seme 42      seme 7
residuo sui suoi punti    7.77e-03    4.77e-04
residuo sulla griglia     3.12e-02    1.23e+03
errore vero                  0.154       0.720
```

Sui duecento istanti in cui è stata controllata, la rete del seme 7 rispetta
la regola della molla **sedici volte meglio** di quella del seme 42. Su una
griglia fitta di istanti che non ha mai visto, la viola **quarantamila volte
peggio**. Fra le due righe c'è un fattore di due milioni e mezzo, e riguardano
la stessa rete nello stesso momento.

Non è dunque soltanto che il residuo basso non garantisce la soluzione giusta:
è che quel residuo basso è stato ottenuto **proprio e soltanto nei punti che
si stanno guardando**. La rete ha imparato a essere impeccabile all'esame e
sregolata fuori.

`````{tab} Elementare

È la scorciatoia di cui parlavamo tre paragrafi fa, quella che il peso 100
doveva impedire. Uno studente che consegna un foglio con una riga dritta sullo
zero non sta violando la regola della molla: un peso fermo al centro, senza
nessuno che lo sposti, resta fermo, e la regola dice esattamente questo. Il
professore, che la soluzione non la conosce e sa solo verificare la regola,
non ha nulla da eccepire. L'unica cosa che distingue quel foglio da quello
giusto è la partenza, ed è un punto solo contro una curva intera.

E c'è il trucco in più, quello che spiega i due residui così diversi. Il
professore controlla duecento punti, sempre gli stessi. Lo studente lo ha
capito, e ha imparato a stare in riga *esattamente lì*: fra un punto di
controllo e l'altro, dove nessuno guarda, la sua curva fa uno strappo violento
e strettissimo, largo meno della distanza fra due controlli. Sul registro del
professore il compito è quasi perfetto; il disegno è sbagliato. Non è furbizia
di nessuno, è che l'unico modo che ha di prendere voti è stare in riga dove si
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
paga per intero, $u_\theta(0) = 1{,}0006$, e si tiene il residuo quasi nullo
dappertutto. È il fenomeno che la prossima sezione chiamerà mancanza di
**ordine causale**: la loss somma residui su punti sparsi nel dominio e nulla
obbliga la rete a propagare in avanti nel tempo l'informazione della partenza.

Il secondo è il **campionamento finito**, e smentisce una rassicurazione che
avevamo dato noi. Nella scheda di apertura si diceva che fra un punto di
collocazione e il successivo «ci pensa la regolarità della rete, che non può
oscillare selvaggiamente». Qui invece lo fa: misurato su una griglia da
200 000 istanti, il residuo di questa rete tocca un massimo di $6{,}6 \cdot
10^2$ a $t = 1{,}26$, con un picco largo $0{,}037$ secondi, mentre i due punti
di collocazione più vicini stanno a $1{,}189$ e $1{,}205$, cioè a $0{,}058$
secondi di distanza. **Il picco è più stretto del passo di campionamento e sta
nel buco fra due punti.** Una rete `tanh` con tre strati da 32 neuroni ha
abbastanza capacità per infilare una guglia dove non viene interrogata, e
duecento punti su dieci secondi non bastano a impedirglielo. La regolarità
della rete è un argomento asintotico, non una garanzia a $N_c$ finito: dice
che infittendo i punti la cosa si chiude, non che sia già chiusa.

`````

Il seme 7 non è nemmeno un caso isolato. Rilanciando lo stesso programma su
sei semi, misurando a fine addestramento il residuo sui punti di collocazione
di ciascuna corsa, e ordinando le sei **dalla loss migliore alla peggiore**,
si ottiene questo:

| seme | residuo finale | errore massimo | attraversamenti dello zero |
|---:|---:|---:|---:|
| 7 | $3{,}4 \cdot 10^{-4}$ | 0,720 | 1 |
| 3 | $2{,}1 \cdot 10^{-3}$ | 0,629 | 4 |
| 42 | $7{,}8 \cdot 10^{-3}$ | **0,154** | 6 |
| 1 | $1{,}7 \cdot 10^{-2}$ | 0,212 | 5 |
| 0 | $2{,}7 \cdot 10^{-2}$ | 0,259 | 5 |
| 2 | $3{,}1 \cdot 10^{-2}$ | 0,289 | 5 |

Le due corse con il residuo **più basso in assoluto** sono le due sbagliate, e
sono anche quelle che restano più lontane dalle sei oscillazioni della
soluzione vera: una e quattro, contro le cinque o sei delle altre. Solo il
seme 42 le completa tutte. Fra le altre quattro il residuo torna a essere una
guida sensata (chi ce
l'ha più basso sbaglia meno), il che rende la trappola ancora più insidiosa:
la loss è informativa finché la rete sta risolvendo il problema giusto, e
smette di esserlo esattamente quando serve, cioè quando ha smesso di
risolverlo. Due corse su sei, su un'equazione che si risolve a mano in mezza
pagina.

Due conseguenze pratiche, e sono le più utili di tutta la sezione. La prima:
in una PINN **la loss non è una pagella**. In un problema di apprendimento
ordinario, una loss di validazione che scende è una buona notizia; qui il
residuo può scendere allontanandosi dalla risposta, e senza una soluzione di
riferimento (che nei casi veri non c'è, altrimenti non useremmo una PINN) non
c'è nulla che lo segnali. La seconda: un risultato ottenuto con un solo seme
non è un risultato. Il modo minimo di lavorare seriamente con questi metodi è
rilanciare con qualche seme diverso e guardare quanto le risposte si somigliano
fra loro, che è l'unica cosa che si può fare quando la risposta giusta non si
conosce.

E una terza, che costa pochissimo e che i due residui affiancati suggeriscono
da sé: **il residuo va sempre misurato anche dove la rete non è stata
addestrata**, su una griglia fitta o su punti estratti di nuovo. È il
corrispettivo dell'insieme di validazione dei capitoli sull'apprendimento, e
per la stessa ragione: un punteggio calcolato dove il modello si è allenato
misura anche quanto bene ha imparato a compiacere quel campione. Chi
ricampiona i punti di collocazione a ogni epoca, che è il rimedio più diffuso,
sta di fatto togliendo alla rete l'esame su cui prepararsi.

Niente di tutto questo smentisce il metodo, e non è il caso di esagerare in
senso opposto: quattro corse su sei ricostruiscono un'oscillazione smorzata
riconoscibile, partendo da un punto e da una regola, il che resta notevole. Ma
«di solito funziona» non è una garanzia, e su un problema da manuale, con
soluzione nota, tre parametri e una sola variabile, a separare la corsa buona
da quella fallita è stato il seme del generatore casuale. È il motivo per cui
la prossima sezione mette in fila i limiti prima delle applicazioni, e non
dopo.

## Né regressione né solutore: una terza via

Fermiamoci a guardare che cosa è successo, perché è facile passarci sopra.

`````{tab} Elementare

Confrontiamo con i due mestieri che già conosciamo. La **regressione** dei
capitoli sul machine learning è unire i puntini: senza puntini non parte
nemmeno, e per disegnare questa curva le sarebbero servite decine di misure
sparse su tutti i 10 secondi. La nostra rete ha ricevuto **zero misure**: un
punto di partenza, una regola, fine; la fisica ha fatto il lavoro dei dati. Il
**solutore classico** visto in apertura di capitolo (il conto a passettini
sulla griglia di istanti, quello del caffè), invece, la curva la sa
calcolare, ma avanza a passettini su una griglia di istanti e restituisce una
tabella: vuoi il valore tra due righe? Tocca a te indovinare quanto vale in
mezzo, tirando una linea fra i due valori vicini (si dice *interpolare*).
Vuoi proseguire oltre l'ultimo istante? Rifai il conto. La PINN restituisce
una **funzione**: chiedile il valore a $3{,}7$ secondi, o in qualunque altro
punto, e risponde all'istante, perché la soluzione ormai abita dentro la rete
(continua, senza griglia, interrogabile ovunque). Onestà d'obbligo: su questo
problemino il solutore classico resta molto più veloce; il vantaggio emerge
quando dati e legge vanno mescolati, come vedremo tra un attimo.

`````

`````{tab} Superiore

Rispetto alla **regressione pura**: minimizzare solo il termine dati richiede
$N_d$ grande e non promette nulla tra un campione e l'altro, mentre qui il
residuo vincola $u_\theta$ su tutto il dominio e bastano le condizioni
iniziali (il termine di fisica agisce come una regolarizzazione infinitamente
informata). Rispetto a un **integratore classico** (Eulero, Runge–Kutta):
quello discretizza il tempo con passo $h$, propaga sequenzialmente e offre
garanzie di convergenza con errore $O(h^p)$; la PINN sostituisce la
propagazione con un'ottimizzazione globale non convessa; nessuna garanzia
formale, costo superiore di ordini di grandezza su un problema standard come
questo, ma soluzione *mesh-free* e continua, valutabile (e derivabile) in
qualunque punto.

Quanto alla storia, un'onestà dovuta: l'idea non nasce nel 2019, e nemmeno
nel 1998. La loss delle PINN, esattamente com'è scritta qui (un MLP che
approssima la soluzione, il residuo minimizzato ai punti di collocazione e le
condizioni imposte **come penalità nella stessa funzione obiettivo**), è di
Dissanayake e Phan-Thien nel **1994** {cite}`dissanayake1994neural`, e reti
addestrate a risolvere equazioni differenziali compaiono già in Lee e Kang nel
1990 {cite}`lee1990neural`. Isaac Lagaris, Aristidis Likas e Dimitrios
Fotiadis, nel 1998, pubblicano la variante che di solito viene citata come
capostipite {cite}`lagaris1998artificial`, ed è utile distinguerla perché non
è la stessa cosa: Lagaris costruisce la soluzione di prova in modo che
condizioni iniziali e al contorno siano soddisfatte *esattamente*, per
costruzione, e resta da minimizzare il solo residuo. È il vincolo imposto
*a priori*; la PINN, come la formulazione del 1994, lo impone invece come
penalità, una scelta che semplifica il metodo ma che, come abbiamo appena
visto con il seme sfortunato e come vedremo nella prossima sezione, ha un
costo. Ma nel 1994 le derivate della rete andavano ricavate con
formule scritte a mano, caso per caso, e l'ottimizzazione girava su CPU
dell'epoca: l'idea restò di nicchia per vent'anni. Quando Maziar Raissi, Paris
Perdikaris e George Karniadakis la rilanciano nel 2019
{cite}`raissi2019physics`, la differenza non è concettuale ma
infrastrutturale: autograd generale e maturo {cite}`paszke2019pytorch`; le due
chiamate a `torch.autograd.grad` di poco fa, e GPU per l'addestramento. A
volte, nella ricerca, l'idea giusta deve solo aspettare i suoi attrezzi.

`````

## Il problema inverso, in tre righe di codice

Chiudiamo con la variazione promessa in apertura di capitolo: quella che, più
di ogni altra, giustifica l'esistenza delle PINN. Finora la rigidezza $k$ la
conoscevamo. Ribaltiamo la situazione: la molla è dentro una scatola chiusa,
$k$ non lo sappiamo, ma un sensore ci ha regalato 25 misure della posizione,
equispaziate lungo i dieci secondi e sporcate da un rumore casuale di
ampiezza tipica $0{,}05$, cioè un ventesimo dello spostamento iniziale. Nel
codice cambia pochissimo:

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

dove `t_oss` e `u_oss` sono i tensori delle misure. E la `loss_iniziale`?
Non serve più: l'ancoraggio che prima spettava alle condizioni iniziali ora
lo danno le 25 misure, e il loro termine ne prende il posto (e il peso). Non
è cambiato nulla nel meccanismo: `k_appreso` è entrato nella lista dei parametri
dell'ottimizzatore, il gradiente della loss scende anche lungo di lui, e a
ogni passo Adam aggiusta insieme la curva *e* la legge, finché le due cose
non vanno d'accordo con le osservazioni. Partendo dal valore volutamente
sbagliato $k = 1$, la stima si stacca, sale e si assesta in prossimità del
valore vero $k = 4$ con cui avevamo generato le misure, ricostruendo al
passaggio l'intera traiettoria: la stima del parametro fisico è un
sottoprodotto della stessa discesa del gradiente.

E c'è un dettaglio che vale la pena raccogliere, dopo la brutta figura di
poco fa. Qui la traiettoria ricostruita è **più accurata** di quella del
problema diretto della stessa pagina, pur essendo il problema più difficile
dei due. Il motivo è tutto nella disposizione degli ancoraggi: nel problema
diretto la rete aveva un solo punto fermo, l'istante zero, e più si andava
avanti nel tempo più era libera di inventare; qui ha venticinque misure
sparse su tutto l'intervallo, che la tengono per mano fino in fondo. Sono
rumorose e sono poche, ma sono *dappertutto*, ed è quello che conta.

Sembra poco: tre righe. È moltissimo: è il medico legale che risale all'ora
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
  curva regolare, alla quale si può chiedere di rispettare una regola.
- Il punteggio da abbassare somma due voci: le violazioni della regola nei
  punti di controllo aperti a caso (i **punti di collocazione**) e gli scarti
  sulla partenza, cioè il punto da cui si parte e la pendenza con cui si
  parte (per un problema esteso nello spazio, anche cosa succede ai bordi).
  Se ci sono misure, entrano come terza voce. Alla partenza si dà più peso,
  perché è l'unico ancoraggio che c'è.
- **La curva dev'essere liscia**: se è fatta di segmenti dritti incollati
  uno dopo l'altro, come quelli che escono dalla ReLU, non ha curvatura da
  nessuna parte, e il professore non vedrebbe più il pezzo più importante
  della regola. Per questo qui si torna alla vecchia S centrata nello zero,
  che invece piega dolcemente dappertutto.
- **Il registratore deve annotare anche i propri conti.** Per sapere quanto
  la curva sale in un istante gli si chiede di riavvolgere i calcoli; ma
  quella pendenza serve poi altre due volte, per ricavarne la curvatura e per
  far arrivare la correzione fino ai pesi. Quindi, con un'apposita opzione,
  gli si dice di tenere memoria anche del conto della pendenza. È il motivo
  per cui un giro di addestramento di una PINN costa più di uno normale.
- Sulla molla con attrito (massa 1, attrito 0,4, rigidezza 4) la rete
  ricostruisce l'oscillazione senza aver mai visto un solo valore della
  soluzione oltre la partenza: un'oscillazione ogni 3,2 secondi circa e
  ampiezza scesa a circa un settimo dopo 10 secondi. Non però con la
  precisione che la loss lascerebbe credere: lo scarto dalla curva vera resta
  attorno al decimo, e si concentra nella seconda metà, lontano dall'unico
  ancoraggio.
- **Un punteggio basso non vuol dire risposta giusta**, ed è la lezione da
  portarsi via. Su sei ripartenze dello stesso identico programma, le due che
  ottengono il punteggio migliore sono le due che sbagliano di più: si sono
  accontentate di una curva che rispetta la regola ma non parte da dove
  doveva. Con una regola sola e nessuna misura in mezzo, «rispettare la
  regola» non basta a inchiodare la risposta.
- Non è unire i puntini (di puntini non ce n'è nessuno) e non è un conto
  fatto a passettini su una griglia di istanti: quello che resta alla fine è
  una **curva intera**, che risponde a qualunque istante le si chieda, anche
  a 3,7 secondi, anche in mezzo a due punti qualsiasi, senza tabelle da
  interpolare.
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
  corse con $\mathcal{L}_{\text{fisica}}$ più bassa ($3{,}4\cdot10^{-4}$ e
  $2{,}1\cdot10^{-3}$) sono quelle con l'errore più grande (0,72 e 0,63):
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
