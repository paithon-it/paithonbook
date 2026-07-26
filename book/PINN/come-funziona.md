# La legge dentro la loss

Per tutto il libro, autograd ha fatto un solo mestiere. Nel capitolo su
PyTorch l'abbiamo raccontato come un registratore che annota i calcoli e li
riavvolge all'indietro; da allora, ogni chiamata a `backward()` ha risposto
sempre alla stessa domanda: *se ritocco questo peso, quanto cambia l'errore?*
Derivate della loss **rispetto ai pesi**, milioni di volte, per addestrare
classificatori, traduttori, generatori.

Questa sezione comincia con un colpo di scena: la domanda si può cambiare.
Autograd non sa che cosa siano "i pesi": sa derivare qualunque uscita del
grafo rispetto a qualunque suo ingresso. Se gli chiediamo la derivata
dell'uscita della rete **rispetto all'input**, otteniamo qualcosa che finora
non ci era mai servito: data una rete $u_\theta(t)$ che riceve un istante $t$
e restituisce un numero, possiamo calcolare $u_\theta'(t)$ e $u_\theta''(t)$
in qualunque punto, esatte a meno della precisione di macchina. La rete smette
di essere soltanto una scatola addestrabile: diventa una **funzione
derivabile**, un oggetto matematico a pieno titolo. E a una funzione
derivabile si può chiedere di rispettare un'equazione differenziale.

Tutto il metodo delle PINN sta in questa mossa. Vediamola all'opera.

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
   velocità in quello stesso punto.

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
\lambda \underbrace{\Big[ \big(u_\theta(0) - u_0\big)^2
+ \big(u_\theta'(0) - v_0\big)^2 \Big]}_{\text{condizioni iniziali}},
$$

dove i $t_j$ sono gli $N_c$ **punti di collocazione** (istanti sparsi nel
dominio, casuali o equispaziati, nei quali esigiamo il rispetto
dell'equazione) e $\lambda > 0$ bilancia i due termini. Per una PDE su un
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
omogenea con residuo nullo ovunque. Sono le condizioni iniziali a
selezionare *la* soluzione tra le infinite; per questo in pratica si sceglie
$\lambda$ ben maggiore di 1: quelle due condizioni sono l'unico ancoraggio,
e vanno difese dal peso schiacciante degli altri termini.

`````

```{figure} ../figures/pinn-schema.svg
:name: fig-pinn-schema
:alt: "Schema del metodo PINN: le coordinate di input entrano in una rete MLP che produce la soluzione candidata u; da questa partono il ramo di autograd, che calcola le derivate rispetto all'input e compone il residuo dell'equazione, e il ramo dei dati e delle condizioni iniziali; entrambi confluiscono nella loss totale, che con la backpropagation aggiorna i pesi della rete."
:width: 100%

L'anatomia di una PINN: la stessa rete alimenta il ramo della fisica (il
residuo, via autograd) e quello dei dati; la loss li somma e la
backpropagation chiude il cerchio.
```

In {numref}`fig-pinn-schema` c'è il metodo per intero. Vale la pena fissare il
ramo color ocra: quelle derivate *rispetto all'input* non compaiono in
nessun'altra architettura di questo libro. È il pezzo nuovo, ed è tutto qui.

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

Il film completo lo conosce chiunque abbia giocato con una molla: il peso
oscilla su e giù, circa una volta ogni 3,2 secondi con i nostri numeri, e ogni
oscillazione è un po' più bassa della precedente, perché l'attrito ruba
energia a ogni passaggio. Dopo 10 secondi l'ampiezza è scesa a circa un
settimo di quella iniziale. Questa è la curva che lo studente del compito in
classe deve disegnare, e che la nostra rete dovrà imparare senza vederne
neppure un punto, tranne la partenza.

`````

`````{tab} Superiore

È un'equazione lineare del secondo ordine a coefficienti costanti: si
risolve con l'equazione caratteristica $m\lambda^2 + c\lambda + k = 0$,
ovvero $\lambda^2 + 0{,}4\,\lambda + 4 = 0$. Il discriminante è negativo
($0{,}16 - 16 < 0$): radici complesse coniugate
$\lambda = -\gamma \pm i\,\omega_d$, con

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
partenza, e il motivo è istruttivo. La ReLU è fatta di due semirette: una rete
di sole ReLU calcola una funzione *lineare a tratti*, la cui derivata prima è
a gradini e la cui **derivata seconda è zero quasi ovunque**. Ma nel nostro
residuo compare $u''$: per una rete ReLU sarebbe identicamente nullo (tranne
nei punti di piega, dove non esiste proprio), e il termine principale
dell'equazione diventerebbe invisibile alla loss. La `tanh`, al contrario, è
liscia (derivabile infinite volte, con derivate continue a ogni ordine) e
infatti è la scelta standard delle PINN (funzionano bene anche il seno e la
softplus: l'importante è la regolarità). La vecchia S centrata nello zero,
pensionata dalla ReLU nel deep learning "normale", qui si prende la rivincita.

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
prospettiva da cui siamo partiti. E la rete è minuscola (tre strati nascosti
da 32 neuroni), perché la funzione da rappresentare è una curva liscia in una
dimensione, non ImageNet.

Il cuore del metodo sono due chiamate a `torch.autograd.grad`, con due
argomenti che non avevamo mai usato:

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
valori, uno per punto di collocazione, e autograd (che di suo calcola prodotti
vettore–jacobiana) con un vettore di uni restituisce in un colpo solo tutte le
200 derivate. Poiché ogni $u_j$ dipende soltanto dal suo $t_j$, non c'è alcuna
mescolanza: nella colonna `u_t` la riga $j$ è esattamente $u_\theta'(t_j)$.

Il secondo è `create_graph=True`, e senza non funzionerebbe niente: chiede ad
autograd di *registrare anche il calcolo della derivata*, così che la derivata
resti a sua volta derivabile. Ci serve due volte. Primo, per derivare di
nuovo: `u_tt` è la derivata di `u_t`, quindi il grafo di `u_t` deve esistere.
Secondo, più sottile: `u_t` e `u_tt` finiscono *dentro la loss*, e quando
chiamiamo `loss.backward()` il gradiente deve poter attraversare anche il
calcolo delle derivate per arrivare fino ai pesi. È una derivata di una
derivata (il registratore che registra sé stesso) ed è il motivo per cui ogni
epoca di una PINN costa più di un'epoca di regressione ordinaria.

Notare infine il peso $100$ sulla `loss_iniziale`: come detto sopra, il
termine di fisica da solo sarebbe felicissimo con la soluzione nulla
$u \equiv 0$; le due condizioni iniziali sono l'unica cosa che gliela vieta, e
vanno protette. Trentamila epoche di Adam {cite}`kingma2015adam` dopo, il
verdetto, confrontando con la soluzione analitica calcolata in NumPy:

```python
# La soluzione analitica, per dare i voti alla rete
gamma = c / (2 * m)                        # 0.2
omega_d = np.sqrt(k / m - gamma ** 2)      # sqrt(3.96) ~ 1.98997

t_test = np.linspace(0.0, 10.0, 500)
u_esatta = np.exp(-gamma * t_test) * (
    np.cos(omega_d * t_test) + (gamma / omega_d) * np.sin(omega_d * t_test)
)

with torch.no_grad():   # solo valutazione: registratore spento
    t_torch = torch.tensor(t_test, dtype=torch.float32).reshape(-1, 1)
    u_pinn = rete(t_torch).squeeze().numpy()

print(f"errore massimo: {np.abs(u_pinn - u_esatta).max():.1e}")
```

Lanciando il programma, la loss parte dall'ordine del centinaio (all'inizio la
rete viola allegramente sia la fisica sia le condizioni iniziali) e precipita
di molti ordini di grandezza; su CPU il tutto richiede qualche minuto. Alla
fine, su 500 punti di verifica sparsi lungo tutti i 10 secondi, lo scarto
massimo dalla formula esatta scende all'ordine del millesimo: le due curve
sono indistinguibili a occhio. E ripetiamolo, perché è il punto dell'intera
sezione: la rete non ha mai visto un solo valore della soluzione. Solo la
partenza e la legge.

## Né regressione né solutore: una terza via

Fermiamoci a guardare che cosa è successo, perché è facile passarci sopra.

`````{tab} Elementare

Confrontiamo con i due mestieri che già conosciamo. La **regressione** dei
capitoli sul machine learning è unire i puntini: senza puntini non parte
nemmeno, e per disegnare questa curva le sarebbero servite decine di misure
sparse su tutti i 10 secondi. La nostra rete ha ricevuto **zero misure**: un
punto di partenza, una regola, fine; la fisica ha fatto il lavoro dei dati. Il
**solutore numerico** visto in apertura di capitolo, invece, la curva la sa
calcolare, ma avanza a passettini su una griglia di istanti e restituisce una
tabella: vuoi il valore tra due righe? Interpoli. Vuoi proseguire oltre
l'ultimo istante? Rifai il conto. La PINN restituisce una **funzione**:
chiedile il valore a $3{,}7$ secondi, o in qualunque altro punto, e risponde
all'istante, perché la soluzione ormai abita dentro la rete (continua, senza
griglia, interrogabile ovunque). Onestà d'obbligo: su questo problemino il
solutore classico resta molto più veloce; il vantaggio della PINN emerge
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

Quanto alla storia, un'onestà dovuta: l'idea non nasce nel 2019. Isaac
Lagaris, Aristidis Likas e Dimitrios Fotiadis pubblicano nel 1998 un metodo
che è, a tutti gli effetti, questo {cite}`lagaris1998artificial`: MLP come
soluzioni di prova di ODE e PDE, addestrati a minimizzare il residuo sui punti
di collocazione. Ma nel 1998 le derivate della rete andavano ricavate con
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
sporcate di rumore. Nel codice cambia pochissimo:

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

dove `t_oss` e `u_oss` sono i tensori delle misure. Non è cambiato nulla
nel meccanismo: `k_appreso` è entrato nella lista dei parametri
dell'ottimizzatore, il gradiente della loss scende anche lungo di lui, e a
ogni passo Adam aggiusta insieme la curva *e* la legge, finché le due cose
non vanno d'accordo con le osservazioni. Partendo dal valore volutamente
sbagliato $k = 1$, la stima si stacca, sale e si assesta in prossimità del
valore vero $k = 4$ con cui avevamo generato le misure, ricostruendo al
passaggio l'intera traiettoria. Nessun ciclo esterno di
tentativi ed errori, nessun solutore da richiamare mille volte: la stima
del parametro fisico è un sottoprodotto della stessa discesa del gradiente.

Sembra poco: tre righe. È moltissimo: è il medico legale che risale all'ora
del decesso, il geofisico che deduce la struttura del sottosuolo dalle onde
sismiche, l'ingegnere che stima l'usura di un componente dai sensori; la
famiglia di problemi in cui le PINN non hanno rivali comodi. La prossima
sezione è dedicata a loro.

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
- Sull'oscillatore smorzato ($m=1$, $c=0{,}4$, $k=4$) la PINN riproduce la
  soluzione analitica $u(t)=e^{-0{,}2t}(\cos\omega_d t +
  0{,}1005\,\sin\omega_d t)$, $\omega_d=\sqrt{3{,}96}$, senza aver visto un
  solo dato oltre le condizioni iniziali.
- L'idea è del 1998 {cite}`lagaris1998artificial`; l'esplosione del 2019
  {cite}`raissi2019physics` arriva quando autograd e GPU la rendono
  praticabile.
- **Problema inverso**: basta promuovere un coefficiente a `nn.Parameter`
  per stimarlo da poche misure rumorose, insieme alla soluzione. È la mossa
  che rende uniche le PINN.
```
