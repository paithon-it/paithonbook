# Il problema più semplice: i bandit a più braccia

Nel 1933, sulle pagine di *Biometrika*, William R. Thompson pone una domanda
che nasce da un disagio pratico {cite}`thompson1933likelihood`. In una
sperimentazione clinica si assegnano i pazienti a due trattamenti e si aspetta
la fine per sapere quale funzioni meglio. Ma a metà strada un'idea di quale sia
il migliore già ce l'abbiamo: continuare ad assegnare metà dei pazienti al
trattamento che sta perdendo è il prezzo che si paga per essere sicuri.
Thompson si chiede se quel prezzo si possa ridurre spostando via via
l'assegnazione verso il trattamento che sta andando meglio, senza per questo
smettere di raccogliere prove sull'altro.

È il **dilemma fra esplorare e sfruttare** annunciato nella panoramica del
capitolo, e qui si presenta nella forma più pura che esista, perché manca tutto
il resto: nessuno stato che cambia, nessuna conseguenza differita, nessun
merito da distribuire su una catena di mosse. Solo la tensione, nuda.

Il nome viene dallo slang americano: la macchinetta da casinò con la leva si
chiama *one-armed bandit*, il bandito con un braccio solo, perché ti deruba con
educazione. Una fila di macchinette, ognuna con una probabilità di vincita
diversa e ignota, è un **bandito a più braccia**. La domanda è quale leva
tirare, e quante volte, sapendo che ogni tiro speso a informarsi è un tiro non
speso a guadagnare.

## Un solo stato, molte leve

`````{tab} Elementare

Hai dieci leve davanti. Ognuna, quando la tiri, ti dà una somma che cambia ogni
volta: alcune leve sono in media generose, altre in media avare, ma nessuna
è costante e tu non sai quali siano quali. Hai mille tiri.

Tutto ciò che puoi fare è tenere un quaderno: per ogni leva, la media di quanto
ti ha reso finora. Quella media è la tua **stima**. All'inizio è pessima
perché si basa su un tiro o due; con l'uso migliora.

Il quaderno si aggiorna senza rifare la somma da capo, con una regola che vale
la pena guardare in faccia perché ritorna dappertutto in questo libro:

> stima nuova = stima vecchia + passo × (quello che ho appena visto − stima vecchia)

Cioè: sposto la stima verso la sorpresa, di un tanto deciso dal passo. Se il
passo è "uno diviso il numero di volte che ho tirato questa leva", si ottiene
esattamente la media di tutti i tiri. Se invece il passo lo tengo **fisso**, le
osservazioni recenti pesano di più e quelle vecchie svaniscono piano piano: è
quello che serve se le leve cambiano carattere nel tempo, cosa che nel mondo
reale succede sempre.

`````

`````{tab} Superiore

Il problema ha $k$ azioni. Ogni azione $a$ ha un valore vero
$q_*(a) = \mathbb{E}[R_t \mid A_t = a]$, ignoto, e la ricompensa osservata è
una realizzazione rumorosa attorno a quel valore. Non c'è stato: la
distribuzione delle ricompense non dipende da cosa è successo prima. Un bandit
è, se si vuole, un MDP con un solo stato.

La stima naturale di $q_*(a)$ è la **media campionaria**

$$
Q_t(a) = \frac{\sum_{i<t} R_i \cdot \mathbb{1}[A_i = a]}{\sum_{i<t} \mathbb{1}[A_i = a]},
$$

che si calcola in forma incrementale, senza tenere in memoria la storia. Se
$Q_n$ è la stima dopo $n-1$ tiri della stessa leva e $R_n$ è l'$n$-esima
ricompensa,

$$
Q_{n+1} = Q_n + \frac{1}{n}\big(R_n - Q_n\big).
$$

È la forma canonica di ogni regola di apprendimento di questo libro:
*stima $\leftarrow$ stima $+$ passo $\cdot$ errore*. La ritroveremo identica
nel TD, dove l'errore è l'errore temporale, e imparentata nella discesa del
gradiente, dove il passo è il learning rate.

Sostituendo $1/n$ con un passo costante $\alpha \in (0,1]$ si ottiene invece

$$
Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1-\alpha)^{n-i} R_i ,
$$

una **media pesata esponenzialmente sul recente**: i pesi decadono
geometricamente all'indietro. Non converge (continua a inseguire), ed è
esattamente ciò che serve quando il problema è **non stazionario**, cioè quando
$q_*(a)$ cambia nel tempo. Il caso stazionario è l'eccezione, non la regola.

`````

## Il costo di essere avidi

Un agente **avido** (*greedy*) tira sempre la leva con la stima più alta. Il
guaio è che la stima più alta all'inizio è quasi sempre sbagliata, e l'errore
si auto-conferma: se la leva davvero migliore ha avuto sfortuna nei primi due
tiri, la sua stima resta bassa, non viene più scelta, e nessuno la corregge
mai. L'agente si chiude dentro una convinzione senza aver mai raccolto le prove
per smentirla.

Il banco di prova standard, introdotto da Sutton e Barto
{cite}`sutton2018reinforcement`, misura quanto costi: dieci leve i cui valori
veri sono estratti da una normale standard, ricompense con rumore unitario,
mille tiri, e tutto ripetuto su duemila banchi diversi per mediare la fortuna.
Su quel banco l'agente avido, negli ultimi cento tiri, sceglie la leva migliore
solo nel **36,7%** dei casi: due volte su tre sta tirando la leva sbagliata,
dopo mille tentativi.

Basta pochissimo per cambiare le cose. Con $\varepsilon$-greedy, cioè una leva
a caso una volta ogni dieci, si sale all'**80,2%**. La ricetta è quella già
vista nella panoramica, e la sua virtù è di non avere praticamente parametri
da tarare. Il suo difetto, però, è altrettanto chiaro: quando esplora, esplora
**a casaccio**. Tira con la stessa probabilità la leva che potrebbe essere la
seconda migliore e quella che ha già dimostrato dieci volte di essere pessima.
Le tre idee che seguono spendono l'esplorazione meglio.

## Tre modi di esplorare meglio di un dado

### Valori iniziali ottimisti

L'idea più economica non aggiunge una riga di codice: cambia solo il valore da
cui partono le stime.

Sul banco di prova i valori veri stanno attorno allo zero. Se inizializziamo
tutte le stime a $+5$, ogni leva promette molto più di quanto mantenga: qualunque
leva si tiri, la ricompensa **delude** e la sua stima scende sotto quelle delle
leve non ancora provate. L'agente, pur restando avido, gira su tutte le leve
per il semplice fatto di rimanere sistematicamente deluso. Sul banco di prova
arriva all'**86,6%**, il risultato migliore delle strategie qui elencate.

È un trucco, però, e conviene dire perché. L'ottimismo si esaurisce: dopo che
tutte le leve sono state provate abbastanza, la spinta a esplorare sparisce.
Su un problema stazionario va benissimo; su uno non stazionario, dove il mondo
cambia e servirebbe tornare a esplorare, non serve a niente. Come scrivono
Sutton e Barto, l'inizio del tempo capita una volta sola, e non conviene
puntarci troppo.

### UCB: esplorare in proporzione a quanto poco si sa

`````{tab} Elementare

Il difetto di $\varepsilon$-greedy è che il dado non guarda in faccia nessuno.
Ma fra le leve non scelte ce ne sono di due tipi diversissimi: quelle che
abbiamo provato venti volte e sono chiaramente mediocri, e quelle che abbiamo
provato una volta sola, per cui non sappiamo davvero niente. Le prime non
meritano un altro tiro, le seconde sì.

L'idea è aggiungere alla stima di ogni leva un **bonus di ignoranza**: quanto
più raramente l'ho tirata, tanto più genoroso è il bonus. Poi si sceglie, senza
dadi, la leva con la somma più alta. Una leva mediocre ma poco esplorata può
vincere il confronto proprio grazie al bonus; ogni volta che la si tira il
bonus cala, finché la sua mediocrità non emerge e smette di essere scelta.

Il bonus cresce anche col passare del tempo, e non è un dettaglio: significa
che una leva trascurata a lungo torna prima o poi in cima alla lista. Nessuna
leva viene abbandonata per sempre, ma le peggiori vengono ricontrollate sempre
più di rado.

`````

`````{tab} Superiore

L'**Upper Confidence Bound** sceglie

$$
A_t = \arg\max_{a} \left[\, Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \,\right],
$$

dove $N_t(a)$ è il numero di volte che $a$ è stata scelta prima di $t$ e
$c > 0$ regola quanto pesa l'incertezza (le azioni mai provate si trattano come
massimamente urgenti). Il termine sotto radice è, a meno di costanti, la
larghezza di un intervallo di confidenza sulla media di $a$: il numeratore
$\ln t$ cresce con il tempo, il denominatore $N_t(a)$ con l'uso. Il nome dice
il principio: **ottimismo di fronte all'incertezza**, cioè agire come se ogni
azione valesse il massimo compatibile con i dati raccolti, e lasciare che siano
i dati a smentire.

Il decadimento logaritmico non è decorativo. Lai e Robbins
{cite}`lai1985asymptotically` dimostrano che, per questa classe di problemi,
nessun algoritmo può avere un **rimpianto**

$$
\mathcal{R}_T = T \max_a q_*(a) - \mathbb{E}\!\left[\sum_{t=1}^{T} R_t\right]
$$

che cresca meno che logaritmicamente in $T$: perdere qualcosa è inevitabile,
la domanda è solo quanto. Auer, Cesa-Bianchi e Fischer
{cite}`auer2002finite` mostrano che UCB1 raggiunge quel limite, con una
garanzia valida a ogni istante finito e non solo asintoticamente. Confrontato:
$\varepsilon$-greedy con $\varepsilon$ costante ha rimpianto **lineare**,
perché continua a sbagliare una frazione fissa delle volte per sempre.

Sul banco di prova, con $c = 2$, UCB sceglie la leva migliore l'**85,9%** delle
volte. Il limite pratico è che la formula presuppone un problema stazionario e
un numero maneggiabile di azioni: portarla di peso nel reinforcement learning
con approssimazione di funzione, dove "quante volte ho visto questo stato" non
è nemmeno ben definito, non funziona, ed è il motivo per cui la sezione
sull'esplorazione nel deep RL dovrà inventarsi altro.

`````

### Il bandit a gradiente: preferenze, non valori

`````{tab} Elementare

Le strategie viste finora stimano *quanto vale* ogni leva e poi decidono. Se ne
può fare a meno: si può imparare direttamente una **preferenza**, un voto senza
unità di misura, e tirare le leve con probabilità proporzionale a quei voti.

La regola di aggiornamento è di buon senso: se la ricompensa appena incassata è
**migliore della media** di quelle ricevute finora, alzo il voto della leva che
ho tirato e abbasso quello di tutte le altre; se è peggiore, faccio l'opposto.

Quel confronto con la media è il pezzo importante e si chiama **termine di
riferimento**. Senza, l'algoritmo confronterebbe la ricompensa con lo zero, che
è un numero arbitrario: se tutte le leve pagano attorno a mille, tutte le
ricompense sembrano ottime e i voti salgono tutti insieme senza distinguere
nulla. Con il riferimento, quel che conta non è quanto ho preso, ma quanto ho
preso **rispetto al solito**.

`````

`````{tab} Superiore

Si mantiene una preferenza $H_t(a) \in \mathbb{R}$ per ogni azione, e la policy
è una softmax sulle preferenze:

$$
\pi_t(a) = \Pr\{A_t = a\} = \frac{e^{H_t(a)}}{\sum_{b} e^{H_t(b)}} .
$$

Le preferenze non stimano nulla, contano solo le loro differenze (aggiungerne
mille a tutte non cambia la policy). L'aggiornamento è una salita stocastica
sul gradiente della ricompensa attesa:

$$
H_{t+1}(A_t) = H_t(A_t) + \alpha\,\big(R_t - \bar{R}_t\big)\big(1 - \pi_t(A_t)\big),
\qquad
H_{t+1}(a) = H_t(a) - \alpha\,\big(R_t - \bar{R}_t\big)\,\pi_t(a)
\;\; \forall a \neq A_t ,
$$

dove $\bar{R}_t$ è la media delle ricompense fino a $t$, cioè la **baseline**.

Vale la pena riconoscere che cosa si sta guardando: è **REINFORCE con
baseline**, il metodo a gradiente di policy del capitolo sul deep
reinforcement learning, nel caso degenere di un solo stato. La stessa
struttura (una distribuzione parametrica sulle azioni, un aggiornamento
proporzionale alla ricompensa scostata da un riferimento) che là si scriverà
come $\nabla_\theta \log \pi_\theta(a\mid s)\,A_t$, con il vantaggio $A_t$ al
posto di $R_t - \bar{R}_t$. Il *vantaggio* dell'actor-critic nasce qui, e nasce
per la stessa ragione: ridurre la varianza senza spostare la media del
gradiente.

Che la baseline serva davvero si misura. Sul banco di prova centrato in zero il
metodo arriva all'**84,1%**. Traslando **tutte** le ricompense di $+4$, cosa
che non cambia in nulla la difficoltà del problema (le differenze fra le leve
sono identiche), con la baseline si resta all'**83,8%**, mentre togliendola si
crolla al **48,5%**. La baseline non è un'ottimizzazione: è ciò che rende
l'algoritmo indifferente all'origine della scala delle ricompense.

`````

## Alla prova: duemila banchi da mille tiri

I numeri citati qui sopra non sono presi da un paper, sono usciti dal codice
che segue. Le quattro strategie basate sui valori vivono in un solo blocco,
perché differiscono solo per come scelgono l'azione e per come iniziano.

```python
import numpy as np

K, PASSI, PROVE = 10, 1000, 2000     # 10 leve, 1000 tiri, 2000 banchi di prova

def prova(eps, q0=0.0, c=None, alpha=None):
    rng = np.random.default_rng(20260807)
    q_vero = rng.normal(0, 1, size=(PROVE, K))   # il valore vero di ogni leva
    ottima, righe = q_vero.argmax(axis=1), np.arange(PROVE)
    Q = np.full((PROVE, K), q0, dtype=float)     # le nostre stime
    N = np.zeros((PROVE, K))                     # quante volte ho tirato ogni leva
    centri = np.zeros(PASSI)
    for t in range(1, PASSI + 1):
        if c is None:
            a = Q.argmax(axis=1)
            caso = rng.random(PROVE) < eps       # ogni tanto, una leva a caso
            a = np.where(caso, rng.integers(0, K, PROVE), a)
        else:                                    # UCB: stima + incertezza
            bonus = np.where(N == 0, 1e6, c * np.sqrt(np.log(t) / np.maximum(N, 1e-9)))
            a = (Q + bonus).argmax(axis=1)
        r = rng.normal(q_vero[righe, a], 1.0)    # la ricompensa e' rumorosa
        N[righe, a] += 1
        passo = alpha if alpha else 1.0 / N[righe, a]   # media incrementale
        Q[righe, a] += passo * (r - Q[righe, a])        # vecchia + passo * errore
        centri[t-1] = (a == ottima).mean()
    return 100 * centri[-100:].mean()

print(f"greedy               {prova(eps=0.0):5.1f}%")
print(f"eps-greedy 0,01      {prova(eps=0.01):5.1f}%")
print(f"eps-greedy 0,1       {prova(eps=0.1):5.1f}%")
print(f"ottimista Q1=5       {prova(eps=0.0, q0=5.0, alpha=0.1):5.1f}%")
print(f"UCB c=2              {prova(eps=0.0, c=2.0):5.1f}%")

# greedy                36.7%
# eps-greedy 0,01       59.1%
# eps-greedy 0,1        80.2%
# ottimista Q1=5        86.6%
# UCB c=2               85.9%
```

Una riga merita attenzione, ed è quella di `eps-greedy 0,01`: al **59,1%** dopo
mille tiri sembra il peggiore dei rimedi, ma sta ancora salendo. Esplorando una
volta su cento impiega dieci volte più tempo a farsi un'idea di tutte le leve,
e alla fine supererà $\varepsilon = 0{,}1$, che invece continuerà per sempre a
buttare un tiro su dieci. La classifica dipende da quanto è lunga la partita, e
questa è una morale generale: **un iperparametro di esplorazione si sceglie
guardando l'orizzonte**, non la prima mille tiri.

Il bandit a gradiente ha una struttura diversa e sta in un blocco a sé, dove si
vede anche l'esperimento sulla baseline.

```python
import numpy as np

K, PASSI, PROVE = 10, 1000, 2000

def gradiente(alpha=0.1, baseline=True, shift=0.0):
    rng = np.random.default_rng(20260807)
    q_vero = rng.normal(shift, 1, size=(PROVE, K))
    ottima, righe = q_vero.argmax(axis=1), np.arange(PROVE)
    H = np.zeros((PROVE, K))          # preferenze: non sono valori, sono voti
    media_r, centri = np.zeros(PROVE), np.zeros(PASSI)
    for t in range(1, PASSI + 1):
        p = np.exp(H - H.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)                    # softmax
        a = (p.cumsum(axis=1) < rng.random((PROVE, 1))).sum(axis=1).clip(0, K-1)
        r = rng.normal(q_vero[righe, a], 1.0)
        scelta = np.zeros((PROVE, K)); scelta[righe, a] = 1.0
        base = media_r if baseline else 0.0                  # il termine di confronto
        H += alpha * (r - base)[:, None] * (scelta - p)      # sali sul gradiente
        media_r += (r - media_r) / t
        centri[t-1] = (a == ottima).mean()
    return 100 * centri[-100:].mean()

print(f"gradiente, ricompense centrate su 0   {gradiente():5.1f}%")
print(f"gradiente, ricompense centrate su +4  {gradiente(shift=4.0):5.1f}%")
print(f"  ... senza baseline                  {gradiente(shift=4.0, baseline=False):5.1f}%")

# gradiente, ricompense centrate su 0    84.1%
# gradiente, ricompense centrate su +4   83.8%
#   ... senza baseline                   48.5%
```

## Dove si incontrano davvero

Un bandit non è un giocattolo teorico, ed è probabilmente la parte di
reinforcement learning che più spesso finisce in produzione, proprio perché
rinuncia a tutto il resto.

**Test A/B, e il loro superamento.** Un test A/B classico manda metà del
traffico a ciascuna variante fino alla fine dell'esperimento: è la
sperimentazione clinica di Thompson, con gli stessi costi. Un'allocazione
adattiva sposta progressivamente il traffico verso la variante che sta
vincendo, e paga in complessità statistica (i dati non sono più raccolti in
modo indipendente dalla decisione) quello che guadagna in denaro non buttato.

**Esplorazione nei sistemi di raccomandazione.** Un catalogo ha continuamente
oggetti nuovi, di cui nessuno sa nulla: mostrarli è esplorare, e non mostrarli
mai garantisce che nessuno saprà mai se erano buoni. È il problema dell'avvio a
freddo visto nel capitolo sulla raccomandazione, letto dal lato della decisione
invece che da quello della rappresentazione.

**Ricerca di iperparametri.** Qui il collegamento è letterale. Il *successive
halving* e **Hyperband** del capitolo sul machine learning sono algoritmi di
bandit: ogni configurazione è una leva, addestrarla per un'epoca è un tiro, e
il problema si chiama *best-arm identification*, che è la variante in cui non
interessa massimizzare le ricompense lungo la strada ma solo indovinare alla
fine qual era la leva migliore.

Fra il bandit e il reinforcement learning pieno c'è un gradino intermedio che
copre buona parte delle applicazioni reali: il **bandit contestuale**, in cui
prima di scegliere si osserva una descrizione della situazione (chi è
l'utente, che ora è, da quale pagina arriva) e la leva migliore dipende da
quella. C'è uno stato, quindi, ma le azioni non lo influenzano: il contesto
successivo arriva dal mondo, non da ciò che abbiamo fatto.

Ed è esattamente il pezzo che manca. Quando le azioni cominciano a determinare
in quale situazione ci si troverà dopo, il problema smette di essere una
sequenza di scelte indipendenti e diventa una **catena**: una mossa fatta ora
può essere pagata o riscossa fra venti mosse, e bisogna capire a quale mossa
attribuirne il merito. Serve un'impalcatura più grande, ed è quella della
prossima sezione.

```{admonition} Da ricordare
:class: important
- Un **bandit a più braccia** è un problema di decisione senza stato: $k$
  azioni, ricompense rumorose, e il solo dilemma fra esplorare e sfruttare. È
  un MDP con un solo stato.
- Le stime si aggiornano con *stima $\leftarrow$ stima $+$ passo $\cdot$
  errore*: con passo $1/n$ si ottiene la media, con passo costante una media
  pesata sul recente, che è ciò che serve se il problema **non è stazionario**.
- L'agente **avido** si chiude in una convinzione mai verificata (36,7% di
  scelte ottime sul banco di prova standard); $\varepsilon$-greedy lo risolve
  a costo zero (80,2%) ma esplora **a casaccio**.
- **Valori iniziali ottimisti** (86,6%): esplorazione gratis, ma si esaurisce e
  non serve sui problemi non stazionari. **UCB** (85,9%): esplora in proporzione
  all'incertezza, e raggiunge il rimpianto logaritmico che Lai e Robbins
  dimostrano essere il minimo possibile.
- Il **bandit a gradiente** impara preferenze invece di valori, ed è REINFORCE
  con baseline in miniatura. La baseline non è un dettaglio: traslando le
  ricompense di $+4$, senza di essa si passa dall'84% al 48%.
- Si incontrano davvero in test A/B adattivi, esplorazione nei sistemi di
  raccomandazione e ricerca di iperparametri (Hyperband è *best-arm
  identification*). Il gradino successivo è il **bandit contestuale**, dove
  c'è uno stato ma le azioni non lo cambiano.
```
