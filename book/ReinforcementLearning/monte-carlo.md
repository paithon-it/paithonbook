# Giocare fino in fondo: i metodi Monte Carlo

Nel 1946, a Los Alamos, Stanisław Ulam era in convalescenza e passava le
giornate a fare solitari. A un certo punto si chiese quale fosse la
probabilità che un solitario venisse. Provò a calcolarla con la combinatoria,
si arenò, e gli venne l'idea che avrebbe cambiato mezzo secolo di scienza
applicata: invece di calcolare la probabilità, **giocare cento partite e
contare quante finiscono bene**. Ne parlò con John von Neumann, e Nicholas
Metropolis propose per il metodo un nome preso dal casinò dove uno zio di Ulam
andava a perdere i soldi presi in prestito dai parenti
{cite}`metropolis1987beginning`.

L'idea è tutta lì, e serve esattamente al punto in cui la sezione precedente
si è fermata. La *value iteration* sa calcolare i valori, ma pretende la mappa
dell'ambiente: le probabilità di transizione $P$ e le ricompense $R$. Se la
mappa non c'è, resta una via che non chiede nulla a nessuno: far vivere
all'agente molte partite intere, guardare come sono andate, e fare la media.

## Giocare, e poi fare la media

Il ritorno $G_t$ è già definito: la somma scontata delle ricompense da un certo
istante fino alla fine dell'episodio. Il valore $V^\pi(s)$ è il ritorno
*atteso* partendo da $s$. Un valore atteso si stima con una media campionaria,
e i campioni sono le partite.

`````{tab} Elementare

Vuoi sapere quanto vale, negli scacchi, una certa posizione. Un modo c'è, e
non richiede di capire niente di scacchi: da quella posizione gioca mille
partite fino allo scacco matto, segnati com'è finita ogni volta, e fai la
media. Se in media si vince, la posizione è buona.

I metodi **Monte Carlo** fanno questo, e la parola difficile non nasconde
niente di più. L'agente gioca un episodio dall'inizio alla fine, poi torna
indietro con la matita e, per ogni situazione attraversata, si annota quanto
ha raccolto **da lì in avanti**. Ripetuto molte volte, quel quaderno di
annotazioni diventa la stima del valore di ogni situazione: basta fare la
media di tutte le righe che parlano della stessa casella.

Nessuna mappa, nessuna formula sull'ambiente: solo partite giocate e una
media. Il prezzo è dichiarato subito: bisogna arrivare **alla fine** della
partita prima di poter scrivere qualsiasi cosa.

`````

`````{tab} Superiore

Ogni volta che un episodio attraversa lo stato $s$ si parla di **visita** a
$s$. Il metodo **a prima visita** stima $V^\pi(s)$ come la media dei ritorni
che seguono la *prima* visita a $s$ in ciascun episodio; quello **a ogni
visita** media i ritorni che seguono *tutte* le visite:

$$
V(s) \;=\; \frac{1}{|\mathcal{T}(s)|} \sum_{t \in \mathcal{T}(s)} G_t ,
$$

dove $\mathcal{T}(s)$ è l'insieme degli istanti in cui $s$ è stato visitato
(solo le prime visite, nella variante a prima visita).

La versione a prima visita ha una giustificazione immediata: i ritorni raccolti
sono variabili aleatorie **indipendenti e identicamente distribuite** con media
$V^\pi(s)$ e varianza finita, quindi per la legge dei grandi numeri la media
converge al valore vero, e l'errore standard cala come $1/\sqrt{n}$ con $n$
ritorni mediati. Ogni stima è **non distorta**. La variante a ogni visita non è
i.i.d. (i ritorni di uno stesso episodio sono correlati) e la sua stima è
distorta per $n$ finito, ma converge anch'essa e si estende meglio
all'approssimazione di funzione {cite}`sutton2018reinforcement`.

Il punto strutturale: qui **non c'è bootstrapping**. Il bersaglio è il ritorno
osservato, non una stima costruita a partire da altre stime. Ogni stato si
stima per conto proprio, e la stima di uno stato non dipende dalla stima degli
altri.

`````

## Che cosa cambia rispetto alla programmazione dinamica

Vale la pena mettere i due metodi uno accanto all'altro, perché la differenza
non è di efficienza ma di **che cosa serve sapere**.

La programmazione dinamica guarda **un passo in avanti ma in tutte le
direzioni**: per aggiornare $V(s)$ somma su tutti gli stati d'arrivo possibili,
pesandoli con le probabilità di transizione. Ha bisogno di quelle probabilità,
e in cambio non le deve stimare.

Monte Carlo guarda **in una direzione sola ma fino in fondo**: segue la
traiettoria realmente accaduta, dall'inizio alla fine dell'episodio, e ignora
le strade non prese. Non ha bisogno di sapere nulla dell'ambiente, e in cambio
paga in rumore.

Da questa differenza discendono tre conseguenze pratiche.

- Monte Carlo funziona anche quando l'ambiente è una **scatola nera** o un
  simulatore: basta saperci giocare, non saperlo descrivere. È spesso molto più
  facile scrivere un simulatore che scrivere la sua tabella di transizione.
- Il costo di stimare un singolo stato **non dipende dal numero di stati**. Se
  interessa il valore di una manciata di posizioni, si giocano partite da
  quelle e basta, senza spazzare l'intero spazio come fa la programmazione
  dinamica.
- Gli errori **non si propagano**. Una stima sbagliata in uno stato non
  contamina i vicini, perché nessuno la usa come bersaglio.

## Tre partite, coi numeri

Riprendiamo l'MDP in miniatura della {numref}`fig-mdp`, con $\gamma = 0{,}9$.
Stavolta fingiamo di **non** conoscere le transizioni: l'agente si limita a
giocare seguendo una policy che di norma sale verso l'obiettivo ma ogni tanto
tentenna. Ecco tre episodi, con le ricompense incassate lungo la strada.

1. $s_0 \xrightarrow{\,0\,} s_1 \xrightarrow{\,+10\,} s_2$
2. $s_0 \xrightarrow{\,-1\,} s_0 \xrightarrow{\,0\,} s_1 \xrightarrow{\,+10\,} s_2$
3. $s_0 \xrightarrow{\,0\,} s_1 \xrightarrow{\,-1\,} s_0 \xrightarrow{\,0\,} s_1 \xrightarrow{\,+10\,} s_2$

I ritorni si calcolano **all'indietro**, che è il modo economico di farlo:
partendo dalla fine, $G \leftarrow r + \gamma\,G$ a ogni passo indietro.
Nel secondo episodio, per esempio: dall'ultimo $s_1$ il ritorno è $10$; dal
secondo $s_0$ è $0 + 0{,}9 \times 10 = 9$; dal primo $s_0$ è
$-1 + 0{,}9 \times 9 = 7{,}1$.

| episodio | ritorni osservati |
|:--|:--|
| 1 | $G(s_0) = 9$; $G(s_1) = 10$ |
| 2 | $G(s_0) = 7{,}1$, poi $G(s_0) = 9$; $G(s_1) = 10$ |
| 3 | $G(s_0) = 6{,}39$, poi $G(s_0) = 9$; $G(s_1) = 7{,}1$, poi $G(s_1) = 10$ |

Adesso la media. **A prima visita** si conta una riga per episodio:

$$
V(s_0) = \frac{9 + 7{,}1 + 6{,}39}{3} = 7{,}50,
\qquad
V(s_1) = \frac{10 + 10 + 7{,}1}{3} = 9{,}03 .
$$

**A ogni visita** entrano tutte le righe, cinque per $s_0$ e quattro per $s_1$:

$$
V(s_0) = \frac{9 + 7{,}1 + 9 + 6{,}39 + 9}{5} = 8{,}10,
\qquad
V(s_1) = \frac{10 + 10 + 7{,}1 + 10}{4} = 9{,}28 .
$$

Due numeri diversi dagli stessi dati, ed entrambi legittimi: sono due
stimatori diversi della stessa quantità. E nessuno dei due tende ai $9$ e $10$
che la value iteration aveva calcolato nella sezione precedente, per una
ragione che conviene fissare: là si calcolava $V^*$, il valore della policy
**ottima**; qui si stima $V^\pi$, il valore della policy **che ha giocato
davvero**, tentennamenti compresi. Con tre episodi, per di più: la legge dei
grandi numeri ha bisogno di ben altro.

```python
gamma = 0.9

# Ogni episodio e' una lista di (stato, ricompensa incassata subito dopo).
episodi = [
    [("s0", 0.0), ("s1", 10.0)],
    [("s0", -1.0), ("s0", 0.0), ("s1", 10.0)],
    [("s0", 0.0), ("s1", -1.0), ("s0", 0.0), ("s1", 10.0)],
]

def ritorni(episodio):
    """Ritorni G_t, calcolati all'indietro: G <- r + gamma * G."""
    G, fuori = 0.0, []
    for stato, r in reversed(episodio):
        G = r + gamma * G
        fuori.append((stato, G))
    return list(reversed(fuori))

def monte_carlo(episodi, prima_visita=True):
    somma, conteggio = {}, {}
    for episodio in episodi:
        visti = set()
        for stato, G in ritorni(episodio):
            if prima_visita and stato in visti:
                continue           # a prima visita: le repliche non contano
            visti.add(stato)
            somma[stato] = somma.get(stato, 0.0) + G
            conteggio[stato] = conteggio.get(stato, 0) + 1
    return {s: somma[s] / conteggio[s] for s in somma}

print(monte_carlo(episodi, prima_visita=True))
# {'s0': 7.496666666666667, 's1': 9.033333333333333}
print(monte_carlo(episodi, prima_visita=False))
# {'s0': 8.098, 's1': 9.275}
```

Nulla nel codice conosce l'ambiente: legge una lista di partite già giocate.
È tutta la differenza con la sezione precedente.

## Dalla valutazione al controllo

Stimare il valore di una policy è metà del lavoro. Per **migliorarla** si
riusa lo schema della policy iteration: si valuta, si rende la policy greedy
rispetto ai valori stimati, si rivaluta. Con una differenza che sembra un
dettaglio tecnico e invece è il tema di tutto il capitolo.

`````{tab} Elementare

C'è una trappola. Se l'agente, dopo aver imparato che una certa mossa è buona,
la gioca sempre, le altre mosse non le prova più. E se non le prova più, non
scoprirà mai che una di quelle era migliore: il suo voto resterà per sempre
quello sbagliato del primo tentativo. Il quaderno delle medie ha una colonna
che non si aggiorna più.

Per questo un agente Monte Carlo che vuole *migliorare* (e non solo misurare)
deve continuare a fare mosse che non crede ottime. È lo stesso dilemma fra
esplorare e sfruttare che ritroveremo nella prossima sezione, e qui si presenta
nella forma più cruda: senza esplorazione, il metodo semplicemente non vede i
dati che gli servirebbero.

`````

`````{tab} Superiore

Il problema è che $Q^\pi(s,a)$ si può stimare solo per le coppie $(s,a)$ che
compaiono nei dati, e una policy deterministica ne genera una sola per stato.
Ci sono due rimedi classici.

Il primo è l'ipotesi degli **inizi esplorativi**: ogni episodio comincia da una
coppia $(s,a)$ estratta a caso, con probabilità non nulla per tutte. È comoda
nella teoria e quasi sempre inapplicabile, perché richiede di poter piazzare
l'agente dove si vuole.

Il secondo, praticabile, è restare su policy **$\varepsilon$-soft**, cioè con
$\pi(a\mid s) \ge \varepsilon/|\mathcal{A}|$ per ogni azione: la
$\varepsilon$-greedy della prossima sezione è il caso tipico. Il *policy
improvement theorem* continua a valere ristretto a questa classe, quindi
l'alternanza valuta-migliora converge, ma converge alla migliore policy
$\varepsilon$-soft, non alla migliore in assoluto {cite}`sutton2018reinforcement`.

La rinuncia è reale, e la via d'uscita è il paragrafo seguente: separare la
policy che **genera** i dati da quella che si sta **valutando**.

`````

## Imparare da una policy e giudicarne un'altra

Qui sta il concetto che questa sezione deve al resto del libro, perché più
avanti verrà usato tre volte senza essere più spiegato.

Chiamiamo $\pi$ la policy che vogliamo valutare (*target*) e $b$ quella che ha
effettivamente generato le partite (*comportamento*). Se $\pi = b$ siamo nel
caso **on-policy** visto finora. Se differiscono, siamo **off-policy**, ed è la
situazione interessante: imparare da un archivio di partite giocate da altri,
da un controllore preesistente, da un esperto umano, oppure da una versione
precedente di sé stessi.

`````{tab} Elementare

Il problema è che i dati raccontano la storia sbagliata. Se il giocatore che
ha lasciato le partite era prudente e la strategia che vuoi giudicare è
audace, le partite audaci nell'archivio sono poche, e mediarle tutte allo
stesso modo darebbe un giudizio sulla prudenza, non sull'audacia.

Il rimedio è **pesare** le partite invece di contarle tutte uguali. Una partita
che la strategia audace avrebbe giocato spesso e che il prudente ha giocato di
rado vale molto, perché è rara e informativa; una partita tipica del prudente
e che l'audace non farebbe mai vale poco o niente. Il peso è semplicemente il
rapporto fra quanto era probabile quella sequenza di mosse per l'una e per
l'altra.

Una condizione però serve, ed è di buon senso: l'archivio deve **contenere**
tutto ciò che la strategia da giudicare potrebbe fare. Se l'audace giocherebbe
una mossa che il prudente non ha mai provato nemmeno una volta, di quella mossa
non si può dire nulla, e nessun peso può inventare i dati mancanti.

`````

`````{tab} Superiore

La condizione di buon senso si chiama **copertura**: $\pi(a\mid s) > 0$ deve
implicare $b(a\mid s) > 0$. Ne segue che $b$ deve essere stocastica dove
differisce da $\pi$, mentre $\pi$ può tranquillamente essere deterministica
(ed è il caso che interessa nel controllo, dove $\pi$ è la greedy).

Il peso è il **rapporto di importance sampling**. La probabilità della
traiettoria $A_t, S_{t+1}, \dots, S_T$ sotto una policy è il prodotto dei
termini $\pi(A_k\mid S_k)\,P(S_{k+1}\mid S_k, A_k)$, e nel rapporto fra le due
policy accade la cosa che rende il metodo praticabile:

$$
\rho_{t:T-1}
= \prod_{k=t}^{T-1} \frac{\pi(A_k\mid S_k)\,P(S_{k+1}\mid S_k,A_k)}
                          {b(A_k\mid S_k)\,P(S_{k+1}\mid S_k,A_k)}
= \prod_{k=t}^{T-1} \frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)} .
$$

Le probabilità di transizione **si cancellano**, identiche a numeratore e
denominatore. Il correttore non dipende dall'MDP, che infatti non conosciamo:
dipende solo dalle due policy e dalle azioni osservate. È il motivo per cui
l'off-policy è possibile senza modello.

Poiché $\mathbb{E}\big[\rho_{t:T-1}\,G_t \mid S_t = s\big] = V^\pi(s)$, si può
stimare in due modi. L'**importance sampling ordinario** fa la media semplice
dei ritorni pesati; quello **pesato** normalizza per la somma dei pesi:

$$
V_{\text{ord}}(s) = \frac{\sum_{t\in\mathcal{T}(s)} \rho_{t:T-1}\,G_t}{|\mathcal{T}(s)|},
\qquad
V_{\text{pes}}(s) = \frac{\sum_{t\in\mathcal{T}(s)} \rho_{t:T-1}\,G_t}
                          {\sum_{t\in\mathcal{T}(s)} \rho_{t:T-1}} .
$$

Il compromesso fra i due è una lezione statistica che vale oltre il RL.
L'ordinario è **non distorto** ma la sua varianza può essere illimitata,
perché un rapporto può valere dieci o mille e moltiplicare un singolo ritorno
per quella cifra. Il pesato è **distorto** (la distorsione svanisce al crescere
dei campioni) ma il peso di un singolo ritorno non supera mai $1$, e la sua
varianza converge a zero anche quando quella dei rapporti è infinita, un
risultato di Precup, Sutton e Dasgupta. In pratica si preferisce quasi sempre
il pesato {cite}`sutton2018reinforcement`.

`````

Un esempio piccolo rende concreto il numero. Supponiamo che $b$ scelga fra due
azioni tirando una moneta ($b = 0{,}5$ per entrambe) e che $\pi$ sia
deterministica. Una partita di tre mosse in cui $b$ ha per caso scelto ogni
volta l'azione che anche $\pi$ avrebbe scelto ha peso

$$
\rho = \frac{1}{0{,}5}\cdot\frac{1}{0{,}5}\cdot\frac{1}{0{,}5} = 8 .
$$

Per $\pi$ quella traiettoria è otto volte più probabile che per $b$, e quindi
conta otto volte tanto. Se invece a un certo punto $b$ ha scelto un'azione che
$\pi$ non sceglierebbe mai, il fattore diventa $0$ e l'intera partita, da
quell'istante in poi, esce dal conto. Si vede subito anche il difetto: bastano
poche mosse perché i pesi diventino minuscoli o enormi, ed è il motivo per cui
l'off-policy su traiettorie lunghe è fragile.

```{admonition} Dove ritorna
:class: seealso
Il rapporto $\rho$ non resta in questa sezione.

- Nel **PPO** è il rapporto $\rho_t = \pi_\theta(a_t\mid s_t) /
  \pi_{\theta_\text{old}}(a_t\mid s_t)$ fra la policy nuova e quella che ha
  raccolto i dati: lo stesso oggetto, troncato a un passo. Il *clipping* di PPO
  è, letteralmente, un tetto messo a quel peso perché non esploda.
- Nell'**offline RL** l'archivio è tutto ciò che c'è, la policy di
  comportamento non si può interrogare oltre, e il problema della copertura
  diventa il problema centrale del capitolo.
- Nell'**RLHF** il modello che si sta ottimizzando si allontana passo dopo
  passo da quello che ha prodotto le risposte giudicate dagli umani: è la
  stessa deriva, tenuta a bada dallo stesso rapporto (più una penalità che
  misura quanto ci si è allontanati dal modello di partenza).
```

## Il ponte verso le differenze temporali

Restano due difetti, e sono quelli che la prossima sezione viene a risolvere.

Il primo è che bisogna **arrivare alla fine**. Un metodo Monte Carlo non
aggiorna niente finché l'episodio non termina, il che lo esclude dai compiti
continui (un impianto che non si spegne mai, un agente che non muore) e lo
rende lento quando gli episodi sono lunghi.

Il secondo è la **varianza**. Il ritorno di una singola partita è la somma di
molte ricompense, ognuna con la sua dose di caso: come stima è corretta in
media ma ballerina, e servono molti episodi per stabilizzarla.

L'idea che li risolve entrambi è di una semplicità irritante: invece di
aspettare il ritorno vero, usare la ricompensa del prossimo passo più la
**stima già disponibile** dello stato in cui si finisce. Si aggiorna subito, e
si sostituisce una somma rumorosa di molti termini con un termine osservato e
una stima. Si guadagna in varianza, si perde in correttezza (la stima usata
come bersaglio può essere sbagliata: è la distorsione del *bootstrapping*), e
nasce il temporal-difference learning.

Le tre famiglie si dispongono allora su due assi, ed è la mappa da tenere a
mente per tutto il resto del capitolo:

| | quanto guarda avanti | serve il modello? | bootstrapping |
|:--|:--|:--|:--|
| Programmazione dinamica | un passo, su **tutti** gli stati d'arrivo | sì | sì |
| Monte Carlo | **fino alla fine**, su una traiettoria sola | no | no |
| Differenze temporali | un passo, su una traiettoria sola | no | sì |

Manca una casella, quella in mezzo fra un passo e tutta la partita, e non è
vuota: i metodi a $n$ passi e le tracce di eleggibilità la riempiono con
continuità, dal TD puro al Monte Carlo puro, regolando una sola manopola. Ne
diremo alla fine della prossima sezione.

```{admonition} Da ricordare
:class: important
- Un metodo **Monte Carlo** stima il valore di uno stato come **media dei
  ritorni** osservati partendo da lì: nessun modello dell'ambiente, solo
  episodi giocati fino in fondo.
- **A prima visita** conta un ritorno per episodio ed è non distorto; **a ogni
  visita** li conta tutti. Entrambi convergono, con errore che cala come
  $1/\sqrt{n}$.
- Non c'è **bootstrapping**: il bersaglio è il ritorno vero, quindi le stime
  non si contaminano fra loro, ma hanno varianza alta e arrivano solo a
  episodio finito.
- Per **migliorare** una policy, e non solo misurarla, serve esplorazione:
  inizi esplorativi (teorici) o policy $\varepsilon$-soft (pratiche).
- L'**importance sampling** permette di valutare una policy $\pi$ con dati
  generati da un'altra policy $b$, pesando le traiettorie con
  $\rho = \prod \pi(a_k\mid s_k)/b(a_k\mid s_k)$. Le probabilità di transizione
  si cancellano, quindi non serve il modello. Serve la **copertura**.
- La variante **pesata** dell'importance sampling è distorta ma molto più
  stabile di quella ordinaria, e in pratica si preferisce.
```
