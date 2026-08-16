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
si è fermata. Il metodo di là, quello che riscriveva i numeri delle caselle
finché non si assestavano (l’*iterazione dei valori*, in inglese *value
iteration*), sa calcolare i valori ma pretende la mappa: per ogni mossa, dove
si finisce e quanto si incassa. Se la mappa non c'è, resta una via che non
chiede nulla a nessuno: far vivere all'agente molte partite intere, guardare
come sono andate, e fare la media.

## Giocare, e poi fare la media

Il **ritorno** è quello definito nella sezione precedente: quanto si raccoglie
in tutto da un certo istante fino alla fine della partita, contando meno ciò
che arriva tardi. Il **valore** di una situazione è il ritorno *medio* partendo
da lì, e una media si stima nel modo più ovvio che ci sia: si prendono tanti
casi e si fa la loro media. Qui i casi sono le partite giocate.

`````{tab} Elementare

Vuoi sapere quanto vale, negli scacchi, una certa posizione. Un modo c'è, e
non richiede di capire niente di scacchi: da quella posizione gioca mille
partite fino allo scacco matto, e segnati com'è finita ogni volta, un punto se
hai vinto e zero se hai perso. La media di quei mille numeri è la percentuale
di vittorie, e se è alta la posizione è buona.

I metodi **Monte Carlo** fanno questo, e la parola difficile non nasconde
niente di più. L'agente gioca una partita dall'inizio alla fine, poi torna
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
ritorni mediati. Ogni stima è **non distorta**.

La variante a ogni visita è invece distorta per $n$ finito, e conviene dire da
dove viene la distorsione, perché la spiegazione naturale è sbagliata: **non**
dal fatto che i ritorni di uno stesso episodio siano correlati (una media di
variabili correlate, in numero fissato e con la stessa media marginale, resta
non distorta: la correlazione muove la varianza, non il valore atteso). Viene
dal fatto che il **denominatore è aleatorio**: il numero di visite non è deciso
in anticipo, ed è correlato con il numeratore, perché un episodio che passa
molte volte per lo stesso stato contribuisce molte righe, e quelle righe non
sono un campione qualunque dei ritorni possibili. È il classico
stimatore-rapporto, dove l'attesa del rapporto non è il rapporto delle attese;
di quanto e in che verso sbagli dipende dal problema, e il conto di poco più
avanti, dove la variante a ogni visita dà un numero più alto dell'altra, ne è un
esempio e non una regola. La distorsione svanisce al crescere degli episodi, e
la variante si estende meglio all'approssimazione di funzione
{cite}`sutton2018reinforcement`.

Il punto strutturale: qui **non c'è bootstrapping**. Il bersaglio è il ritorno
osservato, non una stima costruita a partire da altre stime. Ogni stato si
stima per conto proprio, e la stima di uno stato non dipende dalla stima degli
altri.

`````

## Che cosa cambia rispetto alla programmazione dinamica

**Programmazione dinamica** è il nome che Bellman diede al modo di procedere
della sezione precedente (con lo scrivere programmi per il computer non c'entra
niente: «programmazione», per lui, voleva dire pianificazione), quello che trova i valori girando e rigirando su tutte
le caselle con la mappa in mano: da qui in avanti lo useremo come nome
collettivo delle sue due ricette, l'iterazione dei valori e quella della
pagella (*value iteration* e *policy iteration*). Vale la pena metterlo accanto
a Monte Carlo, perché la differenza fra i due non è di efficienza ma di **che
cosa serve sapere**.

La programmazione dinamica guarda **un passo in avanti ma in tutte le
direzioni**: per calcolare il valore di una casella tiene conto di tutte le
caselle in cui quella mossa potrebbe far finire, dando a ciascuna un peso pari
alla sua probabilità.

«Dare un peso» è il gesto che torna in tutto il resto del capitolo, e conviene
vederlo una volta su due numeri. È la media che si fa a scuola quando una prova
conta il doppio di un'altra: invece di sommare e dividere per quanti sono, si
moltiplica ogni numero per quanto conta e poi si somma. Se una mossa porta
nella casella A otto volte su dieci e nella casella B due volte su dieci, e A
vale $10$ mentre B vale $0$, quella mossa vale
$0{,}8 \times 10 + 0{,}2 \times 0 = 8$: non $5$, che sarebbe la media semplice
fra $10$ e $0$, perché in B ci si finisce di rado.

Alla programmazione dinamica quei pesi servono, e quindi le serve conoscere le
probabilità; in cambio non le deve stimare.

Monte Carlo guarda **in una direzione sola ma fino in fondo**: segue la
traiettoria realmente accaduta, dall'inizio alla fine dell'episodio, e ignora
le strade non prese. Non ha bisogno di sapere nulla dell'ambiente, e in cambio
paga in **rumore**: è la parola che si usa per il fatto che una misura,
ripetuta, non dà mai due volte lo stesso numero.

Da questa differenza discendono tre conseguenze pratiche.

- Monte Carlo funziona anche quando l'ambiente è una **scatola nera** o un
  simulatore: basta saperci giocare, non saperlo descrivere. Scrivere un
  programma che simula un gioco è spesso molto più facile che compilare
  l'elenco, mossa per mossa e con tutte le probabilità, di dove quel gioco può
  portare.
- Il costo di stimare un singolo stato **non dipende dal numero di stati**. Se
  interessa il valore di una manciata di posizioni, si giocano partite da
  quelle e basta, senza passare in rassegna tutte le altre come fa la
  programmazione dinamica.
- Gli errori **non si propagano**. Il voto di una casella esce solo da quello
  che è successo davvero nelle partite, e nessun'altra casella se ne serve per
  calcolare il proprio: un voto sbagliato resta sbagliato dov'è, e non contagia
  i vicini.

## Tre partite, coi numeri

Riprendiamo il mondo a tre caselle della {numref}`fig-mdp`, quello in cui
salire non costa nulla, restare fermi o tornare indietro costano $1$, e
arrivare all'obiettivo paga $10$. Vale lo stesso sconto di prima, $0{,}9$: un
premio che arriva una mossa più tardi conta nove decimi. Stavolta però fingiamo
di **non** conoscere dove porta ogni mossa. L'agente si limita a giocare,
seguendo una strategia che di norma sale verso l'obiettivo ma ogni tanto
tentenna, ed ecco tre sue partite, con le ricompense incassate lungo la strada.

1. $s_0 \xrightarrow{\,0\,} s_1 \xrightarrow{\,+10\,} s_2$
2. $s_0 \xrightarrow{\,-1\,} s_0 \xrightarrow{\,0\,} s_1 \xrightarrow{\,+10\,} s_2$
3. $s_0 \xrightarrow{\,0\,} s_1 \xrightarrow{\,-1\,} s_0 \xrightarrow{\,0\,} s_1 \xrightarrow{\,+10\,} s_2$

Ogni riga si legge da sinistra a destra, e il numero sopra la freccia è quello
che si incassa facendo quel passo, col suo segno: $+10$ vuol dire dieci punti
guadagnati, $-1$ un punto perso. Nella prima partita, da $s_0$ si sale a $s_1$
senza incassare nulla, e da $s_1$ si arriva all'obiettivo incassando $+10$.

`````{tab} Elementare

I punti raccolti si contano **all'indietro**, ed è il modo comodo di farlo: si
parte dalla fine, dove il totale è $0$ perché dopo l'arrivo non c'è più niente
da raccogliere, e a ogni passo indietro si moltiplica per $0{,}9$ il totale che
si aveva e ci si aggiunge la ricompensa di quel passo.

La prima partita è la più corta e serve a scaldarsi. Da $s_1$ si arriva
all'obiettivo incassando $10$, e l'obiettivo vale $0$: quindi da $s_1$ in avanti
si sono raccolti $10 + 0{,}9 \times 0 = 10$. Da $s_0$, un passo prima, non si
incassa nulla e si finisce in un posto che vale $10$: quindi
$0 + 0{,}9 \times 10 = 9$.

Prendiamo adesso la seconda partita. Dall'ultimo $s_1$ mancava solo il premio, quindi
$10$. Dal $s_0$ che veniva subito prima: quel passo non paga nulla e porta in un
posto che vale $10$, quindi $0 + 0{,}9 \times 10 = 9$. Dal primo $s_0$: quel
passo costa $1$ e porta in un posto che vale $9$, quindi
$-1 + 0{,}9 \times 9 = 7{,}1$.

La terza partita è più lunga e si fa allo stesso modo, sempre partendo dalla
fine: l'ultimo $s_1$ vale $10$; il $s_0$ prima di lui $0 + 0{,}9 \times 10 = 9$;
il $s_1$ prima ancora costa $1$ e porta in un posto che vale $9$, quindi
$-1 + 0{,}9 \times 9 = 7{,}1$; e il primo $s_0$ di tutti non paga nulla e porta
in un posto che vale $7{,}1$, quindi $0 + 0{,}9 \times 7{,}1 = 6{,}39$.

| partita | raccolto da $s_0$ in avanti | raccolto da $s_1$ in avanti |
|:--|:--|:--|
| 1 | $9$ | $10$ |
| 2 | $7{,}1$ la prima volta, $9$ la seconda | $10$ |
| 3 | $6{,}39$ la prima volta, $9$ la seconda | $7{,}1$ la prima volta, $10$ la seconda |

Adesso la media, e ci sono due modi di farla. Il primo conta **una riga per
partita**, quella della prima volta che si è passati di lì, e si chiama *a prima
visita*: per $s_0$ si mediano $9$, $7{,}1$ e $6{,}39$, cioè
$22{,}49 : 3 = 7{,}4966\ldots$; per $s_1$ si mediano $10$, $10$ e $7{,}1$, cioè
$27{,}1 : 3 = 9{,}0333\ldots$. Il secondo conta **tutte le righe**, ripassaggi
compresi, e si chiama *a ogni visita*: per $s_0$ i numeri diventano cinque
($9 + 7{,}1 + 9 + 6{,}39 + 9 = 40{,}49$, diviso $5$ fa $8{,}098$) e per $s_1$
quattro ($10 + 10 + 7{,}1 + 10 = 37{,}1$, diviso $4$ fa $9{,}275$). Da qui in
avanti li scriveremo arrotondati alla seconda cifra, $7{,}50$ e $9{,}03$ da una
parte, $8{,}10$ e $9{,}28$ dall'altra, ma i numeri veri sono questi, e sono
quelli che stampa il codice più sotto.

`````

`````{tab} Superiore

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
V(s_0) = \frac{9 + 7{,}1 + 6{,}39}{3} = \frac{22{,}49}{3} \simeq 7{,}50,
\qquad
V(s_1) = \frac{10 + 10 + 7{,}1}{3} = \frac{27{,}1}{3} \simeq 9{,}03 .
$$

**A ogni visita** entrano tutte le righe, cinque per $s_0$ e quattro per $s_1$:

$$
V(s_0) = \frac{9 + 7{,}1 + 9 + 6{,}39 + 9}{5} = 8{,}098,
\qquad
V(s_1) = \frac{10 + 10 + 7{,}1 + 10}{4} = 9{,}275 .
$$

`````

Per ogni casella sono usciti due numeri diversi dagli stessi identici dati, ed
entrambi sono legittimi: sono due modi di stimare la stessa cosa, e nella pratica si usano tutti e due. Quello a prima
visita è il più facile da giustificare: ogni partita porta un numero solo, e
numeri che vengono da partite diverse non si influenzano a vicenda, che è
esattamente la condizione in cui fare una media è la cosa giusta da fare.
Quello a ogni visita conta anche i ripassaggi, quindi butta via meno dati, e
per questo regge meglio quando le partite sono poche e il quaderno resta quasi
vuoto: è la situazione del capitolo successivo, dove il quaderno viene
sostituito da una rete neurale. Con tante partite la scelta non cambia il
risultato, perché tutti e due finiscono sul valore vero.

Tutti e quattro i numeri, però, restano sotto quelli che la sezione precedente
aveva calcolato sullo stesso mondo, che erano $9$ per $s_0$ e $10$ per $s_1$, e
la ragione conviene fissarla:
là si calcolava il valore della strategia **migliore possibile**, qui si misura
quello della strategia **che ha giocato davvero**, tentennamenti compresi. Sono
due domande diverse, e la seconda non può avere una risposta più alta della
prima. Con tre partite sole, per di più: una media si assesta sul valore vero
quando i casi mediati sono tanti, e tre non lo sono.

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

Misurare quanto vale una strategia è metà del lavoro; l'altra metà si chiama
**controllo**, ed è trovarne una migliore. Si riusa lo schema della pagella
della sezione precedente: si misura, poi in ogni situazione si tiene la mossa
che secondo quelle misure rende di più (si dice che la strategia si rende
*greedy*, cioè avida: prende sempre quello che al momento sembra il meglio), e
si ricomincia da capo con la strategia nuova. Con una differenza che sembra un
dettaglio tecnico e invece è il tema di tutto il capitolo.

`````{tab} Elementare

C'è una trappola. Se l'agente, dopo aver imparato che una certa mossa è buona,
la gioca sempre, le altre mosse non le prova più. E se non le prova più, non
scoprirà mai che una di quelle era migliore: il suo voto resterà per sempre
quello sbagliato del primo tentativo. Il quaderno delle medie ha una colonna
che non si aggiorna più.

Per questo un agente Monte Carlo che vuole *migliorare* (e non solo misurare)
deve continuare a fare mosse che non crede ottime. È lo stesso dilemma fra
esplorare e sfruttare che la sezione sulle leve aveva isolato in apertura di
capitolo, e qui si presenta nella forma più cruda: senza esplorazione, il
metodo semplicemente non vede i dati che gli servirebbero.

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
$\varepsilon$-greedy già vista sui bandit è il caso tipico. Il *policy
improvement theorem* continua a valere ristretto a questa classe, quindi
l'alternanza valuta-migliora converge, ma converge alla migliore policy
$\varepsilon$-soft, non alla migliore in assoluto {cite}`sutton2018reinforcement`.

La rinuncia è reale, e la via d'uscita è il paragrafo seguente: separare la
policy che **genera** i dati da quella che si sta **valutando**.

`````

## Imparare da una policy e giudicarne un'altra

Il concetto di questo paragrafo è fra i più riusati del libro: più avanti
ricompare tre volte, e da lì in poi si dà per saputo. Conviene quindi
prendercisi il tempo adesso.

Ci sono due strategie in gioco: quella che vogliamo giudicare e quella che ha
davvero giocato le partite che abbiamo in mano. Se sono la stessa, cioè se si
impara giocando in proprio, si dice che si sta lavorando **on-policy**, ed è il
caso di tutto quello che abbiamo visto finora. Se sono diverse si è
**off-policy**, "fuori dalla propria strategia", ed è la situazione
interessante: imparare da un archivio di partite giocate da altri, da un
programma di controllo che c'era già, da un esperto umano, oppure da una
versione precedente di sé stessi.

`````{tab} Elementare

Il problema è che i dati raccontano la storia sbagliata. Se il giocatore che
ha lasciato le partite era prudente e la strategia che vuoi giudicare è
audace, le partite audaci nell'archivio sono poche, e mediarle tutte allo
stesso modo darebbe un giudizio sulla prudenza, non sull'audacia.

Il rimedio è **pesare** le partite invece di contarle tutte uguali (in inglese
si chiama *importance sampling*, e il nome vuol dire proprio questo: campionare
tenendo conto di quanto ogni caso conta). Una partita che la strategia audace
avrebbe giocato spesso e che il prudente ha giocato di rado vale molto, perché
è rara e informativa; una partita tipica del prudente e che l'audace non
farebbe mai vale poco o niente. Il peso è semplicemente il
rapporto fra quanto era probabile quella sequenza di mosse per l'una e per
l'altra.

Una condizione però serve, ed è di buon senso: l'archivio deve **contenere**
tutto ciò che la strategia da giudicare potrebbe fare. Se l'audace giocherebbe
una mossa che il prudente non ha mai provato nemmeno una volta, di quella mossa
non si può dire nulla, e nessun peso può inventare i dati mancanti.

Quanto pesano davvero quei pesi si vede con un conto piccolo. Poniamo che il
prudente scelga fra due mosse tirando una monetina, e che l'audace sappia
sempre quale vuole. Prendiamo una partita di tre mosse in cui la monetina ha
indovinato per caso tutte e tre le volte la mossa dell'audace. L'audace quella
partita l'avrebbe giocata **sempre**, cioè una volta su una; il prudente ci è
arrivato per fortuna, e la sua fortuna vale una volta su due a ogni mossa, cioè
$\frac12 \times \frac12 \times \frac12 = \frac18$, una volta su otto. Il peso è
il rapporto fra le due: $1$ diviso $\frac18$ fa $8$, e quindi quella partita
conta **otto volte tanto**. Se
invece a un certo punto la monetina ha scelto una mossa che l'audace non
farebbe mai, da lì in avanti quella partita non dice più niente sull'audace, e
il suo peso va a zero. Ecco il difetto, in due righe: bastano poche mosse
perché i pesi diventino minuscoli o enormi, ed è il motivo per cui giudicare le
partite di un altro funziona bene su partite corte e traballa su quelle lunghe.

`````

`````{tab} Superiore

Nelle formule la policy da valutare si chiama $\pi$ (*target*) e quella che ha
generato i dati $b$ (*behavior*, di comportamento).

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

Poiché $\mathbb{E}_b\big[\rho_{t:T-1}\,G_t \mid S_t = s\big] = V^\pi(s)$ (il
pedice non è pignoleria: l'attesa è sulle traiettorie generate da $b$, ed è
tutto il punto), si può stimare in due modi. L’**importance sampling
ordinario** fa la media semplice dei ritorni pesati; quello **pesato**
normalizza per la somma dei pesi:

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
dei campioni) ma il peso di un singolo ritorno non supera mai $1$ e, purché i
ritorni siano limitati, la sua varianza converge a zero anche quando quella
dei rapporti è infinita: un risultato del 2001 di Precup, Sutton e Dasgupta.
In pratica si preferisce quasi sempre il pesato
{cite}`sutton2018reinforcement`.

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

`````

Questo modo di pesare le partite di un altro non resta in questa sezione: è uno
degli attrezzi che il libro riusa di più, e conviene sapere dove ricomparirà.

```{admonition} Dove ritorna
:class: seealso
- Nel **PPO** (*Proximal Policy Optimization*), uno degli algoritmi più usati
  del capitolo successivo, il peso è il rapporto fra quanto la strategia nuova
  e quella che ha raccolto i dati avrebbero giocato la stessa mossa: lo stesso
  oggetto di qui, calcolato su una mossa
  sola invece che su tutta la partita. E siccome un peso che esplode è il
  difetto appena visto, PPO gli mette attorno una fascia, sopra e sotto, oltre
  la quale il peso viene tosato (*clipping*).
- Nell’**offline RL**, cioè imparare da un archivio di partite senza poterne
  giocare altre, quell'archivio è tutto ciò che c'è: la condizione appena vista
  (deve contenere tutto quello che la strategia da giudicare potrebbe fare)
  diventa il problema centrale della sezione che gli è dedicata.
- Nell’**RLHF** (*Reinforcement Learning from Human Feedback*), il modo in cui
  gli assistenti conversazionali imparano dai giudizi delle persone su quale di
  due risposte sia migliore, il programma che si sta migliorando si allontana
  passo dopo passo da quello che aveva prodotto le risposte giudicate: è la
  stessa deriva fra chi ha giocato e chi si giudica, tenuta a bada dallo stesso
  rapporto e da una regola in più: al programma che si sta migliorando vengono
  tolti punti quanto più le sue risposte si allontanano da quelle del programma
  di partenza, così che non possa cambiare troppo in fretta.
```

## Il ponte verso le differenze temporali

Restano due difetti, e sono quelli che la prossima sezione viene a risolvere.

Il primo è che bisogna **arrivare alla fine**. Un metodo Monte Carlo non
aggiorna niente finché l'episodio non termina, il che lo esclude dai compiti
continui (un impianto che non si spegne mai, un agente che non muore) e lo
rende lento quando gli episodi sono lunghi.

Il secondo è che i numeri **ballano**. Il ritorno di una singola partita è la
somma di molte ricompense, ognuna con la sua dose di caso: in media è giusto, ma
preso una volta sola può capitare lontanissimo dal vero, e servono molti episodi
perché la media si assesti. Quanto ballano lo misura la **varianza**, cioè
quanto i valori si sparpagliano attorno alla loro media.

L'idea che li risolve entrambi è di una semplicità irritante: invece di
aspettare il ritorno vero, usare la ricompensa del prossimo passo più la
**stima già disponibile** della situazione in cui si finisce. Si aggiorna
subito, e si sostituisce una somma rumorosa di molti termini con un termine
osservato e una stima sola. Usare una propria stima per aggiornarne un'altra ha
un nome, **bootstrapping** (alla lettera "tirarsi su per i lacci delle
scarpe"), e ha un costo: la stima presa come bersaglio può essere sbagliata, e
allora la correzione tira nella direzione sbagliata. E non tira a caso, tira
sempre dalla stessa parte, almeno finché le stime non si assestano: nel
labirinto tutte le caselle partono da zero, che è meno del loro valore vero,
quindi ogni bersaglio costruito su di esse è più basso del vero, e ogni
correzione tira verso il basso. Un errore che ha un verso non si cancella
facendo la media di tante correzioni, e per questo ha un nome suo, la
**distorsione**; è il prezzo di
non aspettare la fine: numeri molto più stabili, appoggiati però a un bersaglio
che potrebbe non essere quello giusto. Nasce così l'apprendimento per
**differenze temporali**, in inglese *temporal-difference*, che tutti abbreviano
in **TD**.

Le tre famiglie si dispongono allora su due assi, ed è la mappa da tenere a
mente per tutto il resto del capitolo:

| | quanto guarda avanti | serve la mappa dell'ambiente? | si corregge appoggiandosi alle proprie stime (*bootstrapping*) |
|:--|:--|:--|:--|
| Programmazione dinamica | un passo, su **tutte** le caselle in cui si può finire | sì | sì |
| Monte Carlo | **fino alla fine**, su una partita sola | no | no |
| Differenze temporali | un passo, su una partita sola | no | sì |

In quella tabella manca una riga, ed è quella in mezzo: guardare avanti non un
passo solo e nemmeno fino alla fine, ma due passi, o tre, o dieci. Non è una
possibilità teorica, è una manopola vera, che va con continuità dalle
differenze temporali pure al Monte Carlo puro; la fine della prossima sezione è
dedicata a lei.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un metodo **Monte Carlo** stima quanto vale una situazione nel modo più
  diretto che ci sia: si giocano molte partite intere, e per ogni situazione
  attraversata ci si segna sul quaderno quanto si è raccolto **da lì in
  avanti**. Il valore è la media di quelle righe. Nessuna mappa dell'ambiente,
  solo partite giocate fino in fondo.
- Rispetto al metodo della sezione precedente cambia **che cosa bisogna
  sapere**: quello guarda un passo avanti in tutte le direzioni possibili e
  pretende la mappa dell'ambiente, Monte Carlo guarda in una direzione sola ma
  fino in fondo e non pretende niente. Basta saper giocare, non saper
  descrivere il gioco. E se interessano poche situazioni, si giocano partite
  solo da quelle, senza passare in rassegna tutte le altre.
- Le medie si possono fare in due modi, contando una riga per partita (la prima
  volta che si è passati di lì) oppure contandole tutte, ripassaggi compresi.
  Danno numeri un po’ diversi, sono tutti e due legittimi, e con tante partite
  finiscono nello stesso posto.
- Ogni situazione si giudica per conto suo, su quello che è successo davvero:
  un voto sbagliato non contagia le caselle vicine, perché nessuno lo usa per
  calcolare il proprio. Il prezzo sono i due difetti dichiarati fin dall'inizio:
  i numeri ballano parecchio (una partita sola è una somma di tanti colpi di
  fortuna) e non si scrive niente finché la partita non è finita.
- Per **migliorare** la strategia, e non solo misurarla, l'agente deve
  continuare a giocare mosse che non crede le migliori: se non le prova più,
  quella colonna del quaderno resta per sempre al voto sbagliato del primo
  tentativo.
- Si può giudicare una strategia con partite giocate da un'altra, purché le si
  **pesi** invece di contarle tutte uguali: una partita che la strategia da
  giudicare avrebbe giocato spesso e l'altra di rado conta molto, una che la
  prima non farebbe mai non conta niente. Il peso è solo il rapporto fra quanto
  erano probabili quelle mosse per l'una e per l'altra, e per questo non serve
  sapere nulla dell'ambiente. Serve però che l'archivio contenga tutto ciò che
  la strategia da giudicare potrebbe fare.
- I pesi però sono fragili: bastano poche mosse perché diventino minuscoli o
  enormi (tre mosse tirate a sorte e indovinate pesano già otto volte tanto, e
  una sola mossa che la strategia da giudicare non farebbe mai manda a zero
  tutto il resto della partita). Giudicare le partite di un altro funziona
  bene sulle partite corte, e diventa traballante su quelle lunghe.
```

`````

`````{tab} Superiore

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
- L’**importance sampling** permette di valutare una policy $\pi$ con dati
  generati da un'altra policy $b$, pesando le traiettorie con
  $\rho = \prod \pi(a_k\mid s_k)/b(a_k\mid s_k)$. Le probabilità di transizione
  si cancellano, quindi non serve il modello. Serve la **copertura**.
- La variante **pesata** dell'importance sampling è distorta ma molto più
  stabile di quella ordinaria, e in pratica si preferisce.
```

`````
