# Metodi a gradiente di policy

Seul, marzo 2016. Alla trentasettesima mossa della seconda partita contro il
campione Lee Sedol, il programma AlphaGo appoggia una pietra in un punto che
nessun giocatore professionista avrebbe scelto: i commentatori pensano a un
errore. Era invece una mossa che, secondo le stime del programma stesso, un
umano avrebbe giocato circa una volta su diecimila. Lee Sedol si alza dal
tavolo per un quarto d'ora. Quella mossa non era stata copiata da nessun
archivio di partite: era il frutto di una *strategia* appresa giocando milioni
di volte contro se stesso.

Come si insegna a una macchina una strategia? Nei metodi basati sul valore,
che abbiamo incontrato con il Q-learning, impariamo a stimare *quanto vale*
uno stato o un'azione, la funzione $V(s)$ e la funzione $Q(s,a)$, e ne
ricaviamo l'azione migliore scegliendo di volta in volta quella con valore più
alto. I metodi a **gradiente di policy** ribaltano la prospettiva: invece di
valutare e poi decidere, imparano *direttamente a decidere*.

## Imparare la policy, non il valore

`````{tab} Elementare

Immagina di allenare un cane. Un approccio è compilare una tabella mentale che
assegna a ogni situazione un "punteggio di bontà" per ciascun comportamento
possibile, e poi far scegliere al cane il comportamento col punteggio più alto.
L'altro approccio, più diretto, è modellare la *tendenza* del cane: rendere più
probabili i comportamenti che in passato hanno fruttato un premio, meno
probabili quelli finiti male. Non calcoliamo un punteggio per poi decidere:
regoliamo direttamente le probabilità con cui l'animale sceglie.

Una **policy** è esattamente questo: una funzione che, data una situazione,
dice con quale probabilità compiere ciascuna azione.

`````

`````{tab} Superiore

Una policy parametrica $\pi_\theta(a \mid s)$ è una distribuzione di
probabilità sulle azioni $a$ condizionata allo stato $s$, controllata dai
parametri $\theta$ (i pesi di una rete neurale). L'obiettivo è massimizzare il
**ritorno atteso**:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[ R(\tau) \big],
\qquad R(\tau) = \sum_{t=0}^{T} \gamma^{t}\, r_t ,
$$

dove $\tau=(s_0,a_0,r_0,s_1,\dots)$ è una *traiettoria* generata seguendo la
policy, $r_t$ è la ricompensa al passo $t$ e $\gamma\in[0,1)$ è il fattore di
sconto, che pesa meno il futuro lontano. Rispetto ai metodi basati sul valore,
ottimizzare $\pi_\theta$ direttamente gestisce con naturalezza gli spazi di
azioni continui e le policy stocastiche.

`````

## REINFORCE: premiare ciò che ha funzionato

Il primo algoritmo di questa famiglia, dovuto a Ronald Williams
{cite}`williams1992simple`, ha
un'idea tanto semplice da sembrare ingenua: gioca un'intera partita, guarda
com'è andata, e poi *aumenta la probabilità delle azioni che hanno portato a
ricompense alte*, diminuendo quella delle azioni seguite da ricompense basse.

`````{tab} Elementare

È il metodo "prova e ricorda". Il giocatore fa una partita intera, dall'inizio
alla fine. Se ha vinto, si dice: "qualunque cosa io abbia fatto, rendila più
probabile la prossima volta". Se ha perso, il contrario. Ripetuto migliaia di
volte, questo semplice riflesso spinge il comportamento verso le mosse buone
senza che nessuno debba mai spiegare *perché* siano buone.

`````

`````{tab} Superiore

REINFORCE stima il gradiente di $J(\theta)$ tramite il *policy gradient
theorem* {cite}`sutton2000policy`:

$$
\nabla_\theta J(\theta) =
\mathbb{E}\Big[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t \Big],
$$

dove $G_t=\sum_{k\ge t}\gamma^{\,k-t} r_k$ è il ritorno osservato a partire dal
passo $t$. Il termine $\nabla_\theta \log \pi_\theta(a_t\mid s_t)$ indica *come*
ritoccare i parametri per rendere più probabile l'azione $a_t$; moltiplicandolo
per $G_t$, quel ritocco viene amplificato quando l'esito è stato buono e
invertito quando è stato cattivo. L'aggiornamento è quindi

$$
\theta \leftarrow \theta + \alpha\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t ,
$$

con $\alpha$ il passo di apprendimento. Un'avvertenza di rigore: il gradiente
esatto dell'obiettivo scontato conterrebbe un fattore $\gamma^{\,t}$ davanti a
ciascun addendo della somma; la prassi, che seguiamo qui, lo omette, ottenendo
una direzione leggermente distorta rispetto a $\nabla_\theta J(\theta)$ ma che
non soffoca il segnale dei passi lontani nel tempo. Chi ha letto la sezione
sui bandit
riconosce la struttura: il *bandit a gradiente* era esattamente questo, in un
mondo con un solo stato, dove la softmax sulle preferenze $H(a)$ faceva le
veci di $\pi_\theta(a\mid s)$. Anche il rimedio che segue è già comparso là.

Il punto debole è la **varianza**:
$G_t$ dipende dall'intero seguito casuale della partita, e le stime risultano
rumorose e lente a convergere.

`````

## Actor-Critic: chi agisce e chi giudica

Come si abbatte quella varianza? L'idea è affiancare al giocatore un giudice
che commenta le mosse una per una, senza aspettare la fine
({numref}`fig-actor-critic`).

```{figure} ../figures/actor-critic.svg
:name: fig-actor-critic
:alt: Il riquadro Agente contiene Attore e Critico; il Critico passa il vantaggio all'Attore, che invia un'azione all'Ambiente, il quale restituisce stato successivo e ricompensa.
:width: 85%

L'architettura Actor-Critic. L'attore decide l'azione, il critico la valuta e
gli restituisce un segnale (il *vantaggio*); l'ambiente risponde con lo stato
successivo e la ricompensa.
```

`````{tab} Elementare

Immagina due ruoli. L'**attore** è chi gioca: decide le mosse. Il **critico**
è un allenatore a bordo campo che, mossa dopo mossa, mormora "meglio del
previsto" oppure "peggio del previsto". L'attore non deve più aspettare la fine
della partita per sapere com'è andata: riceve un giudizio immediato a ogni
passo e corregge subito la rotta. Impara più in fretta e in modo più stabile.

`````

`````{tab} Superiore

Nell'architettura **Actor-Critic** convivono due reti. L'*attore* è la policy
$\pi_\theta(a\mid s)$; il *critico* stima la funzione valore $V_\phi(s)$. Al
ritorno grezzo $G_t$ si sostituisce il **vantaggio** (*advantage*), di solito
l'errore di differenza temporale:

$$
A_t = r_t + \gamma\, V_\phi(s_{t+1}) - V_\phi(s_t) .
$$

$A_t$ misura di quanto l'azione compiuta ha superato le *aspettative*
codificate dal critico: è positivo se l'esito è stato migliore del previsto. La
regola diventa $\nabla_\theta \log\pi_\theta(a_t\mid s_t)\,A_t$, e la varianza
scende per due vie che conviene distinguere. Sottrarre la *baseline*
$V_\phi(s_t)$ non distorce il gradiente; sostituire il ritorno $G_t$ con
$r_t+\gamma V_\phi(s_{t+1})$ è invece *bootstrapping*, e rende la stima
distorta finché il critico è impreciso. Si scambia varianza con *bias*: è il
compromesso al cuore dell'actor-critic. Attore e critico si addestrano
insieme: il critico affina le sue stime, l'attore le usa come segnale.

`````

## A3C e PPO: gli algoritmi che funzionano davvero

Su queste fondamenta poggiano gli algoritmi usati in pratica. **A3C**
(*Asynchronous Advantage Actor-Critic* {cite}`mnih2016asynchronous`) fa
girare molti
attori in parallelo su copie dell'ambiente: le loro esperienze scorrelate
rendono l'addestramento più stabile e sfruttano le CPU multi-core. **PPO**
(*Proximal Policy Optimization* {cite}`schulman2017proximal`) è oggi lo
standard di fatto, per robustezza e semplicità.

```{figure} ../figures/ppo-2017.svg
:name: fig-ppo-clipping
:alt: "Dalla policy vecchia, al centro, si dipartono due frecce. La prima è un passo breve che resta dentro una fascia consentita disegnata attorno al punto di partenza, e viene accettata. La seconda è un salto lungo che esce dalla fascia: l'aggiornamento viene tagliato al bordo, e oltre quel bordo non porta più alcun vantaggio."
:width: 84%

Il guinzaglio di PPO. Non impedisce di migliorare: toglie il premio a chi
prova a migliorare troppo in una volta, e così l'aggiornamento resta vicino
alla policy che ha generato i dati.
```

Il taglio disegnato in {numref}`fig-ppo-clipping` esiste perché i dati
invecchiano in fretta. Le esperienze raccolte sono state generate dalla policy
vecchia, e valgono a stimare il gradiente solo finché la nuova le somiglia; un
passo troppo lungo userebbe dati che non descrivono più il comportamento
dell'agente.

`````{tab} Elementare

Il rischio, quando aggiorni una strategia, è esagerare: un solo passo troppo
lungo può rovinare l'apprendimento di ore intere. PPO fa esattamente ciò che
suggerisce il nome (*proximal*, "vicino"): cambia la policy solo di poco per
volta, senza mai allontanarsi troppo da quella attuale. Piccoli passi prudenti,
ma tanti.

`````

`````{tab} Superiore

PPO massimizza un obiettivo "tosato" (*clipped*):

$$
L^{\text{CLIP}}(\theta) =
\mathbb{E}\big[\min\!\big(\rho_t A_t,\ \operatorname{clip}(\rho_t,\,1-\epsilon,\,1+\epsilon)\,A_t\big)\big],
$$

dove $\rho_t = \dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$
è il rapporto tra la nuova e la vecchia policy, e $\epsilon$ (tipicamente
$0{,}2$) fissa quanto le è concesso spostarsi. Quel rapporto non è un
espediente inventato qui: è il **rapporto di importance sampling** incontrato
nel capitolo precedente, quello che permette di valutare una policy con dati
generati da un'altra, troncato a un passo solo. E il suo difetto è lo stesso
già visto là: può assumere valori enormi e mandare in aria la stima. Il
*clipping* annulla
l'incentivo a spingere $\rho_t$ fuori dall'intervallo $[1-\epsilon,1+\epsilon]$:
la policy migliora a piccoli passi controllati, senza gli aggiornamenti
distruttivi che affliggevano i primi metodi.

Detto così il tosaggio sembra un trucco, e invece è l'approssimazione
economica di un'idea precisa che lo precede. **TRPO** (*Trust Region Policy
Optimization*) pone il problema come massimizzazione **vincolata**: si
massimizza lo stesso obiettivo con importance sampling, ma imponendo che la
nuova policy resti vicina alla vecchia in **divergenza di Kullback-Leibler**,

$$
\max_\theta\ \mathbb{E}\big[\rho_t A_t\big]
\quad \text{soggetto a} \quad
\mathbb{E}\big[D_{\mathrm{KL}}(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta)\big]
\le \delta .
$$

Il vincolo definisce una **regione di fiducia**, cioè l'intorno entro il quale
l'approssimazione lineare dell'obiettivo è ancora credibile. La ragione per cui
serve è quella che il capitolo ha già raccontato con la metafora del
guinzaglio: una policy non è un modello supervisionato qualunque, perché
determina i dati che raccoglierà, e un passo troppo lungo non produce un errore
recuperabile ma una policy che smette di visitare gli stati utili.

Il prezzo di TRPO è computazionale: risolvere quel vincolo richiede
un'approssimazione del secondo ordine con la matrice di informazione di Fisher,
gestita con gradiente coniugato e ricerca di linea. Funziona, ed è pesante e
scomodo da implementare. PPO osserva che l'effetto che si vuole (non
allontanarsi troppo) si ottiene quasi tutto con un `min` e un `clip` dentro un
normale ottimizzatore del primo ordine. È il motivo per cui ha vinto: **non è
più corretto di TRPO, è abbastanza corretto e infinitamente più semplice**, e
in ingegneria quello è di solito il tipo di vittoria che conta.

`````

## Pensare prima di agire: la ricerca ad albero Monte Carlo

Finora la policy ha sempre risposto d'istinto: stato dentro, azione fuori, un
passaggio nella rete. Ma un giocatore forte, prima di muovere, **pensa**:
prova mentalmente qualche continuazione, valuta dove porta, sceglie. Quel
pensare ha un algoritmo, si chiama **ricerca ad albero Monte Carlo** (MCTS), e
il libro lo nominerà spesso da qui in avanti (AlphaGo, AlphaZero, MuZero, e i
modelli linguistici che ragionano esplorando più strade). Vale la pena vederlo
una volta per bene, anche perché è un vecchio amico travestito.

`````{tab} Elementare

Il problema è che le continuazioni sono troppe. Agli scacchi, dopo tre mosse a
testa, i seguiti sono milioni; nel Go, molti di più. Esaminarle tutte è
impossibile, quindi bisogna guardare a fondo **solo dove conviene**. Ma per
sapere dove conviene bisognerebbe aver già guardato. È lo stesso dilemma dei
bandit a più braccia: sfruttare quello che sembra buono, o esplorare quello di
cui si sa poco?

MCTS lo risolve costruendo un albero delle possibilità **a poco a poco**,
ripetendo migliaia di volte lo stesso giro di quattro mosse:

1. **Selezione.** Si scende dall'inizio seguendo, a ogni bivio, la mossa che
   ha il punteggio migliore, dove «migliore» mette insieme quanto ha reso
   finora e quanto poco è stata provata.
2. **Espansione.** Quando si arriva a un bivio con una mossa mai tentata, si
   aggiunge quel ramo all'albero.
3. **Simulazione.** Da lì si tira dritto fino alla fine della partita, in fretta
   e alla buona (nella versione originale, a caso), solo per farsi un'idea
   grezza di come va a finire.
4. **Risalita.** Il risultato torna indietro lungo la strada percorsa, e ogni
   nodo attraversato aggiorna la propria media e il proprio conteggio.

Il bello è che l'albero cresce **storto, e di proposito**: profondissimo sulle
linee promettenti, largo appena un dito su quelle che non convincono. Nessuno
gli ha detto quali fossero: lo ha scoperto giocandoci.

E la mossa da fare, alla fine, non è quella con la media migliore: è quella
**più visitata**. Sembra strano, ed è più solido: una media alta può venire da
due prove fortunate, mentre un ramo visitato mille volte ha resistito a mille
occasioni di essere abbandonato.

`````

`````{tab} Superiore

La formulazione standard è **UCT** (*Upper Confidence bounds applied to
Trees*), di Kocsis e Szepesvári {cite}`kocsis2006bandit`, costruita sopra il
framework di ricerca di Coulom {cite}`coulom2006efficient`. L'idea è di
trattare **ogni nodo come un bandit indipendente** sulle sue mosse, e in fase
di selezione scegliere

$$
a^\star = \arg\max_a \left[\, Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}}
\,\right],
$$

dove $N(s)$ è il numero di visite al nodo, $N(s,a)$ quelle al figlio, e
$Q(s,a) = W(s,a)/N(s,a)$ la media dei ritorni osservati passando di lì. È
**letteralmente UCB1**, la formula della sezione sui bandit, applicata a ogni
bivio: stesso ottimismo di fronte all'incertezza, stesso decadimento
logaritmico. Il contributo di UCT è mostrare che applicandola ricorsivamente
la stima alla radice converge a quella minimax, con garanzie sull'errore di
campionamento.

La risalita aggiorna $N$ e $W$ lungo il cammino; nei giochi a due giocatori il
ritorno si alterna di segno a ogni livello, perché ciò che è buono per me è
cattivo per l'avversario.

**AlphaGo e AlphaZero cambiano due dei quattro passi**, ed è lì che entrano le
reti. Il termine di esplorazione diventa **PUCT**, pesato da una probabilità a
priori fornita dalla rete di policy,

$$
U(s,a) = c_{\text{puct}}\, P(s,a)\,
\frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)},
$$

così che la ricerca guardi per prime le mosse che la rete considera plausibili
invece di trattarle tutte alla pari. E la **simulazione casuale sparisce**:
al suo posto la rete di valore $v_\theta(s)$ stima direttamente chi sta
vincendo, il che elimina il rumore delle partite giocate a caso, che era il
tallone d'Achille del metodo pre-2016.

Il risultato è quello che rende possibile il ciclo di *self-play*: **la ricerca
gioca meglio delle reti che la guidano**. La distribuzione delle visite alla
radice, normalizzata, è una policy migliorata rispetto a $P(s,\cdot)$, e
diventa il bersaglio su cui la rete si addestra. MCTS, in questa lettura, è un
**operatore di miglioramento della policy**: lo stesso ruolo che nella
programmazione dinamica ha il passo di *policy improvement*, ottenuto con la
ricerca invece che con un massimo esatto.

`````

L'idea è più generale del gioco da tavolo, ed è il motivo per cui conviene
averla in tasca: **quando si può simulare, si può pensare**. MuZero la usa
senza conoscere le regole, imparando un modello latente su cui cercare; il
capitolo sul RL basato su modello ci torna sopra; e i modelli linguistici che
esplorano più catene di ragionamento prima di rispondere fanno, con altri
nomi, la stessa cosa.

### In pratica: le visite si concentrano

Che l'albero cresca storto non è un modo di dire, ed è la cosa più facile da
verificare. Prendiamo un albero giocattolo di profondità quattro con sedici
foglie di valore noto e nascondiamo la migliore in mezzo alle altre.

```python
import math
import numpy as np

# Un albero giocattolo: profondità 4, due mosse per nodo, 16 foglie.
# I valori delle foglie li conosciamo, così sappiamo qual è la risposta giusta.
PROFONDITA, RAMI = 4, 2
rng = np.random.default_rng(7)
valori_foglie = rng.uniform(0, 1, RAMI ** PROFONDITA)
valori_foglie[6] = 0.98                      # la foglia buona, nascosta in mezzo
migliore = int(valori_foglie.argmax())
PRIME = (RAMI ** PROFONDITA - 1) // (RAMI - 1)   # indice della prima foglia

def figli(nodo):
    return [nodo * RAMI + 1 + k for k in range(RAMI)]

def foglia(nodo):
    return nodo >= PRIME

N, W = {0: 0}, {0: 0}                        # visite e somma dei ritorni

def uct(nodo, c=1.4):
    """UCB1 applicato a un bivio dell'albero: è la formula della sezione bandit."""
    padre = N[nodo]
    def punteggio(f):
        if N.get(f, 0) == 0:
            return float("inf")              # mai provato: massimamente urgente
        return W[f] / N[f] + c * math.sqrt(math.log(padre) / N[f])
    return max(figli(nodo), key=punteggio)

def simula(nodo):
    """Discesa a caso fino a una foglia: la stima grezza di questo nodo."""
    while not foglia(nodo):
        nodo = int(rng.choice(figli(nodo)))
    return valori_foglie[nodo - PRIME]

for _ in range(2000):
    nodo, cammino = 0, [0]
    while not foglia(nodo) and all(N.get(f, 0) > 0 for f in figli(nodo)):
        nodo = uct(nodo)                                          # 1. SELEZIONE
        cammino.append(nodo)
    if not foglia(nodo):
        nodo = next(f for f in figli(nodo) if N.get(f, 0) == 0)   # 2. ESPANSIONE
        cammino.append(nodo)
        N[nodo] = W[nodo] = 0
    ritorno = simula(nodo)                                        # 3. SIMULAZIONE
    for n in cammino:                                             # 4. RISALITA
        N[n] += 1
        W[n] += ritorno

print("visite ai due rami dalla radice:", [N[f] for f in figli(0)])
print("valore medio dei due rami      :", [round(float(W[f] / N[f]), 3)
                                           for f in figli(0)])

visite_foglie = np.array([N.get(PRIME + i, 0) for i in range(RAMI ** PROFONDITA)])
print(f"foglia migliore: {migliore} (valore {valori_foglie[migliore]:.2f})")
print(f"quota delle visite andata lì: "
      f"{visite_foglie[migliore] / visite_foglie.sum():.1%}")
print(f"tirando a caso sarebbe stata: {1 / len(valori_foglie):.1%}")
```

Dalla radice, il ramo che porta alla foglia buona riceve **1922 visite su
2000** e l'altro 78: dopo poche decine di prove la ricerca ha smesso di
sprecare tempo di là. In fondo all'albero, il **58%** di tutte le visite
finisce sulla foglia migliore, contro il $6{,}2\%$ che le toccherebbe tirando a
caso. Nessuno ha detto all'algoritmo dove guardare, e la formula che gliel'ha
fatto scoprire è la stessa che nel capitolo sui bandit sceglieva fra due leve.

## Da AlphaGo ad AlphaZero

Torniamo alla mossa 37. AlphaGo {cite}`silver2016mastering` non era un solo
algoritmo, ma una sintesi: una rete di policy che proponeva mosse promettenti,
una rete di valore che stimava chi fosse in vantaggio, e la ricerca ad albero
Monte Carlo appena vista, che usava entrambe per esplorare in profondità solo
le linee più sensate.

```{figure} ../figures/alphago-2016.svg
:name: fig-alphago
:alt: "Ciclo chiuso: il self-play genera partite che il sistema gioca contro sé stesso; dalle partite si addestrano due reti, quella di policy che propone le mosse e quella di valore che stima chi sta vincendo; le due reti guidano a loro volta una ricerca ad albero Monte Carlo, che gioca meglio di entrambe e produce le partite del giro successivo."
:width: 92%

Il giro che si alimenta da solo. La ricerca ad albero gioca meglio delle reti
che la guidano, e le partite che ne escono diventano il materiale con cui
quelle reti migliorano.
```

Il ciclo di {numref}`fig-alphago` è il motivo per cui i suoi successori
poterono fare a meno delle partite umane. Se la ricerca produce mosse migliori di quelle che
le reti sanno proporre, allora il sistema ha una fonte di supervisione interna:
non gli serve un maestro, gli basta giocare contro sé stesso e imparare da
dove la ricerca lo ha portato. Le reti erano state addestrate prima su partite umane,
poi affinate con RL giocando contro se stesse.

Un anno dopo, **AlphaGo Zero** {cite}`silver2017mastering` elimina persino le
partite umane: parte dalle sole regole del Go e impara *tabula rasa*, dal
nulla, soltanto affrontando copie di sé, fino a battere nettamente la versione
che aveva sconfitto Lee Sedol. Nel 2018 **AlphaZero** {cite}`silver2018general`
generalizza la ricetta: la stessa architettura, senza ritocchi per gioco,
padroneggia Go, scacchi e shogi, superando i migliori programmi
specializzati. È la dimostrazione più limpida di cosa nasce dall'unione di RL,
ricerca ad albero e reti profonde.

## Un ultimo salto: allineare i modelli linguistici

Lo stesso meccanismo (aumentare la probabilità di ciò che riceve un giudizio
positivo) è oggi al cuore dell'addestramento dei modelli linguistici.

```{figure} ../figures/instructgpt-2022.svg
:name: fig-instructgpt
:alt: "Schema in tre stadi: alcune persone ordinano per preferenza più risposte allo stesso prompt; da questi ordinamenti si addestra un reward model che impara ad assegnare punteggi; il reward model guida infine l'ottimizzazione della policy del modello linguistico, che genera, viene valutata e aggiornata."
:width: 100%

Il giudizio umano che diventa un numero. Le persone non danno voti: mettono in
fila delle risposte, ed è il reward model a tradurre quell'ordine in un
punteggio che l'ottimizzazione sa usare.
```

Il dettaglio di {numref}`fig-instructgpt` che vale la pena notare è il primo
stadio: alle persone si chiede di **ordinare**, non di valutare. Confrontare
due risposte è un giudizio che gli esseri umani danno con buona coerenza fra
loro; assegnare un voto da uno a dieci molto meno, e su scale diverse.
Nell'**RLHF** (*Reinforcement Learning from Human Feedback*; Christiano et
al., 2017; {cite}`ouyang2022training`) le risposte del modello sono
l'"azione", dei valutatori umani indicano quali preferiscono, e le loro
preferenze addestrano un *modello di ricompensa* che fa da critico. Con PPO si
ritocca poi la policy del modello (la sua tendenza a produrre certe risposte),
verso ciò che gli umani apprezzano. La stessa idea che ha portato una macchina
a giocare la mossa 37 aiuta oggi un assistente a rispondere in modo utile e
onesto.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- I metodi a gradiente di policy imparano **direttamente a decidere**: invece
  di dare un voto a ogni mossa e poi scegliere la migliore, regolano la
  *tendenza* dell'agente, come si allena un cane rendendo più probabili i
  comportamenti che hanno fruttato un premio. È la via naturale quando le
  mosse possibili non sono un menu di poche voci.
- **REINFORCE** è il "prova e ricorda": si gioca una partita intera e, se è
  andata bene, si rende più probabile tutto ciò che si è fatto. Semplice, ma
  lento e altalenante, perché il giudizio arriva solo alla fine.
- **Actor-Critic** affianca al giocatore un allenatore a bordo campo che
  commenta ogni mossa ("meglio del previsto", "peggio del previsto"):
  l'apprendimento diventa più rapido e più stabile. **A3C** fa giocare molti
  attori in parallelo; **PPO** cambia la strategia solo di poco per volta
  (passi piccoli e prudenti, ma tanti) ed è oggi lo standard.
- La **ricerca ad albero Monte Carlo** è il "pensare prima di muovere":
  migliaia di volte si scende nell'albero delle possibilità scegliendo dove
  conviene, si prova un ramo nuovo, si tira fino alla fine e si riporta
  indietro il risultato. L'albero cresce **storto di proposito**, profondo
  dove promette e appena accennato altrove, e la mossa scelta è la **più
  visitata**, non quella con la media più alta.
- **AlphaGo** e **AlphaZero** uniscono la strategia, la stima di chi sta
  vincendo e quella esplorazione ad albero; con l'**RLHF** lo stesso
  meccanismo, guidato dalle preferenze delle persone, allinea i modelli
  linguistici.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- I metodi a gradiente di policy apprendono **direttamente**
  $\pi_\theta(a\mid s)$: adatti ad azioni continue e strategie stocastiche.
- **REINFORCE** aumenta la probabilità delle azioni seguite da ritorni alti,
  ma soffre di varianza elevata.
- **Actor-Critic** aggiunge un critico $V_\phi(s)$ che fornisce il *vantaggio*
  $A_t$, riducendo la varianza. **A3C** parallelizza gli attori; **PPO** limita
  i passi con il *clipping* ed è lo standard attuale.
- **MCTS/UCT** applica UCB1 a ogni nodo dell'albero,
  $Q(s,a) + c\sqrt{\ln N(s)/N(s,a)}$, e alterna selezione, espansione,
  simulazione e risalita. AlphaZero sostituisce il termine di esplorazione con
  **PUCT**, pesato dalla policy a priori, e la simulazione casuale con la rete
  di valore. La distribuzione delle visite alla radice è una **policy
  migliorata**: è l'operatore che rende possibile il *self-play*.
- **AlphaGo/AlphaZero** uniscono policy, valore e ricerca ad albero; **RLHF**
  applica PPO all'allineamento degli LLM.
```
`````
