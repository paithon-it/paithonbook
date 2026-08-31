# Decidere ogni tanto invece che sempre

Che cosa resta, di tutto quello che un agente ha imparato, quando gli si cambia
l'obiettivo?

La domanda ha un banco di prova classico, ed è una pianta di quattro stanze
collegate da quattro porte, che sono gli unici passaggi fra una stanza e
l'altra:

```text
#############
#     #     #
#     #     #
#           #      <- porta fra le due stanze di sopra
#     #     #
#     #     #
## ####     #      <- porta fra le due stanze di sinistra
#     #### ##      <- porta fra le due stanze di destra
#     #     #
#     #     #
#           #      <- porta fra le due stanze di sotto
#     #     #
#############
```

Un agente che si muove di una casella per volta impara benissimo ad arrivare
alla casella che gli si è indicata: gli servono qualche migliaio di passi e ci
arriva. Poi si sposta il traguardo in un'altra stanza, e ricomincia da capo,
compresa la parte che sapeva già a memoria: come uscire dalla stanza in cui si
trova, che ha percorso mille volte e che non è cambiata di un pixel.

Il guaio è nel **formato** di ciò che ha imparato. Quello che ha in mano è la
tabella dei valori del
{doc}`capitolo sul reinforcement learning </ReinforcementLearning/overview>`,
un voto per ogni coppia (casella, mossa), e quei voti dicono quanto conviene
una mossa *per quel traguardo lì*:
cambiato il traguardo, ogni numero è da rifare. Quello che vorremmo salvare,
invece di una mossa, è un pezzo di comportamento: «esci da questa stanza», che
vale per tutti i traguardi che stanno di là.

## Un pezzo di comportamento che si può chiamare per nome

`````{tab} Elementare

In una partita di pallacanestro nessuno decide dove mettere il piede. Si
decide **lo schema**: quello che in settimana la squadra ha provato cento
volte, e che poi in campo esce come un gesto solo.

Uno schema ha tre cose. C'è un momento in cui lo si può chiamare, e non è
sempre: a metà campo, con la palla in mano, mentre dalla propria area non
avrebbe senso. C'è quello che succede dentro, la sequenza di tagli e blocchi
che ognuno conosce a memoria. E c'è un modo di finire, deciso in anticipo: il
tiro parte, oppure lo schema si rompe e si torna a giocare a braccio. Le prime
due cose ce le ha anche una sequenza di mosse qualunque; la terza no, ed è
quella che fa la differenza, perché consente di chiamare lo schema e poi
smettere di pensarci, sapendo che a un certo punto qualcuno dirà che è finito.

Chi chiama lo schema decide di rado, una volta ogni dieci secondi invece che
dieci volte al secondo, e i suoi dieci secondi sono un gesto solo anche per
lui: ha chiamato una cosa, e quella cosa o riesce o no. Chi sta dentro lo
schema, in quei dieci secondi, non decide affatto: esegue.

`````

`````{tab} Superiore

Il quadro delle **opzioni** {cite}`sutton1999options` formalizza la cosa in una
tripla. Un'opzione $\omega = \langle I_\omega, \pi_\omega, \beta_\omega\rangle$
è fatta di

- un **insieme di avvio** $I_\omega \subseteq S$, gli stati da cui l'opzione
  può partire;
- una **sotto-politica** $\pi_\omega(a \mid s)$, che sceglie le azioni
  elementari finché l'opzione è in corso;
- una **condizione di terminazione** $\beta_\omega(s) \in [0,1]$, la
  probabilità che l'opzione finisca in $s$.

L'agente ha quindi due politiche a due livelli: una **politica sulle opzioni**
$\pi_\Omega(\omega \mid s)$, che sceglie quale opzione avviare, e le
sotto-politiche $\pi_\omega$, che la eseguono. La terza componente è ciò che
distingue un'opzione da una macro generica: una macro è un gruppo di azioni
qualunque, un'opzione è un gruppo di azioni che sa dire quando è finito, ed è
quel «sa dire» a renderla innestabile in un MDP.

Innestabile, ma non a costo zero. Un'opzione dura $k$ passi, e $k$ è una
variabile casuale: gli istanti in cui si decide non sono più equispaziati, e il
processo visto dal livello alto è un **semi-MDP**. L'aggiornamento cambia di
conseguenza,

$$
Q(s,\omega) \leftarrow Q(s,\omega) + \alpha
\Big[ r_{1} + \gamma r_{2} + \dots + \gamma^{k-1} r_{k}
      + \gamma^{k} \max_{\omega'} Q(s',\omega') - Q(s,\omega) \Big],
$$

dove $s$ è lo stato in cui l'opzione è stata avviata, $s'$ quello in cui è
terminata, $r_1,\dots,r_k$ le ricompense raccolte per strada, $\gamma$ lo
sconto e $\alpha$ il passo di apprendimento. Il salto è tutto in
quell'esponente: lo sconto si applica $k$ volte invece che una, perché fra le
due decisioni è passato $k$ e non $1$.

Da qui i tre problemi che il campo si porta dietro, e vale la pena tenerli
distinti perché hanno risposte diverse: trovare i **sotto-obiettivi**, trovare
la **politica sulle opzioni**, trovare le **sotto-politiche**.

`````

Quel pezzo di comportamento, nel gergo del campo, si chiama **opzione**, ed è
la parola che da qui in avanti userà tutta la sezione: dove sopra si legge
«schema», nella letteratura e nel codice si legge *option*.

## Quello che si guadagna, e quello che si paga

`````{tab} Elementare

Uno schema provato una volta serve in ogni partita. È lì che sta il guadagno:
non si rimpara a uscire dal pressing ogni volta che cambia l'avversario, si
rimpara solo quando chiamarlo. Ed è anche il motivo per cui una squadra nuova
può cominciare a giocare in una settimana: gli schemi arrivano già fatti dai
giocatori che li hanno imparati altrove.

Il conto da pagare c'è, e nessuna delle sue voci è piccola.

C'è il costo di **inventarli**: gli schemi non nascono da soli, e chi li
prepara deve già sapere dove il gioco si blocca e quali movimenti lo sbloccano.

C'è il costo di **sceglierli**, che è diverso dal primo: provarli tutti è fuori
discussione, perché con quattro gesti possibili le sequenze lunghe dieci sono
più di un milione, e in una settimana se ne provano una dozzina. Qualunque
repertorio è una scommessa su quali dodici.

E c'è il costo che si dimentica, quello sulla **qualità del gioco**. Uno schema
porta dove porta, e ci mette il suo tempo: chi gioca soltanto a schemi non vede
il pallone perso dall'avversario con il canestro libero davanti, chiama la sua
sequenza, la esegue per bene, e arriva dall'altra parte tre secondi dopo che
sarebbe potuto arrivare da solo in due passi. Giocare a gesti grossi fa perdere
le scorciatoie, sempre.

Da quest'ultimo viene la sola scelta sensata: gli schemi si **aggiungono** al
gioco libero e non lo sostituiscono. Chi ha in mano tutti e due chiama lo
schema quando il campo è chiuso e va da solo quando la strada è aperta, e così
quel terzo costo non lo paga più.

`````

`````{tab} Superiore

Il guadagno principale è in **campioni**: una sotto-politica appresa una volta
si riusa per traguardi diversi e per compiti diversi, quindi la gerarchia è
anzitutto una forma di transfer. Il secondo guadagno è
sull’**esplorazione**: esplorare avviando opzioni copre distanze che una
passeggiata casuale di azioni elementari non copre quasi mai, e questo è il
motivo per cui la gerarchia si incontra sempre insieme al problema delle
ricompense rade.

I costi sono di tre specie, e l'ultima è quella che si dimentica.

- **La conoscenza del dominio.** Il quadro originale dà i sotto-obiettivi per
  scontati. Quando non lo sono, trovarli è il problema, e non un
  preliminare.
- **La complessità combinatoria.** Il numero di macro cresce in modo
  esponenziale nella loro lunghezza, quindi enumerarle è escluso e la politica
  di alto livello va approssimata come tutto il resto.
- **La qualità.** Una politica che contiene macro può essere **peggiore** di una
  fatta di sole azioni elementari, perché una macro scavalca percorsi più corti
  che le azioni singole avrebbero trovato. La causa non sta nell'algoritmo ma
  nell'aver ristretto l'insieme delle traiettorie percorribili. Da
  qui la scelta di progetto standard, che le opzioni si **aggiungano**
  all'insieme delle azioni elementari invece di rimpiazzarlo: così l'ottimo
  raggiungibile resta quello di prima, e le opzioni possono solo far arrivare
  prima.

`````

### In pratica: quattro stanze, con e senza opzioni

Il risparmio di esperienza, le scorciatoie perse e la ragione per cui le
opzioni si aggiungono alle mosse invece di rimpiazzarle si misurano tutti e tre
nello stesso banco di prova, che è la pianta di poco fa. Le opzioni sono otto,
cioè due per ogni porta, una per ciascuna delle due stanze che quella porta
collega, e ognuna conduce alla sua porta per la via più breve dentro la stanza,
terminando quando ci arriva.

Poi si allenano tre agenti sullo stesso problema, dando a ogni passo elementare
una ricompensa di $-1$: il segno negativo fa sì che l'agente, cercando il
massimo, cerchi la strada più corta, e i valori che impara si leggono come
«quanti passi mi mancano, cambiato di segno». Il primo agente ha le sole
quattro mosse, il secondo le mosse **più** le opzioni, il terzo le sole
opzioni. Ogni **episodio** parte da una casella sorteggiata e finisce quando
l'agente tocca il traguardo.

```python
import numpy as np
from collections import deque

MAPPA = ["#############", "#     #     #", "#     #     #", "#           #",
         "#     #     #", "#     #     #", "## ####     #", "#     #### ##",
         "#     #     #", "#     #     #", "#           #", "#     #     #",
         "#############"]
MOSSE = [(-1, 0), (1, 0), (0, -1), (0, 1)]
libera = lambda s: MAPPA[s[0]][s[1]] == " "
CELLE = [(r, c) for r in range(13) for c in range(13) if libera((r, c))]
vicini = lambda s: [(s[0]+dr, s[1]+dc) for dr, dc in MOSSE
                    if libera((s[0]+dr, s[1]+dc))]

# Una porta e' una cella libera con due soli vicini, e opposti fra loro.
PORTE = [s for s in CELLE if len(vicini(s)) == 2
         and vicini(s)[0][0] + vicini(s)[1][0] == 2 * s[0]
         and vicini(s)[0][1] + vicini(s)[1][1] == 2 * s[1]]
assert len(PORTE) == 4, PORTE

# Le stanze sono le componenti connesse che restano togliendo le porte.
STANZA, resto = {}, [s for s in CELLE if s not in PORTE]
for s in resto:
    if s in STANZA:
        continue
    coda, etichetta = deque([s]), len(set(STANZA.values()))
    STANZA[s] = etichetta
    while coda:
        x = coda.popleft()
        for v in vicini(x):
            if v not in STANZA and v not in PORTE:
                STANZA[v] = etichetta
                coda.append(v)
assert len(set(STANZA.values())) == 4

def politica_verso(porta, st):
    """La sotto-politica di un'opzione: il cammino piu' breve fino a `porta`
    restando dentro la stanza `st`. L'insieme di avvio e' la stanza piu' le
    porte che vi si aprono; la terminazione e' l'arrivo."""
    dentro = {s for s in resto if STANZA[s] == st}
    dentro |= {p for p in PORTE if any(STANZA.get(v) == st for v in vicini(p))}
    dist, coda = {porta: 0}, deque([porta])
    while coda:
        x = coda.popleft()
        for v in vicini(x):
            if v in dentro and v not in dist:
                dist[v] = dist[x] + 1
                coda.append(v)
    return {s: min((a for a in range(4)
                    if (s[0]+MOSSE[a][0], s[1]+MOSSE[a][1]) in dist),
                   key=lambda a: dist[(s[0]+MOSSE[a][0], s[1]+MOSSE[a][1])])
            for s in dist if s != porta}

OPZIONI = [(p, politica_verso(p, st)) for p in PORTE
           for st in sorted({STANZA[v] for v in vicini(p)})]
IDX = {s: i for i, s in enumerate(CELLE)}
N, NA = len(CELLE), 4 + len(OPZIONI)
print(f"{N} caselle libere, {len(PORTE)} porte, {len(OPZIONI)} opzioni")

def tabelle(meta):
    """Dove si finisce e in quanti passi, per ogni stato e ogni azione.
    L'ambiente e' deterministico, quindi si calcola una volta sola."""
    dove = np.zeros((N, NA), int)
    passi = np.zeros((N, NA), int)
    valida = np.zeros((N, NA), bool)
    for s, i in IDX.items():
        for a in range(4):
            v = (s[0] + MOSSE[a][0], s[1] + MOSSE[a][1])
            dove[i, a] = IDX[v if libera(v) else s]
            passi[i, a], valida[i, a] = 1, True
        for j, (porta, pol) in enumerate(OPZIONI):
            if s not in pol:
                continue
            x, k = s, 0
            while x != porta and x != meta:        # si ferma anche sul traguardo
                m = MOSSE[pol[x]]
                x = (x[0] + m[0], x[1] + m[1])
                k += 1
            dove[i, 4+j], passi[i, 4+j], valida[i, 4+j] = IDX[x], k, True
    return dove, passi, valida

def allena(meta, usa, episodi=300, semi=5, tetto=300):
    """`usa` e' la maschera che dice quali azioni l'agente ha a disposizione."""
    dove, passi, valida = tabelle(meta)
    ammesse, fine = valida & usa, IDX[meta]
    costo_iniziale, costo_finale = [], []
    for seme in range(semi):
        rng = np.random.default_rng(seme)
        Q = np.where(ammesse, -60.0, -np.inf)
        storia = []
        for _ in range(episodi):
            s, tot = int(rng.integers(N)), 0
            while s != fine and tot < tetto:
                lista = np.flatnonzero(ammesse[s])
                a = (rng.choice(lista) if rng.random() < 0.1
                     else int(lista[np.argmax(Q[s, lista])]))
                s2, k = int(dove[s, a]), int(passi[s, a])
                tot += k
                futuro = 0.0 if s2 == fine else Q[s2, ammesse[s2]].max()
                # l'aggiornamento semi-MDP: -k perche' ogni passo costa 1
                Q[s, a] += 0.25 * (-k + futuro - Q[s, a])
                s = s2
            storia.append(tot)
        costo_iniziale.append(np.mean(storia[:50]))
        lunghezze = []                       # la politica imparata, senza esplorare
        for s0 in range(N):
            s, tot = s0, 0
            while s != fine and tot < tetto:
                a = int(np.argmax(np.where(ammesse[s], Q[s], -np.inf)))
                tot += int(passi[s, a])
                s = int(dove[s, a])
            lunghezze.append(tot)
        costo_finale.append(np.mean(lunghezze))
    return np.mean(costo_iniziale), np.mean(costo_finale)

def ottimo(meta):
    """Il minimo possibile: cammino piu' breve da ogni cella, a mosse singole."""
    dist, coda = {meta: 0}, deque([meta])
    while coda:
        x = coda.popleft()
        for v in vicini(x):
            if v not in dist:
                dist[v] = dist[x] + 1
                coda.append(v)
    assert len(dist) == N, (len(dist), N)
    return np.mean([dist[s] for s in CELLE])

M_MOSSE = np.zeros((N, NA), bool); M_MOSSE[:, :4] = True
M_OPZ = np.zeros((N, NA), bool);   M_OPZ[:, 4:] = True

for nome, meta in (("una porta", PORTE[3]), ("un angolo di stanza", (11, 11))):
    print(f"\ntraguardo: {nome} {meta}   "
          f"(il minimo possibile e' {ottimo(meta):.1f} passi)")
    for etichetta, usa in (("solo le quattro mosse", M_MOSSE),
                           ("mosse piu' opzioni", M_MOSSE | M_OPZ),
                           ("solo le opzioni", M_OPZ)):
        iniziale, finale = allena(meta, usa)
        print(f"   {etichetta:22s}  primi 50 episodi: {iniziale:6.1f} passi"
              f"   politica finale: {finale:6.1f}")
```

```text
104 caselle libere, 4 porte, 8 opzioni

traguardo: una porta (10, 6)   (il minimo possibile e' 8.5 passi)
   solo le quattro mosse   primi 50 episodi:   85.1 passi   politica finale:   20.1
   mosse piu' opzioni      primi 50 episodi:   21.6 passi   politica finale:   17.3
   solo le opzioni         primi 50 episodi:   15.7 passi   politica finale:   14.7

traguardo: un angolo di stanza (11, 11)   (il minimo possibile e' 10.4 passi)
   solo le quattro mosse   primi 50 episodi:  108.6 passi   politica finale:   25.3
   mosse piu' opzioni      primi 50 episodi:   94.5 passi   politica finale:   17.2
   solo le opzioni         primi 50 episodi:  303.6 passi   politica finale:  300.5
```

Tutti i numeri sono medie sulle centoquattro caselle di partenza, e vanno letti
con questo in mente: il minimo possibile di $8{,}5$ vuol dire che un agente
perfetto, partendo da una casella sorteggiata, ci mette in media otto passi e
mezzo.

Nella prima tabella il traguardo è una porta, cioè esattamente ciò che le
opzioni sanno raggiungere, e il guadagno si vede subito: nei primi cinquanta
episodi l'agente a sole mosse ne spende $85{,}1$ per arrivare, quello che ha
anche le opzioni $21{,}6$, cioè un quarto. È il risparmio di esperienza, e viene
tutto da un fatto: una decisione al posto di una stanza intera.

Nella stessa tabella c'è però il prezzo, e sta nell'ultima riga. L'agente che
ha **solo** le opzioni impara ancora più in fretta ($15{,}7$), e poi si ferma su
una politica da $14{,}7$ passi contro gli $8{,}5$ che si potrebbero fare: va di
porta in porta, e le scorciatoie dentro le stanze non le può nemmeno
rappresentare. In trecento episodi nessuno dei tre scende agli $8{,}5$, quindi
le opzioni comprano velocità e non un cammino più corto di quello che le mosse
consentirebbero; ma quel $14{,}7$ misura il repertorio di chi ha solo le
opzioni, e non la sua lentezza nell'imparare.

La seconda tabella è la stessa cosa detta in modo brutale. Il traguardo è in un
angolo di stanza, dove nessuna opzione conduce, e l'agente a sole opzioni
**non ci arriva mai**: i suoi $300{,}5$ passi sono la sbarra che il conto si dà
per non girare all'infinito (poco più di trecento, perché la sbarra ferma
l'agente alla fine dell'opzione in corso e non nel mezzo). Chi ha le mosse
insieme alle opzioni arriva, e con la politica migliore delle tre ($17{,}2$),
perché le opzioni gli sono servite a propagare i valori fra le stanze e le
mosse a coprire l'ultimo tratto.

## Dove i sotto-obiettivi non li dà nessuno

Nel banco di prova qui sopra i sotto-obiettivi, cioè le porte, li ha trovati
una regola di due righe: sono le caselle con due soli vicini. È un lusso della
pianta a stanze. Nel mondo vero il repertorio non lo detta nessuno, e trovarlo
è il problema, non il preliminare. Le risposte si dividono in due famiglie, e
si distinguono per **chi decide che cosa**.

La prima famiglia non decide niente in anticipo: fa imparare le opzioni
**insieme** alla politica che le sceglie. L’**option-critic**
{cite}`bacon2017optioncritic` estende alle opzioni lo stesso gradiente con cui
il {doc}`gradiente di policy </DeepReinforcementLearning/policy-gradient>`
migliora una politica, e lo usa per aggiustare tre cose alla volta: che cosa fa
un'opzione, quando finisce, e quale opzione conviene chiamare. Nessuno deve
scrivere un sotto-obiettivo; bisogna solo dire **quante** opzioni si vogliono,
che è come dire a una squadra «preparatene otto» invece di dettargliele.

La seconda famiglia divide i compiti: il livello alto **nomina** un obiettivo,
il livello basso lo raggiunge come crede. L'immagine è quella di un feudo, dove
il signore assegna il territorio e non entra nel merito di come lo si lavora, e
il nome se lo portano dietro le **FeUdal Networks** {cite}`vezhnevets2017feudal`:
un modulo *manager* lavora a passo lento e fissa obiettivi, un modulo *worker*
li traduce in azioni elementari a ogni battito dell'ambiente. Il manager non
dice mai come fare: dice dove andare.

Che le due famiglie si incontrino proprio dove la ricompensa arriva di rado ha
una ragione. L’**h-DQN** {cite}`kulkarni2016hdqn` mette insieme le due metà del
problema: un livello alto sceglie un obiettivo (in un gioco, «arriva alla
scala», «prendi la chiave»), un livello basso sceglie le mosse per
raggiungerlo, e il premio che tiene in piedi il livello basso se lo dà l'agente
stesso quando l'obiettivo è raggiunto. Senza quel premio interno il livello
basso non avrebbe niente da inseguire, perché il punteggio del gioco può
arrivare migliaia di passi dopo, quando la porta attraversata è dimenticata da
un pezzo.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un’**opzione** è un pezzo di comportamento che si chiama per nome, come uno
  schema in una partita: si sa da dove lo si può far partire, che cosa succede
  dentro, e soprattutto **quando è finito**. È quest'ultima cosa a distinguerlo
  da una sequenza di mosse qualunque.
- Chi comanda decide **di rado** (quale schema), chi esegue non decide affatto,
  e il guadagno sta tutto lì: un pezzo imparato una volta si riusa per obiettivi
  diversi, invece di rimpararlo da capo a ogni cambio di traguardo.
- Il conto da pagare ha tre voci: qualcuno gli schemi li deve inventare;
  provarli tutti è impossibile; e chi gioca a gesti grossi **perde le
  scorciatoie**, perché lo schema porta dove porta anche quando bastavano due
  passi.
- Per questo gli schemi si **aggiungono** al gioco libero e non lo
  sostituiscono: chi ha in mano tutti e due chiama lo schema quando serve e va
  da solo quando la strada è aperta. Nelle quattro stanze si vede in numeri:
  chi ha solo gli schemi impara prima di tutti e poi resta fermo a quasi il
  doppio del cammino più corto, e se il traguardo è in un punto dove nessuno
  schema conduce non ci arriva mai.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un’**opzione** {cite}`sutton1999options` è la tripla
  $\omega = \langle I_\omega, \pi_\omega, \beta_\omega\rangle$: insieme di
  avvio, sotto-politica, condizione di terminazione. L'agente ha due livelli,
  la politica sulle opzioni $\pi_\Omega(\omega \mid s)$ e le sotto-politiche
  $\pi_\omega(a \mid s)$.
- Un'opzione dura $k$ passi con $k$ casuale, quindi il processo visto dall'alto
  è un **semi-MDP** e l'aggiornamento porta $\gamma^{k}$ al posto di $\gamma$,
  con le ricompense del tratto accumulate e scontate.
- I guadagni sono l’**efficienza in campioni** (una sotto-politica appresa una
  volta si riusa e si trasferisce) e l’**esplorazione**, perché avviare opzioni
  copre distanze che una passeggiata di azioni elementari non copre.
- Il costo che si dimentica è la **qualità**: una politica con macro può essere
  peggiore di una di sole azioni elementari, perché scavalca percorsi più corti.
  Per questo le opzioni si aggiungono all'insieme delle azioni invece di
  rimpiazzarlo, e l'ottimo raggiungibile resta quello di prima.
- Trovare i sotto-obiettivi è il problema aperto: l’**option-critic**
  {cite}`bacon2017optioncritic` impara sotto-politiche e terminazioni con un
  teorema del gradiente di policy per le opzioni, chiedendo solo *quante*
  opzioni; le **FeUdal Networks** {cite}`vezhnevets2017feudal` separano un
  manager lento che fissa obiettivi da un worker che li esegue; l’**h-DQN**
  {cite}`kulkarni2016hdqn` innesta i due livelli su una ricompensa intrinseca.
```
`````

La gerarchia, insomma, cambia l'unità di misura del tempo: il livello alto vede
una partita fatta di poche mosse grosse, e quanto duri ciascuna glielo dice il
livello basso. Resta in piedi il punto su cui l'h-DQN si appoggia, cioè la
ricompensa che l'agente si dà da sé, comparsa qui senza spiegazioni. È il
problema della prossima sezione, e senza risolverlo la gerarchia non parte: un
livello basso che deve imparare a raggiungere una porta ha bisogno di sapere
subito se ci è arrivato, mentre il punteggio dell'ambiente, nei giochi a
ricompensa rada, può tardare migliaia di passi.
