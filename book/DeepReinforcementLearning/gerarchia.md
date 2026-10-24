# Decidere ogni tanto invece che sempre

Che cosa resta, di tutto quello che un agente ha imparato, quando gli si cambia
l'obiettivo?

La domanda ha un banco di prova classico, una pianta di quattro stanze
collegate da quattro porte, gli unici passaggi fra una stanza e
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

Un agente che si muove di una casella per volta impara benissimo a raggiungere
la casella indicata: gli serve qualche migliaio di passi. Se poi si sposta il
traguardo in un'altra stanza, ricomincia da capo, compresa la parte che
sapeva già a memoria: come uscire dalla stanza in cui si trova, che ha percorso
mille volte e che non è cambiata di un pixel.

Il limite sta nel formato di ciò che ha imparato. L'agente ha in mano la
tabella dei valori del {doc}`capitolo sul reinforcement learning
</ReinforcementLearning/overview>`, un voto per ogni coppia (casella, azione),
e quei voti dicono quanto conviene un'azione *per quel traguardo*: [cambiato il
traguardo, ogni numero è da rifare]{.pt-matita}. Vorremmo quindi salvare, invece di
un'azione, un pezzo di comportamento: «esci da questa stanza», che vale per
tutti i traguardi che stanno di là.

## Un pezzo di comportamento che si può chiamare per nome

`````{tab} Elementare

In una partita di pallacanestro nessuno decide dove mettere il piede. Si
decide lo schema: quello che in settimana la squadra ha provato cento
volte, e che poi in campo esce come un gesto solo.

Uno schema ha tre cose. C'è un momento in cui lo si può chiamare, e non è
sempre: a metà campo, con la palla in mano, mentre dalla propria area non
avrebbe senso. C'è quello che succede dentro, i passaggi e i movimenti senza
palla che ognuno ha provato in settimana. Infine c'è un modo di finire, deciso
in anticipo: il tiro parte, oppure lo schema si rompe e si torna a giocare a
braccio. Una sequenza di passi imparata a memoria ed eseguita sempre uguale ha
anche lei un inizio, un corpo e una fine; [lo schema vero, invece, guarda il
campo mentre lo si esegue]{.pt-matita}, adatta i movimenti a come si mette la
difesa, e finisce prima o dopo a seconda di come va. È questo che consente di
chiamarlo e poi smettere di pensarci, sapendo che sarà lui a dire quando è
finito.

Chi chiama lo schema decide di rado, una volta ogni dieci secondi invece che
dieci volte al secondo, e i suoi dieci secondi sono un gesto solo anche per lui:
ha chiamato una cosa, e quella cosa o riesce o no. Chi sta dentro lo schema, in
quei dieci secondi, non sceglie che cosa giocare: esegue, e si adatta soltanto a
quello che vede.

`````

`````{tab} Superiore

Il quadro delle opzioni {cite}`sutton1999options` formalizza questa struttura
in una tripla. Un'opzione $\omega = \langle I_\omega, \pi_\omega,
\beta_\omega\rangle$ è fatta di

- un **insieme di avvio** $I_\omega \subseteq S$, gli stati da cui l'opzione
  può partire;
- una **sotto-politica** $\pi_\omega(a \mid s)$, che sceglie le azioni
  elementari finché l'opzione è in corso;
- una **condizione di terminazione** $\beta_\omega(s) \in [0,1]$, la
  probabilità che l'opzione finisca in $s$.

L'agente opera quindi su due livelli: una **politica sulle opzioni**
$\pi_\Omega(\omega \mid s)$, che sceglie quale opzione avviare, e le
sotto-politiche $\pi_\omega$, che eseguono l'opzione scelta. [La differenza con
una macro classica sta nel ciclo chiuso]{.pt-matita}. Una macro è una sequenza
fissa di azioni elementari, eseguita senza guardare; un'opzione, invece, ha una
sotto-politica che guarda lo stato a ogni passo e una terminazione che dipende
dallo stato (anche in modo stocastico), quindi reagisce a ciò che succede e
finisce quando serve. La sequenza fissa resta un caso particolare, un'opzione
*semi-markoviana* (che guarda la storia dall'avvio) in cui sotto-politica e
terminazione dipendono soltanto da quanti passi sono trascorsi
{cite}`sutton1999options`.

Un'opzione dura però $k$ passi, con $k$ variabile casuale. Di conseguenza gli
istanti in cui si decide non sono più equispaziati, e il processo visto dal
livello alto è un **semi-MDP**. Anche l'aggiornamento cambia,

$$
Q(s,\omega) \leftarrow Q(s,\omega) + \alpha
\Big[ r_{1} + \gamma r_{2} + \dots + \gamma^{k-1} r_{k}
      + \gamma^{k} \max_{\omega'} Q(s',\omega') - Q(s,\omega) \Big],
$$

dove $s$ è lo stato in cui l'opzione è stata avviata, $s'$ quello in cui è
terminata, $r_1,\dots,r_k$ le ricompense raccolte per strada, $\gamma$ lo
sconto e $\alpha$ il passo di apprendimento. Il salto è tutto in
quell'esponente: lo sconto si applica $k$ volte invece di una, perché fra le
due decisioni sono trascorsi $k$ passi e non $1$.

Restano così aperti i problemi del campo: identificare i sotto-obiettivi,
apprendere la politica sulle opzioni e apprendere le sotto-politiche. Per i
sotto-obiettivi le risposte standard sono di specie diversa: cercare gli stati
collo di bottiglia, per cui passano molte delle traiettorie che arrivano a
destinazione (le porte di una pianta a stanze sono l'esempio da manuale)
{cite}`mcgovern2001automatic`; imparare abilità senza ricompensa, massimizzando
l'informazione mutua fra l'abilità scelta e gli stati che visita
{cite}`eysenbach2019diversity`; oppure rinunciare a sotto-obiettivi espliciti e
apprendere le opzioni insieme alla politica che le sceglie, come fa
l'option-critic {cite}`bacon2017optioncritic`.

`````

Quel pezzo di comportamento, nel gergo della ricerca, si chiama **opzione** (in
inglese *option*, il nome corrente nella letteratura e nel codice).

## Quello che si guadagna, e quello che si paga

`````{tab} Elementare

Uno schema provato una volta serve in ogni partita. È lì che sta il guadagno:
non si rimpara a uscire dal pressing ogni volta che cambia l'avversario, si
impara solo quando chiamarlo. Per questo una squadra nuova
può cominciare a giocare in una settimana: gli schemi arrivano già fatti dai
giocatori che li hanno imparati altrove.

Il secondo guadagno si vede in una squadra alle prime armi. Provare un gesto
alla volta, a caso, la tiene a girare nella propria metà campo: per arrivare
sotto il canestro avversario servono molti gesti giusti in sequenza, e a caso
non escono quasi mai. Uno schema intero, anche scelto alla cieca, la porta
dall'altra parte al primo tentativo, e solo lì scopre che cosa le serve
imparare.

I costi sono tre, e nessuno è piccolo.

Il primo è inventare gli schemi: non nascono da soli, e chi li
prepara deve già sapere dove il gioco si blocca e quali movimenti lo sbloccano.

Il secondo è scegliere tra gli schemi, che è un problema diverso: provarli
tutti è fuori discussione, perché con quattro gesti possibili le sequenze
lunghe dieci sono più di un milione, e in una settimana se ne provano una
dozzina. Qualunque repertorio è una scommessa su quali dodici scegliere.

Il terzo è quello che si dimentica, e riguarda la qualità del gioco. Uno schema
porta dove porta, e ci mette il suo tempo: chi gioca soltanto a schemi non vede
il pallone perso dall'avversario con il canestro libero davanti, chiama la sua
sequenza, la esegue per bene, e arriva dall'altra parte tre secondi più tardi
di quanto avrebbe fatto da solo in due passi. [Giocare a gesti grossi fa perdere
le scorciatoie]{.pt-matita}.

Da quest'ultimo viene la scelta abituale: gli schemi si aggiungono al gioco
libero e non lo sostituiscono. Una squadra che ha in mano tutti e due chiama lo
schema quando il campo è chiuso e va da sola quando la strada è aperta, e così
evita quel terzo costo. Non del tutto: se lo schema porta abbastanza bene, la
squadra si abitua a chiamarlo, e non prova più la scorciatoia che avrebbe
trovato da sola.

`````

`````{tab} Superiore

Il vantaggio principale sta nell'efficienza campionaria: una sotto-politica
appresa una volta si riusa per traguardi diversi e per compiti diversi, quindi
la gerarchia è anzitutto una forma di transfer. Il secondo vantaggio riguarda
l'esplorazione: avviare opzioni copre distanze che una passeggiata casuale di
azioni elementari non copre quasi mai; per questo la gerarchia si adotta
tipicamente in presenza di ricompense rade.

I costi sono di tre specie, e l'ultima è quella che si dimentica.

- La conoscenza del dominio. Il quadro originale dà i sotto-obiettivi per
  scontati. Quando non lo sono, trovarli è il problema, e non un
  preliminare.
- La complessità combinatoria. Il numero di macro cresce in modo
  esponenziale con la loro lunghezza, quindi enumerarle è escluso e la politica
  di alto livello va approssimata come tutto il resto.
- La qualità. [Una politica che contiene macro può essere peggiore di una fatta
  di sole azioni elementari]{.pt-matita}, perché una macro scavalca percorsi
  più corti che le azioni singole avrebbero trovato. La causa non sta
  nell'algoritmo ma nell'aver ristretto l'insieme delle traiettorie
  percorribili. Da qui la scelta progettuale standard di aggiungere le opzioni
  all'insieme delle azioni elementari invece di rimpiazzarlo: così l'ottimo
  raggiungibile resta quello di prima. Raggiungibile, però, non vuol dire
  raggiunto: le opzioni cambiano quali stati l'agente visita, e un'opzione
  sufficientemente efficace sottrae all'apprendimento le visite che servivano a
  scoprire la strada migliore. Si impara più in fretta all'inizio e si arriva
  all'ottimo più tardi, o non ci si arriva; che il saldo sia positivo dipende
  da quanto le opzioni sono adatte al compito.

Già il lavoro fondativo del quadro avverte che troppe opzioni possono
rallentare anche la pianificazione, come i macro-operatori usati senza misura
(l’*utility problem*) {cite}`sutton1999options`. Inoltre Jong, Hester e Stone
rilevano che aggiungere opzioni alle azioni elementari aiuta solo in contesti
limitati e può peggiorare l'apprendimento, perché distorce l'esplorazione
{cite}`jong2008utility`. Dal lato del guadagno, il risparmio di campioni viene
in buona parte dall'apprendimento *intra-option*: ogni azione eseguita aggiorna
il valore di tutte le opzioni che in quello stato l'avrebbero scelta, non solo
di quella in corso {cite}`sutton1999options`.

`````

### In pratica: quattro stanze, con e senza opzioni

Il risparmio di esperienza, le scorciatoie perse e la ragione per cui le
opzioni si aggiungono alle mosse invece di rimpiazzarle si misurano tutti e tre
nello stesso banco di prova: la pianta a quattro stanze. Le opzioni sono otto: ogni
porta ne ha due, una per ciascuna delle due stanze che collega. Ognuna arriva
alla sua porta per la via più breve dentro la stanza, e finisce lì.

Si allenano poi tre agenti sullo stesso problema, dando a ogni passo elementare
una ricompensa di $-1$: il segno negativo spinge l'agente, nel massimizzare il
ritorno, a trovare la strada più corta, e i valori appresi si leggono come
«passi mancanti al traguardo, cambiati di segno». Il primo agente ha le sole
quattro mosse, il secondo le mosse più le opzioni, il terzo le sole opzioni.
Ogni episodio parte da una casella sorteggiata e finisce quando l'agente tocca
il traguardo. Tutti e tre esplorano a caso il $10\%$ delle volte, correggono le
stime con un passo di $0{,}25$, non scontano il futuro (il compito finisce
sempre) e partono pessimisti, con un valore di $-60$ per ogni azione, molto
sotto quelli veri.

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
        Q = np.where(ammesse, -60.0, -np.inf)    # partenza molto pessimista
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
                # l'aggiornamento semi-MDP: -k perche' ogni passo costa 1;
                # gamma = 1, perche' il compito finisce sempre
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

La politica finale e il minimo possibile sono medie sulle centoquattro caselle
di partenza (la prima anche sui cinque semi); i primi cinquanta episodi sono
invece medie su partenze sorteggiate, cinquanta per ciascun seme, e contano
anche i passi spesi a esplorare. Il minimo di $8{,}5$ è il numero medio di
passi di un agente perfetto che parta da una casella qualsiasi.

Nella prima tabella il traguardo è una porta, cioè ciò che le
opzioni sanno raggiungere, e il guadagno si vede subito: nei primi cinquanta
episodi l'agente a sole mosse ne spende $85{,}1$ per arrivare, quello che ha
anche le opzioni $21{,}6$, cioè un quarto. È il risparmio di esperienza, e
deriva da questo: una sola decisione copre una stanza intera.

L'ultima riga della stessa tabella ha una lettura meno ovvia. L'agente che ha
solo le opzioni impara ancora più in fretta ($15{,}7$) e poi si ferma su una
politica da $14{,}7$ passi contro gli $8{,}5$ possibili, ma quel $14{,}7$ non è il
limite del suo repertorio: qui il traguardo è una porta, l'opzione in corso si
ferma appena ci arriva, e anche con le sole opzioni la strada da $8{,}5$ esiste.
È l'apprendimento a non arrivarci, e in trecento episodi non ci arriva nessuno
dei tre: le opzioni accelerano l'apprendimento, non accorciano il cammino.

La seconda tabella mostra il limite di questo repertorio. Il traguardo è in un
angolo di stanza, dove nessuna opzione conduce, e l'agente a sole opzioni non
ci arriva mai: i suoi $300{,}5$ passi rappresentano il tetto fissato nella
simulazione per non girare all'infinito (poco più di trecento, perché il limite
ferma l'agente alla fine dell'opzione in corso e non nel mezzo). L'agente che
combina mosse e opzioni raggiunge invece il traguardo, ottenendo la politica
migliore tra le tre ($17{,}2$ passi dopo trecento episodi): le opzioni gli sono
servite a propagare i valori fra le stanze e le mosse a coprire l'ultimo
tratto.

Trecento episodi, però, fotografano la partenza. Prolungando l'addestramento, i
due agenti che hanno le mosse si scambiano di posto:

```python
for nome, meta in (("una porta", PORTE[3]), ("un angolo", (11, 11))):
    for etichetta, usa in (("solo le quattro mosse", M_MOSSE),
                           ("mosse piu' opzioni", M_MOSSE | M_OPZ)):
        _, finale = allena(meta, usa, episodi=1000)
        print(f"{nome:10s} {etichetta:22s} dopo mille episodi: {finale:5.1f}")
```

```text
una porta  solo le quattro mosse  dopo mille episodi:   8.6
una porta  mosse piu' opzioni     dopo mille episodi:  14.0
un angolo  solo le quattro mosse  dopo mille episodi:  10.6
un angolo  mosse piu' opzioni     dopo mille episodi:  13.3
```

Chi ha solo le mosse arriva a meno di mezzo passo dal minimo; chi ha anche le
opzioni resta sopra di cinque passi e mezzo sulla porta e di quasi tre
sull'angolo. Le opzioni che l'hanno fatto partire adesso lo trattengono, e la
spiegazione più plausibile è quella del costo che si dimentica: sono abbastanza
efficaci, l'agente le chiama, e non compie le visite che avrebbero rivelato la
strada più corta. L'esperimento misura il ritardo, non la sua causa. [Aggiungere
le opzioni lascia raggiungibile la strada migliore, ma non garantisce che la si
trovi]{.pt-matita}.

## Dove i sotto-obiettivi non li dà nessuno

Nel banco di prova delle quattro stanze una regola di due righe ha trovato i
sotto-obiettivi, cioè le porte: sono le caselle con due soli vicini. È un lusso
della pianta a stanze, perché nel mondo vero nessuno detta il repertorio. Le
architetture che se ne fanno carico si dividono in due famiglie, secondo chi
decide che cosa.

La prima famiglia non decide niente in anticipo: fa imparare le opzioni insieme
alla politica che le sceglie. L’**option-critic** {cite}`bacon2017optioncritic`
applica alle opzioni il metodo del {doc}`gradiente di policy
</DeepReinforcementLearning/policy-gradient>`, e lo usa per aggiustare tre cose
alla volta: che cosa fa un'opzione, quando finisce, e quale opzione conviene
chiamare. Nessuno deve scrivere un sotto-obiettivo; bisogna solo dire quante
opzioni si vogliono. Il gradiente della terminazione allunga l'opzione dove
continuarla vale più della media delle alternative, e la chiude dove vale meno.
In formule, il gradiente del valore rispetto ai parametri $\vartheta$ della
terminazione è una somma pesata, sugli stati e sulle opzioni incontrati, del
termine

$$
-\,\frac{\partial\beta_{\omega,\vartheta}(s')}{\partial\vartheta}\,
\big(Q_\Omega(s',\omega) - V_\Omega(s')\big),
$$

dove $Q_\Omega(s,\omega)$ è il valore di proseguire con l'opzione $\omega$
dallo stato $s$ e
$V_\Omega(s)=\sum_{\omega}\pi_\Omega(\omega\mid s)\,Q_\Omega(s,\omega)$ la
sua media sulle opzioni che la politica di alto livello sceglierebbe lì.
Lasciate libere, però, le terminazioni degenerano in opzioni da un passo o in
un'opzione sola che fa tutto, e in pratica si aggiunge un piccolo costo per ogni
cambio di opzione {cite}`harb2018waiting`.

La seconda famiglia divide i compiti: il livello alto nomina un obiettivo, il
livello basso lo raggiunge come crede. L'immagine è quella di un feudo in cui
il signore assegna il territorio senza entrare nel merito di come viene
lavorato; da qui il nome delle **FeUdal Networks**
{cite}`vezhnevets2017feudal`, ispirate a Dayan e Hinton. Un modulo *manager*
lavora a passo lento e fissa obiettivi, un modulo *worker* li traduce in azioni
elementari a ogni passo dell'ambiente. L'obiettivo è una direzione in uno
spazio latente appreso, e il worker riceve un premio interno per quanto si
muove lungo quella direzione: il manager dice dove andare, non come.

[Le due famiglie servono soprattutto dove la ricompensa arriva di
rado]{.pt-matita}: una decisione di alto livello copre in una volta sola un
tratto che a passi elementari richiederebbe una fortuna rara. Nella seconda
famiglia, però, il problema si ripresenta un livello più giù. Un livello basso
che impara a raggiungere un obiettivo ha bisogno di sapere quando ci è
arrivato, ma il punteggio del gioco può arrivare migliaia di passi dopo, quando
la porta attraversata è ormai lontana nel tempo. L’**h-DQN**
{cite}`kulkarni2016hdqn` affronta questo punto: un livello alto sceglie un
obiettivo (in un gioco, «arriva alla scala», «prendi la chiave»), un livello
basso sceglie le azioni per raggiungerlo, e quando l'obiettivo è raggiunto
l'agente stesso dà al livello basso un premio interno. Quel premio non entra
nel punteggio del gioco: serve soltanto a dire al livello basso «ci sei
arrivato». L'h-DQN, però, non trova gli obiettivi da sé: su *Montezuma's
Revenge* li ricava da un rivelatore di oggetti scritto a mano (la scala, la
chiave, la porta), ricevendo dall'esterno ciò che si voleva apprendere. Mostra
che due livelli e un premio interno superano le ricompense rade quando gli
obiettivi ci sono; trovarli resta il compito dei metodi che li apprendono, come
l’option-critic e le FeUdal Networks.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un’opzione è un pezzo di comportamento che si chiama per nome, come uno
  schema in una partita: si sa da dove lo si può far partire, che cosa succede
  dentro e quando è finito. A differenza di una sequenza di passi imparata a
  memoria, lo schema guarda il campo mentre lo si esegue: si adatta a come si mette
  la difesa, e finisce quando serve.
- Chi comanda decide di rado (quale schema), chi esegue non sceglie che cosa
  giocare, e il guadagno sta tutto lì: un pezzo imparato una volta si riusa
  per obiettivi diversi, invece di rimpararlo da capo a ogni cambio di
  traguardo.
- I costi sono tre: qualcuno deve inventare gli schemi; provarli tutti è
  impossibile; e chi gioca a gesti grossi perde le scorciatoie, perché lo
  schema porta dove porta anche quando bastavano due passi.
- Per questo gli schemi si aggiungono al gioco libero e non lo
  sostituiscono: chi ha in mano tutti e due chiama lo schema quando serve e va
  da solo quando la strada è aperta. Nelle quattro stanze si vede nei numeri:
  chi ha solo gli schemi impara prima di tutti, ma se il traguardo è in un
  punto dove nessuno schema conduce non ci arriva mai. Chi ha schemi e mosse
  parte più in fretta, ma alla lunga resta qualche passo sopra chi ha solo le
  mosse, probabilmente perché gli schemi che portano abbastanza bene gli
  tolgono la voglia di cercare la scorciatoia.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un’opzione {cite}`sutton1999options` è la tripla
  $\omega = \langle I_\omega, \pi_\omega, \beta_\omega\rangle$: insieme di
  avvio, sotto-politica, condizione di terminazione. È a ciclo chiuso
  (sotto-politica e terminazione guardano lo stato); la macro classica,
  sequenza fissa di azioni, ne è la versione cieca, un'opzione semi-markoviana
  che dipende soltanto dai passi trascorsi. L'agente ha due
  livelli, la politica sulle opzioni $\pi_\Omega(\omega \mid s)$ e le
  sotto-politiche $\pi_\omega(a \mid s)$.
- Un'opzione dura $k$ passi con $k$ casuale, quindi il processo visto dall'alto
  è un semi-MDP e l'aggiornamento porta $\gamma^{k}$ al posto di $\gamma$,
  con le ricompense del tratto accumulate e scontate.
- I guadagni sono l’efficienza in campioni (una sotto-politica appresa una
  volta si riusa e si trasferisce) e l’esplorazione, perché avviare opzioni
  copre distanze che una passeggiata di azioni elementari non copre.
- Il costo che si dimentica è la qualità: una politica che sceglie solo fra
  opzioni può essere peggiore di una di sole azioni elementari, perché scavalca
  percorsi più corti. Per questo le opzioni si aggiungono all'insieme delle
  azioni invece di rimpiazzarlo, mantenendo inalterato l'ottimo raggiungibile.
  Raggiungibile, però, non significa raggiunto: un'opzione sufficientemente
  efficace sottrae le visite necessarie a scoprire la traiettoria ottimale. È
  la spiegazione più plausibile del ritardo misurato nelle quattro stanze (a
  mille episodi, 14,0 passi contro 8,6).
- Trovare i sotto-obiettivi è il problema aperto: l’option-critic
  {cite}`bacon2017optioncritic` impara sotto-politiche e terminazioni con un
  teorema del gradiente di policy per le opzioni, chiedendo solo *quante*
  opzioni; le FeUdal Networks {cite}`vezhnevets2017feudal` separano un
  manager lento che fissa obiettivi da un worker che li esegue; l’h-DQN
  {cite}`kulkarni2016hdqn` innesta i due livelli su una ricompensa intrinseca.
```
`````

La gerarchia cambia l'unità di misura del tempo: il livello alto vede una
partita fatta di poche decisioni, e il livello basso ne stabilisce la durata.
Resta aperto il punto su cui l'h-DQN si appoggia: la ricompensa interna che
l'agente si dà quando raggiunge l'obiettivo assegnato. Nelle quattro stanze non
serviva, perché il $-1$ a ogni passo è un segnale continuo; serve dove la
ricompensa dell'ambiente è rada. I premi interni per la novità e per la
sorpresa sono il cuore della {doc}`sezione sull'esplorazione
<esplorazione-e-ricompensa>`.
