# Le reti a energia: da Hopfield a LeCun

L'8 ottobre 2024 l'Accademia reale svedese delle scienze annuncia il premio
Nobel per la fisica: va a John Hopfield e Geoffrey Hinton «per scoperte e
invenzioni fondamentali che rendono possibile l'apprendimento automatico con
reti neurali artificiali». La notizia lascia interdetti parecchi addetti ai
lavori — lo stesso Hinton, raggiunto al telefono in un albergo della
California, si dice sbalordito — e per giorni rimbalza la stessa domanda: che
cosa c'entra la *fisica*? Hopfield e Hinton non hanno scoperto particelle né
misurato onde gravitazionali: hanno costruito reti neurali.

La risposta della giuria è seria, ed è la porta d'ingresso di questa sezione.
Le reti premiate non *assomigliano* a sistemi fisici: si comportano
esattamente come tali. Possiedono una grandezza chiamata **energia**, definita
con la stessa matematica dei materiali magnetici, e la loro dinamica la fa
scendere, come una pallina che rotola verso il fondo di una valle. In questo
quadro *ricordare* significa scivolare in un minimo di energia, e *imparare*
significa scolpire il paesaggio: scavare valli nei punti dove vogliamo che la
rete vada a finire.

E perché ce ne occupiamo proprio qui, a metà del capitolo sui world model?
Nella sezione precedente abbiamo visto agenti che imparano dentro il proprio
sogno, con modelli che provano a *generare* il futuro dettaglio per dettaglio.
La proposta alternativa di LeCun — le architetture JEPA della prossima sezione
— rinuncia a generare e giudica invece la *compatibilità* tra presente e
futuro; ed è scritta da cima a fondo nella lingua delle funzioni di energia.
Questa sezione serve a imparare quella lingua, partendo da dove è nata: una
rete del 1982 che sa fare una cosa sola, ma la sa fare in un modo che colpì
tutti. Ricordare come ricordiamo noi.

## Ricordare per vicinanza, non per indirizzo

La memoria di un computer funziona per **indirizzo**: ogni dato abita in una
casella numerata, e per recuperarlo bisogna conoscere il numero esatto —
sbagli una cifra e ottieni un dato qualsiasi. La memoria umana funziona per
**contenuto**: bastano tre note stonate fischiettate da un passante per farti
riaffiorare l'intera canzone, un profumo per restituirti una cucina di
trent'anni fa, mezza faccia intravista da un autobus per completare nome e
cognome. Non forniamo indirizzi: forniamo *frammenti*, e il ricordo si
completa da solo. I tecnici la chiamano **memoria associativa**.

Nel 1982 John Hopfield — un fisico della materia condensata prestato alla
biologia, dal 1980 al California Institute of Technology — mostra come
costruirne una con neuroni artificiali {cite}`hopfield1982neural`. La sua
mossa è quella di un fisico: notare che un gruppo di neuroni binari, ciascuno
«acceso» o «spento», collegati da pesi simmetrici, è matematicamente identico
a un materiale magnetico in cui ogni atomo ha uno spin che punta in su o in
giù e sente l'influenza dei vicini. E di sistemi così la fisica sa tutto, a
partire dalla domanda giusta: qual è l'energia di ogni configurazione, e
verso dove scende?

```{figure} ../figures/energia-paesaggio.svg
:name: fig-energia-paesaggio
:alt: Paesaggio di energia con tre valli i cui minimi, segnati in teal, sono i ricordi memorizzati; una pallina ocra etichettata «ricordo parziale / rumoroso» parte da un punto alto e una freccia terracotta la accompagna nel fondo della valle più vicina. L'asse verticale è l'energia E, quello orizzontale lo stato della rete.
:width: 92%

Il paesaggio di energia di una rete di Hopfield: ogni ricordo memorizzato è
una valle, e il richiamo è la discesa dallo stato corrotto al minimo più
vicino.
```

La {numref}`fig-energia-paesaggio` contiene, in un solo disegno, tutta l'idea:
i ricordi sono valli, lo stato attuale della rete è una pallina, e la fisica
del sistema — l'energia che può solo scendere — fa il lavoro di richiamo al
posto nostro.

`````{tab} Elementare

Ogni ricordo che la rete ha memorizzato scava una valle nel paesaggio della
figura. Lo stato della rete in un dato momento è una pallina appoggiata da
qualche parte su quel profilo. Dare alla rete un indizio — un ricordo
parziale, o rovinato — significa posare la pallina in un punto alto del
pendio, vicino a una valle ma non sul fondo. Poi non c'è altro da fare: la
pallina rotola, e può soltanto scendere, finché si ferma nel punto più basso
nei paraggi. Se l'indizio somigliava al ricordo B più che agli altri, il
fondo più vicino è proprio la valle di B: arrivarci *è* ricordare, con tutti
i dettagli che l'indizio non conteneva. La melodia stonata del passante ti
deposita sul fianco della valle della canzone giusta, e la discesa fa il
resto: il ricordo non lo *cerchi*, ci *cadi dentro*.

Due avvertenze oneste. Primo: la pallina scende nella valle più *vicina*,
non necessariamente in quella *giusta* — un indizio troppo rovinato può
depositarti sul pendio sbagliato, e da lì si finisce nel ricordo sbagliato
con la stessa naturalezza. Secondo: il paesaggio ha una capienza. Se provi a
scavare troppe valli in poco spazio, i fianchi si fondono e compaiono conche
a metà strada tra due ricordi: «ricordi fantasma» che nessuno ha mai
memorizzato. Una rete con 25 neuroni, come quella che costruiremo tra poco,
regge tre o quattro ricordi: oltre, va in confusione tutta insieme.

`````

`````{tab} Superiore

La rete è un vettore di $N$ neuroni binari $s_i \in \{-1, +1\}$, collegati da
pesi **simmetrici** ($w_{ij} = w_{ji}$) e senza auto-connessioni
($w_{ii} = 0$). A ogni stato $s$ è associata l'energia

$$
E(s) = -\frac{1}{2}\, s^\top W s = -\frac{1}{2} \sum_{i \neq j} w_{ij}\, s_i s_j,
$$

dove $W$ è la matrice dei pesi e la somma percorre tutte le coppie di
neuroni: una coppia collegata da peso positivo abbassa l'energia quando i due
neuroni concordano, e la alza quando discordano (per pesi negativi vale
l'opposto). La dinamica è l'**aggiornamento asincrono**: si sceglie un
neurone $i$, si calcola il suo campo locale $h_i = \sum_j w_{ij} s_j$ e si
pone $s_i \leftarrow \operatorname{sign}(h_i)$, lasciando tutto il resto
fermo. Se il neurone cambia segno, l'energia varia di
$\Delta E = -2\,|h_i| < 0$: **ogni aggiornamento la fa scendere o la lascia
invariata, mai salire**. Poiché gli stati sono in numero finito ($2^N$) ed
$E$ è limitata dal basso, la discesa termina in un punto fisso — un minimo
locale dell'energia. È qui che serve la simmetria dei pesi: senza di essa
non esisterebbe alcuna funzione che la dinamica fa scendere, e la rete
potrebbe girare in tondo per sempre.

Le valli si scolpiscono con la **regola di Hebb**, dal neuropsicologo Donald
Hebb che nel 1949 la propose per le sinapsi biologiche — l'idea che sarebbe
poi stata riassunta nello slogan «i neuroni che si attivano insieme si
legano insieme». Per memorizzare i pattern
$\xi^1, \dots, \xi^P$, ciascuno un vettore di $\pm 1$:

$$
w_{ij} = \frac{1}{N} \sum_{\mu=1}^{P} \xi_i^{\mu}\, \xi_j^{\mu}
\qquad (i \neq j),
$$

dove $\xi_i^{\mu}$ è l'$i$-esimo bit del pattern $\mu$: ogni pattern
rafforza i legami tra i propri bit concordi, e così diventa (con alta
probabilità, se i pattern sono quasi ortogonali) un minimo locale di $E$.
La capienza però è limitata: l'analisi di meccanica statistica di Daniel
Amit, Hanoch Gutfreund e Haim Sompolinsky (1985–1987), con i metodi dei
vetri di spin, fissa la capacità a circa $0{,}138\,N$ pattern — tollerando una
piccola frazione di bit errati nel richiamo. Oltre quella soglia il
recupero non degrada dolcemente: collassa. E anche sotto soglia il
paesaggio contiene minimi non richiesti: gli opposti $-\xi^{\mu}$ di ogni
pattern e miscele spurie di tre o più ricordi.

`````

## Una memoria che si ripara da sola, in poche righe

Tutto questo si può toccare con mano. Il codice che segue costruisce una rete
di Hopfield completa in NumPy: tre lettere stilizzate su una griglia 5×5
(quindi $N = 25$ neuroni), la regola di Hebb per i pesi, la funzione di
energia e il richiamo per aggiornamento asincrono a partire da una versione
corrotta, con il 24% dei pixel invertiti.

```python
import numpy as np

rng = np.random.default_rng(42)

# Tre lettere stilizzate 5x5: '#' = pixel acceso (+1), '.' = spento (-1)
LETTERE = {
    "T": ["#####",
          "..#..",
          "..#..",
          "..#..",
          "..#.."],
    "L": ["#....",
          "#....",
          "#....",
          "#....",
          "#####"],
    "X": ["#...#",
          ".#.#.",
          "..#..",
          ".#.#.",
          "#...#"],
}

def a_vettore(disegno):
    """Da lista di stringhe a vettore di +1/-1."""
    return np.array([1 if c == "#" else -1
                     for riga in disegno for c in riga])

def a_righe(s):
    """Da vettore di +1/-1 a cinque stringhe stampabili."""
    griglia = np.where(s == 1, "#", ".").reshape(5, 5)
    return ["".join(riga) for riga in griglia]

pattern = np.array([a_vettore(d) for d in LETTERE.values()])   # forma (3, 25)
N = pattern.shape[1]

# Regola di Hebb: somma dei prodotti esterni, diagonale a zero
W = (pattern.T @ pattern) / N
np.fill_diagonal(W, 0.0)

def energia(s):
    """E(s) = -1/2 s^T W s (soglie nulle)."""
    return -0.5 * s @ W @ s

def richiama(s, max_passate=10):
    """Aggiornamento asincrono fino a un punto fisso (minimo locale)."""
    s = s.copy()
    for _ in range(max_passate):
        cambiato = False
        for i in rng.permutation(N):        # un neurone alla volta
            campo = W[i] @ s                # campo locale sul neurone i
            nuovo = np.sign(campo) if campo != 0 else s[i]
            if nuovo != s[i]:
                s[i] = nuovo
                cambiato = True
        if not cambiato:                    # nessun cambiamento: stato stabile
            break
    return s

def corrompi(s, quanti=6):
    """Inverte 'quanti' bit scelti a caso (6 su 25 = 24%)."""
    s = s.copy()
    indici = rng.choice(N, size=quanti, replace=False)
    s[indici] = -s[indici]
    return s

for nome, disegno in LETTERE.items():
    originale = a_vettore(disegno)
    rumoroso = corrompi(originale)          # 24% dei pixel invertiti
    recuperato = richiama(rumoroso)
    esito = ("recuperato" if np.array_equal(recuperato, originale)
             else "NON recuperato")
    print(f"{nome}:  E = {energia(rumoroso):+.2f} "
          f"-> {energia(recuperato):+.2f}  ({esito})")
    print("   corrotto   richiamato")
    for r1, r2 in zip(a_righe(rumoroso), a_righe(recuperato)):
        print(f"   {r1}      {r2}")
    print()
```

Eseguendolo, tutte e tre le lettere riemergono intatte dalle loro versioni
sfigurate. Per la T, ad esempio, l'energia scende da $-2{,}08$ dello stato
corrotto a $-11{,}20$ del pattern richiamato:

```text
T:  E = -2.08 -> -11.20  (recuperato)
   corrotto   richiamato
   #.###      #####
   ..#..      ..#..
   #.#.#      ..#..
   .##..      ..#..
   .###.      ..#..
```

Vale la pena notare tre dettagli del codice, perché sono la teoria in forma
eseguibile: la diagonale di $W$ è azzerata (niente auto-connessioni), i
neuroni si aggiornano *uno alla volta* in ordine casuale (la discesa
asincrona che non fa mai salire $E$), e il ciclo si ferma quando nessun
neurone vuole più cambiare — un minimo locale, cioè un ricordo. Onestà
statistica: con tre pattern su 25 neuroni siamo proprio al limite della
capienza $0{,}138 \times 25 \approx 3{,}4$; le tre lettere sono state scelte
quasi ortogonali tra loro, e con corruzioni casuali diverse dal seme fissato
il recupero perfetto riesce circa nove volte su dieci. Nelle altre, la
pallina finisce in un minimo spurio: il limite non è nel codice, è nella
matematica.

## Alzare la temperatura: le macchine di Boltzmann

La pallina di Hopfield ha un difetto di fabbrica: può solo scendere. Se
l'indizio la deposita sul pendio sbagliato, finisce nella valle sbagliata — o
in un ricordo fantasma — e da lì non esce più. E c'è un limite più profondo:
la rete *ricorda*, ma non *inventa*; i suoi neuroni coincidono con i pixel
del pattern, senza spazio per rappresentazioni interne. A metà anni Ottanta
Geoffrey Hinton e Terrence Sejnowski, con David Ackley, propongono la
**macchina di Boltzmann** {cite}`ackley1985learning`, che aggiunge alla rete
di Hopfield esattamente due ingredienti: la **temperatura** e i **neuroni
nascosti**. Il nome è un omaggio a Ludwig Boltzmann, uno dei padri della meccanica
statistica: come vedremo, all'equilibrio la rete visita gli stati con le
stesse probabilità con cui un sistema fisico caldo visita le proprie
configurazioni.

`````{tab} Elementare

La temperatura è una scossa. Immagina la pallina ferma in una conca che non
è la valle giusta: se il paesaggio resta immobile, non ne uscirà mai. Ora
scuoti tutto, come una biglia in una scatola da scarpe: con scossoni forti
la biglia salta fuori anche dalle valli profonde e gira dappertutto; con
scossoni deboli resta confinata nei fondovalle. Il trucco è scuotere forte
all'inizio e sempre più piano — è la mossa del fabbro, che scalda il metallo
e lo lascia raffreddare lentamente perché gli atomi trovino da soli la
disposizione migliore — così la biglia ha modo di uscire dalle conche
mediocri finché può, e di assestarsi in una valle profonda quando la calma
torna.

I neuroni nascosti, invece, sono taccuini interni: neuroni che non
corrispondono a nessun pixel del dato ma servono alla rete per annotare
regolarità sue («qui c'è una riga verticale», «questi due angoli vanno
insieme»). E l'apprendimento diventa un confronto tra due modi di stare al
mondo: nella fase di *veglia* la macchina osserva i dati veri e registra
quali coppie di neuroni si accendono insieme; nella fase di *sogno* viene
lasciata libera di produrre stati per conto suo, e si registra la stessa
cosa. Poi i pesi si ritoccano per rinforzare ciò che accade da svegli più
che in sogno, e indebolire il contrario. Si smette quando i sogni sono
indistinguibili dalla veglia: a quel punto la macchina si è fatta un modello
dei dati. Il guaio, come vedremo, è che sognare «per bene» richiedeva tempi
biblici.

`````

`````{tab} Superiore

Nella macchina di Boltzmann l'aggiornamento del neurone $i$ diventa
stocastico:

$$
P(s_i = +1) = \sigma\!\left(\frac{2 h_i}{T}\right)
= \frac{1}{1 + e^{-2 h_i / T}},
$$

dove $h_i = \sum_j w_{ij} s_j$ è il campo locale, $\sigma$ la sigmoide già
incontrata nel capitolo sulle reti neurali e $T > 0$ la temperatura. Per
$T \to 0$ si ritrova l'aggiornamento deterministico di Hopfield; per $T$
grande la rete accetta spesso anche mosse che *alzano* l'energia, e può
quindi evadere dai minimi locali (abbassare $T$ gradualmente è la *ricottura
simulata*). All'equilibrio termico la rete visita gli stati secondo la
distribuzione di Boltzmann–Gibbs

$$
P(s) = \frac{e^{-E(s)/T}}{Z},
\qquad
Z = \sum_{s'} e^{-E(s')/T},
$$

dove $Z$ — la **funzione di partizione** — somma su tutti i $2^N$ stati
possibili: è lei che rende la rete un vero modello probabilistico, ed è lei
che costerà carissima. I neuroni si dividono in **visibili** (dove si
presentano i dati) e **nascosti** (variabili latenti che catturano
regolarità di ordine superiore). L'apprendimento massimizza la
verosimiglianza dei dati sui visibili, e il gradiente ha una forma di
contrasto di rara eleganza:

$$
\Delta w_{ij} \;\propto\; \langle s_i s_j \rangle_{\text{dati}}
- \langle s_i s_j \rangle_{\text{modello}},
$$

dove il primo termine è la correlazione media tra i neuroni $i$ e $j$ con i
visibili bloccati sui dati (fase positiva, la «veglia») e il secondo la
stessa correlazione con la rete libera di campionare da sé (fase negativa,
il «sogno»). Il problema pratico è tutto nel secondo termine: stimarlo
richiede di portare all'equilibrio una catena di Markov su uno spazio
esponenziale, per *ogni* passo di gradiente. È questo doppio ciclo a rendere
l'algoritmo originale inutilizzabile oltre i problemi giocattolo.

`````

La via d'uscita arriva quasi vent'anni dopo, ed è di nuovo di Hinton: la
**contrastive divergence** {cite}`hinton2002training`. L'idea è rinunciare al
sogno completo: invece di far girare la catena fino all'equilibrio, la si fa
partire *dai dati* e la si ferma dopo un solo passo (o pochi), usando quel
sogno appena abbozzato come surrogato della fase negativa. Il gradiente che
ne esce è distorto, ma in pratica funziona — soprattutto sulle **macchine di
Boltzmann ristrette** (RBM), la variante in cui i collegamenti esistono solo
tra strato visibile e strato nascosto, così che ogni strato si campiona in
blocco, in parallelo. Fu proprio la coppia RBM più contrastive divergence,
impilata strato su strato, a rimettere in moto il deep learning a metà anni
Duemila, quando addestrare reti profonde sembrava impossibile: un ruolo
storico che va riconosciuto con onestà, insieme al suo epilogo — di lì a
pochi anni ReLU, GPU e dataset più grandi avrebbero reso quel pre-training
superfluo, e oggi le RBM non si usano quasi più. Il *linguaggio* con cui
erano scritte, invece, è vivo e vegeto: è il tema del prossimo paragrafo.

## L'energia come compatibilità: la cornice di LeCun

Nel 2006 Yann LeCun e i suoi collaboratori pubblicano un lungo tutorial
{cite}`lecun2006tutorial` che compie il gesto inverso rispetto a Hopfield:
non costruire una rete a energia, ma mostrare che *quasi ogni modello di
apprendimento può essere letto come una funzione di energia*. La ricetta è
di una generalità spiazzante. Si prende una funzione $E(x, y)$ che assegna un
numero a ogni coppia formata da un input $x$ e una possibile risposta $y$:
energia bassa se la coppia è **compatibile** (questa immagine con questa
etichetta, questa frase con questa traduzione, questo presente con questo
futuro), alta se non lo è. Rispondere significa cercare la $y$ che rende
l'energia minima; imparare significa dare forma alla superficie —
**abbassare** l'energia delle coppie giuste e **alzarla**, o tenerla alta,
per quelle sbagliate. La memoria di Hopfield è il caso speciale in cui $x$ è
il ricordo corrotto e $y$ quello completo; la macchina di Boltzmann è il caso
in cui sull'energia si costruisce una probabilità. Ma la cornice è molto più
larga, e contiene una liberazione.

`````{tab} Elementare

Pensa a un buttafuori davanti a una festa a coppie. Il suo mestiere è
giudicare la coppia che ha davanti: questi due stanno bene insieme, passano;
questi due no. Nota che cosa *non* gli serve: non deve conoscere tutte le
persone della città, né compilare una classifica completa di tutti gli
abbinamenti possibili con le percentuali esatte che sommano a cento. Gli
basta un giudizio di compatibilità, coppia per coppia. Un modello a energia
è questo buttafuori: dare la *probabilità* di ogni risposta possibile — come
fanno i modelli probabilistici — è un lavoro immane, perché per dire «70%»
su una risposta devi aver messo in conto *tutte* le altre; dire «questa
coppia sì, quella no» è enormemente più economico, e per moltissimi compiti
basta e avanza.

C'è però un pericolo, e ha un nome preciso: il **collasso**. Immagina il
buttafuori pigro che ha scoperto la scorciatoia perfetta: dire sempre sì.
Chiunque si presenti, passa. Nessuna coppia si lamenta mai — e il suo
giudizio non vale più niente. Se durante l'addestramento premi il modello
solo quando dà energia bassa alle coppie giuste, la soluzione più comoda è
dare energia bassa *a tutto*. I rimedi sono due, e li ritroveremo: fargli
vedere anche coppie sbagliate e pretendere che le respinga (allenarlo *per
contrasto*), oppure costruire la porta così stretta che far passare tutti
gli sia fisicamente impossibile (vincolarlo *per costruzione*).

`````

`````{tab} Superiore

Un modello a energia (*energy-based model*, EBM) è una funzione
$E_\theta(x, y)$ a valori reali, con parametri $\theta$, che misura quanto
$x \in \mathcal{X}$ e $y \in \mathcal{Y}$ siano compatibili: valori bassi per
le coppie compatibili, alti per le altre. L'inferenza è un problema di
ottimizzazione:

$$
\hat{y} = \arg\min_{y \in \mathcal{Y}} E_\theta(x, y),
$$

dove $\hat{y}$ è la risposta predetta: nessuna somma su $\mathcal{Y}$, solo
una ricerca del minimo. Un modello probabilistico si ottiene come caso
particolare tramite la distribuzione di Gibbs:

$$
P_\theta(y \mid x) = \frac{e^{-\beta E_\theta(x, y)}}
{\displaystyle\int_{\mathcal{Y}} e^{-\beta E_\theta(x, y')} \, dy'},
$$

dove $\beta > 0$ è una temperatura inversa e il denominatore è la funzione
di partizione $Z_\theta(x)$ — l'integrale (o la somma) su *tutte* le
risposte possibili. Quando $\mathcal{Y}$ è grande o continuo e ad alta
dimensione ($y$ = un'immagine, un video, una frase), $Z_\theta(x)$ è
intrattabile: è il muro contro cui si è schiantata la macchina di Boltzmann.
La tesi del tutorial è che per decidere, ordinare o pianificare serve solo
l'$\arg\min$, che di $Z$ non ha alcun bisogno: rinunciare alla
normalizzazione non è una perdita ma un vantaggio computazionale.

Il prezzo della libertà è il **collasso**. Se la loss $\mathcal{L}$ si
limitasse ad abbassare l'energia sulle coppie del training set,
$\min_\theta \sum_k E_\theta(x_k, y_k)$, la soluzione banale sarebbe
un'energia costante e bassa ovunque: superficie piatta, modello inutile. Le
contromisure si dividono in due famiglie. I **metodi contrastivi** alzano
esplicitamente l'energia su coppie sbagliate $(x, \tilde{y})$ — con loss a
margine del tipo
$\mathcal{L} = E_\theta(x, y) + \max\!\big(0,\, m - E_\theta(x, \tilde{y})\big)$,
dove $m$ è il margine richiesto tra coppia giusta e sbagliata; la
contrastive divergence appartiene a questa famiglia, e in alta dimensione
tutti i suoi membri soffrono dello stesso male: i controesempi non bastano
mai a puntellare un'intera superficie. I **metodi regolarizzati o
architetturali** impediscono invece il collasso per costruzione, limitando
il *volume* dello spazio a bassa energia: colli di bottiglia sulle variabili
latenti, vincoli di sparsità, termini che impongono varianza alle
rappresentazioni. È la famiglia su cui LeCun scommette per i world model —
e la prossima sezione mostra perché.

`````

## Tre ponti

Il lessico appena costruito non è un pezzo da museo: attraversa mezza AI
contemporanea, e conviene fissare tre ponti prima di proseguire.

Il primo guarda all'indietro, al capitolo sui **modelli di diffusione**: lo
*score* che vi abbiamo incontrato — il gradiente della log-densità dei dati,
$\nabla_x \log p(x)$ — è esattamente $-\nabla_x E(x)$ per un'energia mai
normalizzata, e generare per denoising progressivo è, alla lettera, scendere
lungo un paesaggio di energia a partire dal rumore.

Il secondo porta alla **prossima sezione**: la JEPA di LeCun è
un'architettura a energia senza normalizzazione — una $E(x, y)$ che giudica
la compatibilità tra un pezzo di mondo osservato e uno da predire, difesa
dal collasso con metodi regolarizzati, non contrastivi. Tutto il discorso
che faremo lì poggia sulle definizioni di questa pagina.

Il terzo ponte è una curiosità documentata che chiude un cerchio
sorprendente: le reti di Hopfield «moderne», con stati continui ed energia
riprogettata, hanno capacità esponenziale e una regola di aggiornamento che
coincide con la formula dell'attenzione dei Transformer
{cite}`ramsauer2021hopfield` — il paper si intitola, non a caso, *Hopfield
Networks is All You Need*, e rilegge la scaled dot-product attention di
{cite}`vaswani2017attention` (capitolo sui Transformer) come il richiamo di
una memoria associativa: le query sono indizi, le key–value i ricordi. La
memoria del 1982 e il meccanismo che regge i modelli di linguaggio parlano,
matematicamente, la stessa lingua. Forse il Nobel a Hopfield e Hinton non
premiava soltanto il passato.

```{admonition} Da ricordare
:class: important
- La **rete di Hopfield** {cite}`hopfield1982neural` è una memoria
  associativa: i ricordi sono minimi di un'energia
  $E(s) = -\tfrac{1}{2} s^\top W s$, la regola di Hebb scava le valli e
  l'aggiornamento asincrono — che non aumenta mai $E$ — completa i ricordi
  corrotti scendendo nel minimo più vicino. Capienza: circa $0{,}138\,N$
  pattern, poi il richiamo collassa.
- La **macchina di Boltzmann** {cite}`ackley1985learning` aggiunge
  temperatura (aggiornamenti stocastici per evadere dai minimi sbagliati) e
  neuroni nascosti; impara per contrasto tra fase di veglia (dati) e fase di
  sogno (campioni del modello). Era lentissima: la **contrastive
  divergence** {cite}`hinton2002training` la rese praticabile, e le RBM
  contribuirono a far ripartire il deep learning a metà anni Duemila.
- La cornice dell'**energy-based learning** {cite}`lecun2006tutorial`: ogni
  modello è una $E(x, y)$ che misura la compatibilità tra input e risposta;
  inferire è $\arg\min_y E$, imparare è abbassare l'energia delle coppie
  giuste e alzarla sulle sbagliate. I modelli probabilistici sono il caso
  particolare normalizzato — e la funzione di partizione $Z$ è proprio il
  costo che conviene evitare.
- Il pericolo è il **collasso** (energia bassa ovunque); i rimedi sono
  **contrastivi** (controesempi respinti) o **regolarizzati/architetturali**
  (poco volume a bassa energia per costruzione): la JEPA della prossima
  sezione sceglie i secondi.
- La regola di aggiornamento delle reti di Hopfield moderne coincide
  matematicamente con l'attenzione dei Transformer
  {cite}`ramsauer2021hopfield`: il linguaggio delle energie non è storia, è
  attualità.
```
