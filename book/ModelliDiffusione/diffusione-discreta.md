# Quando lo stato è fatto di simboli: diffondere il testo

Chi risolve un sudoku non procede da sinistra a destra e dall'alto in basso.
Guarda la griglia, trova la casella che il resto costringe di più, ci scrive il
numero, e da quel numero ricava altre costrizioni; poi cerca la prossima
casella più determinata, e avanti così. L'ordine lo decide la griglia, un passo
alla volta, e cambia da partita a partita.

Un modello linguistico ordinario fa il contrario: scrive una parola dopo
l'altra, da sinistra a destra, e quello che ha scritto non lo tocca più.
Funziona benissimo e ha due conseguenze che si pagano tutti i giorni: per
scrivere mille parole servono mille passaggi nella rete, uno per parola; e per
riempire un buco in mezzo a un testo già scritto bisogna far finta che il
seguito non ci sia.

Serve un altro modo di rovinare, e la strada che l'ha trovato assomiglia al
sudoku. Nel sudoku, del resto, la regola è tutta lì: in una riga ogni cifra può
comparire una volta sola, quindi appena se ne scrive una certe caselle restano
con una sola possibilità, e sono quelle che conviene riempire per prime.

## Rovinare un simbolo: cancellarlo

`````{tab} Elementare

Se non si può aggiungere un pizzico di rumore a una parola, che cosa le si può
fare che sia gradualmente peggiorabile? Tre risposte sono state provate, e una
sola ha funzionato bene.

*Sostituirla con un'altra a caso.* Si prende una parola e con una certa
probabilità la si scambia con una qualunque del vocabolario. Aumentando quella
probabilità si arriva a un testo del tutto casuale. Funziona, ed è scomodo: chi
deve ripulire non sa quali parole siano state toccate e quali no, e deve
decidere anche questo.

*Spostarla verso i vicini.* Ha senso solo se i simboli hanno un ordine, come i
livelli di grigio di un pixel quantizzato, e non ne ha nessuno per un
vocabolario, dove «gatto» non è più vicino a «gatta» che a «treno».

*Cancellarla.* Al posto della parola si mette un segnaposto che dice «qui c'era
qualcosa e non lo sai». Aumentando il tempo si cancella una frazione crescente
di parole, finché non resta che una fila di segnaposti. È la strada che ha
vinto, per una ragione che si vede subito: **chi ripulisce sa esattamente dove
guardare**. Le parole rimaste sono vere e non si toccano; il lavoro è
ricostruire quelle cancellate, e il modello sa quali sono.

C'è di più, ed è una sorpresa piacevole. Il compito «ecco un testo con dei buchi,
riempili» è esattamente quello su cui una generazione di modelli linguistici è
stata addestrata a partire dal 2018, e nessuno all'epoca la chiamava diffusione.
La differenza fra quei modelli e questi sta in due dettagli: la frazione di
parole cancellate, che là era fissa e qui viene sorteggiata fra zero e tutto, e
il fatto che qui esiste una procedura per **generare** partendo da una fila di
soli segnaposti, invece di limitarsi a riempire i buchi di un testo dato.

`````

`````{tab} Superiore

Il processo in avanti si specifica con una matrice di transizione
$\mathbf{Q}_t$ sul vocabolario, che agisce indipendentemente su ogni posizione:
$q(x_t \mid x_{t-1}) = \operatorname{Cat}(x_t; \mathbf{Q}_t\,
\mathbf{e}_{x_{t-1}})$. È una catena di Markov sui simboli, esattamente nel
senso della {doc}`sezione di matematica </Matematica/catene-di-markov>`, e la
scelta di $\mathbf{Q}_t$ definisce la famiglia:

- **uniforme**: con probabilità $\beta_t$ il simbolo diventa uno qualsiasi del
  vocabolario. La stazionaria è l'uniforme.
- **ordinale**: la transizione favorisce i simboli vicini, sensata solo su
  alfabeti con una metrica (pixel quantizzati, note musicali).
- **assorbente**: con probabilità $\beta_t$ il simbolo diventa il simbolo
  speciale $\texttt{[MASK]}$, dal quale non si esce più. La stazionaria è la
  sequenza di soli $\texttt{[MASK]}$.

La terza è quella che si è imposta, e la ragione è strutturale: lo stato
$\mathbf{x}_t$ **dichiara** quali posizioni sono state corrotte, quindi il
posteriore $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ è una delta su
tutte le posizioni non mascherate, che restano fisse, e quello che resta da
imparare vive sulle sole posizioni mascherate. Con la transizione uniforme
nessuna posizione è al sicuro: il modello deve anche inferire *dove* è avvenuta
la corruzione, e l'obiettivo variazionale è molto più rumoroso. La
fattorizzazione per posizione, da sola, non discrimina fra le tre varianti:
viene dall'indipendenza per posizione del processo in avanti, non dalla scelta
di $\mathbf{Q}_t$ {cite}`austin2021structured`.

Il nucleo di perturbazione, per la variante assorbente, ha forma chiusa e
indipendente per posizione:

$$
q(x_t^{(i)} \mid x_0^{(i)}) =
\alpha_t\,\delta_{x_0^{(i)}} + (1-\alpha_t)\,\delta_{\texttt{[MASK]}},
$$

cioè ogni posizione sopravvive con probabilità $\alpha_t$ e viene mascherata
con probabilità $1-\alpha_t$, indipendentemente dalle altre. È l'analogo
esatto del campionamento *simulation-free* del caso continuo: per avere
$\mathbf{x}_t$ si sorteggia una maschera e basta.

`````

## L'obiettivo si riduce a una cosa già nota

`````{tab} Elementare

Si prende una frase, si sorteggia quanta parte cancellarne, si cancella, si
chiede al modello di indovinare le parole mancanti, e lo si punisce in
proporzione a quanto sbaglia. Nient'altro: una scelta a più risposte, ripetuta
su ogni posizione cancellata.

Sviluppando il conto che dice quanto il modello è bravo, e semplificando tutto
quello che si semplifica, resta esattamente questo e nient'altro. Non ci sono
rumori gaussiani, non c'è nessun punteggio da imparare, non c'è nessuna
equazione differenziale: c'è la somma degli errori sulle posizioni cancellate,
ciascuno moltiplicato per un peso.

Il peso però conta, e dice una cosa sensata. Quando quasi tutto è cancellato,
indovinare è quasi impossibile e ogni singola risposta vale poco; quando è
cancellata una parola sola, il contesto la determina quasi del tutto e
sbagliarla è grave. Il peso che il conto produce dà esattamente questo
andamento.

`````

`````{tab} Superiore

Per la variante assorbente il limite variazionale collassa in una forma
notevolmente semplice. Detto $\mathcal{M}_t$ l'insieme delle posizioni
mascherate a tempo $t$,

$$
\mathcal{L} = \mathbb{E}_{t\sim\mathcal{U}[0,1]}\,
\mathbb{E}_{\mathbf{x}_t}\left[
\frac{-\alpha_t'}{1-\alpha_t}
\sum_{i\in\mathcal{M}_t}
-\log p_\theta\big(x_0^{(i)}\mid \mathbf{x}_t\big)\right],
$$

cioè una **cross-entropia sulle sole posizioni mascherate**, pesata da un
fattore che dipende solo dal programma di mascheramento
{cite}`sahoo2024simple,shi2024simplified`. Il segno merita un secondo: il
programma scende da $\alpha_0=1$ a $\alpha_1=0$, quindi $\alpha_t'$ è negativa
e il coefficiente $-\alpha_t'/(1-\alpha_t)$ è positivo, come deve essere una
perdita fatta di termini positivi. Non compaiono né il punteggio né una
divergenza fra gaussiane: il modello risolve una classificazione sul
vocabolario, posizione per posizione.

Da qui il legame con il **modellamento mascherato del linguaggio** alla BERT
{cite}`devlin2019bert`: a meno del peso, è la stessa somma valutata a un solo
valore di $t$ invece che su tutti. La corrispondenza vale a meno di un dettaglio
che conta. BERT sceglie il $15\%$ delle posizioni, ma di quelle sostituisce con
$\texttt{[MASK]}$ soltanto i quattro quinti: un decimo lo rimpiazza con un
simbolo a caso, un decimo lo lascia intatto, e la perdita si calcola su tutte e
tre le specie. La frazione davvero cancellata è quindi il $12\%$, e fra le
posizioni da indovinare ce ne sono che non sono buchi. Un modello di diffusione
mascherata è un BERT addestrato su **tutte** le frazioni di maschera, con la
cancellazione sempre applicata, più una procedura di campionamento che lo
trasforma in un generatore: l'equivalenza fra i due obiettivi, a meno dei pesi,
è dimostrata in {cite}`austin2021structured`, e rende disponibile un decennio di
lavoro sull'addestramento mascherato.

Esiste anche una formulazione a tempo continuo che ricalca fedelmente il caso
gaussiano, con al posto del gradiente della log-densità il **rapporto fra
probabilità di stati vicini**, $p_t(\mathbf{y})/p_t(\mathbf{x})$ per
$\mathbf{y}$ che differisce da $\mathbf{x}$ in una posizione
{cite}`lou2024discrete`. Quel rapporto è la controparte discreta del punteggio
(dove non si può derivare, si divide), il processo è una catena di Markov a
tempo continuo con la sua matrice generatrice, e l'obiettivo che lo stima è una
divergenza di Bregman anziché una quadratica. La struttura è la stessa; cambia
l'aritmetica.

`````

## Il prezzo del parallelismo

Il vantaggio annunciato di questa famiglia è che le posizioni si possono
riempire **in parallelo** invece che una per volta. Il vantaggio è reale e ha
un prezzo esatto, che si misura su un linguaggio abbastanza piccolo da poterne
enumerare tutte le frasi.

```python
import itertools
from collections import Counter
import numpy as np

# Un linguaggio di quattro parole: tre bit, e il numero di uni deve essere
# pari. Ogni bit da solo e' cinquanta e cinquanta, ogni coppia di bit e'
# indipendente, ma i tre insieme sono legatissimi: due qualsiasi decidono
# il terzo.
L = 3
LINGUA = [w for w in itertools.product((0, 1), repeat=L) if sum(w) % 2 == 0]
print(LINGUA)                # -> [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

MASCHERA = -1

def probabilita_di_uno(stato, i):
    """p(bit i = 1 | quello che si e' gia' scoperto), per enumerazione."""
    compatibili = [w for w in LINGUA
                   if all(s == MASCHERA or w[j] == s
                          for j, s in enumerate(stato))]
    return sum(1 for w in compatibili if w[i] == 1) / len(compatibili)

def campiona(passi, n=20000):
    rng = np.random.default_rng(0)
    fuori = []
    for _ in range(n):
        stato = [MASCHERA] * L
        ordine = list(rng.permutation(L))
        # le posizioni si scoprono a gruppi: `passi` gruppi in tutto
        quanti = [L // passi + (1 if k < L % passi else 0) for k in range(passi)]
        for q in quanti:
            gruppo = [ordine.pop() for _ in range(q)]
            # tutte insieme, ciascuna dalla propria probabilita' condizionata
            scelte = {i: int(rng.random() < probabilita_di_uno(stato, i))
                      for i in gruppo}
            for i, v in scelte.items():
                stato[i] = v
        fuori.append(tuple(stato))
    return fuori

for passi in (1, 2, 3):
    conta = Counter(campiona(passi))
    n = sum(conta.values())
    valide = sum(v for w, v in conta.items() if sum(w) % 2 == 0) / n
    freq = sorted(v / n for v in conta.values())
    print(f"passi {passi}: valide {valide:.3f}, risultati diversi"
          f" {len(conta)}, dal {freq[0]:.3f} al {freq[-1]:.3f}")
# -> passi 1: valide 0.499, risultati diversi 8, dal 0.123 al 0.129
# -> passi 2: valide 1.000, risultati diversi 4, dal 0.249 al 0.252
# -> passi 3: valide 1.000, risultati diversi 4, dal 0.249 al 0.252
```

`````{tab} Elementare

Con tre caselle, che possono valere zero o uno, le combinazioni possibili sono
otto; la regola del numero pari di uni ne ammette quattro, e sono quelle
stampate. Guardato un bit alla volta non si vede niente: ciascuno è cinquanta e
cinquanta, e anche due qualsiasi non sanno niente l'uno dell'altro. Il legame
c'è solo fra tutti e tre insieme, e si dice in cinque parole: **due qualsiasi
decidono il terzo**. Se le prime due sono uno e zero, la terza deve essere uno,
se no gli uni sono dispari.

Riempire le tre posizioni **in un colpo solo**, ciascuna sorteggiata dalla
propria probabilità, è come tirare tre monetine: esce una qualunque delle otto
combinazioni, e infatti il conto le trova tutte e otto con la stessa frequenza,
fra $0{,}123$ e $0{,}129$. Metà delle parole generate non appartiene al
linguaggio: il modello ha dimenticato la regola. Non ha sbagliato nessun conto,
ogni singola probabilità era giusta; ha sbagliato a usarle insieme, perché
scriverne una cambia le altre e il colpo solo non lo lascia succedere.

Con **due passi** il problema sparisce, e non per fortuna. Le prime due caselle
si possono scoprire insieme proprio perché, prese in due, non hanno niente da
dirsi: tirarle nello stesso momento è lecito. È la terza che dipende da tutte e
due, e quando tocca a lei le altre sono già scritte. Risultato: solo parole
valide, e un quarto ciascuna.

Questo è il compromesso della diffusione sui simboli, e vale ovunque. Il numero
di passi che servono non dipende dalla lunghezza del testo ma da **quanto le
parti si condizionano a vicenda**. Dove il testo è prevedibile si può scrivere
molto in parallelo; dove ogni parola cambia il senso delle altre, i passi
tornano tanti. Un modello che promette mille parole in dieci passaggi sta
scommettendo che il testo sia in gran parte prevedibile, ed è una scommessa che
qualche volta si perde.

E c'è un modo di scegliere quali posizioni scoprire per prime: quelle su cui il
modello ha meno dubbi. Sono le caselle che il resto ha già quasi deciso, cioè
esattamente la mossa con cui si comincia un sudoku.

Il risparmio, poi, non è pieno. Chi scrive una parola alla volta si tiene in
mano il lavoro già fatto sulle parole precedenti e non lo rifà; qui, a ogni
passo, si riguarda la frase intera da capo. Dieci passi su mille parole
convengono, perché sono dieci letture invece di mille; cento passi su cento
parole non convengono affatto.

`````

`````{tab} Superiore

Il fenomeno è la **fattorizzazione indipendente del passo parallelo**. Il
modello stima le marginali condizionate $p_\theta(x^{(i)}\mid\mathbf{x}_t)$ per
ogni posizione mascherata, e scoprirne $k$ insieme equivale a campionare dal
prodotto

$$
\prod_{i\in G} p_\theta\big(x^{(i)}\mid\mathbf{x}_t\big)
\;\neq\;
p_\theta\big(\{x^{(i)}\}_{i\in G}\mid\mathbf{x}_t\big) ,
$$

che coincide con la congiunta solo se le posizioni del gruppo sono
condizionatamente indipendenti dato $\mathbf{x}_t$. Nel linguaggio di parità non
lo sono, e il risultato è la distribuzione prodotto delle marginali, cioè
l'uniforme su $\{0,1\}^3$: il conto sul linguaggio di parità la mostra, con
$0{,}499$ di parole valide e otto stati equifrequenti invece di quattro.

La quantità che governa l'errore è la **multi-informazione** del gruppo
scoperto insieme, condizionata a ciò che è già noto, cioè la divergenza fra la
congiunta e il prodotto delle sue marginali,

$$
\mathrm{TC}\big(G \mid \mathbf{x}_t\big) =
\sum_{i\in G} H\big(x^{(i)} \mid \mathbf{x}_t\big)
- H\big(\{x^{(i)}\}_{i\in G} \mid \mathbf{x}_t\big) .
$$

Scoprire in parallelo è esatto quando quella quantità è nulla, e l'errore
cresce con essa. Per due sole posizioni coincide con l'informazione mutua; da
tre in su non si riduce a nessuna somma di informazioni mutue a coppie, e il
linguaggio di parità lo mostra nel modo più netto: ogni coppia di bit ha
informazione mutua esattamente nulla, e le tre posizioni insieme hanno
multi-informazione di un bit. È la ragione per cui due passi bastano e uno no.
Ne segue la strategia usata in pratica, che è adattiva: a ogni passo si
scoprono le posizioni su cui il modello è **più sicuro** (entropia più bassa),
perché lì la dipendenza residua dalle altre tende a essere minore
{cite}`ghazvininejad2019mask`. È il sudoku dell'apertura, tradotto in un
criterio.

Le conseguenze pratiche, in ordine di quanto pesano:

- **Nessun riuso della cache.** Un modello autoregressivo riusa le
  rappresentazioni delle posizioni precedenti a costo zero; qui ogni passo
  ricalcola tutta la sequenza, quindi il costo per passo è quello di un
  passaggio intero. Il guadagno esiste solo se i passi sono molti meno delle
  posizioni.
- **Verosimiglianza solo come limite.** Si ottiene un bound variazionale e non
  il valore esatto, quindi i confronti di perplessità con i modelli
  autoregressivi vanno letti sapendo che si sta confrontando un limite con un
  valore.
- **Riempimento gratuito.** Completare un buco in mezzo a un testo, o generare
  rispettando vincoli su posizioni sparse, è il caso normale invece che un
  adattamento: si parte da uno stato in cui quelle posizioni sono già scritte.
- **Calcolo regolabile a piacere.** Lo stesso modello si può usare con dieci
  passi o con cento, scambiando qualità e latenza al momento della richiesta,
  senza riaddestrare.

`````

## Dove sta questa famiglia

`````{tab} Elementare

I modelli linguistici a diffusione esistono e sono addestrati su scala vera.
Dove restano indietro rispetto a chi scrive una parola alla volta (si chiamano
modelli autoregressivi) è la generazione lunga, ed è esattamente il posto in
cui il conto sul linguaggio a tre bit dice che devono restare indietro: nel
linguaggio una parola dell'inizio può decidere una parola della fine, e ogni
passo parallelo paga una parte di quel legame.

Dove la famiglia ha già un vantaggio chiaro è nei compiti in cui il testo va
riempito invece che scritto di seguito: completare un modulo, correggere un
pezzo di codice in mezzo a un file, generare rispettando vincoli sparsi. Lì la
scrittura da sinistra a destra è un impedimento, e il sudoku è il modo giusto
di procedere.

E gli stessi metodi si applicano a qualunque cosa sia fatta di simboli e abbia
una struttura da rispettare. Le molecole sono reticoli di atomi legati fra
loro, le proteine sono sequenze di amminoacidi, un programma è una sequenza di
simboli con una grammatica. Sono tutti casi in cui riempire in un ordine deciso
dai vincoli è più naturale che scrivere da sinistra a destra.

`````

`````{tab} Superiore

La fattorizzazione dice anche dove cercare la distanza dai modelli
autoregressivi, ed è una previsione verificabile: la distanza deve concentrarsi
sui compiti in cui la dipendenza fra posizioni lontane è forte, cioè sulla
generazione lunga, e assottigliarsi dove il contesto determina quasi tutto,
cioè sulla comprensione. È quello che si osserva sui modelli linguistici a
diffusione mascherata addestrati su scala {cite}`nie2025llada`.

Da qui tre osservazioni di prospettiva, tutte verificabili sul conto fatto e
nessuna delle quali richiede di sapere quale modello sia uscito quando.

La prima è che il limite viene dalla **fattorizzazione**, e non dalla diffusione
in sé. Un passo parallelo esatto richiederebbe di campionare dalla congiunta
delle posizioni scoperte, e nessuno sa farlo a costo lineare. Le tecniche che
riducono il divario (scoperta adattiva per entropia, correzione delle posizioni
già scritte, gruppi piccoli in punti critici) lavorano tutte su questa
quantità.

La seconda è che il caso continuo e quello discreto si descrivono con lo stesso
principio, al livello in cui la probabilità si sposta: il rapporto fra
probabilità di stati vicini fa il mestiere del gradiente della log-densità, la
matrice generatrice quello della deriva, e le tre letture (variazionale, del
punteggio, del flusso) si riscrivono tutte {cite}`lai2026principles`. La
traduzione però non è letterale, ed è la stessa fonte a dirlo: in uno spazio
finito non c'è nessuna geometria euclidea, quindi il gradiente rispetto allo
stato non ha un analogo diretto, e un cammino di probabilità non determina una
sola matrice generatrice. Chi ha capito il caso continuo può leggere il discreto
come una traduzione, e questo vale anche nel verso pratico: le tecniche di guida
della {doc}`sezione su guida e allineamento </ModelliDiffusione/guida>` hanno
tutte una controparte discreta.

La terza è che questa famiglia rende naturali i compiti di **riempimento
vincolato**, che è la forma di molti problemi importanti fuori dal linguaggio:
generazione di molecole con proprietà imposte, progettazione di sequenze
proteiche con siti fissati, sintesi di programmi con un'interfaccia data. In
tutti l'ordine di scrittura è un vincolo artificiale, e toglierlo è un
guadagno.

`````

Resta da registrare un conto in sospeso. Anche qui la strada del ritorno ha una
formula esatta, la matrice generatrice invertita, come nel continuo ce l'aveva
il risultato di Anderson della {doc}`sezione sul limite continuo
</ModelliDiffusione/sde-e-ode>`. Ma la dinamica esatta scopre una posizione per
volta, perché il processo in avanti corrompe ogni posizione per conto suo e due
cancellazioni nello stesso istante hanno probabilità trascurabile: ogni volta
che se ne scoprono due insieme si sta saltando, e il salto è
un'approssimazione {cite}`campbell2022continuous`. Il regime in cui questa
famiglia conviene, cioè quello parallelo, è esattamente quello in cui
l'approssimazione morde. C'è però un rovescio, e va detto per intero: qui
l'errore si azzera dopo un numero finito di passi, uno per posizione, mentre
nel continuo si azzera solo al limite.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- A una parola non si può aggiungere un pizzico di rumore, perché fra due
  simboli non c'è niente in mezzo. Il modo che funziona è **cancellarla**,
  sostituendola con un segnaposto: chi ripulisce sa esattamente dove guardare,
  e le parole rimaste sono vere.
- L'obiettivo che ne esce è una vecchia conoscenza: **indovinare le parole
  cancellate**, con la frazione di cancellazione sorteggiata invece che fissa.
  È il compito su cui una generazione di modelli linguistici è stata
  addestrata, più una procedura per generare da zero.
- Il vantaggio è riempire **in parallelo** invece che una parola alla volta, e
  in qualunque ordine. Il prezzo è esatto e si misura: scoprire più posizioni
  insieme le tratta come indipendenti, e dove non lo sono il risultato è
  spazzatura. Sul linguaggio a tre bit, un passo solo produce metà parole
  sbagliate; due passi zero. E il risparmio non è pieno: a ogni passo si
  riguarda la frase intera da capo, quindi conviene solo se i passi sono molti
  meno delle parole.
- Quindi il numero di passi necessari non dipende dalla lunghezza del testo ma
  da **quanto le parti si condizionano a vicenda**, ed è la ragione per cui in
  pratica si scoprono per prime le posizioni su cui il modello è più sicuro:
  è il sudoku, dove si comincia dalla casella più costretta.
- Dove questa famiglia vince già oggi è nei compiti di **riempimento**:
  completare un buco in mezzo a un file, rispettare vincoli su posizioni
  sparse, generare molecole o sequenze con parti fissate.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Il processo in avanti è una catena di Markov sul vocabolario con matrice
  $\mathbf{Q}_t$; la variante **assorbente** (verso $\texttt{[MASK]}$) si è
  imposta perché lo stato dichiara dove è avvenuta la corruzione, e il
  posteriore si fattorizza sulle sole posizioni mascherate. Il nucleo è
  indipendente per posizione, quindi il campionamento è *simulation-free*.
- Il limite variazionale collassa in una **cross-entropia sulle posizioni
  mascherate**, pesata da $-\alpha_t'/(1-\alpha_t)$, che è positivo perché il
  programma scende: nessun punteggio, nessuna KL fra gaussiane. È il
  modellamento mascherato alla BERT esteso a tutte le frazioni di maschera, e
  a meno dei pesi, più un campionatore.
- Scoprire $k$ posizioni insieme campiona dal **prodotto** delle marginali
  invece che dalla congiunta, ed è esatto solo se sono condizionatamente
  indipendenti; la quantità che misura l'errore è la multi-informazione del
  gruppo, non le informazioni mutue a coppie. Sul linguaggio di parità le mutue
  a coppie sono tutte nulle e un passo solo dà lo stesso $0{,}499$ di parole
  valide e otto stati invece di quattro; due passi danno la distribuzione
  esatta. La strategia pratica è scoprire per prime le posizioni a entropia
  minima.
- Costi: **nessun riuso della cache** (ogni passo ricalcola l'intera
  sequenza), verosimiglianza disponibile solo come **limite**. Guadagni:
  riempimento vincolato naturale, e calcolo regolabile al momento della
  richiesta senza riaddestrare.
- La formulazione a tempo continuo sostituisce il gradiente della log-densità
  con il **rapporto fra probabilità di stati vicini** e la deriva con la
  matrice generatrice: la struttura del caso continuo si traduce, guida
  compresa, ma non alla lettera, perché in uno spazio finito il gradiente
  rispetto allo stato non ha un analogo diretto.
```
`````

La strada è percorsa tutta, allora: da una ricetta per rovinare le immagini a
una teoria che dice quale percorso seguire, quanto in fretta, in che direzione
piegarlo, e che cosa fare quando lo stato non è fatto di numeri. Quello che
resta uguale in tutte le versioni è la mossa iniziale, ed è la ragione per cui
questa famiglia ha vinto dove altre si erano fermate: invece di chiedere a una
rete di produrre in un colpo qualcosa di complicato, le si chiede mille volte
di sistemare qualcosa di poco rovinato, e si mettono in fila le mille
risposte.
