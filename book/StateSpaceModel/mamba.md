# Mamba: selezione e scan

Nella sezione precedente abbiamo costruito S4 e i suoi parenti: uno *state
space model* nasce come sistema dinamico continuo e, una volta discretizzato,
diventa una ricorrenza lineare a stato fisso (a ogni passo lo stato di prima si
riduce un po’, ci si somma quello che entra adesso, e da lì si legge l'uscita),
con la sua doppia natura (ricorrente per l'inferenza, convoluzionale per
l'addestramento). È una macchina potente e a lungo raggio. Ha però un limite di
fondo, che finora abbiamo lasciato sullo sfondo: è **invariante nel tempo**.

Invariante nel tempo, in gergo *lineare tempo-invariante* (LTI), vuol dire che
le sue tre regole (di quanto lo stato si riduce, come l'ingresso vi entra, come
se ne legge l'uscita) sono le stesse a ogni passo. La stessa
matrice di transizione governa il primo token e il millesimo; lo stesso filtro
scorre su tutta la sequenza, indifferente a ciò che legge. È proprio questa
rigidità a regalare a S4 la forma «tutto insieme» (un unico filtro fisso, in
gergo *kernel*, che si applica ovunque) ma è anche la sua cecità: un SSM LTI
non può **scegliere**, in base al contenuto, su cosa concentrarsi e cosa
lasciar cadere. Tratta la parola importante e la parola di riempimento esattamente
allo stesso modo.

L'idea di Mamba, proposta da Albert Gu e Tri Dao nel 2023 {cite}`gu2023mamba`,
è tanto semplice da enunciare quanto delicata da realizzare: rendere l'SSM
**selettivo**. Lasciare cioè che le regole della ricorrenza dipendano da ciò
che entra, così che il modello possa decidere, parola per parola, cosa
propagare e cosa dimenticare. È lo stesso salto che nel capitolo precedente
separava le architetture in cui il passato sbiadisce sempre alla stessa
velocità, decisa una volta per tutte, da quelle in cui è la parola in arrivo a
decidere quanto sbiadire; qui ci arriviamo dall'altra sponda, quella dei
sistemi dinamici.

## La selettività (S6)

Il cuore di Mamba è un SSM di tipo S4 in cui le regole non sono più decise una
volta per tutte: le sceglie, parola per parola, ciò che sta entrando. Gli
autori chiamano **S6** questo meccanismo, e il nome inganna, perché sembra il
numero d'ordine di una serie, il modello che viene dopo S5, e non lo è: è
un'abbreviazione introdotta di passaggio, «i modelli S4 con un meccanismo di
selezione, calcolati con uno scan». Di progressione non ce n'è nessuna, tanto
che S5, incontrato fra le tappe verso il linguaggio, è il modello di un altro
gruppo di ricerca.

La conseguenza tecnica è importante
e va guardata in faccia: se le regole cambiano a ogni parola, il sistema non è
più invariante nel tempo. E un sistema che cambia regola strada facendo non ha
un filtro unico: la forma «tutto insieme», che era il segreto
dell'addestramento veloce di S4, semplicemente non è più applicabile. Bisognerà
procurarsi un'altra strada per lavorare in parallelo, e sarà lo *scan* di cui
parliamo qui sotto. Ma prima il guadagno, che ripaga il sacrificio.

`````{tab} Elementare

Immagina un buttafuori all'ingresso di un locale. Un SSM invariante nel tempo è
un tornello: chiunque arrivi, stesso trattamento, stessa spinta in avanti. Il
buttafuori di Mamba, invece, **guarda in faccia** chi passa e decide sul momento:
questo lo faccio entrare e me lo segno, quest'altro lo lascio perdere, di
quest'altro ancora mi dimentico subito. La regola non è scritta una volta per
tutte: cambia a ogni persona, in base a chi è.

Perché ci interessa? Perché apre la porta a un tipo di ragionamento che un
tornello non potrà mai fare: quello che dipende dal **contenuto**. Prendi il
gioco del «copia solo le parole in maiuscolo» in mezzo a un fiume di parole
minuscole: serve decidere, parola per parola, se questa va tenuta o buttata. Un
sistema che tratta tutti i token allo stesso modo fallisce; uno che sa scegliere,
no. È la differenza tra un metal detector regolato una volta all'aeroporto e una
guardia attenta che valuta caso per caso.

`````

`````{tab} Superiore

In un SSM LTI i parametri $(\bar{\mathbf{A}}, \bar{\mathbf{B}}, \mathbf{C}, \Delta)$ sono costanti lungo la
sequenza. Mamba li rende **funzioni dell'input**. Detta $\mathbf{x}_t$ l'attivazione al
passo $t$ e $N$ la dimensione dello stato dell'SSM (per canale):

$$
\mathbf{B}_t = \mathrm{Linear}_N(\mathbf{x}_t),
\qquad
\mathbf{C}_t = \mathrm{Linear}_N(\mathbf{x}_t),
\qquad
\Delta_t = \mathrm{softplus}\!\big(p + \mathrm{Linear}_1(\mathbf{x}_t)\big),
$$

dove $\mathrm{Linear}_N$ proietta $\mathbf{x}_t$ in un vettore di dimensione $N$, $p$ è un
parametro scalare appreso e $\mathrm{softplus}(z) = \log(1 + e^z)$ garantisce un
passo $\Delta_t > 0$. La matrice $\mathbf{A}$, diagonale, resta un **parametro fisso**: non
dipende dal token. Ma la discretizzazione (la scelta *zero-order hold* usata da
Mamba, che nella sezione precedente ha dato $\bar{\mathbf{A}} = \exp(\Delta \mathbf{A})$) fa passare
$\Delta_t$ *dentro* la transizione:

$$
\bar{\mathbf{A}}_t = \exp(\Delta_t\, \mathbf{A}),
\qquad
\mathbf{h}_t = \bar{\mathbf{A}}_t\, \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t\, x_t,
\qquad
y_t = \mathbf{C}_t\, \mathbf{h}_t,
$$

dove $\bar{\mathbf{A}}_t$ è la transizione discreta al passo $t$ e $\bar{\mathbf{B}}_t$ è il termine
di ingresso, entrambi ottenuti da $\Delta_t$ (la ricorrenza è scritta per un
canale: $x_t$ è la componente dell'attivazione $\mathbf{x}_t$ su quel canale, ed è
un numero). Poiché $\Delta_t$ dipende da $\mathbf{x}_t$,
anche $\bar{\mathbf{A}}_t$ diventa **di fatto data-dipendente**, pur partendo da una $\mathbf{A}$
fissa: un $\Delta_t$ grande apre la memoria al nuovo token, un $\Delta_t$ vicino a
zero la lascia scorrere via quasi immutata. Il prezzo è la perdita
dell'invarianza temporale: non esiste più un unico kernel
$\bar{\mathbf{K}} = (\mathbf{C}\bar{\mathbf{B}},\, \mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}},\, \dots)$, perché $\bar{\mathbf{A}}_t, \bar{\mathbf{B}}_t, \mathbf{C}_t$
cambiano a ogni passo. La forma convoluzionale svanisce; resta la sola forma
ricorrente, e con essa il problema di come addestrarla in parallelo.

`````

Il guadagno concettuale è quello che gli autori chiamano *ragionamento basato
sul contenuto*. Il gioco delle parole da copiare e di quelle da lasciar
cadere, che è il compito di *selective copying*, un SSM invariante nel tempo lo
sbaglia: sa ricordare a lungo, ma la sua dinamica è la stessa per tutti. Lo
stesso vale per le *induction
heads*, il meccanismo con cui un modello, visto una volta lo schema «A è
seguito da B», lo completa la volta successiva: richiede di agganciare il
presente a un preciso episodio passato, cioè di scegliere *cosa* propagare. La
selettività di Mamba dà all'SSM proprio questa capacità di decisione che gli
mancava. È lo stesso gesto che, nel capitolo precedente, le valvole decise dai
dati (i *gate*) conferivano alle attenzioni lineari: qui arriva vestito da
sistema dinamico, ma la sostanza è la medesima.

## Lo scan hardware-aware

Rinunciare alla convoluzione sembra un disastro per l'efficienza: la forma
«tutto insieme» era ciò che rendeva S4 addestrabile in fretta. Per fortuna la
ricorrenza lineare ha due proprietà che ci salvano, e conviene tenerle
distinte. La prima è che **comporre due passi dà ancora un passo dello stesso
tipo**: due aggiornamenti consecutivi si possono fondere in uno solo, che ha la
stessa forma di ciascuno dei due. La seconda è che quella composizione è
**associativa**: raggruppare i passi in un modo o nell'altro dà lo stesso
risultato, come in una somma di tanti numeri, dove si può cominciare a sommare
da dove si vuole.

È la seconda a essere decisiva, perché autorizza a fondere i passi a coppie,
poi a gruppi di quattro, di otto, invece di percorrerli in fila da sinistra a
destra. Questo è il *parallel scan*, dove «scan» è la passata che percorre la
sequenza accumulando i risultati parziali. I conti da fare, a seconda di come
si raggruppa, restano tanti quanti erano oppure crescono un poco. Quello che
crolla è l’**attesa**: raddoppiando la lunghezza della sequenza si aggiunge un
turno soltanto, e dove prima c'erano mille passi in fila adesso ci sono una
decina di turni. È il compromesso tipico del calcolo parallelo, dove si
accettano più conti in cambio di meno attesa.

Ogni turno tiene occupati migliaia di core della GPU: quelli generici, però,
non le sue unità dedicate a moltiplicare matrici, e in fondo alla pagina
vedremo che è un problema. La convoluzione se n'è andata, ma il parallelismo
resta.

La {numref}`fig-scan-parallelo` mette le due strade sullo stesso orologio.

```{figure} ../figures/scan-parallelo.svg
:name: fig-scan-parallelo
:alt: Due schemi affiancati della stessa ricorrenza su dodici passi, con lo stesso asse verticale dei turni, numerati da 0 a 11. A sinistra, in fila: una scala di pallini pieni scende in diagonale, una posizione per turno, e per arrivare in fondo ne servono undici. A destra, a raddoppio: quattro righe di frecce in cui la distanza fra le posizioni composte raddoppia (1, 2, 4, 8), i pallini pieni passano da 2 a 4 a 8 a 12, e dopo il quarto turno tutte le righe sotto restano vuote perché non c'è più niente da fare.
:width: 90%

Le due strade per svolgere la stessa ricorrenza su dodici passi, messe sullo
stesso orologio: in verticale i turni, in orizzontale le posizioni della
sequenza. A sinistra si va in fila, una composizione per turno, e il risultato
definitivo (il pallino pieno) avanza di una posizione alla volta: servono
undici turni. A destra si compongono le posizioni distanti prima 1, poi 2, poi
4, poi 8, e siccome a ogni turno raddoppia il tratto di sequenza già riassunto,
dopo quattro turni ogni posizione ha il suo risultato. Le due strade danno gli
stessi numeri; cambia solo quanto c'è da aspettare.
```

Non basta però l'algoritmo. Mamba deve fare i conti anche con il modo in cui
una scheda grafica tiene i dati, la sua **gerarchia di memoria**, ed è qui che
sta la parte «hardware-aware».

`````{tab} Elementare

Prima lo *scan*, che è la parola inglese per «passata»: il modo di svolgere in
fretta un conto che sembra doversi fare in fila. Immagina una classe che deve
sommare mille numeri scritti alla lavagna. Un solo ragazzo che parte dal primo
e va avanti fa quasi mille addizioni, una dopo l'altra: nessuno può aiutarlo,
perché per fare la sua somma deve aspettare quella di prima. Se invece i
ragazzi si mettono in coppia, e ogni coppia somma i suoi due numeri, in un
colpo solo i mille numeri diventano cinquecento; poi cinquecento diventano
duecentocinquanta, e così via. Dopo dieci giri si è arrivati in fondo, perché
dimezzando mille dieci volte si arriva a uno. Di addizioni se ne fanno più o
meno quante prima, qualcuna in più secondo come si raggruppa, ma il tempo di
attesa crolla, perché a ogni giro lavorano tutti insieme. (Una differenza con
la classe c'è: al modello non serve solo il totale finale, ma il totale fino a
ciascuna posizione. È il conto della figura qui sopra, e si raggruppa allo
stesso modo.) La ricorrenza di Mamba si può svolgere così: non è una
somma, ma si comporta come una somma, nel senso che si può cominciare a
raggruppare i passi da dove si vuole. Ed è per questo che perdere la forma
«tutto insieme» non è la catastrofe che sembrava.

Poi l’*hardware*. Pensa a un contabile che deve tenere la somma corrente di una
lunghissima lista di movimenti. Ha due posti dove lavorare: un foglietto sulla
scrivania, piccolo ma a portata di mano, e un archivio in cantina, enorme ma
lontano (ogni discesa in cantina costa tempo). Il modo stupido è scendere in
archivio a ogni riga, per depositare e riprendere il totale. Il modo furbo è tenere il
foglietto sulla scrivania: ci scrivi sopra la somma corrente, la aggiorni
movimento dopo movimento senza mai muoverti, e scendi in cantina una volta
sola alla fine, per archiviare il totale.

Mamba fa esattamente questo. Il foglietto veloce è la memoria interna della
scheda grafica, l'archivio lontano è la sua memoria principale. Attenzione a
non confonderli con il riassunto del modello: qui si parla della scrivania su
cui la scheda fa i suoi conti, non di ciò che il modello ricorda del testo. Il
modello carica una
volta i parametri, svolge tutta la ricorrenza sul «foglietto» e riporta in
archivio solo il risultato, senza mai scrivere in cantina gli ingombranti
stati intermedi. E c'è un secondo trucco da contabile parsimonioso: quei
totali intermedi, se servono di nuovo per correggere i conti (la fase di
addestramento all'indietro), non li conserva (li **ricalcola** al volo, perché
rifare la somma costa meno che tenere in archivio migliaia di fogli).

`````

`````{tab} Superiore

Conviene scrivere l'operatore dello scan, perché senza di lui resta un nome.
Posto $h_t = a_t\,h_{t-1} + b_t$ (una singola componente dello stato: nel caso
diagonale $a_t$ e $b_t$ sono le componenti corrispondenti di
$\bar{\mathbf{A}}_t$ e di $\bar{\mathbf{B}}_t x_t$, e sono numeri), ogni passo è
la coppia $(a_t, b_t)$ e comporne due dà

$$
(a_1, b_1) \bullet (a_2, b_2) = (a_2 a_1,\; a_2 b_1 + b_2),
$$

dove il fattore di sinistra è il passo che viene prima. La famiglia è dunque
chiusa (il risultato è ancora una coppia dello stesso tipo) e l'operatore è
**associativo**, perché lo è la composizione di funzioni: è questa seconda
proprietà a permettere di riassociare l'albero dello scan. Non è invece
commutativo, e non potrebbe esserlo: l'ordine dei fattori è l'ordine della
sequenza.

Delle due versioni classiche dello scan conviene tenere presente la differenza,
perché più avanti ne scriveremo una sola. Detta $L$ la lunghezza della
sequenza, quella **a raddoppio** compone a ogni turno le posizioni distanti
prima 1, poi 2, poi 4: raggiunge la profondità $O(\log L)$, ma con un lavoro
$O(L\log L)$, cioè qualche operazione più del necessario. Quella di
**Blelloch**, con una passata che sale e una che scende, ha la stessa
profondità $O(\log L)$ e lavoro $O(L)$, come la versione sequenziale. In
entrambe il numero di turni crolla da $L$ al suo logaritmo, ed è il numero di
turni ciò che si paga in attesa.

La GPU, dal canto suo, ha una memoria ad alta capacità ma lenta, la **HBM**, e
una memoria molto più piccola e veloce, la **SRAM** on-chip. Il collo di
bottiglia di una ricorrenza selettiva è che lo stato espanso ha forma
$(\texttt{batch}, L, D, N)$ (batch per lunghezza per canali per dimensione
dello stato) e materializzarlo tutto in HBM sarebbe proibitivo in memoria e in
banda. Mamba lo evita con la **fusione dei kernel** (*kernel fusion*): carica i
parametri $(\Delta, \mathbf{A}, \mathbf{B}, \mathbf{C})$ dalla HBM alla SRAM, esegue *in* SRAM la
discretizzazione e la ricorrenza tramite il parallel scan, e riporta in HBM
soltanto l'output $\mathbf{y}$ di dimensione $(\texttt{batch}, L, D)$. Lo stato espanso
non viene mai scritto nella memoria lenta: nasce e muore in SRAM.

A questo si aggiunge la **ricomputazione** (*recomputation*).
Nell'addestramento, il passo all'indietro (*backward*) ha bisogno degli stati
intermedi $\mathbf{h}_t$ per calcolare i gradienti; salvarli tutti costerebbe memoria
quanto materializzare lo stato espanso. Mamba non li salva: li **ricalcola**
durante il backward, rifacendo la ricorrenza. È lo stesso compromesso del
*gradient checkpointing* (si spende un po’ di calcolo in più per risparmiare
molta memoria) e permette al selective scan di avere lo stesso profilo di
memoria di un'implementazione ottimizzata dell'attenzione, senza mai pagare il
costo dello stato espanso in HBM.

`````

Conviene vedere, ridotta all'osso, la ricorrenza che lo scan calcola in
fretta. Il codice che segue si può leggere anche senza saper programmare: le
prime righe dicono che cosa entra, e il ciclo `for` (che vuol dire «per ogni
passo, ripeti quanto segue») è la vasca da bagno di inizio capitolo, quella in
cui il livello cala da solo e risale con l'acqua che entra, scritta in Python.
A ogni giro il livello di prima viene ridotto un po’, si aggiunge quello che
entra adesso, e si legge il risultato.

```python
import torch

# SSM selettivo, un canale: stato h di dimensione N.
# I parametri B, C, delta dipendono dal token (indice t); A e' fisso.
def ssm_selettivo(x, A, B, C, delta):
    # x: (L,)   input del canale
    # A: (N,)   diagonale fissa (valori negativi, per stabilita')
    # B, C: (L, N)  generati da x, cambiano a ogni passo
    # delta: (L,)   passo di discretizzazione, generato da x
    L, N = B.shape
    h = torch.zeros(N, dtype=x.dtype, device=x.device)
    y = torch.empty_like(x)
    for t in range(L):
        A_bar = torch.exp(delta[t] * A)   # A-bar_t = exp(delta_t A), diagonale
        B_bar = delta[t] * B[t]           # discretizzazione semplificata di B
        h = A_bar * h + B_bar * x[t]      # h_t = A-bar_t h_{t-1} + B-bar_t x_t
        y[t] = torch.dot(C[t], h)         # y_t = C_t . h_t
    return y
```

Il ciclo `for` è la forma ricorrente, quella dell'inferenza: costo e memoria
costanti per token, un aggiornamento dopo l'altro. Quel ciclo però si può evitare in due modi, e li proviamo tutti e due.

La prima riguarda la sezione precedente: se le regole **non** cambiano da un
passo all'altro, lo stesso risultato si ottiene con un filtro unico che scorre
sulla sequenza. Congeliamo allora i tre parametri che dipendevano dal token,
costruiamo quel filtro e confrontiamo.

```python
torch.manual_seed(0)
L, N = 12, 4
x = torch.randn(L, dtype=torch.float64)
A = -torch.rand(N, dtype=torch.float64) - 0.5      # autovalori negativi
B_fisso = torch.randn(N, dtype=torch.float64)
C_fisso = torch.randn(N, dtype=torch.float64)
delta = torch.full((L,), 0.4, dtype=torch.float64)

# stessi parametri a ogni passo: il sistema e' invariante nel tempo (LTI)
y_ric = ssm_selettivo(x, A, B_fisso.repeat(L, 1), C_fisso.repeat(L, 1), delta)

# il kernel K_j = C A-bar^j B-bar: quanto pesa ancora un ingresso di j passi fa
A_bar = torch.exp(0.4 * A)
B_bar = 0.4 * B_fisso
K = torch.stack([(C_fisso * A_bar**j * B_bar).sum() for j in range(L)])

# la convoluzione causale con quel kernel, scritta a mano
y_conv = torch.stack([(K[: t + 1] * torch.flip(x[: t + 1], (0,))).sum()
                      for t in range(L)])

print("ricorrenza vs convoluzione, scarto massimo:",
      (y_ric - y_conv).abs().max().item())
```

Il secondo modo è quello di questa sezione: anche quando le regole
cambiano a ogni passo, e il filtro unico non esiste più, il *parallel scan*
calcola **esattamente lo stesso** vettore `y` del ciclo, raggruppando i passi
invece di percorrerli in fila.

```python
def scan_parallelo(a, b):
    """Ricorrenza h_t = a_t h_{t-1} + b_t svolta a raddoppio.

    Ogni passo e' la coppia (a_t, b_t), e comporne due da'
    (a1, b1) . (a2, b2) = (a2 a1, a2 b1 + b2): l'operazione e'
    associativa, quindi i passi si possono raggruppare a piacere.
    """
    a, b = a.clone(), b.clone()
    salto = 1
    while salto < a.shape[0]:
        a_prec, b_prec = a[:-salto].clone(), b[:-salto].clone()
        b[salto:] = a[salto:] * b_prec + b[salto:]
        a[salto:] = a[salto:] * a_prec
        salto *= 2          # 1, 2, 4, 8, ...: log L giri invece di L
    return b                # b_t contiene ora h_t

# parametri che cambiano a ogni passo: il sistema e' selettivo
B = torch.randn(L, N, dtype=torch.float64)
C = torch.randn(L, N, dtype=torch.float64)
delta = torch.rand(L, dtype=torch.float64) * 0.5 + 0.1
y_ciclo = ssm_selettivo(x, A, B, C, delta)

A_bar = torch.exp(delta[:, None] * A)          # (L, N)
B_bar = delta[:, None] * B * x[:, None]        # (L, N)
H = scan_parallelo(A_bar, B_bar)               # tutti gli stati in una volta
y_scan = (C * H).sum(dim=1)

print("ciclo vs scan parallelo, scarto massimo:",
      (y_ciclo - y_scan).abs().max().item())
```

Entrambi gli scarti sono dell'ordine di $10^{-16}$, cioè zero a meno
dell'ultima cifra che un calcolatore riesce a rappresentare: le tre forme
calcolano la stessa funzione. È, ancora una volta, la doppia natura che
accomuna tutta questa famiglia di modelli: una forma parallela per addestrare
in fretta, una forma ricorrente a costo costante per generare.

## Il blocco Mamba

Il meccanismo selettivo è il motore; attorno gli serve una carrozzeria. Il
**blocco Mamba** nasce fondendo due pezzi già noti: il blocco H3
{cite}`fu2023h3`, che per primo aveva adattato gli SSM al linguaggio mettendo
attorno al nucleo ricorrente una valvola (la stessa idea del capitolo
precedente: due rami che si moltiplicano, e uno regola quanto dell'altro lascia
passare), e il *gated MLP*, cioè lo strato che nei Transformer segue
l'attenzione, dotato anche lui di una valvola. Il risultato è un unico mattone omogeneo, che si
impila su se stesso a formare l'intera rete: non si alternano blocchi di tipo
diverso, come nei Transformer, ce n'è uno solo, ripetuto.

```{figure} ../figures/blocco-mamba.svg
:name: fig-blocco-mamba
:alt: Diagramma del blocco Mamba. Dal basso, l'ingresso si divide in due rami dopo una proiezione lineare. Il ramo principale attraversa in sequenza una convoluzione causale monodimensionale (Conv1d), un'attivazione SiLU e l'SSM selettivo (S6). Il ramo parallelo attraversa una sola attivazione SiLU. I due rami si incontrano in un gating moltiplicativo, il cui risultato passa per una proiezione lineare di uscita. Un tratteggio scavalca l'intero blocco e si richiude su un simbolo di somma: è la connessione residua.
:width: 85%

Il blocco Mamba, che è l'unico tipo di stazione della catena e si ripete
uguale decine di volte. In basso il pezzo in arrivo si sdoppia: la copia
principale (a sinistra) passa per tre lavorazioni, la copia parallela (a
destra) per una sola e diventa la valvola che regola quanto della prima
lasciar passare. In alto le due si moltiplicano e una proiezione rimette il
pezzo nella forma di partenza. Il tratteggio che scavalca tutto è la
scorciatoia che fa arrivare il pezzo di partenza anche in cima, così che le
lavorazioni aggiungano al pezzo invece di sostituirlo.
```

Seguiamo il percorso di {numref}`fig-blocco-mamba` dal basso verso l'alto.

`````{tab} Elementare

Immagina una piccola catena di montaggio. Il pezzo grezzo (il token) entra e
viene subito **sdoppiato** in due copie che seguono strade diverse. La copia
principale passa per tre stazioni: prima una che le fa dare un'occhiata ai
pochi vicini immediati (una convoluzione locale), poi un ammorbidimento
(l'attivazione), poi il cuore selettivo che decide cosa ricordare del lungo
passato. La seconda copia prende una scorciatoia con un solo ammorbidimento e
diventa una specie di **rubinetto**: alla fine i due rami si reincontrano e il
rubinetto regola quanto del ramo principale lasciar passare, moltiplicandoli
insieme. Un'ultima proiezione rimette il pezzo nella forma di partenza. Tutto
qui: un solo tipo di stazione, ripetuto in verticale decine di volte. Niente
attenzione, niente strati aggiuntivi: la stessa macchina, dall'inizio alla
fine.

`````

`````{tab} Superiore

Detta $\mathbf{u}$ l'attivazione in ingresso al blocco, il flusso è:

1. **Proiezione in ingresso**: due proiezioni lineari espandono $\mathbf{u}$ (fattore
   $E=2$) in due rami, $\mathbf{x}$ (principale) e $\mathbf{z}$ (di *gating*).
2. **Convoluzione causale 1D**: una `Conv1d` a finestra corta scorre sul ramo
   $\mathbf{x}$ lungo la dimensione temporale. È «causale» perché ogni posizione vede
   solo il proprio passato immediato (nessuna fuga di informazione dal futuro)
   ed è la stessa idea di filtro che scorre vista per le reti convoluzionali,
   qui ridotta a una dimensione e a una manciata di passi. Fornisce un
   contesto locale a basso costo prima dell'SSM.
3. **Attivazione SiLU**: si applica $\mathrm{SiLU}(x) = x\,\sigma(x)$ (nota anche
   come *Swish*), la parente liscia della ReLU incontrata tra le funzioni di
   attivazione, dove $\sigma$ è la sigmoide.
4. **SSM selettivo (S6)**: il ramo attraversa il nucleo selettivo descritto in
   apertura di sezione, con $\mathbf{B}_t, \mathbf{C}_t, \Delta_t$ generati
   dall'input e calcolato via parallel scan.
5. **Gating moltiplicativo**: l'uscita dell'SSM viene moltiplicata elemento per
   elemento dal ramo parallelo passato per SiLU, $\mathbf{y} \odot \mathrm{SiLU}(\mathbf{z})$. È il
   *gate* che regola, canale per canale, quanto dell'uscita ricorrente lasciar
   passare.
6. **Proiezione in uscita**: una proiezione lineare riporta il risultato alla
   dimensione del modello.

Il blocco è avvolto da una **normalizzazione** (LayerNorm o RMSNorm) e da una
**connessione residua**, come in un Transformer. La differenza è che questo
mattone è *l'unico* mattone: non si alternano blocchi di attenzione e blocchi
*feed-forward*, si impila sempre lo stesso.

`````

## Cosa ottiene Mamba

Messi insieme i pezzi (selettività per il ragionamento basato sul contenuto,
scan hardware-aware per l'efficienza, un blocco unico per l'architettura) che
cosa se ne ricava?

`````{tab} Elementare

Due cose, soprattutto. La prima è **il lavoro che non esplode quando il testo
si allunga**: mentre un Transformer, per raddoppiare la lunghezza, quadruplica
il lavoro, Mamba lo raddoppia soltanto. È questo, e nient'altro, che si intende
quando in queste pagine si legge «costo **lineare**»: il lavoro cresce di pari
passo con la lunghezza. Nella generazione parola per parola il vantaggio si
sente, perché a ogni parola nuova il modello non deve rileggersi tutto quello
che ha scritto finora: gli basta il suo riassunto, che è sempre della stessa
misura.

La seconda è la **portata**, ed è il punto in cui conviene essere precisi su
dove è stata misurata. Le sequenze da un milione di passi su cui Mamba continua
a migliorare non sono testo: sono suono grezzo (dove un passo è un campione
sonoro: in un secondo di registrazione ce ne stanno circa sedicimila, quindi un
milione di campioni è poco più di un minuto) e
sequenze di DNA (dove un passo è una lettera del genoma). Sul linguaggio i
contesti provati nell'articolo restano molto più corti, dell'ordine delle
migliaia di parole. Che la stessa ricetta funzioni su tre materiali così
diversi è comunque il segno che il meccanismo non ha niente di specificamente
linguistico.

`````

`````{tab} Superiore

Il bilancio, in termini di meccanismi e non di classifiche:

- **Costo lineare** nella lunghezza della sequenza, in tempo e memoria, contro
  il costo quadratico dell'attenzione piena.
- **Inferenza a memoria costante**: lo stato ricorrente sostituisce la cache
  chiave-valore, che in un Transformer cresce con il contesto e va riletta a
  ogni token generato. È da qui che viene il vantaggio di throughput in
  generazione.
- **Scaling** verificato fino a sequenze dell'ordine di $10^6$ passi, cioè su
  ordini di grandezza dove l'attenzione piena non è praticabile. Le misure a
  quella lunghezza sono su **audio** grezzo e **genomica**; sul linguaggio i
  contesti dell'articolo restano di qualche migliaio di token.
- Il meccanismo non è specifico del testo: gli stessi blocchi si addestrano su
  audio e su DNA, dove le sequenze sono lunghe e non hanno una struttura a
  token discreti come il linguaggio.

`````

Un'ultima onestà, che vale come promemoria e come ponte. Mamba ha avuto un
percorso editoriale movimentato. L'articolo comparve alla fine del 2023 come
*preprint*: messo online a disposizione di tutti prima che qualcuno lo avesse
giudicato. Il giudizio, nella ricerca, lo danno le riviste e i convegni, che
affidano ogni lavoro ad altri studiosi del campo; e il primo convegno a cui
Mamba fu sottoposto, ICLR, nel 2024 lo respinse, con un rifiuto che fece
discutere. Pochi mesi dopo un altro convegno, COLM, lo ha accettato e gli ha
assegnato un premio come uno dei lavori migliori dell'anno.

Il vaglio, quindi, c'è stato.

Di nodi aperti, però, ne restano due. Lo *scan* selettivo non sfruttava appieno
le unità di calcolo matriciale delle GPU: un dettaglio ingegneristico che
sembra minore e in pratica pesa parecchio. E lo stato di dimensione fissa, che
è la forza di Mamba in efficienza, resta il suo limite quando serve ritrovare
un dettaglio preciso in un contesto molto lungo. Sono
proprio questi i nodi che la sezione successiva scioglie: Mamba-2 riscrive il
selective scan come una moltiplicazione di matrici (recuperando i *tensor
core* della GPU) e, nel farlo, svela una parentela inattesa. Perché dietro
l'SSM selettivo, vedremo, si nasconde di nuovo l'attenzione: le due famiglie
che abbiamo raccontato da capitoli diversi sono, alla fine, due viste della
stessa cosa.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- S4 tratta ogni parola con la **stessa regola**: è un tornello. Ha memoria
  lunga, ma non sa scegliere. **Mamba** {cite}`gu2023mamba` mette al suo posto
  un buttafuori, che guarda in faccia chi passa e decide sul momento quanto
  scriverne nel riassunto e quanto lasciar cadere. È questa la **selettività**.
- Si paga un prezzo: se la regola cambia a ogni parola, non esiste più un
  filtro unico, e il modo «tutto insieme» di fare i conti se ne va. Resta il
  modo passo dopo passo.
- Il prezzo si recupera con lo **scan**, cioè svolgendo la catena a gruppi
  invece che in fila (a coppie, poi a quattro, poi a otto): di operazioni se ne
  fanno più o meno quante prima, ma i turni di attesa crollano. In più Mamba tiene i conti nella
  memoria piccola e vicina della scheda grafica, come il contabile che non
  scende in cantina a ogni riga, e i risultati intermedi che gli serviranno
  dopo li **rifà** invece di conservarli.
- Il **blocco Mamba** è un'unica stazione, ripetuta decine di volte: il pezzo
  si sdoppia, una copia passa per la lavorazione lunga (uno sguardo ai vicini,
  un ammorbidimento, il cuore selettivo), l'altra fa da valvola, e alla fine
  le due si moltiplicano. Niente attenzione, nessun altro tipo di stazione.
- Cosa se ne ricava: **il lavoro cresce di pari passo con la lunghezza** (testo
  doppio, lavoro doppio, non quadruplo), la memoria durante la generazione non
  cresce mai, e si reggono sequenze dell'ordine del milione di passi, misurate
  però fuori dal linguaggio (un minuto di suono grezzo, un tratto di genoma).
  Uscito nel 2023 come articolo non ancora giudicato da nessuno (*preprint*),
  respinto dal convegno ICLR nel 2024 e
  pubblicato lo stesso anno al convegno COLM, che lo ha premiato: è anche una
  buona lezione su come funziona il giudizio nella ricerca.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- S4 è **tempo-invariante** (LTI): stessi parametri a ogni passo, quindi un
  kernel di convoluzione fisso, ma nessuna capacità di scegliere in base al
  contenuto. **Mamba** {cite}`gu2023mamba` rende l'SSM **selettivo** (S6).
- Nella selettività $\mathbf{B}_t, \mathbf{C}_t, \Delta_t$ diventano **funzione dell'input**; $\mathbf{A}$
  resta fissa, ma poiché $\bar{\mathbf{A}}_t = \exp(\Delta_t \mathbf{A})$ e $\Delta_t$ dipende da
  $\mathbf{x}_t$, anche la transizione è di fatto data-dipendente. Si rompe l'invarianza
  temporale: **niente più convoluzione**, serve uno scan.
- Il guadagno è il **ragionamento basato sul contenuto** (*selective copying*,
  *induction heads*) che un SSM LTI non può fare. È lo stesso salto dei gate
  data-dipendenti delle attenzioni lineari, raggiunto dal versante dei sistemi
  dinamici.
- Comporre due passi della ricorrenza dà un passo dello stesso tipo (la
  famiglia è chiusa) e la composizione è **associativa**: da qui il **parallel
  scan**, con profondità $O(\log L)$ e lavoro $O(L)$ nella versione di Blelloch
  ($O(L\log L)$ in quella a raddoppio), su unità generiche e non
  sui tensor core. Le ottimizzazioni hardware-aware (*kernel fusion* in SRAM e
  **ricomputazione** nel backward) evitano di materializzare lo stato espanso
  in HBM.
- Il **blocco Mamba** fonde il blocco H3 {cite}`fu2023h3` con un *gated MLP*
  ($E=2$): proiezione in ingresso → Conv1d causale → SiLU → SSM selettivo →
  gating moltiplicativo con ramo parallelo (SiLU) → proiezione in uscita, con
  normalizzazione e residui. Un solo tipo di blocco, senza attenzione né MLP a
  parte.
- Cosa ottiene: **tempo lineare** nella lunghezza, inferenza a **memoria
  costante** (nessuna KV cache che cresce), scaling verificato fino a
  $\sim 10^6$ passi su audio grezzo e genomica (sul linguaggio, contesti molto
  più corti), e lo stesso impianto valido per tutte e tre le modalità. Uscito
  nel 2023 come articolo non ancora giudicato da nessuno (*preprint*), respinto
  dal convegno ICLR nel 2024 e pubblicato lo stesso anno al convegno COLM, che lo ha premiato
  (*Outstanding Paper*), con i limiti che Mamba-2 affronterà.
```

`````
