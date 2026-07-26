# Mamba: selezione e scan

Nella sezione precedente abbiamo costruito S4 e i suoi parenti: uno *state
space model* nasce come sistema dinamico continuo e, una volta discretizzato,
diventa una ricorrenza lineare a stato fisso,
$h_t = \bar A\, h_{t-1} + \bar B\, x_t$, con la sua doppia natura (ricorrente
per l'inferenza, convoluzionale per l'addestramento). È una macchina potente e
a lungo raggio. Ha però un limite di fondo, che finora abbiamo lasciato sullo
sfondo: è **invariante nel tempo**.

Invariante nel tempo, in gergo *lineare tempo-invariante* (LTI), vuol dire che
i suoi parametri $\bar A, \bar B, C$ sono gli stessi a ogni passo. La stessa
matrice di transizione governa il primo token e il millesimo; lo stesso filtro
scorre su tutta la sequenza, indifferente a ciò che legge. È proprio questa
rigidità a regalare a S4 la forma convoluzionale (un unico *kernel* fisso che
si applica ovunque) ma è anche la sua cecità: un SSM LTI non può
**scegliere**, in base al contenuto, su cosa concentrarsi e cosa lasciar
cadere. Tratta la parola importante e la parola di riempimento esattamente
allo stesso modo.

L'idea di Mamba, proposta da Albert Gu e Tri Dao in un preprint del 2023
{cite}`gu2023mamba`, è tanto semplice da enunciare quanto delicata da
realizzare: rendere l'SSM **selettivo**. Lasciare cioè che i parametri della
ricorrenza dipendano dall'input, così che il modello possa decidere, token per
token, cosa propagare e cosa dimenticare. È lo stesso salto che nel capitolo
precedente separava il decadimento fisso di RetNet dai gate appresi della GLA,
ma raggiunto dall'altra sponda, quella dei sistemi dinamici.

## La selettività (S6)

Il meccanismo di Mamba si chiama **S6**: è l'SSM di tipo S4 con in più la
*selezione*, in cui i parametri diventano funzione del token. La conseguenza
tecnica è importante e va guardata in faccia: se $\bar B$, $C$ e il passo di
discretizzazione cambiano a ogni parola, il sistema non è più
tempo-invariante. E un sistema tempo-variante non ha un kernel di convoluzione
fisso: la forma convoluzionale, che era il segreto dell'addestramento veloce
di S4, semplicemente non è più applicabile. Bisognerà procurarsi un'altra
strada per parallelizzare, e sarà lo *scan* della prossima sezione. Ma prima
il guadagno, che ripaga il sacrificio.

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

In un SSM LTI i parametri $(\bar A, \bar B, C, \Delta)$ sono costanti lungo la
sequenza. Mamba li rende **funzioni dell'input**. Detta $x_t$ l'attivazione al
passo $t$ e $N$ la dimensione dello stato dell'SSM (per canale):

$$
B_t = \mathrm{Linear}_N(x_t),
\qquad
C_t = \mathrm{Linear}_N(x_t),
\qquad
\Delta_t = \mathrm{softplus}\!\big(p + \mathrm{Linear}_1(x_t)\big),
$$

dove $\mathrm{Linear}_N$ proietta $x_t$ in un vettore di dimensione $N$, $p$ è un
parametro scalare appreso e $\mathrm{softplus}(z) = \log(1 + e^z)$ garantisce un
passo $\Delta_t > 0$. La matrice $A$, diagonale, resta un **parametro fisso**: non
dipende dal token. Ma la discretizzazione (la scelta *zero-order hold* usata da
Mamba, che nella sezione precedente ha dato $\bar A = \exp(\Delta A)$) fa passare
$\Delta_t$ *dentro* la transizione:

$$
\bar A_t = \exp(\Delta_t\, A),
\qquad
h_t = \bar A_t\, h_{t-1} + \bar B_t\, x_t,
\qquad
y_t = C_t\, h_t,
$$

dove $\bar A_t$ è la transizione discreta al passo $t$ e $\bar B_t$ è il termine
di ingresso, entrambi ottenuti da $\Delta_t$. Poiché $\Delta_t$ dipende da $x_t$,
anche $\bar A_t$ diventa **di fatto data-dipendente**, pur partendo da una $A$
fissa: un $\Delta_t$ grande apre la memoria al nuovo token, un $\Delta_t$ vicino a
zero la lascia scorrere via quasi immutata. Il prezzo è la perdita
dell'invarianza temporale: non esiste più un unico kernel
$\bar K = (C\bar B,\, C\bar A\bar B,\, \dots)$, perché $\bar A_t, \bar B_t, C_t$
cambiano a ogni passo. La forma convoluzionale svanisce; resta la sola forma
ricorrente, e con essa il problema di come addestrarla in parallelo.

`````

Il guadagno concettuale è quello che gli autori chiamano *ragionamento basato
sul contenuto*. Un SSM invariante nel tempo può ricordare a lungo, ma non può
**selezionare**: se gli si chiede di copiare solo certi token e ignorarne
altri in base a ciò che sono (il compito di *selective copying*) inciampa,
perché la sua dinamica è la stessa per tutti. Lo stesso vale per le *induction
heads*, il meccanismo con cui un modello, visto una volta lo schema «A è
seguito da B», lo completa la volta successiva: richiede di agganciare il
presente a un preciso episodio passato, cioè di scegliere *cosa* propagare. La
selettività di Mamba dà all'SSM proprio questa capacità di decisione che gli
mancava. È lo stesso gesto che, nel capitolo precedente, i gate
data-dipendenti conferivano alle attenzioni lineari: qui arriva vestito da
sistema dinamico, ma la sostanza è la medesima.

## Lo scan hardware-aware

Rinunciare alla convoluzione sembra un disastro per l'efficienza: la forma
convoluzionale era ciò che rendeva S4 addestrabile in fretta. Per fortuna la
ricorrenza lineare ha una proprietà che ci salva: è **associativa**. Comporre
due passi di una ricorrenza lineare dà ancora un passo dello stesso tipo, e
questo permette di calcolarla non uno alla volta da sinistra a destra, ma con
un *parallel scan* (o *associative scan*): un algoritmo classico, che risale a
Blelloch, capace di svolgere l'intera ricorrenza con lavoro totale $O(L)$ ma
in tempo parallelo proporzionale a $\log L$, sfruttando a pieno le migliaia di
core di una GPU. La convoluzione se n'è andata, ma il parallelismo resta.

Non basta però la matematica: Mamba deve fare i conti con la **gerarchia di
memoria** della scheda grafica, ed è qui che sta la parte «hardware-aware».

`````{tab} Elementare

Pensa a un contabile che deve tenere la somma corrente di una lunghissima
lista di movimenti. Ha due posti dove lavorare: un foglietto sulla scrivania,
piccolo ma a portata di mano, e un archivio in cantina, enorme ma lontano
(ogni discesa in cantina costa tempo). Il modo stupido è scendere in archivio
a ogni riga, per depositare e riprendere il totale. Il modo furbo è tenere il
foglietto sulla scrivania: ci scrivi sopra la somma corrente, la aggiorni
movimento dopo movimento senza mai muoverti, e scendi in cantina una volta
sola alla fine, per archiviare il totale.

Mamba fa esattamente questo. Il foglietto veloce è la memoria interna della
GPU; l'archivio lontano è la sua memoria principale. Il modello carica una
volta i parametri, svolge tutta la ricorrenza sul «foglietto» e riporta in
archivio solo il risultato, senza mai scrivere in cantina gli ingombranti
stati intermedi. E c'è un secondo trucco da contabile parsimonioso: quei
totali intermedi, se servono di nuovo per correggere i conti (la fase di
addestramento all'indietro), non li conserva (li **ricalcola** al volo, perché
rifare la somma costa meno che tenere in archivio migliaia di fogli).

`````

`````{tab} Superiore

La GPU ha una memoria ad alta capacità ma lenta, la **HBM**, e una memoria
molto più piccola e veloce, la **SRAM** on-chip. Il collo di bottiglia di una
ricorrenza selettiva è che lo stato espanso ha forma $(B, L, D, N)$ (batch per
lunghezza per canali per dimensione dello stato) e materializzarlo tutto in
HBM sarebbe proibitivo in memoria e in banda. Mamba lo evita con la **fusione
dei kernel** (*kernel fusion*): carica i parametri $(\Delta, A, B, C)$ dalla
HBM alla SRAM, esegue *in* SRAM la discretizzazione e la ricorrenza tramite il
parallel scan, e riporta in HBM soltanto l'output $y$ di dimensione
$(B, L, D)$. Lo stato espanso non viene mai scritto nella memoria lenta: nasce
e muore in SRAM.

A questo si aggiunge la **ricomputazione** (*recomputation*).
Nell'addestramento, il passo all'indietro (*backward*) ha bisogno degli stati
intermedi $h_t$ per calcolare i gradienti; salvarli tutti costerebbe memoria
quanto materializzare lo stato espanso. Mamba non li salva: li **ricalcola**
durante il backward, rifacendo la ricorrenza. È lo stesso compromesso del
*gradient checkpointing* (si spende un po' di calcolo in più per risparmiare
molta memoria) e permette al selective scan di avere lo stesso profilo di
memoria di un'implementazione ottimizzata dell'attenzione, senza mai pagare il
costo dello stato espanso in HBM.

`````

Vale la pena vedere, ridotta all'osso, la ricorrenza che lo scan calcola in
fretta. Nella sua forma sequenziale (un token alla volta, come la scriverebbe
una RNN) per un singolo canale è questa:

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
    h = torch.zeros(N)
    y = torch.empty(L)
    for t in range(L):
        A_bar = torch.exp(delta[t] * A)   # A-bar_t = exp(delta_t A), diagonale
        B_bar = delta[t] * B[t]           # discretizzazione semplificata di B
        h = A_bar * h + B_bar * x[t]      # h_t = A-bar_t h_{t-1} + B-bar_t x_t
        y[t] = torch.dot(C[t], h)         # y_t = C_t . h_t
    return y
```

Il ciclo `for` è la forma ricorrente, quella dell'inferenza: costo e memoria
costanti per token, un aggiornamento dopo l'altro. Il parallel scan calcola
*esattamente lo stesso* vettore `y`, ma senza il ciclo sequenziale, sfruttando
l'associatività per svolgere i passi in parallelo. È, ancora una volta, la doppia
natura che accomuna tutta questa famiglia di modelli: una forma parallela per
addestrare in fretta, una forma ricorrente a costo costante per generare.

## Il blocco Mamba

Il meccanismo selettivo è il motore; attorno gli serve una carrozzeria. Il
**blocco Mamba** nasce fondendo due ingredienti già noti: il blocco H3
{cite}`fu2023h3`, che per primo aveva adattato gli SSM al linguaggio circondando
il nucleo ricorrente di una struttura di *gating* moltiplicativo, e il classico
*gated MLP* dei Transformer. Il risultato è un unico blocco omogeneo, con un
fattore di espansione $E = 2$ (le proiezioni interne raddoppiano la dimensione),
che si impila su se stesso a formare l'intera rete. Non ci sono blocchi di
attenzione, non ci sono strati *feed-forward* separati: un solo tipo di mattone,
ripetuto.

```{figure} ../figures/blocco-mamba.svg
:name: fig-blocco-mamba
:alt: Diagramma del blocco Mamba. Dal basso, l'ingresso si divide in due rami dopo una proiezione lineare. Il ramo principale attraversa in sequenza una convoluzione causale monodimensionale (Conv1d), un'attivazione SiLU e l'SSM selettivo (S6). Il ramo parallelo attraversa una sola attivazione SiLU. I due rami si incontrano in un gating moltiplicativo, il cui risultato passa per una proiezione lineare di uscita. Attorno al blocco, una connessione residua e una normalizzazione.
:width: 85%

Il blocco Mamba: la proiezione in ingresso apre due rami. Quello principale
passa per Conv1d causale, SiLU e SSM selettivo; quello parallelo per una SiLU
che fa da *gate*. Il gating moltiplicativo li ricongiunge, poi la proiezione
di uscita: il tutto avvolto da normalizzazione e connessione residua.
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

Detta $u$ l'attivazione in ingresso al blocco, il flusso è:

1. **Proiezione in ingresso**: due proiezioni lineari espandono $u$ (fattore
   $E=2$) in due rami, $x$ (principale) e $z$ (di *gating*).
2. **Convoluzione causale 1D**: una `Conv1d` a finestra corta scorre sul ramo
   $x$ lungo la dimensione temporale. È «causale» perché ogni posizione vede
   solo il proprio passato immediato (nessuna fuga di informazione dal futuro)
   ed è la stessa idea di filtro che scorre vista per le reti convoluzionali,
   qui ridotta a una dimensione e a una manciata di passi. Fornisce un
   contesto locale a basso costo prima dell'SSM.
3. **Attivazione SiLU**: si applica $\mathrm{SiLU}(x) = x\,\sigma(x)$ (nota anche
   come *Swish*), la parente liscia della ReLU incontrata tra le funzioni di
   attivazione, dove $\sigma$ è la sigmoide.
4. **SSM selettivo (S6)**: il ramo attraversa il nucleo della sezione precedente,
   con $B_t, C_t, \Delta_t$ generati dall'input e calcolato via parallel scan.
5. **Gating moltiplicativo**: l'uscita dell'SSM viene moltiplicata elemento per
   elemento dal ramo parallelo passato per SiLU, $y \odot \mathrm{SiLU}(z)$. È il
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

Due cose, soprattutto. La prima è **velocità che non peggiora quando il testo
si allunga**: mentre un Transformer, per raddoppiare la lunghezza, quadruplica
il lavoro, Mamba lo raddoppia soltanto (il costo cresce di pari passo con la
sequenza, non più in fretta). Nella generazione parola per parola questo si
traduce in un modello sensibilmente più scattante. La seconda è la
**portata**: Mamba regge sequenze lunghissime, dell'ordine del milione di
passi, là dove un Transformer soffocherebbe. E a parità di qualità pesa meno:
un modello Mamba regge il confronto con Transformer grandi il doppio. Non solo
testo, per giunta: la stessa ricetta dà buoni risultati anche su segnali audio
e su sequenze di DNA.

`````

`````{tab} Superiore

I risultati riportati nel preprint sono, in sintesi:

- **Costo lineare** nella lunghezza della sequenza, in tempo e memoria, contro il
  costo quadratico dell'attenzione piena.
- **Inferenza** con throughput circa $5\times$ superiore a quello di un
  Transformer di taglia comparabile, grazie allo stato ricorrente a memoria
  costante (nessuna cache chiave-valore che cresce con il contesto).
- **Scaling** verificato fino a sequenze dell'ordine di $10^6$ passi.
- Un **Mamba-3B** che eguaglia in qualità Transformer di taglia circa doppia sul
  *language modeling*.
- Risultati di primo piano anche fuori dal linguaggio, su **audio** e
  **genomica**, a conferma che il meccanismo non è specifico del testo.

`````

Un'ultima onestà, che vale come promemoria e come ponte. Mamba è un **preprint
del 2023**: la sua sede di pubblicazione non è confermata, e come sempre nella
ricerca recente non tutto ciò che il primo articolo annuncia si è retto
intatto alla prova del tempo. Lo *scan* selettivo, in particolare, non
sfruttava appieno le unità di calcolo matriciale delle GPU: un dettaglio
ingegneristico che sembra minore ma pesa parecchio in pratica. E lo stato di
dimensione fissa, che è la forza di Mamba in efficienza, resta il suo limite
quando serve ritrovare un dettaglio preciso in un contesto molto lungo. Sono
proprio questi i nodi che la sezione successiva scioglie: Mamba-2 riscrive il
selective scan come una moltiplicazione di matrici (recuperando i *tensor
core* della GPU) e, nel farlo, svela una parentela inattesa. Perché dietro
l'SSM selettivo, vedremo, si nasconde di nuovo l'attenzione: le due famiglie
che abbiamo raccontato da capitoli diversi sono, alla fine, due viste della
stessa cosa.

```{admonition} Da ricordare
:class: important
- S4 è **tempo-invariante** (LTI): stessi parametri a ogni passo, quindi un
  kernel di convoluzione fisso, ma nessuna capacità di scegliere in base al
  contenuto. **Mamba** {cite}`gu2023mamba` rende l'SSM **selettivo** (S6).
- Nella selettività $B_t, C_t, \Delta_t$ diventano **funzione dell'input**; $A$
  resta fissa, ma poiché $\bar A_t = \exp(\Delta_t A)$ e $\Delta_t$ dipende da
  $x_t$, anche la transizione è di fatto data-dipendente. Si rompe l'invarianza
  temporale: **niente più convoluzione**, serve uno scan.
- Il guadagno è il **ragionamento basato sul contenuto** (*selective copying*,
  *induction heads*) che un SSM LTI non può fare. È lo stesso salto dei gate
  data-dipendenti delle attenzioni lineari, raggiunto dal versante dei sistemi
  dinamici.
- La ricorrenza lineare è **associativa**, quindi si calcola con un **parallel
  scan** (lavoro $O(L)$, tempo parallelo $\log L$). Le ottimizzazioni
  hardware-aware (*kernel fusion* in SRAM e **ricomputazione** nel backward)
  evitano di materializzare lo stato espanso in HBM.
- Il **blocco Mamba** {cite}`fu2023h3` fonde il blocco H3 con un *gated MLP*
  ($E=2$): proiezione in ingresso → Conv1d causale → SiLU → SSM selettivo →
  gating moltiplicativo con ramo parallelo (SiLU) → proiezione in uscita, con
  normalizzazione e residui. Un solo tipo di blocco, senza attenzione né MLP a
  parte.
- Cosa ottiene: **tempo lineare** nella lunghezza, inferenza $\sim 5\times$ più
  veloce dei Transformer, scaling fino a $\sim 10^6$ passi, un Mamba-3B alla pari
  con Transformer di taglia doppia, e risultati forti anche su audio e genomica.
  Resta un preprint 2023, con i limiti che Mamba-2 affronterà.
```
