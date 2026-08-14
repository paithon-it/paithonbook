# Panorama e limiti

Abbiamo attraversato due capitoli (l'attenzione lineare e gli *state space
model*) che sembravano raccontare storie diverse: uno partiva dai Transformer e
toglieva all'attenzione il pezzo che la faceva costare tanto, l'altro dai
sistemi dinamici continui e li misurava a intervalli. Eppure siamo arrivati, ogni volta, allo stesso posto. Vale la
pena, ora che li abbiamo entrambi in mano, mettere i pezzi in fila e chiedersi
cosa abbiamo davvero costruito, dove regge e dove no.

Il punto di partenza era un difetto ben preciso. Nel capitolo sui Transformer,
confrontandoli con le RNN, avevamo trovato il loro tallone d'Achille: far
guardare ogni parola a tutte le altre costa in tempo e memoria **al quadrato**
nella lunghezza della sequenza. Raddoppiare il testo quadruplica il lavoro. Da
lì nascevano le finestre di contesto limitate e una vasta ricerca su come
collegare le parti di un testo lungo senza convocare ogni volta l'assemblea
plenaria. Le architetture di questi due capitoli sono una delle risposte più
promettenti a quel problema. E la tesi che le tiene insieme, ripetuta di
sezione in sezione, è una sola. Tengono tutte un riassunto che non si allarga
mai, lo aggiornano parola per parola, e lo stesso conto lo sanno fare in due
modi: tutto insieme quando c'è da imparare, un pezzo alla volta quando c'è da
rispondere. In gergo si chiamano **reti ricorrenti lineari a stato di
dimensione fissa**: si addestrano in parallelo come un Transformer e fanno
inferenza un token alla volta a costo costante come una RNN.

## Un'unica famiglia

Ricapitoliamo l'immagine con cui il capitolo precedente ha descritto questa
memoria. Funziona come uno schedario: a ogni parola si archivia una
voce nuova, formata da un'**etichetta** e da un **contenuto**, e per rileggere
si presenta un'etichetta e si riceve indietro ciò che le assomiglia di più. Lo
schedario ha un numero fisso di **cassetti**, sempre quello, e ogni voce nuova
lascia un segno un po' in tutti.
Prima di archiviare la voce nuova, però, quello che c'è già viene sbiadito un
po': è la **transizione**, ed è l'unica cosa su cui le architetture di questi
due capitoli sono davvero diverse fra loro. Da RetNet a Mamba, cambia come e
quanto si sbiadisce, e chi lo decide.

`````{tab} Elementare

Immagina un unico apparecchio con poche manopole. Il corpo della macchina è
sempre lo stesso: una memoria che a ogni parola scrive una nuova voce e ne
rilegge le vecchie. Cambiare architettura non vuol dire cambiare macchina, ma
girare tre manopole. La prima decide **quanto è grande** la memoria. La
seconda decide **come sbiadisce** il passato quando arriva il presente: si può
non sbiadire affatto, sbiadire tutto in blocco della stessa quantità, sbiadire
cassetto per cassetto in modo diverso, oppure (la versione più raffinata)
cancellare *di mira* solo la vecchia voce che sta per essere riscritta. Non
sbiadire affatto, però, non vuol dire tenere tutto: i cassetti restano quelli,
le voci continuano ad ammucchiarsi una sopra l'altra, e più avanti in questa
pagina vedremo che è il limite di fondo di tutta la famiglia. La terza manopola
decide se queste scelte sono **fisse**, uguali
per ogni parola, o se invece è la parola stessa a deciderle, momento per
momento. RetNet, GLA, DeltaNet, Mamba: sono lo stesso apparecchio con le
manopole in posizioni diverse. Nomi e sigle diversi per un solo schema. E le
posizioni si combinano: c'è chi gira insieme la manopola dello sbiadire in
blocco e quella del cancellare di mira, e si chiama **Gated DeltaNet**.

`````

`````{tab} Superiore

In formule, lo stato è una matrice $\mathbf{S}_t \in \mathbb{R}^{d\times d}$, una
memoria che associa **chiavi a valori**; si scrive per prodotto esterno e si
legge per proiezione:

$$
\mathbf{S}_t = \mathbf{S}_{t-1}\, (\text{transizione}_t) + \mathbf{v}_t\, \mathbf{k}_t^\top,
\qquad
\mathbf{o}_t = \mathbf{S}_t\, \mathbf{q}_t,
$$

dove $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ sono query, chiave e valore del token $t$ (gli stessi
introdotti nell'attenzione dei Transformer) e $\mathbf{v}_t \mathbf{k}_t^\top$ è la nuova coppia
scritta in memoria. La transizione moltiplica **a destra**, e non è un
dettaglio di scrittura: con lo stato fatto di colonne indicizzate dalle
chiavi, è da quel lato che il fattore agisce sui canali di chiave (e che
$\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top$ cancella la traccia lasciata da $\mathbf{k}_t$); a sinistra
sbiadirebbe i canali dei valori, che è un'altra cosa. Nelle sezioni precedenti
di questo capitolo la stessa transizione compariva **a sinistra**,
$\mathbf{h}_t = \bar{\mathbf{A}}\mathbf{h}_{t-1} + \dots$, e non è una contraddizione: lì lo stato
era il vettore colonna $\mathbf{h}$ di un singolo canale, cioè una riga di $\mathbf{S}$
trasposta, e trasporre scambia i due lati. L'unico caso in cui il lato non
conta davvero è quello di un fattore **scalare**, come l'$\alpha_t$ di Mamba-2,
che commuta con tutto.

Gli assi di progetto sono tre, e ciascuno ha un prezzo e un guadagno.

**1. La dimensione dello stato.** Quanto è grande $d$ (o, per gli SSM, la
dimensione $N$ dello stato per canale). Uno stato più grande è una memoria più
capiente (più coppie chiave-valore ci stanno senza pestarsi i piedi) ma costa
più calcolo e più memoria a ogni passo, $O(d^2)$ per l'aggiornamento. È la
manopola della capacità grezza.

**2. La struttura della transizione.** È la vera firma di ogni architettura, e
la tabella unificante del capitolo precedente la metteva in fila per struttura
via via più ricca:

$$
\underbrace{\mathbf{I}}_{\text{lin. attn}}
\;\to\;
\underbrace{\alpha_t \mathbf{I}}_{\text{RetNet, Mamba-2}}
\;\to\;
\underbrace{\mathrm{Diag}(\alpha_t)}_{\text{GLA}}
\;\to\;
\underbrace{\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top}_{\text{DeltaNet}}
\;\to\;
\underbrace{\alpha_t\,(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)}_{\text{Gated DeltaNet}}
$$

dove $\alpha_t$ è un fattore di **oblio** compreso fra $0$ e $1$ (un numero
solo dove moltiplica l'identità, un valore per canale dove compare dentro
$\mathrm{Diag}$) e $\beta_t \in (0,1)$ la
**forza di riscrittura** della delta rule. Si va dall'accumulo puro (identità,
non si dimentica nulla) al decadimento scalare uniforme, a quello diagonale
per-canale, alla correzione mirata di Householder che *cancella* la vecchia
associazione prima di scrivere la nuova, fino alla combinazione dei due
(decadimento globale *più* correzione mirata) del Gated DeltaNet
{cite}`yang2024gateddelta`. Due precisazioni, perché la fila non è una scala
regolare. La prima: i primi tre gradini sono nidificati (ciascuno contiene il
precedente come caso particolare), ma il passo da $\mathrm{Diag}(\alpha_t)$
alla delta rule non è un'inclusione, perché il decadimento diagonale e la
correzione mirata di Householder sono capacità **complementari** e nessuna
delle due contiene l'altra. La seconda: quello che il Gated DeltaNet unisce
non è la coppia appena nominata. Il suo $\alpha_t$ è uno **scalare**, quindi
mette insieme il decadimento *globale* (il secondo gradino) con la delta rule,
e contiene il secondo e il quarto ma non il terzo: il gating per canale della
GLA resta fuori anche dall'ultimo gradino. Lungo tutta la catena, però, il
conto è lo stesso: si paga in complessità della transizione (via via più
difficile da rendere parallelizzabile) ciò che si guadagna in *state tracking*
e in *recall* preciso.

**3. Il grado di dipendenza dai dati.** La transizione può essere **fissa**
(scelta a priori, uguale per ogni token, come il $\gamma$ di RetNet o il
decadimento della RWKV-4) oppure **data-dipendente**, generata dall'input
token per token, come in GLA, DeltaNet e Mamba {cite}`gu2023mamba`. La
dipendenza dai dati è ciò che compra il *ragionamento basato sul contenuto*:
decidere cosa tenere e cosa lasciar cadere in base a *ciò che si legge*, non
solo a quanto tempo è passato. È il salto che separa un metal detector
regolato una volta per tutte da una guardia che valuta caso per caso.

Su questa mappa gli SSM non sono un'isola. La **dualità stato-attenzione** (SSD)
di Mamba-2 {cite}`dao2024mamba2`, che abbiamo visto nella sezione su Mamba-2,
dimostra che un SSM con transizione **scalare per identità** ($\alpha_t \mathbf{I}$) è
*esattamente* un'attenzione lineare mascherata: è la seconda riga della tabella,
raggiunta dal versante dei sistemi dinamici invece che da quello
dell'attenzione. Le due famiglie che abbiamo raccontato in capitoli separati
sono, alla lettera, due viste della stessa cosa.

`````

## Il collo di bottiglia dello stato fisso

Fin qui i pregi. Ora il limite, che va detto senza giri di parole: **un
riassunto di taglia fissa non può fare tutto ciò che fa l'attenzione piena**.
Non è un difetto di come è stato costruito, di quelli che prima o poi qualcuno
aggiusta: è la conseguenza dell'essere di taglia fissa. Il punto in cui si vede
è ritrovare alla lettera, in un contesto lunghissimo, un dettaglio preciso
letto centinaia di pagine prima; nel gergo del campo, il *recall associativo
esatto*.

`````{tab} Elementare

La differenza è quella tra un quaderno di appunti e una biblioteca. L'attenzione
piena dei Transformer è la biblioteca: conserva *ogni* parola letta, e quando le
chiedi «cosa diceva esattamente quella frase a pagina 900?» va allo scaffale e la
ripesca alla lettera. Il prezzo è doppio. Prima lo spazio: la biblioteca cresce
senza fine, un ripiano per ogni pagina. Poi, e conta di più, il lavoro: ogni
pagina nuova va confrontata con tutte quelle che sono già sugli scaffali, e
così un libro lungo il doppio non costa il doppio ma il quadruplo. È il costo
quadratico che volevamo evitare.

Le architetture di questi due capitoli sono invece un quaderno di appunti di
taglia fissa. A ogni pagina che leggi aggiorni i tuoi appunti: riassumi,
sovrascrivi, cancelli il vecchio per far posto al nuovo. Il quaderno costa
pochissimo: resta sempre dello stesso spessore per quante pagine tu legga. Ma
proprio perché non cresce, non può contenere tutto: se dopo mille pagine ti
chiedo di **citare a memoria** una frase precisa di pagina 900, il quaderno ti
dà il senso generale, non le parole esatte. Le hai riassunte, non trascritte.
Che sia proprio così lo si misura con due prove fatte apposta: nascondere una
frase in un testo lunghissimo e chiedere di ripescarla alla lettera (è *l'ago
nel pagliaio*), oppure riempire la memoria di centinaia di coppie nome-numero e
chiedere a bruciapelo il numero di un nome qualsiasi. Questo è il compromesso: memoria che costa poco e non cresce, in cambio della
rinuncia al ricordo alla lettera di ogni singolo dettaglio.

`````

`````{tab} Superiore

La ragione è di capacità d'informazione. Uno stato $\mathbf{S} \in \mathbb{R}^{d\times
d}$ ha un numero finito di gradi di libertà: come osservato già nel lavoro sui
*fast weight programmer* {cite}`schlag2021linear`, in dimensione $d$ non
esistono più di $d$ direzioni mutuamente ortogonali. Attenzione a come si legge
questo limite, perché la lettura sbagliata è la più comoda: l'interferenza fra
associazioni **non compare oltre una soglia**. Come si è visto nel capitolo
precedente, con chiavi casuali il *crosstalk* cresce da subito, come
$\sqrt{N/d}$ nel numero $N$ di coppie scritte (in questa formula, e solo qui,
$N$ conta le coppie: non è la dimensione dello stato di un SSM, che nel resto
del capitolo porta la stessa lettera), e intorno a $N \approx d$ vale
ormai quanto il valore che si sta cercando. Non c'è un punto in cui la memoria
«si riempie»: c'è un degrado continuo, che a un certo punto diventa
intollerabile per il compito che si ha davanti. L'attenzione piena non ha
questo tetto: la sua «memoria» è la KV
cache, che conserva **tutte** le coppie chiave-valore dei token passati, al
prezzo di crescere linearmente con la lunghezza (ed è quel prezzo a rendere il
costo complessivo quadratico).

Questo divario si misura con i benchmark di **recall**. Nel *needle in a
haystack* si nasconde un fatto preciso (l'ago) in un contesto molto lungo (il
pagliaio) e si chiede al modello di recuperarlo verbatim. In **MQAR**
(*Multi-Query Associative Recall*) si presentano molte coppie chiave-valore e
si interroga il modello su chiavi arbitrarie. Sono proprio i compiti su cui la
dimensione dello stato diventa il collo di bottiglia. I progressi nella
transizione aiutano (la delta rule di DeltaNet, che *riscrive* invece di
accumulare, sposta in avanti la frontiera proprio perché usa meglio lo spazio
disponibile) ma non spostano il tetto:
finché lo stato è di taglia fissa, per il retrieval esatto su contesti
sufficientemente lunghi l'attenzione piena resta superiore. Non è una gara che
le ricorrenze lineari possano vincere sul suo stesso terreno; è una gara che
conviene **non giocare da sole**.

`````

## Il meglio dei due mondi: gli ibridi

Se una delle due vince sul ricordo alla lettera e l'altra sul costo, la mossa
ovvia è non scegliere: pochi strati di biblioteca dove serve ripescare la
citazione esatta, molti strati di quaderno per tutto il resto. È la strada che
ricorre in tutti i lavori recenti, e sono le architetture **ibride**: alternano
**pochi strati di attenzione piena** a **molti strati lineari o
SSM**. Il costo che cresce al quadrato non
sparisce, ma lo paga una minoranza di strati, e finché il contesto non diventa
smisurato pesa poco sul totale.

`````{tab} Elementare

È la logica di una squadra ben assortita. In un'inchiesta giornalistica non
metti solo archivisti né solo cronisti: tieni pochi archivisti (quelli che
sanno ripescare il documento esatto quando serve la citazione precisa) e molti
cronisti veloci che tengono il filo del racconto senza rileggersi ogni volta
tutto l'archivio. La stragrande maggioranza del lavoro la fanno i cronisti, a
costo basso; gli archivisti intervengono nei pochi momenti in cui l'esattezza
è decisiva. Le architetture ibride sono organizzate così: qualche strato che
conserva tutto e ricorda alla lettera, il resto a memoria costante. Sono fatti
così **Jamba** e **Samba**, e le versioni miste di architetture che abbiamo già
incontrato. Non è un compromesso al ribasso, è la divisione dei compiti che
oggi rende meglio.

`````

`````{tab} Superiore

L'idea ricorre in tutti i lavori recenti, con dosaggi diversi. **Jamba**
(AI21 Labs, 2024) intervalla strati di attenzione e strati Mamba in una
proporzione sbilanciata verso questi ultimi, aggiungendo esperti selettivi
(*mixture-of-experts*), e regge contesti molto lunghi con una occupazione di
memoria contenuta {cite}`lieber2024jamba`. **Samba** {cite}`ren2024samba`
(Microsoft, 2024) combina strati Mamba con strati di **attenzione a finestra
scorrevole** (*sliding-window attention*): l'attenzione locale copre il
contesto ravvicinato, Mamba porta la memoria a lungo raggio, e insieme
estrapolano a lunghezze molto oltre quella di addestramento. La stessa ricetta
appare come variante ibrida sia del Gated DeltaNet {cite}`yang2024gateddelta`
(combinato con attenzione a finestra o globale) sia di Mamba-2
{cite}`dao2024mamba2`, il cui articolo studia esplicitamente l'aggiunta di
pochi strati di attenzione a uno stack SSM.

La tendenza è la stessa in tutti questi lavori, e il messaggio non è «l'ibrido
vince sempre»: è qualcosa di più solido e più modesto. I due ingredienti hanno
punti di forza **complementari** (recall verbatim l'uno, costo e memoria
costanti l'altro), e complementare vuol dire che mescolarli in proporzione
sbilanciata (poca attenzione, molta ricorrenza) costa poco e rende quasi
quanto l'attenzione piena. È il motivo per cui la ricetta ricompare, con
dosaggi diversi, in architetture nate da gruppi che non si parlano.

`````

## Dove sta andando

Resta la domanda che aleggia su tutto il percorso: queste architetture sono i
«killer dei Transformer»? La risposta onesta è no, e non perché siano deboli,
ma perché la domanda è mal posta. Non stiamo assistendo a una sostituzione, ma
all'assestamento di un **ecosistema misto**, in cui strumenti diversi occupano
nicchie diverse.

Dove le ricorrenze lineari e gli SSM danno il meglio è chiaro, e sono
territori di crescente importanza. Il **contesto lunghissimo**, dove il costo
quadratico dell'attenzione piena diventa proibitivo e uno stato che non cresce
è un enorme vantaggio. Poi l'**inferenza a memoria costante**. Un Transformer,
per non rileggersi tutto a ogni parola che scrive, tiene da parte quello che ha
già calcolato: è la *KV cache*, l'archivio delle etichette e dei contenuti di
tutte le parole viste, e si gonfia a ogni parola generata. Un modello a stato
fisso non ne ha bisogno, e la memoria che occupa mentre scrive resta quella con
cui è partito, il che è decisivo quando si serve il modello a molti utenti in
parallelo. Poi gli scenari in **streaming**, dove i dati arrivano
in flusso continuo e non si può rileggere tutto da capo a ogni passo. E infine i
**dispositivi con poca memoria** (telefoni, sistemi embedded), dove una
memoria fissa e prevedibile vale più di qualche punto di qualità sul ricordo
alla lettera.

Conviene chiudere con la stessa prudenza con cui, nel capitolo sui
Transformer, avevamo messo in guardia dalle profezie: questo campo brucia in
fretta le previsioni. Già lì, tra le tendenze future, gli *state space model*
comparivano come la linea di ricerca che rimetteva in gioco idee ricorrenti
proprio dove l'attenzione costa troppo, ed è la storia che questi due capitoli
hanno raccontato per esteso. La lezione di fondo, però, è la stessa
dell'intero libro: nessuna architettura vince per sempre. L'attenzione non ha
«ucciso» le RNN, e le RNN lineari non uccideranno l'attenzione. Chi conosce le
idee semplici che stanno sotto (una memoria in cui ogni parola archivia una
voce nuova, un modo di sbiadire il passato che decide cosa dimenticare, due
forme dello stesso calcolo) non insegue le mode: le legge, e riconosce lo
stesso scheletro sotto il prossimo nome che farà rumore.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **Una sola famiglia**: attenzione lineare (RetNet, GLA, DeltaNet, RWKV,
  xLSTM) e *state space model* (S4, Mamba) sono lo stesso apparecchio, una
  memoria di taglia fissa che a ogni parola scrive una voce nuova e rilegge le
  vecchie. Si addestrano tutti insieme, in parallelo, e generano una parola
  alla volta con una memoria che non cresce mai.
- **Tre manopole di progetto**: quanto è grande la memoria (la capacità
  grezza); **come sbiadisce il passato** quando arriva il presente (non
  dimenticare nulla, sbiadire tutto in blocco, sbiadire cassetto per cassetto,
  oppure cancellare di mira la vecchia voce che sta per essere riscritta); e se
  queste scelte sono fisse per ogni parola oppure decise dalla parola stessa,
  che è ciò che compra il ragionamento basato sul contenuto. Sbiadire e
  cancellare di mira la vecchia voce non sono uno il perfezionamento
  dell'altro: fanno cose diverse, e c'è un'architettura che le usa tutt'e due
  insieme, il **Gated DeltaNet**. È DeltaNet con in più la manopola dello
  sbiadire, quella che sbiadisce tutto in blocco; lo sbiadire cassetto per
  cassetto, invece, resta fuori anche da lui.
- **La dualità** di Mamba-2 {cite}`dao2024mamba2` dimostra che uno *state space
  model* che sbiadisce tutto in blocco è esattamente un'attenzione lineare che
  guarda solo all'indietro: le due famiglie sono due viste della stessa cosa.
- **Il limite onesto**: una memoria che non cresce è un quaderno di appunti, non
  una biblioteca. Va benissimo per il senso del discorso, ma se dopo mille
  pagine chiedi di **citare alla lettera** una frase di pagina 900, il quaderno
  non ce l'ha: l'aveva riassunta, non trascritta. L'attenzione piena conserva
  ogni parola letta e su quel compito resta superiore, al prezzo di uno scaffale
  che cresce senza fine. Si misura con due prove fatte apposta: nascondere una
  frase in un testo lunghissimo e chiedere di ripescarla (è *l'ago nel
  pagliaio*), e riempire la memoria di centinaia di coppie nome-numero per poi
  chiedere a bruciapelo il numero di un nome qualsiasi.
- **Gli ibridi** sono la ricetta che ricorre in tutti i lavori recenti: pochi
  strati di attenzione piena (gli archivisti, che ripescano la citazione esatta
  quando serve) intervallati a molti strati a memoria fissa (i cronisti, che
  tengono il filo a costo basso). Fanno così Jamba {cite}`lieber2024jamba`,
  Samba {cite}`ren2024samba` e le varianti ibride di Gated DeltaNet
  {cite}`yang2024gateddelta` e Mamba-2.
- **Prospettiva sobria**: non un «killer dei Transformer» ma un **ecosistema
  misto**. Le ricorrenze lineari danno il meglio sui testi lunghissimi, quando
  la memoria deve restare costante, sui dati che arrivano in flusso continuo e
  sui dispositivi con poca memoria. Nessuna architettura vince per sempre: chi
  conosce le idee semplici riconosce lo stesso scheletro sotto ogni nuovo nome.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- **Una sola famiglia**: attenzione lineare (RetNet, GLA, DeltaNet, RWKV, xLSTM)
  e *state space model* (S4, Mamba) sono tutte **RNN lineari a stato fisso**
  $\mathbf{S}_t = \mathbf{S}_{t-1}\, (\text{transizione}_t) + \mathbf{v}_t \mathbf{k}_t^\top$, con lettura
  $\mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t$. Si addestrano in parallelo, fanno
  inferenza ricorrente a memoria costante per token.
- **Tre manopole di progetto**: la dimensione dello stato (capacità), la
  struttura della transizione ($\mathbf{I} \to \alpha_t \mathbf{I} \to
  \mathrm{Diag}(\alpha_t) \to \mathbf{I}-\beta_t \mathbf{k}_t \mathbf{k}_t^\top
  \to \alpha_t(\mathbf{I}-\beta_t \mathbf{k}_t \mathbf{k}_t^\top)$, via via più
  ricca, ma non è una scala in cui ogni gradino contiene il precedente:
  decadimento per canale e cancellazione mirata fanno cose diverse, e l'ultimo
  gradino unisce quest'ultima con il decadimento **globale**, non con quello
  per canale), e quanto è data-dipendente (fisso vs generato
  dall'input, che compra il ragionamento basato sul contenuto).
- **La dualità SSD** di Mamba-2 {cite}`dao2024mamba2` dimostra che un SSM a
  transizione scalare ($\alpha_t \mathbf{I}$) è esattamente un'attenzione
  lineare mascherata: SSM e attenzione lineare sono due viste della stessa cosa.
- **Il limite onesto**: uno stato di dimensione fissa è un **collo di
  bottiglia** per il *recall associativo esatto* su contesti lunghissimi, e non
  perché si riempia a una certa soglia: l'interferenza fra associazioni cresce
  da subito, come $\sqrt{N/d}$, e intorno a $N\approx d$ coppie scritte vale
  quanto il valore cercato. L'attenzione piena, che conserva ogni token nella
  KV cache, resta superiore sul retrieval verbatim (benchmark *needle in a
  haystack*, MQAR): al prezzo del costo quadratico.
- **Gli ibridi** sono la ricetta che ricorre in tutti i lavori recenti: pochi
  strati di attenzione piena intervallati a molti strati lineari/SSM (Jamba
  {cite}`lieber2024jamba`, Samba {cite}`ren2024samba`, le varianti ibride di
  Gated DeltaNet {cite}`yang2024gateddelta` e Mamba-2). Recall esatto dove
  serve, costo basso per il resto.
- **Prospettiva sobria**: non un «killer dei Transformer» ma un **ecosistema
  misto**. I punti di forza delle ricorrenze lineari sono il contesto
  lunghissimo, l'inferenza a memoria costante, lo streaming e i dispositivi con
  poca memoria. Nessuna architettura vince per sempre: chi conosce le idee
  semplici riconosce lo stesso scheletro sotto ogni nuovo nome.
```

`````
