# L'energia come compatibilità: la cornice di LeCun

Nel 2006 Yann LeCun, con Sumit Chopra, Raia Hadsell, Marc'Aurelio Ranzato e Fu
Jie Huang, pubblica un lungo articolo didattico {cite}`lecun2006tutorial` che
compie il gesto inverso rispetto a Hopfield: non costruire una rete a energia,
ma mostrare che *quasi ogni modello di apprendimento può essere letto come una
funzione di energia*.

La ricetta è di una generalità spiazzante. Invece di dare un'energia a una
configurazione sola, la si dà a una **coppia**: da una parte quello che si ha
davanti (una foto, una frase, il presente), dall'altra una risposta possibile
(la parola che descrive la foto, la traduzione della frase, il fotogramma
successivo). Energia bassa se i due sono **compatibili**, cioè se stanno bene
insieme; alta se non c'entrano niente l'uno con l'altra. Una foto di gatto con
la parola «gatto» sta in basso; la stessa foto con la parola «aeroplano» sta
in alto. In simboli si scrive $E(\mathbf{x}, y)$, dove $\mathbf{x}$ è quello
che si ha davanti e $y$ la risposta candidata.

Rispondere, allora, significa cercare la risposta che rende l'energia minima;
imparare significa dare forma al paesaggio, **abbassare** l'energia delle
coppie giuste e **alzarla**, o tenerla alta, su quelle sbagliate. È una
cornice molto più larga di quel che sembra, e contiene una liberazione.

`````{tab} Elementare

Pensa a un buttafuori davanti a una festa a coppie. Il suo mestiere è
giudicare la coppia che ha davanti: questi due stanno bene insieme, passano;
questi due no. Nota che cosa *non* gli serve: non deve conoscere tutte le
persone della città, né compilare una classifica completa di tutti gli
abbinamenti possibili con le percentuali esatte che sommano a cento. Gli basta
un giudizio di compatibilità, coppia per coppia. Un modello a energia è questo
buttafuori: dare la *probabilità* di ogni risposta possibile (come fanno i
modelli probabilistici) è un lavoro immane, perché per dire «70%» su una
risposta devi aver messo in conto *tutte* le altre; dire «questa coppia sì,
quella no» è enormemente più economico, e per moltissimi compiti basta e
avanza.

C'è però un pericolo, e ha un nome preciso: il **collasso**. Immagina il
buttafuori pigro che ha scoperto la scorciatoia perfetta: dire sempre sì.
Chiunque si presenti, passa. Nessuna coppia si lamenta mai, e il suo giudizio
non vale più niente. Se durante l'addestramento premi il modello solo quando
dà energia bassa alle coppie giuste, la soluzione più comoda è dare energia
bassa *a tutto*. I rimedi sono due, e li vedremo fra poco: si può istruire il
buttafuori, oppure cambiargli la porta.

`````

`````{tab} Superiore

Un modello a energia (*energy-based model*, EBM) è una funzione
$E_\theta(\mathbf{x}, y)$ a valori reali, con parametri $\theta$, che misura quanto
$\mathbf{x} \in \mathcal{X}$ e $y \in \mathcal{Y}$ siano compatibili: valori bassi per
le coppie compatibili, alti per le altre. L'inferenza è un problema di
ottimizzazione:

$$
\hat{y} = \arg\min_{y \in \mathcal{Y}} E_\theta(\mathbf{x}, y),
$$

dove $\hat{y}$ è la risposta predetta: nessuna somma su $\mathcal{Y}$, solo
una ricerca del minimo. «Solo», però, va preso per quello che è: il minimo
esiste sotto ipotesi (continuità di $E_\theta$ e compattezza di $\mathcal{Y}$,
o coercività dell'energia), e quando $\mathcal{Y}$ è ad alta dimensione
trovarlo è a sua volta un'ottimizzazione non convessa, con gli stessi minimi
locali del resto del capitolo. Il vantaggio è di non dover sommare su
$\mathcal{Y}$, non di avere l'inferenza gratis. Un modello probabilistico si
ottiene come caso
particolare tramite la distribuzione di Gibbs:

$$
P_\theta(y \mid \mathbf{x}) = \frac{e^{-\beta E_\theta(\mathbf{x}, y)}}
{\displaystyle\int_{\mathcal{Y}} e^{-\beta E_\theta(\mathbf{x}, y')} \, dy'},
$$

dove $\beta > 0$ è una temperatura inversa e il denominatore è la funzione di
partizione **condizionata** $Z_\theta(\mathbf{x})$, l'integrale (o la somma) su
*tutte* le risposte possibili: un parente stretto della $Z(\theta)$ della
sezione precedente, ma non lo stesso oggetto, perché lì si integrava sui dati
e qui sulle risposte. Quando $\mathcal{Y}$ è grande o continuo e ad alta
dimensione ($y$
= un'immagine, un video, una frase, e allora sarebbe più onesto scriverlo
$\mathbf{y}$: il libro tiene $y$ tondo perché la stessa formula deve valere
quando $y$ è un'etichetta), $Z_\theta(\mathbf{x})$ può addirittura non esistere,
e quando esiste è intrattabile: è il muro
della sezione precedente. La tesi del tutorial è che per decidere, ordinare o
pianificare serve solo l’$\arg\min$, che di $Z$ non ha alcun bisogno:
rinunciare alla normalizzazione non è una perdita ma un vantaggio
computazionale.

`````

Due modelli che il capitolo ha già incontrato stanno dentro questa cornice, e
vale la pena vedere come, perché i due casi sono diversi. La macchina di
Boltzmann è quello facile: sull'energia si costruisce una probabilità, ed è
proprio il caso da cui la cornice si libera.

La memoria di Hopfield è più sottile. Lì l'energia non giudica una coppia:
guarda una cosa sola, la configurazione della rete. Il frammento rovinato che
le diamo in pasto non entra nel conto dell'energia e non sposta il paesaggio
di un millimetro, sceglie soltanto il punto da cui far partire la discesa. E
la risposta non è il fondovalle più basso di tutti: se lo fosse, quella rete
restituirebbe sempre lo stesso ricordo qualunque indizio le si desse, e non
sarebbe una memoria. La risposta è il fondovalle in cui si finisce partendo di
lì.

Che la risposta possa non essere la migliore in assoluto non è un imbarazzo,
ed è l'articolo stesso a metterlo in conto: in molte situazioni reali un
modello che va a cercarsi la risposta ne trova una approssimata, che può
essere il fondovalle più basso di tutti oppure no.

## Il collasso, e le due famiglie di rimedi

Il collasso, il buttafuori che dice sempre sì, conviene guardarlo da vicino
partendo da come si addestra un modello, perché è lì che si previene. Gli si
mostra una coppia, si guarda che voto le dà, e si ritocca il modello perché
quel voto somigli di più a quello che volevamo. La regola con cui si decide
«di quanto ha sbagliato» è una scelta, e di regole possibili ce ne sono
parecchie: sono quelle a decidere se il collasso è possibile oppure no.

L'articolo di LeCun le mette in fila in una tabella e accanto a ciascuna
scrive una cosa sola: quanto dislivello quella regola pretende fra una coppia
giusta e una sbagliata prima di dichiararsi soddisfatta. Quel dislivello ha un
nome, **margine**, ed è la misura di quanto una regola protegga dal collasso:
una superficie piatta non ha dislivelli, quindi una regola che ne pretende
qualcuno non l'accetterà mai.

La prima regola della lista è anche la più ingenua, quella che si limita ad
abbassare l'energia sui dati veri senza mai guardare nient'altro. Nella sua
casella del margine c'è scritto «none», nessuno {cite}`lecun2006tutorial`,
cioè con un'architettura qualunque non protegge affatto (l'articolo mostra poi
qualche architettura speciale in cui va bene lo stesso). Non è un difetto
sottile da manuale avanzato: è la prima riga della tabella.

`````{tab} Elementare

Come si costringe il buttafuori a fare sul serio? Due modi, ed è utile
tenerli distinti perché tutta la discussione dei prossimi anni ruota su
questa scelta.

Il primo: portargli davanti anche le coppie sbagliate e pretendere che le
respinga. Quelle coppie sbagliate, fabbricate apposta per fargliele
respingere, nel gergo del campo si chiamano **controesempi**, ed è la parola
con cui torneranno. Il metodo funziona, e ha un difetto che si vede subito
appena le coppie possibili diventano tante: quanti controesempi devi mostrare?
Per ogni coppia giusta ne esiste un numero enorme di sbagliate, e mostrargliene
un pugno alla volta è come puntellare un tendone con tre paletti.

E i controesempi non sono tutti uguali: quello che insegna qualcosa è la coppia
sbagliata che al buttafuori sembra più giusta di tutte, cioè l'impostore
migliore. Ma per trovarlo bisogna passare in rassegna la fila, che è esattamente
il lavoro del rispondere, e va rifatto a ogni passo dell'addestramento. Il
difetto è quindi doppio: i controesempi servono a migliaia, e trovarne uno buono
costa quanto rispondere a una domanda.

Il secondo: cambiare la porta invece di istruire il buttafuori. Se al posto
della porta c'è una fessura, di là non passa una folla qualunque cosa lui
dica: si costruisce il modello in modo che il numero di risposte a cui *può*
dare energia bassa sia limitato in partenza. Nessun controesempio da andare a
cercare.

E il fatto che il primo metodo non regga quando le risposte possibili sono
tantissime (per una foto sono più di quante se ne possano contare) è
esattamente l'argomento su cui poggia la proposta di LeCun per i *world
model*, i modelli che si costruiscono un'idea di come va il mondo per poterlo
prevedere, e a cui è dedicato più avanti un capitolo intero.

`````

`````{tab} Superiore

Le contromisure si dividono in due famiglie, secondo la distinzione che LeCun
ha reso canonica nel documento di posizione del 2022 {cite}`lecun2022path` e
che è diventata il vocabolario corrente del campo.

I **metodi contrastivi** alzano esplicitamente l'energia su risposte
sbagliate. Il tutorial definisce a questo scopo la *most offending incorrect
answer* $\bar{y}$ (la risposta scorretta con l'energia più bassa, cioè la più
insidiosa) e su di essa costruisce le loss a margine, per esempio la hinge

$$
\mathcal{L} = \max\!\big(0,\; m + E_\theta(\mathbf{x}, y) - E_\theta(\mathbf{x}, \bar{y})\big),
$$

dove $m > 0$ è il margine preteso fra coppia giusta e coppia sbagliata. E qui
c'è un costo che di solito passa sotto silenzio: $\bar{y}$ è a sua volta un
$\arg\min$ su $\mathcal{Y}$, cioè **un'inferenza completa a ogni passo di
addestramento**. Il problema dei metodi contrastivi non è soltanto quanti
controesempi servano; è che trovarne uno *buono* costa quanto rispondere. La
massima verosimiglianza appartiene alla stessa famiglia: il suo termine
contrastivo è la log-partizione, che solleva l'energia di *ogni* risposta con
forza proporzionale alla sua verosimiglianza, e nel limite $\beta \to \infty$
la loss NLL degenera nella loss del percettrone generalizzata, che ne solleva una sola,
quella a energia minima {cite}`lecun2006tutorial`. Contrastive divergence, NCE
e le loss a margine sono tutte varianti di una stessa domanda: **quali
risposte tirare su, e con che forza**. Il male comune, in alta dimensione, è
che i controesempi non bastano mai a puntellare un'intera superficie.

I **metodi regolarizzati o architetturali** impediscono il collasso per
costruzione, limitando il *volume* dello spazio a bassa energia invece di
sollevarlo punto per punto: colli di bottiglia sulle variabili latenti,
vincoli di sparsità, quantizzazione dei codici (il VQ-VAE del capitolo
sull'audio è esattamente questo) e termini che impongono varianza o
decorrelazione alle rappresentazioni. L'intuizione è che un modello con pochi
gradi di libertà non *può* dare energia bassa a tutto, e allora non serve
alcun controesempio a impedirglielo. È la famiglia su cui LeCun scommette per
i world model, ed è il motivo per cui, nel capitolo sui *world model*, la JEPA
si difende dal collasso senza mai fabbricare un solo controesempio.

`````

Due strade, dunque, e quale sia quella giusta non è affatto deciso. La
questione compare anche nell'elenco che LeCun ripete nelle sue conferenze,
quello delle quattro cose a cui il campo dovrebbe rinunciare: è la terza delle
quattro, e la sezione che segue la riprende per l'ultima volta insieme alle
altre.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un modello a energia può giudicare **coppie**: questo con questo sta bene
  insieme, questo con quest'altro no. È il buttafuori davanti alla festa, e
  gli basta un giudizio alla volta: non deve conoscere tutta la città né
  compilare la classifica di tutti gli abbinamenti possibili con le
  percentuali esatte.
- Rispondere, allora, è cercare la risposta che sta meglio con quello che si
  ha davanti. Imparare è abbassare il terreno sotto le coppie giuste e
  alzarlo sotto quelle sbagliate.
- Il pericolo ha un nome, il **collasso**, ed è il buttafuori pigro che dice
  sempre sì. Se durante l'addestramento si premia soltanto il sì alle coppie
  giuste, la scorciatoia perfetta è dire sì a tutti: nessuno si lamenta, e il
  giudizio non vale più niente.
- I rimedi sono due, e la scelta fra loro è una discussione ancora aperta:
  mostrargli anche le coppie sbagliate e pretendere che le respinga, oppure
  stringere la porta, cioè costruirlo in modo che dire sì a tutti gli sia
  fisicamente impossibile. Il primo va in crisi quando le coppie sbagliate
  sono troppe da mostrare, ed è quasi sempre il caso; il secondo è la
  scommessa di LeCun per i modelli del mondo.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- La cornice dell’**energy-based learning** {cite}`lecun2006tutorial`: ogni
  modello è una $E(\mathbf{x}, y)$ che misura la compatibilità fra input e risposta;
  inferire è $\arg\min_y E$, imparare è abbassare l'energia delle coppie
  giuste e alzarla sulle sbagliate. I modelli probabilistici sono il caso
  particolare normalizzato, e $Z$ è il costo che conviene evitare.
- Il pericolo è il **collasso**: energia bassa ovunque. Nella tabella delle
  loss del tutorial, quella che si limita ad abbassare l'energia sui dati ha
  margine «none», cioè non protegge affatto con un'architettura qualunque.
- Due famiglie di rimedi: **contrastivi** (alzare l'energia su risposte
  sbagliate, a partire dalla *most offending incorrect answer*) e
  **regolarizzati/architetturali** (limitare per costruzione il volume dello
  spazio a bassa energia). I primi non scalano in alta dimensione; i secondi
  sono la scommessa di LeCun per i world model.
- Vista da qui, la massima verosimiglianza è un metodo contrastivo: solleva
  l'energia di tutte le risposte, pesandole con la loro probabilità. Da cui
  il costo, e da cui l'idea di sostituirla.
```
`````
