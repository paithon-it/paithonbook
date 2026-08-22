# Modelli latenti e inferenza variazionale

In una scuola preparatoria inglese, trentatré ragazzi fanno gli esami di
quattro materie: le materie classiche, il francese, l’inglese e la matematica.
I voti si somigliano più di quanto dovrebbero. Chi va bene in una tende ad
andare bene anche nelle altre, e quel «tende» si può misurare: si prendono due
materie, si guardano le due graduatorie della classe e si chiede quanto vadano
d’accordo, con un numero che vale 0 se non c’entrano niente l’una con l’altra e
1 se sono la stessa identica graduatoria. Charles Spearman fa il conto per ogni
coppia di materie, e poi, per ciascuna materia, la media dei tre confronti con
le altre tre: gli vengono 0,77 per le materie classiche, 0,72 per il francese,
0,70 per l’inglese, 0,67 per la matematica {cite}`spearman1904general`. Sono
numeri alti, e fin qui nessuno si stupisce: i bravi sono bravi.

Poi Spearman fa una cosa che con la scuola non c’entra niente. Mette gli stessi
ragazzi davanti a due suoni quasi identici e chiede quale dei due sia più
acuto. È un compito da orecchio, non da studio: non si copia, non si ripassa la
sera prima, e l’unico allenamento che conta è la musica. E il risultato mette in fila le quattro materie
**nello stesso ordine di prima**: 0,60 con le materie classiche, 0,56 col
francese, 0,45 con l’inglese, 0,39 con la matematica. (Sono le correlazioni
grezze su tutta la scuola; restringendole ai ventidue ragazzi che studiavano
musica salgono tutte, e l’ordine resta identico.)

Distinguere due note non ha niente a che vedere con declinare *rosa*, e non ha
niente a che vedere con risolvere un’equazione. Eppure chi fa meglio l’una
tende a fare meglio anche le altre, e sempre nella stessa graduatoria. La
spiegazione che Spearman propone nel 1904 è tutta nella sua forma: sotto a
tutte queste prove c’è **una quantità sola**, che nessuno ha misurato e che
nessuno misurerà mai, e ogni prova è quella quantità più il proprio scarto. Le
materie non si somigliano fra loro: si somigliano perché sono figlie della
stessa cosa.

Se quella quantità esista davvero, e che cosa sia, è oggetto di una discussione
che dura da oltre un secolo, e in questo libro non prendiamo posizione: quello
che ci serve non è la conclusione, è **la mossa**. La mossa è sopravvissuta
alla discussione, ha preso un nome (**variabile latente**, dal latino *latere*,
«stare nascosto») e ha una macchina matematica che la rende operativa, che
Spearman inventò per sostenere la sua tesi e che oggi si chiama analisi
fattoriale. Quella macchina è l’antenata di tutto questo capitolo, e ne è la
versione più semplice: quella in cui la quantità nascosta e le cose visibili
sono legate da somme e moltiplicazioni, e da nient’altro.

Anche l’altra metà del titolo si scioglie qui. **Inferenza** è il mestiere di
risalire alla causa nascosta a partire da quello che si vede, cioè fare la
strada al contrario; **variazionale** dice come lo faremo, e cioè rinunciando
alla risposta esatta e cercando la migliore dentro una famiglia di risposte
semplici, scelta da noi. La seconda metà del capitolo non fa altro che questo.

## La mossa: spiegare il visibile con l’invisibile

Chi vuole costruire una macchina che fabbrica dati nuovi ha davanti un compito
scoraggiante: scrivere una formula per la probabilità di un dato. Quanto è
probabile *questa* fotografia?

La domanda suona strana, perché una fotografia c’è o non c’è, e conviene
scioglierla subito visto che regge tutto il capitolo: vuol dire **quanto ci si
aspettava di vedere una cosa così**. La foto di un gatto nero su un muro è
probabile; la stessa foto con il muro fatto di puntini colorati a caso non lo
è. E chi sa rispondere sa anche fabbricare, perché sapere quali immagini sono
attese è sapere quali produrre.

Il guaio è che quella formula nessuno la sa scrivere. Il dato è enorme
(un’immagine a colori piccola sono già centinaia di migliaia di numeri) e le
sue parti sono legate fra loro in modi che sfuggono: due pixel vicini hanno
quasi sempre lo stesso colore, tranne sui contorni, e dove passano i contorni
dipende da che cosa c’è nella foto.

La mossa della variabile latente è cambiare domanda. Invece di descrivere il
dato, si descrive **come è nato**: prima si sorteggia qualcosa che non si vede,
e poi, a partire da quel qualcosa, si sorteggia il dato. Il modello si scrive
allora in due pezzi, e sono due pezzi semplici; la complicazione che si vede
nasce dal fatto che il primo dei due non lo si osserva mai.

`````{tab} Elementare

Nel cassetto ci sono due sacchetti di biglie. In uno le biglie sono piccole,
sui 12 millimetri, con un paio di millimetri di variazione da una all’altra;
nell’altro sono grandi, sui 20 millimetri, sempre con la sua variazione. Tu
peschi a occhi chiusi: prima una monetina decide il sacchetto, poi peschi una
biglia da lì, e misuri solo la biglia. Quale sacchetto fosse, non lo guardi mai
e non lo scrivi da nessuna parte.

Fai mille pescate e disegni l’istogramma delle misure, cioè il grafico che
dice, per ogni misura, quante biglie ci sono cadute. Non viene una gobba sola:
ne vengono due, una intorno a 12 e una intorno a 20, e in mezzo un
avvallamento. Eppure dentro ciascun sacchetto le misure erano la cosa più
semplice del mondo, una gobba e basta. La forma complicata (due gobbe) non l’ha
messa nessuno: è comparsa perché una parte della storia, cioè quale sacchetto,
è rimasta nascosta.

Se qualcuno ti dicesse a ogni pescata da quale sacchetto viene la biglia, il
conto sarebbe una banalità: guardi il sacchetto, sai la sua gobba, hai finito.
Il conto diventa difficile proprio perché quel dato manca. Tutto quello che
segue nasce da lì, ed è anche la ragione per cui la mossa paga.

Nei casi che ci interessano i sacchetti non sono due, sono infiniti: al posto
delle due scatole c’è un righello, si sorteggia un punto qualunque, e una
regola dice attorno a che misura stanno le biglie del sacchetto che sta lì.

E i sacchetti nessuno te li ha mostrati. Torna ai due del cassetto e cambiali
di poco, 12 e 14 millimetri con la stessa variazione: l’istogramma fa una gobba
sola, e a guardarla non diresti mai che i sacchetti erano due. Sei tu a
supporli, perché supponendoli i conti tornano più semplici, ed è una scommessa
che può anche non pagare.

`````

`````{tab} Superiore

Un **modello a variabile latente** non scrive $p(\mathbf{x})$ direttamente:
scrive una distribuzione congiunta su ciò che si osserva e su ciò che non si
osserva, e ottiene la prima **marginalizzando** la seconda, cioè sommando su
tutti i valori che la variabile nascosta poteva prendere:

$$
p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})\, \mathrm{d}\mathbf{z},
$$

dove $\mathbf{x}$ è il dato osservato, $\mathbf{z}$ la variabile latente,
$p(\mathbf{z})$ il **prior** (la distribuzione da cui $\mathbf{z}$ viene
sorteggiato, scelta da noi e di solito semplicissima), $p_\theta(\mathbf{x}
\mid \mathbf{z})$ la **verosimiglianza** del dato dato il latente, e $\theta$ i
parametri del modello generativo. L’integrale diventa una somma quando
$\mathbf{z}$ è discreto.

Il caso discreto il libro l’ha già visto e non l’ha chiamato così: nella
mistura di gaussiane della sezione su riduzione e clustering, $z \in \{1,
\dots, K\}$ è la componente da cui l’esempio proviene, $p(z = k) = \pi_k$ il
suo peso e $p(\mathbf{x} \mid z = k) = \mathcal{N}(\mathbf{x};
\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ la sua campana, con
$\boldsymbol{\mu}_k$ il centro della componente $k$ e
$\boldsymbol{\Sigma}_k$ la sua covarianza; la densità osservata

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k\,
\mathcal{N}(\mathbf{x};\, \boldsymbol{\mu}_k,\, \boldsymbol{\Sigma}_k)
$$

può essere multimodale pur essendo fatta di soli pezzi unimodali: con $K = 2$ è
la densità a due gobbe che nasce da due sole campane, una per sacchetto. Le
gobbe però non sono garantite, e la soglia si calcola: due componenti di ugual
peso e ugual larghezza ne danno due soltanto se i centri distano più di due
deviazioni standard, e sotto quella soglia la densità torna a una gobba sola
pur restando una mistura.

Il caso continuo generalizza la stessa costruzione: se
$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ e
$p_\theta(\mathbf{x} \mid \mathbf{z}) = \mathcal{N}\big(\mathbf{x};\,
f_\theta(\mathbf{z}),\, \sigma^2 \mathbf{I}\big)$, dove $f_\theta$ è una rete
neurale e $\sigma^2$ la varianza del rumore che il decoder aggiunge (da non
confondere con la varianza della zona proposta dall’encoder, che comparirà
nella terza sezione), allora $p_\theta(\mathbf{x})$ è una **mistura infinita** di gaussiane
sferiche, i cui centri sono le uscite della rete e i cui pesi sono dati dal
prior. Una rete deterministica più due gaussiane elementari bastano quindi a
descrivere una distribuzione che non si saprebbe scrivere in nessun altro modo.

Con $f_\theta$ **lineare** il modello diventa la PCA probabilistica, la cui
soluzione a massima verosimiglianza individua il **sottospazio** generato dalle
prime $L$ componenti principali e non le singole direzioni (con $f_\theta$
lineare, cioè $f_\theta(\mathbf{z}) = \mathbf{W}\mathbf{z}$ e $\mathbf{z}$ di
dimensione $L$, la matrice $\mathbf{W}$ è determinata a meno di una rotazione),
e nel limite $\sigma^2 \to 0$ la codifica si riduce alla proiezione
ortogonale, cioè alla PCA della sezione su
riduzione e clustering. Cambiando l’ipotesi sul rumore, da una sola varianza
per tutte le componenti osservate a una varianza per ciascuna, si ottiene
l’**analisi fattoriale**, che è esattamente il modello di Spearman: un fattore
comune a tutte le prove più uno scarto proprio di ciascuna, e nel lavoro del
1904 quello scarto è già misurato prova per prova (per le materie classiche
sta al fattore comune come 1 sta a 99, per la matematica come 26 a 74). Stessa
struttura, con una moltiplicazione di matrici al posto della rete. Più tardi
sono venute la forma a più fattori e la sua stima a massima verosimiglianza,
non l’idea.

`````

```{figure} ../figures/due-gobbe-da-due-campane.svg
:name: fig-due-gobbe
:alt: "Due riquadri affiancati. A sinistra, sotto il titolo «i due sacchetti, uno per volta», due campane distinte sullo stesso asse dei millimetri: una centrata su 12 e intestata «le piccole», una centrata su 20 e intestata «le grandi». Una freccia porta al riquadro di destra, intestato «i due insieme, sacchetto non scritto», dove una curva sola, che è la somma delle due, ha due gobbe della stessa altezza, una su 12 e una su 20, con un avvallamento segnato a 16."
:width: 100%

Pezzi semplici, risultato complicato. A sinistra le due campane, una per
sacchetto, ciascuna con la sua unica gobba e ciascuna pesata metà, perché metà
delle pescate viene di lì; a destra la loro somma, cioè quello che si misura
quando il sacchetto non lo si guarda mai. La forma a due gobbe non l’ha
disegnata nessuno: è comparsa perché una parte della storia è rimasta nascosta.
(Le curve sono lisce: è la forma verso cui l’istogramma tende quando le pescate
sono tantissime.)
```

Guardando {numref}`fig-due-gobbe` si capisce anche perché conviene: chi volesse
descrivere la curva di destra senza sapere dei sacchetti dovrebbe inventarsi
una formula per una cosa a due gobbe, mentre a noi sono bastate due gobbe
semplici (nel disegno si chiamano **campane**, che è il loro nome consueto) e
la regola con cui si sceglie il sacchetto.

## Il prezzo: la somma che non si può fare

La mossa costa, e conviene dire subito quanto, perché è il problema che il
capitolo passa il tempo ad aggirare.

```{figure} ../figures/modello-latente-generare-e-inferire.svg
:name: fig-modello-latente
:alt: "Due riquadri collegati da due frecce. Nel riquadro in alto, intestato «quello che non si vede», una curva a campana con tre punti sorteggiati sotto di essa e la scritta «z: una nuvola semplice, scelta da noi». Nel riquadro in basso, intestato «quello che si vede», diciotto punti sparsi in modo irregolare e la scritta «x: i dati veri, sparsi come capita». Una freccia continua scende dal riquadro di sopra a quello di sotto, etichettata «generare, un passaggio della rete»; una freccia tratteggiata risale da quello di sotto a quello di sopra, etichettata «risalire, da quale punto sarà venuto?»."
:width: 88%

Le due direzioni della stessa freccia. Scendere è facile: si sorteggia un punto
là sopra e si applica la regola che porta da lui al dato. Risalire, cioè
chiedersi da quale punto di sopra possa essere venuto un dato che si ha in
mano, è il problema che il capitolo risolve, e costa in due modi diversi, che
il testo qui sotto separa. (Le due lettere del disegno sono i nomi con cui
questa materia le chiama da sempre: **z** la cosa nascosta, **x** il dato che
si vede.)
```

La freccia tratteggiata di {numref}`fig-modello-latente` costa in due modi
diversi, che è bene tenere separati perché il capitolo li affronta con due
strumenti distinti.

**Prima difficoltà: la somma.** Per sapere quanto è probabile un dato bisogna
considerare tutti i valori che la causa nascosta poteva prendere, e sommarli
pesandoli. Con due sacchetti sono due addendi. Ma la causa nascosta di cui
parleremo non è la scelta fra due scatole: è una fila di numeri (nel capitolo
ne useremo otto, che è quanto basta a comprimere una cifra scritta a mano), e
ciascuno può valere qualunque cosa. Gli addendi diventano allora infiniti, il
che di per sé non sarebbe un guaio, perché somme di infiniti addendi si fanno
da secoli, purché la cosa da sommare sia semplice.

Il guaio è un altro. A trasformare la causa nascosta nel dato ci pensa una
**rete neurale**, cioè la macchina dei capitoli precedenti: milioni di numeri
messi in fila che si moltiplicano e si sommano, e che nessuno saprebbe
riassumere in una formula. Con quella in mezzo, la somma **non si sa scrivere**
in nessun modo utile; e provare a tentoni, misurandola in tanti punti sparsi,
chiede un numero di punti che si moltiplica a ogni numero in più della causa
nascosta.

**Seconda difficoltà, ed è quella che sorprende: neanche tirare a sorte
funziona.** La via d’uscita ovvia sarebbe sorteggiare un po’ di valori del
latente, guardare quanto ciascuno spiega bene il dato, e fare la media. In
poche dimensioni si fa. In molte no, e la ragione è che quasi tutti i valori
sorteggiati spiegano il dato in modo pessimo: la media di mille numeri quasi
nulli e di un numero grande dipende tutta da quell’uno, che quasi mai capita di
pescare. La sezione centrale del capitolo lo misura invece di dirlo.

## La stessa idea, dentro quattro macchine

Questa idea il libro la mette al lavoro in **quattro** punti, e in nessuno dei
quattro la spiega fino in fondo. Due sono già passati: quando trasforma il
suono in simboli per poterlo scrivere come si scrive un testo, nel capitolo
sull’audio, e quando recinta le mosse che un programma può permettersi di
provare, in quello sul deep reinforcement learning. Due arriveranno: quando
comprime un’immagine per poterla generare su un computer di casa, e quando
spreme un fotogramma di videogioco in pochi numeri. Ogni volta il libro
rimanda la fattura, dicendo che lo vedrà più avanti o che gli basta il ruolo
dei due termini. L’ultima sezione la paga, e ripassa i quattro punti uno per
uno adesso che la macchina è nota.

Dove stia questa famiglia rispetto alle altre lo mette in fila, più avanti, il
capitolo sulla **verosimiglianza** esatta. «Verosimiglianza» è proprio quel
numero di poco fa, quanto il modello si aspettava di vedere il dato; e là i
modelli generativi del libro sono ordinati tutti insieme secondo una cosa sola,
che cosa ciascuno sa dirne. Quella mappa vive là, e qui non ne facciamo una
seconda. Basta l’essenziale: i modelli di questo capitolo quel numero lo sanno
dire **per difetto**, cioè restituiscono un valore che sta di sicuro sotto a
quello vero. Di che cosa sia fatto il divario si sa benissimo, e la terza
sezione lo dice; quanto valga, no. Perché ci si debba accontentare, e perché
accontentarsi convenga, è la storia della terza sezione.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- La mossa di questo capitolo è **spiegare quello che si vede con qualcosa che
  non si vede**: prima si sorteggia una causa nascosta, poi da quella si
  sorteggia il dato. La cosa nascosta si chiama **variabile latente**.
- Il guadagno è che **pezzi semplici danno un risultato complicato**: due
  sacchetti con una gobba ciascuno, se le misure tipiche dei due sono abbastanza
  lontane, producono un istogramma a due gobbe, e nessuno ha dovuto scrivere la
  forma a due gobbe.
- Il prezzo è che per sapere quanto è probabile un dato bisogna **considerare
  tutte le cause nascoste possibili**, e quando sono tante quel conto non si
  fa. Non si fa nemmeno tirando a sorte, perché quasi tutte le cause
  sorteggiate spiegano il dato malissimo.
- Il libro monta questa idea in **quattro** punti senza mai spiegarla fino in
  fondo, due già letti e due che verranno; l’ultima sezione dice quali. È il
  buco che questo capitolo riempie.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un **modello a variabile latente** definisce
  $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z})\,
  p(\mathbf{z})\, \mathrm{d}\mathbf{z}$: prior semplice, verosimiglianza
  condizionale semplice, marginale arbitrariamente complicata.
- Con $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ e
  $p_\theta(\mathbf{x} \mid \mathbf{z}) = \mathcal{N}(\mathbf{x};
  f_\theta(\mathbf{z}), \sigma^2 \mathbf{I})$ si ottiene una **mistura infinita
  di gaussiane** con centri $f_\theta(\mathbf{z})$. La mistura di gaussiane
  finita del {doc}`capitolo sul machine learning </MachineLearning/overview>` è lo stesso oggetto con $\mathbf{z}$
  discreto; con $f_\theta$ lineare si ottiene la PCA probabilistica, e
  l’analisi fattoriale è la variante con una varianza di rumore per ciascuna
  componente osservata.
- La marginale è **intrattabile**: nessuna forma chiusa, e la stima Monte Carlo
  dal prior ha varianza che esplode con la dimensione di $\mathbf{z}$, perché
  quasi tutti i campioni cadono dove
  $p_\theta(\mathbf{x} \mid \mathbf{z})$ è trascurabile.
- Da qui il programma del capitolo: rinunciare al valore esatto di
  $\log p_\theta(\mathbf{x})$ e ottimizzare un **limite inferiore**, che si
  paga con un secondo modello (l’encoder) e si guadagna in trattabilità.
```

`````

## Comprimere, ricostruire, usare

Tre sezioni, e ciascuna toglie un pezzo al problema. La prima parte dalla
strada più corta, l’**autoencoder**, cioè una rete che impara a comprimere e a
ricostruire senza che nessuno le parli di probabilità: funziona benissimo per
comprimere e fallisce per generare, e il perché di quel fallimento è il modo
migliore per capire che cosa manchi. La seconda è il cuore: la
**verosimiglianza intrattabile**, il limite inferiore che la sostituisce
(l’**ELBO**), i suoi due termini letti come ricostruzione e costo di
descrizione, e un trucco senza il quale la macchina non si potrebbe addestrare
in un tempo ragionevole, perché in mezzo c’è un sorteggio e le correzioni, da
sole, un sorteggio non lo attraversano. La terza guarda che cosa si fa con quel riassunto
nascosto una volta che c’è: la manopola con cui gli si può chiedere di tenere
separate le cose di cui il dato è fatto (la luce, l’inclinazione, il soggetto),
il riassunto fatto di simboli invece che di numeri, e i quattro punti del libro
in cui questa macchina è al lavoro.

Il capitolo che segue prende lo stesso problema, fabbricare dati nuovi e
plausibili, e lo attacca dal lato opposto: butta via la probabilità e mette un
giudice. Conviene arrivarci sapendo che cosa si sta buttando via.
