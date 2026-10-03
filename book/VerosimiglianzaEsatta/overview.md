# Modelli a verosimiglianza esatta

```{image} ../figures/aperture/verosimiglianza-esatta.png
:class: pt-apertura only-light
:width: 100%
:alt: Una bilancia a due piatti pesa un piccolo quadro contro una pila di pesi.
```

```{image} ../figures/aperture/verosimiglianza-esatta-scura.png
:class: pt-apertura only-dark
:width: 100%
:alt: Una bilancia a due piatti pesa un piccolo quadro contro una pila di pesi.
```

Il generatore di volti del {doc}`capitolo sulle GAN </GAN/overview>` sforna a
ripetizione persone che non esistono. Facciamogli una domanda diversa da tutte
quelle che gli abbiamo fatto finora. Non «fammi un volto», ma: prendi *questa*
fotografia, guardala, e dimmi quanto è probabile che una fotografia così venga
fuori dal mondo che hai imparato.

Il generatore non ha uno sportello a cui rivolgere quella domanda. Il falsario
ha imparato a produrre, non a valutare: il numero che misura quanto un dato è
plausibile non compare da nessuna parte nel suo addestramento, e non c'è modo
di estrarlo dai suoi pesi. Il {doc}`capitolo sui modelli di diffusione
</ModelliDiffusione/overview>`, appena chiuso, sta un gradino più in là: lì quel
numero esiste, ma non è quello che il modello ottimizza. In teoria ottimizza
un limite inferiore, una stima prudente che sta sotto il valore vero; in
pratica nemmeno quello, ma una sua versione ripesata, che dà campioni migliori
e non è più un limite.

La terza risposta è una famiglia di modelli costruiti in modo che quel numero
esca esatto, e che per valutarlo basti un solo passaggio della rete. Il numero
serve a tre mestieri: comprimere i dati senza perdere niente, scegliere fra
ipotesi alternative la più probabile (fra due trascrizioni di una frase
registrata, la più plausibile), e accorgersi che un dato non somiglia a quelli
su cui il modello è stato addestrato. Sono i mestieri per cui la famiglia è
rimasta in servizio anche negli anni in cui, sulla qualità delle immagini
generate, la diffusione la superava nettamente.

## La parola, prima della mappa

Verosimiglianza è una parola dei {doc}`richiami di probabilità e statistica
</Matematica/probabilita-statistica>`, dove serviva a scegliere i parametri:
fissati i dati, quanto è probabile ciò che si è visto se il modello fosse
questo, letto come funzione dei parametri $\theta$. Qui la si guarda dall'altro
capo. Fissato il modello $p_\theta$, la verosimiglianza di un dato $\mathbf{x}$
è il numero $p_\theta(\mathbf{x})$ che il modello gli assegna, letto come
«quanto mi aspettavo di vedere una cosa così»: alto se il dato è di quelli su
cui il modello avrebbe scommesso, basso se lo coglie di sorpresa. Il nome è
scomodo e il concetto no: è un voto, e a differenza dell'energia del
{doc}`capitolo sui modelli a energia </ModelliEnergia/overview>` è
**normalizzato**, cioè sommato su tutti i dati possibili fa esattamente uno.
Quando il dato è fatto di numeri continui la somma diventa un integrale, e
$p_\theta$ prende il nome di densità: un'altezza di curva più che una
probabilità, che può anche superare uno, mentre a fare uno è l'area sotto la
curva.

Quel «fa esattamente uno» è tutto il problema, ed è il filo che lega i modelli
a verosimiglianza esatta ai modelli a energia. Normalizzare un punteggio vuol
dire dividerlo per la somma dei punteggi di tutti i dati possibili, e quella
somma non si può fare: il capitolo sui modelli a energia mostrerà quanto il
conto sia fuori portata. Le strade sono allora due: rinunciare alla
normalizzazione e cavarsela lo stesso (è il capitolo sui modelli a energia),
oppure costruire il modello in modo che venga normalizzato da sé, senza mai
fare quel conto. È la strada di questo.

## La mappa

I capitoli generativi del libro sono cinque, e per la prima volta li mettiamo
tutti insieme: {doc}`modelli latenti </ModelliLatenti/overview>`,
le GAN, la diffusione,
questo e i modelli a energia. L'asse su cui li
ordiniamo è uno solo: che rapporto ha il modello con la probabilità del
dato. È un taglio fra i tanti possibili, e altrove nel libro le stesse cose
sono ordinate secondo altri assi (le quattro famiglie dell'auto-supervisione si
ordinano secondo che cosa impedisce la risposta vuota, che è tutta un'altra
domanda). Qui contano solo tre risposte.

La prima risposta è che il modello quel numero non ce l'ha affatto. Sa produrre
campioni e nient'altro, e la probabilità non compare in nessuna delle sue
formule. È il caso delle GAN, di cui si dice che definiscono una **densità
implicita**: il modello è una procedura per estrarre campioni, e la densità
$p(\mathbf{x})$ non è scritta da nessuna parte. Può anche non esistere affatto.
Se il generatore parte da un latente con meno numeri di quanti ne abbia
un'immagine, le immagini che produce stanno tutte su una varietà di dimensione
più bassa, una superficie senza spessore dentro lo spazio delle immagini, e su
una superficie senza spessore una densità, che è probabilità per unità di
volume, non si può definire (la {doc}`sezione sull'addestramento avversario
</GAN/come-funziona>` incontra lo stesso fatto quando il duello si inceppa). Del
modello si possono stimare alcune proprietà generando tanti campioni e
misurandoli, come fa il FID, che confronta le statistiche di un mucchio di
immagini generate con quelle di un mucchio di immagini vere; la probabilità di
un singolo dato no.

La seconda risposta è che il modello ce l'ha approssimata. Ha di che parlare
di probabilità, ma quel che ottimizza e quel che sa dire è un surrogato. I VAE,
gli autoencoder variazionali del capitolo sui modelli latenti, danno un limite
inferiore, l’{doc}`ELBO </ModelliLatenti/il-salto-probabilistico>` (il limite
inferiore sull'evidenza, cioè sulla verosimiglianza): si sa che il valore vero
sta più in alto, non di quanto. I modelli a energia danno il voto a meno di una
costante che nessuno conosce: bastano per dire quale di due dati è più
plausibile, non per stampare una percentuale. I modelli di diffusione
addestrano su una versione ripesata dello stesso genere di limite; nella loro
formulazione continua il valore esatto si può ottenere {cite}`song2021score`,
ma passando per la soluzione di un'equazione differenziale e per una stima
fatta a campione di un termine che calcolare per intero costerebbe troppo, cioè
con un lavoro che nessuno fa a ogni immagine.

La terza risposta è che il modello ce l'ha esatta. Restituisce
$\log p(\mathbf{x})$, il logaritmo della probabilità del dato, in un passaggio
solo. Si lavora sul logaritmo perché la probabilità di un'immagine intera è un
numero con migliaia di zeri dopo la virgola, e il logaritmo lo rende
maneggevole e trasforma i prodotti di probabilità in somme. Due strade portano
lì, e sono le prime due sezioni. La prima scrive la probabilità del dato come
prodotto di probabilità condizionate, un valore alla volta, ciascuno sapendo
quelli che lo precedono: sono i modelli autoregressivi, che il libro conosce sul
testo e sull'audio e che qui incontriamo sulle immagini. La seconda costruisce
una trasformazione invertibile che porta i dati su una distribuzione di base
nota, di solito una gaussiana, e ricava la probabilità con un cambio di
variabile: sono i flussi normalizzanti, e la parola «flusso» è la stessa del
*flusso rettificato* con cui il capitolo precedente genera immagini.

## Il prezzo

Nessuna delle due strade è gratis: in tutte e due la probabilità esatta si
paga con un vincolo sulla forma del modello, e per gli autoregressivi anche
con il tempo di generazione.

Un modello autoregressivo deve garantire che ogni valore dipenda soltanto da
quelli che lo precedono. Il testo un ordine ce l'ha per natura; un'immagine no,
quindi lo si fissa, e poi il divieto va imposto dentro una rete convoluzionale,
che per costruzione guarda in tutte le direzioni. Il costo maggiore arriva
quando si genera, perché si procede un valore alla volta: una fotografia a
colori di $256 \times 256$ pixel ha quasi duecentomila valori, e chiede
altrettanti passaggi della rete, uno dopo l'altro.

Un flusso paga altrove. Perché la trasformazione si possa invertire ogni strato
dev'essere invertibile, e questo esclude quasi tutto quello che il libro ha
usato finora: gli strati che riducono la dimensione, come il *pooling* e le
proiezioni su meno coordinate, e le attivazioni che mandano punti diversi nello
stesso valore, come la ReLU, che manda a zero tutti i negativi. Un flusso
conserva il numero di coordinate: entra con un milione di numeri ed esce con un
milione di numeri. Per lavorare su qualcosa di più piccolo deve farsi mettere
davanti un compressore, come succede nella diffusione latente, e la
probabilità esatta vale allora per la versione compressa, non per l'immagine.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- C'è una domanda che al generatore di volti delle GAN non si può proprio
  fare: «questa fotografia, quanto è probabile?». Non ha lo sportello a cui
  rivolgerla: ha imparato a fabbricare, non a giudicare.
- I modelli si mettono in fila secondo che cosa sanno dire di quel numero.
  Alcuni niente (le GAN). Alcuni un numero prudente: i modelli
  latenti e quelli a diffusione danno un valore che sta di sicuro sotto a
  quello vero, senza dire di quanto; nel caso della diffusione il valore vero
  si può anche ottenere, ma con un secondo lavoro lungo e a parte. Alcuni
  soltanto un confronto: i modelli a energia sanno dire
  quale di due dati è più plausibile, non stampare una percentuale, perché al
  loro voto manca una costante che nessuno conosce. E alcuni lo sanno
  esatto, ed è la famiglia di questo capitolo.
- Le strade per saperlo esatto sono due. A pezzi in fila: si taglia il
  dato in pezzetti, si mette in fila e si moltiplicano le probabilità, come si
  fa da sempre con il testo. Per deformazione: si costruisce una macchina
  che si può usare nei due sensi e che porta i dati su una nuvola semplice, la
  probabilità si legge di là, e si tiene conto di quanto la macchina ha stirato
  o schiacciato proprio in quel punto.
- Tutte e due si pagano in libertà di progetto. La prima costringe a generare
  un pezzetto alla volta, e su una fotografia a colori i pezzetti sono quasi
  duecentomila. La seconda vieta di buttare via qualunque cosa, quindi vieta di
  comprimere.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Asse di classificazione: il rapporto del modello con $p(\mathbf{x})$.
  Densità implicita (GAN: campionamento sì, valutazione no; con un latente
  di dimensione minore del dato la distribuzione sta su una varietà e una
  densità rispetto al volume non esiste); densità
  esplicita approssimata (VAE: ELBO, cioè limite inferiore; EBM:
  $-E_\theta(\mathbf{x})$ a meno di una costante di normalizzazione ignota;
  diffusione: bound
  variazionale, ottimizzato in versione ripesata, con il valore esatto
  ottenibile via probability-flow ODE
  {cite}`song2021score` a costo non trascurabile); densità esplicita
  trattabile (autoregressivi e flussi).
- Autoregressivi: $\log p(\mathbf{x}) = \sum_i \log p(x_i \mid
  \mathbf{x}_{<i})$, ogni fattore una distribuzione normalizzata (una softmax
  su 256 livelli, o una miscela di logistiche discretizzate). Valutazione in un
  passaggio (*teacher forcing*), campionamento in $D$ passaggi sequenziali,
  con $D$ il numero di valori che compongono il dato.
- Flussi:
  $\log p(\mathbf{x}) = \log p_Z(f(\mathbf{x})) + \log \lvert \det \partial f / \partial \mathbf{x} \rvert$,
  con $f$ invertibile. Con gli strati di accoppiamento valutazione e
  campionamento costano un passaggio ciascuno; nei flussi autoregressivi (MAF,
  IAF) uno dei due ne costa $D$. In ogni caso $f$ è vincolata a essere un
  diffeomorfismo, quindi a conservare la dimensione.
- La verosimiglianza esatta si paga in vincoli: sull'architettura (maschere
  di causalità, invertibilità con un determinante trattabile) e, per gli
  autoregressivi, nel costo del campionamento. Sulla qualità dei campioni di
  immagini la famiglia è rimasta a lungo dietro la diffusione, e dal 2025 i
  flussi autoregressivi a Transformer {cite}`zhai2025tarflow` ne hanno chiuso
  buona parte del divario; la sua utilità per la compressione e la valutazione
  non ne è mai dipesa.
```

`````

## Due meccanismi, e un bilancio

Prima gli autoregressivi sulle immagini: come si fissa un ordine su una
griglia di pixel, come si impone a una convoluzione di guardare solo i pixel
già letti, e il difetto, il punto cieco, che quel vincolo porta con sé. Poi i
flussi normalizzanti: il cambio di variabile per le densità, che è tutta la
matematica del capitolo, il vincolo di invertibilità, e il modo di soddisfarlo
con un determinante che in generale costa il cubo del numero di coordinate e
che negli strati di accoppiamento si riduce a una somma. Infine il bilancio: a
che cosa serve il numero (comprimere, scegliere fra ipotesi, riconoscere un
dato anomalo), il fallimento istruttivo che ha smontato la più ovvia di queste
applicazioni, e il ponte verso il *flow matching* che il capitolo precedente ha
già usato.
