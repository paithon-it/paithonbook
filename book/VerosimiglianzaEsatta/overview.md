# Modelli a verosimiglianza esatta

Immagina di avere fra le mani il generatore di volti del capitolo sulle GAN,
quello che sforna a ripetizione persone che non esistono, e di fargli una
domanda diversa da tutte quelle che gli abbiamo fatto finora. Non «fammi un
volto», ma: prendi *questa* fotografia, guardala, e dimmi quanto è probabile.

Non è che risponde male. Non ha uno sportello a cui rivolgere quella domanda.
Il falsario ha imparato a **produrre**, non a **valutare**: il numero che
misura quanto un dato è plausibile non compare da nessuna parte nel suo
addestramento, e non c'è modo di estrarlo dai suoi pesi. Il capitolo sui
modelli di diffusione, appena chiuso, sta un gradino più in là: lì quel numero
esiste, ma quello che il modello ottimizza non è quel numero: è una stima
prudente che gli sta sotto, e per avere il valore vero bisogna fare un secondo
lavoro, lungo e a parte.

Questo capitolo racconta la terza risposta: la famiglia di modelli che quel
numero lo restituisce **esatto**, con un solo passaggio della rete, perché è
costruita apposta. La cosa vale la pena non per pignoleria, ma perché quel
numero è la stessa cosa di tre mestieri diversi (comprimere, riconoscere ciò
che è fuori posto, confrontare due modelli senza chiamare un giudice), ed è la
ragione per cui questa famiglia, che nella corsa alle immagini ha perso, non è
affatto uscita di scena.

## La parola, prima della mappa

**Verosimiglianza** l'abbiamo già incontrata nei richiami di matematica, alla
sezione sulla probabilità, dove serviva a scegliere i parametri: quanto è
probabile ciò che ho visto, se il modello fosse questo. Qui la guardiamo
dall'altro capo, a modello ormai fissato: è il valore che quel modello assegna
a un dato, letto come «quanto mi aspettavo di vedere una cosa così». Alta se il
dato è di quelli su cui il modello avrebbe scommesso, bassa se lo coglie di
sorpresa. Il nome è scomodo e il concetto no: è un voto, e a differenza
dell'energia del capitolo che segue è un voto **normalizzato**, cioè sommato su
tutti i dati possibili fa esattamente uno.

Quel «fa esattamente uno» è tutto il problema, ed è il filo che tiene insieme
questo capitolo e il prossimo. Sommare su tutti i dati possibili non si può, e
il capitolo sui modelli a energia mostrerà quanto quel conto sia fuori portata.
Le strade sono allora due: rinunciare alla normalizzazione e cavarsela lo
stesso (è il prossimo capitolo), oppure **costruire il modello in modo che
venga normalizzato da sé**, senza mai fare quel conto. È la strada di questo.

## La mappa

Vale la pena disporre in ordine le famiglie che il libro ha incontrato, perché
è la prima volta che le mettiamo tutte insieme, e l'asse su cui le ordiniamo è
uno solo: **che rapporto ha il modello con la probabilità del dato**. È un
taglio fra i tanti possibili, e va detto: altrove nel libro le stesse cose sono
ordinate secondo altri assi (le quattro famiglie dell'auto-supervisione, per
dire, si ordinano secondo che cosa impedisce la risposta vuota, che è tutta
un'altra domanda). Qui contano solo tre risposte.

**Non ce l'ha affatto.** Il modello sa produrre campioni e nient'altro; la
probabilità non compare in nessuna delle sue formule. È il caso delle **GAN**:
si dice che definiscono una **densità implicita**, dove densità è il nome
tecnico di quel voto e implicita vuol dire esattamente «c'è ma non si può
guardare». Se ne può stimare qualche proprietà generando tanti campioni e
misurandoli, che è quello che fa il FID, ma la probabilità di un singolo dato
no.

**Ce l'ha approssimata.** Il modello ha di che parlare di probabilità, ma quel
che ottimizza e quel che sa dire è un surrogato. I **VAE**, gli autoencoder
variazionali del capitolo precedente, danno un limite inferiore, l'ELBO: si sa
che il valore vero sta più in alto, non di quanto. I **modelli a energia**
danno il voto a meno di una costante che nessuno conosce: bastano per dire
quale di due dati è più plausibile, non per stampare una percentuale. I
**modelli di diffusione** addestrano su un limite come i VAE; nella loro
formulazione continua il valore esatto si può ottenere {cite}`song2021score`,
ma passando per la soluzione di un'equazione differenziale e per una stima
fatta a campione di un pezzo per cui una formula chiusa non esiste, cioè con un
lavoro che nessuno fa a ogni immagine.

**Ce l'ha esatta.** Il modello restituisce $\log p(\mathbf{x})$, giusto, in un
passaggio. Due strade portano lì, e sono le due sezioni di questo capitolo. La
prima spezza il dato in pezzi e li mette in fila, moltiplicando le probabilità
uno dopo l'altro: sono i **modelli autoregressivi**, che il libro conosce da
tempo sul testo e sull'audio, e che qui incontriamo sulle immagini. La seconda
non spezza niente e deforma: costruisce una trasformazione **invertibile** dai
dati a una gaussiana, e legge la probabilità di là. Sono i **flussi
normalizzanti**, ed è la parola «flusso» che il capitolo precedente ha usato
per il *rectified flow* senza mai dire da dove venisse.

## Il prezzo

Nessuna delle due strade è gratis, e il prezzo si paga nella stessa moneta: la
libertà dell'architettura.

Un modello autoregressivo, per essere onesto, deve garantire che ogni pezzo
veda soltanto quelli che lo precedono. Fin qui nessun problema, il testo un
ordine ce l'ha per natura. Su un'immagine bisogna inventarselo, e poi bisogna
imporre quel divieto dentro una rete che di suo guarda in tutte le direzioni.
Il costo vero però arriva quando si genera: un pezzo alla volta, e i pezzi di
un'immagine sono decine di migliaia.

Un flusso paga altrove. Perché la trasformazione si possa invertire, ogni
strato dev'essere invertibile, e questo esclude quasi tutto quello che il libro
ha usato finora (non si può schiacciare, non si può buttare via niente, non si
può nemmeno cambiare il numero di coordinate). Un flusso non comprime: entra
con un milione di numeri ed esce con un milione di numeri. Ed è, come vedremo,
la ragione strutturale per cui ha perso.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- C'è una domanda che al generatore di volti delle GAN non si può proprio
  fare: «questa fotografia, quanto è probabile?». Non risponde male, non ha
  proprio lo sportello. Ha imparato a fabbricare, non a giudicare.
- I modelli si mettono in fila secondo che cosa sanno dire di quel numero.
  Alcuni **niente** (le GAN). Alcuni **qualcosa di approssimato**: i modelli
  del capitolo sulla diffusione e quelli del capitolo che segue sanno dire chi
  è più plausibile fra due dati, non stampare una percentuale. E alcuni lo
  sanno **esatto**, ed è la famiglia di questo capitolo.
- Le strade per saperlo esatto sono due. **A pezzi in fila**: si taglia il
  dato in pezzetti, si mette in fila e si moltiplicano le probabilità, come si
  fa da sempre con il testo. **Per deformazione**: si costruisce una macchina
  che si può usare nei due sensi e che porta i dati su una nuvola semplice, e
  la probabilità si legge di là.
- Tutte e due si pagano in libertà di progetto. La prima costringe a generare
  un pezzetto alla volta, e i pezzetti di un'immagine sono decine di migliaia.
  La seconda vieta di buttare via qualunque cosa, quindi vieta di comprimere.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Asse di classificazione: **il rapporto del modello con $p(\mathbf{x})$**.
  Densità **implicita** (GAN: campionamento sì, valutazione no); densità
  **esplicita approssimata** (VAE: ELBO, cioè limite inferiore; EBM:
  $-E_\theta(\mathbf{x})$ a meno di $\log Z$ ignoto; diffusione: bound
  variazionale, con il valore esatto ottenibile via probability-flow ODE
  {cite}`song2021score` a costo non trascurabile); densità **esplicita
  trattabile** (autoregressivi e flussi).
- **Autoregressivi**: $\log p(\mathbf{x}) = \sum_i \log p(x_i \mid
  \mathbf{x}_{<i})$, ogni fattore una softmax normalizzata. Valutazione in un
  passaggio (*teacher forcing*), campionamento in $D$ passaggi sequenziali.
- **Flussi**: $\log p(\mathbf{x}) = \log p_Z(f(\mathbf{x})) + \log \lvert \det
  \partial f / \partial \mathbf{x} \rvert$, con $f$ invertibile. Valutazione e
  campionamento entrambi in un passaggio; in cambio $f$ è vincolata a essere
  un diffeomorfismo, quindi a **conservare la dimensione**.
- Nessuna delle due paga in fedeltà del modello di probabilità: pagano in
  vincoli architetturali. È il baratto che il capitolo mette a fuoco, e la
  ragione per cui la famiglia ha perso la corsa alle immagini senza perdere
  quella all'utilità.
```

`````

## Come è organizzato il capitolo

Due meccanismi e un bilancio. Prima gli **autoregressivi sulle immagini**: come
si impone un ordine a una griglia di pixel, come si costringe una convoluzione
a guardare solo indietro, e il difetto famoso che quella costrizione porta con
sé. Poi i **flussi normalizzanti**: il cambio di variabile che è tutta la
matematica del capitolo, il vincolo di invertibilità e il modo furbo di
soddisfarlo, e perché il determinante, che in generale costa un'eternità, qui
si legge in un colpo d'occhio. Infine il bilancio: a che cosa serve davvero
sapere quel numero, un fallimento istruttivo che ha smontato una delle sue
applicazioni più ovvie, e il ponte verso il *flow matching* che il capitolo
precedente ha già usato.
