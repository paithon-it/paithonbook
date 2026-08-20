# Efficienza: il modello che si addestra non è quello che si usa

Molti insetti vivono due vite in un corpo solo. La larva è fatta per mangiare:
lenta, molle, con un apparato digerente che occupa quasi tutto lo spazio
disponibile. L’adulto è fatto per volare e riprodursi, ed è un animale
completamente diverso. Nessuno dei due sarebbe capace di fare il mestiere
dell’altro.

Con questa immagine si apre uno dei lavori che questo capitolo racconta, e la
seconda metà della frase è la tesi di tutto il capitolo:

> Many insects have a larval form that is optimized for extracting energy and
> nutrients from the environment and a completely different adult form that is
> optimized for the very different requirements of traveling and reproduction.
> In large-scale machine learning, we typically use very similar models for the
> training stage and the deployment stage despite their very different
> requirements: […]
>
> Hinton, Vinyals e Dean, *Distilling the Knowledge in a Neural Network*
> (2015) {cite}`hinton2015distilling`

Cioè: nell’apprendimento automatico su larga scala, per addestrare e per
mettere in funzione usiamo di solito modelli molto simili, benché le due cose
chiedano cose molto diverse. E le chiedono davvero. Addestrare vuol dire
macinare per settimane una montagna di dati, e chi lo fa può permettersi
qualunque lentezza: nessuno sta aspettando. Rispondere vuol dire stare dentro
la memoria di una macchina e restituire qualcosa mentre una persona guarda lo
schermo. Larva e farfalla, e quasi sempre si spedisce la larva.

Questo capitolo è su come si costruisce la farfalla.

## Il problema, in due numeri

Prima di parlare di rimedi conviene guardare la malattia, e per una volta si
guarda con un conto che si fa a mente.

Un modello ha dei **parametri**, e un parametro è semplicemente un numero che
la rete ha imparato e che va tenuto da qualche parte. Si chiamano anche **pesi**, perché ciascuno dice quanto conta un collegamento
fra due neuroni: due parole, una cosa sola, e da qui in avanti valgono l'una
per l'altra.

Ogni numero costa spazio, e quanto costa lo dicono i **bit**, le cifre binarie
con cui un calcolatore scrive tutto. Otto bit fanno un **byte**, quindi un
numero da trentadue bit occupa quattro byte. Il resto è moltiplicazione.

```python
PARAMETRI = 7_000_000_000     # sette miliardi di parametri
SCHEDA_GB = 16                # la scheda grafica ha sedici gigabyte di memoria

print(f"{'formato':<9} {'bit':>4} {'peso':>10}   ci sta nella scheda?")
for nome, bit in (("float32", 32), ("float16", 16), ("int8", 8), ("int4", 4)):
    gb = PARAMETRI * bit / 8 / 1e9
    print(f"{nome:<9} {bit:>4} {gb:>7.1f} GB   {'sì' if gb < SCHEDA_GB else 'no'}")
```

```text
formato    bit       peso   ci sta nella scheda?
float32     32    28.0 GB   no
float16     16    14.0 GB   sì
int8         8     7.0 GB   sì
int4         4     3.5 GB   sì
```

Le due righe agli estremi sono lo **stesso modello**, con gli stessi parametri,
addestrato una volta sola. Nella prima non ci sta; nell’ultima ci starebbe
quattro volte e mezzo. (La colonna di destra conta i soli **pesi**: per far
girare davvero il modello serve dell’altro spazio, e la riga a sedici bit, che
lascia due gigabyte scarsi di margine, nella pratica è più stretta di quanto
sembri.) Fra le due non c’è nessun addestramento in più: c’è solo un modo
diverso di scrivere gli stessi numeri.

Il resto del capitolo è la spiegazione di perché questo non sia un imbroglio, e
di che cosa si paga in cambio, perché qualcosa si paga sempre.

## Le tre leve

Un modello troppo grande si può stringere in tre modi, e sono tre modi
davvero diversi: non è la stessa idea vista da tre angoli, sono tre operazioni
che agiscono su cose diverse e si possono usare tutte e tre insieme.

**Meno bit per parametro.** I parametri restano tutti, e resta la forma della
rete: cambia solo quante cifre si tengono di ciascun numero. È la leva del
conto qui sopra, si chiama **quantizzazione**, ed è quella che rende di più
per quanto costa.

**Meno parametri.** La rete resta grande com’è, ma una parte dei suoi
collegamenti viene messa a zero e non serve più tenerla né moltiplicarla. Si
chiama **potatura**, ed è la leva che promette di più e mantiene meno, per una
ragione che riguarda il modo in cui i calcolatori fanno i conti.

**Un modello più piccolo che impara dal grande.** Qui non si stringe niente: si
costruisce un secondo modello, piccolo dall’inizio, e gli si insegna a
comportarsi come il primo. Si chiama **distillazione**, ed è l’unica delle tre
in cui il modello finale è un oggetto nuovo.

Le tre si vedono meglio disegnate su una **matrice** di pesi
({numref}`fig-tre-leve`). Matrice è solo il nome che si dà a una griglia di
numeri, righe e colonne come una tabella: uno strato di rete neurale **è** una
di quelle griglie, e far lavorare lo strato vuol dire moltiplicare la griglia
per i numeri che le arrivano. Sulla stessa griglia si vede subito che le tre
leve agiscono su cose diverse: una sulle sfumature, una sulle caselle, una
sulla forma della griglia.

```{figure} ../figures/tre-leve.svg
:name: fig-tre-leve
:alt: "Quattro griglie affiancate che rappresentano la stessa matrice di pesi. La prima, «com’è», è una griglia otto per otto in cui le caselle hanno decine di sfumature diverse. La seconda, «meno bit», ha la stessa griglia otto per otto ma le sfumature sono soltanto 4, ripetute. La terza, «meno pesi», ha la griglia otto per otto con 33 caselle vuote e tratteggiate e 31 piene. La quarta, «più piccolo», è una griglia quattro per quattro, con le sue sfumature tutte diverse. Sotto ciascuna, quanti pesi restano."
:width: 100%

Il primo riquadro è la matrice com’è; gli altri tre sono le tre leve, sulla
stessa matrice. Togliere bit non cambia quante caselle ci sono, cambia quante
sfumature diverse una casella può avere: qui il secondo riquadro ne ammette
quattro, che sarebbero due bit per casella. Togliere pesi lascia le sfumature e
svuota le caselle. Fare un modello più piccolo cambia la griglia.
```

C’è poi una quarta cosa che si fa per andare più veloci, ed è di natura
diversa: **non tocca il modello**. Riguarda come lo si fa lavorare mentre
risponde, ed è materia di due capitoli che vengono più avanti, quello sui
Transformer e quello su MLOps, che la costruiscono ciascuno per una ragione
sua. L’ultima parte di questo capitolo dice quali sono, e perché stiano lì e
non qui.

## Tre piani, tre mestieri

Il libro parla di efficienza in tre punti, e conviene distinguerli subito,
perché è facile andarli a cercare nel posto sbagliato.

Il capitolo sulla **GPU** spiega l’hardware: com’è fatta la memoria di una
scheda, perché i byte che viaggiano contano più dei conti che si fanno, come si
scrive un calcolo che la sfrutti. È il piano di sotto.

Il capitolo su **MLOps** spiega il servizio: come si mette un modello dietro a
un indirizzo a cui altri programmi possano rivolgersi, che cosa si promette a
chi lo usa, come si misura se sta rispettando la promessa. È il piano di
sopra.

Questo capitolo sta in mezzo, e spiega il **meccanismo**: perché quattro bit
bastino, che cosa si rompe quando non bastano, che cosa perde davvero uno
studente che imita il maestro. Non dice come si mette in produzione: dice
perché la cosa che si mette in produzione funziona.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Addestrare e rispondere sono **due mestieri diversi**: il primo può essere
  lento quanto vuole, il secondo deve stare in una macchina e rispondere subito.
  Quasi sempre però si mette in produzione lo stesso identico modello che si è
  addestrato.
- Il conto che spiega tutto: un modello da sette miliardi di parametri pesa 28
  GB se ogni parametro si scrive con trentadue bit, e 3,5 GB se se ne usano
  quattro. È lo **stesso modello**, scritto in un altro modo.
- Le tre leve per stringerlo sono davvero tre cose diverse: **meno bit** per
  parametro (la quantizzazione), **meno parametri** (la potatura), oppure un
  **modello nuovo e più piccolo** che impara dal grande (la distillazione).
- Qualcosa si paga sempre, e la parte utile di questo capitolo è quella: non
  che le tre leve esistano, ma che cosa costano.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- I **pesi** di un modello occupano $P \cdot b / 8$ byte, con $P$ il numero di
  parametri e $b$ i bit per parametro: passare da $b = 32$ a $b = 4$ è un
  fattore otto a parità di $P$. Non è tutta la memoria che serve, e il capitolo
  lo aggiunge man mano: la quantizzazione a gruppi si porta dietro le sue scale
  (il sei per cento in più a quattro bit con gruppi da sessantaquattro), e in
  servizio ci sono le attivazioni e la cache che cresce con la conversazione.
- Le tre leve agiscono su fattori diversi dello stesso prodotto: la
  **quantizzazione** su $b$, la **potatura** e la **distillazione** su $P$. Le
  ultime due lo fanno però in modi opposti: la potatura svuota una matrice che
  resta della sua forma, la distillazione cambia la forma.
- Sono componibili e si compongono davvero: un modello distillato si quantizza,
  e un modello quantizzato si pota. Quello che non è componibile è il
  **budget di errore**: ogni leva ne consuma un pezzo, e le perdite non si
  sommano in modo prevedibile.
- Il tempo di risposta è governato da un’altra grandezza, la **banda di
  memoria**, e non dalla dimensione in sé: è il modello roofline del capitolo
  sulla GPU, e l’ultima parte di questo capitolo ci torna sopra.
```

`````

La prima leva è quella che rende di più e si spiega peggio, perché la domanda
che si porta dietro è imbarazzante: come fa un modello a funzionare uguale se
gli si buttano via ventotto bit su trentadue? La sezione che segue risponde, e
la risposta comincia da una cosa che facciamo tutti al supermercato.
