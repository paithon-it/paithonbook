# Un vocabolario solo: fusione tardiva e fusione precoce

Chiedi a un sistema come quello della sezione precedente di *disegnare* un
gatto nero che salta su un muro, e ti risponderà con delle parole. Magari con
delle ottime parole: la descrizione di un gatto nero che salta su un muro,
scritta bene. Non è pigrizia e non è un rifiuto: è la forma dell'ultimo strato.
Qualunque cosa quel modello abbia capito guardando, per uscire deve passare da
una softmax su un vocabolario, e in quel vocabolario ci sono soltanto parole.

Vale la pena guardarla in faccia, questa asimmetria, perché non è un dettaglio
di implementazione. Un modello costruito innestando un encoder visivo su un
modello di linguaggio, come quelli della sezione precedente
{cite}`liu2023visual`, calcola

$$
p_\theta(y_t \mid y_{<t},\, E(I)),
\qquad y_t \in V_{\text{testo}},
$$

dove $I$ è l'immagine, $E$ l'encoder visivo con il suo connettore, $y_t$ il
token prodotto al passo $t$ e $V_{\text{testo}}$ il vocabolario di uscita.
L'immagine sta a destra della barra verticale: è una **condizione**, non un
valore che $y_t$ possa assumere. Il sistema ha un occhio in ingresso e una
bocca in uscita, e nessuna mano.

Da qui la domanda ingenua che è anche quella giusta: e se anche l'immagine
avesse i suoi token? Se in $V$ ci fossero, accanto alle parole, dei simboli che
stanno per pezzi di immagine, la stessa softmax potrebbe emetterli, e produrre
un disegno diventerebbe *lo stesso gesto* che produrre una frase. Il resto
della sezione è il prezzo di questa idea.

## Un alfabeto anche per i pixel

L'attrezzo per costruire quei simboli non è nuovo in questo libro: lo abbiamo
già montato per il suono, nella sezione sui **codec neurali** del capitolo
sull'audio. Là il problema era identico nella forma: un'onda è una grandezza
continua, un modello di linguaggio mangia simboli discreti, e un alfabeto per
il suono in natura non esiste, va costruito. La risposta era la **vector
quantization** del VQ-VAE {cite}`oord2017neural`: un elenco finito di
vettori-prototipo (il *codebook*), ogni vettore latente sostituito dal
prototipo più vicino, e di lui si conserva solo il numero di posizione. Quel
numero è il token.

Sui pixel si fa esattamente la stessa cosa, cambiando solo che cosa entra
nell'encoder. Il gesto è quello: comprimere, arrotondare al prototipo più
vicino, tenere l'indice.

```{figure} ../figures/vq-vae-rappresentazioni-discrete.svg
:name: fig-vq-vae
:alt: "L'immagine entra in un encoder che produce una griglia di vettori continui. Accanto, il codebook appreso elenca un numero finito di vettori-prototipo. Ogni vettore della griglia viene sostituito dal prototipo più vicino, e di ciascuno resta soltanto l'indice: la griglia di vettori diventa così una griglia di numeri interi."
:width: 100%

Da grandezza continua ad alfabeto. L'arrotondamento al prototipo più vicino è
tutto il trucco: perde qualcosa, ma trasforma un'immagine in una sequenza di
numeri interi, cioè in qualcosa che un modello di linguaggio sa leggere.
```

Il passaggio che conta in {numref}`fig-vq-vae` è l'ultimo, quando del vettore
resta solo l'indice. Da lì in poi un'immagine e una frase sono la stessa cosa
per il modello, sequenze di simboli da un vocabolario finito, ed è questo che
rende pensabile trattarle nello stesso Transformer.

`````{tab} Elementare

Immagina un mosaicista con un magazzino di tessere. Non ha tutti i colori del
mondo: ha un catalogo di 8.192 tessere diverse, ognuna con il suo numero. Per
riprodurre una fotografia la divide in quadratini di 16 pixel per lato, e per
ogni quadratino sceglie dal catalogo la tessera che gli somiglia di più. Alla
fine, invece della fotografia, ha una lista di numeri di catalogo: uno per
quadratino, letti riga per riga come si legge una pagina.

Facciamo il conto su una foto di 512 pixel per lato. I quadratini sono 32 per
riga e 32 per colonna, in tutto 1.024, e altrettanti sono i numeri della lista.
Per distinguere 8.192 tessere bastano 13 cifre binarie, quindi l'intera
fotografia sta in 13.312 bit, poco più di 1,6 kilobyte, contro i 786 kilobyte
della foto grezza: circa 470 volte meno. E il punto non è il risparmio (per
quello esistono già i formati di compressione), è che adesso l'immagine è una
**lista di simboli presi da un elenco fisso**, esattamente come una frase è una
lista di parole prese da un dizionario. Da qui in poi, per una macchina che
completa sequenze, disegnare e scrivere sono lo stesso mestiere.

Due cose sono andate perse per strada. La tessera scelta non è quasi mai
identica al quadratino originale, è la più vicina che c'era in magazzino, e la
differenza è sparita per sempre. E per fare della griglia una lista abbiamo
dovuto decidere un ordine di lettura, riga per riga, come per il testo: ma una
fotografia non ha un verso di lettura, quell'ordine ce lo siamo inventato noi.

`````

`````{tab} Superiore

Sia $C = \{e_1, \dots, e_K\}$ il codebook appreso e $z$ il vettore latente
prodotto dall'encoder per una porzione di immagine. La quantizzazione è la
stessa vista per l'audio,

$$
k^\star = \arg\min_{k \in \{1, \dots, K\}} \lVert z - e_k \rVert^2,
$$

con $k^\star$ token della porzione, e l'addestramento attraversa l'$\arg\min$
non differenziabile con lo *straight-through estimator* già discusso in quel
capitolo. Quel che cambia rispetto all'audio è la forma del dominio: non una
sequenza monodimensionale di frame, ma un reticolo bidimensionale di patch, che
va **linearizzato** (di norma in ordine raster) per diventare una sequenza.

Con i parametri del tokenizzatore di Chameleon {cite}`chameleon2024mixed`,
$K = 8192$ e un'immagine $512 \times 512$ ridotta a $1024$ token: ogni token
copre $512^2/1024 = 256$ pixel, cioè una patch di $16 \times 16$, e vale
$\log_2 8192 = 13$ bit. L'immagine intera occupa $1024 \cdot 13 = 13312$ bit
contro i $512 \cdot 512 \cdot 3 \cdot 8 = 6\,291\,456$ bit del formato grezzo,
un fattore di compressione di circa $472$. Il tasso è irrigidito per
costruzione: qualunque sia il contenuto, l'immagine costa sempre 1.024 token,
né uno di più per una scena complessa né uno di meno per un muro bianco.

Il vocabolario del modello diventa allora l'unione
$V = V_{\text{testo}} \cup V_{\text{img}}$; in Chameleon un vocabolario BPE da
65.536 voci che include le 8.192 dell'immagine. Da notare l'ordine di grandezza
del contesto: un'immagine occupa quanto un migliaio di token di testo, cioè
diverse centinaia di parole, e con un'attenzione quadratica nella lunghezza
della sequenza questo è il vincolo che domina tutto il resto (è il tema della
prossima sezione, sulla risoluzione).

`````

## Dove si incontrano i due flussi

Con i token visivi in mano possiamo dare alle due parole del titolo un
significato preciso. La differenza fra fusione **tardiva** e fusione
**precoce** non è quanta informazione si scambiano le due modalità: è a quale
profondità della rete cominciano a scambiarsela, e se condividono o no i
parametri che le elaborano.

`````{tab} Elementare

Pensa a una redazione. Nella prima, il fotografo e chi scrive lavorano in
stanze separate: il fotografo guarda le sue immagini e passa a chi scrive un
foglietto con quello che ha visto, poi esce di scena. Chi scrive prende il
foglietto e fa il suo mestiere, che è mettere in fila delle parole. La
collaborazione avviene alla fine, vicino all'uscita, e il prodotto è sempre e
soltanto un testo: da quella redazione non esce mai una fotografia, perché chi
tiene la penna non ha mai avuto in mano una macchina fotografica.

Nella seconda redazione c'è una persona sola, e davanti a sé ha una cassa
tipografica come quelle dei caratteri mobili: nelle caselle ci sono le lettere,
e nelle caselle accanto ci sono le tessere del mosaico di cui parlavamo prima.
Sono nella stessa cassa, si prendono con lo stesso gesto. Chi compone può
mettere in riga tre parole, poi mille tessere, poi altre due parole, e
rileggendo la riga non distingue fra le due cose: sono tutti pezzi di piombo.
Questa persona può produrre un'immagine per la stessa ragione per cui può
produrre una frase, e cioè perché i pezzi stanno tutti nella stessa cassa.

Il prezzo si intuisce già: la prima redazione la metti in piedi assumendo un
fotografo che sa già fotografare e uno scrittore che sa già scrivere; la
seconda devi formarla da capo, e chi la compone dovrà imparare due mestieri
insieme, con la stessa testa.

`````

`````{tab} Superiore

**Fusione tardiva.** Due encoder addestrati separatamente producono
rappresentazioni che si incontrano vicino all'uscita. Nel caso estremo,
l'addestramento contrastivo della sezione su CLIP, l'unica interazione fra le
modalità è un prodotto scalare fra due vettori. Nel caso dei connettori della
sezione precedente, l'interazione è più profonda (le patch entrano come
prefisso o via cross-attention) ma i pesi restano in gran parte quelli di due
modelli pre-addestrati per conto proprio, e soprattutto la testa di uscita è
una softmax su $V_{\text{testo}}$: la capacità generativa è asimmetrica per
costruzione.

**Fusione precoce.** Le due modalità diventano token dello stesso vocabolario
$V$ **all'ingresso**, e da lì in avanti non esistono più due flussi: c'è una
sequenza sola, $s = (s_1, \dots, s_n)$ con $s_t \in V$, che un unico
Transformer attraversa dal primo strato all'ultimo con la stessa attenzione e
gli stessi pesi. L'obiettivo è quello del capitolo sui modelli di linguaggio,
senza aggiunte:

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{n} \log p_\theta(s_t \mid s_{<t}),
$$

dove $\theta$ sono i parametri dell'unico modello e la somma corre su tutti i
token della sequenza, visivi e testuali indifferentemente. È l'impostazione di
Chameleon {cite}`chameleon2024mixed`: sequenze **miste**, in cui un'immagine è
un blocco di 1.024 token delimitato da due simboli speciali di apertura e
chiusura, e sequenze di addestramento che alternano i due tipi in ordine
arbitrario (testo con immagini in mezzo, immagini con didascalie, pagine web
intere).

La conseguenza è quella che cercavamo: siccome la softmax finale copre tutto
$V$, il modello può emettere un token visivo dove prima poteva emettere solo
una parola. Generare un'immagine è campionare 1.024 volte dalla stessa softmax
da cui si campionano le parole, e passare i risultati al decoder del
tokenizzatore. Nessuna testa aggiuntiva, nessun secondo modello: la
simmetria non è una funzionalità in più, è un effetto della scelta del
vocabolario.

`````

La stessa ricetta si estende oltre le immagini ferme. Emu3 {cite}`wang2024emu3`
tokenizza allo stesso modo testo, immagini **e video**, con un quantizzatore
che comprime anche nel tempo (quattro fotogrammi consecutivi diventano un solo
strato di token, altrimenti un secondo di ripresa costerebbe quanto le
ventiquattro o trenta fotografie che lo compongono) e addestra da zero un unico
Transformer con l'unico obiettivo di predire il token successivo. Lo stesso
modello genera e capisce: nella stessa sequenza di simboli sta la richiesta di
produrre un'immagine e la domanda su un'immagine già data. Che il video entri
quasi senza modifiche è un'indicazione: quel che rende la fusione precoce
interessante non è la modalità in sé, è che il formato «sequenza di simboli» le
assorbe tutte.

## Perché non l'hanno fatto subito

A leggerla così, la fusione precoce sembra la cosa ovvia da fare, e viene da
chiedersi perché la strada battuta sia stata a lungo l'altra. Le ragioni sono
due, ed entrambe dicono qualcosa sui modelli in generale.

La prima è economica ed è la più banale: la fusione tardiva riusa. Un encoder
visivo pre-addestrato e un modello di linguaggio pre-addestrato esistono già,
sono costati a qualcun altro, e il connettore che li unisce si addestra con
risorse alla portata di un laboratorio universitario. La fusione precoce non
riusa niente, perché il suo vocabolario non è quello di nessun modello
esistente: il pre-addestramento va rifatto da zero, su migliaia di miliardi di
token.

La seconda è più interessante, ed è che il modello che ne esce è **più difficile
da addestrare**. Non «più lento»: instabile. Le curve divergono, e lo fanno
tardi, quando una fetta importante del calcolo è già stata spesa.

`````{tab} Elementare

Immagina due cantanti che condividono un solo microfono e un solo amplificatore
con una sola manopola del volume. Il primo canta piano, il secondo forte. Per
farsi sentire, il primo alza un po' la voce; allora il secondo, per non essere
coperto, alza la sua; e il primo di nuovo. Nessuno dei due sta facendo niente
di sbagliato, ciascuno sta solo cercando di farsi sentire, ma il livello sale e
sale, e a un certo punto l'amplificatore non ce la fa più: quel che esce
dall'altoparlante non è più musica, è un fischio.

Nel modello a fusione precoce i due cantanti sono le due modalità, e
l'amplificatore condiviso sono i pesi. Testo e immagini hanno statistiche
diverse (una tessera di mosaico è molto meno prevedibile di un articolo
determinativo, e la loro «voce» dentro la rete ha ampiezze diverse), ma passano
per gli stessi parametri, e ciascuna, per contare qualcosa nel risultato, tende
a farsi un po' più grossa. I numeri interni crescono, lentamente, per milioni
di passi; e poiché sono memorizzati con una precisione finita, prima o poi si
esce dall'intervallo in cui quei numeri hanno ancora un senso, e
l'addestramento salta. La cura è quella che userebbe un fonico: mettere un
limitatore esattamente nel punto in cui il segnale entra
nell'amplificatore, così che nessuno dei due possa alzare la voce oltre una
certa soglia. Nessuno perde il diritto di parola, si toglie solo la possibilità
di urlare.

`````

`````{tab} Superiore

Il meccanismo documentato nel lavoro su Chameleon {cite}`chameleon2024mixed`
parte da una proprietà innocua della softmax: è invariante per traslazione,
$\mathrm{softmax}(z) = \mathrm{softmax}(z + c)$, e quindi il suo risultato non
dice nulla sul livello assoluto dei logit. Con pesi condivisi fra modalità
dalle statistiche diverse, ciascuna può allora «competere» con l'altra alzando
un po' la norma delle proprie attivazioni, senza che la funzione di perdita se
ne accorga. Le norme di query e chiavi crescono, con esse i logit
$QK^\top/\sqrt{d_k}$ che entrano nella softmax dell'attenzione, e quando quei
valori escono dall'intervallo in cui l'aritmetica a precisione ridotta (bf16)
rappresenta ancora qualcosa, l'addestramento diverge. La crescita è lenta, ed è
questo a renderla insidiosa: nel lavoro citato il problema compare sopra gli 8
miliardi di parametri e i mille miliardi di token, e le divergenze arrivano
quando è già stato percorso un buon 20–30% dell'addestramento. È la ragione per
cui non lo si scopre addestrando modelli piccoli.

Due accorgimenti, entrambi di normalizzazione, lo tengono a bada. Il primo è la
**query-key normalization**: si normalizza $Q$ e $K$ *prima* del prodotto
scalare,

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(
\frac{\mathrm{LN}(Q)\,\mathrm{LN}(K)^{\top}}{\sqrt{d_k}}\right) V,
$$

dove $\mathrm{LN}$ è la layer normalization, $d_k$ la dimensione delle chiavi e
$V$ sono qui i valori dell'attenzione (non il vocabolario di poco fa):
i logit dell'attenzione smettono di dipendere dalla scala delle attivazioni, e
la loro crescita è limitata alla sorgente. Il secondo riguarda **dove** stanno
le normalizzazioni nel blocco. Nella disposizione *pre-norm* usuale il flusso
residuo riceve l'uscita non normalizzata del sotto-strato,
$x \leftarrow x + F(\mathrm{LN}(x))$, e nulla impedisce alla norma di $x$ di
crescere strato dopo strato; spostando la normalizzazione a valle,
$x \leftarrow x + \mathrm{LN}(F(x))$, quel che si somma al flusso residuo è
sempre di scala controllata. Nel lavoro citato la normalizzazione di query e
chiavi serve a entrambe le taglie di modello: a quello da 7 miliardi di
parametri basta affiancarle il dropout, mentre quello da 34 miliardi chiede la
riorganizzazione delle norme, che con il dropout non va d'accordo e finisce per
sostituirlo.

Resta l'ultima softmax, quella sul vocabolario, dove la stessa invarianza per
traslazione lascia scivolare i logit tutti insieme verso l'alto (in un modello
di solo testo il fenomeno ha già un nome, *logit drift*). Lì si aggiunge alla
perdita un termine $10^{-5}\log^2 Z$, dove $Z$ è la funzione di partizione
della softmax, cioè la somma degli esponenziali dei logit: non tocca le
probabilità, che dal livello assoluto non dipendono, ma toglie al modello la
libertà di farlo crescere.

Vale la pena isolare la lezione generale, che va oltre la multimodalità: quando
più sorgenti eterogenee condividono gli stessi parametri, la competizione fra
di esse si scarica sulle **norme**, e la stabilità va difesa esattamente dove
quelle norme entrano in una softmax, che della loro crescita non si accorge
finché non è troppo tardi.

`````

C'è poi un costo di natura diversa, e non si cura con una normalizzazione:
**quantizzare butta via**. L'encoder continuo della fusione tardiva conserva
tutto quello che ha visto in un vettore di numeri reali; il tokenizzatore lo
arrotonda a una voce di catalogo, e la differenza non è recuperabile da nessuna
parte a valle. Con i numeri di prima, ogni patch di $16 \times 16$ pixel (768
byte di colori) diventa uno fra 8.192 simboli: quel che sopravvive è la
struttura grossa, non il tratto sottile. Il caso peggiore è il testo dentro
l'immagine, perché un carattere di un documento fotografato sta proprio nella
scala di dettaglio che l'arrotondamento cancella, ed è una limitazione che i
lavori su questi modelli riconoscono apertamente. La sezione successiva, sulla
risoluzione, mostrerà che il problema si sposta ma non sparisce nemmeno con gli
encoder continui.

## La terza via: due obiettivi, un modello solo

A questo punto la domanda diventa più precisa. La fusione precoce ci serviva
per un motivo solo: rendere l'immagine qualcosa che il modello possa
**emettere**, non solo leggere. Ma quel motivo richiede davvero che l'immagine
sia fatta di simboli discreti? Il capitolo sui modelli di diffusione ha
mostrato un modo completamente diverso di generare un'immagine, che con i
vocabolari non ha niente a che fare: partire dal rumore e toglierne un velo
alla volta, come si è visto nella sezione sul funzionamento della diffusione.

Transfusion {cite}`zhou2024transfusion` prende sul serio l'ipotesi: un solo
Transformer, un solo insieme di parametri, ma **due obiettivi diversi** a
seconda del tipo di token che sta trattando. Sul testo, la predizione del token
successivo di sempre. Sull'immagine, la diffusione: niente codebook, niente
arrotondamento, le patch restano vettori continui (latenti di un autoencoder,
come nella diffusione latente) e il modello impara a stimarne il rumore.

`````{tab} Elementare

Una persona che sa scrivere e dipingere non usa la stessa tecnica per le due
cose, e non c'è ragione di obbligarla. Il testo lo scrive da sinistra a destra,
una parola dopo l'altra, e mentre sceglie una parola può rileggere solo quel
che ha già scritto: quello che verrà dopo non esiste ancora. Il quadro no: il
quadro lo abbozza tutto insieme e poi ci ripassa sopra, e mentre ritocca
l'angolo in alto a sinistra guarda anche l'angolo in basso a destra, perché
sono già lì tutti e due e devono stare insieme.

È esattamente questa la differenza che il modello si porta dentro. Quando
elabora le parole, ogni posizione guarda solo all'indietro. Quando elabora i
pezzi di un'immagine, dentro quel blocco tutti guardano tutti, avanti e
indietro, perché non c'è un verso di lettura da rispettare: è la stessa
obiezione che avevamo lasciato in sospeso quando abbiamo dovuto decidere,
arbitrariamente, di leggere il mosaico riga per riga.

Il vantaggio pratico si vede subito: siccome l'immagine non viene mai
arrotondata a un catalogo di tessere, il dettaglio fine non viene buttato via
in partenza. Il vantaggio di principio è che si smette di fingere che un
disegno sia una frase.

`````

`````{tab} Superiore

Due meccanismi reggono la costruzione.

**L'attenzione mista.** La maschera è causale fra i token di testo, come in
qualunque decoder, ma **bidirezionale all'interno di ciascun blocco immagine**:
le patch della stessa immagine si vedono tutte a vicenda, mentre continuano a
vedere solo il passato per quel che riguarda il testo che le precede. La
motivazione è che l'ordine raster è una finzione: gli elementi di un'immagine
sono co-presenti, non successivi, e imporre loro una causalità è un vincolo
gratuito.

**I due obiettivi.** La perdita è la somma dei due termini che il libro ha già
incontrato separatamente,

$$
\mathcal{L} = \mathcal{L}_{\text{LM}} + \lambda\, \mathcal{L}_{\text{DDPM}},
$$

dove $\mathcal{L}_{\text{LM}}$ è la cross-entropia sul token successivo
calcolata sulle sole posizioni testuali, $\mathcal{L}_{\text{DDPM}}$ è
l'errore quadratico sul rumore stimato calcolato sulle sole posizioni visive
(la $\mathbb{E}\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2$ della sezione
sulla diffusione) e $\lambda$ pesa il secondo rispetto al primo (nel lavoro
originale $\lambda = 5$). Le due perdite non riguardano parametri diversi:
attraversano gli stessi strati, e ogni peso del Transformer riceve gradienti da
entrambe. Restano specifici della modalità soltanto gli innesti agli estremi
(l'embedding dei token di testo da una parte, i pochi strati che impacchettano
e spacchettano le patch latenti dall'altra), che è il minimo indispensabile per
far entrare due tipi di dato nella stessa sequenza.

In generazione il modello lavora in due modalità e passa dall'una all'altra da
sé: campiona parole finché non emette il simbolo speciale che apre
un'immagine, a quel punto accoda un blocco di patch di puro rumore, esegue su
quel blocco il ciclo di denoising del capitolo sulla diffusione (con tutto il
testo precedente che fa da condizionamento, tramite l'attenzione), chiude il
blocco e torna a scrivere parole.

`````

La maschera di attenzione è l'unico pezzo davvero nuovo, e sta in poche righe.
Costruiamola per una sequenza di nove posizioni: tre token di testo, un blocco
immagine di quattro patch, altri due token di testo.

```python
import numpy as np

def maschera_mista(segmenti):
    """Maschera di attenzione per una sequenza mista di testo e immagini.

    segmenti: lista di coppie (tipo, lunghezza), con tipo "testo" o "immagine".
    Restituisce una matrice booleana: consentito[i, j] e' True se la posizione
    i puo' guardare la posizione j."""
    tipi, blocchi = [], []
    for indice, (tipo, lunghezza) in enumerate(segmenti):
        tipi.extend([tipo] * lunghezza)        # tipo di ogni posizione
        blocchi.extend([indice] * lunghezza)   # a quale segmento appartiene
    n = len(tipi)
    blocchi = np.array(blocchi)
    e_immagine = np.array([t == "immagine" for t in tipi])

    consentito = np.tril(np.ones((n, n), dtype=bool))     # causale: j <= i
    stesso_blocco = blocchi[:, None] == blocchi[None, :]
    dentro_immagine = stesso_blocco & e_immagine[:, None] & e_immagine[None, :]
    return consentito | dentro_immagine    # nel blocco immagine, anche avanti


m = maschera_mista([("testo", 3), ("immagine", 4), ("testo", 2)])
for riga in m:
    print("".join("X" if v else "." for v in riga))
```

L'uscita disegna la struttura meglio di qualunque descrizione (`X` dove
l'attenzione è permessa, `.` dove è vietata):

```text
X........
XX.......
XXX......
XXXXXXX..
XXXXXXX..
XXXXXXX..
XXXXXXX..
XXXXXXXX.
XXXXXXXXX
```

Le prime tre righe e le ultime due sono la scaletta causale che conosciamo. Le
quattro righe centrali sono il blocco immagine: ciascuna vede tutte e quattro
le patch, comprese quelle che vengono dopo, e vede tutto il testo che precede.
Nessuna riga di testo, invece, guarda avanti. In PyTorch questa matrice si
passa a `nn.MultiheadAttention` come `attn_mask`, con l'avvertenza che lì la
convenzione è rovesciata (per una maschera booleana, `True` significa
*vietato*): si passa `torch.from_numpy(~m)`.

## Quando serve un vocabolario comune

Le tre strade non si superano a vicenda, e la scelta si fa sul meccanismo.

Se il compito è **capire** (descrivere una foto, rispondere a domande su un
grafico, leggere un documento), la fusione tardiva è la scelta ragionevole e lo
resta: riusa due modelli già addestrati, non paga il dazio della
quantizzazione, e chiede solo un connettore. Pagare per una simmetria che non
si userà è cattiva ingegneria.

Se il compito è **produrre immagini e testo dentro lo stesso sistema**, o
peggio alternarli (una risposta che contiene un diagramma, una figura corretta
alla luce di quanto detto due paragrafi prima), allora la strada del connettore
chiede due modelli e una cerniera fra i due, e la cerniera è esattamente il
punto in cui l'informazione si perde. Un vocabolario comune, o un modello con
due obiettivi, tiene tutto in un contesto solo.

Onestà d'obbligo, per chiudere: il campo non ha deciso, e i sistemi reali sono
spesso ibridi (un encoder continuo per capire, un decoder generativo per
produrre, dentro lo stesso prodotto). L'argomento a favore della fusione
precoce non è che generi meglio, perché su questo confrontare i sistemi ha
poco senso e invecchia in fretta: è che la condivisione dei parametri
*dovrebbe* produrre un **trasferimento** fra le due direzioni. Imparare a
disegnare un gatto dovrebbe aiutare a riconoscerlo, perché per generarlo
bisogna sapere com'è fatto, mentre per descriverlo spesso basta indovinare
dalla didascalia (il problema che l'overview del capitolo ha chiamato
allucinazione visiva).

La prova che deciderebbe la questione ha allora questa forma: a parità di
parametri, di dati e di calcolo, la capacità di *capire* di un modello a
fusione precoce migliora quando gli si insegna anche a *generare*? Se sì, il
vocabolario comune non è una comodità architetturale, è un modo di imparare
meglio, e vale il costo del pre-addestramento da zero. Se le due capacità si
limitano a convivere senza aiutarsi, la fusione precoce resta una scelta di
ingegneria, giustificata quando serve un solo sistema al posto di due, e il
connettore continuerà a vincere per la ragione più semplice del mondo: costa
meno.

```{admonition} Da ricordare
:class: important
- Un modello che innesta un encoder visivo su un modello di linguaggio
  {cite}`liu2023visual` è **asimmetrico**: l'immagine è una condizione, e la
  softmax finale copre solo il vocabolario del testo. Non può generare
  immagini perché non ha simboli per dirle.
- La **quantizzazione vettoriale** del VQ-VAE {cite}`oord2017neural`, già usata
  per il suono nei codec neurali, dà all'immagine i suoi simboli: un codebook
  di prototipi, l'indice del più vicino come token. Con un codebook da 8.192
  voci, un'immagine $512 \times 512$ diventa 1.024 token da 13 bit.
- **Tardiva** è la fusione di due encoder addestrati a parte che si incontrano
  vicino all'uscita; **precoce** è un vocabolario unico all'ingresso, con un
  solo Transformer e un solo obiettivo autoregressivo
  {cite}`chameleon2024mixed`, esteso al video da {cite}`wang2024emu3`. Solo la
  seconda genera, perché la simmetria è nel vocabolario.
- Il conto della fusione precoce è doppio: il pre-addestramento va rifatto da
  zero, e con pesi condivisi fra modalità dalle statistiche diverse le **norme
  crescono**, finché i logit dell'attenzione escono dall'intervallo in cui
  l'aritmetica a precisione ridotta ha ancora senso. Si difende normalizzando
  query e chiavi prima del prodotto scalare e spostando le normalizzazioni a
  valle dei sotto-strati.
- **Quantizzare butta via**: ogni patch di $16 \times 16$ pixel diventa uno fra
  8.192 simboli, e il dettaglio fine (il testo dentro una foto) è il primo a
  sparire.
- **Transfusion** {cite}`zhou2024transfusion` tiene un modello solo con due
  obiettivi (autoregressivo sul testo, diffusione sull'immagine) e
  un'attenzione **mista**: causale fra le parole, bidirezionale dentro il
  blocco immagine, perché un'immagine non ha un ordine di lettura.
- La domanda aperta non è quale generi meglio, ma se condividere i parametri
  produca **trasferimento** fra capire e generare. Se sì, il vocabolario comune
  vale il suo costo; se no, l'innesto vince perché costa meno.
```
