# Imparare a vedere senza etichette

Le fotografie su cui AlexNet si è addestrata nel 2012, un milione e
duecentomila, non sono piovute dal cielo: qualcuno le ha guardate a una a una e
ha confermato che dentro c'era davvero la cosa che il nome prometteva. ImageNet
è stata costruita così, per anni, da decine di migliaia di persone reclutate su
piattaforme di micro-lavoro e pagate a cottimo. È il lavoro invisibile su cui
poggia la sezione sul transfer learning: quando scarichiamo una rete
«pre-addestrata» stiamo prendendo in prestito il tempo di quegli annotatori. E
quel lavoro non cresce insieme al problema: le etichette costano, e le costa
qualcuno a mano, una per una; l'elenco delle categorie utili cambia da un
mestiere all'altro, e per moltissimi settori non esiste affatto. Le immagini
*senza* etichetta, al contrario, sono praticamente infinite.

Da questa asimmetria nasce la domanda: si può insegnare a una rete a descrivere bene un'immagine **senza che nessuno dica mai che cosa c'è
nella foto**? Quella descrizione, la lista di numeri in cui la rete riassume
un'immagine, si chiama la sua **rappresentazione**, ed è la cosa che vogliamo:
il resto si costruisce sopra. La risposta è sì, e per arrivarci bisogna
rovesciare l'uso di uno strumento appena costruito. Nella sezione sulla data
augmentation le trasformazioni erano un freno, un modo di impedire alla rete di
imparare a memoria. Qui non frenano niente: qui l'augmentation **è il segnale
di addestramento**.

## Un compito la cui risposta è già nei dati

Fin qui, nel libro, una rete imparava perché qualcuno le diceva la risposta:
si chiama apprendimento **supervisionato**, come uno studente con l'insegnante
accanto che corregge. Il meccanismo di questa sezione si chiama invece
**auto-supervisionato** perché la correzione se la dà da sé, e la definizione
sta in una riga: si inventa un compito (un **pretesto**, *pretext task*) la cui
risposta corretta è ricavabile dai dati stessi, senza che nessuno la scriva.
Risolverlo non interessa a nessuno; interessa quello che il modello è costretto
a capire per riuscirci, e che resta nell’**encoder** (la parte della rete che
trasforma l'immagine nella sua lista di numeri, il riassunto interno di cui si
diceva) quando il pretesto si butta via. È un'idea che il libro ritroverà in
ogni campo: nel {doc}`capitolo sui Transformer </Transformers/overview>` regge il pre-addestramento dei modelli
di linguaggio, in quello sull'audio fa imparare a wav2vec 2.0 e a HuBERT la
struttura del parlato da migliaia di ore mai trascritte.

Sul testo un buon pretesto si trova subito: si copre una parola e si chiede di
indovinarla, e la risposta giusta è la parola che si è coperta. Un'immagine non
ha parole, e quel che segue è la ricerca di un pretesto altrettanto buono per i
pixel, in due famiglie: rendere il modello **indifferente** alle trasformazioni
che non cambiano il contenuto, oppure fargli **ricostruire** ciò che è
nascosto.

## Due viste della stessa foto

La prima famiglia si chiama **contrastiva**, perché il modello impara mettendo
a confronto: non basta avvicinare due cose, bisogna insieme allontanarne
altre. Si prende un'immagine e la si trasforma **due volte**, a caso e in modo
indipendente: due ritagli diversi, due variazioni di colore, magari una
sfocatura. Si ottengono due *viste* della stessa scena, e al modello si chiede
una cosa sola: che le due viste della stessa immagine vengano descritte in modo
simile fra loro, e diverso da come descrive le viste di tutte le altre immagini
del gruppo che sta guardando in quel momento (il **batch**, cioè la manciata di
esempi che si elaborano insieme). Quelle altre viste, i rivali da cui il gemello
va distinto, hanno un nome che tornerà spesso: si chiamano i **negativi**. È la
ricetta di SimCLR {cite}`chen2020simple`, il lavoro che nel 2020 ha mostrato
che per far funzionare l'idea non serve nessuna macchineria in più: bastano un
encoder normale, due trasformazioni scelte bene e batch abbastanza grandi.

`````{tab} Elementare

Ritaglia da una stessa fotografia due francobolli: uno prende il muso del
gatto, l'altro la coda e un pezzo di muro. Fai lo stesso con altre novantanove
fotografie e mescola tutti i duecento ritagli sul tavolo. Il gioco è: per ogni
ritaglio, ritrovare il suo gemello, cioè l'altro pezzo che veniva dalla stessa
foto.

Nessuno ha dovuto dire che nella foto c'era un gatto: la risposta giusta la
conosciamo per costruzione, perché i due ritagli li abbiamo fatti noi. Eppure
per vincere bisogna capire parecchio: che muso e coda appartengono allo stesso
animale, che quel pelo e quel muro stanno insieme. Bisogna farsi un'idea di
*che cosa* raffigura ogni ritaglio, perché è l'unico modo di riconoscere il
gemello in mezzo a centonovantotto estranei. Almeno, così sembra: fra poco
vedremo che c'è anche un modo di vincere senza aver capito niente, ed è il
problema centrale di tutta questa storia.

`````

`````{tab} Superiore

Sia $\mathcal{B} = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ un batch di $N$
immagini. Da ciascuna si campionano due trasformazioni indipendenti
$T, T' \sim \mathcal{T}$, dove $\mathcal{T}$ è la famiglia di trasformazioni
ammesse, e si ottengono le viste $\tilde{\mathbf{x}} = T(\mathbf{x})$ e
$\tilde{\mathbf{x}}' = T'(\mathbf{x})$, in tutto $2N$. Un
encoder $f_\theta$ (una ResNet, o un ViT) trasforma ogni vista nella
rappresentazione $\mathbf{h} = f_\theta(\tilde{\mathbf{x}})$; una **testa di proiezione**
$g_\phi$, un piccolo MLP, la porta in uno spazio più piccolo,
$\mathbf{z} = g_\phi(\mathbf{h})$,
dove si calcola la perdita. A valle si riusa $\mathbf{h}$, non $\mathbf{z}$: la proiezione impara
a buttare via proprio l'informazione che la loss chiede di ignorare (il colore,
la posizione del ritaglio) e a un compito diverso quell'informazione può
servire.

La perdita è la **NT-Xent** (*normalized temperature-scaled cross entropy*),
cioè una InfoNCE {cite}`oord2018representation` calcolata sulle viste. Per la
coppia positiva $(i, j)$:

$$
\ell_{i,j} = -\log
\frac{\exp\big(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau\big)}
{\displaystyle\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]}\,
\exp\big(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau\big)},
\qquad
\mathrm{sim}(\mathbf{u},\mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}
{\lVert \mathbf{u} \rVert \, \lVert \mathbf{v} \rVert},
$$

dove $\mathbf{z}_i$ e $\mathbf{z}_j$ sono le proiezioni delle due viste della stessa immagine,
$\mathrm{sim}$ è la similarità coseno, $\tau > 0$ la temperatura e la somma al
denominatore corre sulle altre $2N-1$ viste del batch, che fanno da
**negativi**. La perdita totale è la media di $\ell_{i,j}$ su tutte le $2N$
coppie ordinate. È la stessa InfoNCE che il libro usa per allineare immagini e
didascalie, con una differenza sostanziale: là il positivo è la didascalia
scritta da una persona, qui è una seconda copia deformata della stessa immagine.
La supervisione non viene dal linguaggio, viene dalla trasformazione.

`````

In PyTorch tutto questo sta in una decina di righe, e la parte da guardare è
**da dove esce la risposta giusta**: non la scrive nessuno, viene fuori
soltanto dall'ordine in cui abbiamo impilato le viste. Se le prime $N$ righe
sono le viste A e le seconde $N$ sono le viste B nello stesso ordine, la
gemella della riga $i$ è la riga $i+N$, e questo il computer lo sa fare da sé.

```python
import torch
import torch.nn.functional as F

def nt_xent(z, tau=0.5):
    """z: (2N, d). Le prime N righe sono le viste A, le seconde le viste B
    nello stesso ordine: la gemella della riga i e' la riga i+N."""
    n = z.shape[0] // 2
    z = F.normalize(z, dim=1)          # sulla sfera: il prodotto e' un coseno
    sim = (z @ z.t()) / tau            # (2N, 2N) coseni su temperatura
    sim.fill_diagonal_(float("-inf"))  # nessuna vista e' positiva di se stessa
    riga = torch.arange(2 * n, device=z.device)
    bersagli = (riga + n) % (2 * n)    # la gemella della riga i: i+N, oppure i-N
    return F.cross_entropy(sim, bersagli)   # cross-entropy per riga, mediata
```

Non compare nessuna etichetta e nessun elenco di categorie: la risposta giusta è
un numero di riga, e a fornirlo è stato soltanto il modo in cui abbiamo impilato
le due viste.

## Scegliere le trasformazioni è scrivere il compito

Ed eccoci al rovesciamento annunciato. Nella sezione sulla data augmentation le
trasformazioni servivano a non far imparare a memoria, e la regola era: sono
ammesse se **non cambiano l'etichetta**. Qui un'etichetta non c'è, e le
trasformazioni non decorano il compito, lo **definiscono**: chiedere che due
viste finiscano vicine significa dire al modello *a che cosa deve essere
indifferente*. Se ruotiamo, gli insegniamo che l'orientamento non conta; se
cambiamo colore, che il colore non conta. L'elenco delle trasformazioni
ammesse **è** la specifica di ciò che il modello considererà «la stessa cosa».

La coppia che conta, negli esperimenti di SimCLR, è **ritaglio casuale più
disturbo del colore**, e la ragione per cui il secondo è indispensabile è la
lezione più generale di questa sezione.

`````{tab} Elementare

Torniamo ai ritagli sul tavolo. C'è un modo di barare, e un modello pigro lo
trova subito: due ritagli della stessa fotografia hanno quasi sempre gli stessi
colori. La foto del gatto sul divano rosso è rossastra dappertutto, quella della
spiaggia è azzurra e sabbia dappertutto. Basta allora guardare la tinta media,
senza capire *niente* di quello che raffigurano.

È il tipo di scorciatoia che rovina un pretesto: il modello vince il gioco e non
impara nulla di utile. Il rimedio è togliergli l'indizio, cambiando il colore
dei due ritagli in modo indipendente, uno più caldo e uno più freddo. A quel
punto la tinta non è più una prova, e l'unica informazione rimasta per
riconoscere il gemello è quella che volevamo fin dall'inizio: la forma delle
cose.

`````

`````{tab} Superiore

Il problema è quello che si chiama *shortcut learning*: se una statistica di
basso livello basta a risolvere il pretesto, l'ottimizzazione la userà, perché è
la via meno costosa verso una loss bassa. Qui la statistica è la distribuzione
dei colori: due patch della stessa immagine hanno istogrammi cromatici molto
simili, già quasi sufficienti a identificare la coppia. Gli autori di SimCLR lo
mostrano confrontando gli istogrammi delle patch e concludono che comporre il
ritaglio con il **color distortion** (jitter di luminosità, contrasto,
saturazione e tinta, più una conversione in scala di grigi applicata con
probabilità bassa) è indispensabile: nessuna delle due trasformazioni, presa da
sola, produce rappresentazioni utili, mentre la loro composizione sì
{cite}`chen2020simple`.

La regola che se ne ricava vale ben oltre questo caso. In un compito
auto-supervisionato **le invarianze imposte e le scorciatoie disponibili sono la
stessa cosa vista da due lati**: ogni trasformazione applicata toglie un
indizio, e l'indizio che *non* togliamo diventa la strada che il modello
prenderà.

`````

Che la scorciatoia esista si verifica in una ventina di righe, senza addestrare
nulla. Costruiamo duecento immagini finte, ciascuna con una propria dominante di
colore, ne estraiamo due ritagli casuali a testa e proviamo ad accoppiarli
usando **soltanto** l’**istogramma** dei colori, cioè il conto di quanti pixel
di ogni tinta ci sono in un ritaglio, senza sapere dove stanno. È la carta
d'identità cromatica di un'immagine, e ignora completamente la forma.

```python
import numpy as np

rng = np.random.default_rng(0)
N, LATO, RITAGLIO, BIN = 200, 64, 24, 8

# Ogni "immagine" ha una dominante di colore propria (la sua firma cromatica)
# piu' una texture casuale: due ritagli qualunque ne ereditano la dominante.
dominante = rng.uniform(0.2, 0.8, size=(N, 1, 1, 3))
rumore = 0.15 * rng.standard_normal((N, LATO, LATO, 3))
immagini = np.clip(dominante + rumore, 0, 1)

def ritaglia(img):
    i, j = rng.integers(0, LATO - RITAGLIO, size=2)
    return img[i:i + RITAGLIO, j:j + RITAGLIO]

def istogramma(r):
    # 8 bin per canale, concatenati e normalizzati: 24 numeri per ritaglio
    h = [np.histogram(r[..., c], bins=BIN, range=(0, 1))[0] for c in range(3)]
    h = np.concatenate(h).astype(float)
    return h / h.sum()

def accoppia(vista_a, vista_b):
    # per ogni ritaglio di A, il ritaglio di B con l'istogramma piu' vicino
    d = ((vista_a[:, None, :] - vista_b[None, :, :]) ** 2).sum(-1)
    return (d.argmin(axis=1) == np.arange(len(vista_a))).mean()

def colore_casuale(r):
    # jitter di colore: guadagno per canale + luminosita', estratti per ritaglio
    g = rng.uniform(0.6, 1.4, size=3)
    return np.clip(r * g + rng.uniform(-0.2, 0.2), 0, 1)

A = [ritaglia(x) for x in immagini]
B = [ritaglia(x) for x in immagini]

nudi_a = np.array([istogramma(r) for r in A])
nudi_b = np.array([istogramma(r) for r in B])
jit_a = np.array([istogramma(colore_casuale(r)) for r in A])
jit_b = np.array([istogramma(colore_casuale(r)) for r in B])

print("solo ritaglio:          ", round(accoppia(nudi_a, nudi_b), 3))
print("ritaglio + colore:      ", round(accoppia(jit_a, jit_b), 3))
print("livello del caso:       ", round(1 / N, 3))
```

```text
solo ritaglio:           0.955
ritaglio + colore:       0.05
livello del caso:        0.005
```

Con il solo ritaglio l'istogramma dei colori risolve il 95,5% degli
abbinamenti: un modello addestrato in quelle condizioni non ha nessun motivo di
guardare le forme. Aggiungendo il disturbo del colore la stessa scorciatoia
crolla al 5%. Il metro di paragone è quanto prenderebbe tirando a indovinare:
con duecento candidati fra cui scegliere si azzecca una volta su duecento, cioè
lo 0,5%. Il 5% resta dieci volte tanto, e va detto: il disturbo non cancella
l'indizio, lo rende inaffidabile, e questo basta perché al modello convenga
cercarne uno migliore. Il punto è strutturale: la difficoltà del pretesto non
sta nei dati, sta nelle trasformazioni che abbiamo scelto.

## Il prezzo dei negativi

Fin qui i negativi sono i compagni di batch: i rivali del gemello sono le altre
viste che stanno sul tavolo in quel preciso momento. È una scelta comoda, e
diventa subito il vincolo che comanda su tutto. Con pochi rivali il gioco è
facile e si impara poco: ritrovare il gemello fra tre è quasi ovvio, fra
quattromila no. Servono quindi batch enormi, e un batch enorme va tenuto tutto
insieme nella memoria degli acceleratori mentre si calcolano le correzioni:
SimCLR arriva a batch da $4096$ immagini e per reggerli lavora su $128$
acceleratori in parallelo, il che taglia fuori chiunque non abbia un centro di
calcolo.

Da qui la mossa che scioglie il nodo, e conviene dire subito che è arrivata
**prima**: MoCo è di qualche mese anteriore a SimCLR, e non nasce come sua
risposta ma come attacco allo stesso problema, già noto. La mossa è staccare
l'una dall'altra due cose che fin qui erano la stessa, **quanti rivali il
modello vede** e **quante immagini si elaborano insieme**. Perché mai i
negativi devono essere per forza i compagni di batch?

`````{tab} Elementare

Immagina di dover riconoscere il ritaglio gemello non fra i duecento che hai sul
tavolo adesso, ma fra tutti quelli visti nell'ultima ora, tenuti in una scatola
che funziona come una coda: ogni volta che ne arrivano di nuovi li metti sopra e
butti via i più vecchi, così resta grande sempre uguale, comunque sia grande il
mazzo che guardi in una volta sola.

C'è però un guaio. Chi giudica la somiglianza sei tu, e tu impari in
continuazione: i ritagli descritti mezz'ora fa lo sono stati con criteri diversi
da quelli di adesso, e confrontarli con i nuovi è come paragonare misure prese
con due righelli diversi. La soluzione è affidare le descrizioni a una copia di
te stesso che cambia idea molto lentamente, aggiornandosi di un millesimo alla
volta: abbastanza aggiornata da essere sensata, abbastanza lenta perché tutta la
scatola resti confrontabile.

`````

`````{tab} Superiore

**MoCo** {cite}`he2020momentum` riformula l'apprendimento contrastivo come la
costruzione di un **dizionario dinamico**. Una nota sulle date, perché l'ordine
in cui questa sezione presenta i metodi è logico e non cronologico: MoCo esce in
preprint nel novembre 2019 e SimCLR nel febbraio 2020, quindi la coda non è una
risposta a SimCLR ma una soluzione indipendente al medesimo problema, che allora
si poneva rispetto ai *memory bank* della generazione precedente.
La vista-ancora passa in un encoder
$f_{\theta_q}$ che produce la *query* $\mathbf{q}$; le altre viste passano in un secondo
encoder $f_{\theta_k}$ che produce le *chiavi*, accumulate in una **coda** FIFO
di dimensione fissa (nel lavoro originale $K = 65\,536$ elementi): a ogni passo
si accodano le chiavi del mini-batch corrente e si scartano le più vecchie. La
perdita è una InfoNCE il cui denominatore somma sulla coda, non sul batch: il
numero di negativi non dipende più dalla dimensione del batch.

Resta il problema della **coerenza**: se $\theta_k$ cambiasse a ogni passo come
$\theta_q$, chiavi accodate in momenti diversi sarebbero prodotte da encoder
diversi e i loro prodotti scalari non sarebbero confrontabili. La soluzione è
non addestrare l'encoder delle chiavi per retropropagazione, ma mantenerlo come
**media mobile esponenziale** di quello delle query:

$$
\theta_k \;\leftarrow\; m\, \theta_k + (1 - m)\, \theta_q,
$$

dove $m \in [0, 1)$ è il coefficiente di momento, vicinissimo a uno (nel paper
$m = 0{,}999$) e $\theta_q$ sono i parametri aggiornati dal gradiente. Con $m$
piccolo il metodo peggiora nettamente, e senza momento non converge affatto: è
proprio la lentezza a rendere confrontabili chiavi entrate in coda centinaia di
passi prima di quelle con cui vengono confrontate (con $K = 65\,536$ e
mini-batch da $256$, la coda copre esattamente $256$ passi di addestramento).

`````

La copia lenta, quella che si aggiorna di un millesimo alla volta (si chiama
**media mobile**) e che fa da riferimento senza mai prendere punteggio, è la
stessa costruzione che il libro incontra nel {doc}`capitolo sui world
model </WorldModels/overview>`. Qui tiene coerente un dizionario di negativi; fra poco servirà a farne
del tutto a meno.

## Toglierli del tutto

Fin qui la logica sembrava inaggirabile. Se non c'è nessuno da cui distinguersi,
che cosa impedisce al modello di descrivere *tutte* le immagini allo stesso
identico modo? Sarebbe la risposta perfetta secondo il punteggio: due viste
descritte in modo identico non lasciano niente da correggere, quindi il modello
non ha nessun motivo di cercare oltre. Quella risposta vuota, che vince il gioco
senza aver guardato niente, si chiama **collasso**, e i negativi esistono
apposta per renderla impossibile. Nel 2020 BYOL
{cite}`grill2020bootstrap` fa la cosa che secondo quel ragionamento non dovrebbe
funzionare: elimina i negativi, tiene due reti e chiede a una di predire
l'uscita dell'altra. E non collassa.

`````{tab} Elementare

Ci sono un allievo e un insegnante. Guardano la stessa foto, ma da due ritagli
diversi. L'insegnante compila la sua scheda; l'allievo, con davanti soltanto il
proprio ritaglio, deve **indovinare che cosa ha scritto l'insegnante**, e il
punteggio misura quanto le due schede combaciano. Nient'altro in gioco: nessuna
foto rivale da respingere, nessuna penalità, solo «avvicinati a quello che dice
lui».

Se i due fossero alla pari la scappatoia si vedrebbe subito: intendersi su una
riga sola, «un'immagine», da scrivere su qualunque cosa capiti sotto gli occhi.
Schede identiche, punteggio pieno, e nessuno dei due che abbia mai guardato
davvero. È il collasso di cui si diceva, e a impedirlo qui non c'è nessuna forza
che allontani: ci sono due dissimmetrie.

La prima riguarda l'insegnante: non è una seconda rete assunta a parte, è una
**copia lenta dell'allievo**, i suoi criteri rimescolati poco alla volta con
quelli dell'allievo di oggi. Verrebbe da chiedersi che cosa si possa mai
imparare da una copia di sé stessi, e la risposta è che l'insegnante non sa di
più: sa **un'altra cosa**, perché sta guardando l'altro ritaglio. Il sapere non
arriva da lui, arriva dal confronto fra due sguardi diversi sulla stessa scena;
lui serve a tenere fermo il metro mentre l'allievo si muove. E siccome non
prende punteggio, non ha alcun motivo di semplificarsi la vita: non può mettersi
d'accordo con l'allievo su una risposta comoda per entrambi, perché non ha voce
in capitolo.

La seconda sta nel percorso della risposta: solo l'allievo ha una **testa di
predizione**, un passaggio in più con cui rielaborare la propria scheda prima
del confronto. Non gli si chiede di scrivere la scheda dell'insegnante, gli si
chiede di scriverne una **da cui** quella dell'insegnante si possa ricavare. È
una richiesta più debole, come chiedere l'indirizzo di casa invece del percorso
esatto per arrivarci: chi dà l'indirizzo ha detto abbastanza, e resta libero di
averci pensato per una strada tutta sua. Le due schede possono così restare
diverse senza che nessuno venga penalizzato, ed è proprio quella libertà a
togliere alla risposta vuota il suo vantaggio.

Che tanto bastasse ha sorpreso tutti, autori compresi: il fatto è solido e
riproducibile, la spiegazione è arrivata dopo, un pezzo alla volta.

`````

`````{tab} Superiore

BYOL mantiene due reti. Quella **online**, di parametri $\theta$, è fatta di
encoder, proiettore e **testa di predizione** $q_\theta$; quella **target**, di
parametri $\xi$, ha solo encoder e proiettore. Date due viste $\mathbf{v}$ e
$\mathbf{v}'$ della
stessa immagine, si minimizza l'errore quadratico fra la predizione della rete
online e la proiezione della rete target, entrambe normalizzate:

$$
\mathcal{L}_{\theta,\xi} =
\left\lVert
\frac{q_\theta(\mathbf{z}_\theta)}{\lVert q_\theta(\mathbf{z}_\theta) \rVert_2}
- \frac{\mathbf{z}'_\xi}{\lVert \mathbf{z}'_\xi \rVert_2}
\right\rVert_2^2
= 2 - 2 \cdot
\frac{\langle q_\theta(\mathbf{z}_\theta),\, \mathbf{z}'_\xi \rangle}
{\lVert q_\theta(\mathbf{z}_\theta) \rVert_2 \cdot \lVert \mathbf{z}'_\xi \rVert_2},
$$

dove $\mathbf{z}_\theta$ è la proiezione della vista $\mathbf{v}$ nella rete
online, $\mathbf{z}'_\xi$
quella della vista $\mathbf{v}'$ nella rete target e $q_\theta$ la testa di predizione;
la perdita si simmetrizza scambiando le due viste. Il gradiente scende **solo**
su $\theta$, mentre i parametri target seguono la solita media mobile,
$\xi \leftarrow m\, \xi + (1-m)\, \theta$, con $m$ inizializzato a $0{,}996$ e
portato verso uno durante l'addestramento.

Il punto è che la soluzione costante, pur essendo un minimo della perdita,
empiricamente non viene mai raggiunta, e a tenerne lontana la dinamica sono, per
quanto mostrano le analisi teoriche disponibili (condotte su modelli lineari
semplificati {cite}`tian2021understanding`), due asimmetrie. La prima: il ramo
target non riceve gradiente (**stop-gradient**), non può «accordarsi» con
l'altro e si limita a inseguirlo in ritardo. La seconda: la testa $q_\theta$ è
presente da un lato solo, quindi l'obiettivo effettivo dell'encoder online non è
produrre $\mathbf{z}'_\xi$, è produrre qualcosa da cui $q_\theta$ *possa
predire* $\mathbf{z}'_\xi$, che è un vincolo più debole. Un lavoro successivo, SimSiam
{cite}`chen2021exploring`, ha isolato il pezzo indispensabile, lo stop-gradient,
mostrando che la media mobile è utile ma non necessaria; e una prima spiegazione
molto discussa, che attribuiva l'anti-collasso alla batch normalization
(statistiche calcolate sul batch, quindi un contrasto implicito fra immagini), è
stata smentita dagli autori stessi di BYOL, riaddestrandolo con una
normalizzazione che del batch non sa nulla {cite}`richemond2020byol`. Il
meccanismo, per quanto se ne è capito, è dinamico: non una forza repulsiva, ma
una traiettoria di ottimizzazione che, nei fatti, non passa per il punto
degenere; una dimostrazione per il caso generale ancora non c'è.

`````

## Insegnare a sé stessi

C'è un terzo modo di formulare la stessa idea, e cambia quello che le due reti
si scambiano: non più una scheda di numeri da far somigliare, ma una
**ripartizione di fiducia** fra molte caselle, da riprodurre com'è (in termini
tecnici, una *distribuzione di probabilità*). È lo schema della
**distillazione**, cioè un modello che impara imitando le risposte di un altro
invece delle etichette vere, con la
particolarità che l'insegnante non è un modello più grande già addestrato, ma di
nuovo la copia lenta dello studente. Da qui il nome, DINO
{cite}`caron2021emerging`, contrazione di *self-distillation with no labels*.

`````{tab} Elementare

Cambia la forma del compito. L'insegnante non compila più una scheda libera:
sceglie fra un elenco fisso di caselle (sessantacinquemila, nel lavoro
originale: il numero è una scelta di progetto, e conviene che sia grande
perché il modello possa fare distinzioni fini) e
distribuisce la sua fiducia fra quelle («70% la casella 4012, 20% la 891, il
resto sparso»). Le caselle non significano niente in partenza, nessuno ha detto
che cosa sono: è il modello a decidere, addestrandosi, che cosa finisce dove.
L'allievo vede un ritaglio diverso e deve produrre la stessa ripartizione.

Restano due modi di barare, e a ciascuno corrisponde una contromisura. Vediamoli
con tre caselle invece di sessantacinquemila, che il conto si segue a mente.

Il primo modo è mettere sempre tutto nella stessa casella: se l'insegnante
risponde «100% la prima» su qualunque immagine, indovinare è banale e nessuno
dei due ha guardato niente. La contromisura è tenere il conto di quanto
l'insegnante ha usato ciascuna casella finora e **togliere quella media** prima
di leggere la risposta. Se la prima casella se la gioca sempre lui, il conto
medio la penalizza: dove diceva «$0{,}9$, $0{,}05$, $0{,}05$», tolta la media
$(0{,}9,\ 0{,}05,\ 0{,}05)$ resta $(0,\ 0,\ 0)$, cioè nessuna informazione, e
quella strada smette di pagare.

Il secondo modo è l'opposto, e nasce proprio dalla cura: spalmare la fiducia in
parti uguali, «33% ciascuna», che è un altro modo di non dire niente. La
contromisura è rendere la risposta dell'insegnante **più decisa**, allargando le
differenze fra le caselle prima di passarla all'allievo: dove lui aveva
$(0{,}40,\ 0{,}35,\ 0{,}25)$, quello che arriva all'allievo è più vicino a
$(0{,}49,\ 0{,}35,\ 0{,}15)$, con la prima casella salita, l'ultima scesa e il
distacco fra le due più che raddoppiato. Le due correzioni tirano in direzioni
opposte e si tengono a vicenda.

E c'è un fenomeno che vale la pena raccontare, il più sorprendente di questa
storia. Una famiglia di reti che il libro incontrerà nel capitolo sui
Transformer lavora dividendo l'immagine in tessere e decidendo, a ogni
passaggio, quanto guardare ciascuna: quella decisione è un numero per tessera,
e siccome è un numero si può colorare, ottenendo una specie di mappa di calore
di dove la rete stava guardando. Se si disegna quella mappa per una rete
addestrata con questo gioco, ci si vedono comparire i **contorni degli
oggetti**: la sagoma del cane staccata dallo sfondo, con una precisione che
nessuno ha chiesto. Nel materiale di addestramento non c'era una sola immagine
ritagliata a mano, e il perché è comprensibile: fra un ritaglio e l'altro
l'unica cosa che resta uguale è il soggetto, mai lo sfondo, quindi al gioco
conviene imparare a isolarlo.

`````

`````{tab} Superiore

Lo studente $g_{\theta_s}$ e l'insegnante $g_{\theta_t}$ hanno la stessa
architettura (un ViT {cite}`dosovitskiy2021image`, oppure una CNN) e producono
ciascuno un vettore di dimensione $K$ (nel lavoro originale $K = 65\,536$),
trasformato in distribuzione da una softmax con temperatura. La perdita è la
cross-entropia fra le due distribuzioni:

$$
\min_{\theta_s} \; H\big(P_t(\mathbf{x}), P_s(\mathbf{x}')\big),
\qquad
H(a, b) = -\sum_{i=1}^{K} a^{(i)} \log b^{(i)},
$$

dove $P_t$ e $P_s$ sono le distribuzioni prodotte da insegnante e studente su
due viste diverse $\mathbf{x}$ e $\mathbf{x}'$ della stessa immagine. I
parametri $\theta_t$ seguono la solita media mobile di $\theta_s$, con stop-gradient sul ramo
dell'insegnante. Le viste seguono la strategia **multi-crop**: due ritagli
globali (oltre metà dell'area) a entrambe le reti e alcuni ritagli locali
piccoli solo allo studente, che deve dunque predire dalla parte il tutto.

L'anti-collasso è affidato a due operazioni sull'uscita dell'insegnante che si
oppongono l'una all'altra. Il **centering** sottrae un vettore $\mathbf{c}$,
aggiornato come media mobile della media del batch,

$$
\mathbf{c} \;\leftarrow\; m\, \mathbf{c} + (1 - m)\,
\frac{1}{N}\sum_{i=1}^{N} g_{\theta_t}(\mathbf{x}_i),
$$

dove $N$ è, come sopra, la dimensione del batch e $m$ un coefficiente di media
mobile: il
centro insegue la risposta media dell'insegnante e impedisce che una coordinata
domini per tutte le immagini. Da solo, però, spinge verso la distribuzione
uniforme. Lo **sharpening** fa l'opposto: la temperatura dell'insegnante è
tenuta più bassa di quella dello studente ($0{,}04$–$0{,}07$ contro $0{,}1$), il
che rende $P_t$ più piccata e scoraggia l'uniformità. L'equilibrio fra le due
tiene il sistema lontano da entrambe le forme di collasso.

Il fenomeno più notevole non è nella perdita ma in ciò che si osserva dopo: le
mappe di auto-attenzione del token di classe, nell'ultimo strato di un ViT
addestrato così, si concentrano sui contorni degli oggetti, al punto che
soglializzandole si ottengono maschere di segmentazione grossolane ma sensate
{cite}`caron2021emerging`. La struttura non emerge altrettanto nettamente in un
ViT addestrato con le etichette, il che suggerisce che a produrla sia il
pretesto: l'unica cosa stabile fra un ritaglio e l'altro è il soggetto, non lo
sfondo, e l'obiettivo premia le rappresentazioni che lo isolano.

`````

## Nascondere tre quarti dell'immagine

L'altra grande famiglia non chiede al modello di riconoscere niente: gli chiede
di **ricostruire**. È il gioco della parola coperta, quello con cui si
pre-addestrano i modelli di linguaggio, trasportato sui pixel: la stessa idea
che nell'audio muove wav2vec 2.0 e HuBERT. Sulle immagini l'operazione è stata a
lungo deludente, e il MAE (*masked autoencoder*, cioè una rete che impara a
rimettere a posto quello che le si è coperto) {cite}`he2022masked` ha mostrato
che il problema era la dose.

`````{tab} Elementare

Se copro una parola su sette in una pagina, per indovinarle serve conoscere bene
la lingua. Se copro un pixel su sette in una fotografia, per indovinarli non
serve sapere niente: basta guardare i pixel intorno e fare una media, perché un
pixel somiglia moltissimo ai suoi vicini. Le immagini sono **ridondanti** nello
spazio in un modo in cui il testo non lo è: ogni parola porta informazione sua,
ogni pixel ripete in gran parte quella del pixel accanto.

Ecco perché il gioco funziona solo se si esagera. La fotografia si taglia prima
in tessere quadrate, come un mosaico, e poi se ne coprono tre su quattro: i
buchi diventano così grandi che nessuna media dei vicini li riempie, e per
indovinare che cosa c'è sotto un rettangolo enorme bisogna aver capito la scena
(«è un cane, quindi lì c'è la zampa che continua»). Il compito smette di essere
un esercizio di ritocco e diventa un esercizio di comprensione.

Il lavoro è diviso fra due pezzi, ed è qui il regalo. Il primo, quello grosso,
guarda **soltanto le tessere rimaste scoperte**, cioè un quarto del totale, e
si costruisce l'idea della scena; il secondo, molto più piccolo, prende quella
idea e disegna quello che c'era sotto le tessere coperte. Al pezzo grosso, che
è quello caro, tocca dunque un quarto del materiale, e finito il gioco è
l'unico che si tiene: il piccolo si butta via, perché serviva solo a definire
il compito. Nascondere tanto rende l'esercizio più difficile *e* più
economico, cosa che quasi mai capita.

`````

`````{tab} Superiore

L'immagine è divisa in patch come in un ViT {cite}`dosovitskiy2021image` e se ne
maschera una frazione molto alta, campionata uniformemente a caso (nel lavoro
originale il 75%). L'architettura è deliberatamente **asimmetrica**: l'encoder,
grande, riceve in ingresso *soltanto* le patch visibili, senza alcun segnaposto
per quelle mancanti; il decoder, molto più leggero, riceve la sequenza completa,
cioè le rappresentazioni delle patch visibili più un token appreso ripetuto in
ogni posizione mancante, e ricostruisce i pixel delle sole patch mascherate con
perdita quadratica. Finito il pretraining il decoder si getta via: serviva a
definire il compito.

Le due proprietà si sostengono a vicenda. La ridondanza spaziale del segnale
implica che con una frazione mascherata bassa il compito sia risolvibile per
estrapolazione locale, senza alcuna rappresentazione semantica; per lo stesso
motivo il testo, discreto e denso di informazione, si accontenta del 15% di BERT
{cite}`devlin2019bert`, e il parlato sta nel mezzo (wav2vec 2.0 maschera circa
la metà dei tratti). E poiché l'encoder elabora solo il 25% dei token, il costo
del passaggio in avanti scende **all'incirca in proporzione**, e appena di più.
Vale la pena essere precisi, perché il quadratico dell'attenzione fa spesso dire
più di quanto sia vero: in un blocco Transformer quasi tutte le moltiplicazioni
(proiezioni $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$, proiezione d'uscita, MLP) sono **lineari** in $N$, e
solo il prodotto $N \times N$ fra query e chiavi è quadratico. È quest'ultimo, e
soltanto lui, a scendere a un sedicesimo passando da $N$ a $N/4$; ma alle taglie
in gioco pesa poco. Contando i FLOP di un blocco come $24Nd^2$ (parte lineare)
più $4N^2d$ (parte quadratica), per un ViT-B/16 con $N = 196$ patch e
$d = 768$ il termine quadratico è circa il 4% del totale, per un ViT-L
($d = 1024$) il 3%: passando da $N = 196$ a $N = 49$ il blocco scende al 24%
del costo iniziale, cioè poco meno di un quarto, non a un sedicesimo. Gli autori
misurano un
pretraining complessivamente tre o più volte più rapido a parità di
architettura, e il fattore è minore di quattro per una ragione precisa: il
numero misurato è il tempo di addestramento nel suo complesso, e il decoder, che
la sequenza la riceve completa, dal mascheramento non guadagna nulla. Resta il
punto: il compito diventa più difficile e insieme più economico.

La differenza di fondo rispetto ai metodi contrastivi è **dove** finisce la
difficoltà. Là stava nelle trasformazioni scelte a mano, cioè nelle invarianze
imposte dal progettista; qui in una sola manopola, la frazione mascherata, e
nessuna augmentation artigianale è necessaria (il MAE usa poco più del ritaglio
casuale). Il prezzo è che la perdita vive nello spazio dei pixel, e obbliga il
modello a spendere capacità anche su dettagli imprevedibili e irrilevanti: è
l'obiezione che il capitolo sui world model porterà alle architetture
generative.

`````

## Come si capisce se ha funzionato

Un modello addestrato così non classifica niente: di ogni immagine dà soltanto
il suo riassunto interno, la lista di numeri di cui si diceva all'inizio. Come
si misura se quei riassunti sono buoni? Si potrebbe rifinirlo, cioè lasciarlo
imparare ancora un po’ su un compito vero con le etichette, e guardare quanto ci
prende alla fine. Ma quel numero non basta, perché mescola due cose diverse:
quanto era buono il riassunto di partenza e quanto è andata bene la rifinitura.

`````{tab} Elementare

Lo strumento standard si chiama **sondaggio lineare**, ed è un esame con le
mani legate. Si prende l'encoder e lo si blocca: da qui in poi non impara più
nulla. Poi gli si affianca un giudice fatto apposta debole: un classificatore
così semplice che da solo non saprebbe riconoscere niente, perché tutto quello
che sa fare è tracciare una linea dritta fra i riassunti che l'encoder produce
(i gatti di qua, i cani di là), aiutandosi con le etichette di un piccolo
dataset di prova. Se un giudice così sprovveduto supera l'esame, il merito non
può essere suo: vuol dire che nei riassunti dell'encoder gatti e cani erano
**già** separati. La debolezza del giudice è la garanzia dell'esame.

C'è però un limite: l'esame promuove solo ciò che si separa con una linea
dritta. Un encoder potrebbe aver capito tutto e averlo scritto in una forma più
contorta, che la linea dritta non sa leggere; l'esame lo boccerebbe lo stesso.
Per questo nessun voto, da solo, chiude la questione.

`````

`````{tab} Superiore

Lo strumento standard è il **sondaggio lineare** (*linear probing*): si congela
completamente l'encoder, si buttano via testa di proiezione e decoder, e sopra
le rappresentazioni si addestra un solo strato lineare con softmax, usando le
etichette di un dataset di valutazione. Nient'altro: nessun gradiente entra
nell'encoder, nessuna non linearità viene aggiunta. Se basta un iperpiano (una
retta, quando le dimensioni sono due) a separare le classi, allora
l'informazione era **già dentro** la rappresentazione, e in una forma
direttamente utilizzabile, perché il classificatore lineare non ha nessuna
capacità di costruirla da sé. La debolezza dello strumento è il suo pregio.

Va detto con altrettanta chiarezza che il sondaggio misura una cosa precisa, la
**separabilità lineare**, non tutta l'informazione presente. Un encoder può
codificare il contenuto di un'immagine in una forma che va letta con una
funzione non lineare, e la sonda lo penalizzerebbe: infatti i metodi
generativi come il MAE tendono a fare peggio sotto sonda lineare che sotto
fine-tuning completo, mentre per i contrastivi le due misure sono più allineate.
Lo strumento incorpora un'ipotesi su come la rappresentazione verrà usata, e per
questo gli si affianca spesso una sonda ancora più spartana, la classificazione
a $k$ vicini più prossimi, che non addestra proprio niente.

`````

La prova più severa è però un'altra: **cambiare compito**. Rilevamento e
segmentazione, di cui questo capitolo si occupa a parte, non chiedono di dire
che cosa c'è nella foto, chiedono di dire *dove*: e per rispondere non basta un
riassunto che descriva bene l'immagine tutta insieme, ne serve uno che resti
preciso zona per zona, angolo per angolo. Può quindi succedere che un encoder
superi l'esame con la linea dritta sulla classificazione e poi, messo a fare da
base di un rilevatore, non regga: vuol dire che ha imparato un riassunto buono
per una domanda sola. È il criterio
che conta, perché è il motivo per cui addestriamo in anticipo: non risolvere il
pretesto, ma avere un punto di partenza utile per compiti che, mentre ci
addestravamo, non sapevamo nemmeno quali sarebbero stati.

## Dove si mette la difficoltà

La varietà dei nomi nasconde un'unica struttura. Chi mette a confronto (SimCLR,
MoCo), chi predice senza rivali (BYOL, DINO) e chi ricostruisce quello che ha
coperto (il MAE, e si chiamano **generativi** proprio perché il compito è
produrre di nuovo un pezzo di immagine) risolvono lo stesso problema:
fabbricare un compito la cui risposta è già nei dati e che sia abbastanza
difficile da non poter essere risolto per scorciatoia. Quello che li distingue è
**dove** mettono la difficoltà.

I contrastivi la mettono nelle **trasformazioni**: il compito è facile per
costruzione (riconoscere il gemello) e diventa difficile perché le due viste
sono state rese diverse apposta. Siamo noi a decidere quali differenze il
modello deve imparare a ignorare, e quella decisione è conoscenza nostra sul
problema, messa a mano dentro il compito. I generativi la mettono nella
**quantità di informazione tolta**: nessuna invarianza scelta da noi, una sola
manopola da girare, quanto si copre. Il prezzo è che il conto lo si paga sui
pixel, e ricostruire i pixel vuol dire dover indovinare anche il granello di
polvere e il riflesso, cioè spendere fatica su dettagli che a nessuno
interessano. I predittivi stanno in mezzo: da loro la difficoltà non sta né
nelle trasformazioni né nella quantità coperta, ma nel modo in cui le due reti
sono fatte diverse l'una dall'altra, ed è quella differenza a tenere il sistema
lontano dalla risposta vuota.

Tre posti, ma non sono tutti, e conviene dirlo subito. Ce n'è un quarto, e sta
dove nessuno dei tre guarda: la difficoltà si mette **dentro il riassunto**,
chiedendo che i numeri che lo compongono dicano ciascuno una cosa propria
invece di ripetersi a vicenda. Niente rivali da allontanare, e nessun
bisogno che le due reti siano fatte diverse: la condizione che tiene lontana la
risposta vuota è scritta direttamente nel punteggio. Il capitolo
sull'auto-supervisione presenta questa quarta famiglia insieme alle altre tre,
e poi rilegge tutte e quattro secondo una seconda domanda, che cosa impedisce
in ciascuna al modello di rispondere sempre la stessa cosa.

Da qui il libro prosegue in due direzioni che chiudono il cerchio. Nel capitolo
sui world model la JEPA porta la difficoltà in un posto ancora diverso: si
maschera come nel MAE, ma si predice la **rappresentazione** della parte
nascosta invece dei suoi pixel, e le augmentation artigianali spariscono del
tutto. Nel {doc}`capitolo su visione e linguaggio </VisioneLinguaggio/overview>` il gemello da ritrovare non è più
una seconda vista della stessa foto ma la sua **didascalia**: si mescolano sul
tavolo le immagini e le frasi, e si chiede di riappaiarle. È lo stesso gioco,
con lo stesso identico conto dietro, e cambia soltanto da dove viene il
segnale: non da una deformazione che abbiamo applicato noi, ma dal fatto che
qualcuno, pubblicando quell'immagine, ci ha scritto accanto che cosa c'era.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Le etichette le scrivono delle persone, costano, e non bastano mai; le foto
  senza etichetta sono infinite. L'idea è **inventare un gioco la cui risposta
  giusta la conosciamo per costruzione**, senza che nessuno debba scriverla, e
  tenere quello che il modello è stato costretto a capire per vincerlo.
- Il gioco più semplice: ritagliare da ogni foto due pezzi e chiedere, in mezzo
  a centinaia di ritagli mescolati, di **ritrovare il gemello**. Gli altri
  ritagli, i rivali, si chiamano *negativi*.
- Qui le deformazioni non servono più a non far imparare a memoria: **sono il
  compito**. Scegliendole diciamo al modello a che cosa deve essere
  indifferente, e ogni indizio che dimentichiamo di togliere diventa una
  **scorciatoia**: senza disturbare i colori, due ritagli della stessa foto si
  riconoscono dalla sola tinta media, e il modello vince senza aver capito
  niente.
- Avere tanti rivali è utile ma costa: si può tenerli in una **scatola-coda**,
  alimentata da una copia lenta di sé stessi perché le descrizioni vecchie e
  nuove restino confrontabili. Oppure toglierli del tutto, mettendo di fronte
  un allievo e un insegnante che è una copia lenta dell'allievo: sorprende che
  non collassi sulla risposta vuota, ma non collassa.
- L'altra grande famiglia non chiede di riconoscere, chiede di **ricostruire**:
  si copre tre quarti dell'immagine e si fa indovinare cosa c'era sotto. Tanto
  serve, perché un pixel somiglia troppo ai suoi vicini: con pochi buchi basta
  fare una media e non si impara nulla.
- Per capire se ha funzionato si usa un **esame con le mani legate**: si blocca
  il modello e gli si affianca un giudice così debole da non poter aggiungere
  niente di suo. Se passa l'esame, il merito è del modello. La prova più severa
  però è un'altra: cambiare compito.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Le etichette costano e non scalano; le immagini non etichettate sono
  abbondanti. L'apprendimento **auto-supervisionato** inventa un **pretesto** la
  cui risposta è ricavabile dai dati, e ne conserva l'encoder, non il compito.
- Nel metodo **contrastivo** {cite}`chen2020simple` due viste della stessa
  immagine devono avvicinarsi fra loro e allontanarsi da quelle delle altre
  (perdita **NT-Xent**, una InfoNCE con temperatura $\tau$ sulle $2N$ viste).
- Qui la **data augmentation non regolarizza: definisce il compito**. Scegliere
  le trasformazioni significa dire al modello a che cosa essere indifferente, e
  ogni indizio non rimosso diventa una **scorciatoia**: senza disturbo del
  colore, due ritagli della stessa foto si appaiano dal solo istogramma.
- I negativi costano batch enormi. **MoCo** {cite}`he2020momentum` li mette in
  una **coda** alimentata da un encoder aggiornato per **media mobile**, così
  restano numerosi e coerenti nel tempo; **BYOL** {cite}`grill2020bootstrap` li
  elimina e non collassa grazie all’**asimmetria** fra le due reti (testa di
  predizione da un lato, media mobile e stop-gradient dall'altro), un fatto
  robusto la cui spiegazione è arrivata dopo il risultato.
- **DINO** {cite}`caron2021emerging` distilla lo studente da una copia lenta di
  sé, con **centering** e **sharpening** che si bilanciano contro le due forme
  di collasso; nelle mappe di attenzione del ViT emergono i contorni degli
  oggetti, senza che nessuna segmentazione sia stata fornita.
- Il **MAE** {cite}`he2022masked` maschera circa il 75% delle patch (contro il
  15% di BERT: le immagini sono spazialmente **ridondanti**, e con pochi buchi
  il compito si risolve per interpolazione) e ricostruisce i pixel con un
  encoder che vede solo le patch visibili e un decoder leggero, poi buttato via.
- La valutazione canonica è il **sondaggio lineare** sull'encoder congelato: se
  una retta separa le classi, l'informazione era già nella rappresentazione.
  Misura la separabilità lineare, non tutto; la prova più severa è il
  trasferimento a rilevamento e segmentazione.
```

`````
