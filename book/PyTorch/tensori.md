# Tensori e autograd

Dentro PyTorch ogni cosa è un tensore: l'immagine da classificare, la frase
tradotta in numeri, e ognuno dei numeri che la rete si tiene dentro e che
impara col tempo, i suoi **pesi**. Il nome stesso della libreria lo porta
scritto: *Torch*, la torcia, era la libreria di partenza, e *Py* dice che
adesso si guida da Python. Accanto ai tensori vive il secondo protagonista,
più discreto ma decisivo: **autograd**, il meccanismo che osserva i calcoli
mentre avvengono e sa risalire alle derivate. Capire questi due oggetti
significa capire il motore su cui gira tutto il deep learning moderno.

## Che cos'è un tensore

Scalari, vettori e matrici li abbiamo già incontrati nel {doc}`capitolo di algebra
lineare </Matematica/overview>`: un numero solo, una fila di numeri, una tabella di numeri. Il tensore
è semplicemente il passo successivo, la stessa idea portata avanti finché si
vuole: si continua ad aggiungere direzioni lungo cui i numeri si estendono, e
ciascuna di quelle direzioni si chiama **asse**.

```{figure} ../figures/tensori-scala.svg
:name: fig-tensori-scala
:alt: "Quattro oggetti in fila, collegati da frecce, ciascuno con sotto il nome e la sua forma: un singolo quadrato (scalare, rank 0), una riga di quattro celle (vettore, rank 1, forma (4,)), una griglia 3 per 3 (matrice, rank 2, forma (3, 3)) e una pila di tre griglie 3 per 3 (tensore 3D, rank 3, forma (3, 3, 3))."
:width: 90%

La scala dei tensori. Ogni gradino aggiunge un asse: da un numero solo, a una
fila, a una griglia, a una pila di griglie.
```

`````{tab} Elementare

Un tensore è un contenitore di numeri con un certo numero di "assi", cioè di
direzioni lungo cui si estende ({numref}`fig-tensori-scala`):

- un **numero solo** (per esempio la temperatura, $23{,}5$) è un tensore a zero
  assi: uno *scalare*;
- una **fila di numeri** (i voti di uno studente) è un tensore a un asse: un
  *vettore*;
- una **tabella** di numeri (i pixel di una foto in scala di grigi) ha due
  assi: una *matrice*;
- una **pila di tabelle** (una foto a colori: una griglia per il rosso, una
  per il verde, una per il blu) ha tre assi.

Due parole per due cose, e sono inglesi perché così le troverai scritte nel
codice: il numero di assi si chiama **rank**, le lunghezze lungo ciascun asse
formano la **shape** (la "forma"). Una foto a colori $256 \times 256$ è, in
PyTorch, un tensore di shape $(3, 256, 256)$: rank 3, e i tre colori vengono
scritti **per primi**, prima delle due misure della griglia. È una convenzione,
non una legge di natura (altre librerie mettono i colori in fondo), e conviene
saperlo perché quando la forma non torna il primo sospetto è averla invertita.

`````

`````{tab} Superiore

Formalmente un tensore di rank $r$ è un array multidimensionale i cui elementi
si indicizzano con $r$ indici, $T_{i_1 i_2 \dots i_r}$. Lo caratterizzano due
attributi:

- il **rank** (o numero di assi), $r$;
- la **shape** $(n_1, n_2, \dots, n_r)$, la dimensione lungo ciascun asse.

Scalare, vettore e matrice sono i casi $r = 0, 1, 2$. Nel deep learning si
lavora costantemente con rank più alti: in PyTorch un *batch* di immagini RGB
è un tensore $(N, C, H, W)$ (numero di esempi, canali, altezza, larghezza)
quindi rank 4 (l'ordine *channels-first*, diverso dal $(N, H, W, C)$ di altre
librerie). A differenza dell'oggetto matematico "tensore" (che porta con sé
regole di trasformazione tra sistemi di coordinate), qui il termine indica
soltanto la struttura dati: un array $n$-dimensionale con un `dtype` omogeneo
(`float32`, `int64`, …).

`````

## Creare tensori e farci i conti

Un tensore si può fabbricare in tre modi: scrivendo i numeri a mano in una
lista Python, chiedendo a PyTorch di riempirlo lui (`zeros`, `ones`, `randn`, fra un
attimo), oppure partendo da un array di **NumPy**, la libreria di calcolo
numerico del {doc}`capitolo su Python </Python/overview>`. Con
quest'ultima PyTorch va d'accordo così bene che i due si passano i dati senza
nemmeno ricopiarli.

```python
import torch

s = torch.tensor(3.14)                     # scalare, rank 0
v = torch.tensor([1.0, 2.0, 3.0])          # vettore, rank 1
M = torch.tensor([[1., 2.], [3., 4.]])     # matrice, rank 2

M.shape        # torch.Size([2, 2])
M.dtype        # torch.float32

torch.zeros(2, 3)        # matrice 2x3 di zeri
torch.ones(5)            # vettore di uno
torch.randn(3, 3)        # numeri a caso, quasi tutti fra -2 e 2, centrati sullo zero
torch.arange(0, 10, 2)   # da 0 a 10 di 2 in 2, 10 escluso: tensor([0, 2, 4, 6, 8])
```

Le ultime quattro righe fabbricano tensori pieni senza che si debba scrivere i
numeri a mano, e la terza, quella che li sorteggia, non è un capriccio: una
rete comincia la sua vita con dei numeri a caso dentro, e riempire un tensore
di zeri o di numeri sorteggiati è il gesto con cui la si mette al mondo.

Sui tensori valgono le operazioni dell'algebra lineare che già conosciamo
(somma, prodotto per scalare, prodotto fra matrici) e ogni riga viene eseguita
nell'istante in cui la scrivi, esattamente come in NumPy:

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([10., 20., 30.])

a + b            # tensor([11., 22., 33.])
a * b            # prodotto elemento per elemento
a.sum()          # tensor(6.)  -> calcolato SUBITO
a @ b            # prodotto scalare: 1·10 + 2·20 + 3·30 = tensor(140.)
a.reshape(3, 1)  # nuova forma: gli stessi numeri in colonna, 3x1
```

`````{tab} Elementare

Tre cose da notare. Primo: il risultato compare all'istante, con i numeri già
dentro (puoi controllare ogni passaggio come su una calcolatrice, senza
"avviare" nulla). Secondo: il simbolo `@` è quello del prodotto fra matrici, e
su due semplici file di numeri come queste fa la cosa più elementare che quel
prodotto sappia fare, cioè moltiplicarle a due a due e sommare tutto
($1 \cdot 10 + 2 \cdot 20 + 3 \cdot 30 = 140$); è il **prodotto scalare**, e
il conto è scritto per esteso nel commento apposta perché lo si possa
rifare. Terzo: il **broadcasting**, che conosciamo già da NumPy,
funziona identico. Se scrivi `a + 5`, PyTorch capisce da solo che vuoi sommare
$5$ a *ciascuno* dei tre numeri, come un insegnante che alza di un punto tutti
i voti della classe senza bisogno di scrivere la regola tre volte: il
risultato è `tensor([6., 7., 8.])`.

`````

`````{tab} Superiore

L'interoperabilità con NumPy è alla pari: `torch.from_numpy(arr)` e
`t.numpy()` convertono nei due sensi **condividendo la memoria** (nessuna
copia: modificare l'uno modifica l'altro). Le regole di broadcasting sono le
stesse di NumPy, gli assi si allineano da destra e le dimensioni compatibili
(uguali, o pari a 1) si espandono virtualmente: una matrice $(3, 4)$ più un
vettore $(4,)$ produce una $(3, 4)$. Il prodotto matriciale è `@` (ovvero
`torch.matmul`), con la stessa semantica di NumPy anche sui vettori rank-1:
`a @ b` tra due vettori è direttamente il prodotto scalare, senza bisogno di
reshape. A proposito di `reshape`: restituisce una **vista** (stessa memoria,
solo un modo diverso di leggerla) quando la disposizione dei dati lo permette,
e altrimenti copia in silenzio; `t.view(...)` la vista la pretende, e solleva
un errore invece di copiare quando non può darla. La condizione non è la
contiguità, come si legge spesso: è che la nuova forma sia compatibile con gli
**stride** esistenti, cioè con i passi con cui si cammina lungo ciascun asse.
Spezzare o fondere assi già contigui fra loro riesce anche su un tensore non
contiguo: se `x` è $(8,4)$ e `y = x[::2]` (quindi $(4,4)$ con stride $(8,1)$,
non contigua), `y.view(4, 2, 2)` passa e condivide la memoria, mentre
`y.view(16)` solleva. Il messaggio d'errore lo dice esattamente, e vale la
pena leggerlo alla lettera: *view size is not compatible with input tensor's
size and stride (at least one dimension spans across two contiguous
subspaces). Use `.reshape(...)` instead.* Molte operazioni
esistono in variante *in-place* col suffisso underscore (`t.add_(1)`,
`t.zero_()`): risparmiano memoria ma, come vedremo, vanno evitate sui tensori
tracciati da autograd.

`````

## Lo stesso codice su CPU e GPU

Ogni tensore vive su un *device*. Di default è la CPU; spostarlo su una GPU
(se c'è) è una chiamata a `.to()`, e tutto il resto del codice non cambia.

```python
# se una scheda grafica c'è usa quella, altrimenti la CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

M = torch.randn(1000, 1000)   # una matrice grande: un milione di numeri
M = M.to(device)              # trasloca dove dice `device`
prodotto = M @ M              # calcolato dove vive il tensore
print(prodotto.shape, prodotto.device)
# su una macchina senza scheda grafica: torch.Size([1000, 1000]) cpu
```

La prima riga è un modo compatto di scrivere una scelta, e si legge da
sinistra: prendi `"cuda"` *se* c'è una scheda utilizzabile, *altrimenti*
`"cpu"`. È il gesto con cui comincia quasi ogni programma PyTorch, e da qui in
avanti lo ritroveremo identico. L'ultima riga stampa dove il risultato è
finito, e su una macchina senza scheda grafica stampa `cpu`: non è un guasto,
è la scelta della prima riga che si vede all'opera. Su un computer con una
scheda NVIDIA la stessa identica riga stamperebbe `cuda:0`.

La matrice è di mille per mille perché su matrici piccole la differenza fra i
due dispositivi non si vede: il vantaggio della scheda grafica comincia quando
i conti da fare sono tanti.

`````{tab} Elementare

La regola è una sola: **i conti avvengono dove stanno i numeri**. Se il
tensore è sulla CPU, calcola la CPU; se lo sposti sulla scheda grafica,
calcola lei, e per le moltiplicazioni tra matrici grandi può essere decine o
centinaia di volte più veloce, perché una GPU è nata per fare migliaia di
piccoli conti in parallelo (in origine, i pixel dei videogiochi). L'unica
attenzione: due tensori possono lavorare insieme solo se stanno sullo stesso
dispositivo (non puoi sommare un numero che sta in cucina con uno che sta in
garage senza prima spostarne uno).

`````

`````{tab} Superiore

Il pattern idiomatico è definire `device` una volta all'inizio e spostare
modello e batch con `.to(device)`; su Apple Silicon il device si chiama
`"mps"`. Operazioni tra tensori su device diversi sollevano un errore
esplicito (nessun trasferimento implicito, che nasconderebbe costi: il
passaggio CPU↔GPU attraversa il bus PCIe ed è lento rispetto al calcolo). Le
stesse moltiplicazioni tra matrici che scriviamo qui sono, sull'hardware,
migliaia di prodotti scalari eseguiti in parallelo dai kernel CUDA/cuDNN: è il
motivo per cui le GPU (nate per la grafica) sono diventate lo strumento del
deep learning. Il codice resta identico; cambia solo la velocità.

`````

## Autograd: la derivata calcolata da sola

Addestrare una rete significa girare le sue tante manopole interne (i
**parametri**, cioè i pesi di cui si diceva in apertura) finché l'errore che
commette non diventa piccolo. Quell'errore ha un nome che ricorrerà per tutto
il libro: si chiama **loss**, la perdita.

Per sapere da che parte girare ciascuna manopola serve il **gradiente**. È di
nuovo una derivata, ma stavolta la cosa che si sposta di un soffio non è il
dato che entra: è la manopola. Il gradiente dice, per ogni singolo peso, che
cosa succede alla loss se quel peso lo si alza appena: sale o scende, e di
quanto. Chi vuole meno errore gira ogni manopola dalla parte in cui il numero
scende. Farlo a mano per una rete con milioni di pesi è impensabile, e qui
entra la **differenziazione automatica** (*autodiff*), il vero cuore di
PyTorch.

```{figure} ../figures/extra-backpropagation-spiegata.svg
:name: fig-autograd-due-passate
:alt: "Una rete con tre neuroni in ingresso, due nello strato nascosto e uno in uscita, percorsa in due sensi. Sotto, due frecce opposte: quella verso destra è etichettata «forward: dati verso previsione», quella verso sinistra «backward: errore verso correzioni». Accanto al neurone d'uscita, dei raggi segnalano l'errore commesso."
:width: 96%

I due sensi di marcia: all'andata i dati attraversano la rete e producono una
previsione, al ritorno l'errore risale la stessa strada e diventa una
correzione per ogni peso incontrato.
```

Nella {numref}`fig-autograd-due-passate` c'è però una cosa che il disegno non
può mostrare, e che spiega un comportamento di PyTorch altrimenti sorprendente:
per poter tornare indietro, l'andata deve **ricordare**. Una rete è
fatta di **strati**, cioè di stazioni in fila: i dati entrano dalla prima,
ognuna li trasforma un po’ e passa il risultato alla successiva, finché
dall'ultima esce la risposta. Il viaggio di andata si chiama **passata in
avanti** (in inglese *forward pass*).

Ora, quel viaggio non si limita a calcolare la risposta: a ogni stazione
appunta anche il risultato di passaggio, perché al ritorno servirà. Quindi più
stazioni ha la rete, cioè più è **profonda**, più appunti restano in memoria
durante l'andata.

Ecco come si accende tutto questo, sull'esempio più piccolo possibile: un solo
numero al posto di una rete, così si può controllare il risultato a mente. La
prima riga chiede a PyTorch di tenere d'occhio `x`; le due dopo fanno un conto
e chiedono la strada del ritorno.

```python
x = torch.tensor(3.0, requires_grad=True)   # "osserva questo tensore"

y = x**2 + 2*x            # y = x² + 2x: il grafo si costruisce da solo
y.backward()              # passata all'indietro

x.grad                    # la derivata di y in x=3  ->  tensor(8.)
```

`````{tab} Elementare

Immagina un **registratore** acceso mentre fai i conti: annota ogni singola
operazione eseguita a partire dai tensori "osservati". Alla fine gli chiedi:
"com'è cambiato il risultato al variare di questo ingresso?" e lui,
riavvolgendo il nastro all'indietro, ti risponde con la derivata esatta.

Il riavvolgimento non è una magia, ed è la parte che vale la pena capire.
Nessun conto complicato è complicato *tutto insieme*: è una catena di gesti
elementari, un'elevazione al quadrato, una moltiplicazione, una somma. Di
ciascuno di questi la derivata è nota una volta per tutte, come una tabellina.
Quindi chi ha annotato la catena può ripercorrerla al contrario, applicare a
ogni anello la sua tabellina e comporre i risultati: alla fine del nastro ha in
mano la derivata dell'intera catena, senza averla mai scritta.

È ciò che fa autograd: dichiarando `requires_grad=True` accendi il
registratore su `x`; ogni calcolo successivo viene annotato; e la chiamata
`y.backward()` riavvolge il nastro, depositando la derivata in `x.grad`. Nel
codice sopra $y = x^2 + 2x$, e il numero che esce è $8$: lo stesso che nella
pagina di apertura del capitolo avevamo controllato a mano spostando $x$ da
$3$ a $3{,}01$. Non abbiamo scritto nessuna formula di derivata: l'ha
ricostruita la libreria.

`````

`````{tab} Superiore

Autograd implementa la differenziazione automatica in modalità *reverse*
(*reverse-mode autodiff*). Ogni operazione su tensori con
`requires_grad=True` aggiunge un nodo al grafo dinamico delle computazioni;
`y.backward()` percorre il grafo a ritroso applicando la **regola della
catena** vista nel capitolo di analisi e ottimizzazione:

$$
\frac{\partial \mathcal{L}}{\partial \theta}
= \sum_{j=1}^{m} \frac{\partial \mathcal{L}}{\partial z_j}\,
  \frac{\partial z_j}{\partial \theta},
\qquad \text{cioè} \qquad
\nabla_{\theta} \mathcal{L}
= \left( \frac{\partial \mathbf{z}}{\partial \theta} \right)^{\!\top}
  \nabla_{\mathbf{z}} \mathcal{L},
$$

dove $\mathbf{z}$ raccoglie le $m$ quantità intermedie che dipendono da
$\theta$, $\partial \mathbf{z} / \partial \theta$ è la matrice delle loro
derivate (la **Jacobiana**) e $\nabla_{\mathbf{z}} \mathcal{L}$ è il gradiente
già calcolato a valle. Composta lungo tutto il grafo, questa regola è
precisamente l'algoritmo di **backpropagation** del capitolo precedente.

Nella riga qui sopra ci sono due cose che una catena a un solo cammino non
direbbe, e sono esattamente le due che contano nella pratica. La prima è la
**trasposta**: la modalità reverse non costruisce mai la Jacobiana, calcola
direttamente il prodotto fra la sua trasposta e il vettore che arriva da valle
(un *vector-Jacobian product*, uno per nodo), ed è da lì che viene il costo di
una sola passata, quale che sia il numero di parametri. Materializzare la
Jacobiana costerebbe una passata per ciascuna uscita. La seconda è la
**sommatoria**: un parametro che alimenta più rami riceve un contributo per
ramo, e i contributi si sommano. È la ragione strutturale per cui `.grad` è un
`+=` e non un `=`, e il punto in cui il grafo smette di essere una catena.

Quattro dettagli operativi che incontreremo di continuo. I gradienti si
**accumulano**: una `backward()` successiva, su un nuovo forward, somma in
`x.grad` invece di sovrascrivere, per questo il training loop azzera i
gradienti a ogni passo. Ripetere la *stessa* chiamata sullo stesso grafo,
invece, solleva un errore, e la ragione dice che cosa `backward()` faccia
davvero: percorrendolo **libera** i valori intermedi salvati durante l'andata,
quelli della {numref}`fig-autograd-due-passate`. Il grafo resta, gli appunti
per percorrerlo no. Chiederli in prestito è `retain_graph=True`, e serve tutte
le volte che da una sola passata in avanti partono due passate all'indietro
(due loss che pescano da un tronco comune, il generatore di una GAN che
alimenta due obiettivi); costa memoria, quindi non si mette per abitudine.
Secondo: il gradiente si deposita solo sulle **foglie** del grafo, i tensori
creati da noi con `requires_grad=True`, e i parametri di un modello lo sono
tutti. Su un tensore intermedio, cioè prodotto da un'operazione, `.grad`
resta `None` con tanto di avviso: per leggerlo a metà strada si chiama
`y.retain_grad()` prima del backward. Terzo: il blocco
`with torch.no_grad():` sospende la registrazione, indispensabile in
valutazione, quando i gradienti non servono e il grafo sarebbe solo memoria
sprecata. Infine `t.detach()` restituisce una vista del tensore staccata dal
grafo, e le operazioni in-place sui tensori tracciati vanno evitate perché
possono invalidare i valori salvati per la passata a ritroso.

`````

Resta da dire che cosa succede quando il registratore **non** serve, ed è il
caso più comune di tutti: il modello ha finito di imparare e lo si sta soltanto
usando. Gli si dà una foto, lui risponde, e nessuno ha intenzione di correggere
niente. Lì tutti quegli appunti sono peso morto, e si può dire in anticipo di
non prenderli: è il comando `torch.no_grad()`, che si scrive attorno al pezzo
di codice da cui non ci si aspetta nessuna correzione. Attenzione alla
sfumatura, perché è il punto in cui si sbaglia: `no_grad()` non svuota una
memoria già piena, impedisce che si riempia. Lo useremo a ogni valutazione, da
qui alla fine del capitolo.

Tensori e autograd sono i due oggetti su cui poggia tutto il resto del
capitolo: dalla prossima sezione non si farà che comporli.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un **tensore** è una scatola di numeri con un certo numero di assi: un
  numero solo, una fila, una tabella, una pila di tabelle. Il numero di assi si
  chiama **rank**, le lunghezze lungo gli assi sono la **shape**. Una foto a
  colori è una pila di tre tabelle, una per colore.
- Ogni riga di conti viene eseguita **subito**, con i numeri già dentro, e le
  regole sono quelle di NumPy, la libreria già vista nel {doc}`capitolo su Python </Python/overview>`:
  sommare un numero a tutta una fila si scrive una volta sola.
- Ogni tensore vive su un **dispositivo**, la CPU o la scheda grafica: i conti
  avvengono dove stanno i numeri, e il codice non cambia, cambia solo la
  velocità.
- **Autograd** è il registratore: `requires_grad=True` lo accende su un
  tensore, `.backward()` riavvolge il nastro e deposita la derivata. Nessuna
  formula scritta a mano.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un **tensore** generalizza scalari, vettori e matrici a un numero qualunque
  di assi: lo descrivono **rank** (numero di assi) e **shape** (forma). In
  PyTorch le immagini sono *channels-first*: $(N, C, H, W)$.
- L'esecuzione è immediata e le regole (broadcasting compreso) sono quelle di
  NumPy, con cui i tensori si scambiano dati senza copie.
- Ogni tensore vive su un **device** (`"cpu"`, `"cuda"`, `"mps"`): i conti
  avvengono dove stanno i numeri, e il codice non cambia.
- **Autograd** calcola i gradienti da solo: `requires_grad=True` accende il
  registratore, `.backward()` riavvolge il nastro; è la backpropagation, cioè
  la regola della catena applicata a ritroso sul grafo dei calcoli.
- `t.view(...)` pretende una vista e passa solo se la nuova forma è compatibile
  con gli **stride**, non solo sui tensori contigui; `reshape` copia quando non
  può fare altrimenti.
```
`````
