# Comprimere e ricostruire, e perché non basta

Un museo ha in magazzino più quadri di quanti ne stiano alle pareti, e un
archivista che li deve schedare. Per ogni quadro scrive una scheda molto più
piccola dell’originale, e la mette in un cassetto. Se le schede servono a
qualcosa lo si scopre chiamando un copista, dandogli una scheda e nient’altro,
e guardando se il quadro che ridipinge somiglia a quello di partenza. Se
somiglia, la scheda conteneva l’essenziale.

I due imparano insieme, ed è il punto: una scheda è buona rispetto a chi la
deve leggere, mai in assoluto. Se il copista sa già dipingere una
cornice dorata, l’archivista non ha bisogno di annotarla; se non lo sa, quella
riga sulla scheda va spesa. Nessuno dei due ha ricevuto istruzioni su che cosa
sia importante in un quadro: se lo sono divisi lavorando.

Questa macchina è già comparsa nella {doc}`sezione sui codec
neurali </Audio/codec-neurali>`, per comprimere il suono: è l’autoencoder, una
coppia di reti con la forma di una clessidra, larga alle due estremità e
strettissima in mezzo ({numref}`fig-autoencoder-clessidra`, che sta là).
L’encoder («chi codifica») riduce il dato a un **codice** $\mathbf{z}$, detto
anche latente; il decoder («chi decodifica») ricostruisce il dato dal solo
codice. Nel museo l’archivista è l’encoder, il copista il decoder e la scheda
il codice; la classe Python del capitolo si chiama `Clessidra`.

Il codice è un vettore $\mathbf{z} \in \mathbb{R}^L$, cioè un punto dello
spazio latente a $L$ dimensioni ($L = 8$ nell’esperimento sulle cifre). Fra
due codici c’è una distanza, e quindi ha senso parlare di codici vicini, del
punto a metà strada fra due codici, di un cammino dall’uno all’altro.

Qui interessa una domanda diversa da quella dell’audio: un autoencoder che sa
comprimere tutti i dati sa anche generarne di nuovi? In generale no, e la
ragione non sta in quanto è brava la rete. L’obiettivo non dice dove debbano
cadere i codici, quindi un codice scelto a caso non ha motivo di decodificarsi
in un dato plausibile.

## La strozzatura è il compito

Dell’autoencoder, qui, interessa più il codice che produce che quanto comprima.
La parte stretta in mezzo, la dimensione $L$ del codice, da cui deve passare
tutto, si chiama **strozzatura** (in inglese *bottleneck*).

`````{tab} Elementare

Due metà e una strozzatura in mezzo, come il collo di una clessidra.
L’archivista riduce il quadro a una scheda di pochi numeri, il copista da quella
scheda prova a ritirare fuori il quadro, e la pagella è una sola per tutti e
due: quanto la copia somiglia all’originale, un quadro alla volta. Una scheda di
due numeri si potrebbe segnare come un punto su un foglio, una di otto come un
punto in un posto che non si disegna ma si tratta allo stesso modo: le schede
stanno su una mappa, alcune vicine e altre lontane.

Quella strozzatura il museo la vuole. Se all’archivista fosse concesso scrivere
una scheda lunga quanto il quadro, la scriverebbe uguale al quadro, il copista
la ricopierebbe, e i due avrebbero imparato a fotocopiare. Dovendo stare in
poche righe, l’archivista è costretto a decidere che cosa conta, ed è quella
decisione a interessarci.

È il primo modo di impedire la fotocopia, ma non l’unico. Il museo può lasciare
la scheda lunga, anche più del quadro, e far pagare l’inchiostro: ogni riga
scritta costa, e allora ogni quadro ne riempie poche, righe diverse per quadri
diversi, mentre le altre restano bianche. All’inizio l’archivista non c’era:
per ogni quadro il copista provava una scheda dopo l’altra e teneva quella che,
con meno inchiostro, gli bastava a ridipingerlo. Su migliaia di pezzetti di
fotografie di boschi e di prati ogni riga finiva per descrivere un piccolo
contorno inclinato, come quelli a cui rispondono le cellule della vista nel
nostro cervello. Cercare quadro per quadro, però, è lento: si è aggiunto allora
un archivista che indovina al primo colpo la scheda che la ricerca troverebbe,
e alla fine la ricerca è sparita del tutto. Resta l’archivista che scrive da
solo la scheda, pagando l’inchiostro.

Un altro modo è sporcare. Il museo mette qualche macchia su ogni quadro prima
di darlo all’archivista, e il copista prende il voto solo se restituisce il
quadro pulito: una fotocopia ricopierebbe anche le macchie.

Un altro ancora si capisce con un ritratto. Se la testa gira un po’ a destra, la
scheda se ne deve accorgere, o la copia avrebbe la testa dritta; se sulla tela
compare un graffio, la scheda può ignorarlo, perché nessun quadro del museo
differisce da un altro per un graffio. Il museo allora mette un prezzo su quanto
la scheda si muove quando il quadro si muove di poco: un quadro toccato appena
deve dare una scheda quasi uguale, e una fotocopia, che ricopia ogni ritocco per
intero, il prezzo lo paga tutto. Da sola la richiesta porterebbe a scrivere la
stessa scheda per tutti i quadri; il copista, che deve ridipingerli diversi,
tira dall’altra parte. Nelle prove la scheda che ne esce resta sensibile ai
cambiamenti che portano da un quadro vero a un altro quadro vero, e sorda a
quelli che nessun pittore farebbe, purché i quadri veri, come si suppone,
differiscano fra loro per poche cose. Il prezzo però si conta solo sui quadri
del museo, dove si danno i voti: davanti a una tela diversa da tutte nessuno ha
controllato, e la scheda può saltare quanto vuole.

Questa regola ha una scappatoia. L’archivista scrive tutto diviso per mille, e
la scheda cambia pochissimo qualunque cosa succeda al quadro; il copista
moltiplica per mille prima di dipingere, e la copia esce perfetta. Il prezzo è
salvo e nessuno dei due ha imparato niente. Il rimedio è che i due usino la
stessa tabella, quella che dice quanto ogni punto della tela pesa su ciascuna
riga della scheda: chi divide per mille scrivendo divide per mille anche
dipingendo, e un punto che sulla tela vale 4 diventa 0,004 sulla scheda e
0,000004 sulla copia, che non somiglia più a niente.

Fra le macchie e la scheda che si muove poco c’è un legame. Per restituire
pulito un quadro appena macchiato il copista deve fare in modo che la copia
cambi meno del quadro sporcato: è la stessa richiesta, fatta alla copia invece
che alla scheda, e vale finché le macchie restano leggere e la copia somiglia
già bene al quadro.

Con la strozzatura, l’inchiostro, le macchie o il prezzo sul movimento, il museo
dà sempre il voto a una scheda alla volta, guardando il quadro da cui viene:
nessuno stende mai le schede sulla mappa per vedere come stanno messe, vicine o
lontane. E quale che sia la regola, sulla scheda finiscono le cose che il
copista non saprebbe indovinare da sé (che soggetto è, com’è composto, quali
colori dominano) e non finiscono quelle che sa rimettere da solo (la grana della
tela, il modo in cui uno sfondo sfuma). Quelle il copista non le ricorda quadro
per quadro: ne dipinge una tipica, sempre la stessa per tutti i quadri che hanno
la stessa scheda. Dalla scheda il copista non inventa niente; e in mano non ha
mai avuto una scheda che non venisse da un quadro vero.

`````

`````{tab} Superiore

Un autoencoder è una coppia di funzioni parametriche,
$e_\phi: \mathbb{R}^D \to \mathbb{R}^L$ e $d_\theta: \mathbb{R}^L \to
\mathbb{R}^D$, addestrate insieme a minimizzare l’errore di
ricostruzione

$$
\mathcal{L}(\phi, \theta) = \frac{1}{N} \sum_{i=1}^{N}
\ell\big(\mathbf{x}_i,\; d_\theta(e_\phi(\mathbf{x}_i))\big),
$$

dove $\mathbf{x}_i$ è l’$i$-esimo esempio, $D$ la dimensione del dato, $L$
quella del codice $\mathbf{z}_i = e_\phi(\mathbf{x}_i)$, $N$ il numero di
esempi e $\ell$ una misura di scarto fra dato e ricostruzione, sommata sulle
$D$ componenti (errore quadratico, oppure cross-entropia per componente come
nell’addestramento sulle cifre scritte a mano: è la somma sui pixel a fare del
risultato un costo «per cifra» e non «per pixel»). Il vincolo $L \ll D$ è la
strozzatura, e senza un vincolo il problema è vuoto: con $L \ge D$ basta
prendere $d_\theta$ e $e_\phi$ inverse l’una dell’altra (l’identità, per dire)
e la loss tocca il suo minimo senza che nessuno abbia imparato niente. La
strozzatura, però, non è l’unico vincolo possibile, ed è per questo che sta fra
le condizioni e non nella definizione: ce ne sono altri, che di strozzatura non
ne hanno. Uno chiede che di ogni codice si accendano pochissime componenti, e
di componenti ne tiene più di quante erano quelle di partenza (lo *sparse
autoencoder*, che torna più avanti nel capitolo sull’{doc}`interpretabilità
</Interpretabilita/attribuzione-e-meccanicistica>`); un altro fa ricostruire il
dato da una sua copia sporcata di rumore (il *denoising autoencoder*, che il
capitolo sui {doc}`modelli a energia </ModelliEnergia/oltre-la-partizione>`
riprende più avanti); il terzo pretende che il codice cambi poco quando il dato
cambia poco, ed è l’**autoencoder contrattivo**. Scritti come obiettivi, sono
tre modifiche di $\mathcal{L}$: la sparsità aggiunge un termine $\lambda \sum_j
\lvert z_{ij} \rvert$, con $L$ anche maggiore di $D$; il denoising sostituisce
$\ell(\mathbf{x}_i, d_\theta(e_\phi(\mathbf{x}_i)))$ con
$\mathbb{E}_{\tilde{\mathbf{x}} \sim C(\tilde{\mathbf{x}} \mid \mathbf{x}_i)}\,
\ell(\mathbf{x}_i, d_\theta(e_\phi(\tilde{\mathbf{x}})))$, dove $C$ è il
processo che sporca (rumore gaussiano, pixel azzerati)
{cite}`vincent2008extracting`; il contrattivo di Rifai e colleghi
{cite}`rifai2011contractive` aggiunge il quadrato della norma di Frobenius
dello jacobiano dell’encoder, $\lambda\,\lVert \partial e_\phi(\mathbf{x}_i) /
\partial \mathbf{x}_i \rVert_F^2$. In tutti e tre i casi l’identità smette di
essere una soluzione: con la sparsità e con la contrazione perché costa, con il
rumore perché l’identità lo lascia dov’è.

La versione sparsa ha un antenato senza encoder, la **codifica sparsa** di
Olshausen e Field {cite}`olshausen1996emergence`: un decoder lineare
$\mathbf{B} \in \mathbb{R}^{D \times L}$, il dizionario, anche sovracompleto,
e per ogni dato il codice

$$
\mathbf{z}^\star(\mathbf{x}) = \arg\min_{\mathbf{z}}\;
\tfrac12\,\lVert \mathbf{x} - \mathbf{B}\mathbf{z} \rVert_2^2
+ \lambda \sum_{j=1}^{L} S(z_j),
$$

dove $S$ è una penalità di sparsità: Olshausen e Field ne provano diverse, fra
cui $\log(1 + z^2)$ e $\lvert z \rvert$, con risultati qualitativamente simili.
Con $S(z) = \lvert z \rvert$ il problema in $\mathbf{z}$ è convesso, ed è il
*basis pursuit denoising* di Chen, Donoho e Saunders {cite}`chen1998atomic`, la
forma che si usa di solito; il problema congiunto in $\mathbf{z}$ e $\mathbf{B}$
non lo è, e lo si affronta alternando la ricerca dei codici a dizionario fisso
con l’aggiornamento del dizionario a codici fissi, come in k-means. Addestrato
su pezzetti di immagini naturali, il dizionario si riempie di rilevatori di
bordi localizzati e orientati, simili ai campi recettivi delle cellule semplici
della corteccia visiva primaria. Il prezzo è l’inferenza, perché ogni codice è
un’ottimizzazione. La *decomposizione sparsa predittiva* (PSD) di Kavukcuoglu,
Ranzato e LeCun {cite}`kavukcuoglu2008fast` aggiunge un encoder
$e_\phi(\mathbf{x}) = \mathbf{G}\tanh(\mathbf{W}\mathbf{x} + \mathbf{c})$, con
$\mathbf{W} \in \mathbb{R}^{L \times D}$, $\mathbf{c} \in \mathbb{R}^{L}$ e
$\mathbf{G}$ diagonale, e un termine
$\alpha\,\lVert \mathbf{z} - e_\phi(\mathbf{x}) \rVert_2^2$ nella stessa
funzione obiettivo, con $\alpha$ che pesa quanto il codice deve restare vicino
alla predizione dell’encoder: l’addestramento alterna la minimizzazione in
$\mathbf{z}$, partendo da $e_\phi(\mathbf{x})$, con un passo di gradiente su
$\mathbf{B}$ e su $\phi$, e a regime il codice costa una passata in avanti. Con
$\alpha = 0$ si torna alla codifica sparsa; lo sparse autoencoder è il passo
successivo, che per produrre il codice tiene soltanto l’encoder (il decoder
resta, ed è il dizionario) e mette la penalità direttamente sulla sua uscita.

Che cosa impari il contrattivo lo dice un argomento, sostenuto dagli esperimenti
più che da un teorema. Da sola la penalità darebbe un encoder costante, da sola
la ricostruzione (con $L \ge D$) l’identità; il compromesso è un encoder che,
vicino agli esempi di addestramento, contrae in quasi tutte le direzioni e resta
sensibile soltanto lungo poche, diverse da punto a punto. Se i dati si
concentrano vicino a una varietà di dimensione bassa (l’*ipotesi della varietà*,
che qui si assume), quelle poche sono le sue direzioni tangenti, perché lungo di
esse stanno gli esempi vicini che la ricostruzione deve tenere distinti, e le si
legge nei vettori singolari destri associati ai pochi valori singolari grandi
dello jacobiano {cite}`rifai2011contractive,goodfellow2016deep`. Lontano dai
dati la penalità non viene mai calcolata, e lì l’encoder può anche dilatare. Per
un encoder di un solo strato a sigmoide,
$\mathbf{z} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$ (qui $\sigma$ è la
sigmoide, non una deviazione standard), lo jacobiano è
$\operatorname{diag}\big(z_j(1-z_j)\big)\,\mathbf{W}$ e la penalità ha forma
chiusa, $\sum_{j=1}^{L} \big(z_j(1-z_j)\big)^2 \sum_{k=1}^{D} W_{jk}^2$, con $j$
sulle componenti del codice e $k$ su quelle del dato. Si calcola in $O(DL)$,
come l’errore di ricostruzione, e mostra le due vie per contrarre: pesi piccoli
(con un encoder lineare è la sola, e la penalità si riduce al weight decay
$\lVert\mathbf{W}\rVert_F^2$) oppure unità sature, dove $z_j(1-z_j)$ si annulla.
Con un encoder più profondo la forma chiusa non c’è, e lo jacobiano esatto costa
$L$ passate all’indietro per esempio; Rifai e colleghi lo evitano impilando
encoder di uno strato, addestrati uno alla volta, ciascuno con la sua forma
chiusa. Serve però una scala nel decoder, perché un encoder che moltiplica per
una costante piccola $c$ e un decoder che divide per $c$ azzerano la penalità
senza aver imparato niente: i pesi legati, $\mathbf{W}$ nell’encoder e
$\mathbf{W}^\top$ nel decoder, lo impediscono.

Il legame con il denoising è stretto. Con rumore gaussiano di deviazione
standard $s$ piccola, errore quadratico e una ricostruzione che resta vicina
all’identità,
$d_\theta(e_\phi(\mathbf{x})) = \mathbf{x} + o(1)$ per $s \to 0$, la loss del
denoising vale l’errore di ricostruzione più $s^2$ volte la stessa penalità,
calcolata però sullo jacobiano dell’intera ricostruzione $d_\theta \circ e_\phi$
invece che del solo encoder. Senza l’ultima condizione nello sviluppo resta un
termine dello stesso ordine $s^2$, proporzionale allo scarto di ricostruzione e
alle derivate seconde di $d_\theta \circ e_\phi$. E per un autoencoder di
capacità illimitata la ricostruzione ottima sposta ogni punto in cui la densità
è positiva di $s^2\,\nabla_{\mathbf{x}} \log p(\mathbf{x})$, cioè lungo lo
*score*, il gradiente del logaritmo della densità: lo stesso oggetto che il
denoising score matching, trattato più avanti nel capitolo sui {doc}`modelli a
energia </ModelliEnergia/oltre-la-partizione>`, stima per costruzione, e che
per un denoising di forma particolare aveva già trovato Vincent
{cite}`vincent2011connection`. Tutte e due le affermazioni valgono a meno di
termini $o(s^2)$ {cite}`alain2014regularized`. Lo score, poi, basta a generare,
ma non pescando un codice, bensì camminando nello spazio dei dati con una catena
di Markov guidata da $d_\theta(e_\phi(\mathbf{x})) - \mathbf{x}$, che gli stessi
Alain e Bengio costruiscono con Metropolis-Hastings.

Due osservazioni che tornano utili subito. La prima: nella definizione di
$\mathcal{L}$ non compare nessuna distribuzione. Il $p(\mathbf{x})$ di poco fa
si legge nell’ottimo, non sta scritto nella loss, e non c’è un $p(\mathbf{z})$,
né niente da cui pescare un codice; c’è una funzione che comprime, una che
decomprime e uno scarto da minimizzare. La seconda: la loss vincola i codici
solo uno per uno, tramite la propria ricostruzione, e non dice nulla su come i
codici stiano fra loro; e dove il codice tace il decoder non inventa, perché con
errore quadratico, e con la cross-entropia per componente, il decoder ottimo a
encoder fissato restituisce la {doc}`media condizionata
</RetiNeurali/da-dove-viene-la-loss>` $\mathbb{E}[\mathbf{x} \mid \mathbf{z}]$,
la stessa per tutti i dati che hanno quel codice. Nessuna delle due è una
dimenticanza da correggere in un secondo momento: sono la definizione, e da lì
discende tutto il resto della sezione.

`````

## L’autoencoder lineare è la PCA

Se encoder e decoder sono funzioni lineari, più un termine costante,
l’autoencoder ritrova la PCA (*principal component analysis*), l’analisi delle
componenti principali della {doc}`sezione su riduzione e clustering
</MachineLearning/riduzione-clustering>`: le $L$ direzioni lungo cui i dati
variano di più. L’analisi fattoriale di Spearman, con cui il capitolo si apre,
appartiene alla stessa famiglia lineare, con un’ipotesi diversa sul rumore.

`````{tab} Elementare

Mettiamo che a tutti e due sia vietato essere creativi: ogni numero della
scheda dev’essere una miscela fissa di quello che sta sulla tela, «tanto di
questo più tanto di quello» (tre volte il primo punto, meno il secondo, più
metà del terzo, e così via), la stessa miscela per tutti i quadri, e ogni
quadro ridipinto dev’essere a sua volta una miscela fissa dei numeri della
scheda. In queste condizioni non resta niente da inventare, e il meglio che i
due possono fare è già noto: disporre i quadri su una mappa piatta, tesa lungo
le poche direzioni in cui differiscono di più, che è la cosa che nella sezione
su riduzione e clustering si chiamava analisi delle componenti principali. Su
quella mappa, poi, i due possono tracciare gli assi come vogliono, dritti o
girati: la mappa resta la stessa, e le copie anche.

Dei due divieti, però, quello che decide è il secondo. Se al copista tocca
comunque una miscela fissa, i quadri che ridipinge cadono sulla mappa piatta
comunque, e all’archivista si può concedere qualunque libertà senza che cambi
niente.

Un permesso serve a tutti e due, ed è l’unico: partire dal quadro medio del
museo e annotare soltanto di quanto il quadro che hanno davanti se ne discosta.
Senza, la mappa è costretta a passare per il quadro fatto di niente, la tela
bianca, e quasi mai è quella giusta.

Detto altrimenti: archivista e copista liberi da quei divieti sono la stessa
vecchia macchina con un permesso in più, quello di incurvare la mappa, e a
incurvarla è il copista. La differenza fra le due macchine spiega quando
conviene l’una e quando l’altra: se i quadri stanno davvero su una mappa
piatta, incurvarla non serve; se stanno su una superficie piegata, una mappa
piatta la può solo approssimare.

`````

`````{tab} Superiore

Con $e_\phi$ e $d_\theta$ affini e $\ell$ l’errore quadratico, il minimo della
loss si raggiunge quando la ricostruzione $d_\theta(e_\phi(\mathbf{x}))$ è la
proiezione ortogonale di $\mathbf{x}$ sul sottospazio affine che passa per la
media dei dati ed è generato dalle prime $L$ componenti principali dei dati
centrati {cite}`bourlard1988auto,baldi1989neural`, e il codice ne è un sistema
di coordinate. Baldi e Hornik dimostrano anche la parte che interessa a chi
addestra: se la covarianza dei dati ha rango pieno e autovalori distinti, quel
minimo globale è l’unico minimo locale, e ogni altro punto critico (le
proiezioni su sottospazi generati da altri autovettori, di dimensione al più
$L$) è una sella. Sul caso lineare la discesa del gradiente non ha dove
impantanarsi; con una non linearità nel decoder questa garanzia non c’è più. E
la centratura conta: con mappe puramente lineari e dati non centrati il minimo
è il sottospazio dei primi $L$ vettori singolari destri della matrice grezza (è
il teorema di Eckart e Young {cite}`eckart1936approximation`: la migliore
approssimazione di rango $L$ in norma di Frobenius), che passa per l'origine e
in generale non coincide con quello della PCA. A farsene carico è il termine
additivo, ed è la ragione per cui le `nn.Linear` della `Clessidra` ce l'hanno.
Con una precisazione che conta: la soluzione è unica solo a meno di un cambio
di base nel latente, cioè l’autoencoder lineare recupera il *sottospazio* di
massima varianza, non le singole direzioni principali né il loro ordinamento;
per ritrovare quelle serve un vincolo in più, l’ortonormalità delle direzioni e
l’ordinamento per varianza decrescente, che la PCA impone e l’autoencoder no.

È lo stesso modello lineare-gaussiano dell’apertura del capitolo, nella
versione a rumore isotropo: la soluzione a massima verosimiglianza della PCA
probabilistica individua il sottospazio principale per qualunque varianza di
rumore, e nel limite di rumore infinitesimo la ricostruzione si riduce alla
proiezione ortogonale, cioè alla PCA. L’analisi fattoriale di Spearman è
invece la stessa famiglia con una varianza di rumore per ciascuna componente
osservata, e lì una soluzione in forma chiusa non c’è.

Attenzione poi a dove vanno messe le non linearità, perché il vincolo è
asimmetrico. Bourlard e Kamp dimostrano la metà negativa della faccenda, ed è
quella che sorprende: in una rete a tre strati con uscita lineare, mettere una
non linearità nello strato nascosto non serve a niente: sotto l’ottimo lineare
non si scende. La ragione è che le ricostruzioni sono l’immagine del decoder, e
con un decoder affine quell’immagine è un sottospazio affine comunque sia
fatto l’encoder. A piegare la superficie è il decoder; la non linearità
dell’encoder serve ad atterrarci sopra meglio. La `Clessidra` addestrata sulle
cifre scritte a mano ce l’ha da tutte e due le parti, e tutto il resto (la
strozzatura, la loss, l’assenza di probabilità) è identico.

`````

Da qui in avanti l’autoencoder non è più lineare. Con una {doc}`funzione di
attivazione </RetiNeurali/funzioni-attivazione>` non lineare dentro il
decoder, le ricostruzioni $d_\theta(\mathbf{z})$ formano una superficie curva
di dimensione $L$ invece di un sottospazio piatto; quella dentro l’encoder
serve a scegliere, per ogni dato, un buon punto su quella superficie. Adesso lo
guardiamo lavorare su dati veri.

## Trenta righe, e funziona

Le cifre scritte a mano di `scikit-learn` sono immagini di 8 pixel per lato,
cioè 64 numeri, e sono 1797. Le comprimiamo in otto numeri, che è un ottavo
del dato, e chiediamo alla rete di rifarle.

Lo scarto $\ell$ è la cross-entropia binaria della {doc}`sezione sulla teoria
dell’informazione </Matematica/teoria-informazione>`, sommata sui $D = 64$
pixel:

$$
\ell(\mathbf{x}, \hat{\mathbf{x}}) = -\sum_{k=1}^{D}
\big[\, x_k \log \hat{x}_k + (1 - x_k) \log (1 - \hat{x}_k) \,\big],
$$

dove $x_k \in [0, 1]$ è il grigio del pixel $k$ e $\hat{x}_k$ la probabilità
di «scuro» che il decoder gli assegna. Invece di contare i grigi di differenza
fra originale e copia, misura quanto il decoder si è sbilanciato su ogni pixel
e quanto ci ha azzeccato, e dà un numero che si legge in unità di
informazione.

Una riga del codice con gli autoencoder non c'entra niente,
`torch.set_num_threads(1)`: chiede a PyTorch di fare i conti su un nucleo
solo, e serve perché gli stessi numeri escano su qualunque macchina.[^thread]

```python
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits

torch.manual_seed(0)
# un thread solo: cosi' i numeri stampati qui sotto sono gli stessi su
# qualunque macchina, e su dati piccoli come questi e' anche piu' veloce
torch.set_num_threads(1)

# 1797 cifre scritte a mano, 8x8 pixel, riportate fra 0 (chiaro) e 1 (scuro)
X = torch.tensor(load_digits().data / 16.0, dtype=torch.float32)


class Clessidra(nn.Module):
    """Encoder e decoder, con la strozzatura in mezzo."""

    def __init__(self, latente=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(64, 48), nn.ReLU(),
                                     nn.Linear(48, latente))
        self.decoder = nn.Sequential(nn.Linear(latente, 48), nn.ReLU(),
                                     nn.Linear(48, 64))


rete = Clessidra()
opt = torch.optim.Adam(rete.parameters(), lr=3e-3)
for passo in range(4000):
    # il decoder esce in logit; la cross-entropia li confronta col grigio vero
    perdita = F.binary_cross_entropy_with_logits(
        rete.decoder(rete.encoder(X)), X, reduction="sum") / len(X)
    opt.zero_grad()
    perdita.backward()
    opt.step()

print(f"errore di ricostruzione:   {perdita.item():.1f} nat per cifra")

# il metro di paragone: chi non guarda la cifra e dichiara, per ogni pixel,
# il grigio medio che quel pixel ha su tutte le 1797 cifre
marginale = F.binary_cross_entropy(
    X.mean(0).expand_as(X), X, reduction="sum") / len(X)
print(f"chi non guarda la cifra:  {marginale.item():.1f} nat per cifra")
```

```text
errore di ricostruzione:   16.3 nat per cifra
chi non guarda la cifra:  27.1 nat per cifra
```

L’errore si misura in nat, parenti stretti dei bit dei {doc}`richiami di
matematica </Matematica/teoria-informazione>`: il bit conta le scelte fra due,
il nat fa lo stesso conto con il numero 2,718… (quello che i matematici
chiamano $e$) al posto del 2, e un nat vale circa 1,44 bit. Sono i nat che una
cifra costa in media a chi la deve indovinare un pixel alla volta, e più sono,
peggio si è scommesso. (Con due riserve. La formula è pensata per pixel tutti
bianchi o tutti neri, e su grigi intermedi non è più una probabilità vera: i
suoi nat vanno letti come un metro per confrontare due macchine, non come una
misura assoluta. E le cifre su cui si misura sono le stesse su cui la rete si è
addestrata, perché qui interessa il confronto fra due macchine e non quanto
bene la rete se la cavi su cifre mai viste.) Da solo il 16,3 non direbbe
niente, e per questo c’è la seconda
riga. Il confronto giusto non è con chi tira a caso, che è un bersaglio troppo
facile, ma con chi ha guardato bene tutte le cifre e non guarda quella che
deve rifare: per ogni pixel dichiara il grigio che quel pixel ha in media, e
nient’altro. Quello spende 27,1 nat. L’autoencoder, con otto numeri, ne spende
16,3: tre quinti, avendo compresso la cifra in un ottavo dello spazio. Quel
27,1 tornerà nella {doc}`sezione sul latente che si usa
</ModelliLatenti/il-latente-che-si-usa>`, dove sarà il segno di un guasto.

```python
LIVELLI = " .:-=+*#%"


def affianca(*immagini):
    """Le immagini 8x8 stampate una accanto all'altra, in caratteri."""
    griglie = [(im.reshape(8, 8) * 8).round().long().clamp(0, 8) for im in immagini]
    return "\n".join("   ".join("".join(LIVELLI[i] for i in g[r]) for g in griglie)
                     for r in range(8))


with torch.no_grad():
    codici = rete.encoder(X)
    ricostruite = torch.sigmoid(rete.decoder(codici))

print("quattro cifre vere")
print(affianca(*X[:4]))
print("\nle stesse, rifatte a partire da otto numeri")
print(affianca(*ricostruite[:4]))
```

```text
quattro cifre vere
  :*=         **:        :%*       =%*
  *%+%:       *%=       :%%#      =*-%:
 :%. *=      :%%-       =*=%      . **
 :*  ==     =%%%.        -%*       .%*
 :=  ==       %%:       =*%          **
 :*  *=       %%-      =%%:           +=
 .#:+*        %%-      :*%%*:      =::#=
  -*+         *%+        :*%=      =**=

le stesse, rifatte a partire da otto numeri
  -#*:        =#:        -%*       -#+
  #*+#.      .%%=       .###.     :*+#:
 :#..#-      =%%-       :**#      .:+%.
 :+  ++     .*%%.       .=#=       .*#.
 :=  ==      :%%.       +%#          +*
 .*  *:      .#%-      =%%*.          *-
  *=*#       .#%-      :###*.      ::-#=
  -#*.        +#=        :#%+      =##+
```

Le quattro cifre vere sono uno zero, un uno, un due e un tre; le prime due si
leggono a colpo d’occhio, il due si riconosce dalla base larga nella penultima
riga, il tre bisogna proprio saperlo. Ma non è questo il punto. Il punto è che
la riga di sotto ripete la riga di sopra, tratto per tratto, e ci arriva
partendo da otto numeri soli. La compressione funziona, e il resto della
sezione non la mette in dubbio.

## Il cammino che si perde

Adesso la domanda che ci interessa. Prendiamo due cifre vere, i loro codici
$\mathbf{z}_0$ e $\mathbf{z}_1$, e camminiamo in linea retta dall’uno
all’altro: $\mathbf{z}_t = (1 - t)\,\mathbf{z}_0 + t\,\mathbf{z}_1$, con $t$ che
va da 0 a 1, facendo decodificare ogni tappa. Su ciascuna delle otto posizioni
si fa la stessa cosa: se il primo codice comincia con 3 e il secondo con 7, a
metà strada quel numero vale 5 e a un quarto vale 4. Se lo spazio latente fosse
ben organizzato, dovremmo vedere una cifra trasformarsi con continuità
nell’altra.

Per sapere se le tappe cadono dove il decoder è già stato serve anche una
distanza fra codici, ed è quella euclidea, $\lVert \mathbf{z} - \mathbf{z}'
\rVert = \sqrt{\textstyle\sum_j (z_j - z'_j)^2}$: il teorema di Pitagora con
otto cateti invece di due. Fra i codici $(3, 1)$ e $(7, 4)$, per esempio, è
$\sqrt{4^2 + 3^2} = 5$. Il metro di paragone è la spaziatura tipica, la
distanza mediana fra un codice vero e il suo vicino più prossimo.

```python
with torch.no_grad():
    partenza, arrivo = codici[0], codici[1]      # lo zero e l'uno di prima
    tappe = torch.stack([partenza + t * (arrivo - partenza)
                         for t in torch.linspace(0, 1, 5)])
    print(affianca(*torch.sigmoid(rete.decoder(tappe))))

# quanto dista ogni tappa dal codice vero piu' vicino, e quanto distano di
# solito due codici veri vicini
fra_codici = torch.cdist(codici, codici)
fra_codici.fill_diagonal_(float("inf"))
spaziatura = fra_codici.min(1).values.median()
dalle_tappe = torch.cdist(tappe, codici).min(1).values
print("\ndal codice vero piu' vicino: "
      + "  ".join(f"{d:.2f}" for d in dalle_tappe))
print(f"spaziatura tipica:           {spaziatura:.2f}")
```

```text
  -#*:       .++.       .=*:       .+*.        =#:
  #*+#.      **+#.      +*+*       -%#=       .%%=
 :#..#-     .#..#:      *-:*.      *#++       =%%-
 :+  ++     .*:.+.     .***+      .*##=      .*%%.
 :=  ==     .*:.=.      *##=       -#%:       :%%.
 .*  *:     .#..*.      *-=+       :*#=       .#%-
  *=*#       +==*       -===       :#*=       .#%-
  -#*.       .**.       .+*:        +#-        +#=

dal codice vero piu' vicino: 0.00  2.31  3.32  3.56  0.00
spaziatura tipica:           2.34
```

Agli estremi ci sono lo zero e l’uno, riconoscibili. In mezzo l’anello dello
zero si stringe e si riempie di grigio a poco a poco, senza mai chiudersi del
tutto; poi si storce; poi resta una barra spessa che non è ancora un uno. La
prima delle tre tappe intermedie è ancora uno zero, uno zero che si sta
sfaldando; le altre due non si lasciano chiamare per nome: non sono cifre.

Le distanze dicono dove si trovano quelle tappe. La prima tappa intermedia
dista dal codice vero più vicino quanto due codici veri vicini distano fra loro
(2,31 contro 2,34); la seconda e la terza circa una volta e mezza la
spaziatura, in un tratto dove il decoder non ha mai avuto un codice da
ricostruire, e produce quello che gli riesce.

Lo stesso succede, e peggio, provando a inventare da zero. «Inventare» qui vuol
dire una cosa precisa: si guarda dove stanno i codici veri (il loro centro, e
quanto sono sparpagliati attorno a quel centro, una posizione per volta), si
pesca un punto a caso in quella zona, cioè da una gaussiana con quella media e
quelle deviazioni standard, e lo si fa decodificare. È il modo più ragionevole
di provarci. Guardare una posizione per volta è però una semplificazione, e si
prova anche a guardarle tutte insieme.

```python
with torch.no_grad():
    sorteggiati = codici.mean(0) + codici.std(0) * torch.randn(500, 8)
    inventate = torch.sigmoid(rete.decoder(sorteggiati))

print("quattro cifre decodificate da codici sorteggiati")
print(affianca(*inventate[:4]))

lontananza = torch.cdist(sorteggiati, codici).min(1).values.median()

print(f"\nfra un codice vero e il suo vicino:   {spaziatura:.2f}")
print(f"fra un codice sorteggiato e i veri:   {lontananza:.2f}"
      f"   ({lontananza / spaziatura:.1f} volte la spaziatura)")

# la stessa pesca guardando anche come le otto posizioni vanno d'accordo
# fra loro, invece di una per volta
with torch.no_grad():
    scala = torch.linalg.cholesky(torch.cov(codici.T))
    accoppiati = codici.mean(0) + torch.randn(500, 8) @ scala.T
accoppiata = torch.cdist(accoppiati, codici).min(1).values.median()

print(f"lo stesso, guardandole insieme:       {accoppiata:.2f}"
      f"   ({accoppiata / spaziatura:.1f} volte la spaziatura)")
```

```text
quattro cifre decodificate da codici sorteggiati
  =%%.       .=%%%+     .*%*       :#*
 +%%%%       *+.%%.     *%#%.     .#+*-
 #%.+#      .#  =+      #==#.     .: ++
 -+  .:     .-          ++#+         #-
 -*  ++     :=.:.       =%%-        :#:
 #%%.*%     :#-+=       #%#*        **:.
 +%%+#%      +-=:       =#*+       .#+=:
  =%%%:      .#*:       .*#=       :*+.

fra un codice vero e il suo vicino:   2.34
fra un codice sorteggiato e i veri:   5.20   (2.2 volte la spaziatura)
lo stesso, guardandole insieme:       4.21   (1.8 volte la spaziatura)
```

Le quattro immagini hanno l’aria di cifre e non lo sono: la seconda e la
quarta hanno tratti che si interrompono a
metà, la prima è un anello troppo grasso, la terza una forma piena, e nessuna
delle quattro si lascia chiamare per nome. Ma il numero conta più delle
immagini,
perché dice il perché invece del sintomo. I codici veri stanno a poco più
di due unità l’uno dall’altro; un codice sorteggiato dista più di cinque dal
più vicino dei codici veri. Sorteggiare in quello spazio vuol dire finire, di
norma, a più del doppio della distanza che separa due codici veri vicini: è
terra mai battuta, ed è la regola più che l’eccezione. E non dipende dal modo
semplice di pescare. Sorteggiando da una gaussiana con covarianza piena, che
tiene conto anche di come le otto posizioni variano insieme, il divario scende
da 2,2 spaziature a 1,8 e non sparisce: i codici veri non stanno in una nuvola
a forma di campana, nemmeno inclinata.

## Perché la strozzatura non basta

L’obiettivo dell’autoencoder contiene soltanto la ricostruzione, e niente in
$\mathcal{L}$ dice dove debbano stare i codici l’uno rispetto all’altro. Il
difetto sta lì, nell’obiettivo, e una rete più grande non lo ripara.

`````{tab} Elementare

Che cosa è stato chiesto ai due, in tutto? Una cosa sola, e per milioni di
volte: «la copia somiglia all’originale?». Su quella pagella non compare da
nessuna parte la richiesta di tenere le schede in ordine nel cassetto, né
quella di riempire i vuoti fra una scheda e l’altra, né quella di dipingere
qualcosa di sensato partendo da una scheda che nessuno ha mai scritto. Quello
che non si chiede non si ottiene, e qui non è stato chiesto.

E c’è un motivo per aspettarsi anche di peggio, che non è dimostrato ma è
plausibile: se l’unica cosa che conta è che ogni quadro torni indietro
riconoscibile, all’archivista conviene tenere lontani fra loro i gruppi di
schede, perché più sono distanti, meno rischia che il copista li confonda. E
allontanare i gruppi, a parità di schede, vuol dire allargare i vuoti in
mezzo. Il cassetto ne esce con le schede addossate in
qualche angolo, larghe distese vuote in mezzo, e nessun confine che dica dove
finisce la zona buona. Per rileggere va benissimo. Per pescare, no: non si sa
dove pescare, e quasi ovunque si peschi non c’è niente.

Manca quindi una regola su dove vanno messe le schede, e sono due cose
insieme: una forma decisa in anticipo per il cassetto, così si sa dove pescare,
e un voto che pretenda quella forma accanto al voto sulla somiglianza. Non una
scheda scritta meglio: una regola sull’insieme.

`````

`````{tab} Superiore

L’autoencoder ottimizza la sola ricostruzione, e nella sua loss non compare
nulla che riguardi la distribuzione dei codici che produce. Quella
distribuzione, l’**aggregato**

$$
q_\phi(\mathbf{z}) = \mathbb{E}_{\mathbf{x} \sim p_{\text{dati}}}
\big[\delta\big(\mathbf{z} - e_\phi(\mathbf{x})\big)\big],
$$

dove $\delta$ è la delta di Dirac (l’encoder qui è deterministico, quindi ogni
dato contribuisce un punto solo) e $p_{\text{dati}}$ la distribuzione da cui
gli esempi provengono, è una misura concentrata sugli $N$ codici degli esempi
di addestramento. Campionarla si può, ma vuol dire soltanto ripescare un codice
già visto, codificando un esempio a caso; fra un codice e l’altro la sua forma
non è vincolata da niente, e da lei non si ottengono codici nuovi. Generare
richiede esattamente quello, cioè una distribuzione da cui pescare un
$\mathbf{z}$ mai visto prima di decodificarlo. (Nella sezione seguente lo
stesso simbolo $q_\phi(\mathbf{z})$ tornerà con l’encoder diventato stocastico:
là le delta saranno gaussiane, l’aggregato sarà una loro mistura e prenderà il
nome con cui la letteratura lo chiama, *posterior aggregata*.) Sostituirla a
posteriori con una gaussiana adattata ai codici è la scorciatoia ovvia, e i
rapporti appena misurati dicono quanto costa: $2{,}2$ spaziature con covarianza
diagonale, $1{,}8$ con covarianza piena. La gaussiana copre una regione che
$q_\phi(\mathbf{z})$ non occupa.

E non c’è nemmeno niente che si opponga alla dilatazione del latente.
L’argomento è euristico: a parità del resto, codici più distanti fra loro si
ricostruiscono meglio, perché il decoder ha meno occasioni di confonderli, e
nella loss non compare nessun termine che paghi quella distanza. I rapporti
appena misurati, però, quella dilatazione non la possono vedere: sono rapporti
fra due lunghezze del latente, e moltiplicare tutti i codici per una costante le
moltiplica tutte e due. Misurano il disaccordo di forma di poco sopra, non la
scala. Si dice, con
formula spiccia, che il latente non è regolarizzato, e la regolarizzazione
che manca riguarda la distribuzione dei codici, non i pesi.

Da qui il programma della sezione seguente. Servono due cose insieme, e sono le
due che il nome «autoencoder variazionale» tiene una per parola: una
distribuzione bersaglio $p(\mathbf{z})$ scelta in anticipo (così si sa dove
pescare) e un termine nella loss che spinga i codici a distribuirsi come lei.
La sorpresa, e il motivo per cui la sezione è lunga, è che quel termine non si
inventa: cade fuori da solo dal tentativo, tutt’altro, di massimizzare la
verosimiglianza dei dati.

`````

Prima di tirare le somme, una precisazione su che cosa non è in discussione.
L’autoencoder resta il modo giusto di comprimere, ed è così che lo usano i
capitoli che gli chiedono di comprimere: nella {doc}`sezione sui codec
neurali </Audio/codec-neurali>` per fabbricare un alfabeto del suono, e nella
{doc}`sezione su Stable Diffusion </ModelliDiffusione/stable-diffusion>` per
rimpicciolire un’immagine di quarantotto volte. In tutti e due i casi ha un
ingrediente in più, diverso dall’uno all’altro: per l’immagine il termine che
la {doc}`sezione sull’ELBO </ModelliLatenti/il-salto-probabilistico>` ricava,
tenuto con un peso piccolissimo; per il suono il latente fatto di simboli della
sezione sul {doc}`latente che si usa </ModelliLatenti/il-latente-che-si-usa>`.
Il compito che gli si affida, però, resta comprimere. Il difetto misurato qui
riguarda un mestiere diverso, fabbricare dati nuovi a partire da un codice
sorteggiato, che a un compressore nessuno ha mai chiesto e che nessuna quantità
di addestramento gli fa venire.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un autoencoder è un archivista e un copista che si allenano insieme: il
  primo riduce ogni dato a una scheda di pochi numeri, il secondo ricostruisce
  il dato dalla sola scheda, e il voto è uno solo, quanto la copia somiglia
  all’originale.
- La strozzatura è la richiesta, non un limite: potendo scrivere una scheda
  lunga quanto il quadro, i due imparerebbero a fotocopiare. Altre regole
  fanno lo stesso lavoro senza strozzatura: l’inchiostro che si paga, quadri
  sporcati da ripulire, una scheda che si muove poco quando il quadro cambia
  poco.
- Comprimere funziona: otto numeri bastano a rifare una cifra scritta a mano in
  modo che si riconosca.
- Generare a partire da una scheda no. Camminando in linea retta fra due schede
  vere si incontrano punti che non vogliono dire niente, e pescando una scheda a
  caso si finisce, di norma, a quasi il doppio della distanza che separa due
  schede vere vicine, o anche più.
- La colpa non è dell’archivista: nella sua pagella non compariva l’ordine del
  cassetto. Quello che manca è una regola su dove vanno messe le schede, ed
  è fatta di due pezzi, una forma decisa in anticipo per il cassetto e un voto
  che quella forma la pretenda; la sezione seguente li ricava senza inventarli.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un autoencoder addestra $e_\phi$ e $d_\theta$ sulla sola ricostruzione, e
  nella sua definizione non compare nessuna distribuzione. La strozzatura
  $L \ll D$ è uno dei vincoli che gli impediscono di fotocopiare, non la
  definizione: al suo posto vanno la sparsità del codice, il rumore
  sull’ingresso o la penalità contrattiva sullo jacobiano dell’encoder.
- Con $e_\phi$ e $d_\theta$ affini ed errore quadratico ritrova il
  sottospazio affine che passa per la media dei dati ed è generato dalle prime
  $L$ componenti principali {cite}`bourlard1988auto,baldi1989neural`, a meno
  di un cambio di
  base: è la PCA della sezione su riduzione e clustering. Senza il termine
  additivo, su dati non centrati, quel sottospazio passa per l’origine e in
  generale non è quello della PCA. Basta una non linearità nel decoder
  perché cada il vincolo di affinità e la superficie che ritrova possa essere
  curva.
- La ricostruzione riesce (16,3 nat per cifra contro i 27,1 di chi dichiara il
  grigio medio di ogni pixel senza guardare la cifra: tre quinti, su cifre da
  64 pixel compresse in 8 numeri); il campionamento no: un codice sorteggiato
  da una gaussiana adattata ai codici dista dai codici veri $2{,}2$ volte la
  loro spaziatura tipica ($1{,}8$ con covarianza piena).
- La causa è strutturale: l’aggregato $q_\phi(\mathbf{z})$ non è vincolato da
  nulla, e niente nella loss paga la distanza fra i codici, quindi niente
  si oppone a un latente dilatato e pieno di vuoti.
- Serve quindi un prior $p(\mathbf{z})$ dichiarato e un termine che avvicini i
  codici a lui. La sezione seguente non lo aggiunge a mano: lo ricava.
```

`````

Resta un avvertimento, prima di andare avanti. L’esperimento è andato
benissimo, e ha risposto a una domanda diversa da quella che avevamo in testa.
Ottimizzare la sola ricostruzione e sperare in uno spazio latente ordinato vuol
dire scambiare la grandezza che si misura con quella che si vuole, un errore
che torna spesso, in forme diverse. La {doc}`sezione
sull’ELBO </ModelliLatenti/il-salto-probabilistico>` lascia l’autoencoder com’è
e ne cambia l’obiettivo: da «ricostruire il dato» a «rendere probabile il
dato».


[^thread]: Il motivo è che le somme in virgola mobile non sono associative:
sommando gli stessi numeri in un ordine diverso il risultato cambia nell’ultima
cifra, perché a ogni passo il totale parziale viene arrotondato a quante cifre
il formato può tenere, e arrotondare un totale grande insieme a un addendo
piccolo ne perde un pezzo. PyTorch, per andare più veloce, spezza ogni somma
lunga fra i nuclei di calcolo disponibili e poi rimette insieme i pezzi: quanti
sono i pezzi dipende da quanti nuclei ha la macchina, quindi macchine diverse
sommano in ordini diversi. Su una rete addestrata per quattromila passi
quelle ultime cifre si accumulano, e alla fine si vedono: il costo di
descrizione del $\beta$-VAE a $\beta = 0{,}5$, nella sezione sul latente che si
usa, senza quella riga vale 6,04 lasciando lavorare quattro nuclei e 6,03
usandone uno solo. La regola è più larga di
PyTorch: fissare il seme non basta, dato che il seme
governa i sorteggi e non l’ordine delle somme. E la riparazione non costa
niente, anzi. Su tensori piccoli come questi (1797 cifre da 64 numeri) un
thread solo è più veloce, e di parecchio. Non è un paradosso: spartire il
lavoro e rimettere insieme i pezzi ha un costo fisso, che qui si paga
quattromila volte, e ogni volta a ciascun nucleo tocca un pezzo di conto
talmente piccolo che il tempo per consegnarglielo supera il tempo per farlo. Il
parallelismo rende quando a ciascuno tocca abbastanza da fare, ed è la stessa
ragione per cui non si chiamano quattro muratori a spostare un mattone.
