# Un pixel alla volta

Il libro sa già generare una cosa alla volta. Il capitolo sui Transformer
scrive testo una parola alla volta; quello sull'audio produce suono un token
alla volta, e prima ancora, con WaveNet, un campione d'onda alla volta. La
ricetta è sempre la stessa: si mette il dato in fila, si insegna alla rete a
indovinare il pezzo successivo dati i precedenti, e la probabilità dell'intero
è il prodotto delle probabilità dei pezzi. Nessuna approssimazione: quel
prodotto **è** la probabilità, non una sua stima, e ogni fattore è una
probabilità vera perché è una scelta fra un numero finito di possibilità, con i
pesi che sommano a uno.

Chiediamoci allora la cosa ovvia: e un'immagine? È più vecchia di quanto si
creda, come domanda. Nel gennaio del 2016 Aäron van den Oord, Nal Kalchbrenner
e Koray Kavukcuoglu {cite}`oord2016pixel` costruiscono una rete che «predice in
sequenza i pixel di un'immagine lungo le due dimensioni spaziali»; WaveNet, che
il capitolo sull'audio racconta come la pietra miliare del suono generato, è la
sorella minore di questo lavoro, stesso laboratorio e stesso anno, con l'onda
al posto della griglia.

Servono due cose, e la seconda è tutto il mestiere.

## Primo: un ordine

Una frase un ordine ce l'ha, un'immagine no. Bisogna sceglierne uno e non
cambiarlo più: si va **riga per riga, da sinistra a destra**, come quando si
legge, e dentro un pixel a colori si mette anche un ordine fra i tre canali,
cioè fra il rosso, il verde e il blu.
La scelta è arbitraria e nessuno pretende che sia la migliore: pretende solo di
essere fissa, perché è rispetto a essa che «prima» e «dopo» vogliono dire
qualcosa.

Fatto questo, la probabilità di un'immagine è il prodotto di quella di ogni
pixel dato tutto ciò che nella lettura viene prima. Per una figurina di
$32 \times 32$ in bianco e nero sono 1.024 fattori, uno per pixel, e ciascuno è
una distribuzione su 256 livelli di grigio: una probabilità vera, che somma a
uno, non un punteggio da normalizzare chissà come.

## Secondo: una convoluzione che guarda solo indietro

Qui arriva l'ostacolo, e ha una forma precisa. Il capitolo sul deep learning ha
speso pagine per spiegare perché su un'immagine si usa una convoluzione e non
uno strato denso: perché una convoluzione guarda un intorno, cioè i vicini di
casa in **tutte** le direzioni. Ma «tutte le direzioni» qui è esattamente ciò
che non si può fare: se nel prevedere un pixel la rete sbircia quelli che
vengono dopo, sta barando, e la probabilità che ne esce non vale niente.

Il rimedio è brutale e funziona: si prende il filtro e si **azzerano** le
caselle che guardano nel futuro. Il filtro resta un quadrato, ma metà del
quadrato è spenta per sempre.

E qui serve una distinzione che sembra un cavillo e non lo è. Al **primo**
strato, fra le caselle da spegnere c'è anche quella centrale, cioè il pixel
stesso che stiamo cercando di indovinare: se restasse accesa, la rete
imparerebbe in tre secondi a copiare la risposta dalla domanda, e avremmo un
modello con verosimiglianza perfetta e utilità zero. È la maschera detta di
**tipo A**. Dal secondo strato in poi, invece, quella casella centrale non
contiene più il pixel vero: contiene il riassunto che il primo strato ne ha
fatto, e quel riassunto il pixel vero non l'ha mai visto. Spegnerla sarebbe uno
spreco, e si lascia accesa: è la maschera di **tipo B**.

## Il codice, e una sorpresa

La causalità non si dichiara, si verifica, e il modo più diretto è chiederla al
gradiente: se muovendo un pixel l'uscita in una certa posizione non cambia,
quel pixel non è entrato nel conto.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)


class ConvMascherata(nn.Conv2d):
    """Convoluzione che puo' guardare solo i pixel gia' visitati.

    Ordine di scansione: riga per riga, da sinistra a destra. Il tipo 'A'
    esclude anche il pixel centrale (serve al primo strato: se lo vedesse, il
    modello imparerebbe a copiarlo); il tipo 'B' lo include, perche' da li' in
    poi il centro non e' piu' il pixel vero ma un riassunto legittimo.
    """

    def __init__(self, tipo, *a, **kw):
        super().__init__(*a, **kw)
        _, _, kh, kw_ = self.weight.shape
        m = torch.ones(kh, kw_)
        m[kh // 2, kw_ // 2 + (1 if tipo == "B" else 0):] = 0   # resto della riga
        m[kh // 2 + 1:] = 0                                      # tutte le righe sotto
        self.register_buffer("maschera", m)

    def forward(self, x):
        return self._conv_forward(x, self.weight * self.maschera, self.bias)


def campo_visivo(n_strati, rif=(8, 4), n=9):
    """Quali pixel entrano DAVVERO nel conto per il pixel `rif`?

    Non lo deduciamo dalle maschere: lo chiediamo al gradiente. Se muovendo un
    pixel l'uscita in `rif` non cambia, quel pixel non e' stato guardato.
    """
    strati = [ConvMascherata("A", 1, 8, 3, padding=1), nn.ReLU()]
    for _ in range(n_strati - 2):
        strati += [ConvMascherata("B", 8, 8, 3, padding=1), nn.ReLU()]
    strati.append(ConvMascherata("B", 8, 1, 3, padding=1))
    x = torch.zeros(1, 1, n, n, requires_grad=True)
    nn.Sequential(*strati)(x)[0, 0, rif[0], rif[1]].backward()
    return x.grad[0, 0].abs() > 0


N, RIF = 9, (8, 4)
prima = torch.tensor([[(r * N + c) < (RIF[0] * N + RIF[1]) for c in range(N)]
                      for r in range(N)])

for L in (6, 12, 24):
    visto = campo_visivo(L, RIF, N)
    print(f"{L:2d} strati | pixel del futuro guardati: {int((visto & ~prima).sum())}"
          f" | pixel del passato mai guardati: {int((prima & ~visto).sum())}")

print("\ncampo visivo con 24 strati ('#' visto, '.' passato mai visto, "
      "' ' futuro):")
visto = campo_visivo(24, RIF, N)
for r in range(N):
    print("   " + " ".join("#" if visto[r, c] else ("." if prima[r, c] else " ")
                           for c in range(N)))
```

```text
 6 strati | pixel del futuro guardati: 0 | pixel del passato mai guardati: 24
12 strati | pixel del futuro guardati: 0 | pixel del passato mai guardati: 6
24 strati | pixel del futuro guardati: 0 | pixel del passato mai guardati: 6

campo visivo con 24 strati ('#' visto, '.' passato mai visto, ' ' futuro):
   # # # # # # # # #
   # # # # # # # # #
   # # # # # # # # #
   # # # # # # # # #
   # # # # # # # # #
   # # # # # # # # .
   # # # # # # # . .
   # # # # # # . . .
   # # # #
```

La prima colonna dei numeri è la buona notizia, e va letta come una prova: a
qualunque profondità, **zero** pixel del futuro entrano nel conto. Le maschere
fanno il loro mestiere.

La seconda colonna è la sorpresa. Con sei strati ventiquattro pixel del passato
restano fuori, e ci si sta: la rete non arriva così lontano, basta farla più
profonda. Con dodici ne restano fuori sei. Con ventiquattro ne restano fuori
**ancora sei**. Non è una questione di profondità: c'è una zona che quelle
maschere non raggiungeranno mai, per quanto si insista, ed è il triangolo che
nella mappa sale a destra del pixel da indovinare. Si chiama **punto cieco**
(*blind spot*), e il lavoro che l'ha diagnosticato lo dice in una riga: le
PixelCNN «hanno un punto cieco nel campo recettivo che non può essere usato per
fare predizioni» {cite}`oord2016conditional`.

Il perché è geometrico e si vede sulla mappa. Ogni strato allarga la vista di
una casella per lato, ma verso destra può salire di una sola colonna per ogni
riga che sale, perché la maschera taglia la riga corrente all'altezza del
centro. Il risultato è un cono che si apre verso l'alto a sinistra e resta
chiuso verso l'alto a destra, mentre il passato vero, quello dell'ordine di
lettura, comprende tutta la riga di sopra fino in fondo.

```{figure} ../figures/campo-cieco.svg
:name: fig-campo-cieco
:alt: "Griglia di nove per nove quadretti con il pixel da indovinare in basso al centro. Al crescere della profondità i quadretti visti si accendono allargandosi verso l'alto a sinistra; i quadretti che vengono dopo nell'ordine di lettura restano spenti, e un triangolo di sei quadretti appena sopra e a destra del pixel da indovinare resta spento pur venendo prima: è il punto cieco."
:width: 92%

Il campo visivo cresce con la profondità, il punto cieco no. In terracotta il
pixel da indovinare, in teal (il verde-azzurro scuro) quello che la rete
arriva a guardare; i quadretti tratteggiati in ocra vengono prima nell'ordine
di lettura, quindi la rete avrebbe tutto il diritto di guardarli, e non li
guarderà mai. Quelli grigi vengono dopo, e restano spenti giustamente.
```

Il disegno lo mostra strato per strato ({numref}`fig-campo-cieco`): il cono si
allarga fin dove arriva, e i sei quadretti tratteggiati che salgono a destra
del pixel da indovinare restano dove sono anche quando tutto il resto si è
acceso.

La riparazione, nello stesso lavoro, è dividere il filtro in due pile che
lavorano in parallelo: una **verticale**, che guarda tutte le righe di sopra
senza nessuna maschera e quindi cresce a rettangolo, e una **orizzontale**, che
guarda solo la riga corrente da sinistra. Le due uscite si sommano dopo ogni
strato, e insieme «catturano l'intero campo recettivo». Il punto cieco
sparisce, e la verosimiglianza migliora; quel lavoro cambia però anche le
funzioni di attivazione, quindi il miglioramento misurato è delle due modifiche
insieme.

`````{tab} Elementare

Il punto cieco merita un'immagine, perché è il genere di guasto che nessuno si
aspetta e che salta fuori solo misurando.

Immagina di dover indovinare una parola coperta in un testo, con il permesso di
leggere tutto quello che viene prima. Il permesso ce l'hai, ma gli occhiali che
ti hanno dato hanno una fessura di forma sbagliata: leggi benissimo tutta la
colonna a sinistra, leggi le righe di sopra ma solo fino a un certo punto verso
destra, e più sali più il pezzo di destra ti sfugge. Nessuno ti sta impedendo
di leggerlo: sono gli occhiali, cioè lo strumento con cui guardi, e cambiare
occhiali più spessi non serve, perché la fessura ha quella forma a
prescindere.

Sono due difetti diversi, e la tabella li separa. Con sei strati la rete non
arriva lontano abbastanza: è miopia, e si cura con la profondità. Con
ventiquattro strati la miopia è passata e restano sempre gli stessi sei pixel:
quello non è più un limite di potenza, è un limite di **forma**, e si cura
soltanto cambiando la forma dello strumento. È esattamente ciò che il lavoro
del 2016 fa, mettendo due finestre invece di una: una che guarda tutte le
righe di sopra per intero e una che guarda la riga corrente da sinistra.

La lezione vale ben oltre le immagini, ed è una di quelle che tornano: quando
un modello non arriva a un risultato, la prima domanda non è «quanto lo devo
fare più grande», è «c'è qualcosa che questa architettura non può fare in linea
di principio?». Nel primo caso si spende calcolo, nel secondo lo si butta.

`````

`````{tab} Superiore

La fattorizzazione è la regola della catena applicata a un ordinamento totale
dei pixel:

$$
p(\mathbf{x}) = \prod_{i=1}^{n^2} p\big(x_i \mid x_1, \dots, x_{i-1}\big),
$$

con $x_i$ l’$i$-esimo pixel in ordine di scansione (e, sui colori, i tre canali
ordinati dentro ciascun pixel). Ogni fattore è una categorica su 256 livelli,
quindi normalizzata per costruzione: $\log p(\mathbf{x})$ è **esatta** e si
ottiene in un solo passaggio in avanti, perché durante l'addestramento tutti i
contesti sono disponibili insieme (*teacher forcing*). È l'asimmetria
caratteristica della famiglia: valutare costa un passaggio, campionare ne costa
$n^2$.

Le maschere realizzano il vincolo di causalità a livello di pesi:
$\mathbf{W} \leftarrow \mathbf{W} \odot \mathbf{M}$ con $\mathbf{M}$ binaria,
tipo A al primo strato ($\mathbf{M}$ azzera il centro) e tipo B dopo. La
composizione di strati mascherati resta causale perché la causalità è chiusa
per composizione, ed è ciò che il test sul gradiente verifica: detta
$\hat{x}_i$ l'uscita della rete in posizione $i$, si ha $\partial \hat{x}_i /
\partial x_j = 0$ per ogni $j \geq i$ nell'ordine di scansione, a qualunque
profondità.

Il punto cieco è il prezzo della realizzazione, non del vincolo. Detto in
termini di campo recettivo: la maschera tronca la riga corrente al centro,
quindi l'espansione verso destra guadagna al più una colonna per ogni riga
guadagnata in alto. Il bordo destro del campo sale quindi a $45^\circ$ invece
di coprire tutto il semipiano che l'ordinamento consentirebbe, e ciò che resta
fra quella retta e il bordo dell'immagine non è raggiungibile a nessuna
profondità: con filtri $3 \times 3$ il punto cieco arriva a coprire, dicono gli
autori, «fino a un quarto del campo recettivo potenziale». La riparazione
di {cite}`oord2016conditional` fattorizza il filtro in **due pile**: la
verticale, non mascherata, sulle righe strettamente superiori (campo
rettangolare, nessun punto cieco), e l'orizzontale, mascherata, sulla riga
corrente; l'uscita della verticale entra nell'orizzontale con una $1\times1$, e
i due rami si sommano dopo ogni blocco. Lo stesso lavoro sostituisce la ReLU
con un'unità *gated* in stile LSTM, e aggiunge il condizionamento su un vettore
esterno, che è il motivo per cui il titolo parla di generazione
**condizionale**. Le due modifiche insieme portano il PixelCNN, su CIFAR-10, da
$3{,}14$ a $3{,}03$ bit per dimensione (quanti bit costa in media ogni numero
dell'immagine: più basso è meglio), a un soffio dal $3{,}00$ del PixelRNN, che
però è il modello lento.

Fra i successori, **PixelCNN++** {cite}`salimans2017pixelcnn` sostituisce la
categorica su 256 livelli con una **miscela di logistiche discretizzate**: per
una categorica il livello 128 e il 129 sono due simboli senza alcuna relazione,
mentre una miscela continua e poi discretizzata recupera l'ordinamento dei
valori; gli autori riportano che così l'addestramento accelera, e il conto su
CIFAR-10 scende a $2{,}92$.

`````

## Perché non ha vinto sulle immagini, e dove è tornata

Il conto è impietoso, ed è tutto nel campionamento. Valutare la probabilità di
un'immagine costa **un** passaggio della rete; generarne una ne costa uno per
pixel, in fila, perché il pixel numero mille ha bisogno che il
novecentonovantanovesimo sia già stato deciso. Su una fotografia a colori di
$256 \times 256$ sono $256 \times 256 \times 3 = 196.608$ passaggi sequenziali
per una sola immagine,
contro l'unico passaggio di una GAN. Non è una differenza di efficienza: è una
differenza di categoria.

Ed è per questo che l'idea, sulle immagini, è tornata da un'altra porta.
Nessuna legge dice che i pezzi da mettere in fila debbano essere i pixel. Se
prima si comprime l'immagine in una griglia piccola di simboli presi da un
catalogo (il VQ-GAN del capitolo sulle GAN: $256 \times 256$ diventano $16
\times 16$, cioè 256 simboli), allora i passaggi sequenziali da 196.608
diventano 256, settecentosessantotto volte meno, e a metterli in fila può
pensare un Transformer. L'autoregressione sulle immagini non è morta: si è
spostata di un piano, dai pixel ai token, ed è la forma in cui oggi si trova
dentro i modelli che disegnano e parlano.

Resta però una cosa che si perde in quel trasloco, e riguarda proprio questo
capitolo: la verosimiglianza che si calcola sui token è quella dei **token**,
non quella dell'immagine. Il passaggio dal catalogo ai pixel è una perdita, e
oltre quella perdita il numero non parla più. Chi vuole $\log p$ dell'immagine
vera deve restare sui pixel, o cambiare famiglia: ed è la sezione che segue.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- È la ricetta del testo, applicata a una griglia. Si sceglie un **ordine**
  (riga per riga, come si legge), e la probabilità di un'immagine è il prodotto
  di quella di ogni pixel dato tutto quello che viene prima. Nessuna
  approssimazione, e per misurarla basta un passaggio solo.
- Il costo è mettere il bavaglio alla convoluzione: si **spengono** le caselle
  del filtro che guardano nel futuro. Al primo strato si spegne anche quella
  centrale, perché lì c'è il pixel che stiamo cercando di indovinare, e vederlo
  sarebbe copiare la risposta dalla domanda.
- Il bavaglio si porta dietro un guasto che nessuno aveva previsto: un triangolo
  di pixel che vengono **prima** e che la rete non guarderà mai, per quanto la
  si faccia profonda. Si chiama **punto cieco**, e non si cura con la
  profondità: si cura cambiando la forma della finestra, cioè mettendone due.
- Generare costa carissimo, perché va fatto un pixel alla volta e in fila: su
  una fotografia sono quasi duecentomila passaggi, contro l'unico di una GAN.
  Per questo oggi la stessa idea si applica a pezzi più grossi dei pixel (i
  simboli di catalogo del VQ-GAN), dove i passaggi diventano poche centinaia.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- $\log p(\mathbf{x}) = \sum_i \log p(x_i \mid \mathbf{x}_{<i})$ su un
  ordinamento totale dei pixel, ogni fattore una categorica su 256 livelli:
  normalizzazione per costruzione, verosimiglianza **esatta**, valutazione in un
  passaggio (*teacher forcing*), campionamento in $\mathcal{O}(n^2)$ passaggi
  sequenziali.
- Causalità imposta **sui pesi**: $\mathbf{W} \leftarrow \mathbf{W} \odot
  \mathbf{M}$, maschera di tipo A al primo strato (azzera il centro) e di tipo
  B dopo. La causalità è chiusa per composizione, e il test sul gradiente la
  verifica invece di darla per buona.
- Il **punto cieco** {cite}`oord2016conditional` è un artefatto della
  realizzazione, non del vincolo: il bordo destro del campo recettivo sale a
  $45^\circ$ invece di coprire il semipiano che l'ordinamento consentirebbe, e
  con filtri $3 \times 3$ arriva a un quarto del campo potenziale.
  Riparazione: fattorizzare in pila **verticale** (non mascherata) e
  **orizzontale** (mascherata), sommate dopo ogni blocco; su CIFAR-10, con le
  unità *gated*, da $3{,}14$ a $3{,}03$ bit per dimensione.
- **PixelCNN++** {cite}`salimans2017pixelcnn` sostituisce la softmax a 256 vie
  con una **miscela di logistiche discretizzate**, che recupera l'ordinamento
  fra livelli adiacenti perso dalla categorica: $2{,}92$ su CIFAR-10.
- Il collo di bottiglia è il campionamento sequenziale. Spostando
  l'autoregressione dai pixel a **token discreti** (VQ-GAN) si passa da
  $196.608$ a $256$ passaggi, al prezzo di una verosimiglianza che è quella dei
  token e non dell'immagine.
```

`````
