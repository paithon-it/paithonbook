# Il flusso che si può invertire

La sezione precedente ha ottenuto la probabilità esatta spezzettando: tanti
pezzi, ognuno col suo voto, e il prodotto. Questa la ottiene senza spezzare
niente, con una mossa che a prima vista sembra un imbroglio.

L'idea è questa. Non proviamo a descrivere la distribuzione dei dati, che è
complicatissima. Proviamo invece a costruire una macchina che li raddrizza:
prende una fotografia e la porta in un punto di una nuvola semplice, una
gaussiana, dove sappiamo dire tutto perché la formula ce l'abbiamo. Se quella
macchina si può usare anche al contrario, allora abbiamo due cose in un colpo
solo. Per generare: si pesca un punto a caso nella gaussiana e lo si fa
tornare indietro, e quello che esce è una fotografia. Per valutare: si
manda avanti la fotografia, si legge quanto è probabile il punto in cui è
finita, e si corregge di un fattore che diremo fra poco.

Questa famiglia si chiama dei **flussi normalizzanti**, e le due parole vanno
prese una per volta. «Normalizzante» qui non vuol dire «che fa venire uno»:
vuol dire «che porta verso la normale», cioè verso la gaussiana. «Flusso» è il
paragone con un fluido, perché la nuvola dei dati non salta da una forma
all'altra, si sposta un poco alla volta, una trasformazione dopo l'altra.

Resta da fissare un verso, perché i due sensi hanno nomi diversi e si
confondono con facilità. La definizione è quella di Danilo Rezende e Shakir
Mohamed {cite}`rezende2015variational`, «una densità iniziale semplice
trasformata in una più complessa applicando una successione di trasformazioni
invertibili finché non si raggiunge la complessità desiderata», ed è scritta
nel senso della generazione: dalla gaussiana ai dati. A dare il nome è il senso
opposto, quello che si percorre per calcolare la probabilità, dove sono le
trasformazioni inverse a portare i dati verso la normale. L'idea e la parola
vengono da prima, dai lavori di Esteban Tabak e colleghi sulla stima di densità
{cite}`tabak2010density,tabak2013family`, che sono esattamente quelli che
Rezende e Mohamed citano quando introducono il principio.

E qui va detta subito la cosa che chiude un debito col capitolo precedente:
quel «flusso» è la stessa parola del *rectified flow* di Stable Diffusion 3.
È una parentela e non una coincidenza di vocabolario, e in fondo alla prossima
sezione la ricostruiremo per intero.

## Il fattore che nessuno si aspetta

Prima della macchina, la matematica, che è tutta qui e sta in una riga.

Prendiamo una cosa semplicissima: una grandezza che sta fra 0 e 1, distribuita
in modo uniforme. La sua densità vale 1 dappertutto lì dentro, e l'area sotto
la curva fa 1, come dev'essere. Densità e probabilità non sono sinonimi, e
conviene tenerle separate da subito. La densità è quanto è alta la curva in un
punto, la probabilità è l'area che sta sotto la curva in un tratto. Adesso la
stiriamo: la moltiplichiamo per tre e le aggiungiamo uno, così finisce fra 1 e
4. È la stessa grandezza, non abbiamo buttato via niente e non abbiamo aggiunto
niente. Ma il tavolo su cui è stesa è diventato tre volte più largo, e la
stessa quantità d'acqua su un tavolo tre volte più largo sta tre volte più
bassa. La densità di arrivo non vale 1: vale un terzo.

Questo è il punto che rende i flussi diversi da tutto il resto del capitolo.
Senza quel fattore l'area sotto la curva non fa più uno, e un numero la cui
area non fa uno non è una probabilità. Ed è qui che l'imbroglio si
scioglie: la probabilità esatta senza spezzare niente si ottiene davvero, a
patto di pagare questo fattore.
Verifichiamolo, perché è il genere di cosa che si capisce meglio vedendola.

```python
import numpy as np

rng = np.random.default_rng(0)

# x e' uniforme fra 0 e 1: la sua densita' vale 1 dappertutto li' dentro, e
# infatti l'area sotto la curva fa 1. Adesso stiriamo: y = 3x + 1, lo stesso
# intervallo disteso su una lunghezza tripla. La quantita' d'acqua non cambia,
# il tavolo si allarga.
x = rng.random(2_000_000)
y = 3 * x + 1

bordi_x = np.linspace(0, 1, 61)                    # 60 caselle su [0, 1]
prima, _ = np.histogram(x, bins=bordi_x, density=True)
bordi = np.linspace(1, 4, 61)                      # 60 caselle su [1, 4]
dopo, _ = np.histogram(y, bins=bordi, density=True)

print(f"densita' di x (uniforme su [0,1]), misurata: {prima.mean():.3f}")
print(f"densita' di y (uniforme su [1,4]), misurata: {dopo.mean():.3f}")
print(f"rapporto fra le due: {prima.mean() / dopo.mean():.2f}"
      f"  <- e' lo stiramento, 3")
print()
print(f"area sotto la densita' di y, col fattore:   "
      f"{(dopo * np.diff(bordi)).sum():.3f}")
print(f"area sotto la densita' di y, senza fattore: "
      f"{(prima.mean() * np.diff(bordi)).sum():.3f}"
      f"  <- non e' una probabilita'")
```

```text
densita' di x (uniforme su [0,1]), misurata: 1.000
densita' di y (uniforme su [1,4]), misurata: 0.333
rapporto fra le due: 3.00  <- e' lo stiramento, 3

area sotto la densita' di y, col fattore:   1.000
area sotto la densita' di y, senza fattore: 3.000  <- non e' una probabilita'
```

In una dimensione il fattore è lo stiramento, cioè di quanto la trasformazione
allunga o accorcia. Attenzione al verso, perché è la trappola, e la trappola
non sta nello stiramento: sta in quale delle due densità si sta chiedendo. Sul
tavolo abbiamo chiesto la densità di ciò che esce, e stirando è scesa; qui si
chiede quella di ciò che entra, cioè della fotografia, mentre la macchina la
stira verso la gaussiana, e allora quel fattore va moltiplicato. In molte
dimensioni la trasformazione può allungare in una direzione, accorciare in
un'altra e ruotare il tutto, e allora il fattore giusto è quello che dice di
quante volte è cambiato il volume: il determinante della tabella delle
derivate, la *jacobiana*.

`````{tab} Elementare

Il determinante l'abbiamo già incontrato nei {doc}`richiami di matematica
</Matematica/determinante-e-volume>`: dice di quante volte una trasformazione
cambia l'area di un quadratino, il volume di un cubetto, e in mille dimensioni
la stessa cosa senza più niente da disegnare. E passando per due
trasformazioni in fila le variazioni si moltiplicano: un'area che triplica e
poi raddoppia è cresciuta sei volte.

Qui però cambia una cosa, ed è la sola da portarsi dietro. Là il quadretto
poteva stare dovunque sul foglio e il numero era sempre quello, perché lo
stiramento era lo stesso dappertutto. Una macchina che raddrizza le fotografie
non è così: schiaccia in un punto e allarga in quello a fianco. Il numero è
quello del punto in cui si sta guardando, e cambia se ci si sposta.

La regola dei flussi si legge allora in italiano, senza formule: *la densità di
una fotografia è la densità del punto dove la fotografia finisce, moltiplicata
per quanto la macchina ha stirato lo spazio lì attorno*.

Il «moltiplicata» sorprende, ed è il punto in cui tutti si sbagliano. La
macchina lavora nel verso che porta le fotografie sulla gaussiana. Se prende un
pezzetto piccolo di fotografie e lo stira su una zona grande della gaussiana,
quel pezzetto si porta a casa tutta l'acqua di quella zona e la tiene in poco
posto: lì l'acqua è alta, e quelle fotografie sono probabili. Se invece lo
schiaccia in un puntino, si accontenta dell'acqua di un puntino, e lì le
fotografie sono rare.

Nel verso della generazione la macchina è l'inversa, e la regola si capovolge
con lei: là si divide, e una zona schiacciata diventa probabile.

C'è però una cosa che una macchina così non può fare, ed è buttare via. Se
all'andata lasciasse per strada anche un solo numero, al ritorno dovrebbe
inventarselo, e quella che esce non sarebbe più la fotografia che era entrata.
Per questo un flusso esce con esattamente tanti numeri quanti ne sono entrati.

E adesso il guaio. Calcolare un determinante costa, e costa tantissimo: il
conto generale vuole all'incirca tante operazioni quanto è il cubo del lato
della tabella, quindi per mille righe per mille colonne siamo a mille per mille
per mille, cioè un miliardo di operazioni, da rifare per ogni immagine e a ogni
passo dell'addestramento. E mille si raggiunge subito, perché la tabella ha una
riga e una colonna per ogni numero dell'immagine: mille numeri sono una
figurina di 32 pixel per lato in bianco e nero. Su una fotografia vera non se
ne parla nemmeno.

`````

`````{tab} Superiore

Il cambio di variabile per una $f: \mathbb{R}^D \to \mathbb{R}^D$
diffeomorfa, con $\mathbf{z} = f(\mathbf{x})$:

$$
p_X(\mathbf{x}) = p_Z\big(f(\mathbf{x})\big)\,
\left\lvert \det \frac{\partial f}{\partial \mathbf{x}}(\mathbf{x}) \right\rvert,
\qquad\text{cioè}\qquad
\log p_X(\mathbf{x}) = \log p_Z\big(f(\mathbf{x})\big)
+ \log \left\lvert \det \mathbf{J}_f(\mathbf{x}) \right\rvert .
$$

Qui $\mathbf{J}_f(\mathbf{x})$ è la jacobiana di $f$ in $\mathbf{x}$, cioè la
tabella delle derivate parziali $(\mathbf{J}_f)_{ij} = \partial f_i/\partial
x_j$, e $p_Z$ è la densità di base, che nel seguito è una gaussiana standard.
Il valore assoluto serve perché il determinante è con segno (dice anche se
la trasformazione ribalta l'orientamento) mentre a noi interessa solo il
rapporto fra i volumi. Componendo più trasformazioni i logaritmi si sommano,
ed è la ragione per cui in pratica si scrive tutto in scala logaritmica: una
successione $f = f_K \circ \dots \circ f_1$ dà

$$
\log\lvert\det \mathbf{J}_f(\mathbf{x})\rvert
= \sum_{k=1}^{K} \log\lvert\det \mathbf{J}_{f_k}(\mathbf{h}_{k-1})\rvert,
\qquad \mathbf{h}_0 = \mathbf{x}, \quad \mathbf{h}_k = f_k(\mathbf{h}_{k-1}),
$$

dove ogni jacobiana si valuta nel punto in cui quel passo lavora, e non in
$\mathbf{x}$: è la stessa dipendenza dal punto che ha la formula a un passo
solo, scritta per una composizione.

Due vincoli cadono da qui, ed entrambi pesano.

Il primo è che $f$ dev'essere invertibile, quindi in particolare
$D_{\text{ingresso}} = D_{\text{uscita}}$: un flusso non riduce la dimensione,
mai. Il secondo è il costo: il determinante di una matrice $D \times D$ costa
$\mathcal{O}(D^3)$ con i metodi generali, e $D$ qui è il numero di pixel per
canali. Per $32 \times 32$ in scala di grigi, $D = 1024$ e il conto è
$\approx 10^9$ operazioni per esempio per passo, con l'aggravante che serve
anche il gradiente di quel determinante. Impraticabile.

`````

## Il trucco: metà ferma, metà mossa

La via d'uscita è cambiare la domanda. Invece di calcolare in fretta il
determinante di una matrice qualunque, si costruisce la trasformazione in modo
che il suo determinante sia già scritto.

La ricetta si chiama **strato di accoppiamento**, e il gesto è questo: si
spaccano le coordinate in due metà. La prima metà passa intatta, non la si
tocca. La seconda metà viene scalata e traslata, e i numeri con cui la si scala
e la si trasla sono decisi dalla prima metà, quella che è passata intatta.
Poi si scambiano i ruoli e si ripete, così che tutte le coordinate prima o poi
vengano trasformate e prima o poi facciano da guida.

L'idea è del 2014, di NICE {cite}`dinh2015nice`, dove però la seconda metà
veniva soltanto traslata e non scalata: una traslazione non cambia i volumi,
quindi quegli strati il fattore non lo toccavano affatto, e a cambiarlo restava
un solo strato di scala in cima alla pila. La scala, che è quella che rende il
fattore interessante, arriva con RealNVP {cite}`dinh2017density`, ed è la forma
che si usa oggi e che il flusso sulle due lune mette in pratica.

Tre proprietà cadono tutte insieme, ed è per questo che la ricetta ha vinto.

Si inverte a occhio. Per tornare indietro serve sapere con che cosa si è
scalato e traslato, e quei numeri dipendono solo dalla prima metà, che è
arrivata intatta: si legge, si ricalcolano scala e traslazione, si disfa. Non
serve invertire nessuna rete.

Il determinante è gratis. La prima metà non cambia, quindi la parte
corrispondente della tabella delle derivate è l'identità; la seconda metà
dipende dalla prima in un modo complicatissimo, ma quel blocco della tabella
sta tutto da una parte della diagonale, e dall'altra parte ci sono soltanto
zeri. Una tabella fatta così si chiama *triangolare*, e il suo determinante è
il prodotto di quello che sta *sulla* diagonale, senza nessun altro conto. Cioè:
il determinante è il prodotto delle scale, che sono numeri che abbiamo già in
mano. Da un miliardo di operazioni a una moltiplicazione per ogni coordinata
scalata, e le coordinate scalate sono metà: sulla figurina di 32 pixel per lato
di poco fa, cinquecentododici invece di un miliardo.

La rete che decide non ha vincoli. Ed è il punto più bello, quello che
sfugge a una prima lettura: la rete che, guardando la prima metà, produce scala
e traslazione non deve essere invertibile, e infatti non lo è. Può essere
qualunque cosa, profonda quanto si vuole, con le funzioni di attivazione che si
vogliono. L'invertibilità del flusso non sta nel pezzo che impara: sta nel modo
in cui i pezzi sono montati.

## Un flusso vero, sulle due lune

Mettiamolo alla prova su due lune, il banco di prova della {doc}`sezione sulle
SVM a kernel </MachineLearning/svm-kernel>`: due archi intrecciati, una forma
che nessuna retta separa e nessuna gaussiana descrive. Addestriamo un flusso a
raddrizzarli, con una loss sola, la verosimiglianza. Poi facciamo le tre
domande che contano: si inverte davvero? È davvero una densità? E sa
distinguere le lune dal resto del piano?

```python
import math

import torch
import torch.nn as nn
from sklearn.datasets import make_moons

torch.manual_seed(0)


class Accoppiamento(nn.Module):
    """Una coordinata passa intatta; l'altra viene scalata e traslata in
    funzione della prima.

    Il jacobiano e' triangolare per costruzione, quindi il suo determinante e'
    il prodotto degli elementi sulla diagonale: qui una sola scala, che il
    passo restituisce insieme al risultato. La rete che decide scala e
    traslazione NON deve essere invertibile, e infatti non lo e'.
    """

    def __init__(self, scambia, nascosto=64):
        super().__init__()
        self.scambia = scambia
        self.rete = nn.Sequential(nn.Linear(1, nascosto), nn.Tanh(),
                                  nn.Linear(nascosto, nascosto), nn.Tanh(),
                                  nn.Linear(nascosto, 2))

    def _st(self, fissa):
        s, t = self.rete(fissa.unsqueeze(1)).chunk(2, dim=1)
        return torch.tanh(s).squeeze(1), t.squeeze(1)   # tanh: scale sane

    def _ricomponi(self, fissa, mobile):
        return (torch.stack([mobile, fissa], 1) if self.scambia
                else torch.stack([fissa, mobile], 1))

    def avanti(self, x):                      # dati -> latente
        fissa, mobile = (x[:, 1], x[:, 0]) if self.scambia else (x[:, 0], x[:, 1])
        s, t = self._st(fissa)
        return self._ricomponi(fissa, mobile * torch.exp(s) + t), s

    def indietro(self, z):                    # latente -> dati
        fissa, mobile = (z[:, 1], z[:, 0]) if self.scambia else (z[:, 0], z[:, 1])
        s, t = self._st(fissa)
        return self._ricomponi(fissa, (mobile - t) * torch.exp(-s))


class Flusso(nn.Module):
    """Sei accoppiamenti a turni alterni: cosi' ogni coordinata viene
    trasformata e ogni coordinata fa da guida."""

    def __init__(self, n=6):
        super().__init__()
        self.passi = nn.ModuleList(Accoppiamento(i % 2 == 1) for i in range(n))

    def avanti(self, x):
        logdet = torch.zeros(len(x))
        for p in self.passi:
            x, s = p.avanti(x)
            logdet = logdet + s
        return x, logdet

    def indietro(self, z):
        for p in reversed(self.passi):
            z = p.indietro(z)
        return z

    def log_densita(self, x):
        """Il cambio di variabile, scritto: log p(x) = log p(z) + log|det|."""
        z, logdet = self.avanti(x)
        log_gauss = -0.5 * (z ** 2).sum(1) - math.log(2 * math.pi)
        return log_gauss + logdet


X, _ = make_moons(2000, noise=0.06, random_state=0)
X = torch.tensor(X, dtype=torch.float32)
X = (X - X.mean(0)) / X.std(0)

flusso = Flusso()
opt = torch.optim.Adam(flusso.parameters(), lr=3e-3)
for passo in range(1500):
    perdita = -flusso.log_densita(X).mean()          # verosimiglianza, e basta
    opt.zero_grad(); perdita.backward(); opt.step()

# --- Prova 1: e' davvero invertibile? Andata e ritorno, e si controlla.
with torch.no_grad():
    z, _ = flusso.avanti(X)
    errore = (flusso.indietro(z) - X).abs().max().item()
print(f"errore massimo andata e ritorno: {errore:.2e}")

# --- Prova 2: e' davvero una densita'? Si integra su una griglia fitta.
# In due dimensioni la quadratura si puo' ancora fare, e la rifacciamo due
# volte sugli stessi pesi, con il fattore |det| e senza: e' il modo di vedere
# che non e' una rifinitura.
g = torch.linspace(-6, 6, 601)
gx, gy = torch.meshgrid(g, g, indexing="ij")
griglia = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
area = (g[1] - g[0]) ** 2
with torch.no_grad():
    z_g, logdet_g = flusso.avanti(griglia)
    log_gauss_g = -0.5 * (z_g ** 2).sum(1) - math.log(2 * math.pi)
print(f"integrale della densita', col fattore:   "
      f"{((log_gauss_g + logdet_g).exp().sum() * area).item():.4f}")
print(f"integrale della densita', senza fattore: "
      f"{log_gauss_g.exp().sum().mul(area).item():.4f}")

# --- Prova 3: la densita' distingue le lune dal resto del piano?
fuori = torch.rand(2000, 2) * 8 - 4
with torch.no_grad():
    print(f"log-densita' media sulle lune:  {flusso.log_densita(X).mean():.2f}")
    print(f"log-densita' media a caso:      {flusso.log_densita(fuori).mean():.2f}")
```

```text
errore massimo andata e ritorno: 1.40e-06
integrale della densita', col fattore:   1.0000
integrale della densita', senza fattore: 0.2394
log-densita' media sulle lune:  -1.31
log-densita' media a caso:      -169.04
```

Le cinque righe vanno lette una per una, perché nessuna è scontata.

La prima è la prova che la macchina si usa nei due sensi: andata e ritorno
riportano al punto di partenza con un errore di poco più di un milionesimo, che
è il rumore dei numeri a trentadue bit e non un'approssimazione del metodo. La
seconda e la terza sono quelle che importano alla verosimiglianza esatta, e
vanno lette insieme: la densità del modello, integrata numericamente su una
griglia che la copre tutta, fa uno, e non perché qualcuno l'abbia normalizzata
a mano. Fa uno perché il cambio di variabile lo garantisce, e la riga dopo lo
mostra togliendo il fattore dagli stessi identici pesi: senza, l'area scende a
0,24, e un numero la cui area non fa uno non è una probabilità. In quel numero
non c'è niente da leggere, dipende dal flusso che è uscito da questo
addestramento; l'unica cosa che conta è che non faccia uno. Fare uno è
esattamente la differenza fra questa famiglia e quella del {doc}`capitolo sui
modelli a energia </ModelliEnergia/overview>`, dove quel conto non si può fare
e tutto il capitolo gira attorno a come evitarlo. Le ultime due dicono che il
modello ha imparato dov'è la roba: sulle lune assegna circa $-1{,}3$, su punti
presi a caso nel quadrato circa $-169$. Fra i due ci sono quasi
centosessantotto nat, e sono logaritmi: in scala normale vuol dire che la
densità tipica sulle lune sta più di settanta ordini di grandezza sopra quella
di un punto pescato a caso.

Quello che le cinque righe dicono con i numeri si può anche guardare. La
{numref}`fig-flusso-lune` segue gli stessi punti mentre attraversano i sei
accoppiamenti, e fa vedere la cosa che nessun numero stampato mostra. A ogni
passo si muove una sola delle due coordinate, e l'altra resta esattamente
dov'è, che è il vincolo da cui il determinante viene gratis. Nel mezzo la
nuvola si allarga a più del doppio, e poi si richiude sulla gaussiana.

```{figure} ../figures/lune-si-raddrizzano.svg
:name: fig-flusso-lune
:alt: Una nuvola di punti dentro un riquadro con un reticolo di riferimento. All'inizio i punti disegnano due archi intrecciati, le due lune, uno in un colore e uno nell'altro. A ogni passo l'intera nuvola si deforma, ma si sposta lungo una sola direzione per volta: prima solo in verticale, poi solo in orizzontale, e così alternando per sei passi. A metà strada la nuvola si allarga fino a occupare quasi tutto il riquadro, poi si richiude. Alla fine gli archi non ci sono più e i punti formano una macchia tonda centrata sull'origine, con i due colori mescolati. Due righe di testo sotto il riquadro dicono, a ogni passo, quale delle due direzioni si sta muovendo e quanto è larga la nuvola nei due sensi.
:width: 95%

Le due lune diventano una gaussiana in sei accoppiamenti. A ogni passo si muove
una sola coordinata, e la larghezza di quella ferma non cambia; l'altra
intanto si allarga, si stringe, e alla fine vale uno su tutti e due gli assi.
```

## Glow, e il limite che non si toglie

Fra l'accoppiamento e i modelli che hanno fatto notizia c'è un passo, e lo fa
**Glow** {cite}`kingma2018glow`. Il problema che risolve è piccolo e concreto:
scambiare le due metà a turni alterni è una scelta fissa, decisa da chi
progetta, e con tante coordinate le scelte fisse costano. Glow la sostituisce
con una **convoluzione invertibile $1 \times 1$**, che è il modo elegante di
dire «una permutazione appresa, anzi qualcosa di più generale di una
permutazione, e comunque una tabella che si sa invertire e di cui si sa
calcolare il determinante». Il guadagno lo misurano gli autori confrontando i
due modelli interi: su CIFAR-10 il costo passa dai $3{,}49$ bit per dimensione
di RealNVP ai $3{,}35$ di Glow, che oltre alla convoluzione cambia anche la
normalizzazione interna e il modo di dividere le coordinate. Con la stessa
ricetta escono i volti a $256 \times 256$ del 2018, quelli che si trasformano
l'uno nell'altro tirando una riga nello spazio latente.

E qui va detto come è andata a finire. I flussi, sulle
immagini, hanno perso, e non per un dettaglio di ingegneria: per il vincolo di
partenza. Una trasformazione invertibile conserva la dimensione, quindi un
flusso su fotografie di $512 \times 512$ a colori deve finire con 786.432
numeri, tanti quanti ne sono entrati, senza poterne buttare via uno.
Confrontalo con la diffusione latente del capitolo precedente, che sulla stessa
fotografia di numeri ne muove 16.384 perché ha il permesso di comprimere prima:
quarantotto volte meno, ed è lo stesso fattore 48 che quel capitolo aveva già
contato. Quel permesso i flussi non ce l'hanno per costruzione. È il prezzo
dell'esattezza, scritto nella definizione stessa della famiglia.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un flusso non prova a descrivere i dati: costruisce una macchina che li
  raddrizza, portandoli su una nuvola semplice di cui sappiamo tutto. Se la
  macchina si usa nei due sensi, generare è pescare un punto nella nuvola e
  farlo tornare indietro.
- Chi deforma lo spazio deve pagare un fattore di correzione: la stessa
  acqua su un tavolo tre volte più largo sta tre volte più bassa. Senza quel
  fattore l'area sotto la curva non fa più uno, e un numero la cui area non fa
  uno non è una probabilità. Il conto sul tavolo allargato di tre volte lo
  mostra: con il fattore l'area fa 1,000, senza farebbe 3.
- Il fattore, in molte dimensioni, costa un'eternità da calcolare. Il trucco è
  costruire la macchina in modo che sia già scritto: metà delle coordinate
  passano intatte, l'altra metà viene scalata e traslata in base alla prima. E
  il pezzo che decide come, cioè la rete che impara, non ha nessun vincolo:
  l'invertibilità sta nel montaggio, non nel motore.
- Il prezzo lo si paga sulla taglia: una macchina che si usa nei due sensi non
  può buttare via niente, quindi non può comprimere. Sulle fotografie, dove
  comprimere è tutto, questa famiglia ha perso la corsa. Il numero esatto che
  restituisce, però, serve ancora, ed è la prossima sezione.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Cambio di variabile: $\log p_X(\mathbf{x}) = \log p_Z(f(\mathbf{x})) +
  \log\lvert\det \mathbf{J}_f(\mathbf{x})\rvert$, con $f$ diffeomorfa e
  scritta nel verso che porta i dati al latente; scritta nel verso opposto lo
  stesso termine cambia segno. I logaritmi dei determinanti si sommano lungo
  la composizione, ciascuno valutato nel punto in cui il suo passo lavora.
- Due vincoli: $f$ conserva la dimensione, e $\det \mathbf{J}$ costa
  $\mathcal{O}(D^3)$ in generale.
- Strato di accoppiamento: partizione $\mathbf{x} = (\mathbf{x}_a,
  \mathbf{x}_b)$, con $\mathbf{z}_a = \mathbf{x}_a$ e $\mathbf{z}_b =
  \mathbf{x}_b \odot \exp(s(\mathbf{x}_a)) + t(\mathbf{x}_a)$, con $\exp$
  elemento per elemento. La forma
  additiva ($s \equiv 0$) è di NICE {cite}`dinh2015nice` ed è a volume
  costante, $\det \mathbf{J} = 1$; la scala è di RealNVP
  {cite}`dinh2017density`. La jacobiana è triangolare a blocchi con identità in
  alto a sinistra, quindi $\log\lvert\det\rvert = \sum_{i \in b}
  s_i(\mathbf{x}_a)$, perché $s$ è il *logaritmo* della scala: costo lineare. L'inversa è esplicita, e $s, t$ possono essere reti arbitrarie
  e non invertibili.
- Glow {cite}`kingma2018glow` sostituisce la permutazione fissa fra i due
  blocchi con una convoluzione $1\times1$ invertibile, il cui determinante
  costa $\mathcal{O}(c^3)$ nei soli canali (e si abbatte ulteriormente con la
  parametrizzazione LU).
- Il limite strutturale è la conservazione della dimensione: nessun
  collo di bottiglia, quindi nessuna compressione. Su $512\times512\times3$
  sono $786.432$ dimensioni anche in uscita, contro le $16.384$ del latente
  di Stable Diffusion (il fattore 48 già contato nel capitolo precedente). È la
  ragione per cui la famiglia è marginale nella
  generazione di immagini e resta viva nella stima di densità, nell'inferenza
  variazionale e come base teorica dei metodi continui.
```

`````
