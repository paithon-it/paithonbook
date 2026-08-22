# Il flusso che si può invertire

La sezione precedente ha ottenuto la probabilità esatta spezzettando: tanti
pezzi, ognuno col suo voto, e il prodotto. Questa la ottiene senza spezzare
niente, con una mossa che a prima vista sembra un imbroglio.

L'idea è questa. Non proviamo a descrivere la distribuzione dei dati, che è
complicatissima. Proviamo invece a costruire una **macchina che li raddrizza**:
prende una fotografia e la porta in un punto di una nuvola semplice, una
gaussiana, dove sappiamo dire tutto perché la formula ce l'abbiamo. Se quella
macchina si può usare anche al contrario, allora abbiamo due cose in un colpo
solo. Per **generare**: si pesca un punto a caso nella gaussiana e lo si fa
tornare indietro, e quello che esce è una fotografia. Per **valutare**: si
manda avanti la fotografia, si legge quanto è probabile il punto in cui è
finita, e si corregge di un fattore che diremo fra poco.

Questa famiglia si chiama dei **flussi normalizzanti**. Il nome dice il mestiere
in due parole: una successione di trasformazioni che porta i dati verso la
normale, cioè la gaussiana. Nella forma in cui si trova nell'apprendimento
automatico la definizione è quella di Danilo Rezende e Shakir Mohamed
{cite}`rezende2015variational`, «una densità iniziale semplice trasformata in
una più complessa applicando una successione di trasformazioni invertibili
finché non si raggiunge la complessità desiderata». Il nome e il principio però
vengono da prima, dai lavori di Esteban Tabak e colleghi sulla stima di densità
{cite}`tabak2010density,tabak2013family`, che sono esattamente quelli che
Rezende e Mohamed citano nel
momento in cui l'idea entra nelle reti neurali.

E qui va detta subito la cosa che chiude un debito col capitolo precedente:
quel «flusso» è la stessa parola del *rectified flow* di Stable Diffusion 3.
Non è una coincidenza di vocabolario, è una parentela, e in fondo alla prossima
sezione la ricostruiremo per intero.

## Il fattore che nessuno si aspetta

Prima della macchina, la matematica, che è tutta qui e sta in una riga.

Prendiamo una cosa semplicissima: una grandezza che sta fra 0 e 1, distribuita
in modo uniforme. La sua densità vale 1 dappertutto lì dentro, e l'area sotto
la curva fa 1, come dev'essere. Adesso la stiriamo: la moltiplichiamo per tre e
le aggiungiamo uno, così finisce fra 1 e 4. È la stessa grandezza, non abbiamo
buttato via niente e non abbiamo aggiunto niente. Ma il tavolo su cui è stesa è
diventato tre volte più largo, e la stessa quantità d'acqua su un tavolo tre
volte più largo sta tre volte più bassa. La densità di arrivo non vale 1: vale
un terzo.

Questo è il punto che rende i flussi diversi da tutto il resto del capitolo.
Senza quel fattore l'area sotto la curva non fa più uno, e un numero la cui
area non fa uno non è una probabilità.
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

bordi = np.linspace(1, 4, 61)                      # 60 caselle su [1, 4]
misurata, _ = np.histogram(y, bins=bordi, density=True)

print(f"densita' di x (uniforme su [0,1]), misurata: {1.0:.3f}")
print(f"densita' di y (uniforme su [1,4]), misurata: {misurata.mean():.3f}")
print(f"rapporto fra le due: {1 / misurata.mean():.2f}  <- e' lo stiramento, 3")
print()
print(f"area sotto la densita' di y, col fattore:   "
      f"{(misurata * np.diff(bordi)).sum():.3f}")
print(f"area sotto la densita' di y, senza fattore: "
      f"{(1.0 * np.diff(bordi)).sum():.3f}  <- non e' una probabilita'")
```

```text
densita' di x (uniforme su [0,1]), misurata: 1.000
densita' di y (uniforme su [1,4]), misurata: 0.333
rapporto fra le due: 3.00  <- e' lo stiramento, 3

area sotto la densita' di y, col fattore:   1.000
area sotto la densita' di y, senza fattore: 3.000  <- non e' una probabilita'
```

In una dimensione il fattore è lo stiramento, cioè di quanto la trasformazione
allunga o accorcia. In molte dimensioni la trasformazione può allungare in una
direzione, accorciare in un'altra e ruotare il tutto, e allora il fattore
giusto è quello che dice di quante volte è cambiato il **volume**. Ha un nome:
si chiama **determinante**, e qui va calcolato sulla tabella delle derivate
della trasformazione, che di suo si chiama *jacobiana*. Sono le due parole
nuove della sezione, e la seconda è soltanto il nome di una tabella.

`````{tab} Elementare

Il determinante si capisce in un disegno.

Prendi un quadratino disegnato sul tavolo, di lato uno, e applicagli la
trasformazione. Il quadratino diventa un'altra figura: magari un rettangolo
allungato, magari un rombo storto. Il determinante conta quante volte l'area è
cambiata: due se è diventata grande il doppio, un mezzo se è diventata la metà.
Quando la figura si ribalta, come allo specchio, il numero esce col meno
davanti, e del meno non ce ne facciamo niente. In tre dimensioni è il volume, e
in mille il conto è lo stesso.

E se il quadratino passa per due trasformazioni in fila, le variazioni si
moltiplicano: un'area che triplica e poi raddoppia è cresciuta sei volte.

La regola dei flussi si legge allora in italiano, senza formule: *la
probabilità di una fotografia è la probabilità del punto dove la fotografia
finisce, moltiplicata per quanto la macchina ha stirato lo spazio lì attorno*.

Il «moltiplicata» sorprende, ed è il punto in cui tutti si sbagliano. La
macchina lavora nel verso che porta le fotografie sulla gaussiana. Se prende un
pezzetto piccolo di fotografie e lo stira su una zona grande della gaussiana,
quel pezzetto si porta a casa tutta l'acqua di quella zona e la tiene in poco
posto: lì l'acqua è alta, e quelle fotografie sono probabili. Se invece lo
schiaccia in un puntino, si accontenta dell'acqua di un puntino, e lì le
fotografie sono rare.

Nel verso della generazione la macchina è l'inversa, e la regola si capovolge
con lei: là si divide, e una zona schiacciata diventa probabile. Chi gira nei
due sensi però non butta via niente per strada: quello che perde all'andata, al
ritorno dovrebbe indovinarlo.

E adesso il guaio. Calcolare un determinante costa, e costa tantissimo: per una
tabella di mille righe per mille colonne il conto generale richiede all'incirca
un miliardo di operazioni, e va rifatto **per ogni immagine e a ogni passo
dell'addestramento**. Mille righe per mille colonne è una figurina di 32 pixel
per lato in bianco e nero. Su una fotografia vera non se ne parla nemmeno.

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

Il valore assoluto serve perché il determinante è **con segno** (dice anche se
la trasformazione ribalta l'orientamento) mentre a noi interessa solo il
rapporto fra i volumi. Componendo più trasformazioni i logaritmi si sommano,
ed è la ragione per cui in pratica si scrive tutto in scala logaritmica: una
successione $f = f_L \circ \dots \circ f_1$ dà
$\log\lvert\det \mathbf{J}_f\rvert = \sum_\ell \log\lvert\det
\mathbf{J}_{f_\ell}\rvert$.

Due vincoli cadono da qui, ed entrambi pesano.

Il primo è che $f$ dev'essere **invertibile**, quindi in particolare
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
che il suo determinante sia **già scritto**.

La ricetta si chiama **strato di accoppiamento**, e il gesto è questo: si
spaccano le coordinate in due metà. La prima metà **passa intatta**, non la si
tocca. La seconda metà viene scalata e traslata, e i numeri con cui la si scala
e la si trasla sono decisi **dalla prima metà**, quella che è passata intatta.
Poi si scambiano i ruoli e si ripete, così che tutte le coordinate prima o poi
vengano trasformate e prima o poi facciano da guida.

L'idea è del 2014, di NICE {cite}`dinh2015nice`, dove però la seconda metà
veniva soltanto traslata e non scalata: una traslazione non cambia i volumi,
quindi lì il fattore di correzione valeva uno tondo e non c'era niente da
calcolare. La scala, che è quella che rende il fattore interessante, arriva con
RealNVP {cite}`dinh2017density`, ed è la forma che si usa oggi e che il flusso
sulle due lune mette in pratica.

Tre proprietà cadono tutte insieme, ed è per questo che la ricetta ha vinto.

**Si inverte a occhio.** Per tornare indietro serve sapere con che cosa si è
scalato e traslato, e quei numeri dipendono solo dalla prima metà, che è
arrivata intatta: si legge, si ricalcolano scala e traslazione, si disfa. Non
serve invertire nessuna rete.

**Il determinante è gratis.** La prima metà non cambia, quindi la parte
corrispondente della tabella delle derivate è l'identità; la seconda metà
dipende dalla prima in un modo complicatissimo, ma quel blocco della tabella
sta sotto la diagonale e il determinante di una tabella triangolare è
semplicemente il prodotto di quello che sta *sulla* diagonale. Cioè: il
determinante è il prodotto delle scale, che sono numeri che abbiamo già in
mano. Da un miliardo di operazioni a una moltiplicazione per ogni coordinata
scalata: sulla figurina di 32 pixel per lato di poco fa, cinquecento invece di
un miliardo.

**La rete che decide non ha vincoli.** Ed è il punto più bello, quello che
sfugge a una prima lettura: la rete che, guardando la prima metà, produce scala
e traslazione **non deve essere invertibile**, e infatti non lo è. Può essere
qualunque cosa, profonda quanto si vuole, con le funzioni di attivazione che si
vogliono. L'invertibilità del flusso non sta nel pezzo che impara: sta nel modo
in cui i pezzi sono montati.

## Un flusso vero, in venti righe

Mettiamolo alla prova su due lune, il banco di prova che il libro ha già usato
per le SVM a kernel: due archi intrecciati, una forma che nessuna retta separa
e nessuna gaussiana descrive. Addestriamo un flusso a raddrizzarli, con una
loss sola, la verosimiglianza. Poi facciamo le tre domande che contano: si
inverte davvero? È davvero una densità? E sa distinguere le lune dal resto del
piano?

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
print(f"log-verosimiglianza media per punto: {-perdita.item():.3f} nat")

# --- Prova 1: e' davvero invertibile? Andata e ritorno, e si controlla.
with torch.no_grad():
    z, _ = flusso.avanti(X)
    errore = (flusso.indietro(z) - X).abs().max().item()
print(f"errore massimo andata e ritorno: {errore:.2e}")

# --- Prova 2: e' davvero una densita'? Si integra su una griglia fitta.
# In due dimensioni la quadratura si puo' ancora fare, e vale come verifica
# del fatto che il fattore |det| non e' decorativo: senza, non farebbe 1.
g = torch.linspace(-6, 6, 601)
gx, gy = torch.meshgrid(g, g, indexing="ij")
griglia = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
with torch.no_grad():
    p = flusso.log_densita(griglia).exp()
area = (g[1] - g[0]) ** 2
print(f"integrale della densita' sulla griglia: {(p.sum() * area).item():.4f}")

# --- Prova 3: la densita' distingue le lune dal resto del piano?
fuori = torch.rand(2000, 2) * 8 - 4
with torch.no_grad():
    print(f"log-densita' media sulle lune:  {flusso.log_densita(X).mean():.2f}")
    print(f"log-densita' media a caso:      {flusso.log_densita(fuori).mean():.2f}")
```

```text
log-verosimiglianza media per punto: -1.307 nat
errore massimo andata e ritorno: 1.40e-06
integrale della densita' sulla griglia: 1.0000
log-densita' media sulle lune:  -1.31
log-densita' media a caso:      -169.04
```

Le cinque righe vanno lette una per una, perché ciascuna dice una cosa diversa
e nessuna è scontata.

La prima è la loss, e da sola non dice granché. La seconda è la prova che la
macchina si usa nei due sensi: andata e ritorno riportano al punto di partenza
con un errore di poco più di un milionesimo, che è il rumore dei numeri a
trentadue bit e non un'approssimazione del metodo. La terza è quella che
importa a questo capitolo: la densità del modello, integrata su tutto il piano,
fa **uno**, e non perché qualcuno l'abbia normalizzata a mano. Fa uno perché il
cambio di variabile lo garantisce, e togliendo il termine `logdet` dal codice
non farebbe più uno. È esattamente la differenza fra questa famiglia e quella
del {doc}`capitolo sui modelli a energia </ModelliEnergia/overview>`, dove quel conto non si può fare e tutto il capitolo
gira attorno a come evitarlo. Le ultime due vanno lette insieme, e dicono che
il modello ha imparato dov'è la roba: sulle lune assegna circa $-1{,}3$, su
punti presi a caso nel quadrato circa $-169$. Sono centosessantotto nat di
differenza, e siccome quei numeri sono logaritmi, in scala normale vuol dire un
rapporto di più di $10^{72}$.

## Glow, e il limite che non si toglie

Fra l'accoppiamento e i modelli che hanno fatto notizia c'è un passo, e lo fa
**Glow** {cite}`kingma2018glow`. Il problema che risolve è piccolo e concreto:
scambiare le due metà a turni alterni è una scelta fissa, decisa da chi
progetta, e con tante coordinate le scelte fisse costano. Glow la sostituisce
con una **convoluzione invertibile $1 \times 1$**, che è il modo elegante di
dire «una permutazione appresa, anzi qualcosa di più generale di una
permutazione, e comunque una tabella che si sa invertire e di cui si sa
calcolare il determinante». Il guadagno lo misurano gli autori: su CIFAR-10 il
costo passa dai $3{,}49$ bit per dimensione di RealNVP a $3{,}35$, e con la
stessa ricetta escono i volti a $256 \times 256$ del 2018, quelli che si
trasformano l'uno nell'altro tirando una riga nello spazio latente.

E qui il capitolo deve essere onesto su come è andata a finire. I flussi, sulle
immagini, hanno perso, e non per un dettaglio di ingegneria: per il vincolo di
partenza. Una trasformazione invertibile **conserva la dimensione**, quindi un
flusso su fotografie di $512 \times 512$ a colori deve muovere 786.432 numeri
dall'inizio alla fine, senza mai poterne buttare via uno. Confrontalo con la
diffusione latente del capitolo precedente, che sulla stessa fotografia di
numeri ne muove 16.384 perché ha il permesso di comprimere prima: quarantotto
volte meno, ed è lo stesso fattore 48 che quel capitolo aveva già contato.
Quel permesso i flussi non ce l'hanno per costruzione. È il prezzo dell'esattezza,
scritto nella definizione stessa della famiglia.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **flusso** non prova a descrivere i dati: costruisce una macchina che li
  **raddrizza**, portandoli su una nuvola semplice di cui sappiamo tutto. Se la
  macchina si usa nei due sensi, generare è pescare un punto nella nuvola e
  farlo tornare indietro.
- Chi deforma lo spazio deve pagare un **fattore di correzione**: la stessa
  acqua su un tavolo tre volte più largo sta tre volte più bassa. Senza quel
  fattore l'area sotto la curva non fa più uno, e un numero la cui area non fa
  uno non è una probabilità. Il conto sul tavolo allargato di tre volte lo
  mostra: con il fattore l'area fa 1,000, senza farebbe 3.
- Il fattore, in molte dimensioni, costa un'eternità da calcolare. Il trucco è
  costruire la macchina in modo che sia **già scritto**: metà delle coordinate
  passano intatte, l'altra metà viene scalata e traslata in base alla prima. E
  il pezzo che decide come, cioè la rete che impara, **non ha nessun vincolo**:
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
- **Cambio di variabile**: $\log p_X(\mathbf{x}) = \log p_Z(f(\mathbf{x})) +
  \log\lvert\det \mathbf{J}_f(\mathbf{x})\rvert$, con $f$ diffeomorfa. I
  logaritmi dei determinanti si sommano lungo la composizione.
- Due vincoli: $f$ conserva la dimensione, e $\det \mathbf{J}$ costa
  $\mathcal{O}(D^3)$ in generale.
- **Strato di accoppiamento**: partizione $\mathbf{x} = (\mathbf{x}_a,
  \mathbf{x}_b)$, con $\mathbf{z}_a = \mathbf{x}_a$ e $\mathbf{z}_b =
  \mathbf{x}_b \odot \exp(s(\mathbf{x}_a)) + t(\mathbf{x}_a)$. La forma
  additiva ($s \equiv 0$) è di NICE {cite}`dinh2015nice` ed è a volume
  costante, $\det \mathbf{J} = 1$; la scala è di RealNVP
  {cite}`dinh2017density`. La jacobiana è triangolare a blocchi con identità in
  alto a sinistra, quindi $\log\lvert\det\rvert = \sum_i s_i(\mathbf{x}_a)$:
  costo lineare. L'inversa è esplicita, e $s, t$ possono essere reti arbitrarie
  e non invertibili.
- **Glow** {cite}`kingma2018glow` sostituisce la permutazione fissa fra i due
  blocchi con una **convoluzione $1\times1$ invertibile**, il cui determinante
  costa $\mathcal{O}(c^3)$ nei soli canali (e si abbatte ulteriormente con la
  parametrizzazione LU).
- Il limite strutturale è la **conservazione della dimensione**: nessun
  collo di bottiglia, quindi nessuna compressione. Su $512\times512\times3$
  sono $786.432$ dimensioni da trasportare, contro le $16.384$ del latente di
  Stable Diffusion (il fattore 48 già contato nel capitolo precedente). È la
  ragione per cui la famiglia è marginale nella
  generazione di immagini e resta viva nella stima di densità, nell'inferenza
  variazionale e come base teorica dei metodi continui.
```

`````
