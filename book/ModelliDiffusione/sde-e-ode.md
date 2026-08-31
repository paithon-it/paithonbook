# Il limite continuo: una sola equazione, in avanti e all'indietro

Nel 1982 Brian Anderson, ingegnere del controllo automatico, pubblica su una
rivista di processi stocastici un articolo di quattordici pagine intitolato
*Reverse-time diffusion equation models* {cite}`anderson1982reverse`. Il
problema che aveva per le mani era di tutt'altro genere: dato un segnale
rumoroso osservato fino a un certo istante, stimare che cosa fosse successo
prima. Il risultato che ottiene è generale: **un processo casuale che si
diffonde in avanti nel tempo si può percorrere all'indietro**, e l'equazione
che lo descrive è la stessa di prima più un termine correttivo.

Quel termine correttivo dice, in ogni punto, da che parte stanno le zone dove i
dati sono più fitti: è il gradiente della log-densità. Per quasi quarant'anni
l'articolo resta un risultato tecnico citato da poche decine di lavori; poi nel
2020 qualcuno si accorge che rovinare un'immagine con del rumore è esattamente
una diffusione in avanti, che generare è percorrerla all'indietro, e che quel
termine correttivo è la sola cosa che una rete debba imparare
{cite}`song2021score`. La formula era già scritta, e aspettava soltanto che
qualcuno avesse il problema giusto.

Questa pagina fa il passaggio dai mille passi di DDPM al tempo continuo. Il
guadagno è sostanziale e non estetico: nel continuo le due tradizioni della
diffusione diventano due casi della stessa equazione, si scopre che accanto
alla strada casuale ne esiste una deterministica che produce le stesse
immagini, e le quattro cose che una rete può imparare si rivelano quattro modi
di scrivere lo stesso oggetto.

## Che cosa resta quando i passi diventano infiniti

`````{tab} Elementare

Le due ricette per rovinare un'immagine, quella che aggiunge rumore sempre più
forte e quella che a ogni passo restringe un po' l'immagine e ci aggiunge un
pizzico di rumore, sembrano diverse. Scritte una sotto l'altra hanno però la
stessa forma: **il nuovo valore è il vecchio, più uno spostamento sistematico,
più uno scossone sorteggiato**. Cambia solo quanto valgono i due pezzi.

Ora immagina di prendere quella ricetta e di applicarla non mille volte, ma un
milione, con passi mille volte più corti. Quello che si ottiene al limite è una
descrizione continua: invece di dire «a ogni passo fai così», si dice «in ogni
istante il valore deriva un po' in questa direzione e trema un po' con questa
intensità». Il primo pezzo si chiama **deriva** e dice dove il valore è
trascinato; il secondo si chiama **diffusione** e dice quanto trema.

Sul tremore serve una precisazione, perché è controintuitiva. Gli scossoni non
si accumulano come uno spostamento normale: sommando cento scossoni a caso, chi
va a destra e chi a sinistra si compensano parecchio, e quello che resta cresce
come la **radice** del numero di scossoni, non come il numero. Ecco perché
nella ricetta il tremore compare con una radice quadrata del passo, mentre la
deriva compare con il passo intero. È lo stesso fatto della legge dei grandi
numeri, visto al contrario: la media di tante prove si stabilizza come uno
diviso radice di enne, e la loro somma si sparpaglia come radice di enne.

Il gioco vale la candela per una ragione pratica: nel continuo la ricetta si
descrive con due sole funzioni del tempo, e tutte le varianti che negli anni
sono state proposte si scrivono scegliendo quelle due. Invece di una famiglia di
metodi diversi, una sola equazione con due manopole.

`````

`````{tab} Superiore

I due schemi di iniezione del rumore si scrivono, su una griglia temporale di
passo $\Delta t$, nella forma comune

$$
\mathbf{x}_{t+\Delta t} \;\approx\; \mathbf{x}_t
+ \mathbf{f}(\mathbf{x}_t, t)\,\Delta t
+ g(t)\sqrt{\Delta t}\;\boldsymbol{\epsilon}_t ,
\qquad \boldsymbol{\epsilon}_t\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
$$

con $\mathbf{f}$ la **deriva** (un vettore, funzione dello stato e del tempo) e
$g$ il **coefficiente di diffusione** (uno scalare, funzione del solo tempo).
La radice su $\Delta t$ viene dalla natura del processo, non da una scelta di
notazione: gli incrementi di un processo di Wiener hanno varianza proporzionale
al tempo, quindi deviazione standard proporzionale alla sua radice.

Per $\Delta t\to 0$ la ricorrenza converge all'equazione differenziale
stocastica **in avanti**

$$
\mathrm{d}\mathbf{x}(t) = \mathbf{f}(\mathbf{x}(t),t)\,\mathrm{d}t
+ g(t)\,\mathrm{d}\mathbf{w}(t),
\qquad \mathbf{x}(0)\sim p_{\text{dati}},
$$

dove $\mathbf{w}(t)$ è un **processo di Wiener** standard: parte da zero, ha
incrementi indipendenti, e $\mathbf{w}(t)-\mathbf{w}(s)$ è gaussiano di media
nulla e varianza $t-s$. È continuo ovunque e derivabile in nessun punto, ed è
la ragione per cui la scrittura $\mathrm{d}\mathbf{w}$ non si può dividere per
$\mathrm{d}t$.

La cosa che rende trattabile tutto il resto è la scelta di una **deriva
affine**, $\mathbf{f}(\mathbf{x},t) = f(t)\,\mathbf{x}$ con $f$ scalare. Sotto
questa ipotesi il processo condizionato al dato iniziale resta gaussiano a ogni
istante, e il **nucleo di perturbazione** ha forma chiusa:

$$
p_t(\mathbf{x}_t\mid\mathbf{x}_0)
= \mathcal{N}\!\big(\mathbf{x}_t;\; \alpha_t\,\mathbf{x}_0,\;
\sigma_t^2\,\mathbf{I}\big),
\qquad
\alpha_t = \exp\!\left(\int_0^t f(u)\,\mathrm{d}u\right),
$$

$$
\sigma_t^2 = \alpha_t^2 \int_0^t \frac{g^2(s)}{\alpha_s^2}\,\mathrm{d}s .
$$

Qui $\alpha_t$ è quanto resta del dato di partenza e $\sigma_t^2$ la varianza
del rumore accumulato. La conseguenza operativa ha un nome, ed è il motivo per
cui questi modelli si addestrano in tempi umani: il campionamento è **senza
simulazione** (*simulation-free*). Per avere $\mathbf{x}_t$ a un istante
qualsiasi non serve integrare l'equazione passo per passo, basta sorteggiare
$\boldsymbol{\epsilon}$ e scrivere
$\mathbf{x}_t = \alpha_t\mathbf{x}_0 + \sigma_t\boldsymbol{\epsilon}$.

`````

## Le due famiglie: chi lascia esplodere la varianza e chi la conserva

Scelte la deriva e la diffusione, il processo è deciso. Le due tradizioni della
diffusione corrispondono a due scelte, e portano nomi che dicono che cosa
succede alla larghezza della nuvola dei dati.

`````{tab} Elementare

**La prima famiglia non tocca l'immagine e ci versa sopra rumore sempre più
forte.** Non c'è nessuna deriva: il dato resta dov'è, e attorno a esso la
nuvola si allarga senza limite. Alla fine il rumore è così forte che
dell'immagine non si distingue più niente, ma tecnicamente c'è ancora, sepolta.
Il nome dice il comportamento: la varianza esplode.

**La seconda famiglia invece restringe l'immagine mentre aggiunge rumore**, e
lo fa con un dosaggio preciso: quanto toglie di segnale, tanto aggiunge di
rumore, così che la larghezza complessiva della nuvola resti sempre più o meno
uguale a uno. È la ricetta di DDPM, e il nome dice anche questo: la varianza si
conserva.

La differenza pratica sta nella taglia dei numeri con cui la rete lavora. Nella
prima famiglia i valori diventano enormi verso la fine, e la rete deve
maneggiare grandezze che cambiano di ordini di grandezza; nella seconda restano
sempre attorno a uno, che per una rete neurale è la condizione comoda. La
seconda ce li tiene senza fare niente; chi usa la prima li riscala prima di
darli alla rete, e fatto quel riscalamento il vantaggio si annulla.

Il fatto notevole è che sono la stessa equazione con due manopole diverse.
Prima del 2020 erano due letterature separate, con due vocabolari e due insiemi
di trucchi; nel continuo la separazione sparisce.

`````

`````{tab} Superiore

**Variance Exploding (VE), la famiglia dei modelli score-based.** Nessuna
deriva, e la diffusione cresce con il programma di rumore scelto:

$$
\mathbf{f}(\mathbf{x},t) = \mathbf{0},
\qquad
g(t) = \sqrt{\frac{\mathrm{d}\sigma^2(t)}{\mathrm{d}t}} ,
$$

da cui $\alpha_t = 1$ e $\sigma_t = \sigma(t)$: il segnale resta intatto e il
rumore cresce senza limite. La marginale terminale non tende a nessuna
distribuzione fissa, e come prior si usa
$\mathcal{N}(\mathbf{0},\sigma_{\max}^2\mathbf{I})$ con $\sigma_{\max}$
dell'ordine della massima distanza fra due dati.

**Variance Preserving (VP), la famiglia di DDPM.**

$$
\mathbf{f}(\mathbf{x},t) = -\tfrac{1}{2}\beta(t)\,\mathbf{x},
\qquad
g(t) = \sqrt{\beta(t)} ,
$$

da cui

$$
\alpha_t = \exp\!\left(-\tfrac{1}{2}\int_0^t \beta(u)\,\mathrm{d}u\right),
\qquad
\sigma_t^2 = 1 - \alpha_t^2 ,
$$

cioè $\alpha_t^2+\sigma_t^2 = 1$ esattamente: se i dati hanno varianza unitaria
la mantengono per tutta la traiettoria, e la marginale terminale tende a
$\mathcal{N}(\mathbf{0},\mathbf{I})$. Il legame con la notazione discreta della
sezione precedente è $\alpha_t = \sqrt{\bar{\alpha}_t}$.

Esiste anche una **sub-VP**, che usa la stessa deriva con
$g^2(t)=\beta(t)\,(1-\alpha_t^4)$, da cui $\sigma_t^2=(1-\alpha_t^2)^2$: la
varianza resta sotto quella della VP a ogni istante, e nell'articolo che la
introduce dà verosimiglianze migliori della VP.

Il rapporto $\alpha_t/\sigma_t$ è il **rapporto segnale-rumore**, e le tre
famiglie si distinguono soltanto per come lo fanno scendere e per come
riscalano lo stato lungo la strada. Questa osservazione è il punto di partenza
della riformulazione di Karras e colleghi {cite}`karras2022elucidating`, che
tratta la scala come una riparametrizzazione libera e riscrive tutte le
varianti in un unico spazio di progetto.

`````

Che il limite continuo dica la verità si controlla in venti righe, ed è il modo
più rapido di convincersene: si simula l'equazione a piccoli passi e si
confronta la nuvola che ne esce con quella che il conto esatto prevede.

```python
import numpy as np

rng = np.random.default_rng(0)

# programma di rumore VP, lo stesso di DDPM in versione continua
B_MIN, B_MAX, T = 0.1, 20.0, 1.0
beta = lambda t: B_MIN + t * (B_MAX - B_MIN)
alpha = lambda t: np.exp(-0.5 * (B_MIN * t + 0.5 * (B_MAX - B_MIN) * t**2))
sigma = lambda t: np.sqrt(1 - alpha(t)**2)

# dati: due sole possibilita', cosi' tutto il resto e' calcolabile a mano
MODI = np.array([-1.5, 1.5])
x0 = rng.choice(MODI, size=20000)

# 1) l'equazione simulata a piccoli passi
N = 1000
dt = T / N
x = x0.copy()
fotografie = {}
for k in range(N):
    t = k * dt
    x = (x - 0.5 * beta(t) * x * dt
         + np.sqrt(beta(t) * dt) * rng.normal(size=x.shape))
    if k + 1 in (N // 4, N // 2, N):
        fotografie[(k + 1) * dt] = x.copy()

# 2) la stessa cosa in forma chiusa, senza simulare niente
print("  t     simulata          forma chiusa")
for t, xs in fotografie.items():
    chiusa = alpha(t) * x0 + sigma(t) * rng.normal(size=x0.shape)
    print(f"{t:5.2f}   {xs.mean():+.3f} {xs.std():.3f}     "
          f"{chiusa.mean():+.3f} {chiusa.std():.3f}")
# ->  0.25   +0.002 1.282     -0.001 1.286
# ->  0.50   -0.005 1.048     -0.015 1.051
# ->  1.00   +0.005 0.999     +0.008 0.995
```

Media e larghezza coincidono a ogni istante, e la seconda colonna si ottiene
senza fare mille passi: una moltiplicazione e una somma. La deviazione a
$t=0{,}25$ vale $1{,}28$ e non $1$ perché i dati di partenza hanno già una
larghezza loro, essendo due punti a distanza tre.

## L'equazione all'indietro, e il suo unico ingrediente ignoto

`````{tab} Elementare

Riavvolgere una diffusione sembra impossibile, e per una singola traiettoria lo
è: il fumo che esce da un camino non rientra, e da una nuvola di fumo non si
ricostruisce la scintilla che l'ha prodotta. Il risultato di Anderson dice una
cosa più sottile e più utile: **la singola traiettoria non si riavvolge, ma la
distribuzione sì**. Non si può sapere da quale scintilla venga quel fumo lì; si
può però costruire un processo che, partendo da nuvole di fumo, produce
scintille distribuite esattamente come quelle vere.

L'equazione all'indietro ha la stessa forma di quella in avanti, con un termine
in più: in ogni punto, oltre alla deriva e al tremore, c'è una spinta che punta
verso le zone dove i dati sono più densi. Quella spinta è il **punteggio**, il
verso della salita di cui si parlava, e la cosa importante è che è **l'unica
cosa ignota**. Deriva e tremore li abbiamo scelti noi quando abbiamo deciso
come rovinare le immagini: sono nostri, li conosciamo esattamente. Della strada
del ritorno manca un pezzo solo, ed è quello che la rete impara.

C'è una cosa che sorprende. Andando all'indietro si continua ad aggiungere
rumore. Sembra assurdo, visto che si sta cercando di ripulire; e invece i due
termini lavorano insieme. La spinta verso le zone dense tira verso i dati, il
rumore permette di esplorare invece di precipitare sul primo posto buono, e il
tremore **si spegne** man mano che si procede: all'inizio del ritorno è forte e
si vaga, verso la fine è quasi nullo e comanda solo la spinta. È esattamente la
ricetta con cui i fisici campionano una distribuzione complicata, con una
temperatura che si abbassa piano piano finché il sistema non si posa.

`````

`````{tab} Superiore

Il teorema di Anderson afferma che il processo invertito nel tempo, associato
alla SDE in avanti, è a sua volta una diffusione, retta da

$$
\mathrm{d}\bar{\mathbf{x}}(t) =
\Big[\mathbf{f}(\bar{\mathbf{x}}(t),t)
- g^2(t)\,\nabla_{\mathbf{x}}\log p_t(\bar{\mathbf{x}}(t))\Big]\mathrm{d}t
+ g(t)\,\mathrm{d}\bar{\mathbf{w}}(t),
\qquad
\bar{\mathbf{x}}(T)\sim p_{\text{prior}},
$$

integrata da $t=T$ a $t=0$ (quindi $\mathrm{d}t<0$), con
$\bar{\mathbf{w}}$ un processo di Wiener nel tempo invertito. Qui
$\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$ è il **punteggio** (*score*) della
marginale all'istante $t$, cioè il gradiente rispetto a $\mathbf{x}$ del
logaritmo della densità dei dati rumorosi.

Il contenuto del teorema è forte e va enunciato con precisione: le marginali
del processo invertito **coincidono** con quelle del processo in avanti a ogni
istante, come uguaglianza e non come approssimazione asintotica. Le
singole traiettorie invece non si corrispondono, ed è la ragione per cui il
generatore non ricostruisce l'immagine da cui il rumore era partito.

L'ingrediente ignoto è uno solo. Scelti $\mathbf{f}$ e $g$, il resto della
dinamica è determinato dal punteggio, e imparare il punteggio è il problema
che la sezione precedente ha già risolto in forma discreta: il minimo della
loss quadratica sul rumore è la media condizionata
$\mathbb{E}[\boldsymbol{\epsilon}\mid\mathbf{x}_t]$, e questa è
$-\sigma_t\nabla\log p_t(\mathbf{x}_t)$.

**La lettura come dinamica di Langevin.** Ponendo $\mathbf{f}\equiv\mathbf{0}$
e riparametrizzando il tempo con $s := T-t$, la SDE all'indietro diventa

$$
\mathrm{d}\bar{\mathbf{x}}_s
= 2\tau(s)\,\nabla\log \pi_s(\bar{\mathbf{x}}_s)\,\mathrm{d}s
+ \sqrt{2\tau(s)}\,\mathrm{d}\mathbf{w}_s,
\qquad
\tau(s) := \tfrac{1}{2}g^2(T-s),
$$

dove $\pi_s := p_{T-s}$ è la densità bersaglio a quell'istante. Ha la forma
della dinamica di Langevin con cui i {doc}`modelli a energia
</ModelliEnergia/oltre-la-partizione>` campionano, e le differenze sono tre. La
temperatura $\tau(s)$ **decresce** lungo il percorso (è un *annealing*, non una
temperatura fissa); il bersaglio cambia a ogni istante invece di essere sempre
lo stesso; e la deriva è il doppio di quella di Langevin, che a parità di
diffusione vuole $\tau\nabla\log\pi$. E la differenza morde: con la temperatura
congelata questa equazione avrebbe per stazionaria $\pi^2$ e non $\pi$, e la
Langevin vera, dove serve, si aggiunge a parte come correttore. Quello che nei
modelli a energia è una catena lunghissima da far mescolare qui diventa un
percorso guidato di durata prefissata.

`````

## La strada deterministica, quella che non sorteggia niente

Fin qui il ritorno è casuale. La domanda che chiude il quadro è se il caso
serva davvero.

`````{tab} Elementare

Immagina due modi di riportare a casa una folla dispersa in una piazza. Il
primo dà a ciascuno una spinta verso casa e lo lascia anche barcollare un po';
il secondo assegna a ogni punto della piazza una direzione precisa e chiede a
tutti di seguirla senza sbandare. I percorsi individuali sono diversissimi, e
tuttavia si può fare in modo che **in ogni istante la folla sia sparpagliata
allo stesso modo**: stessa densità in ogni punto della piazza, a ogni ora.

Questo secondo modo esiste, e si costruisce a partire dal primo togliendo il
tremore e dimezzando la spinta: metà è esattamente quanto serve a rimpiazzare
il rimescolamento che si è tolto. Le conseguenze sono tre.

*Si può percorrere nei due sensi.* Senza caso, il percorso è una linea
tracciata: dai dati al rumore e dal rumore ai dati si passa lungo la stessa
strada. Ogni immagine ha quindi il **suo** rumore, quello da cui si ottiene, e
lo si può trovare percorrendo la strada all'incontrario. È così che si modifica
un'immagine esistente invece di generarne una da zero: la si porta indietro
fino al suo rumore, si cambia qualcosa nella richiesta, e la si riporta avanti.

*Si può dire quanto è probabile ciò che si è generato.* Lungo una strada
tracciata si può tenere il conto di quanto lo spazio si allarga e si stringe, e
quel conto è precisamente il {doc}`determinante
</Matematica/determinante-e-volume>`, cioè quanto una trasformazione gonfia lo
spazio. Con la strada casuale quel conto non si può fare.

*Servono meno passi.* Le traiettorie senza tremore sono lisce, e una curva
liscia si approssima bene con pochi segmenti. Le traiettorie casuali sono
frastagliate e chiedono passi corti. Quasi tutti i generatori veloci di oggi
percorrono la strada deterministica, e la {doc}`sezione sui campionatori veloci
</ModelliDiffusione/campionatori-veloci>` racconta come.

Il prezzo si vede solo quando la spinta non è quella esatta ma quella che la
rete ha imparato, cioè sempre. Sulla strada tracciata un errore commesso a metà
percorso resta lì e si porta avanti fino in fondo; su quella casuale il
barcollamento rimescola a ogni passo, e ogni rimescolamento riporta la folla
verso la densità che dovrebbe avere in quel momento, cancellando in parte gli
errori vecchi. Con la spinta esatta le due strade darebbero lo stesso
risultato; con una spinta approssimata la casuale arriva più vicino, ed è per
questo che chi vuole il massimo della qualità visiva usa spesso un misto, con
del caso all'inizio e nessuno alla fine.

`````

`````{tab} Superiore

Accanto alla SDE all'indietro esiste un'equazione **deterministica** con le
stesse marginali a ogni istante, la **ODE del flusso di probabilità**
(*probability flow ODE*, PF-ODE):

$$
\frac{\mathrm{d}\tilde{\mathbf{x}}(t)}{\mathrm{d}t}
= \mathbf{f}(\tilde{\mathbf{x}}(t),t)
- \tfrac{1}{2}g^2(t)\,\nabla_{\mathbf{x}}\log
p_t(\tilde{\mathbf{x}}(t)) .
$$

Il confronto con la SDE all'indietro mostra le due sole differenze: il termine
di rumore è sparito, e il coefficiente del punteggio è dimezzato. Non è una
coincidenza dei conti: l'una e l'altra soddisfano la stessa equazione di
evoluzione per la densità, e il fattore $\tfrac12$ è esattamente quanto serve a
compensare il contributo diffusivo che si è tolto.

Le tre conseguenze, in forma precisa:

- **Invertibilità.** La PF-ODE definisce un flusso deterministico
  $\Phi_{t\to s}$, quindi una biiezione fra la distribuzione dei dati e il
  prior. Da qui l’*inversione DDIM*, che serve a editing, interpolazione nello
  spazio del rumore e attribuzione.
- **Verosimiglianza esatta.** Trattando la PF-ODE come un
  {doc}`flusso normalizzante continuo </VerosimiglianzaEsatta/a-che-serve>` si
  ottiene
  $\log p_0(\mathbf{x}_0) = \log p_T(\mathbf{x}_T) + \int_0^T
  \nabla\!\cdot\!\mathbf{v}_t(\mathbf{x}_t)\,\mathrm{d}t$,
  con $\mathbf{v}_t$ il campo di velocità della ODE e la divergenza stimata
  alla Hutchinson. È il conto che la SDE non permette.
- **Errore di discretizzazione.** Un integratore di ordine $p$ su una
  traiettoria liscia accumula errore $O(h^p)$ con $h$ il passo; le traiettorie
  della SDE hanno regolarità di Hölder $1/2$ e i metodi stocastici si fermano a
  ordini bassi. È la ragione strutturale per cui i campionatori a pochi passi
  lavorano sulla ODE.

Il rovescio, misurato in letteratura e riconoscibile a occhio nei campioni: a
parità di modello la SDE produce campioni più diversi e spesso migliori sui
punteggi percettivi, perché la stocasticità corregge in corsa gli errori del
punteggio stimato, mentre la ODE li integra fedelmente. Da qui i campionatori
ibridi, che usano rumore nella prima parte del percorso e la sola ODE nella
seconda.

`````

Le due strade e la loro equivalenza si controllano in poche righe, e senza
addestrare niente: su due sole modalità il punteggio esatto si scrive a mano,
quindi si può isolare la matematica dalla rete.

```python
def punteggio(x, t):
    """Il punteggio esatto della mistura, senza nessuna rete."""
    a, s = alpha(t), sigma(t)
    d = x[:, None] - a * MODI[None, :]
    w = np.exp(-0.5 * (d / s)**2)
    w /= w.sum(axis=1, keepdims=True)
    return -(w * d).sum(axis=1) / s**2

n_camp, M = 5000, 500
dt = T / M

# la strada deterministica: dal rumore ai dati, senza sorteggiare niente
y = rng.normal(size=n_camp)
for k in range(M):
    t = T - k * dt
    y = y + dt * 0.5 * beta(t) * (y + punteggio(y, t))

# la strada casuale: la stessa cosa con il tremore acceso
z = rng.normal(size=n_camp)
for k in range(M):
    t = T - k * dt
    z = (z + dt * (0.5 * beta(t) * z + beta(t) * punteggio(z, t))
         + np.sqrt(beta(t) * dt) * rng.normal(size=n_camp))

for nome, v in (("ODE", y), ("SDE", z)):
    print(nome, round(float((v < 0).mean()), 4),
          round(float(v[v < 0].mean()), 4), round(float(v[v > 0].mean()), 4),
          round(float(np.abs(np.abs(v) - 1.5).mean()), 4))
# -> ODE 0.501 -1.5003 1.5 0.0055
# -> SDE 0.4998 -1.5001 1.5005 0.0135
```

Le due strade arrivano nello stesso posto: metà dei campioni su ciascun modo, e
i modi esattamente a $\pm 1{,}5$. L'ultima colonna è di quanto in media un
campione manca il proprio modo, e dice la differenza fra le due: la strada
casuale lascia una dispersione due volte e mezzo più larga, perché continua ad
aggiungere rumore fino all'ultimo passo.

E la strada deterministica è davvero una strada, cioè si percorre nei due
sensi.

```python
ts = np.linspace(1e-3, T, M + 1)
campo = lambda x, t: -0.5 * beta(t) * (x + punteggio(x, t))

partenza = rng.choice(MODI, size=6) + 0.01 * rng.normal(size=6)
x = partenza.copy()
for k in range(M):                      # dai dati al rumore
    x = x + (ts[k + 1] - ts[k]) * campo(x, ts[k])
rumore = x.copy()
for k in range(M, 0, -1):               # e ritorno, sulla stessa strada
    x = x - (ts[k] - ts[k - 1]) * campo(x, ts[k])

print(np.round(rumore, 3))
# -> [-0.354  2.779  0.572 -0.04   0.562 -0.623]
print(round(float(np.abs(x - partenza).max()), 4))   # -> 0.0077
```

Sei immagini portate al rumore e riportate indietro tornano dove erano, a meno
di otto millesimi. Quel residuo è l'errore di chi integra a passi finiti, non
un difetto della strada: è esattamente la grandezza che i campionatori veloci
cercano di ridurre a parità di passi.

## Quattro modi di dire la stessa cosa

Chi legge il codice di più di una libreria incontra reti che predicono cose
apparentemente diverse. Sono la stessa rete.

`````{tab} Elementare

Data un'immagine rovinata, ci sono quattro domande che si possono fare, e
rispondere a una qualsiasi permette di rispondere a tutte le altre.

- **Qual era il disturbo?** È la domanda di DDPM.
- **Qual era l'immagine pulita?** È la domanda che sembra più naturale.
- **In che direzione salire?** È il punteggio.
- **Con quale velocità muoversi?** È una combinazione delle prime due, e si
  chiama velocità.

Sono quattro forme della stessa informazione perché l'immagine rovinata è fatta
di due pezzi, l'immagine pulita e il disturbo, mescolati in proporzioni note.
Sapendo l'immagine rovinata e uno dei due pezzi, l'altro si ricava sottraendo.

La scelta non cambia la matematica e cambia i conti. Chiedere il disturbo
funziona male quando di disturbo ce n'è pochissimo, perché di quel disturbo
nell'immagine resta una traccia minuscola e va indovinato tutto a partire da
lì; chiedere l'immagine pulita funziona male dall'altra parte, dove di immagine
non c'è quasi più niente. La quarta domanda, la velocità, è stata inventata
proprio per questo: mescola le due in modo da restare ben posta a tutti e due
gli estremi, ed è quella che si usa quando il percorso deve essere accorciato a
pochi passi.

`````

`````{tab} Superiore

Con $\mathbf{x}_t = \alpha_t\mathbf{x}_0 + \sigma_t\boldsymbol{\epsilon}$, le
quattro parametrizzazioni sono legate da identità esatte:

$$
\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t
- \sigma_t\hat{\boldsymbol{\epsilon}}}{\alpha_t},
\qquad
\mathbf{s}_\theta(\mathbf{x}_t,t)
= -\frac{\hat{\boldsymbol{\epsilon}}}{\sigma_t},
\qquad
\hat{\mathbf{v}} = \alpha_t\hat{\boldsymbol{\epsilon}}
- \sigma_t\hat{\mathbf{x}}_0 ,
$$

dove $\hat{\boldsymbol{\epsilon}}$ è la predizione del rumore,
$\hat{\mathbf{x}}_0$ quella del dato pulito, $\mathbf{s}_\theta$ il punteggio e
$\hat{\mathbf{v}}$ la **velocità**. Una rete addestrata su una qualsiasi delle
quattro si converte nelle altre senza riaddestramento, e le librerie infatti
espongono un parametro di configurazione per sceglierla.

L'equivalenza è algebrica, il condizionamento numerico no. Per $t\to 0$ si ha
$\sigma_t\to 0$: la conversione $\hat{\boldsymbol{\epsilon}}\mapsto
\mathbf{s}$ divide per $\sigma_t$ e amplifica l'errore, mentre
$\hat{\mathbf{x}}_0$ è ben posta. Per $t\to T$ è il contrario, perché
$\alpha_t\to 0$ e ricostruire $\hat{\mathbf{x}}_0$ significa dividere per
qualcosa che tende a zero. La **$\mathbf{v}$-prediction**
{cite}`salimans2022progressive` interpola fra le due e resta ben condizionata a
entrambi gli estremi; è la parametrizzazione standard nella distillazione e
nei programmi di rumore che portano il rapporto segnale-rumore esattamente a
zero all'ultimo passo.

Le funzioni di costo corrispondenti differiscono solo per un peso dipendente da
$t$:

$$
\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}
\Big[\,w(t)\,\lVert \hat{\boldsymbol{\epsilon}}(\mathbf{x}_t,t)
- \boldsymbol{\epsilon}\rVert^2\Big] ,
$$

e ogni scelta di parametrizzazione equivale a una scelta di $w(t)$ nella
formulazione sul rumore. È il quadro unificante che la letteratura ha messo a
fuoco solo dopo il 2022 {cite}`lai2026principles`, e spiega perché confronti
fra articoli che dichiaravano obiettivi diversi risultassero poi difficili da
interpretare: cambiava il peso, non l'obiettivo.

`````

## In pratica

```python
# la conversione fra le quattro parametrizzazioni, che nessuna libreria
# nasconde ma tutte chiamano in modo diverso
def converti(eps, x_t, t):
    a, s = alpha(t), sigma(t)
    x0 = (x_t - s * eps) / a
    score = -eps / s
    v = a * eps - s * x0
    return x0, score, v

t = 0.5
x_t = alpha(t) * MODI[0] + sigma(t) * 0.3        # un esempio costruito a mano
eps_vero = 0.3
x0, score, v = converti(eps_vero, x_t, t)
print(round(float(x0), 6), round(float(MODI[0]), 6))     # -> -1.5 -1.5
print(round(float(v), 6),
      round(float(alpha(t) * eps_vero - sigma(t) * MODI[0]), 6))
# -> 1.523836 1.523836
```

La prima riga è il controllo che conta: partendo dal rumore vero e
dall'immagine rovinata si ricostruisce l'immagine pulita, e viene esattamente
quella di partenza. Le quattro domande sono la stessa domanda.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Le due ricette per rovinare un'immagine hanno la stessa forma (valore
  vecchio, più uno spostamento sistematico, più uno scossone sorteggiato) e nel
  limite dei passi infinitamente corti diventano **una sola equazione** con due
  manopole: la **deriva**, che dice dove il valore è trascinato, e la
  **diffusione**, che dice quanto trema.
- Le due famiglie storiche sono due scelte di quelle manopole: una lascia
  l'immagine ferma e fa esplodere il rumore, l'altra restringe l'immagine
  mentre aggiunge rumore così che la larghezza resti sempre attorno a uno. La
  seconda tiene i numeri in una scala comoda senza doverli riscalare.
- La strada del ritorno esiste ed è **la stessa equazione con un termine in
  più**, la spinta verso le zone dense. Quella spinta è l'unica cosa ignota, e
  la rete impara solo quella. Andando all'indietro il tremore c'è ancora ma si
  spegne piano piano, come una temperatura che si abbassa.
- Accanto alla strada casuale ce n'è una **deterministica** che attraversa le
  stesse nuvole negli stessi istanti. Si percorre nei due sensi (quindi ogni
  immagine ha il suo rumore, e si può modificare un'immagine esistente),
  permette di dire quanto è probabile ciò che si genera, e si accorcia con meno
  passi. In cambio dà campioni un po' meno vari.
- Le quattro cose che una rete può predire (il disturbo, l'immagine pulita, la
  direzione di salita, la velocità) sono **la stessa informazione**: sapendone
  una si ricavano le altre tre. Cambia solo dove i conti restano precisi.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Il limite $\Delta t\to 0$ dà $\mathrm{d}\mathbf{x} =
  \mathbf{f}(\mathbf{x},t)\mathrm{d}t + g(t)\mathrm{d}\mathbf{w}$. Con deriva
  **affine** il nucleo è gaussiano in forma chiusa,
  $p_t(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}(\alpha_t\mathbf{x}_0,
  \sigma_t^2\mathbf{I})$, e il campionamento è *simulation-free*.
- **VE**: $\mathbf{f}=\mathbf{0}$, $g=\sqrt{\mathrm{d}\sigma^2/\mathrm{d}t}$.
  **VP**: $\mathbf{f}=-\tfrac12\beta\mathbf{x}$, $g=\sqrt{\beta}$, con
  $\alpha_t^2+\sigma_t^2=1$. Le due differiscono per come fanno scendere il
  rapporto segnale-rumore e per il riscalamento.
- **Anderson (1982)**: $\mathrm{d}\bar{\mathbf{x}} = [\mathbf{f} -
  g^2\nabla\log p_t]\mathrm{d}t + g\,\mathrm{d}\bar{\mathbf{w}}$, con marginali
  **identiche** a quelle del processo in avanti. Il punteggio è l'unico termine
  ignoto; con $\mathbf{f}=\mathbf{0}$ l'equazione è una dinamica di Langevin con
  temperatura $\tau(s)=\tfrac12 g^2(T-s)$ che decresce.
- **PF-ODE**: $\dot{\tilde{\mathbf{x}}} = \mathbf{f} -
  \tfrac12 g^2\nabla\log p_t$, stesse marginali, traiettorie deterministiche.
  Dà invertibilità, verosimiglianza esatta alla Hutchinson e traiettorie lisce
  (quindi integratori di ordine alto). La SDE resta preferibile per la
  diversità, perché la stocasticità corregge gli errori del punteggio stimato.
- Le quattro parametrizzazioni sono legate da
  $\hat{\mathbf{x}}_0 = (\mathbf{x}_t-\sigma_t\hat{\boldsymbol{\epsilon}})/
  \alpha_t$, $\mathbf{s}=-\hat{\boldsymbol{\epsilon}}/\sigma_t$,
  $\hat{\mathbf{v}}=\alpha_t\hat{\boldsymbol{\epsilon}}-\sigma_t
  \hat{\mathbf{x}}_0$, e differiscono solo per il peso $w(t)$ nella loss. La
  $\mathbf{v}$-prediction è ben condizionata a entrambi gli estremi, dove le
  altre due degenerano.
```
`````

La diffusione ha smesso di essere una ricetta e ha una struttura: si sceglie
come rovinare, il ritorno è determinato, e l'unica cosa da imparare è una
funzione. Resta però una domanda che l'impianto lascia aperta. Il percorso è
stato costruito rovinando i dati con del rumore gaussiano, e sono quelle nuvole
gaussiane a decidere la forma delle traiettorie: c'è un modo di scegliere il
percorso invece di ereditarlo, e magari di sceglierlo dritto?
