# Flow matching: scegliere la strada invece di ereditarla

Nel 1781 Gaspard Monge presenta all'Accademia delle Scienze di Parigi una
memoria sulla *théorie des déblais et des remblais*, cioè sugli sterri e sui
riporti. Il problema è concreto e da cantiere: c'è un mucchio di terra di una
certa forma, bisogna spostarlo per ottenerne uno di un'altra forma, e ogni
badilata costa in proporzione a quanto lontano la si porta. Qual è il piano di
trasporto che costa meno? La domanda si rivelerà molto più profonda del
cantiere che l'aveva generata: ci vorranno centosessant'anni perché Leonid
Kantorovič la riformuli in una veste risolvibile, e quella riformulazione è un
pezzo del lavoro sull'impiego ottimale delle risorse per cui, nel 1975, gli
daranno il Nobel per l'economia.

La diffusione, guardata da lontano, è un problema di questa famiglia. Da una
parte c'è un mucchio di terra a forma di «tutte le fotografie di gatti»,
dall'altra un mucchio a forma di «rumore»: generare significa spostare il
secondo sul primo. La sezione precedente ha ottenuto un piano di trasporto per
una via indiretta, costruendo un processo che rovina i dati e poi invertendolo.
La domanda è se non si possa fare la cosa ovvia: **scegliere il percorso,
invece di ereditarlo dal modo in cui si è deciso di rovinare le immagini**.

La risposta è sì, il metodo si chiama **flow matching**, e la strada più
semplice fra due punti risulta essere anche la più economica da percorrere.

## Invece di rovinare, trasportare

`````{tab} Elementare

Cambia il punto di vista, e cambiano le domande che vengono naturali.

Nel punto di vista della diffusione si decide **come rovinare**: si stabilisce
quanto rumore aggiungere e a che ritmo, e quel che segue è determinato. Il
percorso che i dati fanno per diventare rumore lo si subisce, ed è quello che il
programma di rumore impone.

Nel punto di vista del trasporto si decide **il percorso**. Si dice: questa
fotografia deve trasformarsi in questo particolare mucchio di rumore, e ci
arriverà passando di qui. Fatta la scelta, quello che resta da imparare è una
sola cosa, un **campo di velocità**: in ogni punto dello spazio e in ogni
istante, in che direzione e a che velocità muoversi. Un campo di velocità è
esattamente quello che un meteorologo disegna sulle carte dei venti, con una
frecciolina per ogni punto: chi ci si trova dentro viene portato dove la
freccia indica.

Generare, allora, è lasciarsi portare: si parte da un punto sorteggiato nel
rumore, si guarda la freccia, ci si sposta un pochino, si guarda la freccia
nuova, e avanti così fino a destinazione. Nessun caso, nessun sorteggio dopo il
primo.

C'è una condizione da rispettare, e ha a che fare con la conservazione della
materia. Se il campo di venti deve trasformare esattamente un mucchio in un
altro, non può inventare terra dove non c'era né farla sparire: quello che entra
in una regione deve essere quello che esce dalle altre. Detta così sembra
un'ovvietà; scritta in formule è un'equazione precisa, che lega il campo di
velocità alla densità che si vuole spostare, e che va rispettata perché il
trasporto sia legittimo.

`````

`````{tab} Superiore

Un **flusso** è una famiglia di mappe $\Phi_t:\mathbb{R}^D\to\mathbb{R}^D$
generata da un campo di velocità $\mathbf{u}_t$ attraverso l'equazione
differenziale ordinaria

$$
\frac{\mathrm{d}}{\mathrm{d}t}\Phi_t(\mathbf{x})
= \mathbf{u}_t\big(\Phi_t(\mathbf{x})\big),
\qquad \Phi_0 = \mathrm{id} .
$$

Il flusso spinge in avanti la densità iniziale, e la famiglia di densità
$p_t := (\Phi_t)_{\#}p_0$ che ne risulta si chiama **percorso di probabilità**
(*probability path*). La condizione di consistenza fra campo e densità è
l’**equazione di continuità**

$$
\frac{\partial p_t}{\partial t}
+ \nabla\!\cdot\!\big(p_t\,\mathbf{u}_t\big) = 0 ,
$$

che è la conservazione della massa scritta in forma locale: la densità cambia
in un punto esattamente di quanto il flusso ne porta dentro meno quanto ne
porta fuori. La coppia $(p_t,\mathbf{u}_t)$ si dice allora *compatibile*.

Il rovesciamento di prospettiva rispetto alla sezione precedente è tutto qui.
Là si fissava una SDE in avanti e si ricavava il campo che la inverte, un
oggetto determinato da $\mathbf{f}$, $g$ e dal punteggio. Qui si fissa
direttamente il percorso $p_t$ che interpola fra $p_0 = p_{\text{dati}}$ e
$p_1 = p_{\text{prior}}$, e si cerca un $\mathbf{u}_t$ compatibile con esso.

Il campo compatibile **non è unico**: all'equazione di continuità si può
aggiungere qualunque campo $\mathbf{w}$ con $\nabla\!\cdot\!(p_t\mathbf{w})=0$
senza cambiare l'evoluzione delle densità. Il flow matching ne sceglie uno, e a
parità di percorso gaussiano quello che sceglie coincide con la PF-ODE della
sezione precedente: il dizionario fra i due linguaggi lo mostra in una riga.
Quello che il rovesciamento di prospettiva aggiunge è la libertà di scegliere
il percorso, non un campo diverso da percorrere.

`````

## Il bersaglio che non si può calcolare, e il trucco che lo rende calcolabile

`````{tab} Elementare

L'idea di far imparare a una rete il campo dei venti si scontra subito con un
ostacolo. Per insegnarle qualcosa bisogna poterle dire che cosa avrebbe dovuto
rispondere; e la velocità giusta in un punto dipende da **tutti** i modi in cui
ci si può arrivare. In quel punto passano infinite fotografie diverse in
viaggio verso infiniti rumori diversi, ciascuna con la sua velocità, e quella
giusta è la loro media. Calcolarla vorrebbe dire fare un integrale su tutto
l'archivio, per ogni punto e per ogni istante.

Il trucco che sblocca la situazione è lo stesso con cui la rete ha imparato il
verso della salita, ed è uno dei più usati in tutto il machine learning. Invece
di chiedere alla rete la velocità **media** su tutti i viaggi che passano di
lì, le si chiede la velocità di **un** viaggio specifico: si prende una
fotografia, si sorteggia un rumore di arrivo, si decide che quei due sono gli
estremi del viaggio, e a quel punto la velocità è banale da scrivere, perché il
viaggio lo abbiamo disegnato noi.

Il fatto sorprendente, e il motivo per cui il metodo funziona, è che
**allenarsi sulla velocità del singolo viaggio porta esattamente allo stesso
posto** che allenarsi su quella media. Non è un'approssimazione: le due
funzioni di costo hanno lo stesso punto di minimo. La ragione è quella che
rende speciale l'errore quadratico: chi cerca di indovinare un numero
sorteggiato, e viene giudicato su quanto sbaglia al quadrato, ha come strategia
migliore rispondere **la media** di quel numero. Quindi una rete allenata a
indovinare le singole velocità, proprio perché non può indovinarle tutte,
finisce per rispondere la loro media, che è quello che ci serviva.

`````

`````{tab} Superiore

L'obiettivo naturale sarebbe il **flow matching** puro,

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\;\mathbf{x}\sim p_t}
\Big[\big\lVert \mathbf{v}_\theta(\mathbf{x},t)
- \mathbf{u}_t(\mathbf{x})\big\rVert^2\Big],
$$

inutilizzabile perché $\mathbf{u}_t$ è definito da una marginalizzazione che
non si sa calcolare. Si introduce allora una variabile di condizionamento
$\mathbf{z}$ (tipicamente il dato $\mathbf{x}_0$, oppure la coppia
$(\mathbf{x}_0,\mathbf{x}_1)$ di estremi del viaggio) tale che il percorso e la
velocità **condizionati** siano noti in forma chiusa, e si minimizza il
**conditional flow matching**

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,\;\mathbf{z},\;
\mathbf{x}\sim p_t(\cdot\mid\mathbf{z})}
\Big[\big\lVert \mathbf{v}_\theta(\mathbf{x},t)
- \mathbf{u}_t(\mathbf{x}\mid\mathbf{z})\big\rVert^2\Big] .
$$

Il teorema che regge il metodo {cite}`lipman2023flow` è che i due obiettivi
differiscono per una costante indipendente da $\theta$, quindi

$$
\nabla_\theta\mathcal{L}_{\text{FM}}
= \nabla_\theta\mathcal{L}_{\text{CFM}} ,
$$

e hanno lo stesso minimizzatore, che è la media condizionata

$$
\mathbf{u}_t(\mathbf{x})
= \mathbb{E}\big[\mathbf{u}_t(\mathbf{x}\mid\mathbf{z})
\;\big|\;\mathbf{x}_t=\mathbf{x}\big] .
$$

La dimostrazione è un conto di due righe che usa una sola proprietà: il minimo
di $\mathbb{E}[\lVert a - Y\rVert^2]$ rispetto ad $a$ è $\mathbb{E}[Y]$. È
esattamente la struttura del **denoising score matching** di Vincent, dove il
bersaglio intrattabile (il punteggio della marginale) veniva sostituito dal
bersaglio banale (il rumore iniettato), con la stessa garanzia. Riconoscere che
sono lo stesso teorema applicato due volte fa risparmiare metà della
letteratura.

`````

## Scegliere il percorso, e il più semplice è una retta

`````{tab} Elementare

Deciso di scegliersi il percorso, quale scegliere?

Il percorso che la diffusione impone è una curva, e non per un capriccio: nasce
dal fatto che a ogni istante si mescola un po' meno immagine con un po' più
rumore, secondo un dosaggio che cambia continuamente. Se si disegna il viaggio
di una singola fotografia verso il suo rumore, quello che viene fuori è una
strada tortuosa.

Ma niente obbliga a quel percorso. Si può stabilire che il viaggio sia la
**linea retta**: al tempo zero la fotografia, al tempo uno il rumore, e in
mezzo la miscela che sta esattamente a metà strada quando il tempo è a metà.
Con questa scelta la velocità del viaggio è costantissima e si scrive senza
pensarci: è la differenza fra il punto di arrivo e quello di partenza, uguale
in ogni istante. Non c'è niente di più semplice da far imparare a una rete.

Questa è l'idea del **flusso rettificato**, ed è la strada che i generatori di
immagini più recenti hanno preso.

C'è però un'insidia da capire bene, perché il metodo si spiega spesso male.
Ogni singolo viaggio è una retta; il **campo dei venti** che ne risulta no. In
un punto in cui passano molti viaggi diretti in posti diversi, la freccia è la
loro media, e seguendo le medie non si percorre nessuna delle rette: si fa una
curva. È come una folla in cui ciascuno cammina dritto verso casa propria: il
flusso complessivo, visto dall'alto, gira.

Il rimedio esiste e si chiama **raddrizzamento**. Si fa girare il modello una
volta, e si guarda dove ciascun rumore di partenza va a finire: si ottengono
così delle coppie che il modello stesso ha accoppiato, invece che accoppiate a
caso. Rifacendo l'addestramento su quelle coppie, i viaggi non si incrociano
quasi più, e il campo dei venti diventa davvero dritto. A quel punto un solo
passo, o due, bastano.

`````

`````{tab} Superiore

Un percorso condizionato **gaussiano** si scrive in generale

$$
p_t(\mathbf{x}\mid\mathbf{x}_0)
= \mathcal{N}\big(\mathbf{x};\;\alpha_t\mathbf{x}_0,\;
\sigma_t^2\mathbf{I}\big),
$$

esattamente la stessa forma del nucleo di perturbazione della sezione
precedente, con $\alpha_0=1,\sigma_0=0$ e, all'altro capo,
$\alpha_1=0,\sigma_1=1$: esattamente per il rettificato, solo al limite per la
diffusione, ed è il rapporto segnale-rumore terminale non nullo di cui la
distillazione dovrà tenere conto. Le scelte di $(\alpha_t,\sigma_t)$ danno le
varianti:

| percorso | $\alpha_t$ | $\sigma_t$ | velocità condizionata |
|---|---|---|---|
| diffusione (VP) | $\exp(-\tfrac12\int_0^t\beta)$ | $\sqrt{1-\alpha_t^2}$ | $\dot\alpha_t\mathbf{x}_0+\dot\sigma_t\boldsymbol{\epsilon}$ |
| rettificato | $1-t$ | $t$ | $\boldsymbol{\epsilon}-\mathbf{x}_0$ |

Nel caso rettificato l'interpolazione è
$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\,\boldsymbol{\epsilon}$ e la velocità
condizionata è **costante nel tempo**, il che rende l'obiettivo

$$
\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}
\Big[\big\lVert \mathbf{v}_\theta\big((1-t)\mathbf{x}_0
+ t\boldsymbol{\epsilon},\,t\big)
- (\boldsymbol{\epsilon}-\mathbf{x}_0)\big\rVert^2\Big],
$$

cioè tre righe di codice {cite}`liu2023rectified`.

**La rettitudine è condizionata, non marginale.** Il campo marginale
$\mathbf{u}_t(\mathbf{x}) = \mathbb{E}[\boldsymbol{\epsilon}-\mathbf{x}_0\mid
\mathbf{x}_t=\mathbf{x}]$ è una media su tutti gli accoppiamenti compatibili, e
le sue traiettorie sono curve anche quando ogni traiettoria condizionata è una
retta. È il punto in cui la divulgazione del metodo scivola più spesso.

Il **reflow** rimedia iterando. Dal modello addestrato si ricava
l'accoppiamento deterministico $(\boldsymbol{\epsilon},
\Phi(\boldsymbol{\epsilon}))$ indotto dalla ODE, e si riaddestra usando
**quelle coppie** invece di accoppiamenti indipendenti. L'accoppiamento indotto
non fa incrociare le traiettorie, quindi il nuovo campo marginale è più vicino
a quello condizionato; iterando, le traiettorie si raddrizzano e il numero di
passi necessari crolla. Il prezzo è che ogni giro di reflow richiede di
generare un insieme di coppie con il modello corrente, e che l'accuratezza si
degrada leggermente a ogni giro.

Un avvertimento sulla parentela con il trasporto ottimo, perché il nome
inganna. Un accoppiamento che non fa incrociare le traiettorie è **monotono**,
e in una dimensione la mappa monotona è effettivamente la soluzione del
problema di Monge con costo quadratico. In più dimensioni questo non basta: il
reflow non fa crescere il costo di trasporto a ogni giro, ma il limite non è in
generale la mappa ottima, e nessuno di questi metodi risolve il problema di
Monge. La coincidenza vale in dimensione uno e va lasciata lì.

`````

Il vantaggio delle strade dritte si misura, e il conto si fa senza addestrare
niente: su dati costituiti da due sole possibilità il campo di velocità esatto
si scrive a mano, per il percorso rettificato come per quello della diffusione.

```python
import numpy as np

MODI = np.array([-1.5, 1.5])
T_MIN = 1e-3

# --- percorso rettificato: x_t = (1-t) x0 + t z, quindi x_t|x0 ha
#     media (1-t) x0 e deviazione t
def x0_atteso(x, t):
    d = x[:, None] - (1 - t) * MODI[None, :]
    w = np.exp(-0.5 * (d / t)**2)
    w /= w.sum(axis=1, keepdims=True)
    return (w * MODI[None, :]).sum(axis=1)

def velocita_retta(x, t):
    return (x - x0_atteso(x, t)) / t

# --- percorso della diffusione (VP), cioe' il campo della PF-ODE
B_MIN, B_MAX = 0.1, 20.0
beta = lambda t: B_MIN + t * (B_MAX - B_MIN)
alpha = lambda t: np.exp(-0.5 * (B_MIN * t + 0.5 * (B_MAX - B_MIN) * t**2))
sigma = lambda t: np.sqrt(1 - alpha(t)**2)

def velocita_diffusione(x, t):
    a, s = alpha(t), sigma(t)
    d = x[:, None] - a * MODI[None, :]
    w = np.exp(-0.5 * (d / s)**2)
    w /= w.sum(axis=1, keepdims=True)
    punteggio = -(w * d).sum(axis=1) / s**2
    return -0.5 * beta(t) * (x + punteggio)

rng = np.random.default_rng(0)
z = rng.normal(size=4000)

def integra(campo, passi):
    ts = np.linspace(1.0, T_MIN, passi + 1)
    x = z.copy()
    for k in range(passi):
        x = x + (ts[k + 1] - ts[k]) * campo(x, ts[k])
    return x

errore = lambda x: float(np.abs(np.abs(x) - 1.5).mean())

print("passi   retta    diffusione")
for passi in (1, 2, 4, 8, 16, 64, 256):
    print(f"{passi:5d}   {errore(integra(velocita_retta, passi)):.4f}   "
          f"{errore(integra(velocita_diffusione, passi)):.4f}")
# -> passi   retta    diffusione
# ->     1   1.4992   0.8112
# ->     2   0.4845   0.7442
# ->     4   0.0227   0.4300
# ->     8   0.0020   0.1204
# ->    16   0.0017   0.0343
# ->    64   0.0016   0.0135
# ->   256   0.0015   0.0104
```

La tabella dice due cose, e la seconda è la più istruttiva.

La prima è il risultato atteso: con il percorso rettificato **otto passi**
bastano ad arrivare più vicino di quanto il percorso della diffusione arrivi
con duecentocinquantasei. Sull'ultima riga, però, il confronto va letto con una
riserva: le due colonne non hanno lo stesso pavimento. A un millesimo dalla
fine il percorso rettificato ha $\sigma_t=0{,}001$ e quello della diffusione
$0{,}0105$, dieci volte tanto, e quei soli residui valgono $0{,}0016$ e
$0{,}0084$. Dei $0{,}0104$ della riga a duecentocinquantasei passi, quindi,
quasi tutto è residuo e non errore di integrazione: il divario vero è quello
delle righe a quattro e a otto passi, dove è di quasi venti e di sessanta
volte. Il valore $0{,}0015$ su cui la prima colonna si appoggia non è errore di
integrazione ma il tempo che si ferma a un millesimo invece che a zero: a
quell'istante il segnale è ancora rimpicciolito di un millesimo, e
$1{,}5\times0{,}001$ fa esattamente $0{,}0015$.

La seconda è che con **un passo solo** il percorso rettificato fa peggio della
diffusione, e finisce a metà strada fra i due modi. È la conferma numerica
dell'insidia: al tempo uno la freccia, che è una media, punta verso la media
dei dati, perché a quell'istante l'immagine di partenza è del tutto dimenticata
e i due modi pesano uguale. Ogni viaggio è una retta, il campo che li media no,
e un passo solo lo dimostra.

Che sia proprio l'incrocio delle traiettorie il problema si vede guardando dove
finisce ciascun punto di partenza.

```python
griglia = np.array([-3.0, -1.5, -0.15, 0.15, 1.5, 3.0])
arrivo = griglia.copy()
ts = np.linspace(1.0, T_MIN, 401)
for k in range(400):
    arrivo = arrivo + (ts[k + 1] - ts[k]) * velocita_retta(arrivo, ts[k])
print(np.round(arrivo, 4))            # -> [-1.5013 -1.4996 -1.4973  1.4973  1.4996  1.5013]
print(bool(np.all(np.diff(arrivo) > 0)))          # la mappa e' monotona -> True
```

La mappa che il campo induce è **monotona**: chi parte più a destra arriva più
a destra, e le traiettorie non si scavalcano mai. L'unica eccezione è il punto
esattamente a zero, dove i due modi si equivalgono e la velocità è nulla: è lo
spartiacque fra i due bacini, un punto solo su tutta la retta. È questa
monotonia che il raddrizzamento sfrutta, ed è anche il motivo per cui in una
dimensione (e soltanto lì) la mappa che ne esce coincide con quella del
trasporto ottimo.

## La stessa cosa vista da due parti

`````{tab} Elementare

A leggere gli articoli i due punti di vista sembrano due mondi.

Non li separa quasi niente. Un percorso di diffusione è un percorso di trasporto
particolare, quello che si ottiene mescolando immagine e rumore con un certo
dosaggio; il campo di velocità che lo realizza si ricava dal punteggio con una
formula, e viceversa. Una rete addestrata in un modo si può usare nell'altro
riscrivendo la sua uscita.

Quello che cambia sono tre cose pratiche, ed è per esse che il vocabolario
nuovo si è imposto. La prima: potendo scegliere il percorso, si sceglie quello
che costa meno passi da percorrere. La seconda: il flow matching non chiede che
il punto di arrivo sia rumore gaussiano, quindi permette di andare da una
distribuzione qualsiasi a un'altra qualsiasi, per esempio da fotografie diurne
a fotografie notturne, senza passare dal rumore. La terza: le formule sono più
corte, e formule più corte fanno commettere meno errori.

`````

`````{tab} Superiore

Il dizionario fra i due linguaggi si scrive in una riga. Dato un percorso
gaussiano $\mathbf{x}_t = \alpha_t\mathbf{x}_0 + \sigma_t\boldsymbol{\epsilon}$,
il campo di velocità marginale e il punteggio sono legati da

$$
\mathbf{u}_t(\mathbf{x}) =
\frac{\dot\alpha_t}{\alpha_t}\,\mathbf{x}
+ \left(\frac{\dot\alpha_t}{\alpha_t}\sigma_t^2
- \dot\sigma_t\sigma_t\right)
\nabla_{\mathbf{x}}\log p_t(\mathbf{x}) ,
$$

e la relazione si inverte. Le due formulazioni sono quindi la stessa famiglia
di modelli in due parametrizzazioni, e la differenza fra le loss è ancora una
volta un peso $w(t)$.

Le tre differenze che restano sono operative e sostanziali:

- **Libertà nel percorso.** $(\alpha_t,\sigma_t)$ diventano una scelta di
  progetto invece che una conseguenza del programma di rumore, e si sceglie il
  percorso che minimizza la curvatura delle traiettorie.
- **Estremi arbitrari.** Il conditional flow matching non richiede
  $p_1=\mathcal{N}(\mathbf{0},\mathbf{I})$: basta poter campionare da entrambe
  le sponde e disporre di un accoppiamento. Da qui la traduzione fra domini e i
  ponti fra distribuzioni, che con la formulazione a SDE richiedevano
  costruzioni ad hoc.
- **Semplicità della loss.** Nessun programma di rumore da tarare, nessun peso
  da riequilibrare a mano, un solo campo da regredire.

Sono queste tre ragioni, e non un vantaggio teorico, ad aver portato le
architetture di punta (Stable Diffusion 3 e i modelli della sua generazione)
alla formulazione rettificata, come racconta la {doc}`sezione sui Diffusion
Transformer </ModelliDiffusione/diffusion-transformer>`.

`````

Una nota di lessico, perché aprendo il codice di una libreria si incontrano
tutti e due i vocabolari nello stesso file. «Programma di rumore» e «percorso
di probabilità» indicano la stessa scelta; «predire il rumore» e «predire la
velocità» sono due uscite della stessa rete, convertibili l'una nell'altra; e
un campionatore chiamato *flow matching Euler* e uno chiamato *DDIM* fanno, sul
percorso rettificato, lo stesso identico passo. Sapere che i due dizionari
traducono lo stesso testo evita di cercare differenze dove ci sono soltanto
nomi diversi.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Nel flow matching si **sceglie il percorso** fra i dati e il rumore, invece di
  ereditarlo dal modo in cui si è deciso di rovinare le immagini. Quello che
  la rete impara è un **campo di venti**: in ogni punto e in ogni istante,
  dove andare e quanto in fretta.
- La velocità giusta in un punto è la media su tutti i viaggi che ci passano, e
  quella media non si sa calcolare. Il trucco è chiedere alla rete la velocità
  di **un solo viaggio**, che è banale perché il viaggio lo abbiamo disegnato
  noi: allenarsi così porta esattamente allo stesso risultato, perché chi deve
  indovinare un numero sorteggiato e viene giudicato sul quadrato dell'errore
  risponde la media.
- Il percorso più semplice è la **retta**, e la sua velocità è costante: la
  differenza fra arrivo e partenza. A quattro passi la retta sbaglia quasi
  venti volte meno del percorso della diffusione, a otto sessanta volte meno.
- Ma la retta è quella del **singolo viaggio**: il campo che ne risulta,
  essendo una media, curva ancora. Con un passo solo si finisce a metà strada
  fra i due gruppi di dati. Il rimedio è il **raddrizzamento**: si guarda dove
  il modello porta ciascuna partenza, e si riaddestra su quelle coppie.
- Diffusione e flow matching sono la stessa famiglia in due linguaggi. Quello
  che il secondo aggiunge sono tre libertà pratiche: scegliere il percorso,
  poter andare da una distribuzione qualsiasi a un'altra qualsiasi, e formule
  più corte.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un flusso è generato da $\dot\Phi_t = \mathbf{u}_t(\Phi_t)$, e la coppia
  $(p_t,\mathbf{u}_t)$ deve soddisfare l’**equazione di continuità**
  $\partial_t p_t + \nabla\!\cdot\!(p_t\mathbf{u}_t)=0$. Il campo compatibile
  con un dato $p_t$ non è unico, e la PF-ODE è un'altra sua soluzione.
- $\mathcal{L}_{\text{FM}}$ è intrattabile; $\mathcal{L}_{\text{CFM}}$, che
  regredisce sulla velocità **condizionata**, ha lo stesso gradiente e lo
  stesso minimo, che è $\mathbb{E}[\mathbf{u}_t(\mathbf{x}\mid\mathbf{z})\mid
  \mathbf{x}_t]$. È il teorema del denoising score matching applicato di nuovo.
- Percorso **rettificato**: $\alpha_t=1-t$, $\sigma_t=t$, velocità condizionata
  $\boldsymbol{\epsilon}-\mathbf{x}_0$, costante nel tempo. La rettitudine è
  **condizionata**: il campo marginale è una media e le sue traiettorie
  curvano ancora. Il **reflow** riaddestra sull'accoppiamento indotto dalla ODE
  e le raddrizza, a costo di generare le coppie e di degradare un poco.
- Il legame con il punteggio è
  $\mathbf{u}_t = \frac{\dot\alpha_t}{\alpha_t}\mathbf{x} +
  (\frac{\dot\alpha_t}{\alpha_t}\sigma_t^2 - \dot\sigma_t\sigma_t)
  \nabla\log p_t$, quindi le due formulazioni differiscono per una
  riparametrizzazione e un peso.
- **Non è trasporto ottimo.** La mappa monotona coincide con la soluzione di
  Monge solo in dimensione uno; in dimensione maggiore il reflow abbassa il
  costo di trasporto senza raggiungere l'ottimo.
```
`````

Con il flow matching il percorso è diventato una scelta di progetto, e la
scelta migliore accorcia il viaggio. Resta il fatto che quel viaggio va
comunque percorso, e che percorrerlo significa risolvere numericamente
un'equazione differenziale: è un problema con una letteratura di due secoli
alle spalle, e conviene usarla.
