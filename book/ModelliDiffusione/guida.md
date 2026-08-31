# Piegare la generazione: guida, vincoli, preferenze

Il sorpasso sulle GAN del 2021 lo firmano Prafulla Dhariwal e Alex Nichol
{cite}`dhariwal2021diffusion`, e lo devono per metà a un'architettura migliore
e per metà a un'aggiunta che, a leggerla, sembra un espediente: si addestra un
classificatore a riconoscere le categorie **su immagini rumorose**, e durante
la generazione si spinge l'immagine, a ogni passo, nella direzione che al
classificatore piace di più.

L'espediente non era un espediente. Discendeva da una riga di teorema di
Bayes, e quella riga tiene insieme tutti i modi di dire a un modello che cosa
generare: come gli si chiede di seguire un testo, come gli si impone un
vincolo che non aveva mai visto, e come lo si piega verso quello che le
persone preferiscono. Dice anche che cosa quelle tecniche fanno alla
distribuzione dei risultati, e la risposta è più violenta di quella che di
solito si racconta.

## Una riga di Bayes, e tutte le guide

`````{tab} Elementare

Il modello sa una cosa sola: in ogni punto, in che direzione muoversi per
rendere l'immagine più credibile. Chiamiamola la bussola della verosimiglianza.
Quello che manca è una seconda bussola, che indichi la direzione verso ciò che
si è chiesto.

Il conto che le mette insieme è quello che si fa ogni volta che si aggiorna
un'opinione con un indizio, e sta in una riga: **la direzione verso «immagini
credibili che contengono un gatto» è la direzione verso «immagini credibili»
più la direzione verso «cose che a un riconoscitore di gatti sembrano gatti»**.
Due bussole che si sommano, e la somma è la strada.

Da qui le due ricette. La prima, quella del 2021, prende la seconda bussola da
un riconoscitore addestrato a parte: si guarda quanto il riconoscitore è
convinto, si calcola in che direzione ritoccare i pixel per convincerlo di
più, e si somma. Funziona, e ha due costi: bisogna avere un riconoscitore, e
bisogna averlo addestrato su immagini rumorose, perché è su quelle che verrà
interrogato (un riconoscitore normale, davanti a un'immagine mezza distrutta,
risponde a caso).

La seconda ricetta, quella che si usa oggi, si accorge che il riconoscitore si
può togliere: la stessa rete, interrogata due volte (con e senza la richiesta),
fornisce la differenza fra le due bussole senza bisogno di nessun altro pezzo.
La {doc}`sezione su Stable Diffusion </ModelliDiffusione/stable-diffusion>` la
racconta per esteso: è lì che si vede perché il classificatore si può
togliere.

Resta però un punto che quel racconto lascia in ombra. La riga di Bayes dice
di sommare le due bussole così come sono, senza moltiplicare la seconda per
niente; nella pratica la si moltiplica per sette, o per dieci. Quel numero si
chiama la **forza della guida**, il conto non lo prevede, ed è stato aggiunto
perché funziona. Quello che fa alla distribuzione dei risultati si può
misurare invece che raccontare.

`````

`````{tab} Superiore

Dal teorema di Bayes, $p(\mathbf{x}\mid c)\propto p(\mathbf{x})\,p(c\mid
\mathbf{x})$, e prendendo il gradiente del logaritmo rispetto a $\mathbf{x}$ la
costante di normalizzazione sparisce:

$$
\nabla_{\mathbf{x}}\log p_t(\mathbf{x}\mid c)
= \nabla_{\mathbf{x}}\log p_t(\mathbf{x})
+ \nabla_{\mathbf{x}}\log p_t(c\mid\mathbf{x}) .
$$

Il primo termine è il punteggio che il modello già conosce; il secondo è
l'unica cosa che il condizionamento aggiunge. Tutte le tecniche di guida sono
modi diversi di procurarsi quel secondo termine.

**Classifier guidance** {cite}`dhariwal2021diffusion`. Si addestra un
classificatore $p_\psi(c\mid
\mathbf{x}_t, t)$ sui dati **rumorosi** a tutti i livelli, e si usa

$$
\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta
- s\,\sigma_t\,\nabla_{\mathbf{x}}\log p_\psi(c\mid\mathbf{x}_t,t),
$$

con $s$ la forza. Il fattore $\sigma_t$ converte fra punteggio e predizione del
rumore. Il requisito del classificatore addestrato sul rumore è la parte
scomoda: un classificatore ordinario ha gradienti privi di senso su un ingresso
fuori distribuzione.

**Classifier-free guidance** {cite}`ho2022classifier`. Si osserva che, per la
stessa riga di Bayes,
$\nabla\log p_t(c\mid\mathbf{x}) = \nabla\log p_t(\mathbf{x}\mid c)
- \nabla\log p_t(\mathbf{x})$, cioè la differenza fra le due predizioni della
stessa rete. Da qui

$$
\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t,
\varnothing) + w\big(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,c)
- \boldsymbol{\epsilon}_\theta(\mathbf{x}_t,\varnothing)\big) .
$$

**Il punto che va dichiarato.** Per $w=1$ questa è esattamente
$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,c)$, cioè il campionamento dalla
condizionata vera, e la formula di Bayes è rispettata; per $w>1$ non lo è più,
e si estrapola oltre il punto che il teorema autorizza. Quello che resta da
mettere in conto è la lettura di ricambio che circola, cioè che il risultato
equivalga a campionare dalla distribuzione inclinata

$$
q_w(\mathbf{x}) \;\propto\; p(\mathbf{x})\,p(c\mid\mathbf{x})^{w},
$$

lettura che sarebbe corretta se la guida si applicasse una volta sola, alla
distribuzione finale. Applicata a ogni istante lungo la traiettoria non lo è,
perché la composizione di punteggi inclinati a tempi diversi non è il punteggio
dell'inclinata al tempo finale {cite}`du2023reduce`, e Bradley e Nakkiran
{cite}`bradley2024classifier` lo dimostrano per i due campionatori in uso. La
differenza si misura.

`````

## Che cosa la guida fa davvero alla distribuzione

Il banco di prova è quello che il capitolo usa da qualche sezione, e qui
guadagna una cosa. Al posto delle immagini ci sono numeri su una riga,
raccolti in due mucchietti, e adesso ogni mucchietto ha anche una sua
larghezza, così si può misurare di quanto la guida lo stringe. Delle due
bussole si conosce la formula esatta, quindi quello che si misura è la guida e
non gli errori di una rete.

```python
import numpy as np

MU, S, T_MIN = np.array([-1.5, 1.5]), 0.5, 1e-3
B_MIN, B_MAX = 0.1, 20.0
beta = lambda t: B_MIN + t * (B_MAX - B_MIN)
alpha = lambda t: np.exp(-0.5 * (B_MIN * t + 0.5 * (B_MAX - B_MIN) * t**2))

def score_condizionato(x, t, k):
    """Con la classe nota i dati sono una gaussiana sola: punteggio esatto."""
    a = alpha(t)
    v = a * a * S * S + (1 - a * a)
    return -(x - a * MU[k]) / v

def score_libero(x, t):
    a = alpha(t)
    v = a * a * S * S + (1 - a * a)
    d = x[:, None] - a * MU[None, :]
    w = np.exp(-0.5 * d * d / v)
    w /= w.sum(axis=1, keepdims=True)
    return -(w * d).sum(axis=1) / v

def genera(forza, n=20000, passi=400):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    ts = np.linspace(1.0, T_MIN, passi + 1)
    for i in range(passi):
        t, dt = ts[i], ts[i + 1] - ts[i]
        libero = score_libero(x, t)
        guidato = libero + forza * (score_condizionato(x, t, 1) - libero)
        x = x + dt * (-0.5 * beta(t) * x - 0.5 * beta(t) * guidato)
    return x

# la distribuzione che la lettura corrente attribuisce alla guida
xg = np.linspace(-8, 8, 400_001)
gauss = lambda m: np.exp(-0.5 * ((xg - m) / S)**2)
p = 0.5 * gauss(MU[0]) + 0.5 * gauss(MU[1])
p_classe = gauss(MU[1]) / (gauss(MU[0]) + gauss(MU[1]))

print("forza |  campionatore vero  |  p(x) p(c|x)^w")
for forza in (1.0, 3.0, 7.5):
    x = genera(forza)
    q = p * p_classe**forza
    q /= np.trapezoid(q, xg)
    m = np.trapezoid(xg * q, xg)
    s = np.sqrt(np.trapezoid((xg - m)**2 * q, xg))
    print(f"{forza:5.1f} |  {x.mean():+.3f}  {x.std():.3f}      |  "
          f"{m:+.3f}  {s:.3f}")
# -> forza |  campionatore vero  |  p(x) p(c|x)^w
# ->   1.0 |  +1.499  0.498      |  +1.500  0.500
# ->   3.0 |  +2.019  0.311      |  +1.505  0.494
# ->   7.5 |  +2.570  0.247      |  +1.508  0.490
```

`````{tab} Elementare

I dati veri, per la classe scelta, hanno centro in $1{,}5$ e larghezza $0{,}5$.
Con la forza a uno il campionatore li ritrova, e l'ultimo millesimo di scarto è
l'errore dei passi finiti: è la riga di controllo che dice che il conto è
giusto.

La colonna di destra è la previsione. C'è una formula che circola e che dice
quanto la guida dovrebbe spostare le cose, e quella colonna la applica: prevede
che alzando la forza fino a sette e mezzo il centro si sposti di otto millesimi
e la larghezza cali di un centesimo, cioè quasi niente.

Quello che succede davvero è un'altra storia, e sono due cose insieme. La
distribuzione **si restringe**: da $0{,}5$ scende a $0{,}31$ e poi a $0{,}25$,
cioè a metà. E **si sposta**: il centro passa da $1{,}5$ a $2{,}02$ e poi a
$2{,}57$, che è nella coda della distribuzione vera, dove di esempi ce n'era
uno su sessanta.

Il secondo effetto è quello che in pratica si vede e si chiama «sovracottura»:
colori più saturi del vero, contrasti più duri, composizioni tutte uguali. Qui
i numeri stanno al posto dei pixel, e uscire dalla zona dei dati vuol dire
mettere colori più accesi di quanti se ne trovino nelle foto vere. A forza
sette e mezzo, che è il valore con cui Stable Diffusion esce di serie, i
campioni non vengono dalla distribuzione dei dati con quell'etichetta: vengono
da una distribuzione più stretta della metà e centrata un'unità più in là.

Le cure che si usano derivano tutte da questa lettura. Si può **rimettere in
scala** l'immagine guidata, riportandone l'ampiezza a quella che aveva la
bussola condizionata da sola: si smorza l'esagerazione senza cambiare la
direzione, e in pratica si mescolano le due versioni, perché il riscalamento
puro dà immagini spente. Si può **tagliare i valori estremi** a ogni passo, non
a una soglia decisa una volta per tutte ma a un livello scelto ogni volta
guardando quanto sono grandi i valori di quell'immagine. E si può **accendere
la guida solo in un tratto** del percorso invece che lungo tutto: non costa
niente, e per scegliere il tratto bisogna sapere dove il danno nasce.

Detto tutto questo, la guida non è un difetto da togliere. Senza, i generatori
che disegnano su richiesta seguono la richiesta così poco da risultare
inservibili: il gatto viene, l'acquerello no. Il prezzo qui misurato si paga
apposta, e conoscerlo è diverso dal subirlo.

`````

`````{tab} Superiore

La colonna di destra è la distribuzione inclinata $q_w \propto p\,p(c\mid
\cdot)^w$ calcolata per quadratura sulla stessa griglia.

Per $w=1$ le due colonne coincidono, come devono: la guida a uno è il
campionamento dalla condizionata.

Per $w>1$ divergono, e non di poco. Il campionatore produce media $2{,}57$ e
deviazione $0{,}247$ a $w=7{,}5$, contro $1{,}508$ e $0{,}490$ della
distribuzione inclinata: la media è spostata di più di due deviazioni
standard dei dati veri, e la larghezza è dimezzata. **La guida applicata lungo
la traiettoria non campiona dalla distribuzione inclinata**, e la formula che
si cita per giustificarla descrive un oggetto diverso da quello che si ottiene.

La ragione è che l'operazione di inclinazione e quella di diffusione non
commutano. Il punteggio inclinato al tempo $t$,
$\nabla\log p_t + w\nabla\log p_t(c\mid\cdot)$, non è il punteggio al tempo $t$
della distribuzione che si otterrebbe diffondendo $q_w$: l'errore si accumula a
ogni passo e produce una deriva sistematica verso l'esterno del supporto.

Le mitigazioni in uso derivano tutte da questa diagnosi:

- **CFG rescale** {cite}`lin2024common`: si riporta la deviazione standard
  della predizione guidata a quella della predizione **condizionata**, non
  della non condizionata, e si mescola il risultato con la predizione guidata
  secondo un peso $\phi$, perché il riscalamento puro spegne le immagini.
  Corregge la deriva di scala senza toccare la direzione.
- **Dynamic thresholding** {cite}`saharia2022photorealistic`: a ogni passo si
  satura $\hat{\mathbf{x}}_0$ a un quantile alto dei suoi valori assoluti
  invece che a una soglia fissa, e si riscala. Corregge la fuoriuscita
  dall'intervallo dei dati.
- **Intervallo di guida** {cite}`kynkaanniemi2024guidance`: si applica la guida
  solo per $t$ in una fascia, lasciando $w=1$ agli estremi. Le due ragioni non
  sono simmetriche: ad alto rumore la guida amplifica una differenza che nei
  modelli veri resta grande fra le due predizioni, e fa danno; a basso rumore
  non fa danno, semplicemente non serve, e spegnerla costa meno senza cambiare
  quasi niente nel risultato. È la mitigazione con il miglior rapporto fra
  guadagno e costo, perché è gratuita.

Resta da dire, per non lasciare l'impressione che la guida sia un errore, che
senza di essa i modelli condizionati al testo producono immagini che seguono la
richiesta troppo poco per essere utili, e i punteggi percettivi con guida sono
molto migliori. Lo scambio è deliberato, e conoscerne il prezzo esatto è
diverso dal subirlo.

`````

## Guidare con qualunque misura, senza riaddestrare

`````{tab} Elementare

La riga di Bayes non chiede che la seconda bussola venga da un classificatore.
Chiede solo che ci sia un modo di dire **quanto un'immagine è vicina a quello
che si vuole**, e che quel modo si possa derivare, cioè che sappia rispondere
alla domanda «in che direzione ritoccare i pixel per migliorare un po'?».

Quel modo può essere qualsiasi cosa. La somiglianza a una fotografia di
riferimento, la fedeltà a un disegno di contorni, il rispetto di una palette di
colori, l'assenza di volti riconoscibili, il verdetto di un modello che valuta
la qualità estetica; e fuori dalle immagini, l'energia di una molecola o
l'aderenza a una legge fisica. Nessuna di queste richiede di toccare il modello
generativo. Si paga però al momento di generare: per sapere in che direzione
ritoccare i pixel bisogna ripercorrere la rete all'indietro, e un passo guidato
è un'andata più un ritorno, contro la sola andata di un passo normale.

C'è un accorgimento che fa la differenza fra funzionare e non funzionare, ed è
istruttivo. La misura va applicata **non all'immagine rumorosa che si ha in
mano, ma alla stima dell'immagine pulita** che il modello sa già produrre in
ogni istante. Un misuratore di qualità estetica davanti a un'immagine mezza
distrutta risponde a caso, e la sua indicazione sarebbe rumore; davanti alla
stima di come quell'immagine finirà, risponde sensatamente. È il motivo per cui
i metodi che aggiungono vincoli a un modello di diffusione hanno tutti la
stessa forma, e passano tutti da quella stima.

C'è un caso in cui questa sola idea copre una famiglia intera di problemi.
Ricostruire la parte mancante di una foto, tirar fuori i dettagli da
un'immagine troppo piccola, rimettere a fuoco una mossa, ricostruire una TAC
dalle sue proiezioni: sono tutte la stessa domanda, «quale immagine, passata
attraverso questo apparecchio, avrebbe dato la misura che ho in mano?». Cambia
l'apparecchio, non il procedimento, e il modello di diffusione fa da memoria di
com'è fatta un'immagine plausibile.

Il limite è che questa è un'approssimazione, non un conto esatto: la stima
dell'immagine pulita è una media su tutti i finali possibili, e all'inizio del
percorso quella media è sfocata e la spinta è imprecisa. Chiedendo troppo si
ottengono immagini che soddisfano la misura e non somigliano più a niente,
perché la spinta ha portato fuori dal territorio delle immagini plausibili;
chiedendo troppo poco, immagini bellissime che la misura non la rispettano.

`````

`````{tab} Superiore

La **guida senza addestramento** (*training-free guidance*) sostituisce
$\nabla\log p_t(c\mid\mathbf{x}_t)$ con il gradiente di una funzione di merito
$h$ qualsiasi, valutata sulla stima del dato pulito:

$$
\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta
- \eta_t\,\sigma_t\,
\nabla_{\mathbf{x}_t}\, h\big(\hat{\mathbf{x}}_0(\mathbf{x}_t)\big),
\qquad
\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t
- \sigma_t\boldsymbol{\epsilon}_\theta}{\alpha_t} ,
$$

dove $\eta_t>0$ è la forza della spinta, che si fa crescere al calare del
rumore (la lettera $\lambda$ fa già due mestieri da queste parti, il peso
dell'ELBO e il logaritmo del rapporto segnale-rumore). Il gradiente rispetto a
$\mathbf{x}_t$ attraversa la rete, quindi ogni passo guidato costa una passata
all'indietro in più.

La giustificazione poggia su due gradini, e solo il primo è esatto. Il primo è
la formula di Tweedie, che dà la stima del dato pulito come media a posteriori,
$\hat{\mathbf{x}}_0 = \mathbb{E}[\mathbf{x}_0\mid\mathbf{x}_t]$, ed è
un'identità, non un'approssimazione (vale a rete ottima, cioè quando
$\boldsymbol{\epsilon}_\theta$ è davvero
$\mathbb{E}[\boldsymbol{\epsilon}\mid\mathbf{x}_t]$). Il secondo è lo scambio
fra la funzione e la media,
$\mathbb{E}[h(\mathbf{x}_0)\mid\mathbf{x}_t]\approx
h(\mathbb{E}[\mathbf{x}_0\mid\mathbf{x}_t])$, che è un salto di Jensen
{cite}`chung2023dps`: esatto solo per $h$ affine, di segno noto per $h$ convessa
o concava, e tanto più largo quanto più la posteriore è larga, cioè per $t$
grande. È la ragione formale per cui la guida si applica con peso crescente
verso la fine, e per cui i metodi migliori stimano anche la covarianza della
posteriore invece di usare la sola media.

La famiglia copre i **problemi inversi** in modo uniforme. Con
$h(\mathbf{x}) = -\lVert \mathbf{A}\mathbf{x}-\mathbf{y}\rVert^2$ e
$\mathbf{A}$ l'operatore di misura si ottengono inpainting (maschera),
super-risoluzione (sottocampionamento), deblurring (convoluzione) e ricostruzione
tomografica, tutti con lo stesso codice e un $\mathbf{A}$ diverso. Il modello
di diffusione fa da prior, e il conto è un campionamento approssimato dalla
posteriore $p(\mathbf{x}\mid\mathbf{y})$.

Il limite è quello dello scambio fra funzione e media, e va riportato: si sta
campionando da un'approssimazione della posteriore, non dalla posteriore. Con
$\eta$ grande il campione soddisfa il vincolo e lascia la varietà dei dati; con
$\eta$ piccolo resta plausibile e non soddisfa il vincolo. Il compromesso non
ha una scelta canonica.

`````

## Quando la guida non basta: piegare il modello

`````{tab} Elementare

La guida agisce al momento della generazione e lascia il modello com'è. C'è un
altro modo, più costoso e più profondo: cambiare il modello perché produca
direttamente quello che si vuole.

Il primo approccio tratta il percorso di generazione come una sequenza di
decisioni e usa gli strumenti del reinforcement learning: si genera, si dà un
voto al risultato, e si rendono più probabili i passi che hanno portato ai voti
alti. Il voto può venire da un giudice automatico addestrato a prevedere il
gradimento umano, o da una misura oggettiva come «quanto il testo descrive
davvero l'immagine».

Il secondo evita del tutto il giudice. Si raccolgono **coppie**: due immagini
per la stessa richiesta, e l'indicazione di quale delle due piace di più. Poi
si addestra il modello a rendere più probabile la preferita e meno probabile
l'altra. Il vantaggio è che si salta il passaggio più fragile dell'intera
catena, cioè addestrare un giudice che rimanga onesto.

C'è un accorgimento che tiene in piedi la cosa. Non si guarda quanto il modello
trova probabile la preferita, ma **di quanto** la trova più probabile di come
la trovava prima di cominciare: si tiene da parte una copia congelata del
modello di partenza e si misura sempre la differenza rispetto a quella. Senza
il paragone, il modo più rapido di rendere probabile la preferita sarebbe
dimenticare tutto il resto.

Il pericolo però non sparisce, cambia solo indirizzo. Un modello addestrato a
massimizzare un voto impara a massimizzare **quel voto**, non la cosa che il
voto doveva misurare: se il giudice premia i colori saturi perché nei dati di
addestramento le foto belle erano sature, il modello impara a saturare tutto;
se premia le immagini con molti dettagli, impara a riempirle di dettagli
inutili. Nel secondo metodo il giudice non c'è, ma il voto sì, nascosto dentro
le coppie: la preferenza raccolta è quella di chi ha guardato, e il modello
impara a piacere a quelle persone in quelle condizioni. Il fenomeno ha un nome,
**reward hacking**, e lo racconta per esteso la
{doc}`sezione sull'esplorazione e la ricompensa
</DeepReinforcementLearning/esplorazione-e-ricompensa>`.

`````

`````{tab} Superiore

Due famiglie, entrambe importate dall'allineamento dei modelli linguistici, che
la {doc}`sezione sul post-addestramento </Transformers/post-training>`
descrive.

**Ottimizzazione della ricompensa.** La catena di denoising si tratta come un
processo decisionale di Markov: stato $\mathbf{x}_t$, azione il passo verso
$\mathbf{x}_{t-1}$, ricompensa $r(\mathbf{x}_0,c)$ assegnata solo alla fine.
Si applica un gradiente di policy sulla catena {cite}`black2024training`,
oppure, quando la ricompensa è differenziabile, si retropropaga direttamente
attraverso il campionatore, il che richiede di tenere in memoria l'intera
traiettoria o di ricalcolarla a tratti.

**Ottimizzazione diretta della preferenza.** Date coppie $(\mathbf{x}^+,
\mathbf{x}^-)$ per lo stesso $c$, si massimizza un obiettivo che dipende dal
rapporto delle verosimiglianze fra modello corrente e modello di riferimento,
senza addestrare un modello di ricompensa. La versione per la diffusione
sostituisce la log-verosimiglianza (intrattabile) con il suo limite
variazionale {cite}`wallace2024diffusion`, ottenendo un obiettivo che si valuta
su un singolo passo di rumore sorteggiato, cioè senza percorrere la traiettoria.
Il costo per coppia resta quello di un addestramento ordinario moltiplicato per
le quattro valutazioni di rete che servono: due modelli, quello corrente e
quello di riferimento, su due immagini.

**Il fallimento tipico** è il *reward hacking*. Il modello di ricompensa è una
funzione appresa su una distribuzione di immagini, e l'ottimizzazione la porta
fuori da quella distribuzione, dove il punteggio è alto e la qualità no; nella
seconda famiglia la ricompensa non è un modello a parte ma resta implicita nel
rapporto di verosimiglianze, e il guasto si sposta sulle preferenze raccolte
invece di sparire. Le contromisure standard sono un termine di
regolarizzazione verso il modello di partenza (una divergenza KL, con il suo
coefficiente), il rimescolamento periodico dei dati di preferenza, e
l'interruzione anticipata guidata da una valutazione indipendente. Nessuna
delle tre risolve il problema, lo rallentano.

`````

## In quale tratto del percorso nasce lo spostamento

Se il danno si concentra in un tratto del percorso, spegnere la guida lì non
costa niente e ne recupera buona parte. Per scoprire quale sia il tratto la si
accende in uno solo per volta e si guarda di quanto ci si è spostati alla fine.
Le finestre vanno prese della stessa ampiezza, se no si confonde la durata con
il livello di rumore.

```python
def genera_a_intervallo(forza, da, a, n=20000, passi=400):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    ts = np.linspace(1.0, T_MIN, passi + 1)
    divari = []
    for i in range(passi):
        t, dt = ts[i], ts[i + 1] - ts[i]
        libero = score_libero(x, t)
        divario = score_condizionato(x, t, 1) - libero
        # fuori dalla finestra la forza torna a uno, cioe' nessuna spinta
        # in piu' di quella che il teorema di Bayes autorizza
        qui = forza if da <= t <= a else 1.0
        if qui != 1.0:
            divari.append(np.abs(divario).mean())
        x = x + dt * (-0.5 * beta(t) * x
                      - 0.5 * beta(t) * (libero + qui * divario))
    return x, np.mean(divari)

def riga(nome, ris):
    x, divario = ris
    print(f"{nome:22} {x.mean():+.3f}   {x.std():.3f}    "
          f"{x.mean() - 1.5:+.3f}   {divario:.3f}")

print(f"{'':22} media  larghezza  scarto  divario")
print(f"{'bersaglio':22} +1.500   0.500")
riga("guida sempre accesa", genera_a_intervallo(7.5, 0.0, 1.0))
for da, a in ((0.8, 1.0), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0.0, 0.2)):
    riga(f"accesa fra {da} e {a}", genera_a_intervallo(7.5, da, a))
# ->                        media  larghezza  scarto  divario
# -> bersaglio              +1.500   0.500
# -> guida sempre accesa    +2.570   0.247    +1.070   0.052
# -> accesa fra 0.8 e 1.0   +1.661   0.492    +0.161   0.028
# -> accesa fra 0.6 e 0.8   +2.030   0.417    +0.530   0.119
# -> accesa fra 0.4 e 0.6   +2.199   0.232    +0.699   0.203
# -> accesa fra 0.2 e 0.4   +1.857   0.240    +0.357   0.137
# -> accesa fra 0.0 e 0.2   +1.546   0.427    +0.046   0.027
```

La misura dice una cosa netta e una che il banco di prova non può dire.

Quella netta: **il danno si fa in mezzo alla strada**, e le due colonne di
destra si muovono insieme. Accendendo la guida soltanto nel primo quinto, dove
il rumore è massimo, il centro si sposta di $0{,}161$, e le due bussole lì
distano $0{,}028$; soltanto nell'ultimo, dove i dati sono vicini, lo
spostamento è $0{,}046$ e la distanza $0{,}027$. Nella fascia fra $0{,}4$ e
$0{,}6$ la distanza sale a $0{,}203$ e lo spostamento a $0{,}699$. La ragione è
tutta lì: a rumore massimo l'immagine è ancora tutta grana, la classe non si
vede, le due bussole indicano quasi la stessa direzione, e moltiplicare per
sette e mezzo una differenza che è quasi zero non sposta niente; alla fine il
soggetto è ormai deciso e di nuovo le due coincidono. È in mezzo che divergono,
ed è lì che la moltiplicazione morde.

Le cinque righe non si sommano, e non devono: $0{,}161$ più $0{,}530$ e via
dicendo fa più di $1{,}070$. Un campione spinto fuori strada in una finestra
viene poi in parte ritirato verso il mucchio dalla bussola libera nelle
finestre seguenti, quindi le spinte si mangiano fra loro invece di
accumularsi.

Quella che il banco non può dire è dove convenga accendere la guida davvero.
Chi la usa sui modelli veri fa l'opposto di quello che questi numeri
suggeriscono: la accende **in mezzo** e la spegne ai due estremi
{cite}`kynkaanniemi2024guidance`. La contraddizione è apparente, perché qui
manca metà del conto. Due gaussiane ad alto rumore si confondono del tutto,
mentre in un modello vero la richiesta decide la composizione fin dai primi
passi e le due predizioni restano lontane: lì la guida fa danno sul serio. E a
basso rumore un banco a una dimensione non ha dettagli fini da saturare. Questi
numeri misurano il costo della guida là dove il banco lo sa produrre; il
guadagno, che è la ragione per cui la guida si usa, un banco con una gaussiana
sola per classe non lo può mostrare affatto.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Tutte le guide vengono da una riga di Bayes: la direzione verso «immagini
  credibili e che contengono un gatto» è la direzione verso «immagini
  credibili» più la direzione verso «cose che sembrano gatti». Il primo pezzo
  il modello lo sa; il secondo è tutto quello che il condizionamento aggiunge.
- Quella riga autorizza a sommare le due bussole così come sono. Alzare la
  **forza della guida** oltre l'uno vuol dire spingersi oltre quello che il
  conto autorizza, e ha un prezzo misurabile: sul banco di prova la
  distribuzione si restringe alla metà e il suo centro finisce nella coda,
  dove di esempi ce n'era uno su sessanta.
- La guida non è però un difetto da togliere: senza, la richiesta viene
  seguita così poco che il risultato non serve a niente. Il prezzo si paga
  apposta.
- Le cure sono tre, e la più economica costa zero: accendere la guida solo in
  un tratto del percorso. Sul banco lo scarto si concentra in mezzo alla
  strada, dove le due bussole divergono di più; ai due estremi, dove indicano
  quasi la stessa direzione, la guida non sposta quasi niente.
- La seconda bussola può venire da **qualunque misura** che sappia dire in che
  direzione ritoccare i pixel: somiglianza a una foto, rispetto di un contorno,
  gradimento estetico, energia di una molecola. Nessun riaddestramento, ma ogni
  passo costa quasi il doppio. L'accorgimento decisivo è applicare la misura
  alla **stima dell'immagine pulita**, non a quella rumorosa.
- Se la guida non basta si cambia il modello, con il reinforcement learning o
  con le preferenze a coppie. Il pericolo è sempre lo stesso: un modello che
  massimizza un voto impara a massimizzare quel voto, non la cosa che il voto
  doveva misurare.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- $\nabla\log p_t(\mathbf{x}\mid c) = \nabla\log p_t(\mathbf{x}) +
  \nabla\log p_t(c\mid\mathbf{x})$: **classifier guidance** stima il secondo
  termine con un classificatore addestrato sui dati rumorosi,
  **classifier-free** con la differenza fra due predizioni della stessa rete.
- Per $w=1$ la guida senza classificatore campiona dalla condizionata; per
  $w>1$ non campiona da $q_w\propto p\,p(c\mid\cdot)^w$, contrariamente a
  quanto si legge spesso, perché inclinazione e diffusione non commutano
  {cite}`bradley2024classifier`. A $w=7{,}5$ il campionatore dà media
  $2{,}570$ e deviazione $0{,}247$ contro $1{,}508$ e $0{,}490$ dell'inclinata.
- Mitigazioni: **CFG rescale** (riporta la scala a quella della predizione
  condizionata, mescolando), **dynamic thresholding** (satura a un quantile),
  **intervallo di guida** (guida solo in un tratto). Sul banco lo scarto si
  concentra a metà percorso: guidare soltanto nell'ultimo quinto lo riduce da
  $1{,}070$ a $0{,}046$. Nei modelli veri la fascia utile è quella centrale,
  perché lì il divario fra le due predizioni non si annulla come qui.
- **Guida senza addestramento**: si sostituisce il secondo termine con
  $\nabla h(\hat{\mathbf{x}}_0)$ per una $h$ qualsiasi derivabile. Copre
  inpainting, super-risoluzione, deblurring e tomografia con lo stesso codice.
  Poggia su due gradini: la formula di Tweedie, che è esatta, e lo scambio
  $\mathbb{E}[h(\mathbf{x}_0)]\approx h(\mathbb{E}[\mathbf{x}_0])$, che è un
  salto di Jensen, esatto solo per $h$ affine e tanto peggiore quanto più la
  posteriore è larga.
- **Allineamento**: gradiente di policy sulla catena di denoising vista come
  MDP, oppure ottimizzazione diretta della preferenza su coppie. Il fallimento
  tipico è il *reward hacking*, che la regolarizzazione KL rallenta senza
  risolvere.
```
`````

Le due strade restano diverse fino in fondo: la guida non tocca il modello e
paga a ogni generazione, l'allineamento tocca il modello una volta e paga
all'addestramento. Quello che hanno in comune è il modo di sbagliare, cioè
spingere verso un bersaglio più stretto di quello che si voleva davvero, e in
tutti e due i casi il prezzo si può misurare invece di scoprirlo dopo.

Tutto questo, però, dà per scontata una cosa: che lo stato su cui
si lavora sia fatto di numeri che si possono sporcare un pochino alla volta. Un
pixel è un numero, e aggiungergli un centesimo ha senso. Una parola no: fra
«gatto» e «cane» non c'è niente in mezzo, e sommare del rumore gaussiano a un
simbolo non produce un simbolo un po' più rumoroso, produce niente.
