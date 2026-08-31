# Da mille passi a uno: insegnare a saltare

Un maestro di scacchi che calcola dieci mosse avanti e un principiante che ne
calcola due giocano lo stesso gioco con costi diversissimi. Se però il maestro
gioca un migliaio di partite e il principiante le studia tutte, può arrivare a
riconoscere le posizioni e a fare la mossa giusta senza calcolare niente: ha
sostituito il ragionamento con la memoria della sua conclusione.

Questi metodi fanno esattamente questo. C'è un modello che sa percorrere bene
la traiettoria, un passo alla volta, e ce n'è un altro che impara a **saltare**
da un capo all'altro. Il mestiere si chiama **distillazione**, come nella
{doc}`sezione sul modello piccolo che imita
</Efficienza/un-modello-piccolo-che-imita>`, con una differenza da mettere
subito in chiaro: là a rimpicciolire era la rete, qui la rete resta grande
uguale e a rimpicciolire è il numero di passi. Non si tratta di percorrere
meglio, che era il mestiere della {doc}`sezione sui campionatori veloci
</ModelliDiffusione/campionatori-veloci>` e ha un limite invalicabile: si
tratta di cambiare l'oggetto che si impara. E il risultato è che oggi
un'immagine può uscire da **una sola** valutazione della rete.

## Insegnare a fare in un passo quello che il maestro fa in due

`````{tab} Elementare

La ricetta più diretta è anche la prima che ha funzionato, e sta in una riga:
si prende il modello che sa fare mille passi, gli si chiede di farne due, e si
addestra un secondo modello a ottenere lo stesso risultato con **un** passo
solo. Il secondo modello a quel punto sa fare in cinquecento passi quello che
il primo faceva in mille.

E poi lo si rifà. Il secondo modello diventa il maestro, un terzo impara a
dimezzarlo ancora, e si scende a duecentocinquanta. Ogni giro dimezza, quindi
dieci giri portano da mille a uno. Il conto è quello del foglio piegato a metà:
poche piegature, e gli strati sono tantissimi.

Due cose vanno dette perché il metodo si capisca davvero.

La prima è che ogni giro è un addestramento a sé, non una rifinitura, con il
maestro da interrogare due volte per ogni esempio. I giri però si accorciano
man mano, perché a ogni passaggio ci sono meno passi da imitare, e Salimans e
Ho riportano che l'intera catena non costa più che addestrare il modello di
partenza. Il metodo trasforma tempo di generazione in tempo di addestramento, a
un cambio conveniente, ed è per questo che si è diffuso.

La seconda è che il modo in cui la rete descrive la propria risposta diventa
importante. Chiedendole «qual era il disturbo?» si ottiene una risposta da cui
l'immagine va ancora ricavata, e all'inizio del ritorno quel passaggio è
disastroso: lì dell'immagine è rimasto quasi niente, quindi per tirarla fuori
bisogna dividere per quel quasi niente, e ogni errore sul disturbo viene
moltiplicato. All'altro capo succede il rovescio, ed è mal posta la domanda
«com'era l'immagine pulita?». Chi fa passi lunghi attraversa tutti e due gli
estremi in un colpo solo, e gli serve una descrizione che regga a tutti e due:
quella che mescola disturbo e immagine pulita è nata proprio per questo,
insieme a questo metodo.

`````

`````{tab} Superiore

La **progressive distillation** {cite}`salimans2022progressive` addestra uno
studente $\boldsymbol{\epsilon}_\phi$ a riprodurre in un passo il risultato di
due passi del maestro $\boldsymbol{\epsilon}_\theta$. Detto
$\Psi^{(2)}_{t\to t''}$ il risultato di due passi DDIM del maestro da $t$ a
$t''$ passando per $t'$, il bersaglio dello studente è

$$
\mathbf{x}_{t''}^{\text{obiettivo}} = \Psi^{(2)}_{t\to t''}(\mathbf{x}_t),
\qquad
\mathcal{L}(\phi) = \mathbb{E}\Big[w(t)\big\lVert
\Psi^{(1)}_{t\to t''}(\mathbf{x}_t;\phi)
- \mathbf{x}_{t''}^{\text{obiettivo}}\big\rVert^2\Big] .
$$

Terminato il giro, lo studente diventa il maestro e si ripete: $K$ giri
riducono i passi di un fattore $2^K$, quindi dieci giri portano da $1024$ a
$1$.

Due dettagli tecnici sono decisivi e vanno riportati.

Il primo è la **parametrizzazione**. Con passi lunghi, a $t$ **grande**
l'obiettivo sul rumore ha varianza esplosiva perché lì $\alpha_t\to 0$, e
ricostruire
$\hat{\mathbf{x}}_0 = (\mathbf{x}_t-\sigma_t\hat{\boldsymbol{\epsilon}})/\alpha_t$
amplifica ogni errore di $\hat{\boldsymbol{\epsilon}}$ di $1/\alpha_t$;
all'altro estremo vale il contrario. La $\mathbf{v}$-prediction,
$\hat{\mathbf{v}} = \alpha_t \boldsymbol{\epsilon}-\sigma_t\mathbf{x}_0$, nasce
in questo articolo e per questa ragione: resta ben condizionata a entrambi gli
estremi, dove le altre degenerano. Gli autori riportano che anche la predizione
diretta di $\mathbf{x}_0$ regge la distillazione; la $\mathbf{v}$ è quella che
si è imposta.

Il secondo è il **programma di rumore a SNR terminale nullo**. Se all'ultimo
passo $\alpha_T$ non è esattamente zero, il modello a un passo riceve un
ingresso che contiene ancora un residuo di segnale, che in generazione non
c'è: la discrepanza fra addestramento e uso è invisibile con mille passi e
diventa dominante con uno.

Il limite del metodo è strutturale: lo studente insegue le **traiettorie** del
maestro, quindi ne eredita gli errori e non può superarlo. Il tetto è la
qualità del maestro, e ogni giro ci si avvicina da sotto.

`````

## Far combaciare le distribuzioni invece delle traiettorie

`````{tab} Elementare

Inseguire le traiettorie ha un difetto che si capisce con un'immagine.
Immagina di dover imparare a disegnare copiando i tratti di un maestro, uno per
uno, nell'ordine esatto in cui li fa. Se il maestro sbaglia un tratto tu lo
copi, e il tuo massimo è la sua bravura. Ma quello che al committente interessa
è il **disegno finito**, non la sequenza dei tratti: se ne arrivi con uno
altrettanto bello per un'altra strada, va bene uguale, e magari meglio.

Da qui la seconda famiglia di metodi. Invece di chiedere allo studente di
ripercorrere il cammino del maestro, gli si chiede che le immagini che produce
**siano distribuite come** quelle del maestro. È una richiesta più debole sui
singoli casi e più forte nell'insieme, e ha una conseguenza notevole: uno
studente così può risultare **migliore** del maestro, perché non è obbligato a
riprodurne gli errori.

Come si misuri se due mucchi di immagini sono distribuiti allo stesso modo è il
problema tecnico, e le risposte sono due. La prima usa un giudice che impara a
distinguere le immagini vere da quelle dello studente, che è l'idea delle
{doc}`reti avversarie </GAN/overview>`. La seconda, più stabile, confronta le
due distribuzioni attraverso il *verso della salita* di ciascuna: si tiene un
modello che conosce quello dei dati veri e uno che impara quello dello
studente, e si spinge lo studente finché i due non coincidono.

`````

`````{tab} Superiore

La **distillazione per corrispondenza di distribuzione** sostituisce l'errore
sulle traiettorie con una divergenza fra la distribuzione $p_\phi$ dello
studente e quella $p_\theta$ del maestro. La formulazione più usata minimizza
la KL inversa,

$$
D_{\mathrm{KL}}\big(p_\phi \,\|\, p_\theta\big),
$$

il cui gradiente rispetto ai parametri dello studente dipende dalle due
distribuzioni **solo attraverso i loro punteggi**:

$$
\nabla_\phi D_{\mathrm{KL}}
= -\,\mathbb{E}\Big[\big(\nabla\log p_\theta(\mathbf{x})
- \nabla\log p_\phi(\mathbf{x})\big)^\top
\frac{\partial \mathbf{x}}{\partial \phi}\Big] .
$$

Il primo punteggio è il maestro; il secondo si stima con un modello di
diffusione ausiliario addestrato in linea sui campioni dello studente. Il
gradiente si annulla quando i due punteggi coincidono, cioè quando le due
distribuzioni sono uguali. In pratica si aggiunge un termine di ricostruzione
su un piccolo insieme di coppie per ancorare lo studente, e spesso un
discriminatore avversario che accelera la convergenza.

La differenza sostanziale rispetto alla distillazione progressiva è che il
vincolo è **a livello di distribuzione** e non di traiettoria. Ne segue che lo
studente può superare il maestro, cosa impossibile per costruzione nella
famiglia precedente, ed è stato osservato: modelli a uno o due passi che
battono il maestro a molti passi su punteggi percettivi, pur avendo una
copertura delle modalità peggiore. Il prezzo è un addestramento a tre reti in
gioco, con la stabilità che ne consegue.

`````

## La condizione di consistenza

`````{tab} Elementare

C'è un modo di ottenere il salto senza avere un maestro da imitare, e nasce da
un'osservazione tanto semplice che sembra inutile finché non la si guarda bene.

Prendi una traiettoria, quella che va da un certo rumore a una certa immagine.
Adesso mettiti in un punto qualsiasi di quella traiettoria, a metà, a un
quarto, ovunque. La domanda «dove va a finire questo percorso?» ha sempre la
stessa risposta, perché il percorso è uno solo. Punti diversi, istanti diversi,
stessa destinazione.

Sembra un'ovvietà, ed è invece una richiesta fortissima quando la si impone a
una funzione da imparare. Si chiede a una rete di rispondere «dove si va a
finire da qui», e si pretende che risponda **la stessa cosa** per tutti i punti
di una stessa traiettoria. Una rete che soddisfa questa richiesta è, per
definizione, un generatore a un passo: le si dà un rumore qualsiasi e risponde
direttamente l'immagine.

La cosa notevole è che la richiesta si può imporre senza sapere quale sia la
risposta giusta. Basta prendere due punti vicini della stessa traiettoria e
chiedere che la rete risponda uguale su tutti e due; ripetuto su tante coppie,
questo incolla insieme la risposta lungo tutta la traiettoria. E l'ancora che
impedisce alla rete di rispondere sempre la stessa cosa (che soddisferebbe la
richiesta ed è inutile) è il punto di arrivo, dove la risposta giusta si
conosce: a rumore zero, l'immagine è quella che si sta guardando.

`````

`````{tab} Superiore

Sia $\{\mathbf{x}_t\}$ la soluzione della PF-ODE. Si cerca una funzione
$\mathbf{f}_\phi(\mathbf{x},t)$ che soddisfi la **condizione di
autoconsistenza**

$$
\mathbf{f}_\phi(\mathbf{x}_t,t) = \mathbf{f}_\phi(\mathbf{x}_s,s)
\quad\text{per ogni } s,t \text{ sulla stessa traiettoria},
$$

con la **condizione al contorno**
$\mathbf{f}_\phi(\mathbf{x},\varepsilon) = \mathbf{x}$, dove $\varepsilon$ è un
istante piccolo e fissato in cui la traiettoria si ferma invece di arrivare a
zero (e non ha niente a che vedere con il rumore $\boldsymbol{\epsilon}$). La
condizione si impone per costruzione, con una parametrizzazione del tipo

$$
\mathbf{f}_\phi(\mathbf{x},t)
= c_{\text{skip}}(t)\,\mathbf{x} + c_{\text{out}}(t)\,F_\phi(\mathbf{x},t),
\qquad
c_{\text{skip}}(\varepsilon)=1,\; c_{\text{out}}(\varepsilon)=0 .
$$

Senza la condizione al contorno la funzione costante soddisferebbe la
consistenza ed è la soluzione degenere.

Sono i **consistency model** {cite}`song2023consistency`, e si addestrano in
due modi. La **distillazione di consistenza** usa un maestro per fare il passo
$t\to s$ e impone $\mathbf{f}_\phi(\mathbf{x}_t,t)\approx
\mathbf{f}_{\phi^-}(\hat{\mathbf{x}}_s,s)$, con $\phi^-$ una copia a media
mobile dei parametri che fa da bersaglio, e che stabilizza sensibilmente
l'addestramento per la stessa ragione della rete-target del {doc}`Deep
Q-Network </DeepReinforcementLearning/dqn>`, cioè che la stima non deve
inseguire un bersaglio che si muove insieme a lei (là la copia è periodica
invece che a media mobile). L’**addestramento di consistenza** fa a meno del
maestro sostituendo il passo con uno stimatore non distorto del punteggio
ricavato dal rumore iniettato, e addestra da zero.

Il campionamento a un passo è
$\mathbf{x}_\varepsilon = \mathbf{f}_\phi(\mathbf{x}_T,T)$.
Quello a pochi passi alterna salto e re-iniezione di rumore: si salta a
$\hat{\mathbf{x}}_0$, si riporta il risultato al livello di rumore $\tau_k$
sorteggiando, si salta di nuovo. Due o quattro giri di questo tipo recuperano
gran parte del divario con il maestro, ed è il regime in cui questi modelli si
usano davvero.

`````

## Il quadro che tiene insieme tutto: imparare la mappa, non la velocità

`````{tab} Elementare

Le tre famiglie sembrano tre trucchi diversi. Guardate da un passo indietro
sono la stessa mossa.

Un modello di diffusione ordinario impara la **velocità istantanea**: dove
andare adesso, per un tratto infinitesimo. Per sapere dove si finisce bisogna
integrarla, cioè fare tanti passi. Tutti e tre imparano invece qualcosa che
contiene già l'integrale: la **destinazione**, oppure la velocità **media** su
un tratto lungo.

E la velocità media è la chiave, perché rende ovvio il perché del guadagno.
Andando da Milano a Roma la velocità istantanea cambia di continuo, e per
sapere dove si arriva bisogna seguirla minuto per minuto; ma se qualcuno ti
dice la velocità media dell'intero viaggio, la destinazione la calcoli con una
moltiplicazione. Un passo solo, e il risultato è esatto invece che
approssimato, perché la media è definita proprio così.

Da qui la famiglia generale, che invece di una destinazione sola impara la
mappa fra **due istanti qualsiasi**: dove si arriva partendo da qui a
quest'ora e viaggiando fino a quell'altra. Chi impara questa mappa può fare un
passo solo, due, o venti, con lo stesso modello, e scegliere di volta in volta
quanto pagare. I metodi che portano nomi diversi negli articoli sono casi
particolari di questa: la destinazione finale, o il tratto lungo, o quello
corto.

`````

`````{tab} Superiore

Il quadro unificante è quello delle **mappe di flusso** (*flow maps*). Un
modello di diffusione ordinario apprende il campo istantaneo
$\mathbf{u}_t(\mathbf{x})$, e la generazione richiede di integrarlo. I metodi a
pochi passi apprendono invece direttamente l'operatore di soluzione

$$
\Phi_{t\to s}(\mathbf{x}) := \mathbf{x} + \int_t^s
\mathbf{u}_r(\mathbf{x}_r)\,\mathrm{d}r ,
$$

o una sua restrizione. Le famiglie si collocano così:

| metodo | che cosa apprende |
|---|---|
| diffusione, flow matching | $\mathbf{u}_t$, la velocità istantanea |
| consistency model | $\Phi_{t\to 0}$, la sola destinazione |
| consistency trajectory model | $\Phi_{t\to s}$ per $s$ qualsiasi |
| mean flow | $\bar{\mathbf{u}}(\mathbf{x},t,s) := \frac{1}{s-t}\int_t^s \mathbf{u}_r\,\mathrm{d}r$ |

L'ultima riga rende evidente perché un passo basti: per definizione
$\Phi_{t\to s}(\mathbf{x}) = \mathbf{x} + (s-t)\,\bar{\mathbf{u}}$, quindi un
singolo passo di Eulero con la velocità **media** è esatto, mentre lo stesso
passo con la velocità **istantanea** ha errore $O((s-t)^2)$. Tutto il guadagno
sta in questa sostituzione, e tutto il costo sta nel fatto che
$\bar{\mathbf{u}}$ dipende da due tempi invece che da uno, quindi la rete ha un
ingresso in più e un problema più difficile da approssimare.

La consistenza in tempo continuo si scrive come una condizione differenziale
sulla mappa,
$\frac{\mathrm{d}}{\mathrm{d}t}\Phi_{t\to s}(\mathbf{x}_t)=\mathbf{0}$ lungo
la traiettoria, che sviluppata dà
$\partial_t\Phi + (\nabla_{\mathbf{x}}\Phi)\,\mathbf{u}_t = \mathbf{0}$: un
prodotto Jacobiano-vettore, calcolabile con la differenziazione automatica in
avanti a costo di una passata in più. È la forma in cui i metodi più recenti
addestrano senza discretizzare {cite}`lai2026principles`.

`````

Le due identità che reggono i generatori a un passo si controllano in dieci
righe, sullo stesso banco di prova delle sezioni precedenti: la consistenza
(fermarsi a metà strada e ripartire porta allo stesso posto) e l'esattezza del
passo singolo con la velocità media.

```python
import numpy as np

MODI = np.array([-1.5, 1.5])
T_MIN = 1e-3
B_MIN, B_MAX = 0.1, 20.0
beta = lambda t: B_MIN + t * (B_MAX - B_MIN)
alpha = lambda t: np.exp(-0.5 * (B_MIN * t + 0.5 * (B_MAX - B_MIN) * t**2))
sigma = lambda t: np.sqrt(1 - alpha(t)**2)

def eps(x, t):
    a, s = alpha(t), sigma(t)
    d = x[:, None] - a * MODI[None, :]
    w = np.exp(-0.5 * (d / s)**2)
    w /= w.sum(axis=1, keepdims=True)
    return (w * d).sum(axis=1) / s

campo = lambda x, t: -0.5 * beta(t) * x + 0.5 * beta(t) / sigma(t) * eps(x, t)

def percorri(x, da, a, passi=400):
    ts = np.linspace(da, a, passi + 1)
    for k in range(passi):
        x = x + (ts[k + 1] - ts[k]) * campo(x, ts[k])
    return x

rng = np.random.default_rng(0)
z = rng.normal(size=8)

destinazione = percorri(z.copy(), 1.0, T_MIN)     # f(z, 1): dove si va a finire
print(np.round(destinazione, 4))
# -> [ 1.4842 -1.4845  1.4995  1.4829 -1.4972  1.493   1.5109  1.5051]

# autoconsistenza: fermarsi lungo la strada e ripartire da li'
for t_meta in (0.6, 0.3, 0.1):
    a_meta = percorri(z.copy(), 1.0, t_meta)
    scarto = np.abs(percorri(a_meta, t_meta, T_MIN) - destinazione).max()
    print(t_meta, round(float(scarto), 6))
# -> 0.6 0.000888
# -> 0.3 0.001641
# -> 0.1 0.002068
```

Gli scarti sono dell'ordine del millesimo, e sono l'errore dei passi grossolani
usati per la prova: la consistenza vale esattamente sulla traiettoria vera. È
la proprietà che una rete addestrata a rispettarla eredita, e che la rende un
generatore a un passo.

```python
velocita_media = (destinazione - z) / (T_MIN - 1.0)
print(np.round(z + (T_MIN - 1.0) * velocita_media, 4))
# -> [ 1.4842 -1.4845  1.4995  1.4829 -1.4972  1.493   1.5109  1.5051]

print(np.round(z + (T_MIN - 1.0) * campo(z, 1.0), 4))
# -> [ 0.1258 -0.1322  0.6408  0.105  -0.536   0.3618  1.3047  0.9476]
```

Le due righe dicono tutto quello che c'è da dire sui generatori a un passo. Un
solo passo con la velocità **media** riproduce la destinazione cifra per cifra,
perché la media è definita esattamente così; lo stesso passo con la velocità
**istantanea** finisce ovunque tranne che sui due modi. La difficoltà non sta
nel campionare, sta nell'imparare quella media.

## Che cosa si perde

Il conto va chiuso onestamente, perché i modelli a pochi passi hanno tre costi
che i confronti pubblicati non sempre mettono in evidenza.

Il primo è la **varietà**. Comprimere mille passi in uno significa affidare a
una sola valutazione tutta la scelta, e le distribuzioni che ne escono sono
sistematicamente più concentrate: i giudizi sulla qualità visiva restano buoni,
ma il modello copre meno tipi di immagine diversi. Sui volti la cosa si vede a
occhio, generando qualche centinaio di campioni.

Il secondo è la **controllabilità**. Le tecniche della {doc}`sezione su guida e
allineamento </ModelliDiffusione/guida>` agiscono modificando la direzione a
ogni passo; con un passo solo il punto in cui intervenire è uno, e la guida va
incorporata nella distillazione invece che applicata al momento. È il motivo
per cui un modello distillato arriva spesso con la forza della guida già
fissata dentro.

Il terzo è che **il confronto onesto si fa a parità di valutazioni della
rete**, e non a parità di passi. Un metodo a due passi che ne usa due per passo
costa quanto uno a quattro passi, e le tabelle degli articoli non sempre lo
dichiarano.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La prima ricetta è la **distillazione progressiva**: uno studente impara a
  fare in un passo quello che il maestro fa in due, poi diventa maestro a sua
  volta. Dieci giri portano da mille passi a uno, e nel complesso costano
  quanto addestrare il modello di partenza; lo studente però non può superare
  il maestro, perché ne copia i tratti.
- La seconda chiede una cosa più debole sul singolo caso e più forte
  nell'insieme: che le immagini prodotte **siano distribuite come** quelle del
  maestro. Così lo studente può risultare migliore, perché non è obbligato a
  copiarne gli errori.
- La terza parte da un'ovvietà: da qualunque punto di una traiettoria, la
  destinazione è la stessa. Chiedere a una rete di rispondere sempre la stessa
  cosa lungo una traiettoria la trasforma in un generatore a un passo, e
  l'ancora che impedisce la risposta banale è il punto di arrivo, dove la
  risposta si conosce.
- Tutte e tre fanno la stessa mossa: invece della **velocità istantanea**
  imparano qualcosa che contiene già l'integrale, cioè la destinazione o la
  **velocità media** sul tratto. Un passo con la velocità media è esatto per
  definizione; con quella istantanea non lo è, e il conto lo mostra.
- I costi: meno varietà nelle immagini, guida più difficile da applicare
  dopo, e confronti che vanno letti a parità di valutazioni della rete e non
  di passi.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- **Distillazione progressiva**: lo studente riproduce in un passo due passi
  DDIM del maestro, e si itera; $K$ giri dividono i passi per $2^K$. Vuole una
  parametrizzazione ben condizionata a entrambi gli estremi, di solito la
  $\mathbf{v}$-prediction, e un programma di rumore a SNR terminale nullo; ha
  come tetto la qualità del maestro.
- **Corrispondenza di distribuzione**: si minimizza
  $D_{\mathrm{KL}}(p_\phi\|p_\theta)$, il cui gradiente dipende dalle due
  distribuzioni solo attraverso i punteggi; il secondo si stima con un modello
  ausiliario addestrato in linea. Il vincolo è distribuzionale, quindi lo
  studente può superare il maestro.
- **Consistency model**: $\mathbf{f}_\phi(\mathbf{x}_t,t)=
  \mathbf{f}_\phi(\mathbf{x}_s,s)$ lungo la PF-ODE, con
  $\mathbf{f}_\phi(\mathbf{x},\varepsilon)=\mathbf{x}$ imposta dalla
  parametrizzazione $c_{\text{skip}}\mathbf{x}+c_{\text{out}}F_\phi$. Si
  addestra per distillazione o da zero, con una rete bersaglio a media mobile.
- Il quadro unificante è quello delle **mappe di flusso**: si apprende
  $\Phi_{t\to s}$ invece di $\mathbf{u}_t$. Poiché $\Phi_{t\to s}(\mathbf{x}) =
  \mathbf{x}+(s-t)\bar{\mathbf{u}}$, un passo di Eulero con la velocità
  **media** è esatto, mentre con quella istantanea l'errore è $O((s-t)^2)$. La
  consistenza in tempo continuo è
  $\partial_t\Phi + (\nabla_{\mathbf{x}}\Phi)\mathbf{u}_t=\mathbf{0}$, un
  prodotto Jacobiano-vettore.
- I costi da dichiarare: copertura delle modalità inferiore, guida da
  incorporare nella distillazione, e confronti da fare a parità di valutazioni
  della rete.
```
`````

Resta l'ultima domanda pratica, quella che separa un generatore da uno
strumento: come si dice a un modello **che cosa** generare. Fin qui il modello
produce campioni dalla distribuzione che ha imparato, e basta; la sezione su
guida e allineamento mostra come si piega quella distribuzione verso ciò che si
vuole, e quanto è possibile piegarla prima che si rompa.
