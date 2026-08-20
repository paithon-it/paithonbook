:::{only} html
```{image} _static/logo-light.svg
:class: pt-copertina-logo only-light
:alt: Paithon Book
```

```{image} _static/logo-dark.svg
:class: pt-copertina-logo only-dark
:alt: Paithon Book
```
:::

```{raw} html
<p class="pt-claim" aria-hidden="true">Il Libro di Intelligenza Artificiale che spiega
<span class="pt-claim__el">due</span> <span class="pt-claim__sup">volte.</span></p>
```

# Il Libro di Intelligenza Artificiale che spiega due volte.

L'intelligenza artificiale ha la fama di essere una materia per pochi, e in
giro si trovano soprattutto due cose: articoli che promettono tutto senza
spiegare niente, e testi che spiegano bene ma danno per acquisiti corsi
universitari di analisi e statistica. Questo libro prova a stare nel mezzo, in
italiano, partendo da zero e senza saltare i passaggi difficili. Ogni concetto
che conta è spiegato **due volte**, una con un'analogia di tutti i giorni e una
con la notazione e le formule per intero. Non è la stessa spiegazione in
versione corta e in versione lunga: sono due strade per lo stesso posto, e
prendendone una non perdi l'altra.

```{epigraph}
Invece di cercare di produrre un programma che simuli la mente adulta, perché non provare piuttosto a produrne uno che simuli quella di un bambino?

<p class="attribution">Alan Turing, <i>Computing Machinery and Intelligence</i>,&nbsp;1950</p>
```

% L'attribuzione della citazione qui sopra e’ scritta in HTML perche’ al sito
% serve la classe CSS `attribution`. In stampa il blocco raw sparisce e la
% citazione resterebbe senza autore, quindi si ripete per il solo LaTeX.

:::{only} latex
*Alan Turing, «Computing Machinery and Intelligence», 1950.*
:::

:::{only} html
E **«due volte» ha un secondo senso**, che riguarda come queste pagine sono
scritte. Le stende un'intelligenza artificiale; le rilegge **un'altra AI**,
che alla stesura non ha partecipato e ha un compito solo, cercare l'errore;
e alla fine passano da me, ed è quel passaggio a decidere che cosa resta.
Questo libro è, alla lettera, **l'AI che spiega se stessa**. Perché il metodo
è severo, su che cosa il libro scommette, e perché il segno qui sopra è un
triangolo impossibile, sta nella {doc}`Prefazione </prefazione>`.
:::

:::{only} html
Da che parte stare lo decidi tu, pagina per pagina o una volta per tutte: il
comando è in alto nella barra, <span class="pt-livello-inline"
aria-hidden="true"><svg class="pt-livello__icona
pt-livello__icona--elementare" viewBox="0 0 16 16" width="15"
height="15"><rect x="1.6" y="7" width="5.2" height="6.4" rx="1.3"
fill="currentColor"/><rect x="9.2" y="2.6" width="5.2" height="10.8" rx="1.3"
fill="none" stroke="currentColor" stroke-width="1.4"
opacity="0.55"/></svg><svg class="pt-livello__icona
pt-livello__icona--superiore" viewBox="0 0 16 16" width="15" height="15"><rect
x="1.6" y="7" width="5.2" height="6.4" rx="1.3" fill="currentColor"/><rect
x="9.2" y="2.6" width="5.2" height="10.8" rx="1.3"
fill="currentColor"/></svg></span>. Vale per tutto il libro
in un colpo solo, e si cambia quando vuoi.
:::

:::{only} html
Questa versione online del libro **è gratuita** ed **è in continuo aggiornamento** per stare al passo con le innovazioni di questa materia: aggiungiamo regolarmente sezioni, argomenti ed esempi in Python. Il codice usa **PyTorch**, NumPy e scikit-learn, e per provarlo non devi installare niente. Dove in alto compare **Esegui il codice**, quel capitolo ha un *notebook*: una copia della pagina in cui i blocchi di codice, invece di stare lì solo da leggere, si eseguono uno dopo l'altro. Il collegamento lo apre su Google Colab, un servizio gratuito che fa girare il codice su una macchina di Google (serve un account Google). Qualche pagina è essa stessa un notebook, e si riconosce dall'icona a razzo 🚀 in alto a destra: lì il codice si esegue restando nel libro. Altrove si legge qui e si copia dove preferisci.

Che cosa è cambiato, e quando, sta scritto: {doc}`Aggiornamenti </aggiornamenti>` è il registro delle sezioni nuove e delle correzioni, una voce per pubblicazione, con il link alla pagina toccata. Questa è la versione **{{ versione }}** ({{ data_versione }}).
:::

:::{only} latex
Il libro **è gratuito** ed è in continuo aggiornamento per stare al passo con
questa materia: sezioni, argomenti ed esempi si aggiungono di continuo. Il
codice usa **PyTorch**, NumPy e scikit-learn, e online si esegue senza
installare niente, dentro il libro o su Google Colab. Qui il codice si legge e
si copia; per eseguirlo, l'indirizzo è nel colophon.
:::

::::{only} html
:::{container} pt-scarica
[**Scarica il libro intero in PDF** · versione {{ versione }}](https://github.com/paithon-it/paithonbook/releases/latest/download/paithon-book.pdf)

[**Leggilo nel browser**](https://doi.org/10.5281/zenodo.21947219)

Tutti i capitoli in un file solo, impaginato per la lettura e per la stampa. Dove qui il libro anima una figura, là trovi tre fermi immagine e l'indirizzo per vederla muoversi. Leggerlo nel browser vuol dire aprirlo su Zenodo, l'archivio del CERN dove il libro è depositato: quel collegamento è il suo **DOI**, l'identificativo permanente che porta sempre all'ultima versione, la mostra pagina per pagina e dà la scheda per citarla.
:::
::::

---

## Il percorso

{{ n_capitoli_lettere|capitalize }} capitoli, dall'alfabeto (Python e matematica) alle frontiere (Transformer, modelli di diffusione, agenti LLM, generazione audio, produzione con l'MLOps). Se qui sotto trovi molti nomi che non ti dicono niente, sei nel posto giusto:
è l'elenco di quello che saprai alla fine. I capitoli si leggono nell'ordine, e l'ordine è pensato per costruire un mattone alla volta, ognuno dando per acquisito quello prima; chi le basi già le ha salta dove gli pare.

<div class="pt-chapters"> <a class="pt-card"
href="Introduzione/overview.html"> <span class="pt-card-num"></span> <span
class="pt-card-title">Introduzione</span> <span class="pt-card-desc">Che cos'è
l'intelligenza artificiale, da ELIZA a oggi, e il primo programma da
eseguire.</span> </a> <a class="pt-card" href="Python/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Python</span> <span
class="pt-card-desc">Il linguaggio dell'AI: basi, NumPy, pandas e
matplotlib.</span> </a> <a class="pt-card" href="Matematica/overview.html">
<span class="pt-card-num"></span> <span class="pt-card-title">Richiami di
matematica</span> <span class="pt-card-desc">Algebra lineare, ottimizzazione,
probabilità, teoria dell'informazione, analisi numerica e la matematica di un
LLM.</span> </a> <a
class="pt-card" href="MachineLearning/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Machine
Learning</span> <span class="pt-card-desc">Imparare dai dati: apprendimento
supervisionato, overfitting e metriche, alberi ed ensemble, SVM, clustering e
riduzione della dimensionalità, dati che cambiano.</span> </a> <a
class="pt-card" href="RetiNeurali/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Reti Neurali</span>
<span class="pt-card-desc">Dal percettrone alla backpropagation: come una rete
impara davvero.</span> </a> <a class="pt-card" href="PyTorch/overview.html">
<span class="pt-card-num"></span> <span class="pt-card-title">PyTorch</span>
<span class="pt-card-desc">Tensori, autograd, moduli e training loop: il
framework del libro.</span> </a> <a class="pt-card" href="GPU/overview.html">
<span class="pt-card-num"></span> <span class="pt-card-title">GPU e calcolo
parallelo</span> <span class="pt-card-desc">Sotto il cofano dell'hardware:
architettura, memoria, kernel, GEMM e tensor core, Flash Attention e
parallelismo su più GPU.</span> </a> <a class="pt-card" href="Efficienza/overview.html"> <span class="pt-card-num"></span> <span class="pt-card-title">Efficienza</span> <span class="pt-card-desc">Il modello che si addestra non è quello che si usa: quantizzazione e componenti anomale, potatura e biglietto della lotteria, distillazione, e perché starci in memoria non è rispondere in fretta.</span> </a> <a class="pt-card"
href="DeepLearning/overview.html"> <span class="pt-card-num"></span> <span
class="pt-card-title">Deep Learning</span> <span class="pt-card-desc">Reti
convoluzionali, ottimizzazione, regolarizzazione, architetture storiche e una
rete sola per molti compiti.</span> </a> <a class="pt-card"
href="VisioneArtificiale/overview.html"> <span class="pt-card-num"></span>
<span class="pt-card-title">Visione Artificiale</span> <span
class="pt-card-desc">Classificare, riconoscere, segmentare: le macchine che
vedono.</span> </a> <a class="pt-card"
href="Ricerca/overview.html"> <span class="pt-card-num"></span> <span
class="pt-card-title">Ricerca e pianificazione</span> <span
class="pt-card-desc">Guardare avanti nelle mosse possibili: alberi di ricerca,
euristiche e A*, minimax e potatura alfa-beta, e le tre cose che quando
mancano cambiano tutto.</span> </a> <a class="pt-card"
href="ReinforcementLearning/overview.html"> <span class="pt-card-num"></span>
<span class="pt-card-title">Reinforcement Learning</span> <span
class="pt-card-desc">Imparare per tentativi: i bandit, gli MDP e le funzioni
di valore, i metodi Monte Carlo, il Q-learning.</span> </a> <a class="pt-card"
href="DeepReinforcementLearning/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Deep Reinforcement
Learning</span> <span class="pt-card-desc">Quando il RL incontra le reti
profonde: DQN e policy gradient, controllo continuo, RL basato su modello,
imitazione, offline RL, esplorazione e reward hacking.</span> </a> <a class="pt-card"
href="NaturalLanguageProcessing/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Natural Language
Processing</span> <span class="pt-card-desc">Dal testo ai numeri: token,
embedding, modelli n-gram, reti ricorrenti, traduzione e attenzione, entità,
parsing, dialogo e chatbot.</span> </a> <a
class="pt-card" href="Transformers/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Transformers</span>
<span class="pt-card-desc">L'architettura che ha cambiato tutto:
self-attention, GPT e BERT, i grandi modelli linguistici, Mixture of Experts,
il post-addestramento (RLHF, DPO), retrieval e RAG, multimodalità e
multilingua.</span> </a> <a class="pt-card"
href="AttenzioneLineare/overview.html"> <span class="pt-card-num"></span>
<span class="pt-card-title">Attenzione lineare</span> <span
class="pt-card-desc">Oltre il costo quadratico: dal trucco del kernel alla
delta rule, con RetNet, RWKV e xLSTM.</span> </a> <a class="pt-card"
href="StateSpaceModel/overview.html"> <span class="pt-card-num"></span> <span
class="pt-card-title">State Space Model</span> <span class="pt-card-desc">Da
S4 a Mamba: sequenze a tempo lineare dai sistemi dinamici, fino a Mamba-2 e
Mamba-3.</span> </a> <a class="pt-card" href="VisioneLinguaggio/overview.html">
<span class="pt-card-num"></span> <span class="pt-card-title">Visione e
linguaggio</span> <span class="pt-card-desc">Modelli che vedono e parlano:
addestramento contrastivo, connettori, fusione precoce e il costo del
dettaglio.</span> </a> <a class="pt-card" href="Agenti/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Agenti e applicazioni
LLM</span> <span class="pt-card-desc">Quando gli LLM agiscono: tool use e
ReAct, RAG avanzato, context engineering, valutazione.</span> </a> <a
class="pt-card" href="IngegneriaLLM/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Prompt, contesto e
loop</span> <span class="pt-card-desc">Programmare gli LLM a parole su tre
livelli concentrici: prompt engineering, context engineering e loop
engineering.</span> </a> <a class="pt-card"
href="SistemiMultiAgente/overview.html"> <span class="pt-card-num"></span>
<span class="pt-card-title">Sistemi multi-agente</span> <span
class="pt-card-desc">Quando gli agenti sono molti: costo del coordinamento,
topologie, consenso, apprendimento per rinforzo multi-agente e sciami.</span>
</a> <a class="pt-card" href="Audio/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Audio oltre la
voce</span> <span class="pt-card-desc">Dallo spettrogramma alla generazione:
AudioSet, wav2vec, codec neurali, MusicGen.</span> </a> <a class="pt-card"
href="SpeechRecognition/overview.html"> <span class="pt-card-num"></span>
<span class="pt-card-title">Speech Recognition</span> <span
class="pt-card-desc">Dalla voce al testo e ritorno: allineamento, CTC,
Whisper, sintesi vocale.</span> </a> <a class="pt-card"
href="ModelliLatenti/overview.html"> <span class="pt-card-num"></span> <span
class="pt-card-title">Modelli latenti</span> <span class="pt-card-desc">Spiegare
i dati con qualcosa che non si vede: autoencoder, ELBO e riparametrizzazione,
il latente che il resto del libro monta senza spiegarlo.</span> </a> <a class="pt-card"
href="GAN/overview.html"> <span class="pt-card-num"></span> <span
class="pt-card-title">GAN</span> <span class="pt-card-desc">Due reti che si
sfidano: il gioco avversario che genera immagini.</span> </a> <a
class="pt-card" href="ModelliDiffusione/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Modelli di
Diffusione</span> <span class="pt-card-desc">Rumore e ritorno: da DDPM a
Stable Diffusion e ai Diffusion Transformer.</span> </a> <a class="pt-card"
href="VerosimiglianzaEsatta/overview.html"> <span class="pt-card-num"></span>
<span class="pt-card-title">Verosimiglianza esatta</span> <span
class="pt-card-desc">I modelli che sanno dire quanto &egrave; probabile un
dato: un pixel alla volta, oppure deformando lo spazio.</span> </a> <a
class="pt-card" href="ModelliEnergia/overview.html"> <span
class="pt-card-num"></span> <span
class="pt-card-title">Modelli a energia</span> <span class="pt-card-desc">Il
paesaggio invece della probabilit&agrave;: da Hopfield e Boltzmann allo score
matching, la lingua in cui sono scritte diffusione e JEPA.</span> </a> <a
class="pt-card" href="AutoSupervisione/overview.html"> <span
class="pt-card-num"></span> <span
class="pt-card-title">Auto-supervisione</span> <span class="pt-card-desc">Il
paradigma che regge quasi tutto il libro, detto una volta sola: come si
fabbrica un segnale dai dati, le quattro famiglie (contrasto, distillazione,
ridondanza, ricostruzione) e il dibattito sul rinforzo.</span> </a> <a
class="pt-card" href="WorldModels/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">World Model</span>
<span class="pt-card-desc">Simulatori interni della realtà: dai sogni di Ha
&amp; Schmidhuber alla via di LeCun (JEPA).</span> </a> <a class="pt-card"
href="GraphNeuralNetwork/overview.html"> <span class="pt-card-num"></span>
<span class="pt-card-title">Graph Neural Network</span> <span
class="pt-card-desc">Reti che imparano su dati a grafo: message passing, GCN,
GraphSAGE e GAT, dalle molecole alla scoperta di farmaci e alle
raccomandazioni.</span> </a> <a class="pt-card"
href="SistemiRaccomandazione/overview.html"> <span class="pt-card-num"></span>
<span class="pt-card-title">Sistemi di Raccomandazione</span> <span
class="pt-card-desc">Dal Netflix Prize alla raccomandazione neurale: i modelli
che scelgono per noi.</span> </a> <a class="pt-card"
href="SerieTemporali/overview.html"> <span class="pt-card-num"></span> <span
class="pt-card-title">Serie temporali</span> <span
class="pt-card-desc">Prevedere dal passato: decomposizione e modelli classici
(ARIMA, Holt-Winters), validazione temporale e metriche, forecasting neurale
(TCN, DeepAR, N-BEATS, Transformer e foundation model).</span> </a> <a
class="pt-card" href="PINN/overview.html"> <span class="pt-card-num"></span>
<span class="pt-card-title">Physics-Informed Neural Networks</span> <span
class="pt-card-desc">Le leggi della fisica dentro la loss: reti che rispettano
le equazioni differenziali.</span> </a> <a class="pt-card"
href="MLOps/overview.html"> <span class="pt-card-num"></span> <span
class="pt-card-title">MLOps</span> <span class="pt-card-desc">Dal notebook
alla produzione: versioning, serving, monitoraggio del drift, LLMOps e
deploy.</span> </a> <a class="pt-card"
href="Interpretabilita/overview.html"> <span class="pt-card-num"></span> <span
class="pt-card-title">Interpretabilità e XAI</span> <span
class="pt-card-desc">Aprire la scatola nera: modelli trasparenti e importanza
delle feature, spiegazioni locali (LIME, SHAP, controfattuali), attribuzione
(Grad-CAM, integrated gradients) e interpretabilità meccanicistica.</span>
</a> <a class="pt-card" href="AIResponsabile/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">AI responsabile</span>
<span class="pt-card-desc">Equità e bias algoritmico, privacy (differential
privacy, federated learning) e robustezza agli attacchi avversari, attaccare e
difendere un LLM (prompt injection, jailbreak), allineamento (RLHF, DPO) e
governance (l'AI Act europeo).</span> </a> <a
class="pt-card" href="Conclusioni/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Conclusioni</span>
<span class="pt-card-desc">Bilancio del viaggio e letture per
proseguire.</span> </a> </div>

---

## Livelli di Complessità

Il libro presenta due livelli di complessità: **Elementare** e **Superiore**:

- *Elementare*: racconta il concetto con un'analogia di tutti i giorni, senza
  dare niente per acquisito. Se l'argomento è nuovo, si parte da qui.

- *Superiore*: lo rifà con la notazione, le formule e i riferimenti. Presume
  le basi, e arriva dove l'Elementare non può.

Nessuno dei due è il riassunto dell'altro, e non c'è da sceglierne uno per
sempre: si può leggere un capitolo tutto all'Elementare, uno tutto al
Superiore, oppure lo stesso due volte, prima di qua e poi di là.

:::{only} html
**Come si sceglie.** Ogni concetto ha due schede, e si apre quella che si
vuole. L'icona a **due barrette** in alto nella barra, invece, imposta il
livello di **tutto il libro** in un colpo, e la scelta resta salvata nel
browser: la ritrovi al ritorno, capitolo dopo capitolo. Le due cose convivono,
e l'interruttore decide solo da dove *parte* ogni pagina: se leggi
all'Elementare e vuoi sbirciare una formula, sbirciala, due paragrafi più
sotto sarai di nuovo all'Elementare.
:::

:::{only} latex
Su carta le due schede si leggono una dopo l'altra: prima l'Elementare, poi la
Superiore.
:::

Un esempio: la funzione ReLU, ai due livelli.

`````{tab} Elementare

La ReLU è un filtro, e per parlare di filtri serve una scrittura che conviene
fissare subito: $f(x)$ si legge «quello che il filtro restituisce quando gli si
dà $x$». La $x$ è il numero che entra, $f(x)$ quello che esce.

Immagina allora uno sportello che smista movimenti di denaro: entra una somma
$x$ in Euro (€) e lo sportello lascia passare soltanto gli importi positivi. Se
$x$ è positivo, esce $x$ tale e quale; se $x$ è negativo, cioè un movimento
che toglierebbe denaro invece di aggiungerne, non passa niente ed esce $0$, e
lo stesso vale per un movimento nullo. Tutto qui: la ReLU lascia passare quello che entra quando è positivo,
e blocca completamente il resto.

Esempi:

$$
\begin{aligned}
x &= 10, & f(x) &= 10 \\
x &= -3, & f(x) &= 0
\end{aligned}
$$

E perché occuparsene? Perché un filtro così banale, ripetuto milioni di volte,
è uno dei mattoni delle reti neurali: è la regola che decide quali segnali
proseguono dentro la rete e quali si fermano lì. Il capitolo sulle reti neurali
racconta perché proprio questa funzioni meglio di alternative in apparenza più
raffinate.

`````

`````{tab} Superiore

La funzione di attivazione ReLU (Rectified Linear Unit) è definita come:

$$
f(x) = x^+ = \max(0, x)
$$


Questa funzione può anche essere definita «a tratti» (*piecewise*) nel seguente modo:

$$
f(x) = \begin{cases}
    x & \text{se } x > 0, \\
    0 & \text{altrimenti}.
\end{cases}
$$

Questa funzione prende un input $x$ e restituisce $x$ se $x$ è positivo; altrimenti, restituisce zero.
La ReLU è ampiamente utilizzata nelle reti neurali perché introduce una non linearità essenziale e la sua derivata vale esattamente $1$ per $x > 0$: durante la *backpropagation*, **lungo i cammini attivi**, il gradiente non si attenua per colpa dell'attivazione, e sparisce la saturazione che affligge sigmoide e tangente iperbolica.

Attenzione però a non chiedere alla ReLU più di quanto dia. Il gradiente che attraversa uno strato è $\mathbf{W}^\top \mathrm{diag}(\mathbb{1}[z>0])$: l'attenuazione la producono i pesi e le unità spente, non l'attivazione. Che il segnale sopravviva a molti strati dipende quindi dalla **scala dell'inizializzazione**, e con quella sbagliata si vede subito: attraversando cinquanta strati di sole ReLU, con la scala di Xavier (giusta per la tangente iperbolica) il gradiente si attenua di sette ordini di grandezza, e con una scala un po’ troppo grande esplode di otto. È proprio la ReLU a richiedere una scala sua, il fattore $2$ di He, perché azzera metà delle unità: la frazione di derivate nulle, misurata, è $0{,}50$.

In $x = 0$ la funzione non è derivabile (il grafico ha un punto angoloso); il sottodifferenziale è l'intervallo $[0, 1]$ e nella pratica, PyTorch compreso, si adotta la convenzione $f'(0) = 0$.

`````

---

## Supporta il progetto

Questo libro è gratuito e lo resterà. Se ti è servito, ecco come puoi darmi
una mano:

:::{only} html
- Metti una ⭐ e condividi il progetto GitHub [Paithon Book](https://github.com/paithon-it/paithonbook)
- Supporta il progetto per contribuire al suo sviluppo ❤️ <span style="display: inline-block; vertical-align: middle;">
    <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="paithon.it" data-color="#B5532C" data-emoji="🔋"  data-font="Cookie" data-text="Ricarica la mia energia" data-outline-color="#1A1A1A" data-font-color="#ffffff" data-coffee-color="#C9A961" ></script>
</span>

- Aiutaci a scoprire errori e migliorare il progetto: selezionando un pezzo di testo in qualunque pagina compare un pulsante che apre una segnalazione (una *issue*) già compilata su GitHub, dove la correzione viene discussa, e chi ha segnalato l'errore è citato nel
commit che la applica.
- Manda i tuoi commenti ✉ a *info@paithon.it*. Servono solo a migliorare il
  libro.
:::

:::{only} latex
- Metti una stella al progetto su GitHub e condividilo:
  `github.com/paithon-it/paithonbook`.
- Aiutaci a scoprire gli errori: ogni pagina del libro online ha un modo
  rapido per aprire una segnalazione, e chi segnala un errore viene citato
  nel commit che lo corregge.
- Manda i tuoi commenti a *info@paithon.it*. Servono solo a migliorare il
  libro.
:::

---
