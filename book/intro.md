```{image} _static/logo-light.svg
:class: pt-copertina-logo only-light
:alt: paithon book
```

```{image} _static/logo-dark.svg
:class: pt-copertina-logo only-dark
:alt: paithon book
```

```{raw} html
<p class="pt-claim" aria-hidden="true">Il Libro di Intelligenza Artificiale che spiega
<span class="pt-claim__el">due</span> <span class="pt-claim__sup">volte.</span></p>
```

# Il Libro di Intelligenza Artificiale che spiega due volte.

In alto nella barra trovi questo comando, <span class="pt-livello-inline"
aria-hidden="true"><svg class="pt-livello__icona
pt-livello__icona--elementare" viewBox="0 0 16 16" width="15"
height="15"><rect x="1.6" y="7" width="5.2" height="6.4" rx="1.3"
fill="currentColor"/><rect x="9.2" y="2.6" width="5.2" height="10.8" rx="1.3"
fill="none" stroke="currentColor" stroke-width="1.4"
opacity="0.55"/></svg><svg class="pt-livello__icona
pt-livello__icona--superiore" viewBox="0 0 16 16" width="15" height="15"><rect
x="1.6" y="7" width="5.2" height="6.4" rx="1.3" fill="currentColor"/><rect
x="9.2" y="2.6" width="5.2" height="10.8" rx="1.3"
fill="currentColor"/></svg></span>, per scegliere **come** leggere: l'analogia
quotidiana o la trattazione formale con le formule. La scelta vale per tutto
il libro e si cambia quando vuoi.

```{epigraph}
Gli artisti comprendono che i matematici hanno un modo tutto loro di guardare il mondo, che può far loro percepire le cose in modo diverso.

-- Marcus du Sautoy
```

Addentrarsi le prime volte nel mondo del Machine Learning e dell'Intelligenza Artificiale con Python può sembrare una sfida complessa, ma con la giusta guida può diventare un viaggio affascinante e gratificante. Paithon Book è qui per rendere questo percorso più semplice e accessibile, offrendo risorse chiare e pratiche, tutte in italiano, per chiunque voglia imparare, indipendentemente dal proprio livello di partenza.

Questa versione online del libro **è gratuita** ed **in continuo aggiornamento** per stare al passo con le innovazioni del campo: aggiungiamo regolarmente sezioni, argomenti ed esempi in Python. Il codice del libro usa **PyTorch**, NumPy e scikit-learn. Dove in alto compare **Esegui il codice**, la pagina ha un notebook compagno che si apre su Colab con le sue celle in ordine; due capitoli hanno anche un notebook proprio, con l'icona 🚀. Altrove il codice si legge qui e si copia nel proprio editor.

---

## Il percorso

{{ n_capitoli_lettere|capitalize }} capitoli, dall'alfabeto (Python e matematica) alle frontiere (Transformer, modelli di diffusione, agenti LLM, generazione audio, produzione con l'MLOps). Ogni capitolo si legge da solo, ma l'ordine è pensato per costruire un mattone alla volta.

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
probabilità, teoria dell'informazione, analisi numerica.</span> </a> <a
class="pt-card" href="MachineLearning/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Machine
Learning</span> <span class="pt-card-desc">Imparare dai dati: modelli
supervisionati, overfitting, metriche, iperparametri.</span> </a> <a
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
parallelismo su più GPU.</span> </a> <a class="pt-card"
href="DeepLearning/overview.html"> <span class="pt-card-num"></span> <span
class="pt-card-title">Deep Learning</span> <span class="pt-card-desc">Reti
convoluzionali, ottimizzazione, regolarizzazione e architetture
storiche.</span> </a> <a class="pt-card"
href="VisioneArtificiale/overview.html"> <span class="pt-card-num"></span>
<span class="pt-card-title">Visione Artificiale</span> <span
class="pt-card-desc">Classificare, riconoscere, segmentare: le macchine che
vedono.</span> </a> <a class="pt-card"
href="ReinforcementLearning/overview.html"> <span class="pt-card-num"></span>
<span class="pt-card-title">Reinforcement Learning</span> <span
class="pt-card-desc">Imparare per tentativi: MDP, funzioni di valore e
Q-learning.</span> </a> <a class="pt-card"
href="DeepReinforcementLearning/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Deep Reinforcement
Learning</span> <span class="pt-card-desc">Quando il RL incontra le reti
profonde: DQN e policy gradient.</span> </a> <a class="pt-card"
href="NaturalLanguageProcessing/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Natural Language
Processing</span> <span class="pt-card-desc">Dal testo ai numeri: token,
embedding, reti ricorrenti, traduzione e attenzione.</span> </a> <a
class="pt-card" href="Transformers/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Transformers</span>
<span class="pt-card-desc">L'architettura che ha cambiato tutto:
self-attention, GPT, BERT e multimodalità.</span> </a> <a class="pt-card"
href="AttenzioneLineare/overview.html"> <span class="pt-card-num"></span>
<span class="pt-card-title">Attenzione lineare</span> <span
class="pt-card-desc">Oltre il costo quadratico: dal trucco del kernel alla
delta rule, con RetNet, RWKV e xLSTM.</span> </a> <a class="pt-card"
href="StateSpaceModel/overview.html"> <span class="pt-card-num"></span> <span
class="pt-card-title">State Space Model</span> <span class="pt-card-desc">Da
S4 a Mamba: sequenze a tempo lineare dai sistemi dinamici, fino a Mamba-2 e
Mamba-3.</span> </a> <a class="pt-card" href="Agenti/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Agenti e applicazioni
LLM</span> <span class="pt-card-desc">Quando gli LLM agiscono: tool use e
ReAct, RAG avanzato, context engineering, valutazione.</span> </a> <a
class="pt-card" href="IngegneriaLLM/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Prompt, contesto e
loop</span> <span class="pt-card-desc">Programmare gli LLM a parole su tre
livelli concentrici: prompt engineering, context engineering e loop
engineering.</span> </a> <a class="pt-card" href="Audio/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Audio oltre la
voce</span> <span class="pt-card-desc">Dallo spettrogramma alla generazione:
AudioSet, wav2vec, codec neurali, MusicGen.</span> </a> <a class="pt-card"
href="SpeechRecognition/overview.html"> <span class="pt-card-num"></span>
<span class="pt-card-title">Speech Recognition</span> <span
class="pt-card-desc">Dalla voce al testo e ritorno: allineamento, CTC,
Whisper, sintesi vocale.</span> </a> <a class="pt-card"
href="GAN/overview.html"> <span class="pt-card-num"></span> <span
class="pt-card-title">GAN</span> <span class="pt-card-desc">Due reti che si
sfidano: il gioco avversario che genera immagini.</span> </a> <a
class="pt-card" href="ModelliDiffusione/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Modelli di
Diffusione</span> <span class="pt-card-desc">Rumore e ritorno: da DDPM a
Stable Diffusion e ai Diffusion Transformer.</span> </a> <a class="pt-card"
href="ModelliEnergia/overview.html"> <span class="pt-card-num"></span> <span
class="pt-card-title">Modelli a energia</span> <span class="pt-card-desc">Il
paesaggio invece della probabilit&agrave;: da Hopfield e Boltzmann allo score
matching, la lingua in cui sono scritte diffusione e JEPA.</span> </a> <a
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
class="pt-card-title">MLOps</span> <span class="pt-card-desc">Il capitolo
conclusivo: dal notebook alla produzione, versioning, serving, monitoraggio
del drift, LLMOps e deploy.</span> </a> <a class="pt-card"
href="Interpretabilita/overview.html"> <span class="pt-card-num"></span> <span
class="pt-card-title">Interpretabilità e XAI</span> <span
class="pt-card-desc">Aprire la scatola nera: modelli trasparenti e importanza
delle feature, spiegazioni locali (LIME, SHAP, controfattuali), attribuzione
(Grad-CAM, integrated gradients) e interpretabilità meccanicistica.</span>
</a> <a class="pt-card" href="AIResponsabile/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">AI responsabile</span>
<span class="pt-card-desc">Equità e bias algoritmico, privacy (differential
privacy, federated learning) e robustezza agli attacchi avversari,
allineamento (RLHF, DPO) e governance (l'AI Act europeo).</span> </a> <a
class="pt-card" href="Conclusioni/overview.html"> <span
class="pt-card-num"></span> <span class="pt-card-title">Conclusioni</span>
<span class="pt-card-desc">Bilancio del viaggio e letture per
proseguire.</span> </a> </div>

---

## Livelli di Complessità

Il libro presenta due livelli di complessità: **Elementare** e **Superiore**:

- *Elementare*: spiega i concetti nel modo più semplice e accessibile possibile, con analogie concrete e quotidiane. È ideale per chi è alle prime armi negli argomenti trattati nel libro.

- *Superiore*: approfondisce i concetti in modo più formale e dettagliato. Si presume che il lettore abbia una comprensione di base consolidata e sia pronto per esplorare aspetti più complessi.

Alternare la lettura tra questi due livelli può favorire una comprensione più profonda: si può leggere un capitolo tutto all'Elementare, tutto al Superiore, o costruire gradualmente la propria competenza passando dall'uno all'altro.

**Come si sceglie.** Ogni concetto ha due schede, e si apre quella che si
vuole. Ma se sai già da che parte vuoi stare, in alto nella barra (accanto
alla ricerca e al chiaro/scuro), c'è un'icona a **due barrette**: la prima
piena e la seconda vuota quando sei all'Elementare, entrambe piene quando
passi al Superiore. Premendola imposti il livello di **tutto il libro** in un
colpo, e la scelta resta salvata nel browser: la ritrovi al ritorno, capitolo
dopo capitolo.

Le due cose convivono. L'interruttore decide da dove *parte* ogni pagina; aprire una singola scheda al livello opposto resta un'eccezione locale, che non contagia i paragrafi dopo. Se leggi all'Elementare e vuoi sbirciare una formula, sbirciala: due paragrafi più sotto sarai di nuovo all'Elementare.

Ad esempio, ecco la funzione ReLU nelle due modalità:

````{tab} Elementare

Immagina di avere un controllo, denominato $f(x)$, per il trasferimento di una somma di denaro $x$ in Euro (€). Se il valore della transazione è positivo, $x > 0$, il filtro $f(x)$ permette il passaggio esattamente di $x$ €. Se la transazione ha un valore negativo o zero, il controllo blocca la transazione e restituisce $0$, evitando così prelievi non autorizzati dal conto del destinatario. In altre parole, la funzione ReLU permette il passaggio di una quantità uguale a quella in ingresso solo se quest'ultima è positiva, altrimenti blocca completamente il passaggio (restituendo zero).

Esempi:

$
\begin{align*}
x &= 10, & f(x) &= 10 \\
x &= -3, & f(x) &= 0 \\
\end{align*}
$

````

````{tab} Superiore

La funzione di attivazione ReLU (Rectified Linear Unit) è definita come:

$$
f(x) = x^+ = \max(0, x)
$$


Questa funzione può anche essere definita "a tratti" (*piecewise*) nel seguente modo:

$$
f(x) = \begin{cases}
    x & \text{se } x > 0, \\
    0 & \text{altrimenti}.
\end{cases}
$$

Questa funzione prende un input $x$ e restituisce $x$ se $x$ è positivo; altrimenti, restituisce zero. \
La ReLU è ampiamente utilizzata nelle reti neurali perché introduce una non linearità essenziale con una derivata costante per i valori positivi di $x$, il che rende più efficiente il calcolo della *backpropagation* durante la fase di addestramento.

````

---

## Supporta il progetto

Ogni progetto ha bisogno di supporto per crescere e migliorare. I progetti open source come questo si basano sul sostegno e l'entusiasmo della comunità. Se questo libro ti è stato utile o credi nel suo valore, ecco come puoi mostrarmi il tuo apprezzamento:

- Metti una ⭐ e condividi il progetto GitHub [Paithon Book](https://github.com/paithon-it/paithonbook)
- Supporta il progetto per contribuire al suo sviluppo ❤️ <span style="display: inline-block; vertical-align: middle;">
    <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="paithon.it" data-color="#B5532C" data-emoji="🔋"  data-font="Cookie" data-text="Ricarica la mia energia" data-outline-color="#1A1A1A" data-font-color="#ffffff" data-coffee-color="#C9A961" ></script>
</span>

- Aiutaci a scoprire errori e migliorare il progetto aprendo un Issue tramite le pagine del sito.
- Invia i tuoi feedback ✉ a *info@paithon.it*. Saranno utilizzati esclusivamente per migliorare e arricchire il libro.

---
