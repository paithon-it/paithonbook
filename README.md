# Paithon Book

<a href="https://book.paithon.it/main/"><img src="book/_static/social/og-book.png" width="840" alt="Paithon Book: Il Libro di Intelligenza Artificiale che spiega due volte."></a>

**Il Libro di Intelligenza Artificiale che spiega due volte.** Machine
Learning, Deep Learning e Reinforcement Learning con Python: in italiano,
gratis, in aggiornamento continuo.

<p>
  <a href="https://book.paithon.it/main/"><img alt="Leggi online" src="https://img.shields.io/badge/Leggi%20online-book.paithon.it-B5532C?style=flat-square"></a>
  <a href="https://github.com/paithon-it/paithonbook/releases/latest/download/paithon-book.pdf"><img alt="Scarica il PDF" src="https://img.shields.io/github/v/release/paithon-it/paithonbook?style=flat-square&label=PDF&color=B5532C"></a>
  <a href="https://doi.org/10.5281/zenodo.21947219"><img alt="DOI 10.5281/zenodo.21947219" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21947219-2D5A5C?style=flat-square"></a>
  <a href="https://jupyterbook.org"><img alt="Jupyter Book" src="https://img.shields.io/badge/costruito%20con-Jupyter%20Book-2D5A5C?style=flat-square"></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/framework-PyTorch-C9A961?style=flat-square"></a>
  <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/deed.it"><img alt="Licenza testi CC BY-NC-ND 4.0" src="https://img.shields.io/badge/testi-CC%20BY--NC--ND%204.0-1A1A1A?style=flat-square"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="Licenza codice Apache 2.0" src="https://img.shields.io/badge/codice-Apache%202.0-1A1A1A?style=flat-square"></a>
</p>

Questa è la versione online di **Paithon Book**, di Francesco Messina: un
percorso completo (dall'algebra lineare ai Transformer) scritto con la
tradizione della divulgazione italiana. Rigoroso nei contenuti, accessibile
nel tono.

👉 **Leggilo qui: [book.paithon.it](https://book.paithon.it/main/)**

📄 **Oppure scaricalo**: [il libro intero in
PDF](https://github.com/paithon-it/paithonbook/releases/latest/download/paithon-book.pdf),
impaginato per la lettura e per la stampa; dalla griglia dei capitoli sulla
home si scarica anche il **singolo capitolo**, per chi ne vuole uno solo.
Dove il libro online anima una figura, la stampa dà tre fermi immagine e
l'indirizzo per vederla muoversi.

Il libro cambia: le sezioni nuove e le correzioni sono registrate, versione per
versione, in
[Aggiornamenti](https://book.paithon.it/main/aggiornamenti.html).

## Due livelli di lettura

Ogni concetto chiave è spiegato due volte, con un sistema di schede affiancate:

- **Elementare**, analogie concrete e quotidiane, zero prerequisiti: deve
  poterlo leggere uno studente di liceo (o chiunque, sotto l'ombrellone).
- **Superiore**, la trattazione formale: definizioni, notazione, formule,
  riferimenti ai paper originali.

I due livelli si alternano nello stesso capitolo: si può leggere tutto
all'Elementare, tutto al Superiore, o saltare dall'uno all'altro. Un
**interruttore nella barra in alto** (icona a due barrette, accanto a ricerca e
chiaro/scuro) imposta il livello di partenza di tutto il libro e lo ricorda;
aprire una singola scheda al livello opposto resta un'eccezione locale, che non
contagia i paragrafi successivi. Per questo ogni livello è scritto per reggersi
da solo.

Non sono due gradi di dettaglio ma due presentazioni della stessa struttura: a
ogni passaggio della scheda formale corrisponde un gesto nell'analogia, e chi
si costruisce l'intuizione sull'Elementare non deve disimparare niente aprendo
la Superiore.

## Indice

Il libro si apre con una **Prefazione**; poi i capitoli, raggruppati in parti
che spezzano l'indice per aree.

| # | Capitolo | In breve |
|---|----------|----------|
| 1 | **Introduzione** | Che cos'è l'AI, da ELIZA a oggi; il primo notebook eseguibile |
| 2 | **Python** | Le basi del linguaggio, NumPy, pandas e matplotlib |
| 3 | **Richiami di matematica** | Algebra lineare, sistemi lineari, ortogonalità e proiezioni, determinante, ottimizzazione, probabilità, disuguaglianze di concentrazione, catene di Markov, teoria dell'informazione, analisi numerica, e la matematica che sta dentro un modello linguistico |
| 4 | **Machine Learning** | Apprendimento supervisionato, overfitting, metriche, iperparametri, spline e modelli additivi, alberi e metodi ensemble (Random Forest, gradient boosting, XGBoost), bootstrap, SVM e kernel trick, classificatori generativi (naive Bayes, analisi discriminante), riduzione della dimensionalità e clustering (PCA, t-SNE/UMAP, k-means, DBSCAN) con le metriche per valutarlo, distribution shift, processi gaussiani |
| 5 | **Reti Neurali** | Percettrone, funzioni di attivazione, backpropagation |
| 6 | **PyTorch** | Tensori, autograd, `nn.Module`, training loop, `Dataset` e `DataLoader`, i tre errori più comuni, dal notebook agli script, replicare un paper, prestazioni e scala |
| 7 | **GPU e calcolo parallelo** | Sotto il cofano: architettura ed esecuzione, gerarchia di memoria, kernel e Triton, GEMM e tensor core, Flash Attention, parallelismo dati/tensor/pipeline e FSDP |
| 8 | **Efficienza** | Il modello che si addestra non è quello che si usa: quantizzazione (scala e granularità, componenti anomale degli LLM, PTQ e QAT, GPTQ e AWQ), potatura e sparsità strutturata, ipotesi del biglietto della lotteria, distillazione e bersagli morbidi, e perché starci in memoria non sia rispondere in fretta |
| 9 | **Deep Learning** | Reti convoluzionali, ottimizzazione e regolarizzazione, architetture storiche, una rete sola per molti compiti |
| 10 | **Visione Artificiale** | Classificazione e transfer learning, data augmentation, apprendimento auto-supervisionato (SimCLR, MoCo, BYOL, DINO, MAE), detection e segmentazione, geometria e profondità, rendering neurale (NeRF, splatting), style transfer |
| 11 | **Ricerca e pianificazione** | Guardare avanti senza imparare niente: spazio degli stati e albero di ricerca, ricerca cieca e in profondità iterativa, euristiche ammissibili e consistenti, A*, minimax e potatura alfa-beta, funzione di valutazione ed effetto orizzonte, e le tre ipotesi che cadendo aprono il rinforzo |
| 12 | **Reinforcement Learning** | Imparare per tentativi: bandit a più braccia, MDP e funzioni valore, metodi Monte Carlo, differenze temporali e Q-learning |
| 13 | **Deep Reinforcement Learning** | DQN e policy gradient, actor-critic (A2C/A3C/PPO), MCTS e AlphaGo, controllo continuo (DDPG/TD3/SAC), RL basato su modello (MuZero, Dreamer), imitazione e clonazione comportamentale, offline RL (Decision Transformer), esplorazione e reward hacking |
| 14 | **Natural Language Processing** | Strumenti classici, embedding, tokenizzatori a sotto-parole (BPE, WordPiece, SentencePiece, byte), classificazione, n-gram, RNN/LSTM, seq2seq, POS/NER, parsing, dialogo |
| 15 | **Transformers** | Self-attention, architettura, Mixture of Experts, GPT/BERT/ViT/CLIP, LLM e scaling, post-training (RLHF/DPO), RAG, multimodalità e multilingua |
| 16 | **Attenzione lineare** | Dall'attenzione quadratica alla ricorrenza lineare; delta rule e gate; DeltaNet, GLA, RetNet, RWKV, xLSTM |
| 17 | **State Space Model** | Dai sistemi dinamici a S4; Mamba e il selective scan; dualità SSD, Mamba-2 e Mamba-3; ibridi e limiti |
| 18 | **Visione e linguaggio** | Modelli che vedono e parlano: addestramento contrastivo (CLIP, SigLIP), connettori (Flamingo, Q-Former, proiettore), fusione precoce e vocabolario comune, il costo del dettaglio (tessere, documenti), allucinazione visiva e azioni come token |
| 19 | **Agenti e applicazioni LLM** | Tool use e ReAct, RAG avanzato (HyDE, reranking, Self-RAG), context engineering, architetture multi-agente e valutazione |
| 20 | **Prompt, contesto e loop** | Programmare gli LLM a parole su tre livelli concentrici: prompt engineering (ruoli, few-shot, chain-of-thought, self-consistency), context engineering (le quattro mosse, i guasti del contesto, il PRP), loop engineering (il ciclo con validation gate, split maker/checker, governance) |
| 21 | **Sistemi multi-agente** | Quando gli agenti sono molti: il costo del coordinamento (Amdahl, composizione degli errori), topologie (supervisore, lavagna, mercato), protocolli e consenso (Condorcet, dibattito, bizantini), MARL e self-play, sciami (ACO, PSO) e società simulate |
| 22 | **Audio oltre la voce** | Dal suono alle feature (spettrogramma, mel, MFCC), classificazione (AudioSet, AST), rappresentazioni auto-supervisionate (wav2vec 2.0, HuBERT), codec neurali (VQ-VAE, RVQ, EnCodec), generazione (WaveNet, AudioLM, MusicGen) |
| 23 | **Speech Recognition** | Dalla voce al testo: allineamento e CTC, modelli con attenzione, Whisper, sintesi vocale (TTS) |
| 24 | **Modelli latenti** | Spiegare i dati con una causa che non si vede: autoencoder e il suo limite, verosimiglianza intrattabile, ELBO e trucco della riparametrizzazione, beta-VAE e latente discreto, e i quattro punti del libro in cui erano già al lavoro |
| 25 | **Generative Adversarial Network** | Il gioco avversario, applicazioni ed evoluzioni |
| 26 | **Modelli di Diffusione** | Da DDPM al limite continuo (SDE, ODE del flusso di probabilità), flow matching, Stable Diffusion, Diffusion Transformer (DiT, Sora), campionatori veloci, generatori a un passo, guida e allineamento, diffusione sul testo |
| 27 | **Verosimiglianza esatta** | I modelli che sanno dire quanto è probabile un dato: autoregressivi sulle immagini (PixelCNN, maschere e punto cieco), flussi normalizzanti (cambio di variabile, RealNVP, GLOW), a che serve la verosimiglianza e dove sbaglia |
| 28 | **Modelli a energia** | Il paesaggio invece della probabilità: memoria associativa di Hopfield, macchine di Boltzmann e contrastive divergence, i modi di aggirare la funzione di partizione (Langevin, score matching, NCE), la cornice di LeCun, EBM di oggi (IGEBM, JEM) |
| 29 | **Auto-supervisione** | Il segnale fabbricato dai dati: la banda informativa e la torta di LeCun, le quattro famiglie (contrastiva, distillazione, riduzione di ridondanza con Barlow Twins e VICReg, mascheramento), collasso e misura, il dibattito sul rinforzo |
| 30 | **World Model** | Simulatori interni: Ha & Schmidhuber, Dreamer, la via di LeCun (JEPA), l'inferenza attiva, simulatori video e dibattito |
| 31 | **Graph Neural Network** | Dati relazionali e message passing: GCN, GraphSAGE, GAT; node embedding e knowledge graph, applicazioni (molecole, raccomandazione, frodi, traffico) e limiti (oversmoothing) |
| 32 | **Sistemi di Raccomandazione** | Dal Netflix Prize alla matrix factorization e alla raccomandazione neurale |
| 33 | **Serie temporali** | Prevedere dal passato: decomposizione, ARIMA e Holt-Winters, validazione temporale e metriche (MASE, pinball loss), forecasting neurale (TCN, DeepAR, N-BEATS, Transformer, foundation model) |
| 34 | **Physics-Informed Neural Networks** | Le equazioni differenziali dentro la loss; operatori neurali |
| 35 | **MLOps** | Dal notebook alla produzione: versionamento, pipeline di dati, serving e quantizzazione, metriche di servizio (TTFT, TPOT, goodput), monitoraggio del drift, LLMOps e deploy, il conto in energia |
| 36 | **Interpretabilità e XAI** | Aprire la scatola nera: modelli trasparenti, importanza delle feature, LIME/SHAP/controfattuali, Grad-CAM e integrated gradients, interpretabilità meccanicistica |
| 37 | **AI responsabile** | Equità e bias, privacy (differential privacy, federated learning) e robustezza agli attacchi avversari, sicurezza degli LLM (prompt injection, jailbreak, red teaming), allineamento (RLHF/DPO) e governance (l'AI Act europeo) |
| 38 | **Conclusioni** | Bilancio e letture per proseguire |

## Il codice si esegue

Il framework di riferimento per tutto il codice è **PyTorch** (con NumPy e
scikit-learn dove appropriato), e per provarlo non serve installare niente.

I capitoli con del codice hanno il loro **notebook compagno** in `notebooks/`,
e lo si vede dalla pagina: dove in cima compare *Esegui il codice*, quel
collegamento apre su Google Colab una copia in cui i blocchi, invece di stare
lì solo da leggere, si eseguono uno dopo l'altro. Due pagine sono esse stesse
un notebook (il primo programma dell'Introduzione e la verifica numerica
dell'Attenzione lineare) e si riconoscono dal razzo 🚀 fra i comandi in cima
alla pagina: lì il codice gira restando nel libro, senza passare da Colab.

I notebook non si scrivono a mano, li ricava dalle pagine
`scripts/genera-notebook.py`; e quando una pagina o un notebook cambiano, la CI
li riesegue e confronta quello che stampano con i numeri scritti nel libro
(`.github/workflows/verifica-notebook.yml`). Un «esegui» che consegna un errore
è peggio che non offrirlo.

## Come è fatto

- **Sorgente**: `book/`, Markdown [MyST](https://myst-parser.readthedocs.io/)
  e notebook Jupyter, indice in `book/_toc.yml`, che raggruppa i capitoli in
  parti.
- **Figure**: SVG geometriche disegnate a mano in `book/figures/`, su palette
  fissa (terracotta `#B5532C`, teal `#2D5A5C`, ocra `#C9A961`, warm-black
  `#1A1A1A`, cream `#F8F5EE`). Niente immagini generate, niente stock. Sono
  numerate per capitolo (`Fig. 3.2`) e il numero non si scrive a mano: lo
  calcola `book/_ext/pt_figure.py` dall'indice. Toccandole si aprono a schermo
  intero, perché in una colonna da telefono le etichette dentro un diagramma
  diventano illeggibili.
- **Animazioni**: dove il tempo è il contenuto (una discesa che converge, una
  finestra che scorre, un token generato dopo l'altro) la figura si muove. I
  generatori stanno in `animazioni/`: quelli sotto `animazioni/svg/` calcolano
  in Python e scrivono `@keyframes` CSS dentro l'SVG, gli altri rendono con
  Manim. Un solo file per figura, chiaro e scuro compresi; in stampa ogni clip
  diventa tre fermi immagine.
- **Tema**: la firma visiva arriva dal submodule
  [`paithon-it/brand`](https://github.com/paithon-it) (`book/_static/brand`),
  con light/dark mode automatica.
- **Bibliografia**: `book/references.bib`, citata nel testo capitolo per
  capitolo.
- **PDF**: `scripts/genera-pdf.py` compone il libro con LuaLaTeX, converte le
  figure e sostituisce le animazioni con i fermi immagine; `--capitoli` ne
  ritaglia uno per capitolo. Il risultato è quello allegato a ogni
  [release](https://github.com/paithon-it/paithonbook/releases).
- **Deploy**: a ogni push su `book/` GitHub Actions ricostruisce e pubblica il
  libro (`.github/workflows/call-deploy-book.yml`).

## Costruire il libro in locale

```bash
# 1. Clona il repo (con il submodule del tema, o resti senza firma visiva)
git clone --recurse-submodules https://github.com/paithon-it/paithonbook.git
cd paithonbook

# 2. Installa le dipendenze
pip install -r requirements.txt

# 3. Costruisci
jb build book

# 4. Guarda il risultato
python -m http.server 8080 --directory book/_build/html
# poi apri http://localhost:8080/
```

Oltre duecento pagine in un paio di minuti. I notebook non vengono rieseguiti a
ogni build (`execute_notebooks: auto` esegue solo quelli che non hanno gli
output salvati), quindi il tempo è quasi tutto Sphinx.

Se una modifica non compare, la prima sospettata è la cache: `jb clean book`
ricostruisce da zero, e `jb build book --all` rifà tutte le pagine senza
buttare via l'ambiente.

**Installate dal `requirements.txt`, non a mano.** `pip install jupyter-book`
oggi vi darebbe la **2**, che non è la versione successiva di questo strumento
ma un altro programma (si chiama mystmd, non usa Sphinx e con il
`_config.yml` di questo libro non c'entra niente). Il `requirements.txt`
installa `teachbooks`, che vincola la serie giusta.

Il PDF è un altro mestiere e ha altre dipendenze
(`pip install -r requirements-strumenti.txt`, più TeX Live e il Chromium di
Playwright): `python3 scripts/genera-pdf.py` lo costruisce in
`book/_build/stampa/`.

### Quando conviene buildare

Per correggere un refuso non serve: la CI costruisce a ogni push. Serve invece
quando la domanda riguarda qualcosa che il sorgente non dice, e sono più cose
di quante sembri:

- **che numero prende una figura.** Lo mette un'estensione che riscrive i
  contatori di Sphinx: nel sorgente il numero non c'è, e finché non si builda
  non esiste;
- **se un `{numref}` o un `{cite}` risolve.** Un riferimento a un'ancora
  sbagliata non è un errore di sintassi: la pagina si costruisce lo stesso, e
  una citazione che non risolve non lascia nemmeno un link morto, sparisce. Per
  stanarli, `jb build book -n --all` segnala ogni riferimento che non trova
  destinazione, e `python3 scripts/coerenza.py --solo cite,numref,ref` fa lo
  stesso in un attimo, senza buildare;
- **come viene una pagina stampata**, che è un rendering diverso da quello a
  schermo e con regole proprie: in stampa i link non esistono, la
  composizione tipografica sì, e certi difetti vivono in un formato solo.

## Contribuire

- ✍️ **Seleziona la frase** che non torna, direttamente nel libro: compare un
  pulsante *Segnala* che apre una issue già compilata con la citazione, la
  sezione e il file sorgente.
- 🐛 Oppure [apri una issue](https://github.com/paithon-it/paithonbook/issues) a
  mano. Le regole e la concessione di licenza necessaria sono in
  [`CONTRIBUTING.md`](CONTRIBUTING.md).
- ✉️ Feedback e proposte: **info@paithon.it**.
- ⭐ Se il libro ti è utile, una stella al repo aiuta il progetto a farsi
  trovare.
- ❤️ Puoi anche [offrire una pizza](https://buymeacoffee.com/paithon.it)
  all'autore.

## Come citarlo

Il libro è depositato su **Zenodo** e ha un DOI:
**[10.5281/zenodo.21947219](https://doi.org/10.5281/zenodo.21947219)**. Quello
è il DOI *di tutte le versioni*: apre sempre l'ultima depositata, e da lì si
legge il PDF nel browser senza scaricarlo. Ogni versione ne ha poi uno suo,
scritto sulla sua scheda, e va usato quando si cita un passaggio che potrebbe
cambiare, perché questo libro cambia davvero.

La citazione pronta la dà il pulsante **«Cite this repository»** nella colonna
di destra, che GitHub costruisce da [`CITATION.cff`](CITATION.cff): quel file
non si scrive a mano, lo genera `python3 scripts/genera-citazione.py` dal
registro delle versioni, così il numero non può divergere da quello del libro.

```bibtex
@book{paithonbook,
  author    = {Messina, Francesco},
  title     = {Paithon Book. Il Libro di Intelligenza Artificiale che spiega due volte},
  publisher = {paithon.it},
  year      = {2026},
  doi       = {10.5281/zenodo.21947219},
  url       = {https://book.paithon.it/main/}
}
```

## Licenza

Il progetto adotta un **doppio regime di licenza**:

- **Testi e figure**, [CC BY-NC-ND
  4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.it) (file
  [`LICENSE`](LICENSE)): puoi condividerli citando la fonte, senza usi
  commerciali né opere derivate. Vale per i `.md` e i `.ipynb` sotto `book/` e
  per le SVG in `book/figures/`.
- **Codice degli esempi**, [Apache 2.0](LICENSE-CODE) (file
  [`LICENSE-CODE`](LICENSE-CODE)): puoi usarlo, modificarlo e integrarlo nei
  tuoi progetti, anche commerciali, mantenendo le note di copyright e di
  licenza. Include una concessione esplicita di brevetto. Sono i frammenti di
  codice dentro i capitoli e i notebook.

Tutto il resto del repository, la meccanica del libro (`book/_static/`,
`book/_templates/`), gli script e i generatori (`scripts/`, `animazioni/`), è
**© Francesco Messina (paithon.it), tutti i diritti riservati**.

Materiale di terzi, con le sue licenze: il tema
[`sphinx-book-theme`](https://github.com/executablebooks/sphinx-book-theme)
(BSD-3-Clause), i font Fraunces, Inter e JetBrains Mono
([SIL OFL 1.1](https://openfontlicense.org/), il cui testo viaggia con i file nel submodule), e il submodule
[`paithon-it/brand`](https://github.com/paithon-it/brand), che ha una licenza
propria.

Per contribuire (e per la concessione che serve, dato che la licenza dei testi
vieta le opere derivate), vedi [`CONTRIBUTING.md`](CONTRIBUTING.md).

© 2019–2026 Francesco Messina · [paithon.it](https://www.paithon.it)
