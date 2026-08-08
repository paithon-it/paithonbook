# Paithon Book

**Il Libro di Intelligenza Artificiale che spiega due volte.** Machine
Learning, Deep Learning e Reinforcement Learning con Python: in italiano,
gratis, in aggiornamento continuo.

<p>
  <a href="https://book.paithon.it/main/"><img alt="Leggi online" src="https://img.shields.io/badge/Leggi%20online-book.paithon.it-B5532C?style=flat-square"></a>
  <a href="https://jupyterbook.org"><img alt="Jupyter Book" src="https://img.shields.io/badge/costruito%20con-Jupyter%20Book-2D5A5C?style=flat-square"></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/framework-PyTorch-C9A961?style=flat-square"></a>
  <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/deed.it"><img alt="Licenza testi CC BY-NC-ND 4.0" src="https://img.shields.io/badge/testi-CC%20BY--NC--ND%204.0-1A1A1A?style=flat-square"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="Licenza codice Apache 2.0" src="https://img.shields.io/badge/codice-Apache%202.0-1A1A1A?style=flat-square"></a>
</p>

Questa è la versione online del libro **"PAIthon: Machine Learning, Deep
Learning e Reinforcement Learning con Python"** di Francesco Messina: un
percorso completo (dall'algebra lineare ai Transformer) scritto con la
tradizione della divulgazione italiana. Rigoroso nei contenuti, accessibile
nel tono.

👉 **Leggilo qui: [book.paithon.it](https://book.paithon.it/main/)**

Il libro cambia: le sezioni nuove e le correzioni sono registrate, versione per
versione, in
[Aggiornamenti](https://book.paithon.it/main/aggiornamenti.html).

## Due livelli di lettura

Ogni concetto chiave è spiegato due volte, con un sistema di schede affiancate:

- **Elementare**, analogie concrete e quotidiane, zero prerequisiti: deve
  poterlo leggere uno studente di liceo (o chiunque, sotto l'ombrellone).
- **Superiore**, la trattazione formale: definizioni, notazione, formule,
  riferimenti ai paper originali.

I due livelli si alternano nello stesso capitolo: si può leggere tutto all'Elementare, tutto al Superiore, o saltare dall'uno all'altro. Un **interruttore nella barra in alto** (icona a due barrette, accanto a ricerca e chiaro/scuro) imposta il livello di partenza di tutto il libro e lo ricorda; aprire una singola scheda al livello opposto resta un'eccezione locale, che non contagia i paragrafi successivi. Per questo ogni livello è scritto per reggersi da solo.

## Indice

| # | Capitolo | In breve |
|---|----------|----------|
| 1 | **Introduzione** | Che cos'è l'AI, da ELIZA a oggi; il primo notebook eseguibile |
| 2 | **Python** | Le basi del linguaggio, NumPy, pandas e matplotlib |
| 3 | **Richiami di matematica** | Algebra lineare, ottimizzazione, probabilità, teoria dell'informazione, analisi numerica |
| 4 | **Machine Learning** | Apprendimento supervisionato, overfitting, metriche, iperparametri, alberi e metodi ensemble (Random Forest, gradient boosting, XGBoost), SVM e kernel trick, riduzione della dimensionalità e clustering (PCA, t-SNE/UMAP, k-means, DBSCAN), distribution shift, processi gaussiani |
| 5 | **Reti Neurali** | Percettrone, funzioni di attivazione, backpropagation |
| 6 | **PyTorch** | Tensori, autograd, `nn.Module`, training loop, prestazioni e scala |
| 7 | **GPU e calcolo parallelo** | Sotto il cofano: architettura ed esecuzione, gerarchia di memoria, kernel e Triton, GEMM e tensor core, Flash Attention, parallelismo dati/tensor/pipeline e FSDP |
| 8 | **Deep Learning** | Reti convoluzionali, ottimizzazione e regolarizzazione, architetture storiche |
| 9 | **Visione Artificiale** | Classificazione e transfer learning, data augmentation, apprendimento auto-supervisionato (SimCLR, MoCo, BYOL, DINO, MAE), detection e segmentazione, style transfer |
| 10 | **Reinforcement Learning** | MDP, value/policy iteration, Q-learning |
| 11 | **Deep Reinforcement Learning** | DQN e policy gradient, actor-critic (A2C/A3C/PPO), controllo continuo (DDPG/TD3/SAC), RL basato su modello (MuZero, Dreamer), offline RL (Decision Transformer), esplorazione e reward hacking |
| 12 | **Natural Language Processing** | Strumenti classici, embedding, tokenizzatori a sotto-parole (BPE, WordPiece, SentencePiece), classificazione, n-gram, RNN/LSTM, seq2seq, POS/NER, parsing, dialogo |
| 13 | **Transformers** | Self-attention, architettura, Mixture of Experts, GPT/BERT/ViT/CLIP, LLM e scaling, post-training (RLHF/DPO), RAG |
| 14 | **Attenzione lineare** | Dall'attenzione quadratica alla ricorrenza lineare; delta rule e gate; DeltaNet, GLA, RetNet, RWKV, xLSTM |
| 15 | **State Space Model** | Dai sistemi dinamici a S4; Mamba e il selective scan; dualità SSD, Mamba-2 e Mamba-3; ibridi e limiti |
| 16 | **Visione e linguaggio** | Modelli che vedono e parlano: addestramento contrastivo (CLIP, SigLIP), connettori (Flamingo, Q-Former, proiettore), fusione precoce e vocabolario comune, il costo del dettaglio (tessere, documenti), allucinazione visiva e azioni come token |
| 17 | **Agenti e applicazioni LLM** | Tool use e ReAct, RAG avanzato (HyDE, reranking, Self-RAG), context engineering, architetture multi-agente e valutazione |
| 18 | **Prompt, contesto e loop** | Programmare gli LLM a parole su tre livelli concentrici: prompt engineering (ruoli, few-shot, chain-of-thought, self-consistency), context engineering (le quattro mosse, i guasti del contesto, il PRP), loop engineering (il ciclo con validation gate, split maker/checker, governance) |
| 19 | **Sistemi multi-agente** | Quando gli agenti sono molti: il costo del coordinamento (Amdahl, composizione degli errori), topologie (supervisore, lavagna, mercato), protocolli e consenso (Condorcet, dibattito, bizantini), MARL e self-play, sciami (ACO, PSO) e società simulate |
| 20 | **Audio oltre la voce** | Dal suono alle feature (spettrogramma, mel, MFCC), classificazione (AudioSet, AST), rappresentazioni auto-supervisionate (wav2vec 2.0, HuBERT), codec neurali (VQ-VAE, RVQ, EnCodec), generazione (WaveNet, AudioLM, MusicGen) |
| 21 | **Speech Recognition** | Dalla voce al testo: allineamento e CTC, modelli con attenzione, Whisper, sintesi vocale (TTS) |
| 22 | **Generative Adversarial Network** | Il gioco avversario, applicazioni ed evoluzioni |
| 23 | **Modelli di Diffusione** | Da DDPM a Stable Diffusion e ai Diffusion Transformer (DiT, Sora) |
| 24 | **Modelli a energia** | Il paesaggio invece della probabilità: memoria associativa di Hopfield, macchine di Boltzmann e contrastive divergence, i modi di aggirare la funzione di partizione (Langevin, score matching, NCE), la cornice di LeCun, EBM di oggi (IGEBM, JEM) |
| 25 | **World Model** | Simulatori interni: Ha & Schmidhuber, Dreamer, la via di LeCun (JEPA), simulatori video e dibattito |
| 26 | **Graph Neural Network** | Dati relazionali e message passing: GCN, GraphSAGE, GAT; node embedding, applicazioni (molecole, raccomandazione, frodi, traffico) e limiti (oversmoothing) |
| 27 | **Sistemi di Raccomandazione** | Dal Netflix Prize alla matrix factorization e alla raccomandazione neurale |
| 28 | **Serie temporali** | Prevedere dal passato: decomposizione, ARIMA e Holt-Winters, validazione temporale e metriche (MASE, pinball loss), forecasting neurale (TCN, DeepAR, N-BEATS, Transformer, foundation model) |
| 29 | **Physics-Informed Neural Networks** | Le equazioni differenziali dentro la loss; operatori neurali |
| 30 | **MLOps** | Dal notebook alla produzione: versionamento, pipeline di dati, serving e quantizzazione, metriche di servizio (TTFT, TPOT, goodput), monitoraggio del drift, LLMOps e deploy |
| 31 | **Interpretabilità e XAI** | Aprire la scatola nera: modelli trasparenti, importanza delle feature, LIME/SHAP/controfattuali, Grad-CAM e integrated gradients, interpretabilità meccanicistica |
| 32 | **AI responsabile** | Equità e bias, privacy (differential privacy, federated learning) e robustezza agli attacchi avversari, sicurezza degli LLM (prompt injection, jailbreak, red teaming), allineamento (RLHF/DPO) e governance (l'AI Act europeo) |
| 33 | **Conclusioni** | Bilancio e letture per proseguire |

Il framework di riferimento per tutto il codice è **PyTorch** (con NumPy e
scikit-learn dove appropriato). Due capitoli (Introduzione e Attenzione
lineare) hanno un notebook eseguibile, con i pulsanti per Colab e per
l'esecuzione nel browser (Thebe); negli altri il codice è da leggere e
copiare.

## Come è fatto

- **Sorgente**: `book/`, Markdown [MyST](https://myst-parser.readthedocs.io/)
  e notebook Jupyter, indice in `book/_toc.yml`.
- **Figure**: SVG geometriche disegnate a mano in `book/figures/`, su palette fissa (terracotta `#B5532C`, teal `#2D5A5C`, ocra `#C9A961`, warm-black `#1A1A1A`, cream `#F8F5EE`). Niente immagini generate, niente stock.
- **Tema**: la firma visiva arriva dal submodule [`paithon-it/brand`](https://github.com/paithon-it) (`book/_static/brand`), con light/dark mode automatica.
- **Bibliografia**: `book/references.bib`, citata nel testo capitolo per capitolo.
- **Deploy**: a ogni push su `book/` GitHub Actions ricostruisce e pubblica il libro (`.github/workflows/call-deploy-book.yml`).

## Costruire il libro in locale

```bash
# 1. Clona il repo (con il submodule del tema)
git clone --recurse-submodules https://github.com/paithon-it/paithonbook.git
cd paithonbook

# 2. Installa le dipendenze
pip install -r requirements.txt

# 3. Costruisci
jb build book

# 4. Servi in locale
python -m http.server 8080
# poi apri http://localhost:8080/book/_build/html/
```

Se qualcosa non torna: `jb clean book && jb build book` ricostruisce da zero.
Non serve buildare in locale per contribuire: il CI costruisce a ogni push.

## Contribuire

- ✍️ **Seleziona la frase** che non torna, direttamente nel libro: compare un pulsante *Segnala* che apre una issue già compilata con la citazione, la sezione e il file sorgente.
- 🐛 Oppure [apri una issue](https://github.com/paithon-it/paithonbook/issues) a mano. Le regole e la concessione di licenza necessaria sono in [`CONTRIBUTING.md`](CONTRIBUTING.md).
- ✉️ Feedback e proposte: **info@paithon.it**.
- ⭐ Se il libro ti è utile, una stella al repo aiuta il progetto a farsi trovare.
- ❤️ Puoi anche [offrire una pizza](https://buymeacoffee.com/paithon.it) all'autore.

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
`book/_templates/`), gli script e i generatori, è **© paithon.it, tutti i
diritti riservati**.

Materiale di terzi, con le sue licenze: il tema
[`sphinx-book-theme`](https://github.com/executablebooks/sphinx-book-theme)
(BSD-3-Clause), i font Fraunces, Inter e JetBrains Mono
([SIL OFL 1.1](https://openfontlicense.org/), il cui testo viaggia con i file nel submodule), e il submodule
[`paithon-it/brand`](https://github.com/paithon-it/brand), che ha una licenza
propria.

Per contribuire (e per la concessione che serve, dato che la licenza dei testi
vieta le opere derivate), vedi [`CONTRIBUTING.md`](CONTRIBUTING.md).

© 2018–2026 Francesco Messina · [paithon.it](https://www.paithon.it)
