# animazioni/

Sorgenti delle figure animate del libro. Due strade, che non competono:

| | `animazioni/*.py` (Manim) | `animazioni/svg/*.py` (SVG animato) |
|---|---|---|
| uscita | `book/figures/*.gif` raster | `book/figures/*.svg`, testo |
| peso | ~570 KB a clip | **~6 KB** a figura |
| in git | blob binario nuovo a ogni render | diffabile |
| stampa | tre fermi immagine (`fermi.py`) | idem, e da fermo è già lo stato finale |
| serve per | LaTeX denso, curve, coreografie | geometria, barre, reti, griglie |
| dipendenze | Docker + Manim + LaTeX | nessuna (`cairosvg` solo per il provino) |

**Quale usare.** Di default l'SVG: costa meno, si legge in git, e in stampa si
risolve da solo. Si passa a Manim quando serve davvero ciò che solo lui dà —
formule LaTeX composte, grafici di funzione con tangenti, coreografie con molti
oggetti che si inseguono. Non è una gerarchia di qualità: è che *rendere
fotogrammi* e *descrivere keyframe* risolvono problemi diversi.

## La strada Manim

Un file `.py` per animazione, che importa il tema del design system
(`from paithon_anim import *`) ed eredita da `ScenaPaithon` implementando
`costruisci()`.

Lo stile (palette, font, ritmo, firma) vive nel brand, in
`book/_static/brand/motion/` — la stessa cartella che usa il sito, così le clip
del libro e quelle degli articoli sono coerenti. Si modifica nel repo `brand`.

Non si renderizza a mano: se ne occupa il driver della skill, che gira Manim
nell'immagine Docker ufficiale e converte in GIF con l'`ffmpeg` dell'host.

```bash
# nuova animazione da template
python3 .claude/skills/anima-manim/driver.py nuova "regola della catena"

# render (l'output finisce in book/figures/, come le SVG)
python3 .claude/skills/anima-manim/driver.py render animazioni/regola-della-catena.py
```

Istruzioni complete, regole di composizione e gotchas:
[`.claude/skills/anima-manim/SKILL.md`](../.claude/skills/anima-manim/SKILL.md).
Esempi di riferimento: `.claude/skills/anima-manim/esempi/`.

## La strada SVG

Come per Manim, il motore vive nel brand, in `book/_static/brand/motion/`
(`paithon_svg.py`, accanto a `paithon_anim.py`): è la stessa cartella che monta
il sito, e si modifica nel repo `brand`. Qui restano i generatori e `genera.py`,
che tiene l'unica cosa che sa del libro, cioè che le figure vanno in
`book/figures/`.

Un file `.py` per figura in `animazioni/svg/`, che espone `NOME`, `TITOLO` e
`costruisci() -> Figura`. La matematica resta in Python — l'algoritmo gira
davvero — ma in uscita non ci sono pixel: qualche `@keyframes` CSS su pochi
elementi, e i fotogrammi intermedi li calcola il browser.

```bash
python3 animazioni/svg/genera.py                    # tutte
python3 animazioni/svg/genera.py percettrone-impara # una sola
```

**La regola che tiene in piedi tutto**: il disegno *fermo* è lo stato
**finale**, scritto con coordinate e attributi veri, senza nessun `transform`;
l'animazione parte dalla trasformazione inversa e finisce sull'identità. Così
il riposo non dipende dal CSS — regge in stampa, nei PDF, con
`prefers-reduced-motion` e in qualunque rasterizzatore — e non c'è modo che i
due stati divergano. È l'equivalente di «l'ultimo fotogramma deve reggere da
solo» per Manim, ma verificato dalla struttura invece che a occhio.

`genera.py` scrive anche un **provino** PNG in `~/.cache/paithon-svg/`:
è lo stato di riposo rasterizzato, cioè esattamente ciò che vedrà la stampa.
Va aperto con `Read` prima di pubblicare — ha già intercettato un errore di
segno che avrebbe mandato in stampa una retta con la pendenza sbagliata.

`scrivi()` rifiuta il file se contiene colori fuori palette, uno `<script>` o
XML malformato.

### Quando la figura disegna un esperimento, e non un calcolo

Un generatore che **calcola** (una funzione, una geometria, un algoritmo su
dati costruiti) dà gli stessi byte su qualunque macchina, e `--verifica` fa il
suo mestiere. Un generatore che **addestra una rete** no: dove c'è un
`kthvalue`, un `argmax` o un ordinamento fra valori appaiati, l'ultimo bit
decide quale peso sopravvive, e due CPU con ordini di riduzione BLAS diversi
(stesso seme, stesso torch) prendono da lì in poi due strade diverse. L'SVG
cambia, `--verifica` dice «da rigenerare» su ogni macchina che non sia quella
che ha committato, e un cancello che non può tornare verde viene aggirato.

La regola per queste figure: **il dato si misura una volta e si committa, il
disegno è una funzione pura del dato.**

```bash
python3 animazioni/svg/genera.py --misura NOME  # riesegue e riscrive il dato
python3 animazioni/svg/genera.py NOME           # ridisegna, e guarda il provino
```

Il generatore espone in più una `misura()`, che esegue l'esperimento, lo
collauda e scrive `animazioni/dati/<nome>.json` con dentro il seme, la
configurazione e la data; `costruisci()` legge quel json e non ha bisogno
nemmeno di importare `torch`. Le asserzioni che difendono la didascalia girano
in tutti e due i posti: sul dato committato a ogni disegno, perché un dato
committato è un dato che nessuno riapre più, e sull'esperimento quando lo si
rimisura. Il json porta con sé la configurazione con cui è stato prodotto e il
caricamento la riconfronta con quella del generatore, così chi ritocca un
parametro e non rimisura trova un rifiuto invece di una figura che disegna una
rete che non esiste più. E senza il json il generatore si ferma e dice come
produrlo: una figura che si inventa i propri numeri è peggio di una figura che
manca.

**E l'arrotondamento del dato si conta, non si sceglie a occhio.** Nel json i
numeri vanno arrotondati, o il file si riempie di cifre che non significano
niente e in un diff non si legge più. Ma tagliare corto butta via misura: un
float32 porta circa nove cifre decimali significative, e su un valore da
qualche centesimo sei decimali ne tengono cinque. Non è una questione di
principio: nel ciclo di addestramento uno scostamento passava da -0,060395777
a -0,060396 e la sua ordinata da 336,4499894 a 336,4500138, cioè scavalcava
l'arrotondamento a `.1f` e il punto disegnato si spostava di un decimo di
pixel. La prova che decide è una sola, e va fatta la prima volta: si rigenera
l'SVG e si guarda se cambia rispetto a quello committato.

Le due strade non si scelgono a gusto. Se l'algoritmo è deterministico
dappertutto, calcolare nel generatore è più semplice e si fa così; se dentro
c'è un addestramento, il dato si separa. E non è più una cosa da ricordarsi:
`genera.py --verifica` guarda anche i sorgenti, e un generatore che importa
torch senza esporre `misura()`, o che lo importa al livello del modulo (cioè
lo fa pagare anche a chi disegna), lo rifiuta. Chi vuole sapere quante figure
sono in questa classe le conta invece di fidarsi di un elenco scritto a mano
(oggi sono tre, e l'ancoraggio a inizio riga non è pignoleria: senza, il
conto tira dentro anche `genera.py`, che di torch parla soltanto):

```bash
grep -lE '^[[:space:]]*(import torch|from torch)' animazioni/svg/*.py
```

## Online la clip, a stampa tre fermi immagine

Nel PDF il movimento non c'è, e una figura ferma sola perde proprio la cosa
per cui l'animazione era stata fatta. Al suo posto vanno **tre fotogrammi
affiancati** più l'indirizzo della pagina che la muove: non sono l'animazione,
ma dicono che c'era un prima e un dopo.

```bash
python3 animazioni/fermi.py             # i tre PNG per tutte e 35
python3 animazioni/fermi.py --provino   # i fogli a contatto, DA GUARDARE
python3 animazioni/fermi.py --verifica  # sono allineati alle animazioni?
                                        # (per impronta, non per data:
                                        #  le date non sopravvivono al
                                        #  checkout della CI)
```

Escono in `book/figures/fermi/`, sono **tracciati**, e valgono per tutte e due
le strade: le GIF di Manim (fotogrammi con PIL) e gli SVG animati (Chromium,
che ferma l'animazione CSS su un istante preciso). Gli istanti di default sono
10%, 50% e 90% del ciclo; dove non rendono giustizia si scrive in
`fermi.toml`, dopo aver guardato il provino.

Il vecchio `--striscia` non c'è più: faceva metà di questo lavoro, per le sole
GIF, con un meccanismo suo.

## Le animazioni del libro

Si anima solo dove **il tempo è il contenuto**, cioè dove una figura ferma perde
davvero informazione. Architetture, tassonomie e confronti restano SVG. Il tetto
è **5–10 clip per capitolo**, un tetto, non una quota da riempire.

**Chi manca non si scrive qui, si conta.** Fino ad agosto 2026 questa riga
diceva «i capitoli non ancora coperti sono la lista dei prossimi», e la lista
era in testa a chi l'aveva scritta: nel frattempo il libro è passato da quindici
capitoli a trentacinque e le clip sono rimaste quindici, tutte nella metà
vecchia. Chi leggeva la parte nuova non ne vedeva nessuna. Adesso lo dice il
controllo, che non può dimenticarsene:

```bash
python3 scripts/coerenza.py --solo animazioni
```

Elenca i capitoli a zero clip e quelli che sforano il tetto. **Zero clip fa
fallire il controllo**, e c'è un modo solo per sbloccarlo: dichiarare il
capitolo in `animazioni/senza-clip.toml`, scrivendo perché lì il tempo non è il
contenuto.

Sembra una formalità e non lo è, perché la distinzione che conta non è fra un
capitolo animato e uno fermo (fermo può essere la scelta giusta, e per le
Conclusioni lo è): è fra **essersela chiesta e non essersela chiesta**, e
dall'esterno le due cose si somigliano al punto da confondersi. Elencare e
basta non è bastato: la prima versione di questa riga diceva «è un elenco da
guardare», e infatti nessuno l'ha guardato. Una riga da scrivere invece si
nota, e costa meno di un minuto.

| Sorgente | Figura nel libro | Sezione |
|---|---|---|
| `svg/alfabeta-pota.py` | `fig-alfabeta-pota` | `Ricerca/giocare-contro-qualcuno.md` |
| `svg/apertura-flusso.py` | `fig-apertura-flusso` | `VisioneArtificiale/geometria-e-profondita.md` |
| `svg/attacco-epsilon.py` | `fig-attacco-epsilon` | `AIResponsabile/privacy-e-robustezza.md` |
| `svg/autovettori.py` | `fig-autovettori` | `Matematica/algebra-lineare.md` |
| `svg/bootstrap-si-accumula.py` | `fig-bootstrap-accumula` | `MachineLearning/il-bootstrap.md` |
| `svg/bpe-fusioni.py` | `fig-bpe-fusioni` | `NaturalLanguageProcessing/tokenizzatori.md` |
| `svg/broadcasting-si-stende.py` | `fig-broadcasting-si-stende` | `Python/numpy.md` |
| `svg/cammino-latente.py` | `fig-cammino-latente` | `ModelliLatenti/il-salto-probabilistico.md` |
| `svg/campo-cieco.py` | `fig-campo-cieco` | `VerosimiglianzaEsatta/pixel-per-pixel.md` |
| `svg/catena-si-assesta.py` | `fig-catena-si-assesta` | `Matematica/catene-di-markov.md` |
| `svg/cancello-che-respinge.py` | `fig-cancello-che-respinge` | `IngegneriaLLM/loop-engineering.md` |
| `svg/ciclo-addestramento.py` | `fig-ciclo-addestramento` | `PyTorch/addestramento.md` |
| `svg/ciclo-agente.py` | `fig-ciclo-agente` | `Agenti/agenti-e-tool-use.md` |
| `svg/credito-spalmato.py` | `fig-credito-spalmato` | `AutoSupervisione/dibattito-rl.md` |
| `svg/ctc-collassa.py` | `fig-ctc-allineamento` | `SpeechRecognition/modelli-asr.md` |
| `svg/decodifica-per-differenza.py` | `fig-decodifica-per-differenza` | `VisioneLinguaggio/vedere-quel-che-non-ce.md` |
| `svg/deriva-ks.py` | `fig-deriva-ks` | `MLOps/monitoring-e-drift.md` |
| `svg/diffusione-avanti.py` | `fig-diffusione-avanti` | `ModelliDiffusione/come-funziona.md` |
| `svg/dqn-stabilita.py` | `fig-dqn-stabilita` | `DeepReinforcementLearning/dqn.md` |
| `svg/dropout.py` | `fig-dropout` | `DeepLearning/ottimizzazione-regolarizzazione.md` |
| `svg/euclide-scende.py` | `fig-euclide-scende` | `Introduzione/overview.md` |
| `svg/finestra-spettrogramma.py` | `fig-finestra-spettrogramma` | `Audio/dal-suono-alle-feature.md` |
| `svg/flash-attention-blocchi.py` | `fig-flash-attention-blocchi` | `GPU/flash-attention.md` |
| `svg/formiche-feromone.py` | `fig-formiche-feromone` | `SistemiMultiAgente/sciami-e-simulazioni.md` |
| `svg/frontiera-che-si-allarga.py` | `fig-frontiera` | `Ricerca/esplorare-lo-spazio.md` |
| `svg/gan-inseguimento.py` | `fig-gan-inseguimento` | `GAN/come-funziona.md` |
| `svg/gradiente-svanisce.py` | `fig-gradiente-svanisce` | `DeepLearning/ottimizzazione-regolarizzazione.md` |
| `svg/gradienti-integrati.py` | `fig-gradienti-integrati` | `Interpretabilita/attribuzione-e-meccanicistica.md` |
| `svg/hopfield-ricorda.py` | `fig-hopfield-ricorda` | `ModelliEnergia/memoria-associativa.md` |
| `svg/il-codice-si-accorcia.py` | `fig-il-codice-si-accorcia` | `AutoSupervisione/capire-e-accorciare.md` |
| `svg/kmeans-converge.py` | `fig-kmeans-converge` | `MachineLearning/riduzione-clustering.md` |
| `svg/learning-rate.py` | `fig-learning-rate` | `DeepLearning/ottimizzazione-regolarizzazione.md` |
| `svg/origine-mobile.py` | `fig-walk-forward-validazione` | `SerieTemporali/validazione-e-feature.md` |
| `svg/percettrone-impara.py` | `fig-percettrone-impara` | `RetiNeurali/percettrone.md` |
| `svg/pinn-residuo.py` | `fig-pinn-residuo` | `PINN/come-funziona.md` |
| `svg/potatura-che-assottiglia.py` | `fig-potatura` | `Efficienza/meno-pesi.md` |
| `svg/scan-parallelo.py` | `fig-scan-parallelo` | `StateSpaceModel/mamba.md` |
| `svg/spartito-in-fila.py` | `fig-spartito-in-fila` | `Audio/generazione-audio.md` |
| `svg/vetrina-si-ordina.py` | `fig-vetrina-si-ordina` | `SistemiRaccomandazione/raccomandazione-neurale.md` |
| `svg/sogno-diverge.py` | `fig-sogno-diverge` | `WorldModels/mondi-in-miniatura.md` |
| `svg/xor-non-separabile.py` | `fig-xor-non-separabile` | `RetiNeurali/percettrone.md` |
| `svg/xor-si-piega.py` | `fig-xor-si-piega` | `RetiNeurali/percettrone.md` |
| `attenzione-mascherata.py` | `fig-attenzione-mascherata` | `Transformers/architettura.md` |
| `backpropagation.py` | `fig-backpropagation-animata` | `RetiNeurali/backpropagation.md` |
| `convoluzione.py` | `fig-convoluzione-animata` | `DeepLearning/reti-convoluzionali.md` |
| `diffusione-denoising.py` | `fig-diffusione-denoising` | `ModelliDiffusione/come-funziona.md` |
| `generazione-autoregressiva.py` | `fig-generazione-autoregressiva` | `Transformers/llm.md` |
| `iterazione-valore.py` | `fig-iterazione-valore` | `ReinforcementLearning/mdp-valore.md` |
| `limite-centrale.py` | `fig-limite-centrale` | `Matematica/probabilita-statistica.md` |
| `message-passing.py` | `fig-message-passing-animato` | `GraphNeuralNetwork/message-passing.md` |
| `stato-ricorrente.py` | `fig-stato-ricorrente` | `AttenzioneLineare/dalla-softmax-alla-ricorrenza.md` |

Dove la scena mostra dei numeri, li **calcola**: l'iterazione di valore esegue
davvero Bellman sulla griglia, la ricorrenza dell'attenzione lineare somma
davvero i prodotti esterni, la convoluzione convolve. Nessun numero è scritto a
mano, così non può smentire il testo.
