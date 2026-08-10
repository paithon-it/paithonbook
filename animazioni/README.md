# animazioni/

Sorgenti delle figure animate del libro. Due strade, che non competono:

| | `animazioni/*.py` (Manim) | `animazioni/svg/*.py` (SVG animato) |
|---|---|---|
| uscita | `book/figures/*.gif` raster | `book/figures/*.svg`, testo |
| peso | ~570 KB a clip | **~6 KB** a figura |
| in git | blob binario nuovo a ogni render | diffabile |
| stampa | serve la `-striscia.png` | si congela da sé |
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

## Online la clip, a stampa la striscia

Ogni animazione produce **due** artefatti in `book/figures/`, e sono la stessa
figura su due supporti:

- `<nome>.gif` — è quella richiamata dal `{figure}` nei file MyST: vive nel
  libro online, dove il movimento si può guardare;
- `<nome>-striscia.png` — 2–3 fotogrammi affiancati, la versione ferma per la
  **stampa** e il PDF. Non è referenziata da nessun `.md`: si sostituisce alla
  GIF quando si impagina l'edizione cartacea.

Si generano insieme con `--striscia`. Per questo ogni scena è scritta in modo
che **l'ultimo fotogramma regga da solo**: è quello che finisce sulla carta.

## Le animazioni del libro

Si anima solo dove **il tempo è il contenuto**, cioè dove una figura ferma perde
davvero informazione. Architetture, tassonomie e confronti restano SVG. Il tetto
è **5–10 clip per capitolo** — un tetto, non una quota da riempire.

Questa è la **prima ondata**: un capitolo per clip, i concetti più ostici del
libro. I capitoli non ancora coperti sono la lista dei prossimi.

| Sorgente | Figura nel libro | Sezione |
|---|---|---|
| `svg/percettrone-impara.py` | `fig-percettrone-impara` | `RetiNeurali/percettrone.md` |
| `svg/xor-non-separabile.py` | `fig-xor-non-separabile` | `RetiNeurali/percettrone.md` |
| `svg/autovettori.py` | `fig-autovettori` | `Matematica/algebra-lineare.md` |
| `svg/gradiente-svanisce.py` | `fig-gradiente-svanisce` | `DeepLearning/ottimizzazione-regolarizzazione.md` |
| `svg/dropout.py` | `fig-dropout` | `DeepLearning/ottimizzazione-regolarizzazione.md` |
| `svg/learning-rate.py` | `fig-learning-rate` | `DeepLearning/ottimizzazione-regolarizzazione.md` |
| `limite-centrale.py` | `fig-limite-centrale` | `Matematica/probabilita-statistica.md` |
| `backpropagation.py` | `fig-backpropagation-animata` | `RetiNeurali/backpropagation.md` |
| `convoluzione.py` | `fig-convoluzione-animata` | `DeepLearning/reti-convoluzionali.md` |
| `attenzione-mascherata.py` | `fig-attenzione-mascherata` | `Transformers/architettura.md` |
| `generazione-autoregressiva.py` | `fig-generazione-autoregressiva` | `Transformers/llm.md` |
| `stato-ricorrente.py` | `fig-stato-ricorrente` | `AttenzioneLineare/dalla-softmax-alla-ricorrenza.md` |
| `diffusione-denoising.py` | `fig-diffusione-denoising` | `ModelliDiffusione/come-funziona.md` |
| `iterazione-valore.py` | `fig-iterazione-valore` | `ReinforcementLearning/mdp-valore.md` |
| `message-passing.py` | `fig-message-passing-animato` | `GraphNeuralNetwork/message-passing.md` |

Dove la scena mostra dei numeri, li **calcola**: l'iterazione di valore esegue
davvero Bellman sulla griglia, la ricorrenza dell'attenzione lineare somma
davvero i prodotti esterni, la convoluzione convolve. Nessun numero è scritto a
mano, così non può smentire il testo.
