# La via di LeCun: predire nello spazio delle idee

Il 27 giugno 2022 Yann LeCun deposita su OpenReview (la piattaforma dove di
solito si caricano gli articoli in attesa di revisione) un documento di 62
pagine intitolato *A Path Towards Autonomous Machine Intelligence*
{cite}`lecun2022path`. Già il sottotitolo è insolito: «versione 0.9.2», come
un software non ancora finito. E insolito è tutto il resto: non è un paper di
risultati, con esperimenti e tabelle, ma un **documento di posizione** (la
visione dell'autore su come costruire macchine intelligenti nei prossimi dieci
anni) messo online apposta perché chiunque potesse commentarlo, criticarlo,
smontarlo pubblicamente. Un premio Turing che espone il proprio programma di
ricerca, in bozza dichiarata, alle obiezioni di tutti: in un'epoca in cui si
tende a mostrare solo ciò che già funziona, è una mossa da notare.

Dentro c'è un'architettura per **agenti autonomi** fatta di sei moduli. La
**percezione** stima lo stato del mondo a partire dai sensori; il **world
model** (il cuore del progetto) predice come quello stato evolverà, anche in
risposta ad azioni immaginate; il modulo di **costo** misura il «disagio»
dell'agente, e ha due parti: una cablata e immutabile (l'analogo del dolore e
del piacere) e un *critico* appreso che impara a prevedere il costo futuro;
l'**attore** propone le azioni; la **memoria a breve termine** tiene traccia
degli stati recenti del mondo; e il **configuratore** sovrintende, regolando
gli altri moduli a seconda del compito. L'agente può agire in due modi: in
modalità reattiva, dove la percezione pilota direttamente l'azione, o in
modalità deliberata, dove usa il world model per *simulare* sequenze di azioni
e scegliere quella dal costo previsto più basso (un'eco della distinzione tra
pensiero veloce e pensiero lento resa celebre da Daniel Kahneman, che LeCun
richiama esplicitamente).

Sei moduli sono tanti, e in gran parte sono ancora sulla carta. Ma tutto il
progetto sta o cade su una domanda sola: **come si addestra il world model?**
La risposta di LeCun è: con l'auto-supervisione, guardando (come il neonato
dell'inizio del capitolo) enormi quantità di video senza alcuna etichetta,
imparando a prevedere ciò che viene dopo. Fin qui, niente di nuovo: anche i
mondi in miniatura di Ha e Schmidhuber facevano qualcosa di simile. La rottura
è nel *dove* si fa la previsione.

## Perché non predire i pixel

Nella sezione sui mondi in miniatura la previsione avveniva già in uno spazio
compresso (i 32 numeri del VAE) ma quella compressione era addestrata a
**ricostruire i pixel**: la qualità del codice si misurava sulla fedeltà del
disegno. LeCun propone di tagliare anche quel cordone. Il futuro, osserva, ha
due proprietà che rendono la previsione nei pixel una pessima idea: è
**molteplice** (da uno stesso presente possono seguire tanti futuri diversi,
tutti plausibili) ed è pieno di **dettagli irrilevanti**, il suo esempio
ricorrente è un albero in un video: nessun modello potrà mai prevedere la
posizione esatta di ogni foglia mossa dal vento, e soprattutto *non serve a
niente* provarci.

`````{tab} Elementare

Un bicchiere è in bilico sul bordo del tavolo. «Cadrà e andrà in pezzi»: lo
prevedi in un decimo di secondo, e questa previsione ti basta per allungare la
mano. Ora prova invece a prevedere la *fotografia esatta* della scena tra due
secondi: dove sarà ogni scheggia, come si rifletterà la luce su ogni
frammento, che forma avrà la macchia d'acqua sul pavimento. Impossibile, e del
tutto inutile: nessuna decisione sensata dipende dalla forma della terza
scheggia. Un modello costretto a prevedere l'immagine pixel per pixel ha
esattamente questo problema, due volte. Primo: spreca quasi tutta la sua
capacità a studiare dettagli che non contano nulla. Secondo: siccome i futuri
possibili sono tanti (le schegge possono disporsi in mille modi) e lui deve
produrre *una* immagine sola, la scelta meno penalizzata è la media di tutti i
futuri; una foto fantasma, sfocata, in cui mille rotture diverse si
sovrappongono. La proposta di LeCun: non prevedere la foto, prevedere il
*succo* («bicchiere in pezzi sul pavimento, acqua sparsa»), cioè prevedere
nello **spazio delle idee**, dove i mille futuri diversi nei dettagli
diventano un futuro solo, quello che conta.

`````

`````{tab} Superiore

Se si addestra un predittore $g$ a minimizzare l'errore quadratico
$\mathbb{E}\,\lVert y - g(x) \rVert^2$ su un futuro $y$ intrinsecamente
stocastico, l'ottimo è la media condizionata $g^*(x) = \mathbb{E}[y \mid x]$:
quando i modi della distribuzione sono molti e distinti, la loro media è
un'immagine sfocata che non corrisponde a *nessun* futuro reale; è la ragione
per cui la predizione video nei pixel produce fantasmi lattiginosi. La
proposta di {cite}`lecun2022path` è la **JEPA** (*Joint-Embedding Predictive
Architecture*): due encoder mappano contesto e target nello spazio delle
rappresentazioni, $s_x = f_\theta(x)$ e $s_y = \bar{f}_{\bar{\theta}}(y)$, e
un **predictor** $g_\phi$ opera interamente lì:

$$
E(x, y) = \big\lVert\, g_\phi(s_x, z) - s_y \,\big\rVert_2^2,
$$

dove $E$ è l'energia della coppia, $z$ è un'eventuale variabile latente che
assorbe la molteplicità dei futuri (quale dei tanti esiti plausibili si è
realizzato), e $\theta$, $\bar{\theta}$, $\phi$ sono i parametri dei due
encoder e del predictor. Il collegamento con il capitolo precedente è
letterale: una JEPA **è** un modello a energia non normalizzato; la
compatibilità tra presente e futuro è l'errore di predizione nello spazio
latente, l'inferenza è la solita $\arg\min$, e della funzione di partizione
non c'è alcun bisogno. La libertà nuova sta nell'encoder del target: poiché
$y$ non va ricostruito ma solo *rappresentato*, $\bar{f}$ può legittimamente
buttare via informazione. I gradi di libertà imprevedibili e irrilevanti (le
foglie, i riflessi) possono semplicemente non arrivare nello spazio in cui si
calcola la loss: è una scelta d'architettura, non una speranza.

`````

La {numref}`fig-jepa-architettura` mette i due mondi uno sopra l'altro. Nel
pannello A il decoder generativo deve tornare fino ai pixel, e la loss lo
punisce su ogni foglia che trema; nel pannello B la previsione parte dal
contesto e arriva al target senza mai uscire dallo spazio delle
rappresentazioni: i dettagli irrilevanti restano fuori dalla porta.

```{figure} ../figures/jepa-architettura.svg
:name: fig-jepa-architettura
:alt: "Confronto a due pannelli. Sopra, l'architettura generativa: dal contesto un decoder disegna ogni pixel del futuro e la loss confronta pixel per pixel la predizione sfocata con il futuro reale, sprecando capacità sui dettagli imprevedibili. Sotto, la JEPA: un encoder in teal trasforma il contesto in un embedding, un encoder target tratteggiato aggiornato per media mobile esponenziale trasforma il target, un predictor in terracotta predice l'embedding del target e la loss confronta i due embedding nello spazio delle rappresentazioni."
:width: 100%

Generativa contro JEPA: la prima predice il futuro nei pixel (e deve
indovinare anche l'irrilevante), la seconda lo predice nello spazio delle
rappresentazioni, dove l'irrilevante non è mai entrato.
```

## Il ritorno del collasso

Chi ha letto il capitolo sui modelli a energia sa già dove si nasconde la
trappola, perché è la stessa del buttafuori pigro: se la loss premia solo la
vicinanza tra embedding predetto e embedding del target, la soluzione più
comoda non è capire il mondo (è **appiattirlo**). Basta che i due encoder
imparino a produrre sempre lo stesso vettore, qualunque cosa vedano:
predizione perfetta, energia zero ovunque, rappresentazioni che non
distinguono un gatto da un lampadario. È il **collasso**, e per le JEPA è il
pericolo numero uno, perché qui (a differenza dei modelli generativi, ancorati
ai pixel veri) anche il *bersaglio* è prodotto da una rete che avrebbe tutto
l'interesse a barare. Nel documento del 2022 LeCun indica la famiglia di
rimedi che preferisce: quella **regolarizzata**, già incontrata in chiusura
della sezione scorsa. Ma nei sistemi JEPA costruiti davvero da Meta la difesa
concreta è un'altra, più semplice e più sottile.

`````{tab} Elementare

Immagina un allievo e un insegnante. L'allievo guarda la parte visibile della
foto e prova a *descrivere* che cosa c'è nella parte coperta; l'insegnante,
che vede la foto intera, scrive la descrizione giusta; il voto misura quanto
le due descrizioni combaciano. Se allievo e insegnante potessero mettersi
d'accordo, la truffa sarebbe immediata: rispondere entrambi, sempre, «boh»
(descrizioni identiche, voti perfetti, e nessuno dei due che abbia mai
guardato la foto). Il trucco che rompe la truffa: come insegnante si usa una
**copia lenta dell'allievo**. Non è una seconda rete addestrata a parte: è
l'allievo stesso *com'era in media nelle ultime settimane* (i suoi pesi,
mescolati poco alla volta). Questo insegnante non ascolta le lamentele sul
voto (nessun segnale di addestramento lo raggiunge: tecnicamente, non riceve
gradiente) e cambia idea solo al ritmo a cui l'allievo migliora *davvero*. Con
lui non ci si può accordare al ribasso: l'unico modo di prendere buoni voti è
inseguire le sue descrizioni, che si muovono piano e non colludono. È una
soluzione empirica (perché funzioni così bene è ancora oggetto di studio) ma
funziona.

`````

`````{tab} Superiore

Il rimedio usato dai sistemi JEPA di Meta è **architetturale**: asimmetria
tra i due encoder. L'encoder del target non viene addestrato per
retropropagazione ma mantenuto come **media mobile esponenziale** (EMA,
*exponential moving average*) dei pesi dell'encoder di contesto:

$$
\bar{\theta} \;\leftarrow\; \tau\, \bar{\theta} + (1 - \tau)\, \theta,
$$

dove $\theta$ sono i pesi dell'encoder di contesto, $\bar{\theta}$ quelli
dell'encoder target e $\tau$ un momento vicino a 1 (in I-JEPA parte da 0,996):
a ogni passo il target si sposta di una frazione millesimale verso l'encoder
corrente. All'EMA si accompagna lo **stop-gradient**: la loss non si propaga
mai attraverso il ramo del target, che è puro riferimento. La coppia EMA +
stop-gradient impedisce la discesa coordinata dei due encoder verso la
costante (il bersaglio insegue, non collude) ed è la stessa scoperta empirica
che aveva sorpreso la comunità con BYOL nel 2020: niente coppie negative,
niente termini contrastivi, eppure niente collasso. Una comprensione teorica
completa del *perché* manca ancora, ed è giusto dirlo; il documento del 2022
{cite}`lecun2022path` discute anche l'alternativa esplicitamente regolarizzata
(varianza mantenuta sopra una soglia, covarianze fuori diagonale penalizzate,
alla VICReg), ma I-JEPA e V-JEPA, nei paper, si affidano all'asimmetria EMA.

`````

## I-JEPA: la scommessa alla prova delle immagini

Nel documento del 2022 la JEPA è soprattutto un diagramma. La prima
incarnazione convincente arriva l'anno dopo, dal gruppo di LeCun a Meta AI:
**I-JEPA** (*Image-based JEPA*) {cite}`assran2023self`, presentata alla
conferenza CVPR. Gli ingredienti li conosciamo tutti: l'encoder è un Vision
Transformer (il ViT che nel capitolo sui Transformer tagliava l'immagine in
tessere {cite}`dosovitskiy2021image`) e il compito è un indovinello di
mascheramento: dato un solo blocco di *contesto* dell'immagine, prevedere che
cosa c'è in quattro blocchi *bersaglio* nascosti. La novità è tutta nel **che
cosa** si prevede: non i pixel dei blocchi mancanti, ma le loro
**rappresentazioni**, calcolate dall'encoder target aggiornato per EMA.

`````{tab} Elementare

È il gioco della cartolina strappata. Ti mostro una cartolina a cui mancano
quattro rettangoli e ti chiedo: che cosa c'era lì? Non ti chiedo di
*ridisegnare* i pezzi mancanti: quello sarebbe il compito generativo, e ti
costringerebbe a inventare dettagli che non puoi sapere. Ti chiedo di
*descriverli*: «lì continua il muso del cane, girato verso destra». A
correggerti è la copia lenta di te stesso, che ha visto la cartolina intera e
ha scritto le sue descrizioni. Due dettagli fanno la differenza. Primo: i
rettangoli nascosti sono *grandi*, per indovinare un pezzo grande devi aver
capito la scena («è un cane, quindi là sotto c'è una zampa»), mentre per un
buchino basta allungare i bordi, senza capire niente. Secondo: al modello non
servono i trucchi artigianali con cui di solito si addestrano questi sistemi
(versioni ritagliate, specchiate, ricolorate della stessa foto, scelte a mano
da chi progetta). Basta l'indovinello. E i risultati danno ragione alla
scommessa: con appena l'1% delle etichette di ImageNet (una dozzina di foto
etichettate per categoria) I-JEPA classifica meglio dei metodi che
ricostruiscono i pixel, imparando in una frazione del loro tempo di calcolo.

`````

`````{tab} Superiore

L'encoder di contesto (un ViT) elabora solo le patch visibili del blocco di
contesto; un **predictor** (un ViT più stretto) riceve $s_x$ e, per ciascuno
dei $M = 4$ blocchi bersaglio, token posizionali che indicano *dove*
prevedere; la loss è la distanza $L_2$ media tra le rappresentazioni predette
e quelle prodotte dall'encoder target:

$$
\mathcal{L} = \frac{1}{M} \sum_{i=1}^{M}
\big\lVert\, \hat{s}_y^{(i)} - s_y^{(i)} \,\big\rVert_2^2,
$$

dove $\hat{s}_y^{(i)}$ e $s_y^{(i)}$ sono le rappresentazioni (a livello di
patch) predette e bersaglio del blocco $i$. Un dettaglio architetturale è
decisivo: l'encoder target elabora l'immagine **intera**, e i bersagli si
ottengono mascherando la sua *uscita*, non il suo ingresso; così ogni
rappresentazione-bersaglio incorpora il contesto globale ed è semanticamente
ricca. Niente augmentation artigianali: nessun crop multiplo, nessun jitter di
colore. I numeri del paper {cite}`assran2023self`: su ImageNet-1K con l'**1%
delle etichette**, un ViT-H/14 pre-addestrato con I-JEPA raggiunge il 73,3% di
accuratezza top-1 (77,3% per il ViT-H/16 a risoluzione 448), contro il 71,5%
di MAE (il metodo generativo che ricostruisce i pixel mascherati) e il 69,7%
di iBOT (lì con un ViT-B/16), e il pre-addestramento del ViT-H/14 richiede
meno di 1200 ore-GPU (meno di 72 ore su 16 A100), oltre dieci volte meno di
MAE a parità di architettura. Predire rappresentazioni invece di pixel,
dunque, non solo funziona: costa anche molto meno.

`````

## Dal fotogramma al film: V-JEPA

Le immagini erano il banco di prova; il progetto di LeCun, però, parla di
*futuro*, e il futuro vive nei video. **V-JEPA** {cite}`bardes2024revisiting`
(2024) trasporta lo schema dalla dimensione spaziale a quella
spazio-temporale: si maschera una regione del video e si prevedono le sue
rappresentazioni a partire dal resto. C'è una finezza che rivela quanto i
video siano una bestia diversa: i fotogrammi vicini sono quasi identici,
quindi se la maschera coprisse zone diverse in fotogrammi diversi il modello
potrebbe barare copiando dal fotogramma accanto. La maschera è perciò un
**tubo**: la stessa regione spaziale, estesa lungo *tutta* la durata della
clip, e generosa: in media viene coperto circa il 90% del video. Addestrato
così su due milioni di video pubblici, senza etichette, senza testo e senza
ricostruzione, V-JEPA produce rappresentazioni che (con la rete congelata e
solo una piccola testa di classificazione sopra) raggiungono l'81,9% su
Kinetics-400 (riconoscere l'azione: chi nuota, chi suona) e il 72,2% su
Something-Something-v2, un benchmark che richiede di capire il *movimento*
(«spingere qualcosa da sinistra a destra»), non solo l'aspetto. È il segnale
che il modello ha imparato qualcosa sulla dinamica delle scene, non un
catalogo di texture.

## V-JEPA 2: il world model tocca il mondo

Nel giugno 2025 arriva il passo successivo, ed è quello che riporta tutta
questa storia al punto di partenza del capitolo: usare il modello per *agire*.
**V-JEPA 2** {cite}`assran2025vjepa` scala la ricetta (un encoder da 1,2
miliardi di parametri, pre-addestrato su oltre un milione di ore di video da
internet) e i numeri di comprensione salgono di conseguenza: 77,3% su
Something-Something-v2, stato dell'arte nell'anticipazione delle azioni su
Epic-Kitchens-100 (prevedere, guardando una cucina in soggettiva, che cosa
farà la persona nel prossimo secondo), buoni risultati di video question
answering una volta allineato con un modello di linguaggio.

Ma la parte concettualmente nuova è **V-JEPA 2-AC** (*action-conditioned*):
sopra l'encoder congelato viene addestrato un predictor **condizionato sulle
azioni** del robot, usando meno di 62 ore di video robotici *non etichettati*
del dataset pubblico DROID. Il risultato è un world model operativo: dato lo
stato attuale e una sequenza di azioni candidate, predice (sempre nello spazio
delle rappresentazioni), dove finirà il mondo. La pianificazione funziona per
immaginazione, come nei Dreamer ma senza ricompense: si dà al robot
un'**immagine-obiettivo** (la tazza sopra il piatto), il modello simula gli
effetti di sequenze di azioni alternative e sceglie quella il cui esito
previsto è più vicino all'obiettivo. Portato su bracci robotici Franka in due
laboratori *mai visti* durante l'addestramento, senza alcuna dimostrazione né
messa a punto specifica, il sistema esegue prese e spostamenti di oggetti
nuovi: pianificazione robotica **zero-shot**. I compiti sono semplici
(afferrare, posare, spostare) e l'orizzonte di pianificazione è corto; ma è il
punto esatto in cui la via di LeCun smette di essere un diagramma e tocca,
letteralmente, il mondo fisico.

## Tre famiglie per imparare senza etichette

Vale la pena fermarsi e mettere ordine, perché in questo libro abbiamo ormai
incontrato tutti e tre i grandi modi di imparare senza annotatori umani.

`````{tab} Elementare

Tre studenti, stessi libri, nessun professore. Il primo studia **ricopiando
con i buchi**: cancella pezzi del testo e si allena a riscriverli identici,
parola per parola o pixel per pixel; è il metodo *generativo*, quello di BERT
con le frasi (gli «esercizi a buchi» del capitolo sui Transformer) e di MAE
con le foto. Il secondo studia col **gioco delle coppie**: mescola foto e
didascalie e impara a dire quali vanno insieme e quali no; è il metodo
*contrastivo*, quello di CLIP, che avvicina ogni immagine alla sua descrizione
e la allontana dalle altre. Il terzo (la via JEPA) studia **prevedendo il
riassunto**: copre un pezzo e, invece di ricopiarlo, ne prevede la
*descrizione*, confrontandola con quella di una copia lenta di sé. Non è una
classifica: il primo metodo ha vinto nel linguaggio, il secondo ha unito
immagini e parole, il terzo scommette sul futuro e sul video. Sono tre
risposte diverse alla stessa domanda: che cosa, di ciò che manca, vale la pena
prevedere?

`````

`````{tab} Superiore

**Generativa**: si ricostruisce l'input nello spazio dell'input. Il masked
language modeling di BERT {cite}`devlin2019bert` predice i token mascherati
con una cross-entropia sul vocabolario; MAE fa lo stesso con i pixel delle
patch mascherate, con loss $L_2$. Funziona magnificamente sul testo (dove i
token sono discreti e la softmax rappresenta senza sforzo l'incertezza) e
resta più goffa su segnali continui ad alta dimensione, dove l'equivalente
della softmax non esiste e ricostruire costringe a modellare l'irrilevante: è
l'argomento centrale di {cite}`lecun2022path`. **Contrastiva**: si impara una
geometria, avvicinando le coppie compatibili e allontanando quelle
incompatibili (CLIP {cite}`radford2021learning` con la loss InfoNCE su coppie
immagine–didascalia). Nel lessico del capitolo precedente: energia abbassata
sulle coppie giuste e *alzata esplicitamente* sui controesempi, con la nota
difficoltà di trovarne mai abbastanza in alta dimensione. **Predittiva nello
spazio latente**: la famiglia JEPA; energia = errore di predizione tra
embedding, nessuna ricostruzione, nessuna coppia negativa, collasso evitato
per asimmetria architetturale (EMA, stop-gradient) o per regolarizzazione
esplicita (varianza/covarianza). È la più giovane delle tre, e quella su cui
pesa la scommessa più grossa.

`````

## Una scommessa aperta

Chiudiamo con l'onestà dovuta. Quella raccontata in questa sezione è una
**linea di ricerca in corso**, non un traguardo raggiunto. Le rappresentazioni
JEPA sono eccellenti e costano poco, e V-JEPA 2-AC ha mostrato che un world
model auto-supervisionato può guidare un robot vero; ma dell'architettura a
sei moduli del 2022 la maggior parte resta sulla carta: la JEPA gerarchica che
dovrebbe pianificare su più scale temporali, il configuratore, il ragionamento
a lungo orizzonte. I critici, dal canto loro, fanno notare che la storia
recente non è stata tenera con le previsioni di insufficienza: i modelli
generativi, scalati abbastanza, continuano a esibire capacità che «non
avrebbero dovuto» avere, e i generatori di video più spinti producono scene la
cui coerenza fisica, almeno in apparenza, cresce con la scala. Se per capire
il mondo serva davvero smettere di generarlo, o se generare *sia* un modo di
capire, è esattamente la domanda su cui il campo è spaccato, e LeCun, come
ricordato in apertura di capitolo, ci ha scommesso la carriera, lasciando Meta
per una startup dedicata ai world model. La prossima sezione attraversa il
fronte opposto del dibattito: i simulatori generativi di video, da Sora a
Genie, e la domanda se un modello che *disegna* futuri plausibili abbia capito
la fisica o abbia solo imparato a imitarla.

## Una mini-JEPA in PyTorch

Tutti i pezzi della sezione (encoder, copia lenta EMA, predictor, loss tra
embedding) stanno comodamente in una pagina di PyTorch. L'esperimento è
volutamente in miniatura: ogni «immagine» è una scena finta di 8 patch,
generate da un contenuto latente comune più rumore; il modello vede 6 patch di
contesto e deve prevedere l'**embedding** (non i valori!) delle 2 patch
mascherate. Il commento chiave è sull'EMA: è lei che tiene il sistema lontano
dal collasso.

```python
import copy
import torch
from torch import nn

torch.manual_seed(0)

DIM_PATCH, DIM_EMB = 16, 32
N_PATCH, N_CONTESTO = 8, 6          # per scena: 6 patch visibili, 2 mascherate

# Mappa fissa dal "contenuto" della scena all'aspetto delle patch
PROIEZIONE = torch.randn(4, DIM_PATCH)

def genera_batch(n=256):
    """Ogni scena nasce da un contenuto latente comune alle sue 8 patch."""
    contenuto = torch.randn(n, 1, 4)               # il "succo" della scena
    patch = contenuto @ PROIEZIONE                 # come il succo appare
    return patch + 0.25 * torch.randn(n, N_PATCH, DIM_PATCH)  # dettagli casuali

# Encoder (l'allievo), predictor, ed encoder target (la copia lenta)
encoder = nn.Sequential(
    nn.Linear(DIM_PATCH, 64), nn.ReLU(), nn.Linear(64, DIM_EMB))
predictor = nn.Sequential(
    nn.Linear(DIM_EMB, 64), nn.ReLU(), nn.Linear(64, DIM_EMB))

encoder_target = copy.deepcopy(encoder)
for p in encoder_target.parameters():
    p.requires_grad_(False)          # stop-gradient: il bersaglio non si allena

@torch.no_grad()
def aggiorna_target(tau=0.996):
    """EMA: il target insegue lentamente l'encoder. È l'anti-collasso:
    non ricevendo gradiente, non può 'mettersi d'accordo' con l'encoder
    per appiattire tutti gli embedding sulla stessa costante."""
    for p, p_t in zip(encoder.parameters(), encoder_target.parameters()):
        p_t.mul_(tau).add_((1.0 - tau) * p)

opt = torch.optim.Adam(
    list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3)

for passo in range(1, 601):
    patch = genera_batch()                          # (256, 8, 16)
    # contesto -> embedding riassuntivo (media delle 6 patch visibili)
    s_x = encoder(patch[:, :N_CONTESTO]).mean(dim=1)        # (256, 32)
    # target -> embedding calcolato dalla copia lenta, senza gradiente
    with torch.no_grad():
        s_y = encoder_target(patch[:, N_CONTESTO:]).mean(dim=1)  # (256, 32)
    s_y_pred = predictor(s_x)                       # predizione tra embedding
    loss = nn.functional.mse_loss(s_y_pred, s_y)    # loss nello spazio latente

    opt.zero_grad()
    loss.backward()
    opt.step()
    aggiorna_target()                               # un passetto di EMA

    if passo in (1, 100, 200, 400, 600):
        # se gli embedding collassassero, questa varietà scenderebbe verso 0
        varieta = s_y.std(dim=0).mean().item()
        print(f"passo {passo}: loss {loss.item():.4f}  "
              f"varietà degli embedding {varieta:.3f}")
```

Eseguendolo, la loss crolla in un centinaio di passi (da circa 0,23 a meno di
0,01), mentre la «varietà» degli embedding (la deviazione standard media tra
scene diverse) non solo resta lontana da zero, ma *cresce*, da circa 0,4 a
1,0: il modello impara a prevedere il *contenuto* delle patch nascoste (che è
condiviso con il contesto) e ignora il rumore (che non è prevedibile), senza
appiattire le rappresentazioni. In scala giocattolo, è esattamente il
comportamento per cui I-JEPA e V-JEPA sono progettate. Ciò che qui manca (il
ViT al posto del piccolo MLP, i token posizionali che dicono al predictor
*dove* prevedere, milioni di immagini e di ore di video) è ingegneria; la
logica è tutta in queste righe.

```{admonition} Da ricordare
:class: important
- Nel 2022 LeCun pubblica su OpenReview *A Path Towards Autonomous Machine
  Intelligence* {cite}`lecun2022path`: non un paper di risultati ma un
  progetto di architettura; sei moduli (percezione, world model, costo,
  attore, memoria a breve termine, configuratore) attorno a un world model
  appreso per auto-supervisione.
- **Predire nei pixel è la strada sbagliata**: il futuro è molteplice e
  pieno di dettagli irrilevanti; la minimizzazione dell'errore quadratico
  produce la media sfocata dei futuri. La **JEPA** predice nello spazio
  delle rappresentazioni: è un'architettura a energia non normalizzata, dove
  l'energia è l'errore di predizione tra embedding.
- Il pericolo è il solito **collasso** (embedding costanti, energia bassa
  ovunque); la difesa usata dai sistemi reali è l'asimmetria **EMA +
  stop-gradient**: l'encoder target è una copia lenta che non riceve
  gradiente e non può colludere.
- **I-JEPA** {cite}`assran2023self` (CVPR 2023): un ViT predice le
  rappresentazioni di quattro blocchi mascherati dal contesto; niente
  augmentation artigianali; con l'1% delle etichette di ImageNet batte i
  metodi a ricostruzione di pixel (73,3% contro 71,5% di MAE) con oltre
  dieci volte meno calcolo.
- **V-JEPA** {cite}`bardes2024revisiting` porta lo schema al video (maschere
  a tubo estese su tutta la clip); **V-JEPA 2** {cite}`assran2025vjepa`
  scala a oltre un milione di ore di video e, con meno di 62 ore di video
  robotici non etichettati, ottiene pianificazione robotica **zero-shot**
  su bracci Franka mai visti: il world model tocca il mondo fisico.
- Tre famiglie di auto-supervisione: **generativa** (ricostruisci: BERT,
  MAE), **contrastiva** (avvicina/allontana: CLIP), **predittiva nello
  spazio latente** (JEPA). La partita tra generare e predire-nelle-idee è
  aperta: la prossima sezione visita l'altra sponda.
```
