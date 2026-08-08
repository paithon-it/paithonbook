# Quando la diffusione incontra i Transformer: DiT

C'è un'ironia nascosta nella storia raccontata fin qui. Il cuore di ogni
modello di diffusione che abbiamo incontrato (il DDPM del 2020, lo Stable
Diffusion del 2022) è la U-Net {cite}`ronneberger2015u`: un'architettura nata
nel 2015 a Friburgo per **segmentare cellule al microscopio**, come ricordiamo
dal capitolo sulla visione artificiale. Per anni nessuno l'ha messa in
discussione: le si è aggiunta un po' di attenzione ai piani bassi, le si è
iniettato l'embedding del tempo, ma l'impalcatura (convoluzioni, discesa,
risalita, *skip connections*) è rimasta quella del microscopio.

Nel 2023 William Peebles, allora dottorando a Berkeley, e Saining Xie,
professore alla New York University, si fanno la domanda che a quel punto del
libro dovrebbe suonare familiare: la U-Net serve *davvero*? O anche qui, come
era successo nel 2017 per la traduzione con la caduta delle reti ricorrenti,
vale il titolo di quel paper: *attention is all you need*
{cite}`vaswani2017attention`? La loro risposta si chiama **DiT**, *Diffusion
Transformer* {cite}`peebles2023scalable`, presentata alla International
Conference on Computer Vision del 2023. E la storia ha un seguito che ne
misura il peso: un anno dopo ritroveremo lo stesso Peebles, insieme a Tim
Brooks, alla guida di un progetto chiamato Sora.

## Affettare la scheda: le patch come parole

DiT non butta via tutto. Il trasloco nello spazio latente della sezione
precedente resta: c'è ancora l'archivista (il VAE che comprime le immagini in
schede compatte) e la diffusione lavora ancora sulle schede, non sui pixel
{cite}`rombach2022high`. A cambiare mestiere è solo il restauratore: via la
U-Net, dentro un Transformer. Ma un Transformer mangia *sequenze di token*,
non griglie di numeri. Come dargli in pasto un latente? La mossa è già nel
nostro repertorio: è la stessa, identica, del Vision Transformer incontrato
nel capitolo sui Transformer {cite}`dosovitskiy2021image`.

`````{tab} Elementare

La scheda dell'archivista (per le immagini di ImageNet su cui DiT viene
addestrato) è una piccola griglia: 32 caselle per lato, con quattro numeri in
ogni casella. Il Vision Transformer ci ha insegnato il trucco per trasformare
una griglia in una frase: tagliarla in **tessere**, come un mosaico, e mettere
le tessere in fila come se fossero parole. Qui le tessere sono quadratini di 2
caselle per lato: $16 \times 16 = 256$ tessere, cioè una "frase" di 256
parole.

Da qui in poi il lavoro lo conosciamo dal capitolo sui Transformer: la torre
di lettori. A ogni piano, ogni tessera guarda *tutte* le altre (per capire
quanta neve c'è sull'orecchio del gatto aiuta guardare anche la tessera con la
coda, dall'altra parte della scheda) e poi rielabora per conto suo quello che
ha visto. All'ultimo piano, ogni tessera riconsegna la propria porzione di
rumore stimato, e le porzioni ricomposte formano la mappa completa:
esattamente ciò che il restauratore deve indicare a ogni passo di pulitura. La
differenza con la U-Net è di principio: le convoluzioni davano la precedenza
ai vicini di casa per costruzione, il Transformer non privilegia nessuno (dove
conviene guardare lo impara da solo, tessera per tessera).

`````

`````{tab} Superiore

L'ingresso è il latente rumoroso $z_t \in \mathbb{R}^{32 \times 32 \times 4}$
(immagini $256 \times 256$ compresse dal VAE con fattore $f = 8$). Il
*patchify* lo suddivide in patch quadrate di lato $p$ e proietta linearmente
ciascuna in un embedding di dimensione $d$: si ottiene una sequenza di
$N = (32/p)^2$ token (usiamo $N$, perché in questo capitolo $T$ è già il
numero di passi di diffusione), a cui si somma un positional encoding
(sinusoidale
bidimensionale, fisso) che ne registra la posizione nella griglia. Segue una
pila di blocchi Transformer del tutto standard (multi-head self-attention più
MLP, con residual e layer normalization, come nel capitolo sui Transformer) e
una testa lineare finale che da ogni token ricostruisce la sua patch di rumore
stimato (nel DiT originale, anche i parametri della covarianza), riassemblata
poi in un tensore della stessa forma dell'ingresso: la rete resta una
$\epsilon_\theta(z_t, t)$, cambia solo ciò che ha dentro.

Il lato $p$ della patch è la manopola del calcolo: con $p = 2$ i token sono
$N = 256$, con $p = 4$ sono 64, con $p = 8$ sono 16. Dimezzare $p$ quadruplica
i token e, con essi, almeno quadruplica il costo di una passata a parità di
parametri, e l'attenzione, quadratica nel numero di token, cresce ancora più
in fretta. Da qui la nomenclatura del paper: quattro taglie (DiT-S, B, L, XL)
per tre patch (/8, /4, /2), dodici modelli che, vedremo tra poco, sono il vero
esperimento del lavoro.

`````

## Il condizionamento entra dalle manopole: adaLN-zero

Manca un pezzo: la rete deve sapere *a che punto della scala* sta lavorando
(il passo $t$) e *che cosa* deve disegnare (nel DiT originale una classe di
ImageNet, nei discendenti un testo). La U-Net sommava l'embedding del tempo
alle feature di ogni blocco; con i token in fila si aprono più strade, e
Peebles e Xie le mettono a confronto. Potrebbero accodare tempo e classe come
token in più della frase, o farli consultare via cross-attention, come fa
Stable Diffusion col prompt. Vince invece la soluzione più discreta,
battezzata **adaLN-zero**: il condizionamento non entra nella conversazione
(regola le manopole).

`````{tab} Elementare

Immagina che la torre di lettori abbia una regia, collegata con l'auricolare a
ogni piano. La regia non suggerisce parole: dà istruzioni di *regolazione*. A
ogni piano dice quanto alzare o abbassare il volume di ciò che passa, come
spostarne il tono, e soprattutto **quanto del lavoro di quel piano deve finire
nel risultato**: da "intervieni a piena forza" a "lascia passare tutto
intatto". Le istruzioni dipendono dal momento: se siamo ai primi passi della
pulitura (quasi tutta neve) o agli ultimi ritocchi, se si sta disegnando un
gatto o un faro.

Il "-zero" del nome è un'astuzia da cantiere: all'inizio
dell'addestramento tutte le manopole di intervento partono da **zero**.
Ogni piano, il primo giorno, lascia passare tutto senza toccare nulla, e
impara strada facendo quanto farsi sentire. Sembra pigrizia, ma è il modo
più stabile di cominciare: nessun piano rovina il lavoro degli altri prima
di aver imparato il proprio.

`````

`````{tab} Superiore

Ricordiamo dal capitolo sui Transformer la layer normalization: normalizza
ogni token a media zero e varianza uno, poi riscala con un guadagno e un bias
appresi, uguali per tutti gli input. L'**adaptive layer norm** (adaLN) rende
guadagno e bias *funzioni del condizionamento*: un piccolo MLP riceve
$c = \mathrm{emb}(t) + \mathrm{emb}(y)$ (embedding sinusoidale del passo più
embedding della classe) e produce, per ciascun sotto-strato di ciascun blocco,
tre vettori $(\beta_c, \gamma_c, \alpha_c)$:

$$
\mathrm{adaLN}(h) = (1 + \gamma_c) \odot \mathrm{LN}(h) + \beta_c,
\qquad
x \leftarrow x + \alpha_c \odot \mathrm{Sottostrato}\big(\mathrm{adaLN}(x)\big),
$$

dove $\mathrm{LN}$ è la normalizzazione *senza* parametri appresi, $\gamma_c$
e $\beta_c$ sono scala e traslazione dettate dal condizionamento, $\odot$ è il
prodotto elemento per elemento e $\alpha_c$ è un *gate* che dosa il contributo
del sotto-strato prima della somma residua. Il suffisso *zero* sta
nell'inizializzazione: l'ultimo strato dell'MLP parte azzerato, quindi
$\gamma_c = \beta_c = \alpha_c = 0$ e ogni blocco all'inizio è l'**identità**;
la rete comincia come un tubo vuoto e i blocchi si accendono gradualmente.
L'idea di modulare le normalizzazioni ha un precedente illustre che
conosciamo: l'AdaIN con cui StyleGAN {cite}`karras2019style` inietta lo stile
nel generatore. Nelle ablazioni del paper, adaLN-zero batte sia i token
in-context sia la cross-attention a parità di calcolo, e costa quasi nulla,
perché produce vettori di manopole, non token aggiuntivi da far partecipare
all'attenzione.

`````

## La qualità scala con il calcolo

E qui arriva il risultato che ha fatto scuola, più del punteggio in sé. Dodici
modelli (quattro taglie per tre dimensioni di patch) ordinati per **Gflops**,
i miliardi di operazioni di una singola passata in avanti. Il FID (il
punteggio di qualità dei campioni già usato in questo capitolo: più basso,
meglio) scende al crescere dei Gflops con una regolarità impressionante, e non
importa *da dove* vengano i Gflops: più strati, più larghezza o più token per
via di patch più piccole, la curva è la stessa. È l'eco esatta delle **leggi
di scala** viste per i modelli di linguaggio {cite}`kaplan2020scaling`,
{cite}`hoffmann2022training`: la qualità non dipende da trovate
architetturali, ma dal calcolo investito, e in modo abbastanza prevedibile da
poter pianificare l'investimento.

In cima alla curva, il modello più grande con le patch più piccole: DiT-XL/2,
675 milioni di parametri, che su ImageNet $256 \times 256$ condizionato alla
classe raggiunge un FID di 2,27, misurato con classifier-free guidance come
per i concorrenti del confronto (meglio di tutti i modelli di diffusione
precedenti, compresi quelli con U-Net di Dhariwal e Nichol
{cite}`dhariwal2021diffusion` e il latent diffusion di Rombach e colleghi
{cite}`rombach2022high`). La U-Net, dunque, non era essenziale: il suo
vantaggio induttivo (sapere in partenza che i pixel vicini contano) si può
comprare con dati e calcolo, e oltre una certa scala il Transformer scala
meglio. È la parabola delle CNN contro i ViT, già vista nel capitolo sui
Transformer, replicata dentro la diffusione.

## Dal fotogramma al minuto: Sora

Perché questa storia conta oltre i benchmark lo dice il video. Il 15 febbraio
2024 OpenAI presenta **Sora**, un modello che da una descrizione testuale
genera video fino a un minuto, accompagnato da un rapporto tecnico dal titolo
programmatico: *Video generation models as world simulators*
{cite}`brooks2024video`. Sul piano architetturale il rapporto è esplicito, e
vale la pena riportare solo ciò che dichiara davvero: Sora «è un diffusion
transformer»; i video vengono compressi da una rete in uno spazio latente e
scomposti in **spacetime patches**, tessere che si estendono nello spazio *e
nel tempo*, usate come token del Transformer; l'addestramento avviene su video
e immagini di durate, risoluzioni e proporzioni variabili; e la qualità dei
campioni migliora «sensibilmente» al crescere del calcolo di addestramento; il
confronto mostrato è tra lo stesso modello a calcolo base, quadruplo e
trentaduplo. È la ricetta DiT, estesa di una dimensione: dove il ViT affettava
un'immagine, qui si affetta un blocco di fotogrammi.

Altrettanto va detto ciò che il rapporto *non* dice: quanti parametri, su
quali dati, con quali dettagli architetturali; è un documento aziendale con
dimostrazioni scelte, non un articolo passato da revisione. E il titolo
rilancia una tesi che il rapporto stesso incrina, documentando bicchieri che
si rovesciano senza rompersi e liquidi che sfidano la gravità: generare video
credibili significa aver *capito* il mondo, o solo averne imparato le
apparenze? È la domanda del capitolo sui **World Model**, più avanti nel
libro, dove i video generativi verranno discussi proprio come candidati
simulatori. Qui registriamo il fatto architetturale: le tessere del ViT,
passate per DiT, sono arrivate al cinema.

## Stable Diffusion 3: due corsie e linee dritte

Il 2024 è anche l'anno in cui la ricetta DiT arriva al text-to-image di
punta. Patrick Esser, Robin Rombach e il resto del gruppo di Stable
Diffusion pubblicano il lavoro dietro **Stable Diffusion 3**
{cite}`esser2024scaling`, presentato a ICML 2024 e premiato tra i migliori
articoli della conferenza. Le novità sono due, e la seconda è la più
profonda.

La prima è l'architettura, battezzata **MM-DiT** (*multimodal DiT*). In Stable
Diffusion il testo era un consulente esterno: trasformato una volta per tutte
dall'encoder di CLIP {cite}`radford2021learning`, veniva solo *consultato*
dalla U-Net via cross-attention (l'informazione fluiva in un senso solo). In
MM-DiT il prompt e il latente dell'immagine sono **due sequenze di token alla
pari**, due corsie con pesi propri (proiezioni e MLP separati per modalità)
che si incontrano in un'attenzione congiunta sulla sequenza concatenata: a
ogni blocco, i token del testo guardano quelli dell'immagine e viceversa. La
famiglia arriva fino a 8 miliardi di parametri e la loss di validazione scende
con la scala in modo regolare (di nuovo le leggi di scala) e predice bene la
qualità giudicata dalle persone. L'impostazione ha fatto scuola: la riprende,
tra gli altri, FLUX (2024), del gruppo di autori originali di Stable
Diffusion.

La seconda novità tocca il cuore probabilistico del capitolo: Stable
Diffusion 3 abbandona la catena di rumore di DDPM per il **rectified
flow**, un caso particolarmente semplice del *flow matching* di Yaron
Lipman e colleghi {cite}`lipman2023flow` (la formulazione con
interpolazioni lineari è dovuta anche a Xingchao Liu e colleghi, sempre
del 2023). L'idea merita le due lenti.

```{figure} ../figures/flow-matching-traiettorie-dritte.svg
:name: fig-traiettorie-dritte
:alt: "Due percorsi che uniscono gli stessi due estremi, il rumore e l'immagine. Il primo, della diffusione classica, è una linea tortuosa che serpeggia e richiede moltissimi passi brevi per essere seguita senza errore. Il secondo, del flow matching, è quasi un segmento rettilineo, e bastano pochi passi lunghi per percorrerlo."
:width: 92%

Stessi estremi, due strade. Su una linea dritta si può camminare a grandi
falcate senza uscire di strada; su una tortuosa no, e i passi devono essere
tanti e piccoli.
```

{numref}`fig-traiettorie-dritte` dice anche perché il guadagno sia in
*campionamento* e non in qualità. Il numero di passi necessari non dipende da
quanto la meta sia lontana ma da quanto il percorso curvi: raddrizzarlo non
cambia dove si arriva, cambia quante volte bisogna fermarsi a chiedere la
direzione.

`````{tab} Elementare

Il restauratore di questo capitolo toglie la neve un velo alla volta:
mille passi lungo un sentiero tortuoso, con tanto di scossoni. L'idea
nuova è quasi insolente: perché seguire un sentiero? Prendi la schermata
di neve e l'immagine finita, traccia una **linea dritta** tra le due, e
insegna alla rete una sola cosa: in ogni punto della linea, *in che
direzione si cammina*.

Facciamo i conti su un pixel. Nella neve vale 0,2; nell'immagine finita vale
0,8. A metà strada vale la media: 0,5. E la direzione di marcia è sempre la
stessa, dall'inizio alla fine: $0{,}8 - 0{,}2 = +0{,}6$ per l'intero cammino.
Su una strada dritta non serve fermarsi mille volte a ricontrollare la mappa:
bastano pochi passi lunghi, perché la direzione non cambia. È questo il motivo
per cui i modelli a rectified flow generano in poche decine di passi ciò che a
DDPM ne costava mille. Con un'onestà da mettere subito a verbale: le strade
*apprese* non escono mai perfettamente dritte (la rete impara una media di
tante linee che si incrociano), quindi qualche controllo lungo il percorso
serve ancora. Ma è la differenza tra un tornante di montagna e una provinciale
con qualche curva.

`````

`````{tab} Superiore

Si fissa una scala continua $t \in [0, 1]$ (dato pulito a $t = 0$, rumore puro
a $t = 1$, coerente con il verso del capitolo) e si collega ogni dato al
rumore con un'**interpolazione lineare**:

$$
x_t = (1 - t)\,x_0 + t\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I),
$$

dove $x_0$ è un dato del training set (in Stable Diffusion 3, un latente)
ed $\epsilon$ il rumore gaussiano. Lungo questo segmento la velocità è
costante: $\mathrm{d}x_t/\mathrm{d}t = \epsilon - x_0$. Il modello
$v_\theta(x_t, t)$ viene addestrato a regredirla:

$$
\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{x_0,\, \epsilon,\, t}
\Big[\, \big\lVert v_\theta(x_t, t) - (\epsilon - x_0) \big\rVert^2 \,\Big],
$$

dove $v_\theta$ è il campo di velocità appreso, con parametri $\theta$, e
l'attesa è su un dato, un rumore e un istante estratti a caso: la stessa
struttura da "regressione con bersaglio noto" della loss di DDPM, con la
velocità al posto del rumore. Per generare si integra l'ODE
$\mathrm{d}x/\mathrm{d}t = v_\theta(x, t)$ da $t = 1$ (rumore) a $t = 0$, per
esempio con passi di Eulero: niente termine stocastico, come già nel
campionatore DDIM incontrato in questo capitolo, ma qui il campo dell'ODE è
appreso direttamente, non ricavato a posteriori da un predittore di rumore.

Perché bastano meno passi? Il campo appreso in un punto è la media delle
velocità di tutte le coppie $(x_0, \epsilon)$ le cui interpolazioni
passano di lì: le traiettorie marginali non sono esattamente rette, ma
risultano molto meno curve di quelle dell'ODE associata alla diffusione
variance-preserving, e l'errore di discretizzazione a parità di passi è
più piccolo. Il flow matching di Lipman e colleghi
{cite}`lipman2023flow` mostra inoltre che la famiglia dei cammini
gaussiani è generale e include i cammini della diffusione come caso
particolare: DDPM diventa un punto in uno spazio di scelte progettuali.
Esser e colleghi {cite}`esser2024scaling` confrontano sistematicamente le
varianti e adottano il rectified flow con un campionamento di $t$
concentrato sugli istanti intermedi, i più difficili: in pratica Stable
Diffusion 3 genera in poche decine di passi.

`````

## Un DiT in miniatura

Come per la spirale di punti con cui abbiamo smontato DDPM, il modo migliore
di fissare l'architettura è costruirla in piccolo. Il codice che segue è un
DiT completo ma in miniatura: patchify di un latente fittizio, blocchi con
attenzione e MLP modulati da adaLN-zero, ricomposizione delle patch. Non c'è
addestramento (servirebbero i dati e le ore di GPU) ma ogni forma è
verificabile a mano, e il ciclo di addestramento sarebbe *lo stesso* visto per
DDPM: cambia solo la rete interrogata.

```python
import math
import torch
from torch import nn

torch.manual_seed(0)

def embedding_tempo(t, dim=128):
    """Embedding sinusoidale del passo t: da (B,) a (B, dim)."""
    freq = torch.exp(-math.log(10000.0) * torch.arange(dim // 2) / (dim // 2))
    ang = t.float().unsqueeze(1) * freq.unsqueeze(0)     # (B, dim/2)
    return torch.cat([ang.sin(), ang.cos()], dim=1)      # (B, dim)

class BloccoDiT(nn.Module):
    """Attenzione + MLP, con modulazione adaLN-zero dal condizionamento."""
    def __init__(self, d=128, teste=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)  # LN "nuda"
        self.attn = nn.MultiheadAttention(d, teste, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))
        # dal condizionamento: shift, scale e gate per i due sotto-strati
        self.manopole = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.manopole.weight)   # adaLN-ZERO: blocco = identita'
        nn.init.zeros_(self.manopole.bias)

    def forward(self, x, c):
        # x: (B, N, d) token del latente; c: (B, d) tempo + classe
        b1, g1, a1, b2, g2, a2 = self.manopole(c).chunk(6, dim=1)  # 6 x (B, d)
        h = self.norm1(x) * (1 + g1.unsqueeze(1)) + b1.unsqueeze(1)
        att, _ = self.attn(h, h, h, need_weights=False)
        x = x + a1.unsqueeze(1) * att            # gate a1: vale 0 all'inizio
        h = self.norm2(x) * (1 + g2.unsqueeze(1)) + b2.unsqueeze(1)
        x = x + a2.unsqueeze(1) * self.mlp(h)    # gate a2, idem
        return x

class MiniDiT(nn.Module):
    """DiT minimale: patchify, blocchi Transformer, patch di rumore in uscita."""
    def __init__(self, canali=4, lato=32, patch=2, d=128, blocchi=4, classi=10):
        super().__init__()
        self.canali, self.lato, self.patch = canali, lato, patch
        n_token = (lato // patch) ** 2                       # (32/2)^2 = 256
        self.patchify = nn.Conv2d(canali, d, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, n_token, d))  # posizioni apprese
        self.emb_classe = nn.Embedding(classi, d)
        self.blocchi = nn.ModuleList([BloccoDiT(d) for _ in range(blocchi)])
        self.finale = nn.Linear(d, patch * patch * canali)   # token -> sua patch

    def forward(self, z, t, y):
        # z: (B, 4, 32, 32) latente rumoroso; t: (B,) passo; y: (B,) classe
        x = self.patchify(z)                     # (B, d, 16, 16)
        x = x.flatten(2).transpose(1, 2)         # (B, 256, d): i token
        x = x + self.pos
        c = embedding_tempo(t, x.shape[-1]) + self.emb_classe(y)   # (B, d)
        for blocco in self.blocchi:
            x = blocco(x, c)
        x = self.finale(x)                       # (B, 256, patch*patch*4)
        # ricompone le patch: l'uscita ha la stessa forma dell'ingresso
        B, g, p, C = z.shape[0], self.lato // self.patch, self.patch, self.canali
        x = x.view(B, g, g, p, p, C)             # (B, 16, 16, 2, 2, 4)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, self.lato, self.lato)
        return x                                 # (B, 4, 32, 32): rumore stimato

modello = MiniDiT()
z = torch.randn(2, 4, 32, 32)      # due latenti fittizi, come quelli del VAE
t = torch.randint(0, 1000, (2,))   # un passo di rumore per ciascuno
y = torch.randint(0, 10, (2,))     # una classe per ciascuno
print(modello(z, t, y).shape)      # torch.Size([2, 4, 32, 32])
print(sum(p.numel() for p in modello.parameters()))  # 1225616: ~1.2 milioni
```

Vale la pena di fare a mano il *desk-check* dell'inizializzazione: con i pesi
delle `manopole` a zero, ogni `g` e `b` vale zero e ogni gate `a` pure, quindi
ogni blocco restituisce `x` invariato; la rete appena costruita è una catena
di identità, come promesso da adaLN-zero. Il DiT vero differisce nei numeri
(28 blocchi e $d = 1152$ nella taglia XL, contro i nostri 4 e 128),
nell'uscita (predice anche la covarianza) e nel positional encoding, ma non
nella logica: quella sta tutta qui.

## Il conto, e la lezione

Chiudiamo il capitolo con la stessa onestà con cui l'abbiamo aperto.
Addestrare questi modelli è fuori dalla portata individuale, e la direzione è
quella di un rincaro: già lo Stable Diffusion del 2022 era costato centinaia
di migliaia di dollari di GPU, i suoi successori a miliardi di parametri
costano multipli non dichiarati, e su Sora OpenAI non pubblica né costi né
dimensioni. La scala è diventata un ingrediente della ricetta (le curve
FID–Gflops lo dicono senza giri di parole) e la scala si paga.
L'**inferenza**, però, è un'altra storia: i pesi di DiT e di Stable Diffusion
3 si scaricano, girano su una scheda video da videogiochi, e il rectified flow
ha reso la generazione più veloce, non più lenta. L'asimmetria vista per
Stable Diffusion (addestrare è per pochi, usare è per molti) si è accentuata
in entrambe le direzioni.

E la lezione finale è quella che questo libro ripete dal primo capitolo. Nel
2015 la diffusione nasce da un'analogia termodinamica; nel 2024 genera un
minuto di video da una frase. In mezzo, nessun colpo di genio isolato: un
autoencoder variazionale del 2014 nato per comprimere, una U-Net del 2015 nata
per i microscopi {cite}`ronneberger2015u`, l'attenzione nata nel 2015 per
tradurre e promossa a protagonista nel 2017 {cite}`vaswani2017attention`, le
patch di un ViT del 2021 nato per classificare {cite}`dosovitskiy2021image`
(mattoni progettati per tutt'altro, ricombinati con pazienza da gruppi diversi
in anni diversi). Chi vi racconta questa storia come una successione di
rivoluzioni improvvise ve la racconta male: è una storia di ricombinazioni, e
il prossimo mattone, con ogni probabilità, è già su uno scaffale che abbiamo
attraversato senza fermarci.

```{admonition} Da ricordare
:class: important
- **DiT** {cite}`peebles2023scalable` sostituisce la U-Net con un
  Transformer: il latente si affetta in patch-token come nel ViT
  {cite}`dosovitskiy2021image`, l'attenzione rimpiazza le convoluzioni; il
  VAE e la diffusione nel latente restano quelli di
  {cite}`rombach2022high`.
- Il condizionamento su $t$ e classe entra via **adaLN-zero**: scala,
  traslazione e gate della layer norm generati da un MLP del condizionamento,
  con inizializzazione a zero (ogni blocco parte come identità).
- Risultato chiave: il **FID scende con i Gflops** in modo regolare, comunque
  li si spenda, le leggi di scala {cite}`kaplan2020scaling` arrivano alla
  diffusione; DiT-XL/2 (675 milioni di parametri) tocca FID 2,27, con
  classifier-free guidance, su ImageNet $256 \times 256$.
- **Sora** {cite}`brooks2024video` dichiara un diffusion transformer su
  *spacetime patches* di video compressi, con qualità che cresce col
  calcolo: la ricetta DiT estesa al tempo. Se ciò faccia dei video
  generativi dei "simulatori di mondo" è la domanda del capitolo sui World
  Model.
- **Stable Diffusion 3** {cite}`esser2024scaling` adotta l'MM-DiT (testo e
  immagine come due flussi di token alla pari) e passa al **rectified
  flow** {cite}`lipman2023flow`: interpolazioni lineari dato–rumore, una
  velocità appresa per regressione, generazione integrando un'ODE in poche
  decine di passi.
- Addestrare resta un affare da data center; **usare** no: i pesi aperti e
  il latente compresso tengono l'inferenza alla portata di una GPU
  domestica. La storia del capitolo è ricombinazione di mattoni noti, non
  una rivoluzione improvvisa.
```
