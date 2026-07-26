# Mondi in miniatura: imparare sognando

C'è un esperimento, nel 2018, che sembra uscito da un racconto più che da un
laboratorio di machine learning. David Ha e Jürgen Schmidhuber prendono un
livello di *Doom* (lo storico sparatutto) in cui bisogna schivare palle di
fuoco, e ci allenano un agente che il gioco vero, durante l'allenamento, non
tocca mai: guarda migliaia di partite giocate a caso, si costruisce una copia
compressa e approssimativa del gioco dentro le proprie reti neurali, e si
allena esclusivamente lì dentro; nel proprio «sogno», l'hanno chiamato proprio
così. Riportato nel gioco autentico, schiva le palle di fuoco ben oltre la
soglia che definisce il livello «risolto» {cite}`ha2018world`.

L'articolo ha un titolo di due parole, *World Models* (presentato a NeurIPS
2018 come *Recurrent World Models Facilitate Policy Evolution*) e contiene un
secondo primato: sul gioco di guida *CarRacing-v0*, una pista vista dall'alto
generata a caso a ogni partita, lo stesso schema è il primo sistema dichiarato
in grado di *risolvere* il compito: punteggio medio 906 su 100 piste, contro
la soglia richiesta di 900.

L'idea di fondo è antica e molto umana. Quando attraversi la strada non
ragioni sui fotoni che colpiscono la retina: consulti un modello mentale del
traffico («se quell'auto mantiene la velocità, tra tre secondi è qui») e provi
le azioni *nel modello* prima che nel mondo. Un **world model** è questo: una
copia interna, compressa e imparata, dell'ambiente, dentro cui pensare costa
poco e sbagliare non fa male. In questa sezione smontiamo la ricetta di Ha e
Schmidhuber, seguiamo la discendenza fino a DreamerV3 e ai diamanti di
Minecraft, e ricostruiamo i tre moduli in PyTorch.

## Tre lettere per un pilota: V, M e C

La ricetta ha tre ingredienti dai nomi minimalisti: **V** come *visione*,
**M** come *memoria*, **C** come *controller*. V comprime ogni fotogramma in
un piccolo codice; M, una rete ricorrente, impara come quel codice evolve in
risposta alle azioni; C (minuscolo) legge codice e memoria e decide. La
{numref}`fig-world-model-vmc` mostra il giro completo: l'azione di C torna
all'ambiente, che produce il fotogramma successivo. E mostra l'anello
tratteggiato che rende speciale l'architettura: M può alimentare se stesso,
sostituendosi all'ambiente. È il circuito del sogno, e ci arriviamo tra poco.

```{figure} ../figures/world-model-vmc.svg
:name: fig-world-model-vmc
:alt: "Pipeline del world model: l'ambiente produce un fotogramma che V comprime in un codice z di 32 numeri; M, una rete ricorrente, predice il prossimo codice; C, un controller lineare da 867 parametri, sceglie l'azione che torna all'ambiente. Un anello tratteggiato sopra M indica il sogno, in cui la predizione di M rientra come suo input."
:width: 100%

I tre moduli di Ha e Schmidhuber: nel gioco vero il ciclo passa
dall'ambiente; nel sogno l'anello tratteggiato lo sostituisce.
```

### V come Visione: il mondo in trentadue numeri

Un fotogramma di *CarRacing*, ridotto a $64 \times 64$ pixel a colori, sono
$64 \times 64 \times 3 = 12\,288$ numeri. Troppi, e quasi tutti ridondanti:
alla guida non servono i singoli fili d'erba, serve sapere dove curva la
strada e dove sta l'auto. V è un **autoencoder variazionale** (VAE) (lo
abbiamo conosciuto nel capitolo sui Modelli di Diffusione
{cite}`kingma2014auto`) addestrato a comprimere ogni fotogramma in un codice
$z$ di appena 32 numeri: quasi quattrocento volte meno.

`````{tab} Elementare

Immagina di dover descrivere la schermata di gioco a un amico al telefono. Non
gli detti i 12.288 puntini colorati uno per uno: dici «curva a sinistra, auto
al centro, erba sui bordi» (poche informazioni, quelle giuste). Il VAE fa lo
stesso, ma nessuno gli ha suggerito *quali* informazioni tenere: le ha scelte
da solo, perché il suo allenamento è un gioco di andata e ritorno (comprimi il
fotogramma in 32 numeri, poi prova a ridisegnarlo dal solo riassunto). Se il
disegno somiglia all'originale, il riassunto conteneva l'essenziale; se non
somiglia, quei 32 numeri vanno usati meglio. Come al telefono: se dalla tua
descrizione l'amico disegna una scena quasi uguale, la descrizione era buona.

`````

`````{tab} Superiore

L'encoder del VAE mappa il fotogramma $x \in \mathbb{R}^{64 \times 64
\times 3}$ in una distribuzione gaussiana sullo spazio latente:

$$
q_\phi(z \mid x) = \mathcal{N}\!\big(z;\, \mu_\phi(x),\,
\mathrm{diag}(\sigma_\phi^2(x))\big),
\qquad
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon,
\quad \epsilon \sim \mathcal{N}(0, I),
$$

dove $\mu_\phi(x)$ e $\sigma_\phi(x)$ sono media e deviazione standard
prodotte da una pila di convoluzioni con parametri $\phi$,
$z \in \mathbb{R}^{32}$ è il codice latente e la seconda uguaglianza è il
*trucco della riparametrizzazione*, che rende campionabile e derivabile il
passaggio. L'addestramento massimizza l'ELBO, ricostruzione più
regolarizzazione KL verso la prior $\mathcal{N}(0, I)$, come visto nel
capitolo sui Modelli di Diffusione; qui non lo rideriviamo. Nel paper V viene
addestrato *per primo*, in modo non supervisionato, su fotogrammi raccolti da
una policy casuale; il decoder serve solo in addestramento. Su *CarRacing* V
pesa circa 4,3 milioni di parametri.

`````

### M come Memoria: la fisica del gioco in una RNN

Un fotogramma compresso è una fotografia, non un film: non dice cosa
succederà. Il secondo modulo impara la **dinamica**: dato il codice di adesso
e l'azione scelta, quale sarà il codice di poi? M è una LSTM, la rete
ricorrente con i *gate* che decidono cosa ricordare e cosa dimenticare,
incontrata nel capitolo sul Natural Language Processing
{cite}`hochreiter1997long`. Solo che qui la «frase» da proseguire non è fatta
di parole ma di codici $z$: M vive nel piccolo mondo dei 32 numeri, senza mai
toccare i pixel, perciò è veloce ed economica.

`````{tab} Elementare

Pensa a un'amica che ha giocato mille partite. Le descrivi la situazione in
una frase («sono a metà curva, sto accelerando») e lei ti dice come prosegue:
«esci largo verso l'erba». Non le serve *vedere* lo schermo: le basta il
riassunto, perché la fisica del gioco ce l'ha in testa. Una sfumatura conta:
l'amica onesta non risponde con una certezza ma con un ventaglio, «quasi
sempre esci largo; ogni tanto la tieni». M è costruita così: per ogni
situazione prevede le diverse continuazioni possibili, ciascuna con la sua
probabilità, come le previsioni del tempo che dicono «pioggia al 70%» invece
di giurare sul sole. Il futuro di un gioco (e del mondo) non è mai scritto del
tutto, e un modello che finge di saperlo mente.

`````

`````{tab} Superiore

M è una **MDN-RNN**: una LSTM (256 unità nascoste su *CarRacing*) la cui
testa di uscita è una *mixture density network*, l'idea proposta da
Christopher Bishop nel 1994 per far predire a una rete un'intera
distribuzione anziché un valore. Il modello stima

$$
P(z_{t+1} \mid a_t, z_t, h_t)
= \sum_{k=1}^{K} \pi_k(h_t)\,
\mathcal{N}\!\big(z_{t+1};\, \mu_k(h_t),\, \sigma_k^2(h_t)\big),
$$

dove $h_t$ è lo stato nascosto della LSTM dopo aver letto la storia fino al
passo $t$, $a_t$ è l'azione, $K = 5$ è il numero di componenti della miscela,
i pesi $\pi_k$ (una softmax, sommano a 1) dicono quanto è probabile ciascuna
«versione del futuro», e $\mu_k$, $\sigma_k$ ne danno centro e incertezza.
L'addestramento minimizza la log-verosimiglianza negativa dei codici osservati
nelle partite raccolte; su *Doom*, M predice anche la probabilità che
l'episodio finisca (l'agente muore). Su *CarRacing* M pesa in tutto poco più
di 400.000 parametri, e la miscela è la manopola con cui, tra poco, regoleremo
quanto il «mondo interno» è capriccioso.

`````

### C come Controller: il pilota minimalista

E qui la sorpresa: dopo un compressore da milioni di parametri e una memoria
da centinaia di migliaia, il modulo che *decide* è una moltiplicazione di
matrice:

$$
a_t = W_c\,[z_t ; h_t] + b_c,
$$

dove $[z_t ; h_t]$ è la concatenazione del codice visivo e dello stato della
memoria ($32 + 256 = 288$ numeri), $W_c$ è una matrice $3 \times 288$ e $b_c$
tre bias, uno per azione: sterzo, acceleratore, freno. Totale: **867
parametri**. Non è un vezzo: è la tesi dell'articolo. Se V e M hanno digerito
davvero il mondo, per agire bene basta un riflesso lineare: tutta
l'intelligenza sta nel modello, non nel controllore. E un controllore così
piccolo si può addestrare senza gradiente: gli autori usano CMA-ES, una
strategia evolutiva che a ogni generazione fa «gareggiare» 64 varianti del
controllore e ricombina le migliori. Con 867 numeri da scegliere, l'evoluzione
basta e avanza.

## Allenarsi nel sogno

Fin qui M ha un ruolo ancillare: aiutare C fornendogli un po' di futuro
anticipato. Ma guardate l'anello tratteggiato della
{numref}`fig-world-model-vmc`. M predice il prossimo codice $z$; e se quel
codice, invece di confrontarlo con la realtà, lo ridessimo in pasto a M come
input del passo successivo? Il modello comincia a raccontarsi il gioco da
solo, un passo dopo l'altro: niente più ambiente, niente pixel, solo codici
che generano codici. Ha e Schmidhuber lo chiamano *dream*, e l'esperimento su
*Doom* (lo scenario *Take Cover* di VizDoom, la versione del gioco usata nella
ricerca) è tutto qui: C viene addestrato **esclusivamente** dentro il sogno e
poi trasferito, senza ritocchi, nel gioco vero.

`````{tab} Elementare

È il pilota che la sera prima della gara ripassa il circuito a occhi chiusi,
curva per curva, i piloti veri lo fanno davvero: costa zero benzina e zero
incidenti. Ma c'è un tallone d'Achille: se nella tua testa una curva è più
dolce che in pista, impari una traiettoria che domani ti manda nella ghiaia.
All'agente di Ha e Schmidhuber successe qualcosa di più subdolo: dentro il
sogno scoprì dei *trucchi*. Trovò modi di muoversi per cui i mostri,
semplicemente, non sparavano quasi mai, e a volte riusciva perfino a far
svanire le palle di fuoco. Stava barando non al gioco ma al *proprio sogno*,
sfruttandone i difetti, come uno studente che si prepara all'esame
inventandosi da solo domande facili. Punteggi splendidi nel mondo immaginato,
figuraccia in quello vero. Il rimedio è elegante: rendere il sogno *più
capriccioso* del gioco reale, così che i trucchi smettano di funzionare.
Allenato in quel simulatore cattivo, l'agente trovò il gioco vero quasi
riposante: vi sopravvisse in media *più a lungo* che nel proprio sogno.

`````

`````{tab} Superiore

Un *rollout* nel modello è la catena

$$
a_t = C(z_t, h_t), \qquad
z_{t+1} \sim P_\tau(\,\cdot \mid a_t, z_t, h_t),
$$

dove il campionamento dalla miscela avviene a **temperatura** $\tau$: un
parametro che gonfia ($\tau > 1$) o spegne ($\tau \to 0$) l'incertezza della
distribuzione predetta. Il problema strutturale è che C viene ottimizzato
*contro M*, non contro l'ambiente: ogni errore sistematico del modello diventa
una risorsa da sfruttare, e la ricerca di policy trova politiche avversarie al
proprio stesso mondo interno; nel paper, rollout in cui i mostri non sparano
mai, o in cui certi movimenti «estinguono» le palle di fuoco. Con $\tau$ basso
il sogno è docile e l'inganno prospera: punteggi onirici altissimi, transfer
disastroso. La temperatura scelta, $\tau = 1{,}15$, rende il mondo interno più
stocastico di quello vero e funziona da regolarizzazione: punteggio medio
$918 \pm 546$ nel sogno e $1092 \pm 556$ nell'ambiente reale (soglia di
risoluzione: 750) (l'agente va *meglio* nella realtà che nella propria
immaginazione). Il limite di fondo però resta: il disallineamento tra $P_\tau$
e la vera dinamica non si annulla, e gli errori si accumulano lungo il
rollout; ragione per cui i sogni utili sono brevi.

`````

## Dai sogni ai diamanti: la linea Dreamer

*World Models* era una dimostrazione su due videogiochi. Trasformarla in un
metodo generale è stato in buona parte il lavoro di Danijar Hafner e colleghi:
Dreamer (2020) impara i comportamenti interamente nell'**immaginazione
latente**, con attore e critico addestrati su rollout sognati; DreamerV2
(2021) è il primo agente a livello umano sul benchmark Atari imparando dentro
un world model; DreamerV3, pubblicato su *Nature* nel 2025
{cite}`hafner2023mastering`, affronta più di 150 compiti (robot simulati,
Atari, navigazione 3D) con la **stessa identica configurazione**, senza
ritocchi per dominio. Il risultato simbolo: applicato così com'è a Minecraft,
è il primo algoritmo a raccogliere **diamanti** partendo da zero, senza
dimostrazioni umane né curricula. Arrivarci richiede una catena lunghissima di
sotto-obiettivi (legno, banco da lavoro, picconi via via migliori, ferro da
fondere, scavi in profondità) con ricompense rarissime lungo il cammino: il
tipo di compito su cui, come abbiamo visto con *Montezuma's Revenge*, il DQN
si arena.

È qui il raccordo con il capitolo sul Deep Reinforcement Learning: i world
model sono la risposta **model-based** alla fame di campioni dei metodi
*model-free*.

`````{tab} Elementare

Ricordate il conto pagato in apertura di capitolo: al DQN servono decine di
milioni di fotogrammi per imparare un gioco Atari dove a un umano bastano
minuti {cite}`mnih2015human`. Ogni esperienza serve solo ad aggiustare di un
soffio le valutazioni, come uno studente che di un'intera lezione trattiene
una riga. Un world model spreme la stessa esperienza molto di più: ogni
partita vera migliora la copia interna del gioco, e dentro la copia ci si
allena quanto si vuole, al solo costo dell'elettricità. L'idea, in piccolo, ha
più di trent'anni: si chiama Dyna, l'architettura con cui Richard Sutton nel
1991 faceva alternare a un agente mosse vere e mosse «ripassate» in un
modellino imparato del labirinto {cite}`sutton1991dyna` (un antenato a caselle
dei sogni di Dreamer).

`````

`````{tab} Superiore

Un metodo *model-free* come il DQN stima direttamente valori o policy
dall'esperienza; un metodo *model-based* impara anche un modello della
dinamica $p(s_{t+1} \mid s_t, a_t)$ e lo usa per generare transizioni
sintetiche. Dyna {cite}`sutton1991dyna` è lo schema capostipite: gli
aggiornamenti di $Q$ attingono sia da transizioni reali sia da transizioni
simulate dal modello appreso, mescolando apprendimento e pianificazione. I
Dreamer ne sono l'erede profondo: un modello ricorrente dello stato (RSSM),
con una componente deterministica e una stocastica, apprende la dinamica nello
spazio latente; attore e critico vengono addestrati per retropropagazione
attraverso rollout immaginati con orizzonte breve (una quindicina di passi)
per contenere l'accumulo degli errori del modello; DreamerV3 aggiunge
normalizzazioni robuste (osservazioni, ricompense, ritorni) che rendono gli
stessi iperparametri validi su domini radicalmente diversi
{cite}`hafner2023mastering`. Il guadagno è l'efficienza campionaria; il tetto
è la qualità del modello: la policy è buona quanto il sogno in cui è
cresciuta, e su dinamiche caotiche o eventi rari i modelli restano il punto
debole. Il confronto con i metodi model-free, competitivi quando i campioni
costano poco, è tutt'altro che chiuso.

`````

Una nota di prospettiva: oggi «world model» è anche un'etichetta di moda per
i grandi modelli generativi di video, promossi a simulatori del mondo
fisico. La parentela concettuale c'è; la capacità di sostenere un intero
ciclo di apprendimento dentro il modello, come qui, resta materia di ricerca.

## I tre moduli in PyTorch

Chiudiamo con lo scheletro di V, M e C: pochi tensori, forme esplicite nei
commenti. Manca il training (il decoder e la loss del VAE, la
log-verosimiglianza della miscela per M, CMA-ES per C) ma il flusso dei dati è
quello vero.

```python
import torch
from torch import nn

class EncoderVAE(nn.Module):
    """V: comprime un fotogramma 3x64x64 in un codice z di 32 numeri."""
    def __init__(self, dim_z=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),    # -> (32, 31, 31)
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),   # -> (64, 14, 14)
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),  # -> (128, 6, 6)
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(), # -> (256, 2, 2)
            nn.Flatten(),                                # -> 1024
        )
        self.mu = nn.Linear(1024, dim_z)       # media del codice
        self.logvar = nn.Linear(1024, dim_z)   # log-varianza del codice

    def forward(self, x):                      # x: (B, 3, 64, 64)
        h = self.conv(x)                       # (B, 1024)
        mu, logvar = self.mu(h), self.logvar(h)
        eps = torch.randn_like(mu)             # riparametrizzazione
        return mu + torch.exp(0.5 * logvar) * eps   # z: (B, 32)

class ModelloRNN(nn.Module):
    """M: dato il codice e l'azione, predice il codice del passo dopo.
    (Versione deterministica; il paper usa una miscela di gaussiane.)"""
    def __init__(self, dim_z=32, dim_a=3, dim_h=256):
        super().__init__()
        self.lstm = nn.LSTM(dim_z + dim_a, dim_h, batch_first=True)
        self.testa = nn.Linear(dim_h, dim_z)   # media del prossimo z

    def forward(self, z, a, stato=None):       # z: (B, T, 32), a: (B, T, 3)
        ingresso = torch.cat([z, a], dim=-1)   # (B, T, 35)
        h, stato = self.lstm(ingresso, stato)  # h: (B, T, 256)
        return self.testa(h), stato            # z predetto: (B, T, 32)

class Controller(nn.Module):
    """C: policy lineare da codice e memoria all'azione."""
    def __init__(self, dim_z=32, dim_h=256, dim_a=3):
        super().__init__()
        self.lineare = nn.Linear(dim_z + dim_h, dim_a)   # 288*3+3 = 867

    def forward(self, z, h):                   # z: (B, 32), h: (B, 256)
        # azioni in [-1, 1]; gas e freno andrebbero poi riportati in [0, 1]
        return torch.tanh(self.lineare(torch.cat([z, h], dim=-1)))
```

E questo è il circuito del sogno: dieci passi interamente nello spazio dei
codici, con M che fa da ambiente a se stesso. Le azioni qui sono casuali;
nell'addestramento vero le sceglierebbe C, e il punteggio sognato
guiderebbe l'evoluzione dei suoi 867 parametri.

```python
V, M, C = EncoderVAE(), ModelloRNN(), Controller()
print(sum(p.numel() for p in C.parameters()))   # 867: il pilota è minuscolo

x = torch.rand(1, 3, 64, 64)     # un fotogramma finto: batch 1, RGB, 64x64
z = V(x).unsqueeze(1)            # (1, 1, 32): il codice, come sequenza di 1 passo
stato = None                     # memoria (h, c) della LSTM, vuota all'inizio

for t in range(10):              # dieci passi di sogno: nessun ambiente
    a = torch.rand(1, 1, 3) * 2 - 1        # azione casuale in [-1, 1]
    z, stato = M(z, a, stato)              # il codice sognato: (1, 1, 32)

# il controller legge codice e memoria e restituisce i tre comandi
h = stato[0].squeeze(0)          # stato nascosto della LSTM: (1, 256)
comandi = C(z.squeeze(1), h)     # (1, 3): sterzo, acceleratore, freno
```

```{admonition} Da ricordare
:class: important
- Un **world model** è una copia interna, compressa e imparata,
  dell'ambiente: pensare e sbagliare lì dentro costa quasi nulla.
- La ricetta di Ha e Schmidhuber (2018): **V**, un VAE che comprime il
  fotogramma in 32 numeri; **M**, una LSTM che predice la *distribuzione*
  del prossimo codice; **C**, una policy lineare da 867 parametri.
  L'intelligenza sta nel modello, non nel controllore.
- Il **sogno** è un rollout in cui M alimenta se stesso. Rischio: sfruttarne
  i difetti (i mostri che «non sparano mai»); rimedio: alzare la
  **temperatura** $\tau$, rendendo il sogno più incerto del mondo vero.
- Su *VizDoom: Take Cover* l'agente addestrato solo nel sogno supera nel
  gioco reale la soglia di risoluzione (1092 contro 750).
- La linea **Dreamer** porta l'idea a maturità: DreamerV3 (*Nature*, 2025)
  impara nell'immaginazione latente, usa gli stessi iperparametri su più di
  150 compiti e trova i diamanti in Minecraft senza dimostrazioni umane.
- È la risposta **model-based** alla fame di campioni del DQN, con un
  antenato preciso: Dyna di Sutton (1991). Il limite resta la qualità del
  modello: la policy è buona quanto il sogno in cui è cresciuta.
```
