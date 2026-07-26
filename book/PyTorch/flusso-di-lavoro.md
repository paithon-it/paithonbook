# Il flusso di lavoro: dal problema al modello

Chiedi a chi lavora con le reti neurali qual è la parte difficile, e quasi
nessuno risponderà "scrivere il modello". Il modello sono venti righe, e le
sezioni precedenti le hanno già mostrate tutte: tensori, `nn.Module`, il loop
dei cinque passi. La parte difficile è **l'ordine delle mosse**: sapere che
cosa si guarda per primo, che cosa si cambia quando il numero non sale, e
quando fermarsi. È un mestiere, e come tutti i mestieri ha una sua sequenza
fissa che si impara una volta e poi si ripete su qualunque problema: che si
tratti di prevedere il prezzo di una casa o di riconoscere un tumore in una
lastra, le stazioni da attraversare sono sempre quelle.

## Sei stazioni, sempre le stesse

La {numref}`fig-flusso-pytorch` le mette in fila. Vale la pena guardarla una
volta per intero: è la mappa non solo di questa sezione, ma di tutto quello che
si fa quando si addestra un modello.

```{figure} ../figures/flusso-di-lavoro-pytorch.svg
:name: fig-flusso-pytorch
:alt: Sei riquadri numerati collegati in anello (problema, dati, modello, addestramento, valutazione, uso) con una freccia tratteggiata che dalla valutazione torna al modello e segnala il ciclo di miglioramento.
:width: 90%

Il flusso di lavoro di un progetto PyTorch. Il percorso si attraversa una
volta in linea retta e poi decine di volte in circolo tra le stazioni 3 e 5:
è lì, non nella scrittura del modello, che si consuma il tempo.
```

Le prime due stazioni non hanno niente a che vedere con PyTorch e sono quelle
che decidono l'esito: capire **che cosa si vuole predire** e **da quali dati**.
Le tre centrali sono il ciclo vero e proprio. L'ultima è quella che quasi
sempre si dimentica di pianificare, e che il
[capitolo sull'MLOps](../MLOps/overview.md) riprende per esteso.

`````{tab} Elementare
È lo stesso ordine con cui si prepara una ricetta nuova. Prima decidi che
piatto vuoi (la stazione 1), poi controlli che cosa hai in dispensa (2). Solo
allora scegli il procedimento (3) e cucini (4). Poi (ed è il passaggio che
distingue chi cucina bene) **assaggi** (5), e l'assaggio non lo fai sul
cucchiaio che hai già leccato: usi una porzione che non hai ancora toccato,
altrimenti ti convinci che sia buono solo perché lo hai fatto tu. Se manca
sale, torni indietro e cambi *una cosa sola*, altrimenti al secondo assaggio
non saprai se è merito del sale o del tempo di cottura. Alla fine scrivi la
ricetta su un foglio (6), perché fra un mese non te la ricorderai.
`````

`````{tab} Superiore
Formalizzato: si fissa uno spazio di ipotesi $\mathcal{F}$ (l'architettura),
una funzione di perdita $\mathcal{L}$ e un algoritmo di ottimizzazione; si
stima $\theta$ minimizzando il rischio empirico sul training set; si misura il
rischio su un campione indipendente per stimare la generalizzazione. Le
stazioni 3–5 sono un ciclo di ricerca su iperparametri e architettura, guidato
dalla metrica di **validazione**, e ogni decisione presa guardando quel numero
lo consuma un po', perché il set di validazione diventa a poco a poco parte
dell'addestramento. Per questo il test set si tocca **una volta sola**, alla
fine: è l'unica stima onesta che rimane. Il capitolo sul machine learning
tratta per esteso questa contabilità in [overfitting e
validazione](../MachineLearning/overfitting-validazione.md).
`````

## Un problema di cui conosciamo già la risposta

Il modo migliore per imparare un flusso di lavoro è percorrerlo su un problema
*truccato*: uno di cui conosciamo la soluzione in anticipo, così da poter
verificare a colpo d'occhio se il modello l'ha trovata. Costruiamo dei dati
con una formula nota (una retta di pendenza $0{,}7$ e intercetta $0{,}3$) e
poi buttiamo via la formula, lasciando al modello solo i punti.

```python
import torch
from torch import nn

torch.manual_seed(42)          # stessi numeri casuali a ogni esecuzione

# I parametri "veri": il modello dovrà ritrovarli da solo, senza mai vederli.
peso_vero, bias_vero = 0.7, 0.3

X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)   # 50 punti, shape (50, 1)
y = peso_vero * X + bias_vero                   # shape (50, 1)

taglio = int(0.8 * len(X))                      # 80% per addestrare, 20% per il test
X_train, y_train = X[:taglio], y[:taglio]       # (40, 1)
X_test,  y_test  = X[taglio:], y[taglio:]       # (10, 1)
```

Due dettagli meritano attenzione, perché tornano in ogni progetto.
`unsqueeze(dim=1)` aggiunge l'asse delle *feature*: `nn.Linear` vuole una
matrice $(N, d)$, non un vettore di $N$ numeri, e dimenticarlo è il primo dei
[tre errori più comuni](errori-comuni.md). E `manual_seed` fissa il
generatore di numeri casuali: senza, due esecuzioni dello stesso codice danno
risultati diversi e non si capisce più se un miglioramento viene dalla
modifica o dalla fortuna.

Il modello è la retta più semplice che si possa scrivere: un `nn.Linear` con
un ingresso e un'uscita, cioè esattamente due numeri da imparare.

```python
class RegressioneLineare(nn.Module):
    def __init__(self):
        super().__init__()
        self.strato = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.strato(x)

modello = RegressioneLineare()
print(modello.state_dict())   # peso e bias, per ora casuali
```

Adesso il ciclo, con una sola aggiunta rispetto alla sezione precedente: ogni
tanto ci si ferma a misurare anche sui dati messi da parte.

```python
criterio = nn.L1Loss()                                     # errore assoluto medio
ottimizzatore = torch.optim.SGD(modello.parameters(), lr=0.01)

for epoca in range(1000):
    modello.train()
    y_pred = modello(X_train)
    perdita = criterio(y_pred, y_train)
    ottimizzatore.zero_grad()
    perdita.backward()
    ottimizzatore.step()

    if epoca % 200 == 0:                                   # il "termometro"
        modello.eval()
        with torch.no_grad():
            perdita_test = criterio(modello(X_test), y_test)
        print(f"epoca {epoca:>4} | train {perdita.item():.4f} "
              f"| test {perdita_test.item():.4f}")

print(modello.state_dict())
```

Alla fine `state_dict()` stampa due numeri molto vicini a $0{,}7$ e $0{,}3$:
non identici, perché la discesa del gradiente si ferma quando è *abbastanza*
vicina, ma le prime due cifre tornano. È una verifica che nella maggior parte
dei problemi veri non potremo mai fare, e proprio per questo vale la pena
farla almeno una volta: qui sappiamo con certezza che la macchina funziona.

`````{tab} Elementare
La `L1Loss` è la scelta più leggibile che ci sia: è la **distanza media** tra
quello che il modello dice e quello che dovrebbe dire. Se stampa $0{,}05$ e
stiamo predicendo dei prezzi in euro, il modello sbaglia in media di cinque
centesimi. Un numero che si può raccontare a chiunque, senza spiegare che cosa
sia un quadrato di un errore. Il valore stampato *sul test* è quello che conta:
è il voto preso su domande mai viste.
`````

`````{tab} Superiore
`nn.L1Loss` calcola l'errore assoluto medio
$\mathcal{L} = \frac{1}{N}\sum_i |\hat{y}_i - y_i|$, mentre `nn.MSELoss` media
i quadrati. La differenza pratica sta nei gradienti e negli *outlier*: il
gradiente della L1 rispetto al residuo è $\pm 1$, costante, quindi un punto
molto lontano non domina l'aggiornamento, la L1 è robusta; con il lr fissato,
però, il modello non converge esattamente ma oscilla in un intorno di ampiezza
$\sim \eta$ attorno all'ottimo. La MSE, il cui gradiente è proporzionale al
residuo, converge in modo più pulito ma insegue gli outlier. La
`nn.SmoothL1Loss` (o *Huber*) è il compromesso: quadratica vicino allo zero,
lineare lontano. Qui la scelta è quasi indifferente perché i dati sono
esattamente su una retta: la loss finale è limitata solo dalla granularità dei
passi.
`````

## Loss e ultimo strato: una scelta che dipende dal problema

La domanda "quale loss uso?" ha una risposta quasi meccanica, e il tipo di
problema la determina insieme alla forma dell'ultimo strato. Vale la pena
tenere questa tabella sott'occhio: metà degli errori dei principianti nascono
da una riga sbagliata qui.

| Tipo di problema | Ultimo strato | Funzione di perdita | Per leggere l'output |
|---|---|---|---|
| Regressione (un numero) | `nn.Linear(d, 1)` | `nn.MSELoss` o `nn.L1Loss` | niente, è già il numero |
| Classificazione binaria | `nn.Linear(d, 1)` | `nn.BCEWithLogitsLoss` | `torch.sigmoid` |
| Classificazione a $K$ classi | `nn.Linear(d, K)` | `nn.CrossEntropyLoss` | `torch.softmax(dim=1)` |
| Multi-etichetta ($K$ sì/no) | `nn.Linear(d, K)` | `nn.BCEWithLogitsLoss` | `torch.sigmoid` |

`````{tab} Elementare
La riga da leggere con più attenzione è la terza. Quando le classi sono più di
due, il modello non dà probabilità: dà dei **punteggi grezzi**, uno per
classe, che possono essere negativi o enormi. La funzione di perdita se li
aspetta proprio così: è lei a trasformarli in probabilità, al suo interno.
Aggiungere la trasformazione anche nel modello significa farla due volte, e il
risultato è un modello che impara male senza dare nessun errore: nessun
messaggio rosso, solo numeri che non salgono. È il bug più silenzioso di
tutti.
`````

`````{tab} Superiore
`BCEWithLogitsLoss` e `CrossEntropyLoss` incorporano rispettivamente la
sigmoide e la log-softmax, e vanno alimentate con i **logit**. Il motivo è
numerico: il calcolo congiunto usa il *log-sum-exp trick*, che evita
l'underflow di $\log(\hat{y})$ quando $\hat{y} \to 0$. Le versioni "nude"
(`nn.BCELoss`, `nn.NLLLoss`) esistono per i casi in cui la normalizzazione è
già avvenuta, ma nel dubbio si usa sempre la variante con i logit. Due note di
forma dei tensori: `BCEWithLogitsLoss` vuole target `float32` della stessa
shape dei logit, tipicamente si applica `squeeze()` all'uscita
$(N,1) \to (N,)$; `CrossEntropyLoss` vuole logit $(N,K)$ e target interi
$(N,)$ di dtype `int64`, **non** one-hot. Per classi molto sbilanciate,
entrambe accettano un peso per classe (`weight`, o `pos_weight` per la
binaria), che rialza il contributo della classe rara.
`````

## Il ciclo di miglioramento: una leva alla volta

Il modello gira, il numero non è buono abbastanza. È il momento in cui si
consuma il grosso di un progetto, ed è anche quello in cui si prendono le
decisioni peggiori: si cambiano cinque cose insieme, il risultato migliora, e
non si sa quale delle cinque abbia funzionato, quindi non si sa nemmeno quale
spingere ancora.

`````{tab} Elementare
La regola è quella dell'idraulico: **una chiave alla volta**. E c'è un ordine
sensato in cui provare, dal più efficace al più illusorio.

1. **Più dati, o dati migliori.** È quasi sempre la leva più potente, e quasi
   sempre la più noiosa. Mille esempi in più valgono di solito più di
   qualunque astuzia architetturale.
2. **Addestrare più a lungo**, tenendo d'occhio la curva di validazione: se
   sale, si è passato il punto di fermarsi.
3. **Un modello più capiente**: più strati, più unità per strato. Solo dopo
   aver verificato che il modello *piccolo* non ce la fa: non che i dati siano
   sbagliati.
4. **Il learning rate.** Tra tutti i numeri regolabili è quello che conta di
   più: cambiarlo per un fattore dieci in su o in giù spesso fa la
   differenza tra un modello che impara e uno che non parte.
5. **I freni** (dropout, weight decay), ma solo se il divario tra
   addestramento e validazione si sta allargando.
6. **Cambiare del tutto approccio**: un'altra architettura, o partire da un
   modello già addestrato da altri; il *transfer learning* del [capitolo sulla
   visione](../VisioneArtificiale/classificazione-transfer.md).

E prima di ogni prova: fissa il seme casuale, annota che cosa hai cambiato,
tieni il risultato. Un quaderno di laboratorio, letteralmente.
`````

`````{tab} Superiore
Formalmente si sta esplorando lo spazio degli iperparametri con un budget
limitato, e la sensibilità non è uniforme: il learning rate domina, seguito
dalla dimensione del batch e dalla capacità del modello, mentre molte altre
scelte contano poco. Da qui due pratiche standard. La prima è la **ricerca
casuale** invece della ricerca a griglia: con $n$ prove, la casuale campiona
$n$ valori distinti *per ogni* iperparametro, la griglia molti meno, e con
sensibilità così sbilanciate questo cambia tutto. La seconda è il *learning
rate range test*: si fa crescere $\eta$ esponenzialmente per poche centinaia
di iterazioni e si sceglie il valore poco prima che la loss esploda.

C'è poi una diagnosi da fare **prima** di tutto il resto: verificare che il
modello riesca a fare *overfitting* su un campione minuscolo (due o tre
batch). Se non riesce a mandare a memoria dieci esempi, il problema non sono
gli iperparametri: è un bug (target disallineati, loss sbagliata, gradienti
che non arrivano). Sono cinque minuti che ne risparmiano molti. Il repertorio
completo (regolarizzazione, scheduler, normalizzazione) è nel capitolo sul
[deep learning](../DeepLearning/ottimizzazione-regolarizzazione.md);
l'infrastruttura per non perdere il conto degli esperimenti in [dal notebook
alla produzione](../MLOps/dal-notebook-alla-produzione.md).
`````

## Predire su dati nuovi: tre condizioni e due interruttori

Il modello è addestrato. Arriva un dato mai visto e va dato in pasto alla
rete: è il gesto più semplice del capitolo, ed è quello che fallisce più
spesso. Il dato nuovo deve soddisfare **tre condizioni** (stesso dispositivo,
stesso tipo, stessa forma dei dati di addestramento) e vanno azionati **due
interruttori**.

```python
modello.eval()                                  # interruttore 1: modalità esame
with torch.no_grad():                           # interruttore 2: niente gradienti
    x_nuovo = torch.tensor([[0.95]],            # forma: (1, 1), non (1,)
                           dtype=torch.float32) # tipo: come in addestramento
    x_nuovo = x_nuovo.to(next(modello.parameters()).device)  # stesso dispositivo
    stima = modello(x_nuovo)
print(stima.item())        # ~ 0.7 * 0.95 + 0.3 = 0.965
```

La riga con `next(modello.parameters()).device` è un piccolo trucco che vale
la pena adottare: chiede al modello stesso dove abita, invece di ricordarselo
in una variabile globale che prima o poi si disallinea. Che cosa succede
quando una delle tre condizioni salta (e come si legge il messaggio d'errore
che ne esce) è l'argomento della sezione [sui tre errori più
comuni](errori-comuni.md).

```{admonition} Da ricordare
:class: important
- Il flusso di lavoro ha **sei stazioni**: problema, dati, modello,
  addestramento, valutazione, uso. Le tre centrali si ripetono in ciclo, ed è
  lì che va il tempo.
- Costruirsi un **problema con la risposta nota** (dati generati da una
  formula) è il modo più rapido per verificare che la propria macchina
  funzioni davvero.
- Loss e ultimo strato si scelgono dal **tipo di problema**:
  `MSELoss`/`L1Loss` per la regressione, `BCEWithLogitsLoss` per il sì/no,
  `CrossEntropyLoss` per le $K$ classi; le ultime due **vogliono i logit**.
- Nel ciclo di miglioramento si cambia **una leva alla volta**, in ordine:
  dati, durata, capacità, learning rate, regolarizzazione, architettura.
- Prima di ottimizzare qualunque cosa: verifica che il modello riesca a
  mandare a memoria **due batch**. Se non ci riesce, è un bug, non un
  iperparametro.
- Per predire su dati nuovi servono **tre condizioni** (device, dtype, shape)
  e **due interruttori** (`eval()`, `no_grad()`).
```
