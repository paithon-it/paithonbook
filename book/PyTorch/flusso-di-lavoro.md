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
:alt: Sei riquadri numerati collegati in sequenza (il problema, i dati, il modello, l'addestramento, la valutazione, salvare e usare) e una freccia tratteggiata che dalla quinta stazione torna alla terza, etichettata «non va? si cambia una cosa sola e si riprova».
:width: 90%

Il flusso di lavoro di un progetto PyTorch. Il percorso si attraversa una
volta in linea retta e poi decine di volte in circolo tra le stazioni 3 e 5:
è lì, non nella scrittura del modello, che si consuma il tempo.
```

Le sei stazioni, per nome: **problema**, **dati**, **modello**,
**addestramento**, **valutazione**, **uso**.

Le prime due non hanno niente a che vedere con PyTorch e sono quelle che
decidono l'esito: capire che cosa si vuole predire, e da quali dati. Le tre
centrali (scegliere il modello, addestrarlo, misurarlo) sono il ciclo vero e
proprio, e si ripercorrono decine di volte. L'ultima, mettere il modello al
lavoro per qualcuno che non sia chi l'ha costruito, è quella che quasi sempre
si dimentica di pianificare, e la riprende per esteso il
[capitolo sull'MLOps](../MLOps/overview.md), che si occupa appunto del mestiere
di tenere in piedi modelli che qualcuno usa davvero.

`````{tab} Elementare
È lo stesso ordine con cui si prepara una ricetta nuova. Prima decidi che
piatto vuoi (la stazione 1), poi controlli che cosa hai in dispensa (2). Solo
allora scegli il procedimento (3) e cucini (4). Poi (ed è il passaggio che
distingue chi cucina bene) **assaggi** (5), e l'assaggio non lo fai sul
cucchiaio che hai già leccato: usi una porzione che non hai ancora toccato,
altrimenti ti convinci che sia buono solo perché lo hai fatto tu. Se manca
sale, torni indietro e cambi *una cosa sola*, altrimenti al secondo assaggio
non saprai se è merito del sale o del tempo di cottura. E alla fine il piatto
lo porti in tavola (6), che è il momento in cui scopri se piace anche a chi non
l'ha cucinato: la sola prova che conta, e l'unica che quasi nessuno mette in
conto quando comincia.
`````

`````{tab} Superiore
Formalizzato: si fissa uno spazio di ipotesi $\mathcal{F}$ (l'architettura),
una funzione di perdita $\mathcal{L}$ e un algoritmo di ottimizzazione; si
stima $\theta$ minimizzando il rischio empirico sul training set; si misura il
rischio su un campione indipendente per stimare la generalizzazione. Le
stazioni 3–5 sono un ciclo di ricerca su iperparametri e architettura, guidato
dalla metrica di **validazione**, e ogni decisione presa guardando quel numero
lo consuma un po’, perché il set di validazione diventa a poco a poco parte
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

Vale la pena fermarsi un attimo su quei due nomi, perché è il punto migliore
di tutto il capitolo per capire che cosa sia davvero un peso. La **pendenza**
di una retta è quanto la retta sale ogni volta che ci si sposta di uno verso
destra; l’**intercetta** è l'altezza a cui la retta taglia l'asse verticale, il
punto da cui parte. Nel vocabolario delle reti neurali quei due numeri si
chiamano **peso** e **bias**, ed è la stessa cosa: il peso dice quanto
l'ingresso conta, il bias dove si parte. Una rete vera ne ha milioni invece di
due, ma il mestiere di ciascuno è questo.

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
`unsqueeze(dim=1)` trasforma la fila di cinquanta numeri in una **tabella** di
cinquanta righe e una colonna: gli strati di PyTorch vogliono una riga per
esempio, e su ogni riga le **caratteristiche** di quell'esempio (in inglese
*feature*, ed è la parola che si troverà nel codice: `in_features`,
`out_features`). Qui la caratteristica è una sola, ma la colonna ci vuole lo
stesso, ed è per questo che il conto delle dimensioni è il primo dei
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
# OrderedDict({'strato.weight': tensor([[0.7645]]), 'strato.bias': tensor([0.8300])})
```

Lo `state_dict`, che già conosciamo, qui contiene due soli numeri, ed è
interessante che siano già pieni: nessuno ha ancora addestrato niente, ma un
modello nasce sempre con dei numeri a caso dentro, sorteggiati da PyTorch nel
momento in cui lo si costruisce. È da lì che l'addestramento parte, e per
questo `manual_seed` conta: fissa anche quel sorteggio. (Il $0{,}7645$ che esce
dal caso somiglia al $0{,}7$ vero per pura coincidenza; il bias, $0{,}83$
contro $0{,}3$, è bello lontano.)

Adesso il ciclo. Rispetto alla sezione precedente cambiano tre cose, e vale la
pena dirle prima perché altrimenti sembrano contraddizioni. Primo: qui i
quaranta punti entrano tutti insieme a ogni giro, non a pacchetti, perché sono
quaranta e starebbero in un pacchetto solo; quindi qui «epoca» e «un giro di
correzione» coincidono, mentre su MNIST un'epoca erano quasi mille giri.
Secondo: l'ottimizzatore è SGD e non Adam, perché con due soli parametri il
vantaggio di Adam (un passo diverso per ciascuno) non si vede, e SGD lascia
vedere meglio quello che succede. Terzo: ogni tanto ci si ferma a misurare
anche sui dati messi da parte.

```python
criterio = nn.L1Loss()                                     # errore assoluto medio
# lr e' il learning rate, il "passo" della sezione precedente
ottimizzatore = torch.optim.SGD(modello.parameters(), lr=0.01)

for epoca in range(1000):
    modello.train()
    y_pred = modello(X_train)
    perdita = criterio(y_pred, y_train)
    ottimizzatore.zero_grad()
    perdita.backward()
    ottimizzatore.step()

    if epoca % 199 == 0:                                   # il "termometro"
        modello.eval()
        with torch.no_grad():
            perdita_test = criterio(modello(X_test), y_test)
        print(f"epoca {epoca:>4} | train {perdita.item():.4f} "
              f"| test {perdita_test.item():.4f}")

print(modello.state_dict())
```

Ecco che cosa stampa, che è la parte da guardare:

```text
epoca    0 | train 0.5552 | test 0.5740
epoca  199 | train 0.0103 | test 0.0003
epoca  398 | train 0.0013 | test 0.0138
epoca  597 | train 0.0103 | test 0.0003
epoca  796 | train 0.0013 | test 0.0138
epoca  995 | train 0.0103 | test 0.0003
OrderedDict({'strato.weight': tensor([[0.6968]]), 'strato.bias': tensor([0.3025])})
```

Alla fine `state_dict()` stampa due numeri molto vicini a $0{,}7$ e $0{,}3$:
$0{,}6968$ e $0{,}3025$, cioè $0{,}70$ e $0{,}30$ arrotondati al centesimo. Non
identici, perché la discesa del gradiente si ferma quando è *abbastanza*
vicina. È una verifica che nella maggior parte dei problemi veri non potremo
mai fare, e proprio per questo vale la pena farla almeno una volta: qui
sappiamo con certezza che la macchina funziona.

Guardando la tabella si nota che le ultime righe si ripetono: $0{,}0103$ e
$0{,}0013$ tornano a turno. Non è un caso, ed è la cosa più istruttiva di tutto
l'esempio: verso la fine la perdita non si ferma su un valore, **alterna** fra
quei due, un giro sì e un giro no.

Perché lo faccia si dice in una riga. Questa misura dell'errore corregge sempre
della stessa quantità, che si sia lontanissimi o a un capello dal bersaglio:
sbagliare di $10$ e sbagliare di $0{,}001$ producono la stessa spinta. Non
«frena» avvicinandosi. E il passo è fisso, sempre $0{,}01$. Quindi, arrivata a
un capello dal punto giusto, la correzione lo scavalca; il giro dopo lo
scavalca all'indietro; e da lì in poi ci balla attorno per sempre, con
un'oscillazione grande più o meno quanto il passo.

Ecco perché il "termometro" stampa ogni $199$ epoche e non ogni $200$, che
sembra un capriccio e non lo è. Stampando ogni $200$, cioè un numero pari, si
guarderebbe sempre lo stesso piede del ballo: dopo la prima riga si vedrebbero
quattro righe con lo stesso identico numero, e il modello sembrerebbe fermo
sull'ottimo mentre gli sta girando attorno. Col $199$ i due piedi si vedono
tutti e due, ed è la verità.

`````{tab} Elementare
La `L1Loss` è la scelta più leggibile che ci sia: è la **distanza media** tra
quello che il modello dice e quello che dovrebbe dire. Se stampa $0{,}05$ e
stiamo predicendo dei prezzi in euro, il modello sbaglia in media di cinque
centesimi. Un numero che si può raccontare a chiunque, senza spiegare che cosa
sia un quadrato di un errore.

Il numero da guardare è quello di destra, misurato sui dieci punti messi da
parte: è l'unico preso su domande mai viste. E qui va detto che stiamo
prendendo la stessa scorciatoia della sezione precedente, guardandolo sei volte
durante la corsa: su un problema truccato come questo è innocuo, perché non
stiamo decidendo niente in base a quel numero, lo stiamo solo guardando
scendere. In un progetto vero quel ruolo lo farebbe un terzo mucchio, la
validazione, e il test resterebbe chiuso fino alla fine.
`````

`````{tab} Superiore
`nn.L1Loss` calcola l'errore assoluto medio
$\mathcal{L} = \frac{1}{N}\sum_i |\hat{y}_i - y_i|$, mentre `nn.MSELoss` media
i quadrati. La differenza pratica sta nei gradienti e negli *outlier*: il
gradiente della L1 rispetto al residuo è $\pm 1$, costante, quindi un punto
molto lontano non domina l'aggiornamento, la L1 è robusta; con il lr fissato,
però, il modello non converge esattamente ma oscilla in un intorno di ampiezza
$\sim \eta$ attorno all'ottimo. Qui l'oscillazione è un ciclo limite di
periodo 2, e si misura: il residuo cambia segno tutto insieme, quindi il
gradiente sul bias vale $\pm 1$ e il passo è esattamente $\pm \eta$; sul peso
il gradiente è $\pm \overline{|x_i|} = \pm 0{,}39$, e l'ampiezza scala di
conseguenza. È la ragione per cui il "termometro" stampa a passo dispari: a
passo pari si campionerebbe sempre la stessa fase. La MSE, il cui gradiente è
proporzionale al residuo, converge in modo più pulito ma insegue gli outlier. La
`nn.SmoothL1Loss` (o *Huber*) è il compromesso: quadratica vicino allo zero,
lineare lontano. Qui la scelta è quasi indifferente perché i dati sono
esattamente su una retta: la loss finale è limitata solo dalla granularità dei
passi.
`````

## Loss e ultimo strato: una scelta che dipende dal problema

"Quale loss uso?" è una domanda che ha una risposta quasi meccanica: la decide
il tipo di problema, e insieme a lei decide anche la forma dell'ultimo strato.
I tipi di problema, in fondo, sono quattro, e sono quattro modi di fare una
domanda a un modello: *quanto?*, *sì o no?*, *quale fra tanti?*, *quali fra
tanti?* Ognuno ha la sua riga nella tabella. Le due lettere che vi compaiono
stanno per «quanti numeri entrano nell'ultimo strato» ($d$) e «quante categorie
ci sono» ($K$). Vale la pena tenerla sott'occhio: metà degli errori dei
principianti nascono da una riga sbagliata qui.

| Tipo di problema | Ultimo strato | Funzione di perdita | Per leggere l'output |
|---|---|---|---|
| Regressione (un numero) | `nn.Linear(d, 1)` | `nn.MSELoss` o `nn.L1Loss` | niente, è già il numero |
| Classificazione binaria | `nn.Linear(d, 1)` | `nn.BCEWithLogitsLoss` | `torch.sigmoid` |
| Classificazione a $K$ classi | `nn.Linear(d, K)` | `nn.CrossEntropyLoss` | `torch.softmax(dim=1)` |
| Multi-etichetta ($K$ sì/no) | `nn.Linear(d, K)` | `nn.BCEWithLogitsLoss` | `torch.sigmoid` |

`````{tab} Elementare
Le quattro righe sono i quattro tipi di domanda che si possono fare a un
modello. **Quanto?** (un prezzo, una temperatura): l'ultimo strato dà un numero
solo e lo si legge com'è; nel gergo questo caso si chiama **regressione**, ed è
quello del nostro esempio con la retta. **Sì o no?** (è spam, non è spam): un numero solo,
che poi va schiacciato fra zero e uno per leggerlo come probabilità, ed è ciò
che fa la `sigmoid`. **Quale, fra tanti?** (quale cifra, quale animale): tanti
numeri quante sono le categorie, e vince il più alto. **Quali, fra tanti?** (la
foto contiene un cane *e* un prato *e* una palla): di nuovo tanti numeri, ma
ognuno è un sì/no per conto suo, e più di uno può essere sì. È quello che si
chiama *multi-etichetta*, e la differenza con la riga di sopra è tutta lì: là
si sceglie una risposta, qui se ne accendono quante se ne vuole.

La riga da leggere con più attenzione è la terza. Quando le classi sono più di
due, il modello non dà probabilità: dà dei **punteggi grezzi**, uno per classe,
che possono essere negativi o enormi. Quei punteggi hanno un nome, ed è la
sillaba che si trova nei nomi delle funzioni: si chiamano **logit**, ed è per
questo che la funzione di perdita per il sì/no si chiama
`BCEWithLogitsLoss`, cioè «con i logit». La funzione di perdita se li aspetta
proprio così: è lei a trasformarli in probabilità, al suo interno.

L'ultima colonna della tabella non contraddice questo, e vale la pena essere
espliciti perché è il punto dove ci si incarta. La `softmax` va usata **dopo**,
sul risultato, quando si vuole leggere una probabilità da mostrare a qualcuno;
non va messa **dentro** il modello, come ultimo strato. Sono due righe di
codice che si assomigliano e fanno cose opposte: la prima è una lettura e non
tocca l'addestramento, la seconda fa applicare la trasformazione due volte, una
dal modello e una dalla loss. Il risultato è un modello che impara male senza
dare nessun errore: nessun messaggio rosso, solo numeri che non migliorano. È
il bug più silenzioso di tutti.
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

Il modello gira, e il numero che conta (quello sui dati messi da parte, non
quello sui dati su cui ha studiato) non è buono abbastanza. È il momento in cui
si consuma il grosso di un progetto, ed è anche quello in cui si prendono le
decisioni peggiori: si cambiano cinque cose insieme, il risultato migliora, e
non si sa quale delle cinque abbia funzionato, quindi non si sa nemmeno quale
spingere ancora.

`````{tab} Elementare
La regola è quella dell'idraulico: **una chiave alla volta**. E c'è un ordine
sensato in cui provare, dal più efficace al più illusorio.

1. **Più dati, o dati migliori.** È quasi sempre la leva più potente, e quasi
   sempre la più noiosa. Mille esempi in più valgono di solito più di
   qualunque astuzia architetturale.
2. **Addestrare più a lungo**, tenendo d'occhio l'errore sulla validazione, la
   simulazione d'esame vista nella sezione precedente: se ricomincia a salire,
   si è passato il momento di fermarsi.
3. **Un modello più capiente**: più strati, più unità per strato. Ma solo dopo
   essersi accertati che il modello piccolo non ce la faccia davvero, e che il
   problema non siano invece i dati: un modello grande su dati sbagliati impara
   soltanto a memoria le cose sbagliate.
4. **Il learning rate**, cioè il passo. Tra tutti i numeri regolabili è quello
   che conta di più: cambiarlo per un fattore dieci in su o in giù spesso fa la
   differenza tra un modello che impara e uno che non parte.
5. **I freni**: tutto ciò che rende la vita un po’ più difficile al modello
   mentre studia, apposta perché non si limiti a memorizzare (i nomi che
   incontrerai sono *dropout* e *weight decay*, e li spiega il capitolo sul
   deep learning). Si mettono solo se la distanza fra l'errore in addestramento
   e quello in validazione si sta allargando.
6. **Cambiare del tutto approccio**: un'altra architettura, o partire da un
   modello già addestrato da altri; il *transfer learning* del [capitolo sulla
   visione](../VisioneArtificiale/classificazione-transfer.md).

E prima di ogni prova: fissa il seme casuale, annota che cosa hai cambiato,
tieni il risultato. Un quaderno di laboratorio, letteralmente.

C'è però un controllo che viene **prima** di tutti e sei, e che costa cinque
minuti: prendi due mucchietti di esempi, una ventina in tutto, e addestra su
quelli finché l'errore non è quasi zero. Mandare a memoria venti esempi è alla
portata di qualunque rete, e se la tua non ci riesce non c'è manopola che la
salvi su cinquantamila: c'è un errore da qualche parte nel codice, e stai
girando le manopole sbagliate.
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

La riga con `next(modello.parameters()).device` merita una spiegazione, perché
sembra peggio di quello che è. Un modello, come i suoi dati, sta fisicamente da
qualche parte: nella memoria del processore o in quella della scheda grafica. E
i due possono lavorare insieme solo se stanno nello stesso posto. Quella riga
prende il primo peso che il modello ha in casa, gli chiede dove abita, e ci
manda il dato nuovo. Il vantaggio è che così la risposta viene dal modello
stesso invece che da un appunto scritto altrove nel programma, che prima o poi
qualcuno cambierà senza ricordarsi di aggiornare anche questo.

Che cosa succede quando una delle tre condizioni salta, e come si legge il
messaggio d'errore che ne esce, è l'argomento della sezione [sui tre errori più
comuni](errori-comuni.md).

Il mestiere sta in queste sei stazioni e nel ciclo che le lega. Il resto del
capitolo torna a occuparsi dei pezzi, cominciando da quello che nella pratica
dà più lavoro di tutti: i dati.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il flusso di lavoro ha **sei stazioni**: che cosa voglio predire, con quali
  dati, quale modello, addestrarlo, assaggiarlo, usarlo. Le tre centrali si
  ripetono in circolo, ed è lì che va tutto il tempo.
- Costruirsi un **problema con la risposta nota** (punti generati da una retta
  che si conosce) è il modo più rapido per verificare che la propria macchina
  funzioni davvero: alla fine i due numeri devono tornare.
- La domanda che si fa al modello decide l'ultimo strato e la misura
  dell'errore: quanto? sì o no? quale fra tanti? quali fra tanti? Sbagliare
  questa riga è metà degli errori di chi comincia.
- Nel migliorare un modello si cambia **una cosa alla volta**, e in ordine:
  prima più dati, poi più tempo, poi un modello più grande, poi la manopola
  del passo, poi i freni, e solo alla fine si cambia strada.
- Prima di girare qualunque manopola, il collaudo che costa cinque minuti: il
  modello deve riuscire a **mandare a memoria venti esempi**. Se non ci
  riesce, non è una manopola sbagliata, è un errore nel codice.
- Per dare al modello un dato nuovo servono **tre condizioni** (stesso posto,
  stesso tipo di numeri, stessa forma) e **due interruttori** (modalità esame,
  niente appunti).
```
`````

`````{tab} Superiore
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
`````
