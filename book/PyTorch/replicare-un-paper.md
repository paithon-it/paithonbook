# Replicare un paper

Un articolo scientifico si può leggere in tre modi. Il primo è scorrerlo:
mezz'ora, si ricava l'idea generale e si dimentica in una settimana. Il
secondo è studiarlo: si seguono le derivazioni, si capisce l'argomento. Il
terzo è farlo girare: trasformare le equazioni in `nn.Module`, mandare
avanti un tensore e guardare se esce quello che deve uscire. È il modo più
severo, perché il codice non accetta i passaggi vaghi: dove il testo dice "si
proietta linearmente" bisogna decidere una matrice, e la matrice ha una forma
precisa.

Quello che si impara replicando un paper è un metodo, che vale per qualunque
articolo. Il modello su cui lo mettiamo alla prova usa due strati che arrivano
più avanti: la convoluzione, nella {doc}`sezione sulle reti convoluzionali
</DeepLearning/reti-convoluzionali>`, e l’attenzione multi-testa, nella
{doc}`sezione sull’attenzione </Transformers/attenzione>`.

Per ora contano come scatole di cui si conosce solo che forma entra e che forma
esce, e basta questo: quello che si controlla sono le proprietà che devono
valere comunque, prima e a prescindere da qualunque addestramento (nel gergo si
chiamano *invarianti*), e quelle si verificano dal di fuori, senza aprire le
scatole. Per questo il metodo funziona anche su un articolo di cui non si è
capito tutto, con un limite: gli invarianti sono condizioni necessarie, e
dicono che il montaggio può essere giusto, non che lo sia.

## Il metodo

`````{tab} Elementare
Replicare un paper somiglia a montare un mobile a partire da una fotografia
invece che dalle istruzioni. Si procede così.

Uno: fai l'inventario dei pezzi. Quasi ogni articolo ha una figura
dell'architettura e una tabella di numeri (quanti strati, quanto sono larghi).
Quelle due cose insieme sono la distinta dei materiali.

Due: traduci un'equazione alla volta. Le formule di un paper sono
tipicamente tre o quattro, e ognuna diventa poche righe di codice. Si va in
ordine, e non si passa alla successiva finché la precedente non gira.

Tre: controlla le misure a ogni passo. Dopo ogni pezzo, si manda dentro un
tensore finto e si guarda che forma esce. È l'equivalente del metro da
falegname: se una misura non torna, l'errore è lì, non tre pezzi più avanti.

Quattro: conta i pezzi alla fine. Se il paper dice che il modello ha 86
milioni di parametri e il tuo ne ha 40, hai saltato qualcosa. È la verifica più
economica di tutte, e non richiede di addestrare nulla; non vede però un pezzo
montato al contrario, che ha le stesse viti di quello giusto.

Poi, prima di dichiararlo finito, una spinta. Un mobile può stare in piedi con
un ripiano soltanto appoggiato: da fuori sembra montato, e cede al primo peso.
Si scuote il fianco e si guarda che cosa si muove insieme al resto; quello che
resta fermo non è avvitato a niente. Sul modello il gesto è lo stesso: si dà un
colpo solo, dall'uscita, e si guarda se è arrivato fino a ogni singolo pezzo.
Quelli che non l'hanno sentito non impareranno mai niente, perché la correzione
non li raggiunge. Senza la spinta, il ripiano appoggiato si scopre dopo tre
giorni di addestramento che non porta da nessuna parte.

E se in negozio c'è lo stesso mobile già montato dal fabbricante, la prova più
severa è metterlo accanto al tuo. Un ripiano montato al contrario ha le stesse
viti e le stesse misure di quello giusto, e né il metro né il conto dei pezzi
se ne accorgono; accanto all'originale salta all'occhio. Sul modello vuol dire
dare a tutti e due gli stessi pesi e la stessa immagine, e guardare se escono
gli stessi numeri.

Finché le misure e il conto dei pezzi non tornano, e la spinta non arriva a
tutti, chiedersi se il modello vada bene quanto nella fotografia è prematuro.
Da lì in poi comincia la parte difficile.
`````

`````{tab} Superiore
Formalizzato, il procedimento parte da un inventario dei pezzi (la struttura e
le dimensioni che il paper dichiara) e traduce un'equazione alla volta; a ogni
passo si verificano invarianti controllabili senza addestrare:

1. **Invariante di forma.** Ogni modulo definisce una mappa
   $f: \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_{\text{out}}}$; se ne
   verifica il tipo
   con un tensore casuale della forma dichiarata nel paper. Un `assert` sulla
   shape in uscita è un test unitario a costo zero.
2. **Invariante di conteggio.** Il numero di parametri è una funzione chiusa
   del numero di strati, delle larghezze e della dimensione dell'MLP (oltre che
   della patch e della risoluzione, nel caso del ViT), e i paper lo
   dichiarano. Uno scarto grosso (43 milioni invece di 86) indica un blocco
   mancante o una dimensione sbagliata; un conteggio che coincide all'unità
   esclude gli errori che cambiano il numero di parametri, e soltanto quelli:
   non vede il numero di teste, l'ordine delle operazioni, la posizione dei
   residui.
3. **Invariante di gradiente.** Un `backward()` su una loss finta deve
   produrre `p.grad is not None` per ogni parametro di `named_parameters()`.
   Il controllo vede un ramo staccato per sbaglio con `detach()`, o un
   sotto-modulo che il `forward` non usa mai; non vede un peso creato con
   `torch.tensor(...)` invece che `nn.Parameter`, perché quello fra i
   parametri non c'è e il ciclo non lo visita: lo prende l'invariante di
   conteggio, che esce più basso del dovuto. E `not None` non vuol dire
   diverso da zero (una ReLU che non si accende mai consegna gradienti
   nulli), quindi conviene guardare anche `p.grad.abs().sum()`. Tutti e due i
   difetti si scoprono qui e non dopo tre giorni di addestramento che non
   converge.
4. **Invariante di equivalenza.** A parità di pesi, l'uscita deve coincidere
   con quella di un'implementazione di riferimento, quando ce n'è una. È
   l'unico dei quattro controlli che vede una normalizzazione messa dopo
   l'attenzione invece che prima, un residuo spostato, un dettaglio che il
   paper non dichiara. Senza un riferimento ne resta una versione a costo
   minimo, l'indipendenza fra gli esempi: in `eval()` l'uscita di un esempio
   non deve cambiare se cambiano gli altri del batch.

Gli invarianti sono condizioni necessarie, non sufficienti: soddisfatti tutti
e quattro, dicono che il montaggio può essere quello del paper. Solo a quel
punto ha senso parlare di risultati numerici, ed è lì che comincia la parte
difficile.
`````

## Il caso: il Vision Transformer

Il metodo si prova su un modello molto più grande di quelli usati fin qui, e su
un articolo che ricompare più avanti: *An Image is Worth 16x16 Words*
{cite}`dosovitskiy2021image` (su arXiv nel 2020, alla conferenza ICLR nel 2021),
che ha mostrato come un Transformer applicato direttamente ai riquadri di
un'immagine, senza convoluzioni, regga il confronto con le reti convoluzionali
nella classificazione, a patto di un pre-addestramento su moltissimi dati. È un
buon caso di studio perché l'architettura si scrive in quattro equazioni e i
numeri da verificare sono pubblicati. Della teoria di questo modello non si è
ancora parlato, e questo mostra un fatto preciso: forme, conteggio e gradienti
si controllano senza averla capita; che il montaggio sia anche quello giusto lo
dice soltanto il confronto con un'implementazione di riferimento.

La prima equazione del paper costruisce la sequenza di ingresso. L'immagine si
divide in riquadri che non si sovrappongono, le *patch* (in italiano
quadratini, ma nel codice si troverà la parola inglese); ogni patch si
appiattisce in una fila di numeri, e una matrice $\mathbf{E}$, la stessa per
tutte, la trasforma in un vettore di $D$ numeri. Davanti a tutti si mette un
vettore in più che dall'immagine non viene, il *token di classe*, i cui numeri
si imparano durante l'addestramento come quelli dei pesi; e a ogni vettore si
somma un vettore di posizione, perché le patch, tolte dalla griglia, non
portano più con sé il posto che occupavano: senza, il modello non saprebbe più
quale stava in alto a sinistra.

Nell'articolo l'equazione è questa, con le parentesi quadre che mettono i
vettori in fila e la somma finale che aggiunge le posizioni:

$$
\mathbf{z}_0 = [\, \mathbf{x}_{\text{class}} ;\;
\mathbf{x}_p^{1}\mathbf{E} ;\; \mathbf{x}_p^{2}\mathbf{E} ;\; \dots ;\;
\mathbf{x}_p^{N}\mathbf{E} \,] + \mathbf{E}_{\text{pos}}
$$

dove $\mathbf{x}_p^{i}$ è la $i$-esima patch appiattita, $\mathbf{E} \in
\mathbb{R}^{(P^2 \cdot C) \times D}$ è la proiezione lineare, la stessa per
tutte le patch, $D$ è la lunghezza dei vettori che ne escono (nel codice
`d_modello`), $\mathbf{x}_{\text{class}}$ è il token di classe, messo in testa
alla sequenza, ed $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$
sono le codifiche di posizione. Con immagini $224 \times 224$, patch $P = 16$ e
$C = 3$ canali di colore (rosso, verde e blu, di cui è fatta ogni foto) si
ottengono $N = (224/16)^2 = 196$ patch, ciascuna di $16 \cdot 16 \cdot 3 = 768$
numeri.
Il quadrato viene da lì: $224/16 = 14$ è il numero di quadratini che stanno su
una riga, e l'immagine è una griglia, quindi le righe sono altrettante e i
quadratini in tutto sono $14 \times 14$.

Una coincidenza da segnalare, perché altrimenti confonde: il $768$ appena
calcolato ($16 \cdot 16 \cdot 3$, quanti numeri contiene una patch) e il $768$
che comparirà fra poco nel codice come `d_modello` sono due cose diverse.
Il primo è quanto entra nella proiezione, il secondo quanto ne esce, ed è una
scelta degli autori del paper. Che coincidano vuol dire soltanto che la
proiezione, in questo caso, non cambia il numero di numeri; con patch da $32$
pixel il primo diventerebbe $3072$ e il secondo resterebbe $768$.

Ecco la stessa equazione, in PyTorch:

```python
import torch
from torch import nn

class IncorporazionePatch(nn.Module):
    """Equazione 1 del paper: da immagine a sequenza di token."""

    def __init__(self, canali=3, patch=16, d_modello=768, immagine=224):
        super().__init__()
        n_patch = (immagine // patch) ** 2                    # 196

        # Il trucco: una convoluzione con kernel = stride = patch È la
        # proiezione lineare delle patch appiattite, calcolata tutta insieme.
        self.proiezione = nn.Conv2d(canali, d_modello,
                                    kernel_size=patch, stride=patch)

        # come nel codice di riferimento: classe a zero, posizioni piccole
        self.token_classe = nn.Parameter(torch.zeros(1, 1, d_modello))
        self.posizioni = nn.Parameter(
            0.02 * torch.randn(1, n_patch + 1, d_modello))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proiezione(x)                    # (B, 3, 224, 224) -> (B, 768, 14, 14)
        x = x.flatten(2).transpose(1, 2)          #                  -> (B, 196, 768)
        cls = self.token_classe.expand(B, -1, -1) #                     (B, 1, 768)
        x = torch.cat([cls, x], dim=1)            #                  -> (B, 197, 768)
        return x + self.posizioni                 # broadcast su tutto il batch
```

I commenti a destra del `forward` riportano la forma del tensore dopo ogni
riga: è il controllo delle forme a ogni passo. $B$ è la dimensione del batch,
cioè quante immagini passano insieme, e non cambia lungo il percorso. Si entra
con $(B, 3, 224, 224)$, cioè $B$ immagini a tre canali da $224 \times 224$
pixel. La convoluzione dà $(B, 768, 14, 14)$: la griglia si riduce a
$14 \times 14$ patch, e ciascuna ha ora $768$ componenti. La riga dopo
appiattisce la griglia e scambia due assi: $(B, 196, 768)$, cioè $196$ patch in
fila da $768$ componenti. Aggiunto il token di classe si arriva a
$(B, 197, 768)$, la forma che il modello mantiene identica in tutti e dodici i
blocchi che seguono.

Nel codice le righe da guardare sono due: la convoluzione e il token di classe.

`````{tab} Elementare
La convoluzione. Una mascherina di cartone con un buco quadrato, appoggiata
sopra la foto. Dai pixel che si vedono nel buco esce un pugno di numeri, poi la
mascherina scatta più in là e si ricomincia. `nn.Conv2d` fa questo, e ha due
manopole: quanto è largo il buco (`kernel_size`) e di quanto scatta ogni volta
(`stride`).

Il paper chiede di tagliare l'immagine in quadratini e di proiettare ciascuno.
Alla lettera sono tre gesti: ritagliare, impilare, moltiplicare la pila. Con un
buco largo sedici pixel e uno scatto pure di sedici il gesto è uno solo. La
mascherina si sposta di quanto è larga, quindi ogni quadratino passa sotto una
volta e nessun pixel due volte.

E i conti sono gli stessi. Stessi numeri da imparare (la mascherina ne aggiunge
soltanto uno per ogni numero che esce, una piccola correzione fissa), stessi
prodotti, sistemati in due ordini diversi, e per passare dall'uno all'altro
basta riordinarli. Cambia il lavoro attorno. Chi ritaglia stacca tutti e 196 i
quadratini e li mette da parte, cioè si ritrova sul tavolo una seconda foto
fatta a pezzi grande quanto la prima, e solo dopo moltiplica; la mascherina li
legge dove stanno. Un gesto invece di tre, e più veloce.

Quella pila pesa quanto la foto finché la mascherina scatta di quanto è larga.
Chi la fa scattare di otto pixel con un buco da sedici ritrova lo stesso pixel
dentro più ritagli, e la pila diventa 3,7 volte la foto.

Il token di classe. Sopra la pila dei 196 ritagli si mette un foglio che di
immagine non ha niente. È lo stesso per ogni foto che arriva sul tavolo, e
quello che ci sta scritto lo decide l'addestramento, come per un peso qualunque
della rete. Attraversando la rete quel foglio raccoglie qualcosa da tutti gli
altri, e alla fine la risposta si legge lì, come nel verbale di una riunione,
che tiene il senso di quello che i partecipanti si sono detti senza essere uno
di loro.

Anche i cartellini delle posizioni si imparano allo stesso modo. Ai 197 posti
della fila (i 196 ritagli più il foglio) ne è attaccato uno ciascuno, che dice
da che punto della foto veniva il quadratino. Sono tanti quanti i posti, e
questo legame fra cartellini e posti si fa sentire quando le foto cambiano
misura. Chi dà foto da 384 pixel a un
modello addestrato su foto da 224 si ritrova sul tavolo 576 quadratini (24 per
riga invece di 14) e 197 cartellini in mano. Stamparne di nuovi, vuoti,
butterebbe via quello che il modello aveva imparato sulle posizioni, quindi si
prendono quelli che ci sono e si stirano sulla griglia più grande.

Al foglio c'è un'alternativa. Niente foglio in cima, e alla fine si fa la media
di quello che hanno detto i 196 ritagli. Funziona altrettanto bene a un patto,
scritto in fondo al paper e non nel testo principale: bisogna cambiare
la lunghezza dei passi con cui il modello si corregge mentre impara (il
*learning rate*). Con i passi di prima va peggio, e chi quella riga non l'ha
letta conclude che l'alternativa non funzioni, mentre ha soltanto lasciato i
passi com'erano.
`````

`````{tab} Superiore
L'equivalenza. Proiettare le patch appiattite significa calcolare
$\mathbf{x}_p^{i}\mathbf{E}$ con $\mathbf{E} \in \mathbb{R}^{(P^2C) \times D}$
per ogni $i$. Una `Conv2d` con `kernel_size = stride = P` calcola, per ogni
posizione non sovrapposta, il prodotto scalare tra la finestra e ciascuno dei
$D$ filtri: gli stessi $P^2C \cdot D$ moltiplicatori, riorganizzati. È
identica anche nei parametri, a meno del bias: la `Conv2d` ne ha $D$ che
l'equazione 1 non scrive (la tabella dei conti più avanti li include), e con
il bias la proiezione è affine invece che lineare. Per passare da una forma
all'altra basta portare il filtro $(D, C, P, P)$ in $(D, P^2C)$, con lo stesso
ordine (canale, riga, colonna) con cui si appiattiscono le patch. Il calcolo,
però, è delegato a un kernel ottimizzato invece che a `unfold` seguito da
`nn.Linear`. Con patch non sovrapposte `unfold` non duplica nessun pixel: il
tensore che materializza ha esattamente tanti elementi quanti l'immagine di
partenza ($B \cdot P^2C \cdot N = B \cdot C \cdot 224^2$), e quello che si
risparmia è la copia, non un'esplosione di memoria. L'esplosione arriva quando
il passo è minore della finestra, dove lo stesso pixel cade in più finestre:
con $P = 16$ e passo $8$ l'intermedio è già $3{,}7$ volte l'immagine.

Token di classe e posizioni. Entrambi sono `nn.Parameter`, cioè imparati:
$\mathbf{x}_{\text{class}}$ è la sonda da cui l'equazione 4 legge l'uscita, e
le codifiche di posizione sono *apprese*, non sinusoidali come nel Transformer
originale {cite}`vaswani2017attention` (dove gli autori avevano verificato che
i due tipi danno risultati quasi identici). L'ablazione del ViT (appendice D.4)
trova differenze trascurabili fra codifiche apprese a una dimensione, a due e
relative ($0{,}642$, $0{,}640$ e $0{,}640$ di accuratezza lineare 5-shot su
ImageNet con ViT-B/16), mentre senza alcuna codifica si scende a $0{,}614$: la
posizione serve, la sua forma poco. Due conseguenze pratiche. La prima: la
lunghezza di $\mathbf{E}_{\text{pos}}$ è legata alla risoluzione, quindi
cambiare la dimensione dell'immagine richiede di interpolare le codifiche, non
basta riallocarle. La seconda: l'alternativa al token di classe è il *global
average pooling* sui token delle patch, che funziona altrettanto bene ma
richiede un learning rate diverso; dettaglio che il paper riporta in appendice,
ed esattamente il tipo di nota che fa fallire una replica.
`````

Le equazioni 2 e 3 descrivono il blocco che poi si ripete dodici volte, e in
esse compaiono tre sigle e due parole che la {doc}`sezione sulla struttura del
Transformer </Transformers/architettura>` spiega per esteso.
Qui bastano una riga a testa. La scatola in cui i quadratini si guardano fra
loro, e ognuno raccoglie qualcosa dagli altri, è l'attenzione multi-testa, MSA
nel paper. MLP è una coppia di strati come quelli già
visti, che lavora su ogni posizione per conto suo. LN è la
*LayerNorm*, che rimette i numeri su una scala comoda prima di darli in pasto
alle altre due. *Pre-norm* vuol dire soltanto che quella rimessa in scala
avviene prima delle scatole e non dopo. E la connessione residua è il
`+ z` in fondo a ciascuna riga: quello che la scatola ha prodotto non
sostituisce l'ingresso, gli si somma, così il segnale originale ha sempre una
strada libera per arrivare in fondo.

$$
\begin{aligned}
\mathbf{z}'_{\ell} &= \text{MSA}\big(\text{LN}(\mathbf{z}_{\ell-1})\big) + \mathbf{z}_{\ell-1}, \\
\mathbf{z}_{\ell}  &= \text{MLP}\big(\text{LN}(\mathbf{z}'_{\ell})\big) + \mathbf{z}'_{\ell},
\end{aligned}
\qquad \ell = 1 \dots L
$$

dove $\mathbf{z}_{\ell}$ è la sequenza degli $N+1$ vettori all'uscita del blocco
$\ell$ ($\mathbf{z}_0$ è quella dell'equazione 1), $\mathbf{z}'_{\ell}$ il
valore intermedio dopo l'attenzione, e $L = 12$ il numero dei blocchi.
L'equazione 4 legge la risposta dal solo token di classe, dopo un'ultima
normalizzazione: $\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$, dove l'indice alto
$0$ indica la prima posizione della sequenza, quella del token di classe.

```python
class BloccoTransformer(nn.Module):
    """Equazioni 2 e 3: attenzione multi-testa e MLP, entrambe pre-norm."""

    def __init__(self, d_modello=768, teste=12, d_mlp=3072, dropout=0.1):
        super().__init__()
        # eps=1e-6 come nel riferimento; il default di PyTorch e' 1e-5
        self.norm1 = nn.LayerNorm(d_modello, eps=1e-6)
        self.attenzione = nn.MultiheadAttention(d_modello, teste,
                                                dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_modello, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_modello, d_mlp),
            nn.GELU(),                       # il paper usa GELU, una ReLU smussata
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d_modello),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attenzione(h, h, h, need_weights=False)[0]   # residuo 1
        x = x + self.mlp(self.norm2(x))                           # residuo 2
        return x
```

Il blocco ha due trappole. `nn.MultiheadAttention` restituisce una coppia,
l'uscita e i pesi dell'attenzione: senza il `[0]` che tiene la sola uscita, la
somma col residuo dà un errore di tipo poco comprensibile. E
`batch_first=True` non è il default: senza, il modulo si aspetta i tre assi
nell'ordine (posizione, esempio, numeri) e non (esempio, posizione, numeri).
Questo secondo errore non solleva mai un'eccezione, perché l'attenzione
conserva le forme e il residuo si chiude lo stesso: il modello gira, ma ogni
patch raccoglie informazione dalla patch nello stesso posto delle altre
immagini del batch, invece che dalle altre patch della propria. Lo trova la
prova dell'indipendenza fra gli esempi, che arriva poco più avanti insieme alla
verifica del modello intero.

## Verificare senza addestrare

La tabella 1 del paper dichiara, per la variante **ViT-Base**: $12$ strati,
dimensione nascosta $768$ (la $D$ dell'equazione 1), dimensione dell'MLP
$3072$, $12$ teste di attenzione, 86 milioni di parametri. I primi quattro
numeri li abbiamo copiati nel codice; il quinto no, e ne è una conseguenza solo
in parte. Dipende da strati, dimensione nascosta e dimensione dell'MLP (e da
patch e risoluzione, che la tabella non riporta), ma non dal numero di teste,
perché le teste si spartiscono le colonne di ciascuna proiezione. Il conto si
fa in trenta secondi, senza una GPU e senza dati, e verifica quelle dimensioni;
le teste e l'ordine delle operazioni si controllano in un altro modo.

```python
modello = nn.Sequential(
    IncorporazionePatch(),
    *[BloccoTransformer() for _ in range(12)],
)

finto = torch.randn(2, 3, 224, 224)               # due immagini finte
uscita = modello(finto)
print(uscita.shape)                                # torch.Size([2, 197, 768])

n_parametri = sum(p.numel() for p in modello.parameters() if p.requires_grad)
print(f"{n_parametri:,}")                          # 85,797,120
```

Il conto torna. Rifarlo a mano una volta smaschera ogni svista che cambia il
numero di parametri (un blocco in meno, una dimensione sbagliata), e nessuna di
quelle che lo lasciano com'è:

| Pezzo | Formula | Parametri |
|---|---|---|
| Proiezione delle patch | $768 \cdot 768 + 768$ | $590\,592$ |
| Token di classe | $768$ | $768$ |
| Codifiche di posizione | $197 \cdot 768$ | $151\,296$ |
| Attenzione, per blocco | $4 \cdot (768^2 + 768)$ | $2\,362\,368$ |
| MLP, per blocco | $2 \cdot 768 \cdot 3072 + 3072 + 768$ | $4\,722\,432$ |
| Due LayerNorm, per blocco | $2 \cdot 2 \cdot 768$ | $3\,072$ |
| **Un blocco intero** | $2\,362\,368 + 4\,722\,432 + 3\,072$ | $7\,087\,872$ |
| **12 blocchi** | $12 \cdot 7\,087\,872$ | $85\,054\,464$ |
| **Totale** | $590\,592 + 768 + 151\,296 + 85\,054\,464$ | $\mathbf{85\,797\,120}$ |

Nella prima riga il $768$ a sinistra è quanto entra nella proiezione
($16 \cdot 16 \cdot 3$), quello a destra quanto ne esce (`d_modello`).

Due righe hanno un fattore da giustificare. Il **4** dell'attenzione conta
quattro proiezioni della stessa forma $768 \times 768$ più bias: tre producono
le tre versioni di ogni elemento che l'attenzione mette in gioco (le *query*,
le *chiavi* e i *valori*), la quarta ricompone l'uscita delle teste. Il **2**
della LayerNorm è perché una normalizzazione, dopo aver riportato i numeri su
una scala standard, li riscala di nuovo con due parametri imparati per
componente, un moltiplicatore e uno spostamento: due numeri per ciascuna delle
$768$ componenti, e i normalizzatori per blocco sono due, da cui
$2 \cdot 2 \cdot 768$.

Poco meno di $86$ milioni: è il numero che dichiara il paper. Il nostro conto,
però, si è fermato prima di due pezzi finali, e sono quelli che ci mancano per
arrivare a un modello completo. Il primo è la LayerNorm dell'equazione 4, che
vale $2 \cdot 768 = 1\,536$ con lo stesso conto di prima. Il secondo è la
**testa di classificazione**, cioè lo strato che dai 768 numeri del token di
classe ricava un punteggio per ciascuna delle $K$ classi: sono $768 \cdot K$
pesi, uno per ogni coppia componente-classe, più $K$ bias, uno per classe.

La verifica si può quindi portare fino in fondo su
`torchvision.models.vit_b_16`, che è lo stesso modello con $K = 1000$ classi:

$$
85\,797\,120 + 1\,536 + (768 \cdot 1000 + 1000) = 86\,567\,656,
$$

ed è esattamente il numero che quel modello riporta, fino all'ultima cifra. Il
$+K$ in coda conta: scordarsi i mille bias della testa
farebbe chiudere il conto mille parametri sotto, e in una verifica che si
vanta di essere esatta all'unità mille parametri si vedono.

Se invece il nostro conteggio fosse uscito attorno ai $43$ milioni, il primo
sospetto sarebbero sei blocchi invece di dodici, che danno $43\,269\,888$
parametri; se fosse uscito attorno ai $170$ milioni, un numero di blocchi o una
larghezza raddoppiati. Il controllo costa trenta secondi.

Lo stesso controllo, strato per strato, lo dà `torchinfo`:

```python
from torchinfo import summary
summary(modello, input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params"])
```

Quello che stampa è una tabella, una riga per strato, con la forma in ingresso,
la forma in uscita e quanti parametri quel pezzo si porta dietro; in fondo, la
somma. Serve per due cose: quando la catena si spezza, per vedere in quale riga
la forma smette di combaciare, e quando il totale non torna, per capire su
quale blocco è andato perso. Sul nostro modello il riepilogo in fondo alla
tabella dice `Total params: 85,797,120`, lo stesso numero del conto a mano, ma
stavolta con davanti il dettaglio di dove sta ciascun pezzo.

Forme e conteggio non vedono le sviste che lasciano i numeri come sono: una
normalizzazione messa dopo l'attenzione invece che prima, un residuo spostato,
un dettaglio che il paper non scrive, gli assi scambiati di `batch_first`. Le
vede il confronto con un'implementazione di riferimento, che per il ViT c'è,
`vit_b_16` di torchvision. Si copiano i suoi pesi nel nostro modello, pezzo per
pezzo, si dà a tutti e due la stessa immagine, e le uscite dei dodici blocchi
devono coincidere. Il riferimento non serve addestrato, perché si confronta il
montaggio e non quello che ha imparato.

```python
from torchvision.models import vit_b_16

torch.manual_seed(0)
riferimento = vit_b_16().eval()       # pesi a caso: si confronta il montaggio


def copia_pesi(nostro, rif):
    """Porta nel nostro modello i pesi del riferimento, pezzo per pezzo."""
    with torch.no_grad():
        nostro[0].proiezione.weight.copy_(rif.conv_proj.weight)
        nostro[0].proiezione.bias.copy_(rif.conv_proj.bias)
        nostro[0].token_classe.copy_(rif.class_token)
        nostro[0].posizioni.copy_(rif.encoder.pos_embedding)
        for b, r in zip(nostro[1:], rif.encoder.layers):
            coppie = [(b.norm1, r.ln_1), (b.norm2, r.ln_2),
                      (b.attenzione.out_proj, r.self_attention.out_proj),
                      (b.mlp[0], r.mlp[0]), (b.mlp[3], r.mlp[3])]
            for mio, suo in coppie:
                mio.weight.copy_(suo.weight)
                mio.bias.copy_(suo.bias)
            b.attenzione.in_proj_weight.copy_(r.self_attention.in_proj_weight)
            b.attenzione.in_proj_bias.copy_(r.self_attention.in_proj_bias)


def scarto(nostro, rif, immagini):
    """La differenza massima fra le uscite dei dodici blocchi."""
    catturato = {}
    gancio = rif.encoder.ln.register_forward_hook(
        lambda modulo, ingresso, uscita: catturato.update(z=ingresso[0]))
    with torch.no_grad():
        rif(immagini)                 # il gancio prende l'uscita dei blocchi
        gancio.remove()
        return (nostro(immagini) - catturato["z"]).abs().max().item()


copia_pesi(modello, riferimento)
modello.eval()
immagini = torch.randn(2, 3, 224, 224)
prima = scarto(modello, riferimento, immagini)
for m in modello.modules():
    if isinstance(m, nn.LayerNorm):
        m.eps = 1e-5                      # il default di PyTorch
dopo = scarto(modello, riferimento, immagini)
print(f"scarto dal riferimento:      {prima:.1e}")
print(f"con le LayerNorm a eps=1e-5: {dopo:.1e}")

# senza un riferimento: un esempio non deve dipendere dagli altri del batch
x = torch.randn(4, 197, 768)
sbagliato = BloccoTransformer().eval()
sbagliato.attenzione.batch_first = False      # la seconda trappola
casi = (("batch_first=True ", modello[1]), ("batch_first=False", sbagliato))
with torch.no_grad():
    for nome, blocco in casi:
        uguale = torch.allclose(blocco(x)[:1], blocco(x[:1]), atol=1e-5)
        print(f"un esempio non dipende dagli altri, {nome}: {uguale}")
```

```text
scarto dal riferimento:      0.0e+00
con le LayerNorm a eps=1e-5: 4.5e-02
un esempio non dipende dagli altri, batch_first=True : True
un esempio non dipende dagli altri, batch_first=False: False
```

Con le LayerNorm come nel riferimento lo scarto è zero, cifra per cifra: il
montaggio è quello. Con il default di PyTorch, `eps` a $10^{-5}$ invece che a
$10^{-6}$, lo scarto sale a qualche centesimo, e il confronto trova da solo un
dettaglio che il paper non dichiara e che il conteggio non può vedere, perché
`eps` non è un parametro. Senza un riferimento resta la prova minima: con
`batch_first` sbagliato forme, conteggio e gradienti tornano lo stesso, e
l'indipendenza fra gli esempi no.

## Quando i numeri non tornano

Architettura verificata, e poi? Riprodurre la *struttura* di un paper è
questione di ore; riprodurne i risultati spesso non lo è, e non per colpa di chi
ci prova.

Il ViT è un caso esemplare proprio in questo. La tesi dell'articolo è che
l'architettura raggiunge o supera le reti convoluzionali (le CNN, la famiglia
di modelli per immagini del {doc}`capitolo sul deep learning
</DeepLearning/overview>`) solo dopo essere stata addestrata una prima volta su
quantità di dati enormi, e solo allora rifinita sul compito che interessa: è
quello che si chiama *pre-addestramento*. La parola dice in che ordine si
fanno le cose, e lascia aperto il resto: qui le immagini portano ancora la
loro etichetta, mentre sul testo la stessa mossa si farà senza che nessuno
etichetti niente. Nel paper quelle quantità sono ImageNet-21k, ventunomila
categorie su quattordici milioni di immagini, o il JFT-300M interno a Google,
diciottomila categorie su trecento milioni di immagini mai rese pubbliche.
Addestrato da zero sul solo ImageNet-1k con la ricetta del paper, lo stesso
codice resta sotto le reti convoluzionali di dimensione paragonabile, e questo
è un *risultato* del paper, non un fallimento della replica. Il limite dipende
però anche dalla ricetta: con aumentazione dei dati e regolarizzazione forti, e
senza un'immagine in più, DeiT porta la stessa architettura all’81,8% sul solo
ImageNet-1k, e all’83,4% aggiungendo una distillazione
{cite}`touvron2021training`.

Sapere in anticipo che la riproduzione completa è impossibile cambia
l'obiettivo, e in meglio: si replica l'architettura, la si verifica scaricando
i pesi che gli autori hanno pubblicato, e si addestra su un problema alla
propria portata partendo da quei pesi invece che da zero. Quest'ultima mossa si
chiama *transfer learning*, ed è l'argomento della {doc}`sezione sul transfer
learning </VisioneArtificiale/classificazione-transfer>`.

Quando invece i numeri dovrebbero tornare e non tornano, i sospetti abituali
sono questi:

1. I dati e le trasformazioni. Ritaglio, risoluzione, statistiche di
   normalizzazione, *augmentation* (le variazioni casuali applicate alle
   immagini durante l'addestramento): spesso sono descritte in una riga di
   appendice.
2. Il programma del learning rate, cioè come il passo cambia durante
   l'addestramento: di solito sale piano all'inizio (il riscaldamento, in
   inglese *warmup*) e poi scende,
   per esempio lungo un arco di coseno fino a zero, e il valore di picco
   dipende dalla dimensione del batch. Un paper che dice solo
   "lr $= 10^{-3}$" ne sta omettendo metà.
3. La dimensione del batch e l'accumulo. Chi ha 8 GPU e chi ne ha una non
   stanno addestrando lo stesso modello, a meno di accumulare i gradienti.
4. I freni, cioè tutto quello che si mette apposta per rendere la vita più
   difficile al modello mentre impara. Ce n'è una famiglia intera (weight decay,
   dropout, *label smoothing*, che sfuma le etichette invece di darle secche),
   e li raccoglie la sezione su [come far funzionare
   le reti profonde](../DeepLearning/ottimizzazione-regolarizzazione.md): un paper
   che ne omette uno solo è già un altro esperimento.
5. L'inizializzazione, cioè da quali numeri partono i pesi, quando non è
   quella che la libreria mette di suo.
6. Il protocollo di valutazione: su quale porzione di dati si misura, in
   quanti modi si ritaglia ogni immagine di prova, e se il numero riportato è
   il migliore ottenuto o l'ultimo.
7. Il caso: il seme, e quanti semi sono stati provati.

`````{tab} Elementare
Fra questi sospetti, i due da guardare per primi sono i dati con le loro
trasformazioni e il programma del learning rate. La preparazione dei dati è
quasi sempre raccontata di fretta, in mezza riga d'appendice, e un dettaglio di
quella riga può spostare il risultato quanto un cambio di architettura. E il
learning rate quasi mai è un numero fisso: sale piano all'inizio (il
*riscaldamento*) e poi scende lungo l'addestramento, quindi chi legge solo il
valore di picco sta copiando un terzo dell'informazione.

Quel valore di picco, poi, è tarato sul vassoio che gli autori avevano sotto
mano. Se hai una GPU sola e il paper ne usava otto, il vassoio è molto più
piccolo, la direzione in cui il modello si corregge viene da molti meno esempi,
e lo stesso passo lungo di prima si dà quasi alla cieca. Le strade sono due:
accorciare il passo, oppure accumulare le correzioni di più vassoi piccoli e
muoversi una volta sola, che è un modo di fingere un vassoio grande su una
macchina piccola. Il trucco imita bene, non alla perfezione: certi strati si
tarano sulla media di quello che hanno davanti, e davanti hanno ancora il
vassoio piccolo. Chi addestra a quelle scale sceglie allora strati che
rimettono in scala ogni esempio per conto suo, come la LayerNorm.

Restano due sviste che, mentre succedono, non danno nessun segnale. Una
riguarda i freni: quello che tira di continuo i pesi verso lo zero (il *weight
decay*) di solito si mette sulla gran parte dei pezzi, ma non su quelli che
servono soltanto a rimettere i numeri in scala; frenare anche loro non rompe
niente, e sposta il risultato di poco, in un verso che dipende dal modello.
L'altra è il righello: se il paper riporta il risultato migliore fra molte
prove, o la media di più ritagli della stessa foto di prova, e tu riporti
l'ultimo numero che ti è uscito, una parte della differenza viene da come si
misura e non dal modello.
`````

`````{tab} Superiore
In dettaglio, i punti su cui una replica si perde.

Programma del learning rate. Una forma molto diffusa è warmup lineare per
$T_w$ passi seguito da decadimento a coseno fino a zero. Il valore di picco non
è trasferibile tra batch di dimensione diversa: la *linear scaling rule*
{cite}`goyal2017accurate` prescrive $\eta \propto B$ per SGD, con un warmup che
la tiene stabile nei primi passi, e vale finché $B$ resta sotto una soglia
oltre la quale l'accuratezza peggiora rapidamente (circa $8\,000$ immagini nelle
loro prove su ImageNet); con Adam e AdamW l'analisi via equazioni differenziali
stocastiche {cite}`malladi2022sdes` suggerisce invece $\eta \propto \sqrt{B}$,
insieme a memorie più corte per le due medie: moltiplicando il batch per
$\kappa$, $\eta$ si moltiplica per $\sqrt{\kappa}$, $1-\beta_1$ e $1-\beta_2$
per $\kappa$, e $\epsilon$ si divide per $\sqrt{\kappa}$. È la regola da cui si
parte, da ricontrollare sul proprio problema. Un paper che riporta solo $\eta$
senza $B$, warmup e schedule non è replicabile alla lettera. Nel ViT i valori
stanno in un'appendice (tabella 3): batch $4096$, warmup lineare di $10\,000$
passi, Adam con $\beta_1 = 0{,}9$ e $\beta_2 = 0{,}999$, decadimento lineare
del learning rate nel pre-addestramento su JFT e su ImageNet-21k e a coseno su
ImageNet, weight decay $0{,}1$, $0{,}03$ e $0{,}3$ rispettivamente, e su
ImageNet anche il taglio del gradiente a norma globale $1$.

Accumulo dei gradienti. Il batch efficace è
$B_{\text{eff}} = B_{\text{micro}} \times k \times n_{\text{GPU}}$, dove $k$
sono i passi di accumulo: si eseguono $k$ `backward()` e un solo
`optimizer.step()`, ricordando di dividere la loss per $k$ se la riduzione è
la media. Non è del tutto equivalente a un batch grande vero: le statistiche
della batch normalization restano calcolate sul micro-batch, ragione per cui i
lavori che scalano molto preferiscono LayerNorm o GroupNorm.

Weight decay. Per convenzione si esclude da bias e parametri di
normalizzazione, e in PyTorch lo si fa passando a AdamW due *parameter group*
distinti, uno con `weight_decay=0`. Quanto pesi la scelta dipende dal modello:
su ResNet-50 l'esclusione sposta l'accuratezza di qualche centesimo o decimo di
punto, e nelle prove di He e colleghi verso il basso {cite}`he2019bag`; per gli
altri modelli non c'è una cifra da dare per buona.

Protocollo di valutazione. Se il paper usa una media esponenziale dei pesi
(EMA), o *test-time augmentation*, o riporta la metrica migliore sulla
validazione invece dell'ultima, confrontarsi con l'addestramento nudo dà una
differenza sistematica che non ha nulla a che vedere con l'architettura.
`````

La disciplina è la stessa della sezione sul
[flusso di lavoro](flusso-di-lavoro.md): si cambia un sospetto alla volta e
si registra. Ed è utile sapere che il problema è riconosciuto e studiato: la
comunità ha risposto con i *reproducibility checklist* adottati dalle grandi
conferenze, che chiedono agli autori di dichiarare esattamente questi punti
{cite}`pineau2021improving`.

### Il diario, che è la metà che nessuno scrive

«Si registra» merita più di due parole, perché è la pratica che distingue tre
giorni di lavoro da tre giorni di lavoro buttati. La forma minima è una riga
per esperimento, scritta prima di lanciarlo:

> *Esperimento 7. Ipotesi: la differenza viene dal warmup, che nel paper è di
> 10k passi e nel mio di 500. Cambio solo quello. Mi aspetto che la loss
> iniziale smetta di impennarsi. Esito: …*

Tre proprietà rendono utile questo rito, e tolta una delle tre non funziona più.
Una cosa alla volta, altrimenti il risultato non attribuisce il merito a
nessuno dei due cambi. L'ipotesi prima del risultato, perché scritta dopo si
adatta sempre a ciò che è successo, e si finisce per credere di aver capito. E
soprattutto si annota anche quello che non ha funzionato: è la metà che
nessuno scrive, ed è l'unica che impedisce di riprovare fra due settimane la
stessa cosa senza ricordarsene.

Gli strumenti che tracciano gli esperimenti (nel {doc}`capitolo su MLOps
</MLOps/overview>`) rendono tutto questo cercabile e condivisibile, e non c'è
ragione di non usarli. Ma registrano bene i parametri e i numeri, e non
registrano l'unica cosa che non si può ricostruire dopo: perché si era provato.
Quella va scritta a mano.

```{admonition} Onestà intellettuale
:class: note
"Non sono riuscito a riprodurre il risultato" è un esito legittimo, e si
scrive: quanto ci si è avvicinati, che cosa si è provato, che cosa mancava.
Una replica fallita e documentata è informazione utile per tutti; una
replica dichiarata riuscita senza esserlo, no. Vale anche per il proprio
lavoro: se un risultato dipende da un seme fortunato, non è un risultato.
```

Il metodo vale più del caso su cui l'abbiamo provato: si applica identico a
qualunque articolo che dichiari un'architettura e dei numeri.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Replicare un articolo è il modo più affidabile di capirlo: il codice non
  accetta i passaggi vaghi. Dove il testo dice «si proietta», il codice deve
  dire con che cosa e di che misura.
- Le mosse sono quattro: inventario dei pezzi, una formula alla volta,
  metro da falegname a ogni passo (mandi dentro un dato finto e guardi che
  forma esce), conteggio dei pezzi alla fine.
- Nessuna delle quattro richiede di addestrare niente, e nessuna richiede di
  aver capito che cosa fanno i pezzi dentro: bastano le misure in entrata e in
  uscita.
- Il conteggio dei pezzi costa trenta secondi: se l'articolo dice 86 milioni e
  a te ne escono 43, ne hai montata metà. Non vede però un pezzo montato al
  contrario, che si vede soltanto mettendo il tuo modello accanto a uno già
  montato, con gli stessi pesi dentro.
- E prima di dichiararlo finito, una spinta: un colpo dall'uscita, e si
  guarda se arriva fino a ogni pezzo. Quelli che restano fermi non sono
  avvitati a niente, e non impareranno mai.
- Riprodurre il montaggio è quasi sempre possibile; riprodurre i
  risultati spesso no, perché mancano i dati o metà delle istruzioni. Dirlo
  è parte del lavoro, non un'ammissione di sconfitta.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Replicare un paper è il modo più affidabile di capirlo: il codice non tollera
  i passaggi vaghi.
- Il metodo: inventario dei pezzi, un'equazione alla volta, e a ogni passo
  gli invarianti, cioè le proprietà che devono valere comunque, prima di
  qualunque addestramento. Sono quattro: forma in uscita, numero di parametri,
  gradiente su ogni parametro dopo un `backward()`, uscita uguale a quella di
  un riferimento a parità di pesi (o almeno indipendenza fra gli esempi del
  batch). Sono condizioni necessarie, non sufficienti.
- Nel ViT, una `Conv2d` con `kernel_size = stride = patch` *è* la proiezione
  lineare delle patch: riconoscere queste equivalenze fa parte del mestiere.
- ViT-Base ha $85\,797\,120$ parametri senza LayerNorm finale né testa (la
  testa ne aggiunge $768 \cdot K + K$): il conto si rifà a mano e smaschera
  ogni svista che cambia il numero di parametri, non il numero di teste né
  l'ordine delle operazioni. Il confronto con `vit_b_16` a pesi copiati vede
  anche quelle, e trova l’`eps` delle LayerNorm, $10^{-6}$ invece del default
  $10^{-5}$.
- Riprodurre l’architettura è quasi sempre possibile; riprodurre i
  risultati spesso no: dati non pubblici, iperparametri omessi, hardware
  diverso. Dirlo è parte del lavoro.
```
`````

Resta una domanda che il metodo da solo non risolve: una volta che il modello
è giusto, come lo si fa girare in fretta, e su più schede? È l'argomento di
{doc}`prestazioni e scala <prestazioni>`.
