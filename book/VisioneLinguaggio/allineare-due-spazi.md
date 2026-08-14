# Un solo spazio per le immagini e le parole

Le mille categorie di ImageNet, il dataset su cui si è addestrata mezza storia
della visione artificiale, contengono circa centoventi razze di cane e nessuna
classe «persona» (ci sono uno «sposo», un «giocatore di baseball» e un
«sommozzatore», ma l'essere umano in quanto tale non è una categoria). Non è una
svista, è la conseguenza di come nasce un classificatore: qualcuno decide una
lista, qualcun altro etichetta milioni di immagini secondo quella lista, e il
modello impara a rispondere sempre alla stessa domanda, quale delle mille.
Fuori da quell'elenco non esiste niente. Un tram non esiste, una radiografia
non esiste, e «un gatto nero che salta sul muro» non esiste nemmeno come
domanda: non è una classe, è una frase.

Nel capitolo sulla visione abbiamo visto la via d'uscita standard, il
**transfer learning**: si prende una rete pre-addestrata, si toglie la testa,
se ne monta una nuova dimensionata sulle proprie classi e la si addestra su
esempi etichettati a mano. Funziona, e resta il modo normale di costruire un
classificatore quando le classi sono poche e stabili. Ma il conto si paga ogni
volta che l'elenco cambia: una classe in più vuole immagini nuove, etichette
nuove, un addestramento nuovo e un modello nuovo da mettere in servizio. Quello
che il sistema sa dire viene deciso una volta per tutte, prima di partire.

La domanda di questa sezione è se si possa costruire un modello a cui le classi
si dicano **a parole**, nel momento in cui servono. La risposta comincia da un
cambio di domanda.

## Non «che cosa è», ma «quale di queste»

Prima dell'idea serve un attrezzo, e il libro lo ha già costruito. Nel capitolo
sul linguaggio abbiamo visto che una parola si può scrivere come un **vettore**,
cioè una fila di numeri, e che quelle file di numeri formano una **mappa del
significato**: *gatto* e *felino* finiscono vicini, *gatto* e *mercoledì*
lontanissimi, e quanto due cose siano vicine lo dice un solo numero fra $-1$ e
$+1$. Lo spazio di cui parla questa sezione è quella mappa, con un'aggiunta:
dentro non ci vanno soltanto le parole, ci vanno anche le fotografie. La foto di
un gatto nero su un muro deve finire **più vicina** alla frase «un gatto nero su
un muro» di quanto sia a «una scodella di minestra». Più vicina, si badi, non
sovrapposta: è una differenza che tornerà a farsi sentire alla fine della
sezione.

L'idea, resa celebre da CLIP {cite}`radford2021learning` nel 2021, è di
addestrare due reti separate, un **encoder di immagini** e un **encoder di
testo**, a scrivere le loro uscite su quell'unica mappa (in gergo: nello stesso
**spazio vettoriale**). Il compito non è più assegnare un'etichetta: è appaiare.
Dato un mucchietto di immagini e il mucchietto mescolato delle loro didascalie,
il modello deve dire chi va con chi.

```{figure} ../figures/clip-testo-e-immagini.svg
:name: fig-clip-matrice
:alt: "Un gruppo di immagini passa nell'encoder visivo e un gruppo di didascalie nell'encoder testuale; i due insiemi di vettori vengono confrontati a due a due in una matrice di similarità. Le celle sulla diagonale, che corrispondono agli abbinamenti corretti, vanno massimizzate; tutte le altre, gli abbinamenti sbagliati, vanno minimizzate."
:width: 96%

La matrice è il compito. Ogni riga porta con sé una risposta giusta e tante
sbagliate, e sono queste ultime, gratuite e numerose, a fare il grosso del
lavoro.
```

Il conto di {numref}`fig-clip-matrice` spiega perché conti tanto la dimensione
del **batch**, cioè quante coppie si mettono sul tavolo insieme a ogni passo di
addestramento (ci si torna più avanti in questa pagina). Con $N$ coppie ogni
riga porta una risposta giusta e $N - 1$ sbagliate: raddoppiare il batch
raddoppia le alternative sbagliate che ogni immagine deve scartare, e l'esame si
fa più difficile.

Le celle da riempire, però, sono $N^2$, e raddoppiando il batch diventano quattro
volte tante. Attenzione a che cosa costa: non i prodotti fra numeri che riempiono
la tabella, che accanto al lavoro dei due encoder sono briciole (gli encoder
elaborano $N$ immagini e $N$ testi, e quel conto cresce del doppio insieme al
batch). A crescere di quattro volte è **l'ingombro**: quella tabella va tenuta
tutta insieme in memoria, e ogni riga, per fare la sua media, ha bisogno di tutte
le colonne, cioè anche dei risultati calcolati sulle altre schede grafiche, che
vanno quindi radunati a ogni passo. La difficoltà dell'esame cresce del doppio,
l'ingombro per prepararlo di quattro volte: è questa forbice a rendere cari i
batch molto grandi, ed è esattamente il nodo che una variante successiva
scioglierà.

`````{tab} Elementare

Immagina un tavolo con quattro fotografie e, in disordine, quattro didascalie
ritagliate dal giornale. Nessuno ti dice che cosa raffigurano le foto: ti si
chiede solo di appaiarle. Il gioco sembra più povero di «riconosci il soggetto»,
e invece chiede la stessa cosa per vie traverse, perché per appaiare bene devi
comunque aver capito che nella prima foto c'è un gatto su un muro e che quella
didascalia parla di un gatto su un muro.

Il vantaggio è che questo gioco non ha bisogno di nessuno che prepari le
risposte. Le didascalie esistono già: ogni immagine pubblicata sul web arriva
con del testo attaccato, la frase sotto la foto, la descrizione alternativa che
serve a chi non vede, il titolo del prodotto in un catalogo. Sono coppie
già appaiate, gratis, a milioni: per addestrare CLIP ne sono state raccolte
quattrocento milioni. Nessuno le ha etichettate, nessuno ha deciso una lista di
categorie. È supervisione, ma **naturale**: viene dal fatto che gli esseri
umani, quando pubblicano un'immagine, ci scrivono accanto che cosa c'è.

`````

`````{tab} Superiore

Formalmente si apprendono due funzioni, $f_{\text{img}}$ e $f_{\text{txt}}$,
che portano rispettivamente un'immagine e una sequenza di token in un unico
spazio di rappresentazione $\mathbb{R}^d$ (in CLIP, tramite una proiezione
lineare posta in cima a ciascun encoder). L'encoder visivo è una CNN o un
Vision Transformer {cite}`dosovitskiy2021image`; quello testuale è un
Transformer con maschera causale, da cui si preleva la rappresentazione
dell'ultimo token. Le uscite vengono **normalizzate**,

$$
\mathbf{I}_i = \frac{f_{\text{img}}(\tilde{\mathbf{I}}_i)}{\lVert f_{\text{img}}(\tilde{\mathbf{I}}_i) \rVert_2},
\qquad
\mathbf{T}_j = \frac{f_{\text{txt}}(\mathbf{t}_j)}{\lVert f_{\text{txt}}(\mathbf{t}_j) \rVert_2},
$$

dove $\tilde{\mathbf{I}}_i$ è l'immagine $i$-esima del batch (il tensore grezzo,
quello che l'overview chiamava $\mathbf{I}$), $\mathbf{t}_j$ la didascalia
$j$-esima e $\mathbf{I}_i, \mathbf{T}_j \in \mathbb{R}^d$ i due embedding.
Adottiamo per questa sezione la notazione del paper di CLIP, in cui $\mathbf{I}$
e $\mathbf{T}$ denotano gli embedding normalizzati e non i dati grezzi: da qui
in poi $\mathbf{I}_i$ è un vettore, non un reticolo di pixel. Anche il grassetto
va letto, perché porta l'altra metà dell'informazione: $\mathbf{T}_j$ è un
vettore, mentre il $T$ tondo che si incontra altrove nel capitolo (il numero di
token di un prompt) è un conteggio. I due embedding vivono così sulla sfera
unitaria: il loro prodotto scalare $\langle \mathbf{I}_i, \mathbf{T}_j \rangle$ è esattamente il
coseno dell'angolo fra i due, un numero in $[-1, 1]$.

Il compito di pretesto è una classificazione a $N$ vie *definita dal batch
stesso*: data l'immagine $i$, indovinare quale delle $N$ didascalie presenti sia
la sua. Non c'è alcuna ontologia fissata a priori, e il «vocabolario» delle
descrizioni è aperto quanto la lingua. È un caso di apprendimento
auto-supervisionato di famiglia **contrastiva**, quella che il capitolo sui
world model metterà accanto alla generativa e alla predittiva nello spazio
latente: si impara una geometria, avvicinando ciò che va insieme e
allontanando ciò che non va insieme.

`````

```{figure} ../figures/vlm-contrastivo.svg
:name: fig-vlm-contrastivo
:alt: A sinistra due torri, l'encoder delle immagini e l'encoder del testo, che producono ciascuno un vettore normalizzato; le due frecce convergono in uno spazio condiviso rappresentato come una sfera unitaria su cui i due vettori sono vicini. A destra la matrice quattro per quattro delle similarità coseno del batch, con la diagonale piena di terracotta e i valori più alti, e tutte le altre celle chiare con valori bassi.
:width: 85%

Due encoder, una mappa sola. Le somiglianze di un gruppo di $N$ coppie formano
una tabella $N \times N$: sulla diagonale gli abbinamenti giusti, in tutte le
altre caselle quelli sbagliati. Il disegno a sinistra è uno schema: quanto le due
frecce siano davvero vicine lo misureremo più avanti, ed è meno di quel che
sembra.
```

La {numref}`fig-vlm-contrastivo` mostra la struttura che ne esce, ed è tutta la
sezione in un disegno. Le due torri lavorano indipendenti, non si scambiano
niente durante il calcolo, e si incontrano solo alla fine, in un prodotto
scalare. Da un batch di $N$ coppie nasce una matrice $N \times N$ di
similarità: sulla diagonale le coppie vere, ovunque altrove i **negativi**,
cioè gli abbinamenti sbagliati che il caso ha messo insieme nello stesso batch.

## L'esame si fa in due sensi

A questo punto serve una **funzione di costo** che dica al modello che cosa fare
di quella tabella (in inglese si chiama *loss*, ed è il nome che si sente più
spesso; in questa sezione le due parole indicano la stessa cosa). La richiesta è
semplice da enunciare: la diagonale in alto, tutto il resto in basso. Il modo di
ottenerlo è la vecchia conoscenza di questo libro, la **cross-entropy**, cioè il
modo di misurare quanto si paga caro sbagliare una domanda a risposta multipla,
applicata qui a una domanda a risposta multipla che il batch costruisce da solo.

`````{tab} Elementare

Guarda la griglia della figura una riga alla volta. La prima riga è
un'interrogazione a risposta multipla: «ecco l'immagine numero uno, quale delle
quattro didascalie è la sua?». Il modello risponde con quattro numeri, e la
risposta giusta è sempre la prima cella, quella sulla diagonale. Il costo
misura quanto la risposta giusta è stata considerata probabile: se il modello
le dà il 90% di fiducia paga pochissimo, se le dà il 25% (come tirando a caso
fra quattro) paga parecchio.

Poi si rifà lo stesso identico esame guardando le **colonne**: «ecco la
didascalia numero uno, quale delle quattro immagini descrive?». Le due
interrogazioni non sono la stessa cosa, perché una didascalia potrebbe essere
la più adatta a una foto senza che quella foto sia la più adatta a lei. Si
fanno entrambe e si fa la media: da qui l'aggettivo **simmetrica** che si
attacca a questa loss.

`````

`````{tab} Superiore

La forma generale è la **InfoNCE**, introdotta da van den Oord e colleghi per
il contrastive predictive coding {cite}`oord2018representation`:

$$
\mathcal{L}_{\text{InfoNCE}} = - \,\mathbb{E}\!\left[\,
\log \frac{\exp\big(s(\mathbf{u}, \mathbf{v}^{+})/\tau\big)}
{\sum_{k=1}^{N} \exp\big(s(\mathbf{u}, \mathbf{v}_k)/\tau\big)} \right],
$$

dove $\mathbf{u}$ è l'ancora, $\mathbf{v}^{+}$ il suo positivo,
$\mathbf{v}_1, \dots, \mathbf{v}_N$ l'insieme dei
candidati (il positivo più $N-1$ negativi), $s(\cdot, \cdot)$ una misura di
compatibilità e $\tau > 0$ la **temperatura**. È, letteralmente, una
cross-entropy su un problema di classificazione a $N$ vie in cui la classe
corretta è «il positivo». (Nel testo originale la compatibilità è una funzione
di punteggio qualsiasi; la $\tau$ esplicita è della variante su similarità
coseno, quella che CLIP adotta, e che qui useremo sempre.)

In CLIP l'ancora è un embedding di immagine, i candidati sono le $N$ didascalie
del batch e la compatibilità è il coseno. Per la direzione immagine → testo:

$$
\ell^{\,\mathrm{I}\to\mathrm{T}}_i = - \log
\frac{\exp\big(\langle \mathbf{I}_i, \mathbf{T}_i \rangle / \tau\big)}
{\sum_{j=1}^{N} \exp\big(\langle \mathbf{I}_i, \mathbf{T}_j \rangle / \tau\big)},
$$

e simmetricamente, scorrendo la colonna $i$ invece della riga $i$, per la
direzione testo → immagine:

$$
\ell^{\,\mathrm{T}\to\mathrm{I}}_i = - \log
\frac{\exp\big(\langle \mathbf{I}_i, \mathbf{T}_i \rangle / \tau\big)}
{\sum_{k=1}^{N} \exp\big(\langle \mathbf{I}_k, \mathbf{T}_i \rangle / \tau\big)}.
$$

La loss finale è la media delle due:

$$
\mathcal{L} = \frac{1}{2N} \sum_{i=1}^{N}
\Big( \ell^{\,\mathrm{I}\to\mathrm{T}}_i + \ell^{\,\mathrm{T}\to\mathrm{I}}_i \Big).
$$

Qui $N$ è la dimensione del batch, $\mathbf{I}_i$ e $\mathbf{T}_j$ gli embedding
normalizzati,
$\langle \mathbf{I}_i, \mathbf{T}_j \rangle$ la loro similarità coseno e $\tau$ la temperatura. Si
noti che il numeratore è lo stesso nelle due direzioni (la coppia vera $(i,i)$)
e a cambiare è solo l'insieme rispetto a cui si normalizza: le didascalie a
parità di immagine, oppure le immagini a parità di didascalia. I gradienti
alzano il coseno della diagonale e abbassano quelli fuori diagonale, con
un'intensità che dipende da quanto ciascun negativo è già vicino: è la
proprietà, tipica della softmax, di occuparsi soprattutto dei concorrenti
credibili.

`````

## Quattro coppie, fatte a mano

Conviene vedere i numeri, perché la temperatura fa una differenza che a parole
non si apprezza. Prendiamo un batch minuscolo, $N = 4$: quattro immagini e le
loro quattro didascalie. Nella tabella qui sotto le righe $\mathbf{I}_1 \dots \mathbf{I}_4$ sono
le quattro immagini, le colonne $\mathbf{T}_1 \dots \mathbf{T}_4$ le quattro didascalie, e ogni
cella dice quanto quell'immagine e quella didascalia si somigliano, su una scala
che va da $-1$ (agli antipodi) a $+1$ (nello stesso punto esatto); in
grassetto le quattro coppie vere. I valori sono plausibili per un modello a
metà addestramento (le coppie vere intorno a $0{,}3$, le altre fra $0$ e
$0{,}15$):

| somiglianza | $\mathbf{T}_1$ | $\mathbf{T}_2$ | $\mathbf{T}_3$ | $\mathbf{T}_4$ |
|---|---|---|---|---|
| $\mathbf{I}_1$ | **0,30** | 0,10 | 0,05 | 0,02 |
| $\mathbf{I}_2$ | 0,08 | **0,28** | 0,12 | 0,04 |
| $\mathbf{I}_3$ | 0,04 | 0,15 | **0,32** | 0,09 |
| $\mathbf{I}_4$ | 0,06 | 0,03 | 0,10 | **0,26** |

`````{tab} Elementare

Guarda la prima riga: la coppia giusta somiglia $0{,}30$, la migliore delle
sbagliate $0{,}10$. Differenze piccole, e il mestiere della **temperatura** è
decidere quanto pesano: è una manopola che amplifica le differenze fra i
punteggi prima di trasformarli in percentuali di fiducia, e più è bassa, più
amplifica.

Con la temperatura di partenza di CLIP, che è bassa ($0{,}07$), quel piccolo
vantaggio viene ingigantito, e i passaggi si possono seguire con una
calcolatrice. Primo: si divide ogni somiglianza per la temperatura, cioè la si
moltiplica per quattordici e rotti. La riga diventa $4{,}3$, poi $1{,}4$, $0{,}7$
e $0{,}3$. Secondo: quei numeri si trasformano in fiducia con un'operazione che
gonfia i grandi molto più dei piccoli, e $4{,}3$ diventa $72{,}7$ mentre $1{,}4$
diventa appena $4{,}2$ (poi $2{,}0$ e $1{,}3$). Terzo: si guarda che fetta è
ciascuno del totale, che è $80{,}2$: alla coppia giusta va $72{,}7$ su $80{,}2$,
cioè il **91%** della fiducia.

Il costo della riga è tanto più basso quanto più alta è quella fetta: al 91% vale
$0{,}099$, e se il modello tirasse a caso, dando il 25% a ciascuna delle quattro,
varrebbe $1{,}386$. Facendo la media sulle quattro righe, e poi anche sulle
colonne, il costo complessivo è $0{,}148$.

Ora alziamo la manopola a $0{,}5$, senza toccare una sola somiglianza.
L'amplificazione quasi sparisce: alla coppia giusta va il 35% della fiducia e
alle tre sbagliate poco meno, fra il 20 e il 24. Il costo sale a $1{,}082$;
per confronto, tirare a caso fra quattro didascalie costerebbe $1{,}386$.
Stessa tabella, stesso ordine corretto: con la manopola alta si paga quasi
quanto tirando a caso, con quella bassa un decimo.

`````

`````{tab} Superiore

Con la temperatura di partenza di CLIP, $\tau = 0{,}07$, le similarità coseno
della prima riga diventano logit dividendo per $\tau$:
$0{,}30/0{,}07 = 4{,}29$, poi $1{,}43$, $0{,}71$ e $0{,}29$. Esponenziando si
ottengono $72{,}7$, $4{,}17$, $2{,}04$ e $1{,}33$, la cui somma è $80{,}2$; le
probabilità sono quindi $0{,}906$, $0{,}052$, $0{,}025$ e $0{,}017$, e il
costo della riga è $-\log 0{,}906 = 0{,}099$ (i logaritmi qui sono naturali,
come vuole la forma esponenziale della softmax). Ripetendo per le altre tre
righe e mediando, la loss in direzione immagine → testo vale $0{,}147$; quella
sulle colonne $0{,}148$; la loss simmetrica $0{,}148$.

Ora rifacciamo il conto **senza cambiare una sola similarità**, solo alzando la
temperatura a $\tau = 0{,}5$. La prima riga diventa $0{,}351$, $0{,}235$,
$0{,}213$, $0{,}201$: la coppia giusta è ancora in testa, ma di un soffio, e la
loss simmetrica sale a $1{,}082$. Per confronto, un modello che tirasse a caso
fra quattro didascalie pagherebbe $\log 4 = 1{,}386$. Con $\tau = 0{,}5$ questa
matrice, che pure è ordinata correttamente, costa quasi quanto tirare a caso;
con $\tau = 0{,}07$ ne costa circa un decimo.

`````

## Perché la temperatura e il batch non sono dettagli

Quel confronto dice una cosa importante: la temperatura non è un parametro
cosmetico, decide *quanto* piccole differenze di somiglianza diventino grandi
differenze di fiducia, e quindi che cosa il modello si sforzi di
correggere.

`````{tab} Elementare

La temperatura è la severità dell'esaminatore. Un esaminatore mite (temperatura
alta) dà a tutte le risposte voti simili: passi anche se la tua risposta giusta
era di poco davanti alle altre, e quindi non hai motivo di migliorare. Un
esaminatore severo (temperatura bassa) amplifica ogni differenza: se la
didascalia giusta è appena più vicina di una sbagliata, per lui è quasi un
errore, e il modello viene spinto ad allargare quel margine. Nei conti di
prima, lo stesso identico compito costava $0{,}15$ con l'esaminatore severo e
$1{,}08$ con quello mite.

La cosa curiosa è che questa severità non la sceglie chi progetta: è un numero
che il modello **impara** insieme a tutto il resto, come i pesi. E c'è un
secondo ingrediente altrettanto poco appariscente: quante didascalie sbagliate
ci sono nel mucchio. Indovinare fra quattro è facile, e un modello che sbaglia
poco impara poco. Indovinare fra trentamila è tutta un'altra cosa, e trentamila
è esattamente l'ordine di grandezza che CLIP usa: il batch non è una scelta di
ingegneria, è la difficoltà dell'esame.

`````

`````{tab} Superiore

In CLIP $\tau$ è **appresa**. In pratica il parametro ottimizzato è il
logaritmo del fattore di scala $1/\tau$, così che la scala resti positiva senza
vincoli espliciti; lo si inizializza al valore corrispondente a $\tau = 0{,}07$
e si impedisce alla scala di superare $100$, perché l'ottimizzazione tenderebbe
altrimenti a farla crescere senza freno (una temperatura che tende a zero rende
la loss arbitrariamente piccola sulle coppie già ordinate bene, e instabile il
gradiente: nel lavoro originale il tetto è motivato proprio dall'instabilità
osservata in addestramento). L'effetto di $\tau$ sulla distribuzione dei pesi è
quello visto nei conti: al calare della temperatura la softmax si fa più
**piccata** e la penalità si concentra sui negativi difficili, quelli con
coseno vicino a quello del positivo. Vale la pena aggiungere dove il parametro
va a finire, perché il $0{,}07$ è un punto di partenza e non un regime di
esercizio: nel modello pubblicato la scala appresa sta **appoggiata al tetto**,
cioè $\tau = 1/100 = 0{,}01$, sette volte più piccata di come è partita.
L'ottimizzazione, lasciata libera, va a sbattere contro il vincolo e ci resta; ci
servirà fra poco, quando si tratterà di capire perché due nuvole di punti non si
avvicinano mai.

Il secondo parametro strutturale è $N$. Il denominatore della InfoNCE somma sui
candidati del batch: i negativi *sono* il batch, non un insieme costruito a
parte. Con $N$ piccolo il compito è banale (la baseline casuale è $\log N$, e
con $N = 4$ vale $1{,}39$) e il segnale di apprendimento è povero; al
crescere di $N$ il compito diventa un ago in un pagliaio e il gradiente informa
molto di più. CLIP addestra con batch da $32\,768$ coppie, distribuiti su
centinaia di GPU. Il prezzo è la struttura stessa della loss: la matrice di
similarità è $N \times N$, il suo costo cresce con il quadrato del batch, e la
normalizzazione della softmax richiede che ogni riga veda *tutte* le colonne,
quindi che gli embedding di tutti i dispositivi vengano radunati insieme a ogni
passo. Torneremo su questo punto fra poco, perché è esattamente il vincolo che
una variante successiva scioglie.

`````

## La loss in dieci righe

Tradotta in PyTorch, tutta la sezione sta in una funzione. Gli embedding
arrivano dai due encoder come due matrici $(N, d)$; il resto è normalizzazione,
un prodotto matriciale e due cross-entropy.

```python
import torch
from torch import nn
import torch.nn.functional as F

# la temperatura si impara: il parametro e' log(1/tau), cosi' la scala,
# che si ottiene esponenziando, e' positiva per costruzione
logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())


def loss_contrastiva(emb_img, emb_txt, logit_scale):
    """emb_img, emb_txt: due tensori (N, d), una riga per elemento del batch."""
    # 1. sulla sfera unitaria: il prodotto scalare diventa un coseno
    I = F.normalize(emb_img, dim=-1)
    T = F.normalize(emb_txt, dim=-1)

    # 2. matrice N x N dei coseni, riscalata dalla temperatura (con il tetto)
    scala = logit_scale.exp().clamp(max=100.0)
    logits = scala * (I @ T.t())

    # 3. la risposta giusta e' sempre sulla diagonale: 0, 1, 2, ... N-1
    bersagli = torch.arange(len(I), device=I.device)

    # 4. una cross-entropy sulle righe, una sulle colonne, e si media
    perdita_i2t = F.cross_entropy(logits, bersagli)
    perdita_t2i = F.cross_entropy(logits.t(), bersagli)
    return (perdita_i2t + perdita_t2i) / 2


# Gli stessi conti fatti a mano poco fa, rifatti dalla libreria: si parte
# direttamente dalla tabella delle somiglianze, saltando i due encoder.
somiglianze = torch.tensor([[0.30, 0.10, 0.05, 0.02],
                            [0.08, 0.28, 0.12, 0.04],
                            [0.04, 0.15, 0.32, 0.09],
                            [0.06, 0.03, 0.10, 0.26]])
bersagli = torch.arange(4)
for tau in (0.07, 0.5):
    logits = somiglianze / tau
    perdita = (F.cross_entropy(logits, bersagli)
               + F.cross_entropy(logits.t(), bersagli)) / 2
    print(f"tau = {tau}: loss simmetrica = {perdita:.3f}")
# tau = 0.07: loss simmetrica = 0.148
# tau = 0.5: loss simmetrica = 1.082
```

Le ultime righe rifanno i conti della tabella di poco fa: la stessa matrice, la
stessa loss, solo la temperatura cambiata. Quei due numeri, $0{,}148$ e
$1{,}082$, si possono così ritrovare invece che crederli sulla parola.

Vale la pena notare che cosa *non* c'è: nessuna etichetta, nessun numero di
classi, nessuna testa di classificazione. L'unica informazione supervisionata è
l'ordine delle righe, cioè il fatto che la didascalia $i$ stava sotto
l'immagine $i$.

## Un solo spazio, due quartieri

Conviene tornare sulla parola «vicino», perché presa alla lettera inganna. Fin
qui si è detto che una foto e la sua didascalia finiscono vicine sulla mappa.
Vicine quanto? La risposta si misura in dieci righe di codice, e non è quella che
ci si aspetta.

`````{tab} Elementare

Prendiamo ottanta fotografie, otto per ciascuno di dieci soggetti (aerei,
gatti, cavalli, navi e così via), più quaranta didascalie, diamole a un modello
CLIP pubblico e misuriamo tutte le vicinanze. Viene fuori questo: una fotografia
somiglia alla didascalia che il modello stesso sceglie per lei circa $0{,}27$, e
a una **qualunque altra fotografia** circa $0{,}76$. Cioè ogni foto è molto più
vicina a una foto che non c'entra niente che alla frase che la descrive.

Non è un guasto, e non impedisce al meccanismo di funzionare: sulle stesse
ottanta immagini, con quel compito facile a dieci categorie, il classificatore
scritto a parole indovina quasi nove volte su dieci. Funziona perché il
confronto che conta non è mai «foto contro frase, in assoluto»: è sempre
«questa foto, con quale delle dieci frasi va meglio?». Fra le frasi la
graduatoria è giusta, ed è tutto quello che serve.

La mappa, insomma, è una sola, ma ci sono due quartieri: le fotografie da una
parte, le frasi dall'altra, e i due gruppi non si toccano mai. Basta una riga
tracciata una volta sola per dire di ogni punto, senza sbagliarne nemmeno uno su
centoventi, se è una foto o una frase. L'addestramento non ha mai chiesto ai due
quartieri di mescolarsi: ha chiesto che, **dentro** al quartiere delle frasi,
quella giusta stesse davanti a tutte le altre. E quello lo ottiene benissimo.

Ne segue una regola pratica che vale la pena ricordare: il numero di somiglianza
fra una foto e una frase non si confronta con quello fra due foto. Sono due
righelli con lo zero in posti diversi, e chi fissa una soglia guardando i secondi
e la applica ai primi sbaglia tutte le volte.

`````

`````{tab} Superiore

Il fenomeno ha un nome, **modality gap**, e una descrizione sistematica in Liang
e colleghi {cite}`liang2022mind`, che trovano le due modalità immerse «a
distanza di braccio» nello spazio che condividono. Il conto qui sotto è rifatto
in casa, e vale la pena scrivere con che cosa, perché senza il protocollo un
numero non si può controllare: `clip-vit-base-patch32`, ottanta fotografie di
CIFAR-10 (otto per ciascuna delle dieci classi) e quaranta didascalie generiche,
quattro per classe sullo stampo `a photo of a {classe}`. La distanza fra i due
centroidi vale $\approx 1{,}1$ (gli embedding stanno sulla
sfera unitaria, dove il massimo possibile è $2$); il coseno medio della coppia
migliore è $\approx 0{,}27$ contro $\approx 0{,}76$ fra due immagini qualunque;
e proiettando tutto sulla direzione che unisce i due centroidi le due nuvole non
si sovrappongono per niente, tanto che una regressione logistica risponde
«immagine o testo?» con accuratezza $1{,}000$ in validazione incrociata. Sulle
stesse ottanta immagini, e con un solo prompt per classe, la classificazione
zero-shot a dieci vie ne prende $0{,}875$: il divario non le impedisce di
funzionare, ed è il punto.
I decimali dipendono dalle scelte appena elencate, e in particolare quel
$0{,}76$ è alto perché le immagini di CIFAR-10 sono $32 \times 32$ e si
somigliano fra loro più di quanto si somiglino fotografie a piena risoluzione:
su queste il valore scende, senza che la forbice si chiuda. L'ampiezza del
divario dipende dunque dal modello e dai dati; la sua esistenza no.

Che non si chiuda non è un difetto dell'ottimizzazione: è ciò che
l'ottimizzazione chiede. La InfoNCE dipende soltanto dai **rapporti** fra le
similarità di una riga, quindi è insensibile a qualunque spostamento in blocco
di una delle due nuvole che non cambi l'ordinamento; il minimo non è «le due
nuvole sovrapposte», è «dentro ogni riga, la coppia vera davanti alle altre». Il
divario nasce per giunta già all'inizializzazione (è l'**effetto cono**: una rete
profonda non addestrata concentra le proprie uscite in un cono stretto, e due
reti diverse danno due coni diversi), e la temperatura bassa di cui si è parlato
lo difende invece di chiuderlo: forzando a mano la sovrapposizione, alla
temperatura originale la perdita *aumenta*.

Due conseguenze per chi costruisce. La prima: coseni cross-modali e coseni
intra-modali vivono su scale diverse, non si confrontano fra loro e non si
mescolano in un'unica soglia. La seconda: le operazioni che presuppongono uno
spazio omogeneo (il centroide fra un'immagine e un testo, un $k$-means su
vettori misti, una soglia assoluta di appartenenza) restituiscono risultati che
sembrano sensati e non lo sono. Quel che è lecito, ed è quanto basta a tutto il
resto della sezione, è l'$\arg\max$ **dentro** una modalità sola.

`````

## Il classificatore che si scrive a parole

Finito l'addestramento, il modello sa fare una cosa sola: dire quanto
un'immagine e un testo si somigliano. Ma quella cosa sola, usata bene, produce
un classificatore che nessuno ha addestrato.

`````{tab} Elementare

Vuoi distinguere gatti, cani e tram? Non serve raccogliere foto né riaddestrare
niente. Scrivi tre frasi: «una foto di un gatto», «una foto di un cane», «una
foto di un tram». Le passi all'encoder di testo, che ti dà tre file di numeri.
Passi la tua immagine all'encoder di immagini, che te ne dà una. Guardi a quale
delle tre è più vicina, cioè calcoli quel numero fra $-1$ e $+1$ tre volte e
tieni la volta in cui è più alto, e hai la risposta. Se domani ti serve anche «una foto di un
vaporetto», aggiungi una riga di testo: il tuo classificatore è cresciuto di
una classe in un secondo, senza una sola immagine di vaporetto.

Questo si chiama **zero-shot**, «a zero esempi», e non è una funzione in più
che qualcuno ha aggiunto: è la stessa identica operazione di prima, l'abbinare,
usata con didascalie che ti sei scritto da solo. Il fenomeno che ha colpito
tutti nel 2021 è che il classificatore scritto a parole, senza aver visto
nemmeno una delle immagini etichettate di ImageNet (sono 1,28 milioni), ci
prendeva quanto la ResNet-50 che su quelle immagini si era addestrata.

C'è però una stranezza da conoscere: **come** scrivi la frase cambia il
risultato. «Una foto di un gatto» funziona meglio della sola parola «gatto», e
la ragione è che il modello ha imparato dalle didascalie del web, che sono
frasi, non parole isolate; presentargli una parola secca è come parlargli in una
lingua un po' diversa da quella su cui si è allenato. E poi c'è l'ambiguità:
«gru» da sola può essere l'uccello o la macchina da cantiere, mentre «una foto
di una gru, l'uccello» chiude la questione. Sistemare la frase vale, su
ImageNet, poco più di un punto percentuale.

Il secondo trucco sta nel non fidarsi di una formulazione sola.
Della stessa classe si scrivono ottanta frasi diverse («una foto di un gatto»,
«un primo piano di un gatto», «una foto sfocata di un gatto»), si fa la media
delle ottanta file di numeri e si usa quella: le stranezze di ciascuna
formulazione si annullano a vicenda e resta quello che le ottanta hanno in
comune, cioè il concetto. Vale altri tre punti e mezzo, e non costa niente,
perché la media si fa una volta sola e prima di guardare qualunque fotografia.
Non è un dettaglio da rifinitura.

`````

`````{tab} Superiore

Dato un insieme di classi candidate $\{c_1, \dots, c_K\}$, si costruisce per
ciascuna un prompt (per esempio `una foto di un {c_k}`), lo si passa
nell'encoder di testo e si normalizza, ottenendo $\mathbf{T}_1, \dots, \mathbf{T}_K$. La
predizione per un'immagine con embedding $\mathbf{I}$ è

$$
\hat{y} = \arg\max_{k \in \{1, \dots, K\}} \; \langle \mathbf{I}, \mathbf{T}_k \rangle .
$$

L'osservazione strutturale, fatta nel paper originale
{cite}`radford2021learning`, è che questa è **letteralmente** una
classificazione lineare: la matrice $[\mathbf{T}_1; \dots; \mathbf{T}_K] \in \mathbb{R}^{K \times d}$
è una matrice di pesi, e l'encoder di testo si comporta come una rete che
*genera* i pesi del classificatore a partire da una descrizione, invece di
stimarli per discesa del gradiente su esempi etichettati. Cambiare l'insieme
delle classi significa rigenerare quella matrice, un'operazione che costa una
forward pass per classe.

Due fenomeni rendono la scelta del prompt non neutrale. Il primo è la
**polisemia**: un'etichetta isolata non disambigua i suoi sensi (l'italiano
«gru» copre l'uccello e la macchina da cantiere, e nei dataset di visione casi
simili sono la norma), mentre un contesto testuale lo fa. Il secondo è uno
**scarto di distribuzione**: nel corpus di pre-addestramento il testo appaiato
a un'immagine è quasi sempre una frase, quindi un input costituito da una sola
parola cade in una regione poco frequentata dello spazio testuale. Il rimedio,
un template fisso come `A photo of a {label}.`, vale nel paper originale un
guadagno di $1{,}3$ punti su ImageNet, e mediare gli embedding di ottanta
template diversi (una forma di ensembling che, essendo fatta sui vettori e non
sulle predizioni, non costa nulla in inferenza) ne aggiunge altri $3{,}5$.

La stessa geometria dà il **recupero cross-modale**: si indicizzano gli
embedding di un archivio di immagini e si interroga l'indice con l'embedding di
una frase, prendendo i $k$ più vicini; oppure il contrario, cercando la
didascalia più adatta a un'immagine. Ricerca semantica di immagini,
deduplicazione, filtraggio di corpora enormi: sono tutti lo stesso prodotto
scalare. E il text encoder così addestrato è riusabile altrove: è lui,
congelato, a tradurre il prompt in vettori dentro Stable Diffusion, come
vedremo nel capitolo sui modelli di diffusione.

`````

## Una sigmoide al posto della softmax

Il vincolo lasciato in sospeso poco fa nasce tutto dalla forma dell'esame. Per
dare le percentuali di una riga bisogna avere sotto gli occhi la riga intera,
cioè tutte le didascalie del gruppo; e se il gruppo è spalmato su duecento schede
grafiche, ogni passo di addestramento comincia radunando i risultati di tutte e
finisce ridistribuendoli. È un costo che cresce con il batch, proprio mentre il
metodo chiede batch grandi.

SigLIP {cite}`zhai2023sigmoid` cambia una cosa sola, e la cambia alla radice:
smette di trattare la riga come una domanda a risposta multipla e tratta ogni
casella come una domanda a sé, con risposta sì o no.

`````{tab} Elementare

Invece di «ecco l'immagine numero uno, quale delle quattro didascalie è la sua?»,
a ogni casella della tabella si fa una domanda indipendente: «voi due andate
insieme, sì o no?». Sedici domandine al posto di quattro interrogazioni.

Il guadagno è che per rispondere a una non serve sapere niente delle altre.
Nessuno deve più radunare la riga intera, il lavoro si può spezzare in pezzi che
viaggiano per conto proprio, e soprattutto cade l'obbligo del mucchio enorme:
l'esame a scelta multipla, per essere difficile, il mucchio grande lo pretendeva;
una domanda sì-o-no è difficile da sola.

Un guaio però c'è, ed è di proporzioni. In una tabella di quattro per quattro le
caselle da «sì» sono quattro e quelle da «no» dodici; con quattromila immagini
per volta diventano quattromila «sì» contro quasi sedici milioni di «no». Chi
rispondesse «no» a tutto avrebbe quasi sempre ragione senza aver imparato niente,
e le prime ore di addestramento se ne andrebbero tutte a scoprire questa
sciocchezza. Il rimedio è dirgliela in partenza: si regala al modello la
conoscenza che «no» è la risposta di gran lunga più frequente, così il tempo lo
può spendere sul resto.

`````

`````{tab} Superiore

«Questa immagine e questa didascalia vanno insieme?» è un problema di
classificazione binaria, e la funzione che gli corrisponde non è la softmax ma la
**sigmoide**:

$$
\mathcal{L}_{\text{sig}} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{N}
\log \sigma\!\Big( z_{ij} \big( t \, \langle \mathbf{I}_i, \mathbf{T}_j \rangle + b \big) \Big),
\qquad
z_{ij} = \begin{cases} +1 & i = j \\ -1 & i \neq j \end{cases}
$$

dove $\sigma$ è la funzione logistica, $t$ il fattore di scala appreso (lo
stesso ruolo di $1/\tau$, parametrizzato anche qui come esponenziale di un
parametro libero), $b$ un bias appreso e $z_{ij}$ l'etichetta binaria della
cella. Si noti la normalizzazione: la somma corre su tutte le $N^2$ celle, ma
il divisore è $N$, così che la loss conti il costo per elemento del batch e non
per cella. Il bias serve a un problema che la softmax non aveva: in un batch le
celle negative sono $N^2 - N$ contro $N$ positive, uno sbilanciamento feroce,
e senza un $b$
inizializzato molto negativo (nel lavoro originale a $-10$) le prime iterazioni
si consumerebbero tutte a correggerlo, invece che a imparare.[^siglip-segno]

[^siglip-segno]: Chi va a controllare sull'articolo troverà l'equazione
    stampata con il segno della similarità rovesciato rispetto a questa. La
    forma scritta qui è quella dello pseudocodice degli autori e
    dell'implementazione di riferimento, ed è l'unica compatibile con la
    motivazione che loro stessi danno per $b = -10$, cioè che i negativi
    partano a costo quasi nullo.

`````

La conseguenza pratica è che ogni pezzo del conto, cioè ogni casella, dipende da
una coppia sola. Non c'è più niente da normalizzare su tutto il batch, il
calcolo si può spezzare in blocchi che si scambiano gli embedding a turno, e
soprattutto la qualità dell'addestramento smette di dipendere dall'avere un
batch enorme. Sul
proprio impianto gli autori misurano due soglie: sotto le sedicimila coppie la
sigmoide stacca la softmax di parecchio, e oltre le trentaduemila nessuna delle
due guadagna più molto. Sono i numeri di quelle prove, non costanti di natura, e
a un altro modello su altri dati verranno diversi; quello che non dipende dai
numeri è la direzione, cioè che alla dimensione del batch viene tolto il ruolo
di prerequisito. È lo stesso allineamento, ottenuto togliendo un vincolo invece
di aggiungere un pezzo.

Vale la pena registrare anche un risultato di metodo, arrivato negli stessi mesi
di CLIP: ALIGN {cite}`jia2021scaling` ha addestrato la stessa architettura a due
torri su oltre un miliardo di coppie immagine-testo raccolte senza costosi
passaggi di filtraggio, praticamente il web così com'è. Il messaggio, che sono
gli autori stessi a formulare, è che la scala del corpus compensa il suo
rumore: la curatela dei dataset di visione non è un prerequisito del metodo.

## Uno spazio allineato non è uno spazio che capisce

Qui finisce la parte in cui tutto funziona meglio del previsto, e comincia
quella che spiega perché il resto di questo capitolo esiste.

Torniamo alla frase che apriva la sezione: «un gatto nero che salta sul muro».
Un modello contrastivo la riconosce benissimo se la foto contiene un gatto, del
nero e un muro. Ma proviamo a chiedergli di distinguere «il gatto sotto il
tappeto» da «il tappeto sotto il gatto», o «il gatto insegue il cane» da «il
cane insegue il gatto»: le due frasi contengono le stesse identiche parole, e
le due immagini gli stessi oggetti. È qui che il meccanismo mostra il fondo.

Il fenomeno è stato reso visibile da **Winoground**
{cite}`thrush2022winoground`, un
insieme di quattrocento esempi costruiti a mano apposta: due immagini e due
didascalie fatte esattamente delle stesse parole in ordine diverso, con il
compito di appaiarle correttamente. Si misura in tre modi: scegliere la
didascalia giusta per ciascuna delle due immagini, scegliere l'immagine giusta
per ciascuna delle due didascalie, e riuscire in tutte e quattro le scelte
insieme. Le prime due misure chiedono di indovinare *entrambe* le volte fra due
possibilità, quindi tirando a caso si prende il 25%; la terza chiede di
ordinare correttamente quattro punteggi, e a caso si prende un sesto, cioè il
16,7%. Il risultato dello studio, enunciato dagli autori stessi, è che nessuno
dei modelli provati fa molto meglio del caso; sulle due misure più difficili
sono tutti *sotto* il livello del caso. (Sulla prima delle tre, la più facile,
qualcuno il caso lo stacca, ed è l'unica in cui succede: chi va a guardare la
tabella dello studio trova una colonna che sembra smentire la frase, e sono le
altre due a contare.) Non è una classifica fra prodotti: è la misura di un
limite che riguarda la famiglia.

La ragione è strutturale, e sta nel gioco stesso che abbiamo descritto: il
modello non è addestrato a *descrivere* un'immagine, è addestrato a
*distinguere* la sua didascalia dalle altre del gruppo, che sono didascalie di
immagini prese a caso. Per vincere, quasi sempre, basta indovinare quali oggetti
compaiono nella foto: se le altre parlano di un tramonto, di una bicicletta e di
una scodella di minestra, riconoscere «gatto» e «muro» è più che sufficiente, e
capire *chi sta sopra chi* non porta nessun vantaggio. La strada più economica
verso un costo basso è trattare la didascalia come un **sacco di concetti**, e
l'ottimizzazione, che è pigra per mestiere, la prende. La sintassi, le
relazioni spaziali, il conteggio, la negazione (una didascalia con «senza» resta
vicinissima alla stessa senza il «senza») sono i primi a rimanere fuori.

Che sia davvero il gioco a produrre quel comportamento, e non un difetto delle
reti, lo si è dimostrato con un esperimento di una semplicità disarmante
{cite}`yuksekgonul2023when`: si prendono le didascalie di un archivio, se ne
**mescolano le parole**, si rifà la ricerca per immagini, e il risultato non
peggiora. Se l'ordine si può buttare via senza pagare pegno, l'ordine il compito
non lo chiedeva. Lo stesso lavoro mostra anche il rovescio, che è la parte utile:
aggiungendo al mucchio, come didascalie sbagliate, delle **versioni permutate
della didascalia giusta**, la stessa identica rete impara l'ordine. Il limite
non era dell'architettura, era di quello che le si chiedeva di distinguere.

Due precisazioni, per onestà. La prima è che quegli esempi, scelti a mano
perché siano difficili, lo sono anche per altre ragioni (alcuni chiedono
conoscenza del mondo, altri sono visivamente ostici), e quindi misurano la
composizionalità insieme a qualcos'altro: il fenomeno è solido, la sua
quantificazione esatta lo è meno. La seconda è che un limite parallelo viene
dalla forma della rappresentazione: un'intera immagine finisce in **un solo**
vettore da qualche centinaio di dimensioni, e un vettore solo non può portare
insieme la scena, la posizione di ogni oggetto e il testo scritto su un
cartello. Il dettaglio fine e i documenti sono un problema a parte, e li
affronta la sezione sulla risoluzione.

Da qui in avanti il capitolo prova a superare entrambi i limiti nello stesso
modo: smettere di chiedere a un prodotto scalare di rappresentare la relazione
fra un'immagine e una frase, e mettere al suo posto un modello di linguaggio che
*legge* l'immagine token per token. Come si innestano gli occhi su un modello
che sa solo leggere è la prossima sezione.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un classificatore a elenco chiuso sa dire soltanto le voci del suo elenco, e
  aggiungerne una costa foto nuove, etichette nuove e un addestramento nuovo. Il
  gioco delle quattro foto e delle quattro didascalie da appaiare sostituisce la
  domanda «che cosa è questo?» con «quale di queste didascalie è la sua?».
- Nessuno prepara le risposte: le didascalie sono già attaccate alle immagini del
  web, e per addestrare CLIP ne sono state raccolte quattrocento milioni.
- Il gioco si fa in due sensi, per righe e per colonne, e si fa la media. Due
  numeri decidono quanto è severo: la **manopola** che amplifica le differenze
  fra le somiglianze prima di trasformarle in percentuali, e **quante didascalie
  sbagliate ci sono nel mucchio**, perché indovinare fra quattro è facile e
  indovinare fra trentamila no.
- Il regalo che ne esce: per costruire un classificatore bastano tre frasi
  scritte a mano, e domani la quarta si aggiunge in un secondo, senza una sola
  fotografia. Conta però **come** si scrive la frase, e conviene scriverne
  ottanta e fare la media.
- Nella variante «sì o no» si smette di chiedere «quale di queste quattro» e si
  chiede a ogni casella «voi due andate insieme?»: nessuno deve più radunare
  tutta la riga, e il mucchio enorme non serve più.
- Immagini e parole finiscono sulla stessa mappa, ma in **due quartieri
  separati**: la coppia giusta è più vicina di tutte le altre coppie, e questo
  basta a farla vincere, ma non è mai vicina quanto due fotografie fra loro. I
  numeri si confrontano fra pari, mai una foto con una frase in assoluto.
- Appaiare non è capire: al gioco si vince riconoscendo gli oggetti, quindi «il
  gatto sotto il tappeto» e «il tappeto sotto il gatto» restano indistinguibili.
  È il motivo per cui esistono le sezioni che seguono.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un classificatore chiuso sa dire solo le classi su cui è stato addestrato, e
  aggiungerne una costa rietichettatura e riaddestramento. L'addestramento
  **contrastivo** immagine-testo {cite}`radford2021learning` sostituisce la
  domanda «che cosa è» con «quale di queste didascalie è la sua».
- La supervisione è **naturale**: le coppie immagine-didascalia esistono già
  sul web (quattrocento milioni per CLIP), nessuno le etichetta e nessuna lista
  di categorie viene decisa in anticipo.
- La loss è una **InfoNCE simmetrica** {cite}`oord2018representation`:
  embedding normalizzati, matrice $N \times N$ di coseni divisi per la
  temperatura $\tau$, cross-entropy sulle righe e sulle colonne con la diagonale
  come risposta corretta.
- $\tau$ e $N$ sono parte del metodo, non dell'implementazione: la temperatura
  (appresa, e a fine addestramento appoggiata al tetto, $\tau = 0{,}01$) decide
  quanto la distribuzione è piccata, e i negativi vengono dal batch, quindi un
  batch piccolo rende il compito troppo facile. A crescere con $N^2$ non è il
  calcolo, che accanto ai due encoder è trascurabile, ma la **memoria** della
  matrice e l'**all-gather** fra i dispositivi.
- Le due modalità restano in **due regioni disgiunte** dello spazio condiviso, il
  *modality gap* {cite}`liang2022mind`: il contrastivo ottimizza un ordinamento
  dentro il batch, che è invariante per traslazione di una delle due nuvole.
  Conseguenza operativa: un coseno cross-modale non si confronta con un coseno
  intra-modale.
- La **classificazione zero-shot** è una conseguenza, non una funzione in più:
  si scrive una didascalia per classe e si prende la più simile. La forma del
  prompt conta perché il modello ha imparato su frasi, non su parole isolate.
  La stessa geometria dà il recupero di immagini per descrizione.
- **SigLIP** {cite}`zhai2023sigmoid` sostituisce la softmax di riga con una
  sigmoide per coppia: niente normalizzazione globale, niente raduno degli
  embedding fra le GPU, buon addestramento anche con batch piccoli.
  **ALIGN** {cite}`jia2021scaling` mostra che il metodo regge un miliardo di
  coppie raccolte dal web senza curatela.
- Allineare non è capire: la loss premia il riconoscimento degli oggetti e non
  le relazioni fra loro, e il modello si comporta in buona parte come un
  **sacco di concetti**. Winoground {cite}`thrush2022winoground` rende visibile
  il fallimento, ARO {cite}`yuksekgonul2023when` ne isola la causa (mescolare le
  parole della didascalia non peggiora il recupero) e mostra che con negativi
  permutati la stessa rete impara l'ordine: il limite è dell'obiettivo, non
  dell'architettura. Da qui le architetture delle sezioni successive.
```

`````
