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

L'idea, resa celebre da CLIP {cite}`radford2021learning` nel 2021, è di
addestrare due reti separate, un **encoder di immagini** e un **encoder di
testo**, a scrivere le loro uscite nello stesso spazio vettoriale. Il compito
non è più assegnare un'etichetta: è appaiare. Dato un mucchietto di immagini e
il mucchietto mescolato delle loro didascalie, il modello deve dire chi va con
chi.

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
I_i = \frac{f_{\text{img}}(x_i)}{\lVert f_{\text{img}}(x_i) \rVert_2},
\qquad
T_j = \frac{f_{\text{txt}}(t_j)}{\lVert f_{\text{txt}}(t_j) \rVert_2},
$$

dove $x_i$ è l'immagine $i$-esima del batch, $t_j$ la didascalia $j$-esima e
$I_i, T_j \in \mathbb{R}^d$ i due embedding, che vivono così sulla sfera
unitaria: il loro prodotto scalare $\langle I_i, T_j \rangle$ è esattamente il
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

Due encoder, uno spazio solo: le similarità di un batch formano una matrice
$N \times N$ in cui le coppie vere stanno sulla diagonale e tutto il resto è
un negativo.
```

La {numref}`fig-vlm-contrastivo` mostra la struttura che ne esce, ed è tutta la
sezione in un disegno. Le due torri lavorano indipendenti, non si scambiano
niente durante il calcolo, e si incontrano solo alla fine, in un prodotto
scalare. Da un batch di $N$ coppie nasce una matrice $N \times N$ di
similarità: sulla diagonale le coppie vere, ovunque altrove i **negativi**,
cioè gli abbinamenti sbagliati che il caso ha messo insieme nello stesso batch.

## Una cross-entropy che guarda in due direzioni

A questo punto serve una funzione di costo che dica al modello che cosa fare di
quella matrice, e la richiesta è semplice da enunciare: la diagonale in alto,
tutto il resto in basso. Il modo di ottenerlo è la vecchia conoscenza di questo
libro, la cross-entropy, applicata però a un problema di scelta multipla che il
batch costruisce da solo.

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
\log \frac{\exp\big(s(u, v^{+})/\tau\big)}
{\sum_{k=1}^{N} \exp\big(s(u, v_k)/\tau\big)} \right],
$$

dove $u$ è l'ancora, $v^{+}$ il suo positivo, $v_1, \dots, v_N$ l'insieme dei
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
\frac{\exp\big(\langle I_i, T_i \rangle / \tau\big)}
{\sum_{j=1}^{N} \exp\big(\langle I_i, T_j \rangle / \tau\big)},
$$

e simmetricamente, scorrendo la colonna $i$ invece della riga $i$, per la
direzione testo → immagine:

$$
\ell^{\,\mathrm{T}\to\mathrm{I}}_i = - \log
\frac{\exp\big(\langle I_i, T_i \rangle / \tau\big)}
{\sum_{k=1}^{N} \exp\big(\langle I_k, T_i \rangle / \tau\big)}.
$$

La loss finale è la media delle due:

$$
\mathcal{L} = \frac{1}{2N} \sum_{i=1}^{N}
\Big( \ell^{\,\mathrm{I}\to\mathrm{T}}_i + \ell^{\,\mathrm{T}\to\mathrm{I}}_i \Big).
$$

Qui $N$ è la dimensione del batch, $I_i$ e $T_j$ gli embedding normalizzati,
$\langle I_i, T_j \rangle$ la loro similarità coseno e $\tau$ la temperatura. Si
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
non si apprezza. Prendiamo un batch minuscolo, $N = 4$, con similarità coseno
plausibili per un modello a metà addestramento (le coppie vere intorno a
$0{,}3$, le altre fra $0$ e $0{,}15$):

| $\langle I_i, T_j \rangle$ | $T_1$ | $T_2$ | $T_3$ | $T_4$ |
|---|---|---|---|---|
| $I_1$ | **0,30** | 0,10 | 0,05 | 0,02 |
| $I_2$ | 0,08 | **0,28** | 0,12 | 0,04 |
| $I_3$ | 0,04 | 0,15 | **0,32** | 0,09 |
| $I_4$ | 0,06 | 0,03 | 0,10 | **0,26** |

Con la temperatura di partenza di CLIP, $\tau = 0{,}07$, i valori della prima
riga diventano logit dividendo per $\tau$: $0{,}30/0{,}07 = 4{,}29$, poi
$1{,}43$, $0{,}71$ e $0{,}29$. Esponenziando si ottengono $72{,}7$, $4{,}17$,
$2{,}04$ e $1{,}33$, la cui somma è $80{,}2$; le probabilità sono quindi
$0{,}906$, $0{,}052$, $0{,}025$ e $0{,}017$, e il costo della riga è
$-\log 0{,}906 = 0{,}099$ (i logaritmi qui sono naturali, come vuole la forma
esponenziale della softmax). Ripetendo per le altre tre righe e mediando, la
loss in direzione immagine → testo vale $0{,}147$; quella sulle colonne
$0{,}148$; la loss simmetrica $0{,}148$.

Ora rifacciamo il conto **senza cambiare una sola similarità**, solo alzando la
temperatura a $\tau = 0{,}5$. La prima riga diventa $0{,}351$, $0{,}235$,
$0{,}213$, $0{,}201$: la coppia giusta è ancora in testa, ma di un soffio, e la
loss simmetrica sale a $1{,}082$. Per confronto, un modello che tirasse a caso
fra quattro didascalie pagherebbe $\log 4 = 1{,}386$. Con $\tau = 0{,}5$ questa
matrice, che pure è ordinata correttamente, costa quasi quanto tirare a caso;
con $\tau = 0{,}07$ ne costa circa un decimo.

## Perché la temperatura e il batch non sono dettagli

Quel confronto dice una cosa importante: la temperatura non è un parametro
cosmetico, decide *quanto* piccole differenze di coseno diventino grandi
differenze di probabilità, e quindi che cosa il modello si sforzi di
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
coseno vicino a quello del positivo.

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
```

Vale la pena notare che cosa *non* c'è: nessuna etichetta, nessun numero di
classi, nessuna testa di classificazione. L'unica informazione supervisionata è
l'ordine delle righe, cioè il fatto che la didascalia $i$ stava sotto
l'immagine $i$.

## Il classificatore che si scrive a parole

Finito l'addestramento, il modello sa fare una cosa sola: dire quanto
un'immagine e un testo si somigliano. Ma quella cosa sola, usata bene, produce
un classificatore che nessuno ha addestrato.

`````{tab} Elementare

Vuoi distinguere gatti, cani e tram? Non serve raccogliere foto né riaddestrare
niente. Scrivi tre frasi: «una foto di un gatto», «una foto di un cane», «una
foto di un tram». Le passi all'encoder di testo, che ti dà tre vettori. Passi
la tua immagine all'encoder di immagini, che te ne dà uno. Guardi a quale dei
tre è più vicino, e hai la risposta. Se domani ti serve anche «una foto di un
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
ImageNet, poco più di un punto percentuale, e mediare i vettori di ottanta
formulazioni diverse ne vale altri tre e mezzo: non è un dettaglio da
rifinitura.

`````

`````{tab} Superiore

Dato un insieme di classi candidate $\{c_1, \dots, c_K\}$, si costruisce per
ciascuna un prompt (per esempio `una foto di un {c_k}`), lo si passa
nell'encoder di testo e si normalizza, ottenendo $T_1, \dots, T_K$. La
predizione per un'immagine con embedding $I$ è

$$
\hat{y} = \arg\max_{k \in \{1, \dots, K\}} \; \langle I, T_k \rangle .
$$

L'osservazione strutturale, fatta nel paper originale
{cite}`radford2021learning`, è che questa è **letteralmente** una
classificazione lineare: la matrice $[T_1; \dots; T_K] \in \mathbb{R}^{K \times d}$
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

Il vincolo lasciato in sospeso poco fa è la normalizzazione globale. La softmax
di una riga ha bisogno di conoscere l'intera riga, cioè tutte le didascalie del
batch; se il batch è distribuito su duecento GPU, ogni passo di addestramento
comincia con un raduno degli embedding e finisce con la loro distribuzione
all'indietro. È un costo di comunicazione che cresce con il batch, proprio
mentre il metodo chiede batch grandi.

SigLIP {cite}`zhai2023sigmoid` cambia una cosa sola, e la cambia alla radice:
smette di trattare la riga come una domanda a risposta multipla e tratta ogni
cella come una domanda a sé, con risposta sì o no. «Questa immagine e questa
didascalia vanno insieme?» è un problema di classificazione binaria, e la
funzione che gli corrisponde non è la softmax ma la **sigmoide**:

$$
\mathcal{L}_{\text{sig}} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{N}
\log \sigma\!\Big( z_{ij} \big( t \, \langle I_i, T_j \rangle + b \big) \Big),
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
si consumerebbero tutte a correggerlo, invece che a imparare.

La conseguenza pratica è che ogni addendo della somma dipende da una sola
coppia. Non c'è più niente da normalizzare su tutto il batch, il calcolo si può
spezzare in blocchi che si scambiano gli embedding a turno, e soprattutto la
qualità dell'addestramento smette di dipendere dall'avere un batch enorme: sotto
la soglia delle sedicimila coppie la sigmoide stacca nettamente la softmax, e
oltre le trentaduemila nessuna delle due guadagna più molto, il che toglie
alla dimensione del batch il ruolo di prerequisito. È lo stesso allineamento,
ottenuto togliendo un vincolo invece di aggiungere un pezzo.

Vale la pena registrare anche un risultato di metodo, arrivato dallo stesso
periodo: ALIGN {cite}`jia2021scaling` ha addestrato la stessa architettura a due
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

Il fenomeno è stato isolato da **Winoground** {cite}`thrush2022winoground`, un
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
sono tutti *sotto* il livello del caso. Non è una classifica fra prodotti: è la
misura di un limite che riguarda la famiglia.

La ragione è strutturale, e sta nella loss che abbiamo scritto. Il modello non
è addestrato a *descrivere* un'immagine, è addestrato a *distinguere* la sua
didascalia dalle altre $N-1$ del batch, che sono didascalie di immagini prese a
caso. Per vincere quel gioco, quasi sempre, basta indovinare quali oggetti
compaiono nella foto: se le altre parlano di un tramonto, di una bicicletta e di
una scodella di minestra, riconoscere «gatto» e «muro» è più che sufficiente, e
capire *chi sta sopra chi* non porta nessun vantaggio. La strada più economica
verso una loss bassa è trattare la didascalia come un **sacco di concetti**, e
l'ottimizzazione, che è pigra per mestiere, la prende. La sintassi, le
relazioni spaziali, il conteggio, la negazione (una didascalia con «senza» resta
vicinissima alla stessa senza il «senza») sono i primi a rimanere fuori.

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
  (appresa) decide quanto la distribuzione è piccata, e i negativi vengono dal
  batch, quindi un batch piccolo rende il compito troppo facile.
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
  **sacco di concetti**. È il fenomeno misurato da Winoground
  {cite}`thrush2022winoground`, e la ragione per cui servono le architetture
  delle sezioni successive.
```
