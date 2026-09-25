# Imparare a imparare in fretta

Di esempi ce ne sono dieci, e il compito è nuovo. Che cosa si fa?

La risposta ovvia è la meno utile. Si prende una rete già addestrata su tutti i
compiti conosciuti e le si dà una ripassata sui dieci esempi. Funziona quando i
compiti si somigliano molto, e quando la famiglia è varia funziona male per una
ragione controintuitiva: una rete addestrata su tutti i compiti insieme impara
la loro media, e la media di una famiglia varia può non somigliare a nessuno
dei suoi membri.

C'è un'altra strada, e il nome che porta dice già la mossa: si chiama
**meta-apprendimento**, cioè apprendimento sull'apprendimento. Invece di
addestrare una rete a risolvere i compiti, la si addestra a essere il punto di
partenza da cui il compito successivo si impara in fretta.

## La posizione di partenza, invece della risposta

L'algoritmo che ha dato forma canonica a questa idea si chiama **MAML** (da
*Model-Agnostic Meta-Learning*), e la sua mossa sta tutta in che cosa sceglie di
misurare. Si paga con un conto in più: per correggere il punto di partenza
bisogna seguire anche l'effetto dei passi di adattamento.

`````{tab} Elementare

Uno che ha suonato per anni il violino, poi la viola, poi il violoncello, si
siede davanti a un contrabbasso e dopo cinque minuti ci cava qualcosa. Non
perché sappia suonare il contrabbasso: quello non l'ha mai toccato. Perché gli
anni sugli altri archi gli hanno messo le mani, l'orecchio e il modo di tenere
l'arco in una posizione da cui il contrabbasso è a cinque minuti di distanza.

La differenza con l'apprendimento multi-compito è tutta qui, ed è una
differenza su che cosa si è allenato. Là si imparavano più strumenti insieme
per suonarli tutti; qui non interessa suonare bene i tre di prima. Interessa
arrivare a essere uno da cui il quarto si impara in fretta.

Detta così sembra un gioco di parole, e invece cambia l'allenamento. Chi vuole
suonare bene i tre strumenti si esercita sui tre strumenti. Chi vuole arrivare
pronto al quarto deve esercitarsi in un modo strano: prendere ogni tanto uno
strumento messo da parte, darsi cinque minuti, e poi guardare come suona
dopo quei cinque minuti. Se suona male, a essere corretta non è la tecnica di
quei cinque minuti ma la posizione di partenza da cui li si era cominciati.

Che è anche la ragione per cui l'altra strada fallisce sulle famiglie varie.
Chi si esercita sempre e solo a suonare bene i tre strumenti che ha, senza mai
provare a ripartire da zero su un quarto, finisce per aggiustarsi addosso
una posizione buona per quei tre e per nient'altro: comoda, e ferma. Da lì i
cinque minuti non bastano.

E c'è un modo in cui va anche peggio, che è la cosa più sorprendente di tutte,
e più avanti si vede in numeri. Quella posizione comoda è comoda *perché* è un
punto di equilibrio: chi ci sta dentro e prova a muoversi in fretta verso il
contrabbasso non ci arriva a metà strada, esce dall'equilibrio e basta, e dopo
i cinque minuti suona peggio di quando ha cominciato. Chi invece la
posizione se l'è scelta apposta per potersi muovere, in cinque minuti si
avvicina.

E c'è un confine da tenere presente, perché è netto. Tutto questo vale finché
il quarto strumento è ancora un arco. Metti in mano a quella persona una
tromba, e la posizione delle mani non serve a niente: nel migliore dei casi è
neutra, nel peggiore ha abitudini da disimparare, e cinque minuti diventano
sei mesi. Una posizione di partenza è buona per una famiglia, e quale sia
quella famiglia lo decide chi allena: è lui che sceglie da quale mucchio pescare
gli strumenti dell'allenamento, e quel mucchio è la promessa che sta facendo.

`````

`````{tab} Superiore

MAML sta per *Model-Agnostic Meta-Learning* {cite}`finn2017maml`, e
model-agnostic vuol dire che non prescrive un'architettura: si applica a
qualunque modello addestrato per discesa del gradiente, e infatti gli autori lo
provano su regressione, classificazione e apprendimento per rinforzo.

Si parte da una **distribuzione di compiti** $p(\mathcal{T})$: non un compito,
una famiglia da cui si sorteggia. L'ottimizzazione è a due livelli.

Il **ciclo interno** simula l'adattamento. Sorteggiato un compito
$\mathcal{T}_i$ e presi $k$ suoi esempi, si fanno uno o pochi passi di discesa
a partire dai parametri correnti $\theta$:

$$
\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}^{\text{s}}_{\mathcal{T}_i}(\theta) ,
$$

dove $\alpha$ è il passo interno e $\mathcal{L}^{\text{s}}_{\mathcal{T}_i}$ la
perdita sui $k$ esempi con cui ci si adatta. Il **ciclo esterno** aggiorna $\theta$
guardando quanto valgono i parametri adattati, e non $\theta$ stesso:

$$
\theta \leftarrow \theta - \beta \nabla_\theta
\sum_{\mathcal{T}_i \sim p(\mathcal{T})}
\mathcal{L}^{\text{q}}_{\mathcal{T}_i}(\theta_i') ,
$$

con $\beta$ il passo esterno e $\mathcal{L}^{\text{q}}_{\mathcal{T}_i}$ la
perdita su esempi nuovi dello stesso compito. Qui sta tutto: il gradiente si
prende rispetto a $\theta$ di una perdita valutata in $\theta_i'$, che di
$\theta$ è funzione. Derivare attraverso il passo di adattamento chiama in causa
le derivate seconde, ed è il costo dell'algoritmo. Per un passo interno la
regola della catena dà

$$
\nabla_\theta\,\mathcal{L}^{\text{q}}_{\mathcal{T}_i}(\theta_i') = \big(\mathbf{I} - \alpha\,\nabla^2_\theta \mathcal{L}^{\text{s}}_{\mathcal{T}_i}(\theta)\big)\,\nabla_{\theta'}\mathcal{L}^{\text{q}}_{\mathcal{T}_i}(\theta')\big|_{\theta' = \theta_i'},
$$

dove $\nabla^2_\theta$ è l'hessiana della perdita interna, quella di
adattamento, mentre il gradiente a destra è della perdita esterna. L'hessiana
non si forma mai, basta il suo prodotto per un vettore, che costa un secondo
passaggio all'indietro; ma per derivare attraverso $m$ passi interni bisogna
tenere in memoria il grafo di tutti e $m$, e la memoria cresce linearmente con
$m$. La versione del primo ordine, già provata nel lavoro originale, butta via
il termine con l'hessiana e usa
$\nabla_{\theta'}\mathcal{L}^{\text{q}}_{\mathcal{T}_i}(\theta_i')$ come se
$\theta_i'$ non dipendesse da $\theta$: sulle immagini gli autori la trovano
quasi equivalente e più veloce. Reptile {cite}`nichol2018first` rinuncia anche a
quel gradiente: fa $m > 1$ passi interni e sposta $\theta$ verso i parametri che
ne escono, $\theta \leftarrow \theta + \beta\,(\theta_i^{(m)} - \theta)$. Il
numero di passi non è un dettaglio: con uno solo la mossa diventa
$\theta - \alpha\beta\,\nabla_\theta\mathcal{L}_{\mathcal{T}_i}(\theta)$, la
discesa del gradiente sulla perdita media dei compiti, e sono i passi successivi
a portare dentro i termini che premiano l'adattamento. Le due perdite, infine,
non si calcolano sugli stessi esempi: quella del ciclo interno sull'insieme di
supporto, quella del ciclo esterno sull'insieme di interrogazione, come nel
codice sulle sinusoidi. L'obiettivo che ne esce si legge «$\theta$ è un punto da
cui pochi passi bastano», e non «$\theta$ è bravo sui compiti visti»: sono due
proprietà diverse, e la prima si ottiene solo scrivendola nella funzione
obiettivo.

La valutazione ha una forma sua, **$N$-way $k$-shot**: si costruisce un compito
con $N$ classi e $k$ esempi per classe, si dà al modello l’**insieme di
supporto** (i $N \cdot k$ esempi su cui adattarsi) e lo si interroga
sull’**insieme di interrogazione**. Quello che si misura è la prestazione
dopo l'adattamento, che è una grandezza diversa dalla prestazione del
modello, e per questo la generalizzazione ordinaria fra addestramento e prova
non basta a descriverla.

Il punto di rottura è nella parola *distribuzione*. Tutto il metodo poggia
sull'ipotesi che il compito di prova venga da $p(\mathcal{T})$; su un compito
fuori distribuzione l'inizializzazione non ha nessuna ragione di essere
migliore di una casuale, e può essere peggiore, perché codifica regolarità che
là non valgono. Definire la famiglia è parte del progetto, non un dettaglio
dell'esperimento.

La famiglia su cui il metodo fu presentato, e quella su cui gira il codice qui
sotto, è $p(\mathcal{T}) = \{\,x \mapsto A\sin(x + \varphi)\,\}$ con
$A \sim \mathcal{U}[0{,}1,\,5]$ e $\varphi \sim \mathcal{U}[0,\pi)$. La sua
media si calcola in chiuso, e spiega in anticipo il termine di paragone:
$\mathbb{E}_\varphi[\sin(x+\varphi)] = \frac{2}{\pi}\cos x$, quindi

$$
\mathbb{E}_{\mathcal{T}}\big[f_{\mathcal{T}}(x)\big]
= \mathbb{E}[A]\cdot\frac{2}{\pi}\cos x \approx 1{,}62\cos x ,
$$

una cosinusoide sola, di ampiezza ridotta. Della famiglia è un membro anche
lei ($A = 1{,}62$, $\varphi = \pi/2$, tutti e due dentro il loro intervallo), ma
uno solo, e non predice nessuno degli altri: una rete addestrata
congiuntamente su $p(\mathcal{T})$ converge lì, e da lì tutte le altre onde
sono lontane.

`````

### In pratica: dieci punti su un'onda mai vista

L'esperimento con cui questa idea fu presentata usa la famiglia più semplice
che si possa disegnare: le onde, o *sinusoidi*, cioè le curve che salgono e
scendono regolarmente, tutte della stessa forma ma ciascuna con la propria
altezza e il proprio punto di partenza. Sono un buon banco di prova perché si
somigliano (sono tutte onde) e sono diverse (una è alta e comincia in cima,
un'altra è bassa e comincia in fondo). Il compito è indovinare *quale* onda,
avendone visti dieci punti.

La famiglia serve anche a far vedere in anticipo che cosa impara la rete
allenata su tutte le onde insieme, che sarà il termine di paragone, perché la
sua media si calcola. Sommando tutte le onde della famiglia e dividendo, quello
che resta è un'unica curva, bassa e sempre uguale, che sta più o meno a metà
strada fra tutte. È un'onda anche lei, ma una sola, e nessuna delle altre le
somiglia. Chi si allena su tutte le onde insieme converge lì.

Si confrontano tre punti di partenza, dando a tutti e tre lo stesso
adattamento, cioè cinque passi di aggiustamento sui dieci punti: una rete presa
a caso, una allenata su tutte le onde insieme, e una meta-addestrata. Dare a
tutti e tre lo stesso adattamento è precisamente il confronto che interessa,
perché la domanda è da quale partenza quei passi lì funzionano.

```python
import torch

torch.set_num_threads(1)   # su una macchina carica i thread si ostacolano

def pesi(gen, misure=((1, 40), (40, 40), (40, 1))):
    """La rete come lista esplicita di tensori: serve perche' il ciclo interno
    deve produrre una lista NUOVA di parametri, senza toccare quella vecchia."""
    p = []
    for entra, esce in misure:
        w = torch.randn(entra, esce, generator=gen) * (2.0 / entra) ** 0.5
        p += [w.requires_grad_(), torch.zeros(esce, requires_grad=True)]
    return p

def rete(x, p):
    h = torch.relu(x @ p[0] + p[1])
    h = torch.relu(h @ p[2] + p[3])
    return h @ p[4] + p[5]

def compito(gen):
    """Un membro della famiglia: ampiezza e fase sorteggiate."""
    A = torch.rand(1, generator=gen) * 4.9 + 0.1
    fase = torch.rand(1, generator=gen) * torch.pi
    return lambda x: A * torch.sin(x + fase)

def punti(f, n, gen):
    x = torch.rand(n, 1, generator=gen) * 10 - 5
    return x, f(x)

def adatta(p, x, y, passi, alfa, grafo):
    """Il ciclo interno. Con grafo=True la catena resta derivabile, ed e' cio'
    che permette al ciclo esterno di derivare ATTRAVERSO l'adattamento."""
    for _ in range(passi):
        perdita = ((rete(x, p) - y) ** 2).mean()
        g = torch.autograd.grad(perdita, p, create_graph=grafo)
        p = [w - alfa * gw for w, gw in zip(p, g)]
    return p

ITER, LOTTO = 1000, 8

# --- meta-addestramento: si valuta il DOPO, non l'adesso
gen = torch.Generator().manual_seed(1)
maml = pesi(gen)
opt = torch.optim.Adam(maml, lr=1e-3)
for _ in range(ITER):
    perdita = 0.0
    for _ in range(LOTTO):
        f = compito(gen)
        xs, ys = punti(f, 10, gen)      # insieme di supporto
        xq, yq = punti(f, 10, gen)      # insieme di interrogazione
        adattati = adatta(maml, xs, ys, 1, 0.01, grafo=True)
        perdita = perdita + ((rete(xq, adattati) - yq) ** 2).mean()
    opt.zero_grad(); (perdita / LOTTO).backward(); opt.step()

# --- il termine di paragone: la stessa rete allenata su TUTTE le sinusoidi
gen2 = torch.Generator().manual_seed(1)
insieme = pesi(gen2)
opt2 = torch.optim.Adam(insieme, lr=1e-3)
for _ in range(ITER):
    perdita = 0.0
    for _ in range(LOTTO):
        f = compito(gen2)
        x, y = punti(f, 20, gen2)
        perdita = perdita + ((rete(x, insieme) - y) ** 2).mean()
    opt2.zero_grad(); (perdita / LOTTO).backward(); opt2.step()

# --- la prova: 100 sinusoidi mai viste, stesso adattamento per tutti e tre
import statistics
prova = torch.linspace(-5, 5, 200).reshape(-1, 1)
righe = {}
for etichetta, p0 in (("a caso", pesi(torch.Generator().manual_seed(3))),
                      ("allenata su tutte", insieme),
                      ("MAML", maml)):
    g = torch.Generator().manual_seed(7)      # le stesse 100 sinusoidi per tutti
    prima, dopo = [], []
    for _ in range(100):
        f = compito(g)
        xs, ys = punti(f, 10, g)
        with torch.no_grad():
            prima.append(((rete(prova, p0) - f(prova)) ** 2).mean().item())
        p1 = adatta([w.detach().requires_grad_() for w in p0],
                    xs, ys, 5, 0.01, grafo=False)
        with torch.no_grad():
            dopo.append(((rete(prova, p1) - f(prova)) ** 2).mean().item())
    righe[etichetta] = (statistics.median(prima), statistics.median(dopo),
                        sum(1 for a, b in zip(prima, dopo) if b < a))

print("errore quadratico mediano su 100 sinusoidi mai viste")
print("(mediano e non medio: una singola divergenza rende la media inutile)")
print(f"   {'':20s} {'prima':>8s} {'dopo 5 passi':>13s}   migliora in")
for etichetta, (a, b, quante) in righe.items():
    print(f"   {etichetta:20s} {a:8.2f} {b:13.2f}   {quante:3d} casi su 100")

with torch.no_grad():
    u = rete(prova, insieme)
    print(f"\nla rete allenata su tutte oscilla fra {u.min():.2f} e {u.max():.2f}:")
    print("e' la media della famiglia: un'onda sola, di ampiezza ridotta")
```

```text
errore quadratico mediano su 100 sinusoidi mai viste
(mediano e non medio: una singola divergenza rende la media inutile)
                           prima  dopo 5 passi   migliora in
   a caso                   4.14          4.87    59 casi su 100
   allenata su tutte        2.12          5.34    17 casi su 100
   MAML                     2.08          1.74    77 casi su 100

la rete allenata su tutte oscilla fra -0.96 e 1.84:
e' la media della famiglia: un'onda sola, di ampiezza ridotta
```

I numeri sono errori: più bassi, meglio la curva prevista ricalca l'onda vera.
E la colonna da guardare per prima è quella di sinistra, perché è la sorpresa.
Senza adattamento la rete meta-addestrata ($2{,}08$) e quella allenata su tutte
le onde ($2{,}12$) prendono lo stesso voto, e nessuna delle due è una buona
previsione: sono due curve quasi ferme accanto a onde che salgono e scendono, e
infatti la seconda oscilla fra $-0{,}96$ e $1{,}84$, cioè attorno alla curva
media di poco fa. Il meta-addestramento non ha prodotto un modello
migliore.

La differenza sta tutta nella colonna dopo. Cinque passi di aggiustamento sui
dieci punti portano la rete meta-addestrata da $2{,}08$ a $1{,}74$, e la
migliorano in settantasette onde su cento. Portano quella allenata su tutte da
$2{,}12$ a $5{,}34$, e la migliorano in diciassette: da quel punto di partenza
quegli stessi passi fanno danno. La curva media è un posto comodo dove stare
fermi, e cinque passi lanciati verso un'onda precisa la portano fuori di lì
senza arrivare da nessuna parte.

La riga della rete presa a caso va letta con un'avvertenza, perché sembra
contraddirsi: migliora in cinquantanove casi su cento e ha il numero peggiore.
Le due colonne sono mediane calcolate su insiemi diversi di cento numeri, non
la mediana delle differenze, e da una partenza casuale i miglioramenti sono
piccoli mentre i pochi peggioramenti sono enormi. Il conteggio dei casi e la
mediana rispondono a due domande diverse, e qui danno risposte diverse.

Quello che il meta-addestramento ha ottimizzato, insomma, non si vede
guardando la rete ferma: si vede soltanto guardando che cosa le succede quando
impara. Ed è proprio così che era stata definita la cosa da migliorare.

La tabella riporta mediane, e non medie, perché con passi di dimensione
fissa capita che su qualche onda i cinque passi non convergano affatto: basta
uno di quei casi, e la media di cento numeri la decide lui.

## Confrontare invece di adattare

MAML, davanti al compito nuovo, adatta: fa qualche passo di aggiustamento, cioè
di discesa del gradiente, a partire da un punto scelto bene. Per la
classificazione con pochi esempi c’è un’altra famiglia di metodi, che al momento
della prova non adatta niente. Impara prima uno spazio in cui le classi si
riconoscono per vicinanza, e il compito nuovo lo risolve confrontando. La forma
più semplice sono le **reti prototipiche** di Snell, Swersky e Zemel
{cite}`snell2017prototypical`: ogni classe è rappresentata dalla media dei suoi
esempi di supporto nello spazio appreso, il suo *prototipo*, e un esempio nuovo
va alla classe del prototipo più vicino. L’allenamento a episodi, cioè su
compiti $N$-way $k$-shot sorteggiati come quelli della prova, viene dalle
*matching networks* di Vinyals e colleghi {cite}`vinyals2016matching`.

`````{tab} Elementare

Il portiere di un grande albergo ha visto passare migliaia di facce, e ha
imparato che cosa guardare: la distanza fra gli occhi, la forma del naso e del
mento. Ha imparato anche che cosa lasciar perdere: i capelli, che la gente
cambia; gli occhiali, che si tolgono; la luce della hall, che non è mai la
stessa. Stamattina arrivano cinque ospiti nuovi, e di ciascuno vede la faccia
una volta sola, al banco. Nel pomeriggio uno rientra, e il portiere lo saluta
per nome. Non ha imparato niente di nuovo: ha confrontato. Un portiere al primo
giorno confronterebbe tutto, e un ospite che rientra con il cappello e il sole
alle spalle gli sembrerebbe un’altra persona. Detto con i numeri: di una faccia
si possono prendere quaranta misure, e l’occhio allenato ne guarda poche. È come
se mettesse le facce in fila per somiglianza, vicine quelle che si somigliano
nelle cose che contano e lontane le altre, e questa sistemazione delle facce si
chiama *spazio appreso*.

Se di un ospite ha visto la faccia più volte, se ne fa un ritratto medio, e chi
entra va all’ospite il cui ritratto gli somiglia di più; con una faccia sola il
ritratto è quella faccia. Fra due ospiti il confine cade a metà strada fra i due
ritratti.

Quell’occhio si è allenato facendo tante volte la prova vera: cinque facce a
caso, viste una volta (gli *esempi di supporto*), poi altre facce da assegnare.
Ogni prova così si chiama *episodio*. Ogni volta che sbagliava, a correggersi
non era quello che sapeva di quelle cinque persone, che il giorno dopo erano
partite, ma quello che guardava. Per questo funziona anche con gli ospiti di
domani, purché si distinguano per le stesse cose.

Il confine è quel «purché». Davanti a due gemelle identiche che si distinguono
soltanto per il taglio di capelli, il portiere guarda con grande sicurezza le
cose sbagliate, e fa peggio del principiante, che almeno guardava tutto.

`````

`````{tab} Superiore

Sia $f_\theta : \mathbb{R}^d \to \mathbb{R}^{d'}$ la rete che porta un esempio
nello spazio appreso. In un episodio con insieme di supporto $S_j$ per ciascuna
delle $N$ classi, il prototipo della classe $j$ è

$$
\mathbf{c}_j = \frac{1}{\lvert S_j \rvert}
\sum_{\mathbf{x} \in S_j} f_\theta(\mathbf{x}),
$$

e un esempio d’interrogazione riceve

$$
p_\theta(y = j \mid \mathbf{x}) =
\frac{\exp\big(-\lVert f_\theta(\mathbf{x}) - \mathbf{c}_j \rVert^2\big)}
{\sum_{j'=1}^{N}
\exp\big(-\lVert f_\theta(\mathbf{x}) - \mathbf{c}_{j'} \rVert^2\big)} .
$$

Si minimizza $-\log p_\theta(y \mid \mathbf{x})$ sulle domande di episodi
sorteggiati fra le classi di addestramento, nello stesso formato $N$-way
$k$-shot della prova {cite}`snell2017prototypical`; al momento della prova non
c’è nessun passo di gradiente, solo $N$ medie e $N$ distanze. Da MAML cambia il
costo, perché non ci sono né ciclo interno né derivate seconde, e cambia il
perimetro: il metodo è fatto per la classificazione, mentre MAML si applica
anche alla regressione e al rinforzo.

Con la distanza euclidea al quadrato il classificatore è lineare nello spazio
appreso. Posto $\mathbf{z} = f_\theta(\mathbf{x})$,

$$
-\lVert\mathbf{z} - \mathbf{c}_j\rVert^2 = 2\,\mathbf{c}_j^{\!\top}\mathbf{z}
- \lVert\mathbf{c}_j\rVert^2 - \lVert\mathbf{z}\rVert^2 ,
$$

e l’ultimo termine è uguale per tutte le classi, quindi il confine fra due
classi è l’iperpiano che biseca il segmento fra i loro prototipi. La non
linearità, se serve, sta tutta in $f_\theta$. Gli autori giustificano la media
come prototipo con le divergenze di Bregman, di cui l’euclidea al quadrato è un
caso, e trovano che rende molto più del coseno: la ragione, che danno come
congettura, è che il coseno non è una divergenza di Bregman. Con $k = 1$ il
prototipo è l’esempio stesso, e il metodo diventa equivalente alle matching
networks.

Nel codice l’embedding è lineare, $f_\theta(\mathbf{x}) = \mathbf{W}\mathbf{x}$,
e il gradiente si scrive in forma chiusa (è la ragione per cui qui non serve
autograd). Con $\mathbf{u}_{qj}$ lo scarto fra la domanda $q$ e la media del
supporto della classe $j$, misurati tutti e due nello spazio di partenza, il
logit è $-\lVert\mathbf{W}\mathbf{u}_{qj}\rVert^2$ e

$$
\nabla_{\mathbf{W}}\mathcal{L} = -2\,\mathbf{W} \sum_{q,j} g_{qj}\,
\mathbf{u}_{qj}\mathbf{u}_{qj}^{\!\top},
$$

dove $g_{qj}$ è la derivata della perdita rispetto al logit, cioè la
probabilità meno l’indicatrice della classe giusta, divisa per il numero di
domande. Imparare $\mathbf{W}$ vuol dire imparare la metrica
$\mathbf{W}^{\!\top}\mathbf{W}$.

Il punto di rottura è quello di MAML, spostato dall’inizializzazione allo
spazio. $f_\theta$ impara le direzioni che separano le classi della
distribuzione di addestramento, e su classi che si separano lungo altre
direzioni può aver schiacciato proprio quelle che servono.

`````

Il blocco costruisce una famiglia di classi in cui la cosa da imparare è chiara:
quaranta misure, di cui contano quattro direzioni. I centri delle classi stanno
lungo quelle quattro, il rumore su tutte e quaranta. Lo spazio appreso è una
tabella di numeri, la matrice $\mathbf{W}$, che trasforma le quaranta misure di
ogni esempio. La si allena a episodi da cinque classi con un esempio ciascuna
(5-way 1-shot) su 64 classi, sempre con un esempio anche quando la prova ne darà
cinque, e poi la si prova su 64 classi mai viste della stessa famiglia e su 64
di una famiglia diversa, che ha le sue quattro direzioni altrove.

```python
import numpy as np

rng = np.random.default_rng(0)
d, r = 40, 4                   # quaranta misure, e contano quattro direzioni
# le direzioni che contano, e quelle di un'altra famiglia
famiglia = np.linalg.qr(rng.normal(size=(d, r)))[0]
altra = np.linalg.qr(rng.normal(size=(d, r)))[0]

def classi(n, direzioni):
    """n centri di classe, sparsi soltanto lungo le direzioni date."""
    return rng.normal(0, 3, (n, r)) @ direzioni.T

def episodio(centri, k, q=5, N=5):
    """N classi a caso, k esempi di supporto e q domande per ciascuna."""
    scelte = centri[rng.choice(len(centri), N, replace=False)]
    # il rumore sta su tutte le misure
    supporto = scelte[:, None] + rng.normal(0, 1, (N, k, d))
    domande = scelte[:, None] + rng.normal(0, 1, (N, q, d))
    return supporto, domande.reshape(-1, d), np.repeat(np.arange(N), q)

def accuratezza(W, centri, k, episodi=1000):
    giuste = totale = 0
    for _ in range(episodi):
        S, Q, y = episodio(centri, k)
        prototipi = S.mean(1) @ W.T            # nello spazio appreso
        dist = (((Q @ W.T)[:, None] - prototipi[None]) ** 2).sum(-1)
        giuste, totale = giuste + (dist.argmin(1) == y).sum(), totale + len(y)
    return giuste / totale

base = classi(64, famiglia)                    # le classi dell'addestramento
W = np.eye(d)                     # si parte dalle misure così come sono
for _ in range(2000):             # addestramento a episodi, 5-way 1-shot
    S, Q, y = episodio(base, 1)
    U = Q[:, None, :] - S.mean(1)[None]        # domanda meno prototipo
    logit = -((U @ W.T) ** 2).sum(-1)
    p = np.exp(logit - logit.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    p[np.arange(len(y)), y] -= 1               # g: probabilità meno indicatrice
    M = np.einsum("qk,qki,qkj->ij", p / len(y), U, U)
    W += 0.01 * 2 * W @ M                      # il gradiente è -2 W M

nuove, estranee = classi(64, famiglia), classi(64, altra)
for nome, centri in [("classi nuove, stessa famiglia", nuove),
                     ("classi di un'altra famiglia", estranee)]:
    for k in (1, 5):
        grezze = accuratezza(np.eye(d), centri, k)
        apprese = accuratezza(W, centri, k)
        print(f"{nome}, {k}-shot: misure grezze {grezze:.1%},"
              f" spazio appreso {apprese:.1%}")
resto = np.linalg.svd(famiglia, full_matrices=True)[0][:, r:]   # le altre 36
dentro = np.linalg.norm(W @ famiglia) / np.sqrt(r)
fuori = np.linalg.norm(W @ resto) / np.sqrt(d - r)
print(f"fattore medio: {dentro:.2f} sulle 4 direzioni che contano,"
      f" {fuori:.2f} sulle altre 36")
```

```text
classi nuove, stessa famiglia, 1-shot: misure grezze 76.1%, spazio appreso 88.6%
classi nuove, stessa famiglia, 5-shot: misure grezze 91.2%, spazio appreso 95.6%
classi di un'altra famiglia, 1-shot: misure grezze 79.1%, spazio appreso 64.7%
classi di un'altra famiglia, 5-shot: misure grezze 92.3%, spazio appreso 80.6%
fattore medio: 0.53 sulle 4 direzioni che contano, 0.17 sulle altre 36
```

Sulle classi nuove della stessa famiglia lo spazio appreso porta il
riconoscimento da un esempio solo da 76,1% a 88,6%, e da cinque esempi da 91,2%
a 95,6%: con un esempio solo il rumore pesa di più, ed è lì che togliere quello
che non conta rende di più. L’ultima riga dice che cosa è stato imparato, senza
che nessuno lo avesse detto alla matrice: le differenze lungo le quattro
direzioni che contano escono moltiplicate in media per 0,53, quelle lungo le
altre trentasei per 0,17, cioè circa tre volte più piccole. Sulle classi
dell’altra famiglia lo stesso schiacciamento cade sulle direzioni sbagliate, e
lo spazio appreso fa peggio delle misure grezze: 64,7% contro 79,1% da un
esempio, 80,6% contro 92,3% da cinque.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il meta-apprendimento non allena una rete a risolvere i compiti che ha
  visto: la allena a essere un buon punto di partenza per il compito
  successivo, quello di cui esistono dieci esempi.
- L'allenamento è strano apposta: si prende un compito messo da parte, ci si
  concede qualche passo di adattamento, e si guarda com'è andata dopo
  quei passi. È quel «dopo» a essere migliorato, non il «prima».
- La strada ovvia (allenare una rete sola su tutti i compiti insieme e poi
  ripassarla) fallisce quando la famiglia è varia, perché quella rete impara la
  media dei compiti, e la media di solito non somiglia a nessuno di loro:
  sulle onde della prova è un'onda bassa e sempre la stessa, e le altre non le
  somigliano.
  E fallisce due volte, perché quella curva media è un posto comodo dove stare
  fermi: i pochi passi di aggiustamento la portano fuori di lì e la lasciano a
  metà, cioè peggiorano invece di migliorare.
- Il confine è la famiglia: una buona posizione di partenza lo è per gli
  strumenti che le somigliano. Su un compito che sta fuori non aiuta, e può
  perfino portarsi dietro abitudini da disimparare.
- L’altra strada non adatta niente: impara che cosa guardare, e davanti alle
  classi nuove confronta. Ogni classe diventa il ritratto medio dei suoi
  esempi, e la cosa nuova va al ritratto più somigliante. Funziona per le
  classi della stessa famiglia; con classi che si distinguono per altre cose,
  guarda nel posto sbagliato.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- MAML {cite}`finn2017maml` ottimizza un'inizializzazione $\theta$ a due
  livelli: il ciclo interno adatta,
  $\theta_i' = \theta - \alpha\nabla_\theta\mathcal{L}^{\text{s}}_{\mathcal{T}_i}(\theta)$,
  e il ciclo esterno aggiorna $\theta$ sul valore della perdita calcolata in
  $\theta_i'$ su esempi nuovi dello stesso compito. Derivare attraverso
  l'adattamento chiama le derivate seconde: è il costo del metodo.
- L'obiettivo ottimizzato è la prestazione dopo l'adattamento, che è una
  proprietà diversa dalla prestazione tout court, e la si ottiene solo
  scrivendola nella funzione obiettivo.
- *Model-agnostic* vuol dire che serve solo che il modello si addestri per
  discesa del gradiente: gli autori lo provano su regressione, classificazione
  e rinforzo.
- La valutazione è $N$-way $k$-shot con insieme di supporto e di
  interrogazione, perché quello che va misurato è la velocità di
  adattamento e non la generalizzazione ordinaria.
- Il punto di rottura sta nella distribuzione $p(\mathcal{T})$: fuori da essa
  l'inizializzazione non ha ragione di aiutare, e può nuocere.
- Le reti prototipiche imparano $f_\theta$ a episodi e classificano al
  prototipo più vicino, la media del supporto: nessun ciclo interno, un
  classificatore lineare nello spazio appreso con la distanza euclidea al
  quadrato, e lo stesso punto di rottura di MAML, le classi fuori dalla
  distribuzione di addestramento {cite}`snell2017prototypical`.
```
`````

Quello che il capitolo ha costruito, dalla prima sezione a qui, è sempre la
stessa cosa vista da tre angoli: la posizione in cui una rete si trova prima di
affrontare un compito, che vale più del compito per cui era nata. La
profondità la costruisce a scala, dal bordo grezzo alla forma intera; il
multi-compito la fa servire a più mestieri insieme; il meta-apprendimento la
sceglie in modo che il mestiere successivo costi poco. Il {doc}`capitolo sulla
visione artificiale </VisioneArtificiale/overview>` la porta dentro un dominio
solo, le immagini, dove i mestieri hanno nomi precisi: dire che cosa c'è, dire
dov'è, ritagliarne il contorno.
