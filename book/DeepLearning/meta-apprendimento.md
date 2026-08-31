# Imparare a imparare in fretta

Di esempi ce ne sono dieci, e il compito è nuovo. Che cosa si fa?

La risposta ovvia è la meno utile. Si prende una rete già addestrata su tutti i
compiti conosciuti e le si dà una ripassata sui dieci esempi. Funziona quando i
compiti si somigliano molto, e quando la famiglia è varia funziona male per una
ragione controintuitiva: una rete addestrata su tutti i compiti insieme impara
la loro **media**, e la media di una famiglia varia può non somigliare a nessuno
dei suoi membri.

C'è un'altra strada, e il nome che porta dice già la mossa: si chiama
**meta-apprendimento**, cioè apprendimento sull'apprendimento. Invece di
addestrare una rete a risolvere i compiti, la si addestra a essere il punto di
partenza da cui il compito successivo si impara in fretta.

## La posizione di partenza, invece della risposta

`````{tab} Elementare

Uno che ha suonato per anni il violino, poi la viola, poi il violoncello, si
siede davanti a un contrabbasso e dopo cinque minuti ci cava qualcosa. Non
perché sappia suonare il contrabbasso: quello non l'ha mai toccato. Perché gli
anni sugli altri archi gli hanno messo le mani, l'orecchio e il modo di tenere
l'arco in una posizione da cui il contrabbasso è a cinque minuti di distanza.

La differenza con l'apprendimento multi-compito è tutta qui, ed è una
differenza su che cosa si è allenato. Là si imparavano più strumenti insieme
per suonarli tutti; qui non interessa suonare bene i tre di prima. Interessa
arrivare a essere uno **da cui il quarto si impara in fretta**.

Detta così sembra un gioco di parole, e invece cambia l'allenamento. Chi vuole
suonare bene i tre strumenti si esercita sui tre strumenti. Chi vuole arrivare
pronto al quarto deve esercitarsi in un modo strano: prendere ogni tanto uno
strumento messo da parte, darsi cinque minuti, e poi guardare **come suona
dopo quei cinque minuti**. Se suona male, a essere corretta non è la tecnica di
quei cinque minuti ma la posizione di partenza da cui li si era cominciati.

Che è anche la ragione per cui l'altra strada fallisce sulle famiglie varie.
Chi si esercita sempre e solo a suonare bene i tre strumenti che ha, senza mai
provare a **ripartire da zero** su un quarto, finisce per aggiustarsi addosso
una posizione buona per quei tre e per nient'altro: comoda, e ferma. Da lì i
cinque minuti non bastano.

E c'è un modo in cui va anche peggio, che è la cosa più sorprendente di tutte,
e più avanti si vede in numeri. Quella posizione comoda è comoda *perché* è un
punto di equilibrio: chi ci sta dentro e prova a muoversi in fretta verso il
contrabbasso non ci arriva a metà strada, esce dall'equilibrio e basta, e dopo
i cinque minuti suona **peggio** di quando ha cominciato. Chi invece la
posizione se l'è scelta apposta per potersi muovere, in cinque minuti si
avvicina.

E c'è un confine da tenere presente, perché è netto. Tutto questo vale finché
il quarto strumento è ancora un arco. Metti in mano a quella persona una
tromba, e la posizione delle mani non serve a niente: nel migliore dei casi è
neutra, nel peggiore ha abitudini da disimparare, e cinque minuti diventano
sei mesi. Una posizione di partenza è buona **per una famiglia**, e quale sia
quella famiglia lo decide chi allena: è lui che sceglie da quale mucchio pescare
gli strumenti dell'allenamento, e quel mucchio è la promessa che sta facendo.

`````

`````{tab} Superiore

L'algoritmo che ha dato la forma canonica a questa idea è **MAML**
(*Model-Agnostic Meta-Learning*) {cite}`finn2017maml`. Model-agnostic vuol dire
che non prescrive un'architettura: si applica a qualunque modello addestrato
per discesa del gradiente, e infatti gli autori lo provano su regressione,
classificazione e apprendimento per rinforzo.

Si parte da una **distribuzione di compiti** $p(\mathcal{T})$: non un compito,
una famiglia da cui si sorteggia. L'ottimizzazione è a due livelli.

Il **ciclo interno** simula l'adattamento. Sorteggiato un compito
$\mathcal{T}_i$ e presi $k$ suoi esempi, si fanno uno o pochi passi di discesa
a partire dai parametri correnti $\theta$:

$$
\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta) ,
$$

dove $\alpha$ è il passo interno. Il **ciclo esterno** aggiorna $\theta$
guardando quanto valgono i parametri **adattati**, e non $\theta$ stesso:

$$
\theta \leftarrow \theta - \beta \nabla_\theta
\sum_{\mathcal{T}_i \sim p(\mathcal{T})}
\mathcal{L}_{\mathcal{T}_i}(\theta_i') ,
$$

con $\beta$ il passo esterno. Qui sta tutto: il gradiente si prende rispetto a
$\theta$ di una perdita valutata in $\theta_i'$, che di $\theta$ è funzione.
Derivare attraverso il passo di adattamento chiama in causa le derivate
seconde, ed è il costo dell'algoritmo. L'obiettivo che ne esce si legge
«$\theta$ è un punto da cui pochi passi bastano», e non «$\theta$ è bravo sui
compiti visti»: sono due proprietà diverse, e la prima si ottiene solo
scrivendola nella funzione obiettivo.

La valutazione ha una forma sua, **$N$-way $k$-shot**: si costruisce un compito
con $N$ classi e $k$ esempi per classe, si dà al modello l’**insieme di
supporto** (i $N \cdot k$ esempi su cui adattarsi) e lo si interroga
sull’**insieme di interrogazione**. Quello che si misura è la prestazione
**dopo l'adattamento**, che è una grandezza diversa dalla prestazione del
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

una cosinusoide sola, di ampiezza piccola, che **non appartiene alla
famiglia**. Una rete addestrata congiuntamente su $p(\mathcal{T})$ converge lì,
ed è il punto: non impara una sinusoide sbagliata, impara una curva che nella
famiglia non c'è.

`````

### In pratica: dieci punti su un'onda mai vista

L'esperimento con cui questa idea fu presentata usa la famiglia più semplice
che si possa disegnare: le **onde**, cioè le curve che salgono e scendono
regolarmente, tutte della stessa forma ma ciascuna con la propria altezza e il
proprio punto di partenza. Sono un buon banco di prova perché si somigliano
(sono tutte onde) e sono diverse (una è alta e comincia in cima, un'altra è
bassa e comincia in fondo). Il compito è indovinare *quale* onda, avendone visti
dieci punti.

La famiglia serve anche a far vedere in anticipo che cosa impara il termine di
paragone, perché la sua media si calcola. Sommando tutte le onde della famiglia
e dividendo, quello che resta è un'unica curva, bassa e sempre uguale, che sta
più o meno a metà strada fra tutte e che nella famiglia **non c'è**. Chi si
allena su tutte le onde insieme converge lì.

Si confrontano tre punti di partenza, dando a tutti e tre lo stesso
adattamento, cioè cinque passi di aggiustamento sui dieci punti: una rete presa
a caso, una allenata su tutte le onde insieme, e una meta-addestrata. Dare a
tutti e tre lo stesso adattamento è precisamente il confronto che interessa,
perché la domanda è da quale partenza **quei passi lì** funzionano.

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
    print("e' la media della famiglia, non una sua sinusoide")
```

```text
errore quadratico mediano su 100 sinusoidi mai viste
(mediano e non medio: una singola divergenza rende la media inutile)
                           prima  dopo 5 passi   migliora in
   a caso                   4.14          4.87    59 casi su 100
   allenata su tutte        2.12          5.34    17 casi su 100
   MAML                     2.08          1.74    77 casi su 100

la rete allenata su tutte oscilla fra -0.96 e 1.84:
e' la media della famiglia, non una sua sinusoide
```

I numeri sono errori: più bassi, meglio la curva prevista ricalca l'onda vera.
E la colonna da guardare per prima è quella di sinistra, perché è la sorpresa.
Senza adattamento la rete meta-addestrata ($2{,}08$) e quella allenata su tutte
le onde ($2{,}12$) prendono lo stesso voto, e nessuna delle due è una buona
previsione: sono due curve quasi ferme accanto a onde che salgono e scendono, e
infatti la seconda oscilla fra $-0{,}96$ e $1{,}84$, cioè attorno alla curva
media di poco fa. **Il meta-addestramento non ha prodotto un modello
migliore.**

La differenza sta tutta nella colonna dopo. Cinque passi di aggiustamento sui
dieci punti portano la rete meta-addestrata da $2{,}08$ a $1{,}74$, e la
migliorano in settantasette onde su cento. Portano quella allenata su tutte da
$2{,}12$ a $5{,}34$, e la migliorano in diciassette: da quel punto di partenza
quegli stessi passi fanno **danno**, ed è il peggioramento annunciato dalla
scena. La curva media è un posto comodo dove stare fermi, e cinque passi
lanciati verso un'onda precisa la portano fuori di lì senza arrivare da nessuna
parte.

La riga della rete presa a caso va letta con un'avvertenza, perché sembra
contraddirsi: migliora in cinquantanove casi su cento e ha il numero peggiore.
Le due colonne sono mediane calcolate su insiemi diversi di cento numeri, non
la mediana delle differenze, e da una partenza casuale i miglioramenti sono
piccoli mentre i pochi peggioramenti sono enormi. Il conteggio dei casi e la
mediana rispondono a due domande diverse, e qui danno risposte diverse.

Quello che il meta-addestramento ha ottimizzato, insomma, non si vede
guardando la rete ferma: si vede soltanto guardando che cosa le succede quando
impara. Ed è proprio così che era stata definita la cosa da migliorare.

Un'ultima nota sul come si misura, che vale oltre questo esperimento. La tabella
riporta **mediane**, e non medie, perché con passi di dimensione fissa capita
che su qualche onda i cinque passi non convergano affatto: basta uno di quei
casi, e la media di cento numeri la decide lui. Una prima versione di questa
prova riportava medie su quaranta onde, e alla riga di mezzo dava un errore di
trentacinque miliardi che non descriveva nessuna delle quaranta.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il **meta-apprendimento** non allena una rete a risolvere i compiti che ha
  visto: la allena a essere un **buon punto di partenza** per il compito
  successivo, quello di cui esistono dieci esempi.
- L'allenamento è strano apposta: si prende un compito messo da parte, ci si
  concede qualche passo di adattamento, e si guarda com'è andata **dopo**
  quei passi. È quel «dopo» a essere migliorato, non il «prima».
- La strada ovvia (allenare una rete sola su tutti i compiti insieme e poi
  ripassarla) fallisce quando la famiglia è varia, perché quella rete impara la
  **media** dei compiti, e la media di solito non è nessuno di loro: sulle onde
  della prova è una curva bassa e sempre uguale, che nella famiglia non c'è.
  E fallisce due volte, perché quella curva media è un posto comodo dove stare
  fermi: i pochi passi di aggiustamento la portano fuori di lì e la lasciano a
  metà, cioè peggiorano invece di migliorare.
- Il confine è la **famiglia**: una buona posizione di partenza lo è per gli
  strumenti che le somigliano. Su un compito che sta fuori non aiuta, e può
  perfino portarsi dietro abitudini da disimparare.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- **MAML** {cite}`finn2017maml` ottimizza un'inizializzazione $\theta$ a due
  livelli: il ciclo interno adatta,
  $\theta_i' = \theta - \alpha\nabla_\theta\mathcal{L}_{\mathcal{T}_i}(\theta)$,
  e il ciclo esterno aggiorna $\theta$ sul valore di $\mathcal{L}$ calcolata in
  $\theta_i'$. Derivare attraverso l'adattamento chiama le derivate seconde: è
  il costo del metodo.
- L'obiettivo ottimizzato è la prestazione **dopo** l'adattamento, che è una
  proprietà diversa dalla prestazione tout court, e la si ottiene solo
  scrivendola nella funzione obiettivo.
- *Model-agnostic* vuol dire che serve solo che il modello si addestri per
  discesa del gradiente: gli autori lo provano su regressione, classificazione
  e rinforzo.
- La valutazione è **$N$-way $k$-shot** con insieme di supporto e di
  interrogazione, perché quello che va misurato è la **velocità di
  adattamento** e non la generalizzazione ordinaria.
- Il punto di rottura sta nella distribuzione $p(\mathcal{T})$: fuori da essa
  l'inizializzazione non ha ragione di aiutare, e può nuocere.
```
`````

Quello che il capitolo ha costruito, dalla prima sezione a qui, è sempre la
stessa cosa vista da tre angoli: la posizione in cui una rete si trova prima di
affrontare un compito, che vale più del compito per cui era nata. La
profondità la costruisce a scala, dal bordo grezzo alla forma intera; il
multi-compito la fa servire a più mestieri insieme; il meta-apprendimento la
sceglie in modo che il mestiere successivo costi poco. Il capitolo sulla
visione artificiale la porta dentro un dominio solo, le immagini, dove i
mestieri hanno nomi precisi: dire che cosa c'è, dire dov'è, ritagliarne il
contorno.
