# Una rete, molti compiti: l'apprendimento multi-compito

Chi ha studiato il latino racconta spesso di aver capito meglio l'italiano.
Non è un modo di dire affettuoso verso il liceo classico: è che le due materie
poggiano sulla stessa impalcatura, e faticare sulla declinazione di una
costringe a guardare in faccia una struttura che nell'altra si usava senza
accorgersene. Imparare due cose imparentate insieme non costa il doppio, e a
volte costa meno che impararne una sola.

L'idea che questo valga anche per una rete neurale ha un articolo di
riferimento e una data: Rich Caruana, 1997, in un lavoro il cui titolo è
semplicemente *Multitask Learning* {cite}`caruana1997multitask`. La tesi è
netta: addestrare una rete su più compiti collegati, in parallelo e con una
rappresentazione condivisa, **migliora la generalizzazione su ciascuno di
essi**. Non è un trucco per risparmiare memoria: è il compito in più che
insegna qualcosa al compito principale.

Vale la pena affrontarla qui perché è una tecnica che il libro incontra
dappertutto senza mai chiamarla per nome. Il rilevatore di oggetti che predice
insieme la classe e le coordinate del riquadro fa multi-compito. Il modello
linguistico pre-addestrato su decine di obiettivi diversi fa multi-compito. Il
sistema di raccomandazione che stima insieme la probabilità di un clic e quella
di un acquisto fa multi-compito, e la seconda è rara quanto preziosa.

## Un tronco, tante teste

Prima del perché, la forma: come è fatta materialmente una rete che fa più cose.

`````{tab} Elementare

La struttura si disegna in un attimo: un **tronco** condiviso, che elabora
l'ingresso, e in cima tante **teste** quante sono le cose da predire, una per
compito. Il tronco impara una rappresentazione buona per tutti; ogni testa la
traduce nella risposta che le serve.

L'immagine giusta è quella di un ufficio: c'è un archivio comune, dove il
materiale viene letto e ordinato una volta sola, e poi ci sono gli uffici
specializzati che da quello stesso archivio ricavano risposte diverse. Nessuno
rilegge i documenti da capo per ogni domanda.

Non è l'unica forma possibile. A volte i compiti sono parenti ma non abbastanza
da poter condividere tutto: allora si tengono reti separate e si chiede solo
che **non si allontanino troppo** l'una dall'altra, con una penalità che le
tiene vicine. Costa di più, ma non obbliga due compiti diversi a usare per
forza la stessa identica rappresentazione.

Nella pratica si finisce quasi sempre in mezzo: **condiviso in basso, separato
in alto**. In basso una rete impara cose generiche (i bordi, le forme, la
struttura della frase) che servono a chiunque; in alto cose specifiche del
compito, che è giusto restino separate.

`````

`````{tab} Superiore

La forma standard è la **condivisione dura** (*hard parameter sharing*): un
tronco $g_\phi$ comune e $T$ teste $h_{\theta_t}$, addestrati minimizzando una
somma pesata

$$
\mathcal{L} = \sum_{t=1}^{T} \lambda_t \,
\mathcal{L}_t\big(h_{\theta_t}(g_\phi(\mathbf{x})),\, y_t\big).
$$

L'alternativa è la **condivisione morbida** (*soft sharing*): $T$ reti
separate, ciascuna con i propri parametri, legate da un termine di
regolarizzazione che ne penalizza la distanza, per esempio $\sum_{t \neq s}
\lVert \phi_t - \phi_s \rVert_2^2$. Costa $T$ volte i parametri ma non impone
una rappresentazione unica: si usa quando i compiti sono affini ma non
sovrapponibili.

Le forme miste condividono gli strati bassi e lasciano divergere gli alti, ed è
quasi sempre la scelta pratica, per la ragione che il capitolo ha già
stabilito parlando di rappresentazioni gerarchiche: le feature generiche
stanno in basso e quelle specifiche in alto, quindi il punto in cui separare le
teste è una decisione su **quanto in alto arriva la parentela** fra i compiti.

Da notare che il *transfer learning* già incontrato è lo stesso schema disteso
nel tempo: là i compiti si affrontano in sequenza (si pre-addestra su uno, si
rifinisce sull'altro), qui in parallelo. La differenza pratica è che il
sequenziale può dimenticare il primo compito mentre impara il secondo, e il
parallelo no, perché il primo è ancora nella loss.

`````

## Perché funziona: il compito in più fa da freno

`````{tab} Elementare

Il guadagno viene da tre parti, e conviene tenerle distinte perché non sono la
stessa cosa.

Il primo è il più ovvio: **più segnale**. Il compito in più porta con sé altre
etichette, quindi altre occasioni di capire com'è fatto l'ingresso. È
particolarmente prezioso quando il compito che ci sta a cuore ha poche
etichette (costano, o sono rare) e quello di contorno ne ha tante.

Il secondo è più sottile ed è il vero motivo per cui la cosa funziona: il
compito in più fa da **freno**. Una rete lasciata sola con un compito trova la
scorciatoia più comoda per risolverlo, e le scorciatoie sono proprio ciò che
non generalizza. Se la stessa rappresentazione deve servire anche a un secondo
compito, quelle scorciatoie smettono di essere convenienti, perché al secondo
non servono. La rete è spinta verso soluzioni più generali, che è la
definizione stessa di regolarizzazione.

Il terzo è di attenzione: certi compiti **dicono alla rete dove guardare**. Se
per rispondere alla seconda domanda serve un dettaglio che per la prima
sembrava trascurabile, la rete impara comunque a rappresentarlo, e magari
scopre che serviva anche alla prima.

`````

`````{tab} Superiore

L'argomento di Caruana è **statistico**, non ingegneristico: i compiti
condividono un **bias induttivo**. Ogni algoritmo di apprendimento ne ha uno,
cioè un insieme di assunzioni implicite che rendono preferibili certe ipotesi;
addestrare su più compiti significa cercare l'ipotesi che soddisfa *tutti* i
loro bias insieme, il che restringe lo spazio delle soluzioni ammissibili.

Restringere lo spazio delle ipotesi con informazione **vera** è la definizione
operativa di regolarizzazione: il rumore specifico di un compito viene mediato
via, la struttura comune sopravvive. Nei termini del compromesso
bias-varianza già incontrato, si accetta un po' di bias in cambio di molta
varianza in meno, ed è per questo che il guadagno è massimo dove la varianza è
alta, cioè con **pochi dati per il compito principale**. Con dataset
abbondanti l'effetto si assottiglia fino a sparire, e a volte si inverte.

Una nota che allaccia il resto del libro: la **distinzione fra multi-compito e
apprendimento auto-supervisionato è meno netta di quanto sembri**. Un compito
inventato apposta perché la rete impari qualcosa (predire la rotazione di
un'immagine, ricostruire una parte mascherata) è un compito ausiliario a tutti
gli effetti, con la comodità che le sue etichette sono gratis. Le due
letterature descrivono lo stesso meccanismo arrivandoci da due direzioni.

`````

## Quando invece fa danno

C'è un però grosso, ed è la ragione per cui questa non è una tecnica da usare
sempre.

`````{tab} Elementare

Funziona **se i compiti sono imparentati**. Se non lo sono, si contendono la
stessa rappresentazione e finiscono peggio di quando erano separati. Studiare
latino aiuta l'italiano; studiare latino la sera prima di una gara di nuoto
non aiuta il nuoto, e toglie ore all'allenamento.

Il meccanismo del danno è concreto: il tronco ha una capacità finita, e ogni
pezzo di quella capacità speso per un compito che non c'entra è un pezzo tolto
a quello che conta. Peggio: i due compiti possono chiedere al tronco cose
incompatibili, e allora ogni passo che accontenta l'uno scontenta l'altro, e
l'addestramento passa il tempo a oscillare invece di migliorare.

C'è poi un problema più prosaico e altrettanto insidioso: **quanto pesa
ciascun compito** nella somma? Se un compito misura un errore in metri e un
altro una probabilità, i loro numeri non sono paragonabili, e chi ha i numeri
più grossi finisce per comandare l'addestramento senza che nessuno l'abbia
deciso. Si può regolare a mano, provando, oppure lasciare che sia la rete a
capire quanto fidarsi di ciascun compito: quelli su cui è molto incerta pesano
meno.

E la domanda a monte, «questi due compiti sono imparentati?», non ha una
formula. Si risponde provando.

`````

`````{tab} Superiore

Il rovescio ha un nome, **trasferimento negativo** (*negative transfer*), e una
diagnosi meccanica proposta da Yu e colleghi: i **gradienti in conflitto**. I
gradienti che due compiti imprimono ai parametri condivisi possono avere
prodotto scalare negativo, e allora ogni passo che aiuta l'uno danneggia
l'altro. **PCGrad** {cite}`yu2020gradient` interviene esattamente lì: quando
$\nabla\mathcal{L}_i \cdot \nabla\mathcal{L}_j < 0$, proietta ciascun gradiente
sul piano ortogonale all'altro prima di sommarli, rimuovendo la sola componente
distruttiva e lasciando intatto il resto.

Una cautela metodologica su questa diagnosi, perché è facile misurarla male: il
prodotto scalare fra gradienti va guardato **durante** l'addestramento e sui
parametri condivisi, non all'inizializzazione. All'inizio le teste sono
casuali e possono assorbire un cambio di segno senza che il tronco se ne
accorga, quindi un conflitto vero fra compiti può benissimo non comparire
come coseno negativo.

Resta il problema dei pesi $\lambda_t$, insidioso perché le loss di compiti
diversi hanno **unità e scale diverse** (una cross-entropia e un errore in
metri non sono commensurabili) e cercarli a mano è un'ottimizzazione in $T-1$
dimensioni, ciascuna delle quali costa un addestramento. La soluzione di
Kendall, Gal e Cipolla {cite}`kendall2018multi` li tratta come funzione
dell'**incertezza omoschedastica** di ciascun compito, parametrizzata da un
$\sigma_t$ **appreso**:

$$
\mathcal{L} = \sum_t \frac{1}{2\sigma_t^2}\,\mathcal{L}_t + \log \sigma_t ,
$$

dove il primo termine abbassa il peso dei compiti rumorosi e il secondo
impedisce la soluzione degenere $\sigma_t \to \infty$, che li azzererebbe
tutti. I pesi smettono di essere iperparametri e diventano parametri.

Sulla domanda a monte, **quali compiti stiano bene insieme**, non c'è una
teoria utilizzabile: le misure di affinità fra compiti sono un'area di ricerca
aperta, e in pratica si procede empiricamente, provando i compiti a coppie e
tenendo quelli che aiutano.

`````

## In pratica: il guadagno si misura, e può essere negativo

L'affermazione «un compito imparentato aiuta, uno estraneo danneggia» si può
verificare, e l'esperimento sta in una pagina. Costruiamo una situazione
realistica: il compito che ci interessa ha **poche etichette** (quaranta), il
compito ausiliario ne ha molte (ottocento). Poi confrontiamo tre addestramenti
sullo stesso identico tronco.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

D, H = 12, 64
N_ETICHETTATI, N_AUSILIARI, N_TEST = 40, 800, 2000   # poche etichette dove servono

def dati(seme):
    g = torch.Generator().manual_seed(seme)
    n = N_AUSILIARI + N_TEST
    X = torch.randn(n, D, generator=g)
    w = torch.randn(D, generator=g)
    nascosto = torch.tanh(X @ w)                 # la quantità che conta davvero
    return X, {
        "principale": nascosto,                  # etichettata solo su 40 esempi
        "parente":    nascosto ** 2,             # dipende dalla STESSA quantità
        "estraneo":   torch.randn(n, generator=g),
    }

def addestra(X, y, ausiliario, seme, passi=800):
    torch.manual_seed(seme)
    tronco = nn.Sequential(nn.Linear(D, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh())
    teste = nn.ModuleDict({k: nn.Linear(H, 1) for k in y})
    ott = torch.optim.Adam(list(tronco.parameters()) + list(teste.parameters()),
                           lr=3e-3)
    for _ in range(passi):
        # il compito principale vede 40 esempi, l'ausiliario ne vede 800
        perdita = F.mse_loss(teste["principale"](tronco(X[:N_ETICHETTATI])).squeeze(-1),
                             y["principale"][:N_ETICHETTATI])
        if ausiliario:
            perdita = perdita + F.mse_loss(
                teste[ausiliario](tronco(X[:N_AUSILIARI])).squeeze(-1),
                y[ausiliario][:N_AUSILIARI])
        ott.zero_grad(); perdita.backward(); ott.step()
    with torch.no_grad():
        pred = teste["principale"](tronco(X[N_AUSILIARI:])).squeeze(-1)
        return F.mse_loss(pred, y["principale"][N_AUSILIARI:]).item()

base = None
for ausiliario in (None, "parente", "estraneo"):
    errori = [addestra(*dati(s)[:2], ausiliario, s) for s in range(5)]
    media = sum(errori) / len(errori)
    if base is None:
        base = media
    nome = ausiliario or "nessuno"
    print(f"ausiliario: {nome:<10} errore sul test {media:.4f}"
          f"   ({100 * (media - base) / base:+.0f}%)")
```

Media su cinque semi, per non leggere il rumore:

- **nessun ausiliario**: errore $0{,}2829$. Quaranta esempi sono pochi, e si
  vede;
- **ausiliario imparentato**: $0{,}0993$, cioè **il 65% di errore in meno**. Il
  secondo compito non condivide le etichette del primo, condivide la *quantità
  nascosta* da cui entrambi dipendono, e ottocento esempi su quella quantità
  hanno insegnato al tronco quello che quaranta non bastavano a insegnare;
- **ausiliario estraneo**: $0{,}3523$, cioè **il 25% di errore in più**. Non è
  neutro: è peggio che non averlo. Le stesse capacità del tronco sono state
  spese per inseguire del rumore, e quelle capacità le ha sottratte al compito
  che contava.

Il terzo numero è il più importante dei tre, perché è quello che di solito non
si racconta. Il multi-compito non è una tecnica che si aggiunge e male che
vada non fa niente: **male che vada fa danno**, e quanto danno dipende da una
domanda che nessuna formula risolve, cioè se i compiti siano davvero
imparentati. Stabilire quali compiti stiano bene insieme è ancora, in larga
parte, un problema aperto: si misura empiricamente, provandoli a coppie, più
che deducendolo.

```{admonition} Da ricordare
:class: important
- L'**apprendimento multi-compito** addestra una rete su più compiti insieme
  con una rappresentazione condivisa: un **tronco** comune e una **testa** per
  compito (*condivisione dura*), oppure reti separate tenute vicine da una
  penalità (*condivisione morbida*).
- Il guadagno ha tre sorgenti distinte: **più segnale** (soprattutto se il
  compito principale ha poche etichette), un effetto di **regolarizzazione**
  (le scorciatoie che servono a un compito solo smettono di convenire) e un
  effetto di **attenzione** (un compito indica alla rete cosa vale la pena
  rappresentare).
- L'argomento di Caruana è statistico: i compiti condividono un **bias
  induttivo**, e cercare l'ipotesi che li soddisfa tutti restringe lo spazio
  delle soluzioni. È regolarizzazione, quindi rende di più dove la varianza è
  alta, cioè con pochi dati.
- Il rovescio è il **trasferimento negativo**: compiti non imparentati si
  contendono la rappresentazione e peggiorano il risultato. La diagnosi
  meccanica sono i **gradienti in conflitto**, e un rimedio è proiettarli
  (**PCGrad**).
- I pesi $\lambda_t$ delle loss non sono commensurabili fra compiti; si
  possono **apprendere** trattandoli come incertezza di ciascun compito
  ($\mathcal{L} = \sum_t \frac{1}{2\sigma_t^2}\mathcal{L}_t + \log\sigma_t$),
  invece di cercarli a mano.
- Misurato su un caso costruito: un ausiliario imparentato toglie il **65%**
  dell'errore, uno estraneo ne **aggiunge il 25%**. Non è una tecnica neutra.
```
