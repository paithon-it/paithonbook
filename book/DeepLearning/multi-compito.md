# Una rete, molti compiti: l'apprendimento multi-compito

Chi ha studiato il latino racconta spesso di aver capito meglio l'italiano. Le
due materie poggiano sulla stessa impalcatura, e faticare sulla declinazione di
una costringe a guardare in faccia una struttura che nell'altra si usava senza
accorgersene. Imparare due cose imparentate insieme non raddoppia la fatica, e
a volte la riduce rispetto a impararne una sola.

Le reti delle {doc}`architetture storiche <architetture-storiche>` imparavano
un compito ciascuna: classificare le fotografie di ImageNet, e nient'altro.
L'idea che imparare più compiti insieme convenga anche a una rete neurale
risale a Rich Caruana e al suo lavoro del 1997, intitolato semplicemente
*Multitask Learning* {cite}`caruana1997multitask`.

Nell’**apprendimento multi-compito** si addestra una rete su più compiti
collegati, tutti insieme, e li si fa partire dallo stesso lavoro preliminare,
cioè dagli stessi numeri intermedi, calcolati una volta sola, da cui ciascun
compito ricava la sua risposta. Questi numeri prendono il nome di
**rappresentazione condivisa**. Se i compiti sono collegati, il compito
principale *generalizza* meglio, cioè sbaglia meno su esempi mai visti. Si noti
che il guadagno viene dall'informazione che il compito in più porta a quello
principale, più che dal risparmio di memoria.

È una tecnica che si incontra dappertutto, quasi sempre senza che qualcuno la
chiami per nome. Ad esempio, il rilevatore di oggetti che dice insieme che cosa
c'è nella foto e in che punto si trova (la categoria e le quattro coordinate
del riquadro che lo racchiude) fa multi-compito. La rete che da una sola
fotografia di una strada stima insieme quanto è lontano ogni pixel e a quale
oggetto appartiene fa multi-compito. Il sistema di raccomandazione che stima
insieme la probabilità che tu clicchi su un prodotto e quella che tu lo compri
fa multi-compito, e la seconda è rara quanto preziosa, perché di acquisti se ne
vedono molti meno che di clic. In questo caso il compito abbondante aiuta
quello raro.

## Un tronco, tante teste

Prima di chiederci perché funziona, vediamo come è fatta materialmente una rete
che fa più cose.

`````{tab} Elementare

La struttura si disegna in un attimo: un **tronco** condiviso, che elabora
l'ingresso, e in cima tante **teste** quante sono le cose da predire, una per
compito. Il tronco è la rappresentazione condivisa, cioè il lavoro fatto una
volta sola e buono per tutti. Ogni testa lo traduce poi nella risposta che le
serve.

Si può pensare a un ufficio: c'è un archivio comune, dove il materiale viene
letto e ordinato una volta sola, e poi ci sono gli sportelli specializzati che
da quello stesso archivio ricavano risposte diverse. In questo modo nessuno
rilegge i documenti da capo per ogni domanda.

L'errore da ridurre, però, è uno solo. A fine giornata non si guarda uno
sportello per volta, ma si somma quanto ha sbagliato ciascuno e ne esce un
numero unico da far scendere, sistemando insieme l'archivio e gli sportelli. Nel
sommare bisogna decidere quanto pesa ciascuno sportello, e questa è la scelta
più delicata di tutte.

Non è l'unica forma possibile. A volte i compiti sono parenti ma non abbastanza
da poter condividere tutto. In questi casi si tengono due reti separate,
ciascuna col suo lavoro, e si chiede soltanto che non si allontanino troppo
l'una dall'altra.

Vicine, qui, significa con numeri simili al loro interno. In concreto, alla fine
di ogni passo si controlla quanto i pesi dell'una differiscono da quelli
dell'altra, e più la differenza cresce più sale la multa che si aggiunge
all'errore. Sono due uffici distinti che però tengono le procedure allineate.
Questa soluzione occupa più memoria, perché le reti sono due, ma non obbliga
due compiti diversi a usare la stessa identica preparazione.

Nella pratica [si finisce quasi sempre in mezzo: condiviso in basso, separato
in alto]{.pt-matita}. In basso una rete impara cose generiche (i bordi, le
forme, la struttura della frase) che servono a chiunque; in alto cose
specifiche del compito, che è giusto restino separate.

I mestieri si possono anche imparare uno alla volta, invece che tutti insieme.
Si arriva in fondo lo stesso, con un rischio che l'altra via non corre: mentre
impara il secondo mestiere l'ufficio riordina l'archivio come conviene a quello
e dimentica il primo, perché nessuno gliene chiede più conto. Imparandoli
insieme, il primo compito resta nell'errore da ridurre ogni giorno e non può
essere lasciato cadere.

`````

`````{tab} Superiore

La forma standard è la **condivisione dura** (*hard parameter sharing*), con un
tronco $g_\phi$ comune e $T$ teste $h_{\theta_t}$, addestrati minimizzando una
somma pesata

$$
\mathcal{L} = \sum_{t=1}^{T} \lambda_t \,
\mathcal{L}_t\big(h_{\theta_t}(g_\phi(\mathbf{x})),\, y_t\big).
$$

Qui $\mathbf{x}$ è l'esempio in ingresso, $\phi$ sono i parametri del tronco
condiviso e $\theta_t$ quelli della testa del compito $t$, $y_t$ è l'etichetta
di quel compito, $\mathcal{L}_t$ la sua perdita e $\lambda_t > 0$ il peso con
cui entra nella somma. Quei pesi sono il punto delicato di tutto il metodo.

L'alternativa è la **condivisione morbida** (*soft sharing*): $T$ reti
separate, ciascuna con i propri parametri, legate da un termine di
regolarizzazione che ne penalizza la distanza, per esempio $\sum_{t \neq s}
\lVert \psi_t - \psi_s \rVert_2^2$, dove $\psi_t$ sono tutti i parametri della
rete del compito $t$ (qui non c'è un tronco condiviso, quindi la $\phi$ della
condivisione dura non servirebbe).

Questa forma richiede $T$ volte i parametri ma non impone una rappresentazione
unica, e per questo si usa quando i compiti sono affini ma non sovrapponibili.
Ha un iperparametro continuo, la forza $\mu$ con cui la penalità entra nella
loss: con $\mu = 0$ le reti sono indipendenti, mentre per $\mu \to \infty$ i
parametri penalizzati diventano uguali, quindi se la penalità esclude gli strati
d'uscita il limite è la condivisione dura.

[Le forme miste condividono gli strati bassi e lasciano separati gli
alti]{.pt-matita}; sono quasi sempre la scelta pratica, per la ragione vista
nella {doc}`sezione sulle rappresentazioni gerarchiche <overview>`: le feature
generiche stanno in basso e quelle specifiche in alto, quindi il punto in cui
separare le teste dipende da quanto in alto arriva la parentela fra i compiti.

Il *transfer learning* (visto con le stesse rappresentazioni gerarchiche) è lo
stesso schema disteso nel tempo: là i compiti si affrontano in sequenza (si
pre-addestra su uno, si rifinisce sull'altro), qui in parallelo. La differenza
pratica è che il sequenziale può dimenticare il primo compito mentre impara il
secondo, e il parallelo no, perché il primo è ancora nella loss.

`````

```{figure} ../figures/multi-compito-tronco-teste.svg
:name: fig-multi-compito-forme
:alt: "Tre modi di far imparare più compiti a una rete, affiancati. A sinistra, «un tronco, tante teste»: dall'ingresso sale un unico blocco condiviso e da lì partono tre frecce verso tre teste, una per compito. Al centro, «due reti tenute vicine»: due colonne separate, ciascuna con il proprio ingresso e la propria testa, collegate da tre frecce tratteggiate a doppia punta, la penalità che impedisce loro di allontanarsi. A destra, «condiviso sotto, separato sopra»: un tronco comune in basso che si biforca a metà in due rami distinti, ciascuno con la sua testa."
:width: 96%

Le tre forme, una accanto all'altra, cambiano il punto in cui passa la linea fra
il lavoro comune e il lavoro separato di ciascun compito: in cima nel primo
disegno, in nessun punto nel secondo (le due reti non condividono niente), a
metà altezza nel terzo.
```

Le tre forme della {numref}`fig-multi-compito-forme` rispondono alla stessa
domanda: fino a che altezza i compiti si somigliano. La prima scommette che si
somiglino fino in cima, e ci rimette se la scommessa è sbagliata. La seconda
non scommette niente, e richiede il doppio dei numeri da imparare; ha però un
iperparametro, la forza della penalità che tiene vicine le due reti, e più lo si
alza più le due reti si somigliano. La terza sceglie l'altezza a cui separare
i compiti.

## Perché funziona: il compito in più fa da freno

Che funzioni è un fatto sperimentale vecchio di quasi trent'anni. Le ragioni
sono più di una e vanno tenute separate, perché dicono cose diverse su quando
aspettarsi un guadagno e quando no.

`````{tab} Elementare

Il motivo più ovvio è la quantità di segnale. Il compito in più porta con sé
altre etichette, cioè altre risposte giuste scritte accanto agli esempi, come
«in questa foto c'è un gatto». Le etichette costano, perché quasi sempre è una
persona a doverle scrivere una per una, e a volte sono semplicemente rare (di
persone che comprano ce ne sono molte meno di persone che guardano).

Ogni etichetta in più è un'occasione in più di capire com'è fatto l'ingresso, e
il guadagno è massimo quando il compito che ci sta a cuore ne ha poche e quello
di contorno ne ha tante. Al contrario, se del compito principale abbiamo già
esempi in abbondanza, questo vantaggio si assottiglia fino a sparire. Il limite
è che le etichette in più aiutano a leggere l'archivio comune, ma ogni sportello
deve imparare il proprio mestiere dai propri esempi, e lì il compito di contorno
non lo può sostituire.

Il secondo motivo è più sottile: il compito in più fa da freno. Una rete
lasciata sola con un compito trova la soluzione di comodo, e le soluzioni di
comodo sono quelle che non generalizzano. [Se la stessa rappresentazione deve
servire anche a un secondo compito, quelle soluzioni non convengono
più]{.pt-matita}, perché al secondo non servono, e la rete è spinta verso
soluzioni più generali.

In gergo questa spinta si chiama regolarizzazione, un nome che si dà a
qualunque accorgimento che, restringendo le strade che la rete può prendere, la
costringa a una risposta che vale in generale invece che a una perfetta sugli
esempi già visti. Il freno rende quando gli esempi sono pochi e la rete è
libera di inventarsi qualunque cosa. Quando gli esempi abbondano le soluzioni
di comodo sono già poche, e il compito in più, invece di aiutare, può
cominciare a ostacolare.

Il terzo motivo è l’attenzione, nel senso di tutti i giorni: certi compiti
dicono alla rete dove guardare. Se per rispondere alla seconda domanda serve un
dettaglio che per la prima sembrava trascurabile, la rete impara comunque a
rappresentarlo, e magari scopre che serviva anche alla prima.

`````

`````{tab} Superiore

L'argomento di Caruana è statistico: i compiti condividono un **bias
induttivo**. Ogni algoritmo di apprendimento ne ha uno, cioè un insieme di
assunzioni implicite che rendono preferibili certe ipotesi. Addestrare su più
compiti significa allora cercare l'ipotesi che soddisfa *tutti* i loro bias
insieme, il che restringe lo spazio delle soluzioni ammissibili.

Caruana ne distingue i meccanismi {cite}`caruana1997multitask`. Il primo è
l’*amplificazione statistica dei dati*: se due compiti usano la stessa feature
nascosta e le loro etichette hanno rumore indipendente, la rete che li impara
insieme stima quella feature mediando i due rumori, come con un campione più
grande. Ne è una conseguenza la *selezione degli attributi*, per cui, con pochi
dati e molti ingressi irrilevanti, due segnali riconoscono meglio quali
ingressi contano.

Nell’*eavesdropping*, l'origliare, una feature facile da imparare per un compito
e difficile per un altro viene appresa grazie al primo e usata dal secondo
attraverso lo strato condiviso. Infine c'è il *bias di rappresentazione*: fra le
soluzioni che la discesa del gradiente può trovare, la rete a più compiti
preferisce quelle che servono a tutti.

In definitiva, [chiedere alla stessa rappresentazione di servire più compiti
restringe lo spazio delle ipotesi con informazione vera]{.pt-matita}, e agisce da
regolarizzazione, perché il rumore specifico di un compito viene mediato via
mentre la struttura comune sopravvive. Nel compromesso bias-varianza (la
{doc}`sezione sull'overfitting </MachineLearning/overfitting-validazione>`) si
accetta un po’ di bias in cambio di molta varianza in meno, perciò il guadagno
è massimo dove la varianza è alta, cioè con pochi dati per il compito
principale. Con dataset abbondanti l'effetto si riduce fino a sparire, e a
volte si inverte.

Maurer, Pontil e Romera-Paredes ne danno la forma quantitativa per $T$ compiti
di $m$ esempi ciascuno, con una rappresentazione condivisa e un predittore per
compito {cite}`maurer2016benefit`. L'eccesso di rischio medio sui compiti,
rispetto alla scelta migliore nelle stesse classi, è limitato con probabilità
almeno $1-\delta$ da tre termini.

Per classi limitate il primo, dovuto alla rappresentazione condivisa, va come
$O\big(1/\sqrt{mT}\big)$, cioè si ripartisce su tutti gli $mT$ esempi; il
secondo, dovuto ai predittori specifici, va come $O\big(1/\sqrt{m}\big)$ e da
$T$ non dipende; il terzo, di confidenza, come
$O\big(\sqrt{\ln(1/\delta)/(mT)}\big)$. Aggiungere compiti abbassa quindi il
primo e non il secondo, per cui il guadagno è massimo quando $m$ è piccolo e il
costo sta nella rappresentazione.

Osserviamo che la distinzione fra multi-compito e apprendimento
auto-supervisionato è meno netta di quanto sembri. Un compito inventato apposta
perché la rete impari qualcosa (predire la rotazione di un'immagine, ricostruire
una parte mascherata) è infatti un compito ausiliario a tutti gli effetti, con
la comodità che le sue etichette sono gratis. Le due letterature descrivono lo
stesso meccanismo arrivandoci da due direzioni.

`````

## Quando invece fa danno

Il multi-compito ha però un limite, e per questo non è una tecnica da usare
sempre.

`````{tab} Elementare

Funziona se i compiti sono imparentati. Se non lo sono, [si contendono la
stessa rappresentazione e finiscono peggio di quando erano
separati]{.pt-matita}. Studiare latino aiuta l'italiano; studiare latino la
sera prima di una gara di nuoto non aiuta il nuoto, e toglie ore
all'allenamento.

Il meccanismo del danno è concreto. Il tronco ha una capacità finita, perché i
numeri che può regolare sono tanti ma non infiniti, e ciò che riesce a tenere a
mente ne è limitato. Di conseguenza, ogni pezzo di capacità speso per un
compito che non c'entra è sottratto al compito che conta. Peggio ancora, i due
compiti possono chiedere al tronco cose incompatibili, e allora ogni passo che
accontenta l'uno scontenta l'altro, e l'addestramento passa il tempo a
oscillare invece di migliorare.

Il litigio si può smussare invece di subirlo, e il rimedio più noto prova a
farlo: a ogni passo si guardano le correzioni che i due compiti chiedono al
tronco, da ciascuna si toglie la parte che tira contro l'altra, e il resto si
lascia intatto. Si chiama **PCGrad**, dall'inglese *projecting conflicting
gradients*, «proiettare i gradienti in conflitto».

L'idea è pulita e la diagnosi che la ispira è fondata, ma sulla cura si discute
ancora. Chi ha rifatto i confronti su larga scala ha infatti trovato che la
semplice somma dei due errori, purché regolata con gli stessi accorgimenti che
si userebbero per un compito solo (i freni contro l'imparare a memoria, il
passo scelto con cura), tiene il passo o fa meglio.

C'è poi un problema più prosaico e altrettanto insidioso, e nasce dal fatto che
l'errore da far scendere è uno solo: si prende quanto la rete sbaglia sul primo
compito, si prende quanto sbaglia sul secondo e si sommano. Quanto deve pesare
ciascuno dei due in quella somma? Se un compito misura un errore in metri e un
altro una probabilità, i loro numeri non sono paragonabili, e chi ha i numeri
più grossi finisce per comandare l'addestramento senza che nessuno l'abbia
deciso.

I pesi si possono regolare a mano, provando, oppure si può lasciare che sia la
rete a capire quanto fidarsi di ciascun compito: quelli su cui è molto incerta
pesano meno. Dichiararsi incerta, però, deve avere una penalità, altrimenti la
strada più comoda è dirsi incertissima su tutto, dare a ogni compito un peso
quasi nullo e ritrovarsi con un errore totale vicino a zero senza aver imparato
niente.

La domanda a monte, «questi due compiti sono imparentati?», non ha una formula,
e si risponde provando.

`````

`````{tab} Superiore

Il rovescio ha un nome, **trasferimento negativo** (*negative transfer*), e una
diagnosi meccanica proposta da Yu e colleghi: i **gradienti in conflitto**. I
gradienti che due compiti imprimono ai parametri condivisi [possono avere
prodotto scalare negativo, e allora ogni passo che aiuta l'uno danneggia
l'altro]{.pt-matita}. **PCGrad** {cite}`yu2020gradient` interviene su quel
punto. Chiamati $\mathbf{g}_i = \nabla_\phi \mathcal{L}_i$ i gradienti dei
compiti rispetto ai soli parametri condivisi, quando $\mathbf{g}_i^\top
\mathbf{g}_j < 0$ si sostituisce $\mathbf{g}_i$ con la sua proiezione sul piano
normale a $\mathbf{g}_j$,

$$
\mathbf{g}_i \leftarrow \mathbf{g}_i - \frac{\mathbf{g}_i^\top \mathbf{g}_j}{\lVert \mathbf{g}_j \rVert^2}\,\mathbf{g}_j ,
$$

e simmetricamente $\mathbf{g}_j$ contro il $\mathbf{g}_i$ originale, prima di
sommarli. In questo modo si toglie la sola componente distruttiva e si lascia
intatto il resto. Con più di due compiti ogni gradiente si proietta contro gli
altri uno alla volta, in ordine casuale, perché il risultato dipende
dall'ordine. Il costo è computazionale, poiché invece di un solo passaggio
all'indietro sulla somma delle perdite ne servono $T$, e in memoria $T$
gradienti della dimensione di $\phi$.

Il beneficio pratico di questa famiglia di metodi è però contestato.
Confronti su larga scala trovano infatti che la somma pesata semplice, purché
regolarizzata e stabilizzata come si farebbe per un compito solo, li eguaglia o
li batte {cite}`kurin2022defense`, e che su compiti di visione e di linguaggio
non producono guadagni oltre l'ottimizzazione ordinaria {cite}`xin2022current`.
Il conflitto fra gradienti resta una buona diagnosi, ma che proiettarli sia la
cura è meno stabilito di quanto la letteratura sui metodi lasci pensare.

Serve una cautela metodologica su questa diagnosi, perché è facile misurarla
male: il prodotto scalare fra gradienti va guardato durante l'addestramento e
sui parametri condivisi, non all'inizializzazione. All'inizio le teste sono
casuali e possono assorbire un cambio di segno senza che il tronco se ne
accorga, quindi un conflitto vero fra compiti può benissimo non comparire come
coseno negativo.

Resta il problema dei pesi $\lambda_t$, insidioso perché le loss di compiti
diversi hanno unità e scale diverse (una cross-entropia e un errore in metri
non sono commensurabili). Inoltre cercarli a mano è una ricerca in $T-1$
dimensioni, in cui ogni combinazione provata richiede un addestramento
completo.

La soluzione di Kendall, Gal e Cipolla {cite}`kendall2018multi` li tratta come
funzione dell’**incertezza omoschedastica** di ciascun compito, parametrizzata
da un $\sigma_t$ appreso. Omoschedastica perché quel $\sigma_t$ dipende dal
compito e non dall'esempio, al contrario della regressione eteroschedastica di
{doc}`Da dove viene la loss </RetiNeurali/da-dove-viene-la-loss>`, da cui prende
però la stessa forma. Per compiti tutti di regressione:

$$
\mathcal{L} = \sum_t \left[\frac{1}{2\sigma_t^2}\,\mathcal{L}_t
+ \log \sigma_t\right] ,
$$

dove il primo termine abbassa il peso dei compiti rumorosi e il secondo
impedisce la soluzione degenere $\sigma_t \to \infty$, che li azzererebbe
tutti. I pesi smettono così di essere iperparametri e diventano parametri.

A pesi della rete fissati, il minimo in $\sigma_t$ si trova in forma chiusa ed è
$\sigma_t^2 = \mathcal{L}_t$, dove il compito pesa $1/(2\mathcal{L}_t)$,
inversamente alla propria loss corrente, e la loss congiunta vale
$\tfrac12\sum_t (1 + \log\mathcal{L}_t)$. In altri termini, il criterio equivale
a minimizzare la somma dei logaritmi delle loss, a meno di costanti, e nessun
compito può comandare solo perché i suoi errori sono numeri grandi. In pratica
si impara $s_t = \log\sigma_t^2$, che evita la divisione per zero e non ha
vincoli di segno.

Il fattore cambia però con il tipo di compito. In un compito di classificazione
$\sigma_t$ ha un'altra funzione, la temperatura di una softmax, e il peso che ne
discende è $1/\sigma_t^2$, senza il $2$. Per una regressione e una
classificazione, ad esempio, la loss congiunta è
$\frac{1}{2\sigma_1^2}\mathcal{L}_1 +
\frac{1}{\sigma_2^2}\mathcal{L}_2 + \log\sigma_1 + \log\sigma_2$, e l'ultimo
passaggio che le dà questa forma è un'approssimazione, esatta soltanto per
$\sigma_2 = 1$.

Sulla domanda a monte, quali compiti stiano bene insieme, non c'è una teoria
utilizzabile. Le misure di affinità fra compiti sono un'area di ricerca
aperta, e in pratica si procede empiricamente, provando i compiti a coppie e
tenendo quelli che aiutano.

`````

## In pratica: il guadagno si misura, e può essere negativo

L'affermazione «un compito imparentato aiuta, uno che non c'entra niente
danneggia» si può verificare con un esperimento.

Inventiamo una **quantità nascosta**, cioè un numero che non si vede, ricavato
dall'ingresso con una regola fissa. Il compito principale, quello che ci sta a
cuore, chiede di indovinare quel numero, e ha pochissimi esempi etichettati,
quaranta. Accanto gli si mette un compito **ausiliario**, cioè un compito di
contorno che serve solo ad aiutare il primo e che ha molti esempi, ottocento.

Proviamo due ausiliari: uno imparentato, che chiede il *quadrato* della stessa
quantità nascosta, e uno che non c'entra niente, il cui bersaglio è puro caso e
che nessuna rete può imparare. A questo punto confrontiamo tre addestramenti
sullo stesso identico tronco (senza ausiliario, con quello imparentato, con
quello inutile), e poi ancora due volte quello inutile, pesato di meno.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)   # su una macchina carica i thread si ostacolano

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
        # bersaglio che non dipende da X: nessuna rete può impararlo
        "rumore":     torch.randn(n, generator=g),
    }

def addestra(X, y, ausiliario, seme, passi=800, peso=1.0):
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
            perdita = perdita + peso * F.mse_loss(
                teste[ausiliario](tronco(X[:N_AUSILIARI])).squeeze(-1),
                y[ausiliario][:N_AUSILIARI])
        ott.zero_grad(); perdita.backward(); ott.step()
    with torch.no_grad():
        pred = teste["principale"](tronco(X[N_AUSILIARI:])).squeeze(-1)
        return F.mse_loss(pred, y["principale"][N_AUSILIARI:]).item()

def prova(ausiliario, peso=1.0):
    """L'errore sul test di ciascuno dei cinque semi."""
    return [addestra(*dati(s)[:2], ausiliario, s, peso=peso) for s in range(5)]

base = prova(None)                       # cinque semi per configurazione
media_base = sum(base) / len(base)
print(f"ausiliario: {'nessuno':<14} errore sul test {media_base:.4f}   (  +0%)")
for ausiliario, peso in (("parente", 1.0), ("rumore", 1.0),
                         ("rumore", 0.1), ("rumore", 0.01)):
    errori = prova(ausiliario, peso)
    media = sum(errori) / len(errori)
    # lo stesso seme con e senza ausiliario: stessi dati, stessa partenza
    per_seme = [100 * (e - b) / b for e, b in zip(errori, base)]
    nome = ausiliario if peso == 1.0 else f"{ausiliario} x{peso}"
    print(f"ausiliario: {nome:<14} errore sul test {media:.4f}"
          f"   ({100 * (media - media_base) / media_base:+4.0f}%;"
          f" semi da {min(per_seme):+.0f}% a {max(per_seme):+.0f}%)")
```

```text
ausiliario: nessuno        errore sul test 0.2829   (  +0%)
ausiliario: parente        errore sul test 0.0993   ( -65%; semi da -82% a -39%)
ausiliario: rumore         errore sul test 0.3523   ( +25%; semi da +15% a +36%)
ausiliario: rumore x0.1    errore sul test 0.3038   (  +7%; semi da -2% a +17%)
ausiliario: rumore x0.01   errore sul test 0.2871   (  +1%; semi da -2% a +7%)
```

Ogni numero è la media di cinque prove che differiscono solo per il seme, cioè
per il punto da cui parte il generatore di numeri casuali, e quindi hanno pesi
iniziali diversi e dati diversi. Una prova sola, infatti, misurerebbe anche la
fortuna.

Presi da soli, i numeri non dicono niente, perché dipendono dalla scala delle
quantità da indovinare, che abbiamo scelto noi; quello che conta è il
confronto. Fra parentesi, accanto alla variazione media, ci sono la più bassa e
la più alta fra i cinque semi, ciascuno confrontato con lo stesso seme senza
ausiliario (stessi dati, stessa partenza).

- nessun ausiliario: errore $0{,}2829$. Quaranta esempi sono pochi, e si
  vede;
- ausiliario imparentato: $0{,}0993$, cioè il 65% di errore in meno, e su
  ogni seme fra il 39% e l’82% in meno. Il
  secondo compito non condivide le etichette del primo, ma la *quantità
  nascosta* da cui entrambi dipendono, e ottocento esempi su quella quantità
  hanno insegnato al tronco più di quanto potessero quaranta;
- ausiliario che non ha niente da insegnare: $0{,}3523$, cioè il 25% di
  errore in più, e per giunta su tutti e cinque i semi, fra il 15% e il 36%;
  è peggio che non averlo. La capacità del tronco spesa a inseguire quel
  bersaglio è capacità sottratta al compito che contava.

Il risultato dell'ausiliario inutile è il più importante dei tre: [il
multi-compito non è una tecnica che si aggiunge senza rischio, perché nel caso
peggiore fa danno]{.pt-matita}. Si noti però che il bersaglio di
quell'ausiliario è rumore puro, un caso estremo che nessuna rete può imparare,
quindi l'esperimento misura con precisione la prima delle due cause di danno,
la capacità del tronco sprecata. Un compito diverso ma *imparabile* fa in
genere molto meno danno di così.

Il $+25\%$, poi, non dice niente su quanto debba pesare ciascun compito. Le
righe `parente` e `rumore` sommano i due errori con peso uguale, cioè
$\lambda_1 = \lambda_2 = 1$ (le $\lambda$ sono i pesi con cui i due compiti
entrano nella somma). È la scelta più innocente possibile, e anche la meno
difendibile, perché sono i pesi a comandare l'addestramento senza che nessuno
li abbia decisi.

Le ultime due righe rifanno le stesse cinque prove con l'ausiliario pesato
$\lambda = 0{,}1$ e $\lambda = 0{,}01$. Nel primo caso il danno medio scende a
$+7\%$, e su almeno un seme l'ausiliario aiuta (da $-2\%$ a $+17\%$); nel
secondo scende a $+1\%$, che sui cinque semi non si distingue da zero (da
$-2\%$ a $+7\%$). Il *trasferimento negativo*, cioè il danno che un compito fa
a un altro, è insomma una proprietà della coppia e del peso che le si dà, non
della sola coppia di compiti.

Un'avvertenza simile serve per il $-65\%$, il risultato migliore dei tre. Il
compito «parente» è imparentato quanto è possibile esserlo: il suo bersaglio si
ricava dalla stessa quantità nascosta del compito principale con un calcolo
fisso (il quadrato), e nient'altro. Non ha quindi una sola difficoltà propria,
e chi ha imparato il principale ha già in mano tutto il necessario per il
«parente».

Il contrario vale quasi, ma manca il segno. Un numero e il suo opposto hanno lo
stesso quadrato, quindi dal «parente» il tronco impara quale combinazione degli
ingressi guardare, cioè la direzione $\mathbf{w}$ della quantità nascosta
$\tanh(\mathbf{x}^\top\mathbf{w})$, ma il segno di quella quantità deve
impararlo dai soli quaranta esempi etichettati. Il $-65\%$ è dunque vicino al
massimo che un compito in più può dare, non a un caso tipico, perché due compiti
davvero distinti condividono una parte della struttura, non tutta.

Che compiti in competizione peggiorino il risultato è documentato anche su
scala ben più grande di questa {cite}`standley2020tasks`, e quanto danno
facciano dipende da una domanda che nessuna formula risolve: se i compiti siano
imparentati. Stabilire quali stiano bene insieme è in larga parte un problema
aperto, e si procede misurando, provandoli a coppie, più che deducendolo.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Una rete può imparare più cose insieme: un tronco comune, che legge
  l'ingresso una volta sola per tutti, e in cima una testa per ogni
  risposta da dare. È l'ufficio con l'archivio comune e gli sportelli
  specializzati.
- Il compito in più aiuta per tre motivi diversi: porta altri esempi;
  fa da freno, perché le soluzioni di comodo per un compito solo smettono di
  convenire quando la stessa preparazione deve servire anche a un altro; e
  indica dove guardare, cioè quali dettagli valeva la pena notare.
- Aiuta soprattutto quando del compito che ci sta a cuore abbiamo pochi
  esempi. Se ne abbiamo tanti, il vantaggio si assottiglia fino a sparire.
- Ma solo se i compiti sono imparentati. Se non lo sono si contendono lo
  stesso tronco e finiscono peggio di quando erano separati: il latino aiuta
  l'italiano, non il nuoto. Non c'è una formula per saperlo prima: si prova.
- Un altro problema pratico è quanto pesa ciascun compito nella somma
  dell'errore: se uno misura metri e l'altro una probabilità, quello con i
  numeri più grossi comanda l'addestramento senza che nessuno l'abbia deciso.
  Si può regolare a mano, oppure lasciare che sia la rete a fidarsi di meno dei
  compiti su cui è più incerta.
- Su un caso costruito apposta, un compito ausiliario imparentato ha tolto il
  65% dell'errore, uno fatto di numeri casuali ne ha aggiunto il 25%. Non è una
  tecnica neutra. Sono però i due estremi, e contati dando ai due compiti lo
  stesso peso nella somma: dando all'ausiliario un peso dieci volte minore, il
  danno medio scende dal 25% al 7%.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- L’apprendimento multi-compito addestra una rete su più compiti insieme
  con una rappresentazione condivisa: un tronco comune e una testa per
  compito (*condivisione dura*), oppure reti separate tenute vicine da una
  penalità (*condivisione morbida*).
- Il guadagno ha più meccanismi, che Caruana distingue: l'amplificazione
  statistica (più segnale sulla stessa feature, soprattutto se il compito
  principale ha poche etichette), la selezione degli attributi e
  l'eavesdropping (un compito indica alla rete che cosa conviene
  rappresentare), il bias di rappresentazione (la ricerca preferisce le
  soluzioni che servono a tutti). Nella forma quantitativa di Maurer e
  colleghi il termine della rappresentazione condivisa si ripartisce su tutti
  gli $mT$ esempi, quello dei predittori specifici no.
- L'argomento di Caruana è statistico: i compiti condividono un bias
  induttivo, e cercare l'ipotesi che li soddisfa tutti restringe lo spazio
  delle soluzioni. È regolarizzazione, quindi rende di più dove la varianza è
  alta, cioè con pochi dati.
- Il rovescio è il trasferimento negativo: compiti non imparentati si
  contendono la rappresentazione e peggiorano il risultato. La diagnosi
  meccanica sono i gradienti in conflitto; proiettarli (PCGrad) è il
  rimedio più noto, ma che serva davvero è contestato dai confronti su larga
  scala.
- Le loss dei compiti non sono commensurabili fra loro, e così i pesi
  $\lambda_t$ non si fissano a occhio; si possono apprendere trattandoli come
  incertezza di ciascun compito
  ($\mathcal{L} = \sum_t [\mathcal{L}_t/(2\sigma_t^2) + \log\sigma_t]$ per una
  regressione, con $1/\sigma_t^2$ al posto di $1/(2\sigma_t^2)$ per una
  classificazione), invece di cercarli a mano; all'ottimo ogni compito pesa
  l'inverso della propria loss, e il criterio diventa la somma dei logaritmi
  delle loss.
- Su un caso costruito, un ausiliario imparentato toglie il 65% dell'errore,
  un ausiliario fatto di puro rumore ne aggiunge il 25%. Non è una tecnica
  neutra. Entrambi i numeri sono estremi (il «parente» è una funzione
  deterministica del bersaglio principale) e valgono per $\lambda_t$ uguali:
  con l'ausiliario pesato $0{,}1$ il danno medio scende al 7%, con $0{,}01$
  all’1%, che sui cinque semi non si distingue da zero.
```
`````

Il tronco condiviso serve più compiti perché la profondità costruisce una scala
di descrizioni, dal bordo alla forma, che non appartiene al compito per cui è
nata. Da qui la domanda del {doc}`meta-apprendimento <meta-apprendimento>`: se
quella scala non appartiene ai compiti che l'hanno costruita, quanti esempi
bastano per farla servire a un compito che non esisteva quando la si costruiva?
