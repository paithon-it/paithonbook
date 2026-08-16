# Una rete, molti compiti: l'apprendimento multi-compito

Chi ha studiato il latino racconta spesso di aver capito meglio l'italiano.
Non è un modo di dire affettuoso verso il liceo classico: è che le due materie
poggiano sulla stessa impalcatura, e faticare sulla declinazione di una
costringe a guardare in faccia una struttura che nell'altra si usava senza
accorgersene. Imparare due cose imparentate insieme non costa il doppio, e a
volte costa meno che impararne una sola.

L'idea che questo valga anche per una rete neurale ha un articolo di
riferimento e una data: Rich Caruana, 1997, in un lavoro il cui titolo è
semplicemente *Multitask Learning* {cite}`caruana1997multitask`.

La tesi è netta. Si addestra una rete su più compiti collegati, tutti insieme, e
si fa in modo che partano dallo stesso lavoro preliminare: gli stessi numeri
intermedi, calcolati una volta sola, da cui poi ciascun compito ricava la sua
risposta. È la **rappresentazione condivisa**. Il risultato è che **ciascuno**
dei compiti funziona meglio su esempi mai visti, che è quello che chiamiamo
*generalizzazione*. Non è un trucco per risparmiare memoria: è il compito in più
che insegna qualcosa al compito principale.

Vale la pena affrontarla qui perché è una tecnica che il libro incontra
dappertutto senza mai chiamarla per nome. Il rilevatore di oggetti che dice
insieme che cosa c'è nella foto e in che punto si trova (la categoria e le
quattro coordinate del riquadro che lo racchiude) fa multi-compito. La rete che
da una sola fotografia di una strada stima insieme quanto è lontano ogni pixel
e a quale oggetto appartiene fa multi-compito. Il sistema di raccomandazione che
stima insieme la probabilità che tu clicchi su un prodotto e quella che tu lo
compri fa multi-compito, e la seconda è rara quanto preziosa: di acquisti se ne
vedono molti meno che di clic, e il compito abbondante aiuta quello raro.

## Un tronco, tante teste

Prima del perché, la forma: come è fatta materialmente una rete che fa più cose.

`````{tab} Elementare

La struttura si disegna in un attimo: un **tronco** condiviso, che elabora
l'ingresso, e in cima tante **teste** quante sono le cose da predire, una per
compito. Il tronco è la rappresentazione condivisa di cui parlavamo: il lavoro
fatto una volta sola e buono per tutti. Ogni testa lo traduce nella risposta
che le serve.

L'immagine giusta è quella di un ufficio: c'è un archivio comune, dove il
materiale viene letto e ordinato una volta sola, e poi ci sono gli uffici
specializzati che da quello stesso archivio ricavano risposte diverse. Nessuno
rilegge i documenti da capo per ogni domanda.

Non è l'unica forma possibile. A volte i compiti sono parenti ma non abbastanza
da poter condividere tutto: allora si tengono due reti separate, ciascuna col
suo lavoro, e si chiede solo che **non si allontanino troppo** l'una
dall'altra. Vicine, qui, vuol dire con numeri simili al loro interno: alla fine
di ogni passo si controlla quanto i pesi dell'una differiscono da quelli
dell'altra, e più la differenza cresce più aumenta la multa. Sono due uffici
distinti che però tengono le procedure allineate. Costa di più, perché le reti
sono due, ma non obbliga due compiti diversi a usare per forza la stessa
identica preparazione.

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

```{figure} ../figures/multi-compito-tronco-teste.svg
:name: fig-multi-compito-forme
:alt: "Tre modi di far imparare più compiti a una rete, affiancati. A sinistra, «un tronco, tante teste»: dall'ingresso sale un unico blocco condiviso e da lì partono tre frecce verso tre teste, una per compito. Al centro, «due reti tenute vicine»: due colonne separate, ciascuna con il proprio ingresso e la propria testa, collegate da tre frecce tratteggiate a doppia punta, la penalità che impedisce loro di allontanarsi. A destra, «condiviso sotto, separato sopra»: un tronco comune in basso che si biforca a metà in due rami distinti, ciascuno con la sua testa."
:width: 96%

Le tre forme, una accanto all'altra. Quello che cambia è **dove** passa la
linea fra ciò che i compiti fanno insieme e ciò che ciascuno fa per conto suo:
nel primo disegno passa in cima, nel secondo non passa affatto (le due reti non
condividono niente), nel terzo passa a metà altezza.
```

Messe in fila come in {numref}`fig-multi-compito-forme`, le tre forme si
rivelano tre risposte alla stessa domanda: fino a che altezza i compiti si
somigliano. La prima scommette che si somiglino fino in cima, e ci rimette se la
scommessa è sbagliata. La seconda non scommette niente, e il conto lo paga in
numeri da imparare, che sono il doppio. La terza sceglie un'altezza, ed è
l'unica delle tre che si può regolare.

## Perché funziona: il compito in più fa da freno

Che funzioni è un fatto sperimentale vecchio di trent'anni; il perché, invece,
non è uno solo. Conviene tenere separate le ragioni, perché dicono cose diverse
su quando aspettarsi un guadagno e quando no.

`````{tab} Elementare

Il guadagno viene da tre parti, e conviene tenerle distinte perché non sono la
stessa cosa.

Il primo è il più ovvio: **più segnale**. Il compito in più porta con sé altre
**etichette**, cioè altre risposte giuste scritte accanto agli esempi, come «in
questa foto c'è un gatto». Le etichette costano, perché quasi sempre è una
persona a doverle scrivere una per una, e a volte sono semplicemente rare (di
persone che comprano ce ne sono molte meno di persone che guardano). Ogni
etichetta in più è un'occasione in più di capire com'è fatto l'ingresso, e il
guadagno è massimo quando il compito che ci sta a cuore ne ha poche e quello di
contorno ne ha tante. Al contrario, se del compito principale abbiamo già
esempi in abbondanza, questo primo vantaggio si assottiglia fino a sparire.

Il secondo è più sottile ed è il vero motivo per cui la cosa funziona: il
compito in più fa da **freno**. Una rete lasciata sola con un compito trova la
scorciatoia più comoda per risolverlo, e le scorciatoie sono proprio ciò che
non generalizza. Se la stessa rappresentazione deve servire anche a un secondo
compito, quelle scorciatoie smettono di essere convenienti, perché al secondo
non servono. La rete è spinta verso soluzioni più generali, e questo è
esattamente ciò che in gergo si chiama **regolarizzazione**: qualunque
accorgimento che, restringendo le strade che la rete può prendere, la costringa
a una risposta che vale in generale invece che a una perfetta sugli esempi già
visti.

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
bias-varianza già incontrato, si accetta un po’ di bias in cambio di molta
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

Il meccanismo del danno è concreto. Il tronco ha una **capacità** finita: i
numeri che può regolare sono tanti, ma sono un numero preciso, e quello che
riesce a tenere a mente è limitato da quanti sono. Ogni pezzo di quella capacità
speso per un compito che non c'entra è un pezzo tolto a quello che conta.
Peggio: i due compiti possono chiedere al tronco cose incompatibili, e allora
ogni passo che accontenta l'uno scontenta l'altro, e l'addestramento passa il
tempo a oscillare invece di migliorare.

C'è poi un problema più prosaico e altrettanto insidioso. Quando la rete impara
due cose insieme, l'errore che si cerca di ridurre è **uno solo**: si prende
quanto sbaglia sul primo compito, si prende quanto sbaglia sul secondo e si
sommano. Ma quanto deve pesare ciascuno dei due in quella somma? Se un compito
misura un errore in metri e un altro una probabilità, i loro numeri non sono
paragonabili, e chi ha i numeri più grossi finisce per comandare l'addestramento
senza che nessuno l'abbia deciso. Si può regolare a mano, provando, oppure
lasciare che sia la rete a capire quanto fidarsi di ciascun compito: quelli su
cui è molto incerta pesano meno.

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

Va detto che il beneficio pratico di questa famiglia di metodi è **contestato**:
confronti su larga scala trovano che la somma pesata semplice, purché
regolarizzata e stabilizzata come si farebbe per un compito solo, li eguaglia o
li batte {cite}`kurin2022defense`, e che su compiti di visione e di linguaggio
non producono guadagni oltre l'ottimizzazione ordinaria {cite}`xin2022current`.
Il conflitto fra gradienti resta una buona **diagnosi**; che proiettarli sia la
**cura** è meno stabilito di quanto la letteratura sui metodi lasci pensare.

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
dell’**incertezza omoschedastica** di ciascun compito, parametrizzata da un
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

L'affermazione «un compito imparentato aiuta, uno che non c'entra niente
danneggia» si può verificare, e l'esperimento sta in una pagina.

L'impianto è questo. Si inventa una **quantità nascosta**: un numero che non si
vede, ricavato dall'ingresso con una regola fissa. Il compito **principale**,
quello che ci sta a cuore, chiede di indovinare proprio quel numero, e di
esempi etichettati ne ha pochissimi, quaranta. Accanto gli si mette un compito
**ausiliario**, cioè un compito di contorno che serve solo ad aiutare il primo,
e di esempi ne ha molti, ottocento. Di ausiliari ne proviamo due: uno
imparentato, che chiede il *quadrato* della stessa quantità nascosta, e uno che
non c'entra niente, il cui bersaglio è puro caso e che nessuna rete può
imparare.

Poi si confrontano tre addestramenti sullo stesso identico tronco: senza
ausiliario, con quello imparentato, con quello inutile. Il codice qui sotto è
lungo, ma si può saltare senza perdere il filo: quello che conta sono i tre
numeri che stampa, commentati subito dopo.

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
        # bersaglio che non dipende da X: nessuna rete può impararlo
        "rumore":     torch.randn(n, generator=g),
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
# cinque semi per tre configurazioni: qualche minuto di attesa
for ausiliario in (None, "parente", "rumore"):
    errori = [addestra(*dati(s)[:2], ausiliario, s) for s in range(5)]
    media = sum(errori) / len(errori)
    if base is None:
        base = media
    nome = ausiliario or "nessuno"
    print(f"ausiliario: {nome:<10} errore sul test {media:.4f}"
          f"   ({100 * (media - base) / base:+.0f}%)")
```

Ogni numero è la media di cinque prove che differiscono solo per il **seme**,
cioè per il punto da cui parte il generatore di numeri casuali: pesi iniziali
diversi, dati diversi. Una prova sola misurerebbe anche la fortuna. E i numeri
presi da soli non dicono niente, perché dipendono da quanto sono grandi le
quantità che stiamo cercando di indovinare, che le abbiamo scelte noi: conta il
confronto fra i tre.

- **nessun ausiliario**: errore $0{,}2829$. Quaranta esempi sono pochi, e si
  vede;
- **ausiliario imparentato**: $0{,}0993$, cioè **il 65% di errore in meno**. Il
  secondo compito non condivide le etichette del primo, condivide la *quantità
  nascosta* da cui entrambi dipendono, e ottocento esempi su quella quantità
  hanno insegnato al tronco quello che quaranta non bastavano a insegnare;
- **ausiliario che non ha niente da insegnare**: $0{,}3523$, cioè **il 25% di
  errore in più**. Non è neutro: è peggio che non averlo. La capacità del tronco
  spesa a inseguire quel bersaglio è capacità sottratta al compito che contava.

Il terzo numero è il più importante dei tre, perché è quello che di solito non
si racconta: il multi-compito non è una tecnica che si aggiunge e male che
vada non fa niente. **Male che vada fa danno.** Va però letto per quello che è:
il bersaglio del terzo braccio è rumore puro, un caso estremo che nessuna rete
può imparare, quindi quello che l'esperimento misura con precisione è la prima
delle due cause di danno, la capacità del tronco sprecata. Un compito diverso
ma *imparabile* fa in genere molto meno danno di così.

C'è una seconda cosa che quel $+25\%$ non dice, ed è la manopola che
l'esperimento non tocca. Nel codice i due errori si sommano con peso uguale,
cioè $\lambda_1 = \lambda_2 = 1$ (le $\lambda$ sono i pesi con cui i due compiti
entrano nella somma): la scelta più innocente possibile, e anche la meno
difendibile, visto che la sezione precedente ha detto che sono proprio quei pesi
a comandare l'addestramento senza che nessuno l'abbia deciso. Rifacendo le
stesse cinque prove con l'ausiliario pesato $\lambda = 0{,}1$ il danno scende a
$+7\%$, e con $\lambda = 0{,}01$ a $+1{,}5\%$. Il trasferimento negativo,
insomma, non è una proprietà della sola coppia di compiti: è una proprietà della
coppia **e** del peso che le si dà.

E per simmetria va detto anche del primo numero, il migliore dei tre. Il compito
«parente» è imparentato quanto è possibile esserlo: il suo bersaglio si ricava
dalla stessa quantità nascosta del compito principale con un conto fisso (il
quadrato), e nient'altro. Non ha una sola difficoltà propria: chi ha imparato
l'uno ha già in mano tutto quello che serve all'altro. Il $-65\%$ è dunque
vicino al tetto di quello che un compito in più può fare, non a un caso tipico:
due compiti davvero distinti condividono una parte della struttura, non tutta.

Che compiti in competizione peggiorino il risultato è comunque un fatto
documentato su scala ben più grande di questa {cite}`standley2020tasks`, e
quanto danno facciano dipende da una domanda che nessuna formula risolve, cioè
se i compiti siano davvero imparentati. Stabilire quali stiano bene insieme è
ancora, in larga parte, un problema aperto: si misura empiricamente,
provandoli a coppie, più che deducendolo.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Una rete può imparare **più cose insieme**: un **tronco** comune, che legge
  l'ingresso una volta sola per tutti, e in cima una **testa** per ogni
  risposta da dare. È l'ufficio con l'archivio comune e gli sportelli
  specializzati.
- Il compito in più aiuta per tre motivi diversi: porta **altri esempi**;
  fa da **freno**, perché le scorciatoie comode per un compito solo smettono di
  convenire quando la stessa preparazione deve servire anche a un altro; e
  **indica dove guardare**, cioè quali dettagli valeva la pena notare.
- Aiuta soprattutto quando del compito che ci sta a cuore abbiamo **pochi
  esempi**. Se ne abbiamo tanti, il vantaggio si assottiglia fino a sparire.
- Ma solo **se i compiti sono imparentati**. Se non lo sono si contendono lo
  stesso tronco e finiscono peggio di quando erano separati: il latino aiuta
  l'italiano, non il nuoto. E non c'è una formula per saperlo prima: si prova.
- Un altro problema pratico è **quanto pesa ciascun compito** nel conto
  dell'errore: se uno misura metri e l'altro una probabilità, quello con i
  numeri più grossi comanda l'addestramento senza che nessuno l'abbia deciso.
  Si può regolare a mano, oppure lasciare che sia la rete a fidarsi di meno dei
  compiti su cui è più incerta.
- Misurato su un caso costruito apposta: un compito ausiliario imparentato ha
  tolto il **65%** dell'errore, uno fatto di numeri casuali ne ha **aggiunto il
  25%**. Non è una tecnica neutra. Sono però i due estremi, e contati dando ai
  due compiti lo stesso peso nella somma: dando all'ausiliario un peso dieci
  volte minore, il danno scende dal 25% al 7%.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- L’**apprendimento multi-compito** addestra una rete su più compiti insieme
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
  meccanica sono i **gradienti in conflitto**; proiettarli (**PCGrad**) è il
  rimedio più noto, ma che serva davvero è contestato dai confronti su larga
  scala.
- I pesi $\lambda_t$ delle loss non sono commensurabili fra compiti; si
  possono **apprendere** trattandoli come incertezza di ciascun compito
  ($\mathcal{L} = \sum_t \frac{1}{2\sigma_t^2}\mathcal{L}_t + \log\sigma_t$),
  invece di cercarli a mano.
- Misurato su un caso costruito: un ausiliario imparentato toglie il **65%**
  dell'errore, un ausiliario fatto di puro rumore ne **aggiunge il 25%**. Non è
  una tecnica neutra. Entrambi i numeri sono estremi (il «parente» è una
  funzione deterministica del bersaglio principale) e valgono a
  $\lambda_t$ uguali: con l'ausiliario pesato $0{,}1$ il danno scende al 7%, con
  $0{,}01$ all'1,5%.
```
`````
