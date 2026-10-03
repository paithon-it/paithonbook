# Un modello piccolo che imita: la distillazione

Un maestro guarda una cifra scritta a mano da un bambino e dice: «È un 7». Un
altro maestro guarda la stessa cifra e dice: «È un 7, ma per un soffio: è
scritta in un modo che poteva farla scambiare per un 1, e in nessun caso la si
sarebbe potuta prendere per un 8».

Il secondo maestro ha detto la stessa cosa del primo più qualcos’altro, e quel
qualcos’altro non riguarda quella cifra soltanto: riguarda come sono fatte le
cifre. Che 7 e 1 si somiglino, e che 7 e 8 no, è una cosa che il maestro ha
imparato in anni di cifre lette, e che l’etichetta «7» da sola non trasmette.

La distillazione è il modo di passare quel sapere a un modello piccolo, e la
domanda interessante è che cosa passi, oltre a quello che dice l’etichetta.

## Che cosa c’è dentro un «quasi»

Le due leve precedenti stringevano un modello già fatto. Qui si fa un’altra
cosa: si costruisce un modello nuovo, piccolo fin dall’inizio, e lo si addestra
anche su quello che il modello grande risponderebbe, dubbi compresi, oltre che
sulle risposte giuste.

`````{tab} Elementare

Un maestro che legge cifre da trent’anni, se glielo si chiede, dice quanto
scommetterebbe su ciascuna delle dieci cifre, quasi tutto sul sette, un pochino
sull’uno, niente sull’otto. Il guaio è che glielo si deve chiedere. Lasciato
fare taglia corto, «sette», con una sicurezza da 0,9999, e i ripensamenti
restano un borbottio che nessuno sente.

Allora gli si mette davanti una manopola, e più la si alza più lui si dilunga.
A uno parla come sempre. A quattro, quel «0,9999» diventa «0,9 sul
sette, 0,09 sull’uno, 0,001 sull’otto», il sette resta il primo e affiora la
forma del dubbio, come le ombre di una fotografia troppo contrastata quando le
si schiarisce. Girata a fondo rovina tutto, perché ogni cifra gli sembra
plausibile e non si capisce più quale avesse scelto. La manopola si chiama
**temperatura**.

Dall’altra parte del banco il modello piccolo, lo studente, impara a
riconoscere le cifre. Ascolta due voci, il registro (dove qualcuno ha scritto
la risposta giusta) e i commenti del maestro, pesate sette parti al maestro e
tre al registro. La proporzione si sceglie: meno ci si fida del maestro, più
parti tornano al registro.

La temperatura però vale per tutti e due, e alzandola si dilunga anche lo
studente: le due risposte si somigliano di più, e la correzione che lo studente
riceve si fa fiacca proprio mentre gli si mostrano le sfumature. Per rimetterla
in forza la si moltiplica per il quadrato della temperatura: a quattro, per
sedici. Il quadrato però è il conto giusto solo con la temperatura molto alta;
a quattro la correzione si era indebolita di meno, e moltiplicata per sedici
esce più forte di com’era. Così lo studente ascolta il maestro più delle sette
parti su dieci che gli erano state assegnate. Funziona lo stesso, perché la
proporzione la si ritocca guardando come vanno le cose; ma chi crede di averla
messa a sette contro tre ha in mano un numero che non racconta quello che
succede in classe.

Una cifra con la sola etichetta insegna una cosa, che quel disegno è un sette.
Col commento del maestro ne insegna dieci, quanto somiglia a ciascuna delle
dieci cifre.

Il guadagno grosso però è meno elegante, e sta nella pila dei fogli che in
fondo all’aula nessuno ha mai etichettato. Il maestro li commenta uno per uno,
e allo studente quei commenti valgono quanto gli altri. Etichettare costa, i
fogli no, ed è per questo che la distillazione si usa più di quanto la si
spieghi. Il maestro però può commentare bene soltanto quello che ha imparato:
uno che ha visto le stesse poche etichette dello studente non ha niente da
aggiungere.

Se il maestro è convinto che una certa quattro malfatta sia un nove, lo
studente ne prende la convinzione, perché gliela sente ripetere con tutte le
sue sfumature, e al maestro dà più retta che al registro: sette parti contro
tre. Un maestro sbagliato fa peggio di nessun maestro.

`````

`````{tab} Superiore

Un classificatore produce dei logit $z_i$ che la softmax trasforma in
probabilità. La distillazione {cite}`hinton2015distilling` introduce una
**temperatura** $T$ nella softmax:

$$
p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}.
$$

Con $T = 1$ si ha la softmax ordinaria; per $T > 1$ la distribuzione si
appiattisce e i rapporti fra le probabilità piccole diventano numericamente
significativi; per $T \to \infty$ tende all’uniforme. Le probabilità così
ottenute dal modello grande sono i **bersagli morbidi**. La temperatura alta
serve solo in addestramento: a inferenza lo studente torna a $T = 1$.

Lo studente si addestra minimizzando una combinazione:

$$
\mathcal{L} = (1-\alpha)\,\mathcal{L}_{\text{dura}}(\mathbf{z}^s, y)
+ \alpha\,T^2 \,\mathrm{KL}\!\big(p^t(T)\,\|\,p^s(T)\big),
$$

dove $\mathcal{L}_{\text{dura}}$ è l’entropia incrociata con l’etichetta vera,
calcolata a $T = 1$ e non alla temperatura della distillazione, e il secondo
termine è la divergenza di Kullback-Leibler fra la distribuzione morbida del
maestro e quella dello studente, calcolate tutt’e due alla stessa temperatura.
La KL differisce dall’entropia incrociata con i bersagli del maestro per
l’entropia del maestro, che non dipende dallo studente: le due danno lo stesso
gradiente.

Nel codice dell’esperimento $\alpha = 0{,}7$ e $T = 4$.

Il fattore $T^2$ non è cosmetico, e la sua derivazione è più fragile di come la
si racconta. Derivando la divergenza rispetto ai logit dello studente si
ottiene $\partial \mathrm{KL}/\partial z^s_i = (p^s_i - p^t_i)/T$, cioè un solo
$1/T$. Il secondo compare linearizzando $p^s_i - p^t_i$, e la linearizzazione
chiede due cose, non una: temperatura alta rispetto ai logit, e logit a
media nulla su ciascun esempio. È il regime in cui il lavoro originale la
ricava, e lì moltiplicare per $T^2$ mantiene il termine morbido sulla scala di
quello duro, così si può cambiare $T$ senza riaggiustare $\alpha$. In quel
limite il gradiente è $(z^s_i - z^t_i)/(K T^2)$, con $K$ il numero delle
classi, cioè quello dell’errore quadratico fra i logit dello studente e quelli
del maestro: la regressione sui logit con cui Ba e Caruana addestravano i loro
modelli imitatori {cite}`ba2014deep` è un caso limite della distillazione.

Fuori da quel regime il compenso è approssimativo, e il regime buono è più
lontano di quanto sembri. Sul maestro che il codice sulle cifre addestra
(logit con scarto tipico intorno a undici), con lo studente appena creato e il
gradiente della divergenza preso rispetto ai suoi logit su tutti gli esempi,
l’esponente locale $\kappa$ di $\|\nabla\| \propto T^{-\kappa}$ vale $1{,}01$
fra $T=1$ e $T=2$ e $1{,}13$ fra $2$ e $4$, e supera $2$ soltanto fra $8$ e
$16$ (lo stampa il conto della temperatura, dopo l’esperimento). A $T=4$, cioè
alla temperatura che il codice usa, moltiplicare per $T^2$ sovracompensa di
3,6 volte. Il risultato dell’esperimento è buono lo stesso, perché $\alpha$
assorbe il resto; ma è il genere di dettaglio che distingue una ricetta
applicata da una capita.

Dove stia il guadagno è meno ovvio di come lo si racconta di solito.
L’argomento tradizionale è che i bersagli morbidi trasportino informazione
sulla struttura delle classi (la «conoscenza oscura»: quali classi il maestro
confonde e quali no) e che questa informazione agisca come un regolarizzatore,
riducendo la varianza dello studente. Il secondo argomento sta nella stessa
pagina del lavoro originale, che lo prende da Buciluă, Caruana e
Niculescu-Mizil {cite}`bucilua2006model`, e si cita molto meno: i bersagli
morbidi si possono calcolare su dati non etichettati, e questo sposta il
problema da «quanti esempi ho» a «quanti esempi il maestro può commentare».
L’esperimento sulle cifre misura il secondo, che è quello che si riesce a
mostrare in modo pulito su un dataset piccolo, e mostra anche quanto tutti e
due dipendano da quello che il maestro ha visto.

Il principio ha varianti che la stessa formula non scrive: far imitare allo
studente anche gli strati intermedi del maestro (FitNets
{cite}`romero2015fitnets`), addestrarlo sulle sequenze che il maestro genera
invece che sulle sole distribuzioni parola per parola {cite}`kim2016sequence`,
distillare già nel preaddestramento, come fa DistilBERT, che toglie a BERT il
40% dei parametri conservandone il 97% delle capacità di comprensione del
linguaggio, con il 60% di velocità in più {cite}`sanh2019distilbert`.

`````

## L’esperimento

Il maestro è una rete larga, addestrata su tutte le etichette. Lo studente è
una rete minuscola, e vede pochissime etichette: centoventi esempi su
ottocentonovantotto. La domanda è che cosa cambi se, oltre a quelle centoventi
etichette, allo studente si lasciano leggere anche i commenti del maestro
su tutti gli esempi, compresi quelli di cui non ha l’etichetta.

```python
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# un thread solo: due esecuzioni di fila danno lo stesso numero. Su un'altra
# macchina le ultime cifre ballano, perche' cambia l'ordine delle somme
torch.set_num_threads(1)

dati = load_digits()
X, Xte, y, yte = train_test_split(dati.data / 16.0, dati.target,
                                  test_size=0.5, random_state=0)
X = torch.tensor(X, dtype=torch.float32)
Xte = torch.tensor(Xte, dtype=torch.float32)
y, yte = torch.tensor(y), torch.tensor(yte)

POCHI = 120                      # le sole etichette che lo studente puo' vedere
Xpoche, ypoche = X[:POCHI], y[:POCHI]
TEMPERATURA = 4.0


def crea(taglie, seme):
    torch.manual_seed(seme)
    strati = []
    for dentro, fuori in zip(taglie, taglie[1:]):
        strati += [nn.Linear(dentro, fuori), nn.ReLU()]
    return nn.Sequential(*strati[:-1])       # l'ultima ReLU non serve


def accuratezza(modello):
    with torch.no_grad():
        return (modello(Xte).argmax(1) == yte).float().mean().item() * 100


def parametri(modello):
    return sum(p.numel() for p in modello.parameters())


def addestra_maestro(Xm, ym):
    """Il maestro: una rete larga, novecento passi sugli esempi che gli si
    danno."""
    maestro = crea([64, 512, 512, 10], seme=0)
    opt = torch.optim.Adam(maestro.parameters(), lr=1e-3)
    for _ in range(900):
        F.cross_entropy(maestro(Xm), ym).backward()
        opt.step()
        opt.zero_grad()
    return maestro


maestro = addestra_maestro(X, y)
print(f"maestro, con tutte le {len(X)} etichette: {accuratezza(maestro):.1f}%")


def morbida(uscita, bersaglio):
    """Quanto lo studente si discosta dai dubbi del maestro. Il fattore T*T
    rimette il termine morbido sulla scala di quello duro."""
    T = TEMPERATURA
    return F.kl_div(F.log_softmax(uscita / T, dim=1),
                    F.softmax(bersaglio / T, dim=1),
                    reduction="batchmean") * T * T


def studente(Xp, yp, Xm=None, logit=None, seme=1):
    """Uno studente minuscolo, addestrato sulle etichette (Xp, yp) e, se ci
    sono, sui commenti `logit` del maestro agli esempi Xm."""
    s = crea([64, 16, 10], seme)
    opt = torch.optim.Adam(s.parameters(), lr=3e-3)
    for _ in range(900):
        perdita = F.cross_entropy(s(Xp), yp)
        if logit is not None:
            perdita = 0.3 * perdita + 0.7 * morbida(s(Xm), logit)
        perdita.backward()
        opt.step()
        opt.zero_grad()
    return accuratezza(s)


def tre_condizioni(maestro, Xp, yp):
    """Tre studenti (semi 1, 2, 3) senza maestro, col maestro sui soli esempi
    etichettati, e col maestro su tutti gli esempi."""
    risultati = []
    for Xm in (None, Xp, X):
        logit = None
        if Xm is not None:
            with torch.no_grad():
                logit = maestro(Xm)
        risultati.append([studente(Xp, yp, Xm, logit, s) for s in (1, 2, 3)])
    return risultati


# tre condizioni, che servono a separare due cose che di solito si confondono:
# i dubbi del maestro, e il fatto che il maestro possa commentare esempi di cui
# lo studente non ha l'etichetta
nomi = ("niente maestro", "maestro sui soli 120", "maestro su tutti gli 898")
for etichetta, prove in zip(nomi, tre_condizioni(maestro, Xpoche, ypoche)):
    media = sum(prove) / len(prove)
    print(f"studente, {etichetta:<22} {media:.1f}%   "
          f"(tre semi: {', '.join(f'{p:.1f}' for p in prove)})")
piccolo = crea([64, 16, 10], seme=1)
print(f"lo studente ha {parametri(maestro) / parametri(piccolo):.0f} volte "
      f"meno parametri del maestro")
```

```text
maestro, con tutte le 898 etichette: 96.9%
studente, niente maestro         90.2%   (tre semi: 90.5, 90.2, 89.8)
studente, maestro sui soli 120   92.6%   (tre semi: 92.7, 92.0, 93.1)
studente, maestro su tutti gli 898 95.8%   (tre semi: 96.1, 95.4, 95.9)
lo studente ha 249 volte meno parametri del maestro
```

Cinque punti e mezzo fra lo studente senza maestro e quello col maestro su
tutti gli esempi, da 90,2 a 95,8, con tre semi che dicono la stessa cosa. Lo
studente col maestro arriva a un punto dal maestro stesso, avendo in mano
duecentoquarantanove volte meno parametri e centoventi etichette invece di
ottocentonovantotto.

La riga di mezzo è quella che serve di più, perché separa due cose che di
solito si raccontano come una sola. Con il maestro che commenta soltanto i
centoventi esempi che lo studente ha già etichettati si guadagnano 2,4 punti:
sono i dubbi, cioè sapere che un certo sette somigliava a un uno e non a un
otto, quali li ha imparati un maestro che di etichette ne ha viste
ottocentonovantotto. Lasciando al maestro commentare anche gli altri
settecentosettantotto, che lo studente non può usare perché non ne ha
l’etichetta, se ne guadagnano altri 3,2. Quindi la spiegazione bella («il
maestro dice dieci cose per esempio invece di una») è vera e vale meno della
metà del risultato. L’altra metà è più prosaica: il maestro trasforma esempi di
cui lo studente non ha l’etichetta in esempi utilizzabili.

Tutti e due i guadagni, però, dipendono da quanto il maestro ne sa più dello
studente, e lo si vede cambiando maestro: uno addestrato sulle stesse
centoventi etichette dello studente, uno sulla sola prima metà dei dati, e
infine il maestro di prima con uno studente che ha tutte le etichette.

```python
meta = len(X) // 2
casi = (("sulle sole 120 dello studente", addestra_maestro(Xpoche, ypoche),
         Xpoche, ypoche),
        (f"sui primi {meta} esempi", addestra_maestro(X[:meta], y[:meta]),
         Xpoche, ypoche),
        ("su tutti, studente con tutte", maestro, X, y))
print(f"{'maestro addestrato':<30}{'maestro':>10}{'senza':>10}"
      f"{'sui suoi':>10}{'su tutti':>10}")
for nome, m, Xp, yp in casi:
    medie = [sum(p) / len(p) for p in tre_condizioni(m, Xp, yp)]
    print(f"{nome:<30}{accuratezza(m):>9.1f}%"
          + "".join(f"{v:>9.1f}%" for v in medie))
```

```text
maestro addestrato               maestro     senza  sui suoi  su tutti
sulle sole 120 dello studente      91.0%     90.2%     90.0%     90.1%
sui primi 449 esempi               94.0%     90.2%     92.0%     93.5%
su tutti, studente con tutte       96.9%     95.7%     95.8%     95.8%
```

Con un maestro che ha visto soltanto le centoventi etichette dello studente i
commenti non portano niente, né sugli esempi etichettati né sugli altri: i
dubbi di un maestro valgono quello che il maestro sa in più. Con un maestro che
la seconda metà dei dati non l’ha mai vista i due guadagni scendono da 2,4 e
3,2 a 1,8 e 1,5, e il secondo più del primo, perché le etichette di quegli
esempi il maestro di prima le aveva viste e i suoi commenti in parte le
passavano allo studente: la cifra che il conto attribuisce ai dati non
etichettati è un tetto. Le due cose insieme fanno la distillazione, e chi ne
racconta solo la prima attribuisce a un meccanismo elegante un guadagno che
viene soprattutto da un meccanismo banale. La riga dello studente con tutte le
etichette dice che cosa il conto non dimostra, cioè che imitare sia meglio che
imparare: con tutte le
ottocentonovantotto etichette lo studente arriva al 95,7% da solo, e il maestro
non aggiunge più niente.

Resta il maestro che sbaglia. Gli si danno etichette sbagliate per tre esempi
su dieci, ciascuna sostituita da una cifra a caso fra le altre nove, e lo si
mette a commentare come prima.

```python
caso = torch.Generator().manual_seed(0)       # un sorteggio a parte
sbagliate = y.clone()
quali = torch.rand(len(y), generator=caso) < 0.3
spostamento = torch.randint(1, 10, (int(quali.sum()),), generator=caso)
sbagliate[quali] = (y[quali] + spostamento) % 10     # sempre una cifra diversa
confuso = addestra_maestro(X, sbagliate)
senza, sui_suoi, su_tutti = [sum(p) / len(p)
                             for p in tre_condizioni(confuso, Xpoche, ypoche)]
print(f"etichette sbagliate date al maestro: "
      f"{(sbagliate != y).float().mean() * 100:.1f}%")
print(f"maestro: {accuratezza(confuso):.1f}%")
print(f"studente senza maestro {senza:.1f}%, col maestro sui 120 "
      f"{sui_suoi:.1f}%, su tutti {su_tutti:.1f}%")
```

```text
etichette sbagliate date al maestro: 30.7%
maestro: 74.1%
studente senza maestro 90.2%, col maestro sui 120 63.0%, su tutti 81.2%
```

Lo studente finisce sotto lo studente senza maestro, e di molto quando il
maestro commenta proprio i suoi centoventi esempi: lì le etichette giuste
pesano tre parti su dieci e i commenti sbagliati sette, e vincono i commenti.
Un maestro sbagliato fa peggio di nessun maestro, almeno con le proporzioni di
questo codice; dando al maestro un peso più piccolo il registro conterebbe di
più.

Un ultimo conto riguarda la ricetta. Il termine del maestro, alzando la
temperatura, si indebolisce, e il codice lo rimette in forza moltiplicandolo per
$T^2$, cioè per sedici. Se la spinta che il maestro dà allo studente scendesse
esattamente come $1/T^2$, l’esponente che si legge raddoppiando la temperatura
varrebbe 2. Lo si misura sullo studente appena creato, prima di ogni
addestramento.

```python
import math


def spinta(z_studente, T):
    """Quanto tira il termine morbido: la norma del suo gradiente rispetto ai
    logit dello studente, su tutti gli esempi."""
    z = z_studente.detach().clone().requires_grad_(True)
    with torch.no_grad():
        bersaglio = F.softmax(maestro(X) / T, dim=1)
    perdita = F.kl_div(F.log_softmax(z / T, dim=1), bersaglio, reduction="sum")
    perdita.backward()
    return z.grad.norm().item()


with torch.no_grad():
    z_nuovo = crea([64, 16, 10], seme=1)(X)
    print(f"scarto tipico dei logit del maestro: {maestro(X).std():.1f}")
for a, b in ((1, 2), (2, 4), (4, 8), (8, 16)):
    calo = spinta(z_nuovo, b) / spinta(z_nuovo, a)
    kappa = -math.log(calo) / math.log(b / a)
    print(f"esponente fra T={a} e T={b}: {kappa:.2f}")
rapporto = spinta(z_nuovo, 4) / spinta(z_nuovo, 1)
print(f"spinta a T=4 rispetto a T=1: {rapporto:.2f}")
print(f"la stessa, moltiplicata per 16: {16 * rapporto:.2f}")
```

```text
scarto tipico dei logit del maestro: 10.6
esponente fra T=1 e T=2: 1.01
esponente fra T=2 e T=4: 1.13
esponente fra T=4 e T=8: 1.59
esponente fra T=8 e T=16: 2.11
spinta a T=4 rispetto a T=1: 0.23
la stessa, moltiplicata per 16: 3.64
```

Fino a $T = 4$ la spinta scende quasi come $1/T$ e non come $1/T^2$, e
l’esponente arriva a 2 solo fra 8 e 16. Alla temperatura del codice il fattore
sedici porta la spinta a 3,6 volte quella di partenza invece che alla stessa:
lo studente ascolta il maestro più delle sette parti su dieci che il codice gli
assegna.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un modello non risponde «7»: risponde con dieci numeri che dicono quanto
  ci crede. L’etichetta vera ne contiene uno; la risposta del maestro li
  contiene tutti e dieci.
- Un maestro ben addestrato è troppo sicuro di sé e i numeri interessanti si
  perdono nelle cifre lontane. Si ammorbidiscono le sue risposte, come si
  schiariscono le ombre di una fotografia troppo contrastata: il soggetto resta
  il più chiaro di tutti, e intanto nel buio ricompare quello che c’era.
- Il guadagno si spezza in due. Uno studente minuscolo con centoventi
  etichette sta al 90,2%; con i commenti del maestro sugli stessi centoventi
  esempi sale a 92,6% (+2,4: i dubbi di un maestro che ha visto più
  etichette); con i commenti anche sugli esempi di cui non ha l’etichetta
  arriva a 95,8% (+3,2 in più). La parte grossa viene dal poter usare esempi
  senza etichetta, che però il maestro aveva visto etichettati.
- Il maestro passa quello che sa in più: uno addestrato sulle stesse
  centoventi etichette dello studente non porta niente.
- Lo studente eredita anche gli errori del maestro, e con sette parti su dieci
  date al maestro gli dà più retta che alle proprie etichette: un maestro
  allenato con tre etichette su dieci sbagliate lo porta sotto lo studente
  senza maestro. Un maestro sbagliato fa peggio di nessun maestro.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- La temperatura nella softmax, $p_i(T) = e^{z_i/T} / \sum_j e^{z_j/T}$,
  appiattisce la distribuzione e rende numericamente significativi i rapporti
  fra le probabilità piccole. Sono i bersagli morbidi.
- La perdita è
  $(1-\alpha)\mathcal{L}_{\text{dura}} + \alpha T^2 \mathrm{KL}(p^t(T)\|p^s(T))$
  {cite}`hinton2015distilling`, con il termine duro a $T=1$ (e $T = 1$ anche a
  inferenza). Il fattore $T^2$ tiene i due termini sulla stessa scala solo a
  temperatura alta rispetto ai logit, dove la distillazione diventa la
  regressione sui logit {cite}`ba2014deep`: a $T=4$, con lo studente appena
  creato, sovracompensa di 3,6 volte, e a riassorbire lo scarto è $\alpha$.
- L’ablazione a tre condizioni separa i due contributi: 90,2% senza maestro,
  92,6% col maestro sui soli esempi etichettati, 95,8% col maestro su tutti
  (maestro al 96,9%, addestrato su 898 etichette). Tutti e due dipendono da
  quanto il maestro sa in più: addestrato sulle sole 120 etichette dello
  studente non porta niente, sulla metà dei dati i guadagni scendono a 1,8 e
  1,5; e con tutte le etichette allo studente spariscono. Il 3,2 dei dati non
  etichettati è un tetto, perché quelle etichette il maestro le ha viste.
- Un maestro con il 30% di etichette sbagliate porta lo studente sotto lo
  studente senza maestro (81,2% contro 90,2%, e 63,0% se commenta solo gli
  esempi etichettati): con $\alpha = 0{,}7$ il termine morbido pesa più di
  quello duro.
- La distillazione è l’unica delle tre leve in cui l’architettura finale si
  sceglie invece di ereditarla: la quantizzazione restituisce la rete che ha
  ricevuto con altri numeri dentro, la potatura la stessa rete con dei buchi, o
  con qualche riga in meno dove il taglio è strutturato.
```

`````

Della distillazione resta questo: il maestro passa allo studente più di quanto
dica l’etichetta, ma soltanto quello che sa in più, e il guadagno grosso viene
dagli esempi che può commentare anche senza etichetta. Con lei le tre leve del
capitolo finiscono, e hanno in comune che riguardano tutte il modello, quanto
spazio occupa e quanti conti chiede. Perché questo non basti a farlo
rispondere in fretta lo spiega la {doc}`sezione che chiude il capitolo
<far-rispondere-in-fretta>`.
