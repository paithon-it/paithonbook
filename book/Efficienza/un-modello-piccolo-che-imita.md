# Un modello piccolo che imita: la distillazione

Un maestro corregge un compito e scrive «7». Un altro maestro corregge lo
stesso compito e scrive: «7, ma per un soffio: quel numero è scritto in un modo
che poteva farlo scambiare per un 1, e in nessun caso lo si sarebbe potuto
prendere per un 8».

Il secondo maestro ha detto la stessa cosa del primo più qualcos’altro, e quel
qualcos’altro non riguarda il compito: riguarda **come sono fatte le cifre**.
Che 7 e 1 si somiglino, e che 7 e 8 no, è una cosa che il maestro ha imparato
in anni di compiti corretti, e che l’etichetta «7» da sola non trasmette.

Questa sezione è su come si passa quella roba lì a un modello piccolo, e su
perché sia più di quanto ci sia nei dati.

## Che cosa c’è dentro un «quasi»

Le due leve precedenti stringevano un modello già fatto. Qui si fa un’altra
cosa: si costruisce un modello nuovo, piccolo fin dall’inizio, e lo si addestra
non sulle risposte giuste ma su **quello che il modello grande risponderebbe**,
dubbi compresi.

`````{tab} Elementare

Un modello che riconosce cifre non restituisce «7». Restituisce dieci numeri,
uno per cifra, che dicono quanto ci crede: qualcosa come «sono quasi certo che
sia un 7; se proprio dovessi sbagliarmi direi 1; che sia un 8 è
inconcepibile».

Il guaio è che un modello grande e ben addestrato è **troppo sicuro di sé**:
alla fine dice 0,9999 sul 7 e praticamente zero su tutto il resto. Le
informazioni interessanti (che l’1 fosse il secondo candidato e l’8 no)
ci sono ancora, ma sono nascoste in cifre così piccole che nessuno le sente.

Il rimedio è ammorbidire, e l’immagine giusta è una fotografia troppo
contrastata: il soggetto è bianco accecante e tutto il resto è nero pesto, e
nel nero i dettagli ci sono ma non si vedono. Schiarendo le ombre il soggetto
resta il più chiaro di tutti, e intanto nel buio ricompare quello che c’era.
Qui è lo stesso: da «0,9999 e nove zeri» si passa a qualcosa come «0,7 sul 7,
0,15 sull’1, 0,001 sull’8» (sono i numeri che si ottengono schiarendo di sei
volte: quanto si schiarisce è una manopola, e fra poco ha un nome). Il 7 resta
il primo, e adesso si vede anche la forma del dubbio.

Questo ammorbidimento ha una manopola, e nel codice qui sotto si chiama
**temperatura**: più la si alza, più la lista si appiattisce. A uno la
fotografia è quella di partenza; sopra, le ombre si schiariscono; portandola
troppo in alto si perde anche il soggetto, perché tutte le cifre finiscono per
sembrare ugualmente probabili.

E qui c’è la prima delle due cose che vale la pena portarsi via. Lo studente
che impara dalle risposte giuste impara **una cosa per esempio**: che quel
disegno lì è un 7. Lo studente che impara dai dubbi del maestro impara **dieci
cose per esempio**: quanto quel disegno assomiglia a ciascuna delle dieci
cifre. Non è che il maestro spieghi meglio, è che dice molte più cose ogni
volta che apre bocca.

La seconda è meno elegante e conta di più, e la misura più sotto la separa
dalla prima. Il maestro può parlare **anche degli esempi di cui nessuno ha
scritto la risposta**: gli si mettono davanti e lui li commenta, e quei
commenti valgono per lo studente esattamente come gli altri. Le etichette
costano, i dati grezzi no, e questo è il motivo per cui la distillazione si usa
più di quanto la si spieghi.

Il punto di rottura è netto e va detto: **lo studente eredita anche gli errori
del maestro**. Se il maestro è convinto che una certa 4 malfatta sia un 9, lo
studente impara quella convinzione, e la impara meglio di quanto imparerebbe la
risposta giusta, perché gliela sente ripetere con tutte le sue sfumature. Un
maestro sbagliato è peggio di nessun maestro.

`````

`````{tab} Superiore

Un classificatore produce dei **logit** $z_i$ che la softmax trasforma in
probabilità. La distillazione {cite}`hinton2015distilling` introduce una
**temperatura** $T$ nella softmax:

$$
p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}.
$$

Con $T = 1$ si ha la softmax ordinaria; per $T > 1$ la distribuzione si
appiattisce e i rapporti fra le probabilità piccole diventano numericamente
significativi; per $T \to \infty$ tende all’uniforme. Le probabilità così
ottenute dal modello grande sono i **bersagli morbidi**.

Lo studente si addestra minimizzando una combinazione:

$$
\mathcal{L} = (1-\alpha)\,\mathcal{L}_{\text{dura}}(\mathbf{z}^s, y)
+ \alpha\,T^2 \,\mathrm{KL}\!\big(p^t(T)\,\|\,p^s(T)\big),
$$

dove $\mathcal{L}_{\text{dura}}$ è l’entropia incrociata con l’etichetta vera e
il secondo termine è la divergenza di Kullback-Leibler fra la distribuzione
morbida del maestro e quella dello studente, calcolate alla stessa temperatura.

Nel codice qui sotto $\alpha = 0{,}7$ e $T = 4$.

Il fattore $T^2$ non è cosmetico, e vale la pena vedere da dove viene perché è
più fragile di come lo si racconta. Derivando la divergenza rispetto ai logit
dello studente si ottiene $\partial C/\partial z_i = (q_i - p_i)/T$, cioè **un
solo** $1/T$. Il secondo compare linearizzando $q_i - p_i$, e quella
linearizzazione vale a **temperatura alta rispetto ai logit**: è il regime in
cui il lavoro originale la ricava, e in quel regime moltiplicare per $T^2$
mantiene il termine morbido sulla scala di quello duro, così si può cambiare
$T$ senza riaggiustare $\alpha$.

Fuori da quel regime il compenso è approssimativo, e conviene saperlo perché il
regime buono è più lontano di quanto sembri. Misurato sul maestro che il codice
qui sotto addestra (logit con scarto tipico intorno a undici), l’esponente
locale di $\|\nabla\| \propto T^{-\alpha}$ vale $1{,}05$ fra $T=1$ e $T=2$,
$1{,}09$ fra $2$ e $4$, e arriva a $2$ soltanto oltre $T=16$. A $T=4$, cioè
alla temperatura che il codice usa, moltiplicare per $T^2$ **sovracompensa di
circa tre volte e mezzo**. Non è un guasto (il risultato dell’esperimento è
buono lo stesso, e $\alpha$ assorbe il resto), è il genere di dettaglio che
distingue una ricetta applicata da una capita.

Vale la pena essere precisi su **dove stia il guadagno**, perché è il punto in
cui la spiegazione divulgativa si allontana dalla letteratura. L’argomento
tradizionale è che i bersagli morbidi trasportino informazione sulla struttura
delle classi (la «conoscenza oscura»: quali classi il maestro confonde e quali
no) e che questa informazione agisca come un regolarizzatore, riducendo la
varianza dello studente. Un argomento complementare, altrettanto valido, è che
i bersagli morbidi si possano calcolare su **dati non etichettati**, e questo
sposta il problema da «quanti esempi ho» a «quanti esempi il maestro può
commentare». L’esperimento qui sotto misura il secondo, che è quello che si
riesce a mostrare in modo pulito su un dataset piccolo.

`````

## L’esperimento

Il maestro è una rete larga, addestrata su tutte le etichette. Lo studente è
una rete minuscola, e vede pochissime etichette: centoventi esempi su
ottocentonovantotto. La domanda è che cosa cambi se, oltre a quelle centoventi
etichette, allo studente si lasciano leggere anche i **commenti del maestro**
su tutti gli esempi, **compresi quelli di cui non ha l’etichetta**.

```python
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# un thread solo: cosi' i numeri stampati sono gli stessi su ogni macchina
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


maestro = crea([64, 512, 512, 10], seme=0)
opt = torch.optim.Adam(maestro.parameters(), lr=1e-3)
for _ in range(900):
    F.cross_entropy(maestro(X), y).backward()
    opt.step()
    opt.zero_grad()
print(f"maestro, con tutte le {len(X)} etichette: {accuratezza(maestro):.1f}%")
with torch.no_grad():
    logit_maestro = maestro(X)

with torch.no_grad():
    logit_su_pochi = maestro(Xpoche)


def morbida(uscita, bersaglio):
    """Quanto lo studente si discosta dai dubbi del maestro. Il fattore T*T
    rimette il termine morbido sulla scala di quello duro."""
    T = TEMPERATURA
    return F.kl_div(F.log_softmax(uscita / T, dim=1),
                    F.softmax(bersaglio / T, dim=1),
                    reduction="batchmean") * T * T


# tre condizioni, che servono a separare due cose che di solito si confondono:
# i dubbi del maestro, e il fatto che il maestro possa commentare esempi di cui
# lo studente non ha l'etichetta
for etichetta, maestro_su in (("niente maestro", None),
                              ("maestro sui soli 120", "pochi"),
                              ("maestro su tutti gli 898", "tutti")):
    prove = []
    for seme in (1, 2, 3):
        studente = crea([64, 16, 10], seme)
        opt = torch.optim.Adam(studente.parameters(), lr=3e-3)
        for _ in range(900):
            perdita = F.cross_entropy(studente(Xpoche), ypoche)
            if maestro_su == "pochi":
                perdita = 0.3 * perdita + 0.7 * morbida(studente(Xpoche),
                                                        logit_su_pochi)
            elif maestro_su == "tutti":
                perdita = 0.3 * perdita + 0.7 * morbida(studente(X),
                                                        logit_maestro)
            perdita.backward()
            opt.step()
            opt.zero_grad()
        prove.append(accuratezza(studente))
    media = sum(prove) / len(prove)
    print(f"studente, {etichetta:<22} {media:.1f}%   "
          f"(tre semi: {', '.join(f'{p:.1f}' for p in prove)})")
print(f"lo studente ha {parametri(maestro) / parametri(studente):.0f} volte "
      f"meno parametri del maestro")
```

```text
maestro, con tutte le 898 etichette: 96.9%
studente, niente maestro         90.2%   (tre semi: 90.5, 90.2, 89.8)
studente, maestro sui soli 120   92.6%   (tre semi: 92.7, 92.0, 93.1)
studente, maestro su tutti gli 898 95.8%   (tre semi: 96.1, 95.4, 95.9)
lo studente ha 249 volte meno parametri del maestro
```

Cinque punti e mezzo fra la prima riga e la terza, con tre semi che dicono la
stessa cosa. Lo studente col maestro arriva a un punto dal maestro stesso,
avendo in mano duecentoquarantanove volte meno parametri e centoventi etichette
invece di ottocentonovantotto.

La riga di mezzo è quella che vale la pena aver misurato, perché **separa due
cose che di solito si raccontano come una sola**. Con il maestro che commenta
soltanto i centoventi esempi che lo studente ha già etichettati, si guadagnano
**2,4 punti**: quello è il valore puro dei dubbi, cioè di sapere che un certo
sette somigliava a un uno e non a un otto. Lasciando al maestro commentare
anche gli altri settecentosettantotto, che lo studente non può usare perché non
ne ha l’etichetta, se ne guadagnano altri **3,2**.

Quindi la spiegazione bella («il maestro dice dieci cose per esempio invece di
una») è vera e vale meno della metà del risultato. L’altra metà, la maggiore,
è più prosaica: il maestro trasforma dati senza etichetta in dati utilizzabili.
Le due cose insieme fanno la distillazione, e chi ne racconta solo la prima
attribuisce a un meccanismo elegante un guadagno che viene soprattutto da un
meccanismo banale.

E vale la pena dire anche che cosa il conto **non** dimostra: non dimostra che
imitare sia meglio che imparare. Se allo studente si dessero tutte e
ottocentonovantotto le etichette vere, il vantaggio si assottiglierebbe fino a
sparire nel rumore. Chi vuole vederlo cambia un solo numero nel codice qui
sopra, `POCHI`, portandolo da 120 a `len(X)`.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un modello non risponde «7»: risponde con **dieci numeri** che dicono quanto
  ci crede. L’etichetta vera ne contiene uno; la risposta del maestro li
  contiene tutti e dieci.
- Un maestro ben addestrato è troppo sicuro di sé e i numeri interessanti si
  perdono nelle cifre lontane. Si **ammorbidiscono** le sue risposte, come si
  schiariscono le ombre di una fotografia troppo contrastata: il soggetto resta
  il più chiaro di tutti, e intanto nel buio ricompare quello che c’era.
- Misurato, e in tre condizioni perché il guadagno si spezza in due: uno
  studente minuscolo con centoventi etichette sta al 90,2%; con i commenti del
  maestro **sugli stessi centoventi esempi** sale a 92,6% (sono i dubbi, +2,4);
  con i commenti anche sugli esempi di cui non ha l’etichetta arriva a 95,8%
  (+3,2 in più). La parte grossa non viene dai dubbi, viene dal poter usare
  dati che nessuno ha etichettato.
- Lo studente eredita anche gli **errori** del maestro, e li impara meglio di
  quanto imparerebbe la risposta giusta. Un maestro sbagliato è peggio di
  nessun maestro.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- La **temperatura** nella softmax, $p_i(T) = e^{z_i/T} / \sum_j e^{z_j/T}$,
  appiattisce la distribuzione e rende numericamente significativi i rapporti
  fra le probabilità piccole. Sono i **bersagli morbidi**.
- La perdita è
  $(1-\alpha)\mathcal{L}_{\text{dura}} + \alpha T^2 \mathrm{KL}(p^t(T)\|p^s(T))$
  {cite}`hinton2015distilling`. Il fattore $T^2$ compensa lo scalamento $1/T^2$
  dei gradienti del termine morbido, e ometterlo fa concludere che il metodo
  non funzioni.
- L’ablazione a tre condizioni separa i due contributi: 90,2% senza maestro,
  92,6% col maestro sui soli esempi etichettati, 95,8% col maestro su tutti
  (maestro al 96,9%). La «conoscenza oscura» vale 2,4 punti, l’uso dei dati non
  etichettati 3,2, e il secondo sparirebbe se lo studente avesse già tutte le
  etichette.
- La distillazione è l’unica delle tre leve in cui il modello finale ha
  un’**architettura diversa** da quella di partenza, e l’unica che richiede un
  addestramento vero e proprio invece di una trasformazione dei pesi.
```

`````

Le tre leve del capitolo finiscono qui, e hanno una cosa in comune che vale la
pena dire adesso: agiscono tutte e tre **sul modello**. Ma un modello che ci
sta in memoria non è ancora un modello che risponde in fretta, e la parte che
segue spiega perché siano due domande diverse, e a quali capitoli il libro
affidi la seconda.
