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

Un maestro che corregge da trent’anni, se glielo si chiede, dice quanto
scommetterebbe su ciascuna delle dieci cifre, quasi tutto sul sette, un pochino
sull’uno, niente sull’otto. Il guaio è che glielo si deve chiedere. Lasciato
fare taglia corto, «sette», con una sicurezza da 0,9999, e i ripensamenti
restano un borbottio che nessuno sente.

Allora gli si mette davanti una manopola, e più la si alza più lui si dilunga.
A uno parla come sempre. A quattro, quel «0,9999» diventa «0,9 sul
sette, 0,1 sull’uno, 0,001 sull’otto», il sette resta il primo e affiora la
forma del dubbio, come le ombre di una fotografia troppo contrastata quando le
si schiarisce. Girata a fondo rovina tutto, perché ogni cifra gli sembra
plausibile e non si capisce più quale avesse scelto. La manopola si chiama
**temperatura**.

Dall’altra parte del banco il modello piccolo, lo studente, impara a
correggere. Ascolta due voci, il registro (dove qualcuno ha scritto la risposta
giusta) e i commenti del maestro, pesate sette parti al maestro e tre al
registro. Anche questa è una manopola, perché meno ci si fida del maestro più
peso torna al registro.

La manopola però è una sola, e alzandola si dilunga anche lo studente: le due
risposte si somigliano di più, e la correzione che ne viene si fa fiacca.
Proprio mentre gli si mostrano le sfumature, gli si abbassa la voce. Per
rialzarla si moltiplica la correzione per il quadrato della manopola: a
quattro, per sedici. Sarebbe il conto esatto solo con la manopola girata a
fondo; alle posizioni vere rialza più di quanto si fosse abbassato, e lo
studente ascolta il maestro oltre le sette parti su dieci assegnate. Regge,
perché la proporzione la si ritocca guardando come vanno le cose; ma chi crede
di averla messa a sette contro tre ha in mano un numero che non racconta quello
che succede in classe.

Un compito che torna col solo voto insegna una cosa, che quel disegno è un
sette. Col commento del maestro ne insegna dieci, quanto somiglia a ciascuna
delle dieci cifre.

Il guadagno grosso però è meno elegante, e sta nella pila dei compiti che in
fondo all’aula nessuno ha mai corretto. Il maestro li commenta uno per uno, e
allo studente quei commenti valgono quanto gli altri. Correggere costa, i fogli
no, ed è per questo che la distillazione si usa più di quanto la si spieghi.

Se il maestro è convinto che una certa quattro malfatta sia un nove, lo
studente impara la convinzione, e la impara meglio di quanto imparerebbe la
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

dove $\mathcal{L}_{\text{dura}}$ è l’entropia incrociata con l’etichetta vera,
calcolata a $T = 1$ e non alla temperatura della distillazione, e il secondo
termine è la divergenza di Kullback-Leibler fra la distribuzione morbida del
maestro e quella dello studente, calcolate tutt’e due alla stessa temperatura.

Nel codice qui sotto $\alpha = 0{,}7$ e $T = 4$.

Il fattore $T^2$ non è cosmetico, e la sua derivazione è più fragile di come la
si racconta. Derivando la divergenza rispetto ai logit dello studente si
ottiene $\partial \mathrm{KL}/\partial z^s_i = (p^s_i - p^t_i)/T$, cioè un solo
$1/T$. Il secondo compare linearizzando $p^s_i - p^t_i$, e la linearizzazione
chiede due cose, non una: **temperatura alta rispetto ai logit**, e logit **a
media nulla** su ciascun esempio. È il regime in cui il lavoro originale la
ricava, e lì moltiplicare per $T^2$ mantiene il termine morbido sulla scala di
quello duro, così si può cambiare $T$ senza riaggiustare $\alpha$.

Fuori da quel regime il compenso è approssimativo, e conviene saperlo perché il
regime buono è più lontano di quanto sembri. Misurato sul maestro che il codice
qui sotto addestra (logit con scarto tipico intorno a undici), l’esponente
locale $\kappa$ di $\|\nabla\| \propto T^{-\kappa}$ vale $1{,}05$ fra $T=1$ e
$T=2$, $1{,}09$ fra $2$ e $4$, e arriva a $2$ soltanto oltre $T=16$. A $T=4$,
cioè alla temperatura che il codice usa, moltiplicare per $T^2$
**sovracompensa di circa tre volte e mezzo**. Non è un guasto (il risultato
dell’esperimento è buono lo stesso, e $\alpha$ assorbe il resto), è il genere
di dettaglio che distingue una ricetta applicata da una capita.

**Dove stia il guadagno** è il punto in cui il racconto corrente si allontana
dal lavoro che cita. L’argomento tradizionale è che i bersagli morbidi
trasportino informazione sulla struttura delle classi (la «conoscenza oscura»:
quali classi il maestro confonde e quali no) e che questa informazione agisca
come un regolarizzatore, riducendo la varianza dello studente. Il secondo
argomento sta nella stessa pagina del lavoro originale e si cita molto meno: i
bersagli morbidi si possono calcolare su **dati non etichettati**, e questo
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

La riga di mezzo è quella che conviene aver misurato, perché **separa due cose
che di solito si raccontano come una sola**. Con il maestro che commenta
soltanto i centoventi esempi che lo studente ha già etichettati, si guadagnano
**2,4 punti**: quello è il valore puro dei dubbi, cioè di sapere che un certo
sette somigliava a un uno e non a un otto. Lasciando al maestro commentare
anche gli altri settecentosettantotto, che lo studente non può usare perché
non ne ha l’etichetta, se ne guadagnano altri **3,2**.

Quindi la spiegazione bella («il maestro dice dieci cose per esempio invece di
una») è vera e vale meno della metà del risultato. L’altra metà, la maggiore,
è più prosaica: il maestro trasforma dati senza etichetta in dati utilizzabili.
Le due cose insieme fanno la distillazione, e chi ne racconta solo la prima
attribuisce a un meccanismo elegante un guadagno che viene soprattutto da un
meccanismo banale.

E conviene dire anche che cosa il conto **non** dimostra: non dimostra che
imitare sia meglio che imparare. Se allo studente si dessero tutte e
ottocentonovantotto le etichette vere, il vantaggio si assottiglierebbe fino a
sparire nel rumore. Chi vuole vederlo cambia un solo numero nel codice qui
sopra, `POCHI`, portandolo da 120 a `len(X)`: le ultime due righe diventano
allora lo stesso esperimento, e le loro etichette vanno lette così.

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
  {cite}`hinton2015distilling`, con il termine duro a $T=1$. Il fattore $T^2$
  tiene i due termini sulla stessa scala **solo a temperatura alta rispetto ai
  logit**: a $T=4$ sovracompensa di circa tre volte e mezzo, e a riassorbire lo
  scarto è $\alpha$.
- L’ablazione a tre condizioni separa i due contributi: 90,2% senza maestro,
  92,6% col maestro sui soli esempi etichettati, 95,8% col maestro su tutti
  (maestro al 96,9%). La «conoscenza oscura» vale 2,4 punti, l’uso dei dati non
  etichettati 3,2, e il secondo sparirebbe se lo studente avesse già tutte le
  etichette.
- La distillazione è l’unica delle tre leve in cui il modello finale ha
  un’**architettura diversa** da quella di partenza: le altre due restituiscono
  la rete che avevano ricevuto, con altri numeri dentro o con dei buchi.
```

`````

Le tre leve del capitolo finiscono qui, e hanno una cosa in comune da dire
adesso: agiscono tutte e tre **sul modello**. Ma un modello che ci sta in
memoria non è ancora un modello che risponde in fretta, e la parte che segue
spiega perché siano due domande diverse, e a quali capitoli il libro affidi la
seconda.
