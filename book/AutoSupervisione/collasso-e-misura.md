# Perché non collassa, e come si fa a saperlo

Un modello addestrato in modo supervisionato si controlla in un attimo: gli si
danno degli esempi che non ha mai visto, si guarda quante volte azzecca, e quel
numero è la risposta. Un modello auto-supervisionato no. Di ogni dato produce
soltanto il proprio riassunto interno, e il suo punteggio (quanto bene ha risolto
il pretesto) non ci dice quello che vogliamo sapere, perché il pretesto lo
abbiamo inventato noi e a nessuno interessa.

Restano quindi due domande aperte, e questa sezione è fatta di quelle. La prima:
che cosa può andare storto senza che il punteggio se ne accorga. La seconda: con
che cosa si sostituisce, il punteggio.

## Il collasso non è uno solo

La sezione precedente ha chiamato **collasso** la risposta vuota, cioè
descrivere tutto allo stesso identico modo. In quella forma è facile da
riconoscere: se ne accorgerebbe chiunque guardasse due riassunti. Il guaio è che
esiste una versione più educata dello stesso guasto, che passa inosservata
proprio perché non è totale.

`````{tab} Elementare

Torniamo alla scheda con le otto caselle.

Il **collasso completo** è quello che già conosciamo: tutte le fotografie
ricevono la stessa scheda. Un disastro, ma un disastro visibile.

Il **collasso parziale** è più subdolo. Le schede sono diverse fra loro, quindi
a prima vista tutto funziona; solo che, guardandole tutte insieme, ci si accorge
che variano solo lungo due o tre direzioni. È come avere otto manopole di cui sei
sono incollate insieme: le puoi girare, la scheda cambia, ma i gradi di libertà
veri sono meno di quelli che hai pagato. Il modello non ti sta mentendo, ti sta
dando meno di quello che sembra.

Perché è insidioso: il punteggio del pretesto può essere ottimo lo stesso. Se il
gioco che gli abbiamo dato si vince con tre direzioni, tre direzioni gli bastano,
e non ha nessun motivo di produrne otto. Il difetto salta fuori dopo, quando quel
riassunto lo si vuole usare per un compito diverso da quello per cui è stato
addestrato, e le direzioni mancanti erano proprio quelle che servivano.

Contro questo guasto serve la regola della varietà, quella che chiede che le
otto caselle dicano otto cose diverse; e va chiesta nel modo giusto. Pretendere
che ogni casella si muova abbastanza da una foto all'altra non basta: due
caselle incollate insieme si muovono tutte e due, e il controllo passa. Il
guasto lo vede solo chi le guarda a due a due, e fa pagare ogni coppia che dice
la stessa cosa.

`````

`````{tab} Superiore

Si distinguono due regimi.

Il **collasso completo** è la soluzione costante,
$f_\theta(\mathbf{x}) = \mathbf{c}$ per ogni $\mathbf{x}$: la rappresentazione
non porta alcuna informazione sull'ingresso. È il minimo banale che le quattro
famiglie della sezione precedente esistono per escludere.

Il **collasso dimensionale** è più fine: le rappresentazioni non sono costanti,
ma occupano un sottospazio di dimensione $r \ll D$ dello spazio $\mathbb{R}^D$
in cui vivono {cite}`jing2022understanding`. Si diagnostica sullo **spettro della matrice di covarianza**
delle rappresentazioni: se gli autovalori decadono bruscamente e solo $r$ di
essi sono sensibilmente diversi da zero, le direzioni effettivamente usate sono
$r$. Una misura riassuntiva comoda è la **dimensione effettiva**, per esempio
$\big(\sum_i \lambda_i\big)^2 / \sum_i \lambda_i^2$ con $\lambda_i$ gli
autovalori, che vale $D$ se lo spettro è piatto e crolla se è concentrato.

Il punto pratico è che la perdita del pretesto non lo vede: un obiettivo
risolvibile in $r$ direzioni non ha alcun incentivo a usarne $D$. Ed è
esattamente la ragione per cui la famiglia della riduzione di ridondanza scrive
più di un vincolo. Uno spettro concentrato non è altro che una covarianza di
rango deficiente, cioè coordinate fortemente dipendenti fra loro, e una
condizione **coordinata per coordinata** non basta a escluderlo: il termine di
varianza di VICReg tiene la deviazione standard di ogni coordinata sopra una
soglia, e due coordinate identiche la superano tutte e due pur portando una
direzione sola. Sul regime dimensionale mordono i termini che guardano le
**coppie**, cioè la fuori diagonale della cross-correlazione di Barlow Twins e
il termine di covarianza di VICReg; il regime costante lo fermano gli altri due,
la varianza e la diagonale. Due tipi di vincolo per due collassi diversi.

`````

## Che cosa garantisce davvero il punteggio dei metodi contrastivi

C'è una lettura elegante dei metodi contrastivi che circola molto, e conviene
esaminarla perché è vera a metà, e la metà che manca è istruttiva.

La lettura è questa. Il punteggio con cui si addestrano quei metodi si chiama
**InfoNCE** {cite}`oord2018representation`, e il nome dice già dove vuole
andare: si sostiene che sia legato all’**informazione mutua** fra le due viste,
cioè a quanto, sapendo una delle due, si diventa meno incerti sull'altra.
L'informazione mutua è appunto la differenza fra l'incertezza che si aveva prima
e quella che resta dopo, e si misura con l'entropia della sezione sulla teoria
dell'informazione. Minimizzare quella perdita, si dice, equivale a massimizzare
l'informazione mutua. Se fosse tutta la storia, avremmo una spiegazione limpida
del perché quei metodi funzionano.

`````{tab} Elementare

Il legame c'è, ma ha un tetto, e il tetto dipende da una cosa che non c'entra
niente con quanto il modello ha capito: **quanti rivali ci sono nel gruppo**.

Il compito è ritrovare il gemello in mezzo a $N$ candidati. Anche riuscendoci
sempre, quanto abbiamo dimostrato di sapere? Abbiamo dimostrato di saper
scegliere fra $N$ cose, che è quanto basta per vincere **quel** gioco e non un
briciolo di più. Se i candidati sono otto, il massimo che quel gioco può
certificare è la capacità di distinguere fra otto; se il modello ne sapesse
mille volte tanto, il gioco non se ne accorgerebbe, perché ha finito le domande
da fare.

Con un modello che indovina sempre, cioè al meglio delle sue possibilità, il
numero certificato cresce col numero dei rivali e si ferma lì, sempre e solo lì.
Allargare il gruppo alza il tetto e costa: ecco perché in questi metodi i gruppi
si fanno enormi.

E se il segreto stesse davvero nell'informazione, chi quel gioco non lo gioca
affatto dovrebbe cavarsela male. Invece se la cava benissimo: far indovinare a
una rete che cosa dice l'altra, o chiedere che le otto caselle dicano otto cose
diverse, sono modi che di informazione non parlano mai, e i riassunti che ne
escono sono ottimi. Quello che i metodi riusciti hanno in comune è un'altra
coppia di cose: la stessa risposta pretesa sulle due viste della stessa foto, e
una regola che chiude la strada alla risposta vuota.

`````

`````{tab} Superiore

La relazione è un **limite inferiore**: detta $\mathcal{L}_{\text{NCE}}$ la
perdita calcolata su $N$ coppie,

$$
I(\mathbf{x}; \mathbf{y}) \;\ge\; \log N - \mathcal{L}_{\text{NCE}},
$$

come mostrano van den Oord e colleghi {cite}`oord2018representation`. La
disuguaglianza è utile ma è **saturata da $\log N$**: anche con
$\mathcal{L}_{\text{NCE}} = 0$, cioè con un critico perfetto, la quantità
certificata non supera $\log N$. Se l'informazione mutua vera è molto maggiore
(e fra due viste della stessa immagine ad alta risoluzione lo è di parecchio),
il limite non dice più niente di interessante: è vero e inservibile.

Ne seguono due conseguenze, da tenere separate. La prima è tecnica: la
dimensione del batch entra nella *garanzia*, non solo nel costo, il che spiega
perché in questi metodi $N$ conti tanto. La seconda è di
interpretazione, ed è la più importante: **la massimizzazione dell'informazione
mutua non può essere la spiegazione del successo di questi metodi**. Se lo
fosse, metodi che l'informazione mutua non la stimano affatto non dovrebbero
funzionare; e invece la sezione precedente ne ha mostrati due che funzionano
benissimo, la distillazione asimmetrica e la riduzione di ridondanza, e nessuno
dei due ha un termine che assomigli a un limite su $I$. Quello che i metodi
riusciti hanno in comune è una **invarianza imposta** più un **meccanismo che
vieta la risposta vuota**, e non una quantità informativa massimizzata.

`````

```python
import math
import torch

torch.manual_seed(0)

# Un caso in cui le due viste si corrispondono PERFETTAMENTE: l'informazione
# che l'una porta sull'altra e' tutta quella che c'e', e non e' poca.
# Domanda: quanta ne puo' certificare la InfoNCE, al meglio delle sue
# possibilita'?

def infonce_al_meglio(n, d=64):
    """Loss InfoNCE con un critico PERFETTO su n coppie, e il limite che ne segue."""
    z = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    sim = (z @ z.t()) / 0.01           # temperatura bassissima: critico ideale
    perdita = torch.nn.functional.cross_entropy(sim, torch.arange(n)).item()
    # I(x; y) >= log N - L   (van den Oord e colleghi), qui in bit
    return perdita, (math.log(n) - perdita) / math.log(2)

print(f"{'coppie nel batch':>18s} {'perdita':>10s} {'bit certificati':>18s} {'log2(N)':>10s}")
for n in (8, 64, 512, 4096):
    perdita, bit = infonce_al_meglio(n)
    print(f"{n:>18d} {perdita:>10.4f} {bit:>18.2f} {math.log2(n):>10.2f}")
```

```text
  coppie nel batch    perdita    bit certificati    log2(N)
                 8     0.0000               3.00       3.00
                64     0.0000               6.00       6.00
               512     0.0000               9.00       9.00
              4096     0.0000              12.00      12.00
```

Le due colonne di destra coincidono riga per riga, ed è tutta la dimostrazione:
con il critico perfetto la perdita è nulla e il numero certificato è
**esattamente** $\log_2 N$. Anche col batch da $4096$ di SimCLR
{cite}`chen2020simple`, che è il metodo contrastivo più noto, quello che si può
garantire sono dodici bit, cioè **poco più di quanto porti una singola
etichetta** su mille classi secondo il conto della prima sezione: dieci bit. Un
batch da quattromila immagini, che per stare in memoria vuole i centoventotto
acceleratori dichiarati in quel lavoro, certifica due bit più di una parola
scritta sotto una foto. Che quei modelli imparino molto di più è fuori
discussione; quindi non è quel limite a spiegare quello che fanno.

## Con che cosa si sostituisce il punteggio

Resta la seconda domanda: come si misura se il riassunto è buono.

Lo strumento standard il libro l'ha già costruito, nel capitolo sulla visione
artificiale, e si chiama **sondaggio lineare**: si congela l'encoder, gli si
affianca un classificatore così semplice da non poter aggiungere niente di suo,
e si guarda se passa l'esame. Se un giudice tanto sprovveduto ci riesce, il
merito è del riassunto.

Qui interessa il seguito, cioè che cosa quello strumento **non** vede, perché è
la parte che si dimentica.

`````{tab} Elementare

Il sondaggio lineare promuove solo ciò che si separa con una linea dritta. Un
riassunto potrebbe contenere tutto quello che serve, scritto però in una forma
più contorta, e quell'esame lo boccerebbe lo stesso. È un metro, non un verdetto.

Poi c'è un difetto che non riguarda lo strumento ma il modo in cui lo si usa: si
misura quasi sempre sulla stessa cosa, cioè «che oggetto c'è in questa foto». È
una domanda sola, e un riassunto può essere bravissimo a rispondere a quella e
inservibile per le altre.

Accanto a quell'esame ne stanno due che costano poco e dicono cose diverse fra
loro. Uno non studia niente: guarda a quali riassunti già visti somiglia quello
nuovo, e ripete la risposta dei vicini senza aggiustare un solo numero. L'altro
non fa domande: conta quante caselle della scheda dicono davvero qualcosa di
proprio, e se sono poche il guaio si vede lì, senza aspettare che a rivelarlo
sia un compito vero.

La prova che conta di più è un'altra, e costa di più: **cambiare
compito**. Prendere quel riassunto e usarlo come base per qualcosa che non
somiglia all'esame, per esempio per dire *dove* si trovano le cose invece che
*che cosa* sono. Lì saltano fuori i riassunti che avevano imparato a rispondere
bene a una domanda sola. Ed è la prova giusta perché è il motivo per cui
addestriamo in anticipo: non per risolvere il pretesto, ma per avere un punto di
partenza utile per compiti che, mentre ci addestravamo, non sapevamo nemmeno
quali sarebbero stati.

`````

`````{tab} Superiore

Il sondaggio lineare misura la **separabilità lineare** delle classi nello
spazio delle rappresentazioni, che è un'ipotesi su come la rappresentazione
verrà consumata, non una misura dell'informazione che contiene. Da qui tre
integrazioni consuete.

La prima è la classificazione a $k$ **vicini più prossimi**, che non addestra
alcun parametro e quindi non può nemmeno rimediare a una geometria scomoda: è
una sonda ancora più spartana, e per questo informativa in modo diverso.

La seconda sono le misure che guardano la **geometria** invece della resa: lo
spettro della covarianza e la dimensione effettiva, che diagnosticano il
collasso dimensionale prima che si manifesti come perdita di prestazioni a
valle.

La terza, e la più severa, è il **trasferimento a compiti strutturalmente
diversi**: rilevamento e segmentazione chiedono una rappresentazione che resti
informativa **zona per zona**, mentre la classificazione premia un riassunto
globale. Un encoder può eccellere sotto sonda lineare e cedere come dorsale di
un rilevatore, e questo scarto è un dato, non un contrattempo: dice che il
pretesto ha selezionato un tipo di informazione e non un altro.

Vale infine la disciplina già enunciata dal capitolo sulla visione: nessuna
singola misura chiude la questione, e un confronto fra metodi condotto su una
sola sonda e un solo dataset misura la sonda quanto i metodi.

`````

Le due domande di apertura hanno quindi la stessa forma di risposta, ed è una
risposta scomoda: non esiste un numero solo. Il collasso si vede guardando la
**geometria** di quello che il modello produce, non il suo punteggio; e la
qualità di un riassunto si vede solo mettendolo a fare **un mestiere che non è
quello per cui è stato addestrato**. Chi cerca in questo campo una metrica unica
da massimizzare sta cercando una cosa che, per come il paradigma è fatto, non
può esserci: se avessimo un punteggio che dice tutto, avremmo anche il compito
vero, e non ci sarebbe stato bisogno di inventarne uno finto.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il collasso ha due forme. Quella **completa**, tutte le schede uguali, si vede
  subito. Quella **parziale** è insidiosa: le schede sono diverse, ma variano
  solo lungo poche direzioni, come otto manopole di cui sei incollate insieme.
  Il punteggio del gioco non se ne accorge, perché se il gioco si vince con tre
  direzioni tre bastano.
- Contro la seconda forma non basta chiedere che ogni casella si muova: due
  caselle incollate insieme si muovono tutte e due e il controllo passa. Serve
  una richiesta sulle **coppie** di caselle, che faccia pagare ogni coppia che
  dice la stessa cosa.
- Sull'idea che i metodi contrastivi funzionino perché «massimizzano
  l'informazione»: il legame c'è ma ha un **tetto**, e il tetto dipende da quanti
  rivali ci sono nel gruppo, non da quanto il modello ha capito. Con un modello
  che indovina sempre, quattromila rivali certificano **dodici bit**, cioè
  appena più dei dieci che porta una singola etichetta.
- Quindi quella non è la spiegazione: metodi che di informazione non parlano
  affatto funzionano benissimo lo stesso.
- Per misurare c'è l’**esame con le mani legate**, il sondaggio lineare. Ma
  promuove solo ciò che si separa con una linea dritta, e si fa quasi sempre
  sulla stessa domanda. La prova che conta davvero è **cambiare compito**: è il
  motivo per cui addestriamo in anticipo.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- **Collasso completo**: $f_\theta(\mathbf{x}) = \mathbf{c}$. **Collasso
  dimensionale**: le rappresentazioni occupano un sottospazio di dimensione
  $r \ll D$. Il secondo si diagnostica sullo spettro della covarianza (o con la
  dimensione effettiva $(\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$) e **la
  perdita del pretesto non lo vede**, perché un obiettivo risolvibile in $r$
  direzioni non ne richiede $D$.
- Sul regime dimensionale mordono i termini che guardano le **coppie** di
  coordinate, la fuori diagonale della cross-correlazione di Barlow Twins e il
  termine di covarianza di VICReg; il regime costante lo fermano la varianza e
  la diagonale. Una condizione coordinata per coordinata non basta: due
  coordinate identiche superano tutte e due la soglia sulla deviazione standard
  e insieme portano una direzione sola.
- **InfoNCE e informazione mutua**:
  $I(\mathbf{x};\mathbf{y}) \ge \log N - \mathcal{L}_{\text{NCE}}$
  {cite}`oord2018representation`, quindi il limite è **saturato da $\log N$**:
  con critico perfetto e $N = 4096$ si certificano $12$ bit, mentre
  l'informazione vera fra due viste è molto maggiore. Dodici bit sono appena
  più dei dieci di un'etichetta su mille classi: la garanzia è debolissima. La
  dimensione del batch entra nella garanzia, non solo nel costo.
- Conseguenza interpretativa: la massimizzazione dell'informazione mutua **non
  spiega** il successo di questi metodi, visto che distillazione asimmetrica e
  riduzione di ridondanza funzionano senza stimarla. Il denominatore comune è
  invarianza imposta più divieto della soluzione degenere.
- **Valutazione**: sondaggio lineare (misura la separabilità lineare, non
  l'informazione), $k$-NN come sonda senza parametri, spettro della covarianza
  per la geometria, e **trasferimento a compiti strutturalmente diversi** come
  prova più severa, perché rilevamento e segmentazione chiedono informazione
  localizzata dove la classificazione premia un riassunto globale.
```

`````
