# Comprimere e ricostruire, e perché non basta

Un museo ha in magazzino più quadri di quanti ne stiano alle pareti, e un
archivista che li deve schedare. Per ogni quadro scrive una scheda molto più
piccola dell’originale, e la mette in un cassetto. Se le schede servono a
qualcosa lo si scopre chiamando un copista, dandogli una scheda e nient’altro,
e guardando se il quadro che ridipinge somiglia a quello di partenza. Se
somiglia, la scheda conteneva l’essenziale.

I due imparano insieme, ed è il punto: una scheda non è buona in assoluto, è
buona **rispetto a chi la deve leggere**. Se il copista sa già dipingere una
cornice dorata, l’archivista non ha bisogno di annotarla; se non lo sa, quella
riga sulla scheda va spesa. Nessuno dei due ha ricevuto istruzioni su che cosa
sia importante in un quadro: se lo sono divisi lavorando, e questo capitolo
li tiene per mano fino in fondo.

Questa macchina il libro l’ha già montata, nel capitolo sull’audio, per
comprimere il suono: si chiama **autoencoder**, e ha la forma di una
**clessidra**, larga alle due estremità e strettissima in mezzo
({numref}`fig-autoencoder-clessidra`, che sta là). «Clessidra» è il nome che
useremo da qui in avanti. I nomi tecnici delle sue due metà sono quelli inglesi
che si trovano nel codice: l’archivista è l’**encoder**, «chi codifica», e il
copista è il **decoder**, «chi decodifica»; la scheda si chiama **codice**, o
**latente**. Useremo le due serie di parole come sinonimi, perché sono la
stessa cosa detta in due modi.

Un’ultima cosa prima di cominciare, e serve per tutto il resto del capitolo.
Una scheda è una fila di numeri, e una fila di numeri si può sempre immaginare
come un **punto**: due numeri sono un punto su un foglio, tre un punto in una
stanza, otto un punto in un posto che non si disegna ma che si tratta allo
stesso modo. Quindi le schede dell’archivio non stanno in un cassetto disordinato:
stanno su una mappa, alcune vicine e altre lontane, e da qui in avanti diremo
tranquillamente «due schede vicine», «una scheda a metà strada fra due» e
«camminare da una scheda all’altra».

Qui la riprendiamo per una ragione diversa da quella dell’audio, e la domanda è
una sola: se un archivista sa riassumere tutti i quadri del museo, sa anche
inventarne uno nuovo?

La risposta è no, ed è un no interessante, perché non dipende da quanto è bravo
l’archivista.

## La strozzatura è il compito

Conviene rimettere a fuoco che cosa fa la clessidra, perché quello che qui
serve non è la compressione, è la **scheda**. E la parte stretta in mezzo alla
clessidra, quella da cui deve passare tutto, si chiama **strozzatura**: qui
sotto la chiamiamo anche «il collo stretto», ed è la stessa cosa.

`````{tab} Elementare

Due metà e un collo stretto in mezzo. La prima metà, l’archivista, prende il
quadro e lo riduce a una fila di pochi numeri; la seconda, il copista, da quei
numeri prova a ritirare fuori il quadro. La pagella è una sola per tutti e due:
quanto la copia somiglia all’originale.

Il collo stretto non è un limite tecnico da subire: **è la richiesta**. Se
all’archivista fosse concesso scrivere una scheda lunga quanto il quadro, la
scriverebbe uguale al quadro, il copista la ricopierebbe, e i due avrebbero
imparato a fotocopiare. È dovendo stare in poche righe che l’archivista è
costretto a decidere che cosa conta e che cosa no, e quella decisione è tutto
ciò che ci interessa.

Sulla scheda finiscono allora le cose che il copista non saprebbe indovinare da
sé (che soggetto è, com’è composto, quali colori dominano) e non finiscono
quelle che sa già (la grana della tela, il modo in cui uno sfondo sfuma):
spenderci una riga non farebbe guadagnare niente, perché il copista le
rimetterebbe comunque. Non che se le ricordi: **ne mette una qualunque**,
sempre la stessa per tutti i quadri che gli somigliano, e a nessuno importa che
sia proprio quella. Vale la pena tenerlo a mente, perché più avanti torna: il
copista non inventa niente, ripete.

`````

`````{tab} Superiore

Un autoencoder è una coppia di funzioni parametriche,
$e_\phi: \mathbb{R}^D \to \mathbb{R}^L$ e $d_\theta: \mathbb{R}^L \to
\mathbb{R}^D$, con $L \ll D$, addestrate insieme a minimizzare l’errore di
ricostruzione

$$
\mathcal{L}(\phi, \theta) = \frac{1}{N} \sum_{i=1}^{N}
\ell\big(\mathbf{x}_i,\; d_\theta(e_\phi(\mathbf{x}_i))\big),
$$

dove $\mathbf{x}_i$ è l’$i$-esimo esempio, $D$ la dimensione del dato, $L$
quella del **codice** $\mathbf{z}_i = e_\phi(\mathbf{x}_i)$, $N$ il numero di
esempi e $\ell$ una misura di scarto fra dato e ricostruzione, **sommata sulle
$D$ componenti** (errore quadratico, oppure cross-entropia per componente come
nel blocco più avanti: è la somma sui pixel a fare del risultato un costo «per
cifra» e non «per pixel»). Il vincolo $L \ll D$ è la strozzatura, e senza di essa il problema è vuoto: con $L \ge D$
basta prendere $d_\theta$ e $e_\phi$ inverse l’una dell’altra (l’identità, per
dire) e la loss va a zero senza che nessuno abbia imparato niente.

Due osservazioni che tornano utili subito. La prima: in questa scrittura non
compare **nessuna distribuzione**. Non c’è un $p(\mathbf{x})$, non c’è un
$p(\mathbf{z})$, non c’è niente da cui campionare; c’è una funzione che
comprime, una che decomprime e uno scarto da minimizzare. La seconda: la loss
vincola i codici solo **uno per uno**, tramite la propria ricostruzione, e non
dice nulla su come i codici stiano fra loro. Nessuna delle due è una
dimenticanza da correggere in un secondo momento: sono la definizione, e da lì
discende tutto il resto della sezione.

`````

## La clessidra è la PCA, quando è dritta

C’è un fatto da mettere qui, perché lega questa macchina a una che il libro ha
già. PCA sta per *principal component analysis*, cioè l’analisi delle
componenti principali della sezione su riduzione e clustering: quella che
cerca le poche direzioni lungo cui i dati differiscono di più e butta via il
resto. E la macchina di Spearman con cui si apre il capitolo, l’analisi
fattoriale, è sua parente stretta.

`````{tab} Elementare

Mettiamo che all’archivista sia vietato essere creativo: le sue schede devono
essere una combinazione fissa dei pixel, «tanto di questo più tanto di quello»,
e basta, senza nessuna decisione presa caso per caso. In queste condizioni non
gli resta niente da inventare, e il meglio che può fare è già noto: schiacciare
i quadri sul **piano** lungo cui differiscono di più, che è la cosa che nella
sezione su riduzione e clustering si chiamava analisi delle componenti
principali. Quale coppia di direzioni scelga dentro quel piano non è deciso:
conta il piano, non gli assi che ci disegna sopra.

Detto altrimenti: la clessidra non è una macchina nuova, è la vecchia a cui è
stato tolto il divieto di piegarsi. La differenza fra le due è tutta lì, e
spiega quando conviene l’una e quando l’altra: se i dati stanno davvero su un
piano, piegarsi non serve; se stanno su una superficie curva, un piano la può
solo approssimare.

`````

`````{tab} Superiore

Con $e_\phi$ e $d_\theta$ **affini** e $\ell$ l’errore quadratico, il minimo
della loss si raggiunge quando la **ricostruzione**
$d_\theta(e_\phi(\mathbf{x}))$ è la proiezione ortogonale di $\mathbf{x}$ sul
sottospazio generato dalle prime $L$ componenti principali dei dati
**centrati** {cite}`bourlard1988auto`, e il codice ne è un sistema di
coordinate. La centratura non è un dettaglio: con mappe puramente lineari e
dati non centrati il minimo è il sottospazio dei primi $L$ vettori singolari
della matrice grezza, che passa per l'origine e in generale non coincide con
quello della PCA. A farsene carico è il termine additivo, ed è la ragione per
cui le `nn.Linear` del blocco più avanti ce l'hanno. Con una
precisazione che conta: la soluzione è unica solo **a meno di un cambio di
base** nel latente, cioè l’autoencoder lineare recupera il *sottospazio* di
massima varianza, non le singole direzioni principali né il loro ordinamento;
per ritrovare quelle serve un vincolo in più, che la PCA impone e l’autoencoder
no.

È lo stesso modello lineare-gaussiano dell’apertura del capitolo, nella
versione a rumore isotropo: la PCA è la soluzione a massima verosimiglianza
della PCA probabilistica nel limite di rumore infinitesimo, mentre l’analisi
fattoriale di Spearman è la stessa famiglia con una varianza di rumore per
ciascuna componente osservata, e lì una soluzione in forma chiusa non c’è.

Attenzione poi a **dove** vanno messe le non linearità. Bourlard e Kamp
dimostrano la metà negativa della faccenda, ed è quella che sorprende: in una
rete a tre strati con uscita lineare, mettere una non linearità nello strato
nascosto non serve a niente, il minimo resta quello lineare. Perché la
superficie su cui giacciono i dati possa essere curva serve un passaggio non
lineare **da tutte e due le parti**, uno nell’archivista e uno nel copista, che
è la `Clessidra` del blocco più avanti.
Tutto il resto (la strozzatura, la loss, l’assenza di probabilità) è identico.

`````

Da qui in avanti la clessidra è curva, cioè ciascuna delle due metà ha dentro
di sé un passaggio che non è una semplice somma pesata: è la **non linearità**
del capitolo sulle reti neurali, ed è ciò che permette a una rete di piegarsi
invece di limitarsi a piani. Adesso la guardiamo lavorare su dati veri.

## Trenta righe, e funziona

Le cifre scritte a mano di `scikit-learn` sono immagini di 8 pixel per lato,
cioè 64 numeri, e sono 1797. Le comprimiamo in **otto** numeri, che è un ottavo
del dato, e chiediamo alla rete di rifarle.

Il voto che le diamo è la **cross-entropia**, il metro introdotto nel
{doc}`capitolo di matematica </Matematica/overview>`, presa un pixel alla
volta: invece di contare i grigi di differenza fra originale e
copia, misura quanto il copista si è sbilanciato su quel pixel e quanto ci ha
azzeccato. È la scelta consueta su immagini a un canale come queste, dove ogni
pixel è un grigio fra bianco e nero, e in cambio dà un numero che si legge in
unità di informazione.

Nel blocco c'è anche una riga che con gli autoencoder non c'entra niente,
`torch.set_num_threads(1)`: chiede a PyTorch di fare i conti su un nucleo
solo, e serve perché gli stessi numeri escano su qualunque macchina.[^thread]

```python
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits

torch.manual_seed(0)
# un thread solo: cosi' i numeri stampati qui sotto sono gli stessi su
# qualunque macchina, e su dati piccoli come questi e' anche piu' veloce
torch.set_num_threads(1)

# 1797 cifre scritte a mano, 8x8 pixel, riportate fra 0 (chiaro) e 1 (scuro)
X = torch.tensor(load_digits().data / 16.0, dtype=torch.float32)


class Clessidra(nn.Module):
    """Encoder e decoder, con la strozzatura in mezzo."""

    def __init__(self, latente=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(64, 48), nn.ReLU(),
                                     nn.Linear(48, latente))
        self.decoder = nn.Sequential(nn.Linear(latente, 48), nn.ReLU(),
                                     nn.Linear(48, 64))


rete = Clessidra()
opt = torch.optim.Adam(rete.parameters(), lr=3e-3)
for passo in range(4000):
    # il decoder esce in logit; la cross-entropia li confronta col grigio vero
    perdita = F.binary_cross_entropy_with_logits(
        rete.decoder(rete.encoder(X)), X, reduction="sum") / len(X)
    opt.zero_grad()
    perdita.backward()
    opt.step()

print(f"errore di ricostruzione:   {perdita.item():.1f} nat per cifra")

# il metro di paragone: chi non guarda la cifra e dichiara, per ogni pixel,
# il grigio medio che quel pixel ha su tutte le 1797 cifre
marginale = F.binary_cross_entropy(
    X.mean(0).expand_as(X), X, reduction="sum") / len(X)
print(f"chi non guarda la cifra:  {marginale.item():.1f} nat per cifra")
```

```text
errore di ricostruzione:   16.3 nat per cifra
chi non guarda la cifra:  27.1 nat per cifra
```

L’errore si misura in **nat**, l’unità di informazione dei richiami di
matematica: sono i nat che si sprecano in media su una cifra scommettendo male
invece di conoscerla già. (Con una riserva, che la terza sezione dichiara per
esteso: su grigi che non sono zeri e uni questo conto è un surrogato, e i suoi
nat vanno letti come un metro di confronto fra due macchine, non come una
misura assoluta.) Da solo il 16,3 non direbbe niente, e per questo c’è la
seconda riga. Il confronto giusto non è con chi tira a caso, che è un
bersaglio troppo facile, ma con chi ha guardato bene **tutte** le cifre e non
guarda quella che deve rifare: per ogni pixel dichiara il grigio che quel pixel
ha in media, e nient’altro. Quello spende 27,1 nat. La clessidra, con otto
numeri, ne spende 16,3, cioè **tre quinti**: comprimere una cifra in un ottavo
dello spazio costa il sessanta per cento di quello che costerebbe non guardarla
affatto. Quel 27,1 tornerà, e in un posto che non ci si aspetta.

```python
LIVELLI = " .:-=+*#%"


def affianca(*immagini):
    """Le immagini 8x8 stampate una accanto all'altra, in caratteri."""
    griglie = [(im.reshape(8, 8) * 8).round().long().clamp(0, 8) for im in immagini]
    return "\n".join("   ".join("".join(LIVELLI[i] for i in g[r]) for g in griglie)
                     for r in range(8))


with torch.no_grad():
    codici = rete.encoder(X)
    ricostruite = torch.sigmoid(rete.decoder(codici))

print("quattro cifre vere")
print(affianca(*X[:4]))
print("\nle stesse, rifatte a partire da otto numeri")
print(affianca(*ricostruite[:4]))
```

```text
quattro cifre vere
  :*=         **:        :%*       =%*
  *%+%:       *%=       :%%#      =*-%:
 :%. *=      :%%-       =*=%      . **
 :*  ==     =%%%.        -%*       .%*
 :=  ==       %%:       =*%          **
 :*  *=       %%-      =%%:           +=
 .#:+*        %%-      :*%%*:      =::#=
  -*+         *%+        :*%=      =**=

le stesse, rifatte a partire da otto numeri
  -#*:        =#:        -%*       -#+
  #*+#.      .%%=       .###.     :*+#:
 :#..#-      =%%-       :**#      .:+%.
 :+  ++     .*%%.       .=#=       .*#.
 :=  ==      :%%.       +%#          +*
 .*  *:      .#%-      =%%*.          *-
  *=*#       .#%-      :###*.      ::-#=
  -#*.        +#=        :#%+      =##+
```

Le quattro cifre vere sono uno zero, un uno, un due e un tre; le prime due si
leggono a colpo d’occhio, il due si riconosce dalla base larga nella penultima
riga, il tre bisogna proprio saperlo. Ma non è questo il punto. Il punto è che la riga di sotto **ripete la riga di sopra**,
tratto per tratto, e ci arriva partendo da otto numeri soli. La compressione
funziona, e il resto della sezione non la mette in dubbio.

## Il cammino che si perde

Adesso la domanda che ci interessa. Prendiamo due cifre vere, guardiamo i loro
otto numeri, e camminiamo **in linea retta** dall’una all’altra, fermandoci
lungo la strada a far dipingere il copista. Camminare in linea retta fra due
file di numeri vuol dire fare la stessa cosa su ciascuna posizione: se la prima
scheda comincia con 3 e la seconda con 7, a metà strada quel numero vale 5, a
un quarto vale 4, e così per tutte e otto le posizioni insieme. Se la mappa
delle schede fosse un posto sensato, dovremmo vedere una cifra trasformarsi con
continuità nell’altra.

```python
with torch.no_grad():
    partenza, arrivo = codici[0], codici[1]      # lo zero e l'uno di prima
    tappe = torch.stack([partenza + t * (arrivo - partenza)
                         for t in torch.linspace(0, 1, 5)])
    print(affianca(*torch.sigmoid(rete.decoder(tappe))))
```

```text
  -#*:       .++.       .=*:       .+*.        =#:
  #*+#.      **+#.      +*+*       -%#=       .%%=
 :#..#-     .#..#:      *-:*.      *#++       =%%-
 :+  ++     .*:.+.     .***+      .*##=      .*%%.
 :=  ==     .*:.=.      *##=       -#%:       :%%.
 .*  *:     .#..*.      *-=+       :*#=       .#%-
  *=*#       +==*       -===       :#*=       .#%-
  -#*.       .**.       .+*:        +#-        +#=
```

Agli estremi ci sono lo zero e l’uno, riconoscibili. In mezzo l’anello dello
zero si stringe e si riempie di grigio a poco a poco, senza mai chiudersi del
tutto; poi si storce; poi resta una barra spessa che non è ancora un uno. La
prima delle tre tappe intermedie è ancora uno zero, uno zero che si sta
sfaldando; le altre due non si lasciano chiamare per nome: **non sono cifre**. Il copista, in quei punti, non c’è mai stato,
e dipinge quello che gli riesce.

Lo stesso succede, e peggio, provando a inventare da zero. «Inventare» qui vuol
dire una cosa precisa: si guarda dove stanno i codici veri (il loro centro, e
quanto sono sparpagliati attorno a quel centro, una posizione per volta), si
pesca un punto a caso in quella zona, e lo si dà al copista. È il modo più
ragionevole di provarci, e sotto si vede come va. Che si guardi una posizione
per volta è una semplificazione, e si paga: tenendo conto anche di come le
posizioni vanno d’accordo fra loro il divario si accorcia, da 2,2 spaziature a
1,8, senza sparire. La forma dell’archivio non è quella di una nuvola semplice,
e non basta correggerne l’inclinazione.

Un’ultima cosa da fissare, perché il numero che segue si regge su quella: la
distanza fra due schede si misura come quella fra due punti su una carta.
Differenza posizione per posizione, ognuna al quadrato, si somma e si prende la
radice: è il teorema di Pitagora con otto cateti invece di due.

```python
with torch.no_grad():
    sorteggiati = codici.mean(0) + codici.std(0) * torch.randn(500, 8)
    inventate = torch.sigmoid(rete.decoder(sorteggiati))

print("quattro cifre decodificate da codici sorteggiati")
print(affianca(*inventate[:4]))

fra_codici = torch.cdist(codici, codici)
fra_codici.fill_diagonal_(float("inf"))
spaziatura = fra_codici.min(1).values.median()
lontananza = torch.cdist(sorteggiati, codici).min(1).values.median()

print(f"\nfra un codice vero e il suo vicino:   {spaziatura:.2f}")
print(f"fra un codice sorteggiato e i veri:   {lontananza:.2f}"
      f"   ({lontananza / spaziatura:.1f} volte la spaziatura)")
```

```text
quattro cifre decodificate da codici sorteggiati
  =%%.       .=%%%+     .*%*       :#*
 +%%%%       *+.%%.     *%#%.     .#+*-
 #%.+#      .#  =+      #==#.     .: ++
 -+  .:     .-          ++#+         #-
 -*  ++     :=.:.       =%%-        :#:
 #%%.*%     :#-+=       #%#*        **:.
 +%%+#%      +-=:       =#*+       .#+=:
  =%%%:      .#*:       .*#=       :*+.

fra un codice vero e il suo vicino:   2.34
fra un codice sorteggiato e i veri:   5.20   (2.2 volte la spaziatura)
```

Le quattro immagini hanno l’aria di cifre e non lo sono, e conviene non
insistere oltre: la seconda e la quarta hanno tratti che si interrompono a
metà, la prima è un anello troppo grasso, la terza una forma piena, e nessuna
delle quattro si lascia chiamare per nome. Ma il numero conta più delle
immagini,
perché dice il **perché** invece del sintomo. I codici veri stanno a poco più
di due unità l’uno dall’altro; un codice sorteggiato dista più di cinque dal
più vicino dei codici veri. Sorteggiare in quello spazio vuol dire finire, di
norma, a **più del doppio** della distanza che separa due schede vere. È terra
mai battuta, e non è una zona di frontiera: è la regola.

## Perché la strozzatura non basta

Conviene dire per bene di chi sia la colpa, perché non è dell’archivista.

`````{tab} Elementare

Guardiamo la pagella con cui i due sono stati giudicati: «la copia somiglia
all’originale?». È l’unica domanda che è stata fatta loro, per milioni di
volte. In quella domanda non compare da nessuna parte la richiesta di tenere le
schede in ordine nel cassetto, né quella di riempire i vuoti fra una scheda e
l’altra, né quella di dipingere qualcosa di sensato partendo da una scheda che
nessuno ha mai scritto. Quello che non si chiede non si ottiene, e qui non è
stato chiesto.

E c’è un motivo per aspettarsi anche di peggio, che non è dimostrato ma è
plausibile: se l’unica cosa che conta è che ogni quadro torni indietro
riconoscibile, all’archivista conviene tenere **lontani fra loro i gruppi** di
schede, perché più sono distanti, meno rischia che il copista li confonda. E
allontanare i gruppi, a parità di schede, vuol dire allargare i vuoti in
mezzo. Il cassetto ne esce con le schede addossate in
qualche angolo, larghe distese vuote in mezzo, e nessun confine che dica dove
finisce la zona buona. Per rileggere va benissimo. Per pescare, no: non si sa
dove pescare, e quasi ovunque si peschi non c’è niente.

Manca quindi qualcosa di preciso, e conviene dirlo con esattezza perché è ciò
che la prossima sezione aggiunge: **manca una regola su dove vanno messe le
schede**. Non una scheda migliore: una regola sull’insieme.

`````

`````{tab} Superiore

L’autoencoder ottimizza la sola ricostruzione, e nella sua loss non compare
nulla che riguardi la distribuzione dei codici che produce. Quella
distribuzione, l’**aggregato**

$$
q_\phi(\mathbf{z}) = \mathbb{E}_{\mathbf{x} \sim p_{\text{dati}}}
\big[\delta\big(\mathbf{z} - e_\phi(\mathbf{x})\big)\big],
$$

dove $\delta$ è la delta di Dirac (l’encoder qui è deterministico, quindi ogni
dato contribuisce un punto solo) e $p_{\text{dati}}$ la distribuzione da cui
gli esempi provengono, è definibile ma **inutilizzabile**: nessuno l’ha
vincolata, non se ne conosce la forma, e soprattutto non ci si sa campionare.
Ma generare richiede esattamente quello, cioè una distribuzione da cui pescare
$\mathbf{z}$ prima di decodificare. (Nella sezione seguente lo stesso simbolo
$q_\phi(\mathbf{z})$ tornerà con l’encoder diventato stocastico: là le delta
saranno gaussiane, e l’aggregato sarà una loro mistura.) Sostituirla a posteriori con una gaussiana adattata ai codici,
come nel blocco qui sopra, è la scorciatoia ovvia, e il rapporto $2{,}2$
misurato è quanto costa: la gaussiana copre una regione che
$q_\phi(\mathbf{z})$ non occupa.

E non c’è nemmeno niente che si opponga alla dilatazione del latente, che è
il modo in cui il difetto si manifesta nella misura qui sopra. L’argomento è
euristico e conviene dirlo: a parità del resto, codici più distanti fra loro si
ricostruiscono meglio, perché il decoder ha meno occasioni di confonderli, e
nella loss non compare nessun termine che paghi quella distanza. Si dice, con
formula spiccia, che il latente **non è regolarizzato**, e la regolarizzazione
che manca non è sui pesi: è sulla distribuzione dei codici.

Da qui il programma della sezione seguente. Servono due cose insieme, e sono le
due che il nome «autoencoder variazionale» tiene una per parola: una
distribuzione bersaglio $p(\mathbf{z})$ **scelta in anticipo** (così si sa dove
pescare) e un termine nella loss che spinga i codici a distribuirsi come lei.
La sorpresa, e il motivo per cui la sezione è lunga, è che quel termine non si
inventa: **cade fuori da solo** dal tentativo, tutt’altro, di massimizzare la
verosimiglianza dei dati.

`````

Prima di tirare le somme, una precisazione su che cosa **non** è in
discussione. La clessidra resta il modo giusto di comprimere, ed è così che il
libro la usa quando le chiede di comprimere: nel capitolo sull’audio per
fabbricare un alfabeto del suono, e più avanti per far stare un’immagine in
sedicimila numeri invece di ottocentomila. (In quei due posti la clessidra ha
in più il pezzo che la sezione seguente aggiunge, ma il mestiere che le si
chiede è questo.) Il difetto misurato qui riguarda
un mestiere diverso, fabbricare dati nuovi, che a un compressore nessuno ha mai
chiesto e che nessuna quantità di addestramento gli fa venire.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **autoencoder** è un archivista e un copista che si allenano insieme: il
  primo riduce ogni dato a una scheda di pochi numeri, il secondo ricostruisce
  il dato dalla sola scheda, e il voto è uno solo, quanto la copia somiglia
  all’originale.
- La **strozzatura è la richiesta**, non un limite: potendo scrivere una scheda
  lunga quanto il quadro, i due imparerebbero a fotocopiare.
- Comprimere funziona: otto numeri bastano a rifare una cifra scritta a mano in
  modo che si riconosca.
- **Generare no.** Camminando in linea retta fra due schede vere si incontrano
  punti che non vogliono dire niente, e pescando una scheda a caso si finisce,
  di norma, a più del doppio della distanza che separa due schede vere.
- La colpa non è dell’archivista: nella sua pagella non compariva l’ordine del
  cassetto. Quello che manca è **una regola su dove vanno messe le schede**, e
  la sezione seguente la ricava senza inventarla.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un autoencoder addestra $e_\phi$ e $d_\theta$ sulla sola ricostruzione, con
  $L \ll D$. Nella sua definizione **non compare nessuna distribuzione**.
- Nel caso lineare con errore quadratico ritrova il sottospazio delle prime
  $L$ componenti principali {cite}`bourlard1988auto`, a meno di un cambio di
  base: è la PCA della sezione su riduzione e clustering. Con una non linearità
  **per parte** è la stessa macchina a cui quel vincolo è stato tolto, e la
  superficie che ritrova può essere curva.
- La ricostruzione riesce (16,3 nat per cifra contro i 27,1 di chi dichiara il
  grigio medio di ogni pixel senza guardare la cifra: tre quinti, su cifre da
  64 pixel compresse in 8 numeri); il campionamento no: un codice sorteggiato
  dista dai codici veri $2{,}2$ volte la loro spaziatura tipica.
- La causa è strutturale: l’aggregato $q_\phi(\mathbf{z})$ non è vincolato da
  nulla, e **niente nella loss paga la distanza fra i codici**, quindi niente
  si oppone a un latente dilatato e pieno di vuoti.
- Serve quindi un prior $p(\mathbf{z})$ dichiarato e un termine che avvicini i
  codici a lui. La sezione seguente non lo aggiunge a mano: lo ricava.
```

`````

Resta una cosa da dire prima di andare avanti, ed è un avvertimento che
conviene tenere. Quello che abbiamo appena visto non è un esperimento andato
male: è un esperimento **andato benissimo**, che ha risposto a una domanda
diversa da quella che avevamo in testa. Chiedere «la copia somiglia
all’originale?» e sperare in un archivio ordinato è la versione in miniatura di
un errore che in questo libro torna spesso, e cioè scambiare la cosa che si
misura con la cosa che si vuole. La sezione seguente non aggiusta la clessidra:
cambia la domanda.


[^thread]: Il motivo è che **le somme in virgola mobile non sono
associative**: sommando gli stessi numeri in un ordine diverso il risultato
cambia nell’ultima cifra, perché a ogni passo il totale parziale viene
arrotondato a quante cifre il formato può tenere, e arrotondare un totale
grande insieme a un addendo piccolo ne perde un pezzo. PyTorch, per andare più
veloce, spezza ogni somma lunga fra i nuclei di calcolo disponibili e poi
rimette insieme i pezzi: quanti sono i pezzi dipende da quanti nuclei ha la
macchina, quindi **macchine diverse sommano in ordini diversi**. Su una rete
addestrata per quattromila passi quelle ultime cifre si accumulano, e alla
fine si vedono: il costo di descrizione dell’ultima sezione, senza quella
riga, vale 6,04 lasciando lavorare i quattro nuclei di questa macchina e 6,03
usandone uno solo. Conviene tenere la regola, perché è più larga di PyTorch:
**fissare il seme non basta**, dato che il seme governa i sorteggi e non
l’ordine delle somme. E la riparazione non costa niente, anzi. Su tensori piccoli come questi (1797 cifre da 64 numeri) un thread solo è
**più veloce**, e di parecchio. Non è un paradosso: spartire il lavoro e rimettere
insieme i pezzi ha un costo fisso, che qui si paga quattromila volte, e ogni
volta a ciascun nucleo tocca un pezzo di conto talmente piccolo che il tempo
per consegnarglielo supera il tempo per farlo. Il parallelismo rende quando a
ciascuno tocca abbastanza da fare, ed è la stessa ragione per cui non si
chiamano quattro muratori a spostare un mattone.
