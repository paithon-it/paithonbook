# Meno bit: che cosa si perde arrotondando

Al supermercato nessuno somma i centesimi. Si arrotonda: due e novantacinque
diventa tre, uno e dieci diventa uno, e alla cassa il totale che si aveva in
testa è sbagliato di qualche decina di centesimi su settanta euro. Nessuno se
ne lamenta, perché la domanda a cui il conto a mente serve a rispondere («ci
sto dentro?») non è cambiata.

Un modello fa la stessa cosa, e per la stessa ragione. Ogni peso è un numero
con sette o otto cifre decimali, ma quel numero non serve a nessuno preso da
solo: serve come uno degli addendi di una somma lunga migliaia di termini, e a
valle di quella somma c’è una decisione. Tenere tutte le cifre di ciascun
addendo è come contare i centesimi.

## Arrotondare, e di quanto si sbaglia

Arrotondare non è approssimare alla buona: è decidere una volta per tutte di
quanto si è disposti a sbagliare su ogni numero. Quel «di quanto» ha una regola
sola, e da lì viene tutto il resto della sezione.

`````{tab} Elementare

Un pacco di riso da 1,29 lo segni 1,50, e una confezione d’acqua da 3,60 la
segni 3,50: il **passo** che ti sei dato è mezzo euro, e ammetti solo i suoi
multipli. Ogni prezzo scivola su quello più vicino, quindi sbagli al massimo
venticinque centesimi, metà passo: venticinque sul riso da poco più di un euro,
venticinque su una bottiglia di champagne da cento.

Su trenta prodotti arrotondi in su e in giù senza una regola, così il peggio
possibile, sette euro e mezzo, non ti capita mai: il totale sbaglia attorno al
mezzo euro. E cresce piano. Per far diventare l’errore dieci volte tanto non
bastano dieci prodotti, ce ne vogliono cento. Una rete neurale non usa mai un
peso alla volta, ne somma centinaia, ed è nella somma che gli arrotondamenti si
mangiano a vicenda.

Regge finché il totale è grosso. Nella lista però ci sono anche i resi, che
tolgono invece di aggiungere. Trenta importi da una decina di euro fanno
trecento euro, e mezzo euro non lo nota nessuno. Con metà resi paghi quaranta
euro, e il mezzo euro comincia a vedersi. Con i resi che coprono quasi tutta la
spesa paghi un euro, e l’errore, rimasto identico, vale metà di quello che
paghi. L’arrotondamento è lo stesso di prima; a essersi ristretto è il totale.
In una rete succede uguale quando i numeri che somma si elidono quasi tutti: il
risultato esce piccolo e l’errore ci pesa sopra.

Il passo però non lo scegli tu fino in fondo. Quattro bit sono sedici importi in
tutto, e otto se ne vanno sotto lo zero per i resi, perché un reso da mille euro
è lontano dallo zero quanto una spesa da mille: dallo zero in su ne restano
sette. Se il prodotto più caro costa tremilacinquecento euro il gradino vale
cinquecento, e più stretto non può essere, o la scala finisce prima della cima e
quel prezzo non lo sai proprio scrivere. È il più caro a decidere il passo di
tutti gli altri.

Quando nel carrello non c’è nemmeno un reso, gli otto gradini sotto lo zero
restano vuoti e metà scala non serve a niente. Allora la sposti tutta da una
parte e ti segni quale gradino vale zero: i gradini utili passano da sette a
quindici senza aggiungere un bit. In cambio quel gradino, quello dello zero, te
lo devi ricordare e portare dietro.

E adesso il carrello che rompe tutto: trenta prodotti da pochi euro e, sotto, un
televisore da tremila. Il passo deve arrivare fin lassù, quindi diventa enorme e
tutta la spesa piccola arrotonda a zero. Il conto che hai in testa dice tremila
euro tondi, alla cassa ne paghi tremilasettanta. Il televisore costa quello che
costa ed è al suo posto; a essere sbagliato è il passo, che vale per tutti.

`````

`````{tab} Superiore

Quantizzare a $b$ bit vuol dire rappresentare un insieme di numeri reali con
un numero fisso di livelli interi, che con la scala presa dal massimo sono
$2^b - 1$ (quindici a quattro bit: il livello più negativo non viene mai
raggiunto, perché la scala è tarata proprio sul valore assoluto più
grande). La forma più usata è **simmetrica**: si fissa una
**scala** $s$ e si pone

$$
q = \mathrm{round}\!\left(\frac{w}{s}\right), \qquad \hat{w} = s\,q,
\qquad s = \frac{\max_i |w_i|}{2^{b-1} - 1},
$$

L’errore di arrotondamento su ciascun peso è limitato da
$|w - \hat{w}| \le s/2$, e non dipende da $w$: è una proprietà del passo, non del
numero. (Troncare $q$ all’intervallo rappresentabile è una prudenza che con la
scala presa dal massimo non scatta mai: serve soltanto se la scala viene da
altro, per esempio da una calibrazione fatta su dati diversi.)

Quello che conta però non è l’errore sul peso, è l’errore sull’uscita. Per un
prodotto scalare $\sum_i w_i x_i$ l’errore è $\sum_i (\hat{w}_i - w_i) x_i$,
una somma di $n$ termini che, se gli arrotondamenti sono approssimativamente
indipendenti e a media nulla, cresce come $\sqrt{n}$.

Perché l’errore **relativo** non si accumuli serve però che anche il segnale
cresca come $\sqrt{n}$, e questa è un’ipotesi **sugli ingressi**, non sugli
arrotondamenti: vale se i termini $w_i x_i$ sono incoerenti fra loro, cioè si
cancellano in parte come farebbe una somma di numeri a segno casuale. È
un’ipotesi che si dà per scontata e non lo è: un neurone addestrato allinea
$\mathbf{w}$ alla configurazione che vuole riconoscere, e quando l’ingresso è
proprio quella configurazione il segnale cresce come $n$ e l’errore relativo
**migliora**; quando invece l’ingresso è quasi ortogonale ai pesi il segnale
quasi si annulla e l’errore relativo peggiora di molto. Misurato sulla stessa
matrice, a parità di errore sui pesi: 13% su ingressi casuali, 0,1% su un
ingresso allineato, 58% su uno quasi ortogonale.

L’ipotesi di media nulla sugli arrotondamenti invece regge, e si può
controllare: fino a tre bit lo scarto medio sta sotto il millesimo del passo e
la correlazione fra errore e peso è nulla. A due bit crolla (correlazione
$-0{,}76$), ma a due bit è già crollato tutto.

Il punto delicato è la definizione della scala, perché $s$ è fissata dal
**massimo in valore assoluto** del gruppo di numeri che condividono la scala. Un
singolo elemento molto più grande degli altri allarga $s$ per tutti, e ogni
altro elemento del gruppo perde risoluzione in proporzione. Il rimedio non è
mai «arrotondare meglio»: è **cambiare chi condivide la scala**, restringendo
il gruppo oppure tenendone fuori i pochi elementi anomali.

Questa è la forma **simmetrica**, che dà per scontato che i numeri stiano
attorno allo zero. Dove non è così (le uscite di una ReLU, per dire, sono tutte
non negative, e metà dei livelli andrebbe sprecata) si usa la forma
**asimmetrica**, che aggiunge un intero $z$, lo *zero-point*, cioè il livello
che rappresenta il valore reale zero:

$$
\hat{w} = s\,(q - z), \qquad q = \mathrm{round}(w/s) + z .
$$

Il meccanismo è lo stesso e il passo lo dettano sempre gli estremi del gruppo
(nella forma simmetrica basta il più grande in valore assoluto, qui servono
tutti e due); cambia solo che il gruppo può stare tutto da una parte. È la forma con
cui il capitolo su MLOps parla di `int8` in produzione
{cite}`jacob2018quantization`, e da qui in avanti si resta sulla simmetrica,
che ha una formula in meno.

`````

## Quanti bit servono davvero

La domanda si misura, e la risposta non è quella che si sente ripetere.

Si prende una matrice di pesi, la si arrotonda a un certo numero di bit, e si
guarda di quanto cambia **il risultato della moltiplicazione**, che è l’unica
cosa che il resto della rete vedrà.

Da qui in avanti il passo ha il suo nome tecnico, **scala**: è la
stessa identica cosa, la larghezza di un gradino, e la parola serve perché
adesso comincia a contare **chi la condivide**. Il conto si fa in due modi: con
una scala sola per tutta la matrice, e con una scala ogni sessantaquattro
pesi.

```python
import torch

torch.manual_seed(0)
# un thread solo: cosi' i numeri stampati sono gli stessi su ogni macchina
torch.set_num_threads(1)


def quantizza(t, bit, gruppo=None):
    """Porta i numeri su 2**bit livelli interi e li riporta indietro.

    Con `gruppo` la scala non e' una per tutto il tensore, ma una ogni
    `gruppo` numeri consecutivi lungo l'ultima dimensione."""
    q = 2 ** (bit - 1) - 1
    if gruppo is None:
        s = t.abs().max()
        return torch.round(t / s * q).clamp(-q - 1, q) * s / q
    f = t.reshape(*t.shape[:-1], -1, gruppo)
    # il `clamp` sul denominatore non e' pedanteria: se un gruppo e' tutto di
    # zeri la scala vale zero e la divisione restituisce `nan` senza avvisare.
    # Non succede sui pesi di una rete addestrata; succede eccome su una rete
    # potata, che e' esattamente la cosa che il capitolo invita a comporre
    s = f.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    return (torch.round(f / s * q).clamp(-q - 1, q) * s / q).reshape(t.shape)


W = torch.randn(256, 512)
x = torch.randn(512, 64)
vero = W @ x


def errore(Wq):
    """Di quanto cambia il risultato, in percentuale."""
    return ((Wq @ x - vero).norm() / vero.norm() * 100).item()


print(f"{'bit':>4} {'una scala per tutto':>21} {'una scala ogni 64 pesi':>24}")
for bit in (8, 6, 4, 3):
    print(f"{bit:>4} {errore(quantizza(W, bit)):>20.2f}% "
          f"{errore(quantizza(W, bit, 64)):>23.2f}%")
```

```text
 bit   una scala per tutto   una scala ogni 64 pesi
   8                 1.04%                    0.60%
   6                 4.22%                    2.43%
   4                18.71%                   10.77%
   3                43.36%                   24.96%
```

Otto bit costano l’uno per cento, e a sei bit si è ancora sotto il cinque: fin
lì si può dire che arrotondare sia quasi gratis. **A quattro bit no**, ed è la
riga da guardare due volte.

Quel diciotto e sette per cento conviene tradurlo, perché da solo non dice
niente. È il rapporto fra la lunghezza del vettore degli errori e quella del
vettore dei risultati veri, e vuol dire che l’uscita **tipica** dello strato è
lontana quasi un quinto dal valore che avrebbe dovuto avere. Non è un
arrotondamento all’ultima cifra, è uno scostamento grosso, che la rete userà
come se fosse il risultato buono. E la distribuzione è peggiore di quel che il
numero lascia intendere: misurato, **un’uscita su otto sbaglia di più del
proprio valore**, e sono le più piccole, cioè proprio quelle su cui una
decisione si gioca per poco. E lo strato dopo prende quei numeri per veri e ci
aggiunge il suo errore. Non c’è una formula semplice per dire quanto lo
scostamento cresca lungo una rete di trenta strati (dipende da che cosa
ciascuno fa), ma la direzione è una sola, e non è verso il basso.

Quindi la frase che si sente dire, «i modelli girano a quattro bit», non
significa che quattro bit bastino ad arrotondare. Significa che a quattro bit
si arriva **facendo qualcosa di più che arrotondare**, e il resto della sezione
è quel qualcosa.

La colonna di destra è il primo pezzo, ed è il più economico: invece di una
scala sola per due milioni di pesi se ne tiene una ogni sessantaquattro. Il
costo si conta: a quattro bit, sessantaquattro pesi occupano trentadue byte, e
una scala in sedici bit ne occupa due, cioè il sei per cento in più. In cambio
l’errore si divide per un fattore **1,74**, e con una costanza notevole: è lo
stesso a otto bit come a tre, e si vede dividendo le due colonne riga per riga.
La ragione è quella del carrello: più piccolo è il gruppo che condivide il
passo, meno un elemento grande può rovinare i suoi vicini.

## Le poche componenti enormi

Sui pesi il problema del carrello si vede poco, perché i pesi di una rete
addestrata sono distribuiti in modo abbastanza regolare. Sulle **attivazioni**,
cioè sui numeri che scorrono da uno strato all’altro, no: nei modelli
linguistici grandi ci sono poche **componenti** (una componente è uno dei
numeri della fila che passa da uno strato al successivo, sempre nella stessa
posizione) che arrivano a valere fino a venti volte tutte le altre
{cite}`dettmers2022llmint8`. Nell’esperimento qui sotto il rapporto è tarato
più in alto ancora, a trentasei volte, per rendere visibile in dieci righe un
effetto che nei modelli veri si accumula su molti strati.

Sono il prodotto da tremila euro dentro il carrello della spesa. E il rimedio è
quello che verrebbe in mente a chiunque alla cassa: quel prodotto lì lo si
conta a parte, per esteso, e si arrotonda tutto il resto.

```python
X = torch.randn(64, 512)
enormi = [7, 133, 401]            # tre componenti su 512, lo 0,6 per cento
X[:, enormi] *= 60
atteso = X @ W.T


def errore_x(Xq):
    return ((Xq @ W.T - atteso).norm() / atteso.norm() * 100).item()


resto = [i for i in range(512) if i not in enormi]
misto = X.clone()
misto[:, resto] = quantizza(X[:, resto], 8)   # la scala si calcola senza le enormi

print(f"la componente normale piu' grande vale {X[:, resto].abs().max():.1f}")
print(f"la componente enorme piu' grande vale  {X[:, enormi].abs().max():.1f}")
print()
print(f"8 bit, una scala per tutto:           {errore_x(quantizza(X, 8)):6.2f}%")
print(f"8 bit, ma le tre enormi tenute intere: {errore_x(misto):6.2f}%")
```

```text
la componente normale piu' grande vale 4.7
la componente enorme piu' grande vale  167.6

8 bit, una scala per tutto:             7.28%
8 bit, ma le tre enormi tenute intere:   0.20%
```

Tre colonne su cinquecentododici, cioè lo 0,6 per cento dei numeri, tenute per
esteso invece che arrotondate, e l’errore passa da poco più del sette per cento
a due decimi. Non è una rifinitura: è la differenza fra un modello che
funziona e uno che farnetica, e la scoperta che quelle componenti esistano e
siano poche è la ragione per cui `int8` è diventato praticabile sui modelli
linguistici {cite}`dettmers2022llmint8`.

`````{tab} Elementare

L’esperimento dice una cosa sola, ed è che il danno non era distribuito. Il
guasto non stava un po’ dentro ogni numero: erano tre a stare larghissimi, e
per colpa loro tutti gli altri sono stati schiacciati. La prova è che togliendo
dal gruppo soltanto quei tre l’errore crolla di trentasei volte: se il danno
fosse stato sparso, spostarne tre su cinquecentododici non avrebbe cambiato
niente.

E non dice che il problema si risolva sempre così. Funziona perché le
componenti enormi sono **poche**: se fossero tante non ci sarebbe niente da
mettere da parte, e si tornerebbe al passo grosso per tutti. Qui le tre erano
note in partenza, ma nei modelli veri non si elencano: si guarda ogni fila di
numeri appena arriva e si mette da parte tutto quello che supera una soglia. Le
posizioni tendono a essere sempre le stesse, ed è questo a rendere il rimedio
economico; ma è una tendenza osservata, non una lista fissa da cui si parte.

Il rimedio completo ha due metà, e la prima si è già vista: stringere il gruppo
che condivide il passo, una scala ogni sessantaquattro numeri invece di una
sola per tutti. Tenere fuori dal gruppo le poche componenti larghe è la
seconda.

Sotto gli otto bit non bastano nemmeno le due insieme, e i metodi che reggono
cambiano il gesto dell’arrotondare. Uno arrotonda un prezzo alla volta e tiene
il conto di quanto ha sbagliato: se il primo prezzo l’ha tirato su di venti
centesimi, quei venti centesimi li toglie a un prodotto che deve ancora
arrotondare, scegliendo quello a cui la correzione dà meno fastidio, così alla
fine il totale torna. L’altro guarda le quantità. Un prodotto che compri in
cinquanta copie, sbagliato di venti centesimi al pezzo, ti sposta il conto di
dieci euro; lo stesso errore su un prodotto comprato una volta sola sposta
venti centesimi. Allora al prezzo del prodotto da cinquanta copie si dà una
scala tutta sua, a gradini fini, e si arrotonda largo il resto. Tutti e due
guardano che cosa quel numero combina nel conto, e non soltanto quanto vale.

`````

`````{tab} Superiore

L’osservazione empirica è che nei Transformer, oltre una certa scala, compaiono
**caratteristiche anomale sistematiche**: un numero piccolo di dimensioni del
canale nascosto assume valori di ordini di grandezza superiori alle altre, in
modo consistente fra token e fra ingressi {cite}`dettmers2022llmint8`. Poiché la
scala di quantizzazione è fissata dal massimo, quelle dimensioni comprimono
tutte le altre in pochi livelli.

Il metodo che ne è nato ha due parti. La prima è la stretta sulla granularità:
si abbandona la scala unica e se ne tiene una **per ogni prodotto interno**,
cioè si stringe al massimo il gruppo che condivide il passo. La seconda è la
decomposizione a precisione mista, che tratta separatamente i due sottospazi:

$$
\mathbf{X}\mathbf{W}^{\top} =
\underbrace{\mathbf{X}_{\mathcal{O}}\mathbf{W}_{\mathcal{O}}^{\top}}_{\text{a piena precisione}}
+ \underbrace{\mathbf{X}_{\bar{\mathcal{O}}}\mathbf{W}_{\bar{\mathcal{O}}}^{\top}}_{\text{a 8 bit}},
$$

dove $\mathcal{O}$ è l’insieme delle dimensioni anomale, determinato **a ogni
prodotto** come l’insieme delle dimensioni che contengono almeno un valore
sopra una soglia (nel lavoro originale $\alpha = 6{,}0$), non fissato una volta
per tutte. Il costo è che una
frazione minuscola del prodotto resta in virgola mobile; il guadagno è che la
scala del resto non è più dettata da loro.

Sotto gli otto bit la decomposizione non basta più, e i metodi che funzionano
smettono di trattare l’arrotondamento come un’operazione locale. **GPTQ**
{cite}`frantar2023gptq` quantizza i pesi di uno strato uno alla volta e, dopo
ogni arrotondamento, corregge i pesi non ancora quantizzati per compensare
l’errore introdotto sull’uscita, usando l’informazione del secondo ordine
stimata su un piccolo insieme di dati. **AWQ** {cite}`lin2024awq` parte da
un’osservazione complementare: non tutti i pesi contano uguale, e quelli che
moltiplicano le attivazioni grandi vanno protetti riscalando i canali prima di
arrotondare. In tutti e due i casi la differenza rispetto alla tabella qui sopra
non è la formula dell’arrotondamento, è che **si guarda che cosa quel peso fa**
invece che soltanto quanto vale.

`````

## Arrotondare dopo, o saperlo già durante

C’è un’ultima distinzione, ed è quella che separa due mestieri.

`````{tab} Elementare

Tutto quello che si è visto finora si fa a modello già addestrato: si prende
una rete che esiste, si arrotondano i suoi numeri, si misura quanto si è perso.
È il modo economico, si fa in minuti, e per otto bit basta quasi sempre.

L’altro modo è dire alla rete, mentre impara, che alla fine i suoi pesi
verranno arrotondati: come un negozio che sa già che la cassa accetta soltanto
i mezzi euro, e i prezzi li sceglie di conseguenza invece di lasciarli a due e
novantasette. La rete tiene i numeri precisi da una parte e fa i conti con
quelli arrotondati, e così si accorge di quando un peso sta in bilico fra due
gradini e lo sposta dove l’arrotondamento gli fa meno male. Costa un
addestramento intero, e per questo si fa solo quando si scende in basso coi bit
e arrotondare a cose fatte non regge.

C’è una difficoltà, ed è graziosa: arrotondare è un’operazione a gradini, e una
funzione a gradini è **piatta** dappertutto tranne che nei salti. Una rete
impara seguendo la pendenza, e sul piano di un gradino non c’è nessuna pendenza
da seguire: il segnale d’apprendimento morirebbe subito. Il rimedio è una
piccola finzione: si fanno i conti in avanti con i valori arrotondati e
all’indietro si fa finta che l’arrotondamento non ci sia. Fanno eccezione i
numeri finiti fuori dalla scala, ai quali il segnale non arriva proprio. Non è
matematicamente pulito, e funziona.

`````

`````{tab} Superiore

La distinzione è fra **quantizzazione post-addestramento** (PTQ), che opera su
pesi già fissati e al più calibra le scale su un piccolo insieme di dati, e
**addestramento consapevole della quantizzazione** (QAT), che inserisce
l’operazione di quantizzazione nel grafo in avanti durante l’ottimizzazione
{cite}`jacob2018quantization`. Nel secondo caso i pesi in virgola mobile
restano come «pesi ombra» e vengono aggiornati normalmente; il passaggio in
avanti usa la loro versione quantizzata.

Il problema tecnico è che $\mathrm{round}(\cdot)$ ha derivata nulla quasi
ovunque e non definita nei punti di salto, quindi il gradiente rispetto ai pesi
ombra sarebbe zero. La soluzione standard è lo **stimatore diretto**
(*straight-through estimator*) {cite}`bengio2013estimating`: nel passaggio
all’indietro si sostituisce la derivata dell’arrotondamento con l’identità
(tipicamente troncata fuori dall’intervallo rappresentabile),

$$
\frac{\partial \hat{w}}{\partial w} \approx
\begin{cases}
1 & \text{se } w \text{ è nell’intervallo rappresentabile},\\
0 & \text{altrimenti}.
\end{cases}
$$

È un gradiente **sbagliato** per costruzione, e la giustificazione è empirica:
punta nella direzione giusta abbastanza spesso da far convergere
l’ottimizzazione. Il costo di QAT è un addestramento completo, e la regola
pratica è la solita: si usa PTQ, si misura, e si passa a QAT solo quando la
misura dice che non basta.

`````

Messe in fila, le cose di questa sezione dicono una cosa sola, e conviene
tenersi quella invece dell’elenco: **la domanda giusta non è quanti bit, è chi
condivide il passo**. Cambiando chi lo condivide si passa dal 18,7% al 10,8% a
parità di bit; togliendo dal gruppo tre numeri su cinquecentododici si passa
dal 7,28% allo 0,20%. Il numero di bit, da solo, non ha spiegato nessuno dei
due salti.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Arrotondare i pesi funziona per due ragioni: l’errore su ciascun numero è al
  massimo **metà passo**, e sommando tanti numeri gli errori vanno in su e in
  giù e si **compensano**. Una rete somma sempre tanti numeri insieme, e questa
  è la sua fortuna.
- Il passo lo decide il numero **più grande** del gruppo che lo condivide.
  Quindi la domanda vera non è «quanti bit», è **chi condivide il passo**:
  misurato, una scala ogni sessantaquattro pesi taglia l’errore di un fattore
  1,74 rispetto a una scala sola per tutta la matrice, e lo stesso fattore vale
  a otto bit come a tre.
- Otto bit costano circa l’uno per cento e si possono considerare gratis.
  **Quattro bit, arrotondando e basta, costano quasi il venti**: chi dice che i
  modelli girano a quattro bit sta parlando di metodi che fanno molto più che
  arrotondare.
- Nei modelli linguistici poche componenti valgono decine di volte le altre e
  rovinano il passo per tutti. Tenendo intere **tre componenti su
  cinquecentododici** l’errore passa dal 7,28% allo 0,20%.
- Si può arrotondare a modello finito (economico, e per otto bit basta) oppure
  dirlo alla rete mentre impara, così si sposta da sola dove l’arrotondamento
  le fa meno male (costa un addestramento intero).
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Quantizzazione simmetrica a $b$ bit: $\hat{w} = s\,\mathrm{round}(w/s)$ con
  $s = \max|w| / (2^{b-1}-1)$. L’errore per elemento è limitato da $s/2$;
  sull’uscita di un prodotto scalare gli errori indipendenti crescono come
  $\sqrt{n}$, quindi l’errore **relativo** non si accumula.
- La **granularità** della scala è la leva più economica: per tensore, per riga,
  per gruppo di $g$ elementi. Misurato su un prodotto $256 \times 512$ per
  $512 \times 64$: a quattro bit, 18,71% con una scala per tutto e 10,77% con
  una scala ogni 64.
- Le **caratteristiche anomale** dei Transformer {cite}`dettmers2022llmint8`
  dettano la scala e schiacciano tutto il resto. La decomposizione a precisione
  mista le tiene fuori: 7,28% contro 0,20% sullo stesso prodotto.
- Sotto gli otto bit servono metodi che non trattino l’arrotondamento come
  locale: **GPTQ** {cite}`frantar2023gptq` compensa sull’uscita l’errore già
  commesso, **AWQ** {cite}`lin2024awq` protegge i canali che moltiplicano le
  attivazioni grandi.
- **PTQ** contro **QAT**: la seconda mette la quantizzazione nel passaggio in
  avanti durante l’addestramento e aggira la derivata nulla di
  $\mathrm{round}$ con lo **stimatore diretto**
  {cite}`bengio2013estimating`, cioè un gradiente deliberatamente sbagliato che
  funziona.
```

`````

Questa leva lascia la rete esattamente com’è: stessi collegamenti, stessa
forma, numeri scritti più corti. La leva che segue fa il contrario, e va a
toccare i collegamenti: ne toglie una parte e li lascia dove sono, che è una
promessa più grande e, come si vedrà, molto più difficile da riscuotere.
