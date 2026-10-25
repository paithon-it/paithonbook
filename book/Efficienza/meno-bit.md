---
titolo_ricerca: "Quantizzazione: meno bit, che cosa si perde"
descrizione: "Come funziona la quantizzazione dei modelli: quanto si sbaglia
  arrotondando, quanti bit servono, le componenti anomale, gli otto bit in
  virgola mobile."
---
# Meno bit: che cosa si perde arrotondando

Al supermercato nessuno somma i centesimi. Si arrotonda: due e novantacinque
diventa tre, uno e dieci diventa uno, e alla cassa il totale che si aveva in
testa è sbagliato di qualche decina di centesimi su settanta euro. Nessuno se
ne lamenta, perché la domanda a cui il conto a mente serve a rispondere («ci
sto dentro?») non è cambiata.

Un modello fa la stessa cosa, e per la stessa ragione. Ogni peso è un numero
con circa sette cifre significative, ma quel numero non serve a nessuno preso
da solo: serve come uno degli addendi di una somma lunga migliaia di termini, e
dopo quella somma c’è una decisione. Tenere tutte le cifre di ciascun addendo,
quindi, è come contare i centesimi.

## Arrotondare, e di quanto si sbaglia

Arrotondare ai multipli di un **passo** $s$ vuol dire sbagliare al più $s/2$
su ogni numero. In altri termini, [scegliere il passo è scegliere di quanto si
è disposti a sbagliare]{.pt-matita}, e da questa scelta discende tutto il resto.

`````{tab} Elementare

Un pacco di riso da 1,29 lo segni 1,50, e una confezione d’acqua da 3,60 la
segni 3,50: il passo che ti sei dato è mezzo euro, e ammetti solo i suoi
multipli. Ogni prezzo scivola su quello più vicino, quindi sbagli al massimo
venticinque centesimi, cioè metà passo: venticinque sul riso da poco più di un
euro, venticinque su una bottiglia di champagne da cento.

Su trenta prodotti arrotondi in su e in giù senza una regola, così il peggio
possibile, sette euro e mezzo, non ti capita mai: il totale sbaglia di qualche
decina di centesimi. L’errore cresce piano: con dieci volte tanti prodotti
diventa poco più di tre volte tanto, e per farlo diventare dieci volte tanto di
prodotti ne servono cento volte tanti. Una rete neurale non usa mai un peso
alla volta, ma ne somma centinaia, e nella somma gli arrotondamenti si
mangiano a vicenda.

Tutto questo regge finché il totale è grosso. Nella lista però ci sono anche i
resi, che tolgono invece di aggiungere. Senza resi, trenta importi da una
decina di euro fanno trecento euro, e mezzo euro non lo nota nessuno. Se metà
sono resi, acquisti e resi quasi si pareggiano e il totale si fa piccolo,
poniamo quaranta euro: il mezzo euro comincia a vedersi. Con i resi che coprono
quasi tutta la spesa paghi un euro, e l’errore, rimasto identico, vale metà di
quello che paghi.

L’arrotondamento è lo stesso di prima; il totale si è ristretto. In una
rete succede uguale quando i numeri che somma si cancellano quasi tutti a
vicenda: il risultato esce piccolo e l’errore ci pesa sopra.

Però non scegli tu il passo fino in fondo. Quattro bit, ad esempio, sono sedici
importi in tutto. Uno è lo zero; otto se ne vanno sotto lo zero per i resi,
perché un reso da mille euro è lontano dallo zero quanto una spesa da mille;
sopra lo zero ne restano sette. Se il prodotto più caro costa
tremilacinquecento euro, per arrivare fin lassù il gradino deve valere
cinquecento. In altre parole, il più caro decide il passo di tutti gli altri.

Un gradino più stretto lascerebbe quel prezzo oltre l’ultimo gradino, e il
prezzo andrebbe segnato più basso di com’è. Conviene soltanto quando i prezzi
oltre la cima sono pochi e di poco, perché in cambio tutti gli altri si
arrotondano più fini.

Quando nel carrello non c’è nemmeno un reso, gli otto gradini sotto lo zero
restano vuoti, e metà dei gradini non serve a niente. Allora li sposti tutti da
una parte e ti segni quale gradino vale zero. In questo modo i gradini utili
passano da sette a quindici senza aggiungere un bit; in cambio devi ricordarti
e portarti dietro quel gradino, quello dello zero.

Il caso che rompe tutto è un carrello con trenta prodotti da pochi euro e,
sotto, un televisore da tremila. Il passo deve arrivare fin lassù, quindi
diventa enorme e tutta la spesa piccola arrotonda a zero. Il conto che hai in
testa dice tremila euro tondi, alla cassa ne paghi tremilasettanta. Il
televisore ha il prezzo che ha ed è al suo posto; sbaglia il passo, che vale
per tutti.

`````

`````{tab} Superiore

Quantizzare a $b$ bit vuol dire rappresentare un insieme di numeri reali con
un numero fisso di livelli interi, che con la scala presa dal massimo sono
$2^b - 1$. A quattro bit, ad esempio, sono quindici: il livello più negativo
non viene mai raggiunto, perché la scala è calibrata proprio sul valore
assoluto più grande. La forma più usata è **simmetrica**: si fissa una
scala $s$ e si pone

$$
q = \mathrm{round}\!\left(\frac{w}{s}\right), \qquad \hat{w} = s\,q,
\qquad s = \frac{\max_i |w_i|}{2^{b-1} - 1}.
$$

Ne segue che l’errore di arrotondamento su ciascun peso è limitato da
$|w - \hat{w}| \le s/2$, e non dipende da $w$: è una proprietà del passo, non
del numero.

Prendere la scala dal massimo, però, è una scelta, e a pochi bit non è la
migliore. Con un tetto $c < \max_i |w_i|$ si pone $s = c/(2^{b-1}-1)$ e si
tronca $q$ a $\pm(2^{b-1}-1)$. In questo modo i pochi pesi oltre il tetto
sbagliano di $|w| - c$ (l’errore di *saturazione*), mentre tutti gli altri si
arrotondano con un passo più fine. Il tetto che minimizza l’errore quadratico
bilancia le due parti, e per distribuzioni note Banner e colleghi lo ricavano
in forma chiusa {cite}`banner2019post`; nel paragrafo «Quanti bit servono
davvero» lo si cerca sui pesi della prova a ogni numero di bit.

Il taglio funziona perché i valori sacrificati sono pochi e non contengono
niente di decisivo. Quando sono proprio loro a portare l’informazione,
come le componenti anomale delle attivazioni dei modelli linguistici, tagliarli
vuol dire sbagliare i numeri che contano.

Conta però l’errore sull’uscita, non quello sul peso. Se il passo
non supera un paio di deviazioni standard dei pesi, l’errore di arrotondamento
$e = \hat{w} - w$ si comporta come una variabile uniforme su $[-s/2,\, s/2]$,
indipendente dal peso, con

$$
\mathbb{E}[e] = 0, \qquad \operatorname{Var}(e) = \frac{1}{s}\int_{-s/2}^{s/2} u^2\,\mathrm{d}u = \frac{s^2}{12}.
$$

Consideriamo ora un prodotto scalare $\sum_i w_i x_i$, il cui errore è
$\sum_i e_i x_i$. Se gli $e_i$ sono indipendenti fra loro e dagli ingressi, la
varianza dell’errore è $n\,\frac{s^2}{12}\,\mathbb{E}[x^2]$, quindi la sua
ampiezza cresce come $\sqrt{n}$.
Se anche pesi e ingressi sono indipendenti e a media nulla, il segnale ha
varianza $n\,\sigma_w^2\,\mathbb{E}[x^2]$, e l’errore relativo sull’uscita vale

$$
\frac{s}{\sqrt{12}\,\sigma_w} = \frac{\max_i |w_i|}{\sqrt{12}\,\sigma_w\,(2^{b-1}-1)},
$$

che non dipende da $n$ e si dimezza a ogni bit in più (i circa sei decibel per
bit della teoria del segnale). La formula anticipa la tabella degli errori per
numero di bit. In particolare, su $256 \times 512$ pesi gaussiani il massimo
vale circa $4{,}6\,\sigma_w$ (lo stampa il codice con le due scale), e a
quattro bit la formula dà $4{,}6/(7\sqrt{12}) \approx 19\%$. Con una scala ogni
sessantaquattro pesi, al massimo globale si sostituisce quello tipico di un
gruppo, circa $2{,}6\,\sigma_w$, e il rapporto fra i due è il fattore costante
fra le due colonne.

Perché l’errore relativo non si accumuli serve però che anche il segnale cresca
come $\sqrt{n}$, e questa è un’ipotesi sugli ingressi, non sugli
arrotondamenti: vale se i termini $w_i x_i$ sono incoerenti fra loro, cioè si
cancellano in parte come farebbe una somma di numeri a segno casuale.

Questa ipotesi si dà per scontata ma non lo è. Un neurone addestrato, infatti,
allinea $\mathbf{w}$ alla configurazione che vuole riconoscere. Quando
l’ingresso è quella configurazione, il segnale cresce come $n$ e l’errore
relativo migliora; quando invece l’ingresso è quasi ortogonale ai pesi, il
segnale quasi si annulla e l’errore relativo peggiora di molto. Fra i due
estremi, a parità di errore sui pesi, l’errore relativo sull’uscita può
cambiare di ordini di grandezza.

L’ipotesi di media nulla sugli arrotondamenti, invece, vale finché il passo non
supera un paio di deviazioni standard dei pesi, e a quattro bit (due terzi di
deviazione) come a tre (una e mezza) il passo sta sotto questa soglia: l’errore
resta centrato, scorrelato dal peso e con la varianza $s^2/12$ del modello
uniforme. L’ipotesi cade quando i livelli sono così pochi che l’arrotondamento
diventa una funzione del peso: a due bit i livelli sono $-s$, $0$
e $s$, il passo vale più di quattro deviazioni, quasi tutti i pesi finiscono
sullo zero e l’errore è in pratica $-w$, fortemente anticorrelato con il peso;
ma a due bit, in ogni caso, il modello è compromesso.

La scala è il punto delicato. Se la si prende dal massimo in valore assoluto
del gruppo di numeri che la condividono, un elemento solo la detta, e se
questo è molto più grande degli altri allarga $s$ per tutti: ogni altro
elemento del gruppo perde risoluzione in proporzione.

Finché ogni numero si arrotonda al livello più vicino, indipendentemente dagli
altri (*round-to-nearest*, RTN), i rimedi agiscono tutti sulla scala: la si
prende da un tetto invece che dal massimo, oppure si cambia chi la condivide,
restringendo il gruppo o tenendone fuori i pochi elementi anomali.

Questa è la forma simmetrica, che dà per scontato che i numeri stiano attorno
allo zero. Dove non è così (le uscite di una ReLU, per esempio, sono tutte non
negative, e metà dei livelli andrebbe sprecata) si usa la forma
**asimmetrica**, che aggiunge un intero $z$, lo *zero-point*, cioè il livello
che rappresenta il valore reale zero:

$$
\hat{w} = s\,(q - z), \qquad q = \mathrm{round}(w/s) + z ,
$$

con $s = (\max_i w_i - \min_i w_i)/(2^b - 1)$, $z = \mathrm{round}(-\min_i w_i
/ s)$ e $q$ troncato a $[0,\, 2^b - 1]$. Su valori tutti non negativi la forma
simmetrica usa soltanto i livelli da $0$ a $2^{b-1}-1$, quella asimmetrica
tutti i $2^b$. Il passo quindi scende da $\max_i w_i/(2^{b-1}-1)$ a $\max_i
w_i/(2^b-1)$, cioè a poco meno della metà, come avere un bit in più.

Il meccanismo è lo stesso, e gli estremi del gruppo dettano sempre il passo
(nella forma simmetrica basta il più grande in valore assoluto, qui servono
tutti e due); cambia solo che il gruppo può stare tutto da una parte. È la
forma con cui la {doc}`sezione su come si serve un modello
</MLOps/deployment-e-serving>` parla di `int8` in produzione
{cite}`jacob2018quantization`. Da qui in avanti, però, si resta sulla
simmetrica, che ha una formula in meno.

`````

## Quanti bit servono davvero

La domanda si misura, e la risposta non è quella che si sente ripetere.

Si prende una matrice di pesi, la si arrotonda a un certo numero di bit, e si
guarda di quanto cambia il risultato della moltiplicazione, che è l’unica
cosa che il resto della rete vedrà.

Il passo si chiama anche **scala**, e da qui in avanti conta chi la condivide.
Per questo il calcolo si fa in due modi, con una scala sola per tutta la
matrice e con una scala ogni sessantaquattro pesi.

```python
import torch

torch.manual_seed(0)
# un thread solo: due esecuzioni di fila danno lo stesso numero. Su un'altra
# macchina le ultime cifre ballano, perche' cambia l'ordine delle somme
torch.set_num_threads(1)


def quantizza(t, bit, gruppo=None):
    """Porta i numeri sui 2**bit - 1 livelli interi da -livello a livello, e
    li riporta indietro.

    Con `gruppo` la scala non e' una per tutto il tensore, ma una ogni
    `gruppo` numeri consecutivi lungo l'ultima dimensione."""
    livello = 2 ** (bit - 1) - 1          # il passo e' massimo / livello
    if gruppo is None:
        massimo = t.abs().max()
        return (torch.round(t / massimo * livello).clamp(-livello - 1, livello)
                * massimo / livello)
    f = t.reshape(*t.shape[:-1], -1, gruppo)
    # il `clamp` sul denominatore non e' pedanteria: se un gruppo e' tutto di
    # zeri il massimo vale zero e la divisione restituisce `nan` senza avvisare.
    # Non succede sui pesi di una rete addestrata; succede eccome su una rete
    # potata, come mostra la sezione sulla potatura
    massimo = f.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    return (torch.round(f / massimo * livello).clamp(-livello - 1, livello)
            * massimo / livello).reshape(t.shape)


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

quattro = quantizza(W, 4) @ x - vero      # gli errori a quattro bit, scala sola
print(f"a 4 bit, uscite che sbagliano piu' del proprio valore: "
      f"{(quattro.abs() > vero.abs()).float().mean() * 100:.1f}%")

# i due massimi che dettano il passo: di tutta la matrice e, in media, di un
# gruppo da 64
sigma = W.std()
gruppi = W.reshape(256, -1, 64).abs().amax(-1)
print(f"massimo di tutta la matrice: {W.abs().max() / sigma:.2f} sigma")
print(f"massimo di un gruppo da 64, in media quadratica: "
      f"{gruppi.pow(2).mean().sqrt() / sigma:.2f} sigma")
```

```text
 bit   una scala per tutto   una scala ogni 64 pesi
   8                 1.04%                    0.60%
   6                 4.22%                    2.43%
   4                18.71%                   10.77%
   3                43.36%                   24.96%
a 4 bit, uscite che sbagliano piu' del proprio valore: 11.9%
massimo di tutta la matrice: 4.56 sigma
massimo di un gruppo da 64, in media quadratica: 2.63 sigma
```

A otto bit l’uscita dello strato si sposta dell’uno per cento, e a sei bit si è
ancora sotto il cinque, quindi fin lì arrotondare costa poco. A quattro bit no,
e quella è la riga da guardare due volte.

Quel diciotto e sette per cento è il rapporto fra la lunghezza del vettore
degli errori e quella del vettore dei risultati veri, e vuol dire che l’uscita
tipica dello strato è lontana quasi un quinto dal valore che avrebbe dovuto
avere. È uno scostamento grosso e non un arrotondamento all’ultima cifra, e la
rete lo userà come se fosse il risultato buono.

La distribuzione, poi, è peggiore di quel che il numero lascia intendere: più
di un’uscita su nove, l’11,9%, sbaglia di più del proprio valore, e sono le più
piccole, cioè quelle su cui una decisione si gioca per poco. Lo strato
successivo prende quei numeri per veri e ci aggiunge il suo errore. Non c’è una
formula semplice per dire quanto lo scostamento cresca lungo una rete di trenta
strati (dipende da che cosa ciascuno fa), ma la direzione è una sola, e non è
verso il basso.

Lo strato da solo, però, non dice quanto costi in accuratezza, perché questo
dipende da quanto sono strette le decisioni che seguono, e si misura sul
modello: la rete di cifre della {doc}`sezione sulla potatura
<meno-pesi>`, arrotondata a quattro bit con una scala ogni sessantaquattro
pesi, classifica come prima.

Quindi la frase che si sente dire, «i modelli girano a quattro bit», non
descrive un arrotondamento con una scala sola: a quattro bit si arriva facendo
qualcosa di più.

La colonna di destra è il primo pezzo, il più economico: invece di una
scala sola per centotrentamila pesi se ne tiene una ogni sessantaquattro. Il
costo si conta: a quattro bit, sessantaquattro pesi occupano trentadue byte, e
una scala in sedici bit ne occupa due, cioè il sei per cento in più. In cambio
l’errore si divide per un fattore 1,74, e con una costanza notevole: è lo
stesso a otto bit come a tre.

La ragione è che il peso più grande detta il passo di tutto il gruppo, e lo
confermano i due massimi. Su tutta la matrice il massimo sta a 4,56 deviazioni
standard, mentre in un gruppo da sessantaquattro sta in media a 2,63, e 4,56
diviso 2,63 fa 1,73. [Più piccolo è il gruppo che condivide il passo, meno un
elemento grande può rovinare i suoi vicini]{.pt-matita}.

C’è un secondo modo di stringere il passo, senza scale in più: prenderlo più
corto di quanto chiederebbe il peso più grande, cioè calcolarlo da un tetto più
basso del massimo. I pochi pesi oltre il tetto finiscono schiacciati
sull’ultimo livello, e in cambio tutti gli altri si arrotondano più fini. Il
tetto si cerca sui soli pesi, provando quelli fra una e cinque deviazioni
standard e tenendo quello che li sbaglia di meno, e poi si guarda che cosa fa
all’uscita.

```python
def tagliata(t, bit, tetto):
    """Una scala sola, presa dal tetto invece che dal peso piu' grande:
    quello che sta oltre il tetto finisce sull'ultimo gradino."""
    livello = 2 ** (bit - 1) - 1
    passo = tetto / livello
    return torch.round(t / passo).clamp(-livello, livello) * passo


print(f"{'bit':>4} {'tetto migliore':>16} {'errore':>8}")
for bit in (8, 6, 4, 3):
    tetti = [k / 100 * sigma for k in range(100, 501)]    # da 1 a 5 sigma
    tetto = min(tetti, key=lambda c: (tagliata(W, bit, c) - W).pow(2).sum())
    print(f"{bit:>4} {tetto / sigma:>10.2f} sigma "
          f"{errore(tagliata(W, bit, tetto)):>7.2f}%")
```

```text
 bit   tetto migliore   errore
   8       4.00 sigma    0.95%
   6       3.28 sigma    3.31%
   4       2.47 sigma   11.42%
   3       1.95 sigma   21.44%
```

A quattro bit il tetto a due deviazioni standard e mezzo porta l’errore dal
18,7% all’11,4%, quasi quanto una scala ogni sessantaquattro pesi e senza
nessuna scala in più da tenere; a tre bit fa meglio dei gruppi. A otto bit serve
poco, perché il tetto migliore sta appena sotto il peso più grande. Il metodo
funziona finché quello che si taglia è poco e non conta. Se invece il valore
più grande è proprio quello che decide il risultato, tagliarlo vuol dire
sbagliare il numero che conta di più, ed è il caso delle componenti enormi.

## Le poche componenti anomale

Sui pesi dei modelli linguistici grandi il problema di un massimo che allarga
il passo per tutti si vede poco, e i pesi si arrotondano a otto bit senza
difficoltà {cite}`xiao2023smoothquant`. Non vale per tutte le reti. Nelle
reti piccole i pesi di canali diversi possono avere intervalli
che differiscono di oltre cento volte, e qualche peso anomalo rende meno precisi
tutti gli altri {cite}`jacob2018quantization`. È lo stesso problema, e il
rimedio è una scala per canale invece che per tutto lo strato.

Sulle attivazioni dei modelli linguistici grandi, cioè sui numeri che scorrono
da uno strato all’altro, il problema invece è molto marcato: ci sono poche
componenti (una componente è uno dei numeri della fila che passa da uno strato
al successivo, di solito nella stessa posizione) che arrivano a valere fino a venti
volte tutte le altre {cite}`dettmers2022llmint8`. Nell’esperimento che segue il
rapporto è fissato ancora più in alto, a trentasei volte, per rendere visibile
in dieci righe un effetto che nei modelli veri si accumula su molti strati.

Sono le **componenti anomale** (in inglese *outlier*), e il rimedio è tenerle
a parte: quelle poche restano per esteso, e si arrotonda tutto il resto.

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

[Basta tenere per esteso tre colonne su cinquecentododici]{.pt-matita}, cioè lo
0,6 per cento dei numeri, invece di arrotondarle, e l’errore passa da poco più
del sette per cento a due decimi. Sui modelli linguistici dai 6,7 miliardi di
parametri in su è la differenza fra arrotondare a otto bit anche le attivazioni
senza perdere accuratezza e perderne molta {cite}`dettmers2022llmint8`. Tenerle
a parte è economico perché sono poche e tendono a stare nelle stesse posizioni.

`````{tab} Elementare

L’esperimento mostra che il danno non era distribuito. Non
stava un po’ dentro ogni numero: erano tre a stare larghissimi, e per colpa
loro tutti gli altri sono stati schiacciati. La prova è che togliendo dal
gruppo soltanto quei tre l’errore crolla di trentasei volte: se il danno fosse
stato sparso, spostarne tre su cinquecentododici non avrebbe cambiato niente.

Non ne segue che il problema si risolva sempre così. Funziona perché le
componenti enormi sono poche: se fossero tante non ci sarebbe niente da mettere
da parte, e si tornerebbe al passo grosso per tutti.

Qui le tre erano note in partenza, ma nei modelli veri non si elencano:
si guarda ogni fila di numeri appena arriva e si mettono da parte tutti i
valori che superano una soglia. Le posizioni tendono a essere sempre le stesse,
e per questo il rimedio è economico; ma è una tendenza osservata, non una lista
fissa da cui si parte.

Il rimedio ha due metà: stringere il gruppo che condivide il passo, e tenere
fuori dal gruppo le poche componenti larghe. C’è anche chi non le tiene fuori,
e prima di arrotondare trasforma i numeri in un modo di cui il risultato finale
non si accorge: sposta una parte della loro grandezza nei pesi per cui vengono
moltiplicate, che la reggono meglio, oppure la spalma un poco su tutte le altre
componenti, così che nessuna resti enorme.

Sotto gli otto bit non bastano nemmeno le due insieme, e i metodi che
funzionano cambiano il modo di arrotondare. Uno arrotonda un prezzo alla volta
e annota di quanto ha sbagliato: se il primo prezzo l’ha tirato su di venti
centesimi, li sottrae ai prezzi ancora da arrotondare, ripartendoli nel modo
che disturba meno, così alla fine il totale torna.

L’altro guarda le quantità. Un prodotto che compri in cinquanta copie,
sbagliato di venti centesimi al pezzo, ti sposta il conto di dieci euro, mentre
lo stesso errore su un prodotto comprato una volta sola ne sposta venti
centesimi. Al prezzo del prodotto da cinquanta copie si dà allora una scala
tutta sua, a gradini fini, e si arrotonda largo il resto. Tutti e due i metodi,
quindi, guardano che effetto ha quel numero sul totale, e non soltanto quanto
vale.

`````

`````{tab} Superiore

L’osservazione empirica è che nei Transformer, oltre una certa scala, compaiono
**caratteristiche anomale sistematiche** (*outlier features*): un numero
piccolo di dimensioni del canale nascosto assume valori fino a venti volte più
grandi delle altre, in modo consistente fra token e fra ingressi. Che
esistessero si sapeva già {cite}`kovaleva2021bert,bondarenko2021understanding`.
La misura alla scala e il metodo per aggirarle vengono dal lavoro che ha
reso `int8` praticabile {cite}`dettmers2022llmint8`: la transizione è netta e
cade intorno ai 6,7 miliardi di parametri, dove le anomale invadono tutti gli
strati concentrandosi in sei dimensioni. Poiché la scala di quantizzazione è
fissata dal massimo, quelle dimensioni comprimono tutte le altre in pochi
livelli.

Il metodo ha due parti. La prima è una granularità più fine: si abbandona la
scala unica e se ne tiene una per ogni riga di $\mathbf{X}$ (una per token) e
una per ogni colonna di $\mathbf{W}^{\top}$ (una per unità d’uscita), e ogni
elemento del prodotto si riporta in virgola mobile moltiplicando per le due
scale dei vettori che l’hanno prodotto. Il gruppo che condivide il passo
diventa così un vettore solo, il più piccolo compatibile con un prodotto
matriciale fra interi. La seconda è la **decomposizione a precisione mista**,
che tratta separatamente i due sottospazi:

$$
\mathbf{X}\mathbf{W}^{\top} =
\underbrace{\mathbf{X}_{\mathcal{O}}\mathbf{W}_{\mathcal{O}}^{\top}}_{\text{a piena precisione}}
+ \underbrace{\mathbf{X}_{\bar{\mathcal{O}}}\mathbf{W}_{\bar{\mathcal{O}}}^{\top}}_{\text{a 8 bit}},
$$

dove $\mathcal{O}$ è l’insieme delle dimensioni anomale, determinato a ogni
prodotto come l’insieme delle dimensioni che contengono almeno un valore
sopra una soglia (nel lavoro originale $\alpha = 6{,}0$), non fissato una volta
per tutte. Il costo è che una frazione minuscola del prodotto resta in virgola
mobile; il guadagno è che la scala del resto non è più dettata da loro.

La decomposizione tiene a parte le anomale; altre due strade le tolgono di mezzo
prima di arrotondare, con una trasformazione che lascia invariato il prodotto.
SmoothQuant {cite}`xiao2023smoothquant` divide ogni canale $j$ di $\mathbf{X}$
per un fattore $s_j$ e moltiplica per lo stesso fattore la colonna $j$ di
$\mathbf{W}$,

$$
\mathbf{X}\mathbf{W}^{\top} =
\big(\mathbf{X}\,\mathrm{diag}(\mathbf{s})^{-1}\big)
\big(\mathbf{W}\,\mathrm{diag}(\mathbf{s})\big)^{\top},
$$

così parte della difficoltà passa dalle attivazioni ai pesi, che la reggono
meglio, e tutti e due si arrotondano a otto bit. Una rotazione ortogonale
$\mathbf{Q}$, con $\mathbf{X}\mathbf{W}^{\top} = (\mathbf{X}\mathbf{Q})
(\mathbf{W}\mathbf{Q})^{\top}$, ridistribuisce invece ogni anomala su tutte le
componenti. È ciò che fa FlashAttention-3 in FP8 su query e chiavi, con una
trasformata di Hadamard a segni casuali {cite}`shah2024flashattention3` (la
racconta la {doc}`sezione su FlashAttention </GPU/flash-attention>`).

Sotto gli otto bit la decomposizione non basta più, e i metodi che funzionano
smettono di trattare l’arrotondamento come un’operazione locale. **GPTQ**
{cite}`frantar2023gptq` cerca, strato per strato, i pesi quantizzati che
riproducono meglio l’uscita su un piccolo insieme di $m$ esempi di calibrazione,
raccolti nelle righe di
$\mathbf{X}_{\text{cal}} \in \mathbb{R}^{m \times d_{\text{in}}}$ come i token
nelle righe di $\mathbf{X}$ nella decomposizione:

$$
\arg\min_{\hat{\mathbf{W}}} \big\lVert \mathbf{X}_{\text{cal}}\mathbf{W}^{\top}
- \mathbf{X}_{\text{cal}}\hat{\mathbf{W}}^{\top}\big\rVert_F^2 .
$$

Osserviamo che il problema si separa per righe di $\mathbf{W}$ (una per unità
d’uscita), e che tutte le righe hanno la stessa matrice del secondo ordine
$\mathbf{H} = 2\mathbf{X}_{\text{cal}}^{\top}\mathbf{X}_{\text{cal}}$. Il metodo
quantizza una colonna alla volta, nello stesso ordine per tutte le righe, e
dopo ogni colonna $j$ corregge i pesi $F$ non ancora quantizzati con
l’aggiornamento di *Optimal Brain Surgeon*,

$$
\boldsymbol{\delta}_F = -\frac{w_j - \mathrm{quant}(w_j)}{[\mathbf{H}_F^{-1}]_{jj}}\,(\mathbf{H}_F^{-1})_{:,j},
$$

che sposta il resto della riga in modo da compensare, al secondo ordine,
l’errore appena commesso sull’uscita. È l’ordine fisso a renderlo praticabile,
perché $\mathbf{H}_F^{-1}$ è la stessa per tutte le righe, e si aggiorna una
volta per colonna invece che una volta per peso.

**AWQ** {cite}`lin2024awq` parte da un’osservazione complementare: non tutti i
pesi hanno la stessa importanza, e quelli che moltiplicano le attivazioni
grandi vanno protetti riscalando i canali prima di arrotondare. In tutti e due
i casi la formula dell’arrotondamento resta quella della tabella dei bit: la
differenza sta nel fatto che si guarda che cosa fa quel peso, e non soltanto
quanto vale.

`````

## Otto bit in virgola mobile

Fin qui si è arrotondato su livelli equidistanti: un passo solo, uguale per i
numeri piccoli e per quelli grandi, e livelli che si contano con numeri interi
(è il formato `int8`). Con gli stessi otto bit si può fare un'altra scelta,
che le GPU con i tensor core adatti eseguono direttamente: un numero in
virgola mobile, come il `float32` e il `bfloat16` di
{doc}`Prestazioni e scala </PyTorch/prestazioni>`, solo molto più corto. È
l’**FP8**, che la {doc}`sezione su GEMM e tensor core </GPU/gemm-e-tensor-core>`
nomina fra i formati con cui i tensor core arrivano al loro picco.

`````{tab} Elementare

Un nastro per misurare ha 255 tacche, sempre quelle, e prima di cominciare lo
si stende fino alla cosa più lunga del mucchio. Se nel mucchio ci sono solo
bottoni, le tacche sono fitte e i bottoni si misurano bene. Basta però un
tavolo nel mucchio perché ogni tacca diventi di mezzo centimetro, anche per i
bottoni. È il modo di arrotondare visto fin qui: un passo solo per tutto il
gruppo, e a dettarlo è il più grande.

Le persone, quando dicono una misura, fanno un'altra cosa: tengono sempre
poche cifre e spostano la virgola. Si dice tre centimetri e mezzo, un metro e
settanta, quattro chilometri e mezzo. L'errore è sempre una piccola parte
della misura, qualunque sia la misura, e non importa che cosa ci sia accanto
nel mucchio. Il passo, cioè, cresce con il numero: stretto sulle cose piccole
e largo sulle grandi.

Otto bit si possono spendere così: uno per il segno, quattro per dire dove va
la virgola, tre per le cifre che contano. Con tre bit di cifre un numero non
sbaglia mai più di un sedicesimo di sé stesso, poco più del 6%, e in media
sbaglia molto meno, attorno al 2,6%. Chi ha bisogno di arrivare più lontano
sposta un bit dalle cifre alla virgola: arriva più lontano, ma sbaglia al più
di un ottavo, il 12,5%.

Le correzioni che la rete calcola mentre impara, che variano di moltissimi
ordini di grandezza, si scrivono di solito nel secondo modo; i pesi e i numeri
che passano da uno strato all'altro nel primo, anche durante l'addestramento.

Anche la virgola ha un limite: con quattro bit si sposta solo di tanti posti, e
un numero può andare da un sessantaquattresimo a 448 volte l'unità. Si sceglie
allora per tutto il gruppo l'unità di misura, millimetri o chilometri, in modo
che il più grande resti sotto 448, e quell'unità è il **fattore di scala**.
Finché il più piccolo non scende sotto il sessantaquattresimo, ogni numero
conserva il suo errore piccolo.

Il confronto dice quando conviene. Sui bottoni soli il nastro vince, perché le
sue tacche fitte sbagliano meno delle tre cifre fisse. Con un tavolo nel
mucchio, invece, vince la virgola. C'è poi una terza strada, già vista: tanti
nastri corti, uno ogni sessantaquattro oggetti, così che il tavolo allarghi le
tacche soltanto del suo gruppetto.

`````

`````{tab} Superiore

Nel campo normale un numero a otto bit in virgola mobile vale
$v = \pm\, 2^{\,k - \beta}\,(1 + m/2^{M})$, con il segno nel primo bit,
l’esponente $k$ su $E$ bit, lo scostamento (*bias*) $\beta$ e la mantissa $m$ su
$M$ bit. I due formati proposti da Micikevicius e colleghi
{cite}`micikevicius2022fp8` sono E4M3 ($E = 4$, $M = 3$, $\beta = 7$) ed E5M2
($E = 5$, $M = 2$, $\beta = 15$). E4M3 rinuncia agli infiniti e riserva al NaN
la sola configurazione con esponente e mantissa tutti a uno (una per segno),
così che il massimo sale da 240 a $1{,}75 \cdot 2^8 = 448$, con minimo normale
$2^{-6}$ e subnormali fino a $2^{-9}$; E5M2 segue le convenzioni IEEE e arriva a
$1{,}75 \cdot 2^{15} = 57\,344$, con minimo normale $2^{-14}$.

L’errore relativo dell’arrotondamento al più vicino è limitato, nel campo
normale, da $2^{-(M+1)}$, il 6,25% per E4M3 e il 12,5% per E5M2,
indipendentemente dalla grandezza del numero, e il suo valore quadratico medio,
con una mantissa distribuita in modo log-uniforme, è circa
$0{,}21 \cdot 2^{-M}$, cioè il 2,6% per E4M3 e il 5,3% per E5M2.

Nella quantizzazione intera simmetrica è limitato invece l’errore assoluto, da
$s/2$ con $s = \max|w|/127$, e quello relativo sale fino al 100% sui valori
piccoli, perché tutto ciò che sta sotto $s/2$ finisce sullo zero. Il suo valore
quadratico medio relativo è $s/(\sqrt{12}\,\sigma_w)$, cioè
$(\max|w|/\sigma_w)/(127\sqrt{12}) \approx (\max|w|/\sigma_w)/440$. Ne segue
che l’intero con una scala per tensore batte l’E4M3 finché il massimo resta
sotto una dozzina di deviazioni standard, $\max|w|/\sigma_w \lesssim 11{,}5$.

Gli autori raccomandano E4M3 per pesi e attivazioni, che chiedono precisione, ed
E5M2 per i gradienti, che chiedono intervallo. L’intervallo di un FP8 resta
comunque stretto rispetto a quello di un `bfloat16`, e per questo in pratica
il formato si accompagna sempre a un fattore di scala,
$\hat{w} = s\,\mathrm{fp8}(w/s)$ con $s = \max|w| / 448$ per E4M3, dove
$\mathrm{fp8}(\cdot)$ è l’arrotondamento al valore rappresentabile più vicino
(che in PyTorch, per E4M3, satura a $\pm 448$), calcolato per tensore o per
blocco.

La proprietà del formato vale finché il rapporto fra il più grande e il più
piccolo valore del gruppo sta dentro l’intervallo: oltre, i piccoli scendono
fra i subnormali o a zero. È la ragione per cui l’addestramento di DeepSeek-V3
usa scale per tessere di $1 \times 128$ elementi sulle attivazioni e per
blocchi di $128 \times 128$ sui pesi, e con quelle adotta E4M3 su tutti i
tensori, gradienti compresi {cite}`liu2024deepseekv3`.

Il vantaggio pratico è dell’hardware: sui tensor core che lo supportano un
prodotto fra matrici in FP8 ha un picco doppio di quello a 16 bit, con
l’accumulo dichiarato in precisione più alta. Ma sulle H800 gli stessi
autori misurano che il tensor core ne conserva circa 14 bit, e riversano le
somme parziali in FP32 ogni 128 elementi.

`````

I due modi di spendere gli otto bit si chiamano con i bit che danno alla virgola
e alle cifre: E4M3 ne dà quattro all’esponente, cioè alla virgola, e tre alla
mantissa, cioè alle cifre; E5M2 cinque e due.

Il confronto si fa sulla matrice `W` e sugli ingressi `x` della prima misura,
in due versioni: la matrice così com'è, e la stessa con un peso su cento
moltiplicato per venti. Per ciascuna si misurano il rapporto fra il peso più
grande e la deviazione standard, e l'errore sull'uscita con l'intero a una
scala sola, con l'intero a una scala ogni 64 pesi, e con i due FP8 a una scala
sola.

```python
def in_fp8(w, formato):
    s = w.abs().max() / torch.finfo(formato).max  # la scala porta il massimo al tetto
    return (w / s).to(formato).to(torch.float32) * s

caso = torch.Generator().manual_seed(1)           # un sorteggio a parte, per non
pochi = torch.rand(W.shape, generator=caso) < 0.01  # spostare quelli delle altre pagine
con_grandi = W.clone()
con_grandi[pochi] *= 20                           # un peso su cento, venti volte più grande

print(f"{'pesi':<14}{'max/sigma':>10}{'int8':>8}{'int8/64':>9}{'e4m3':>8}{'e5m2':>8}")
for nome, pesi in (("tutti simili", W), ("pochi grandi", con_grandi)):
    esatto = pesi @ x
    err = lambda q: ((q @ x - esatto).norm() / esatto.norm() * 100).item()
    print(f"{nome:<14}{(pesi.abs().max() / pesi.std()).item():10.1f}"
          f"{err(quantizza(pesi, 8)):7.2f}%{err(quantizza(pesi, 8, 64)):8.2f}%"
          f"{err(in_fp8(pesi, torch.float8_e4m3fn)):7.2f}%"
          f"{err(in_fp8(pesi, torch.float8_e5m2)):7.2f}%")
```

```text
pesi           max/sigma    int8  int8/64    e4m3    e5m2
tutti simili         4.6   1.04%    0.60%   2.66%   5.22%
pochi grandi        34.5   7.83%    1.55%   2.50%   5.25%
```

Sui pesi così come sono vince l’intero, 1,04% contro 2,66%, e l’1,04% è lo
stesso numero della prima tabella. Con il massimo a 4,6 deviazioni standard,
infatti, il passo comune è fitto, e sbaglia meno delle tre cifre fisse
dell’E4M3.

Se un peso su cento è venti volte più grande, il massimo sale a 34,5
deviazioni standard, sette volte e mezzo più di prima, e l’errore dell’intero a
una scala sola sale con lui, al 7,83%. I due FP8 restano dove erano (2,50% e
5,25%), perché il loro errore è una frazione di ciascun numero e non dipende da
chi altro c’è nel gruppo.

L’intero con una scala ogni 64 pesi resta però il più preciso di tutti (1,55%),
perché i pesi grandi allargano il passo soltanto del loro gruppo. Anche qui,
quindi, [decide chi condivide il passo più che il formato]{.pt-matita}. La
virgola mobile batte l’intero a una scala sola soltanto quando il peso più
grande sta molto sopra tutti gli altri, come nella seconda riga, e il suo
vantaggio vero è un altro: sulle GPU che la supportano un prodotto in FP8 può
andare fino al doppio della velocità di uno a sedici bit.

## Arrotondare dopo, o saperlo già durante

C’è un’ultima distinzione, che riguarda il momento in cui si interviene.

`````{tab} Elementare

Quasi tutti i metodi visti finora si applicano a modello già addestrato: si
prende una rete che esiste, si arrotondano i suoi numeri, si misura quanto si è
perso. È il modo economico, si fa in minuti, e per otto bit basta quasi sempre.
Fa eccezione l’FP8, che si usa anche mentre la rete impara.

L’altro modo è dire alla rete, mentre impara, che alla fine i suoi pesi
verranno arrotondati: come un negozio che sa già che la cassa accetta soltanto
i mezzi euro, e i prezzi li sceglie di conseguenza invece di lasciarli a due e
novantasette. La rete tiene i numeri precisi da una parte e fa i conti con
quelli arrotondati, e così si accorge di quando un peso sta in bilico fra due
gradini e lo sposta dove l’arrotondamento gli fa meno male.

Serve però un altro addestramento, intero o almeno un lungo ripasso della rete
già addestrata, e per questo si fa solo quando si scende in basso coi bit e
arrotondare a cose fatte non regge.

C’è una difficoltà graziosa: arrotondare è un’operazione a gradini, e una
funzione a gradini è piatta dappertutto tranne che nei salti. Una rete
impara seguendo la pendenza, e sul piano di un gradino non c’è nessuna pendenza
da seguire, quindi il segnale d’apprendimento morirebbe subito.

Il rimedio è una piccola finzione: si fanno i conti in avanti con i valori
arrotondati e all’indietro si fa finta che l’arrotondamento non ci sia. Fanno
eccezione i numeri finiti fuori dalla scala, ai quali il segnale non arriva
proprio. Non è matematicamente pulito, ma se la finzione è scelta bene
funziona.

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

È un gradiente sbagliato per costruzione, e a lungo la sua giustificazione è
stata solo empirica. Per una classe semplice di reti esiste però una
dimostrazione. In particolare, su una rete a due strati lineari con attivazione
ReLU binarizzata e ingressi gaussiani, se lo stimatore è scelto bene il
gradiente che produce (il *coarse gradient*) correla in media positivamente con
il gradiente della perdita di popolazione, il suo opposto è una direzione di
discesa e l’algoritmo converge a un punto critico; uno stimatore scelto male
rende l’addestramento instabile vicino a certi minimi locali
{cite}`yin2019understanding`.

Il costo di QAT è un altro addestramento, completo o come rifinitura di una
rete già addestrata, e la regola pratica è la solita: si usa PTQ, si misura, e
si passa a QAT solo quando la misura dice che non basta.

`````

Messi insieme, questi risultati mostrano che il numero di bit fissa l’ordine di
grandezza dell’errore, perché ogni bit in meno lo raddoppia circa, e da otto a
quattro bit lo moltiplica per diciotto.

A parità di bit, però, il modo in cui si sceglie e si condivide il passo lo
sposta quanto parecchi bit. Una scala ogni sessantaquattro pesi
porta il 18,7% al 10,8%, e un tetto scelto bene all’11,4%; togliere dal gruppo
tre numeri su cinquecentododici porta il 7,28% allo 0,20%, quanto cinque bit in
più; e a parità di otto bit, scrivere ciascun numero con la propria virgola
porta il 7,83% al 2,50% quando qualche peso è grande. In nessuno di questi casi
è cambiato il numero di bit.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Arrotondare i pesi funziona per due ragioni: l’errore su ciascun numero è al
  massimo metà passo, e sommando tanti numeri gli errori vanno in su e in
  giù e si compensano. Una rete somma sempre tanti numeri insieme, e questa
  è la sua fortuna.
- Ogni bit in meno raddoppia circa l’errore. A parità di bit, però, conta
  moltissimo il passo, e di solito a deciderlo è il numero più grande del
  gruppo che lo condivide: una scala ogni sessantaquattro pesi taglia l’errore
  di un fattore 1,74 rispetto a una scala sola per tutta la matrice, lo stesso a
  otto bit come a tre, e schiacciare sull’ultimo gradino i pochi pesi più
  grandi fa quasi altrettanto.
- A otto bit l’uscita di uno strato si sposta di circa l’uno per cento; a
  quattro bit, con una scala sola, di quasi il venti. Quanto questo costi in
  accuratezza lo dice soltanto la prova sul modello, e chi dice che i modelli
  girano a quattro bit sta parlando di metodi che fanno più che arrotondare
  con una scala sola.
- Nei modelli linguistici grandi poche componenti delle attivazioni valgono
  molte volte le altre e rovinano il passo per tutti. Tenendo intere tre
  componenti su cinquecentododici l’errore passa dal 7,28% allo 0,20%.
- Con otto bit si può anche tenere la virgola: poche cifre fisse e una virgola
  che si sposta, come quando si dice una misura. L’errore non supera mai un
  sedicesimo del numero, purché il numero stia fra un sessantaquattresimo e 448
  volte l’unità scelta. Sui pesi ordinati l’intero sbaglia meno; la virgola
  batte l’intero a scala sola quando pochi pesi sono molto più grandi degli
  altri; e una scala ogni sessantaquattro pesi batte tutti e due.
- Si può arrotondare a modello finito (economico, e per otto bit basta quasi
  sempre) oppure dirlo alla rete mentre impara, così i pesi si spostano dove
  l’arrotondamento fa meno male (ma serve un altro addestramento, intero o
  almeno un lungo ripasso).
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Quantizzazione simmetrica a $b$ bit: $\hat{w} = s\,\mathrm{round}(w/s)$ con
  $s = \max|w| / (2^{b-1}-1)$. L’errore per elemento è limitato da $s/2$ e,
  finché il passo non supera un paio di deviazioni standard dei pesi, è
  uniforme di varianza $s^2/12$;
  sull’uscita di un prodotto scalare gli errori indipendenti crescono come
  $\sqrt{n}$. Che l’errore relativo non si accumuli richiede in più che
  cresca così anche il segnale; è un’ipotesi sugli ingressi, non sugli
  arrotondamenti.
- La scala è la leva più economica, in due modi: la granularità (per tensore,
  per riga, per gruppo di $g$ elementi) e il tetto da cui la si prende, che a
  pochi bit conviene tenere sotto il massimo {cite}`banner2019post`. Sul
  prodotto $256 \times 512$ per $512 \times 64$, a quattro bit: 18,71% con una
  scala per tutto presa dal massimo, 11,42% con il tetto ottimo, 10,77% con una
  scala ogni 64.
- Le caratteristiche anomale dei Transformer {cite}`dettmers2022llmint8`
  dettano la scala e schiacciano tutto il resto. La decomposizione a precisione
  mista le tiene fuori (7,28% contro 0,20% sullo stesso prodotto); SmoothQuant
  {cite}`xiao2023smoothquant` le sposta nei pesi riscalando i canali, una
  rotazione ortogonale le ridistribuisce, e in tutti e due i casi il prodotto
  non cambia.
- L’FP8 {cite}`micikevicius2022fp8` (E4M3, massimo 448; E5M2, massimo
  $57\,344$) limita nel campo normale l’errore relativo, $2^{-(M+1)}$ con $M$ i
  bit di mantissa, dove l’`int8` limita quello assoluto; a scala unica l’`int8`
  vince finché $\max|w|/\sigma_w \lesssim 11{,}5$ (sui pesi della prova 1,04%
  contro 2,66%), e perde con pochi valori anomali grandi (7,83% contro 2,50%),
  ma una scala ogni 64 pesi lo riporta davanti (1,55%). L’FP8 vuole una scala
  per tensore o per blocco, e il suo vantaggio è il picco doppio dei tensor
  core che lo eseguono.
- Sotto gli otto bit servono metodi che non trattino l’arrotondamento come
  locale: GPTQ {cite}`frantar2023gptq` compensa sull’uscita l’errore già
  commesso, AWQ {cite}`lin2024awq` protegge i canali che moltiplicano le
  attivazioni grandi.
- PTQ contro QAT: la seconda mette la quantizzazione nel passaggio in
  avanti durante l’addestramento e aggira la derivata nulla di
  $\mathrm{round}$ con lo stimatore diretto
  {cite}`bengio2013estimating`, cioè un gradiente deliberatamente sbagliato che,
  scelto bene, su reti semplici ha garanzie di convergenza
  {cite}`yin2019understanding`.
```

`````

Questa leva lascia intatta l’architettura: stessi collegamenti, stessa forma,
numeri scritti più corti. La leva che segue fa il contrario e tocca i
collegamenti: ne toglie una parte e lascia gli altri dove sono.
