# Riconoscere i suoni: classificazione e tagging

Chiudi gli occhi in una stanza e prova a nominare quello che senti: il ronzio
del frigorifero, un'auto che passa, il tuo stesso respiro. Il cervello lo fa
di continuo, in sottofondo, senza fatica, ed è un lavoro sorprendentemente
difficile da imitare. Nella prima sezione di questo capitolo abbiamo imparato
a trasformare un suono in immagine: lo **spettrogramma**, la «parola visibile»
di Potter, con il tempo sull'asse orizzontale, la frequenza su quello
verticale e l'intensità resa dal colore (lo abbiamo costruito passo per passo
in [Dal suono alle feature](dal-suono-alle-feature.md)). Fatto quel passaggio,
la domanda «che suono è questo?» smette di essere un problema di *audio* e
diventa un problema di *visione*. Un abbaiare, un vetro che si rompe, una
corda di chitarra pizzicata: ognuno lascia sullo spettrogramma una firma
diversa (bande, righe verticali, macchie) e riconoscere quella firma è
esattamente ciò che una rete addestrata sulle immagini sa fare bene.

Non è un esercizio di scuola. Sistemi di questo tipo ascoltano le foreste per
sentire il rumore di una motosega dove non dovrebbe esserci, riconoscono il
canto di un uccello dal telefono di un escursionista, avvisano quando in casa
si rompe un vetro. È l'audio *oltre* la voce: non più «cosa hai detto», ma
«cosa sta suonando».

## Dallo spettrogramma alla classe

Il punto di partenza è l'estrazione delle feature vista nella prima sezione:
dall'onda grezza allo **spettrogramma mel**, la versione compressa e
«a misura d'orecchio» dello spettro che abbiamo descritto in
[Dal suono alle feature](dal-suono-alle-feature.md). Una volta che
il suono è quella tabella tempo–frequenza, classificarlo è un compito da rete
convoluzionale, la stessa macchina che nel capitolo di visione riconosceva un
gatto in una foto.

`````{tab} Elementare

Pensa allo spettrogramma come a una **radiografia del suono**: una lastra dove
ogni rumore lascia una sagoma riconoscibile. Un fischio è una riga sottile e
netta che sale; una vocale è fatta di bande orizzontali parallele; un vetro che
si rompe è uno schizzo verticale improvviso, pieno di frequenze alte tutte
insieme. Il medico impara a leggere le lastre a forza di vederne; una rete
neurale fa lo stesso, mostrandole migliaia di spettrogrammi già etichettati
finché non impara a collegare la sagoma al nome del suono. La cosa
sorprendente è che non serve inventare un metodo nuovo: è lo *stesso* tipo di
rete che riconosce i gatti nelle foto, perché ormai il suono, per lei, *è* una
foto.

`````

`````{tab} Superiore

Lo spettrogramma mel è una matrice $\mathbf{S} \in \mathbb{R}^{F \times T}$: $F$ bande
di frequenza (tipicamente 64 o 128) per $T$ finestre temporali. La trattiamo
come un'**immagine a un solo canale** (l'analogo di una foto in scala di
grigi) e la diamo in pasto a una CNN 2D, con i filtri convoluzionali che
scorrono contemporaneamente sull'asse del tempo e su quello della frequenza. È
esattamente la pipeline convoluzione + non linearità + pooling della
[classificazione di
immagini](../VisioneArtificiale/classificazione-transfer.md), con un'unica
differenza concettuale: qui i due assi non sono omogenei (uno è il tempo,
l'altro la frequenza). La conseguenza però non riguarda la **località**, che non
è mai stata in discussione (una firma sonora è un motivo locale nel piano
tempo–frequenza esattamente come un occhio lo è in una foto): riguarda la
**condivisione dei pesi**, cioè l'equivarianza per traslazione. Applicare gli
stessi filtri a ogni banda equivale ad assumere che traslare un motivo non ne
cambi la classe. Lungo il tempo è una simmetria vera, un latrato è un latrato
mezzo secondo dopo; lungo la frequenza no, perché su una scala quasi
logaritmica traslare in su è **trasporre**, e la trasposizione cambia la vocale,
la nota, lo strumento. Funziona lo stesso perché i motivi utili restano locali,
ma è un bias solo approssimato: da qui la pratica di non fare pooling globale
sull'asse delle frequenze, e la scelta dell'AST (poche righe più sotto) di dare
a ogni patch un embedding della sua posizione *in frequenza*, che sarebbe
superfluo se quell'asse fosse davvero simmetrico. Anche il
**transfer learning** si trasporta di peso: si parte spesso da una rete
pre-addestrata su ImageNet e si rifinisce sugli spettrogrammi, replicando il
canale grigio sui tre canali RGB attesi in ingresso.

`````

C'è però una scelta che l'audio impone più della visione. Chiedersi «che
strumento sta suonando?» presuppone che la risposta sia *una*: pianoforte
*oppure* chitarra *oppure* violino. Ma una clip di dieci secondi di strada
cittadina contiene, tutte insieme, il traffico *e* una voce *e* un clacson *e*
il vento. Sono due problemi diversi.

`````{tab} Elementare

Immagina due tipi di domanda. La prima è a crocetta unica: «di questi tre
strumenti, quale senti?»; la risposta è una sola, e le probabilità dei
candidati si fanno concorrenza, se sale una scende un'altra. La seconda è una
lista della spesa: «segna *tutti* i suoni presenti in questa registrazione», e
qui possono essere veri contemporaneamente il traffico, una voce e un cane,
senza togliersi spazio a vicenda. Il primo caso si chiama classificazione a
**etichetta singola**, il secondo **tagging** multi-etichetta. E c'è un terzo
livello, ancora più fine: non solo *quali* suoni, ma *quando* ciascuno inizia
e finisce, come sottotitolare i rumori di un film. Questo si chiama
rilevamento degli eventi sonori.

`````

`````{tab} Superiore

Nella classificazione a **etichetta singola** le classi sono mutuamente
esclusive: si usa una **softmax** sulle $K$ classi e la cross-entropia, come in
visione. La softmax normalizza a somma 1, imponendo la competizione tra le
alternative.

Nel **tagging multi-etichetta** ogni classe è invece una domanda sì/no
indipendente. Si sostituisce la softmax con una **sigmoide** su ciascuna delle
$K$ uscite e si addestra con la **binary cross-entropy** sommata sulle classi:

$$
\hat{y}_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}},
\qquad
\mathcal{L} = -\sum_{k=1}^{K}\Big[\,y_k \log \hat{y}_k + (1-y_k)\log(1-\hat{y}_k)\,\Big],
$$

dove $z_k$ è il logit della classe $k$, $\hat{y}_k \in (0,1)$ la probabilità
*indipendente* che quel suono sia presente e $y_k \in \{0,1\}$ l'etichetta
vera. Nessun vincolo di somma: più classi possono essere «accese» insieme. Il
**rilevamento degli eventi sonori** (*sound event detection*, il cuore delle
sfide DCASE) spinge oltre, chiedendo una predizione per ogni istante (un
tagging *frame per frame* con i confini temporali di ogni evento) e si valuta
con metriche che confrontano gli intervalli predetti con quelli veri.

`````

## I dati: AudioSet

Un modello vale quanto i dati su cui impara, e per i suoni del mondo il
riferimento è **AudioSet**, pubblicato da Google nel 2017
{cite}`gemmeke2017audioset`. La sua scala è ciò che ne cambia le regole.

`````{tab} Elementare

Prima di AudioSet, chi voleva addestrare un classificatore di suoni
raccoglieva qualche migliaio di clip a mano: poche, costose, tutte dello
stesso tipo. AudioSet ribalta la scala: oltre **due milioni** di frammenti da
dieci secondi, ritagliati da video di YouTube, ciascuno con un'etichetta che
dice *quali* suoni contiene (pescati da un catalogo di centinaia di categorie,
dal miagolio al motore diesel al rumore della pioggia). Il rovescio della
medaglia è che le etichette sono «alla buona»: dicono che in quei dieci
secondi *c'è* un cane, ma non in quale secondo abbaia. Poche certezze precise,
ma tantissimi esempi: è un baratto che, con le reti profonde, conviene quasi
sempre.

`````

`````{tab} Superiore

AudioSet conta $2\,084\,320$ clip da 10 secondi e un'**ontologia** di 632
categorie sonore organizzate a gerarchia; di queste, 527 formano il benchmark
di classificazione standard su cui si confrontano i modelli. Le etichette sono
**deboli** (*weak labels*): indicano la presenza di un suono nella clip, senza
localizzazione temporale, ed essendo multi-etichetta si prestano naturalmente
al setup sigmoide + BCE visto sopra. La metrica di riferimento non è
l'accuratezza (inadatta a un problema multi-etichetta e sbilanciato) ma la
**mean Average Precision** (mAP), la media, sulle classi, dell'area sotto la
curva precisione–richiamo. Un dataset grande e debolmente etichettato sposta
il collo di bottiglia: non più «troppi pochi dati», ma «etichette rumorose e
code lunghe di classi rare», un regime in cui contano di più la capienza del
modello e il pre-addestramento della precisione di ogni singola annotazione.

`````

## Quando l'attenzione arriva all'audio: l'AST

Fino al 2021 la mappa era chiara: gli spettrogrammi si classificano con le CNN,
punto. Poi è arrivato l'**Audio Spectrogram Transformer** (AST) di Gong, Chung
e Glass {cite}`gong2021ast`, e ha tolto le convoluzioni dall'equazione.

`````{tab} Elementare

Ricordi il trucco con cui il Transformer ha imparato a guardare le foto? Si
taglia l'immagine in tante tessere quadrate, si mettono in fila come le parole
di una frase, e il modello impara a «guardare» le tessere lontane che contano:
lo abbiamo visto con il [Vision
Transformer](../Transformers/multimodalita.md). L'AST fa esattamente la stessa
cosa, ma sulla radiografia del suono: taglia lo spettrogramma in tessere, le
mette in fila e lascia che l'attenzione colleghi, per esempio, un colpo secco
all'inizio con la sua eco un istante dopo, anche se sulla lastra sono lontani.
Niente filtri che scorrono: solo tessere che si guardano tra loro.

`````

`````{tab} Superiore

L'AST applica un Transformer in stile ViT direttamente allo spettrogramma
log-mel, senza alcuna convoluzione: è il primo modello di classificazione
audio puramente attentivo. Lo spettrogramma viene suddiviso in **patch**
$16 \times 16$ (parzialmente sovrapposte), ciascuna proiettata linearmente in
un embedding e trattata come un token, con un *positional embedding* per la
posizione tempo–frequenza; da lì in poi è il consueto stack di
*self-attention* del capitolo sui Transformer. Il vantaggio è il campo
recettivo globale fin dal primo strato: ogni patch può pesare qualunque altra,
mentre una CNN allarga la propria vista solo strato dopo strato. Sul benchmark
AudioSet completo l'AST raggiunge una mAP di $0{,}459$ con un singolo modello
($0{,}485$ in ensemble), superando le migliori CNN dell'epoca a parità di
configurazione.

Onestà d'obbligo, la stessa del capitolo sui Transformer: rinunciare alla
convoluzione significa rinunciare al suo *bias induttivo* di località, e quel
bias andava «gratis». Senza, servono molti più dati, oppure, come fa l'AST, il
**transfer** dei pesi di un ViT pre-addestrato su ImageNet, adattando gli
embedding di patch e di posizione dallo spazio delle immagini a quello degli
spettrogrammi. Un Transformer audio addestrato da zero su pochi dati resta
dietro a una CNN: l'attenzione paga quando i dati (o il pre-addestramento)
abbondano.

`````

## Un classificatore di suoni in miniatura

I modelli di questa sezione girano su GPU e su milioni di clip; qui ne
costruiamo una versione tascabile che gira sul portatile, per toccare con mano
l'idea di **estrarre feature e decidere**. Non calcoleremo un vero
spettrogramma, per quello c'è la pipeline di [Dal suono alle
feature](dal-suono-alle-feature.md), ma partiremo dalla forma d'onda grezza e
ne ricaveremo due caratteristiche elementari, finestra per finestra.

`````{tab} Elementare

Due misure, semplicissime. La prima è l'**energia**: quanto è «forte» il suono
in quella finestra (grande quando l'onda oscilla ampia, quasi zero nel
silenzio). La seconda è lo **zero-crossing rate**: quante volte l'onda
attraversa lo zero, cioè passa dal positivo al negativo. Un tono basso e pieno
oscilla lentamente e attraversa lo zero *poche* volte; un sibilo o un rumore,
fatto di frequenze alte, lo attraversa *tantissime* volte: è la differenza tra
una «ooo» profonda e una «sss» sibilante. Con queste due sole misure possiamo
già distinguere tre situazioni: silenzio (poca energia), tono (energia alta,
pochi attraversamenti), rumore (energia intermedia, molto meno del tono e molto
più del silenzio, ma tantissimi attraversamenti).

`````

`````{tab} Superiore

Su una finestra di $L$ campioni $x[0], \dots, x[L-1]$ definiamo l'**energia a
breve termine** come potenza media e lo **zero-crossing rate** come frazione di
cambi di segno tra campioni adiacenti:

$$
E = \frac{1}{L}\sum_{n=0}^{L-1} x[n]^2,
\qquad
\mathrm{ZCR} = \frac{1}{2(L-1)}\sum_{n=1}^{L-1}\big|\,\mathrm{sgn}(x[n]) - \mathrm{sgn}(x[n-1])\,\big|,
$$

dove $\mathrm{sgn}(\cdot)$ è il segno del campione. L'energia distingue il
sonoro dal silenzio; lo ZCR è un indicatore grezzo del contenuto in frequenza:
alto per i suoni ricchi di alte frequenze (rumore, fricative), basso per i
toni gravi. Sono, storicamente, tra le prime feature usate per separare parti
sonore e non sonore del parlato: un antenato rudimentale delle feature
spettrali di [Dal suono alle feature](dal-suono-alle-feature.md).

`````

Generiamo un segnale sintetico in tre parti (un tono puro, del silenzio, del
rumore) e classifichiamo ogni finestra con una regoletta a soglie. Il codice è
NumPy puro e deterministico (seed fissato):

```python
import numpy as np

rng = np.random.default_rng(0)   # seed esplicito: risultato riproducibile
fs = 8000                        # frequenza di campionamento (Hz)
dur = 0.15                       # durata di ogni segmento (secondi)
n = int(fs * dur)                # campioni per segmento
t = np.arange(n) / fs

# Tre segmenti: un tono puro, del silenzio, del rumore
tono     = 1.0 * np.sin(2 * np.pi * 200 * t)        # sinusoide a 200 Hz
silenzio = 0.001 * rng.standard_normal(n)           # quasi-zero (fondo)
rumore   = 0.30 * rng.standard_normal(n)            # rumore gaussiano
segnale  = np.concatenate([tono, silenzio, rumore])

def energia(x):
    "Energia a breve termine: potenza media della finestra."
    return float(np.mean(x**2))

def zcr(x):
    "Zero-crossing rate: frazione di cambi di segno tra campioni adiacenti."
    return float(np.mean(np.abs(np.diff(np.sign(x)))) / 2)

L = 400  # lunghezza finestra: 400 campioni = 50 ms
SOGLIA_E, SOGLIA_Z = 0.01, 0.20   # soglie di decisione

def classifica(e, z):
    if e < SOGLIA_E:              # poca energia: nessun suono
        return "silenzio"
    if z > SOGLIA_Z:              # tanti cambi di segno: rumore/sibilo
        return "rumore"
    return "tono"                 # energia alta, pochi cambi: tono/voce

print(f"{'finestra':>8} | {'energia':>9} | {'zcr':>6} | classe")
print("-" * 42)
for i in range(0, len(segnale) - L + 1, L):
    finestra = segnale[i:i+L]
    e, z = energia(finestra), zcr(finestra)
    print(f"{i//L:>8} | {e:>9.4f} | {z:>6.3f} | {classifica(e, z)}")
```

L'output mostra la separazione netta delle tre situazioni:

```text
finestra |   energia |    zcr | classe
------------------------------------------
       0 |    0.5000 |  0.049 | tono
       1 |    0.5000 |  0.048 | tono
       2 |    0.5000 |  0.048 | tono
       3 |    0.0000 |  0.516 | silenzio
       4 |    0.0000 |  0.514 | silenzio
       5 |    0.0000 |  0.486 | silenzio
       6 |    0.0920 |  0.471 | rumore
       7 |    0.1001 |  0.516 | rumore
       8 |    0.0909 |  0.509 | rumore
```

Il tono ha energia alta ($0{,}5$, la potenza media di una sinusoide di
ampiezza 1) e ZCR bassissimo (una sinusoide a 200 Hz campionata a 8 kHz
attraversa lo zero solo ogni 20 campioni: più il suono è acuto, più fitti sono
quei passaggi, ed è tutto il legame fra questa misura e le frequenze). Il
silenzio ha energia praticamente
nulla, e la soglia sull'energia lo cattura *prima* di guardare lo ZCR, che nel
fondo casuale è persino alto, ma non conta più. Il rumore ha energia
intermedia e ZCR elevato. Due numeri per finestra e tre soglie: nessuna rete,
eppure la logica è la stessa dei modelli grandi; *estrarre feature che
separano le classi, poi decidere*. La differenza è che una CNN o un AST
imparano le feature migliori da soli, invece di riceverle scritte a mano.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Una volta che il suono è diventato una **radiografia** (lo spettrogramma),
  riconoscerlo è un problema di **immagini**: la stessa rete che distingue un
  gatto da un cane in una foto distingue un vetro rotto da un clacson in una
  lastra sonora, e può perfino partire da quello che ha già imparato sulle
  foto.
- Ci sono due domande diverse, e non vanno confuse. «Quale di questi suoni è?»
  è a **crocetta unica**, e le risposte si fanno concorrenza. «Quali suoni ci
  sono qui dentro?» è una **lista della spesa**, e possono essere veri tutti
  insieme. Una terza, più difficile, chiede anche *quando* ciascuno comincia e
  finisce.
- **AudioSet** {cite}`gemmeke2017audioset` cambia le regole con la scala: due
  milioni di frammenti da dieci secondi, etichette «alla buona» (dicono che il
  cane c'è, non in quale secondo abbaia). Tanti esempi imprecisi battono pochi
  esempi perfetti, quando la rete è grande.
- Le **tessere** funzionano anche sul suono: si taglia la lastra in quadretti,
  li si mette in fila e si lascia che ciascuno guardi tutti gli altri, anche
  quelli lontani. Costa molti più dati che i filtri che scorrono, e per questo
  di solito si parte da una rete già addestrata sulle immagini.
- Due misure semplicissime (quanto è **forte** il suono, e quante volte l'onda
  **attraversa lo zero**) bastano a separare a mano silenzio, tono e rumore. È
  la stessa idea dei modelli grandi, che però quelle misure se le scelgono da
  soli.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Trasformato il suono in **spettrogramma mel**, riconoscerlo diventa un
  problema di **visione**: una CNN 2D lo tratta come un'immagine a un canale, e
  il transfer learning da ImageNet si trasporta di peso. Attenzione però al
  bias: sull'asse delle frequenze la condivisione dei pesi assume un'invarianza
  per traslazione che i dati non hanno.
- **Etichetta singola** (una sola classe, softmax + cross-entropia) e
  **tagging multi-etichetta** (più suoni insieme, sigmoide + BCE) sono problemi
  diversi; il **rilevamento di eventi sonori** aggiunge il *quando*.
- **AudioSet** {cite}`gemmeke2017audioset` (oltre 2 milioni di clip da 10 s,
  527 classi di benchmark, etichette *deboli*) cambia le regole: contano scala
  e pre-addestramento più della precisione della singola annotazione (metrica:
  mAP).
- L'**Audio Spectrogram Transformer** {cite}`gong2021ast` applica un
  Transformer in stile ViT alle patch dello spettrogramma, senza convoluzioni;
  in cambio del campo recettivo globale, chiede molti dati o il transfer da
  ImageNet.
- Estrarre due feature semplici (**energia** e **zero-crossing rate**) basta
  per separare a mano silenzio, tono e rumore: la stessa idea dei modelli
  grandi, che però le feature se le imparano da soli.
```

`````
