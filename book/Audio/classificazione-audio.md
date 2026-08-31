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

Il punto di partenza è quello costruito nella prima sezione, [Dal suono alle
feature](dal-suono-alle-feature.md): l'onda diventa una tabella con il tempo su
un asse e le frequenze sull'altro, riletta come la sente un orecchio (lo
**spettrogramma mel**). Da lì in avanti il suono diventa un'immagine. E
riconoscere un'immagine è il mestiere delle **reti convoluzionali**, in sigla
CNN: la stessa macchina che nel {doc}`capitolo di visione
</VisioneArtificiale/overview>` riconosceva un gatto in una foto.

`````{tab} Elementare

Lo spettrogramma è una **radiografia del suono**: una lastra dove ogni rumore
lascia una sagoma riconoscibile. Un fischio è una riga sottile e netta che
sale; una vocale è fatta di bande orizzontali parallele; un vetro che
si rompe è uno schizzo verticale improvviso, pieno di frequenze alte tutte
insieme. Il medico impara a leggere le lastre a forza di vederne; una rete
neurale fa lo stesso, mostrandole migliaia di spettrogrammi già etichettati
finché non impara a collegare la sagoma al nome del suono. La cosa
sorprendente è che non serve inventare un metodo nuovo: è lo *stesso* tipo di
rete che riconosce i gatti nelle foto, perché ormai il suono, per lei, *è* una
foto.

Una cosa però il medico la sa, e la rete no: sulla lastra spostarsi a destra e
spostarsi in alto non sono la stessa cosa. La stessa macchia più a destra è lo
stesso latrato mezzo secondo dopo; più in alto è un suono più acuto, cioè
un'altra vocale, un'altra nota, un altro strumento. La rete invece cerca le
sagome dappertutto allo stesso modo, in basso come in alto: se la cava lo
stesso, perché le sagome che contano sono piccole, ma quell'idea, presa alla
lettera, è falsa.

E c'è un regalo in più, che viene proprio da quella somiglianza con le foto. Una
rete che ha già passato mesi a guardare fotografie ha imparato a riconoscere
bordi, macchie, righe, motivi che si ripetono: roba che sulla lastra sonora c'è
eccome. Allora non si riparte da zero: le si fanno vedere spettrogrammi finché
non si riabitua, e in poco tempo diventa brava anche lì.

`````

`````{tab} Superiore

Lo spettrogramma mel è una matrice $\mathbf{S} \in \mathbb{R}^{F \times T}$: $F$ bande
di frequenza (tipicamente 64 o 128) per $T$ finestre temporali. La trattiamo
come un’**immagine a un solo canale** (l'analogo di una foto in scala di
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
ma è un bias solo approssimato, e va tenuto a mente quando si legge una CNN su
spettrogrammi come se fosse una CNN su fotografie. Anche il
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

«Di questi tre strumenti, quale senti?» è una domanda a crocetta unica: la
risposta è una sola, e le probabilità dei candidati si fanno concorrenza, se
sale una scende un’altra. Poi c'è la lista della spesa: «segna *tutti* i suoni
presenti in questa registrazione», e qui possono essere veri contemporaneamente
il traffico, una voce e un cane, senza togliersi spazio a vicenda. La crocetta
si chiama classificazione a **etichetta singola**, la lista della spesa
**tagging** multi-etichetta. E si può chiedere ancora di più: non solo *quali*
suoni, ma *quando* ciascuno inizia e finisce, come sottotitolare i rumori di un
film. Questo si chiama rilevamento degli eventi sonori.

`````

`````{tab} Superiore

Nella classificazione a **etichetta singola** le classi sono mutuamente
esclusive: si usa una **softmax** sulle $C$ classi (le stesse $C$ classi
dell'apertura del capitolo) e la cross-entropia, come in visione. La softmax
normalizza a somma 1, imponendo la competizione tra le alternative.

Nel **tagging multi-etichetta** ogni classe è invece una domanda sì/no
indipendente. Si sostituisce la softmax con una **sigmoide** su ciascuna delle
$C$ uscite e si addestra con la **binary cross-entropy** sommata sulle classi:

$$
\hat{y}_c = \sigma(z_c) = \frac{1}{1 + e^{-z_c}},
\qquad
\mathcal{L} = -\sum_{c=1}^{C}\Big[\,y_c \log \hat{y}_c + (1-y_c)\log(1-\hat{y}_c)\,\Big],
$$

dove $z_c$ è il logit della classe $c$, $\hat{y}_c \in (0,1)$ la probabilità
*indipendente* che quel suono sia presente e $y_c \in \{0,1\}$ l'etichetta
vera. Nessun vincolo di somma: più classi possono essere «accese» insieme. Il
**rilevamento degli eventi sonori** (*sound event detection*, il cuore delle
sfide DCASE, la gara annuale sul rilevamento e la classificazione di scene ed
eventi acustici) spinge oltre, chiedendo una predizione per ogni istante (un
tagging *frame per frame* con i confini temporali di ogni evento) e si valuta
con metriche che confrontano gli intervalli predetti con quelli veri.

`````

## I dati: AudioSet

Un modello vale quanto i dati su cui impara, e per i suoni del mondo il
riferimento è **AudioSet**, pubblicato da Google nel 2017
{cite}`gemmeke2017audioset`. Quello che lo rende diverso da tutto ciò che
c'era prima è quanto è grosso, più che come è fatto.

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

Il catalogo però non si riempie in modo uniforme. La musica e il parlato
compaiono ovunque; il verso di un uccello raro sta in un centinaio di frammenti
su due milioni. E il modello lo si giudica categoria per categoria, facendo poi
la media dei voti, quindi quelle caselle quasi vuote pesano quanto la musica.

`````

`````{tab} Superiore

L'articolo che presenta AudioSet {cite}`gemmeke2017audioset` descrive
un’**ontologia** di 632 categorie sonore organizzate a gerarchia e una prima
raccolta di $1\,789\,621$ segmenti da 10 secondi. La raccolta pubblicata è poi
cresciuta oltre l'articolo, e la versione che si scarica oggi conta
$2\,084\,320$ clip su 527 categorie: sono queste ultime a formare il benchmark
di classificazione standard su cui si confrontano i modelli, e quando si citano
quei due numeri la fonte è la pagina del dataset, non il paper. Le etichette sono
**deboli** (*weak labels*): indicano la presenza di un suono nella clip, senza
localizzazione temporale, ed essendo multi-etichetta si prestano naturalmente
alla coppia sigmoide + BCE. La metrica di riferimento non è
l'accuratezza (inadatta a un problema multi-etichetta e sbilanciato) ma la
**mean Average Precision** (mAP), che per ogni classe fa la media delle
precisioni raggiunte a ciascuna soglia, pesandole con l'aumento di richiamo che
quella soglia porta, e poi media sulle classi. Si massimizza. Un dataset grande
e debolmente etichettato sposta il collo di bottiglia: non più «troppi pochi
dati», ma «etichette rumorose e
code lunghe di classi rare», un regime in cui contano di più la capienza del
modello e il pre-addestramento della precisione di ogni singola annotazione.

`````

## Quando l'attenzione arriva all'audio: l'AST

Fino al 2021 la mappa era chiara: gli spettrogrammi si classificano con una
rete convoluzionale, al più con un po’ di attenzione appiccicata sopra
all'ultimo strato. Poi è arrivato l’**Audio Spectrogram Transformer** (AST)
di Gong, Chung e Glass {cite}`gong2021ast`, e ha buttato via proprio quelle. Le
convoluzioni sono i **filtri che scorrono**: piccole griglie di numeri, larghe
pochi quadretti, che passano sull'immagine un pezzetto alla volta cercando
sempre la stessa cosa (un bordo, una macchia, una riga). Erano il pezzo di
macchina attorno a cui tutto il resto era costruito
({numref}`fig-ast-tessere`).

```{figure} ../figures/ast-tessere.svg
:name: fig-ast-tessere
:alt: A sinistra uno spettrogramma stilizzato con un colpo secco, una banda verticale marcata, e la sua eco più tenue più avanti nel tempo, tagliato in tessere quadrate da una griglia. A destra le stesse tessere in fila come parole, con un arco che collega la tessera del colpo a quella dell'eco, lontane nella fila.
:width: 100%

Lo spettrogramma tagliato in tessere, e le tessere messe in fila: il colpo e
la sua eco, lontani sulla lastra, si guardano direttamente.
```

`````{tab} Elementare

Torna il trucco con cui il Transformer ha imparato a guardare le foto. Si taglia
l'immagine in tante tessere quadrate, le si mette in fila come le parole di una
frase, e poi ogni tessera guarda tutte le altre e decide quali le interessano.
Quel «guardare le altre e scegliere» si chiama **attenzione**, e lo abbiamo
incontrato con il [Vision
Transformer](../Transformers/multimodalita.md).

L'AST fa la stessa identica cosa sulla radiografia del suono: taglia lo
spettrogramma in tessere, le mette in fila e lascia che ciascuna guardi le
altre, collegando per esempio un colpo secco all'inizio con la sua eco un
istante dopo, anche se sulla lastra sono lontani. Niente filtri che scorrono:
solo tessere che si guardano tra loro.

Non è gratis, però, ed è la parte che di solito non si racconta. I filtri che
scorrono portavano con sé un'idea già pronta (quello che conta sta vicino a
quello che gli sta accanto), e a un modello che riceve un'idea in regalo bastano
meno esempi per imparare. Le tessere quell'idea non ce l'hanno e se la devono
costruire dai dati, quindi di esempi ne chiedono molti di più: addestrato da
zero su poca roba, un Transformer audio resta dietro a una rete convoluzionale.
La scappatoia è quella di prima: partire da una rete che ha già guardato milioni
di **fotografie** e riadattarla agli spettrogrammi.

`````

`````{tab} Superiore

L'AST applica un Transformer in stile ViT direttamente allo spettrogramma
log-mel, senza alcuna convoluzione, e si presenta come il primo modello di
classificazione audio puramente attentivo. Lo spettrogramma viene suddiviso in
**patch** $16 \times 16$ (parzialmente sovrapposte), ciascuna proiettata
linearmente in un embedding e trattata come un token, con un *positional
embedding* per la posizione tempo–frequenza; da lì in poi è il consueto stack
di *self-attention* del {doc}`capitolo sui Transformer
</Transformers/overview>`. Il vantaggio è il campo recettivo globale fin dal
primo strato: ogni patch può pesare qualunque altra, mentre una CNN allarga la
propria vista solo strato dopo strato. Sul benchmark AudioSet completo l'AST
raggiunge una mAP di $0{,}459$ con un singolo modello ($0{,}485$ nell'ensemble
più grande), contro lo $0{,}444$ del miglior ibrido CNN più attenzione
dell'epoca a parità di protocollo.

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

I modelli di questa sezione girano su schede grafiche potenti (le **GPU**) e su
milioni di clip; qui ne costruiamo una versione tascabile che gira sul
portatile, per toccare con mano l'idea di **estrarre feature e decidere**. Non
calcoleremo un vero
spettrogramma, per quello c'è la pipeline di [Dal suono alle
feature](dal-suono-alle-feature.md), ma partiremo dalla forma d'onda grezza e
ne ricaveremo due caratteristiche elementari, finestra per finestra.

`````{tab} Elementare

Il suono non si guarda mai tutto insieme. Lo si taglia a fettine di qualche
centesimo di secondo e si misura dentro ciascuna: una fettina si chiama
**finestra**, e le misure si rifanno da capo per ognuna.

Le misure sono due, semplicissime. La prima è l’**energia**: quanto è «forte» il
suono in quella finestra (grande quando l'onda oscilla ampia, quasi zero nel
silenzio). La seconda è quanto spesso l'onda attraversa lo zero, cioè passa dal
positivo al negativo: si chiama **zero-crossing rate**, in sigla `zcr`. Non è un
conteggio ma una frazione, e per questo esce sempre fra 0 e 1: vale $0{,}5$ se
metà delle coppie di campioni vicini
cambia segno, quasi $0$ se non cambia quasi mai. Un tono basso e pieno
oscilla lentamente e attraversa lo zero *poche* volte; un sibilo o un rumore,
fatto di frequenze alte, lo attraversa *tantissime* volte: è la differenza tra
una «ooo» profonda e una «sss» sibilante.

Con queste due sole misure distinguiamo già tre situazioni, purché le si guardi
in un ordine preciso. Prima l'energia: se è quasi zero c'è **silenzio**, e non
serve chiedere altro. Se invece del suono c'è, allora si guarda quante volte
l'onda attraversa lo zero: pochi attraversamenti vuol dire **tono**, tantissimi
vuol dire **rumore**.

`````

`````{tab} Superiore

Su una finestra di $L$ campioni $x[0], \dots, x[L-1]$ definiamo l’**energia a
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

Generiamo un segnale finto in tre parti (un tono puro, del silenzio, del
rumore) e classifichiamo ogni finestra con una regoletta a soglie, per vedere
come le due misure, prese in quest'ordine, separino le tre situazioni. Il
generatore di numeri casuali parte da un valore fissato in partenza, così chi
esegue il codice ottiene esattamente gli stessi numeri e non altri:

```python
import numpy as np

rng = np.random.default_rng(0)   # punto di partenza fissato: numeri sempre uguali
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

L = 400  # finestra: 400 campioni che qui, a 8 kHz, fanno 50 ms
         # (nella prima sezione 400 campioni erano 25 ms perche' li' si
         #  misurava 16.000 volte al secondo invece di 8.000)
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

Il tono ha energia $0{,}5$, e quel numero si può controllare: l'onda oscilla fra
$-1$ e $1$, e la media dei quadrati di un'oscillazione regolare viene sempre
esattamente la metà del quadrato del suo picco, qui $0{,}5$. (È lo stesso conto
per cui i 230 volt della corrente di casa sono il valore *efficace* di un'onda
che di picco arriva a 325.)

Il suo zcr, invece, è bassissimo: $0{,}049$. Ecco il conto. Il tono del codice
fa 200 oscillazioni al secondo e noi misuriamo 8.000 volte al secondo, quindi
ogni oscillazione la campioniamo 40 volte ($8000$ diviso $200$). Ma ogni
oscillazione attraversa lo zero due volte, una salendo e una scendendo:
quindi un attraversamento ogni 20 campioni, e $1$ diviso $20$ fa $0{,}05$,
cioè quello che troviamo nella tabella a meno degli arrotondamenti ai bordi.
Più il suono è acuto, più fitti sono quei passaggi: è tutto il legame fra questa
misura e le frequenze.

Il silenzio ha energia praticamente nulla, e sullo zcr c'è una cosa da
guardare: vale circa $0{,}5$, cioè **quanto quello del rumore**. È come
l'abbiamo costruito: il nostro «silenzio» è rumore anche
lui, solo trecento volte più piccolo. In un fondo così ogni minuscolo sbalzo
casuale attraversa lo zero, esattamente come fanno gli sbalzi grossi del
rumore vero. Contare gli attraversamenti, da solo, non li distingue affatto.

E allora perché la regola funziona? Perché le due domande si fanno in un ordine
preciso: prima l'energia, che manda il silenzio fuori gioco, e solo dopo lo zcr,
che a quel punto deve separare soltanto il tono dal rumore, e lì la differenza è
enorme ($0{,}05$ contro $0{,}5$, dieci volte). Una regola a soglie è una
scaletta e non un elenco di condizioni, e cambiare l'ordine la rompe.

Due numeri per finestra e due soglie, nessuna rete, e la logica è già quella dei
modelli grandi: *estrarre feature che separano le classi, poi decidere*. La
differenza è che una rete convoluzionale o un AST le feature migliori se le
imparano da soli, invece di riceverle scritte a mano.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Una volta che il suono è diventato una **radiografia** (lo spettrogramma),
  riconoscerlo è un problema di **immagini**: la stessa rete che distingue un
  gatto da un cane in una foto distingue un vetro rotto da un clacson in una
  lastra sonora, e può perfino partire da quello che ha già imparato sulle
  foto. Con un'avvertenza: sulla lastra spostare una sagoma a destra è lo stesso
  suono più tardi, spostarla in alto è un suono diverso, e la rete quella
  differenza non la conosce.
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
- **AudioSet** {cite}`gemmeke2017audioset` cambia le regole con la scala: il
  paper definisce l'ontologia (632 categorie), la raccolta pubblicata arriva a
  oltre 2 milioni di clip da 10 s su 527 classi di benchmark, con etichette
  *deboli*. Contano scala e pre-addestramento più della precisione della
  singola annotazione (metrica: mAP).
- L’**Audio Spectrogram Transformer** {cite}`gong2021ast` applica un
  Transformer in stile ViT alle patch dello spettrogramma, senza convoluzioni;
  in cambio del campo recettivo globale, chiede molti dati o il transfer da
  ImageNet.
- Estrarre due feature semplici (**energia** e **zero-crossing rate**) basta
  per separare a mano silenzio, tono e rumore: la stessa idea dei modelli
  grandi, che però le feature se le imparano da soli.
```

`````
