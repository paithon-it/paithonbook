---
titolo_ricerca: "Addestrare un modello video: obiettivo, dati, FVD e VBench"
descrizione: "Che cosa impara un modello che genera video, e da che cosa: flow
  matching sul latente video, spostamento del rumore, cura dei dati e
  didascalie, addestramento per tappe, e perché FVD e VBench si lasciano
  ingannare da un video fermo."
---
# Insegnare il movimento: dati, addestramento, misura

Stable Video Diffusion parte da 577 milioni di clip, circa duecentododici anni
di filmato, e dopo i filtri ne tiene 152 milioni {cite}`blattmann2023stable`.
Quasi tre clip su quattro vengono scartati: scene in cui non si muove niente,
clip coperti di scritte, inquadrature brutte, clip la cui didascalia non
somiglia a ciò che si vede. Lo scarto conta quanto ciò che si tiene, perché
un modello di diffusione impara la distribuzione dei dati che riceve, e per un
video quella distribuzione comprende il movimento: se metà dei clip è
un’inquadratura fissa, metà di ciò che il modello impara è come restare fermo.

Che cosa il modello impara a fare con il tempo dipende da che cosa gli si
chiede di predire a ogni passo, da quali clip vede e in quale ordine, e dal
metro con cui si giudica ciò che genera. Il metro sembra la scelta più
innocua, ed è quella che può premiare proprio il video fermo che la cura dei
dati ha cercato di togliere.

## L’obiettivo e quanto rumore mettere

Wan, HunyuanVideo e Movie Gen, tre generatori di cui sono pubblici i report,
addestrano un {doc}`Transformer di diffusione
</ModelliDiffusione/diffusion-transformer>`, cioè la rete che riceve un latente
rumoroso e predice in che direzione spostarlo. Il latente è
$\mathbf{x}_0\in\mathbb{R}^{F'\times H/c_s\times W/c_s\times C}$, la versione
compressa del clip che l’autoencoder della {doc}`sezione sulla compressione del
tempo </GenerazioneVideo/comprimere-il-tempo>` ricava dai fotogrammi.

L’obiettivo è il {doc}`flow matching </ModelliDiffusione/flow-matching>` nella
forma del rectified flow: il dato e il rumore sono collegati da una retta, e la
rete impara la velocità con cui la si percorre. La convenzione è quella del
libro, con $t=0$ sul dato e $t=1$ sul rumore. Qui $t$ è il tempo della
diffusione e $i$ l’indice del fotogramma: sono due tempi diversi, e il primo
scorre anche su un video fermo. Restano da decidere quale $t$ estrarre a ogni
passo d’addestramento e di quanto spostarlo quando la risoluzione cresce.

`````{tab} Elementare

Un filmato pulito e una schermata di neve, come sul televisore senza antenna: il
filmato diventa neve lungo una strada dritta, e a metà strada è un misto in
parti uguali dei due. Addestrare la rete vuol dire farle imparare una freccia:
in ogni punto della strada, da che parte e a che passo si va verso la neve. Per
ogni esempio si prende un clip vero e una neve estratta a caso, si sceglie un
punto della strada e la rete, che legge anche la descrizione del filmato,
indovina la freccia. La risposta giusta è la stessa lungo tutta la strada, e si
calcola senza fatica: la differenza fra la neve e il filmato. Se un puntino del
filmato vale 0,2 e la neve, nello stesso punto, vale 0,9, la freccia dice «sali
di 0,7», sia all’inizio della strada sia a metà, dove il puntino vale 0,55.

Un dettaglio conta per il video. La freccia riguarda il clip intero, tutti i
fotogrammi insieme, e il punto della strada è uno solo: la neve cade sul clip
con la stessa intensità su ogni fotogramma.

I punti della strada non si pescano alla pari. [Si scelgono più spesso quelli di
mezzo]{.pt-matita}, dove non è ancora chiaro né che cosa fosse il filmato né che
cosa sia neve, e indovinare è più difficile.

Quando si usa il modello, il viaggio va al contrario. Si parte dal fondo della
strada, dove c’è solo neve, senza un’ombra del filmato, e si cammina contro la
freccia: a ogni passo la rete indica la neve e si fa un passetto dalla parte
opposta, finché resta un filmato. Per questo basta imparare la freccia verso la
neve: chi sa da che parte si va verso la neve sa anche da che parte si torna.

L’ultimo gesto riguarda la grandezza del filmato. Una fotografia di sedici
pixel per lato, coperta di neve fitta, si legge a malapena. La stessa scena a
sessantaquattro pixel per lato dice le stesse cose con molti più punti: ogni
pezzetto di scena è raccontato da sedici volte più punti vicini, come una frase
ripetuta sedici volte in una stanza rumorosa. Nella media di sedici punti
vicini la scena resta, mentre la neve, che cade a caso, in parte si cancella.
Solo in parte, però: sedici granelli presi a caso non si compensano tutti, e
nella media il loro disturbo si divide per la radice quadrata di sedici, cioè
per quattro e non per sedici. Basta perché la stessa neve faccia meno danno.
Per rendere l’esercizio difficile quanto prima la neve va addensata, cioè la
strada va spostata verso la neve: sul lato quadruplicato serve neve quattro
volte più intensa.

`````

`````{tab} Superiore

Sia $\mathbf{x}_0=\mathcal{E}(\text{clip})$ il latente, con $\mathcal{E}$
l’encoder del VAE, $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$
un rumore della stessa forma e $\mathbf{c}$ la codifica del testo, una matrice
con un vettore per token. Con un solo $t\in[0,1]$ per clip, uguale su tutti i
fotogrammi latenti, il flow matching è la regressione

$$
\mathbf{x}_t=(1-t)\,\mathbf{x}_0+t\,\boldsymbol{\epsilon},
\qquad
\mathcal{L}(\theta)=\mathbb{E}\Big[\big\lVert
\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{c})
-(\boldsymbol{\epsilon}-\mathbf{x}_0)
\big\rVert_2^2\Big],
$$

dove $\mathbf{v}_\theta$ è il Transformer, l’attesa è su clip, rumore e $t$, e
$t=1/(1+e^{-u})$ con $u\sim\mathcal{N}(0,1)$ è la logit-normale di Movie Gen,
che concentra l’addestramento sugli istanti intermedi {cite}`polyak2024movie`.
Un livello di rumore unico per tutto il clip è coerente con l’attenzione 3D
piena, che mescola tutti i token; dare a ogni fotogramma il proprio $t$
cambierebbe l’obiettivo, ed è ciò che fa Diffusion Forcing nella {doc}`sezione
sul tempo reale </GenerazioneVideo/generare-mentre-scorre>`. In generazione si
va nel verso opposto, da $t=1$ a $t=0$: si parte da $\boldsymbol{\epsilon}$ e
si segue all’indietro la velocità appresa, nel caso più semplice con il passo
di Eulero

$$
\mathbf{x}_{t-\Delta t}=\mathbf{x}_t-\Delta t\,
\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{c}),
$$

dove $\Delta t$ è la lunghezza del passo.

Wan, HunyuanVideo e Movie Gen scrivono lo stesso obiettivo nel verso opposto,
nella notazione $x_t=t\,x_1+(1-t)\,x_0$, con $x_1$ il dato, $x_0$ il rumore e
velocità $x_1-x_0$ {cite}`wan2025wan,kong2024hunyuanvideo,polyak2024movie`.
Con $t_{\text{libro}}=1-t_{\text{report}}$ e il segno del bersaglio cambiato è
la stessa regressione. Movie Gen, in più, lascia anche sul dato un residuo di
rumore $\sigma_{\min}=10^{-5}$; nella convenzione del libro

$$
\mathbf{x}_t=(1-t)\,\mathbf{x}_0+\big(\sigma_{\min}+(1-\sigma_{\min})\,t\big)\,
\boldsymbol{\epsilon},
\qquad
\text{bersaglio }(1-\sigma_{\min})\,\boldsymbol{\epsilon}-\mathbf{x}_0,
$$

e, poiché la logit-normale a media zero è simmetrica, il cambio di verso non ne
tocca la distribuzione. Movie Gen motiva la scelta del flow matching con un
rapporto segnale-rumore terminale nullo, che i programmi di rumore standard
della diffusione non garantiscono, e con la maggiore robustezza alla scelta del
programma. La condizione conta perché in generazione si parte da rumore puro, e
un programma che a $t=1$ conserva un residuo di segnale mostra al modello, in
addestramento, un ingresso che in uso non c’è. CogVideoX arriva alla stessa
condizione per un’altra strada: resta alla diffusione, con la
{doc}`v-prediction e un programma a SNR terminale nullo
</ModelliDiffusione/pochi-passi>` {cite}`yang2025cogvideox`. La $\mathbf{v}$
di quel nome non è la velocità di una retta, come nel flow matching, ma una
combinazione di rumore e dato definita sul processo di diffusione; Imagen Video
la sceglie perché nel video evita lo spostamento di colore nel tempo
{cite}`ho2022imagen`. CogVideoX cura anche l’estrazione del passo. I suoi passi
sono discreti, da 1 a $T$, e invece di farli estrarre uniformemente a ogni
processo del calcolo distribuito divide l’intervallo in tanti sottointervalli
quanti sono i processi, e ciascuno estrae dal proprio: i passi di un batch
coprono l’intervallo in modo più uniforme, e la perdita oscilla meno
{cite}`yang2025cogvideox`.

Stable Video Diffusion, che non usa il flow matching, sposta verso valori più
alti la distribuzione dei livelli di rumore d’addestramento quando passa a una
risoluzione più alta, e trova lo spostamento essenziale per l’affinamento ad
alta risoluzione {cite}`blattmann2023stable`. Nel flow matching lo spostamento
ha la forma che Esser e colleghi danno per Stable Diffusion 3
{cite}`esser2024scaling`:

$$
\tau_s(t)=\frac{s\,t}{1+(s-1)\,t},
\qquad
\frac{\tau_s}{1-\tau_s}=s\,\frac{t}{1-t},
$$

che fissa gli estremi e per $s>1$ porta ogni $t$ verso 1. Poiché $t/(1-t)$ è il
rapporto fra ampiezza del rumore e del segnale, il fattore $s$ moltiplica quel
rapporto e divide l’SNR per $s^2$. La ragione che ne danno è la ridondanza: [a
risoluzione alta lo stesso segnale è ripetuto su più punti vicini, e il rumore
che lo copre si cancella in parte]{.pt-matita}. Per un’immagine costante la
media dei pixel stima il valore con un errore la cui deviazione è $t/(1-t)$
divisa per la radice del numero di pixel, e pareggiare l’errore fra due
risoluzioni dà per $s$ il rapporto fra i lati. Gli stessi autori avvertono che
un’immagine costante non è realistica, e scelgono il valore con uno studio di
preferenze umane: $s=3$ per le immagini a 1024×1024.

HunyuanVideo usa la stessa $\tau_s$, ma in inferenza e per un’altra ragione
{cite}`kong2024hunyuanvideo`. Nella sezione sull’inferenza il report mette il
rumore a $t=1$, come il libro e al contrario della propria formula
d’addestramento, quindi la formula si legge senza conversioni; il fattore,
però, cresce quando i passi diminuiscono, $s=7$ a 50 passi e $s=17$ sotto i
venti, perché i primi passi, quelli più rumorosi, decidono la maggior parte dei
cambiamenti.

`````

Il meccanismo si vede con dei numeri. Ci chiediamo quanto rumore serva, a
risoluzioni diverse, perché ricostruire la stessa scena sia ugualmente
difficile. La scena è un campo liscio, cioè un’immagine che cambia lentamente
da un punto all’altro (una somma di pochi coseni), con varianza 1 e lato $m$ di
16, 32, 64 e 128 pixel. Al posto della rete c’è lo stimatore più semplice, che
non impara niente: divide l’immagine in una griglia di $8\times8$ quadrati, che
a ogni lato coprono la stessa porzione di scena, e in ciascun quadrato
sostituisce i pixel con la loro media. L’errore è quello quadratico medio fra
la stima e la scena.

Quanto al rumore, il dato rumoroso è
$\mathbf{x}_t=(1-t)\,\mathbf{x}_0+t\,\boldsymbol{\epsilon}$, e dividerlo per
$1-t$ non cambia ciò che se ne può ricavare: resta
$\mathbf{x}_0+\frac{t}{1-t}\,\boldsymbol{\epsilon}$. Rispetto a un segnale di
varianza 1, quindi, il rumore ha deviazione $\sigma=t/(1-t)$, e $t=0{,}5$
corrisponde a $\sigma=1$.

```python
import numpy as np

rng = np.random.default_rng(3)
freq = np.arange(4)                  # coseni a bassa frequenza: segnale liscio

def campo(m, n):
    """n immagini m x m, somma di coseni a bassa frequenza, varianza 1."""
    u = (np.arange(m) + 0.5) / m
    c = np.cos(np.pi * np.outer(freq, u))
    a = rng.normal(size=(n, 4, 4))
    x = np.einsum("nij,iy,jx->nyx", a, c, c)
    return x / x.std()

def errore(x, z, sigma):
    """Errore quadratico dello stimatore a blocchi 8 x 8."""
    n, m, _ = x.shape
    b = m // 8
    media = (x + sigma * z).reshape(n, 8, b, 8, b).mean(axis=(2, 4))
    stima = np.repeat(np.repeat(media, b, axis=1), b, axis=2)
    return float(np.mean((stima - x) ** 2))

dati = {}
for m in (16, 32, 64, 128):
    x = campo(m, 24)
    dati[m] = (x, rng.normal(size=x.shape))

print("lato   errore a sigma = 1")
for m, (x, z) in dati.items():
    print(f"{m:4d}   {errore(x, z, 1.0):.4f}")

bersaglio = errore(*dati[16], 1.0)
print("\nlato   sigma che pareggia   t = 0.5 spostato a")
for m in (32, 64, 128):
    lo, hi = 0.5, 20.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if errore(*dati[m], mid) < bersaglio:
            lo = mid
        else:
            hi = mid
    print(f"{m:4d}   {lo:12.2f}   {lo / (1 + lo):18.3f}")
```

```text
lato   errore a sigma = 1
  16   0.3048
  32   0.1266
  64   0.0789
 128   0.0726

lato   sigma che pareggia   t = 0.5 spostato a
  32           1.96                0.662
  64           4.00                0.800
 128           7.70                0.885
```

Con lo stesso rumore l’errore scende da 0,30 a 0,07 al crescere della
risoluzione, e per riportarlo al valore del lato 16 il rumore deve crescere
quasi come il lato: 1,96 al lato 32, 4,00 al 64, 7,70 al 128, contro 2, 4 e 8.
Lo spostamento $\tau_s$ dei tempi fa proprio questo: moltiplica per un fattore
$s$ il rapporto $t/(1-t)$, cioè la deviazione del rumore. Con $s$ uguale al
fattore che pareggia, il punto $t=0{,}5$ va in $\tau_s(0{,}5)=s/(1+s)$, che al
lato 64 vale 0,800, cioè $\tau_4(0{,}5)$. In questo campo liscio, dunque, $s$ è
circa il rapporto fra i lati. Il segnale è costruito apposta: il giocattolo
mostra il meccanismo della ridondanza, mentre i valori di $s$ dei modelli veri
si scelgono per prova.

## Che cosa si dà da vedere al modello: dati, didascalie, tappe

I filtri e le didascalie decidono quale movimento il modello vedrà. L’ordine in
cui glielo si mostra decide quanto costa mostrarglielo, perché i clip lunghi e
nitidi costano molto più dei fotogrammi isolati.

`````{tab} Elementare

Una cineteca ha ricevuto un’enorme quantità di pellicole, e un apprendista
animatore deve imparare il movimento guardandole. Prima servono tre lavori.

Il primo è il taglio. Una pellicola intera è un montaggio di scene incollate, e
l’apprendista deve vederne una alla volta: si cercano gli stacchi e si taglia, e
da un solo rotolo ne escono molte.

Il secondo è la descrizione. Le didascalie che accompagnano i filmati veri sono
scarne, quindi ogni scena riceve il racconto di un narratore automatico, lungo e
dettagliato: chi c’è, dove, come si muove la camera. Quando poi l’apprendista,
a lavoro finito, riceve una richiesta, un secondo narratore la riscrive nello
stesso stile, perché lui ha sempre studiato su racconti lunghi.

Il terzo è la scrematura. Via le scene in cui non si muove niente, come una
telecamera puntata su un muro; via quelle coperte di scritte, quelle brutte,
quelle in cui la descrizione non c’entra con ciò che si vede. Quanto essere
severi non si decide a occhio. Per ogni criterio si provano tre severità, si
scarta il peggior ottavo, il peggior quarto o la peggior metà, e a ciascuna
severità corrisponde un apprendista: [si chiede a delle persone quale dei tre
disegna meglio]{.pt-matita}, e quella è la severità giusta.

Resta l’ordine delle lezioni. Si comincia da fotografie ferme, che costano
poco; poi clip brevi e piccoli; poi clip lunghi e nitidi; in fondo, un ritocco
su un mazzo più piccolo, scelto fra i filmati migliori. Una fotografia è un
filmato di un solo fotogramma, quindi sta nello stesso esercizio, e anche fra i
filmati se ne lasciano alcune, in dosi diverse da una tappa all’altra. Aiutano,
perché i filmati belli sono pochi e a forza di soli filmati si dimentica come si
disegna; ma una fotografia non mostra nessun movimento, e perciò non devono
essere troppe. Le tappe esistono perché una scena grande e lunga costa molto, e
finché l’apprendista non sa ancora disegnare quel prezzo sarebbe sprecato.

`````

`````{tab} Superiore

**Segmentazione.** Stable Video Diffusion rileva i tagli a tre frequenze di
fotogrammi, e dalla stessa raccolta escono così circa quattro volte più clip:
ogni clip è una scena sola, l’unità su cui la distribuzione ha senso
{cite}`blattmann2023stable`.

**Didascalie.** Per ogni clip Stable Video Diffusion ne produce tre sintetiche:
CoCa sul fotogramma centrale, V-BLIP sul clip e un riassunto fatto da un modello
di linguaggio. CogVideoX descrive ogni fotogramma con CogVLM, riassume con
GPT-4, addestra un Llama 2 sui riassunti e infine un modello end-to-end,
CogVLM2-Caption {cite}`yang2025cogvideox`; HunyuanVideo scrive la didascalia
come JSON strutturato (descrizione breve e densa, sfondo, stile, inquadratura,
luce, atmosfera) {cite}`kong2024hunyuanvideo`. Il meccanismo comune è il
**recaptioning**: i testi dei dati veri sono poveri, il modello si addestra su
descrizioni dense prodotte da un modello visione-linguaggio, e in inferenza un
modello di linguaggio porta il prompt dell’utente nello stesso stile (Movie Gen
§3.4.1 {cite}`polyak2024movie`, HunyuanVideo §4.6).

**Filtri.** Ogni clip riceve annotazioni: il flusso ottico, per togliere le
scene statiche; l’OCR, per togliere i clip pieni di testo; un punteggio estetico
e la similarità testo-immagine, entrambi calcolati con CLIP. Le soglie si
tarano su un sottoinsieme casuale di 9,8 milioni di clip, un’annotazione alla
volta: si toglie il 12,5, il 25 o il 50% peggiore, si addestra un modello su
ciò che resta e si confrontano i modelli con preferenze umane a coppie,
riassunte in un punteggio Elo (quello degli scacchi: a ogni confronto il
vincitore guadagna punti in proporzione a quanto era sfavorito, e più alto è
meglio). Dal complesso LVD, 577 milioni di clip e 212 anni di contenuto, si
arriva a LVD-F con 152 milioni.

**Immagini e video insieme.** Video Diffusion Models accoda a ogni video
alcune immagini indipendenti, prese da altri video dello stesso insieme, e
maschera l’attenzione temporale perché le immagini non si mescolino con i
fotogrammi del video né fra loro: per un’immagine la matrice d’attenzione
temporale si riduce all’identità, e ognuna guarda solo sé stessa. Gli autori
leggono l’effetto come una riduzione della varianza del gradiente a spese di un
po’ di bias per l’obiettivo video, cioè come un modo di far stare più esempi
indipendenti in un batch, e provano 0, 4 e 8 immagini per video
{cite}`ho2022video`. Nei modelli con il {doc}`VAE 3D causale
</GenerazioneVideo/comprimere-il-tempo>` il primo fotogramma, codificato da
solo, rende un’immagine un clip di un fotogramma. CogVideoX mette in più nello
stesso batch video di durate e risoluzioni diverse, perché addestrare immagini
(un fotogramma) e video a durata fissa fa nascere due modi generativi separati,
secondo il numero di token {cite}`yang2025cogvideox`.

**Tappe.** Stable Video Diffusion procede per immagini (stadio I), per video
(stadio II) e per un affinamento su un piccolo insieme di video di alta qualità
e risoluzione più alta (stadio III). Wan parte dalle immagini a 256 px e passa
a tre stadi congiunti di immagini e video, sempre con clip di 5 s: video a 192
px e 16 fps con immagini a 256 px, poi tutto a 480 px, poi a 720 px
{cite}`wan2025wan`. HunyuanVideo passa da 256 px a una miscela di 256 e 512,
poi a tre stadi di video (bassa risoluzione e clip brevi, bassa e lunghi, alta
e lunghi), e a ogni stadio mescola ai video delle immagini, in proporzioni
variabili, sia perché i video di qualità scarseggiano sia per prevenire l’oblio
catastrofico, cioè la perdita di ciò che il modello sapeva delle immagini
{cite}`kong2024hunyuanvideo`.

`````

Il costo di ogni tappa lo decide la lunghezza della sequenza, cioè il numero di
token, i pezzi del latente che il Transformer riceve uno per uno. Li contiamo
per un clip di 81 fotogrammi con le formule di Wan ($F'=1+(F-1)/c_t$
fotogrammi latenti, $c_t=4$, $c_s=8$, patch $1\times2\times2$), a due
risoluzioni della stessa famiglia: 1280×720 e 832×480, il formato 480p con cui
gira Wan 1.3B {cite}`huang2025self`.

```python
F, c_t, c_s = 81, 4, 8                # fotogrammi, compressione del VAE
patch_h = patch_w = 2                 # patch 1 x 2 x 2

def token(H, W):
    fl = 1 + (F - 1) // c_t           # fotogrammi latenti
    return fl * (H // c_s // patch_h) * (W // c_s // patch_w)

piccolo, grande = token(480, 832), token(720, 1280)
print("token a 832x480  :", piccolo)
print("token a 1280x720 :", grande)
print(f"costo lineare    : x{grande / piccolo:.2f}")
print(f"costo quadratico : x{(grande / piccolo) ** 2:.2f}")

tenuti, iniziali = 152, 577           # milioni di clip: LVD-F e LVD (Stable Video Diffusion)
print(f"\nclip tenuti dai filtri: {tenuti / iniziali:.3f}")
print(f"clip scartati         : {1 - tenuti / iniziali:.3f}")
```

```text
token a 832x480  : 32760
token a 1280x720 : 75600
costo lineare    : x2.31
costo quadratico : x5.33

clip tenuti dai filtri: 0.263
clip scartati         : 0.737
```

Il passaggio da 832×480 a 1280×720 moltiplica i token per 2,31, il costo delle
parti lineari dello strato per lo stesso fattore e quello dell’attenzione, che è
quadratico nella lunghezza, per 5,33: spendere le prime tappe a risoluzione
bassa è il modo di pagare la parte più cara solo quando il modello genera già
bene immagini e clip brevi. La {doc}`sezione sull’attenzione nello spazio e nel
tempo </GenerazioneVideo/spazio-e-tempo>` misura quanto pesi l’attenzione sul
resto dello strato. Quanto alla cura dei dati, i filtri di Stable Video
Diffusion tengono 0,263 dei clip e ne scartano 0,737, cioè quasi tre su
quattro.

## Da un’immagine a un video

Nella generazione da immagine a video la condizione è il primo fotogramma: il
modello riceve una fotografia e genera il clip che comincia da lì. Stable Video
Diffusion ne addestra due versioni, a 14 e a 25 fotogrammi
{cite}`blattmann2023stable`. Rispetto alla generazione da testo cambiano il
modo in cui la condizione entra nella rete, la {doc}`guida senza
classificatore </ModelliDiffusione/guida>` e il livello di rumore
dell’addestramento.

`````{tab} Elementare

Questa volta non arriva una descrizione ma una fotografia, e il filmato deve
cominciare proprio da lì. La rete la riceve in due modi. Al posto della
descrizione scritta legge un riassunto della fotografia, una lista di numeri
che dice che cosa c’è, preparata da una rete che ha imparato ad accostare le
immagini alle parole. E la fotografia stessa, appena velata di neve, le viene
messa accanto a ogni fotogramma, come una foto di riferimento da tenere
sott’occhio mentre si disegna.

Poi c’è la guida. La rete si interroga due volte, una con la fotografia e una
senza; si parte dalla freccia senza fotografia e le si aggiunge la differenza
fra le due, moltiplicata per un numero, il peso della guida. Qui il peso è
delicato: troppo piccolo, e il filmato si allontana dalla fotografia; troppo
grande, e i colori diventano troppo carichi, come in una foto con la
saturazione al massimo. La via d’uscita, trovata per prova, è farlo crescere
lungo il filmato: piccolo sul primo fotogramma, grande sull’ultimo.

Resta la strada fra filmato e neve. Con la fotografia sempre accanto la rete
ha un aiuto molto forte, e proprio per questo chi la addestra sposta di nuovo
la strada verso la neve, come per i filmati grandi ma per un’altra ragione.

`````

`````{tab} Superiore

Stable Video Diffusion fa entrare l’immagine per due vie. Il suo embedding
CLIP, il vettore che le assegna l’encoder visivo di un {doc}`modello che
allinea immagini e testi </VisioneLinguaggio/allineare-due-spazi>`, prende il
posto della codifica del testo che il modello di base riceveva. L’immagine
stessa, sporcata con un rumore gaussiano di intensità
$\log\sigma\sim\mathcal{N}(-3,\,0{,}5^2)$ e codificata con l’encoder di Stable
Diffusion 2.1, si concatena lungo i canali all’ingresso della U-Net, copiata
su tutti i fotogrammi e senza maschere {cite}`blattmann2023stable`. Chiamiamo
$\mathbf{c}_{\text{img}}$ questa condizione, per distinguerla dalla codifica
del testo $\mathbf{c}$. Scritta per la velocità $\mathbf{v}_\theta$, la guida
senza classificatore diventa

$$
\tilde{\mathbf{v}}^{i}=\mathbf{v}_\theta^{i}(\mathbf{x}_t,t,\varnothing)+
w_i\big(\mathbf{v}_\theta^{i}(\mathbf{x}_t,t,\mathbf{c}_{\text{img}})-
\mathbf{v}_\theta^{i}(\mathbf{x}_t,t,\varnothing)\big),
$$

dove l’apice $i$ prende, della velocità predetta per tutto il clip, la parte
che riguarda il fotogramma latente $\mathbf{x}^{i}$ (è quindi una velocità, non
un fotogramma), $\varnothing$ indica la condizione tolta e $w_i$ il peso della
guida su quel fotogramma. Con un peso unico, osservano gli autori, una guida
troppo debole produce un video incoerente con l’immagine di partenza e una
troppo forte satura i colori; farlo crescere linearmente lungo i fotogrammi,
dal più piccolo al più grande, si è rivelato utile. Nell’affinamento da
immagine a video, infine, lo stesso lavoro sposta di nuovo verso valori più
alti la distribuzione del rumore d’addestramento, e non per la risoluzione:
perché, scrivono, il segnale di condizionamento è molto forte.

`````

## Misurare un video: FVD, VBench e il video fermo

Un generatore si giudica con numeri, e per il video due fra i più diffusi sono
la *Fréchet Video Distance* (FVD) e le dimensioni di VBench. La FVD riassume
ogni clip in un vettore di numeri e misura quanto l’insieme dei clip generati
stia lontano da quello dei clip veri, come il {doc}`FID </GAN/come-funziona>`
per le immagini; ogni dimensione di VBench dà invece un punteggio a una
proprietà dei video generati. Nessuna è sbagliata. Ciascuna però può essere
soddisfatta senza fare la cosa che dovrebbe misurare, e il caso più semplice è
il video che non si muove: ha sfarfallio nullo, come si è visto con la misura
$D$ dell’{doc}`apertura del capitolo </GenerazioneVideo/overview>`.

Un esperimento in una dimensione mette a confronto due di queste misure. La
scena è una striscia di 160 pixel, il video una macchia che si sposta di $v$
pixel a fotogramma su 16 fotogrammi, con un po’ di rumore del sensore.

La **coerenza** è la similarità coseno media fra fotogrammi vicini: letti i due
fotogrammi come vettori di pixel, vale 1 quando sono proporzionali e scende
quando differiscono. È una versione semplificata della consistenza del
soggetto di VBench, che la calcola non sui pixel ma sulle feature di
{doc}`DINO </VisioneArtificiale/senza-etichette>`, una rete auto-supervisionata
che riassume ogni fotogramma in un vettore; si massimizza. Il **grado di
dinamica** è la quota di video in cui la macchia si sposta più di mezzo pixel a
fotogramma, al posto del {doc}`flusso ottico
</VisioneArtificiale/geometria-e-profondita>` che VBench stima fra fotogrammi
vicini; anch’esso si massimizza.

```python
import numpy as np

rng = np.random.default_rng(11)
F, n = 16, 160                        # fotogrammi, pixel della scena 1D
x = np.arange(n)

def video(v, rumore=0.02):
    """Una macchia che si sposta di v pixel a fotogramma, con rumore del sensore."""
    c0 = rng.uniform(10, n - 10 - v * F)
    centri = c0 + v * np.arange(F)
    f = np.exp(-0.5 * ((x[None, :] - centri[:, None]) / 3.0) ** 2)
    return f + rumore * rng.normal(size=f.shape), centri

def coerenza(f):
    """Similarita' coseno media fra fotogrammi vicini (sui pixel, non su DINO)."""
    a, b = f[:-1], f[1:]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return cos.mean()

def dinamico(centri, soglia=0.5):
    """Dinamico: la macchia si sposta in media piu' di 'soglia' pixel a fotogramma."""
    return np.abs(np.diff(centri)).mean() > soglia

print("velocita'  coerenza  dinamica")
for v in (0.0, 1.0, 2.0, 3.0):
    prove = [video(v) for _ in range(200)]
    c = np.mean([coerenza(f) for f, _ in prove])
    d = np.mean([dinamico(ce) for _, ce in prove])
    print(f"{v:8.1f}  {c:8.3f}  {d:8.2f}")

# fotogrammi indipendenti: ognuno e' una scena plausibile, fra loro nessun legame
indip = [np.stack([video(0.0)[0][0] for _ in range(F)]) for _ in range(200)]
print(f"\nfotogrammi indipendenti: coerenza {np.mean([coerenza(f) for f in indip]):.3f}")
```

```text
velocita'  coerenza  dinamica
     0.0     0.988      0.00
     1.0     0.961      1.00
     2.0     0.884      1.00
     3.0     0.770      1.00

fotogrammi indipendenti: coerenza 0.070
```

Il video fermo ha la coerenza più alta (0,988) e la dinamica zero; la coerenza
cala man mano che la macchia va più veloce, fino a 0,770 a tre pixel a
fotogramma. Con i fotogrammi indipendenti, ognuno plausibile e nessuno legato
al vicino, scende a 0,070. Chi massimizza la sola coerenza converge quindi alla
scena immobile, e dei due numeri solo il grado di dinamica se ne accorge.

Resta da vedere che cosa fa una distanza fra insiemi. Un secondo esperimento
costruisce due insiemi indipendenti di clip con una macchia in moto e li
confronta (vero contro vero); poi confronta il primo con il secondo dopo aver
mescolato in ordine casuale i fotogrammi di ogni clip. Le feature vengono da due
estrattori: uno che vede solo il contenuto (la media dei fotogrammi) e uno che
aggiunge il moto (la differenza media assoluta fra fotogrammi vicini). La
distanza è quella di Fréchet; più è bassa, meglio è. L’esperimento si ripete
su cinque estrazioni, perché anche fra due insiemi veri, che sono finiti, la
distanza non è zero, e cambia da un campione all’altro.

```python
import numpy as np

rng = np.random.default_rng(5)
F, n = 16, 160
x = np.arange(n)

def clip():
    """Macchia in moto, da 1 a 3 pixel a fotogramma, in un verso o nell'altro."""
    v = rng.uniform(1, 3) * rng.choice([-1, 1])
    c0 = rng.uniform(10, n - 10 - abs(v) * F) + (abs(v) * F if v < 0 else 0)
    centri = c0 + v * np.arange(F)
    f = np.exp(-0.5 * ((x[None, :] - centri[:, None]) / 3.0) ** 2)
    return f + 0.02 * rng.normal(size=f.shape)

def mescola(f):
    """Stessi fotogrammi, ordine temporale casuale."""
    return f[rng.permutation(F)]

def solo_contenuto(f):
    """Che cosa c'e' nel clip, senza l'ordine: media dei fotogrammi, in 8 fasce."""
    return f.mean(0).reshape(8, -1).mean(1)

def con_moto(f):
    """Contenuto, piu' la differenza media assoluta fra fotogrammi vicini."""
    d = np.abs(np.diff(f, axis=0)).mean(0).reshape(8, -1).mean(1)
    return np.concatenate([solo_contenuto(f), d])

def frechet(a, b):
    """Distanza di Frechet fra due gaussiane stimate sui due insiemi di feature."""
    ma, mb = a.mean(0), b.mean(0)
    sa, sb = np.cov(a.T), np.cov(b.T)
    autov = np.linalg.eigvals(sa @ sb).real.clip(min=0)
    return float(((ma - mb) ** 2).sum() + np.trace(sa) + np.trace(sb)
                 - 2 * np.sqrt(autov).sum())

print("        solo contenuto         contenuto+moto")
print("prova  vero/vero  mescolato   vero/vero  mescolato  rapporto")
for prova in range(1, 6):             # cinque estrazioni indipendenti
    vere1 = [clip() for _ in range(600)]
    vere2 = [clip() for _ in range(600)]
    mescolati = [mescola(f) for f in vere2]
    d = []
    for feat in (solo_contenuto, con_moto):
        A = np.array([feat(f) for f in vere1])
        B = np.array([feat(f) for f in vere2])
        C = np.array([feat(f) for f in mescolati])
        d += [frechet(A, B), frechet(A, C)]
    print(f"{prova:5d}  {d[0]:9.5f}  {d[1]:9.5f}   {d[2]:9.5f}  {d[3]:9.5f}"
          f"  {d[3] / d[2]:8.0f}")
```

```text
        solo contenuto         contenuto+moto
prova  vero/vero  mescolato   vero/vero  mescolato  rapporto
    1    0.00033    0.00033     0.00038    0.06455       168
    2    0.00054    0.00054     0.00064    0.06397       101
    3    0.00031    0.00031     0.00038    0.06693       175
    4    0.00030    0.00030     0.00039    0.06380       165
    5    0.00056    0.00056     0.00064    0.06431       101
```

La media dei fotogrammi non dipende dal loro ordine, quindi per l’estrattore
che vede solo il contenuto l’insieme mescolato coincide con il secondo insieme
vero: in ogni prova le due distanze sono uguali, e il disordine temporale non
si vede affatto. Con l’estrattore che vede anche il moto, la distanza dal
mescolato sta fra 0,06380 e 0,06693 in tutte e cinque le prove, mentre quella
fra due insiemi veri va da 0,00038 a 0,00064. Il rapporto fra le due cambia
quindi molto da un’estrazione all’altra, da 101 a 175, perché il suo
denominatore è rumore di campionamento; stabile è soltanto l’ordine di
grandezza, il centinaio. Il primo estrattore è un caso estremo, costruito
apposta.

`````{tab} Elementare

Una giuria deve giudicare dei quaderni da sfogliare, quelli in cui il disegno
si muove se fai scorrere le pagine col pollice. Il primo criterio è se le
pagine vicine si somigliano: un quaderno con lo stesso disegno su ogni pagina
prende il voto più alto, e più il disegno si muove, meno le pagine vicine si
somigliano e più il voto cala. Serve allora un secondo criterio, quanto si
muove davvero il disegno, e il quaderno fermo ottiene zero. Su questi due voti
meglio vuol dire più alto. Un terzo criterio, lo sfarfallio, va nel verso
opposto: guarda quanto cambia una scena che dovrebbe restare ferma, e meno
cambia, meglio è (nelle tabelle lo si rovescia, così che anche il suo voto
salga quando va meglio). [Chi mira al massimo della sola somiglianza fra
pagine si ferma]{.pt-matita}.

C’è poi un altro modo di giudicare, che non guarda un quaderno alla volta ma
due scaffali interi: uno di quaderni veri e uno di quaderni disegnati dalla
macchina. Il giudice è una rete addestrata altrove a riconoscere azioni,
parente di quella che il FID usa per le immagini, che riconosce oggetti: non
le si chiede il nome dell’azione, le si sbirciano dentro i numeri con cui
riassume ogni quaderno. I quaderni di ciascuno scaffale diventano così una
nuvola di punti, come le città su una cartina, ma con molte coordinate invece
di due. Il voto è la distanza fra le due nuvole, e più è bassa, meglio è.

L’insidia sta nel giudice. Se ha imparato soprattutto che cosa si vede e poco
in che ordine, uno scaffale di quaderni con le pagine rimescolate gli sembra
quasi uguale a uno di quaderni veri: stesso contenuto, voto quasi uguale. Un
giudice che guarda anche come cambia il disegno da una pagina all’altra li
distingue subito. E c’è un limite di vista: il giudice sfoglia sedici pagine
alla volta. Se il personaggio cambia faccia a poco a poco, fra la pagina dieci
e la pagina cento, ogni mazzetto da sedici sembra in ordine, e nessuno
contiene insieme la faccia di prima e quella di dopo.

`````

`````{tab} Superiore

**FVD.** È la distanza di Fréchet fra due gaussiane stimate sulle feature di una
rete convoluzionale 3D, la I3D (*Inflated 3D ConvNet*), addestrata a
riconoscere azioni su Kinetics-400 o -600, prese dai logit o dall’ultimo
pooling {cite}`unterthiner2018towards`:

$$
\mathrm{FVD}=\lVert\boldsymbol{\mu}_R-\boldsymbol{\mu}_G\rVert_2^2+
\operatorname{Tr}\!\Big(\boldsymbol{\Sigma}_R+\boldsymbol{\Sigma}_G-
2\big(\boldsymbol{\Sigma}_R\boldsymbol{\Sigma}_G\big)^{1/2}\Big),
$$

con $R$ i clip veri e $G$ i generati; è il FID spiegato nella
{doc}`sezione su come si misura una GAN </GAN/come-funziona>`
{cite}`heusel2017gans`, con un estrattore video, e si minimizza. Gli autori
conducono uno studio umano su larga scala, in cui la FVD concorda bene con il
giudizio qualitativo, e ammettono che una gaussiana multivariata è raramente
una buona rappresentazione dei dati. Make-A-Video osserva che il modello di
valutazione della FVD si aspetta clip di 16 fotogrammi, mezzo secondo nelle
sue parole (cioè a 32 fotogrammi al secondo), troppo corti per l’uso pratico
{cite}`singer2023makeavideo`. Su un video più lungo, tagliato in clip di 16
fotogrammi consecutivi, un’incoerenza fra istanti più lontani di così (la
deriva lenta dell’aspetto di un soggetto) non sta dentro nessun clip che
l’estrattore guarda.

**Il contenuto.** Ge e colleghi mostrano che una distorsione
spaziale media senza corruzione temporale dà una FVD di 317,10, mentre una
distorsione spaziale minore con forte incoerenza temporale ne dà 310,52, cioè
migliore; scegliendo campioni senza movimento la FVD si abbassa molto senza che
la qualità temporale migliori. La causa è il classificatore supervisionato che
estrae le feature, addestrato su dati dominati dal contenuto, e feature
auto-supervisionate riducono il bias {cite}`ge2024content`. È, in forma
attenuata, il meccanismo dell’estrattore che vede solo il contenuto nel
giocattolo della distanza di Fréchet.

**VBench.** Ha 16 dimensioni gerarchiche, validate con annotazioni umane, e in
ciascuna un punteggio più alto vuol dire meglio {cite}`huang2024vbench`. La
consistenza del soggetto è la media, su ogni fotogramma, fra la similarità di
feature DINO col primo e col precedente. Lo sfarfallio parte dalla differenza
media assoluta fra fotogrammi, la $D$ dell’apertura del capitolo, che si
calcola su video di scene statiche e va minimizzata; VBench la riporta come
$(255-D)/255$, con i pixel fra 0 e 255, così che anche questa dimensione si
massimizzi. Il grado di dinamica usa il flusso ottico stimato da RAFT: per ogni
video la media del 5% dei flussi più grandi decide se il video è statico, e il
punteggio è la quota di video non statici. Gli autori dichiarano un
compromesso fra le dimensioni di coerenza temporale e il grado di dinamica,
perché una scena quasi statica può «barare» ottenendo un’alta coerenza, e nel
supplemento notano che i modelli che generano video statici hanno un vantaggio
ingiusto sulle altre dimensioni temporali. [Una metrica si può soddisfare
senza fare la cosa che misura]{.pt-matita}, e il grado di dinamica esiste per
accorgersene.

`````

Se i video generati rispettino le leggi della fisica è una domanda a parte, e
nessuna delle misure precedenti la pone. La {doc}`sezione sui generatori video e
il dibattito sui world model </WorldModels/simulatori-e-dibattito>` riporta come
la affrontano Kang e colleghi {cite}`kang2024far` e Physics-IQ
{cite}`motamed2025generative`: realismo visivo e correttezza fisica si misurano
a parte, e il primo non predice la seconda.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- La rete impara una freccia: dato un punto della strada fra filmato e neve,
  da che parte si va verso la neve. La risposta giusta è la stessa lungo tutta
  la strada, e vale per il clip intero, con una sola intensità di neve su tutti
  i fotogrammi. Per generare si parte dalla neve e si cammina contro la
  freccia, fino al filmato.
- I punti di mezzo si pescano più spesso. A risoluzione più alta la stessa
  neve fa meno danno, perché molti punti vicini dicono la stessa cosa e nella
  loro media la neve in parte si cancella; per questo la strada va spostata
  verso la neve.
- Il modello impara il movimento che i dati contengono. Si tagliano i rotoli in
  scene, si rimpiazza la descrizione scarna con una lunga, scritta da un
  narratore, e si scartano le scene ferme, piene di scritte, brutte o mal
  descritte, con una severità che scelgono delle persone.
- Si impara per tappe: fotografie, clip piccoli, clip lunghi e nitidi, un
  ritocco su un mazzo scelto fra i migliori. Le fotografie restano mescolate ai
  filmati, ma non troppe, perché non mostrano movimento.
- Per un filmato che parte da una fotografia, la fotografia prende il posto
  della descrizione e sta accanto a ogni fotogramma; il peso della guida
  cresce dal primo all’ultimo fotogramma.
- Un voto solo si può imbrogliare: il quaderno fermo vince sulla somiglianza
  fra pagine e perde sul movimento. Un giudice che guarda poco l’ordine delle
  pagine si accorge appena che sono rimescolate, e uno che sfoglia sedici
  pagine alla volta non vede ciò che cambia piano piano su tutto il quaderno.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L’obiettivo di Wan, HunyuanVideo e Movie Gen è il flow matching sul latente
  video, nella convenzione
  $\mathbf{x}_t=(1-t)\mathbf{x}_0+t\boldsymbol{\epsilon}$ con bersaglio
  $\boldsymbol{\epsilon}-\mathbf{x}_0$ e un $t$ per clip estratto da una
  logit-normale; tutti e tre lo scrivono nel verso opposto ($x_1$ il dato), e
  CogVideoX resta alla v-prediction con SNR terminale nullo.
  In generazione si integra da $t=1$ a $t=0$, contro la velocità appresa.
- Lo spostamento $\tau_s(t)=st/(1+(s-1)t)$ di Stable Diffusion 3 moltiplica
  per $s$ il rapporto rumore-segnale $t/(1-t)$. A risoluzione alta serve più
  rumore perché i pixel vicini ripetono lo stesso segnale: per un’immagine
  costante $s$ è il rapporto fra i lati, e nel campo liscio del giocattolo lo è
  circa. HunyuanVideo usa la stessa $\tau_s$ in inferenza, con $s$ che cresce
  quando i passi diminuiscono.
- La cura dei dati è una pipeline di annotazioni (tagli, tre didascalie,
  flusso ottico, OCR, estetica, CLIP) con soglie tarate su preferenze umane; il
  recaptioning porta didascalie dense in addestramento e riscrive il prompt in
  inferenza. Immagini e video si addestrano insieme (varianza contro bias), per
  tappe di risoluzione e durata.
- Da immagine a video (Stable Video Diffusion): l’embedding CLIP dell’immagine
  al posto del testo, il suo latente rumoroso concatenato sui canali, un peso
  della guida $w_i$ che cresce linearmente lungo i fotogrammi, e il rumore
  d’addestramento spostato in alto perché la condizione è forte.
- La FVD è il FID con un estrattore video e si minimizza; vede poco il
  disordine temporale quando le feature dipendono soprattutto dal contenuto,
  come quelle dell’I3D, e se il video si taglia in clip di 16 fotogrammi
  consecutivi non vede le incoerenze a più lungo raggio. In VBench ogni
  dimensione si massimizza, lo sfarfallio compreso, che vi entra rovesciato;
  un video fermo vince sulle dimensioni di coerenza temporale e perde sul grado
  di dinamica.
```

`````

Nel flow matching su un clip intero ogni fotogramma vede tutti gli altri,
passati e futuri, e ne condivide il livello di rumore. Un video generato mentre
scorre deve invece vedere solo il passato, e imparare a continuare quello che
ha prodotto da sé; come ci si arriva, e fino a che punto il risultato somigli a
un mondo interattivo, lo racconta la {doc}`sezione sul tempo reale
</GenerazioneVideo/generare-mentre-scorre>`.
