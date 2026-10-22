# Quando la diffusione incontra i Transformer: DiT

Il cuore di ogni modello di diffusione incontrato finora, dal DDPM del 2020
allo Stable Diffusion del 2022, è la U-Net {cite}`ronneberger2015u`,
un'architettura nata nel 2015 a Friburgo per segmentare cellule al microscopio,
come ricordiamo dalla {doc}`sezione su detection e segmentazione
</VisioneArtificiale/detection-segmentazione>`. Per anni nessuno l'ha
sostituita: le si è aggiunta un po’ di attenzione, le si è appesa l'etichetta
con il numero del passo, ma l'impalcatura (guardare da lontano e poi da vicino,
con i ponti diretti fra le due viste) è rimasta quella del microscopio.

Nel 2022 si cominciano a cercare alternative. A settembre Fan Bao e colleghi
propongono U-ViT, un Transformer che riceve come token i pezzetti dell'immagine
e insieme il tempo e la condizione, e della U-Net tiene soltanto i ponti lunghi
fra l'inizio e la fine della rete {cite}`bao2023all`. Alla fine dello stesso
anno William Peebles, allora dottorando a Berkeley, e Saining Xie, professore
alla New York University, fanno la domanda nella sua forma più netta: la U-Net
serve *davvero*? O anche qui, come era successo nel 2017 per la traduzione con
la caduta delle reti ricorrenti, vale il titolo di quel paper: *attention is
all you need* {cite}`vaswani2017attention`? La loro risposta si chiama **DiT**,
*Diffusion Transformer* {cite}`peebles2023scalable`, presentata alla
International Conference on Computer Vision del 2023. Il seguito ne misura il
peso: poco più di un anno dopo lo stesso Peebles, insieme a Tim Brooks, guida
un progetto chiamato Sora.

## Affettare la scheda: le tessere come parole

DiT non butta via tutto. Resta la diffusione latente di {doc}`Stable Diffusion
</ModelliDiffusione/stable-diffusion>`: un VAE comprime l'immagine in un
latente $\mathbf{z}$ più piccolo (la «scheda» compatta di Stable Diffusion), e
il processo di rumore lavora su $\mathbf{z}$, non sui pixel
{cite}`rombach2022high`. A cambiare è solo la rete
$\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t)$ che stima il rumore a ogni
passo: via la U-Net, dentro un Transformer. L'obiettivo di
addestramento e il campionatore restano identici.

Un Transformer, però, elabora sequenze di vettori, i token, mentre il latente
è un tensore $4\times32\times32$ organizzato in griglia. Come si passa
dall'uno all'altro? Con il procedimento del Vision Transformer, incontrato nel
{doc}`capitolo sui Transformer </Transformers/overview>`
{cite}`dosovitskiy2021image`: si divide la griglia in tessere quadrate, le
*patch*, e ciascuna diventa un token.

`````{tab} Elementare

La scheda dell'archivista è la stessa cosa che in Stable Diffusion, solo più
piccola: DiT viene addestrato su fotografie di lato dimezzato rispetto a quelle
di prima, quindi la scheda ha 32 caselle per lato invece di 64. In ogni casella
ci sono quattro numeri, come già là: i quattro valori con cui l'archivista
descrive quel pezzetto di quadro.

Il Vision Transformer ci ha insegnato il trucco per trasformare una griglia in
una frase: tagliarla in tessere, come un mosaico, e disporle in sequenza come
se fossero parole. Qui le tessere sono quadratini di 2 caselle per lato, quindi
ne vengono $32 : 2 = 16$ per lato e $16 \times 16 = 256$ in tutto: una "frase"
di 256 parole.

C'è però un'accortezza: mettendo le tessere in fila si perderebbe
l'informazione su dove stavano nella griglia, quindi a ciascuna si appiccica
un'etichetta che dice da che riga e da che colonna viene.

Quanto grosse tagliare le tessere è una scelta: a 4 caselle per lato le parole
sarebbero 64 invece di 256, e leggerle costerebbe quattro volte meno. È la
manopola con cui si decide quanto far lavorare la torre.

Da qui in poi il lavoro lo conosciamo dal capitolo sui Transformer: la torre,
con i suoi piani (in gergo si chiamano blocchi). Le tessere prendono il posto
che là avevano le parole: duecentocinquantasei in fila, e a ogni piano
ognuna si porta dietro i suoi appunti, una lista di numeri lunga sempre uguale.
A ogni piano, ogni tessera guarda *tutte* le altre (ad esempio, per capire
quanto rumore c'è sull'orecchio del gatto aiuta guardare anche la tessera con
la coda, dall'altra parte della scheda) e poi rielabora i propri appunti.
All'ultimo
piano, ogni tessera riconsegna la propria porzione di rumore stimato, e le
porzioni ricomposte formano la mappa completa: la torre, tutta insieme, *è* il
restauratore, e quella mappa è la sua risposta a ogni passo di pulitura.

La differenza con la U-Net è di principio. Le convoluzioni, cioè il modo di
guardare che rende una rete di visione una rete di visione, davano la
precedenza ai vicini di casa per costruzione; il Transformer, invece, non
privilegia nessuno, e dove conviene guardare lo impara durante
l'addestramento, tessera per tessera.

`````

`````{tab} Superiore

L'ingresso è il latente rumoroso $\mathbf{z}_t \in \mathbb{R}^{32 \times 32
\times 4}$ (immagini $256 \times 256$ compresse dal VAE con fattore $f = 8$).
Il *patchify* lo suddivide in patch quadrate di lato $p$ e proietta linearmente
ciascuna in un embedding di dimensione $d$: si ottiene una sequenza di $N =
(32/p)^2$ token (usiamo $N$, perché in questo capitolo $T$ è già il numero di
passi di diffusione; il paper fa il contrario, chiama $T$ i token e $N$ il
numero di strati, quindi le sue tabelle vanno lette con quel dizionario in
mano), a cui si somma un positional encoding (sinusoidale bidimensionale,
fisso) che ne registra la posizione nella griglia. Segue una pila di blocchi
Transformer del tutto standard (multi-head self-attention più MLP, con residual
e layer normalization, come nel capitolo sui Transformer) e una testa lineare
finale che da ogni token ricostruisce la sua patch di rumore stimato (nel DiT
originale, anche i parametri della covarianza), riassemblata poi in un tensore
della stessa forma dell'ingresso: la rete resta una
$\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t)$, cambia solo ciò che ha dentro.

Il lato $p$ della patch è l'iperparametro che regola il calcolo: con $p = 2$ i
token sono $N
= 256$, con $p = 4$ sono 64, con $p = 8$ sono 16. Dimezzare $p$ quadruplica i
token e con essi il costo di una passata a parità di parametri. Ad esempio, nel
paper la taglia XL passa da 29,1 Gflops con $p = 4$ a 118,6 con $p = 2$, cioè
un fattore $4{,}08$, e non di più. Il termine quadratico dell'attenzione c'è,
ma a queste lunghezze di sequenza è piccolo: per blocco pesa $2N^2 d$ contro i
$12Nd^2$ delle proiezioni e dell'MLP, cioè un rapporto $N/(6d)$ che per
DiT-XL/2, dove
la larghezza è $d = 1152$, fa $256/(6 \cdot 1152) \approx 3{,}7\%$ del termine
lineare, e quindi il $3{,}6\%$ del totale. Il rapporto però cresce con il
quadrato del lato: alla stessa larghezza, $512$ pixel portano $N$ a $1024$ e il
quadratico al $15\%$ del lineare, $1024$ pixel lo portano al $59\%$. Lì quel
termine diventa il vincolo, lo stesso problema da cui era partito il
{doc}`capitolo sull'attenzione lineare </AttenzioneLineare/overview>`. Da qui
la nomenclatura del paper: quattro taglie (DiT-S, B, L, XL) per tre patch (/8,
/4, /2), dodici modelli, che costituiscono l'esperimento del lavoro.

`````

## Il condizionamento entra dalle manopole: adaLN-zero

Manca un pezzo. Alla rete bisogna dire due cose che non stanno nell'immagine:
*a che punto della scala* sta lavorando (il numero del passo) e *che cosa* deve
disegnare (nel DiT originale la categoria dell'oggetto, nei suoi
discendenti un testo). Le due insieme si chiamano **condizionamento**: le
informazioni che orientano il lavoro senza far parte di ciò su cui si lavora.
Il nome viene
dalla probabilità condizionata, e il legame è letterale, perché ciò che la rete
impara è la distribuzione delle immagini *data* la richiesta.

La U-Net aveva un modo semplice di riceverle, appenderle come un'etichetta; con
le tessere in fila si aprono più strade, e Peebles e Xie le mettono a
confronto. Potrebbero accodare le due informazioni come parole in più della
frase, o farle consultare a parte, come fa Stable Diffusion con la richiesta
scritta. Vince invece la soluzione più economica, battezzata **adaLN-zero**: il
condizionamento non entra come token nella sequenza, ma modula la
normalizzazione di ogni blocco. Il nome è una sigla. *LN* è la *layer
normalization*, l'operazione con cui il Transformer, prima di ogni
sotto-strato, riporta ciascun token a media zero e varianza uno e poi lo
riscala con un guadagno e uno spostamento appresi; *ada* sta per adattiva,
perché guadagno e spostamento diventano funzioni del condizionamento invece di
parametri fissi; e *zero* è il modo in cui si parte, l'inizializzazione, che
conta più di quanto sembri.

`````{tab} Elementare

La torre ha una regia, e a ogni piano c'è un tecnico che la ascolta in
auricolare: il messaggio è lo stesso per tutti, ma ognuno ne ricava le
regolazioni buone per il proprio piano. La regia non suggerisce parole: dà
istruzioni di *regolazione*. Ogni piano ha due metà, la prima in cui ogni
tessera guarda le altre e la seconda in cui rielabora i propri appunti, e
ciascuna metà ha tre manopole, divise fra i suoi due capi. Due sono
all'ingresso, e dicono quanto alzare o abbassare il volume di ciò che arriva e
come spostarne il tono: cambiano il materiale che quella metà ha davanti da
leggere. La terza è all'uscita, e dice quanto di ciò che la metà ha prodotto va
aggiunto a quello che c'era prima: a fondo scala interviene a piena forza, a
zero il suo lavoro resta nel cassetto e il segnale in arrivo prosegue intatto.

Le istruzioni dipendono dal momento: se siamo ai primi passi della pulitura
(quasi tutto rumore) o agli ultimi ritocchi, se si sta disegnando un gatto o un
faro.

Il "-zero" del nome è un'astuzia da cantiere: il primo giorno di addestramento
tutte le manopole d'uscita sono a zero, e quindi nessun piano tocca niente. I
piani lavorano lo stesso, solo che nessuno li ascolta, e da lì imparano: a fine
giornata si guarda quanto sarebbe servito dare retta a ciascuno, e ognuno alza
la propria manopola di conseguenza. Sembra pigrizia, ma è il modo più stabile
di cominciare: nessun piano rovina il lavoro degli altri prima di aver imparato
il proprio.

È lo stesso trucco delle manopole a zero con cui ControlNet, nella
{doc}`sezione su Stable Diffusion
</ModelliDiffusione/stable-diffusion>`, affianca l'apprendista al restauratore:
un pezzo nuovo che all'inizio tace, e impara da quanto sarebbe servito
ascoltarlo.

`````

`````{tab} Superiore

Ricordiamo dal capitolo sui Transformer la layer normalization: normalizza
ogni token a media zero e varianza uno, poi riscala con un guadagno e un bias
appresi, uguali per tutti gli input. L’adaptive layer norm (adaLN) rende
guadagno e bias *funzioni del condizionamento*: ogni blocco ha il suo piccolo
MLP, che riceve $\mathbf{c} = \mathrm{emb}(t) + \mathrm{emb}(y)$ (embedding
sinusoidale del passo più embedding della classe) e produce, per ciascuno dei
due sotto-strati, tre vettori
$(\boldsymbol{\beta}_c, \boldsymbol{\gamma}_c, \boldsymbol{\alpha}_c)$:

$$
\mathrm{adaLN}(\mathbf{h}) = (1 + \boldsymbol{\gamma}_c) \odot
\mathrm{LN}(\mathbf{h}) + \boldsymbol{\beta}_c,
\qquad
\mathbf{x} \leftarrow \mathbf{x} + \boldsymbol{\alpha}_c \odot
\mathrm{Sottostrato}\big(\mathrm{adaLN}(\mathbf{x})\big),
$$

dove $\mathbf{h}$ e $\mathbf{x}$ sono lo stesso oggetto, la rappresentazione di
un token in ingresso al sotto-strato, $\mathrm{Sottostrato}$ è l'attenzione o
l'MLP del blocco, $\mathrm{LN}$ è la normalizzazione *senza* parametri appresi,
$\boldsymbol{\gamma}_c$ e $\boldsymbol{\beta}_c$ sono scala e traslazione
dettate dal condizionamento, $\odot$ è il prodotto elemento per elemento e
$\boldsymbol{\alpha}_c$ è un *gate* che dosa il contributo del sotto-strato
prima della somma residua. Il pedice $c$ non è decorativo e va letto:
$\boldsymbol{\alpha}_c, \boldsymbol{\beta}_c, \boldsymbol{\gamma}_c$ vengono
dal condizionamento e non hanno niente a che vedere con $\alpha_t$ e $\beta_t$,
che in questo capitolo sono lo schedule del rumore.

Il suffisso *zero* riguarda
l'inizializzazione. Azzerare $\boldsymbol{\alpha}_c$ è sufficiente, perché con
il gate a zero il ramo residuo non passa, e il blocco è l’identità qualunque
cosa facciano scala e traslazione. Il paper dichiara azzerato quello;
l'implementazione di riferimento azzera per comodità l'intero ultimo strato
dell'MLP, e la differenza non si vede, perché finché $\boldsymbol{\alpha}_c =
0$ il gradiente di scala e traslazione è nullo e a muoversi è solo il gate. Il
gradiente del gate no, perché vale il prodotto scalare fra ciò che il
sotto-strato ha prodotto e la direzione in cui l'uscita andrebbe spostata: il
blocco impara quanto aprirsi pur non contribuendo ancora, e la rete, che
comincia come un tubo vuoto, riceve gradiente su tutti i gate insieme.

Nell'implementazione di riferimento questo succede dal secondo passo, perché
anche lo strato d'uscita parte da zero e al primo passo si muove solo lui; nel
`MiniDiT` di «Un DiT in miniatura», che quello strato non lo azzera, succede
dal primo, come mostra la sua ultima stampa.

L'idea di modulare le normalizzazioni ha un
precedente illustre che conosciamo: l'AdaIN con cui StyleGAN
{cite}`karras2019style` inietta lo stile nel generatore.

Nelle ablazioni del
paper adaLN-zero batte sia i token in-context sia la cross-attention, e lo fa
con meno calcolo: $118{,}6$ Gflops contro i $119{,}4$ dell'in-context e i
$137{,}6$ della cross-attention, che è un sedici per cento in più. In Gflops,
quindi, la modulazione aggiunge quasi niente, perché è fatta di vettori e non di
token da far partecipare all'attenzione. In parametri no: gli MLP di
modulazione sono
$226$ dei $675$ milioni di DiT-XL/2, cioè un terzo della rete, e il confronto
delle ablazioni corre a passi di addestramento pari, non a parametri pari.

`````

## Più lavoro, più qualità (e non conta come)

Il risultato che ha fatto scuola vale più di qualunque punteggio ottenuto dal
modello vincente. Peebles e Xie non costruiscono un DiT solo: ne costruiscono
dodici, di taglie diverse, e li ordinano per il lavoro che fanno, non per la
grandezza. In quell'ordine la qualità delle immagini migliora con una
regolarità notevole, e non importa *da dove* quel lavoro venga. È l'eco delle
leggi di scala viste per i modelli di linguaggio (le regolarità con cui la
qualità di un modello cresce al crescere della sua taglia, dei suoi dati e del
calcolo speso) {cite}`kaplan2020scaling,hoffmann2022training`.

Si misurano tre grandezze. Il lavoro si conta in Gflops, i miliardi di
operazioni che costa una passata della rete, dall'ingresso all'uscita; la
grandezza in parametri, cioè i numeri interni della rete; la qualità con il
FID, una distanza fra l'insieme delle immagini generate e quello delle vere:
più è basso, meglio è.

Non ne segue che l'architettura non conti più, che è la conclusione più facile
da trarre e la sbagliata. Lo stesso lavoro contiene anche delle *ablazioni*
(gli esperimenti in cui si cambia un pezzo solo e si guarda che effetto fa) sul
modo di far entrare il condizionamento, e quel modo cambia la qualità in misura
tutt'altro che marginale; adaLN-zero vince quel confronto con meno calcolo
degli altri due. Le due affermazioni stanno insieme senza contraddirsi, a patto
di enunciarle con precisione: fra modelli costruiti tutti allo stesso modo, il
calcolo predice la qualità meglio della sola taglia o del solo numero di
tessere. D'altra parte, come è fatto un blocco del Transformer resta una
scelta, e una scelta sbagliata abbassa la qualità; la regolarità indica dove
spendere il prossimo Gflop, non che l'ingegneria sia finita.

In cima alla curva sta il modello più grande con le tessere più piccole, che
nel paper si chiama DiT-XL/2 (XL è la taglia della rete, il 2 è il lato della
tessera in caselle). È un Transformer da 675 milioni di parametri che genera
immagini a partire dalla categoria richiesta («un cane», «un faro»). Con la
classifier-free guidance e dopo sette milioni di passi di addestramento, il
modello arriva a un FID di $2{,}27$ su ImageNet $256\times256$. Questo
punteggio ne fa il migliore fra i modelli di diffusione allora noti, mettendolo
alla pari con la GAN più forte del momento, StyleGAN-XL ($2{,}30$).

Per confronto, il latent diffusion di Rombach e colleghi {cite}`rombach2022high`
si fermava a $3{,}60$ con $103{,}6$ Gflops (contro i $118{,}6$ di DiT),
mentre la
U-Net sui pixel di Dhariwal e Nichol {cite}`dhariwal2021diffusion` arrivava a
$3{,}94$ con 742 Gflops, e a $4{,}59$ con 1120 nella versione a stadio unico.

La U-Net, dunque, non era essenziale. Il suo vantaggio di partenza (sapere già
in fabbrica che i pixel vicini contano più di quelli lontani) si può comprare
con dati e calcolo, e a calcolo quasi pari il Transformer arriva più in alto;
che la distanza cresca con la scala, il lavoro non lo misura, perché una curva
di scala la traccia solo per i DiT. È la stessa storia già vista nel
{doc}`capitolo sui Transformer
</Transformers/overview>`, dove le reti di visione classiche avevano ceduto il
passo ai Vision Transformer, e adesso si ripete dentro la diffusione.

`````{tab} Elementare

Un risultato prezioso perché prevedibile: se so quanto
miglioro raddoppiando il lavoro, so anche se conviene raddoppiarlo, prima di
spendere i soldi.

Dodici torri, tutte per lo stesso lavoro ma di taglie diverse.
Le misure di torre sono quattro, e più grande è la torre più sono i piani e più
lunghi gli appunti; i modi di tagliare il mosaico sono tre, e più piccole sono
le tessere più ne entrano a ogni piano. Quattro per tre fa dodici.

Ora ordinale per il lavoro che fanno, non per la grandezza: quante operazioni
servono a far passare un'immagine dall'ingresso all'uscita. In quell'ordine, i
risultati migliorano quasi in linea retta. La sorpresa sta nel fatto che non
conta come quel lavoro sia stato speso; che chi lavora di più faccia meglio
sarebbe ovvio. Una torre piccola che legge tante tessere e una torre grande che
ne legge poche, se fanno la stessa quantità di lavoro, arrivano più o meno allo
stesso punto, e la coppia che Peebles e Xie mettono a confronto è fatta così.

Le due manopole che puoi girare (la taglia della torre e la misura delle
tessere) diventano una sola davanti al risultato, e restano due davanti al
computer: una torre grande e una fila lunga di tessere non gli chiedono la
stessa memoria né la stessa attesa. Si gira allora la manopola più adatta alla
propria macchina, tanto a decidere il risultato sarà il totale.

Con un limite, e nei dati si vede bene. Tagliando le tessere ancora più grosse,
otto caselle per lato, ne restano quattro per lato e sedici in tutto: una fila
cortissima, in cui a ciascuna resta poco da guardare. Quelle torri restano
indietro comunque, anche a parità di lavoro con le altre, e la regola vale
finché le tessere restano piccole.

Detta così sembra la fine dell'ingegneria, e non lo è. Nella stessa ricerca si
vede che *a parità di lavoro* il modo di dare le istruzioni alla torre (la
regia con l'auricolare, contro le alternative scartate) cambia parecchio il
risultato. Le due cose convivono: scelto un buon
modo di costruire la torre, da lì in poi conta quanto la fai lavorare.

`````

`````{tab} Superiore

L'affermazione va delimitata, perché è il tipo di regolarità che si generalizza
troppo in fretta. La correlazione fra Gflops e qualità è misurata
su una sola famiglia (i dodici DiT), su un solo compito (generazione
condizionata alla classe), a quattrocentomila passi di addestramento e senza
guidance. Il risultato principale del lavoro sta fuori da quella curva per due
ragioni insieme, e citarne una sola è l'errore più comune: è ottenuto *con* la
classifier-free guidance, e su un modello che ha continuato ad addestrarsi
fino a sette milioni di passi, cioè diciassette volte e mezzo il budget della
curva.

Il meccanismo che resta, e che ha superato la prova del tempo, è più modesto e
più utile dell'enunciato forte: fissato il disegno del blocco,
i Gflops predicono la qualità meglio del numero di parametri. Il paper misura
una correlazione forte, $-0{,}93$ fra Gflops e FID, e una correlazione forte
ammette eccezioni. Le eccezioni hanno un nome: sono le tessere da otto caselle,
un terzo della famiglia. A Gflops quasi uguali ($1{,}41$ contro $1{,}42$)
DiT-S/4 fa $100{,}4$ e DiT-B/8 fa $122{,}7$; e il caso che non lascia scampo è
DiT-XL/8, che spende il ventidue per cento in più di DiT-S/2 e arriva trentotto
punti di FID più indietro ($106{,}4$ contro $68{,}4$). La coppia che il paper
porta come esempio, DiT-S/2 contro DiT-B/4 ($68{,}40$ e $68{,}38$), è fra
quelle che si equivalgono.

E due delimitazioni sugli iperparametri. Profondità e larghezza l'esperimento
non le separa mai, perché le quattro taglie le fissano insieme seguendo le
configurazioni del Vision Transformer: ciò che viene messo a confronto è la
taglia contro il lato della tessera. Profondità, larghezza e lato della tessera
non si equivalgono in memoria né in latenza. Si può pianificare meno di quanto
dica l'enunciato forte: dentro la parte della famiglia in cui la regola tiene
si sceglie l'iperparametro che conviene sull'hardware disponibile, sapendo
che a decidere sarà il totale.

`````

## Dal fotogramma al minuto: la diffusione sui video

Che questa storia conti anche fuori dai laboratori lo dice il video. Il 15
febbraio 2024 OpenAI presenta **Sora**, un modello che da una descrizione
scritta genera filmati fino a un minuto, accompagnato da un rapporto tecnico
dal titolo programmatico: *Video generation models as world simulators*, cioè
«i modelli che generano video come simulatori del mondo»
{cite}`brooks2024video`.

Sul piano dell'architettura il rapporto dichiara poco, e in modo preciso. Sora
«è un diffusion transformer». I video vengono compressi da una rete in schede,
come le fotografie in Stable Diffusion, e le schede vengono tagliate in
**spacetime patches**, tessere che si estendono nello spazio *e nel tempo*: un
quadratino di immagine per un pezzetto di durata. Quelle tessere vengono
messe in fila e date da leggere a un Transformer come prima. L'addestramento
avviene su video e immagini di durate, risoluzioni e proporzioni diverse.

La qualità cresce poi «sensibilmente» al crescere del lavoro speso ad
addestrare: il confronto mostrato è fra lo stesso modello addestrato con il
lavoro base, poi con quattro volte tanto, poi con trentadue volte tanto. Questo
lavoro si misura in un altro modo rispetto ai dodici DiT. Là si contavano le
operazioni di una passata sola e la qualità era un numero, il FID; qui si conta
il calcolo speso per addestrare, e i tre filmati si confrontano a occhio. Sono
due facce della stessa regolarità, le leggi di scala dei modelli di linguaggio,
misurate su due assi diversi. L'impostazione DiT viene estesa di una dimensione:
dove il Vision Transformer affettava un'immagine, qui si affetta un blocco di
fotogrammi.

Il rapporto *non* dice quanti parametri abbia il modello, su quali dati
sia stato addestrato, come sia fatto nel dettaglio. È un documento aziendale
con dimostrazioni scelte da chi lo ha scritto, non un articolo scientifico che
altri ricercatori abbiano controllato prima della pubblicazione.

Il titolo rilancia una tesi che il rapporto stesso incrina, perché ammette di
non riprodurre correttamente la fisica delle interazioni più elementari.
L'esempio che sceglie è il vetro che si rompe, e il filmato mostrato non
rispetta il comportamento di un bicchiere reale. Generare video credibili
significa aver *capito* il mondo, o solo averne imparato le apparenze? È la
domanda della {doc}`sezione sui simulatori del mondo
</WorldModels/simulatori-e-dibattito>`, dove i video generativi sono discussi
come candidati simulatori. Sul piano dell'architettura il fatto è questo: le
tessere del Vision Transformer, passate per DiT, sono arrivate al cinema.

## Due corsie e linee dritte: MM-DiT e rectified flow

Il 2024 è anche l'anno in cui l'architettura DiT arriva ai modelli di punta
che disegnano su richiesta. Patrick Esser, Robin Rombach e il resto del gruppo
di Stable Diffusion pubblicano il lavoro dietro **Stable Diffusion 3**
{cite}`esser2024scaling`, presentato a ICML 2024 e premiato tra i migliori
articoli della conferenza. Le novità sono tre, in ordine di profondità
crescente.

La prima è l'architettura, battezzata **MM-DiT** (*multimodal DiT*, dove le due
modalità sono appunto il testo e l'immagine), e riguarda il rapporto fra le
due. In Stable Diffusion il testo restava esterno alla rete di diffusione. A
leggerlo era la rete di CLIP {cite}`radford2021learning`, un modello del
{doc}`capitolo su visione e linguaggio </VisioneLinguaggio/overview>`
addestrato a mettere in corrispondenza immagini e didascalie; trasformava la
richiesta una volta per tutte in una sequenza di vettori, e da lì in poi la
U-Net poteva soltanto *consultarla* tramite cross-attention: l'informazione
andava in un senso solo.

In MM-DiT il testo e l'immagine diventano due sequenze di token alla pari: due
corsie che conservano ciascuna i propri pesi ma si incontrano nell'attenzione,
così che a ogni blocco le parole guardino le tessere dell'immagine e
viceversa. La famiglia arriva fino a 8 miliardi di parametri, e la scala si
comporta anche qui in modo regolare e prevedibile. L'impostazione ha fatto
scuola: la riprende, tra gli altri, FLUX (2024), del gruppo di autori originali
di Stable Diffusion.

`````{tab} Elementare

Torniamo al restauratore con la commissione del cliente sul tavolo. In
Stable Diffusion il rapporto fra i due era a senso unico: la commissione era
scritta una volta per tutte, il restauratore ci dava un'occhiata quando gli
serviva, e la commissione non cambiava mai. Un consulente che non entra nella
stanza: parla solo quando lo interrogano, e non sente quello che succede sul
tavolo.

MM-DiT fa entrare il consulente nella stanza. Le parole della richiesta e le
tessere dell'immagine si siedono in due file allo stesso
tavolo, e a ogni piano della torre si guardano a vicenda: le tessere guardano
le parole, come già facevano in Stable Diffusion, ma adesso anche le parole
guardano le tessere. La parola
«acquerello» può accorgersi di che cosa sta effettivamente succedendo nel
disegno, e regolarsi. Ciascuna delle due file conserva un mestiere proprio (chi
elabora parole e chi elabora tessere non fa lo stesso lavoro, e ha strumenti
suoi), ma si parlano da pari.

La regia con l'auricolare non se ne va: a ogni piano continua a dire a che
punto della pulitura siamo e che aria generale deve avere il quadro. Quell'aria
generale è la richiesta riassunta in un pugno di numeri, che è cosa diversa
dalle sue parole una per una; le parole per esteso sono adesso l'altra fila al
tavolo, e non arrivano più dall'auricolare.

`````

`````{tab} Superiore

Le due corsie hanno pesi separati per modalità (proiezioni e MLP distinti) e
condividono solo l'operazione di attenzione, calcolata sulla concatenazione
delle due sequenze:

$$
\mathrm{softmax}\!\left(
\frac{[\mathbf{Q}_{\mathrm{img}};\mathbf{Q}_{\mathrm{txt}}]\,
[\mathbf{K}_{\mathrm{img}};\mathbf{K}_{\mathrm{txt}}]^\top}{\sqrt{d_k}}
\right)[\mathbf{V}_{\mathrm{img}};\mathbf{V}_{\mathrm{txt}}],
$$

dove ogni matrice è proiettata con i pesi della propria modalità e
$[\,\cdot\,;\,\cdot\,]$ concatena le righe, cioè le posizioni. La matrice dei
punteggi ha quattro blocchi: i due sulla diagonale sono l'attenzione di ogni
modalità su sé stessa, i due fuori diagonale quella del testo sull'immagine e
dell'immagine sul testo, e il costo cresce con il quadrato della lunghezza
totale. È il compromesso che distingue MM-DiT sia da un Transformer unico sui
token concatenati (che tratterebbe testo e immagine come se avessero la stessa
statistica) sia dalla cross-attention di Stable Diffusion (dove il flusso è
unidirezionale e gli embedding del testo restano quelli fissati dall'encoder).
Il condizionamento su passo e istruzioni globali resta affidato ad adaLN, come
nel DiT originale; a cambiare è soltanto lo statuto del testo, che smette di
essere condizionamento e diventa una delle due sequenze elaborate.

`````

La seconda novità è la meno appariscente e riguarda il VAE, non la rete di
diffusione: il latente passa da 4 a 16 canali, cioè da quattro a sedici numeri
per ogni casella della scheda. Corregge direttamente il limite discusso a
proposito di Stable Diffusion, cioè il soffitto che il compressore impone alla
qualità finale; a parità di fattore di compressione, più canali significano una
ricostruzione nettamente migliore, e nel lavoro di Esser e colleghi il FID di
ricostruzione dell'autoencoder scende da $2{,}41$ con quattro canali a $1{,}06$
con sedici. È una delle ragioni per cui le scritte e i dettagli fini smettono
di essere il difetto caratteristico della famiglia. Il VAE era stato presentato
come un accorgimento per risparmiare calcolo; quei due numeri misurano quanto
pesasse anche sulla qualità.

La terza novità riguarda il modo stesso di andare dal rumore all'immagine.
Stable Diffusion 3 abbandona la catena di rumore di DDPM per il **rectified
flow** {cite}`liu2023rectified`, letteralmente «flusso raddrizzato», che è una
variante particolare di una famiglia di metodi più generale, il *flow matching*
di Yaron Lipman e colleghi {cite}`lipman2023flow`, che la {doc}`sezione
dedicata </ModelliDiffusione/flow-matching>` ha già presentato. Ci interessa
per un effetto: cambia quanti passi servono nel viaggio dal rumore
all'immagine.

```{figure} ../figures/flow-matching-traiettorie-dritte.svg
:name: fig-traiettorie-dritte
:alt: "Due percorsi fra gli stessi due estremi, un cerchio «rumore» a sinistra e un cerchio «dati, immagine» a destra. In alto, in terracotta, quello della diffusione: una linea che serpeggia, con nove puntini a segnare le fermate. In basso, in teal, quello del flow matching: una freccia dritta con quattro sole fermate."
:width: 92%

Stessi estremi, due strade. Su una linea dritta si può camminare a grandi
falcate senza uscire di strada; su una tortuosa no, e i passi devono essere
tanti e piccoli.
```

{numref}`fig-traiettorie-dritte` mostra la differenza e dice dove sta il
guadagno: nel tempo necessario a produrre l'immagine, non nella sua qualità. Il
numero di passi necessari dipende soprattutto da quanto la strada curvi, e
meno da quanto la meta sia lontana (su una retta un solo passo arriva dove deve
arrivare): raddrizzarla non cambia dove si arriva, cambia quante volte bisogna
fermarsi a chiedere la direzione.

`````{tab} Elementare

Il restauratore va dal rumore all'immagine per mille tappe brevi, e la strada
che percorre serpeggia: la direzione da prendere cambia continuamente, perché
il sentiero che la catena di rumore ha tracciato all'andata curva da sé, e il
rimescolamento a ogni tappa lo scuote ancora. Per questo le tappe devono essere
tante e corte: chi tiene la direzione per troppo tempo esce di strada.

Stable Diffusion 3 prende la strada del flusso rettificato, quella della
{doc}`sezione sul flow matching </ModelliDiffusione/flow-matching>`: una linea
dritta fra la scheda tutta rumore e la scheda dell'immagine finita, e alla
rete si insegna una cosa sola, in ogni punto della linea *in che direzione si
cammina*.

Seguiamo un numero solo. Non un pixel, che qui non si tocca: uno
dei numeri di una casella della scheda, su un intervallo che per comodità
prendiamo da 0 a 1. Nella scheda tutta rumore vale 0,2; nella scheda
dell'immagine finita vale 0,8. A metà strada vale la media: 0,5. La marcia,
quindi, va sempre in su, e di quanto lo dice la differenza fra i due estremi:
$0{,}8 - 0{,}2 = 0{,}6$ da guadagnare in tutto. Se decidiamo di farlo in dieci
tappe, ogni tappa sale sempre della stessa quantità, $0{,}6 : 10 = 0{,}06$.
Nessuna sorpresa lungo la strada, perché la strada è dritta.

Su una strada così non serve fermarsi mille volte a ricontrollare la mappa, e
nemmeno le cinquanta delle scorciatoie: ne bastano poche decine. Dritta, però,
per modo di dire, come si è visto con il flusso rettificato: le strade che la
rete impara sono la media di milioni di linee che si incrociano, e curvano un
po'. È la differenza fra un tornante di montagna e una provinciale con qualche
curva. Raddrizzarle ancora, rifacendo l'addestramento sulle coppie che la rete
stessa collega, porterebbe fino a una fermata sola; Stable Diffusion 3 non lo
fa, e si ferma a qualche decina.

`````

`````{tab} Superiore

Si fissa un tempo continuo $t \in [0, 1]$, al posto del passo intero della
catena (dato pulito a $t = 0$, rumore puro a $t = 1$, coerente con il verso
adottato per la diffusione; Liu e colleghi lo scrivono all'inverso, quindi
aprendo il loro articolo la scala va rovesciata), e si collega ogni dato al
rumore con un’interpolazione lineare:

$$
\mathbf{x}_t = (1 - t)\,\mathbf{x}_0 + t\,\boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),
$$

dove $\mathbf{x}_0$ è un dato del training set (in Stable Diffusion 3 un
latente, cioè lo stesso oggetto che nella diffusione latente si scrive
$\mathbf{z}$) ed $\boldsymbol{\epsilon}$ il rumore gaussiano. Lungo questo
segmento la velocità è costante: $\mathrm{d}\mathbf{x}_t/\mathrm{d}t =
\boldsymbol{\epsilon} - \mathbf{x}_0$. Il campo appreso punta dunque *verso il
rumore*, e la generazione lo percorre all'indietro: perciò, camminando nel
verso della generazione, lo stesso numero va letto col segno opposto. Il
modello $\mathbf{v}_\theta(\mathbf{x}_t, t)$ viene addestrato a stimarla:

$$
\mathcal{L}_{\mathrm{CFM}} = \mathbb{E}_{\mathbf{x}_0,\, \boldsymbol{\epsilon},\, t}
\Big[\, \big\lVert \mathbf{v}_\theta(\mathbf{x}_t, t) - (\boldsymbol{\epsilon} - \mathbf{x}_0) \big\rVert^2 \,\Big],
$$

dove $\mathbf{v}_\theta$ è il campo di velocità appreso, con parametri
$\theta$, e l'attesa è su un dato, un rumore e un istante estratti a caso.
L'obiettivo è quello condizionato, l'unico che si sa calcolare, e ha la stessa
struttura da «regressione con bersaglio noto» della loss di DDPM, con la
velocità al posto del rumore. Per generare si integra l'ODE
$\mathrm{d}\mathbf{x}/\mathrm{d}t = \mathbf{v}_\theta(\mathbf{x}, t)$ da $t =
1$ (rumore) a $t = 0$, per esempio con passi di Eulero: niente termine
stocastico, come già nel campionatore DDIM, ma qui il campo dell'ODE è appreso
direttamente, non ricavato a posteriori da un predittore di rumore.

Perché bastano meno passi? Il campo appreso in un punto è la media delle
velocità di tutte le coppie $(\mathbf{x}_0, \boldsymbol{\epsilon})$ le cui
interpolazioni passano di lì: le traiettorie marginali non sono rette, e
risultano molto meno curve di quelle dell'ODE associata alla diffusione
variance-preserving, e l'errore di discretizzazione a parità di passi è più
piccolo.

Il flow matching di Lipman e colleghi {cite}`lipman2023flow` mostra
inoltre che la famiglia dei cammini gaussiani è generale e include i cammini
della diffusione come caso particolare: DDPM diventa un punto in uno spazio di
scelte progettuali. Esser e colleghi {cite}`esser2024scaling` confrontano
sistematicamente le varianti e adottano il rectified flow con un campionamento
di $t$ concentrato sugli istanti intermedi, i più difficili (una logit-normale
centrata su $t = 0{,}5$): in pratica Stable Diffusion 3 genera in poche decine
di passi.

Il *reflow*, cioè la rettificazione ripetuta, riaddestra sulle coppie
generate, raddrizza ancora le traiettorie e in Liu e colleghi porta al passo
singolo. Stable Diffusion 3 non lo fa: addestra un flusso rettificato una volta
sola.

`````

## Un DiT in miniatura

Come per la spirale di punti con cui abbiamo smontato DDPM, il modo migliore di
fissare l'architettura è costruirla in piccolo. Costruiamo quindi un DiT
completo ma in miniatura: si taglia una scheda finta in tessere, le si fa
passare per i blocchi del Transformer con l'attenzione e la modulazione del
condizionamento, e si ricompone il risultato. Le stampe da guardare sono
quattro: la forma del risultato, il numero di pesi, la prova che i parametri di
modulazione partono da zero e quali di essi cominciano a muoversi per primi.
Non c'è addestramento (servirebbero i dati e le ore di calcolo su GPU), ma tutte
le misure tornano, e il ciclo di addestramento sarebbe *lo stesso* visto per
DDPM: cambia solo la rete interrogata.

```python
import math
import torch
from torch import nn

torch.manual_seed(0)

def embedding_tempo(t, dim=128):
    """Embedding sinusoidale del passo t: da (B,) a (B, dim)."""
    freq = torch.exp(-math.log(10000.0) * torch.arange(dim // 2) / (dim // 2))
    ang = t.float().unsqueeze(1) * freq.unsqueeze(0)     # (B, dim/2)
    return torch.cat([ang.sin(), ang.cos()], dim=1)      # (B, dim)

class BloccoDiT(nn.Module):
    """Attenzione + MLP, con modulazione adaLN-zero dal condizionamento."""
    def __init__(self, d=128, teste=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)  # LN "nuda"
        self.attn = nn.MultiheadAttention(d, teste, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))
        # dal condizionamento: shift, scale e gate per i due sotto-strati
        self.manopole = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.manopole.weight)   # adaLN-ZERO: blocco = identita'
        nn.init.zeros_(self.manopole.bias)

    def forward(self, x, c):
        # x: (B, N, d) token del latente; c: (B, d) tempo + classe
        b1, g1, a1, b2, g2, a2 = self.manopole(c).chunk(6, dim=1)  # 6 x (B, d)
        h = self.norm1(x) * (1 + g1.unsqueeze(1)) + b1.unsqueeze(1)
        att, _ = self.attn(h, h, h, need_weights=False)
        x = x + a1.unsqueeze(1) * att            # gate a1: vale 0 all'inizio
        h = self.norm2(x) * (1 + g2.unsqueeze(1)) + b2.unsqueeze(1)
        x = x + a2.unsqueeze(1) * self.mlp(h)    # gate a2, idem
        return x

class MiniDiT(nn.Module):
    """DiT minimale: patchify, blocchi Transformer, patch di rumore in uscita."""
    def __init__(self, canali=4, lato=32, patch=2, d=128, blocchi=4, classi=10):
        super().__init__()
        self.canali, self.lato, self.patch = canali, lato, patch
        n_token = (lato // patch) ** 2                       # (32/2)^2 = 256
        self.patchify = nn.Conv2d(canali, d, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, n_token, d))  # posizioni apprese
        self.emb_classe = nn.Embedding(classi, d)
        self.blocchi = nn.ModuleList([BloccoDiT(d) for _ in range(blocchi)])
        self.finale = nn.Linear(d, patch * patch * canali)   # token -> sua patch

    def forward(self, z, t, y):
        # z: (B, 4, 32, 32) latente rumoroso; t: (B,) passo; y: (B,) classe
        x = self.patchify(z)                     # (B, d, 16, 16)
        x = x.flatten(2).transpose(1, 2)         # (B, 256, d): i token
        x = x + self.pos
        c = embedding_tempo(t, x.shape[-1]) + self.emb_classe(y)   # (B, d)
        for blocco in self.blocchi:
            x = blocco(x, c)
        x = self.finale(x)                       # (B, 256, patch*patch*4)
        # ricompone le patch: l'uscita ha la stessa forma dell'ingresso
        B, g, p, C = z.shape[0], self.lato // self.patch, self.patch, self.canali
        x = x.view(B, g, g, p, p, C)             # (B, 16, 16, 2, 2, 4)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, self.lato, self.lato)
        return x                                 # (B, 4, 32, 32): rumore stimato

modello = MiniDiT()
z = torch.randn(2, 4, 32, 32)      # due latenti fittizi, come quelli del VAE
t = torch.randint(0, 1000, (2,))   # un passo di rumore per ciascuno
y = torch.randint(0, 10, (2,))     # una classe per ciascuno
print(modello(z, t, y).shape)      # torch.Size([2, 4, 32, 32])
print(sum(p.numel() for p in modello.parameters()))  # 1225616: ~1.2 milioni

# la verifica di adaLN-zero, che le due righe sopra non fanno: a pesi
# appena inizializzati ogni blocco deve essere l'identita', per qualunque c
blocco, x, c = modello.blocchi[0], torch.randn(2, 256, 128), torch.randn(2, 128)
print((blocco(x, c) - x).abs().max().item())   # 0.0

# e chi comincia a imparare: dopo un passo all'indietro, quali delle sei
# manopole del primo blocco (b1, g1, a1, b2, g2, a2) ricevono gradiente
modello(z, t, y).pow(2).mean().backward()
g = modello.blocchi[0].manopole.weight.grad.view(6, 128, 128).abs().sum((1, 2))
print([int(v > 0) for v in g])                 # [0, 0, 1, 0, 0, 1]: i due gate
```

Le ultime due stampe dicono più delle prime due. La prima stampa le misure del
risultato, e resterebbe identica anche togliendo tutti e quattro i blocchi, o
cambiando il passo e la classe: dice che i tubi sono collegati, non che l'acqua
ci passi giusta. La seconda conta i parametri, e almeno cambierebbe togliendo
dei blocchi, ma non distingue una rete che funziona da una rotta.

La terza stampa controlla la promessa di adaLN-zero. I parametri di modulazione
partono da zero, quindi ogni blocco dovrebbe restituire il proprio ingresso, per
qualunque condizionamento arrivi: la differenza fra uscita e ingresso deve
essere zero, e la riga stampa zero. Chi scrive codice fa di solito questo
controllo a mente (in gergo un *desk-check*); qui è eseguibile, e se quello zero
smette di uscire l'inizializzazione si è rotta. La quarta controlla l'altra metà
della promessa: dopo un passo all'indietro, dei sei parametri di modulazione del
primo blocco ricevono una correzione soltanto i due gate, quelli che dosano
quanto il blocco contribuisce. Gli altri quattro restano fermi finché i primi
due sono a zero, perché il lavoro del blocco non arriva ancora all'uscita.

Il DiT vero differisce dal nostro nei numeri (nella taglia più grande, 28
blocchi e una dimensione di 1152 per token, contro i nostri 4 e 128), nel fatto
che predice, accanto al rumore, anche quanta incertezza lasciarsi attorno a ogni
passo (i parametri della covarianza), e nel modo di dire a ogni tessera dove si
trova nella griglia, che nel DiT è una tabella di posizioni fissata in partenza
invece dei nostri numeri appresi. La logica è la stessa.

## Addestrare e usare

Addestrare questi modelli è fuori dalla portata individuale, e la direzione è
quella di un rincaro: già lo Stable Diffusion del 2022 chiedeva le
centocinquantamila ore di calcolo che sappiamo, i suoi successori a miliardi di
parametri ne chiedono un multiplo che nessuno dichiara, e su Sora OpenAI
non pubblica né costi né dimensioni. La grandezza è diventata un ingrediente
della ricetta, e le curve di Peebles e Xie la misurano.

Usare un modello già addestrato, l'inferenza, è un'altra storia: si scaricano i
pesi, si fanno girare sulla GPU di un computer da videogiochi, e il rectified
flow ha reso la generazione più veloce, non più lenta. L'asimmetria vista per
Stable Diffusion (addestrare è per pochi, usare è per molti) si è accentuata in
entrambe le direzioni.

Nel 2015 la diffusione nasce da un'analogia termodinamica; nel 2024 genera un
minuto di video da una frase. In mezzo non c'è un colpo di genio isolato: ci
sono quattro mattoni presi da scaffali diversi. L'autoencoder variazionale è
del 2014, ed era nato per inventare immagini nuove e non per comprimerle,
campionando una scheda a caso e facendola ridipingere al copista. Qui gli
tocca il ruolo del compressore, e a inventare pensa qualcun altro.

La U-Net è del 2015 ed era nata per i microscopi
{cite}`ronneberger2015u`. L'attenzione
nasce nel 2014 per tradurre da una lingua all'altra {cite}`bahdanau2015neural`
e diventa protagonista nel 2017 {cite}`vaswani2017attention`. Le tessere sono
quelle di un Vision Transformer del 2021 nato per riconoscere il contenuto
delle fotografie {cite}`dosovitskiy2021image`. Tre di questi pezzi non erano
nati per generare immagini, e il quarto, il VAE, era nato per generarle e qui
comprime; tutti e quattro sono stati ricombinati con pazienza da gruppi diversi
in anni diversi.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- DiT manda in pensione la rete di visione e mette al suo posto la torre del
  capitolo sui Transformer. La scheda dell'archivista si taglia a
  tessere, le tessere si mettono in fila come parole, e ogni tessera guarda
  tutte le altre invece dei soli vicini di casa. L'archivista e il lavoro sulle
  schede compresse restano quelli di Stable Diffusion.
- Le istruzioni (a che punto della pulitura siamo, che cosa disegnare) non
  entrano nella conversazione: arrivano da una regia che ogni piano ascolta e
  traduce nelle proprie manopole. E il primo giorno le manopole d'uscita sono
  a zero, così ogni piano parte lasciando passare tutto, lavora senza che
  nessuno lo ascolti, e impara strada facendo quanto farsi sentire.
- Il risultato che ha fatto scuola: ordinando dodici modelli (quattro
  taglie di torre per tre misure di tessera) per
  il lavoro che fanno, la qualità migliora quasi in linea retta, e non
  conta *come* quel lavoro sia stato speso, purché le tessere non si taglino
  troppo grosse. Non vuol dire che l'architettura
  non conti: a parità di lavoro, il modo di dare le istruzioni cambia ancora
  molto. Vuol dire che, scelto un buon disegno, da lì in poi comanda il
  calcolo.
- Sora applica la stessa ricetta ai video, tagliando tessere che si
  estendono nello spazio *e nel tempo*. Se questo basti a dire che ha «capito»
  il mondo è la domanda del {doc}`capitolo sui World Model </WorldModels/overview>`.
- Stable Diffusion 3 fa tre cose: mette testo e immagine allo stesso
  tavolo invece che uno a consulenza dell'altro, dà all'archivista quattro
  volte più spazio per le sue schede, e sostituisce il sentiero tortuoso con
  una linea dritta, che si percorre a grandi falcate e quindi in molti
  meno passi. Dritta per modo di dire: le strade che la rete impara sono la
  media di milioni di linee che si incrociano, quindi qualche fermata serve
  ancora.
- Allenare questi modelli resta roba da centri di calcolo; usarli no. Ed è
  una storia di mattoni ricombinati, non di rivoluzioni improvvise.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- DiT {cite}`peebles2023scalable` sostituisce la U-Net con un
  Transformer: il latente si affetta in patch-token come nel ViT
  {cite}`dosovitskiy2021image`, l'attenzione rimpiazza le convoluzioni; il
  VAE e la diffusione nel latente restano quelli di
  {cite}`rombach2022high`.
- Il condizionamento su $t$ e classe entra via adaLN-zero: scala,
  traslazione e gate della layer norm generati da un MLP del condizionamento,
  uno per blocco, con inizializzazione a zero: a fare del blocco l'identità è
  il gate.
- Risultato chiave: dentro la famiglia DiT la qualità scala con i Gflops in
  modo regolare (correlazione $-0{,}93$), quasi comunque li si spenda, e le
  leggi di scala {cite}`kaplan2020scaling` arrivano alla diffusione. Il «quasi»
  sono le patch da otto, che a Gflops pari restano indietro di decine di punti
  di FID. Non è la fine dell'architettura: le ablazioni dello stesso lavoro
  mostrano che il disegno del blocco cambia ancora molto il risultato, e che il
  disegno vincente è anche fra i più economici in Gflops.
- Sora {cite}`brooks2024video` dichiara un diffusion transformer su
  *spacetime patches* di video compressi, con qualità che cresce col calcolo
  speso ad addestrare, che è un asse diverso da quello dei dodici DiT:
  l'architettura DiT estesa al tempo. Se ciò faccia dei video
  generativi dei "simulatori di mondo" è la domanda del capitolo sui World
  Model.
- Stable Diffusion 3 {cite}`esser2024scaling` cambia tre cose: l'MM-DiT
  (testo e immagine come due flussi di token alla pari), il latente da 4 a 16
  canali (che alza il soffitto di ricostruzione discusso per Stable
  Diffusion) e il passaggio al rectified flow {cite}`liu2023rectified`,
  della famiglia del *flow matching* {cite}`lipman2023flow`: interpolazioni
  lineari dato–rumore, una velocità appresa per regressione, generazione
  integrando un'ODE in poche decine di passi. Le traiettorie marginali però
  restano curve, e senza il *reflow* i passi non scendono a uno.
- Addestrare resta un affare da data center; usare no: i pesi aperti e
  il latente compresso tengono l'inferenza alla portata di una GPU
  domestica. La storia del capitolo è ricombinazione di mattoni noti, non
  una rivoluzione improvvisa.
```

`````

La ricetta si riassume in tre operazioni (comprimere, sporcare di rumore,
insegnare a ripulire), e l'abitudine con cui l'abbiamo letta, cercare
nell'architettura nuova i mattoni vecchi, tornerà utile quando Sora ricomparirà
fra i {doc}`simulatori del mondo </WorldModels/simulatori-e-dibattito>`. Il
rectified flow riduce il numero di passi raddrizzando la strada, cioè
cambiando il modello. La {doc}`sezione sui campionatori veloci
</ModelliDiffusione/campionatori-veloci>` pone la domanda inversa: quanto si
può ridurre il numero di passi con la rete che si ha già, senza riaddestrare
niente, scegliendo come percorrere la strada.
