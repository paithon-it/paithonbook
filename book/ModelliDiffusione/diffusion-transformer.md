# Quando la diffusione incontra i Transformer: DiT

C'è un'ironia nascosta nella storia raccontata fin qui. Il cuore di ogni
modello di diffusione che abbiamo incontrato (il DDPM del 2020, lo Stable
Diffusion del 2022) è la U-Net {cite}`ronneberger2015u`: un'architettura nata
nel 2015 a Friburgo per **segmentare cellule al microscopio**, come ricordiamo
dal capitolo sulla visione artificiale. Per anni nessuno l'ha messa in
discussione: le si è aggiunta un po’ di attenzione, le si è appesa l'etichetta
con il numero del passo, ma l'impalcatura (guardare da lontano e poi da vicino,
con i ponti diretti fra le due viste) è rimasta quella del microscopio.

Alla fine del 2022 William Peebles, allora dottorando a Berkeley, e Saining
Xie, professore alla New York University, si fanno la domanda che a quel punto
del libro dovrebbe suonare familiare: la U-Net serve *davvero*? O anche qui, come
era successo nel 2017 per la traduzione con la caduta delle reti ricorrenti,
vale il titolo di quel paper: *attention is all you need*
{cite}`vaswani2017attention`? La loro risposta si chiama **DiT**, *Diffusion
Transformer* {cite}`peebles2023scalable`, presentata alla International
Conference on Computer Vision del 2023. E la storia ha un seguito che ne
misura il peso: poco più di un anno dopo ritroveremo lo stesso Peebles,
insieme a Tim Brooks, alla guida di un progetto chiamato Sora.

## Affettare la scheda: le tessere come parole

DiT non butta via tutto. Il trasloco nello spazio latente della sezione
precedente resta: c'è ancora l'archivista (il VAE che comprime le immagini in
schede compatte) e la diffusione lavora ancora sulle schede, non sui pixel
{cite}`rombach2022high`. A cambiare è solo *chi* indovina il rumore a ogni
passo: via la U-Net, dentro un Transformer. Il mestiere del restauratore resta
identico, cambia la persona che lo esercita.

Un Transformer, però, mangia sequenze: parole in fila, una dopo l'altra, che in
gergo si chiamano **token**. Una scheda invece è una griglia di caselle. Come
si dà in pasto una griglia a chi sa leggere solo in fila? La mossa è già nel
nostro repertorio: è la stessa, identica, del Vision Transformer incontrato
nel capitolo sui Transformer {cite}`dosovitskiy2021image`.

`````{tab} Elementare

La scheda dell'archivista è la stessa cosa della sezione precedente, solo più
piccola: DiT viene addestrato su fotografie di lato dimezzato rispetto a quelle
di prima, quindi la scheda ha 32 caselle per lato invece di 64. In ogni casella
ci sono quattro numeri, come già là: sono i quattro valori con cui l'archivista
descrive quel pezzetto di quadro, e quanti siano è una scelta di progetto (con
più numeri per casella la scheda descrive meglio, e pesa di più).

Il Vision Transformer ci ha insegnato il trucco per trasformare una griglia in
una frase: tagliarla in **tessere**, come un mosaico, e mettere le tessere in
fila come se fossero parole. Qui le tessere sono quadratini di 2 caselle per
lato, quindi ne vengono $32 : 2 = 16$ per lato e $16 \times 16 = 256$ in tutto:
una "frase" di 256 parole. Con un'accortezza: mettendo le tessere in fila si
perderebbe l'informazione su dove stavano nella griglia, e allora a ciascuna si
appiccica un'etichetta che dice da che riga e da che colonna viene.

Da qui in poi il lavoro lo conosciamo dal capitolo sui Transformer: la torre di
lettori, con i suoi piani (in gergo si chiamano **blocchi**, ed è il nome che
compare nel codice più avanti). I lettori sono le tessere: a ogni piano ce n'è
uno per tessera, duecentocinquantasei in fila, e ciascuno tiene i suoi appunti,
una lista di numeri lunga sempre uguale. A ogni piano, ogni tessera guarda
*tutte* le altre (per capire quanto rumore c'è sull'orecchio del gatto aiuta guardare
anche la tessera con la coda, dall'altra parte della scheda) e poi rielabora
per conto suo quello che ha visto. All'ultimo piano, ogni tessera riconsegna
la propria porzione di rumore stimato, e le porzioni ricomposte formano la
mappa completa: la torre di lettori, tutta insieme, *è* il restauratore, e
quella mappa è la sua risposta a ogni passo di pulitura. La differenza con la
U-Net è di principio: le convoluzioni, cioè il modo di guardare che rende una
rete di visione una rete di visione, davano la precedenza ai vicini di casa per
costruzione; il Transformer non privilegia nessuno, e dove conviene guardare lo
impara da solo, tessera per tessera.

`````

`````{tab} Superiore

L'ingresso è il latente rumoroso $\mathbf{z}_t \in \mathbb{R}^{32 \times 32 \times 4}$
(immagini $256 \times 256$ compresse dal VAE con fattore $f = 8$). Il
*patchify* lo suddivide in patch quadrate di lato $p$ e proietta linearmente
ciascuna in un embedding di dimensione $d$: si ottiene una sequenza di
$N = (32/p)^2$ token (usiamo $N$, perché in questo capitolo $T$ è già il
numero di passi di diffusione), a cui si somma un positional encoding
(sinusoidale
bidimensionale, fisso) che ne registra la posizione nella griglia. Segue una
pila di blocchi Transformer del tutto standard (multi-head self-attention più
MLP, con residual e layer normalization, come nel capitolo sui Transformer) e
una testa lineare finale che da ogni token ricostruisce la sua patch di rumore
stimato (nel DiT originale, anche i parametri della covarianza), riassemblata
poi in un tensore della stessa forma dell'ingresso: la rete resta una
$\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t)$, cambia solo ciò che ha dentro.

Il lato $p$ della patch è la manopola del calcolo: con $p = 2$ i token sono
$N = 256$, con $p = 4$ sono 64, con $p = 8$ sono 16. Dimezzare $p$ quadruplica
i token e con essi il costo di una passata a parità di parametri: nel paper la
taglia XL passa da 29,1 Gflops con $p = 4$ a 118,6 con $p = 2$, cioè un fattore
$4{,}08$, e non di più. Il termine quadratico dell'attenzione c'è, ma a queste
lunghezze di sequenza vale meno del 4% del totale: per blocco pesa $2N^2 d$
contro i $12Nd^2$ delle proiezioni e dell'MLP, cioè un rapporto $N/(6d)$ che
per DiT-XL/2, dove la larghezza è $d = 1152$, fa
$256/(6 \cdot 1152) \approx 3{,}7\%$. È a risoluzioni
molto maggiori che quel termine diventa il vincolo, ed è il tema del capitolo
sull'attenzione lineare. Da qui la nomenclatura del paper: quattro taglie
(DiT-S, B, L, XL)
per tre patch (/8, /4, /2), dodici modelli che, vedremo tra poco, sono il vero
esperimento del lavoro.

`````

## Il condizionamento entra dalle manopole: adaLN-zero

Manca un pezzo. Alla rete bisogna dire due cose che non stanno nell'immagine:
*a che punto della scala* sta lavorando (il numero del passo) e *che cosa* deve
disegnare (nel DiT originale la categoria dell'oggetto, nei suoi discendenti un
testo). Le due insieme si chiamano il **condizionamento**, che è il termine
usato in tutta questa sezione e vuol dire esattamente quello: le informazioni
che orientano il lavoro senza far parte di ciò su cui si lavora.

La U-Net aveva un modo semplice di riceverle, appenderle come un'etichetta; con
le tessere in fila si aprono più strade, e Peebles e Xie le mettono a
confronto. Potrebbero accodare le due informazioni come parole in più della
frase, o farle consultare a parte, come fa Stable Diffusion con la richiesta
scritta. Vince invece la soluzione più discreta, battezzata **adaLN-zero**: il
condizionamento non entra nella conversazione, regola le manopole. Il nome è
una sigla e conviene scioglierla subito, perché torna spesso: *ada* sta per
adattivo, cioè che si regola secondo il momento; *LN* è il nome di
un'operazione che i Transformer hanno dentro (la *layer normalization* del
capitolo che porta il loro nome, quella che rimette i numeri in un ordine di
grandezza maneggevole prima di ogni passaggio); e *zero* è il modo in cui si
parte, che vedremo fra poco.

`````{tab} Elementare

Immagina che la torre di lettori abbia una regia, collegata con l'auricolare a
ogni piano. La regia non suggerisce parole: dà istruzioni di *regolazione*. A
ogni piano dice quanto alzare o abbassare il volume di ciò che passa, come
spostarne il tono, e soprattutto **quanto di quel piano deve finire nel
risultato**. Quest'ultima manopola misura l'intervento del piano, non il
volume del segnale: a fondo scala il piano interviene a piena forza, a zero
non interviene affatto e quello che ha ricevuto prosegue intatto. Le istruzioni
dipendono dal momento: se siamo ai primi passi della pulitura (quasi tutto
rumore) o agli ultimi ritocchi, se si sta disegnando un gatto o un faro.

Il "-zero" del nome è un'astuzia da cantiere: il primo giorno di addestramento
tutte le manopole d'intervento sono a **zero**, e quindi nessun piano tocca
niente. Ogni piano impara poi strada facendo quanto farsi sentire. Sembra
pigrizia, ma è il modo
più stabile di cominciare: nessun piano rovina il lavoro degli altri prima
di aver imparato il proprio.

`````

`````{tab} Superiore

Ricordiamo dal capitolo sui Transformer la layer normalization: normalizza
ogni token a media zero e varianza uno, poi riscala con un guadagno e un bias
appresi, uguali per tutti gli input. L’**adaptive layer norm** (adaLN) rende
guadagno e bias *funzioni del condizionamento*: un piccolo MLP riceve
$\mathbf{c} = \mathrm{emb}(t) + \mathrm{emb}(y)$ (embedding sinusoidale del
passo più embedding della classe) e produce, per ciascun sotto-strato di
ciascun blocco, tre vettori
$(\boldsymbol{\beta}_c, \boldsymbol{\gamma}_c, \boldsymbol{\alpha}_c)$:

$$
\mathrm{adaLN}(\mathbf{h}) = (1 + \boldsymbol{\gamma}_c) \odot
\mathrm{LN}(\mathbf{h}) + \boldsymbol{\beta}_c,
\qquad
\mathbf{x} \leftarrow \mathbf{x} + \boldsymbol{\alpha}_c \odot
\mathrm{Sottostrato}\big(\mathrm{adaLN}(\mathbf{x})\big),
$$

dove $\mathrm{LN}$ è la normalizzazione *senza* parametri appresi,
$\boldsymbol{\gamma}_c$ e $\boldsymbol{\beta}_c$ sono scala e traslazione
dettate dal condizionamento, $\odot$ è il prodotto elemento per elemento e
$\boldsymbol{\alpha}_c$ è un *gate* che dosa il contributo
del sotto-strato prima della somma residua. Il pedice $c$ non è decorativo e
va letto: $\boldsymbol{\alpha}_c, \boldsymbol{\beta}_c, \boldsymbol{\gamma}_c$
vengono dal **condizionamento** e non
hanno niente a che vedere con $\alpha_t$ e $\beta_t$, che in questo capitolo
sono lo schedule del rumore. Il suffisso *zero* sta
nell'inizializzazione: l'ultimo strato dell'MLP parte azzerato, quindi
$\boldsymbol{\gamma}_c = \boldsymbol{\beta}_c = \boldsymbol{\alpha}_c = 0$ e
ogni blocco all'inizio è l’**identità**;
la rete comincia come un tubo vuoto e i blocchi si accendono gradualmente.
L'idea di modulare le normalizzazioni ha un precedente illustre che
conosciamo: l'AdaIN con cui StyleGAN {cite}`karras2019style` inietta lo stile
nel generatore. Nelle ablazioni del paper, adaLN-zero batte sia i token
in-context sia la cross-attention a parità di calcolo, e costa quasi nulla,
perché produce vettori di manopole, non token aggiuntivi da far partecipare
all'attenzione.

`````

## Più lavoro, più qualità (e non conta come)

E qui arriva il risultato che ha fatto scuola, e che vale più di qualunque
punteggio abbia ottenuto il modello vincente. Peebles e Xie non costruiscono un
DiT solo: ne costruiscono **dodici**, di taglie diverse, e li mettono in fila
non per grandezza ma per **quanto lavoro fanno**. In quell'ordine la qualità
delle immagini migliora con una regolarità impressionante, e non importa *da
dove* quel lavoro venga. È l'eco delle **leggi di scala** viste per i modelli
di linguaggio (le regolarità con cui la qualità di un modello cresce al
crescere della sua taglia, dei suoi dati e del calcolo speso)
{cite}`kaplan2020scaling,hoffmann2022training`.

Due parole sulle unità di misura, perché le tab qui sotto le useranno. Il
lavoro si conta in **Gflops**, i miliardi di operazioni che costa far passare
un'immagine dall'ingresso all'uscita; la grandezza si conta in **parametri**,
cioè quanti numeri interni ha la rete; e la qualità delle immagini si misura
con il FID incontrato all'apertura del capitolo, che confronta il mucchio delle
immagini generate con il mucchio di quelle vere.

Va detto subito che cosa questo **non** significa, perché è il passo che si fa
più facilmente ed è sbagliato: non significa che l'architettura non conti più.
Lo stesso lavoro contiene anche delle *ablazioni* (gli esperimenti in cui si
cambia un pezzo solo e si guarda che effetto fa) sul modo di far entrare il
condizionamento, e a Gflops sostanzialmente pari quel modo cambia la qualità in
misura tutt'altro che marginale: è lì che adaLN-zero vince. Le due
affermazioni stanno insieme senza contraddirsi, a patto di enunciarle con
precisione: **fra modelli costruiti tutti allo stesso modo**, il calcolo
predice la qualità meglio della taglia o del numero di tessere presi da soli.
Come è fatto un piano della torre resta una scelta, e sbagliarla costa; la
regolarità dice dove spendere il prossimo Gflop, non che l'ingegneria sia
finita.

In cima alla curva sta il modello più grande con le tessere più piccole, che
nel paper si chiama DiT-XL/2 (XL è la taglia della torre, il 2 è il lato della
tessera in caselle): un Transformer da seicentosettantacinque milioni di
numeri interni che, generando immagini a partire dalla categoria richiesta («un
cane», «un faro»), raggiunge la qualità dei migliori modelli con U-Net del periodo,
compresi quelli di Dhariwal e Nichol {cite}`dhariwal2021diffusion` e il latent
diffusion di Rombach e colleghi {cite}`rombach2022high`. La U-Net, dunque, non
era essenziale. Il suo vantaggio di partenza (sapere già in fabbrica che i
pixel vicini contano più di quelli lontani) si può comprare con dati e calcolo,
e oltre una certa scala il Transformer cresce meglio. È la stessa storia già
vista nel capitolo sui Transformer, dove le reti di visione classiche avevano
ceduto il passo ai Vision Transformer, e adesso si ripete dentro la diffusione.

`````{tab} Elementare

Ed è un risultato prezioso proprio perché è **prevedibile**: se so quanto
miglioro raddoppiando il lavoro, so anche se vale la pena raddoppiarlo, prima
di spendere i soldi. Ecco come ci si arriva.

Immagina di costruire dodici torri di lettori, tutte con lo stesso mestiere ma
di taglie diverse. Ci sono quattro misure di torre, dalla più piccola alla più
grande, e una torre più grande vuol dire più piani e appunti più lunghi; e ci
sono tre modi di tagliare il mosaico, in tessere grandi, medie o piccole, e più
le tessere sono piccole più sono i lettori seduti a ogni piano. Quattro per tre
fa dodici.

Ora mettile in fila non per quanto sono grandi, ma per **quanto lavoro fanno**:
quante operazioni servono a far passare un'immagine dall'ingresso all'uscita.
In quell'ordine, i risultati migliorano quasi in linea retta. E la sorpresa non
è che chi lavora di più faccia meglio, che sarebbe ovvio: è che **non conta
come** quel lavoro è stato speso. Una torre alta e stretta e una bassa e larga,
se fanno la stessa quantità di lavoro, arrivano più o meno allo stesso punto.
Le tre cose che puoi girare (i piani, la lunghezza degli appunti, la misura
delle tessere) diventano una sola: il totale del lavoro.

Detta così sembra la fine dell'ingegneria, e non lo è. Nella stessa ricerca si
vede che *a parità di lavoro* il modo di dare le istruzioni alla torre (la
regia con l'auricolare di qualche pagina fa, contro le alternative scartate)
cambia parecchio il risultato. Le due cose convivono: scelto un buon
modo di costruire la torre, da lì in poi conta quanto la fai lavorare.

`````

`````{tab} Superiore

Vale la pena delimitare l'affermazione, perché è il tipo di regolarità che si
generalizza troppo in fretta. La correlazione fra Gflops e qualità è misurata
su **una sola famiglia** (i dodici DiT), su **un solo compito** (generazione
condizionata alla classe), a **un solo budget di addestramento** e **senza
guidance**. La cifra titolare del lavoro, invece, è ottenuta *con* la
classifier-free guidance, e non sta su quella curva: sono due misure diverse, e
confonderle è l'errore più comune quando si cita questo risultato.

Il meccanismo che resta, e che ha superato la prova del tempo, è più modesto e
più utile dell'enunciato forte: fissato il disegno del blocco, i Gflops
predicono la qualità meglio del numero di parametri, e sono indifferenti a
quale manopola si giri per aumentarli. Le tre manopole (profondità, larghezza,
lato della tessera) non si equivalgono affatto in memoria né in latenza, ma si
equivalgono rispetto alla qualità finale. È questo che permette di pianificare:
si sceglie la manopola che conviene sull'hardware disponibile, sapendo che il
risultato dipenderà dal totale e non dalla scelta.

`````

## Dal fotogramma al minuto: la diffusione sui video

Che questa storia conti anche fuori dai laboratori lo dice il video. Il 15
febbraio 2024 OpenAI presenta **Sora**, un modello che da una descrizione
scritta genera filmati fino a un minuto, accompagnato da un rapporto tecnico
dal titolo programmatico: *Video generation models as world simulators*, cioè
«i modelli che generano video come simulatori del mondo»
{cite}`brooks2024video`.

Sul piano dell'architettura quel rapporto è esplicito, e vale la pena riportare
solo ciò che dichiara davvero, che è poco ma preciso. Sora «è un diffusion
transformer». I video vengono compressi da una rete in schede, come le
fotografie della sezione precedente, e le schede vengono tagliate in
**spacetime patches**, tessere che si estendono nello spazio *e nel tempo*: non
più un quadratino di immagine, ma un quadratino di immagine per un pezzetto di
durata. Quelle tessere si mettono in fila e si dànno da leggere alla torre,
esattamente come prima. L'addestramento avviene su video e immagini di durate,
risoluzioni e proporzioni diverse. E la qualità cresce «sensibilmente» al
crescere del lavoro speso ad addestrare: il confronto mostrato è fra lo stesso
modello a cui si è fatto fare il lavoro base, poi quattro volte tanto, poi
trentadue volte tanto, che è la regolarità del paragrafo qui sopra vista
all'opera su un prodotto vero. È la ricetta DiT estesa di una dimensione: dove
il Vision Transformer affettava un'immagine, qui si affetta un blocco di
fotogrammi.

Va detto altrettanto chiaramente ciò che il rapporto *non* dice: quanti numeri
interni abbia il modello, su quali dati sia stato addestrato, come sia fatto
nel dettaglio. È un documento aziendale con dimostrazioni scelte da chi lo ha
scritto, non un articolo scientifico che altri ricercatori abbiano controllato
prima della pubblicazione.

E il titolo rilancia una tesi che il rapporto stesso incrina, perché ammette di
non riprodurre correttamente la fisica delle interazioni più elementari.
L'esempio che sceglie è il vetro che si rompe, e il filmato mostrato è un
bicchiere che si rovescia senza rompersi mentre il liquido lo attraversa.
Generare video credibili significa aver *capito* il mondo, o solo averne
imparato le apparenze? È la domanda del capitolo sui **World Model**, più
avanti nel libro, dove i video generativi verranno discussi proprio come
candidati simulatori. Qui registriamo il fatto architetturale: le tessere del
Vision Transformer, passate per DiT, sono arrivate al cinema.

## Due corsie e linee dritte: MM-DiT e rectified flow

Il 2024 è anche l'anno in cui la ricetta DiT arriva ai modelli di punta che
disegnano su richiesta. Patrick Esser, Robin Rombach e il resto del gruppo di
Stable Diffusion pubblicano il lavoro dietro **Stable Diffusion 3**
{cite}`esser2024scaling`, presentato a ICML 2024 e premiato tra i migliori
articoli della conferenza. Le novità sono tre, e vale la pena prenderle in
ordine di profondità crescente.

La prima è l'architettura, battezzata **MM-DiT** (*multimodal DiT*, cioè DiT a
più modalità: il testo e l'immagine sono due modalità), e riguarda il rapporto
fra le due. In Stable Diffusion il testo era un consulente esterno. La rete che lo
leggeva era quella di **CLIP** {cite}`radford2021learning`, un modello del
capitolo su visione e linguaggio addestrato a mettere in corrispondenza
immagini e didascalie; trasformava la richiesta una volta per tutte in una fila
di numeri, e da lì in poi la U-Net poteva soltanto
*consultarlo*; l'informazione andava in un senso solo. In MM-DiT il testo e
l'immagine diventano **due file di tessere alla pari**: due corsie che
conservano ciascuna i propri pesi ma si incontrano nell'attenzione, così che a
ogni piano della torre le parole guardino le tessere dell'immagine e viceversa.
La famiglia arriva fino a 8 miliardi di numeri interni, e la scala si comporta
anche qui in modo regolare e prevedibile. L'impostazione ha fatto scuola: la
riprende, tra gli altri, FLUX (2024), del gruppo di autori originali di Stable
Diffusion.

`````{tab} Elementare

Torniamo al restauratore con la commissione del cliente sul tavolo. Nella
sezione precedente il rapporto fra i due era a senso unico: la commissione era
scritta una volta per tutte, il restauratore ci dava un'occhiata quando gli
serviva, e la commissione non cambiava mai. Un consulente che non entra nella
stanza: parla solo quando lo interrogano, e non sente quello che succede sul
tavolo.

MM-DiT fa entrare il consulente nella stanza. Le parole della richiesta e le
tessere dell'immagine diventano due file di lettori seduti allo stesso
tavolo, e a ogni piano della torre si guardano a vicenda: le tessere guardano
le parole, come già facevano, ma anche le parole guardano le tessere. La parola
«acquerello» può accorgersi di che cosa sta effettivamente succedendo nel
disegno, e regolarsi. Ciascuna delle due file conserva un mestiere proprio (chi
legge parole e chi legge tessere non fa lo stesso lavoro, e ha strumenti suoi),
ma si parlano da pari.

`````

`````{tab} Superiore

Le due corsie hanno pesi separati per modalità (proiezioni e MLP distinti) e
condividono solo l'operazione di attenzione, calcolata sulla concatenazione
delle due sequenze. È il compromesso che distingue MM-DiT sia da un Transformer
unico sui token concatenati (che tratterebbe testo e immagine come se avessero
la stessa statistica) sia dalla cross-attention di Stable Diffusion (dove il
flusso è unidirezionale e gli embedding del testo restano quelli fissati
dall'encoder). Il condizionamento su passo e istruzioni globali resta affidato
ad adaLN, come nel DiT originale; a cambiare è soltanto lo statuto del testo,
che smette di essere condizionamento e diventa una delle due sequenze
elaborate.

`````

La seconda novità è la meno appariscente e riguarda l'archivista, non il
restauratore: il latente passa da 4 a **16 canali**, cioè da quattro a sedici
numeri per ogni casella della scheda. È la correzione diretta
del limite discusso nella sezione precedente, cioè il soffitto che il
compressore impone alla qualità finale; a parità di fattore di compressione,
più canali significano una ricostruzione nettamente migliore, ed è una delle
ragioni per cui le scritte e i dettagli fini smettono di essere il difetto
caratteristico della famiglia. Chi legge il capitolo in ordine ha visto
presentare l'archivista come un accorgimento per risparmiare tempo: questa è la
misura di quanto quell'accorgimento pesasse anche sulla qualità.

La terza novità tocca il cuore del capitolo, cioè il modo stesso di andare dal
rumore all'immagine. Stable Diffusion 3 abbandona la catena di rumore di DDPM
per il **rectified flow** {cite}`liu2023rectified`, letteralmente «flusso
raddrizzato», che è una variante particolare di una famiglia di metodi più
generale, il *flow matching* di Yaron Lipman e colleghi
{cite}`lipman2023flow`. L'idea merita di essere raccontata due volte, una a
parole e una con la
matematica.

```{figure} ../figures/flow-matching-traiettorie-dritte.svg
:name: fig-traiettorie-dritte
:alt: "Due percorsi fra gli stessi due estremi, un cerchio «rumore» a sinistra e un cerchio «dati, immagine» a destra. In alto, in terracotta, quello della diffusione: una linea che serpeggia, con nove puntini a segnare le fermate. In basso, in teal, quello del flow matching: una freccia dritta con tre sole fermate."
:width: 92%

Stessi estremi, due strade. Su una linea dritta si può camminare a grandi
falcate senza uscire di strada; su una tortuosa no, e i passi devono essere
tanti e piccoli.
```

{numref}`fig-traiettorie-dritte` mostra la cosa in un colpo d'occhio, e dice
anche dove sta il guadagno: nel **tempo** che serve a fare l'immagine, non
nella bellezza dell'immagine che ne esce. Il numero di fermate necessarie non
dipende da quanto la meta sia lontana, ma da quanto la strada curvi:
raddrizzarla non cambia dove si arriva, cambia quante volte bisogna fermarsi a
chiedere la direzione.

`````{tab} Elementare

Il restauratore di questo capitolo va dal rumore all'immagine per mille
tappe brevi, con tanto di scossoni, e la strada che percorre serpeggia: la
direzione da prendere cambia continuamente, perché a ogni tappa la rete
ridecide guardando quello che ha davanti, e quello che ha davanti è appena
cambiato per via del rimescolamento. Per questo le tappe devono essere tante e
corte: chi tiene la direzione per troppo tempo esce di strada.

L'idea nuova è quasi insolente: perché seguire una strada tortuosa? Prendi la
scheda tutta rumore e la scheda dell'immagine finita, traccia una **linea
dritta** tra le due, e insegna alla rete una sola cosa: in ogni punto della
linea, *in che direzione si cammina*.

Facciamo i conti su un numero solo. Non un pixel, che qui non si tocca: uno
dei quattro numeri di una casella della scheda, su una scala che per comodità
prendiamo da 0 a 1. Nella scheda tutta rumore vale 0,2; nella scheda
dell'immagine finita vale 0,8. A metà strada vale la media: 0,5. La marcia,
quindi, va sempre in su, e di
quanto lo dice la differenza fra i due estremi: $0{,}8 - 0{,}2 = 0{,}6$ da
guadagnare in tutto. Se decidiamo di farlo in dieci tappe (dieci è una scelta
nostra, per l'esempio), ogni tappa sale sempre della stessa quantità,
$0{,}6 : 10 = 0{,}06$. Nessuna sorpresa lungo la strada, perché la strada è
dritta.

Su una strada così non serve fermarsi cinquanta volte a ricontrollare la
mappa: ne bastano una ventina, o meno, perché la direzione non cambia mai. È
questo il motivo per cui i modelli a rectified flow generano in pochissimi
passi ciò che alla catena di rumore ne costava cinquanta con le scorciatoie e
mille senza.

Con un'onestà da mettere subito a verbale: le strade che la rete impara non
escono mai perfettamente dritte. Il motivo è che le linee dritte tracciate in
addestramento sono milioni, una per ogni coppia (questo rumore, questa
immagine), e molte di esse passano vicinissime le une alle altre andando in
direzioni diverse. La rete, che in quel punto deve dare una risposta sola, dà
la media, e la media di direzioni diverse non è nessuna delle direzioni di
partenza. Qualche controllo lungo il percorso serve quindi ancora, ma è la
differenza fra un tornante di montagna e una provinciale con qualche curva.

`````

`````{tab} Superiore

Si fissa una scala continua $t \in [0, 1]$ (dato pulito a $t = 0$, rumore puro
a $t = 1$, coerente con il verso del capitolo) e si collega ogni dato al
rumore con un’**interpolazione lineare**:

$$
\mathbf{x}_t = (1 - t)\,\mathbf{x}_0 + t\,\boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}),
$$

dove $\mathbf{x}_0$ è un dato del training set (in Stable Diffusion 3, un latente)
ed $\boldsymbol{\epsilon}$ il rumore gaussiano. Lungo questo segmento la velocità è
costante: $\mathrm{d}\mathbf{x}_t/\mathrm{d}t = \boldsymbol{\epsilon} - \mathbf{x}_0$. Il campo appreso punta
dunque *verso il rumore*, e la generazione lo percorre all'indietro: è per
questo che nella scheda Elementare, che cammina nel verso della generazione, lo
stesso numero compare col segno opposto. Il modello $\mathbf{v}_\theta(\mathbf{x}_t, t)$ viene
addestrato a regredirla:

$$
\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{\mathbf{x}_0,\, \boldsymbol{\epsilon},\, t}
\Big[\, \big\lVert \mathbf{v}_\theta(\mathbf{x}_t, t) - (\boldsymbol{\epsilon} - \mathbf{x}_0) \big\rVert^2 \,\Big],
$$

dove $\mathbf{v}_\theta$ è il campo di velocità appreso, con parametri $\theta$, e
l'attesa è su un dato, un rumore e un istante estratti a caso: la stessa
struttura da "regressione con bersaglio noto" della loss di DDPM, con la
velocità al posto del rumore. Per generare si integra l'ODE
$\mathrm{d}\mathbf{x}/\mathrm{d}t = \mathbf{v}_\theta(\mathbf{x}, t)$ da $t = 1$ (rumore) a $t = 0$, per
esempio con passi di Eulero: niente termine stocastico, come già nel
campionatore DDIM incontrato in questo capitolo, ma qui il campo dell'ODE è
appreso direttamente, non ricavato a posteriori da un predittore di rumore.

Perché bastano meno passi? Il campo appreso in un punto è la media delle
velocità di tutte le coppie $(\mathbf{x}_0, \boldsymbol{\epsilon})$ le cui interpolazioni
passano di lì: le traiettorie marginali non sono esattamente rette, ma
risultano molto meno curve di quelle dell'ODE associata alla diffusione
variance-preserving, e l'errore di discretizzazione a parità di passi è
più piccolo. Il flow matching di Lipman e colleghi
{cite}`lipman2023flow` mostra inoltre che la famiglia dei cammini
gaussiani è generale e include i cammini della diffusione come caso
particolare: DDPM diventa un punto in uno spazio di scelte progettuali.
Esser e colleghi {cite}`esser2024scaling` confrontano sistematicamente le
varianti e adottano il rectified flow con un campionamento di $t$
concentrato sugli istanti intermedi, i più difficili: in pratica Stable
Diffusion 3 genera in poche decine di passi.

`````

## Un DiT in miniatura

Come per la spirale di punti con cui abbiamo smontato DDPM, il modo migliore
di fissare l'architettura è costruirla in piccolo. Il codice che segue è un
DiT completo ma in miniatura: si taglia una scheda finta in tessere, le si fa
passare per i piani della torre con l'attenzione e le manopole della regia, e
si ricompone il risultato. Anche qui, se non hai mai programmato non c'è nessun
obbligo di leggere le righe una per una: il testo dopo dice quello che serve.
Non c'è addestramento (servirebbero i dati e le ore di calcolo su GPU) ma tutte
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
```

Quell'ultima riga vale più delle due che la precedono, e conviene dire perché.
La prima stampa le misure del risultato, e resterebbe **identica** anche
togliendo tutti e quattro i piani della torre, o cambiando il passo e la
classe: dice che i tubi sono collegati, non che l'acqua ci passi giusta. La
seconda conta i numeri interni, e almeno cambierebbe togliendo dei piani, ma
non distingue una rete che funziona da una rotta.

La terza riga invece controlla la promessa di adaLN-zero, e la controlla
davvero. Le manopole partono da zero, quindi ogni piano dovrebbe restituire
esattamente quello che ha ricevuto, per qualunque istruzione arrivi dalla
regia: la differenza fra uscita e ingresso deve essere zero, ed è quello che
la riga stampa. È un controllo che chi scrive codice fa di solito a mente
(in gergo un *desk-check*), e qui è stato reso eseguibile: se un giorno quello
zero smette di uscire, vuol dire che l'inizializzazione si è rotta.

Il DiT vero differisce dal nostro nei numeri (nella taglia più grande, 28
piani e appunti da 1152 numeri per lettore, contro i nostri 4 e 128), nel fatto
che
predice qualcosa in più oltre al rumore, e nel modo di dire a ogni tessera
dove si trova nella griglia. Ma non nella logica: quella sta tutta qui.

## Il conto, e la lezione

Chiudiamo il capitolo con la stessa onestà con cui l'abbiamo aperto.
Addestrare questi modelli è fuori dalla portata individuale, e la direzione è
quella di un rincaro: già lo Stable Diffusion del 2022 chiedeva le
centocinquantamila ore di calcolo che sappiamo, i suoi successori a miliardi di
numeri interni ne chiedono un multiplo che nessuno dichiara, e su Sora OpenAI
non pubblica né costi né dimensioni. La grandezza
è diventata un ingrediente della ricetta, e le curve di questa sezione lo
dicono senza giri di parole.

Usare un modello già addestrato, però, è un'altra storia, e in gergo si chiama
**inferenza**, che vuol dire semplicemente far girare un modello già fatto
invece di costruirlo: si scaricano i pesi, si fanno girare sulla GPU di un
computer da videogiochi, e il rectified flow ha reso la generazione più veloce,
non più lenta. L'asimmetria vista per Stable Diffusion (addestrare è per pochi,
usare è per molti) si è accentuata in entrambe le direzioni.

E la lezione finale è quella che questo libro ripete dal primo capitolo. Nel
2015 la diffusione nasce da un'analogia termodinamica; nel 2024 genera un
minuto di video da una frase. In mezzo, nessun colpo di genio isolato, ma
quattro mattoni presi da scaffali diversi. L'autoencoder variazionale è del
2014, e non era nato per comprimere: era nato per **inventare** immagini nuove,
sorteggiando una scheda a caso e facendola ridipingere al copista. Qui gli
tocca il ruolo del compressore, e a inventare pensa qualcun altro. La U-Net
è del 2015 ed era nata per i microscopi {cite}`ronneberger2015u`. L'attenzione
nasce nel 2015 per tradurre da una lingua all'altra e diventa protagonista nel
2017 {cite}`vaswani2017attention`. Le tessere sono quelle di un Vision
Transformer del 2021 nato per riconoscere il contenuto delle fotografie
{cite}`dosovitskiy2021image`. Nessuno di questi pezzi era stato progettato per
generare immagini, e sono stati ricombinati con pazienza da gruppi diversi in
anni diversi. Chi ti racconta questa storia come una successione di rivoluzioni
improvvise te la racconta male: è una storia di ricombinazioni, e il prossimo
mattone, con ogni probabilità, è già su uno scaffale che abbiamo attraversato
senza fermarci.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **DiT** manda in pensione la rete di visione e mette al suo posto la torre di
  lettori del capitolo sui Transformer. La scheda dell'archivista si taglia a
  tessere, le tessere si mettono in fila come parole, e ogni tessera guarda
  tutte le altre invece dei soli vicini di casa. L'archivista e il lavoro sulle
  schede compresse restano quelli della sezione precedente.
- Le istruzioni (a che punto della pulitura siamo, che cosa disegnare) non
  entrano nella conversazione: arrivano da una **regia** che a ogni piano
  regola le manopole. E il primo giorno tutte le manopole sono a zero, così
  ogni piano parte lasciando passare tutto e impara strada facendo quanto
  farsi sentire.
- Il risultato che ha fatto scuola: mettendo in fila dodici modelli (quattro
  taglie di torre per tre misure di tessera) per
  **quanto lavoro fanno**, la qualità migliora quasi in linea retta, e non
  conta *come* quel lavoro sia stato speso. Non vuol dire che l'architettura
  non conti: a parità di lavoro, il modo di dare le istruzioni cambia ancora
  molto. Vuol dire che, scelto un buon disegno, da lì in poi comanda il
  calcolo.
- **Sora** applica la stessa ricetta ai video, tagliando tessere che si
  estendono nello spazio *e nel tempo*. Se questo basti a dire che ha «capito»
  il mondo è la domanda del capitolo sui World Model.
- **Stable Diffusion 3** fa tre cose: mette testo e immagine allo stesso
  tavolo invece che uno a consulenza dell'altro, dà all'archivista quattro
  volte più spazio per le sue schede, e sostituisce il sentiero tortuoso con
  una **linea dritta**, che si percorre a grandi falcate e quindi in molti
  meno passi.
- Allenare questi modelli resta roba da centri di calcolo; **usarli** no. Ed è
  una storia di mattoni ricombinati, non di rivoluzioni improvvise.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- **DiT** {cite}`peebles2023scalable` sostituisce la U-Net con un
  Transformer: il latente si affetta in patch-token come nel ViT
  {cite}`dosovitskiy2021image`, l'attenzione rimpiazza le convoluzioni; il
  VAE e la diffusione nel latente restano quelli di
  {cite}`rombach2022high`.
- Il condizionamento su $t$ e classe entra via **adaLN-zero**: scala,
  traslazione e gate della layer norm generati da un MLP del condizionamento,
  con inizializzazione a zero (ogni blocco parte come identità).
- Risultato chiave: **dentro la famiglia DiT** la qualità scala con i Gflops in
  modo regolare, comunque li si spenda, e le leggi di scala
  {cite}`kaplan2020scaling` arrivano alla diffusione. Non è la fine
  dell'architettura: le ablazioni dello stesso lavoro mostrano che a Gflops
  pari il disegno del blocco cambia ancora molto il risultato.
- **Sora** {cite}`brooks2024video` dichiara un diffusion transformer su
  *spacetime patches* di video compressi, con qualità che cresce col
  calcolo: la ricetta DiT estesa al tempo. Se ciò faccia dei video
  generativi dei "simulatori di mondo" è la domanda del capitolo sui World
  Model.
- **Stable Diffusion 3** {cite}`esser2024scaling` cambia tre cose: l'MM-DiT
  (testo e immagine come due flussi di token alla pari), il latente da 4 a 16
  canali (che alza il soffitto di ricostruzione discusso nella sezione
  precedente) e il passaggio al **rectified flow** {cite}`liu2023rectified`,
  della famiglia del *flow matching* {cite}`lipman2023flow`: interpolazioni
  lineari dato–rumore, una velocità appresa per regressione, generazione
  integrando un'ODE in poche decine di passi.
- Addestrare resta un affare da data center; **usare** no: i pesi aperti e
  il latente compresso tengono l'inferenza alla portata di una GPU
  domestica. La storia del capitolo è ricombinazione di mattoni noti, non
  una rivoluzione improvvisa.
```

`````

La ricetta sta in tre gesti (comprimere, sporcare di rumore, insegnare a
ripulire), ma quello che serve più avanti è l'abitudine con cui è stata letta,
guardare un'architettura nuova cercandoci dentro i mattoni vecchi.
«Verosimiglianza esatta» tiene la stessa domanda, generare, e cambia il metro
di giudizio: chiede a un modello
non soltanto di produrre dati plausibili, ma di dire con un numero preciso
quanto lo sono, e mostra a che cosa serve quel numero fuori dalla generazione,
comprimere e accorgersi di ciò che è fuori posto.
