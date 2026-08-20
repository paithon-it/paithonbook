# Modelli che vedono e parlano

Nel 1929 René Magritte dipinge una pipa su fondo chiaro e, sotto, ci scrive a
mano *Ceci n'est pas une pipe*: questa non è una pipa. Non è un gioco di
parole, è una constatazione esatta. Il disegno non è la pipa, è un velo di
colore steso su una tela; e la frase sotto non è il disegno, è una fila di
segni d'inchiostro. Eppure chi guarda il quadro attraversa quei due confini
senza accorgersene: vede una forma, pensa a un oggetto, gli dà un nome.

Per una macchina quel passaggio è tutto tranne che gratis, ed è stato a lungo
il confine fra due mestieri separati. Da una parte i programmi che guardano,
addestrati a mettere una fotografia in una casella e poi a tacere; dall'altra i
programmi che scrivono, addestrati a indovinare la parola dopo. Farli lavorare
insieme non vuol dire attaccare una telecamera a un generatore di testo: vuol
dire costruire un posto in cui le misure della luce e le parole di una lingua
possano incontrarsi. È quel posto, e i tre modi di costruirlo, l'argomento del
capitolo.

## Due materie che non si somigliano

Perché mettere insieme testo e immagini è un problema, e non una questione di
formato? Perché le due materie prime hanno una natura opposta.

`````{tab} Elementare

Una pagina di libro arriva già tagliata a pezzi: le parole. Sono in numero
finito, ognuna ha un nome, e a separarle ci pensano gli spazi. «Il gatto nero
salta sul muro» sono sei parole per tutti, sempre le stesse, sempre in
quell'ordine.

Una fotografia no. È un tappeto di puntini colorati, dodici milioni in uno
scatto da telefono, ciascuno con tre numeri per il rosso, il verde e il blu.
Nel tappeto non c'è nessuna cucitura che dica «qui finisce il gatto e comincia
il muro»: quel confine lo vediamo noi, non è scritto nei numeri. E se sposti la
macchina di due centimetri, tutti e dodici i milioni di puntini cambiano valore
mentre la scena resta la stessa; la frase, intanto, non si è mossa di una
virgola.

Farli lavorare insieme vuol dire allora due cose: prima dare all'immagine dei
«pezzi», poi decidere dove quei pezzi incontrano le parole.

`````

`````{tab} Superiore

Un testo è una sequenza $\mathbf{x} = (x_1, \dots, x_T)$ con $x_t \in V$, dove $V$ è un
vocabolario finito: discreto, ordinato in una dimensione, già simbolico, perché
l'unità minima porta significato di per sé. Un'immagine è un tensore
$\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$: continuo, ordinato in due dimensioni
e privo di unità naturali. Il pixel non è un simbolo, e non esiste una
segmentazione canonica del reticolo in parti dotate di senso; la segmentazione
è semmai *l'esito* di un modello, non il suo input.

La seconda asimmetria riguarda la metrica: due fotografie della stessa scena
prese a due centimetri di distanza hanno distanza $\ell_2$ enorme nello spazio
dei pixel e contenuto identico. Le invarianze che ci interessano (traslazione,
illuminazione, punto di vista, scala) sono esattamente quelle rispetto a cui la
metrica nativa non è invariante.

Servono quindi due decisioni, e conviene tenerle separate: **come
rappresentare l'immagine come sequenza** di vettori di dimensione fissa, in uno
spazio dove la vicinanza sia somiglianza semantica, e **dove i due flussi si
incontrano**, cioè a quale profondità del sistema informazione visiva e
informazione linguistica smettono di essere separate.

`````

## Fare dell'immagine una sequenza

La prima delle due cose da fare, dare all'immagine dei pezzi, ha una risposta
condivisa da quasi tutti i sistemi di oggi, e il lettore la conosce già: il
**Vision Transformer** {cite}`dosovitskiy2021image` del capitolo sui
Transformer. Conviene richiamarne il gesto, perché tutto il resto ci poggia
sopra, e sta in due mosse.

La prima: si taglia la fotografia in quadratini tutti uguali, che qui chiameremo
**tessere** (in inglese *patch*, ed è la parola che si incontra ovunque, questo
capitolo compreso). La seconda: siccome le tessere, una volta messe in fila, non
ricordano più da quale punto della foto venissero, a ciascuna si attacca
un'etichetta che dice dove stava nella griglia, e quell'etichetta si chiama
**codifica di posizione**.

```{figure} ../figures/vit-2020.svg
:name: fig-vit-patch-token
:alt: "Un'immagine viene divisa in una griglia di patch quadrate; ogni patch viene appiattita e proiettata in un vettore, a cui si somma una codifica di posizione che ne registra il posto nella griglia; la sequenza risultante entra in un encoder Transformer come se fosse una frase."
:width: 100%

Il gesto da richiamare. La codifica di posizione è la parte da non perdere:
senza, la sequenza sarebbe un mucchio di tessere e l'immagine non avrebbe più
un sopra e un sotto.
```

Il passaggio di {numref}`fig-vit-patch-token` merita di essere fissato perché
è la premessa di tutto il capitolo. Una volta che l'immagine è diventata una
sequenza di vettori, la domanda «come si mettono insieme testo e immagine»
smette di riguardare due materie diverse: sono due sequenze, e le sequenze
sappiamo già come si combinano.

`````{tab} Elementare

Si taglia la foto a tessere quadrate, come un mosaico, e si mettono le tessere
in fila indiana come se fossero le parole di una frase. Con un'immagine da
$224 \times 224$ puntini e tessere da $16 \times 16$ ne stanno
$224 : 16 = 14$ per riga e altrettante per colonna, cioè $14 \times 14 = 196$
tessere: una «frase» di 196 pezzi. Ogni tessera diventa una fila di numeri, con
attaccata un'etichetta che dice in che posizione del mosaico stava, altrimenti
la rete non saprebbe quale tessera confina con quale.

Sembra poco, ed è invece il passaggio che apre la porta: dal momento in cui
un'immagine è una fila di pezzi, come le parole di una frase, tutto ciò che
sappiamo fare con le frasi si può provare anche con le foto.

Il prezzo si paga sul numero di tessere, e non in proporzione. Per capire ogni
tessera il modello la confronta con tutte le altre: con 196 tessere i confronti
sono $196 \times 196$, poco meno di quarantamila. Adesso raddoppiamo il lato
della foto, da 224 a 448 puntini: le tessere diventano quattro volte tante
($448 : 16 = 28$ per riga, cioè $28 \times 28 = 784$), e i confronti, che sono
tessere per tessere, sedici volte tanti. È il conto che tornerà in tutto il
capitolo.

`````

`````{tab} Superiore

Il ViT divide l'immagine in patch quadrate di lato $p$, ne ottiene
$N = \lfloor H/p \rfloor \cdot \lfloor W/p \rfloor$ (con $H = W = 224$ e
$p = 16$, esattamente $N = 196$), appiattisce
ciascuna patch in $\mathbb{R}^{3p^2}$ e la proietta linearmente in
$\mathbb{R}^{d}$, sommando un embedding di posizione. Ne esce una matrice
$\mathbf{X} \in \mathbb{R}^{N \times d}$ indistinguibile da una sequenza di token
testuali: da lì in poi l'architettura non sa più, e non ha bisogno di sapere,
da quale modalità arrivino le righe.

Due conseguenze pesano su tutto il capitolo. Il bias induttivo delle CNN
(località ed equivarianza alla traslazione) sopravvive solo nel taglio iniziale
in patch, e per il resto va ricomprato con i dati: è la ragione per cui un ViT
rende meno di una rete convoluzionale quando i dati scarseggiano, e chiede
pre-addestramenti molto grandi. E il costo dell'attenzione, $O(N^2 d)$,
dipende dal quadrato del numero di patch: raddoppiare il lato dell'immagine
quadruplica $N$ e moltiplica per sedici quel costo.

`````

Risolta quella, resta l'altra, ed è la domanda che ha generato le architetture
di cui parleremo: **in quale punto** l'immagine e il testo si incontrano.

## Tre modi di far incontrare due flussi

I due flussi sono quelli di sempre, l'immagine e il testo, adesso che tutti e
due sono diventati una fila di pezzi. Le risposte che hanno resistito sono tre,
e non sono tre epoche destinate a superarsi a vicenda: convivono, e servono a
cose diverse.

La prima **allinea due spazi senza fonderli**: due reti separate, una per le
immagini e una per i testi, imparano a mandare una foto e la sua didascalia in
due punti vicini di una stessa mappa, e a tenere lontane le coppie che non si
corrispondono. Vicini rispetto a che cosa, la sezione apposita se lo
chiede alla fine, perché la risposta è meno ovvia di così. È l'idea di CLIP
{cite}`radford2021learning`, addestrato su 400 milioni di coppie di immagine e
testo raccolte dal web: il modello che ne esce non scrive una riga, ma sa dire
quanto un'immagine e un testo si somigliano.

La seconda **innesta un occhio su un modello di linguaggio già addestrato**:
fra un encoder visivo e un modello che parla bene si costruisce un raccordo (un
**connettore**) che traduce le patch in qualcosa che il modello sappia leggere.
Il grosso dei pesi resta congelato, si addestra il raccordo.

La terza rinuncia alla distinzione: **un solo modello, un solo vocabolario**.
L'immagine viene ridotta a simboli presi da un elenco finito, token come le
parole, e un unico
Transformer li legge e li scrive tutti insieme.

Chiedersi *dove* si incontrano i due flussi è una buona bussola: spiega quasi
sempre cosa un sistema sa fare e cosa gli costa.

`````{tab} Elementare

Per tenerle a mente, pensa a due persone che devono lavorare insieme e non
parlano la stessa lingua.

Nel primo caso non si parlano affatto: hanno però imparato, ciascuno per conto
suo, a segnare quel che hanno in mano su una stessa **mappa**, vicino se le due
cose si somigliano e lontano se non c'entrano niente. Nessuno dei due sa che cosa
abbia scritto l'altro, ma i segni si possono confrontare con un righello, e tanto
basta per ritrovare le cose («quale di queste diecimila foto è il gatto nero sul
muro?»). Per fare conversazione, no.

Nel secondo c'è un interprete che sussurra: la prima persona guarda, e passa
alla seconda, in forma comprensibile, quello che ha visto. Chi parla resta uno
solo, e non cambia mestiere: continua a parlare come ha sempre fatto, e tutto il
lavoro sta nell'insegnare all'interprete a sussurrargli bene.

Nel terzo si insegna a tutti e due la stessa lingua fin dall'inizio: niente più
da tradurre, in compenso nessuno dei due può portarsi dietro quello che aveva
imparato prima, e la scuola va rifatta da capo per entrambi.

Nessuno dei tre ha vinto: il primo cerca, il secondo conversa, il terzo
produce, cioè sa tirar fuori anche un'immagine e non soltanto parole.

`````

`````{tab} Superiore

Il criterio che le distingue è **la profondità alla quale avviene la fusione**.

*Tardiva, nello spazio delle rappresentazioni.* Due encoder producono due
vettori nello stesso $\mathbb{R}^{d}$, e l'unica interazione fra le modalità è
un prodotto scalare alla fine. Gli embedding delle immagini si precalcolano e
cercare in un archivio costa un prodotto matrice-vettore; in cambio non c'è
interazione fine fra parti dell'immagine e parole, e quindi nessuna capacità
generativa.

*Intermedia, tramite connettore.* Un encoder visivo produce
$\mathbf{Z} \in \mathbb{R}^{N \times d}$, e queste righe entrano in un modello di
linguaggio pre-addestrato o come token in testa alla sequenza (un *prefisso*
visivo) o attraverso strati di cross-attention inseriti fra i blocchi, con le
query dal testo e chiavi e valori dall'immagine. Si addestra il connettore
lasciando spesso congelati i due modelli: la variante più economica, e quella
che riusa meglio ciò che esiste già.

*Precoce, nello spazio dei token.* L'immagine viene quantizzata in simboli di
un codebook e concatenata al testo in un'unica sequenza, su cui un solo
Transformer applica la stessa attenzione a tutto. Il modello diventa simmetrico
(emette token visivi come emette parole), al prezzo di un pre-addestramento
intero e dell'informazione persa nella quantizzazione.

Il confine fra le ultime due è meno netto di quanto la tripartizione
suggerisca, e i sistemi reali sono spesso ibridi.

`````

## Quello che serve avere già in mano

Il capitolo poggia su cose viste altrove e non le rispiega. Conviene dire
quali sono, una riga ciascuna.

Dal capitolo sul **linguaggio** serve la **mappa del significato**: l'idea che
una parola si possa scrivere come una fila di numeri, e che su quella mappa
*gatto* e *felino* finiscano vicini mentre *gatto* e *mercoledì* finiscono agli
antipodi, con un numero fra $-1$ e $+1$ a dire quanto. Tutto questo capitolo
consiste nel far entrare le fotografie in quella stessa mappa.

Dal capitolo sui **Transformer** serve la **cross-attention**. È l'attenzione di
sempre, con una differenza: chi fa le domande e chi le riceve sono due sequenze
diverse. Le domande (in gergo, le *query*) vengono dal testo; quel che si va a
consultare (le *chiavi*, per trovare il punto giusto, e i *valori*, cioè quel che
si porta via) viene dall'immagine. È il meccanismo con cui il testo interroga
l'immagine. Dallo stesso capitolo serve l’**instruction tuning**, che trasforma
un modello che descrive immagini in un modello a cui si fanno domande.

Dal capitolo sulla **visione artificiale** serve il **transfer learning**,
riusare una rete pre-addestrata congelandone i pesi: qui è la strategia
dominante.

## Un avvertimento, prima di cominciare

C'è un rischio specifico di questi sistemi, e conviene metterlo sul tavolo
subito: un modello che vede e parla può parlare benissimo *senza* aver
guardato.

`````{tab} Elementare

Immagina uno studente che ha letto migliaia di didascalie di fotografie. Gli
mostri una spiaggia e ti dice che c'è il mare, la sabbia e qualche ombrellone.
Ha ragione quasi sempre, e non perché abbia guardato: perché nelle didascalie
di spiaggia ci sono quasi sempre mare, sabbia e ombrelloni. Il giorno in cui
gli mostri una spiaggia senza ombrelloni, lui te li nomina lo stesso.

Non è il difetto di un modello sfortunato: se le parole giuste si indovinano
dal contesto, guardare diventa facoltativo, e chi è addestrato a indovinare
bene impara a farne a meno.

`````

`````{tab} Superiore

Un modello che genera testo condizionato a un'immagine minimizza
$\mathcal{L}(\theta) = -\sum_t \log p_\theta(y_t \mid y_{<t}, \mathbf{I})$, dove $y_t$ è
il token al passo $t$ e $\mathbf{I}$ l'immagine. Nulla in questa funzione di costo
obbliga il modello a *usare* $\mathbf{I}$: se il **priore linguistico** concentra già la
massa di probabilità sulla parola corretta, il gradiente che spinge a sfruttare
l'informazione visiva è debole, e il modello impara la statistica delle
didascalie invece della scena. È il meccanismo dell’**allucinazione visiva**:
non un incidente, ma quel che l'obiettivo premia.

`````

## Allineare, innestare, fondere

Le cinque sezioni seguono l'ordine delle domande: allineare due spazi,
collegarli, fonderli, pagare il conto della risoluzione, e infine controllare
che il sistema abbia davvero guardato.

- **Allineare due spazi**: due reti separate imparano a segnare una foto e la
  sua didascalia vicine sulla stessa mappa, ed è l'addestramento *contrastivo*
  di CLIP {cite}`radford2021learning`: si impara per contrasto, avvicinando le
  coppie giuste e allontanando le sbagliate. Da lì viene fuori, come effetto collaterale,
  un classificatore che si scrive a parole.
- **Innestare gli occhi**: il raccordo fra un encoder visivo e un modello di
  linguaggio lasciato fermo. Di raccordi ne sono stati provati tre, e ha vinto
  il più semplice.
- **Fusione precoce e tardiva**, cioè presto o tardi lungo il percorso:
  l'immagine ridotta a simboli di un elenco fin dall'ingresso, come le parole.
  Cosa si guadagna a poterla anche produrre, e cosa si perde nell'arrotondarla
  alla voce di catalogo più vicina.
- **Il costo del dettaglio**, cioè quanti puntini si danno da guardare: perché
  una foto grande occupa tanto posto, come la si spezza in riquadri, e cosa
  serve per leggere un documento invece che riconoscere una scena.
- **Vedere quel che non c'è**: un modello che parla di una fotografia senza
  averla guardata, come lo si misura, e i sistemi che dalla percezione passano
  all'azione.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- La pipa di Magritte dice il problema in un colpo: il disegno non è la cosa,
  la parola non è il disegno, e a tenere insieme i tre pezzi è la nostra testa.
  Insegnare quel salto a una macchina è ciò di cui parla il capitolo.
- Le due materie prime hanno nature opposte: una pagina scritta arriva **già
  tagliata a pezzi**, le parole, sempre le stesse per tutti; una fotografia è un
  **tappeto di puntini colorati** senza cuciture, e il confine fra il gatto e il
  muro lo vediamo noi, nei numeri non c'è.
- Il primo passo è sempre lo stesso: tagliare la foto in **tessere** e metterle
  in fila come le parole di una frase, attaccando a ciascuna un'etichetta che
  dice dove stava. Da lì in poi immagine e testo sono due file di pezzi, e le
  file sappiamo già come si mettono insieme. Il prezzo si paga sul numero di
  tessere: se raddoppiano, il lavoro quadruplica.
- La domanda che genera tutto il capitolo è **dove i due flussi si incontrano**,
  e le risposte che hanno retto sono tre: due che non si parlano ma segnano le
  cose sulla stessa mappa (**cerca**), un interprete che sussurra a chi sa già
  parlare (**conversa**), una lingua sola insegnata a tutti e due dall'inizio
  (**produce**). Non si superano a vicenda.
- Il rischio da tenere presente fin da subito: un modello che vede e parla può
  parlare benissimo **senza aver guardato**, come lo studente che ha letto
  migliaia di didascalie di spiaggia e ti nomina gli ombrelloni anche quando non
  ci sono. Non è sfortuna, è quello che l'addestramento premia.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il problema non è di formato ma di **raccordo fra rappresentazioni**: quello
  che manca non è un convertitore, è uno spazio in cui la vicinanza voglia dire
  la stessa cosa per una misura di luce e per una parola.
- Testo e immagine hanno nature opposte: il testo è **discreto e già
  simbolico**, l'immagine un reticolo **continuo** senza unità naturali, dove
  la distanza fra i pixel non misura la distanza fra i significati.
- Il **ViT** {cite}`dosovitskiy2021image` rende l'immagine una sequenza
  tagliandola in patch: con $224 \times 224$ e patch $16 \times 16$ sono 196
  token, e il costo dell'attenzione cresce con $N^2$.
- **Dove i due flussi si incontrano** genera tre famiglie: allineare due spazi
  senza fonderli (CLIP {cite}`radford2021learning`), innestare un connettore su
  un modello di linguaggio già addestrato, o trattare pixel e parole come token
  di un unico vocabolario. Non si superano a vicenda: la prima cerca, la
  seconda conversa, la terza produce.
- L'obiettivo di addestramento non obbliga il modello a **guardare**: quando il
  priore linguistico (la statistica delle didascalie, appresa prima e
  indipendentemente dall'immagine) basta a indovinare la didascalia nasce
  l'allucinazione visiva, un limite strutturale e non un caso sfortunato.
```

`````
