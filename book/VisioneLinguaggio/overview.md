# Modelli che vedono e parlano

Il 7 luglio 1966, al Project MAC del MIT, il gruppo di intelligenza artificiale
diffonde un promemoria di sei pagine intestato *Vision Memo. No. 100*. Si
chiama *The Summer Vision Project*, lo firma Seymour Papert, e attacca così: il
progetto «è un tentativo di impiegare in modo efficace i nostri lavoratori
estivi nella costruzione di una parte significativa di un sistema visivo». A
coordinare le riunioni del progetto c'è un giovane studente, Gerald Sussman.

Gli obiettivi conviene leggerli nell'ordine in cui sono scritti. Primo:
dividere l'immagine ripresa dalla telecamera in regioni che sono «oggetti
probabili», regioni che sono «sfondo probabile» e regioni che sono «caos»
(l'analisi figura-sfondo). Secondo: descrivere quelle regioni. Terzo e ultimo,
la *object identification*, che deve «dare un nome agli oggetti confrontandoli
con un vocabolario di oggetti noti». Per luglio erano previste scene di oggetti
non sovrapposti (palle, mattoncini, cilindri) con facce di colore uniforme e
sfondo omogeneo; ad agosto si sarebbe passati a superfici e sfondi complicati,
e poi a «oggetti come utensili, tazze e simili».

L'aneddoto si racconta di solito come una barzelletta sull'ottimismo di quegli
anni, ma non è la lettura interessante. Papert era tutto tranne che un ingenuo
(tre anni dopo avrebbe scritto con Marvin Minsky *Perceptrons*, il libro che
dei limiti del percettrone a uno strato fece una dimostrazione matematica), e
la scaletta che aveva in mente era giusta: isolare gli oggetti, descriverli,
chiamarli per nome. Non era l'idea a mancare: mancava un modo di far parlare
fra loro una griglia di misure di luce e un vocabolario di parole, e per
trovarlo un'estate non poteva bastare. Ci sono voluti sessant'anni, e quel modo
(o meglio quei modi) è l'argomento del capitolo.

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

Un testo è una sequenza $x = (x_1, \dots, x_T)$ con $x_t \in V$, dove $V$ è un
vocabolario finito: discreto, ordinato in una dimensione, già simbolico, perché
l'unità minima porta significato di per sé. Un'immagine è un tensore
$I \in \mathbb{R}^{H \times W \times 3}$: continuo, ordinato in due dimensioni
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

La prima domanda ha una risposta condivisa da quasi tutti i sistemi di oggi, e
il lettore la conosce già: il **Vision Transformer**
{cite}`dosovitskiy2021image` del capitolo sui Transformer. Vale la pena
richiamarne il gesto, perché tutto il resto ci poggia sopra.

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
$224 \times 224$ puntini e tessere da $16 \times 16$ vengono
$14 \times 14 = 196$ tessere: una «frase» di 196 pezzi. Ogni tessera diventa
una fila di numeri, con attaccata un'etichetta che dice in che posizione del
mosaico stava, altrimenti la rete non saprebbe quale tessera confina con quale.

Sembra poco, ed è invece il passaggio che apre la porta: dal momento in cui
un'immagine è una fila di pezzi, come le parole di una frase, tutto ciò che
sappiamo fare con le frasi si può provare anche con le foto. Il prezzo si paga
sul numero di tessere.

`````

`````{tab} Superiore

Il ViT divide l'immagine in patch quadrate di lato $P$, ne ottiene
$N = HW/P^2$ (con $H = W = 224$ e $P = 16$, esattamente $N = 196$), appiattisce
ciascuna patch in $\mathbb{R}^{3P^2}$ e la proietta linearmente in
$\mathbb{R}^{d}$, sommando un embedding di posizione. Ne esce una matrice
$X \in \mathbb{R}^{N \times d}$ indistinguibile da una sequenza di token
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

Risolta la prima domanda, resta la seconda, ed è quella che ha generato le
architetture di cui parleremo: **in quale punto** l'immagine e il testo si
incontrano.

## Tre modi di far incontrare due flussi

Le risposte che hanno resistito sono tre, e non sono tre epoche destinate a
superarsi a vicenda: convivono, e servono a cose diverse.

La prima **allinea due spazi senza fonderli**: due reti separate, una per le
immagini e una per i testi, imparano a mandare una foto e la sua didascalia
nello stesso punto di uno spazio comune, e lontano le coppie che non si
corrispondono. È l'idea di CLIP {cite}`radford2021learning`, addestrato su 400
milioni di coppie di immagine e testo raccolte dal web: il modello che ne esce
non scrive una riga, ma sa dire quanto un'immagine e un testo si somigliano.

La seconda **innesta un occhio su un modello di linguaggio già addestrato**:
fra un encoder visivo e un modello che parla bene si costruisce un raccordo (un
**connettore**) che traduce le patch in qualcosa che il modello sappia leggere.
Il grosso dei pesi resta congelato, si addestra il raccordo.

La terza rinuncia alla distinzione: **un solo modello, un solo vocabolario**.
L'immagine viene ridotta a simboli discreti, token come le parole, e un unico
Transformer li legge e li scrive tutti insieme.

Chiedersi *dove* si incontrano i due flussi è una buona bussola: spiega quasi
sempre cosa un sistema sa fare e cosa gli costa.

`````{tab} Elementare

Per tenerle a mente, pensa a due persone che devono lavorare insieme e non
parlano la stessa lingua.

Nel primo caso non si parlano affatto: hanno però imparato, ciascuno per conto
suo, a riporre le cose che si somigliano nello stesso cassetto di uno schedario
comune. Serve per ritrovare le cose («quale di queste diecimila foto è il gatto
nero sul muro?»), non per fare conversazione.

Nel secondo c'è un interprete che sussurra: la prima persona guarda, e passa
alla seconda, in forma comprensibile, quello che ha visto. Chi parla resta uno
solo, e impara a fidarsi di ciò che gli viene sussurrato.

Nel terzo si insegna a tutti e due la stessa lingua fin dall'inizio: niente più
da tradurre, ma va rifatto tutto da capo.

Nessuno dei tre ha vinto: il primo cerca, il secondo conversa, il terzo
produce.

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
$Z \in \mathbb{R}^{N \times d}$, e queste righe entrano in un modello di
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

Il capitolo poggia su cose viste altrove e non le rispiega. Dal capitolo sui
**Transformer** servono la **cross-attention**, dove le query vengono da una
sequenza e chiavi e valori da un'altra (è il meccanismo con cui il testo
interroga l'immagine), e
l'**instruction tuning**, che trasforma un modello che descrive immagini in un
modello a cui si fanno domande. Dal capitolo sulla **visione artificiale**
serve il **transfer learning**, riusare una rete pre-addestrata congelandone i
pesi: qui è la strategia dominante.

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
$\mathcal{L}(\theta) = -\sum_t \log p_\theta(y_t \mid y_{<t}, I)$, dove $y_t$ è
il token al passo $t$ e $I$ l'immagine. Nulla in questa funzione di costo
obbliga il modello a *usare* $I$: se il prior linguistico concentra già la
massa di probabilità sulla parola corretta, il gradiente che spinge a sfruttare
l'informazione visiva è debole, e il modello impara la statistica delle
didascalie invece della scena. È il meccanismo dell'**allucinazione visiva**:
non un incidente, ma quel che l'obiettivo premia.

`````

## Come è organizzato il capitolo

Le cinque sezioni seguono l'ordine delle domande: allineare due spazi,
collegarli, fonderli, pagare il conto della risoluzione, e infine controllare
che il sistema abbia davvero guardato.

- **Allineare due spazi**, l'addestramento contrastivo di CLIP
  {cite}`radford2021learning`: due encoder, uno spazio condiviso, e la
  classificazione senza esempi che ne discende come effetto collaterale.
- **Innestare gli occhi**, i connettori fra un encoder visivo e un modello di
  linguaggio congelato: proiezione lineare, query apprese, cross-attention.
- **Fusione precoce e tardiva**, l'immagine ridotta a token discreti: cosa si
  guadagna in simmetria e cosa si perde nella quantizzazione.
- **Il costo del dettaglio**, la risoluzione: perché una foto grande esplode
  nel contesto, come la si spezza in riquadri, e cosa serve per leggere un
  documento invece che riconoscere una scena.
- **Vedere quel che non c'è**, l'allucinazione visiva e la valutazione di
  questi sistemi, fino ai modelli che dalla percezione passano all'azione.

```{admonition} Da ricordare
:class: important
- Nel 1966 il *Summer Vision Project* di Papert voleva separare gli oggetti
  dallo sfondo, descriverli e dar loro un nome «confrontandoli con un
  vocabolario di oggetti noti»: la scaletta era giusta, mancava il modo di
  legare pixel e parole.
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
  prior linguistico basta a indovinare la didascalia nasce l'allucinazione
  visiva, un limite strutturale e non un caso sfortunato.
```
