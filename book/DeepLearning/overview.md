# Deep Learning: perché la profondità conta

Nel 1959 due neuroscienziati, David Hubel e Torsten Wiesel
{cite}`hubel1959receptive`, infilarono un sottile elettrodo nella corteccia
visiva di un gatto e proiettarono forme luminose davanti ai suoi occhi.
Scoprirono che certi neuroni si accendevano solo quando sullo schermo compariva
una linea con una precisa inclinazione: rivelatori di bordi in carne e ossa.
Altri neuroni, più a valle, combinavano quelle risposte in configurazioni più
complesse. Quel lavoro valse loro il premio Nobel nel 1981 e, senza che nessuno
potesse immaginarlo, descriveva già il funzionamento delle reti neurali
profonde. Il deep learning, in fondo, riscopre la stessa idea: costruire la
percezione a strati, dal semplice al complesso.

## Le feature si imparano, non si scrivono a mano

La differenza tra machine learning classico e deep learning non sta in una
matematica esoterica: sta in *chi* decide quali caratteristiche dei dati
contano. Quelle caratteristiche, in gergo, si chiamano feature: sono i
numeri che descrivono un dato e su cui il modello, cioè il programma che impara
dagli esempi, fa i suoi conti.

`````{tab} Elementare

Nel machine learning classico un esperto umano deve prima trasformare i dati
grezzi in "ingredienti" utili. Per riconoscere un gatto in una foto, qualcuno
scrive a mano il codice che conta i bordi, misura quanta parte dell'immagine
è arancione o grigia, individua gli angoli: sono le *feature ingegnerizzate*,
le caratteristiche scelte da un umano. Solo dopo il modello impara a
distinguere i gatti dai cani.

Il deep learning fa un patto diverso: gli dai i pixel grezzi e lascia che sia
la rete, addestrandosi, a inventarsi da sola gli ingredienti giusti.

In cucina la differenza si vede bene. Al primo cuoco le verdure arrivano già
tagliate da un fornitore, sempre allo stesso modo, cubetti da un centimetro,
perché così si è deciso una volta per tutte. Il cuoco cucina, manda il piatto
in sala, e il cliente gli fa sapere quanto ci è andato vicino. Quel giudizio
arriva fino ai fornelli e non oltre, perché il fornitore continua a tagliare
come sempre anche se il piatto torna indietro dieci sere di fila. E se il
coltello ha buttato via la parte buona, la punta degli asparagi finita nello
scarto, alla cottura non resta niente da recuperare.

Al secondo cuoco arriva la verdura intera. Il giudizio del cliente risale
all'indietro tutta la catena, dalla sala ai fornelli e dai fornelli al
tagliere, e la sera dopo cambiano insieme la cottura e il taglio. Nessuno gli
ha detto che gli asparagi vanno tagliati in diagonale; ci arriva perché così i
piatti tornano indietro meno spesso. Il taglio ha smesso di essere una regola
fissa decisa da qualcun altro ed è diventato una parte del mestiere che si
impara.

`````

`````{tab} Superiore

In una pipeline classica di visione artificiale un estrattore fisso e
progettato a mano (SIFT, HOG, filtri di Gabor) mappa l'immagine in un vettore
di feature $\phi(\mathbf{X})$, su cui un classificatore $h$ viene addestrato.
L'estrattore $\phi$ è congelato: non impara nulla dai dati.

Una rete profonda è invece una composizione di trasformazioni parametriche

$$
f_\theta = f_L \circ f_{L-1} \circ \dots \circ f_1 ,
$$

dove $L$ è il numero di strati e tipicamente $f_\ell(\mathbf{Z}) =
\sigma(\mathbf{W}_\ell \mathbf{Z} + \mathbf{b}_\ell)$, con la matrice di pesi
$\mathbf{W}_\ell$, il vettore di bias $\mathbf{b}_\ell$ e la non linearità
$\sigma$ applicata elemento per elemento (il bias si somma colonna per colonna,
e incontreremo anche strati di altra forma, come il pooling). Tutti i parametri
$\theta$, dal primo strato all'ultimo, vengono ottimizzati insieme minimizzando
la loss $\mathcal{L}$ per
retropropagazione (*end-to-end*). L'estrazione delle feature diventa parte del
modello e viene appresa, invece di stare a monte e fissa. È ciò che si chiama
*representation learning*.

`````

## Rappresentazioni gerarchiche: dai bordi agli oggetti

Una rete profonda è fatta di strati: gruppi di neuroni messi in fila, dove
il primo riceve i numeri dell'immagine e ognuno dei successivi riceve quello che
ha prodotto quello prima di lui. (Un neurone, qui, è un pezzetto di conto e non
una cellula: prende dei numeri, li somma dopo averli pesati e ne restituisce
uno.) Cosa impara davvero uno strato?

Nel 2014 Zeiler e Fergus {cite}`zeiler2014visualizing` trovarono il modo di
"visualizzarlo", cioè di risalire, per ogni neurone di una rete
convoluzionale (il tipo di rete fatto apposta per le immagini, di cui parla
la sezione che segue), alla forma che lo fa accendere. Il risultato somigliava
in modo sorprendente alla corteccia di Hubel e Wiesel: i primi strati reagiscono
a bordi e linee orientate; gli strati intermedi a parti riconoscibili (un
occhio, una ruota) e a *texture*, cioè trame che si ripetono, come la stoffa di
un tessuto o il pelo di un animale; gli ultimi strati a interi oggetti. La
profondità costruisce astrazione, un livello alla volta
({numref}`fig-gerarchia-feature`).

```{figure} ../figures/gerarchia-feature.svg
:name: fig-gerarchia-feature
:alt: Quattro riquadri in fila mostrano come una rete profonda costruisce la percezione a strati, dai pixel dell'immagine ai bordi, alle parti, fino all'oggetto intero riconosciuto.
:width: 92%

Come una rete profonda "vede" un gatto. Da sinistra a destra: i pixel grezzi,
i bordi e le linee dei primi strati, le texture e le parti degli strati
intermedi, l'oggetto intero riconosciuto dagli ultimi strati.
```

`````{tab} Elementare

Davanti a una scatola di mattoncini, cominci dai pezzi più piccoli: tratti
dritti, curve, angoli. Poi combini quei tratti in parti riconoscibili: due
cerchi diventano occhi, due triangoli diventano orecchie. Alla fine le parti si
assemblano in un gatto intero.

Man mano che si sale, ogni pezzo copre più tavolo. Un tratto dritto sta in un
dito di spazio, l'orecchio che mette insieme due triangoli prende mezzo palmo,
il gatto finito occupa tutto il tavolino. Nessuno ha allargato niente apposta,
la superficie cresce da sé, perché ogni pezzo raccoglie pezzi che a loro volta
ne avevano già raccolti.

La rete fa esattamente questo, ma senza che nessuno gliel'abbia insegnato:
nessuno le dice "questo è un occhio". Impara da sola che, per riconoscere i
gatti nelle foto, conviene prima trovare i bordi, poi comporli in parti, poi
comporre le parti in animali.

I pezzi dei primi assemblaggi, però, non sanno di gatto. Tratti dritti, curve e
angoli servono identici a chi vuole costruire un cane, una moto o una casa;
sono le orecchie a punta a valere solo per i gatti. Per questo chi ha passato
mesi a costruire gatti non ricomincia dalla scatola quando gli chiedono una
moto. Tiene i pezzi piccoli così come sono, rifà gli ultimi assemblaggi, e in
un pomeriggio ha finito.

`````

`````{tab} Superiore

La chiave è la **composizionalità**. Ogni mappa di feature dello strato $\ell$
è una funzione non lineare delle mappe dello strato precedente, quindi la
complessità delle forme rilevabili cresce con la profondità. In una rete
convoluzionale cresce anche il campo recettivo (*receptive field*): un
neurone dello strato $\ell$ "vede" una porzione di immagine tanto più ampia
quanto più $\ell$ è profondo, perché aggrega l'output di neuroni che a loro
volta aggregano porzioni più piccole.

Formalmente, la rappresentazione allo strato $\ell$ è
$\mathbf{Z}^{[\ell]} = f_\ell(\mathbf{Z}^{[\ell-1]})$, con
$\mathbf{Z}^{[0]} = \mathbf{X}$ l'immagine di ingresso. Le
$\mathbf{Z}^{[\ell]}$ superficiali codificano feature locali e generiche
(bordi, condivisi tra compiti diversi); le $\mathbf{Z}^{[\ell]}$ profonde
codificano feature astratte e specifiche del compito (categorie di oggetti). È
questa gerarchia a rendere così efficace il *transfer learning*: i primi
strati, appresi su un dataset enorme, si riusano quasi invariati su problemi
nuovi.

`````

## Perché proprio adesso

C'è un dettaglio che spiazza chi arriva al deep learning da profano: la
retropropagazione, l'algoritmo che addestra queste reti, fu resa celebre nel
1986 da Rumelhart, Hinton e Williams {cite}`rumelhart1986learning`, e l'idea è
ancora più vecchia[^backprop-storia].
Se l'algoritmo esisteva da decenni, perché il deep learning
è esploso solo dopo il 2012? Perché servivano tre ingredienti tutti insieme:
dati, potenza di calcolo e algoritmi maturi. Il momento-simbolo è
l'autunno del 2012, la competizione ImageNet: una gara annuale in cui i
gruppi di ricerca di mezzo mondo provavano i propri programmi sulle stesse
fotografie, più di un milione, con lo stesso compito (dire che cosa
c'è dentro, scegliendo fra mille categorie) e una classifica finale. Quell'anno
la vince, con un margine mai visto prima, una rete profonda: AlexNet, dal
nome del primo dei suoi autori, Alex Krizhevsky.

`````{tab} Elementare

Per accendere un fuoco servono la legna, l'aria e una scintilla: se manca uno
solo dei tre, non parte. Il deep learning aveva l'idea da decenni ed è rimasto
spento lo stesso.

La legna sono i dati: milioni di fotografie etichettate, cioè con
scritto accanto, a mano, che cosa c'è dentro. Ne servono milioni perché AlexNet
aveva sessanta milioni di numeri da regolare, e a regolarli sono le foto, una
dopo l'altra. Prima di Internet quella catasta non esisteva: descrivere una per
una milioni di immagini era un lavoro fuori portata.

L'aria è il calcolo. Le schede grafiche (GPU) sono nate per i videogiochi e
si sono rivelate perfette per i conti di una rete: un videogioco chiede la
stessa moltiplicazione su un milione di punti dello schermo nello stesso
istante, una rete la chiede su milioni di pesi. La scheda non vede la
differenza e le sbriga in un colpo solo. AlexNet ha bruciato la sua legna su
due schede da videogiocatore.

La scintilla sono gli algoritmi. Una scintilla su legna buona può benissimo
spegnersi, ed è quello che succedeva: reti profonde che non imparavano. A farla
attaccare sono tre accorgimenti, ciascuno contro un guaio preciso.

Il primo guaio sta nella funzione che ogni neurone applica al numero uscito dai
suoi conti. Era una curva che fa da rubinetto strozzato: giri quanto vuoi,
l'acqua che esce è sempre quella. Numero grande o numero piccolo, quello che
passava era più o meno uguale, e la rete non poteva accorgersi della
differenza. C'è di peggio, ed è lo stesso strozzamento visto al ritorno. La
correzione torna indietro dall'ultimo strato verso il primo e attraversa uno di
quei rubinetti a ogni strato che risale; ognuno ne lascia passare una frazione,
e ai primi strati non arrivava quasi niente. Il rimedio è un rubinetto che o è
chiuso o è spalancato: i numeri positivi passano come sono, i negativi
diventano zero. Se entra 5 esce 5, se entra $-3$ esce 0. Si chiama ReLU.

Il secondo guaio è che la rete si impara a memoria le fotografie
dell'addestramento. Sembrerebbe un pregio, ed è il modo più sicuro di fallire:
chi ripete a memoria i compiti dell'anno scorso va benissimo su quelli e male
sul compito di domani, che è l'unico che conta. Il rimedio è spegnere a caso
una parte dei neuroni a ogni passata sulle fotografie: se al giro dopo un
neurone può mancare, gli altri non si appoggiano solo a lui, e quello che la
rete sa finisce distribuito invece che depositato in un punto (**dropout**).

Il terzo guaio è che le foto etichettate costano. Allora si ritagliano e si
specchiano quelle che già ci sono, e da ognuna ne escono molte senza doverne
etichettare altre (**data augmentation**). Nel 2012, per la prima volta, la
legna, l'aria e la scintilla ci sono tutte insieme.

`````

`````{tab} Superiore

Nel 2009 Fei-Fei Li e collaboratori pubblicano ImageNet, un dataset di
milioni di immagini etichettate da persone; la sua competizione annuale, la
**ILSVRC**, ne usa un sottoinsieme di circa 1,2 milioni di immagini di
addestramento distribuite su 1000 categorie. Nel 2012 AlexNet (Krizhevsky,
Sutskever, Hinton) vince proprio la ILSVRC portando l'errore *top-5* dal 26,2%
del miglior metodo classico al 15,3%. Il confronto va letto per quello che è:
il 15,3% è il punteggio della sottomissione, che media le predizioni di sette
reti; la singola rete descritta nell'articolo si ferma al 18,2%, e anche così
il salto è senza precedenti.

I tre ingredienti, in numeri:

- Dati: un dataset abbastanza grande da addestrare una rete con circa 60
  milioni di parametri senza overfittare in modo catastrofico.
- Calcolo: l'addestramento gira su due GPU NVIDIA GTX 580, sfruttando la
  parallelizzazione dei prodotti tra matrici (`cuda`).
- Algoritmi: attivazione ReLU $\sigma(x)=\max(0,x)$ per attenuare il
  *vanishing gradient*, *dropout* come regolarizzazione, *data augmentation*.

Nessuna di queste idee era nuova in senso stretto; nuova era la loro
combinazione, alla scala giusta.

`````

## Profondo, non solo largo

Fin qui abbiamo dato per buono che impilare strati serva a qualcosa. C'è però un
risultato matematico che sembra dire il contrario, e conviene affrontarlo
subito.

Si chiama **teorema di approssimazione universale**, e dice questo: basta *un
solo* strato in mezzo, fra l'ingresso e l'uscita. Purché lo si faccia
abbastanza largo, cioè con abbastanza neuroni, quella rete a un piano solo sa
imitare con la precisione che si vuole qualunque regola che leghi ingressi e
uscite senza salti bruschi.[^universalita]

Una condizione c'è, ed è essenziale. La funzione che ogni neurone applica al
proprio risultato, l’attivazione (la ReLU, per esempio, che azzera i numeri
negativi e lascia passare i positivi), non deve essere un *polinomio*, cioè una
somma di potenze come $3x^2-x+5$. Il motivo è che sommare polinomi dà sempre e
solo altri polinomi, e mai di grado più alto: se l'attivazione si ferma al
quadrato, tutto ciò che la rete sa produrre si ferma al quadrato, per quanti
neuroni le si mettano. Allargare la rete aggiunge addendi, non alza il grado, e
chi resta dentro le parabole non arriverà mai a una curva che parabola non è.
La ReLU non ha questo limite, perché non è una somma di potenze: ha uno
spigolo, nel punto in cui smette di azzerare e comincia a lasciar passare, e
nessuna potenza, né somma di potenze, fa un angolo.

Se una rete "piatta" e larga sa già imitare tutto, perché impilare tanti strati?

`````{tab} Elementare

Il teorema dice che *in teoria* un solo strato basta. Ma "in teoria" nasconde
due fregature. Una sta nel prezzo: quel singolo strato potrebbe aver bisogno di
un numero enorme, impraticabile, di neuroni. L'altra sta nel verbo che il
teorema usa, "esiste". Garantisce che una rete buona ci sia da qualche parte,
non che qualcuno la sappia trovare. A cercarla è l'addestramento, che parte da
numeri buttati a caso e li corregge un poco alla volta guardando gli esempi,
e sapere che il traguardo c'è non dice da che parte muoversi per raggiungerlo,
né quante foto bisognerà guardare per arrivarci.

Il prezzo alto ha una ragione precisa. Quello che manca alla rete piatta è la
possibilità di costruire sopra. In
una rete a un solo strato ogni neurone guarda i pixel grezzi e nient'altro:
nessuno può partire da una forma che un altro ha già trovato per comporla con
una seconda. Un occhio va descritto ogni volta a partire dai pixel, e le
combinazioni di pixel che fanno un occhio sono innumerevoli: cambia la luce,
cambia l'inclinazione, cambia la taglia, e ogni variante va prevista da capo.

Con più strati il secondo lavora su ciò che ha trovato il primo, il terzo su
ciò che ha trovato il secondo. È la differenza tra una ricetta scritta tutta
come "prendi la farina, prendi l'uovo, prendi il burro" e una che a un certo
punto dice "prepara la besciamella", dando per fatto un pezzo di strada già
percorso: la seconda arriva allo stesso piatto con molte parole in meno.

`````

`````{tab} Superiore

Il teorema garantisce che, se l'attivazione $\sigma$ è continua e non
polinomiale (l'ipotesi è essenziale: la soddisfano la sigmoide e la ReLU
{cite}`leshno1993multilayer`, non un $\sigma$ polinomiale, che produrrebbe
solo polinomi), per ogni funzione continua $f:[0,1]^n\to\mathbb{R}$ e
ogni $\varepsilon>0$ esiste una rete a un solo strato nascosto

$$
g(\mathbf{x}) = \sum_{i=1}^{N} v_i\,\sigma\!\big(\mathbf{w}_i^\top \mathbf{x} + b_i\big)
\qquad\text{tale che}\qquad
\sup_{\mathbf{x} \in [0,1]^n}\,\lvert f(\mathbf{x}) - g(\mathbf{x})\rvert < \varepsilon ,
$$

dove $\mathbf{x} \in [0,1]^n$ è il singolo esempio in ingresso (un vettore,
non una matrice di dati), $\mathbf{w}_i \in \mathbb{R}^n$ è il vettore dei
pesi del neurone $i$-esimo, $b_i$ il suo bias e $v_i$ il peso con cui
contribuisce all'uscita.

Due avvertenze sul quantificatore, che è dove il teorema promette meno di quanto
sembri. La prima: è un risultato di **esistenza**, cioè di densità. Dice che la
rete c'è, non che la discesa del gradiente la trovi, né quanti esempi servano
per impararla. La seconda: il teorema da solo non dà nessun limite su $N$,
il numero di neuroni. Quel limite dipende dalla classe di funzioni che si vuole
approssimare, e conviene non generalizzare la frase che si sente più spesso.
Per le classi definite dalla sola regolarità (derivate limitate fino a un certo
ordine) il numero di neuroni cresce esponenzialmente nella dimensione
dell'ingresso, ed è la maledizione della dimensionalità, che colpisce
qualunque schema di approssimazione lineare e non le reti in particolare. Ma
non è una legge universale: Barron {cite}`barron1993universal` individua una
classe più ristretta, definita da una condizione sulla trasformata di Fourier,
per cui l'errore quadratico scende come $O(1/N)$ senza dipendere dalla
dimensione. La crescita esponenziale è una proprietà della classe di funzioni,
non delle reti a uno strato in quanto tali.

Sulla profondità, invece, le separazioni sono nette e dimostrate. Esistono
famiglie di funzioni rappresentabili da reti profonde con un numero di neuroni
*polinomiale* nella profondità, ma che richiedono larghezza *esponenziale* se
ci si limita a un solo strato {cite}`telgarsky2016benefits`; Eldan e Shamir
{cite}`eldan2016power` esibiscono una funzione che una rete con due strati
nascosti rappresenta con un numero di neuroni polinomiale nella dimensione
dell'ingresso, e che una rete con uno solo non riesce ad approssimare oltre
una certa soglia a meno di renderla esponenzialmente larga. Montúfar e colleghi
{cite}`montufar2014number` mostrano che il numero di regioni lineari che una
rete ReLU può generare cresce esponenzialmente con la profondità e solo
polinomialmente con la larghezza. Tradotto: la profondità compra efficienza
espressiva. È più economico comporre trasformazioni che allargarne una sola.

`````

## Quante regioni taglia una rete

Che la profondità arrivi allo stesso risultato con molti meno neuroni è, fin
qui, una frase; sotto c'è un conto, e con la ReLU si può fare a mano, perché
una rete di ReLU è **lineare a tratti**: lo spazio d'ingresso si divide in
**regioni lineari**, e dentro ciascuna la rete è una funzione affine, cioè una
retta (o un piano, o un iperpiano) e basta. Tutta la sua capacità di curvare
sta nel numero e nella disposizione di quelle regioni, quindi contarle è un
modo di misurare quanto una rete può essere complicata. Il confronto
interessante è a parità di parametri: quante regioni compra una rete, con un
bilancio dato, spendendolo in larghezza oppure in profondità.

`````{tab} Elementare

Prendi una striscia di carta lunga e dritta. Il tuo compito è farla a pezzi, e
lo strumento che hai è una matita: ogni tratto che tiri attraversa la striscia
in un punto solo e la divide lì. Conta che il tratto sia netto, perché è il
tratto a fare il pezzo: con un pennarello che sfuma non ci sarebbe niente da
contare. Dieci tratti fanno undici pezzi, cento tratti ne fanno centouno. Ogni
pezzo in più costa un tratto in più, e non c'è modo di fare meglio: la striscia
è distesa, e ogni tratto la tocca in un posto solo.

Adesso il secondo modo. Prima di tirare i tratti, pieghi la striscia a
fisarmonica: dieci pieghe, e i fogli si accatastano uno sull'altro, undici
strati sovrapposti. Ora un tratto solo, tirato con forza, attraversa tutti gli
undici strati insieme, e quando riapri la striscia quel tratto ha lasciato
undici segni invece di uno. Dieci tratti su una striscia piegata fanno il
lavoro di centodieci tratti su una distesa. Piega di nuovo, e i segni si
moltiplicano di nuovo.

Piegare però non è gratis, e qui sta il confronto vero. Un tratto in più costa
una cosa sola. Una piega in più costa molto di più, perché per piegare bene
bisogna decidere come ciascuno dei dieci fogli deve stare rispetto a ciascuno
degli altri dieci: è un costo che va come dieci per dieci, non come dieci. La
domanda diventa quindi se convenga spendere il proprio budget in tratti o in
pieghe, e la risposta è che a parità di spesa le pieghe vincono, e vincono di
molto.

Tre avvertenze, però, e sono la parte che di solito si salta. La prima: quel
numero è quanto le pieghe potrebbero fare, piegando alla perfezione. Piegate
male, due segni cadono nello stesso posto e i pezzi sono meno. La seconda: i
pezzi che escono da una piega sono copie l'uno dell'altro, perché sono
nati dallo stesso tratto passato attraverso più fogli; una striscia distesa,
di tratti indipendenti, ne fa pochi ma li mette dove vuole. La terza, che è la
più importante: avere un milione di pezzi non vuol dire aver disegnato la cosa
giusta. Se quello che serve è una curva liscia, i pezzi contano solo se cadono
dove serve, e il conto dei pezzi su questo non dice niente.

Un'ultima cosa sul tavolo di gioco: qui la carta è una striscia, cioè una cosa
lunga e basta. Con un foglio, che è quello che succede appena gli ingressi
sono più di uno, contare i pezzi diventa molto più difficile, e il vantaggio
delle pieghe cresce ancora.

`````

`````{tab} Superiore

Con un ingresso e un'uscita il conto è elementare. In una rete a uno strato
nascosto di $D$ unità ReLU, ogni unità ha un punto in cui la sua
pre-attivazione cambia segno: uno spigolo sulla retta d'ingresso. Al massimo
$D$ spigoli distinti tagliano la retta in

$$
N_1 = D + 1
$$

tratti, e su ciascuno la rete è affine. I parametri sono $D$ pesi d'ingresso,
$D$ bias, $D$ pesi d'uscita e un bias finale, cioè $3D+1$: il numero di
regioni cresce linearmente con il numero di parametri.

Con $K$ strati da $D$ unità l'argomento si ripete per induzione. Dentro una
regione già formata dai primi $k-1$ strati, ogni pre-attivazione dello strato
$k$ è affine in $x$, quindi ciascuna delle $D$ unità vi aggiunge al più uno
spigolo: quella regione si divide in al più $D+1$ sottoregioni. Perciò

$$
N_K \le (D+1)^K ,
\qquad
P_K = 3D + 1 + (K-1)\,D\,(D+1) ,
$$

con $P_K$ il numero di parametri (per $K=1$ la seconda formula restituisce
$3D+1$, come dev'essere). Le due quantità sono conteggi puri, adimensionali, e
si confrontano solo a parità di $P$: a profondità $K$ fissata, $D$ cresce come
$\sqrt{P/(K-1)}$ e le regioni come $P^{K/2}$, cioè polinomialmente nei
parametri con grado che cresce con la profondità, contro il grado uno della
rete piatta.

Il conto usa quattro ipotesi, e ciascuna limita che cosa se ne può concludere.
L'attivazione è ReLU, o comunque lineare a tratti con un solo spigolo: con
la sigmoide la rete non è lineare a tratti e le regioni non esistono.
L'ingresso ha una sola dimensione: con $D_i$ ingressi il conto si complica, e
Montúfar e colleghi {cite}`montufar2014number` prendono la cosa dall'altro
capo, esibendo reti (di larghezza $D \ge D_i$) che di regioni ne fanno almeno
$\Omega\big((D/D_i)^{(K-1)D_i} D^{D_i}\big)$: stesso verso, divario più largo.
Il $(D+1)^K$ è invece un limite superiore, che le costruzioni a ripiegamento
avvicinano senza toccarlo, e a cui una rete addestrata non si avvicina
affatto. E soprattutto: il numero di regioni misura
l’**espressività**, cioè quali funzioni la rete può rappresentare, e non dice
niente su quale la discesa del gradiente troverà né su come si comporterà sui
dati nuovi. Le regioni di una rete profonda portano dipendenze e simmetrie
(sono, letteralmente, copie ripiegate le une delle altre), quindi tante
regioni non equivalgono a tanta libertà, e non equivalgono in nessun modo a
generalizzare meglio. Quanto poco le due cose siano legate lo dice la doppia
discesa della {doc}`sezione su overfitting e validazione
</MachineLearning/overfitting-validazione>`: una rete con più parametri che
esempi generalizza meglio proprio dopo aver superato il punto in cui riesce a
memorizzarli tutti.

`````

Il confronto si fa in aritmetica, contando i parametri di una rete profonda
piccola e poi quanti neuroni una rete piatta ci mette a spendere altrettanto.

```python
# Un ingresso, un'uscita, K strati nascosti da D neuroni ReLU ciascuno.
def parametri(D, K):
    return 3 * D + 1 + (K - 1) * D * (D + 1)

def regioni(D, K):        # quante ne puo' fare al massimo
    return (D + 1) ** K

bilancio = parametri(10, 5)
print(f"profonda: K=5, D=10 -> {bilancio} parametri, {regioni(10, 5)} regioni")

# La rete a uno strato piu' piccola che quel bilancio se lo puo' permettere.
D = next(d for d in range(1, 10_000) if parametri(d, 1) >= bilancio)
print(f"piatta:   K=1, D={D} -> {parametri(D, 1)} parametri, "
      f"{regioni(D, 1)} regioni")
print(f"rapporto: {regioni(10, 5) / regioni(D, 1):.0f} volte")
```

```text
profonda: K=5, D=10 -> 471 parametri, 161051 regioni
piatta:   K=1, D=157 -> 472 parametri, 158 regioni
rapporto: 1019 volte
```

Cinque strati da dieci neuroni costano $471$ parametri e hanno per tetto
$161.051$ regioni. La rete a uno strato che spende quei parametri, anzi uno in
più, di tetto ne ha $158$: mille volte meno, con la stessa spesa. Il confronto
è fra tetti, e i due non sono ugualmente stretti: quello della rete piatta si
tocca, perché con un solo strato bastano $D$ spigoli distinti, quello della
profonda nessuna costruzione nota lo raggiunge. E vale per quello che la rete
può rappresentare, non per quello che imparerà.

## La pila, in codice

La gerarchia dai bordi agli oggetti si legge anche nel codice. Una piccola
rete convoluzionale scritta in {doc}`PyTorch </PyTorch/overview>` è
letteralmente una pila di strati che vanno dal semplice al complesso. Di che
cosa faccia ciascuno di quegli strati parla la
{doc}`sezione sulle reti convoluzionali </DeepLearning/reti-convoluzionali>`:
qui conta solo vedere la pila.

```python
from torch import nn

# in ingresso: un gruppo di immagini a colori, alte e larghe 128 pixel
model = nn.Sequential(
    nn.Conv2d(3, 32, 3), nn.ReLU(),    # primi strati: bordi e linee
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3), nn.ReLU(),   # strati intermedi: texture e parti
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3), nn.ReLU(),  # parti più grandi, oggetti
    nn.AdaptiveAvgPool2d(1),           # media di ogni foglio di risultati
    nn.Flatten(),
    nn.Linear(128, 10),                # la classe finale (un punteggio per classe)
)
```

I tre numeri dentro `nn.Conv2d(3, 32, 3)` si leggono così: quanti "fogli" di
numeri arrivano (3, cioè rosso, verde e blu), quanti filtri diversi applicare
(32) e quanto è larga la finestra che ciascun filtro guarda per volta (3 per 3
pixel). Ognuno di quei filtri produce un foglio di risultati, e quel foglio ha
un nome che accompagnerà tutto il capitolo: si chiama **feature map**, ed è la
mappa che segna punto per punto dove nell'immagine il filtro ha trovato ciò che
cerca. La riga `nn.AdaptiveAvgPool2d(1)` riduce ciascuna di quelle mappe a un
numero solo, la sua media. L'ultima riga produce dieci numeri, uno per classe:
si chiamano logit, sono punteggi grezzi, e vince il più alto. Ogni
`nn.Conv2d` più avanti nella pila costruisce feature più astratte a partire da
quelle dello strato
precedente: la stessa scala dai bordi agli oggetti della
{numref}`fig-gerarchia-feature`, resa in poche righe.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Nel machine learning classico è una persona a decidere quali caratteristiche
  dei dati contano, e a scriverle nel codice. Una rete profonda se le
  inventa da sola, partendo dai dati grezzi.
- Le costruisce a strati, dal semplice al complesso: prima bordi e linee,
  poi pezzi riconoscibili (un occhio, una ruota), infine l'oggetto intero.
- Non è esplosa prima del 2012 perché servivano tre cose insieme, come la
  legna, l'aria e la scintilla: milioni di fotografie già etichettate,
  schede grafiche abbastanza veloci e tre accorgimenti precisi (la ReLU
  al posto delle funzioni che spegnevano il segnale, il dropout contro
  l'imparare a memoria, il moltiplicare le foto con ritagli e specchiature).
  Nel 2012 c'erano tutte e tre, e alla gara di ImageNet vinse AlexNet.
  (La ReLU è la regola più semplice possibile: i numeri positivi passano come
  sono, i negativi diventano zero.)
- Uno strato solo, se lo si facesse enorme, in teoria basterebbe: il teorema
  però dice che una rete così *esiste*, non che l'addestramento la sappia
  trovare. E la profondità arriva allo stesso risultato con molti meno neuroni,
  perché ogni strato può costruire sopra quello che ha trovato il
  precedente, invece di descrivere ogni forma a partire dai pixel.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Il deep learning apprende le feature dai dati grezzi, invece di
  richiederle ingegnerizzate a mano come il ML classico.
- Le rappresentazioni sono gerarchiche: bordi → texture e parti → oggetti,
  un livello di astrazione per strato.
- È esploso dopo il 2012 (ImageNet, AlexNet) grazie alla triade dati + GPU +
  algoritmi, non a una singola idea nuova.
- Una rete larga e piatta è universale in teoria (con un'attivazione non
  polinomiale), ma la profondità ottiene la stessa espressività con molti
  meno neuroni: comporre conviene. L'universalità però è un'esistenza, non
  un'apprendibilità.
```
`````

[^backprop-storia]: Il conto che sta sotto la retropropagazione (partire
    dall'errore in fondo alla rete e risalire all'indietro, strato per strato,
    per sapere quanto ciascun peso, cioè ciascuno dei numeri regolabili
    della rete, ha contribuito) era già noto ai matematici dal 1970, quando
    Seppo Linnainmaa lo descrisse in tutta generalità: è quella che oggi si
    chiama *differenziazione automatica in modalità inversa*. Paul Werbos la
    applicò alle reti neurali in un lavoro presentato nel 1981 e pubblicato
    l'anno dopo, e a metà anni Ottanta fu riscoperta per conto proprio da LeCun
    e da Parker. Il merito del 1986 è averla mostrata al mondo, con esperimenti
    convincenti, nel posto giusto: un articolo su *Nature*.

[^universalita]: Il teorema arriva in tre tempi. Cybenko lo dimostra nel 1989
    {cite}`cybenko1989approximation` per le attivazioni a forma di S, cioè per
    quelle curve che schiacciano i numeri in un intervallo stretto e che si
    chiamano *sigmoidi*; Hornik lo estende nel 1991
    {cite}`hornik1991approximation` a
    tutte quelle che restano confinate fra due valori; la forma generale,
    quella che copre anche la ReLU, arriva nel 1993 con Leshno e colleghi
    {cite}`leshno1993multilayer`.
