# Classificare descrivendo: analisi discriminante e naive Bayes

Chiedi a un ornitologo come distingua una cornacchia da una gazza e non ti
risponderà con un confine. Ti dirà com'è fatta una cornacchia: grigia e nera,
tozza, coda corta, becco robusto. E com'è fatta una gazza: bianca e nera, più
snella, con quella coda lunghissima che non si può sbagliare. Il confine fra le
due specie non lo ha mai tracciato; ce l'ha in testa come conseguenza di due
descrizioni.

Tutti i classificatori visti finora fanno il contrario. La regressione logistica
cerca la retta che separa meglio; l'albero cerca la sequenza di domande che
separa meglio; la SVM cerca il corridoio più largo fra le due classi. Nessuno di
loro sa com'è fatta una classe: sanno solo dove finisce una e comincia
l'altra, che è un'informazione più povera e, spesso, più difficile da ottenere.

L'altra strada è opposta: si impara a descrivere ciascuna classe, una per
volta, e il confine si ricava dopo, con una riga di conto. La famiglia
si chiama dei modelli **generativi**, e i suoi tre membri classici (analisi
discriminante lineare, quadratica, e naive Bayes) hanno tutti più di
cinquant'anni e sono tutti ancora in uso.

## Due modi di rispondere alla stessa domanda

Alla fine ogni classificatore deve produrre la stessa cosa: dato un esempio,
quanto è probabile che appartenga a ciascuna classe. Cambia da dove ci si arriva.

`````{tab} Elementare

Una moneta raccolta per terra, da uno o da due euro? A occhio lo dice il colore,
ma facciamo come la macchinetta del caffè, che il colore non lo guarda. Sul
tavolo ci sono una bilancia, un calibro e un barattolo di mille monete già
riconosciute.

Versi il barattolo sul foglio e segni ogni moneta come un punto, il peso in
orizzontale e il diametro in verticale. Poi tiri la riga che tiene i due mucchi
più separati. Adesso pesi la moneta nuova, la misuri, guardi da che parte cade.
Della moneta da un euro non hai imparato niente in particolare. Hai imparato
dove finisce.

Oppure il barattolo lo dividi in due mucchi. Delle monete da un euro calcoli
peso medio, diametro medio, e di quanto le singole si scostano di solito da
quelle medie; poi rifai tutto sul mucchio da due. Adesso hai due descrizioni, e
alla moneta nuova fai due domande. Quanto sarebbe strana fra quelle da un euro?
E fra quelle da due? Strana vuol dire lontana dal centro, contata in
scostamenti: se le monete da un euro pesano in media 7,5 grammi e se ne scostano
di solito di un decimo, una da 8 grammi è cinque scostamenti più in là. Vince il
mucchio che la trova meno strana, con una correzione che nel barattolo si legge:
settecento delle mille erano da un euro e trecento da due, quindi a parità di
stranezza la dai da un euro.

Con una descrizione in mano fai una cosa che con la riga non si poteva fare.
Sorteggi un peso e un diametro che le stiano dentro, ed ecco sul foglietto una
moneta da un euro credibile che nel barattolo non c'era. Il nome generativo
viene da qui, ed è la stessa idea, con descrizioni molto più ricche, dei
programmi che inventano immagini e frasi.

Il giorno che nel barattolo finiscono i cinquanta centesimi, ne calcoli media e
scostamenti e la terza descrizione è pronta; le prime due restano quelle di
ieri, mentre la riga andrebbe ritracciata da capo.

E la moneta col bordo ammaccato, che nel calibro non entra dritta? La giudichi
col solo peso: ogni descrizione sa dire quanto pesano le monete di quel taglio
(il taglio di una moneta è il suo valore, uno o due euro), mentre la riga, senza
il diametro, non sa dove metterla.

Togli dal barattolo tutto tranne venti monete, e di ognuna misura anche
spessore, colore del bordo e usura: poche monete, tante misure. Peso medio e
scostamenti delle monete da un euro escono comunque, perché li calcoli su quelle
e basta, e nella media ogni moneta pesa poco. La riga invece, che con cinque
misure non si disegna più su un foglio ma resta un confine netto fra i due
mucchi, la decidono soprattutto le poche monete vicine al confine: con venti
monete basta spostarne una perché giri.

Poi sul tavolo arriva un gettone del luna park, o una moneta straniera, o un
falso fatto male. Sta lontano da tutti e due i centri, e le due descrizioni lo
trovano stranissimo tutte e due. La riga quella parola non ce l'ha: qualunque
cosa le metti sopra cade a destra o a sinistra, e il gettone esce come una
moneta da due euro con la stessa disinvoltura di una vera.

Quello che le due descrizioni sanno dire, e la riga no, è dove sul foglio le
monete sono fitte e dove sono rade. Costruire una descrizione così si chiama
**stima di densità**, e «densità» vuol dire proprio quanto sono fitte, come la
densità degli abitanti di una città. E come gli abitanti di una regione, la
fittezza da spartire è una sola: chi dice che le monete sono fitte in un posto
deve dire che sono rade altrove. Se no basterebbe dire «fitte dappertutto», e
ogni moneta, gettone compreso, sembrerebbe normale.

Con questa regola, una descrizione è tanto migliore quanto meno trova strane le
monete vere del barattolo, purché resti semplice: un centro per mucchio, e
quanto e in che direzione il mucchio si allarga, la sua forma. Una che si
incollasse alle mille monete una per una le troverebbe normalissime, e
troverebbe strana qualunque moneta nuova.

E se il barattolo arrivasse senza cartellini? La descrizione si fa lo stesso,
sul barattolo intero: si cercano due mucchi invece di uno, e si indovina da sé
quale moneta sta in quale. Il gettone resta stranissimo, perché è lontano da
tutte le monete, di qualunque taglio siano.

Con due misure tutto questo regge. Con migliaia, come i puntini di una
fotografia, può ingannare. Ogni foto vera si scosta dalla foto media in mille
piccoli modi, uno per puntino, e così nessuna sta proprio al centro, dove la
descrizione è più fitta: stanno tutte in un anello tutt'intorno, e una foto di
tutt'altro genere può cadere più vicino di loro al punto più fitto. Programmi
che avevano studiato fotografie di animali e di camion hanno trovato le
fotografie di numeri civici, mai viste, più normali di quelle su cui avevano
studiato.

Il conto si paga quando la descrizione che ti sei dato è sbagliata. Hai dato per
buono che ogni taglio faccia un mucchio solo, tondo e compatto, e invece,
mettiamo, le monete da due euro sono di due serie, una più pesante e una più
leggera, cioè due mucchietti staccati. La tua descrizione ne fa la media e
finisce a metà strada, dove monete da due euro non ce ne sono, e da lì sbagli
anche pezzi che una riga tirata a occhio avrebbe messo dalla parte giusta.
Raccontare com'è fatto ogni taglio vuol dire pagare ogni dettaglio raccontato
male, compresi quelli che alla domanda non servivano.

`````

`````{tab} Superiore

La distinzione è quella fra classificatori **discriminativi** e generativi.

Un classificatore discriminativo modella direttamente la posteriore
$p(y \mid \mathbf{x})$ (regressione logistica, alberi, reti) o addirittura solo il
confine di decisione senza probabilità (SVM, percettrone). Un classificatore
generativo modella la congiunta $p(\mathbf{x}, y) = p(\mathbf{x} \mid y)\,p(y)$,
cioè la distribuzione dei dati dentro ciascuna classe più la frequenza delle
classi, e ricava la posteriore con il teorema di Bayes:

$$
p(y = k \mid \mathbf{x}) =
\frac{p(\mathbf{x} \mid y = k)\; \pi_k}
     {\sum_{j} p(\mathbf{x} \mid y = j)\; \pi_j},
\qquad \pi_k = p(y = k).
$$

Il nome «generativo» viene da una proprietà che il discriminativo non ha:
avendo $p(\mathbf{x} \mid y)$ si possono campionare esempi nuovi di una
classe. Il modello non riassume i dati, li sa rifare, ed è la stessa parola
dei modelli che generano immagini e testo, dove la
famiglia è la stessa e cambia solo quanto è espressiva la $p(\mathbf{x} \mid y)$.

Le conseguenze pratiche di modellare $p(\mathbf{x}\mid y)$ invece di
$p(y \mid \mathbf{x})$ sono quattro, e tornano tutte più avanti nel libro:

1. si ottiene una densità, quindi il rilevamento di anomalie e degli input
   fuori distribuzione ha un punteggio naturale, affidabile in poche
   dimensioni;
2. i parametri si stimano in forma chiusa, ciascuno da tutti i dati della
   sua classe e non tutti insieme dentro un'unica ottimizzazione, e questo si
   sente quando gli esempi sono pochi rispetto alle feature;
3. le classi si stimano una alla volta e indipendentemente: aggiungere una
   classe non richiede di riaddestrare le altre, e i dati mancanti si trattano
   marginalizzando invece che imputando;
4. se il modello di $p(\mathbf{x}\mid y)$ è sbagliato, l'errore si paga anche
   dove non serviva: il generativo spende capacità a descrivere aspetti dei dati
   che non contano per la decisione.

La **stima di densità** è il compito su cui poggia quel punteggio: dato un
campione $\mathbf{x}_1,\dots,\mathbf{x}_n$ estratto i.i.d. da una
distribuzione ignota $p$ sullo spazio dei dati $\mathcal{X}$, trovare in una
famiglia $\mathcal{P}$ una densità vicina a $p$. La vicinanza si misura di
solito con la {doc}`divergenza di Kullback-Leibler
</Matematica/teoria-informazione>`. Poiché

$$
\mathrm{KL}(p\,\|\,q) = -H(p) - \mathbb{E}_{p}[\log q],
$$

dove $H(p)$ è l'entropia di $p$ (differenziale, se $p$ è una densità) e non
dipende da $q$, la migliore approssimazione nella famiglia,
$p^\star = \arg\min_{q \in \mathcal{P}} \mathrm{KL}(p\,\|\,q)$, è la $q$ che
rende massimo $\mathbb{E}_p[\log q]$. Quel valore atteso non si conosce, ma il
campione lo stima, e la stima di densità per massima verosimiglianza

$$
\hat p = \arg\max_{q \in \mathcal{P}}
\frac{1}{n}\sum_{i=1}^{n} \log q(\mathbf{x}_i)
$$

è la minimizzazione empirica della KL. Conta anche la famiglia: su una troppo
ricca il massimo degenera, e una componente di mistura che si stringe su un
punto solo manda la verosimiglianza all'infinito, come mostra la
{doc}`sezione su riduzione e clustering <riduzione-clustering>`. Senza
etichette è un compito non supervisionato (misture gaussiane, flussi, modelli
generativi profondi); con le etichette, un classificatore generativo ne risolve
uno per classe, e la $p(\mathbf{x}) = \sum_k \pi_k\, p(\mathbf{x}\mid y=k)$ che
ne risulta è una densità a sua volta. In alta dimensione, però, il punteggio
inganna, perché la regione di densità più alta non coincide con quella in cui
cadono i campioni tipici: flussi, VAE e PixelCNN addestrati su fotografie di
oggetti comuni danno una verosimiglianza più alta a fotografie di numeri
civici mai viste {cite}`nalisnick2019do`, e la {doc}`sezione sugli usi della
verosimiglianza esatta </VerosimiglianzaEsatta/a-che-serve>` racconta perché.

`````

Resta da dire di quale descrizione stiamo parlando. Il caso classico è il più
semplice possibile: ogni classe è una campana gaussiana, cioè una collina di
probabilità con un **centro** (dove sta il tipico esemplare della classe) e una
**forma** (quanto e in quali direzioni gli esemplari se ne allontanano). Due
ingredienti, e si calcolano con due medie: la media dei punti della classe dà il
centro, la media dei prodotti dei loro scarti presi a due a due (peso per
peso, peso per diametro, diametro per diametro) dà la forma, cioè la matrice di
covarianza.

È un caso fortunato, e conviene dire subito perché. Quando le etichette non ci
sono, gli stessi due ingredienti vanno indovinati insieme all'appartenenza
di ciascun punto, e ci vuole una procedura iterativa: è quello che la
{doc}`sezione su riduzione e clustering <riduzione-clustering>` farà con le
misture gaussiane e l'algoritmo EM. Qui le etichette ci sono, quindi non c'è
niente da indovinare.

## Analisi discriminante: lineare o quadratica

Ronald Fisher affronta il problema nel 1936, su dei fiori
{cite}`fisher1936use`. Il botanico Edgar Anderson aveva misurato lunghezza e
larghezza di petali e sepali di centocinquanta iris, cinquanta per ciascuna di
tre specie; la domanda era se quelle quattro misure bastassero a distinguerle.
Conviene riportare l'avvertenza che Fisher mette nel suo stesso articolo,
perché quasi nessuno di quelli che riusano questi dati la conosce: due delle
tre specie vengono dalla penisola di Gaspé, in Québec, mentre la terza, *Iris
virginica*, «differisce dagli altri due campioni per non essere stata raccolta
nella stessa colonia naturale», il che «potrebbe alterare parecchio sia le
medie sia le loro variabilità»: una parte di quello che distingue *virginica*
dalle altre due può venire dal posto, e non dalla specie. Fisher cercava la
combinazione delle quattro misure che separasse al meglio le specie, e il
metodo che ne uscì porta il suo nome. È lo stesso `iris` che il {doc}`capitolo
sull'interpretabilità </Interpretabilita/overview>` darà in pasto a un albero,
ed è probabilmente il dataset più riusato della storia della
statistica.[^eugenics]

[^eugenics]: L'articolo esce sugli *Annals of Eugenics*, che è il nome della
    rivista fino al 1954, e Fisher ne fu a lungo redattore. È un fatto
    bibliografico e non un dettaglio da nascondere: la statistica inferenziale
    del primo Novecento nasce in buona parte dentro quel programma di ricerca,
    e i metodi che ne uscirono sono validi indipendentemente da esso. Il
    {doc}`capitolo sull'AI responsabile </AIResponsabile/equita-e-bias>` torna
    sul rapporto fra strumenti statistici e usi che se ne fanno.

`````{tab} Elementare

Torniamo alle monete, e mettiamo che i due tagli abbiano la stessa forma, nel
senso che variano allo stesso modo (chi è più pesante è anche un po’ più largo,
nella stessa misura per tutte e due), e che a distinguerli sia solo dove sta il
centro.

In questo caso capire da quale taglio viene una moneta nuova è quasi come
chiedersi a quale dei due centri sono più vicino. Il «quasi» sta in due
accortezze. La prima è misurare la distanza nella forma giusta: se il peso di
solito si scosta di un decimo di grammo e il diametro di un centesimo di
millimetro, un decimo di grammo di scarto è normale e un decimo di millimetro
no. La seconda è la solita correzione per quanto sono comuni i due tagli, che
non sparisce nemmeno qui. Il confine che ne esce è una retta: con la distanza
misurata nella forma giusta, i punti ugualmente lontani da due centri stanno
sull'asse del segmento che li unisce, come a geometria, e la correzione lo
sposta soltanto, parallelo a sé stesso. Il metodo porta il nome di Fisher: è
l'analisi discriminante lineare, o LDA.

Fisher, però, non partiva dalle due descrizioni, e ci arrivava da un'altra
parte. Cercava un modo di mescolare le misure in un numero solo, per esempio due
volte il peso in grammi più tre volte il diametro in millimetri, che per una
moneta da un euro (7,5 grammi, 23,25 millimetri) fa circa 85 e per una da due
(8,5 e 25,75) circa 94. Messo come un punto su una linea, quel numero doveva
tenere i due tagli lontani fra loro e ciascuno ben raccolto. I due centri
distano circa $9$, e ciascun mucchio si allarga di circa due decimi attorno al
suo: nove diviso due decimi fa $45$, e la ricetta migliore è quella che rende
più grande questo rapporto fra la distanza dei centri e la larghezza dei mucchi.

Poi si sceglie sulla linea una soglia, sopra da due euro e sotto da uno. Una
moneta da $7{,}5$ grammi e $25$ millimetri e una da $9$ grammi e $24$ millimetri
fanno tutte e due $90$: ogni grammo in più si compensa con due terzi di
millimetro in meno, sempre nello stesso rapporto, e così sul foglio le monete
che cadono proprio sulla soglia stanno su una retta, inclinata come il confine
trovato con la vicinanza ai centri: le due strade arrivano allo stesso confine.
Ed è una retta come quella tirata sul barattolo all'inizio. Cambia come la si
trova, là guardando dove cadono le monete, qui calcolandola dalle due
descrizioni, e finché le descrizioni sono giuste le due finiscono quasi nello
stesso posto.

Dove mettere la soglia, però, la ricetta di Fisher non lo dice. A metà strada
fra i due centri va bene solo se i due tagli sono ugualmente comuni. Con
settecento monete da un euro e trecento da due serve di nuovo la solita
correzione, che sposta la soglia verso il mucchio da due e lascia più spazio
alle monete da uno: mettendola a metà strada si ottiene una retta parallela a
quella giusta, ma troppo vicina al mucchio da un euro.

Con tre tagli (uno, due euro e cinquanta centesimi) un numero solo di solito non
basta. Ne bastano però due, perché tre centri stanno sempre su un foglio piano,
e le differenze fra i centri, misurate nella forma giusta, stanno tutte lì
sopra. Serve quando le misure sono tante. Con cinque misure ogni moneta è un
punto in uno spazio a cinque dimensioni, che non si disegna; ma i tre centri
stanno comunque su un foglio, e per decidere a quale centro una moneta è più
vicina conta soltanto dove cade la sua ombra su quel foglio, perché il pezzo di
distanza che resta fuori è uguale per tutti e tre. L’ombra si dice con due
numeri, e con peso e diametro soltanto il foglio lo hai già.

Se invece i due tagli hanno forme diverse (uno varia tanto in peso, l'altro
tanto in diametro) la vicinanza al centro da sola inganna. Un taglio molto
variabile trova poco strano qualunque valore, e a lasciarlo fare si prenderebbe
tutte le monete dubbie; quindi alla stranezza che trova si aggiunge una tassa,
tanto più alta quanto più quel taglio è sparpagliato. Con una forma sola quella
tassa sarebbe stata uguale per i due tagli e non avrebbe spostato il confine di
un millimetro; con due forme diverse decide. Impari una forma per ciascuna, e il
confine che ne esce si incurva. È l'analisi discriminante quadratica, QDA.

Sembra che convenga sempre la seconda, visto che può fare tutto quello che fa la
prima. Non è così, perché imparare una forma per ciascun taglio vuol dire
stimare il doppio dei numeri con le stesse monete, quindi stimarli peggio. Con
tagli che davvero hanno la stessa forma, la QDA spende numeri per scoprire una
cosa che era già vera e ci rimette; con tagli di forma diversa, la LDA non ha
proprio modo di accorgersene. È il compromesso bias-varianza: una descrizione
troppo rigida contro una troppo libera per le monete che ha.

C'è anche una via di mezzo, e si prende quando le monete a disposizione sono
poche. Si stimano le due forme separate e poi le si tira verso la forma unica,
tenendo un po’ di ciascuna. Quanto tirare è una manopola, e la si ferma dove le
monete tenute da parte per la prova vengono riconosciute meglio.

`````

`````{tab} Superiore

Si assume $p(\mathbf{x} \mid y = k) = \mathcal{N}(\mathbf{x} \mid
\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$. Sostituendo in Bayes e prendendo il
logaritmo, la regola di decisione confronta le **funzioni discriminanti**

$$
\delta_k(\mathbf{x}) = -\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_k)^{\!\top}
\boldsymbol{\Sigma}_k^{-1}(\mathbf{x}-\boldsymbol{\mu}_k)
-\tfrac{1}{2}\log\lvert\boldsymbol{\Sigma}_k\rvert + \log \pi_k ,
$$

e assegna $\mathbf{x}$ alla classe con $\delta_k$ massima. Il primo termine è la
**distanza di Mahalanobis** al quadrato, cioè la distanza euclidea misurata
nella metrica che la covarianza della classe induce: è la formalizzazione di
«misurare la distanza nella forma giusta».

Il confine fra due classi è $\delta_k(\mathbf{x}) = \delta_\ell(\mathbf{x})$, e
la sua forma dipende da una sola ipotesi.

Con covarianze diverse (QDA) i termini
$\mathbf{x}^{\!\top}\boldsymbol{\Sigma}_k^{-1}\mathbf{x}$ non si cancellano, e il
confine è una quadrica (iperbole, ellisse, parabola secondo il caso).

Con covarianze uguali (LDA), cioè $\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$
per ogni $k$, il termine quadratico è lo stesso nelle due funzioni
discriminanti e sparisce nella differenza. Sviluppando:

$$
\delta_k(\mathbf{x}) = \mathbf{x}^{\!\top} \boldsymbol{\Sigma}^{-1}
\boldsymbol{\mu}_k
- \tfrac{1}{2} \boldsymbol{\mu}_k^{\!\top} \boldsymbol{\Sigma}^{-1}
\boldsymbol{\mu}_k + \log \pi_k
+ \underbrace{\bigl(-\tfrac{1}{2}\mathbf{x}^{\!\top}\boldsymbol{\Sigma}^{-1}
\mathbf{x} - \tfrac{1}{2}\log\lvert\boldsymbol{\Sigma}\rvert\bigr)}_{
\text{uguale per ogni } k},
$$

dove il termine raccolto dalla graffa non è costante in $\mathbf{x}$ (è proprio
quello quadratico), ma è lo stesso per tutte le classi, quindi sparisce nella
differenza $\delta_k - \delta_\ell$, che è ciò da cui il confine dipende.
Tolto quello, quel che resta è affine in $\mathbf{x}$: il confine è un
iperpiano. È anche l'enunciato che il collaudo numerico verifica, dato che
`decision_function` restituisce proprio quella differenza.

Il nome ha una storia più lunga di questa derivazione. Fisher nel 1936 non
suppone classi gaussiane: cerca la direzione $\mathbf{a}$ che massimizza il
rapporto fra la dispersione delle medie di classe e quella dentro le classi,

$$
J(\mathbf{a}) = \frac{\mathbf{a}^{\!\top}\mathbf{S}_B\,\mathbf{a}}{\mathbf{a}^{\!\top}\mathbf{S}_W\,\mathbf{a}},
$$

dove $\mathbf{S}_B$ è la covarianza delle medie di classe e $\mathbf{S}_W$ la
covarianza comune dentro le classi; per due classi il massimo è in
$\mathbf{a} \propto \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)$,
la stessa direzione che la regola bayesiana dà con
$\boldsymbol{\Sigma} = \mathbf{S}_W$. La soglia invece nel quoziente non
compare. La regola bayesiana taglia in

$$
\mathbf{a}^{\!\top}\mathbf{x} = \tfrac{1}{2}\,\mathbf{a}^{\!\top}
(\boldsymbol{\mu}_0 + \boldsymbol{\mu}_1) - \log\frac{\pi_1}{\pi_0},
$$

con $\mathbf{a} = \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)$
senza riscalarlo, e tagliare nel punto medio delle medie proiettate dà la stessa
regola solo con classi equiprobabili. Per Fisher la linearità era dunque
un'ipotesi, perché la regola è una combinazione lineare per costruzione; la
lettura gaussiana, venuta dopo, la trasforma in una conseguenza dell'aver
condiviso la covarianza. Con $K$ classi $\mathbf{S}_B$ ha rango al più $K-1$,
quindi il problema agli autovalori generalizzati
$\mathbf{S}_B\mathbf{a} = \lambda\,\mathbf{S}_W\mathbf{a}$, i cui autovettori
sono i punti stazionari di $J$, ha al più $\min(K-1, d)$ autovalori non nulli.
Proiettare su quelle direzioni non perde niente per la regola LDA: nelle
coordinate sbiancate da $\mathbf{S}_W$ i $K$ centri stanno in un sottospazio
affine di dimensione al più $K-1$, e le componenti ortogonali a quel
sottospazio pesano allo stesso modo su tutte le distanze dai centri
{cite}`hastie2009elements`. È da qui che viene la riduzione a $K-1$ dimensioni.

Il conto dei parametri spiega il compromesso. Con $d$ feature e $K$ classi, la
LDA stima $K$ medie più una covarianza, cioè $Kd + d(d+1)/2$ numeri; la QDA
ne stima $K$, cioè $Kd + K\,d(d+1)/2$. Per $d = 20$ e $K = 2$ sono $250$ contro
$460$: quasi il doppio, e la parte che raddoppia è quella difficile, perché
stimare una covarianza è stimare circa $d^2/2$ numeri da dati che ne
informano poco.
Da qui la **regularized discriminant analysis** di Friedman
{cite}`friedman1989regularized`, che interpola fra le due mescolando
$\boldsymbol{\Sigma}_k$ con la covarianza comune. In scikit-learn quella
interpolazione non c'è (`QuadraticDiscriminantAnalysis(reg_param=...)` fa solo
l'altra metà, cioè tira ciascuna $\boldsymbol{\Sigma}_k$ verso l'identità), e
c'è invece un rimedio diverso e complementare,
`LinearDiscriminantAnalysis(shrinkage=...)`, che tira la covarianza comune
verso un multiplo dell'identità:
$(1-\alpha)\hat{\boldsymbol{\Sigma}} +
\alpha\,\frac{\operatorname{tr}\hat{\boldsymbol{\Sigma}}}{d}\mathbf{I}$.
Cura cioè il rumore della stima, non la differenza fra le classi (e vuole
`solver="lsqr"` o `"eigen"`: con il solver predefinito il parametro solleva un
errore invece di essere ignorato, che è il modo giusto di comportarsi).

Due parentele, con la regressione logistica e con le misture gaussiane. La LDA
produce una posteriore che, per due classi, è esattamente una sigmoide di una
funzione affine, cioè la stessa forma funzionale della regressione logistica; e
il legame con le misture gaussiane è ancora più stretto, perché la LDA è una
mistura gaussiana a covarianza condivisa in cui le variabili latenti sono
osservate. All'EM della sezione sul clustering la seconda parentela si legge al
contrario: il passo E, che là dovrà stimare le responsabilità, qui è dato
(valgono $0$ e $1$, e le sanno tutti), e resta il solo passo M, che sono le due
medie e la covarianza comune, eseguito una volta.

`````

Le due situazioni, misurate: due classi con la stessa forma e due classi con
forme diverse, gli stessi $200$ esempi di addestramento, la stessa prova su
ventimila esempi mai visti.

```python
import numpy as np
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.linear_model import LogisticRegression

def genera(n, forma_uguale, seme):
    """Due classi gaussiane, con la stessa forma oppure con forme diverse."""
    r = np.random.default_rng(seme)
    C0 = np.array([[2.0, 1.2], [1.2, 1.0]])
    C1 = C0 if forma_uguale else np.array([[0.6, -0.5], [-0.5, 2.2]])
    y = r.integers(0, 2, n)
    return (np.where(y[:, None] == 0,
                     r.multivariate_normal([0, 0], C0, n),
                     r.multivariate_normal([1.6, 1.2], C1, n)), y)

print(f"{'':30} {'LDA':>13} {'QDA':>13} {'logistica':>13}")
for uguale in (True, False):
    Xte, yte = genera(20_000, uguale, 999)
    col = []
    for M in (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
              LogisticRegression):
        # venti addestramenti da 200 esempi: la deviazione dice quanto ballano
        s = [M().fit(*genera(200, uguale, 10 + k)).score(Xte, yte) for k in range(20)]
        col.append(f"{np.mean(s):.3f} ±{np.std(s):.3f}")
    etichetta = "stessa forma, due classi" if uguale else "forme diverse"
    print(f"{etichetta:30} {col[0]:>13} {col[1]:>13} {col[2]:>13}")
```

```text
                                         LDA           QDA     logistica
stessa forma, due classi        0.728 ±0.003  0.726 ±0.003  0.728 ±0.003
forme diverse                   0.813 ±0.005  0.855 ±0.002  0.810 ±0.007
```

La colonna del $\pm$ è la deviazione standard fra venti addestramenti, cioè
quanto quel numero balla se si ripete tutto: senza di lei il resto della
tabella non si legge. Nella prima riga i tre valori stanno dentro un $\pm 0{,}003$
l'uno dall'altro: sono lo stesso numero scritto tre volte, e la conclusione è
che quando le classi hanno la stessa forma non c'è niente da guadagnare a
imparare due forme (né a passare a un discriminativo). Nella seconda riga la QDA
sta quattro punti sopra le altre due, con la deviazione più piccola di tutte: lì
la differenza è reale.

Notare anche chi resta indietro insieme a chi: LDA e regressione logistica si
muovono appaiate in tutte e due le righe, perché tracciano lo stesso tipo di
confine e differiscono solo su come ne stimano la posizione.

E che il confine della LDA sia davvero una retta non è una cosa da credere
sulla parola. Il collaudo è questo: si prende il punteggio con cui il modello
decide, si cerca la retta che meglio lo imita, e si guarda di quanto i due
si scostano nel punto peggiore. Se il punteggio è una retta lo scarto deve
venire zero.

```python
X, y = genera(4000, True, 1)
lda = LinearDiscriminantAnalysis().fit(X, y)
qda = QuadraticDiscriminantAnalysis().fit(X, y)

r = np.random.default_rng(0)
P = r.normal(0, 3, (500, 2))          # cinquecento punti a caso nel piano
base = np.c_[P, np.ones(len(P))]      # la piu' generale funzione affine del piano

def scarto_dall_affine(decisione):
    """Quanto la funzione di decisione si scosta dalla piu' vicina retta."""
    coef = np.linalg.lstsq(base, decisione(P), rcond=None)[0]
    return np.abs(decisione(P) - base @ coef).max()

print(f"LDA, scarto dall'affine: {scarto_dall_affine(lda.decision_function):.2e}")
print(f"QDA, scarto dall'affine: {scarto_dall_affine(qda.decision_function):.2e}")
```

```text
LDA, scarto dall'affine: 8.88e-15
QDA, scarto dall'affine: 7.21e+00
```

Lo scarto della LDA è $10^{-15}$, cioè un milionesimo di miliardesimo: zero,
e quel che si legge sono gli arrotondamenti del calcolatore.
Il punteggio della LDA è una retta, non le somiglia. La QDA se ne scosta di
$7{,}2$, e nessuna retta la approssima. È la stessa cosa che si ottiene con
l'algebra, dove i termini al quadrato si cancellano fra le due classi perché
sono identici.

```{figure} ../figures/lda-qda-naive-bayes.svg
:name: fig-lda-qda
:alt: "Tre pannelli sugli stessi due gruppi di punti, teal e terracotta, che si sovrappongono in parte. Nel primo, LDA, le due classi hanno la stessa identica ellisse di forma in due posizioni diverse, e il confine fra loro e una retta. Nel secondo, QDA, ogni classe ha la sua ellisse, con orientamenti diversi, e il confine e una curva. Nel terzo, naive Bayes gaussiano, le ellissi hanno gli assi obbligatoriamente paralleli agli assi del grafico, perche l ipotesi di indipendenza vieta le diagonali, e il confine e una curva diversa dalla precedente."
:width: 100%

Le tre ipotesi, disegnate. L'ovale che circonda ciascun gruppo (un’ellisse)
è la forma che quel metodo si concede per descrivere la classe, e da sola decide
la forma del confine: una sola forma, la stessa per le due classi in due
posizioni diverse, dà una retta; due forme diverse danno una curva. Nessuno
dei confini è stato disegnato: sono tutti conseguenze delle ellissi. Il terzo
pannello anticipa il naive Bayes gaussiano, che le ellissi le obbliga a stare
dritte, con gli assi paralleli a quelli del grafico.
```

### La LDA come regressione, e che cosa ne nasce

Con più di due classi alla LDA si arriva anche per regressione, e da quella
strada nascono due sue estensioni: la *flexible discriminant analysis* (FDA),
che incurva il confine, e la *penalized discriminant analysis* (PDA), che regge
quando le misure sono centinaia e ordinate, come i valori di uno spettro
{cite}`hastie1994flexible,hastie1995penalized`. Il punto di partenza è un modo
di sbagliare: la regressione lineare sulle indicatrici delle classi, che con tre
classi può *mascherarne* una.

`````{tab} Elementare

Con tre tagli viene in mente una scorciatoia. A ogni moneta si danno tre voti,
uno per taglio: uno al taglio che è, zero agli altri due (voti così, fatti solo
di uno e di zero, si chiamano *indicatrici*). Per ciascun voto si cerca una
ricetta alla Fisher, tanto peso più tanto diametro, che lo indovini il meglio
possibile, e alla moneta nuova si dà il taglio che prende il voto più alto.

Con le monete da un euro, da cinquanta centesimi e da due, però, i cinquanta
centesimi stanno in mezzo: più larghi di quelle da un euro (24,25 millimetri
contro 23,25) e più stretti di quelle da due (25,75). La ricetta del voto «due
euro» sale andando verso le monete larghe, quella del voto «un euro» scende, e
quella del voto «cinquanta centesimi» dovrebbe salire in mezzo e scendere ai due
lati. Una somma di tanto peso e tanto diametro questo non lo sa fare: resta
quasi piatta, e vicino al centro le altre due la superano. Molti cinquanta
centesimi finiscono sotto un altro nome, e il taglio di mezzo resta
*mascherato*.

Il rimedio è non fissare i voti a uno e zero, ma sceglierli, e il nome dice
proprio questo: *optimal scoring*, i voti scelti al meglio. Per ogni taglio si
cerca il numero che una ricetta sola indovina meglio, e i numeri che escono
mettono l’euro a un capo, i due euro all’altro e i cinquanta centesimi in mezzo,
dove stanno davvero. Con i voti scelti così la regressione ritrova esattamente
la ricetta di Fisher, e il taglio di mezzo non si perde più.

Da lì si può allargare in due direzioni. La ricetta può smettere di essere una
somma e diventare una curva qualunque, e allora il confine fra i tagli si piega
dove serve: non solo nei modi della QDA, che ammette soltanto le curve che
nascono da due forme diverse, ma in tutti quelli che la curva scelta permette. È
la FDA. Oppure le misure sono centinaia e in fila, come il suono della moneta
che cade sul tavolo registrato a duecentocinquantasei altezze diverse. Una
ricetta libera darebbe a ogni altezza un peso tutto suo, alto e basso a caso, e
imparerebbe il rumore di quelle poche monete. Si chiede allora che due altezze
vicine abbiano pesi simili, con una manopola che dice quanto chiederlo, e la
ricetta viene liscia: è la PDA.

`````

`````{tab} Superiore

Sia $\mathbf{Y} \in \{0,1\}^{N \times K}$ la matrice indicatrice delle classi.
La regressione lineare di $\mathbf{Y}$ sulla matrice dei dati $\mathbf{X}$, con
intercetta, seguita dalla regola $\hat{y} = \arg\max_k \hat{Y}_k(\mathbf{x})$, è
un classificatore lineare, e per $K = 2$ la sua direzione è quella di Fisher.
Per $K \ge 3$ no: le $\hat{Y}_k$ sommano a uno in ogni punto, e con i centroidi
quasi allineati la funzione della classe centrale resta quasi costante e viene
superata dalle due esterne. È il *masking*, tanto più probabile quanto più $K$ è
grande rispetto alla dimensione $d$ {cite}`hastie2009elements`.

L’*optimal scoring* sostituisce le indicatrici con punteggi
$\theta_\ell : \{1, \dots, K\} \to \mathbb{R}$, scelti insieme ai coefficienti
per minimizzare

$$
\mathrm{ASR} = \frac{1}{N} \sum_{\ell=1}^{L} \sum_{i=1}^{N}
\big(\theta_\ell(y_i) - \mathbf{x}_i^{\!\top}\boldsymbol{\beta}_\ell\big)^2,
\qquad L \le K - 1,
$$

con i punteggi a media nulla, varianza unitaria e ortogonali fra loro sui dati.
I $\boldsymbol{\beta}_\ell$ coincidono, a meno di una costante, con le
direzioni discriminanti di Fisher, e la LDA si ottiene assegnando la classe del
centroide più vicino nello spazio delle
$\hat{\eta}_\ell(\mathbf{x}) = \mathbf{x}^{\!\top}\boldsymbol{\beta}_\ell$, con
pesi $w_\ell = 1/\big(r_\ell^2(1 - r_\ell^2)\big)$, dove $r_\ell^2$ è il residuo
quadratico medio del punteggio $\ell$ {cite}`hastie1994flexible`. Il calcolo è
una regressione multipla di $\mathbf{Y}$ seguita da un problema agli autovalori
di dimensione $K$,
$\mathbf{Y}^{\!\top}\hat{\mathbf{Y}}\boldsymbol{\theta} =
\lambda\,\mathbf{Y}^{\!\top}\mathbf{Y}\boldsymbol{\theta}$, da cui si scarta il
punteggio costante ($\lambda = 1$); per gli altri $r_\ell^2 = 1 - \lambda_\ell$.

La FDA sostituisce $\mathbf{x}^{\!\top}\boldsymbol{\beta}_\ell$ con una
regressione non parametrica $\eta_\ell(\mathbf{x})$ (spline additive, MARS,
nuclei) e minimizza $\mathrm{ASR} + \gamma \sum_\ell \mathcal{R}(\eta_\ell)$,
con $\mathcal{R}$ il regolarizzatore di quella famiglia. Con un polinomio di
secondo grado i confini sono quadriche, le stesse che darebbe una LDA sulle
feature aumentate dei quadrati e dei prodotti incrociati. Quando la regressione
è lineare su un’espansione $h(\mathbf{x})$ con penalità
$\gamma\,\boldsymbol{\beta}^{\!\top}\boldsymbol{\Omega}\,\boldsymbol{\beta}$, la
FDA diventa la PDA {cite}`hastie1995penalized`, cioè una LDA nello spazio
espanso con la covarianza interna sostituita da
$\mathbf{S}_W + \gamma\boldsymbol{\Omega}$: le direzioni massimizzano
$\mathbf{a}^{\!\top}\mathbf{S}_B\,\mathbf{a}$ sotto
$\mathbf{a}^{\!\top}(\mathbf{S}_W + \gamma\boldsymbol{\Omega})\mathbf{a} = 1$, e
la distanza è quella di Mahalanobis nella stessa metrica. Serve anche senza
espansione, quando i predittori sono già troppi e correlati (i 256 valori di un
log-periodogramma, i pixel di una cifra scritta a mano): con
$\boldsymbol{\Omega}$ che penalizza le differenze fra coefficienti adiacenti, la
metrica pesa meno le combinazioni ruvide, e la direzione discriminante esce
liscia. È parente dello *shrinkage* di scikit-learn, che tira la covarianza
verso un multiplo dell’identità invece che verso la levigatezza.

`````

Il blocco mette alla prova le tre regole sulle monete: un euro, cinquanta
centesimi e due euro, trecento per taglio, con peso e diametro veri e uno
scarto di un decimo su tutte e due le misure. L’optimal scoring è scritto a
mano, perché scikit-learn ha la LDA e la QDA ma non la FDA né la PDA.

```python
import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

rng = np.random.default_rng(0)
# peso in grammi e diametro in millimetri: un euro, 50 centesimi, due euro
centri = np.array([[7.5, 23.25], [7.8, 24.25], [8.5, 25.75]])
y = np.repeat([0, 1, 2], 300)
X = centri[y] + rng.normal(0, 0.1, (900, 2))       # un decimo di scarto

A = np.column_stack([np.ones(len(y)), X])          # con l'intercetta
Y = np.eye(3)[y]                                   # uno al proprio taglio
Y_hat = A @ np.linalg.lstsq(A, Y, rcond=None)[0]   # una regressione per taglio
voto = Y_hat.argmax(1)

# optimal scoring: i punteggi dei tagli che la regressione indovina meglio
N = len(y)
lam, Theta = eigh(Y.T @ Y_hat / N, Y.T @ Y / N)    # autovalori crescenti
# via il punteggio costante (lambda = 1); il segno è arbitrario, un euro in basso
lam, Theta = lam[::-1][1:], Theta[:, ::-1][:, 1:]
Theta *= -np.sign(Theta[0])
eta = Y_hat @ Theta                                # le regressioni sui punteggi
w = 1 / ((1 - lam) * lam)                          # con r^2 = 1 - lambda
centroidi = np.array([eta[y == k].mean(0) for k in range(3)])
distanze = (((eta[:, None, :] - centroidi) ** 2) * w).sum(-1)
punteggio = distanze.argmin(1)
lda = LinearDiscriminantAnalysis().fit(X, y).predict(X)

for nome, pred in [("regressione sulle indicatrici", voto),
                   ("optimal scoring", punteggio), ("LDA", lda)]:
    print(f"{nome:30} sbaglia {np.sum(pred != y):3d} monete su {N},"
          f" di cui {np.sum((pred != y) & (y == 1)):3d} da cinquanta centesimi")
solo_primo = (((eta[:, None, :1] - centroidi[:, :1]) ** 2) * w[:1]).sum(-1).argmin(1)
print(f"con il solo primo punteggio sbaglia {np.sum(solo_primo != y)} monete")
print("primo punteggio (un euro, 50 cent, due euro):", np.round(Theta[:, 0], 2))
print(f"optimal scoring e LDA concordano su {np.sum(punteggio == lda)}"
      f" monete su {N}")
```

```text
regressione sulle indicatrici  sbaglia 117 monete su 900, di cui  97 da cinquanta centesimi
optimal scoring                sbaglia   0 monete su 900, di cui   0 da cinquanta centesimi
LDA                            sbaglia   0 monete su 900, di cui   0 da cinquanta centesimi
con il solo primo punteggio sbaglia 0 monete
primo punteggio (un euro, 50 cent, due euro): [-1.12 -0.19  1.31]
optimal scoring e LDA concordano su 900 monete su 900
```

La regressione sulle indicatrici sbaglia 117 monete su 900, e 97 sono da
cinquanta centesimi: il taglio di mezzo ne perde quasi un terzo. L’optimal
scoring e la LDA non ne sbagliano nessuna e concordano su tutte e 900, perché
sono la stessa regola scritta in due modi. Il primo punteggio mette l’euro a
$-1{,}12$, i due euro a $1{,}31$ e i cinquanta centesimi in mezzo, a $-0{,}19$:
più vicini all’euro, perché nel diametro gli sono più vicini, un millimetro
contro uno e mezzo. E quel primo punteggio da solo basta già a non sbagliarne
nessuna: nelle monete da euro peso e diametro crescono quasi in proporzione, i
tre centri stanno quasi su una retta, ed è proprio il caso in cui la
regressione sulle indicatrici maschera di più e un numero solo, scelto bene,
separa tutto.

## Naive Bayes: l'ipotesi sfacciata che funziona

Il terzo membro della famiglia si ottiene da una semplificazione che, detta ad
alta voce, sembra insostenibile.

`````{tab} Elementare

Descrivere una classe con centro e forma costa: la forma dice anche come le
caratteristiche vanno insieme (se le monete più pesanti sono anche più larghe), e
con venti caratteristiche le coppie da guardare sono $20 \times 19 / 2$, cioè
centonovanta. Tante da imparare.

Il **naive Bayes** taglia corto e dichiara che quelle relazioni non esistono:
dentro una classe, ogni caratteristica va per conto suo. Peso e diametro non si
sanno l'uno dell'altro. Così di ogni classe basta imparare, una caratteristica
per volta, una media e una dispersione. Poi la moneta nuova si giudica su ogni
caratteristica separatamente, e i giudizi che ne escono si moltiplicano fra loro.

L'ipotesi è quasi sempre falsa, e non un po’: in un'email le parole «offerta» e
«gratis» si tirano dietro a vicenda, in una moneta peso e diametro pure. Il nome
lo ammette: *naive* vuol dire ingenuo.

Il fatto strano è che funziona lo stesso, e la ragione è che al classificatore
non serve avere ragione sulle probabilità, gli serve mettere in classifica le
classi nel giusto ordine. Può sbagliare di brutto sul «quanto» (dirà $0{,}999$
dove il vero è $0{,}7$, perché contando due volte prove che erano la stessa prova
si convince troppo) e azzeccare comunque il «quale». Domingos e Pazzani hanno
studiato proprio questo nel 1997 {cite}`domingos1997optimality`, mostrando che
l'insieme dei casi in cui il naive Bayes è ottimo è molto più grande di quello in
cui la sua ipotesi è vera.

Il corollario pratico tocca chi usa questi modelli. Le probabilità del naive
Bayes non si usano come probabilità. Come classifica sono buone, come numeri no.
Se servono probabilità di cui fidarsi (una soglia da tarare, un costo da
calcolare) vanno ricalibrate.

`````

`````{tab} Superiore

L'ipotesi è l'indipendenza condizionata delle feature data la classe:

$$
p(\mathbf{x} \mid y = k) = \prod_{j=1}^{d} p(x_j \mid y = k).
$$

Nel caso gaussiano equivale a imporre $\boldsymbol{\Sigma}_k$ diagonale, e i
parametri di covarianza crollano da $K\,d(d+1)/2$ a $Kd$: per $d = 20$ e
$K = 2$, da $420$ a $40$. Geometricamente, le ellissi di livello hanno gli assi paralleli
agli assi coordinati (nessuna rotazione), che è ciò che mostra il terzo pannello
di {numref}`fig-lda-qda`.

Il risultato di Domingos e Pazzani {cite}`domingos1997optimality` è che la
regione di ottimalità del naive Bayes sotto perdita $0$–$1$ è strettamente più
ampia di quella in cui vale l'indipendenza condizionata: l'errore
sull’*ordinamento* delle posteriori è un evento più raro dell'errore sulle
posteriori stesse, perché la funzione $\arg\max$ è invariante a un'ampia classe
di distorsioni monotone. Le stime restano però mal calibrate, tipicamente
sovrasicure, perché feature correlate contribuiscono evidenza ripetuta al
prodotto: chi ha bisogno delle probabilità e non solo dell'etichetta ricalibri
con lo scaling di Platt o con l'isotonica, che la
{doc}`sezione sulle metriche </MachineLearning/metriche>` mette alla prova.

Nel caso discreto (conteggi di parole) il modello prende il nome di naive Bayes
**multinomiale**, e con lo smoothing di Laplace è la base storica della
classificazione dei testi. Il capitolo sul *Natural Language Processing* lo
tratta in quella veste, con il conto dello smoothing: qui interessa come membro
della famiglia generativa, non come classificatore di documenti.

`````

## Quando il generativo vince: pochi dati

Resta la domanda che decide se questa famiglia serve ancora, e ha una risposta
misurabile. Andrew Ng e Michael Jordan la formulano nel 2001
{cite}`ng2001discriminative`, e il confronto che scelgono è il più pulito
possibile: naive Bayes gaussiano contro regressione logistica. Nel loro naive
Bayes ogni caratteristica ha una media per classe e una varianza sola, comune
alle due classi, e con questa scelta la posteriore è esattamente una sigmoide
di una funzione affine: i due modelli arrivano alla stessa formula per
decidere, e differiscono solo su come ne ricavano i numeri dai dati.
`GaussianNB` di scikit-learn stima invece una varianza per classe, e il suo
confine è una quadrica; per rifare il confronto di Ng e Jordan la varianza va
messa in comune.

Il risultato ha la forma di una gara con due tempi. Il generativo parte meglio:
con pochi esempi è già vicino al meglio che sa fare. Il discriminativo parte
peggio, ma il meglio che sa fare è più alto, e con abbastanza esempi ci
arriva. Su pochi dati vince il primo; su tanti, il secondo.

Rifacendo quel confronto con una colonna in più, la logistica come la si usa
oggi, si vede quanto ne resta.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from scipy.stats import norm

D = 40
rng = np.random.default_rng(0)
MU = rng.choice([-1, 1], D) * 0.35     # le due classi differiscono in ogni feature

def dati(n, r):
    """Due classi gaussiane a feature indipendenti: l'ipotesi naive qui e' VERA."""
    y = r.integers(0, 2, n)
    return r.normal(0, 1, (n, D)) + np.outer(y, MU), y

def nb_condiviso(X, y):
    """Il naive Bayes di Ng e Jordan: una varianza per colonna, comune alle classi."""
    nb = GaussianNB().fit(X, y)
    nb.var_[:] = (X - nb.theta_[y]).var(axis=0)
    return nb

X_test, y_test = dati(20_000, np.random.default_rng(999))

print(f"{'n':>6} {'naive Bayes':>12} {'logistica (default)':>21} {'logistica nuda':>16}")
for n in (20, 40, 80, 200, 600, 2000):
    a, b, c = [], [], []
    for s in range(15):
        r = np.random.default_rng(100 + s)
        X, y = dati(n, r)
        if len(np.unique(y)) < 2:
            continue
        a.append(nb_condiviso(X, y).score(X_test, y_test))
        b.append(LogisticRegression(max_iter=5000).fit(X, y).score(X_test, y_test))
        c.append(LogisticRegression(C=1e6, max_iter=5000).fit(X, y).score(X_test, y_test))
    print(f"{n:6d} {np.mean(a):12.3f} {np.mean(b):21.3f} {np.mean(c):16.3f}")

# il tetto di tutti: la regola di Bayes con le due gaussiane vere, che per
# classi equiprobabili a covarianza identita' indovina Phi(|MU| / 2)
print(f"massimo teorico: {norm.cdf(np.linalg.norm(MU) / 2):.3f}")
```

```text
     n  naive Bayes   logistica (default)   logistica nuda
    20        0.714                 0.708            0.663
    40        0.762                 0.746            0.693
    80        0.811                 0.771            0.722
   200        0.843                 0.817            0.796
   600        0.857                 0.847            0.846
  2000        0.862                 0.859            0.859
massimo teorico: 0.866
```

La terza colonna è la logistica senza regolarizzazione, che è quella del
confronto originale, e su di lei il fenomeno si vede intero: a $n = 80$ il naive
Bayes sta a $0{,}811$ e lei a $0{,}722$, quasi nove punti sotto; a $n = 200$
sono ancora quasi cinque; da $n = 600$ in poi si avvicinano fino quasi a
toccarsi. Con
quaranta caratteristiche e ottanta esempi la logistica ha due esempi per
parametro, e con così poco non impara; il naive Bayes ne stima anche di più
(centoventi: una media per classe e per colonna, e una varianza per colonna),
ma li stima
uno alla volta, ciascuno con tutti i dati della sua classe, e se la cava.

La seconda colonna è la logistica come la si usa oggi, cioè col
`penalty="l2"` che scikit-learn applica per default, ed è il motivo per cui
questo esperimento conviene rifarlo invece di citarlo. Quel freno accorcia il
divario senza chiuderlo: a $n = 80$ il naive Bayes resta quattro punti sopra
($0{,}811$ contro $0{,}771$), a $n = 200$ due e mezzo, a $n = 600$ uno, e a
$n = 2000$ ancora tre millesimi. Il fenomeno del 2001 sopravvive ai
default di oggi; la regolarizzazione ne cambia la misura, e il verso resta
quello.

Una precisazione su questa tabella, che ne dichiara il limite. I dati qui sono
stati fabbricati a feature indipendenti, cioè nel mondo in cui l'ipotesi del
naive Bayes è vera. Questo rende visibile il primo tempo della gara, la
partenza rapida del generativo, e rende invisibile il secondo: se il modello
del naive Bayes è quello giusto, i due metodi hanno lo stesso tetto, e quel
tetto è il massimo teorico del problema ($0{,}866$ di accuratezza, stampato in
fondo all'uscita). Infatti la riga di $n = 2000$ li dà a $0{,}862$ e $0{,}859$,
tutti e due a pochi millesimi da quel tetto: il generativo ci arriva prima, il
discriminativo lo insegue, e il suo asintoto più alto si vede solo quando
l'ipotesi naive è falsa. Con feature
correlate l'asintoto del naive Bayes scende sotto il massimo teorico, e con
abbastanza esempi la logistica lo supera: al vantaggio di stimare poco si somma
il costo di un'ipotesi falsa. Il vantaggio dei pochi dati è reale; non è un
salvacondotto.

## In pratica

```python
from scipy.special import logsumexp

X, y = genera(2000, True, 0)
lda = LinearDiscriminantAnalysis(store_covariance=True).fit(X, y)
print("accuratezza LDA:", round(lda.score(*genera(20_000, True, 999)), 3))

# la LDA ha imparato due gaussiane: da quelle si ricava anche p(x), non solo la
# classe. E' l'unica cosa che un discriminativo non puo' dare.
inversa = np.linalg.inv(lda.covariance_)
_, logdet = np.linalg.slogdet(lda.covariance_)

def log_densita(P):
    """log p(x): quanto e' verosimile un punto, per il modello gia' addestrato."""
    per_classe = [-0.5*np.einsum("ij,jk,ik->i", P - m, inversa, P - m)
                  - 0.5*logdet - np.log(2*np.pi) + np.log(q)
                  for m, q in zip(lda.means_, lda.priors_)]
    return logsumexp(per_classe, axis=0)

fuori = np.array([[14.0, -11.0]])          # un punto che non c'entra niente
print(f"log p(x) di un punto in mezzo ai dati: {log_densita(X[:1])[0]:9.2f}")
print(f"log p(x) di un punto lontanissimo    : {log_densita(fuori)[0]:9.2f}")
print(f"e sullo stesso punto lontano si dichiara sicuro al "
      f"{lda.predict_proba(fuori).max():.2%}")
```

```text
accuratezza LDA: 0.731
log p(x) di un punto in mezzo ai dati:     -2.36
log p(x) di un punto lontanissimo    :   -711.59
e sullo stesso punto lontano si dichiara sicuro al 99.99%
```

Le ultime tre righe sono il gettone fra le monete, misurato. Il punto
$(14, -11)$ non ha niente a che vedere con questi dati, e la densità lo dice
senza esitazioni. I due numeri stampati sono logaritmi, e la differenza fra loro
è di settecentonove: non «settecento volte meno probabile», ma un rapporto di
$10^{308}$, cioè un $1$ seguito da trecentotto zeri. Quel punto, per il modello,
semplicemente non capita. La classificazione dello stesso punto, invece,
esce al $99{,}99\%$ di sicurezza, perché una volta scelto da che parte della
retta si trova non c'è altro da dire.

I due numeri vengono dallo stesso modello, addestrato una volta sola, e sono
la ragione per cui conviene avere in casa un generativo: la sicurezza di un
classificatore è sempre relativa alle classi che conosce, e da sola non
distingue «è certamente una gazza» da «non ho idea di cosa sia, ma se devo
scegliere dico gazza». La densità quella distinzione la fa.

Quando conviene prenderli in considerazione, in concreto:

- come riferimento di partenza: la LDA non ha iperparametri, si addestra in
  un istante su qualunque tabella, e dà un numero contro cui misurare tutto il
  resto. Se il modello elaborato non la batte, il problema è nei dati o
  nell'impostazione, non nel modello;
- con pochi esempi e molte colonne, che è la situazione tipica dei dati
  clinici e sperimentali: è la riga $n = 80$ della tabella;
- quando serve una densità, cioè quando bisogna accorgersi degli esempi che
  non somigliano a niente di visto: è quello che la mistura gaussiana, più
  avanti nel capitolo, farà per segnalare i punti improbabili;
- per schiacciare i dati in poche dimensioni senza perdere le classi: la
  LDA li proietta su al più $K-1$ direzioni, scelte apposta perché su quelle le
  classi si distinguano. È la cugina supervisionata dell'analisi delle
  componenti principali, che la sezione sul clustering costruirà: quella cerca
  le direzioni in cui i dati variano di più, questa quelle in cui le
  classi si distinguono di più, e le due possono benissimo non coincidere.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Ci sono due modi di classificare. Imparare dove passa il confine (la
  logistica, gli alberi, le SVM) e imparare com'è fatta ogni classe, per poi
  ricavarne il confine. Il secondo è quello dell'ornitologo, e si chiama
  generativo.
- Chi sa com'è fatta ogni classe sa anche dire dove i casi sono fitti e dove
  sono radi, la stima di densità, e quindi riconoscere quello che non somiglia
  a nessuna: un gettone fra le monete. Chi ha imparato solo il confine no. Con
  migliaia di misure, come i puntini di una foto, quel conto può ingannarsi.
- LDA: una sola forma condivisa dalle due classi, e il confine viene una
  retta. QDA: una forma per classe, e il confine si incurva. La ricetta di
  Fisher, un numero solo mescolando le misure, dà l'inclinazione della retta;
  dove tagliare dipende anche da quanto è comune ciascuna classe.
- Non conviene sempre la più flessibile: con classi che hanno davvero la stessa
  forma i tre metodi danno lo stesso numero ($0{,}728$, $0{,}726$, $0{,}728$, e
  ballano di $\pm 0{,}003$), mentre con forme diverse la QDA sta quattro punti
  sopra.
- Il naive Bayes dichiara che dentro una classe le caratteristiche non si
  parlano fra loro. È quasi sempre falso, e funziona lo stesso, perché per
  scegliere la classe basta l’ordine, non il valore esatto. Le sue
  probabilità però non vanno usate come probabilità: sono troppo sicure di sé.
- Il generativo dà il meglio con pochi dati: a ottanta esempi e quaranta
  colonne il naive Bayes sta quasi nove punti sopra la logistica non
  regolarizzata, e quattro sopra quella con i freni di oggi. A seicento esempi
  il vantaggio è quasi finito.
- Con tre tagli in fila, dare a ogni moneta voti di uno e zero e indovinarli
  con una ricetta dritta fa perdere il taglio di mezzo; scegliere i voti al
  meglio ritrova la ricetta di Fisher. Da lì la FDA piega il confine come
  serve, e la PDA tiene lisce le ricette quando le misure sono centinaia e in
  fila.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Discriminativo: si modella $p(y \mid \mathbf{x})$. Generativo: si
  modella $p(\mathbf{x} \mid y)\,p(y)$ e si applica Bayes. Il secondo dà in più
  una densità (anomalie, con un punteggio che in alta dimensione inganna), la
  stima classe per classe e un migliore comportamento a pochi dati.
- La regola di decisione confronta
  $\delta_k(\mathbf{x}) = -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_k)^\top
  \boldsymbol{\Sigma}_k^{-1}(\mathbf{x}-\boldsymbol{\mu}_k)
  -\frac{1}{2}\log\lvert\boldsymbol{\Sigma}_k\rvert + \log\pi_k$: distanza di
  Mahalanobis, più il termine di volume e la priore.
- Con $\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$ il termine quadratico si
  cancella nella differenza e $\delta_k$ diventa affine: è la LDA, e la
  linearità è una conseguenza, non un'ipotesi. Verificato numericamente: scarto
  dalla più vicina funzione affine $8{,}9 \cdot 10^{-15}$ per la LDA contro
  $7{,}2$ per la QDA.
- Parametri (medie più covarianze): LDA $Kd + d(d+1)/2$, QDA
  $Kd + K\,d(d+1)/2$, naive Bayes gaussiano $2Kd$ (una media e una varianza per
  classe e per feature, niente termini incrociati). È il compromesso
  bias-varianza sul modello di $p(\mathbf{x}\mid y)$; la *regularized
  discriminant analysis* e lo `shrinkage` interpolano.
- Naive Bayes: $p(\mathbf{x}\mid y) = \prod_j p(x_j \mid y)$. La regione di
  ottimalità sotto perdita $0$–$1$ è più ampia di quella in cui l'ipotesi vale
  {cite}`domingos1997optimality`, perché conta l’$\arg\max$ e non il valore; le
  posteriori restano sovrasicure e vanno ricalibrate.
- Ng e Jordan {cite}`ng2001discriminative`, con un naive Bayes a varianza
  comune fra le classi (la coppia esatta della logistica): il generativo si
  avvicina al proprio asintoto con un numero di esempi che cresce come
  $O(\log d)$ nel numero di feature, il discriminativo ne chiede $O(d)$; in
  cambio l'asintoto del discriminativo ha errore più basso quando l'ipotesi del
  generativo è falsa. A feature indipendenti i due asintoti coincidono
  (accuratezza di Bayes $0{,}866$) e si vede solo la prima metà. L’$\ell_2$ di
  default riduce il divario senza annullarlo: $0{,}811$ contro $0{,}771$ a
  $n = 80$.
- Il quoziente di Fisher dà la direzione
  $\mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)$, non la soglia,
  che dipende dalle priori. La LDA è anche una riduzione di dimensionalità
  supervisionata su al più $\min(K-1, d)$ direzioni, senza perdita per la
  regola LDA, ed è la mistura gaussiana della sezione sul clustering con le
  variabili latenti osservate: resta il solo passo M, eseguito una volta.
- Con $K \ge 3$ la regressione sulle indicatrici maschera le classi centrali;
  l’optimal scoring {cite}`hastie1994flexible` sceglie i punteggi delle
  classi e ritrova la LDA con una regressione multipla e un autoproblema di
  dimensione $K$. Sostituendo la regressione con una non parametrica si ha la
  FDA; con una penalità $\gamma\boldsymbol{\Omega}$ sui coefficienti la PDA
  {cite}`hastie1995penalized`, cioè la LDA con
  $\mathbf{S}_W + \gamma\boldsymbol{\Omega}$.
```

`````

Il giro dei classificatori classici si chiude tornando al punto di partenza da
dietro: la regressione logistica con cui tutto era cominciato e la LDA di
Fisher tracciano lo stesso tipo di confine e non sono lo stesso metodo, perché
una guarda il confine e l'altra guarda le classi. La prossima sezione abbandona
del tutto le etichette e chiede ai dati di raggrupparsi da soli, che è l'unica
domanda a cui nessuno dei modelli visti finora sa rispondere.
