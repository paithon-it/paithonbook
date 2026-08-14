# Rumore e ritorno: come funziona la diffusione

Facciamo un gioco. Ti mostro una fotografia su cui ho steso un velo di
disturbo (un pulviscolo di puntini casuali) e ti faccio una domanda sola:
*quale disturbo ho aggiunto?* Non ti chiedo di ridipingere la foto, né di
dirmi cosa rappresenta: solo di indicare il pulviscolo. Sembra una richiesta
modesta, ma contiene tutto: se sai rispondere, sai anche restaurare; basta
togliere ciò che hai indicato. Adesso ripetiamo il gioco per mille livelli di
rovina, dalla grana appena percettibile al pulviscolo che ha inghiottito tutto. Chi
impara a rispondere a *ogni* livello possiede una scala che scende dal rumore
puro fino a un'immagine; e siccome ogni pulviscolo appena estratto è diverso
dall'altro, in fondo alla scala troverà ogni volta un'immagine diversa. Nuova.

Il capitolo si è aperto con la promessa di smontare questo giocattolo pezzo
per pezzo; questa sezione la mantiene: l'andata, il ritorno, il viaggio dal
rumore all'immagine, uno sguardo sotto il cofano, la scorciatoia di DDIM, la
rete che fa il lavoro e infine tutto il meccanismo in miniatura, funzionante,
in PyTorch. La
{numref}`fig-diffusione-processo` è la mappa del viaggio.

```{figure} ../figures/diffusione-processo.svg
:name: fig-diffusione-processo
:alt: Due file di quattro riquadri, etichettate x0, x1, xt, xT. In alto il processo in avanti, in teal, un paesaggio stilizzato che di riquadro in riquadro si copre di puntini fino a diventare rumore puro, con frecce verso destra. In basso gli stessi quattro riquadri in terracotta e le frecce verso sinistra, ognuna con sopra il simbolo della rete che stima il disturbo, fino a recuperare il paesaggio.
:width: 100%

I due processi della diffusione: l'andata (in alto) rovina l'immagine di
partenza fino al rumore puro, seguendo una ricetta fissa; il ritorno (in
basso) risale la catena un gradino alla volta, e a ogni gradino è una rete a
dire dove sta il disturbo.
```

## L'andata: rovinare con metodo

Il processo in avanti non si impara: è una ricetta fissa, la esegue un
generatore di numeri casuali. Ma non è una distruzione qualsiasi: è una
distruzione *dosata*, e i dosaggi sono scelti con cura, perché è su di essi
che il ritorno farà affidamento. L'ingrediente è sempre lo stesso: rumore
estratto dalla campana di Gauss, la distribuzione normale incontrata nei
richiami di statistica.

`````{tab} Elementare

Un'immagine in bianco e nero è una griglia di numeri: 0 è nero, 1 è bianco.
Seguiamo un solo pixel, un grigio chiaro che vale 0,8. A ogni passo la
ricetta prevede due gesti:

1. **attenua** il valore, moltiplicandolo per un numero appena sotto 1, nel
   passo che prendiamo a esempio, 0,99: il pixel scende a
   $0{,}8 \times 0{,}99 = 0{,}792$;
2. **aggiungi** un piccolo numero estratto a caso dalla campana di Gauss,
   riscalato di un fattore piccolo, qui 0,14. Se l'estrazione dà $-0{,}7$, il
   contributo è $-0{,}7 \times 0{,}14 \approx -0{,}10$, e il pixel finisce a
   circa $0{,}69$. Con un'estrazione diversa, poniamo $+0{,}3$, sarebbe finito
   a circa $0{,}83$.

Perché anche l'attenuazione, e non solo il rumore? Pensa a una tazza di caffè
sempre piena: a ogni giro togli un cucchiaino di caffè e ne versi uno di
latte. Il livello nella tazza non cambia mai (i numeri non esplodono) ma il
contenuto vira, giro dopo giro, dal caffè al latte. Qui il caffè è l'immagine
e il latte è il rumore: dopo mille giri, nella tazza c'è solo latte. I conti
lo confermano: moltiplicare per 0,99 mille volte lascia del valore iniziale
appena $0{,}99^{1000} \approx 0{,}00004$; del nostro 0,8 non resta traccia, il
pixel è ormai un'estrazione pura dalla campana. Ed è successo a *tutti* i
pixel insieme: la foto è diventata pulviscolo che non ricorda nulla di
ciò che era.

Ed è il livello costante della tazza a spiegare da dove escono i due numeri di
prima. Non li ha scelti nessuno separatamente: il quadrato dell'uno più il
quadrato dell'altro deve fare 1
($0{,}99^2 + 0{,}14^2 = 0{,}9801 + 0{,}0196 \approx 1$), altrimenti la tazza
si svuoterebbe o traboccherebbe. Chi decide quanto attenuare ha già deciso,
senza poter fare altrimenti, quanto disturbo aggiungere.

`````

`````{tab} Superiore

Il processo diretto è la catena di Markov già dichiarata all'apertura del
capitolo:

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\ \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\
\beta_t \mathbf{I}\right),
$$

dove $\mathbf{x}_t$ è il dato al passo $t$, $\beta_t \in (0,1)$ è la varianza del
rumore iniettato al passo $t$ e $\mathbf{I}$ è la matrice identità. La successione
$\beta_1, \dots, \beta_T$ è lo **schedule**: in DDPM è lineare, da
$\beta_1 = 10^{-4}$ a $\beta_T = 0{,}02$ su $T = 1000$ passi (veli
sottilissimi all'inizio, più decisi verso la fine). Nessun parametro appreso:
$q$ è fissata una volta per tutte.

Definendo $\alpha_t = 1-\beta_t$ e $\bar{\alpha}_t = \prod_{s=1}^{t}
\alpha_s$, la catena ammette una forma chiusa che salta direttamente da
$\mathbf{x}_0$ a qualunque $\mathbf{x}_t$:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\ \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\
(1-\bar{\alpha}_t)\, \mathbf{I}\right)
\quad\Longleftrightarrow\quad
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},
$$

con $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$; qui $\bar{\alpha}_t$ è la frazione di
segnale originale sopravvissuta al passo $t$ e $1-\bar{\alpha}_t$ la varianza
del rumore accumulato. La forma chiusa esiste perché la somma di gaussiane
indipendenti è ancora gaussiana (richiami di statistica): componendo un passo
dopo l'altro, i coefficienti del segnale si moltiplicano e la varianza si
accumula secondo la ricorrenza $v_t = \alpha_t\, v_{t-1} + \beta_t$ (ogni
iniezione viene attenuata da tutti i passi successivi, non sommata tale e
quale), la cui soluzione è appunto $1-\bar{\alpha}_t$. E i dosaggi
sono calibrati perché la scala resti stabile: se $\mathbf{x}_0$ ha varianza unitaria,
$\mathrm{Var}(\mathbf{x}_t) = \bar{\alpha}_t + (1-\bar{\alpha}_t) = 1$ a ogni passo (il
processo è *variance-preserving*). Con lo schedule di DDPM,
$\bar{\alpha}_T \approx 4 \cdot 10^{-5}$: al passo finale
$q(\mathbf{x}_T \mid \mathbf{x}_0) \approx \mathcal{N}(0, \mathbf{I})$ per qualunque $\mathbf{x}_0$, cioè $\mathbf{x}_T$ è
rumore gaussiano puro, indipendente dal dato di partenza.

`````

In quella ricetta c'è una scorciatoia, e ha una conseguenza che vale la pena
vedere prima di proseguire: per portare un'immagine a un livello di rovina
qualsiasi non serve percorrere la catena un anello per volta, ci si arriva in
un colpo solo. E allora l'andata si può guardare tutta in una volta, come una
manopola con mille tacche: a ogni tacca corrisponde una dose di disegno e una dose di
disturbo, e le due dosi sono decise in partenza, non da quello che è successo
per strada. La {numref}`fig-diffusione-avanti` ne mostra sei.

```{figure} ../figures/diffusione-avanti.svg
:name: fig-diffusione-avanti
:alt: "Sei riquadri affiancati mostrano la stessa figura geometrica alle tacche t = 0, 100, 250, 450, 700 e 1000 del processo di rovina, e si scoprono uno alla volta da sinistra a destra: nel primo il disegno è nitido, nell'ultimo non si distingue più niente. Sotto ogni riquadro due numeri, quanto resta del disegno e quanto disturbo c'è sopra: 1,00 e 0,00; 0,95 e 0,32; 0,72 e 0,69; 0,36 e 0,93; 0,08 e 1,00; 0,01 e 1,00. Le due file di numeri si scambiano il posto fra la seconda tacca e la terza."
:width: 100%

Il verso facile, in sei tacche della manopola. Sotto ogni riquadro, le due
dosi: quanto disegno è rimasto e quanto disturbo c'è sopra. Le due si scambiano
il posto molto prima di metà catena, fra la tacca 100 e la 250 e non alla 500,
ed è la cosa più utile della figura: già alla tacca 450 il disegno è sceso a un
terzo mentre il disturbo è al 93 per cento, e da lì in poi la manopola aggiunge
poco perché non c'è quasi più niente da coprire.
```

Il ritorno, che è la parte difficile, la figura non lo mostra, e non è una
dimenticanza: percorrere quei sei riquadri da destra a sinistra è esattamente
il compito che nessuna formula sa svolgere, ed è il motivo per cui serve una
rete.

## Il ritorno: indovinare il disturbo

Ora entra in scena l'unica cosa che si impara: una rete neurale che riceve due
cose soltanto, l'immagine rovinata e il numero del passo a cui è stata
rovinata, e deve rispondere alla domanda del nostro gioco: *quale rumore è
stato aggiunto?* Non «com'era la foto pulita» (proprio il rumore). È una
scelta meno ovvia di quanto sembri, ed è uno dei motivi per cui DDPM
{cite}`ho2020denoising` funziona così bene.

Da qui in avanti questa rete si chiamerà $\boldsymbol{\epsilon}_\theta$, che è il nome che
le dà la letteratura e che vale la pena saper leggere: la lettera greca
epsilon è il rumore, il pedice theta ricorda che dietro c'è una rete con dei
pesi da imparare. Scritta con gli ingressi accanto, $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$
si legge «il rumore che la rete stima nell'immagine $\mathbf{x}_t$, sapendo che siamo
al passo $t$».

`````{tab} Elementare

L'addestramento è un mazzo di carte per il ripasso, con le soluzioni sul
retro. Si prepara una carta così: pesca una foto vera dall'archivio, pesca un
livello di rovina a caso (poniamo il passo 700 su 1000), genera il pulviscolo
di disturbo e mescola i tre ingredienti con la ricetta dell'andata. Sul fronte
della carta: la foto rovinata e il numero 700. Sul retro: il pulviscolo esatto
che è stato usato; lo conosciamo alla perfezione, perché l'abbiamo fabbricato
noi un istante fa. La rete guarda il fronte, propone la sua risposta, e il
voto è la distanza tra il disturbo indicato e quello vero: differenze piccole,
voto alto. Milioni di carte dopo, la rete ha imparato a rispondere a ogni
livello di rovina.

Ma perché chiedere il disturbo e non direttamente la foto pulita? Prova a
metterti nei panni della rete al passo 900, davanti a una schermata fatta
quasi tutta di rumore: «dimmi la foto originale» è una richiesta da veggente;
dovrebbe inventare di sana pianta dettagli che nel rumore non ci sono più. «Dimmi il
disturbo» è invece un compito dello stesso formato a ogni livello: il
pulviscolo ha sempre lo stesso aspetto statistico, media zero e la stessa
ampiezza tipica, al passo 10 come al passo 990. È come interrogare uno
studente sempre con domande dello stesso tipo, invece che con quesiti la cui
difficoltà cambia in modo selvaggio. Attenzione però a che cosa si equivale e
che cosa no: le due *risposte* portano la stessa informazione, perché chi
conosce il disturbo e la ricetta con cui è stato mescolato ricava la foto con
una sottrazione; le due *domande* no, ed è tutto il punto.

`````

`````{tab} Superiore

Il processo inverso è modellato da una gaussiana con media appresa:

$$
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\ \sigma_t^2 \mathbf{I}\right),
$$

dove $\theta$ sono i parametri della rete e $\sigma_t^2$ è una varianza
fissata, non appresa. DDPM ne prova **due**, $\sigma_t^2 = \beta_t$ e la
varianza del posteriore
$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$,
e le trova sperimentalmente equivalenti sui mille passi; qui useremo la prima,
ma teniamo a mente che sono due, perché a poche decine di passi smetteranno di
equivalersi (ci torniamo con DDIM). Si noti che $\tilde{\beta}_1 = 0$: la
seconda scelta spiega da sé perché all'ultimo passo non si aggiunga rumore
fresco. La scelta di modellare il ritorno
con una gaussiana è legittimata dal risultato di Feller citato in apertura di
capitolo: per passi
$\beta_t$ piccoli, il vero inverso $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ è
approssimativamente gaussiano. Il contributo chiave di Ho, Jain e Abbeel
{cite}`ho2020denoising` è la **riparametrizzazione** della media: invece di
far predire alla rete $\boldsymbol{\mu}_\theta$ direttamente (o la ricostruzione
$\hat{\mathbf{x}}_0$), la si scrive in funzione del rumore stimato,

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t -
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right),
$$

dove $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ è la stima del rumore $\boldsymbol{\epsilon}$ usato per
produrre $\mathbf{x}_t$ dalla forma chiusa. Con questa scelta l'addestramento si
riduce a un errore quadratico medio:

$$
\mathcal{L}_{\text{semplice}}(\theta) = \mathbb{E}_{\mathbf{x}_0,\, \boldsymbol{\epsilon},\, t}
\left[\, \big\lVert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\!\big(
\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\ t
\big) \big\rVert^2 \,\right],
$$

dove $\mathbf{x}_0$ è un dato del training set, $t$ è uniforme su $\{1, \dots, T\}$ e
$\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$: si campiona una tripla, si costruisce $\mathbf{x}_t$
in un colpo solo con la forma chiusa (senza percorrere la catena), e si
confrontano rumore vero e rumore predetto. Si noti che predire $\boldsymbol{\epsilon}$ e
predire $\boldsymbol{\mu}$ sono formulazioni legate da una relazione affine (dato $\mathbf{x}_t$,
l'una si ricava dall'altra) ma non equivalenti come problemi di regressione:
il bersaglio $\boldsymbol{\epsilon}$ ha distribuzione $\mathcal{N}(0, \mathbf{I})$ *a ogni* $t$,
quindi scala costante e ben condizionata.

L'ablazione di DDPM su questo punto va letta con attenzione, perché dice
qualcosa di più stretto del solito «predire il rumore è meglio». A parità di
obiettivo, cioè usando per entrambe il bound variazionale completo, le due
parametrizzazioni **si equivalgono**. Il salto di qualità arriva soltanto
dalla *coppia*: $\boldsymbol{\epsilon}$ insieme alla loss semplificata scritta qui sopra. E
la coppia speculare, $\boldsymbol{\mu}$ insieme alla loss semplificata, il paper la marca
come instabile in addestramento. La lettura corretta è quindi più forte, non
più debole: predire $\boldsymbol{\epsilon}$ non è tanto un bersaglio migliore in sé, quanto
la sola parametrizzazione con cui la loss semplificata addestri davvero.

`````

## Generare: il viaggio dal rumore all'immagine

Finito l'addestramento, il generatore è pronto. Non serve nessuna foto di
partenza: si estrae rumore puro e si percorre la scala all'indietro, un
gradino alla volta, interrogando la rete a ogni passo.

```{figure} ../figures/diffusione-denoising.gif
:name: fig-diffusione-denoising
:alt: "Animazione: un quadrato di rumore casuale in scala di grigi si trasforma progressivamente, attraverso sei stati etichettati t = 1000, 800, 600, 400, 200 e 0, nella cifra 3 disegnata in pixel art. Sotto il quadrato resta ferma la formula del passo inverso. Il rumore non cala in modo liscio: resta fitto fino a t = 600 e la cifra affiora solo verso t = 400."
:width: 70%

Il processo inverso su una cifra: a ogni passo la rete dice dove sta il
disturbo, se ne toglie una scheggia, si riscala quel che resta e se ne
rimette una manciata fresca. Nessuno ha disegnato il 3: è emerso da mille
rimescolamenti.
```

La {numref}`fig-diffusione-denoising` comprime in pochi passi ciò che nel DDPM
originale ne richiede mille. Il gesto vero, però, non è quello che il buon
senso si aspetta, e conviene guardarlo da vicino: **non è una ripulitura
progressiva**. A ogni passo si toglie molto meno di quanto si rimetta, e ciò
che fa emergere l'immagine è un'altra cosa. Si vede anche nella clip, dove il
disturbo non cala in modo liscio ma resta lì a lungo e se ne va tardi.

`````{tab} Elementare

La procedura è un rituale in tre mosse, ripetuto mille volte. Si parte da una
manciata di pulviscolo appena estratto, mai visto prima; poi, dal passo 1.000 al passo 1:

1. **correggi**: mostra alla rete la schermata e il numero del passo, fatti
   dire dov'è il disturbo, e cancellane una scheggia;
2. **alza il volume** di tutto quello che resta sullo schermo, di un soffio:
   un millesimo, un centesimo. Sale l'immagine che sta sotto e sale il rumore
   che le sta sopra, insieme;
3. **rimescola**: getta sopra una manciata di rumore nuovo. All'ultimo passo, e
   solo lì, questa terza mossa si salta.

Le proporzioni sono l'esatto contrario di quello che il buon senso si aspetta,
ed è la cosa più importante di tutta la sezione: **la manciata di rumore nuovo è
dalle sette alle dieci volte più grande della scheggia appena cancellata**, e
lo è per i primi novecento passi su mille. Non un pizzico: una manciata. Per
quasi tutto il viaggio lo schermo si sposta molto più per il rumore che gli si
è buttato sopra che per il disturbo che gli si è tolto. Solo nell'ultimo
decimo del percorso la manciata si rimpicciolisce, fino a pareggiare la
scheggia sull'ultimo gradino, dove poi sparisce del tutto.

Come fa allora a uscirne un'immagine, se a ogni giro si toglie poco e si
rimette molto? Per una differenza che non sta nella quantità ma nella
*direzione*. La scheggia che si cancella è **mirata**: la rete indica dove il
disturbo sta davvero, e la cancellatura punta sempre da quella parte. Il rumore
che si getta è **casuale**, e ogni volta diverso: mille manciate a caso non
formano nessuna figura, si pestano i piedi a vicenda e restano dove sono. Una
spinta piccola ma sempre nella stessa direzione, ripetuta mille volte, arriva
lontano; mille spinte grandi ma a casaccio, no.

E intanto il volume sale a ogni passo, senza che niente lo contrasti.
All'inizio del viaggio l'immagine c'è già, ma è un sussurro: vale meno di un
centesimo della sua ampiezza vera, sepolta sotto una coltre di rumore alta uno.
Alla fine dei mille passi quel sussurro è cresciuto di **centocinquanta
volte**, e il rumore, tirato dalle tre mosse in versi diversi, ha finito per
calare di cento. Nessuno ha ripulito niente un velo alla volta: si è alzata la
voce di qualcosa che era sempre stato lì.

Il rimescolamento, quindi, non è una svista da tollerare: è metà del
meccanismo. Nei primi passi la rete tira a indovinare (nel rumore quasi puro
non c'è ancora niente da vedere) e fidarsi della sua prima proposta
congelerebbe una direzione presa alla cieca; lo scossone tiene la partita
aperta finché l'immagine non si decide da sola. È anche il motivo per cui il
risultato cambia a ogni esecuzione: rumore iniziale diverso, scossoni diversi,
immagine diversa. Il costo si vede: mille interrogazioni della rete per *ogni*
immagine (il conto salato annunciato all'apertura del capitolo).

`````

`````{tab} Superiore

Il campionamento *ancestrale* di DDPM percorre la catena inversa da
$t = T$ a $t = 1$: si parte da $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ e si itera

$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t -
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)
+ \sigma_t \mathbf{z},
\qquad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I}),
$$

dove il primo termine è la media appresa $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$ e $\sigma_t \mathbf{z}$ è
rumore fresco con $\sigma_t = \sqrt{\beta_t}$; al passo finale $t = 1$ si pone
$\mathbf{z} = 0$ e si restituisce la media.

Vale la pena misurare i tre pezzi di quel passo, perché il loro rapporto è
controintuitivo e rovescia la descrizione a parole che di solito lo accompagna.
Il rumore stimato viene sottratto con coefficiente
$\beta_t / (\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t})$, quello fresco
iniettato con deviazione standard $\sigma_t = \sqrt{\beta_t}$: il rapporto
fra i due è

$$
\frac{\sigma_t}{\beta_t / \big(\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}\big)}
= \frac{\sqrt{\alpha_t\,(1-\bar{\alpha}_t)}}{\sqrt{\beta_t}},
$$

che con lo schedule di DDPM vale **da 7 a 10 per i primi novecento passi della
generazione**, cioè da $t = 1000$ fino a $t = 100$ (7,0 a $t = 1000$, 9,5 a
$t = 500$, 9,7 a $t = 250$, 7,0 a $t = 100$, con il massimo di 10,0 attorno a
$t = 350$). Per il novanta per cento del viaggio, dunque, il rumore iniettato
sposta il punto di quasi un ordine di grandezza più di quanto lo sposti la
correzione. Il rapporto cede solo nella coda: negli ultimi cento passi scende
fino a $\sqrt{\alpha_1} \approx 1$, e a $t = 1$ il rimescolamento non c'è
affatto. Cioè: la ripulitura arriva a pesare quanto il rimescolamento soltanto
alla fine, quando l'immagine è già decisa.

Il livello di rumore, allora, come scende? Poco per volta, e per il modo in
cui i tre contributi si compongono, non per la loro taglia. Scomponendo il
passo $t = 500$, dove il rumore presente ha deviazione standard $0{,}9599$: la
riscalatura per $1/\sqrt{\alpha_t}$ lo alza di $+0{,}0049$, la correzione lo
abbassa di $-0{,}0105$ (tutto il proprio valore, perché è allineata al rumore
che c'è), l'iniezione lo rialza di $+0{,}0052$ (molto meno del proprio
$0{,}1002$, perché è indipendente e si somma in varianza). Netto: $-0{,}0004$,
quattro decimillesimi, contro un $\sigma_t$ di $0{,}1$. La correzione vince un
tiro alla fune contro le altre due mosse, e lo vince di pochissimo.

Il segnale, invece, la riscalatura la incassa e basta: nessuno degli altri due
termini lo tocca, e $\sqrt{\bar{\alpha}_t} \to \sqrt{\bar{\alpha}_{t-1}}$
a ogni passo. Sull'intera catena il rapporto segnale/rumore passa da
$\sqrt{\bar{\alpha}_T / (1-\bar{\alpha}_T)} = 0{,}0064$ a circa $100$, un
fattore quindicimila, di cui **157 dall'amplificazione del segnale** e 100
dalla riduzione del rumore. Descrivere il campionamento come «togliere un velo
di rumore alla volta» racconta quindi metà del guadagno e circa un decimo del
gesto (a $t = 500$ la correzione vale $0{,}0105$ su uno spostamento
complessivo di $0{,}1008$), e tace la riscalatura, che è l'altra metà.

Il termine stocastico non è dunque un vezzo, ed è il pezzo più grosso del
passo per quasi tutto il viaggio: il processo inverso è esso stesso una catena
di distribuzioni, non una
funzione deterministica, e campionare da $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$
richiede sia la media sia il rumore di varianza $\sigma_t^2$. Il prezzo
computazionale è esplicito: $T$ valutazioni complete della rete per ogni
campione, con $T = 1000$, tre ordini di grandezza più di una GAN, che genera
in una singola passata in avanti.

`````

## Sotto il cofano: da dove viene il voto, e che cosa impara la rete

Due domande sono rimaste in sospeso. La prima riguarda il modo in cui abbiamo
dato il voto alla rete (misurare la distanza fra il disturbo indicato e quello
vero, e nient'altro): è un colpo di fortuna, o discende da un principio? La
seconda, più profonda: che cosa sta *davvero* imparando la rete, oltre a
vincere al nostro gioco?

`````{tab} Elementare

Alla prima domanda la risposta è che no, non è fortuna, ed è anzi un po' meno
pulita di come l'abbiamo raccontata. Esiste una ricetta generale per costruire
modelli che inventano cose nuove, ed è sempre la stessa: scrivi quanto
sarebbero probabili, secondo il tuo modello, i dati che hai davvero visto, e
regola i pesi perché quel numero sia il più alto possibile. Applicata alla
nostra catena di mille passi, dopo un po' di conti, quella ricetta si riduce
esattamente a «misura la distanza fra il disturbo indicato e quello vero», un
livello di rovina alla volta. Con un dettaglio che DDPM aggiunge di suo: la
ricetta pretenderebbe che i mille livelli contassero in modo diverso l'uno
dall'altro, e DDPM li fa contare tutti uguali. Non è più la ricetta di prima,
è una sua versione sbilanciata verso i passi difficili, quelli pieni di rumore.
Seguita alla lettera dà immagini peggiori; sbilanciata così, migliori. Capita,
e chi scrive queste cose fa bene a dirlo invece di far finta che tutto torni.

La seconda risposta è più bella. Immagina una mappa sterminata in cui ogni
punto è una possibile immagine: ogni
combinazione di pixel, anche le più assurde. Le immagini sensate (gatti, muri,
volti) occupano poche colline; il rumore riempie la pianura infinita
tutt'attorno. La rete, allenandosi a indicare il disturbo, sta imparando senza
saperlo una **bussola**: in ogni punto della mappa, la freccia che indica la
salita (la direzione in cui ritoccare l'immagine per renderla un po' più
credibile). Generare, allora, è partire da un punto a caso della pianura e
seguire la bussola a piccoli passi, con qualche scossone per non incastrarsi
nei fossi. Per anni due scuole hanno lavorato in parallelo (chi diceva
«insegniamo a togliere il rumore» e chi diceva «insegniamo la freccia della
salita»), finché non si è capito che stavano costruendo lo stesso oggetto con
due linguaggi diversi.

`````

`````{tab} Superiore

**Da dove viene la loss.** Come per ogni modello a variabili latenti, la
log-verosimiglianza $\log p_\theta(\mathbf{x}_0)$ (l'*evidenza*) non è calcolabile
direttamente, ma ammette un limite inferiore variazionale (ELBO). Nella
convenzione del libro, dove $\mathcal{L}$ si minimizza, si lavora con il suo
opposto, $\mathcal{L}_{\text{var}} = -\mathrm{ELBO}$ (un limite *superiore*
sulla log-verosimiglianza negativa), ed è questa quantità che si decompone in
$T-1$ divergenze KL, una per ciascun passo interno della catena, più due
termini di bordo: una ricostruzione $-\log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)$ e
un confronto sul prior, $D_{KL}\big(q(\mathbf{x}_T \mid \mathbf{x}_0)\,\|\,p(\mathbf{x}_T)\big)$, che
non contiene parametri e si può ignorare. Ogni KL confronta il **posteriore
condizionato al dato**, $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$, con il passo appreso
$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$; attenzione a non confonderlo con l'inverso vero
$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ incontrato sopra, gaussiano solo per approssimazione:
condizionando anche su $\mathbf{x}_0$, la distribuzione diventa gaussiana *esatta* e
nota in forma chiusa. Tra
gaussiane, ogni KL si riduce a una distanza quadratica tra medie; con la
riparametrizzazione di $\boldsymbol{\mu}_\theta$ vista sopra, ogni termine diventa
$\lVert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta \rVert^2$ moltiplicato per un peso
dipendente da $t$. La $\mathcal{L}_{\text{semplice}}$ è questo obiettivo con i
pesi posti a 1: non più un bound, ma una sua versione ripesata che nella
pratica produce campioni migliori {cite}`ho2020denoising`.

La riponderazione non è un dettaglio implementativo, e vale la pena dirne il
prezzo, perché il capitolo altrimenti userebbe come premessa una proprietà che
qui ritira. Il peso che l'ELBO assegna al passo $t$ è
$\lambda_t = \beta_t^2 / \big(2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)\big)$,
e con lo schedule di DDPM vale $0{,}500$ al passo 1 contro $0{,}0102$ al passo
1000: porli tutti a 1 significa moltiplicare per circa **cinquanta** il peso
relativo dei passi ad alto rumore rispetto a quelli quasi puliti. È voluto, e
Ho e colleghi lo dichiarano (la loss semplificata sottopesa i termini a $t$
piccolo, «così che la rete possa concentrarsi sui compiti di denoising più
difficili a $t$ grande»). La conseguenza è che un modello addestrato con
$\mathcal{L}_{\text{semplice}}$ **non massimizza più la verosimiglianza**: la
sua verosimiglianza è misurabilmente peggiore di quella dello stesso modello
addestrato sul bound vero, e resta lontana da quella dei modelli
autoregressivi, come gli autori scrivono senza girarci intorno. È il
compromesso che spiega una stranezza altrimenti inspiegabile della famiglia: i
modelli di diffusione producono campioni migliori dei modelli che ottimizzano
la verosimiglianza, pur avendo verosimiglianze peggiori.

**Cosa impara la rete.** Dalla forma chiusa dell'andata segue
$\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\boldsymbol{\epsilon} / \sqrt{1-\bar{\alpha}_t}$: il
rumore iniettato è, a meno di un fattore, il punteggio della densità
*condizionata al dato di partenza*. Il passaggio alla densità marginale
$q(\mathbf{x}_t)$ non è un corollario ma un risultato a sé: il minimo della loss
quadratica è la media condizionata
$\boldsymbol{\epsilon}^*(\mathbf{x}_t, t) = \mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t]$, e si può dimostrare (è
il *denoising score matching* di Vincent {cite}`vincent2011connection`) che
essa vale

$$
\boldsymbol{\epsilon}^*(\mathbf{x}_t, t) = -\sqrt{1-\bar{\alpha}_t}\;
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t),
$$

dove $\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)$ (il gradiente della log-densità dei dati
rumorosi, detto **score**) è esattamente la «freccia della salita» verso le
regioni più probabili. È la prospettiva *score-based* sviluppata da Yang Song
e Stefano Ermon, che Song e colleghi portano a compimento nel 2021
{cite}`song2021score`: l'andata è la discretizzazione di un'equazione
differenziale stocastica (SDE) che diffonde i dati nel rumore, e la SDE
inversa (che genera) dipende dai dati soltanto attraverso lo score. DDPM e i
modelli score-based si rivelano così due discretizzazioni dello stesso
processo continuo: una sola teoria, due dialetti.

`````

## Accelerare il ritorno: DDIM

Mille passi per un'immagine sono tanti, e la prima scorciatoia importante non
si fa attendere: arriva a meno di quattro mesi da DDPM, nell'ottobre del 2020
(poi a ICLR l'anno dopo), e sono i *Denoising Diffusion Implicit Models*
(DDIM) di Jiaming Song, Chenlin Meng e Stefano Ermon
{cite}`song2021denoising`. La promessa è
notevole: **lo stesso identico modello già addestrato**, nessun
riaddestramento, campioni di qualità paragonabile in una cinquantina di passi
invece di mille, e ancora accettabili a venti (nel paper, da 10 a 50 volte più
veloce in tempo di calcolo reale).

`````{tab} Elementare

Il restauratore alle prime armi solleva mille velature sottili, con mano
tremante e piccole correzioni casuali a ogni passaggio. Quello esperto ha
capito una cosa: il percorso di pulitura è sempre lo stesso film, e chi lo
conosce non ha bisogno di guardarlo fotogramma per fotogramma; può saltare
alle scene chiave. DDIM è il restauratore esperto: scende la stessa scala, ma
fermandosi solo a venti o cinquanta gradini scelti, con mano ferma e **senza
scossoni**. Niente scossoni significa anche un regalo inatteso: il
procedimento diventa ripetibile. Dallo stesso rumore iniziale esce
sempre, esattamente, la stessa immagine: il rumore di partenza diventa una
specie di codice dell'immagine, e sfumare da un codice a un altro fa sfumare
un'immagine nell'altra.

`````

`````{tab} Superiore

L'osservazione chiave è che $\mathcal{L}_{\text{semplice}}$ dipende dal
processo in avanti solo attraverso le marginali $q(\mathbf{x}_t \mid \mathbf{x}_0)$: mai
attraverso la struttura congiunta della catena. Esiste allora un'intera
famiglia di processi **non markoviani** con le *stesse* marginali, per i quali
la rete già addestrata è altrettanto valida; Song, Meng ed Ermon la
parametrizzano con un grado di stocasticità $\eta$: per $\eta = 1$ si recupera
il campionamento ancestrale di DDPM **nella variante con
$\sigma_t^2 = \tilde{\beta}_t$**, la varianza del posteriore, non in quella
con $\sigma_t^2 = \beta_t$ usata sopra; per $\eta = 0$ il passo inverso
diventa **deterministico**; dato $\mathbf{x}_T$, l'uscita $\mathbf{x}_0$ è una funzione, non un
campione. Qui la distinzione fra le due varianti, irrilevante sui mille passi,
diventa decisiva: a una cinquantina di passi la scelta
$\sigma_t^2 = \beta_t$ produce campioni molto peggiori della variante con
$\tilde{\beta}_t$, che a sua volta è peggiore del caso deterministico. Ed è
anche il motivo per cui DDIM è una scoperta e non una scorciatoia: accorciare
la catena non è solo saltare gradini, è cambiare quanto rumore si rimette a
ogni gradino.

E poiché la generazione non deve più simulare fedelmente una catena
markoviana passo-passo, può percorrere una sottosequenza
$\tau_1 < \dots < \tau_S$ di $\{1, \dots, T\}$ con $S \ll T$. Gli autori
dichiarano qualità paragonabile a quella dei mille passi già fra i venti e i
cento; la loro stessa tabella, letta con attenzione, dice qualcosa di un po'
più prudente, cioè che la parità piena arriva verso i cinquanta, che a venti
qualcosa si paga e che sotto i venti la qualità precipita. È un compromesso
regolabile fra costo e fedeltà, non un pasto gratis. La mappa
deterministica
rumore→immagine rende inoltre significative le interpolazioni in $\mathbf{x}_T$ e la
ricostruzione (quasi) esatta di un'immagine dal suo rumore. Non è un caso che
tutto ciò ricordi la vista continua di poche pagine fa, quella sotto il cofano:
il campionatore
DDIM con $\eta = 0$ è, in effetti, una discretizzazione dell'ODE del flusso di
probabilità associata alla SDE di {cite}`song2021score`.

`````

## Chi indovina il disturbo? Una vecchia conoscenza

Fin qui la rete è stata una scatola con due ingressi (l'immagine rovinata e il
numero del passo) e un'uscita della stessa forma dell'immagine. Ma quale
architettura? Guardiamo i requisiti: entra un'immagine ed esce un'immagine (la
mappa del rumore stimato, pixel per pixel, alla stessa risoluzione). Il
capitolo sulla visione artificiale ci ha già dato lo
strumento su misura: la **U-Net** di Ronneberger, Fischer e Brox
{cite}`ronneberger2015u`, nata nel 2015 per segmentare immagini biomediche.

`````{tab} Elementare

Nel capitolo sulla visione artificiale la U-Net imparava a dire, pixel per
pixel, «questo è cellula, questo è sfondo»; il nostro restauratore le fa fare
lo stesso identico gesto, ma la risposta ora è «ecco quanto rumore c'è su
questo pixel». Il suo trucco è guardare la foto due volte: prima fa qualche
passo indietro, e da lontano il pulviscolo sparisce e restano le forme
grandi; poi si riavvicina per scrivere la risposta punto per punto. E siccome
allontanandosi i dettagli minuti andrebbero perduti (e il disturbo è fatto
proprio di dettagli minuti), la rete tiene dei ponti diretti fra l'andata e
il ritorno, che li traghettano intatti.

Resta da dirle a che punto della scala sta lavorando. Il numero del passo
entra come un'etichetta attaccata alla foto: «questo è il livello 700 su
1000». Così una sola rete serve tutti i livelli di rovina: quando il rumore è
tanto sgrossa le forme, quando è poco rifinisce i dettagli.

`````

`````{tab} Superiore

La stessa rete che nella segmentazione colorava ogni pixel con la sua classe,
qui gli attribuisce la sua quota di rumore: un encoder che comprime, un
decoder che riespande, e le *skip connections* che traghettano i dettagli
fini dalla discesa alla risalita; preziose, perché il rumore è per sua natura
un dettaglio ad alta frequenza.

Resta da iniettare il tempo. Il passo $t$ entra nella rete come **embedding
sinusoidale** (lo stesso trucco degli encoding di posizione dei Transformer
{cite}`vaswani2017attention`, con $t$ al posto della posizione nella frase)
trasformato da un piccolo MLP e sommato alle feature di *ogni* blocco della
U-Net. Così una sola rete serve tutti i mille livelli di rumore: $t$ le dice
a quale punto della scala sta lavorando, se sgrossare forme globali (rumore
alto) o rifinire texture (rumore basso). La U-Net di DDPM aggiunge infine
blocchi di self-attention a **una sola** risoluzione intermedia (nei modelli
$32 \times 32$, la mappa $16 \times 16$, e non le due più basse), dove i pixel
sono già pochi abbastanza perché guardarli tutti insieme costi poco, ma non
così pochi da non avere più struttura da confrontare.

`````

Teniamo a mente questa composizione (una rete di visione, più il tempo cucito
addosso), perché è un equilibrio provvisorio: più avanti nel capitolo vedremo
un'architettura chiamata DiT buttare via la U-Net intera e metterci al suo
posto un Transformer, che invece di guardare l'immagine da lontano e poi da
vicino la taglia in tessere e le tratta come le parole di una frase. Era già
successo nella visione, con i Vision Transformer
{cite}`dosovitskiy2021image`.

## La diffusione in miniatura: una spirale di punti

Tutto il meccanismo sta in poche decine di righe di PyTorch, a patto di
scegliere dati abbastanza piccoli da
vederci dentro. Useremo punti del piano disposti a spirale: ogni «dato» è una
coppia di coordinate, e la diffusione li disperderà in una nuvola gaussiana
per poi imparare a ridisporli. Gli ingredienti sono *esattamente* quelli delle
immagini; cambia solo la taglia.

E se non hai mai programmato, nessun obbligo di leggere le righe una per una:
i quattro blocchi che seguono sono, nell'ordine, i quattro pezzi del racconto
(la spirale e la ricetta per rovinarla, la rete che indovina il disturbo, il
mazzo di carte dell'addestramento, la scala scesa mille volte), e il testo
fra un blocco e l'altro dice tutto quello che serve.

Prima i dati e la ricetta dell'andata:

```python
import numpy as np
import torch
from torch import nn

torch.manual_seed(0)
rng = np.random.default_rng(0)

# --- Dati: 2000 punti disposti a spirale, coordinate in [-1, 1] ---
n = 2000
angolo = 3.0 * np.pi * np.sqrt(rng.uniform(size=n))    # angolo lungo la spirale
raggio = angolo / (3.0 * np.pi)                        # il raggio cresce con l'angolo
spirale = np.stack([raggio * np.cos(angolo),
                    raggio * np.sin(angolo)], axis=1)  # shape (2000, 2)
spirale += 0.02 * rng.standard_normal(spirale.shape)   # leggero spessore del tratto
x0 = torch.tensor(spirale, dtype=torch.float32)        # (2000, 2)

# --- Schedule del rumore: lo stesso di DDPM ---
T = 1000
beta = torch.linspace(1e-4, 0.02, T)       # beta_t, shape (T,)
alpha = 1.0 - beta                         # alpha_t
alpha_bar = torch.cumprod(alpha, dim=0)    # alpha_t barrato, shape (T,)

def rumorizza(x0, t, eps):
    """Forma chiusa dell'andata: x_t dato x_0, per t interi in [0, T-1]."""
    ab = alpha_bar[t].unsqueeze(1)                     # (B, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps    # (B, 2)
```

Una cosa il giocattolo la lascia correre, e conviene dirla perché il capitolo
ci tornerà sopra: la spirale ha deviazione standard circa $0{,}5$ per
coordinata, non $1$, mentre lo schedule di DDPM è tarato su dati a varianza
unitaria. Qui non fa danno (il segnale è la metà del previsto, quindi la
catena lo affoga un po' prima del dovuto, e la spirale riemerge lo stesso);
in Stable Diffusion sì, e vedremo che quella riscalatura diventa una costante
scritta dentro il modello.

Poi la rete. Al posto della U-Net (i dati non sono immagini) basta una piccola
rete a strati densi, un MLP; il numero del passo entra come l'etichetta
attaccata alla foto di cui si parlava sopra, scritta in una forma che la rete
sa leggere (un *embedding sinusoidale*, lo stesso trucco degli encoding di
posizione dei Transformer) e appesa in coda alle coordinate:

```python
def embedding_tempo(t, dim=16):
    """Embedding sinusoidale del passo t: da (B,) a (B, dim)."""
    freq = torch.exp(torch.arange(dim // 2) * (-np.log(10000.0) / (dim // 2)))
    ang = t.float().unsqueeze(1) * freq.unsqueeze(0)   # (B, dim/2)
    return torch.cat([ang.sin(), ang.cos()], dim=1)    # (B, dim)

class PredittoreRumore(nn.Module):
    """La rete epsilon_theta(x_t, t): un MLP al posto della U-Net."""
    def __init__(self, dim_t=16, dim_h=128):
        super().__init__()
        self.dim_t = dim_t
        self.rete = nn.Sequential(
            nn.Linear(2 + dim_t, dim_h), nn.SiLU(),
            nn.Linear(dim_h, dim_h), nn.SiLU(),
            nn.Linear(dim_h, 2),                       # stima del rumore 2D
        )

    def forward(self, x, t):
        emb = embedding_tempo(t, self.dim_t)           # (B, dim_t)
        return self.rete(torch.cat([x, emb], dim=1))   # (B, 2)
```

Il ciclo di addestramento è la loss semplice di DDPM, riga per riga: pesca
un minibatch, un livello di rumore a caso per ciascun punto, il rumore
«vero», e confronta:

```python
modello = PredittoreRumore()
ottimizzatore = torch.optim.Adam(modello.parameters(), lr=2e-3)

for passo in range(30000):
    idx = torch.randint(0, n, (256,))         # minibatch di 256 punti
    batch = x0[idx]                           # (256, 2)
    t = torch.randint(0, T, (256,))           # un livello di rumore per esempio
    eps = torch.randn_like(batch)             # il rumore "vero" (la soluzione)
    x_t = rumorizza(batch, t, eps)            # (256, 2)
    predetto = modello(x_t, t)                # (256, 2), rumore stimato
    loss = ((eps - predetto) ** 2).mean()     # MSE: la loss semplice di DDPM
    ottimizzatore.zero_grad()
    loss.backward()
    ottimizzatore.step()
    if passo % 5000 == 0:
        print(f"passo {passo:5d}  loss {loss.item():.3f}")
```

Infine la discesa della scala, quella delle tre mosse (correggere, riscalare,
rimescolare), che in letteratura si chiama **campionamento ancestrale** perché
percorre la catena all'indietro un anello per volta: rumore gaussiano in
ingresso, mille interrogazioni della rete, spirale in uscita. Le tre mosse
sono segnate nei commenti; la terza si riconosce anche a occhio, perché è
l'unica che chiama di nuovo il generatore di numeri casuali:

```python
@torch.no_grad()
def campiona(n_campioni=1000):
    """Percorre la catena inversa da x_T (rumore puro) a x_0."""
    x = torch.randn(n_campioni, 2)                        # x_T ~ N(0, I)
    for t in reversed(range(T)):
        t_batch = torch.full((n_campioni,), t)            # (B,), tutti uguali a t
        eps_pred = modello(x, t_batch)                    # rumore stimato
        coeff = beta[t] / (1.0 - alpha_bar[t]).sqrt()     # quanto se ne toglie
        # mossa 1 (correggi) e mossa 2 (riscala), in una riga: mu_theta(x_t, t)
        media = (x - coeff * eps_pred) / alpha[t].sqrt()
        if t > 0:
            # mossa 3 (rimescola): sigma_t * z, con sigma_t = sqrt(beta_t).
            # E' il termine piu' grande dei tre, non un ritocco
            x = media + beta[t].sqrt() * torch.randn_like(x)
        else:
            x = media                    # ultimo passo: niente rumore fresco
    return x                             # (n_campioni, 2)

nuovi = campiona()
print(nuovi.shape)   # torch.Size([1000, 2]): mille coppie di coordinate

# "nuovi" e' una parola grossa: verifichiamola senza disegnare niente. Si
# confronta quanto dista un punto generato dal piu' vicino dell'archivio con
# quanto distano fra loro due punti dell'archivio: se i due numeri si
# somigliano, i generati cadono *fra* quelli di partenza, cioe' sulla spirale
# ma in posti dove non c'era nessuno
da_archivio = torch.cdist(nuovi, x0).min(dim=1).values
fra_archivio = torch.cdist(x0, x0).fill_diagonal_(float("inf")).min(dim=1).values
print(f"generato -> archivio: {da_archivio.median():.3f}   "
      f"archivio -> archivio: {fra_archivio.median():.3f}")
# generato -> archivio: 0.009   archivio -> archivio: 0.006
```

Disegnando `nuovi` con un grafico a dispersione si vede la spirale riemergere
dalla nuvola gaussiana, e le due mediane stampate dicono la stessa cosa senza
bisogno di disegnare: un punto generato dista dall'archivio ($0{,}009$) quanto
due punti dell'archivio distano fra loro ($0{,}006$), cioè cade sulla spirale
ma **in mezzo** agli altri, non sopra a uno di loro. Punti *nuovi*, non copie
dei duemila di partenza. (Tutto il ciclo è questione di meno di un minuto su
un portatile; se la
forma esce solo abbozzata, quasi sempre l'addestramento è stato troppo breve:
con poche migliaia di passi la nuvola non si è ancora decisa, i trentamila
del ciclo servono davvero.) Due
esperimenti valgono la pena: interrompere il campionamento a metà strada, per
vedere la forma «mezza decisa»; e passare da questi punti alle immagini, dove
l'unica modifica sostanziale è sostituire l'MLP con una U-Net e le coppie di
coordinate con griglie di pixel; schedule, loss e cicli restano identici,
carattere per carattere.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- L'**andata** non si impara, è una ricetta fissa: a ogni passo il valore di
  ogni pixel si attenua un pochino e riceve un pizzico di disturbo casuale,
  come nella tazza sempre piena in cui a ogni giro un cucchiaino di caffè
  lascia il posto a uno di latte. Dopo mille giri resta solo pulviscolo,
  e c'è una scorciatoia per arrivare a un livello di rovina qualsiasi senza
  ripercorrere i passi uno per uno.
- La rete impara a indicare **il disturbo**, non l'immagine pulita: è una
  domanda dello stesso tipo a ogni livello di rovina, e la risposta esatta la
  conosciamo sempre, perché il disturbo l'abbiamo fabbricato noi (il mazzo di
  carte con le soluzioni sul retro). Chi conosce il disturbo ricava
  l'immagine con una sottrazione.
- **Generare** vuol dire partire da un pulviscolo mai visto e
  ripetere mille volte tre mosse: cancella la scheggia di disturbo che la rete
  ti indica, alza il volume di tutto quello che resta, getta sopra una
  manciata di rumore nuovo (tranne all'ultimo passo, dove la terza mossa si
  salta).
- La cosa da non dimenticare è che **quella manciata è dalle sette alle dieci
  volte più grande della scheggia**, e lo è per i primi novecento passi su
  mille (poi cala, e sull'ultimo gradino le due si pareggiano): il passo non
  «solleva un velo», rimescola
  molto più di quanto pulisca. L'immagine esce fuori lo stesso perché la
  cancellatura è sempre mirata mentre il rumore nuovo è sempre a caso, e perché
  il volume, salendo a ogni passo, alla fine ha fatto crescere di
  centocinquanta volte un'immagine che all'inizio era un sussurro.
- Sotto il cofano la rete sta imparando una **bussola**: in ogni punto della
  mappa sterminata delle immagini possibili, la freccia che indica come
  ritoccare l'immagine per renderla un po' più credibile. Le due scuole che
  hanno lavorato in parallelo per anni («insegniamo a togliere il rumore» e
  «insegniamo la freccia») stavano costruendo lo stesso oggetto.
- **DDIM** è il restauratore esperto: lo stesso modello già addestrato,
  nessun riaddestramento, ma solo venti o cinquanta gradini scelti e mano
  ferma, senza scossoni casuali. In cambio il procedimento diventa
  ripetibile: dallo stesso rumore iniziale esce sempre la stessa immagine.
- La rete che indovina il disturbo è la **U-Net** già vista nella
  segmentazione, che guarda la foto da lontano e poi da vicino tenendo dei
  ponti fra le due viste; il numero del passo entra come un'etichetta
  attaccata alla foto. Più avanti nel capitolo un Transformer prenderà il suo
  posto.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L'**andata** è fissa: $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) =
  \mathcal{N}(\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t \mathbf{I})$ con
  schedule $\beta_t$; la forma chiusa $\mathbf{x}_t =
  \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}$
  salta da $\mathbf{x}_0$ a qualunque passo, e $\mathbf{x}_T$ è rumore
  gaussiano puro.
- La rete $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ impara a predire **il rumore**,
  non l'immagine: bersaglio a scala costante per ogni $t$, loss MSE
  $\mathbb{E}\lVert\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\rVert^2$ {cite}`ho2020denoising`
  (una regressione, stabile come un problema supervisionato).
- Generare = partire da $\mathbf{x}_T \sim \mathcal{N}(0,\mathbf{I})$ e risalire la
  catena in $T$ passi, e ogni passo fa **tre** cose: sottrae una frazione del
  rumore stimato, riscala per $1/\sqrt{\alpha_t}$, inietta rumore fresco di
  deviazione standard $\sigma_t$ (tranne all'ultimo).
- Il rapporto fra iniezione e sottrazione è
  $\sqrt{\alpha_t(1-\bar{\alpha}_t)}/\sqrt{\beta_t}$, cioè **da 7 a 10 per i
  primi novecento passi** (scende a 1 solo negli ultimi cento): si inietta
  molto più di quanto si sottragga. Il livello di rumore cala lo stesso, ma
  pochissimo per passo (quattro decimillesimi a $t = 500$), perché la
  sottrazione è allineata al rumore mentre l'iniezione si somma in varianza. E
  metà del guadagno sul rapporto segnale/rumore (fattore $157$ su $15\,000$
  complessivi) non viene dalla sottrazione ma dalla **riscalatura**.
- Sotto il cofano: la loss è una versione ripesata di $-\mathrm{ELBO}$
  (il **bound variazionale** da minimizzare, con i passi ad alto rumore
  favoriti di un fattore ~50, al prezzo di verosimiglianze peggiori), e
  predire il rumore
  equivale a stimare lo **score** $\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)$;
  DDPM e modelli score-based sono due discretizzazioni della stessa SDE
  {cite}`song2021score`.
- **DDIM** {cite}`song2021denoising`: stesso modello, campionamento
  deterministico ($\eta = 0$) su 20–50 passi; possibile perché la loss dipende
  solo dalle marginali $q(\mathbf{x}_t \mid \mathbf{x}_0)$, non dalla catena
  markoviana. Con $\eta = 1$ si ritrova il campionatore ancestrale nella
  variante $\sigma_t^2 = \tilde{\beta}_t$, non in quella
  $\sigma_t^2 = \beta_t$ usata qui.
- $\boldsymbol{\epsilon}_\theta$ è una **U-Net** {cite}`ronneberger2015u` (la stessa della
  segmentazione) con il passo $t$ iniettato come embedding sinusoidale; nella
  sezione su DiT verrà sostituita da un Transformer.
```

`````
