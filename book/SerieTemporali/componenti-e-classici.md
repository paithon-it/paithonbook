# Componenti e modelli classici: da ARIMA a Holt-Winters

Nel 1970 George Box e Gwilym Jenkins misero in appendice al loro libro una
tabella di numeri che sarebbe diventata la «cavia» di generazioni di
statistici: i passeggeri mensili delle linee aeree internazionali dal 1949 al
1960 {cite}`box2015time`. A guardarla, quella serie racconta tre storie
sovrapposte. Una **crescita** costante, perché era il decennio del boom dei
voli: ogni anno si vola più dell'anno prima. Un **respiro stagionale**: ogni estate un picco,
ogni inverno un avvallamento, puntuali come le stagioni che li causano. E,
sopra a tutto, un **tremolio** irregolare che nessuna regola spiega. Box e
Jenkins notarono anche un dettaglio decisivo: le oscillazioni estive non erano
sempre alte uguali, ma *crescevano* insieme al livello dei voli. È da qui che
parte ogni modello classico di serie temporale: imparare a leggere quelle tre
storie separatamente prima di provare a prevederle.

## Scomporre una serie: trend, stagionalità, residuo

La prima mossa, più vecchia dei calcolatori, è la **decomposizione**: separare
una serie nei suoi ingredienti. Ce ne sono tre, e la {numref}`fig-serie-decomposizione`
li mostra impilati uno sotto l'altro sullo stesso asse del tempo.

```{figure} ../figures/serie-decomposizione.svg
:name: fig-serie-decomposizione
:alt: Quattro pannelli impilati che condividono l'asse del tempo. Dall'alto la serie osservata con tendenza e oscillazione, poi il trend come linea liscia crescente, poi la stagionalità come onda periodica regolare, infine il residuo come piccole barrette attorno allo zero.
:width: 100%

La serie osservata (in alto) come somma di tre parti: una **tendenza** di fondo
(in inglese *trend*, ed è il nome che si usa anche in italiano), una
**stagionalità** che si ripete a intervalli regolari e un **residuo**
irregolare, che oscilla attorno allo zero.
```

`````{tab} Elementare

Dentro la bolletta della luce ci sono tre cose diverse. C'è una **parte
fissa**, il canone, che cambia poco e semmai cresce piano di anno in anno: è il
*trend*, la direzione di fondo. C'è una **parte stagionale**: d'estate il
condizionatore, d'inverno le luci accese di più (un su e giù che torna uguale
ogni anno). E poi c'è l’**imprevisto**: il mese che sei stato in ferie e hai
consumato meno, l'amico ospite che ha lasciato tutto acceso; piccoli scarti che
non seguono nessuna regola. È il *residuo*.

Scomporre una serie vuol dire fare esattamente questo: guardare la bolletta e
dire «di questi 90 euro, 60 sono il canone di base, 25 sono la stagione calda,
e 5 sono capitati così». Una volta separati i tre pezzi, ciascuno diventa più
facile da capire e da prevedere: il canone lo estrapoli, la stagione la ripeti,
e sull'imprevisto puoi solo dire quanto è grande di solito.

`````

`````{tab} Superiore

Si assume che la serie osservata $x_t$ sia composta da tre componenti latenti:
un **trend-ciclo** $T_t$, una **stagionalità** $S_t$ di periodo $m$ (12 per dati
mensili, 4 per trimestrali) e un **residuo** $R_t$. Le due forme canoniche sono
il modello **additivo** e quello **moltiplicativo**:

$$
x_t = T_t + S_t + R_t
\qquad\text{oppure}\qquad
x_t = T_t \times S_t \times R_t .
$$

Nell'additivo l'ampiezza della stagionalità è *costante* in valore assoluto:
l'estate aggiunge sempre lo stesso numero di unità, qualunque sia il livello.
Nel moltiplicativo l'ampiezza è *proporzionale* al livello: l'estate aggiunge
una percentuale, quindi cresce con la serie (proprio come i passeggeri delle
linee aeree). Un modello moltiplicativo si linearizza prendendo il logaritmo,
$\log x_t = \log T_t + \log S_t + \log R_t$, che riporta al caso additivo.

Attenzione al ritorno, perché è il passo che si dimentica: se si modella
$\log x_t$ e poi si esponenzia la previsione, quello che si ottiene è la
**mediana** di $x_{T+h}$, non la media, perché l'esponenziale non commuta col
valore atteso ($\mathbb{E}[e^X] = e^{\mu+\sigma^2/2}$ per una gaussiana). Con
$\sigma = 0{,}3$ la differenza è del $4{,}4\%$ verso il basso, e cresce come
$e^{\sigma^2/2}$. Non è sempre un errore: se si punta alla mediana (ed è il caso
se si giudica col MAE, o con la *pinball loss* al quantile $0{,}5$ della sezione
seguente) esponenziare e basta è esattamente giusto; se serve la media, perché
si sommano le previsioni di più prodotti o perché si giudica con l'RMSE, la
correzione va applicata {cite}`hyndman2021forecasting`.

Le stime pratiche di $T_t$, $S_t$, $R_t$ si ottengono con **medie mobili
centrate** (ogni istante sostituito dalla media dei suoi vicini: la
*classical decomposition*) o con metodi più robusti come STL
{cite}`hyndman2021forecasting`.

`````

Le tre parti si possono rimettere insieme in due modi, e la differenza conta. O
la stagione **aggiunge** sempre la stessa cifra, d'estate tanti euro in più e
sempre quelli, che il giro d'affari sia grande o piccolo: è la forma
**additiva**. Oppure la stagione **moltiplica**, cioè aggiunge una percentuale,
e allora cresce insieme al resto: è la forma **moltiplicativa**, ed è quella dei
passeggeri delle linee aeree con cui si è aperta la sezione, dove i picchi
estivi si alzavano man mano che si volava di più.

Un esempio minuscolo rende concreta la differenza. Prendiamo le vendite di
gelato di una gelateria in quattro trimestri (in migliaia di euro):

$$
x = (\underbrace{20}_{\text{inverno}},\ \underbrace{45}_{\text{primav.}},\ \underbrace{80}_{\text{estate}},\ \underbrace{35}_{\text{autunno}}).
$$

Con un anno solo di dati la direzione di fondo non si vede ancora (servirebbero
più anni per dire se sale o scende): quello che si può stimare è il **livello**
attorno a cui l'anno ruota, cioè la media dei quattro trimestri, che scriviamo
$\bar{x}$ (la barra sopra una lettera vuol dire «media di»):

$$
\bar{x} = (20+45+80+35)/4 = 45.
$$

La **stagionalità additiva**, che qui chiamiamo $S^{\text{add}}$ (le tre
letterine in alto sono solo un'etichetta per distinguerla dalla prossima, non un
esponente), è ciò che ogni trimestre aggiunge o toglie rispetto a quel livello.
I quattro numeri stanno fra parentesi tutti insieme perché sono una lista
ordinata, un valore per trimestre:

$$
S^{\text{add}} = (20-45,\ 45-45,\ 80-45,\ 35-45) = (-25,\ 0,\ +35,\ -10).
$$

Questi scarti sommano a zero: sono scarti *dalla media*, e per definizione di
media quello che sta sopra pareggia esattamente quello che sta sotto. La
**stagionalità moltiplicativa** è invece il rapporto rispetto al livello:

$$
S^{\text{mol}} = \left(\tfrac{20}{45},\ \tfrac{45}{45},\ \tfrac{80}{45},\ \tfrac{35}{45}\right)
\approx (0{,}44,\ 1{,}00,\ 1{,}78,\ 0{,}78),
$$

fattori che hanno **media** $1$ invece di sommare a zero, e il conto si controlla
subito: $0{,}44+1{,}00+1{,}78+0{,}78 = 4$, diviso quattro fa $1$. La lettura è
diversa: l'additivo dice
«d'estate si vendono 35 mila euro *in più* del solito»; il moltiplicativo dice
«d'estate si vende il 78% *in più*» (moltiplicare per $1{,}78$ e aggiungere il
$78\%$ sono la stessa operazione: $1$ è quello che c'era già, $0{,}78$ è quello
che si aggiunge). Se l'anno prossimo la gelateria raddoppia il giro d'affari, e
il livello passa da 45 a 90, il modello additivo continuerebbe a dire +35 mila,
sottostimando l'estate; quello moltiplicativo continua a dire +78%, che su un
livello di 90 vuol dire +70 mila, cioè uno scarto raddoppiato. Il **residuo** è ciò che resta dopo aver
tolto il livello e la stagione: se questa estate avesse fruttato $82$, mentre il
livello ($45$) e lo scarto estivo ($+35$) restano quelli stimati sugli anni
scorsi, il residuo di quel trimestre sarebbe $82 - (45 + 35) = 2$.

## Stazionarietà e differenziazione

L’{doc}`apertura del capitolo </SerieTemporali/overview>` ha presentato la
**stazionarietà** con l'immagine del fiume stabile. Detta per esteso: una serie è stazionaria quando il valore attorno
a cui balla, l'ampiezza con cui balla e il modo in cui due giorni si somigliano
restano gli stessi lungo tutta la serie. L'ultimo punto è quello che conta, e va
detto con precisione: due istanti si somigliano in base a **quanto** distano fra
loro, non a **quando** cadono nel calendario.[^senso-debole]

Perché proprio il terzo, e non è pedanteria: i primi due riguardano *dove* sta
la serie e *quanto* si agita, cose che si vedono a occhio e si aggiustano in
fretta. Il terzo riguarda la memoria, cioè proprio quello da cui una previsione
si tira fuori. Se il legame fra un giorno e il successivo è uno a gennaio e un
altro a luglio, non c'è nessuna regola da imparare: ce ne sono due, e il modello
ne troverà una terza che non vale né qui né lì.

Qui interessa il *perché* quasi tutti i modelli classici la pretendono, e la
ragione è semplice. Un modello si sceglie pochi numeri guardando i dati (li
chiameremo d'ora in poi i suoi **parametri**: la frazione con cui ieri pesa su
oggi, l'ampiezza tipica degli scossoni) e poi li dà per buoni su tutta la serie,
passato e futuro. Se la media scivola verso l'alto, o se l'ampiezza
delle oscillazioni cambia, quei parametri descrivono bene un pezzo di serie e ne
sbagliano un altro. Rendere la serie stazionaria vuol dire toglierle
di dosso trend e stagionalità, così che ciò che resta *balli sempre allo stesso
modo*.

Lo strumento più famoso è la **differenziazione**: sostituire ogni valore con la
sua variazione rispetto al precedente,

$$
\nabla x_t = x_t - x_{t-1}.
$$

Due parole sui simboli, perché tornano per tutto il capitolo. $x_t$ si legge «il
valore al tempo $t$»: la letterina in basso dice *quando*, non moltiplica
niente. E il triangolino rovesciato $\nabla$ è solo un'abbreviazione per «la
differenza fra un valore e quello prima».

Sulla serie $100, 110, 120, 130$, che cresce di $10$ a ogni passo, la
differenziata è $10, 10, 10$: la salita è sparita, resta una costante. Il
meccanismo si vede benissimo, ed è per questo che l'esempio è utile. Ma è anche,
esattamente, il caso in cui differenziare **non** è la mossa giusta, e conviene
capire subito perché.

Le tendenze sono di due tipi, e chiedono due cure diverse. Una tendenza è
**deterministica** quando la serie oscilla attorno a una retta: la retta c'è
davvero, le scosse la fanno sbandare ma non la spostano, e domani si torna sulla
riga di prima. È il caso della serie $100, 110, 120, 130$ di poco fa: la regola
c'è, e uno scarto casuale non la cambia. Una tendenza è invece **stocastica**
(la parola vuol dire «governata dal caso») quando la retta
non c'è: la serie cammina alla cieca, e ogni scossa le sposta il livello **per
sempre**, come il prezzo di un'azione in borsa che dopo un crollo riparte da
dove è arrivato e non da dove sarebbe dovuto essere.

La differenziazione è la cura del secondo caso, e lì è insostituibile. Sul primo
fa un danno, e il danno si vede a mano, senza far girare niente.

Prendi una serie che è davvero una retta più uno scossone casuale al giorno: il
valore di oggi è il punto della retta di oggi, più lo scossone di oggi. Adesso
sottrai il valore di ieri. Della retta resta solo la salita, sempre uguale
(quei $10$ dell'esempio di prima), e accanto a quella resta *lo scossone di oggi
meno quello di ieri*.

Guarda cosa è appena successo a un singolo scossone, mettiamo quello di ieri.
Ieri era il «più» del conto di ieri; oggi è il «meno» del conto di oggi. Lo
stesso numero, in due giorni consecutivi, una volta con un segno e una volta con
l'altro. Se ieri è stato grosso, ieri ha spinto in alto e oggi spinge in basso,
e così per ogni scossone: nella serie differenziata a un valore alto ne segue
regolarmente uno basso. Quella è una regolarità, e nella serie di partenza non
c'era: l'abbiamo fabbricata noi differenziando. Un modello che la guardi si
metterà a spiegarla.

Fatta la prova al computer (trecento punti di retta più rumore, ripetuta su
venti serie fatte allo stesso modo), i due numeri escono sempre gli stessi. Il
primo: fra un giorno e il successivo la somiglianza vale $-0{,}50$, cioè
fortemente negativa, ed è precisamente l'effetto appena descritto. Il secondo:
gli scarti della differenziata, elevati al quadrato e mediati, sono il **doppio**
di quelli del rumore che c'era dentro. Il doppio esatto, e non una quantità a
caso, perché ogni giorno adesso si porta dentro due scossoni invece di uno, e
quando si sommano due cose che non hanno niente a che vedere fra loro a sommarsi
sono i loro quadrati.

La cura del primo caso si chiama **detrendizzazione**: si tira la retta che
segue la salita e si tengono, di ogni punto, gli scarti da quella retta. È
precisamente ciò che faceva il codice dell'introduzione al capitolo con
`polyfit` e `polyval`, e sulla stessa serie di prova la detrendizzata esce con
una somiglianza fra giorni vicini di un paio di centesimi, cioè nulla, e con la
stessa irregolarità che c'era prima. Le due operazioni non sono
intercambiabili: ciascuna guasta il caso che l'altra risolve.

Per la stagionalità si usa la **differenziazione stagionale**
$\nabla_m x_t = x_t - x_{t-m}$, dove $m$ è la lunghezza del ciclo (12 per dati
mensili, 7 per dati giornalieri con un ritmo settimanale): sottrae il valore
dello stesso periodo del ciclo precedente, mese contro stesso mese dell'anno
prima. Prima di modellare si differenzia quanto basta a stabilizzare la serie, e
non di più: differenziare una volta di troppo è un errore vero e ha un nome,
**sovradifferenziazione**. È lo stesso guasto di poco fa, e la ragione è la
stessa: quando non c'era più niente da togliere, sottrarre il giorno prima non
toglie una salita, sparpaglia solo ogni scossone su due giorni.

E adesso la domanda pratica: come si fa a vedere quanta memoria è rimasta dentro
una serie, e di che tipo? Con due grafici a barre che sono il pane quotidiano
dell'analista di serie temporali, l’**ACF** e la **PACF**.

`````{tab} Elementare

Ricopia la serie su un foglio trasparente e fai scivolare la copia indietro di
un giorno: quanto si somigliano, le due? Poi di due giorni, poi di tre. La
risposta, passo per passo, è la **funzione di autocorrelazione** (ACF): un
grafico a barre che dice quanto oggi assomiglia a ieri, all'altro ieri, e così
via. Se le barre restano alte a lungo, la serie ha una memoria lunga.

C'è però un inganno. Se oggi somiglia a ieri e ieri somigliava all'altro ieri,
allora oggi somiglierà all'altro ieri *di rimbalzo*, anche senza un legame
diretto. La **PACF** (autocorrelazione parziale) toglie questo effetto a catena:
misura quanto oggi dipende dal valore di tre giorni fa *una volta scontato* ciò
che passa attraverso ieri e l'altro ieri. È la differenza fra «il nonno somiglia
al nipote» e «il nonno somiglia al nipote al netto del padre».

Si guardano in coppia, perché nessuno dei due da solo dice tutto, e insieme
dicono una cosa in più: il punto in cui le barre si schiacciano di colpo segna
fin dove arriva la memoria. Schiacciarsi vuol dire rientrare nella fascia
sottile attorno allo zero dove cadono le barre di una serie che memoria non ne
ha; e una barra sola che sporge non è la firma di niente, perché su venti barre
capita più spesso che no che una sporga per caso. Di memorie, fra poco, ne
incontreremo due, e ciascuna lascia la firma su un grafico diverso: se a
schiacciarsi di colpo è la PACF, la serie si ricorda i **valori** passati; se è
l'ACF, si ricorda gli **urti** passati. Con una riserva che verrà detta per
esteso: sulle serie vere le due firme si sovrappongono, e questo modo di
leggerle funziona molto meno di
quanto i manuali lascino sperare.

`````

`````{tab} Superiore

L’**autocorrelazione** a ritardo $k$ è

$$
\rho_k = \frac{\gamma(k)}{\gamma(0)}
= \frac{\mathrm{Cov}(x_t,\, x_{t-k})}{\mathrm{Var}(x_t)},
$$

dove $\gamma(k)$ è l'autocovarianza; la sequenza $\{\rho_k\}$ è l'ACF. La
**PACF** $\phi_{kk}$ è invece la correlazione fra $x_t$ e $x_{t-k}$ dopo aver
rimosso la dipendenza lineare dai ritardi intermedi $x_{t-1},\dots,x_{t-k+1}$;
coincide con l'ultimo coefficiente della regressione di $x_t$ su quei $k$
ritardi. Le due funzioni sono la bussola dell'identificazione di Box-Jenkins,
perché hanno firme complementari: un processo **AR($p$)** ha PACF che si
**annulla** dopo il ritardo $p$ e ACF che decade gradualmente; un processo
**MA($q$)** ha ACF che si **annulla** dopo il ritardo $q$ e PACF che decade.
Leggere dove le barre «cadono nel rumore» suggerisce gli ordini $p$ e $q$ da
provare {cite}`box2015time`. Vale però sui processi puri: quando le due memorie
convivono, cioè su un ARMA, nessuna delle due funzioni si annulla e la lettura
non decide niente, ed è la ragione per cui in pratica gli ordini si scelgono
stimando una griglia di modelli invece che guardando un grafico.

«Nel rumore» ha una definizione precisa: dentro la banda
$\pm z_{1-\alpha/2}/\sqrt{n}$, cioè $\pm 1{,}96/\sqrt{n}$ al 95%, che è
l'intervallo in cui cadrebbe un'autocorrelazione campionaria se quella vera
fosse zero ($n$ è il numero di osservazioni). Va letta sapendo che è una banda
**puntuale**, valida un ritardo per volta: su venti ritardi di rumore bianco
puro, la probabilità che almeno una barra esca dalla banda per puro caso supera
il 50%. Una barra fuori non è la firma di niente, ed è esattamente la ragione
per cui il test di Ljung-Box, più avanti, giudica tutte le autocorrelazioni
**insieme** invece che una per una.

`````

## Autoregressione: AR($p$)

Il modello più naturale per una serie con memoria dice: il prossimo valore è una
combinazione dei valori appena passati, più una spinta casuale.

`````{tab} Elementare

Domani la temperatura sarà simile a quella di oggi, con una correzione. Se oggi
fa più caldo della media di stagione, è probabile che anche domani sia sopra la
media, ma un po’ meno: il caldo «rientra» piano verso il normale. Un modello
**autoregressivo** cattura proprio questo: prende gli ultimi valori, li pesa, li
somma, e aggiunge un pizzico di imprevedibile per il resto. «Auto-regressivo»
vuol dire che la serie fa da predittore *a sé stessa*: guarda il proprio
passato, non variabili esterne.

Il numero di passati che guarda è l’**ordine**, ed è la lettera $p$ che compare
fra parentesi nel nome del modello: AR($p$) vuol dire soltanto «autoregressivo
che guarda indietro di $p$ giorni». Un AR(1) guarda solo ieri; un AR(2) guarda
ieri e l'altro ieri. Più passati includi, più la memoria del modello si
allunga, ma anche più parametri devi stimare da una serie che è pur sempre
lunga un tanto.

`````

`````{tab} Superiore

Un processo **autoregressivo di ordine $p$**, AR($p$), è definito da

$$
x_t = c + \phi_1 x_{t-1} + \phi_2 x_{t-2} + \dots + \phi_p x_{t-p} + \varepsilon_t,
$$

dove $\phi_1,\dots,\phi_p$ sono i coefficienti autoregressivi, $c$ una
costante e $\varepsilon_t$ è **rumore bianco**: una sequenza a media nulla,
varianza $\sigma^2$ costante e incorrelata nel tempo. La parte prevedibile è
$c + \sum_{i=1}^p \phi_i x_{t-i}$, e per chiamarla *valore atteso condizionato*
al passato serve un'ipotesi più forte della sola incorrelazione, cioè che la
scossa di oggi non sia prevedibile da ciò che è successo prima; con la sola
incorrelazione quella quantità è la migliore previsione **lineare**, che è meno.

La condizione sui coefficienti riguarda il **polinomio autoregressivo**
$\phi(z) = 1 - \phi_1 z - \dots - \phi_p z^p$, e chiede che le sue radici
stiano **fuori** dal cerchio unitario. Così scritta dà la stazionarietà *e* la
causalità, cioè il fatto che il processo si possa scrivere in funzione delle
sole scosse passate, che è poi ciò che serve per prevedere; per la sola
stazionarietà basterebbe che nessuna radice cada **sul** cerchio. Attenzione
alla convenzione, perché è la sede di un inciampo classico: chi scrive il
polinomio nell'altra forma, $z^p - \phi_1 z^{p-1} - \dots - \phi_p$, chiede le
radici **dentro** il cerchio, e sta dicendo la stessa identica cosa. Per
l'AR(1) tutto questo si riduce alla condizione $|\phi_1| < 1$. In quel
caso la media di lungo periodo è $\mu = c/(1-\phi_1)$ e la serie vi ritorna
dopo ogni scossa.

`````

Vediamolo con i numeri su un AR(1). La regola dice: il valore di domani è una
quota fissa più una frazione del valore di oggi, più una scossa casuale. In
simboli, $x_t = c + \phi\,x_{t-1} + \varepsilon_t$, dove $c$ è la quota fissa,
$\phi$ (la lettera greca *fi*) la frazione, e $\varepsilon_t$ la scossa.
Prendiamo $c = 4$ e $\phi = 0{,}6$, cioè: di quello che c'era ieri ne resta il
60%, e ogni giorno se ne aggiungono 4.

Dove finisce una serie fatta così, se la si lascia andare? In un valore che non
si muove più, cioè quello per cui «il 60% di sé stesso più 4» ridà sé stesso.
Chiamiamolo $\mu$ (si legge *mu*, ed è la lettera con cui in statistica si
indica una media). La condizione si scrive $\mu = 0{,}6\,\mu + 4$, e si risolve
come qualunque equazione di prima media: porti i $\mu$ da una parte,
$\mu - 0{,}6\,\mu = 4$, cioè $0{,}4\,\mu = 4$, quindi $\mu = 10$. È la **media
di lungo periodo**, e in generale vale $c/(1-\phi)$.

Partiamo da un valore alto, $x_0 = 20$, e seguiamo la parte
prevedibile (mettendo a zero il rumore, $\varepsilon_t = 0$):

$$
\begin{aligned}
x_1 &= 4 + 0{,}6 \cdot 20 = 16, \\
x_2 &= 4 + 0{,}6 \cdot 16 = 13{,}6, \\
x_3 &= 4 + 0{,}6 \cdot 13{,}6 = 12{,}16.
\end{aligned}
$$

La serie scivola $20 \to 16 \to 13{,}6 \to 12{,}16$, avvicinandosi a $10$ a ogni
passo: è il **rientro verso la media** (in inglese *mean reversion*). Succede
finché la frazione $\phi$ sta fra $-1$ e $+1$, che è quello che dice la
scrittura $|\phi| < 1$: le due sbarrette vogliono dire «guarda il numero senza
il segno». Se valesse $1$ o più, ogni giorno
ricomincerebbe da dove era arrivato o più in là, e la serie non tornerebbe mai
indietro: è la tendenza stocastica di poco fa. Il rumore, nella realtà,
scompiglia continuamente la discesa, ma la spinta di fondo resta sempre quella
verso $\mu$.

## Da AR ad ARIMA: media mobile, integrazione, stagionalità

L'AR è metà della storia. L'altra metà guarda non ai valori passati, ma agli
**urti** passati: è il modello a **media mobile**, sigla MA. Attenzione, perché
«media mobile» in questo campo indica due cose diverse. La prima, la più comune,
è il modo più semplice di lisciare un grafico: si sostituisce ogni valore con la
media dei suoi vicini, e così il tremolio si attenua e si vede il fondo. È anche
il modo più semplice di tirar fuori la tendenza di una serie, ed è per questo
che il nome ricorre in giro. La seconda è questa, il modello MA, e non è una
media dei valori: è
una media **degli imprevisti**. Sono due mestieri diversi con lo stesso nome, e
d'ora in avanti «media mobile», da sola, indica il modello.

`````{tab} Elementare

Una gita scolastica svuota la gelateria; uno sciopero blocca i voli. L'effetto
di un urto imprevisto non si esaurisce il giorno stesso: si fa sentire ancora
domani, un po’ meno dopodomani, e poi svanisce. Un modello a
media mobile dice proprio questo: il valore di oggi è il livello normale, più
la sorpresa di oggi, più l'eco delle sorprese degli ultimi giorni. Quanto pesa
ciascun giorno passato è un numero che si legge dai dati, non una regola fissa.

Le sorprese però nessuno le misura: nel registro della gelateria ci sono gli
incassi, e la sorpresa di ieri si ricava all'indietro, per differenza fra quello
che ci si aspettava e quello che è arrivato. Il conto all'indietro riesce finché
i pesi restano dentro certi limiti; fuori da quelli il modello è scritto bene,
ma alle sorprese non si risale, e senza quelle non prevede niente.

Le due memorie si possono usare insieme: quella dei valori (l'AR appena visto)
e quella degli urti (il MA). E siccome le serie vere quasi mai stanno ferme
attorno a un valore, prima si raddrizza la serie e poi si modella ciò che
resta. Raddrizzare, qui, vuol dire il trucco già incontrato per le serie che
camminano alla cieca: sostituire ogni valore con la variazione rispetto al
giorno prima. (Quando invece la serie oscilla attorno a una retta, la retta si
toglie prima, fuori dal modello.) Il tutto insieme si chiama **ARIMA**, il
modello di punta di Box e Jenkins, e la sigla è
la somma dei tre pezzi: **AR** la memoria dei valori, **I** (*integrated*) il
raddrizzamento, **MA** la memoria degli urti. Dietro non ci sono che tre
conteggi, quanti valori passati guardare, quante volte raddrizzare la serie,
per quanti giorni far durare l'eco degli urti, e i manuali li scrivono in
quest'ordine fra parentesi, ARIMA($p,d,q$). Se c'è anche una stagionalità, si
rifà lo stesso gioco sul calendario (dicembre si confronta con lo scorso
dicembre): è la
variante **SARIMA**, dove la S sta per *seasonal*, stagionale.

`````

`````{tab} Superiore

Il modello a **media mobile** MA($q$) è

$$
x_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q},
$$

dove $\varepsilon_{t-i}$ sono le scosse casuali dei passi precedenti e
$\theta_1,\dots,\theta_q$ i loro pesi: un urto di oggi non si esaurisce
subito, ma continua a farsi sentire per $q$ passi prima di svanire.

Come per l'AR c'è una condizione sui coefficienti, ma serve a un'altra cosa. Si
chiama **invertibilità** e chiede che le radici di
$1 + \theta_1 z + \dots + \theta_q z^q$ stiano fuori dal cerchio unitario
($|\theta_1| < 1$ per l'MA(1)). Non serve per la stazionarietà, che un MA ha
comunque, essendo una somma finita di scosse a varianza costante. Serve per due
ragioni concrete. La prima: senza di essa il modello **non è identificato**,
perché $\theta$ e $1/\theta$ danno la stessa autocorrelazione
($\rho_1 = \theta/(1+\theta^2)$ vale $0{,}40$ tanto per $\theta = 0{,}5$ quanto
per $\theta = 2$), quindi la stessa ACF, quindi due processi che i due grafici
di poco fa non sanno distinguere. La seconda: solo con essa le scosse
passate $\varepsilon_{t-i}$, che nessuno osserva, si possono ricostruire dai
dati, cioè solo con essa il modello si può usare per prevedere. Una spia utile:
quando una serie è stata **sovradifferenziata**, il $\theta$ stimato finisce
inchiodato a $-1$ o quasi, cioè proprio sul bordo di questa regione.

Mettendo insieme le due idee si ottiene l’**ARMA($p,q$)**, che spiega il valore
odierno con $p$ valori passati e $q$ errori passati. Ma l'ARMA vive solo su
serie stazionarie, e le serie vere quasi mai lo sono. La soluzione di Box e
Jenkins è incorporare la differenziazione nel modello stesso: nasce
l’**ARIMA($p,d,q$)** {cite}`box2015time`. Le tre lettere:

- **AR($p$)**, l'ordine autoregressivo, quanti valori passati;
- **I($d$)**, *integrated*, quante volte si differenzia la serie per renderla
  stazionaria ($d=1$ toglie un trend lineare, $d=2$ una curvatura);
- **MA($q$)**: l'ordine a media mobile, quanti errori passati.

In pratica si differenzia la serie $d$ volte, si adatta un ARMA($p,q$) al
risultato, e si «re-integra» sommando all'indietro per tornare alla scala
originale. Quando la serie ha una stagionalità marcata, si aggiunge un secondo
blocco di termini che agiscono al ritardo stagionale $m$: è il
**SARIMA($p,d,q$)($P,D,Q$)$_m$**, dove le lettere maiuscole $P,D,Q$ sono gli
ordini AR, di differenziazione e MA *stagionali*, e $m$ è la lunghezza del ciclo.
Un SARIMA$(1,1,1)(1,1,1)_{12}$ è, ancora oggi, un ottimo punto di partenza per
una serie mensile con trend e stagionalità annuale.

`````

## Scegliere l'ordine, e poi verificare i residui

Sappiamo che cos'è un ARIMA, e sappiamo che i tre conteggi fra parentesi si
scrivono $p$ (quanti valori passati), $d$ (quante volte si raddrizza la serie) e
$q$ (per quanti giorni dura l'eco degli urti). Resta la domanda pratica, che è
quella che si pone chiunque abbia una serie davanti: **quali numeri ci metto
dentro, e come faccio a sapere se il modello che ne esce va bene?** La risposta
è una procedura in tre tempi, e vale per tutti i modelli di questa famiglia, dal
più piccolo al SARIMA più carico di lettere.

`````{tab} Elementare

**Primo tempo: raddrizzare la serie.** Si toglie la retta se una retta c'è,
oppure si sostituisce ogni valore con la variazione rispetto al giorno prima se
la serie cammina alla cieca. Per decidere se la serie è già a posto ci sono due
esami, e vanno letti insieme: partono da sospetti opposti, e lo stesso responso
che nell'uno vuol dire «è a posto», nell'altro vuol dire il contrario. Da qui
esce già uno dei tre conteggi, quello di mezzo, che dice quante volte si è
raddrizzato.

**Secondo tempo: scegliere gli altri due.** La ricetta dei manuali dice di
leggere i due grafici a barre, l'ACF e la PACF, e dedurli da lì.
Funziona sui casi da libro di testo, e sulle serie vere quasi mai: quando ci
sono insieme la memoria dei valori e quella degli urti, entrambi i grafici
scendono lentamente e non si legge niente.

Allora si fa la cosa onesta: **si provano tutte le combinazioni** entro un
limite ragionevole. Sono una manciata di modelli e il computer li stima tutti in
qualche secondo. Poi si sceglie con un criterio che tiene conto di due cose
insieme: quanto bene il modello spiega i dati, e quanti numeri ha dovuto
inventarsi per riuscirci. Il secondo pezzo conta quanto il primo: aggiungere
numeri migliora sempre l'aderenza ai dati che si hanno sotto gli occhi, quindi
senza una penalità si finirebbe per scegliere ogni volta il modello più grosso,
che è
il modo classico di imparare a memoria invece che imparare la regola.

**Terzo tempo: guardare quello che resta.** Fatta la scelta non si è finito, e
questo è il passo che quasi tutti saltano, che è poi l'unico che dice se il
modello è *utile*.
Un modello buono ha spremuto dalla serie tutta la regolarità che c'era, quindi
ciò che avanza, la differenza fra previsto e osservato, deve essere
indistinguibile dal caso. Se in quello che avanza si vede ancora una
regolarità, vuol dire che il modello se l'è lasciata sfuggire, e va cambiato. E
se non si vede niente si è imparato meno di quanto sembri: vuol dire che con
questi dati nessuna regolarità è saltata fuori, non che non ce ne siano.

Il modello sbagliato lo si riconosce dai suoi errori, non dalle sue previsioni.
Le previsioni sembrano sempre plausibili; gli errori, se guardati, confessano.

`````

`````{tab} Superiore

La procedura è quella di Box e Jenkins nella forma che si usa oggi, in tre
tempi.

**1. Rendere stazionaria la serie.** Si testa, si toglie il necessario, si fissa
$d$ (e $D$ per la parte stagionale). Sovradifferenziare è un errore reale e
riconoscibile: introduce autocorrelazione negativa artificiale al ritardo 1
(esattamente $-0{,}5$ nel caso limite) e gonfia la varianza, e si vede anche dal
$\theta$ stimato, che finisce sul bordo della regione di invertibilità.

Sui test conviene spendere quattro righe, perché sono due e vanno usati
**insieme**. L’**ADF** (Dickey-Fuller aumentato) ha per ipotesi nulla «c'è una
radice unitaria», quindi un $p$-value **basso** dice *stazionaria*; il **KPSS**
ha per ipotesi nulla «la serie è stazionaria», quindi un $p$-value **basso**
dice *non stazionaria*. Il KPSS ha dunque lo stesso verso del Ljung-Box del
passo 3 (si spera di non rifiutare), l'ADF ha il verso contrario, e portare la
regola dell'uno sull'altro è l'errore più facile del capitolo. Usarli in coppia
dà quattro esiti
e non due: concordi in un senso, concordi nell'altro, e i due casi in cui non
concordano, che sono i più informativi, perché dicono che con questi dati la
domanda non si decide e conviene guardare il grafico.

Va aggiunto che il verdetto dipende dai **termini deterministici** che si
mettono nella regressione ausiliaria del test. Si riprenda la serie di prova di
poco fa, una retta più rumore: di radici unitarie non ne ha nessuna. Con la sola
costante l'ADF **non rifiuta** (su venti repliche di quella serie il $p$ medio è
$0{,}96$), e chi segue la ricetta alla lettera differenzia, cioè
sovradifferenzia. Mettendo il trend nella specificazione, la stessa serie sugli
stessi dati dà il verdetto opposto, con un $p$ praticamente nullo. Lo stesso
vale per il KPSS, che a seconda della specificazione ha per ipotesi nulla la
stazionarietà attorno a una costante oppure attorno a un trend: di entrambi i
test va saputo quale delle due domande si è posta. Non è un difetto dei test: è
che stanno rispondendo a domande diverse.

**2. Scegliere gli ordini con un criterio di informazione.** Si stimano tutte
le combinazioni di $(p,q)$ entro una griglia e si prende quella che minimizza
l’**AIC**:

$$
\mathrm{AIC} = 2k - 2\ln \hat{L},
$$

dove $\hat{L}$ è la verosimiglianza massimizzata e $k$ il numero di parametri.
Il primo termine penalizza la complessità, il secondo premia l'aderenza: è lo
stesso compromesso bias-varianza del capitolo sul Machine Learning, espresso in
valuta di verosimiglianza invece che di errore su un set di validazione. Il
**BIC** ($k\ln n - 2\ln\hat L$, con $n$ il numero di osservazioni) penalizza
di più al crescere delle osservazioni e tende a scegliere modelli più piccoli.

Due dettagli che cambiano il numero, e che quindi non sono dettagli. Il primo:
in $k$ entra anche la **varianza dell'innovazione**, non solo i $\phi$, i
$\theta$ e la costante; `statsmodels` la conta (per un ARMA(1,1) con costante
$k=4$), e chi rifà il conto a mano con $k=3$ sbaglia di due unità, cioè
esattamente la soglia sotto la quale l'AIC non distingue niente. Il secondo: il
$2k$ è una correzione **asintotica**, e in campione corto va sostituita con
quella esatta, l’**AICc** $= \mathrm{AIC} + \frac{2k(k+1)}{n-k-1}$, che è
quella che i manuali usano di default sugli ARIMA
{cite}`hyndman2021forecasting`. Con seicento osservazioni la differenza è di
sei centesimi; con quaranta, e sei parametri, supera le due unità e cambia la
scelta.

Una nota che vale più della formula: **l'AIC è una quantità relativa**. Il suo
valore assoluto non significa nulla, contano solo le differenze, e differenze
sotto le due unità non sono evidenza di niente.

E contano solo fra modelli stimati **sugli stessi dati**. È la clausola che
rende l'AIC un criterio invece che un numero, ed è la ragione per cui $d$ si
fissa al passo 1 e non si mette nella griglia: differenziare cambia i dati su
cui la verosimiglianza è calcolata (una serie differenziata una volta ha
un'osservazione in meno), e due AIC così non si possono sottrarre. Vale identico
per le trasformazioni: l'AIC di un modello su $\log x_t$ e quello di un modello
su $x_t$ non vivono nella stessa scala, e la differenza fra i due è dominata dal
cambio di variabile, non dal modello. Su una serie di prova (duecento punti
positivi che partono da un centinaio e si moltiplicano fino a circa sette volte
tanto, con un ARMA(1,1) stimato su ciascuna delle due scale) il logaritmo
«vince» di quasi duemilaquattrocento unità, cioè di più di mille volte la soglia
delle due. Ma duemiladuecento di quelle unità sono soltanto il cambio di
variabile, cioè il termine jacobiano $2\sum_t \log x_t$: rimettendolo al suo
posto, del vantaggio ne resta poco più di un centinaio. Quel termine dipende
**solo da quanto sono grandi i numeri** della serie, non da come si comportano,
ed è la ragione per cui la lunghezza della serie e la sua crescita cambiano i
primi due numeri di questo conto lasciando intatta la morale: il confronto
grezzo stava misurando l'unità di misura.

**3. Verificare i residui.** Se il modello ha catturato la struttura, i residui
$\hat\varepsilon_t = x_t - \hat x_t$ devono essere **rumore bianco**: media
nulla, varianza costante, nessuna autocorrelazione. Due strumenti, uno
qualitativo e uno quantitativo.

Il **Q-Q plot** confronta i quantili empirici dei residui con quelli di una
normale: se stanno su una retta, la distribuzione è normale come si assume. È
veloce e resta un giudizio a occhio.

Il **test di Ljung-Box** è quantitativo e testa congiuntamente le prime $\ell$
autocorrelazioni dei residui:

$$
Q = n(n+2) \sum_{k=1}^{\ell} \frac{\hat\rho_k^2}{n-k},
$$

con $n$ il numero di osservazioni, $\ell$ il numero di ritardi esaminati e
$\hat\rho_k$ l'autocorrelazione campionaria al ritardo $k$. Sotto l'ipotesi
nulla di **assenza** di autocorrelazione, $Q$ si distribuisce *asintoticamente*
come una $\chi^2$: è un'approssimazione per $n$ grande, e regge se i ritardi
esaminati sono pochi rispetto alle osservazioni. Quanti: la regola d'uso è
$\ell = 10$ su una serie senza stagionalità e $\ell = 2m$ su una che ce l'ha,
comunque non oltre $n/5$, e comunque più di $p+q$, altrimenti i gradi di libertà
del prossimo paragrafo diventano zero o negativi
{cite}`hyndman2021forecasting`.

Con quanti gradi di libertà, però, non è un dettaglio. Applicato ai residui di
un ARMA **stimato**, il test va calcolato con $\ell - (p+q)$ gradi di libertà e
non con $\ell$ {cite}`hyndman2021forecasting`: i parametri già spesi per far
aderire il modello ai dati non contano come prove d'innocenza. Ometterlo è la
scorciatoia più diffusa della materia (le librerie lasciano fare, perché il
parametro va passato a mano) ed è **sempre ottimista**: gonfia il $p$-value, e
cioè fa sembrare adeguati modelli che lo sono meno. Sull'ARMA scelto con
seicento osservazioni la differenza fra le due letture è fra $0{,}71$ e
$0{,}51$.

Attenzione anche al verso, perché è controintuitivo: qui **si spera di non
rifiutare**. Un $p$-value alto significa «non c'è evidenza di struttura
residua», cioè il modello va bene; un $p$-value basso significa che qualcosa è
rimasto fuori. È il verso opposto a quello dell'ADF del passo 1.

Non rifiutare
l'ipotesi nulla **non prova** che i residui siano rumore bianco: prova solo che
il test, con quei dati, non ha trovato prove del contrario. È la stessa
asimmetria di ogni test d'ipotesi, e la ragione per cui la diagnostica non
sostituisce la validazione su dati futuri, che il capitolo affronta nella
sezione seguente.

`````

Il ciclo, in una riga: **stima, seleziona, verifica, e se in quello che resta si
vede ancora una regolarità torna indietro**. È la stessa disciplina che nel
Machine Learning tiene separati i dati su cui il modello impara da quelli su cui
lo si esamina, applicata qui a un oggetto diverso.

Le due cose hanno un nome, e conviene averlo prima di vederle all'opera. Il
criterio che sceglie fra i modelli si chiama **AIC**: più è basso, meglio è. Ma
è un numero che vale solo per differenza, e la differenza va guardata con una
soglia in testa: **sotto le due unità l'AIC non sta distinguendo niente**, e due
modelli così vicini sono, per lui, lo stesso modello. Quel due non è una legge
di natura, è la regola d'uso della materia, e nasce da un'osservazione semplice:
scarti più piccoli si ottengono anche solo cambiando una manciata di
osservazioni, quindi non sono prova di niente.

Il test che guarda quello che resta si chiama **Ljung-Box**, dai due statistici
che lo misero a punto, e risponde a una domanda sola: negli errori del modello
si vede ancora una regolarità, o sembrano capitati a caso? Un mucchio di numeri
senza nessuna regolarità dentro si chiama **rumore bianco**, ed è il
complimento più alto che si possa fare agli errori di un modello: vuol dire che
tutto ciò che si poteva spremere è stato spremuto.

La risposta del test è un numero fra $0$ e $1$ chiamato **$p$-value**, e va
letta al contrario di quanto verrebbe naturale: alto vuol dire «nessuna traccia
di regolarità», cioè il modello va bene; vicino a zero vuol dire «una regolarità
c'è, e l'hai lasciata fuori». La soglia d'uso è $0{,}05$, per convenzione.

### In pratica: l'AIC sceglie, Ljung-Box giudica

Si può vedere l'intera procedura su una serie di cui **conosciamo la risposta**,
perché la generiamo noi. La generiamo da un ARMA(2,1): un ARIMA senza il
raddrizzamento (la I di mezzo), perché la serie che ci fabbrichiamo è già
stabile e non c'è niente da raddrizzare. Guarda due valori passati e un urto
passato.

```python
import warnings
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.simplefilter("ignore")     # le convergenze borderline qui non interessano

# Una serie generata da un ARMA(2,1) NOTO: la risposta giusta la sappiamo.
processo = ArmaProcess(ar=np.r_[1, -0.6, -0.25], ma=np.r_[1, 0.4])

def scegli(serie):
    """Stima tutte le combinazioni fino a ordine 3 e le ordina per AIC."""
    esiti = []
    for p in range(4):
        for q in range(4):
            try:
                # d è fissato al passo 1 e NON entra nella griglia: differenziare
                # cambia i dati, e due AIC su dati diversi non si sottraggono
                esiti.append((ARIMA(serie, order=(p, 0, q)).fit().aic, p, q))
            except Exception:
                continue
    return sorted(esiti)

def ljung_box(residui, p, q, lags=10):
    """p-value di Ljung-Box sui residui di un ARMA(p,q) stimato.
    I gradi di libertà sono lags - (p+q): i parametri già spesi per
    adattare il modello non contano come prove d'innocenza."""
    esito = acorr_ljungbox(residui, lags=[lags], model_df=p + q, return_df=True)
    return esito["lb_pvalue"].iloc[0]

for n in (600, 2000):
    rng = np.random.default_rng(0)
    serie = processo.generate_sample(nsample=n, distrvs=rng.standard_normal)
    classifica = scegli(serie)
    posto = [i for i, (_, p, q) in enumerate(classifica) if (p, q) == (2, 1)][0]
    print(f"\ncon {n} osservazioni (il vero modello è ARMA(2,1)):")
    for i, (aic, p, q) in enumerate(classifica[:3]):
        print(f"   ARMA({p},{q})   AIC = {aic:8.1f}   (+{aic - classifica[0][0]:.1f})")
    aic, p, q = classifica[posto]
    print(f"   il VERO ARMA(2,1) è {posto + 1}° a +{aic - classifica[0][0]:.1f}")

    p, q = classifica[0][1], classifica[0][2]
    residui = ARIMA(serie, order=(p, 0, q)).fit().resid
    pv = ljung_box(residui, p, q)
    print(f"   Ljung-Box sul modello scelto: p = {pv:.3f}  ->  "
          f"{'nessuna traccia di struttura residua' if pv > 0.05 else 'resta struttura'}")

# e su un modello deliberatamente troppo povero? (nessun parametro: model_df=0)
pv = ljung_box(ARIMA(serie, order=(0, 0, 0)).fit().resid, 0, 0)
print(f"\nLjung-Box su un modello vuoto (0,0,0): p = {pv:.1e}  ->  resta struttura")
```

Il risultato è più istruttivo di quello che ci si aspetterebbe.

Con **600 osservazioni l'AIC sbaglia**: sceglie un ARMA(1,1) invece del vero
ARMA(2,1). Ma guarda i margini. Il secondo classificato è a $+0{,}4$ dal
primo, il terzo a $+1{,}5$, e il vero ARMA(2,1) arriva quinto a $+1{,}8$: dal
primo all'ultimo di questi, tutti stanno dentro le due unità sotto le quali,
come si è appena detto, l'AIC non distingue niente. L'AIC non ha **scartato** il
modello vero, l'ha messo nella stessa nuvola d'indifferenza degli altri, il che
con seicento osservazioni è la verità.

Il Ljung-Box sul modello scelto dà $p = 0{,}514$: il test, su questi dati, non
trova traccia di struttura residua. Che non è la stessa cosa che dimostrare
l'assenza, come si è avvertito poco fa, ma è tutto quello che una diagnostica
può dare. Il modello «sbagliato» va benissimo, ed è la lezione centrale:
l'obiettivo non è indovinare il modello vero (che sulle serie reali non
esiste), è trovarne uno che non lasci struttura fuori.

Con 2000 osservazioni l'AIC trova l'ordine giusto, e il secondo classificato è
ancora lì a $+0{,}6$. Verrebbe da concludere che basti avere più dati, e sarebbe
una conclusione affrettata, perché quello è un colpo riuscito, uno solo.
Rilanciando lo stesso esperimento venti volte, cambiando ogni volta soltanto il
numero da cui parte il generatore di numeri casuali (il **seme**: è quello che
decide quale, fra le infinite serie che quel modello può produrre, esce
davvero), l'ordine esatto salta fuori una volta su venti con seicento
osservazioni e cinque volte su venti con duemila. Più dati aiutano, quindi, e si
vede; ma non bastano affatto, perché anche con duemila osservazioni l'AIC manca
il modello vero tre volte su quattro.

Non è un difetto dell'AIC: è che l'AIC è fatto per scegliere il modello che
prevede meglio, non per indovinare quello che ha generato i dati, e quando due
modelli spiegano i dati quasi ugualmente bene i due obiettivi non coincidono. A
puntare sull'identificazione è semmai il **BIC**, un parente stretto che
penalizza i parametri tanto più severamente quante più osservazioni ci sono. La
lezione che invece tiene su tutti i semi e a tutte e due le numerosità è
un'altra: il Ljung-Box non rifiuta mai, e il $p$-value più basso osservato in
quaranta esperimenti è $0{,}13$.

Sul modello vuoto, infine, il $p$-value crolla a zero. «Vuoto» vuol dire
letteralmente questo: un ARMA($0,0$) non guarda nessun valore passato e nessun
urto passato, dice soltanto che la serie balla a caso attorno alla propria
media. Su una serie che una memoria invece ce l'ha, quello che avanza è tutta la
memoria, e il test la vede benissimo: è esattamente il suo mestiere.

Un'ultima avvertenza, che la sezione seguente riprenderà da capo: anche
**scegliere** $p$ e $q$ è un modo di usare i dati. Qui le sedici combinazioni
sono state provate su tutta la serie, dal primo giorno all'ultimo. Se adesso
misurassimo quanto sbaglia il modello scelto su quegli stessi giorni, il numero
verrebbe più bello del vero, perché il modello non è stato scelto a occhi
chiusi: è stato scelto avendo già visto anche i giorni che avrebbero dovuto
fargli da esame.

Con l'AIC il danno è piccolo, perché l'AIC non promette di dire quanto il
modello sbaglierà su giorni nuovi: dichiara solo quanto aderisce a quelli che ha
già visto, penalità compresa. Diventa grave quando a scegliere è l'errore
misurato su un pezzo di serie tenuto da parte, che è il meccanismo costruito
nella sezione seguente: lì la scelta va fatta **dentro** quel meccanismo, non
prima.

## Il mondo entra nella serie: SARIMAX e VAR

Fin qui la serie ha spiegato sé stessa con il proprio passato. Ma il gelataio
sa che le vendite dipendono dal meteo, e il negoziante che dipendono dalle
promozioni. Come entrano, quelle informazioni?

`````{tab} Elementare

Le strade sono due, e rispondono a due situazioni diverse.

La prima: hai **una serie da prevedere** e altre informazioni che la
influenzano ma che non ti interessa prevedere (la temperatura, i giorni di
festa, il prezzo di listino). Quelle si chiamano variabili **esogene**, cioè
«che vengono da fuori», e si aggiungono al modello come contributi che si
sommano a quello che la serie già spiega da sola. La sigla diventa SARIMAX, e
la X finale sta proprio per quelle variabili esterne.

C'è una trappola che si scopre sempre troppo tardi. Per prevedere le vendite di
domani con il meteo, ti serve il meteo **di domani**, che non hai. Quindi o è
una cosa che si conosce in anticipo per
costruzione (il calendario, i giorni di chiusura, una promozione già decisa),
oppure va prevista a sua volta, e allora nella previsione finale entrano due
errori invece di uno. Le variabili esogene che aiutano davvero sono quasi
sempre quelle del primo tipo.

La seconda strada: hai **più serie che si influenzano a vicenda** e vuoi
prevederle tutte insieme (il reddito e i consumi, la domanda e il prezzo). Qui
non c'è una principale e delle comparse: ognuna dipende dal proprio passato e
da quello delle altre. È il modello VAR, e la V sta per «vettoriale», che qui
vuol dire solo che al posto di un numero per volta il modello tratta una fila
di numeri per volta, una casella per ciascuna serie.

Il VAR conviene a una condizione: che quelle serie **si aiutino davvero** a
prevedersi. Se non lo fanno, il modello resta lecito, ma non compra previsioni
migliori, e quei parametri, che crescono col quadrato del numero di serie, li
paghi lo stesso.

Esiste un test per verificarlo, e prende il nome dall'economista Clive Granger.
Il nome, però, è la cosa più sbagliata che ha: si dice «causalità di Granger»,
e non dice affatto che una serie **fa succedere** l'altra. Dice solo che il suo
passato **aiuta a indovinarla**. Se il gelato e i condizionatori salgono
insieme d'estate, ciascuno dei due «prevede» l'altro benissimo, ma a farli
salire è il caldo, che non è né l'uno né l'altro. È un'affermazione sui dati,
non sul mondo, e chi la porta fuori da qui come se fosse una prova di causa fa
un errore che è costato mezzo secolo di equivoci.

`````

`````{tab} Superiore

**SARIMAX** aggiunge al SARIMA una componente di regressione su $r$ variabili
esogene $\mathbf{z}_t$:

$$
x_t = \boldsymbol{\beta}^\top \mathbf{z}_t + \eta_t ,
$$

dove $\eta_t$ segue un SARIMA($p,d,q$)($P,D,Q$)$_m$. Si legge bene così: una
regressione ordinaria il cui **errore non è indipendente** ma ha esso stesso
una struttura temporale, il che è precisamente la ragione per cui una
regressione lineare ordinaria su dati temporali dà coefficienti con errori
standard sbagliati.

Il vincolo operativo da tenere a mente: per una previsione a orizzonte $h$
servono i valori $\mathbf{z}_{T+1},\dots,\mathbf{z}_{T+h}$. O sono **noti per
costruzione** (calendario, festività, promozioni pianificate), o vanno previsti,
e la loro incertezza si propaga a quella finale senza che gli intervalli
standard ne tengano conto.

Il modello **VAR($p$)** tratta invece $N$ serie come un vettore
$\mathbf{x}_t \in \mathbb{R}^N$ (nel capitolo $N$ è sempre il numero di serie
osservate insieme, e $n$ resta il numero di osservazioni di una serie; la
lettera $d$ è già impegnata per l'ordine di differenziazione):

$$
\mathbf{x}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{x}_{t-1} + \dots +
\mathbf{A}_p \mathbf{x}_{t-p} + \boldsymbol{\varepsilon}_t ,
$$

con $\mathbf{A}_i$ matrici $N \times N$. La condizione di stazionarietà è la
sorella multivariata di quella dell'AR: le radici di
$\det(\mathbf{I} - \mathbf{A}_1 z - \dots - \mathbf{A}_p z^p)$ devono stare
fuori dal cerchio unitario. Il numero di parametri cresce come $pN^2$ (più le
$N$ intercette e le $N(N+1)/2$ della covarianza delle innovazioni), il che
spiega perché il VAR sia praticabile su poche serie e diventi subito ingestibile
su molte.

Il **test di causalità di Granger** verifica se i ritardi di una serie
migliorano significativamente la previsione di un'altra rispetto ai soli
ritardi di quest'ultima: è un test $F$ fra due regressioni annidate, l'ipotesi
nulla è che i coefficienti aggiuntivi siano tutti nulli, e richiede serie
stazionarie. Va fatto in entrambe le direzioni, perché è asimmetrico. E va letto
sapendo che il verdetto dipende da **che cos'altro c'è nella regressione**: una
Granger-causalità fra due serie può sparire appena se ne aggiunge una terza, ed
è il caso più frequente sui dati veri. Dipende anche dal numero di ritardi che
si è scelto.

Un chiarimento su cosa il test *non* è: non è un test di **validità** del VAR.
Un VAR le cui matrici $\mathbf{A}_i$ risultano (blocco-)diagonali è un modello
perfettamente ben specificato, che si riduce a tanti AR univariati e che
continua ad aggiungere qualcosa rispetto a stimarli separatamente ogni volta
che le innovazioni $\boldsymbol{\varepsilon}_t$ sono correlate
contemporaneamente (per gli intervalli congiunti, e per le funzioni di risposta
d'impulso). Il test di Granger è un test di **esclusione** su un blocco di
coefficienti: risponde a «i ritardi incrociati servono?», non a «il VAR è
lecito?».

E qui va detto forte, perché il nome ha prodotto mezzo secolo di equivoci: la
**causalità di Granger non è causalità**. È **precedenza predittiva**, e
soltanto quella. Due serie guidate da una terza causa comune non osservata si
«Granger-causano» a vicenda allegramente; e una causa vera che agisce più in
fretta del passo di campionamento non viene rilevata affatto. Il test dice «il
passato di $A$ aiuta a prevedere $B$», che è un'affermazione sui dati, non sul
mondo. La **scala della causalità** di Judea Pearl, quella del capitolo di
matematica, serve esattamente a tenere separate queste due cose: il gradino su
cui vive un test di Granger è il primo, quello delle associazioni fra dati
osservati.

`````

## Lisciamento esponenziale: da SES a Holt-Winters

La famiglia ARIMA dichiara la memoria della serie a voce alta: tanti valori
passati, tanti urti passati, un coefficiente per ciascuno. C'è una seconda
famiglia, altrettanto classica, che non dichiara niente e ottiene lo stesso
effetto con molto meno: il **lisciamento esponenziale** (*exponential
smoothing*). Al posto dei coefficienti ha una media, e la fa su tutto il
passato in una volta sola.

`````{tab} Elementare

Per indovinare le vendite di domani potresti fare la media di tutti i giorni
passati. Ma il mese scorso conta davvero quanto ieri? No. Il lisciamento
esponenziale fa una media *pesata*, in cui ieri pesa molto, l'altro ieri un po’
meno, la settimana scorsa ancora meno, e così via a scendere. A ogni passo
indietro il peso si riduce di una stessa frazione, come un ricordo che sbiadisce
sempre allo stesso ritmo: nitido ieri, sfocato la settimana scorsa, quasi niente
un anno fa.

Il conto non chiede di tenere il registro di tutti i giorni passati: basta un
numero, la stima di ieri, che ogni sera si sposta un poco verso quello che è
appena successo. Quanto poco lo decide una manopola. Girata tutta da un lato, il
metodo insegue ogni sussulto e dimentica in fretta; tutta dall'altro, è lento a
cambiare idea. Messa a 30 su 100, la manopola dà a ieri un peso di 30,
all'altro ieri di 21, al giorno prima di quasi 15, e così a scendere.

La versione base tiene conto solo del **livello** (dove sta la serie ora). Ma se
la serie sale con costanza, ti serve anche una stima di *quanto* sale: aggiungi
il **trend**. E se ha un respiro stagionale, aggiungi anche quello. Sono i tre
gradini: livello, poi livello + trend, poi livello + trend + stagionalità. Con
tutti e tre, il metodo si chiama Holt-Winters, dai nomi dei due che lo misero a
punto fra il 1957 e il 1960: Charles Holt per il livello e la tendenza, Peter
Winters per la stagione.

`````

`````{tab} Superiore

Il **lisciamento esponenziale semplice** (SES) tiene solo il livello $\ell_t$ e
lo aggiorna a ogni passo come media pesata fra l'osservazione nuova e la stima
vecchia:

$$
\ell_t = \alpha\, x_t + (1-\alpha)\,\ell_{t-1},
\qquad
\hat{x}_{t+1} = \ell_t,
$$

con $\alpha \in (0,1)$ il fattore di lisciamento. Srotolando la ricorsione
fino all'inizio della serie,
$\ell_t = \alpha \sum_{j=0}^{t-1} (1-\alpha)^j x_{t-j} + (1-\alpha)^t \ell_0$:
i pesi $\alpha(1-\alpha)^j$ **decadono esponenzialmente** e, per $t$ grande
(quando il peso residuo dell'inizializzazione $\ell_0$ è ormai trascurabile),
la loro somma tende a $1$. Con $\alpha = 0{,}3$ valgono
$0{,}30,\ 0{,}21,\ 0{,}147,\ \dots$

Il metodo di **Holt** aggiunge una componente di trend $b_t$:

$$
\ell_t = \alpha\, x_t + (1-\alpha)(\ell_{t-1} + b_{t-1}),
\qquad
b_t = \beta\,(\ell_t - \ell_{t-1}) + (1-\beta)\,b_{t-1},
$$

con previsione a $h$ passi $\hat{x}_{t+h} = \ell_t + h\,b_t$. **Holt-Winters**
aggiunge infine la stagionalità $s_t$ di periodo $m$ (forma additiva):

$$
\begin{aligned}
\ell_t &= \alpha\,(x_t - s_{t-m}) + (1-\alpha)(\ell_{t-1}+b_{t-1}), \\
b_t &= \beta\,(\ell_t - \ell_{t-1}) + (1-\beta)\,b_{t-1}, \\
s_t &= \gamma\,(x_t - \ell_{t-1} - b_{t-1}) + (1-\gamma)\,s_{t-m},
\end{aligned}
\qquad
\hat{x}_{t+h} = \ell_t + h\,b_t + s_{t+h-m(k+1)},
$$

dove $k = \lfloor (h-1)/m \rfloor$: l'indice stagionale ricicla sempre
l'ultimo ciclo stimato, così anche oltre un periodo intero ($h > m$) la
previsione non riferisce mai stagioni non ancora osservate. I tre fattori
$\alpha,\beta,\gamma \in (0,1)$ regolano quanto in fretta livello,
trend e stagionalità si adeguano ai dati nuovi (qui $\gamma$ è un fattore di
lisciamento, non l'autocovarianza $\gamma(k)$ dell'ACF di poco fa: è la
notazione consolidata di questa famiglia, e le due cose non hanno niente a che
vedere). Questi metodi hanno una veste
moderna nei modelli **ETS** (*Error, Trend, Seasonal*) in forma spazio-stato,
che aggiungono un'interpretazione probabilistica e intervalli di previsione
{cite}`hyndman2021forecasting`.

`````

## Quando i classici bastano (o battono il deep learning)

Verrebbe da pensare che, con le reti neurali che il capitolo affronta più
avanti, questi modelli di mezzo secolo fa siano roba da manuale di storia. Non
è così, e conviene dire perché con onestà. La prova più citata sono le
**competizioni M** dell'introduzione al capitolo, quelle di Spyros Makridakis.
Il verdetto, ripetuto edizione dopo edizione, è scomodo per gli entusiasti: i
metodi statistici semplici (ARIMA, Holt-Winters, e loro medie) restano
difficilissimi da battere, e per molti anni hanno superato reti neurali ben
più complesse.

Le ragioni sono tre. La **robustezza**: un modello con pochi parametri ha poco
spazio per rincorrere il rumore, e rincorrere il rumore è il modo migliore di
sbagliare sul futuro, perché il rumore di domani non è quello di ieri. Quel poco
che impara, allora, ha buone probabilità di valere anche sui giorni che non ha
visto, e questo vale pure quando la serie è corta o disturbata. La **frugalità
di dati**: gran parte delle serie reali (le vendite mensili di un prodotto, i
pazienti di un reparto) hanno poche decine o centinaia di osservazioni, troppo
poche per addestrare una rete affamata di dati, più che sufficienti per un
ARIMA. E l’**interpretabilità**. La frazione con cui il passato pesa sul futuro,
la componente stagionale, la forbice dentro cui il modello dichiara che cadrà il
valore vero (il filo rosso dell'introduzione al capitolo): sono oggetti che un
analista legge, discute e difende davanti a chi deve decidere. I numeri interni
di una rete neurale, che sono milioni e non vogliono dire niente presi uno per
uno, no {cite}`hyndman2021forecasting`. La
regola pratica che ne discende attraversa tutto il forecasting serio: un
modello classico è la **linea di base onesta**. Prima si batte quella, poi si
tira in ballo il deep learning.

## In pratica: stimare un AR(1) ai minimi quadrati

Stimare un AR(1) non richiede librerie sofisticate: è una regressione lineare di
$x_t$ sul suo ritardo $x_{t-1}$, cioè si cerca la retta che passa il più vicino
possibile a tutte le coppie (valore di ieri, valore di oggi). «Il più vicino
possibile» in che senso: nel senso che rende minima la somma dei quadrati degli
scarti, che è lo stesso criterio con cui il {doc}`capitolo sul Machine Learning </MachineLearning/overview>`
sceglieva la retta che passa meglio in mezzo ai dati. In statistica quel
criterio ha un nome, il metodo dei **minimi quadrati**. Generiamo una serie dal modello con una
frazione $\phi$ nota e
verifichiamo di saperla recuperare, poi facciamo una previsione a un passo.
Tutto in puro NumPy.

```python
import numpy as np

rng = np.random.default_rng(42)

# --- genera una serie dal modello AR(1): x_t = c + phi * x_{t-1} + rumore ---
phi_vero, c_vero, sigma = 0.6, 4.0, 1.0
n = 500
x = np.zeros(n)
x[0] = c_vero / (1 - phi_vero)                 # parte dalla media di lungo periodo (10)
for t in range(1, n):
    x[t] = c_vero + phi_vero * x[t - 1] + rng.normal(0, sigma)

# --- stima ai minimi quadrati: regredisci x_t su [1, x_{t-1}] ---
y = x[1:]                                       # bersaglio: x_t
Xmat = np.column_stack([np.ones(n - 1), x[:-1]])  # colonne: costante e x_{t-1}
beta, *_ = np.linalg.lstsq(Xmat, y, rcond=None)   # risolve i minimi quadrati
c_hat, phi_hat = beta

print(f"phi vero = {phi_vero:.2f}   phi stimato = {phi_hat:.3f}")
print(f"c vero   = {c_vero:.2f}   c stimato   = {c_hat:.3f}")

# --- previsione one-step dopo l'ultima osservazione ---
x_next = c_hat + phi_hat * x[-1]
print(f"ultima osservazione x_T = {x[-1]:.3f}")
print(f"previsione   x_(T+1)    = {x_next:.3f}")
```

Il $\phi$ stimato cade vicino a $0{,}6$ e la costante vicino a $4$: con
cinquecento osservazioni i minimi quadrati ricostruiscono bene i parametri del
processo che ha generato la serie. La previsione a un passo è semplicemente la
formula del modello applicata all'ultimo valore osservato. Da qui in avanti si
può ripetere il conto in avanti per prevedere più giorni (quanto lontano si
guarda si chiama **orizzonte**). Ricadendo, però, in un guaio: dal secondo
giorno in poi il conto non parte più da un valore osservato, parte da una
previsione, cioè da un numero che può già essere sbagliato, e quello sbaglio si
trascina fino in fondo. La sezione seguente lo riprende per esteso.

`````{tab} Elementare

Due cose, per non prendere il risultato per più di quel che è.

La prima riguarda il metodo. Tirare una retta in mezzo a punti che vengono da
giorni diversi è delicato, perché quei punti non sono indipendenti fra loro: si
tengono per mano. E il guaio che ne verrebbe è questo: la retta viene fuori
sensata, ma il conto che dice *quanto ci si può fidare* di quella retta assume
punti indipendenti, e allora dichiara una precisione che non c'è.

Qui il guaio non si presenta. In un AR(1) fatto come si deve, gli scarti fra il
valore vero e quello che la retta prevede *sono* le scosse casuali del modello,
e le scosse casuali fra loro per mano non si tengono. Si presenterebbe se in
quegli scarti restasse ancora una regolarità, ed è per questo che guardarli è il
passo che non si salta.

La seconda: con cinquecento osservazioni la stima è buona, con cinquanta lo è
molto meno. La prova con cinquecento dà $0{,}635$ contro un vero $0{,}6$, cioè
un po’ alto, ma una prova sola non dice niente sul metodo: dice cosa è capitato
questa volta.
È ripetendo l'esperimento tante volte che salta fuori il difetto vero, e il
difetto vero punta dalla parte opposta: la media delle stime cade **sotto** il
valore vero, e ci cade tanto più quanto la serie è corta. Il colpevole è il modo
stesso di fare il conto: il valore di ieri, quello che facciamo da guida, porta
già dentro la scossa di ieri. Guida e scossa non sono estranee, e la retta che
ne esce viene un filo più piatta di quella vera. Molte serie reali sono corte, e
chi legge quel numero deve saperlo.

`````

`````{tab} Superiore

Due precisazioni, perché il numero è giusto ma il metodo ha un limite che va
detto proprio nel regime in cui vive gran parte delle serie reali.

La prima: l'avvertimento sul SARIMAX («una regressione lineare
ordinaria su dati temporali dà coefficienti con errori standard sbagliati») qui
non morde, perché in un AR(1) ben specificato l'errore è rumore bianco per
costruzione, ed è quando l'errore ha struttura che gli errori standard ordinari
diventano inaffidabili.

La seconda: regredire su un valore **ritardato della stessa serie** non è una
regressione ordinaria fino in fondo, perché il regressore non è indipendente
dall'errore passato, cioè viene meno l'esogeneità stretta. La stima resta
consistente, ma in campione finito è **distorta verso lo zero**, di circa
$(1+3\phi)/n$: con cinquecento osservazioni sono sei millesimi e non si vedono,
con cinquanta sono sei centesimi, cioè il 10% del valore vero. Ripetendo questo
stesso codice ventimila volte si misurano $-0{,}005$ e $-0{,}059$, che è quanto
la formula prevede.

Il punto non è che la distorsione superi l'errore standard, che a cinquanta
osservazioni resta il doppio ($0{,}12$). Il punto è che l'errore standard, a
forza di ripetere la misura, si media via, e la distorsione no, perché punta
sempre dalla stessa parte: passando da cinquecento a cinquanta osservazioni
cresce da un sesto dell'errore standard alla sua metà.

`````

Con questo abbiamo la cassetta degli attrezzi classica: decomposizione per
capire, ARIMA e Holt-Winters per prevedere, ACF e PACF per diagnosticare. La
sezione successiva affronta la domanda che finora abbiamo aggirato, cioè come si
**valida** un modello di serie temporale senza barare col futuro. E poi come si
trasformano le serie in colonne di una tabella, per darle in pasto ai modelli
tabellari già incontrati nel {doc}`capitolo sul Machine Learning </MachineLearning/overview>`.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **Scomporre** una serie vuol dire leggerla come la bolletta della luce: il
  **canone** di fondo (il trend), la **stagione** che torna ogni anno uguale
  (la stagionalità) e l’**imprevisto** che non segue regole (il residuo). La
  stagione può aggiungere sempre la stessa cifra (caso **additivo**) oppure una
  percentuale, e allora cresce insieme al giro d'affari (caso
  **moltiplicativo**): in gelateria, «d'estate 35 mila euro in più» contro
  «d'estate il $78\%$ in più».
- Quasi tutti i modelli classici pretendono una serie **stazionaria**, che balli
  sempre allo stesso modo, cioè attorno alla stessa media, con la stessa
  ampiezza, e in cui due giorni si somiglino in base a **quanto** distano e non
  a **quando** cadono. Per arrivarci ci sono due strade diverse, e vanno tenute
  distinte: se la serie oscilla attorno a una **retta** si stima la retta e si
  tengono gli scarti; se invece cammina alla cieca, e ogni scossa le sposta il
  livello per sempre, si sostituisce ogni valore con la **variazione** rispetto
  al precedente. Usare la seconda al posto della prima non è gratis: lascia
  dentro la serie una regolarità che non c'era. Per capire che memoria resta si
  guardano due grafici a barre: l’**ACF**
  (la funzione di autocorrelazione), quanto oggi assomiglia ai giorni passati, e
  la **PACF** (l'autocorrelazione parziale), quanto ci assomiglia al netto degli
  effetti a catena (il nonno e il nipote, scontato il padre). Si leggono in
  coppia, perché nessuno dei due da solo dice tutto: quando la memoria è di un
  tipo solo, quella dei valori passati si riconosce dalla PACF che si schiaccia
  di colpo, quella degli urti dall'ACF che si schiaccia di colpo, e dove si
  schiaccia è lì che quella memoria finisce.
- Ci sono due memorie. Quella dei **valori** passati (l'autoregressione: domani
  somiglia a oggi, con un rientro verso la media) e quella degli **urti**
  passati (la media mobile: lo sciopero si fa sentire ancora domani, meno
  dopodomani). **ARIMA** le usa insieme su una serie già raddrizzata, e dietro
  la sigla ci sono solo tre conteggi; **SARIMA** rifà lo stesso gioco sul
  calendario, confrontando dicembre con lo scorso dicembre
  {cite}`box2015time`.
- Gli ordini non si indovinano guardando i grafici: **si provano tutte le
  combinazioni** e si sceglie con un criterio che pesa insieme quanto il
  modello spiega e quanti parametri ha speso (l’**AIC**). Poi, e questo è il
  passo che quasi tutti saltano, **si guarda quello che resta**: se negli
  errori si vede ancora una regolarità, il modello se l'è lasciata sfuggire.
  Il modello sbagliato si riconosce dai suoi errori, non dalle sue previsioni.
- Le informazioni esterne entrano in due modi. Come variabili **esogene** in un
  SARIMAX (il meteo, le promozioni), con la trappola che per prevedere domani
  serve il loro valore di domani; oppure, se più serie si influenzano a
  vicenda, prevedendole tutte insieme con un **VAR**, che però conviene solo se
  quelle serie si aiutano davvero a prevedersi. Lo verifica il test di
  **Granger**, e il suo nome contiene la trappola più famosa del capitolo: la
  «causalità di Granger» non è causalità. Dice che il passato di una serie aiuta
  a indovinarne un'altra, non che la faccia succedere. Gelato e condizionatori
  si prevedono a vicenda benissimo, ma a farli salire è il caldo.
- Il **lisciamento esponenziale** è una media del passato in cui ieri pesa
  molto e ogni passo indietro pesa una frazione in meno, come un ricordo che
  sbiadisce. Tre gradini: solo il livello, poi livello più tendenza, poi
  anche la stagione, e con tutti e tre il metodo si chiama Holt-Winters.
- I classici sono **robusti, si accontentano di poche osservazioni e si
  spiegano a chi deve decidere**: nelle competizioni M restano una **linea di
  base** durissima da battere. Prima si supera quella, poi si tira in ballo il
  deep learning {cite}`hyndman2021forecasting`.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- La **decomposizione** separa una serie in **trend**, **stagionalità** e
  **residuo**, in forma **additiva** ($x_t = T_t + S_t + R_t$, oscillazioni di
  ampiezza costante) o **moltiplicativa** ($x_t = T_t \times S_t \times R_t$,
  ampiezza proporzionale al livello, linearizzabile col logaritmo).
- Quasi tutti i modelli classici richiedono la **stazionarietà**, e lo strumento
  dipende dal tipo di non stazionarietà: **differenziare**
  ($\nabla x_t = x_t - x_{t-1}$) contro un trend **stocastico**,
  **detrendizzare** contro un trend **deterministico**. Scambiarle
  sovradifferenzia, cioè lascia $\rho_1 = -0{,}5$ e varianza doppia, con
  $\theta$ inchiodato sul bordo dell'invertibilità. Si testa con ADF e KPSS, che
  hanno ipotesi nulle **opposte**. **ACF** e **PACF** diagnosticano gli ordini
  sui processi **puri**: la PACF si annulla dopo il ritardo $p$ di un AR, l'ACF
  dopo il ritardo $q$ di un MA, e «annullarsi» vuol dire cadere dentro
  $\pm 1{,}96/\sqrt{n}$, che è una banda **puntuale**; su un ARMA non si annulla
  nessuna delle due, e gli ordini si scelgono con una griglia.
- **AR($p$)** spiega il valore con i $p$ passati; **MA($q$)** con i $q$ errori
  passati; **ARIMA($p,d,q$)** unisce i due sulla serie differenziata $d$ volte,
  e **SARIMA** aggiunge i termini stagionali al ritardo $m$ {cite}`box2015time`.
- La **procedura** è in tre tempi: stazionarizzare (fissando $d$), scegliere
  $(p,q)$ minimizzando l’**AIC** $= 2k - 2\ln\hat L$ su una griglia, verificare
  che i residui siano **rumore bianco** con il Q-Q plot e il test di
  **Ljung-Box**, calcolato con $\ell - (p+q)$ gradi di libertà: ometterlo gonfia
  sempre il $p$-value. Attenzione al verso del test: qui si spera di **non**
  rifiutare (il contrario dell'ADF), e l'AIC è una quantità **relativa** e
  confrontabile solo **a parità di dati**, il che è la ragione per cui $d$ e le
  trasformazioni non entrano nella griglia.
- **SARIMAX** aggiunge variabili **esogene** (una regressione il cui errore ha
  a sua volta struttura temporale), al prezzo di doverne conoscere i valori
  futuri. **VAR($p$)** modella $N$ serie insieme, con $pN^2$ parametri: se i
  **ritardi incrociati** non aiutano, il modello resta lecito (e con innovazioni
  correlate contemporaneamente aggiunge ancora qualcosa), ma quei parametri si
  pagano lo stesso. Che i ritardi incrociati servano lo dice il test di
  **Granger**, che è un test di **esclusione** su quei coefficienti, non di
  validità del modello, e che misura **precedenza predittiva**, non causalità.
- Il **lisciamento esponenziale** pesa il passato con pesi che **decadono
  esponenzialmente**: SES (solo livello), Holt (livello + trend), Holt-Winters
  (livello + trend + stagionalità).
- I modelli classici sono **robusti, frugali di dati e interpretabili**: nelle
  competizioni M restano una **linea di base** durissima da battere. Prima si
  supera quella, poi si passa al deep learning {cite}`hyndman2021forecasting`.
```

`````

[^senso-debole]: Quella del testo è la stazionarietà detta *in senso debole*, e si chiama così perché
    guarda solo la media, l'ampiezza delle oscillazioni e le somiglianze a due
    a due (e perché abbia senso chiederlo serve che quelle quantità esistano,
    cioè $\mathbb{E}[X_t^2] < \infty$). La versione forte chiede di più: che
    presa una qualunque manciata di istanti, la loro distribuzione congiunta
    non cambi se si sposta tutta in avanti nel tempo. Nella pratica non si usa
    quasi mai, e non contiene l'altra: un processo a code pesantissime può
    essere stazionario in senso forte senza avere una varianza da tenere
    costante.
