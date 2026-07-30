# La forchetta che non c'era

Apparecchia mezzo tavolo: un piatto, un coltello alla sua destra, un bicchiere.
Fotografa e chiedi a un modello che vede e parla che cosa c'è sul tavolo. È
probabile che nella risposta compaia anche una forchetta, magari «a sinistra del
piatto», con la stessa calma con cui il modello ha nominato il bicchiere. La
forchetta non c'è, e non c'è mai stata.

Il primo istinto è chiamarlo un errore di percezione, come una macchia sul
sensore o un riflesso scambiato per un oggetto. È la lettura sbagliata, e
sbagliarla porta a cercare la soluzione nel posto sbagliato. Il modello non ha
visto male: ha *scritto bene*. Nelle didascalie del mondo, accanto a un piatto e
a un coltello, una forchetta c'è quasi sempre, e chi è addestrato a continuare
frasi plausibili la scrive perché la frase, senza, sarebbe meno plausibile. È il
**priore linguistico** (il *prior* del modello di linguaggio, quello che
l'overview ha già nominato) che vince sull'evidenza visiva, e questa sezione mostra
perché il meccanismo lo renda inevitabile, come si fa a misurarlo, e che cosa si
può fare per contenerlo.

## Un errore che non riguarda gli occhi

L'overview del capitolo ha già dato il nome al fenomeno e ne ha mostrato la
radice nella funzione di costo. Vale la pena scavare un poco più a fondo, perché
la forma precisa dell'argomento dice anche dove si può intervenire.

`````{tab} Elementare

Pensa alla tastiera del telefono, quella che dopo «a domani e buona» propone
«serata». Non sa che cosa stai per dire: sa come vanno a finire le frasi, e
sbaglia così di rado che smettiamo di accorgercene. Un modello che descrive una
fotografia fa esattamente quel mestiere, con una differenza sola: fra i
suggerimenti che gli arrivano c'è anche l'immagine. L'immagine però è un
suggerimento, non un padrone; se è sfocata, se l'angolo del tavolo è tagliato
fuori, se la forchetta ci starebbe benissimo, il suggerimento debole perde e
vince l'abitudine.

Facciamo un conto ipotetico, per capire quanto sia conveniente l'abitudine.
Mettiamo che su cento fotografie in cui compare un coltello, in ottanta ci sia
anche una forchetta. Un modello che non guarda affatto e risponde sempre «sì, c'è
la forchetta» ne azzecca ottanta su cento. Per battere quel punteggio guardando
davvero bisogna fare meglio dell'ottanta per cento, e nessuno gliel'ha chiesto:
l'addestramento premia la risposta media giusta, e la risposta media giusta si
ottiene anche a occhi chiusi. Guardare non è vietato, è semplicemente facoltativo.

`````

`````{tab} Superiore

Un modello con connettore {cite}`liu2023visual` ottimizza la cross-entropia
autoregressiva
$\mathcal{L}(\theta) = -\sum_t \log p_\theta\big(y_t \mid y_{<t}, E(I)\big)$,
dove $y_t$ è il token al passo $t$, $I$ l'immagine ed $E$ l'encoder visivo con
il suo connettore. Il termine dentro il logaritmo si scompone in modo
istruttivo:

$$
\log p_\theta\big(y_t \mid y_{<t}, E(I)\big) =
\underbrace{\log p_\theta\big(y_t \mid y_{<t}\big)}_{\text{priore linguistico}} +
\underbrace{\log \frac{p_\theta\big(y_t \mid y_{<t}, E(I)\big)}{p_\theta\big(y_t \mid y_{<t}\big)}}_{\text{contributo visivo}},
$$

dove il primo addendo è ciò che il modello direbbe a occhi chiusi (la
distribuzione del solo testo, quella che si ottiene marginalizzando l'immagine)
e il secondo è la **mutua informazione puntuale** fra il token e l'immagine,
dato il prefisso.
La riduzione della perdita che il condizionamento sull'immagine può al più
produrre è la mutua informazione condizionata $\mathcal{I}(Y_t; I \mid Y_{<t})$:
dove la didascalia è già prevedibile dal solo testo, quella quantità è piccola,
e con essa il gradiente che spinge il percorso visivo a servire a qualcosa.

Il punto di partenza dell'ottimizzazione aggrava lo sbilanciamento. Il priore
arriva già formato da un pre-addestramento testuale enormemente più lungo,
mentre il connettore è inizializzato a caso e vale, come si è visto, qualche
milione di parametri contro i miliardi del modello di linguaggio. All'iterazione
zero ignorare l'immagine è un ottimo locale, e la discesa del gradiente non ha
alcun motivo di uscirne se non nella misura in cui i dati la costringono.

C'è infine un contributo che nasce nei dati stessi. I corpora di istruzione
visiva generati da un modello di solo testo a partire da didascalie e riquadri
(la ricetta discussa nella sezione sui connettori) contengono, per costruzione,
affermazioni che il generatore non poteva verificare: il bersaglio della
massima verosimiglianza è già popolato di forchette inventate.

`````

È esattamente la radice delle allucinazioni testuali del capitolo sui
Transformer, e quelle «risposte sicure di sé e sbagliate» che il capitolo
sull'MLOps mette fra i bersagli del monitoraggio. Con un'aggravante e
un'attenuante. L'aggravante è che qui una fonte di verità c'era, allegata alla
richiesta, e il modello l'ha ignorata. L'attenuante, se così si può chiamare, è
che proprio perché quella fonte esiste il fenomeno è **misurabile**: su una
domanda di storia bisogna andare a controllare i libri, su una fotografia la
risposta giusta è nella fotografia.

## Il guaio di correggere un tema

Misurabile in linea di principio, però, non vuol dire facile. Chiedi a un
modello «descrivi questa immagine», ottieni sei righe, e prova a dare un voto.
Quale parola è sbagliata? Se il testo dice «un tavolo apparecchiato con piatto,
coltello e forchetta, pronto per il pranzo», l'errore è una parola su dodici, ma
per accorgersene bisogna prima decidere quali parole sono affermazioni sul mondo
e poi verificarle una a una.

La via classica esiste: si fissa un elenco chiuso di categorie di oggetti, si
cercano nel testo generato quelle parole (con una tabella di sinonimi e di
plurali), e si conta la frazione di oggetti nominati che nell'immagine non sono
annotati. Funziona, è stata la prima misura del campo, e porta con sé tre
fragilità che non si possono togliere. Vede solo gli oggetti dell'elenco: un
colore sbagliato, un conteggio sbagliato, una relazione spaziale rovesciata sono
invisibili. Dipende dalla completezza delle annotazioni: un oggetto che c'è
davvero ma che nessuno ha annotato viene contato come allucinazione. E l'analisi
del testo libero resta un'euristica, che inciampa sulle negazioni («non c'è
nessuna forchetta» contiene la parola «forchetta») e sui riferimenti generici.

Il risultato non è una misura sbagliata: è una misura **rumorosa in una
direzione che non si controlla**, e cioè la cosa peggiore che si possa avere in
mano quando si vuole stabilire se un fenomeno esista. La via d'uscita non è un
analizzatore migliore. È cambiare la domanda.

## Una domanda con due sole risposte

L'impostazione che ha reso il problema trattabile è quella di POPE
{cite}`li2023evaluating`, *Polling-based Object Probing Evaluation*: non si
chiede più al modello di descrivere, gli si chiede «c'è una forchetta in questa
immagine?» e si accetta solo sì o no. La risposta è un token, la verità sta
nelle annotazioni, e nessun giudice deve interpretare niente; è la ragione per
cui il protocollo esiste, perché
l'alternativa (un modello che fa da correttore, l'*LLM-as-a-judge* del capitolo
sull'MLOps) porta in dote i propri difetti proprio là dove si vuole misurare un
difetto.

Il cuore del metodo, però, non è il formato binario: è **come si scelgono gli
oggetti assenti**. Chiedere «c'è una zebra?» davanti a una cucina non misura
niente.

`````{tab} Elementare

Un esame a crocette si fa facile o difficile scegliendo le risposte sbagliate da
mettere accanto a quella giusta. Qui è lo stesso, e i modi sono tre.

Il primo: peschi un oggetto a caso dall'elenco del mondo. «C'è una zebra?»
davanti a un tavolo apparecchiato è una domanda regalata: nessuna abitudine
spinge verso il sì.

Il secondo: peschi gli oggetti che nelle fotografie compaiono più spesso in
assoluto. «C'è una persona?» è già più insidiosa, perché nelle immagini raccolte
dal web una persona c'è quasi sempre, e il modello lo ha imparato.

Il terzo, il più cattivo: peschi l'oggetto che va di solito *insieme* a quelli
che ci sono davvero. Davanti al piatto e al coltello, «c'è una forchetta?». Qui
l'abitudine spinge con tutta la sua forza, ed è esattamente la spinta che
vogliamo misurare.

La cosa importante viene alla fine: non si guarda un punteggio, se ne guardano
tre, e si guarda quanto scendono passando dal primo modo al terzo. Quella
discesa non racconta quanto il modello sia bravo. Racconta a quale abitudine si
sta appoggiando.

`````

`````{tab} Superiore

Sia $\mathcal{O}$ l'insieme delle categorie annotate nel corpus e
$\mathcal{O}(I) \subseteq \mathcal{O}$ quelle presenti nell'immagine $I$. Le
domande positive si estraggono da $\mathcal{O}(I)$, quelle negative da
$\mathcal{O} \setminus \mathcal{O}(I)$ secondo tre distribuzioni:

$$
q_{\text{unif}}(o) \propto 1,
\qquad
q_{\text{freq}}(o) \propto \hat{p}(o),
\qquad
q_{\text{cooc}}(o) \propto \sum_{o' \in \mathcal{O}(I)} \hat{p}(o \mid o'),
$$

dove $\hat{p}(o)$ è la frequenza marginale della categoria $o$ nel corpus e
$\hat{p}(o \mid o')$ la sua frequenza condizionata alla presenza di $o'$; nelle
ultime due si prendono i primi $k$ candidati in ordine di punteggio anziché
campionare, con $k$ pari al numero di domande negative che tocca all'immagine.
Le tre condizioni non sono tre difficoltà arbitrarie: sono due
priori diversi messi alla prova separatamente, quello **marginale** e quello
**condizionato alla co-occorrenza**, più un controllo. Il divario fra le
accuratezze nelle tre condizioni è una stima di quanto ciascun priore stia
guidando la risposta.

Tre proprietà del disegno meritano di essere isolate, perché sono ciò che lo
rende una misura e non un sondaggio. L'insieme è **bilanciato**, metà domande
con risposta sì e metà con risposta no, così che entrambe le strategie
degeneri si collochino al livello del caso in accuratezza. La risposta è un
token, quindi il confronto con la verità è esatto e riproducibile. E accanto
alle metriche si riporta la **quota di sì**,
$\hat{\pi} = \frac{1}{n}\sum_{i} \mathbb{1}[\hat{y}_i = \text{sì}]$, che è il
vero strumento diagnostico: un $\hat{\pi}$ lontano da $0{,}5$ dice che il
modello non sta rispondendo alla domanda, sta esprimendo una disposizione.

`````

Perché la quota di sì non sia un ornamento si vede facendo i conti su un test
bilanciato di tremila domande, millecinquecento su oggetti presenti e
millecinquecento su oggetti assenti.

```python
import numpy as np

# Un test bilanciato: 1500 domande su oggetti presenti, 1500 su oggetti assenti.
verita = np.array([1] * 1500 + [0] * 1500)      # 1 = l'oggetto c'e' davvero


def pagella(risposte):
    vp = ((risposte == 1) & (verita == 1)).sum()   # dice si', e c'e'
    fp = ((risposte == 1) & (verita == 0)).sum()   # dice si', e non c'e'
    fn = ((risposte == 0) & (verita == 1)).sum()   # dice no, e invece c'e'
    precisione = vp / (vp + fp)
    richiamo = vp / (vp + fn)
    f1 = 2 * precisione * richiamo / (precisione + richiamo)
    return round(float(f1), 3), round(float((risposte == 1).mean()), 3)


def guarda_davvero(a):
    """Modello che risponde correttamente a una frazione a di ciascuna classe."""
    giuste = round(1500 * a)
    return np.concatenate([
        np.array([1] * giuste + [0] * (1500 - giuste)),      # sui presenti
        np.array([0] * giuste + [1] * (1500 - giuste)),      # sugli assenti
    ])


print(pagella(np.ones(3000, dtype=int)))   # (0.667, 1.0)  dice sempre "si'"
print(pagella(guarda_davvero(0.60)))       # (0.6, 0.5)    guarda, e sbaglia molto
print(pagella(guarda_davvero(0.90)))       # (0.9, 0.5)    guarda bene
```

Il modello che risponde sempre «sì» non guarda mai, sbaglia una domanda su due,
e porta a casa un F1 di $0{,}667$: perché non manca un solo oggetto presente
(richiamo $1$) e paga solo in precisione ($0{,}5$). Un modello che guarda
davvero ma sbaglia due volte su cinque prende $0{,}60$, cioè **meno**. A leggere
il solo F1 si metterebbe in classifica sopra a tutti un modello che
dell'immagine non ha usato un pixel. La quota di sì scioglie l'equivoco in un
colpo: $1{,}0$ contro $0{,}5$, e il primo dei due non sta rispondendo, sta
ripetendo sempre la stessa cosa.

Due onestà, per non trasformare un protocollo in un oracolo. La prima: si misura
l'**esistenza degli oggetti**, e nient'altro; un colore sbagliato, un conteggio
sbagliato, una relazione rovesciata restano invisibili. La seconda: poiché la
misura è pubblica e la strategia per migliorarla è nota, un modello istruito a
dire «no» più spesso guadagna punti senza aver guadagnato un grammo di vista, che
è la solita legge di Goodhart e vale qui come altrove. Del resto la ragione per
descrivere il metodo e non i punteggi è proprio questa: i punteggi sono cronaca,
il disegno dell'esperimento no.

## Il difetto viene da più a monte

Fin qui abbiamo trattato il priore linguistico come un concorrente troppo forte.
Ma c'è un secondo pezzo del meccanismo, e sta prima: a volte l'informazione che
avrebbe permesso di rispondere non è arrivata affatto.

`````{tab} Elementare

Immagina di dover distinguere due pacchi usando soltanto una bilancia. Uno è
pieno di libri, l'altro di piume, e per un caso sfortunato pesano uguale. La
bilancia dice «due chili» a tutti e due, e chiunque legga solo il numero non
potrà mai distinguerli: non perché sia distratto, ma perché nel numero la
differenza non c'è più. Se poi gli chiedi «in quale pacco ci sono i libri?»,
dovrà tirare a indovinare, e tirerà a indovinare secondo l'abitudine, perché non
ha altro.

L'encoder della prima sezione è quella bilancia. Si possono trovare, e si sono
trovate, coppie di fotografie che una persona distingue in mezzo secondo (un
animale girato a destra e lo stesso girato a sinistra, una scarpa allacciata e
la stessa slacciata) e che l'encoder misura quasi identiche: somiglianza
$0{,}96$ su una scala che arriva a $1$. Un secondo strumento, addestrato solo
sulle immagini e senza mai vedere una didascalia, mette le stesse due foto a
$0{,}5$: per lui sono due cose diverse. La differenza non sta nella fotografia,
sta nel righello.

Ed ecco il punto che chiude il cerchio: quando il righello non distingue, il
modello di linguaggio non risponde «non lo so». Riempie il buco con quello che
di solito è vero. Il punto cieco non produce silenzio, produce allucinazione.

`````

`````{tab} Superiore

Il lavoro di Tong e colleghi {cite}`tong2024eyes` costruisce **coppie cieche**
in modo operativo: due immagini $I_1, I_2$ tali che

$$
\big\langle E_{\text{CLIP}}(I_1), E_{\text{CLIP}}(I_2) \big\rangle > 0{,}95
\qquad\text{e}\qquad
\big\langle E_{\text{SSL}}(I_1), E_{\text{SSL}}(I_2) \big\rangle < 0{,}6,
$$

dove $E_{\text{CLIP}}$ è la torre visiva di un modello contrastivo
{cite}`radford2021learning`, $E_{\text{SSL}}$ un encoder auto-supervisionato di
sola visione, e i vettori sono normalizzati, così che il prodotto scalare sia un
coseno. La seconda condizione garantisce che la differenza esista nei pixel; la
prima, che sia scomparsa nell'embedding. Da queste coppie si ricavano domande a
cui una persona risponde quasi sempre correttamente, e gli errori si raggruppano
in nove famiglie ricorrenti: orientamento e direzione, presenza di un dettaglio,
stato e condizione di un oggetto, quantità e conteggio, posizione e relazione
spaziale, colore e aspetto, caratteristiche fisiche e strutturali, testo
scritto, punto di vista e prospettiva.

L'argomento per cui a valle non si recupera è breve. Il decoder vede soltanto
$Z = E(I)$, quindi la catena $I \to Z \to Y$ è markoviana e per qualunque
risposta $Y$ vale la disuguaglianza dell'elaborazione dei dati,
$\mathcal{I}(Y; I) \le \mathcal{I}(Z; I)$, dove $\mathcal{I}$ è la mutua
informazione: la risposta non può portare sull'immagine più informazione di
quanta l'encoder ne abbia fatta passare, per quanto si addestri ciò che viene
dopo. E in forma quantitativa, se il decoder $g$ è lipschitziano di costante
$L$, allora
$\lVert g(z_1) - g(z_2) \rVert \le L \lVert z_1 - z_2 \rVert$: con $z_1 \approx
z_2$ le due risposte sono costrette a somigliarsi, mentre le risposte corrette
sono opposte. Il modello deve rompere il pareggio, e l'unico strumento che gli
resta per farlo è il priore del blocco precedente. Punto cieco a monte e
allucinazione a valle non sono due difetti: sono un difetto e la sua
manifestazione.

`````

Conviene dire in modo esplicito che questo è **lo stesso limite della prima
sezione, visto dall'altro lato**. Là, dal lato del testo, la loss contrastiva
chiedeva solo di distinguere la didascalia vera da $N-1$ didascalie di immagini
prese a caso, e per vincere quel gioco bastava riconoscere gli oggetti: da qui
il comportamento a «sacco di concetti» misurato da Winoground. Qui, dal lato
dell'immagine, vale la conseguenza speculare: qualunque dimensione visiva che
non serva mai a quella discriminazione può essere collassata senza costo, e
l'ottimizzazione la collassa. L'orientamento, il conteggio, la presenza di un
particolare minuto sono precisamente le dimensioni da cui la didascalia di
un'altra fotografia non dipende mai. Un solo buco, due modi di infilarci il
dito.

Il rimedio a monte è coerente con la diagnosi: se un encoder contrastivo perde
ciò che le didascalie non nominano, gli si affianca un encoder auto-supervisionato
di sola visione e si uniscono le due rappresentazioni. Il come cambia il conto:
affiancare i canali di ogni token lascia la sequenza lunga uguale e allarga i
vettori, alternare nella sequenza i token delle due torri raddoppia il contesto
occupato (il conto della sezione sulla risoluzione). In tutti i casi restano due
encoder da far girare invece di uno, e il problema non è cancellato; è spostato.

## Tre rimedi, nessuna cura

Le contromisure che hanno un senso meccanico sono tre, e attaccano il fenomeno
in tre punti diversi della catena: i dati, la decodifica, l'uscita.

**Ancorare la risposta a ciò che si vede.** Invece di chiedere al modello *che
cosa* c'è, gli si chiede anche *dove*: il nome dell'oggetto accompagnato dalle
coordinate del riquadro che lo contiene, emesse come token. Il come è noto: si
normalizzano le coordinate in $[0,1]$, si taglia l'intervallo in un numero fisso
di gradini e si dà a ogni gradino un simbolo del vocabolario. È lo stesso gesto
già visto per i token d'immagine (un continuo tagliato in un numero finito di
simboli), applicato qui a una grandezza continua che non è l'immagine.
L'effetto interessante non è in uscita ma nel gradiente: «una
forchetta» il priore linguistico te lo regala, «una forchetta in $0{,}42$,
$0{,}31$, $0{,}55$, $0{,}60$» no, perché quei quattro numeri dall'abitudine non
si ricavano. Chiedere le coordinate rende l'affermazione verificabile a
posteriori e, in addestramento, rende il percorso visivo l'unica strada per
abbassare la perdita.

**Decodificare per differenza.** Il secondo rimedio non tocca i pesi: cambia
come si sceglie il token.

`````{tab} Elementare

Il trucco è fare la stessa domanda due volte, una a occhi aperti e una a occhi
chiusi (o guardando una versione dell'immagine rovinata di proposito), e tenere
solo la differenza.

Se «forchetta» risulta probabile in entrambi i casi, quella parola non viene
dalla foto: viene dall'abitudine, e allora la si penalizza. Se «coltello» è
probabile solo a occhi aperti, quella parola l'ha vista davvero, e la si premia.
In pratica si sottrae, punto per punto, quello che il modello direbbe comunque.

Una precauzione serve, altrimenti il trucco si rivolta: sottraendo senza freni
si finisce per premiare parole assurde, che a occhi chiusi erano
improbabilissime e a occhi aperti solo un po' meno. Il rimedio è restringere la
scelta in partenza alle parole che a occhi aperti valevano almeno un decimo
della più probabile: dentro quella rosa si confronta, fuori non si guarda. E il
conto da pagare è semplice: due letture invece di una, quindi il doppio del
tempo per ogni parola scritta.

`````

`````{tab} Superiore

Detti $\ell_\theta$ i logit del modello, $x$ il prompt testuale, $I$ l'immagine
e $I'$ una sua versione priva di informazione (nessuna immagine, oppure
un'immagine degradata con rumore), la decodifica contrastiva sceglie il token
successivo secondo

$$
\ell_{\text{cd}}(y_t) = (1 + \alpha)\,\ell_\theta\big(y_t \mid y_{<t}, x, I\big)
- \alpha\,\ell_\theta\big(y_t \mid y_{<t}, x, I'\big),
$$

ristretto all'insieme dei candidati plausibili

$$
\mathcal{V}_t = \Big\{ w \in V \;:\;
p_\theta\big(w \mid y_{<t}, x, I\big) \ge
\beta \max_{w' \in V} p_\theta\big(w' \mid y_{<t}, x, I\big) \Big\},
$$

dove $\alpha \ge 0$ regola la forza della correzione e $\beta \in (0,1)$ (in
pratica intorno a $0{,}1$) è la soglia di plausibilità che impedisce alla
sottrazione di promuovere token del tutto improbabili. Vale la pena notare che
la differenza dei due logit è, a meno delle costanti di normalizzazione, proprio
il **contributo visivo** isolato nella scomposizione all'inizio della sezione:
si sta decodificando sulla mutua informazione puntuale invece che sulla
probabilità totale.

I limiti seguono dalla stessa lettura. È una toppa in decodifica: non aggiunge
informazione, ridistribuisce quella che c'è. Se la distinzione che serve è già
scomparsa in $E(I)$, contrastare con $E(I')$ non la fa ricomparire: la
correzione sposta massa di probabilità fra token, non restituisce una dimensione
che l'encoder ha collassato. E con $\alpha$ grande si penalizza tutto ciò
che è insieme vero e atteso, cioè anche la forchetta nelle foto in cui la
forchetta c'è: si scambia un tipo di errore con l'altro.

`````

**Una seconda passata.** Il terzo rimedio prende la risposta già scritta, la
scompone in affermazioni elementari («c'è un piatto», «c'è una forchetta», «la
forchetta è a sinistra del piatto») e verifica ciascuna con l'immagine in mano,
riscrivendo o togliendo quelle che non passano. La forma delle domande di
verifica è, letteralmente, quella binaria di POPE: il protocollo di valutazione
diventa un componente del sistema. Il costo è la latenza, moltiplicata per il
numero di affermazioni; e il difetto è più
insidioso, perché se il verificatore è un modello della stessa famiglia porta lo
stesso priore, e può confermare con entusiasmo l'errore che avrebbe dovuto
smascherare. Il rimedio funziona nella misura in cui il secondo controllo è
**indipendente** dal primo: un rilevatore di oggetti, un segmentatore a
vocabolario aperto, una persona.

Nessuno dei tre elimina il fenomeno, e conviene dirlo senza attenuanti. Non è
pessimismo: è la conseguenza di come nasce. Finché la funzione di costo premia
la continuazione plausibile e l'immagine è un condizionamento fra gli altri, il
priore resta la strada più economica verso una perdita bassa. I rimedi spostano
il punto di equilibrio, rendono le affermazioni falsificabili, rendono più caro
dire ciò che si direbbe comunque, mettono un secondo paio di occhi. Riducono,
non curano. È la domanda del capitolo sull'AI responsabile, posta a un sistema
che vede: quanto è fragile, davvero, una volta messo nel mondo. E la prossima
pagina la rende meno accademica.

## Dalla percezione all'azione

Se un sistema sa mappare pixel e istruzioni in parole, niente gli impedisce di
mappare pixel e istruzioni in **azioni**, a una condizione: che le azioni si
possano scrivere. E scriverle si può, con il gesto che questo capitolo ha già
fatto due volte.

`````{tab} Elementare

Un braccio robotico riceve sette numeri: di quanto spostare la mano nelle tre
direzioni dello spazio, di quanto ruotarla nei tre versi, quanto stringere la
pinza. Sono numeri continui, e un modello che scrive parole non sa dire numeri
continui: sa scegliere una voce da un elenco. Allora si taglia ciascun numero in
256 gradini, come le tacche di un righello, e si dà a ogni gradino un nome preso
in prestito dal vocabolario, fra le parole che non si usano quasi mai.

Da quel momento «sposta la mano di un centimetro in avanti» è una parola, e
produrre un movimento è la stessa identica operazione che produrre una frase:
scegliere sette parole di fila. Non serve inventare niente di nuovo. Si riusa
tutto quello che il capitolo ha costruito, l'encoder che guarda, il connettore
che traduce, il modello che scrive; cambia soltanto che cosa c'è scritto
nell'elenco finale.

Il prezzo si legge sul righello. Se la mano si può spostare al massimo di cinque
centimetri per volta, i 256 gradini coprono dieci centimetri e distano meno di
quattro decimi di millimetro l'uno dall'altro: il comando non potrà mai essere
più preciso di due decimi di millimetro, che vanno benissimo per afferrare una
tazza e molto meno per infilare un ago.

`````

`````{tab} Superiore

Sia $a \in \mathbb{R}^{7}$ l'azione (tre componenti di traslazione, tre di
rotazione, una per l'apertura della pinza), a cui si aggiunge un indicatore
binario di fine episodio. Ogni componente $j$ viene discretizzata in $B = 256$
gradini uniformi fra $a_j^{\min}$ e $a_j^{\max}$, e l'indice del gradino diventa
un token: se il vocabolario ha già un simbolo per ogni intero fino a mille lo si
riusa così com'è, altrimenti si sovrascrivono le 256 voci meno frequenti. La
politica è allora

$$
\pi_\theta\big(a \mid I, x\big) = \prod_{j=1}^{7}
p_\theta\big(k_j \mid k_{<j},\, E(I),\, x\big),
$$

dove $k_j$ è il token del gradino della componente $j$, $I$ l'osservazione ed
$x$ l'istruzione in lingua naturale. È la fattorizzazione autoregressiva del
capitolo sui modelli di linguaggio, applicata a una sequenza lunga sette, con la
stessa cross-entropia come perdita. È l'impostazione di RT-2
{cite}`brohan2023rt2`, che addestra il modello in **co-fine-tuning** su una
miscela di traiettorie robotiche e di dati visione-linguaggio del web: le
traiettorie insegnano a muoversi, il resto della miscela impedisce al modello di
dimenticare quel che sapeva, ed è la ragione per cui un'istruzione mai comparsa
in nessuna dimostrazione può comunque essere eseguita, dato che il significato
delle parole viene da altrove. OpenVLA {cite}`kim2024openvla` porta la stessa
ricetta in una versione aperta e più piccola, con due accorgimenti che vale la
pena isolare. Gli estremi $a_j^{\min}$ e $a_j^{\max}$ non sono il minimo e il
massimo osservati ma i **quantili all'1% e al 99%** delle azioni di
addestramento, perché un solo campione anomalo allargherebbe la scala e
sprecherebbe i gradini. E l'encoder visivo **concatena per canali** le feature
di un modello contrastivo e di uno auto-supervisionato di sola visione:
esattamente il rimedio suggerito dal blocco precedente, adottato qui perché un
orientamento sbagliato non è più una parola sbagliata.

`````

La discretizzazione sta in poche righe, e il codice serve soprattutto a rendere
visibile che cosa il braccio esegue davvero, che non è mai l'azione richiesta ma
il centro del gradino più vicino.

```python
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Sette gradi di liberta': 3 di traslazione (metri), 3 di rotazione (radianti),
# 1 per l'apertura della pinza. Gli estremi sono i quantili all'1% e al 99%
# delle dimostrazioni, non il minimo e il massimo.
basso = np.array([-0.05, -0.05, -0.05, -0.20, -0.20, -0.20, 0.0])
alto  = np.array([ 0.05,  0.05,  0.05,  0.20,  0.20,  0.20, 1.0])

N_BIN = 256           # i gradini del righello
PRIMO = 32000 - 256   # gli ultimi 256 identificativi di un vocabolario da 32.000


def in_token(a):
    """Da un'azione continua a sette identificativi di token."""
    frazione = (np.clip(a, basso, alto) - basso) / (alto - basso)   # in [0, 1]
    return PRIMO + np.minimum((frazione * N_BIN).astype(int), N_BIN - 1)


def in_azione(token):
    """E ritorno: il centro del gradino, l'unica cosa che il braccio esegue."""
    frazione = (token - PRIMO + 0.5) / N_BIN
    return basso + frazione * (alto - basso)


a = np.array([0.012, -0.004, 0.021, 0.05, -0.11, 0.0, 1.0])
print(in_token(a))                    # [31902 31861 31925 31904 31801 31872 31999]
print(in_azione(in_token(a)))         # [ 0.0119 -0.0041  0.0209  0.0508 -0.1102  0.0008  0.998 ]
print((alto - basso) / (2 * N_BIN))   # [0.0002 0.0002 0.0002 0.0008 0.0008 0.0008 0.002 ]
```

L'ultima riga è l'errore massimo di quantizzazione, mezzo gradino: due decimi di
millimetro sulla traslazione, meno di un millesimo di radiante sulla rotazione.
È un limite noto e accettabile. Quelli che non si liquidano con un numero sono
altri tre, e vale la pena elencarli, perché fra il video di una dimostrazione e
un impianto che lavora ci sono tutti e tre.

Il primo è la **frequenza**. Un modello da decine di miliardi di parametri emette
uno o due comandi al secondo, e sceso a qualche miliardo arriva a cinque o sei;
un controllore classico ne emette decine o centinaia. Finché il compito è
afferrare e spostare va bene, per un movimento che deve
reagire in fretta no, e non è un problema di ingegneria del software: è la
dimensione del modello.

Il secondo è il **modello di errore**. Una parola sbagliata si rilegge, e il
peggio che capita è che qualcuno la creda. Un movimento sbagliato è già
avvenuto, ha spostato un oggetto vero e magari lo ha rotto; non esiste un
pulsante «rigenera». Un impianto industriale ragiona in tassi di guasto che si
contano in parti per milione, e nessun sistema addestrato per massima
verosimiglianza su meno di un milione di dimostrazioni ha oggi argomenti per
promettere quel numero.

Il terzo sono i **dati**. Le traiettorie non si raccolgono dal web: ognuna
richiede un robot vero e una persona che lo guida, e la scala che si raggiunge è
lontanissima dai miliardi di token del testo. E la generalizzazione qui cambia
natura: cambiare robot cambia la relazione fra il comando e il movimento, e una
politica non si trasferisce come si trasferisce un prompt. È il problema del
*sim-to-real* nominato nel capitolo introduttivo a proposito di robotica e
apprendimento per rinforzo, che nessuna quantità di didascalie risolve; ed è
anche la ragione per cui il capitolo sui **world model** è il vicino di casa
naturale di questa pagina, perché provare nell'immaginazione costa meno che
provare sul pezzo.

Resta il fatto che il meccanismo è di una economia notevole. Non c'è
un'architettura per l'azione: c'è la stessa macchina di tutto il capitolo, con
un vocabolario un po' più largo. E c'è, insieme, il motivo per cui questa pagina
viene dopo l'altra e non prima: un sistema che allucina una forchetta scrive una
parola di troppo, lo stesso sistema che comanda una mano allucina un movimento.

## Cinque domande, una bussola

Il capitolo si chiude dove era cominciato, con la domanda su dove si incontrano
i due flussi. **Allineare** due spazi senza fonderli dà un modello che cerca e
non parla, e uno spazio allineato non è uno spazio che capisce. **Innestare** un
occhio su un modello che sa già parlare dà un modello che conversa, e ha vinto
la saldatura più povera, perché comprimere significa scegliere prima di conoscere
la domanda. **Fondere** all'ingresso, in un vocabolario solo, dà un modello che
produce, al prezzo di un pre-addestramento da rifare e di una quantizzazione che
butta via. **Pagare il dettaglio** è il conto che presenta la risoluzione, dove
ogni pixel in più si trasforma in contesto occupato.

E infine **diffidare**, che non è una quinta tecnica ma la disposizione da
tenere davanti alle altre quattro: chiedersi non soltanto dove i due flussi si
sono incontrati, ma se si sono incontrati davvero, o se il modello sta parlando
di una fotografia che non ha guardato.

```{admonition} Da ricordare
:class: important
- L'**allucinazione visiva** non è un errore di percezione: la perdita si
  scompone in un **priore linguistico** più un **contributo visivo**, e dove la
  didascalia è già prevedibile dal testo il gradiente che spinge a guardare è
  debole. Guardare non è vietato, è facoltativo.
- Valutare su descrizioni libere è rumoroso in modo incontrollabile. **POPE**
  {cite}`li2023evaluating` trasforma la valutazione in domande **binarie** su
  oggetti presenti e assenti, con gli assenti scelti a caso, per frequenza o per
  **co-occorrenza**: il divario fra le tre condizioni misura il priore, non la
  bravura.
- Il test va **bilanciato** e letto con la **quota di sì**: un modello che
  risponde sempre «sì» non guarda mai e ottiene comunque un F1 di $0{,}667$, più
  di un modello che guarda davvero e sbaglia due volte su cinque ($0{,}60$).
- Una parte del difetto è **a monte**: esistono coppie di immagini con embedding
  contrastivi quasi identici (coseno oltre $0{,}95$) che un encoder di sola
  visione separa nettamente {cite}`tong2024eyes`. Quel che l'encoder non
  conserva non è recuperabile a valle, e il modello di linguaggio riempie il
  buco con il priore. È il limite composizionale della prima sezione
  {cite}`radford2021learning` visto dal lato dell'immagine.
- Tre rimedi, nessuna cura: **ancorare** la risposta alle coordinate (che il
  priore non sa indovinare), **decodificare per differenza** fra la
  distribuzione con e senza immagine (trattenuta da una soglia di plausibilità),
  una **seconda passata** di verifica (utile solo se indipendente). Riducono,
  non eliminano.
- Discretizzando i comandi di un robot in 256 gradini per grado di libertà,
  l'**azione diventa una sequenza di token** e tutta la macchina del capitolo si
  riusa {cite}`brohan2023rt2`, {cite}`kim2024openvla`. Restano il passo di
  quantizzazione, la frequenza di controllo di pochi hertz, i dati che non si
  raccolgono dal web, e un modello di errore in cui la mossa sbagliata è già
  avvenuta.
```
