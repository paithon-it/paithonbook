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
frasi plausibili la scrive perché la frase, senza, sarebbe meno plausibile.

A quello che il modello si aspetta di leggere *prima* di guardare la fotografia
daremo un nome, e lo useremo per tutta la sezione: **priore linguistico**. In
parole povere è l'abitudine della lingua, quello che di solito viene scritto in
frasi come questa; «priore» perché viene prima, prima dell'immagine e
indipendentemente da essa. La sezione mostra perché quell'abitudine vinca così
spesso sull'evidenza visiva, come si fa a misurarlo, e che cosa si può fare per
contenerlo.

## Un errore che non riguarda gli occhi

L'apertura del capitolo ha già dato un nome al fenomeno, **allucinazione
visiva**, e ne ha mostrato la radice nella funzione di costo, cioè nel
punteggio dell'errore che l'addestramento fa scendere. Conviene scavare un
poco più a fondo, perché la forma precisa dell'argomento dice anche dove si
può intervenire.

`````{tab} Elementare

La tastiera del telefono, dopo «a domani e buona», propone «serata». Non sa
che cosa stai per dire: sa come vanno a finire le frasi, e sbaglia così di
rado che smettiamo di accorgercene. Un modello che descrive una
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

E l'abitudine parte avvantaggiata. Le frasi la tastiera le macina da anni; la
telecamera gliel'hanno attaccata ieri, e il filo fra le due è nuovo di zecca, per
cui il primo giorno non guardare costa meno. Poi c'è il materiale su cui ha
imparato. Le descrizioni delle fotografie le ha scritte qualcuno che aveva
davanti due righe di didascalia e non la foto, e dove leggeva «tavola
apparecchiata» ha messo piatto, coltello e forchetta. Quella forchetta era lì
prima di ogni domanda.

`````

`````{tab} Superiore

Un modello con connettore {cite}`liu2023visual` ottimizza la cross-entropia
autoregressiva
$\mathcal{L}(\theta) = -\sum_t \log p_\theta\big(y_t \mid y_{<t}, E(\mathbf{I})\big)$,
dove $y_t$ è il token al passo $t$, $\mathbf{I}$ l'immagine ed $E$ l'encoder visivo con
il suo connettore. Il termine dentro il logaritmo si scompone in modo
istruttivo:

$$
\log p_\theta\big(y_t \mid y_{<t}, E(\mathbf{I})\big) =
\underbrace{\log p_\theta\big(y_t \mid y_{<t}\big)}_{\text{priore linguistico}} +
\underbrace{\log \frac{p_\theta\big(y_t \mid y_{<t}, E(\mathbf{I})\big)}{p_\theta\big(y_t \mid y_{<t}\big)}}_{\text{contributo visivo}},
$$

dove il primo addendo è ciò che il modello direbbe a occhi chiusi. L'identità
è algebrica e vale sempre; le etichette dei due addendi chiedono un'ipotesi in
più. La rete interrogata senza immagine non calcola il marginale vero
$\mathbb{E}_{\mathbf{I}}\big[p_\theta(y_t \mid y_{<t}, E(\mathbf{I}))\big]$: è un altro percorso
di calcolo, mai addestrato a marginalizzare. Nella misura in cui lo
approssima, il primo addendo stima il priore linguistico e il secondo la
**mutua informazione puntuale** fra il token e l'immagine, dato il prefisso.
La riduzione della perdita che il condizionamento sull'immagine può al più
produrre è la mutua informazione condizionata $\mathcal{I}(Y_t; \mathbf{I} \mid Y_{<t})$:
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

La radice è esattamente quella delle allucinazioni testuali del capitolo sui
Transformer, quelle «risposte fluenti e sbagliate» che il capitolo
sull'MLOps metterà fra i bersagli del monitoraggio. Con un'aggravante e
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

La via classica è del 2018 e si chiama **CHAIR** {cite}`rohrbach2018object`, le
iniziali di *Caption Hallucination Assessment with Image Relevance*. Si fissa un
elenco chiuso di categorie di oggetti e si cercano quelle parole nel testo
generato, con una tabella di sinonimi e di plurali. Poi si conta: di tutti gli
oggetti che il modello ha nominato, quanti non compaiono nell'elenco di quel che
c'è nella fotografia, scritto a mano da chi le ha preparate.
Funziona, è stata la prima misura del campo, ed è il capostipite della famiglia
che guarda quel che il modello scrive di sua iniziativa invece di interrogarlo.
Porta però con sé quattro fragilità che non si possono togliere.

La prima: vede solo gli oggetti dell'elenco, quindi un colore sbagliato, un
conteggio sbagliato, una relazione spaziale rovesciata sono invisibili. La
seconda: dipende dalla completezza delle annotazioni, e un oggetto che c'è
davvero ma che nessuno ha annotato viene contato come allucinazione. La terza:
cercare parole dentro un testo libero è un metodo approssimativo, che va bene in
media e sbaglia sui casi storti, perché inciampa sulle negazioni («non c'è
nessuna forchetta» contiene la parola «forchetta») e sui riferimenti generici. E
la quarta: il punteggio dipende da cose che con l'immagine non c'entrano, cioè
da come è formulata la richiesta e da quanto è lunga la descrizione che ne esce,
perché più si scrive più si rischia di sbagliare. È l'obiezione che muovono gli autori del protocollo di cui parliamo
fra poco {cite}`li2023evaluating`, e che li ha portati a cambiare strada.

Il risultato è una misura che **si muove per ragioni
che con l'immagine non c'entrano**. Chiedi al modello una descrizione più lunga
e il punteggio peggiora; cambia il modo di chiedere e cambia di nuovo, senza che
il modello sia cambiato di una virgola. È la cosa peggiore che si possa avere in
mano quando si vuole stabilire se un fenomeno esista. La via d'uscita non è un
analizzatore migliore. È cambiare la domanda.

## Una domanda con due sole risposte

L'impostazione che ha reso il problema trattabile è quella di POPE
{cite}`li2023evaluating` (le iniziali di *Polling-based Object Probing
Evaluation*, cioè una valutazione che tasta gli oggetti a forza di domande): non
si chiede più al modello di descrivere, gli si chiede «c'è una forchetta in
questa immagine?» e si accetta solo sì o no. La risposta è una parola sola, la
verità sta nell'elenco di quel che c'è, e nessun giudice deve interpretare
niente. È la ragione per cui il protocollo esiste. L'alternativa sarebbe mettere
a correggere un secondo modello di linguaggio (il modello giudice,
l’*LLM-as-a-judge* di cui parlerà il capitolo sull'MLOps), e un secondo modello
si porta dietro i propri difetti proprio là dove si vuole misurarne uno.

Il cuore del metodo, però, sta in **come si scelgono gli oggetti assenti**, più
che nel formato binario. Chiedere «c'è una zebra?» davanti a una cucina non misura
niente.

`````{tab} Elementare

Un esame si fa facile o difficile decidendo che cosa chiedere. Qui si risponde
soltanto sì o no, e la difficoltà sta tutta nell'oggetto assente su cui si
interroga. I modi sono tre.

Il primo: peschi un oggetto a caso dall'elenco delle categorie. «C'è una zebra?»
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
tre, e si guarda quanto scendono passando dalla domanda regalata a quella
cattiva. Quella discesa non racconta quanto il modello sia bravo. Racconta a
quale abitudine si sta appoggiando.

`````

`````{tab} Superiore

Sia $\mathcal{O}$ l'insieme delle categorie annotate nel corpus e
$\mathcal{O}(\mathbf{I}) \subseteq \mathcal{O}$ quelle presenti nell'immagine $\mathbf{I}$. Le
domande positive si estraggono da $\mathcal{O}(\mathbf{I})$, quelle negative da
$\mathcal{O} \setminus \mathcal{O}(\mathbf{I})$ secondo tre distribuzioni:

$$
q_{\text{unif}}(o) \propto 1,
\qquad
q_{\text{freq}}(o) \propto \hat{p}(o),
\qquad
q_{\text{cooc}}(o) \propto \sum_{o' \in \mathcal{O}(\mathbf{I})} \hat{p}(o \mid o'),
$$

dove $\hat{p}(o)$ è la frequenza marginale della categoria $o$ nel corpus e
$\hat{p}(o \mid o')$ la sua frequenza condizionata alla presenza di $o'$; nelle
ultime due si prendono i primi $k$ candidati in ordine di punteggio anziché
campionare, con $k$ pari al numero di domande negative che tocca all'immagine.
Le tre condizioni mettono alla prova separatamente due
priori diversi, quello **marginale** e quello
**condizionato alla co-occorrenza**, più un controllo. Il divario fra le
accuratezze nelle tre condizioni è una stima di quanto ciascun priore stia
guidando la risposta.

Tre proprietà del disegno meritano di essere isolate, perché sono ciò che lo
rende una misura e non un sondaggio. L'insieme è **bilanciato**, metà domande
con risposta sì e metà con risposta no, così che entrambe le strategie
degeneri si collochino al livello del caso in accuratezza. La risposta è un
token, quindi il confronto con la verità è esatto e riproducibile. E accanto
alle metriche si riporta la **quota di sì**,
$\hat{\rho} = \frac{1}{n}\sum_{i} \mathbb{1}[\hat{y}_i = \text{sì}]$, che è il
vero strumento diagnostico: un $\hat{\rho}$ lontano da $0{,}5$ dice che il
modello non sta rispondendo alla domanda, sta esprimendo una disposizione.

`````

Accanto al punteggio conviene guardare sempre un secondo numero, la **quota di
sì**: su tutte le domande fatte, quante volte il modello ha risposto «sì».
Perché non sia un ornamento si vede facendo i conti su un test bilanciato di
tremila domande, millecinquecento su oggetti presenti e millecinquecento su
oggetti assenti.

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

Il primo modello risponde sempre «sì»: non guarda mai, e sbaglia una domanda su
due, perché azzecca tutte le millecinquecento domande sugli oggetti che ci sono e
sbaglia tutte le millecinquecento su quelli che non ci sono. Eppure il punteggio
con cui di solito si riassumono queste prove, l’**F1**, lo premia. Conviene
smontarlo, perché è fatto di due numeri che qui tirano in direzioni opposte. Il
**richiamo** è la quota di oggetti presenti che il modello ha riconosciuto: chi
dice sempre «sì» non se ne lascia sfuggire nemmeno uno, quindi prende il massimo,
$1$. La **precisione** è la quota di volte in cui, avendo detto «sì», aveva
ragione: qui una su due, cioè $0{,}5$. L'F1 non è la loro media normale, che
darebbe $0{,}75$: è la media che tira verso il più piccolo, il doppio del
prodotto diviso la somma, $2 \cdot 1 \cdot 0{,}5 / (1 + 0{,}5) = 0{,}667$. Un
modello che guarda davvero ma sbaglia due volte su cinque, sia sulle domande a
cui va risposto sì sia su quelle a cui va risposto no, si ferma a $0{,}60$:
**meno**.

A leggere il solo F1 si metterebbe in classifica sopra a chi guarda davvero e
sbaglia due volte su cinque un modello che dell'immagine non ha usato un
pixel. La quota di sì scioglie l'equivoco in un
colpo: $1{,}0$ contro $0{,}5$, e il primo dei due non sta rispondendo, sta
ripetendo sempre la stessa cosa.

Il protocollo ha tre limiti. Il primo: si misura
l’**esistenza degli oggetti**, e nient'altro; un colore sbagliato, un conteggio
sbagliato, una relazione rovesciata restano invisibili. Il secondo: poiché la misura è pubblica e la strategia per migliorarla è nota, un modello istruito a
dire «no» più spesso guadagna punti senza aver guadagnato un grammo di vista, e
la quota di sì lo smaschera soltanto se chi legge se la va a guardare. È la
legge di Goodhart, quella per cui una misura, appena diventa un obiettivo,
smette di misurare ciò che misurava, e vale qui come altrove.

Il terzo è il più facile da dimenticare, perché riguarda il confine fra le due
misure e non i loro difetti. Domandare non è far descrivere: qui si misura se il
modello **acconsente** a un oggetto che non c'è, non se lo **nomina** scrivendo
di sua iniziativa. Sono due grandezze diverse, non due letture della stessa, e la
seconda è precisamente quella con cui la sezione si è aperta, il tavolo con la
forchetta. Un modello può rispondere «no, non c'è nessuna forchetta» a chi glielo
chiede e continuare a metterla in tutte le sue descrizioni: per accorgersene
serve ancora una misura sul testo generato, con tutta la sua rumorosità. Le due
si leggono insieme, e nessuna delle due sostituisce l'altra.

## Chi controlla il controllore

C'è un piano superiore della stessa domanda, e questa sezione non sarebbe
onesta a non porlo. Abbiamo chiesto: il modello ha davvero guardato? E abbiamo
risposto con un protocollo di misura. Ma anche il protocollo può rispondere
senza aver guardato.

`````{tab} Elementare

Di un compito in classe gira da mesi la fotocopia con le soluzioni: i voti
alti non dicono più chi ha studiato, dicono chi ha visto la fotocopia. In
questo campo è successo, ed è documentato. Le prove con cui si misurano questi
sistemi sono pubbliche, stanno sul web, e sul web questi sistemi si addestrano:
domande e risposte finiscono nel materiale di studio insieme a tutto il resto.

C'è anche un secondo difetto, più banale e forse peggiore: molte domande si
possono indovinare senza guardare la fotografia. Queste prove, a differenza
delle domande a cui si risponde sì o no, sono a crocette, e la risposta sta
spesso nella domanda stessa, nelle alternative proposte accanto («che animale
c'è nella foto? a) un cane b) una sedia c) un tavolo d) una nuvola») o in cose
che chiunque sa del mondo.

Per fortuna il controllo che li scopre tutti e due è il più semplice che si possa
immaginare: rifare l'esame **togliendo l'immagine**. Quello che il modello porta
a casa a occhi chiusi è quello che non ha imparato guardando. E per sapere quale
dei due difetti si ha davanti, quel voto si mette accanto a quello di un compagno
che ha letto gli stessi libri e non ha mai fatto il corso con le fotografie;
batterlo a occhi chiusi vuol dire che la fotocopia è girata. Chi pubblica un
punteggio senza aver riportato anche quello sta chiedendo di essere creduto sulla
parola.

`````

`````{tab} Superiore

Il fenomeno è stato quantificato da chi ha costruito MMStar
{cite}`chen2024mmstar`, ed è di ampiezza tale da rendere non interpretabili
molti punteggi pubblicati. I due meccanismi sono distinti. Il primo è la
**risolvibilità dal solo testo**: un modello di solo linguaggio fra i più forti,
interrogato senza ricevere alcuna immagine, batte la scelta casuale di oltre
$24$ punti in media su sei benchmark generalisti, perché la risposta si ricava
dalla domanda, dalle opzioni o dalla conoscenza del mondo che il modello ha già. Il secondo è la **fuga di dati**: un modello
ottiene $43{,}6\%$ su un benchmark multimodale senza immagini, cioè $17{,}9$
punti **sopra** il proprio modello di linguaggio di base, che è la firma della
memorizzazione, non della deduzione. Da qui le due misure che gli autori
propongono, il guadagno multimodale (quanto si perde togliendo l'immagine) e la
fuga multimodale (quanto il sistema completo, interrogato anche lui senza
immagini, supera il proprio modello di
linguaggio di base: se lo supera, il di più viene dai dati di addestramento
multimodali, non dal guardare).

Il caso è aggravante proprio per un protocollo come quello appena descritto,
che poggia sulle annotazioni di un corpus fotografico pubblico, cioè su un
corpus che sta nella miscela di addestramento di quasi ogni sistema di cui si
vuole misurare l'allucinazione. Qui serve cautela: che quel
corpus sia nella miscela è noto, che questo gonfi i punteggi è un rischio
documentato altrove e non una misura pubblicata su questo protocollo. Il rimedio
resta comunque quello, e costa una riesecuzione: riportare il punteggio **a
immagine tolta** accanto a quello ordinario. È la stessa mossa che fra poco
troveremo fra i rimedi in decodifica (confrontare la risposta a occhi aperti con
quella a occhi chiusi), portata dalla generazione alla valutazione.

`````

## Il difetto viene da più a monte

Fin qui abbiamo trattato il priore linguistico come un concorrente troppo forte.
Ma c'è un secondo pezzo del meccanismo, e sta prima: a volte l'informazione che
avrebbe permesso di rispondere non è arrivata affatto.

`````{tab} Elementare

Devi distinguere due pacchi e hai soltanto una bilancia. Uno è pieno di libri,
l’altro di piume, e sulla bilancia segnano quasi uguale: due chili tondi il
primo, due chili e un grammo il secondo. Quel grammo la bilancia
lo scrive, ma è così poco che chi legge il numero lo arrotonda via senza
pensarci, e nessuno gli ha mai insegnato che proprio lì stava la risposta. Se poi
gli chiedi «in quale pacco ci sono i libri?», dovrà tirare a indovinare, e tirerà
a indovinare secondo l'abitudine, perché non ha altro.

La bilancia, qui, è l'encoder della prima sezione, quello addestrato sulle
didascalie del web. Si possono trovare, e si sono trovate, coppie di fotografie
che una persona distingue in mezzo secondo (un animale girato a destra e lo
stesso girato a sinistra, una scarpa allacciata e la stessa slacciata) e che
quell'encoder misura quasi identiche: somiglianza sopra $0{,}95$, su una scala
che arriva a $1$. Un secondo strumento, addestrato sulle sole immagini e senza
aver mai visto una didascalia, mette le stesse due foto sotto $0{,}6$: per lui
sono due cose diverse. La differenza non sta nella fotografia, sta nella
bilancia con cui la si pesa, e non è sfortuna: chi ha imparato dalle didascalie
tiene quello che le didascalie nominano, e il verso in cui è girato un animale
le didascalie non lo dicono quasi mai. (Le due coppie di fotografie, del resto,
si sono cercate apposta con quei due numeri in mano: è così che sono state
trovate.)

Ed ecco il punto che chiude il cerchio: quando la bilancia distingue troppo
poco, il modello di linguaggio non risponde «non lo so». Riempie il buco con
quello che di solito è vero. Il punto cieco non produce silenzio, produce
allucinazione.

`````

`````{tab} Superiore

Il lavoro di Tong e colleghi {cite}`tong2024eyes` costruisce **coppie cieche**
in modo operativo: due immagini $\mathbf{I}_1, \mathbf{I}_2$ tali che

$$
\big\langle E_{\text{CLIP}}(\mathbf{I}_1), E_{\text{CLIP}}(\mathbf{I}_2) \big\rangle > 0{,}95
\qquad\text{e}\qquad
\big\langle E_{\text{SSL}}(\mathbf{I}_1), E_{\text{SSL}}(\mathbf{I}_2) \big\rangle < 0{,}6,
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

Perché a valle non si recuperi si adduce di solito un argomento informazionale,
e conviene essere precisi sul fatto che da solo non basta. Il decoder vede
soltanto $\mathbf{Z} = E(\mathbf{I})$, quindi la catena
$\mathbf{I} \to \mathbf{Z} \to Y$ è markoviana e per qualunque risposta $Y$ vale
la disuguaglianza dell'elaborazione dei dati,
$\mathcal{I}(Y; \mathbf{I}) \le \mathcal{I}(\mathbf{Z}; \mathbf{I})$, dove
$\mathcal{I}$ è la mutua informazione: addestrando ciò che viene dopo non si
aggiunge informazione sull'immagine. Vero, e qui **inoffensivo**. L'encoder è una
funzione deterministica e le due immagini della coppia hanno coseno $0{,}95$,
cioè embedding *distinti*: finché $E$ è iniettivo,
$\mathcal{I}(\mathbf{Z}; \mathbf{I}) = H(\mathbf{I})$, con $H$ l'entropia
dell'immagine (finita, perché i pixel sono già quantizzati), e un limite pari a
tutta l'informazione disponibile non vieta niente a nessuno. La disuguaglianza
morderebbe se l'encoder mandasse le due immagini **nello stesso** punto, che non
è ciò che il lavoro citato osserva.

Il limite vero è di **margine**, non di informazione, ed è più istruttivo. La
differenza fra le due immagini sopravvive nell'embedding, ma lungo una direzione
di norma piccolissima, che il coseno pesa poco e che nulla, in addestramento, ha
mai chiesto al decoder di leggere. Se $g$ è il decoder ed è lipschitziano di
costante $L$, allora
$\lVert g(\mathbf{z}_1) - g(\mathbf{z}_2) \rVert \le L \lVert \mathbf{z}_1 - \mathbf{z}_2 \rVert$:
con $\mathbf{z}_1 \approx \mathbf{z}_2$ le due risposte partono costrette a
somigliarsi, mentre le risposte corrette sono opposte. Non è una dimostrazione di
impossibilità, perché $L$ non viene maggiorato e per una rete profonda è enorme;
è la constatazione che quella distinzione la troverebbe solo chi la cercasse, e
che il modello non la cerca, mentre il priore linguistico lo spinge a rompere il
pareggio in un altro modo. Chi volesse l'affermazione
informazionale in senso forte deve introdurre del rumore, che nei sistemi veri
c'è (quantizzazione, precisione ridotta, augmentation in addestramento): allora
$\mathcal{I}(\mathbf{Z}; \mathbf{I})$ cala davvero e la disuguaglianza torna a
mordere. In tutti i casi, punto cieco a monte e allucinazione a valle sono
un difetto e la sua manifestazione.

`````

Conviene dire in modo esplicito che questo è **lo stesso limite della prima
sezione, visto dall'altro lato**. Là, dal lato del testo, il gioco
dell'abbinamento chiedeva solo di distinguere la didascalia vera da quelle di
altre fotografie prese a caso, e per vincerlo bastava riconoscere gli oggetti:
da qui il comportamento a «sacco di concetti», che tratta la frase come un
mucchio di parole senza ordine. Qui, dal lato dell'immagine, vale la
conseguenza speculare: se un tratto della fotografia non serve mai a fare quella
scelta, buttarlo via non costa niente, e l'addestramento, che è pigro per
mestiere, lo butta. Il verso in cui è girato un animale, quanti oggetti ci sono,
un particolare minuto sono precisamente i tratti da cui la didascalia di
un'altra fotografia non dipende mai. Un solo buco, due modi di infilarci il
dito.

Il rimedio a monte è coerente con la diagnosi: se un encoder addestrato sulle
didascalie perde ciò che le didascalie non nominano, gli si affianca un secondo
encoder addestrato sulle sole immagini e si uniscono le due descrizioni. Il come
cambia il conto: si può allungare la fila di numeri di ogni tessera attaccandoci
in coda quella dell'altro encoder, e allora la sequenza resta lunga uguale;
oppure mettere in fila prima i pezzi di un encoder e poi quelli dell'altro, e
allora il posto occupato raddoppia (è il conto della sezione sulla risoluzione). In tutti i
casi restano due encoder da far girare invece di uno, e il problema si sposta
invece di sparire.

## Tre rimedi, nessuna cura

Le contromisure che hanno un senso meccanico sono tre, e attaccano il fenomeno
in tre punti diversi: l'addestramento, il momento in cui la risposta si scrive,
la risposta già scritta.

**Ancorare la risposta a ciò che si vede.** Invece di chiedere al modello *che
cosa* c'è, gli si chiede anche *dove*: il nome dell'oggetto accompagnato dalle
quattro coordinate del riquadro che lo contiene, scritte nella stessa
risposta. I quattro numeri sono le coordinate di due angoli opposti del
rettangolo, l'alto a sinistra e il basso a destra, misurate in frazioni di
immagine: zero a un bordo, uno al bordo opposto. Su come scriverli i due
lavori di riferimento prendono strade opposte, e conviene vederle affiancate.
Kosmos-2 {cite}`peng2023kosmos` taglia quell'intervallo in un numero fisso di
gradini e dà a ogni gradino un simbolo **nuovo**, aggiunto all'elenco da cui
il modello pesca: è il gesto del mosaicista con il suo catalogo di tessere,
applicato qui a una grandezza che non è l'immagine. Shikra
{cite}`chen2023shikra` fa il contrario, e lo rivendica: nessun simbolo nuovo,
nessun gradino, le coordinate sono numeri decimali scritti in lingua naturale
dentro la frase, come li scriverebbe una persona.

Quale delle due si scelga, per il nostro problema cambia poco, ed è questo il
punto: «una forchetta» l'abitudine della lingua te la regala, «una forchetta in
$0{,}42$, $0{,}31$, $0{,}55$, $0{,}60$» no, perché quei quattro numeri
dall'abitudine non si ricavano. In uscita, l'effetto è che l'affermazione
diventa verificabile: chi legge può andare a guardare quel rettangolo. In
addestramento l'effetto è più profondo, e riguarda il gradiente, cioè il
segnale che corregge i pesi: siccome sbagliare le coordinate costa e indovinarle
per abitudine non si può, l'unico modo di abbassare la perdita è passare
davvero per l'immagine.

**Decodificare per differenza.** Il secondo rimedio non tocca i pesi: cambia
come si sceglie il token.

`````{tab} Elementare

Il trucco è fare la stessa domanda due volte: la prima guardando la fotografia,
la seconda guardando la stessa fotografia rovinata di proposito, coperta di
disturbo finché non ci si distingue quasi più niente. Poi si tiene solo la
differenza. Per brevità diremo «a occhi aperti» e «a occhi chiusi», ma nel
secondo caso gli occhi restano socchiusi e non chiusi del tutto, perché
l'immagine c'è ancora, solo che è illeggibile.
(Qualcuno la toglie del tutto, ed è la variante più radicale dello stesso gesto.)

Se «forchetta» risulta probabile in entrambi i casi, quella parola non viene
dalla foto: viene dall'abitudine, e allora la si penalizza. Se «coltello» è
probabile solo a occhi aperti, quella parola l'ha vista davvero, e la si premia.
In pratica si sottrae, punto per punto, quello che il modello direbbe comunque.

Una precauzione serve, altrimenti il trucco si rivolta: sottraendo senza freni
si finisce per premiare parole assurde, che a occhi chiusi erano
improbabilissime e a occhi aperti solo un po’ meno. Il rimedio è restringere la
scelta in partenza alle parole che a occhi aperti valevano almeno un decimo
della più probabile: dentro quella rosa si confronta, fuori non si guarda. E il
conto da pagare è semplice: due letture invece di una, quindi il doppio del
tempo per ogni parola scritta.

Restano due modi di farsi male. Se si spinge forte sulla sottrazione sparisce
anche la forchetta delle tavole in cui la forchetta c'è davvero, e si è barattato
un errore con un altro. E se la bilancia che pesa la fotografia le due cose non
le aveva separate, chiedere due volte non le separa: la differenza rimescola le
parole in classifica, non aggiunge un pixel.

`````

`````{tab} Superiore

Detti $\ell_\theta$ i logit del modello, $\mathbf{x}$ il prompt testuale, $\mathbf{I}$ l'immagine
e $\mathbf{I}'$ la stessa immagine degradata (nel lavoro citato con il rumore gaussiano
del processo diretto di diffusione, aggiunto finché la scena non è più
riconoscibile; varianti successive contrastano invece con l'assenza
dell'immagine), la decodifica contrastiva visiva
{cite}`leng2024mitigating` sceglie il token successivo secondo

$$
\ell_{\text{cd}}(y_t) = (1 + \alpha)\,\ell_\theta\big(y_t \mid y_{<t}, \mathbf{x}, \mathbf{I}\big)
- \alpha\,\ell_\theta\big(y_t \mid y_{<t}, \mathbf{x}, \mathbf{I}'\big),
$$

ristretto all'insieme dei candidati plausibili

$$
\mathcal{V}_t = \Big\{ w \in V \;:\;
p_\theta\big(w \mid y_{<t}, \mathbf{x}, \mathbf{I}\big) \ge
\beta \max_{w' \in V} p_\theta\big(w' \mid y_{<t}, \mathbf{x}, \mathbf{I}\big) \Big\},
$$

dove $\alpha \ge 0$ regola la forza della correzione e $\beta \in (0,1)$ (in
pratica intorno a $0{,}1$) è la soglia di plausibilità che impedisce alla
sottrazione di promuovere token del tutto improbabili. La differenza dei due
logit è, a meno delle costanti di normalizzazione, proprio
il **contributo visivo** isolato nella scomposizione della perdita:
si sta decodificando su una **stima** della mutua informazione puntuale invece
che sulla probabilità totale, con la stessa approssimazione di allora, resa
qui ancora più larga quando $\mathbf{I}'$ è un'immagine degradata e non l'assenza
dell'immagine.

I limiti seguono dalla stessa lettura. È una toppa in decodifica: non aggiunge
informazione, ridistribuisce quella che c'è. Se la distinzione che serve è già
scomparsa in $E(\mathbf{I})$, contrastare con $E(\mathbf{I}')$ non la fa ricomparire: la
correzione sposta massa di probabilità fra token, non restituisce una dimensione
che l'encoder ha collassato. E con $\alpha$ grande si penalizza tutto ciò
che è insieme vero e atteso, cioè anche la forchetta nelle foto in cui la
forchetta c'è: si scambia un tipo di errore con l'altro.

`````

{numref}`fig-decodifica-per-differenza` mostra il meccanismo su un vocabolario
giocattolo di sei parole.

```{figure} ../figures/decodifica-per-differenza.svg
:name: fig-decodifica-per-differenza
:alt: Tre passi di decodifica, uno per parola. In alto la risposta cresce: un piatto, un coltello e un bicchiere. Sotto, per ogni passo, tre colonne di barre sullo stesso vocabolario di sei parole: le probabilità con la foto, quelle con la foto resa illeggibile e quelle che restano dopo aver sottratto le seconde dalle prime. Ai primi due passi la sottrazione conferma o premia la parola che l'immagine porta davvero; al terzo forchetta è la più alta in tutte e due le letture, quindi non viene dalla foto, e sottraendo sprofonda sotto bicchiere, che era seconda.
:width: 96%

Il rimedio visto nel tempo: a ogni parola due letture, una sottrazione e una
scelta. Al terzo passo «forchetta» guida tutte e due le letture, ed è proprio
questo a condannarla: quello che il modello direbbe comunque non viene
dall'immagine. I numeri sono un esempio giocattolo: qui si sottrae per intero
quel che il modello direbbe a occhi chiusi, e si guardano solo le parole che a
occhi aperti valgono almeno un decimo della prima.
```

**Una seconda passata.** Il terzo rimedio prende la risposta già scritta, la
scompone in affermazioni elementari («c'è un piatto», «c'è una forchetta», «la
forchetta è a sinistra del piatto») e verifica ciascuna con l'immagine in mano,
riscrivendo o togliendo quelle che non passano {cite}`yin2023woodpecker`. La
forma delle domande di
verifica è, letteralmente, quella binaria di POPE: il protocollo di valutazione
diventa un componente del sistema. Il costo è la latenza, moltiplicata per il
numero di affermazioni; e il difetto è più
insidioso, perché se il verificatore è un modello della stessa famiglia porta lo
stesso priore, e può confermare con entusiasmo l'errore che avrebbe dovuto
smascherare. Il rimedio funziona nella misura in cui il secondo controllo è
**indipendente** dal primo: un programma addestrato a trovare oggetti in una
foto, uno che li ritaglia sapendo riconoscere anche categorie che non erano nel
suo elenco, oppure una persona.

Nessuno dei tre elimina il fenomeno, e conviene dirlo senza attenuanti. È la
conseguenza di come nasce. Finché la funzione di costo premia
la continuazione plausibile e l'immagine è un condizionamento fra gli altri, il
priore resta la strada più economica verso una perdita bassa. I rimedi spostano
il punto di equilibrio, rendono le affermazioni falsificabili, rendono più caro
dire ciò che si direbbe comunque, mettono un secondo paio di occhi. Riducono,
non curano. È la domanda che tornerà nel capitolo sull'AI responsabile, posta
qui a un sistema che vede: quanto è fragile, davvero, una volta messo nel mondo.
E le pagine che chiudono questa sezione rendono la domanda meno accademica.

## Dalla percezione all'azione

Se un sistema sa mappare pixel e istruzioni in parole, niente gli impedisce di
mappare pixel e istruzioni in **azioni**, a una condizione: che le azioni si
possano scrivere. E scriverle si può, con il gesto che questo capitolo ha già
fatto due volte, per i pezzi d'immagine e poco fa per le coordinate dei
riquadri: si taglia una grandezza continua in gradini e si dà un nome a ogni
gradino.

`````{tab} Elementare

Un braccio robotico riceve sette numeri: di quanto spostare la mano nelle tre
direzioni dello spazio, di quanto ruotarla nei tre versi, quanto stringere la
pinza. Sono numeri continui, e un modello che scrive parole sa soltanto
scegliere una voce da un elenco. Allora si taglia ciascun numero in 256 gradini,
come le tacche di un righello, e si dà a ogni gradino un nome preso in prestito
dal vocabolario, fra le parole che non si usano quasi mai.

Da quel momento «sposta la mano di un centimetro in avanti» è una parola, e
produrre un movimento è la stessa identica operazione che produrre una frase:
scegliere sette parole di fila. Si riusano i pezzi di prima, l'encoder che
guarda, il connettore che traduce, il modello che scrive; cambia soltanto che
cosa c'è scritto nell'elenco finale. E mentre impara a muoversi continua a
leggere fotografie e frasi come faceva prima, per cui può eseguire un ordine che
in nessuna dimostrazione ha mai visto: che cosa sia una banana non gliel'ha
insegnato il robot.

Il prezzo si legge sul righello. Quanto farlo lungo lo dicono le dimostrazioni
raccolte, buttando via l'uno per cento più esagerato a ciascun capo: un solo
strattone finito lì per sbaglio stirerebbe il righello e allontanerebbe le tacche
per tutti. Se la mano si può spostare al massimo di cinque centimetri per volta,
in avanti o all'indietro, il righello è lungo dieci centimetri: i 256 gradini se
li dividono, e distano meno di quattro decimi di millimetro l'uno dall'altro. Il
braccio esegue sempre il gradino più vicino a quel che gli è stato detto, quindi
sbaglia al massimo di mezzo gradino: due decimi di millimetro, che vanno
benissimo per afferrare una tazza e molto meno per infilare un ago.

`````

`````{tab} Superiore

Sia $\mathbf{a} \in \mathbb{R}^{7}$ l'azione (tre componenti di traslazione, tre di
rotazione, una per l'apertura della pinza), a cui si aggiunge un indicatore
binario di fine episodio. Ogni componente $j$ viene discretizzata in $B = 256$
gradini uniformi fra $a_j^{\min}$ e $a_j^{\max}$, e l'indice del gradino diventa
un token: se il vocabolario ha già un simbolo per ogni intero fino a mille lo si
riusa così com'è, altrimenti si sovrascrivono le 256 voci meno frequenti. La
politica è allora

$$
\pi_\theta\big(\mathbf{a} \mid \mathbf{I}, \mathbf{x}\big) = \prod_{j=1}^{8}
p_\theta\big(k_j \mid k_{<j},\, E(\mathbf{I}),\, \mathbf{x}\big),
$$

dove $k_1, \dots, k_7$ sono i token dei gradini delle sette componenti, $k_8$
quello di fine episodio, $\mathbf{I}$ l'osservazione ed $\mathbf{x}$
l'istruzione in lingua naturale. È la fattorizzazione autoregressiva dei
grandi modelli linguistici, vista nella
{doc}`pagina sui grandi modelli linguistici </Transformers/llm>`, applicata a una
sequenza lunga otto, con la stessa cross-entropia
come perdita. È l'impostazione di RT-2
{cite}`brohan2023rt2`, che addestra il modello in **co-fine-tuning** su una
miscela di traiettorie robotiche e di dati visione-linguaggio del web: le
traiettorie insegnano a muoversi, il resto della miscela impedisce al modello di
dimenticare quel che sapeva, ed è la ragione per cui un'istruzione mai comparsa
in nessuna dimostrazione può comunque essere eseguita, dato che il significato
delle parole viene da altrove. OpenVLA {cite}`kim2024openvla` porta la stessa
ricetta in una versione aperta e più piccola, con due accorgimenti da
isolare. Gli estremi $a_j^{\min}$ e $a_j^{\max}$ non sono il minimo e il
massimo osservati ma i **quantili all'1% e al 99%** delle azioni di
addestramento, perché un solo campione anomalo allargherebbe la scala e
sprecherebbe i gradini. E l'encoder visivo **concatena per canali** le feature
di un modello contrastivo e di uno auto-supervisionato di sola visione:
esattamente il rimedio dei due encoder affiancati, adottato qui perché un
orientamento sbagliato non è più una parola sbagliata.

`````

La discretizzazione sta in poche righe, e il codice serve soprattutto a rendere
visibile che cosa il braccio esegue davvero, che quasi mai è esattamente
l'azione richiesta: è il centro del gradino più vicino.

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

L'ultima riga è l'errore massimo dell'arrotondamento a gradini, che è mezzo
gradino: due decimi di millimetro sulla traslazione, meno di un millesimo di
radiante sulla rotazione, cioè meno di un ventesimo di grado. È un limite noto
e accettabile. Quelli che non si liquidano con un numero sono altri tre, e
conviene elencarli, perché fra il video di una dimostrazione e un impianto che
lavora ci sono tutti e tre.

Il primo è la **frequenza**. Un modello da decine di miliardi di parametri emette
fra uno e tre comandi al secondo, e sceso a qualche miliardo arriva a circa
cinque;
un controllore classico ne emette decine o centinaia. Finché il compito è
afferrare e spostare va bene, per un movimento che deve
reagire in fretta no, e a decidere è la
dimensione del modello, non l'ingegneria del software.

Il secondo è **come si sbaglia**. Una parola sbagliata si rilegge, e il
peggio che capita è che qualcuno la creda. Un movimento sbagliato è già
avvenuto, ha spostato un oggetto vero e magari lo ha rotto; non esiste un
pulsante «rigenera». Un impianto industriale ragiona in tassi di guasto che si
contano in parti per milione, e nessun sistema addestrato per massima
verosimiglianza su meno di un milione di dimostrazioni ha oggi argomenti per
promettere quel numero.

Il terzo sono i **dati**. Le traiettorie non si raccolgono dal web: ognuna
richiede un robot vero e una persona che lo guida, e la scala che si raggiunge è
lontanissima dai miliardi di token del testo. E qui la generalizzazione cambia
natura. Cambiare robot cambia la relazione fra il comando e il movimento, e la
regola con cui il modello decide che cosa fare (la sua **politica**) non si
trasferisce come si trasferisce un prompt. È lo stesso scarto fra il mondo
simulato e il mondo vero (il *sim-to-real*) che il capitolo introduttivo nomina
a proposito di robotica, e nessuna quantità di didascalie lo colma. È anche la
ragione per cui il capitolo sui **world model**, cioè i modelli che si
costruiscono una copia mentale del mondo, è il vicino di casa naturale di questa
pagina: provare in quella copia costa meno che provare sul robot vero.

Resta il fatto che il meccanismo è di una economia notevole. Non c'è
un'architettura per l'azione: c'è la stessa macchina di tutto il capitolo, con
un vocabolario un po’ più largo. E c'è, insieme, il motivo per cui l'azione sta
in coda all'allucinazione e non altrove: un sistema che allucina una forchetta
scrive una parola di troppo, lo stesso sistema che comanda una mano allucina un
movimento.

## Quattro mosse, e una diffidenza

Il capitolo si chiude dove era cominciato, con la domanda su dove si incontrano
i due flussi. **Allineare** due spazi senza fonderli, cioè mandare le foto e le
frasi sulla stessa mappa, dà un modello che cerca e non parla; e uno spazio
allineato non è uno spazio che capisce. **Innestare** un occhio su un modello
che sa già parlare dà un modello che conversa, e ha vinto la saldatura più
povera, perché comprimere significa scegliere prima di conoscere la domanda.
**Fondere** all'ingresso, in un vocabolario solo, dà un modello che produce, al
prezzo di un pre-addestramento da rifare e di un arrotondamento a catalogo che
butta via. **Pagare il dettaglio** è il conto che presenta la risoluzione, dove
ogni pixel in più si trasforma in posto occupato nella sequenza.

E infine **diffidare**, che non è una quinta tecnica ma la disposizione da
tenere davanti alle altre quattro: chiedersi non soltanto dove i due flussi si
sono incontrati, ma se si sono incontrati davvero, o se il modello sta parlando
di una fotografia che non ha guardato.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- La forchetta che non c'era viene dall’**abitudine della
  lingua** che vince sulla fotografia, e si chiama **allucinazione visiva**. Nelle
  didascalie del mondo, accanto a un piatto e a un coltello, una forchetta c'è
  quasi sempre, e chi è addestrato a scrivere frasi plausibili la scrive.
  Guardare resta facoltativo.
- Dare un voto a sei righe di descrizione è rumoroso: bisogna decidere quali
  parole sono affermazioni sul mondo e poi controllarle una a una. La via che
  funziona è **cambiare la domanda**: si chiede «c'è una forchetta?» e si accetta
  solo sì o no.
- Le domande difficili sono quelle sull'oggetto che di solito accompagna
  quelli presenti, non quelle a caso («c'è una zebra?»). Non si guarda **un**
  punteggio: se ne guardano tre e si guarda **quanto scendono**, perché quella
  discesa misura l'abitudine e non la bravura.
- Il voto va letto insieme a **quante volte il modello ha detto sì**: chi dice
  sempre sì sbaglia una domanda su due e ottiene comunque un punteggio migliore
  di chi guarda davvero e sbaglia due volte su cinque.
- Una parte del guaio viene da prima, dalla bilancia: esistono coppie di
  fotografie che una persona distingue in un istante e che l'encoder misura quasi
  uguali. La differenza c'è ancora, nel numero che la bilancia scrive, ma è così
  piccola che nessuno ha mai insegnato al modello a guardarla; e lui, invece di
  dire «non lo so», riempie il buco con l'abitudine.
- Tre rimedi, nessuna cura: farsi dire **anche dove** (le coordinate l'abitudine
  non le regala), **chiedere due volte** e tenere la differenza fra occhi aperti e
  occhi chiusi, far **ricontrollare** la risposta da qualcuno di indipendente.
  Riducono, non eliminano.
- Gli stessi pezzi comandano un braccio: si taglia ogni comando in 256 gradini
  come le tacche di un righello e ogni gradino diventa una parola, così muoversi
  è scrivere. Restano il passo del righello, i pochi comandi al secondo, i dati
  che nessuno regala sul web, e il fatto che una mossa sbagliata è già avvenuta.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L’**allucinazione visiva** non è un errore di percezione: la perdita si
  scompone in un **priore linguistico** più un **contributo visivo**, e dove la
  didascalia è già prevedibile dal testo il gradiente che spinge a guardare è
  debole. Guardare resta facoltativo.
- Le due misure sono **costrutti diversi**, non due letture della stessa
  grandezza. **CHAIR** {cite}`rohrbach2018object` conta gli oggetti allucinati in
  una descrizione libera (generativo, rumoroso, sensibile all'istruzione e alla
  lunghezza); **POPE** {cite}`li2023evaluating` misura a che cosa il modello
  acconsente su domande **binarie**, con gli assenti scelti a caso, per frequenza
  o per **co-occorrenza**: il divario fra le tre condizioni misura il priore, non
  la bravura. Si leggono insieme.
- Il test va **bilanciato** e letto con la **quota di sì**: un modello che
  risponde sempre «sì» non guarda mai e ottiene comunque un F1 di $0{,}667$, più
  di un modello che guarda davvero e sbaglia due volte su cinque ($0{,}60$).
- Anche il **benchmark** può rispondere senza aver guardato: molte domande sono
  risolvibili dal solo testo e i test pubblici finiscono nei corpora di
  addestramento {cite}`chen2024mmstar`. Il controllo che costa meno di tutti è
  rieseguire la prova **a immagine tolta** e riportare il divario.
- Una parte del difetto è **a monte**: esistono coppie di immagini con embedding
  contrastivi quasi identici (coseno oltre $0{,}95$) che un encoder di sola
  visione separa nettamente {cite}`tong2024eyes`. La differenza sopravvive, ma
  con un margine così sottile che nulla, in addestramento, ha insegnato al
  decoder a leggerlo: il modello rompe il pareggio con il priore. È il limite
  composizionale della prima sezione {cite}`radford2021learning` visto dal lato
  dell'immagine.
- Tre rimedi, nessuna cura: **ancorare** la risposta alle coordinate (che il
  priore non sa indovinare), **decodificare per differenza** fra la
  distribuzione con l'immagine e quella con l'immagine degradata (trattenuta da
  una soglia di plausibilità), una **seconda passata** di verifica (utile solo se
  indipendente). Riducono, non eliminano.
- Discretizzando i comandi di un robot in 256 gradini per grado di libertà,
  l’**azione diventa una sequenza di token** e tutta la macchina del capitolo si
  riusa {cite}`brohan2023rt2`, {cite}`kim2024openvla`. Restano il passo di
  quantizzazione, la frequenza di controllo di pochi hertz, i dati che non si
  raccolgono dal web, e un modello di errore in cui la mossa sbagliata è già
  avvenuta.
```

`````

Ci portiamo dietro due cose, che poi sono la stessa vista da due lati: l'azione
si può scrivere, e chi scrive azioni sbaglia come sbaglia chi scrive parole,
cioè con sicurezza e senza accorgersene. Quello che qui nessuno fa è il resto
del mestiere: decidere quando è il momento di agire, mettere in fila le mosse di
un lavoro lungo, tenere il conto di che cosa si è già provato. Comincia da lì il
capitolo sugli **agenti**.
