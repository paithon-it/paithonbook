# Privacy e robustezza: dati protetti e attacchi avversari

Nel 2021 un gruppo di ricercatori guidato da Nicholas Carlini diede a GPT-2 (il
modello linguistico di OpenAI addestrato su un'enorme raccolta di testi del
web) un gran numero di frasi da cui partire, e si mise a leggere le risposte.
In mezzo al mare di frasi plausibili ne trovarono alcune che *non erano*
plausibili: erano *vere*. Il modello sputava, parola per parola, il nome
completo di una persona
reale, il suo indirizzo, un numero di telefono, un'email: informazioni
comparse una manciata di volte nei dati di addestramento, e da lì
*memorizzate*. Nessuno aveva chiesto al modello di ricordarle: l'aveva fatto
da solo, come effetto collaterale dell'imparare.

Questo episodio apre il secondo dei tre temi annunciati nell'apertura del
capitolo. Dopo l'equità affrontiamo insieme **privacy** e **robustezza**, che
sono due facce della stessa domanda scomoda: un modello messo davvero nel
mondo, quanto sa tenere un segreto e quanto è facile fargli sbagliare? Il filo
conduttore è che nessuna delle due proprietà si aggiunge alla fine come una
vernice. Vanno costruite dentro l'addestramento, e costano accuratezza.

## I modelli ricordano più di quanto vorremmo

Un modello, in fondo, non è altro che i suoi dati di addestramento compressi:
occupa molto meno spazio di quello che ha letto, e per riuscirci qualcosa lo
butta via. È come la compressione **con perdite** di una foto salvata male,
quella che riemerge sgranata: si perde qualche dettaglio in cambio di spazio.
Solo che ogni tanto, invece di riassumere un dettaglio, lo conserva per
intero. Quando quel dettaglio è un'informazione personale, la memorizzazione
diventa una falla di privacy.

In Europa questo non è solo un problema tecnico, perché esiste una legge che
dice che cosa si può fare con i dati di una persona: il **GDPR**, il
regolamento generale sulla protezione dei dati {cite}`eugdpr2016`. Porta
scritte addosso due date, e sono due per una ragione: la legge
*esiste* dal 2016, ma *si applica* dal 2018, perché a chi doveva mettersi in
regola sono stati concessi due anni di tempo. È una distinzione da tenere a
mente per tutte le leggi di cui parla questo capitolo.

Due sue idee servono più avanti. La prima è che per
trattare i dati di qualcuno bisogna
avere una **base giuridica**, cioè una ragione fra quelle che la legge ammette
(il consenso della persona, l'esecuzione di un contratto, oppure il *legittimo
interesse* di chi tratta i dati, che va però motivato e messo per iscritto). La
seconda è che alla persona restano attaccati dei **diritti** che può esercitare
in qualunque momento: farsi dire quali suoi dati ci sono (accesso, art. 15),
farli correggere (rettifica, art. 16), farli cancellare (cancellazione, art.
17), opporsi al trattamento (art. 21).

```{figure} ../figures/gdpr-e-llm.svg
:name: fig-flusso-dati-personali
:alt: "Il percorso di un dato personale dentro un sistema basato su LLM, in quattro stazioni in fila: raccolta, addestramento (dove i dati diventano pesi del modello), inferenza (il prompt che contiene dati personali) e output. Sotto la seconda e la terza stazione pende la base giuridica che le dovrebbe giustificare: legittimo interesse per l'addestramento, contratto o consenso per l'inferenza. In basso i quattro diritti dell'interessato, accesso, rettifica, cancellazione e opposizione, con la nota che dentro i pesi del modello sono difficili da esercitare."
:width: 100%

Le stazioni che un dato personale attraversa, e la legge che le accompagna. La
terza si chiama *inferenza* ed è il momento in cui il modello, già addestrato,
risponde a una domanda. Addestrare un modello e rispondere a una domanda hanno
bisogno di ragioni diverse: quella buona per il primo non vale automaticamente
per la seconda. E i quattro diritti in basso seguono il dato per tutto il
percorso.
```

Il punto che {numref}`fig-flusso-dati-personali` rende difficile da aggirare è
la seconda stazione, dove i dati diventano **pesi**: i milioni di numeri
interni che l'addestramento aggiusta un pochino alla volta finché il modello
non funziona. Nelle altre stazioni un dato è un dato, sta in un archivio, si
trova e si cancella. Nei pesi non c'è più, e c'è ancora: è stato sciolto
dentro quei milioni di numeri, e una volta finito l'addestramento non lo si
toglie senza rifare tutto da capo. È il motivo per cui il diritto alla
cancellazione, che esiste ed è esercitabile, è tecnicamente scomodo proprio
nel punto in cui servirebbe di più. La conseguenza pratica per chi usa questi
sistemi è meno consolante di quanto piacerebbe: si può chiedere a un fornitore
di cancellare i propri dati dagli archivi e dallo storico delle conversazioni,
e in Europa il fornitore deve rispondere; ma se quei dati sono già finiti
dentro un modello addestrato, quel modello resta com'è.

`````{tab} Elementare

Uno studente che invece di capire la materia impara il libro a memoria,
all'esame non ragiona: se gli capita una domanda vista in aula, recita la
pagina. Molti modelli fanno qualcosa di simile con gli esempi rari o
ripetuti: non ne colgono la regola, li imparano di sbieco così come sono. Due
guai ne seguono. Il primo: dando al modello l'inizio di una frase che c'era
nei dati, questo può completarla *identica*; se in quei dati c'era il tuo
indirizzo, può ripeterlo. Il secondo, più sottile: anche senza fargli sputare
nulla, si può spesso indovinare *se una certa persona era nei dati* di
addestramento, osservando che il modello è stranamente sicuro proprio sui suoi
esempi. È come capire che uno studente ha già visto un compito perché lo
svolge troppo in fretta e senza esitazioni. Sapere «Tizio era nel dataset
dell'ospedale» può essere di per sé un'informazione sensibile.

E più memoria ha lo studente, più pagine recita: i modelli grandi si portano
dentro parola per parola molto più dei piccoli, e a una frase basta comparire
una manciata di volte per restare impressa.

`````

`````{tab} Superiore

I due attacchi hanno nomi precisi. Il **membership inference attack**,
formalizzato da Shokri e colleghi {cite}`shokri2017membership`, decide se un
dato campione $\mathbf{x}$
apparteneva o meno all'insieme di addestramento, sfruttando il divario di
comportamento del modello tra ciò che ha visto e ciò che non ha visto:
tipicamente una loss più bassa, o una confidenza più alta, sugli esempi di
training. È l'evidenza empirica dell’*overfitting* discusso nel capitolo di
Machine Learning, qui riletto come vulnerabilità: più un modello si adatta ai
singoli esempi, più li lascia riconoscere. L’**estrazione di dati di
addestramento** è più aggressiva: Carlini e colleghi
{cite}`carlini2021extracting` mostrarono che da
GPT-2 si potevano recuperare *verbatim* sequenze memorizzate (nomi, recapiti,
frammenti di codice) presenti anche una sola manciata di volte nel corpus. La
memorizzazione cresce con la dimensione del modello e con la ripetizione del
dato: un problema strutturale dei grandi modelli linguistici, non un bug
isolato. Serve quindi una nozione di privacy che sia una *garanzia
matematica*, non un rammendo a posteriori.

`````

## Privacy differenziale: rumore calibrato al singolo

Contro una falla del genere non basta rattoppare a valle: serve una nozione di
riservatezza che si possa *garantire* in partenza, e che valga anche contro un
avversario a cui non abbiamo pensato. La risposta più solida nasce nel 2006
nella comunità crittografica, ed è la **privacy differenziale** di Cynthia
Dwork e colleghi {cite}`dwork2006calibrating`. L'idea, elegante, ribalta la
prospettiva: invece di chiedersi «questo output rivela qualcosa?», si chiede
«l'output cambierebbe se un singolo individuo entrasse o uscisse dai dati?». Se
la risposta è «quasi per niente», allora nessun individuo può essere in
pericolo, perché la sua presenza non lascia traccia rilevabile.

`````{tab} Elementare

C'è un vecchio trucco per fare sondaggi su domande imbarazzanti («hai mai
evaso le tasse?») senza mettere in imbarazzo nessuno. Prima di rispondere,
ognuno lancia in segreto una moneta. Se esce testa dice la verità. Se esce
croce non risponde per sé: lancia una seconda volta e dice «sì» se viene testa,
«no» se viene croce. Ora, se qualcuno ha detto «sì», tu non puoi accusarlo:
forse era solo la moneta.

Eppure la percentuale vera di evasori salta fuori lo stesso, e non per magia:
basta fare il conto. Su mille persone, circa cinquecento hanno avuto testa al
primo lancio e hanno detto la verità; le altre cinquecento hanno risposto a
sorte, e di queste la metà avrà detto «sì» per puro effetto del secondo lancio,
cioè duecentocinquanta. Se i «sì» totali sono quattrocento, quelli sinceri sono
$400 - 250 = 150$: centocinquanta su cinquecento persone sincere, cioè il
$30\%$. E quel $30\%$ vale per tutti e mille, perché a decidere chi sarebbe
stato sincero è stata la moneta e non la persona: i cinquecento sinceri sono un
campione a caso di tutto il gruppo. Il rumore si sottrae proprio perché
sappiamo *quanto* ne abbiamo messo, mentre non sappiamo a chi sia toccato. Ogni
individuo ha la sua *negabilità plausibile*; la statistica collettiva
sopravvive.

La privacy differenziale è questa idea resa una garanzia numerica: al risultato
di un calcolo sui dati si aggiunge un pizzico di caso, *calibrato* in modo che
la presenza o assenza di una singola persona non sposti quasi nulla. Una sola
manopola, chiamata $\varepsilon$ (epsilon), regola il compromesso: piccola vuol
dire più rumore e più privacy, grande vuol dire meno rumore e più precisione ma
meno protezione.

`````

`````{tab} Superiore

Un meccanismo randomizzato $\mathcal{M}$ soddisfa la **$\varepsilon$-privacy
differenziale** se, per ogni coppia di dataset $\mathcal{D}$ e $\mathcal{D}'$
che differiscono per un solo individuo e per ogni insieme di esiti
$\mathcal{S}$,

$$
\Pr[\mathcal{M}(\mathcal{D}) \in \mathcal{S}] \;\le\; e^{\varepsilon}\,\Pr[\mathcal{M}(\mathcal{D}') \in \mathcal{S}].
$$

Qui $\mathcal{M}$ è la procedura (randomizzata) che produce l'output;
$\mathcal{D}$ e $\mathcal{D}'$ sono *dataset vicini*, identici a meno di una
riga; $\varepsilon \ge 0$ è il **budget di privacy**. La disuguaglianza dice che
aggiungere o togliere una persona può moltiplicare la probabilità di *qualunque*
esito al più per $e^{\varepsilon}$: con $\varepsilon = 0{,}5$ il fattore è
$e^{0{,}5}\approx 1{,}65$, uno scarto modesto. Una versione rilassata, la
**$(\varepsilon,\delta)$-DP**, ammette un termine additivo $+\,\delta$ con
$\delta$ piccolissimo: un margine sulla disuguaglianza, che si può leggere
informalmente come una piccola probabilità di eccezione (la lettura precisa è un
po’ più debole di così), ed è la versione che serve per i meccanismi gaussiani
usati nel deep learning.

Come si ottiene? Con il **meccanismo di Laplace**. Data una funzione numerica
$f$, se ne misura la *sensibilità*
$\Delta f = \max_{\mathcal{D},\mathcal{D}'} \lVert f(\mathcal{D})-f(\mathcal{D}')\rVert_1$,
cioè quanto al massimo un singolo individuo può farne variare il valore; poi si
restituisce

$$
\mathcal{M}(\mathcal{D}) = f(\mathcal{D}) + \mathrm{Lap}\!\left(\frac{\Delta f}{\varepsilon}\right),
$$

rumore estratto da una distribuzione di Laplace di scala $b = \Delta f/\varepsilon$.
Più il calcolo è sensibile al singolo, o più $\varepsilon$ è piccolo, più rumore
va aggiunto. Il risultato garantisce esattamente $\varepsilon$-DP.

`````

Un esempio concreto vale la definizione, e si parte da *perché* un conteggio
esatto sia già un problema. Vogliamo pubblicare quanti dipendenti di
un'azienda guadagnano oltre una certa soglia. Se pubblichiamo il numero
esatto, $42$, e il mese dopo una persona se ne va e il numero pubblicato
diventa $41$, abbiamo appena detto a chiunque tenesse il conto quanto
guadagnava quella persona. Nessuno ha diffuso il suo stipendio: è bastato
pubblicare due volte una statistica che sembrava innocua.

Il rimedio è pubblicare il conteggio *sporcato*, e quanto sporco serve lo dice
una domanda sola: di quanto può cambiare quel numero una persona da sola? Di
uno, perché una persona o c'è o non c'è. Quello è il metro con cui si dosa il
rumore. Poi si sceglie quanta protezione si vuole, girando la manopola
$\varepsilon$ di poco fa: la mettiamo a $0{,}5$, che è severa. La taglia dello
sporco si ottiene dividendo la prima cosa per la seconda, $1$ diviso $0{,}5$,
cioè **due unità**: più la manopola è piccola, più sporco esce. Al $42$ vero si
somma quindi un numero estratto a caso attorno allo zero, di solito entro un
paio di unità e ogni tanto molto di più.

Quanto sporco è, in pratica? In `numpy` il meccanismo sta in tre righe, ed è
eseguibile così com'è:

```python
import numpy as np
rng = np.random.default_rng(0)

def conteggio_privato(conteggio_vero, epsilon):
    sensibilita = 1.0                       # un individuo cambia il conteggio di 1
    b = sensibilita / epsilon               # scala del rumore di Laplace
    return conteggio_vero + rng.laplace(0.0, b)

vero = 42
stime = [conteggio_privato(vero, epsilon=0.5) for _ in range(5)]
print("vero:", vero, " privati:", np.round(stime, 1))
# vero: 42  privati: [42.6 40.8 37.  35.2 44. ]
```

Su tante pubblicazioni il rumore si annulla, perché è centrato sullo zero e
sbaglia in eccesso quanto in difetto. Ma di pubblicazioni se ne fa una, e sono
i singoli tiri quelli che conviene guardare. Con una taglia di due unità lo
scarto resta entro tre unità in circa tre casi su quattro, ed è una proprietà
del dado che stiamo tirando, non una cosa che si legge dai cinque numeri; e
infatti tre di questi cinque tiri sono lì attorno. Gli altri due no: uno
sbaglia di cinque unità e l'altro pubblica $35$ dove il vero è $42$. È il
prezzo di $\varepsilon = 0{,}5$ su un conteggio piccolo, e si vede a occhio.

Resta da dire con precisione **che cosa** si è comprato, perché la formula
rassicurante («adesso nessuno può sapere se quella persona c'era») è più forte
del vero, e non è quello che la privacy differenziale promette. Anzi: quella
garanzia lì, «dal risultato non si impara nulla su nessuno», è
dimostrabilmente irraggiungibile, perché un dato pubblicato che non insegna
niente a nessuno non serve a niente {cite}`dwork2014algorithmic`. Quello che
si compra è un limite a quanto si può dedurre, non un divieto di dedurre. Il
patto, detto per esteso, è questo: qualunque numero esca, doveva poter uscire
quasi altrettanto facilmente anche se quella persona non fosse stata
nell'elenco. Quanto «quasi» lo decide la manopola, ed è l'altra faccia della
stessa scelta: con $\varepsilon = 0{,}5$ vuol dire che togliendo quella
persona quel numero sarebbe uscito al più $1{,}65$ volte meno facilmente, poco
più di una volta e mezza. Chi guarda il numero pubblicato può quindi farsi
un'idea sulla presenza di quella persona, e quell'idea può spostarsi: ma di
tanto così, il che fa di quel numero un indizio e non una prova.

E c'è una seconda cosa da cui la privacy differenziale non protegge, ed è
quella che sorprende chi la incontra per la prima volta: le conclusioni
**sulla popolazione**. Se uno studio protetto conclude che il fumo causa il
cancro, un fumatore ne subisce le conseguenze (l'assicurazione che alza il
premio, per esempio) tanto se era nello studio quanto se non c'era. La privacy
differenziale non lo impedisce e non pretende di farlo: dice soltanto che la
*sua partecipazione* non ha cambiato quasi nulla. È una distinzione sottile e
va tenuta, perché è il confine esatto fra ciò che questa tecnica garantisce e
ciò che le viene attribuito.

### Portare la privacy dentro l'addestramento

Sporcare un conteggio è facile: è un numero solo, pubblicato una volta. Una
rete neurale è milioni di numeri, e nessuno li pubblica: si aggiustano un
pochino alla volta, migliaia di volte, e ogni volta guardando i dati. Ogni
passo, quindi, è un'occasione in più di lasciare un'impronta. La ricetta che
ha reso praticabile la privacy differenziale in questo mestiere è la
**DP-SGD** di Abadi e colleghi {cite}`abadi2016deep` (le lettere stanno per
*differentially private stochastic gradient descent*, cioè l'addestramento di
sempre con la privacy differenziale incorporata), e cambia due cose sole.

`````{tab} Elementare

Nell'addestramento normale ogni esempio spinge i pesi del modello nella
direzione che riduce il suo errore. Il problema di privacy è che un esempio
*insolito* può dare una spinta enorme e riconoscibile: la sua impronta resta
nei pesi. DP-SGD fa due cose per cancellare quell'impronta. Primo, mette un
**tetto** alla spinta di ogni singolo esempio: per quanto strano sia, non può
spingere più di tanto. Secondo, alla spinta complessiva del gruppo aggiunge un
po’ di **rumore casuale**, così da confondere il contributo dei singoli. Il
modello impara comunque la tendenza generale (la spingono tutti nella stessa
direzione) ma il segno particolare di ciascuno si perde nel rumore. Si paga in
accuratezza, com'è giusto: la privacy non è mai gratis.

Quanto rumore mettere lo decide la stessa manopola di prima, e qui ogni passo
ne consuma un pezzetto: i passi sono migliaia, e alla fine del conto la
protezione rimasta è spesso molto più fiacca di quella che il nome fa
immaginare. Per questo «qui c'è la privacy differenziale» dice poco finché non
si dice dove la manopola è stata girata.

`````

`````{tab} Superiore

Ad ogni passo, su un minibatch, DP-SGD calcola il gradiente della loss **per
ogni esempio separatamente**,
$\mathbf{g}_i = \nabla_\theta \mathcal{L}(\theta, \mathbf{x}^{(i)}, y^{(i)})$, e
lo sottopone a due operazioni. Il **clipping per-esempio** limita la norma di
ciascun gradiente a una soglia $C$,

$$
\bar{\mathbf{g}}_i = \mathbf{g}_i \,/\, \max\!\left(1,\ \frac{\lVert \mathbf{g}_i \rVert_2}{C}\right),
$$

così nessun campione può influire oltre $C$; poi si aggiunge **rumore gaussiano**
alla somma e si media,

$$
\tilde{\mathbf{g}} = \frac{1}{B}\left( \sum_{i} \bar{\mathbf{g}}_i
   + \mathcal{N}\!\big(\mathbf{0},\ \sigma^2 C^2 \mathbf{I}\big)\right),
\qquad
\theta \leftarrow \theta - \eta\,\tilde{\mathbf{g}}.
$$

Qui $B$ è la dimensione del batch, $\sigma$ il *moltiplicatore di rumore*,
$\eta$ il passo di apprendimento e $\mathbf{I}$ l'identità. Il clipping fissa la
sensibilità del passo (nessun esempio la fa esplodere), il rumore gaussiano
fornisce la garanzia; componendo i molti passi con il *moments accountant*
introdotto nello stesso lavoro si ottiene un budget $(\varepsilon,\delta)$
complessivo. Il **compromesso privacy/utilità** è concreto: Abadi e colleghi
addestrano su MNIST con un budget dell'ordine di $\varepsilon \approx 8$
arrivando attorno al $97\%$ di accuratezza, poco più di un punto sotto la stessa
architettura senza privacy ($98{,}3\%$), e la qualità cala via via che si
stringe $\varepsilon$ ($95\%$ a $\varepsilon = 2$, $90\%$ a
$\varepsilon = 0{,}5$).

Quel $\varepsilon \approx 8$ è il punto in cui la privacy differenziale smette
di essere una garanzia e diventa una casella spuntata. Il fattore in gioco non
è più $e^{0{,}5} \approx 1{,}65$: è $e^{8} \approx 3000$. Formalmente, la
presenza di una singola persona può moltiplicare per tremila la plausibilità
di un esito, il che come promessa vale poco più di un rito. Non è un difetto
del lavoro di Abadi, che è esplicito sui suoi numeri; è la cosa da sapere
quando si legge «questo sistema usa la privacy differenziale» senza il valore
accanto, perché i budget dei sistemi in produzione stanno spesso lì o sopra.
Più privacy, meno accuratezza: la manopola è sempre la stessa, e va guardato
dove è girata.

`````

## Federated learning: portare il modello ai dati

C'è una via complementare alla privacy: non proteggere l'output di un modello
addestrato su dati raccolti in un unico posto, ma **non raccoglierli
affatto**. È l'idea del *federated learning*, proposta da McMahan e colleghi
{cite}`mcmahan2017communication` per addestrare la tastiera predittiva di
milioni di telefoni senza spedire a un server ciò che le persone digitano.

`````{tab} Elementare

Il modo ovvio di addestrare un modello su dati di tanti ospedali sarebbe
raccogliere tutte le cartelle cliniche in un unico grande archivio. Ma quelle
cartelle non devono uscire dall'ospedale. Il *federated learning* rovescia il
verso del viaggio: invece di portare i dati al modello, porta il **modello ai
dati**. Il server manda a ogni ospedale una copia del modello; ognuno lo
allena un po’ sui propri pazienti, in casa; poi rispedisce indietro non i
dati, ma solo il modello aggiornato: cosa ha *imparato*, non cosa ha *visto*.
Il server fonde insieme le versioni in un modello migliore, contando di più
quelle degli ospedali con più pazienti, e ricomincia. Le cartelle non lasciano
mai l'ospedale.

Quel «cosa ha imparato», però, non è muto come sembra: dal modello che torna
al server si riesce a volte a risalire ai pazienti che lo hanno allenato. Per
questo prima di consegnarlo lo si sporca con un pizzico di caso, la moneta di
prima, e il server viene costruito in modo da vedere soltanto la somma di tutti
gli ospedali, mai il pacco di uno. Tenere i dati a casa abbassa il rischio, non
lo cancella.

`````

`````{tab} Superiore

L'algoritmo di riferimento è **FedAvg**. A ogni round, il server invia i pesi
correnti $\theta_t$ a un sottoinsieme di $K$ client; ciascun client $k$ esegue
alcune epoche di discesa del gradiente sui propri $n_k$ dati locali, ottenendo
$\theta_{t+1}^{k}$; il server li ricompone con una **media pesata** dalla
numerosità locale,

$$
\theta_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n}\,\theta_{t+1}^{k},
\qquad n = \sum_{k} n_k.
$$

Il vantaggio è duplice: i dati grezzi restano sul dispositivo e comunicare i
pesi ogni tanto costa molto meno che spedire i dati ad ogni passo. Ma
attenzione a non dichiarare vittoria troppo presto: **i gradienti perdono
informazione**. Zhu e colleghi {cite}`zhu2019deep` hanno mostrato che da
un aggiornamento condiviso si possono talvolta *ricostruire* gli esempi che
l'hanno prodotto. Il federated learning va perciò combinato con la privacy
differenziale (rumore sugli aggiornamenti) e con l'aggregazione sicura, che
lascia vedere al server solo la somma dei contributi, mai il singolo.
Decentrare i dati riduce il rischio, non lo azzera.

`````

## Esempi avversari: ingannare la rete a comando

Passiamo dalla discrezione alla fragilità. Nel 2013 Szegedy e colleghi
{cite}`szegedy2014intriguing` scoprirono una proprietà sconcertante delle reti
neurali: si può prendere un'immagine classificata correttamente, aggiungerle
una perturbazione così piccola da essere **invisibile all'occhio**, e far
cambiare idea alla rete con altissima sicurezza. L'anno dopo Goodfellow,
Shlens e Szegedy spiegarono il fenomeno e ne diedero la ricetta più semplice
{cite}`goodfellow2015explaining`. Il loro esempio è diventato un'icona, e lo
riproduce schematicamente la {numref}`fig-esempio-avversario`.

```{figure} ../figures/esempio-avversario.svg
:name: fig-esempio-avversario
:alt: Tre riquadri in fila collegati da un piu e da un uguale. Nel primo una sagoma stilizzata di panda con etichetta panda 58 per cento. Nel secondo una griglia di rumore impercettibile etichettata rho per il segno del gradiente, con sotto la spiegazione che e’ la mappa di dove spingere ogni pixel. Nel terzo la stessa identica sagoma di panda con l'etichetta errata gibbone 99 per cento in terracotta.
:width: 100%

La ricetta di un esempio avversario. All'immagine di un panda, riconosciuta con
il $57{,}7\%$ di confidenza, si somma un rumore impercettibile: il riquadro di
mezzo è quel rumore, e non è casuale, è la mappa di dove spingere ciascun pixel
per danneggiare al massimo il modello. La *stessa* immagine viene poi
classificata «gibbone» (una scimmia) con il $99{,}3\%$ di confidenza. A occhio
nudo le due immagini sono identiche.
```

C'è un dettaglio che di solito passa inosservato: il modello era sicuro
al $58\%$ quando aveva ragione, ed è sicuro al $99\%$ quando ha torto. La
confidenza che stampa non è una misura di quanto sia affidabile, ed è una delle
ragioni per cui non ci si può appoggiare a quel numero come se fosse una
garanzia.

Una parola sulla lettera greca che compare nel riquadro di mezzo, la $\rho$
(si legge «ro»): è di quanto siamo disposti a sporcare ciascun pixel, il
budget della manomissione. Piccola vuol dire invisibile a occhio, grande vuol
dire che l'immagine comincia a sembrare sgranata, e allora l'inganno non è più
un inganno.

`````{tab} Elementare

La cosa controintuitiva è che la perturbazione non è casuale: è costruita *su
misura* per quel modello. Un rumore a caso non farebbe quasi nulla; questo,
invece, spinge ogni singolo pixel nella direzione precisa che aumenta l'errore
della rete, tutti d'accordo nello stesso verso. Presi uno a uno, gli
spostamenti sono minuscoli: non li vedi. Ma sommati su centinaia di migliaia
di pixel, formano una spinta abbastanza forte da scavallare il confine di
decisione. È come far cadere una persona non con una spinta, ma con mille dita
che premono tutte dallo stesso lato di un soffio ciascuna: singolarmente
impercettibili, insieme irresistibili. Ed è specifico della macchina: a noi il
panda resta un panda.

`````

`````{tab} Superiore

Un avviso sui simboli, prima delle formule. In letteratura il raggio della
perturbazione ammessa si scrive $\varepsilon$, la **stessa lettera** del budget
di privacy differenziale: sono le notazioni standard di due campi diversi, e si
incontrano appena privacy e robustezza si raccontano di seguito.
Il raggio lo chiamiamo $\rho$, perché la $\varepsilon$ della privacy
è dentro il nome delle sue definizioni ($\varepsilon$-DP) e rinominare quella
sarebbe peggio. Con $\delta$ l'incrocio si ripete e il rimedio cambia: la
perturbazione è un **vettore** e si scrive $\boldsymbol{\delta}$, mentre il
margine della $(\varepsilon,\delta)$-DP è uno scalare e resta tondo. Dietro le
due soluzioni c'è una regola sola: si rinomina ciò che si può rinominare senza
rompere un nome proprio, e dove non si può si usa la forma dei simboli. In un
articolo sugli esempi avversari quella $\rho$ si chiamerà
$\varepsilon$.

Il metodo si chiama **Fast Gradient Sign Method** (FGSM). Fissati i pesi
$\theta$, invece di derivare la loss rispetto ai parametri (come
nell'addestramento) la si deriva rispetto all’**input**, e ci si muove nella
direzione che la *aumenta*:

$$
\mathbf{x}_{\text{adv}} = \mathbf{x} + \rho \cdot \operatorname{sign}\!\big(\nabla_{\mathbf{x}} \mathcal{L}(\theta, \mathbf{x}, y)\big).
$$

Qui $\mathbf{x}$ è l'input, $y$ l'etichetta vera, $\mathcal{L}$ la loss,
$\theta$ i pesi (congelati), e $\nabla_{\mathbf{x}} \mathcal{L}$ il gradiente
della loss *rispetto all'input*; $\operatorname{sign}(\cdot)$ ne prende il segno
componente per componente e $\rho$ è il budget di perturbazione, cioè la
massima variazione ammessa per singola componente (una norma $\ell_\infty$).
Prendere il solo segno assegna a ogni componente lo stesso spostamento
$\pm\rho$: la perturbazione è impercettibile per pixel, ma allineata al
gradiente e quindi massimamente dannosa. Nell'esempio originale bastava
$\rho = 0{,}007$ per far passare il panda ($57{,}7\%$) a gibbone ($99{,}3\%$), e gli autori annotano che quel valore
corrisponde al bit meno significativo di una codifica a 8 bit *dopo la
conversione in numeri reali operata dalla rete* (la precisazione conta, perché
sul solo intervallo unitario il bit varrebbe $1/255 \approx 0{,}004$). Il
dettaglio non è pedanteria: una perturbazione più piccola non sopravvivrebbe al
salvataggio del file, quindi $0{,}007$ è il minimo che possa esistere, non un
valore scelto sotto una soglia.

FGSM è un unico passo, ed è per questo un attacco *debole*. La sua versione
iterativa è la **Projected Gradient Descent** (PGD) di Madry e colleghi
{cite}`madry2018towards`: si ripete il passo più volte con ampiezza $\alpha$
piccola, riproiettando ogni volta dentro la palla di raggio $\rho$ attorno
all'input originale,

$$
\mathbf{x}^{t+1} = \Pi_{\mathcal{B}(\mathbf{x},\rho)}\!\Big( \mathbf{x}^{t} + \alpha \operatorname{sign}\!\big(\nabla_{\mathbf{x}} \mathcal{L}(\theta, \mathbf{x}^{t}, y)\big) \Big),
$$

dove $\Pi_{\mathcal{B}(\mathbf{x},\rho)}$ è la proiezione sull'insieme
delle perturbazioni ammesse (la palla $\ell_\infty$ di raggio $\rho$
centrata in $\mathbf{x}$). PGD è considerato l'attacco «di primo ordine» più
forte e, soprattutto, la base della difesa: Madry inquadra la robustezza come un
problema **min-max**,
$\min_\theta \mathbb{E}_{(\mathbf{x},y)}\big[\max_{\boldsymbol{\delta} \in \mathcal{B}(\mathbf{0},\rho)} \mathcal{L}(\theta, \mathbf{x}+\boldsymbol{\delta}, y)\big]$,
in cui l'attaccante (il $\max$ interno, risolto da PGD) e il difensore (il
$\min$ esterno, l'addestramento) giocano l'uno contro l'altro.

`````

## Difese e la corsa agli armamenti

Da qui in avanti è una partita a due, e le mosse sono due. Chi attacca cerca
il modo peggiore di rovinare l'immagine; chi difende allena il modello proprio
su quelle immagini rovinate. La seconda mossa è la difesa più efficace che si
conosca, e si chiama **adversarial training**: durante l'addestramento alla
rete non si mostrano solo gli esempi puliti, ma anche le loro versioni
sabotate, rifabbricate a ogni passo con l'attacco più forte disponibile. Non è
la singola spinta di poco fa: è quella stessa spinta data molte volte di
seguito, ogni volta piccolissima e ogni volta ricalcolata, che è il modo per
trovare la manomissione peggiore invece della prima che capita. La rete impara
così a rispondere
correttamente anche sulle immagini manomesse. Funziona, ma ha un prezzo: è
molto più costoso, perché ogni passo di addestramento contiene un piccolo
attacco al suo interno, e migliora la robustezza entro un certo raggio di
perturbazione spesso a scapito dell'accuratezza sugli esempi intatti.

E c'è una parola da maneggiare con cura, «robusto», perché da sola non vuol
dire niente: vuol dire robusto *contro quale attacco* e *dentro quale
perimetro*.

`````{tab} Elementare

Difendersi dagli esempi avversari somiglia a una rincorsa continua. Si propone
una difesa, sembra reggere, e poco dopo qualcuno trova un attacco nuovo che la
aggira. Molte protezioni annunciate negli anni si sono rivelate illusorie, e
quasi sempre per lo stesso motivo: non fermavano l'avversario, gli rendevano
solo difficile capire da che parte spingere. Quando qualcuno ha trovato il modo
di capirlo lo stesso sono cadute quasi tutte: di nove difese presentate a un
convegno del 2018, sette si reggevano su quel trucco, e sei sono state bucate
del tutto. È una *corsa agli armamenti*, e al momento non esiste una difesa
definitiva. L'unica garanzia solida viene dalla **robustezza certificata**, che
non promette «nessuno passerà» ma dimostra, con un teorema, che *dentro un
raggio preciso* attorno a un'immagine nessuna manomissione può cambiare la
risposta. Il raggio è piccolo, e il modo più usato per ottenerlo si paga due
volte: ogni tanto la macchina non se la sente e preferisce tacere, e la
garanzia vale salvo una piccola probabilità di errore, che sceglie chi
certifica. Resta molto più di «finora nessuno c'è riuscito».

E c'è un secondo modo di attaccare, che non prende di mira il modello finito ma
i suoi compiti di scuola. Chi riesce a infilare esempi propri nei dati con cui
il modello viene addestrato può insegnargli di nascosto una parola d'ordine: un
adesivo su un cartello stradale, una parolina dentro un testo. Su tutto il
resto quel modello si comporta benissimo, e nessun collaudo se ne accorge; ma
quando il segno concordato compare, fa quello che vuole chi glielo ha
insegnato. È la ragione per cui conta sapere da dove vengono i dati con cui un
modello è stato costruito, e non solo quanti ne sono stati usati.

`````

`````{tab} Superiore

Prima delle difese, una parola sul **modello di minaccia**, perché attacchi e
formule fin qui vivono tutti dentro la palla $\ell_\infty$.
Quel perimetro non descrive l'avversario: descrive ciò che è comodo
trattare, perché è differenziabile, proiettabile e quindi ottimizzabile. Le
perturbazioni che contano nel mondo non hanno norma $\ell_p$ piccola: una
rotazione, un ritaglio, un'ombra, un adesivo su un segnale stradale, una frase
riformulata. Un modello robusto a $\rho = 8/255$ non è per questo robusto
a nessuna di quelle. Ne segue che «robusto» è sempre una proprietà relativa a un
perimetro dichiarato e a un attacco misurato, mai un attributo del modello.

Fatta questa premessa, l'onestà impone di ricordare che molte difese euristiche
proposte dopo il 2015 sono state poi aggirate: Athalye e colleghi
{cite}`athalye2018obfuscated` mostrarono che davano una falsa sicurezza per
*gradient masking* (offuscavano il gradiente invece di rimuovere la
vulnerabilità) e cadevano appena l'attaccante lo ricostruiva. L'adversarial
training con PGD è tra i pochi ad aver retto, entro la palla in cui è stato
misurato. In parallelo si è sviluppata la **robustezza certificata**, che
fornisce garanzie dimostrabili: il *randomized smoothing* di Cohen e colleghi
{cite}`cohen2019certified`, per esempio, costruisce da qualsiasi classificatore
una versione «lisciata» per cui si prova un raggio $\ell_2$ entro cui la
predizione è invariante.

Anche qui la garanzia va letta per quello che è. Il teorema riguarda il
classificatore lisciato, non quello di partenza; e siccome il lisciato non è
calcolabile esattamente, predizione e raggio si stimano per campionamento
Monte Carlo, con una procedura che può **astenersi** e la cui garanzia vale con
probabilità almeno $1-\alpha$, dove $\alpha$ lo si sceglie. Non è un certificato
deterministico come quelli che si ottengono propagando intervalli o limitando la
costante di Lipschitz. Le certificazioni coprono raggi ancora modesti, ma
spostano comunque il terreno da «non sono riuscito a romperla» a «si dimostra,
salvo una probabilità di errore che scelgo io, che non si rompe».

Gli esempi avversari agiscono in fase di *inferenza*, su un modello già
addestrato. Esiste una minaccia gemella che agisce in fase di *addestramento*:
il **data poisoning**, in cui l'attaccante inietta esempi malevoli nel dataset
per degradare il modello o piazzarvi una **backdoor**; un innesco segreto (un
piccolo adesivo su un segnale stradale, una parola-chiave in un testo) che, se
presente, fa scattare a comando una risposta scelta dall'attaccante, mentre su
tutti gli altri input il modello si comporta normalmente
{cite}`gu2017badnets`. Chi controlla i dati, controlla il modello: un'altra ragione per
prendere sul serio la provenienza dei dati di addestramento.

`````

## Marchiare il sintetico: watermarking e provenienza

Fin qui il problema era che cosa *entra* in un modello; adesso guardiamo che
cosa ne *esce*. Se un modello genera testo, immagini o voce indistinguibili dal
vero, come si riconosce a posteriori che sono stati fabbricati?

```{figure} ../figures/deepfake-watermarking.svg
:name: fig-watermarking-testo
:alt: "Due istogrammi affiancati, che contano quante parole di ciascuna lista compaiono in un brano. A sinistra il testo naturale: barre verde-azzurre (lista verde) e nere (lista rossa) alternate e tutte della stessa altezza, con sotto la scritta «verdi circa 50 per cento: nessuna traccia». A destra il testo con watermark: le barre verde-azzurre sono più del doppio delle nere, e sotto la scritta «verdi circa 70 per cento: eccesso rilevabile». In basso la legenda dei due colori."
:width: 90%

La filigrana su un testo è uno sbilanciamento. Nessuna parola, presa da sola,
è sospetta: è la proporzione sull'intero brano a non essere quella del caso.
```

La regola del gioco è più semplice di quanto la
{numref}`fig-watermarking-testo` faccia sospettare. Prima di
scrivere ogni parola, il modello tira a sorte: divide in due metà tutte le
parole che potrebbe usare, chiama «verdi» quelle di una metà e «rosse» quelle
dell'altra, e poi sceglie un po’ più spesso del normale fra le verdi. Il
sorteggio sembra casuale ma non lo è: come i dadi di un videogioco, esce da un
calcolo che parte da un numero segreto, e chi conosce quel numero rifà la
stessa identica sequenza di sorteggi tutte le volte che vuole. Così chi vuole
controllare un testo rifà tutti i sorteggi, riconta le parole verdi e vede se
sono troppe. In un testo scritto da una persona sarebbero circa la metà, perché
quel sorteggio la persona non lo conosceva.

E il limite si legge nella figura stessa, in filigrana: quello che si misura è
una **proporzione**, quindi serve abbastanza testo perché lo sbilanciamento si
distingua dal caso. Su una frase corta non c'è niente da misurare, e riscrivere
il brano con parole proprie diluisce l'eccesso fino a cancellarlo.

`````{tab} Elementare

C'è un limite più profondo del testo corto e della riscrittura, e non dipende
da chi attacca ma dal testo stesso: il trucco delle due liste funziona solo
dove il modello aveva davvero una scelta. Se sta scrivendo qualcosa di quasi
obbligato (il seguito di «Barack» è «Obama», e non c'è alternativa) allora o
rispetta il sorteggio e scrive una sciocchezza, o scrive la parola giusta e
non lascia traccia. Sul testo pieno di scelte, come un racconto, la marca si
nasconde benissimo; su codice sorgente, citazioni, elenchi di numeri, quasi
per niente.

Sulle **immagini** l'idea è la stessa e cambiano i mezzi: la filigrana nascosta
sposta di pochissimo migliaia di pixel secondo uno schema segreto, così che
l'occhio non veda niente e un rilevatore che conosce lo schema ritrovi il segno
anche dopo una compressione moderata. SynthID di Google DeepMind fa questo su
immagini, audio e video.

La **provenienza dichiarata** fa l'opposto: invece di nascondere, allega. Lo
standard **C2PA** attacca al file un cartellino con scritto chi l'ha creato,
con quale strumento e come è stato modificato. E chi impedisce di scriverselo
da sé, un cartellino così, e appiccicarlo a un video falso dicendo che l'ha
girato una televisione? Un sigillo, che solo chi possiede una certa chiave
segreta sa produrre e chiunque può controllare senza possederla. Se il sigillo
non torna, il cartellino è falso e si vede subito.

La differenza si vede tutta con una foto dello schermo: porta via il
cartellino e lascia la filigrana, un po’ consumata, perché copia i pixel e
butta il resto. In compenso il cartellino racconta una storia, la filigrana
dice soltanto «sono artificiale».

Né la filigrana né il cartellino chiudono la porta. Chi ha tempo riscrive il
testo con altre parole, ritaglia e ricomprime l'immagine finché il segno non si
legge più, e il cartellino lo stacca in un secondo. Quello che si compra è il
prezzo: far passare per autentico un contenuto fabbricato smette di essere
gratis.

`````

`````{tab} Superiore

Sul testo il meccanismo è diverso e istruttivo. A ogni passo di generazione si
partiziona pseudo-casualmente il vocabolario in una lista "verde" e una
"rossa", con un seme derivato dal **token precedente**, e si aggiunge un piccolo
bias ai logit dei verdi. Il testo resta fluido dove le alternative
plausibili sono molte; e su una sequenza lunga la frazione di token verdi si
scosta dalla frazione attesa $\gamma$ (che è un parametro, non una costante: nel
lavoro originale $0{,}5$, $0{,}25$ e $0{,}1$) in modo statisticamente
rilevabile. Il rilevatore non deve conoscere il testo originale né avere accesso
al modello: gli basta ricalcolare le liste e fare un test d'ipotesi
(Kirchenbauer e colleghi {cite}`kirchenbauer2023watermark`).

Che il seme dipenda dal solo token precedente spiega la fragilità alla
riscrittura: cambiare una parola invalida la lista di quella successiva.

E c'è un limite più strutturale, perché non dipende
dall'attaccante ma dal testo: la marca si può nascondere soltanto dove il
modello aveva davvero una scelta, cioè dove l'entropia della distribuzione sul
token successivo è alta. Su testo a bassa entropia (codice, citazioni,
completamenti quasi deterministici) o non si riesce a marcare, o si marca
rompendo il testo, ed è il caso limite in cui «Barack» è seguito da «Obama»
mentre «Obama» è finito nella lista rossa. Lo stesso lavoro lega esplicitamente
il numero atteso di token verdi all'entropia media della sequenza: le sequenze
ad alta entropia si rilevano con pochi token, quelle a bassa entropia ne
richiedono molti di più. I due parametri del metodo (la quota verde $\gamma$ e
l'entità del bias sui logit) comprano forza di rilevazione in cambio di qualità
del testo, e il tetto che possono raggiungere lo fissa l'entropia, non la sola
lunghezza.

Ed è anche il punto debole: **una parafrasi distrugge la marca**. Basta far
riscrivere il testo a un altro modello e la partizione verde/rossa si dissolve.
Sulle immagini, ridimensionamento, ritaglio, ricompressione o una foto dello
schermo erodono il segnale; i metadati C2PA li cancella uno screenshot.

C'è poi un limite di natura teorica, non di implementazione. Zhang e colleghi
{cite}`zhang2023watermarks` dimostrano che, sotto ipotesi plausibili
sull'avversario (sa giudicare la qualità di un contenuto e sa perturbarlo
restando fra le versioni equivalenti), nessun watermark può essere insieme
impercettibile e robusto: si può sempre costruire una sequenza di
trasformazioni che preserva il significato e cancella la marca.

La conclusione onesta è la stessa della crittografia applicata: il watermarking
non stabilisce cosa è vero, **alza il costo di far passare il sintetico per
autentico**. Non esiste il lucchetto inviolabile, esiste il lucchetto che costa
più della refurtiva.

`````

## Le trenta dita, in pratica

Per toccare con mano il fenomeno non serve una rete profonda: basta il più
semplice dei classificatori, una regressione logistica giocattolo (un
modellino che, dato un esempio con trenta caratteristiche numeriche, stima una
probabilità). Le trenta caratteristiche non rappresentano niente in
particolare, sono numeri estratti a caso, e le due risposte possibili si
chiamano $0$ e $1$: fai conto che $1$ voglia dire «pratica da approvare».
L'esperimento: scegliamo un caso che il modello azzecca, poi spostiamo ogni
caratteristica di un soffio, tutte nella direzione che danneggia di più il
modello (trenta piccole spinte concordi, invisibili una per una: sono le dita
del titolo), e guardiamo la predizione ribaltarsi. 

L'esempio non è scelto a mano: il programma prende il primo caso che il
modello azzecca con una fiducia fra l'$85$ e il $95$ per cento. Su un caso in
cui è sicuro al $100\%$ questa spinta non basterebbe a ribaltarlo.

```python
import numpy as np
rng = np.random.default_rng(0)

# --- dataset giocattolo in dimensione d, da un vero modello logistico ---
d, n = 30, 500
w_true = rng.normal(size=d)
X = rng.normal(size=(n, d))
prob = 1.0 / (1.0 + np.exp(-(X @ w_true)))
y = (rng.random(n) < prob).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# --- regressione logistica addestrata con la discesa del gradiente ---
w, b = np.zeros(d), 0.0
for _ in range(3000):
    p = sigmoid(X @ w + b)
    w -= 0.2 * (X.T @ (p - y) / n)
    b -= 0.2 * np.mean(p - y)

# --- scelta dell'esempio per criterio, non per indice: azzeccato e con
#     fiducia alta ma non assoluta (a fiducia 1,00 questo attacco non basta) ---
p_tutti = sigmoid(X @ w + b)
azzeccati = (p_tutti > 0.5) == (y == 1)
fiducia = np.maximum(p_tutti, 1 - p_tutti)
i = np.flatnonzero(azzeccati & (fiducia > 0.85) & (fiducia < 0.95))[0]

x, yt = X[i].copy(), y[i]
p0 = sigmoid(x @ w + b)

# --- FGSM: un passo lungo il segno del gradiente della loss rispetto a x ---
grad_x = (p0 - yt) * w                      # dL/dx per la cross-entropy logistica
rho = 0.15
x_adv = x + rho * np.sign(grad_x)
p1 = sigmoid(x_adv @ w + b)

def verdetto(p):                            # calcolato, non scritto a mano
    return "corretto" if int(p > 0.5) == int(yt) else "SBAGLIATO"

print(f"esempio scelto: i = {i},  vera etichetta y = {int(yt)}")
print(f"originale:  p(classe 1) = {p0:.3f}  ->  predice {int(p0 > 0.5)}  ({verdetto(p0)})")
print(f"avversario: p(classe 1) = {p1:.3f}  ->  predice {int(p1 > 0.5)}  ({verdetto(p1)})")
print(f"spinta: {rho} per caratteristica; lunghezza complessiva"
      f" {np.linalg.norm(x_adv - x):.2f} contro {np.linalg.norm(x):.2f} dell'input")

# --- lo stesso attacco su tutti gli esempi: quanti se ne ribaltano davvero? ---
segni = np.sign((p_tutti - y)[:, None] * w)
p_adv = sigmoid((X + rho * segni) @ w + b)
ribaltati = azzeccati & ((p_adv > 0.5) != (y == 1))
print(f"ribaltati {ribaltati.sum()} dei {azzeccati.sum()} esempi classificati bene"
      f" ({100 * ribaltati.sum() / azzeccati.sum():.0f}%)")
```

L'output mostra il ribaltamento:

```text
esempio scelto: i = 1,  vera etichetta y = 1
originale:  p(classe 1) = 0.890  ->  predice 1  (corretto)
avversario: p(classe 1) = 0.190  ->  predice 0  (SBAGLIATO)
spinta: 0.15 per caratteristica; lunghezza complessiva 0.82 contro 6.00 dell'input
ribaltati 183 dei 443 esempi classificati bene (41%)
```

Il modello passa da una fiducia dell’$89\%$ nella risposta giusta a una
risposta sbagliata. E la terza riga dice quanto è costato: la spinta
complessiva vale $0{,}82$ contro il $6{,}00$ dell'esempio di partenza, cioè
meno di un settimo. Attenzione però a come si sommano quelle spinte, perché
$0{,}82$ non è trenta volte $0{,}15$, che farebbe $4{,}5$. Le trenta spinte
non tirano nella stessa direzione: ciascuna muove una caratteristica diversa,
e spostamenti in direzioni diverse si compongono come nel teorema di Pitagora,
cioè elevando al quadrato, sommando e poi facendo la radice. Il conto si rifà
a mano: $0{,}15$ al quadrato fa $0{,}0225$, moltiplicato per trenta fa
$0{,}675$, e la radice di $0{,}675$ è $0{,}82$.

L'ultima riga è quella che tiene onesto l'esempio, e va letta. Con una spinta
di questa taglia l'attacco ribalta il $41\%$ degli esempi che il modello
classificava bene: è una frazione, non una certezza. Gli altri resistono per lo
più perché la loro fiducia di partenza è troppo alta perché uno spostamento di
questa taglia basti a scavallare il confine. Il fenomeno è reale e non ha
bisogno di essere gonfiato: che quattro casi su dieci si ribaltino con una
spinta invisibile è già una notizia.

Il codice prova un valore solo di $\rho$, quello scelto in partenza. La
{numref}`fig-attacco-epsilon` rifà lo stesso esperimento su tutta la scala, e
mostra la cosa che il prima e il dopo non possono mostrare: il punto preciso in
cui la risposta si ribalta.

```{figure} ../figures/attacco-epsilon.svg
:name: fig-attacco-epsilon
:alt: "Tre pannelli sovrapposti. In alto trenta barre verde-azzurre, i valori delle trenta caratteristiche dell'esempio scelto. In mezzo trenta trattini terracotta, tutti della stessa ampiezza e disegnati alla stessa scala del pannello sopra, dove si vedono minuscoli: sono le spinte che l'attacco somma a ciascuna caratteristica. In basso un grafico della probabilità della classe giusta al crescere di rho: parte da 0,890, scende e attraversa la soglia di 0,5 a rho uguale a 0,089, segnata da una linea tratteggiata terracotta; a rho uguale a 0,15 vale 0,190 e la classificazione è ribaltata."
:width: 100%

Quanto poco basta. Ognuna delle trenta caratteristiche dell'esempio (in alto)
riceve una spinta della stessa ampiezza $\rho$ (in mezzo, disegnata alla stessa
scala: è minuscola), e la probabilità della risposta giusta scende finché passa
sotto la metà. Succede a $\rho = 0{,}089$: da lì in poi il modello dà la
risposta sbagliata.
```

`````{tab} Elementare

Perché spostamenti così piccoli bastano? Perché sono tanti e tutti d'accordo.
Ogni caratteristica si muove appena, ma tutte e trenta si muovono nel verso
che fa salire l'errore del modello, e trenta soffi concordi sono una spinta
vera. Con le immagini va anche peggio: le caratteristiche sono i pixel, cioè
centinaia di migliaia, e più dita premono, meno forte deve premere ciascuna. È
il motivo per cui al panda dell'esempio famoso è bastato un rumore invisibile.

`````

`````{tab} Superiore

Il gradiente della cross-entropia rispetto all'input, per la regressione
logistica, è semplicemente $(\hat{y}-y)\,\mathbf{w}$: è la riga `grad_x` del
codice. E il risultato mostra la spiegazione *lineare* di Goodfellow: il passo
si muove lungo $\operatorname{sign}\!\big((\hat{y}-y)\,\mathbf{w}\big)$ e sposta
il punteggio (il logit) di $\rho\,\lVert \mathbf{w}\rVert_1$ in modulo,
sempre nel verso che fa crescere la loss. Nell'esempio $y=1$ e $\hat{y}<y$,
quindi la direzione è $-\operatorname{sign}(\mathbf{w})$ e il logit *cala* di
$\rho\,\lVert \mathbf{w}\rVert_1 = 3{,}54$ (il codice non lo stampa, ma
`np.linalg.norm(w, 1)` vale $23{,}58$, e $0{,}15 \times 23{,}58 = 3{,}54$):
quanto basta a far scendere la probabilità da $0{,}890$ a $0{,}190$, perché il
logit di partenza era $2{,}09$. È una quantità che cresce con il numero
di dimensioni. In alta dimensione (dove vivono immagini e testi) bastano tante
piccole spinte concordi per scavallare il confine. La stessa formula in PyTorch
si scriverebbe con `x.requires_grad_(True)`, un passaggio `loss.backward()` e
`x + rho * x.grad.sign()`: identica idea, gradiente rispetto all'input calcolato
in automatico.

`````

Privacy e robustezza si somigliano più di quanto la distanza fra i due
argomenti faccia pensare: in nessuna delle due esiste la proprietà «al
sicuro», esiste una garanzia con accanto il suo prezzo e il suo perimetro. È
il criterio con cui leggere anche la sezione seguente, che porta la stessa
domanda ai modelli di linguaggio. Lì l'attacco non sarà più un rumore
invisibile, sarà una frase scritta in italiano che chiunque può leggere; e il
perimetro da difendere non si riuscirà nemmeno a disegnare dentro il modello.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- I modelli **imparano a memoria** le cose rare o ripetute, come lo studente che
  recita la pagina invece di ragionare. Due conseguenze: dandogli l'inizio di
  una frase che c'era nei dati può completarla identica, e si può spesso
  indovinare *se una certa persona era nei dati* osservando che il modello è
  stranamente sicuro proprio sui suoi esempi.
- In Europa una legge dice cosa si può fare con i dati di una persona, e le dà
  il diritto di sapere quali dati ci sono, farli correggere e farli cancellare.
  Il punto scomodo è che dai **pesi** del modello, una volta addestrato, non si
  tolgono senza rifare tutto: si cancellano dagli archivi, non da lì.
- Il trucco della **moneta lanciata prima di rispondere** protegge la singola
  persona e lascia leggere il totale: si aggiunge un po’ di caso, in quantità
  nota. Una manopola decide quanto: più caso, più protezione e meno precisione.
  Ma non promette che di te non si sappia più nulla, promette che la *tua
  presenza* cambi poco le idee di chi guarda; e non ti protegge dalle
  conclusioni sulla popolazione a cui appartieni.
- Un'altra strada è **non raccogliere i dati affatto**: si manda il modello a
  casa di chi li ha, ognuno lo allena un po’ sui propri e rimanda indietro solo
  quello che ha imparato. Riduce il rischio, non lo azzera.
- Si può far sbagliare una rete a comando con **tante piccole spinte concordi**,
  invisibili una per una. Difendersi è una rincorsa: al momento non esiste una
  difesa definitiva, e «robusto» vuol sempre dire robusto contro un attacco
  preciso e dentro un limite dichiarato.
- Chi controlla i dati di addestramento può anche piazzarci dentro una parola
  d'ordine segreta che, quando compare, fa fare al modello quel che vuole lui.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- I modelli **memorizzano** i dati rari o ripetuti: da qui i *membership
  inference* (capire se un individuo era nel training) e l’**estrazione**
  verbatim di dati sensibili dagli LLM. La memorizzazione è overfitting visto
  come falla di privacy.
- La **privacy differenziale** {cite}`dwork2006calibrating` garantisce che
  l'output cambi al più di un fattore $e^{\varepsilon}$ se un individuo entra o
  esce dai dati, aggiungendo rumore (meccanismo di Laplace) calibrato alla
  sensibilità. È un **limite all'inferenza**, non un'impossibilità di dedurre
  ({cite}`dwork2014algorithmic`: *nothing is learned* è irraggiungibile), e non
  copre le inferenze sulla popolazione. Il valore di $\varepsilon$ va sempre
  guardato: $e^{0{,}5}\approx 1{,}65$, ma $e^{8}\approx 3000$.
  **DP-SGD** {cite}`abadi2016deep` la porta nel deep learning con clipping
  per-esempio + rumore gaussiano, a circa un punto di accuratezza su MNIST.
- Il **federated learning** {cite}`mcmahan2017communication` porta il modello ai
  dati invece del contrario (FedAvg); ma i gradienti condivisi perdono
  informazione, e vanno protetti con DP e aggregazione sicura.
- Gli **esempi avversari** {cite}`goodfellow2015explaining` ingannano una rete
  con perturbazioni impercettibili: **FGSM** somma $\rho$ per il segno del
  gradiente della loss rispetto all'input; **PGD** {cite}`madry2018towards` ne è
  la versione iterativa e la base dell’*adversarial training*. Attenzione al
  simbolo: il raggio della perturbazione qui è $\rho$, mentre negli articoli si
  scrive $\varepsilon$, che in questo capitolo è già il budget di privacy.
- La palla $\ell_p$ è una comodità matematica, non il modello di minaccia: la
  robustezza si dichiara sempre con accanto perimetro e attacco.
- Non esiste difesa definitiva: è una **corsa agli armamenti**. La robustezza
  certificata offre garanzie provate ma su raggi piccoli, e nel caso del
  *randomized smoothing* {cite}`cohen2019certified` sono garanzie probabilistiche
  sul classificatore lisciato; *data poisoning* e *backdoor* attaccano invece in
  fase di addestramento.
```

`````
