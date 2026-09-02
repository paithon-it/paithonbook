# Auto-supervisione: il segnale è già nei dati

Nel 1953 Wilson Taylor, che studiava giornalismo e non calcolatori, aveva un
problema pratico: misurare quanto un testo sia facile da leggere. Le formule in
circolazione contavano sillabe e lunghezza delle frasi, e a lui non
convincevano. Propose allora un metodo diverso e quasi brutale
{cite}`taylor1953cloze`: prendere il testo, cancellare una parola ogni tante e
darlo da completare a dei lettori. Più parole indovinavano, più quel testo era
prevedibile per loro, e quindi facile. Chiamò la procedura *cloze*, dalla
chiusura percettiva di cui parlavano gli psicologi della forma: la tendenza a
completare da sé una figura interrotta.

Taylor voleva misurare i lettori. Sessantasei anni dopo, lo stesso identico
gioco (coprire una parola e farla indovinare) è diventato il modo in cui si
addestrano i modelli di linguaggio, e gli autori di BERT lo dicono in chiaro,
rimandando proprio a Taylor: quel loro esercizio, scrivono, in letteratura si
chiama *compito cloze* {cite}`devlin2019bert`. Nel mezzo sono cambiati **chi lo
fa** e **a che scopo**, non l'esercizio. Non si misura più il lettore, si
fabbrica il lettore.

Quella mossa è comparsa già cinque volte, con cinque nomi diversi e senza che
nessuno si fermasse a dire che era una cosa sola, e oggi regge il
pre-addestramento di quasi tutti i modelli di cui si parla.

## Un compito la cui risposta è già nei dati

Il nome è **apprendimento auto-supervisionato**, e la definizione sta in una
riga: ci si inventa un compito la cui risposta corretta è ricavabile dai dati
stessi, senza che nessuno la scriva. Un compito così si chiama un **pretesto**,
e il nome è onesto: risolverlo non interessa a nessuno. Interessa quello che il
modello è costretto a capire per riuscirci, e che gli resta addosso quando il
pretesto si butta via.

La differenza con l'apprendimento supervisionato dei capitoli precedenti non
sta nell'algoritmo, che è lo stesso, e nemmeno nella rete, che è la stessa.
Sta in **chi scrive la risposta giusta**. Là la scriveva una persona, una per
esempio, a mano; qui la si ricava dal dato con un'operazione meccanica: quale
parola avevo coperto, quale pezzo di immagine avevo ritagliato, quale
fotogramma viene dopo. La risposta c'era già, e noi l'abbiamo solo nascosta per
un momento.

Detta così sembra un trucco contabile. Non lo è, e la ragione è aritmetica.

## Quanta informazione porta una risposta

Le tre grandi famiglie di apprendimento (supervisionato, per rinforzo,
auto-supervisionato) si distinguono di solito per come sono fatte. Conviene
invece guardarle da un'altra parte: per **quanto dice** la risposta con cui il
modello si corregge. Non quanto è giusta: quanto è *grande*.

`````{tab} Elementare

Ci sono tre modi di imparare a riconoscere gli uccelli.

Nel primo, qualcuno ti mostra una fotografia e ti dice una parola:
«cardellino». Hai imparato qualcosa, ma quella parola è tutto quello che hai
ricevuto per quella fotografia: una parola sola, scelta da un elenco di nomi
che qualcuno ha compilato prima.

Nel secondo, ti mostrano la stessa fotografia con mezza immagine coperta e ti
chiedono di disegnare quello che manca. Adesso la risposta giusta è tutta la
metà nascosta e non una parola: con la forma del becco, il colore delle ali,
il ramo che continua, l'ombra che cade dalla parte giusta. Per riempire quel
buco devi aver capito parecchio, e ogni singolo dettaglio che indovini o sbagli
ti dice qualcosa.

Nel terzo, giri tutto il giorno in un bosco a osservare uccelli e la sera
qualcuno ti dice: «oggi hai fatto bene». Punto. È stata una giornata intera, e
di ritorno hai avuto una frase sola, che vale per tutto. Fosse durata un'ora, o
una settimana, la frase sarebbe stata comunque una. Quali degli sguardi che hai
dato erano quelli buoni? Nessuno te lo dice.

La fotografia con la sua parola è l'apprendimento supervisionato. La giornata
nel bosco è l'apprendimento per rinforzo. Il buco da riempire è
l'auto-supervisione, ed è il solo dei tre in cui la correzione che ricevi è
**grande quanto la cosa che stai guardando**.

`````

`````{tab} Superiore

La quantità da guardare è l'informazione portata dal **bersaglio**, cioè dalla
risposta corretta su cui si calcola la perdita. Una scelta fra $K$ possibilità
equiprobabili porta al più $\log_2 K$ bit, che è la definizione di entropia
applicata al caso uniforme, e la ricava
{doc}`Teoria dell'informazione </Matematica/teoria-informazione>`.

Da qui tre conti, e sono conti di **tetto**, non di sostanza:

- un'etichetta su $K = 1000$ classi porta al più $\log_2 1000 \approx 10$ bit,
  e li porta **per immagine**;
- un token su un vocabolario di $V = 128\,000$ porta al più
  $\log_2 128\,000 \approx 17$ bit, ma li porta **per token**, e un esempio di
  pre-addestramento è una finestra di migliaia di token;
- una ricompensa binaria porta **1 bit**, e lo porta **per episodio**, cioè per
  l'intera traiettoria, comunque lunga sia.

L'ultima riga è quella che decide tutto, e non per la dimensione del numero ma
per il denominatore: nel supervisionato e nell'auto-supervisionato il bersaglio
si misura per esempio o per token, nel rinforzo per **episodio**. Un episodio
può essere una mossa oppure diecimila.

`````

Il conto lo facciamo fare al calcolatore, così si può rifare e discutere.

```python
from math import log2

# Quanta informazione porta AL PIU' il bersaglio, cioe' la risposta giusta su
# cui il modello si corregge. E' il tetto del canale: quanto ci passa davvero
# e' un'altra domanda.

def bit_per_scelta(n):
    """Una scelta fra n possibilita' equiprobabili vale log2(n) bit."""
    return log2(n)

CLASSI = 1000         # ImageNet: una foto, una categoria fra mille
VOCABOLARIO = 128000  # un tokenizzatore di oggi
FINESTRA = 8192       # token in un esempio di pre-addestramento
PASSI = 10000         # passi di una partita prima del verdetto

etichetta = bit_per_scelta(CLASSI)
token     = bit_per_scelta(VOCABOLARIO)
testo     = token * FINESTRA
rinforzo  = 1.0       # vinto o perso: una risposta binaria per partita

print(f"{'compito':34s} {'bit per esempio':>16s}")
print("-" * 51)
print(f"{'etichetta su ' + str(CLASSI) + ' classi':34s} {etichetta:16.1f}")
print(f"{'un token su ' + str(VOCABOLARIO):34s} {token:16.1f}")
print(f"{'una finestra di ' + str(FINESTRA) + ' token':34s} {testo:16.1f}")
print(f"{'vinto o perso, a fine partita':34s} {rinforzo:16.1f}")
print()
print(f"la finestra di testo vale {testo/etichetta:,.0f} etichette".replace(",", "."))
print(f"e {testo/rinforzo:,.0f} verdetti di fine partita".replace(",", "."))
print(f"spalmato sui {PASSI} passi della partita, il verdetto")
print(f"vale {rinforzo/PASSI:.4f} bit per passo")
```

```text
compito                             bit per esempio
---------------------------------------------------
etichetta su 1000 classi                       10.0
un token su 128000                             17.0
una finestra di 8192 token                 138983.7
vinto o perso, a fine partita                   1.0

la finestra di testo vale 13.946 etichette
e 138.984 verdetti di fine partita
spalmato sui 10000 passi della partita, il verdetto
vale 0.0001 bit per passo
```

Le due righe da confrontare non sono la prima e l'ultima, che stanno in un
rapporto di dieci a uno: sono la **terza** e l'ultima. La terza riga è il brano
di testo che il modello legge in un colpo solo, ottomila pezzetti di parola:
vale quasi quattordicimila etichette e centotrentanovemila verdetti di fine
partita, e sono i due rapporti che il programma stampa in fondo. È la ragione
per cui una fotografia etichettata «vale» poco e ce ne vogliono milioni, mentre
una pagina di testo che nessuno ha mai guardato può bastare a insegnare
qualcosa.

Adesso però va detta la cosa che rende il conto onesto, perché senza di essa
quei numeri prometterebbero più di quanto possono mantenere.

`````{tab} Elementare

Quei numeri dicono **quanto è grande la risposta giusta**, non quanto il
modello ne ha capito. Sono il diametro del tubo, non l'acqua che ci passa.

E il diametro è già generoso. Il conto tratta ogni dettaglio della metà coperta
come una sorpresa, e sorprese non sono: se il ramo si vede per un tratto,
l'altro tratto lo disegni senza pensarci, e da quel pezzo non hai imparato
niente.

Anche dove la sorpresa c'è, chi riempie il buco può cavarne pochissimo: sfuma
tutto, azzecca una macchia del colore giusto e passa oltre. La risposta era
grande, quello che ha imparato è piccolo. E all'opposto, un solo «hai
sbagliato» detto al momento giusto può insegnare più di mille foto etichettate
male; solo che il momento giusto è la parte difficile, perché nel bosco quella
frase arriva la sera, quando gli sguardi da correggere sono ormai centinaia.

Il confronto regge sui rapporti grossi, quelli da dieci volte in su, e sulla
tendenza; non sui decimali. E va usato per quello: dice dove c'è **spazio** per
imparare, non quanto si impara davvero.

`````

`````{tab} Superiore

Tre precisazioni, e sono tutte nella stessa direzione.

La prima: $\log_2 K$ è l'entropia della distribuzione **uniforme**, cioè un
massimo. Le etichette reali non sono uniformi e i token nemmeno: l'entropia
condizionata di un token dato il contesto è molto minore di $\log_2 V$, ed è il
limite verso cui un buon modello linguistico spinge la propria perdita, senza
poterlo scendere. Quindi i numeri della tabella sono tetti, e il tetto vero è
più basso.

La seconda: l'informazione del bersaglio è un limite superiore
sull'informazione che il gradiente può trasportare, non una misura di ciò che
la rete acquisisce. Fra le due c'è di mezzo l'ottimizzazione, l'architettura e
la scelta del pretesto, e quanto quella scelta pesi lo ha già mostrato
{doc}`Imparare a vedere senza etichette </VisioneArtificiale/senza-etichette>`:
con le trasformazioni sbagliate un modello risolve il pretesto per scorciatoia
e non impara niente.

La terza, ed è quella che il dibattito sul rinforzo userà: la povertà del
segnale nel rinforzo non è solo una questione di quantità. Un bit per episodio
va anche **assegnato**, cioè distribuito fra i passi che hanno contribuito, e
quel problema (l'assegnazione del credito) è duro in modo indipendente dal
numero di bit.

`````

## La torta, e la parola che ci hanno cambiato dentro

L'argomento ha una forma celebre, e la sua storia dice qualcosa sul campo.

Nel dicembre del 2016, a un convegno, Yann LeCun mostra una diapositiva con
una fetta di torta e una frase: «se l'intelligenza è una torta, il grosso della
torta è l'apprendimento **non supervisionato**, la glassa è l'apprendimento
supervisionato, e la ciliegina è l'apprendimento per rinforzo»
{cite}`lecun2016cake`. L'immagine fa il giro del mondo, e la ciliegina diventa
un modo di dire.

Nel 2019, alla stessa diapositiva, LeCun cambia una parola: dove diceva «non
supervisionato» adesso dice «**auto**-supervisionato». Non è una limatura. La
ragione l'ha scritta lui stesso, insieme a Ishan Misra, in un testo del 2021
che è la formulazione più chiara di tutta questa faccenda
{cite}`lecun2021darkmatter`: «non supervisionato» è un termine mal definito e
fuorviante, perché suggerisce che l'apprendimento non usi supervisione affatto,
mentre in realtà l'auto-supervisione «usa molti più segnali di correzione di
quanti ne usino i metodi supervisionati e per rinforzo standard».

Quella frase è il conto sull'informazione del bersaglio, detto in una riga e
dalla persona che l'ha disegnata, la torta.

Conviene essere precisi su che cosa quell'obiezione colpisce, perché «non
supervisionato» resta una parola giusta in un caso e fuorviante nell'altro.
Colpisce l'uso del termine per i metodi che **prevedono una parte del dato a
partire dal resto**: là un segnale di correzione c'è, ed è quello che rende
l'espressione fuorviante. Non colpisce i metodi che non prevedono niente e si
limitano a descrivere la forma dei dati, cioè il raggruppamento, la riduzione
della dimensionalità e la stima di densità: lì la supervisione manca davvero,
e il nome tradizionale non inganna nessuno. La distinzione è tenuta esplicita
in {doc}`Valutare un raggruppamento </MachineLearning/valutare-un-raggruppamento>`;
qui basti sapere che dei due usi solo il primo è quello contestato, ed è
l'unico che si evita.

```{admonition} Una nota sulla fonte
:class: note
Della diapositiva del 2016 circolano moltissime riproduzioni e la frase è
riportata in modo concorde, ma le lastre originali non sono più reperibili
online. L'argomento non poggia su di esse: poggia sul testo del 2021, che è
firmato, datato e leggibile da chiunque. La torta serve come immagine e come
cronologia, non come prova.
```

## Cinque pretesti, un solo meccanismo

Se l'auto-supervisione è il paradigma, di pretesti se ne sono già costruiti
parecchi senza chiamarli per nome, e la tabella che segue li mette in fila. Le
prime quattro righe sono strada percorsa; l'ultima è quella che viene subito
dopo.

| dove | il pretesto | che cosa se ne tiene |
|---|---|---|
| {doc}`Natural Language Processing </NaturalLanguageProcessing/overview>` e {doc}`Transformer </Transformers/overview>` | coprire una parola, o indovinare la prossima | un modello di linguaggio |
| {doc}`Imparare a vedere senza etichette </VisioneArtificiale/senza-etichette>` | ritrovare il ritaglio gemello, oppure ricostruire i tre quarti coperti | un encoder di immagini |
| {doc}`Imparare senza etichette, nell'audio </Audio/rappresentazioni-auto-supervisionate>` | indovinare il tratto di parlato mascherato | rappresentazioni del suono |
| {doc}`Allineare due spazi </VisioneLinguaggio/allineare-due-spazi>` | riappaiare l'immagine con la sua didascalia | uno spazio comune fra vista e lingua |
| {doc}`World model </WorldModels/overview>` | prevedere come continua la scena | un simulatore interno |

Cinque pretesti diversi e un meccanismo solo, che occupa sei capitoli perché il
linguaggio ne prende due. La colonna di mezzo cambia sempre; la colonna di
destra è sempre la stessa cosa, una **rappresentazione**, cioè il riassunto
interno che il modello si costruisce e che tutto il resto usa come materia
prima. Il pezzo di rete che produce quel riassunto si chiama **encoder**, ed è
esattamente quello che si tiene quando il pretesto si butta.

C'è poi un sesto caso, che non è un capitolo ma un organismo vivo. Nelle
neuroscienze teoriche c'è un modo di guardare al cervello, l’**inferenza
attiva**, secondo cui percepire e agire non sono due mestieri distinti ma lo
stesso mestiere: indovinare che cosa c'è là fuori, e muoversi per indovinare
meglio; gli è dedicata
{doc}`Inferenza attiva </WorldModels/inferenza-attiva>`. Qui serve una frase
sola, perché dice del paradigma qualcosa che nessun sistema artificiale può
dire: imparare «non è fondamentalmente diverso dalla percezione; opera
semplicemente su una scala di tempo più lenta» {cite}`parr2022active`.

In un essere vivente, cioè, non esiste una fase di addestramento separata
dall'uso, con le etichette da una parte e il lavoro dall'altra: c'è una cosa
sola che va avanti sempre, e il bersaglio su cui si corregge è il segnale
successivo. L'auto-supervisione non è quindi un espediente ingegneristico
trovato quando le etichette sono finite: è il modo in cui funziona l'unico
sistema che sappiamo imparare davvero, e che noi abbiamo raggiunto per un'altra
strada.

Una nota di vocabolario, per non inciampare più avanti. Lo stesso obiettivo, in
quella letteratura, circola sotto molti nomi: minimizzare la **sorpresa**,
l’**entropia**, l’**errore di predizione** oppure l’**energia libera
variazionale**. Sono quattro modi di dire la stessa cosa, e la scelta
dipende dal mestiere di chi parla: «errore di predizione» dove si spiegano
segnali cerebrali, «energia libera variazionale» dove si fa apprendimento
automatico.

Da qui il capitolo prosegue in quattro sezioni. Prima le **famiglie**: i quattro
modi di fabbricare un pretesto, letti tutti come risposte diverse a una sola
domanda, che è come si impedisce al modello di rispondere sempre la stessa
cosa. Poi il **collasso e la misura**: che cosa va storto, e come si fa a
sapere se ha funzionato quando non c'è nessun punteggio da guardare. Poi
**capire è accorciare**, che affronta la domanda che le prime due si lasciano
alle spalle, cioè *perché* tutto questo funzioni: la risposta che una parte del
campo dà è che prevedere bene obbliga a comprimere, e comprimere obbliga a
capire. Infine la **ciliegina**, cioè il dibattito su quanto conti
l'apprendimento per rinforzo rispetto alla torta dell'auto-supervisione, che è
la parte in cui persone molto autorevoli non sono d'accordo fra loro, e qui
gli argomenti si riportano con i loro nomi giusti e basta.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **Auto-supervisione** vuol dire inventarsi un esercizio la cui risposta
  giusta è già dentro i dati: coprire una parola e farla indovinare, ritagliare
  un pezzo di foto e farlo ritrovare. Nessuno scrive la risposta, si nasconde e
  basta. L'esercizio si butta via; quello che il modello ha dovuto capire per
  farlo si tiene.
- Il gioco è più vecchio dei calcolatori: nel 1953 serviva a misurare quanto un
  testo fosse facile da leggere. Oggi serve a fabbricare chi lo legge.
- La differenza che conta con gli altri modi di imparare è **quanto è grande la
  correzione** che il modello riceve. Una parola sola («cardellino») per una
  fotografia intera; oppure mezza fotografia da ricostruire, cioè migliaia di
  dettagli; oppure un «bravo» a fine giornata, che vale per tutta la giornata.
- Il conto lo fa per bene un programma che si può rilanciare, e lo fa su un
  caso solo: il brano di testo che il modello legge in un colpo, ottomila
  pezzetti di parola. Quel brano vale quasi quattordicimila parole scritte
  sotto una foto e centotrentanovemila «bravo» di fine giornata, cioè quattro
  zeri di differenza da una parte e cinque dall'altra. La parola e il «bravo»,
  invece, distano appena dieci volte.
- Attenzione a non chiedere troppo a quei numeri: dicono quanto è **grande** la
  risposta, non quanto il modello ne ha capito. Sono il diametro del tubo, non
  l'acqua che ci passa.
- L'immagine famosa è la **torta** di Yann LeCun: il grosso è
  l'auto-supervisione, la glassa è imparare dalle etichette, la ciliegina è
  imparare per tentativi e premi. Nel 2016 la fetta grossa si chiamava «non
  supervisionata»; è LeCun stesso ad averle cambiato nome, perché di
  supervisione ce n'è, e ce n'è molta di più.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L'apprendimento **auto-supervisionato** costruisce un **pretesto** il cui
  bersaglio è ricavabile dal dato con un'operazione meccanica, e ne conserva
  l’**encoder**, non il compito. Algoritmo e architettura restano quelli del
  supervisionato: cambia chi produce il bersaglio.
- Il criterio discriminante è l’**informazione del bersaglio**, e soprattutto il
  suo denominatore: $\log_2 K \approx 10$ bit **per immagine** per
  un'etichetta su $K = 1000$ classi; $\log_2 V \approx 17$ bit **per token**
  per un vocabolario da $V = 128\,000$, cioè circa $1{,}4 \cdot 10^5$ bit per
  una finestra da 8192 token; 1 bit **per episodio** per una ricompensa
  binaria, indipendentemente dalla lunghezza dell'episodio.
- Sono **limiti superiori** in ipotesi uniforme, quindi tetti e non misure:
  l'entropia condizionata reale è più bassa, e fra informazione del bersaglio e
  informazione acquisita ci sono ottimizzazione, architettura e qualità del
  pretesto. Il confronto vale sugli ordini di grandezza.
- Nel rinforzo alla scarsità si somma un problema indipendente,
  l’**assegnazione del credito** fra i passi di una traiettoria: pochi bit, e
  per giunta da distribuire.
- La **torta** di LeCun {cite}`lecun2016cake` data la cornice al 2016 e la sua
  revisione al 2019, quando *unsupervised* diventa *self-supervised*. La
  motivazione è scritta in {cite}`lecun2021darkmatter`: «unsupervised» è
  fuorviante perché l'auto-supervisione «usa molti più segnali di correzione»
  del supervisionato e del rinforzo.
- Il paradigma è già istanziato in cinque ambiti e sei capitoli (linguaggio,
  che ne occupa due, visione, audio, visione-linguaggio, e i world model, che
  vengono subito dopo): qui non si ripetono, si unificano.
```

`````
