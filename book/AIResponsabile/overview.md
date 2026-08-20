# AI responsabile: la tecnologia non è neutra

Nel maggio del 2016 la redazione investigativa di ProPublica pubblica
un'inchiesta destinata a diventare un caso di scuola
{cite}`angwin2016machine`. Al centro c'è **COMPAS**, un software venduto ai
tribunali statunitensi che assegna a ogni imputato un punteggio di rischio: la
probabilità, stimata da un algoritmo, che quella persona torni a delinquere.
Non è un dettaglio burocratico: quei punteggi finiscono sotto gli occhi dei
giudici quando decidono se una persona aspetterà il processo a casa o in
carcere, se concedere la libertà vigilata, quanti anni di pena dare.
Julia Angwin e i suoi colleghi ricostruiscono i punteggi di oltre settemila
imputati della contea di Broward, in Florida, e li confrontano con ciò che è
successo davvero nei due anni successivi. Il risultato è netto: tra gli
imputati che *non* avrebbero commesso nuovi reati, quelli neri venivano
etichettati «ad alto rischio» quasi il doppio delle volte rispetto ai bianchi.
La macchina, pensata per essere più imparziale di un giudice in carne e ossa,
aveva ereditato un pregiudizio.

Ereditato da dove, se nessuno lo aveva scritto? Da ciò che c'era scritto nei
dati, ed è il punto che regge tutto il capitolo.
Un sistema del genere non impara chi ha commesso un reato: quel dato non esiste
in nessun archivio. Impara chi è **stato arrestato**, che è un'altra cosa. Se
in un quartiere passano più pattuglie, lì risultano più reati anche quando non
ce ne sono di più, e chi ci abita entra nello storico con più precedenti. Il
modello legge quel registro e ne ricava una regolarità che sui dati è vera e
sulle persone è ingiusta.  Chi misura con
un metro storto ottiene numeri storti, per quanto impeccabile sia il conto che
ci fa sopra.

Un anno prima, in un laboratorio del MIT, la ricercatrice Joy Buolamwini si
era imbattuta in un problema più intimo: i sistemi commerciali di analisi del
volto non riconoscevano la sua faccia. Funzionavano, ma solo se indossava una
maschera bianca. Nel 2018, con Timnit Gebru, misura il fenomeno in modo
sistematico su tre prodotti in commercio, nello studio *Gender Shades*
{cite}`buolamwini2018gender`: gli stessi sistemi che sbagliano a classificare
il genere in meno di un caso su cento per gli uomini dalla pelle chiara
arrivano a sbagliare in oltre un terzo dei casi per le donne dalla pelle
scura. Non un errore casuale, distribuito a caso: un errore che colpisce
sempre gli stessi.

Questi due episodi dicono la stessa cosa, ed è la tesi di apertura di questo
capitolo: **un modello non è uno strumento neutro**. Impara dai dati, e i dati
portano dentro di sé la storia, le disuguaglianze e i punti ciechi di chi li ha
prodotti e raccolti. Un algoritmo può essere impeccabile nel codice e ingiusto
nell'effetto.

`````{tab} Elementare

Una bilancia tarata soltanto su uomini adulti non ha nulla di rotto: fa
esattamente il suo mestiere. Ma se ci sale sopra un bambino segna un peso
sbagliato, perché è stata calibrata su un mondo che il
bambino non lo prevedeva. Un modello di intelligenza artificiale funziona così:
impara da esempi, e se gli esempi ritraggono soprattutto un certo tipo di
persone, funzionerà bene su quelle e peggio su tutte le altre. Il
riconoscimento facciale allenato per lo più su volti chiari sbaglia di più sui
volti scuri: gli altri quasi non li ha visti.

Quello della bilancia è il primo dei due modi in cui il guasto entra. Nel caso
del tribunale è in gioco il secondo, e i due non vanno confusi, perché è
confonderli che porta fuori strada. Lì i dati sugli imputati
neri non mancavano affatto: erano tanti. Il problema è che dicevano un'altra
cosa da quella che sembrava. Nel registro non c'è scritto «ha commesso un
reato», c'è scritto «è stato arrestato», e chi viene arrestato dipende anche da
dove passano le pattuglie. È come giudicare quali quartieri sono più rumorosi
contando le segnalazioni al comune: misuri anche, e forse soprattutto, chi ha
l'abitudine di segnalare. Non serve un programmatore in malafede: basta uno
specchio, e uno specchio storto. Il modello riflette il mondo che gli abbiamo
dato da guardare, difetti compresi.

`````

`````{tab} Superiore

Vale la pena mettere in fila i numeri, perché la loro precisione è il punto.
In *Gender Shades* il tasso di errore nella classificazione del genere passava
dallo $0{,}8\%$ per gli uomini dalla pelle chiara fino al $34{,}7\%$ per le
donne dalla pelle scura {cite}`buolamwini2018gender`: un divario di oltre
quaranta volte tra i due estremi, misurato su prodotti di tre grandi aziende.
Nell'inchiesta su COMPAS, tra chi non recidivava, il tasso di **falsi
positivi** (imputati innocui etichettati ad alto rischio) era del $44{,}9\%$
per gli imputati neri contro il $23{,}5\%$ per i bianchi
{cite}`angwin2016machine`.

Il caso COMPAS nasconde però una sottigliezza che ritroveremo per tutto il
capitolo. Northpointe, l'azienda che produceva il software (oggi Equivant),
rispose con un rapporto il cui titolo è già l'argomento tecnico della disputa:
*COMPAS Risk Scales: Demonstrating Accuracy Equity and **Predictive Parity***
{cite}`dieterich2016compas`. A parità di punteggio assegnato, la quota di chi
veniva davvero riarrestato era la stessa per neri e bianchi. Ed era vero. Il
paradosso è che entrambe le parti avevano ragione, e la ragione è un teorema:
quando i tassi di base delle due popolazioni differiscono, la **parità del
valore predittivo** rivendicata dall'azienda e la parità di *entrambi* i tassi
di errore misurata da ProPublica non possono valere insieme
{cite}`chouldechova2017fair`. Non è un difetto risolvibile con codice migliore:
è un vincolo aritmetico.

Due avvertenze, che la sezione sull'equità svilupperà. La prima: «parità del
valore predittivo» non è sinonimo di «calibrazione», e i teoremi che si citano
in questa materia sono tre e dicono cose diverse. La seconda: quei tassi di
base sono tassi di **riarresto**, non di reato, cioè le grandezze già toccate
dal bias di misura di cui sopra. Qui basti notare che «equo» non è una parola
con un'unica definizione tecnica.

`````

In Europa la reazione a questi problemi è stata anche normativa, e ci riguarda
da vicino. Nel marzo del 2023 il **Garante per la protezione dei dati
personali** italiano è stato la prima autorità occidentale a fermare ChatGPT.
Un'autorità non spegne un servizio per capriccio, quindi conviene dire perché.
Contestava, in sostanza, quattro cose: che agli utenti non fosse stato
spiegato quali dati venissero raccolti; che non ci fosse una ragione ammessa
dalla legge per darli in pasto al modello mentre imparava
(l’**addestramento**, che è la fase in cui il modello guarda gli esempi e da
lì ricava le sue regolarità); che le risposte del sistema attribuissero alle
persone fatti non corrispondenti al vero; e che non esistesse alcun controllo
dell'età di chi lo usava. Il servizio tornò disponibile qualche settimana
dopo, con una pagina che spiegava il trattamento dei dati e uno sbarramento
sull'età.

E soprattutto l'Unione Europea ha approvato nel 2024 il primo regolamento
**orizzontale** al mondo sull'intelligenza artificiale, l’**AI Act**
{cite}`euaiact2024`: orizzontale vuol dire che vale per tutti i settori
insieme, invece di essere una regola per la sanità, una per le banche e una per
i trasporti. La sua idea portante è di guardare anzitutto al *rischio* che
ciascun impiego comporta per le persone, più che alla tecnologia in astratto. È
un'idea che regge finché il sistema ha un impiego preciso, e alla fine del
capitolo vedremo che con i modelli buoni-per-tutto, quelli che l'uso non lo
scelgono, ha dovuto piegarsi. Per ora ci dice che l'AI responsabile non è più
solo un tema da conferenza accademica, ma materia di diritto.

## Che cosa vuol dire «responsabile»

«AI responsabile» è un ombrello che copre diverse preoccupazioni, distinte ma
intrecciate. Sono l'ossatura del capitolo, e quindi si nominano subito.

- **Equità (*fairness*)**: il modello non deve sistematicamente svantaggiare
  gruppi di persone in base a genere, etnia, età o altre **caratteristiche
  protette**. L'espressione viene dal diritto e indica i tratti su cui la legge
  vieta di discriminare: «protette» non vuol dire che il problema non possa
  succedere, vuol dire che quando succede è illecito. È il filo che lega COMPAS
  e *Gender Shades*.
- **Privacy**: i modelli si nutrono di dati, spesso personali. Proteggerli
  significa impedire che un sistema riveli chi c'era fra quegli esempi. Può
  farlo in due modi, e li vedremo tutti e due. Il primo è sputare fuori di sana
  pianta un indirizzo o un numero di telefono veri. Il secondo è più di
  nascosto: su una persona che ha già visto il modello risponde con una
  sicurezza insolita, e da quella sicurezza si capisce che era nei dati, il che
  a volte è già un'informazione delicata («era fra i pazienti di quel reparto»).
- **Robustezza e sicurezza**: un modello dovrebbe rispondere in modo
  prevedibile anche davanti a casi insoliti, o costruiti *apposta* per
  ingannarlo. E si può, con una facilità che sorprende: a un'immagine si
  possono cambiare i colori così poco che l'occhio non se ne accorge, e lo
  stesso programma che diceva «panda» dice «scimmia». Le immagini truccate così
  si chiamano **esempi avversari**.
- **Trasparenza**: poter spiegare *perché* un modello ha deciso così. È lo scopo
  dell’**interpretabilità**, di cui il libro parla in un capitolo dedicato: qui
  la trattiamo come uno strumento al servizio della responsabilità, non come fine
  a sé.
- **Allineamento (*alignment*)**: fare in modo che il comportamento di un
  sistema corrisponda davvero a ciò che chi lo usa intendeva, e non alla lettera
  di come gliel'ha detto. È il guaio del genio della lampada, che esaudisce il
  desiderio esattamente come lo hai pronunciato: si dice *allineato* un sistema
  quando quello che fa e quello che volevamo si sovrappongono.
- **Governance**, che si potrebbe tradurre con «chi comanda e chi risponde»: le
  regole, le verifiche fatte da qualcuno che non sia il costruttore, le
  responsabilità legali. Chi paga quando un modello sbaglia? L'AI Act è un primo
  tentativo di risposta.

Nessuna di queste dimensioni si ottiene con un numero da tenere d'occhio o con
un pezzo di software da installare. Sono proprietà del *sistema nel suo
contesto d'uso*, non del solo codice.

Va detto subito anche che cosa resta fuori, perché la prima cosa che viene in
mente sentendo «pericoli dell'intelligenza artificiale» è spesso quella dei
film. C'è chi ritiene che un giorno una macchina molto più capace di noi possa
sfuggirci di mano in modo irreparabile: è il **rischio esistenziale**, e sulle
sue probabilità le opinioni degli esperti vanno da «è fantascienza» a «bisogna
fermare tutto». Qui non ne parliamo, e non perché sia una domanda sciocca:
perché non esistono ancora metodi per misurarlo, e questo è un libro su ciò
che si sa fare. Ci occupiamo dei danni che si possono misurare adesso, quelli
su cui esistono metriche e correzioni. Il dibattito fra le due preoccupazioni,
che è vivo e serio, lo riprendiamo alla fine del capitolo, dove parliamo delle
regole.

## Perché il tema esplode adesso

I pregiudizi nei sistemi automatici non sono una scoperta del 2016: se ne
discuteva già negli anni Novanta. Ciò che è cambiato è la scala. Fino a poco
fa un modello sbagliato era un problema locale; oggi lo stesso modello prende
(o suggerisce) decisioni su **credito, giustizia, sanità e lavoro** per
milioni di persone contemporaneamente, e spesso lo fa in modo opaco, dietro
un'interfaccia che restituisce solo il verdetto.

`````{tab} Elementare

Se sbagli una ricetta nella tua cucina, a cena siete in quattro a rimediare con
una pizza. Se la stessa ricetta sbagliata viene stampata e servita in
diecimila mense nello stesso giorno, il piccolo errore è diventato un disastro.
Con i modelli succede questo: uno solo, addestrato una volta, viene poi usato
milioni di volte. Un difettuccio che su un singolo caso passerebbe inosservato,
moltiplicato su tutta la popolazione, diventa un'ingiustizia su larga scala. In
più, a differenza di un impiegato a cui puoi chiedere «perché mi hai detto di
no?», molti modelli non sanno spiegarsi: sono scatole che restituiscono un sì o
un no, e basta. Potenza, diffusione capillare e opacità, tutte e tre insieme:
ecco perché il problema è diventato urgente proprio adesso.

`````

`````{tab} Superiore

Tre spostamenti quantitativi hanno reso il tema ineludibile.

1. **Potenza**. I modelli in uso hanno da centinaia di milioni di parametri in
   su: catturano regolarità sottili nei dati, comprese quelle che *vorremmo*
   non imparassero (le correlazioni spurie tra caratteristiche protette ed
   esito).
2. **Diffusione**. Lo stesso modello viene servito a scala di popolazione. Un
   *bias* con effetto trascurabile sul singolo caso diventa, per la legge dei
   grandi numeri, un effetto sistematico su interi gruppi sociali: proprio i
   contesti (credito, giustizia, impiego, servizi essenziali) che l'AI Act
   classifica come **ad alto rischio** {cite}`euaiact2024`.
3. **Opacità**. Le reti profonde sono ottime nel *cosa* (l'accuratezza) e
   povere nel *perché*: la funzione appresa è distribuita su milioni di pesi,
   senza un ragionamento leggibile. Cresce così la distanza tra performance
   predittiva e comprensibilità: la tensione che il capitolo
   sull'interpretabilità affronta di petto.

`````

## Quattro modi in cui un modello fa danno

Il resto del capitolo procede lungo tre assi, dagli effetti più visibili a
quelli più strutturali, e quello di mezzo prende due sezioni invece di una.

Prima l’**equità e i *bias***: da dove nasce un pregiudizio (dai dati, dalle
etichette, da come è stato scelto l'obiettivo), come lo si *misura* e quali
tecniche provano a ridurlo, senza illudersi che esista una cura definitiva.
Per misurarlo non serve niente di nuovo: basta un rilevatore di fumo. Le cose
che possono succedere sono quattro, e torneranno di continuo: c'era un
incendio e ha suonato, c'era e non ha suonato, non c'era e ha suonato, non
c'era e non ha suonato. Sono le quattro caselle di una tabella che
{doc}`Valutare un modello </MachineLearning/metriche>` chiama **matrice di
confusione**. Qui la
differenza è una sola, ed è quella decisiva: la tabella si compila
*separatamente per ogni gruppo di persone*, e poi si confrontano. Da lì escono
i due numeri attorno a cui gira tutto il capitolo: quanti dei casi veri il
sistema riesce a prendere (il **tasso di veri positivi**) e quanti falsi
allarmi dà su chi non c'entrava niente (il **tasso di falsi positivi**).

Poi **privacy, robustezza e sicurezza**, che prendono due sezioni. La prima
chiede come un modello possa, senza volerlo, lasciar trapelare i dati su cui è
stato addestrato, e come un avversario possa manipolarne l'input con
perturbazioni impercettibili (gli **esempi avversari** di poco fa) per fargli
sbagliare a comando. La seconda porta la stessa domanda ai modelli di
linguaggio, dove l'attacco non è più un rumore invisibile ma una frase scritta
in italiano: basta nasconderla dentro una pagina web che il sistema andrà a
leggere, e il modello la esegue come se gliel'avesse data chi lo ha costruito.
Sono due facce della stessa domanda: un modello messo davvero nel mondo, quanto
sa tenere un segreto e quanto è facile fargli sbagliare.

Infine **allineamento e governance**: che cosa significa chiedere a un sistema
potente di perseguire *ciò che intendiamo* e non la lettera di un obiettivo
mal specificato; e quale impalcatura di regole, audit e responsabilità
(dall'AI Act in giù) prova a tenere il tutto entro binari accettabili.

## Non solo un problema tecnico

C'è un'ultima onestà da mettere in chiaro fin da subito, e attraversa tutto il
capitolo. Molte di queste domande **non hanno una risposta puramente tecnica**.
Decidere *quale* nozione di equità far valere, *quanto* rischio è tollerabile,
*chi* paga quando un modello danneggia qualcuno: sono scelte sociali e
politiche, che nessuna formula risolve al posto nostro.

Ce n'è una che viene prima di tutte, e che un capitolo tecnico rischia di
saltare: **chi ha deciso di mettere lì quel sistema, e per risolvere quale
problema**. Quando un tribunale adotta un software di punteggi, il problema che
si sta togliendo dal tavolo non è quasi mai «vorremmo essere più giusti»: sono
i tempi, i costi, la pila di fascicoli da smaltire. E c'è un vantaggio in più,
meno confessabile: davanti a un reclamo un numero si difende meglio di una
motivazione scritta, perché sembra non avere un autore. Tenere presente quel
passo dice che «come rendo equo il modello» non è sempre la prima domanda: a
volte lo è «questo compito va dato a un modello?», e la risposta legittima può
essere no.
Non è una domanda che si chiude con una metrica, ma lasciarla fuori significa
dare per scontata proprio la decisione che ha causato tutto il resto.

`````{tab} Elementare

Devi dividere una torta «in modo giusto». Fette identiche per tutti? Fette
proporzionali a quanto ciascuno ha contribuito a comprarla? Di più a chi ha
più fame? Sono tre idee di giustizia ragionevoli, e portano a tagli diversi:
nessun coltello, per quanto affilato, sceglie da solo quale sia quella
corretta.

Con i modelli è identico, e la sezione sull'equità farà i conti che lo
mostrano. Possiamo pretendere che il modello **sbagli allo stesso modo** su
ogni gruppo, oppure che il suo «sì» **valga lo stesso** per tutti, cioè che
quando dice sì ci prenda ugualmente spesso in ogni gruppo. Sono due richieste
sensate, e le vorremmo tutte e due; ma se una certa cosa, nei dati, capita più
spesso in un gruppo che nell'altro, le due richieste litigano e una va lasciata
andare. Quale, è una decisione che spetta alle persone e non alla matematica:
la tecnica dice quali compromessi esistono, sceglierli resta un atto di
responsabilità umana.

`````

`````{tab} Superiore

La torta ha una controparte formale, ed è il risultato di impossibilità che già
affiorava nel caso COMPAS. Nella forma che serve qui: fissato un classificatore
con punteggi di rischio, la **parità del valore predittivo** (a parità di
predizione positiva, stessa probabilità reale di esito nei due gruppi), la
parità dei **falsi positivi** e la parità dei **falsi negativi** non possono in
generale valere tutte e tre insieme quando i tassi di base dei gruppi
differiscono, se non nei casi degeneri {cite}`chouldechova2017fair`. Due se ne
tengono sempre: è la terza a saltare. La sezione sull'equità mostra l'identità
algebrica da cui discende, e distingue questo enunciato dagli altri due che gli
somigliano e che vengono regolarmente confusi con esso.

La scelta di *quale* criterio privilegiare non discende dai dati: è normativa.
La statistica delimita lo spazio delle opzioni e ne espone i costi; qual è il
compromesso «giusto» è una domanda di valori, che va posta esplicitamente e non
nascosta dentro una funzione obiettivo. E, come si è visto sopra, nemmeno le
premesse del conflitto sono neutrali: i tassi di base che rendono il teorema
mordente sono misure, e una misura può essere a sua volta distorta.

`````

Con questo spirito (tecnico dove la tecnica basta, esplicito dove non basta),
entriamo nel merito, cominciando da dove tutto è iniziato in apertura:
l'equità e i pregiudizi che si annidano nei dati.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un modello **non è neutro**: impara dagli esempi che gli diamo, e si porta
  dietro la storia storta che c'è dentro. Due casi da tenere a mente: il
  software dei tribunali americani, che etichettava «ad alto rischio» quasi il
  doppio degli imputati neri innocui rispetto ai bianchi; e i riconoscitori di
  volti, che sbagliavano quasi mai sugli uomini dalla pelle chiara e in oltre
  un caso su tre sulle donne dalla pelle scura.
- Il pregiudizio entra in due modi diversi, e vanno tenuti separati: perché di
  un gruppo ci sono **pochi esempi**, oppure perché quello che è scritto nei
  dati **non è la cosa che credevamo** (nel registro c'è «arrestato», non «ha
  commesso un reato»).
- **AI responsabile** è un ombrello che copre sei cose: equità, privacy,
  robustezza e sicurezza, trasparenza, allineamento (che il sistema faccia
  quello che intendevamo, non la lettera di quel che gli abbiamo detto) e
  regole.
- Perché adesso: i modelli sono potenti, uno solo serve milioni di persone, e
  quasi nessuno sa spiegare perché ha risposto così. Tutte e tre insieme.
- «Giusto» non è una parola sola. Come per la torta da dividere, ci sono più
  idee di equità tutte ragionevoli, e spesso **non si possono avere insieme**:
  quale pretendere è una scelta che spetta alle persone, non alla matematica.
- E prima ancora: qualcuno ha deciso di mettere lì quel sistema. Chiedersi
  *perché* è parte del mestiere, non una digressione.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un modello **non è neutro**: impara dai dati e ne eredita storia, punti ciechi
  e disuguaglianze. COMPAS (falsi positivi $44{,}9\%$ contro $23{,}5\%$) e
  *Gender Shades* (errore fino al $34{,}7\%$ sulle donne dalla pelle scura contro
  lo $0{,}8\%$ sugli uomini dalla pelle chiara) sono i casi-simbolo.
- **AI responsabile** è un ombrello: equità, privacy, robustezza/sicurezza,
  trasparenza (l'interpretabilità come strumento), allineamento, governance.
- Il tema è urgente **adesso** per tre spostamenti insieme (modelli potenti,
  diffusi a scala di popolazione, opachi) con impatto reale su credito,
  giustizia, sanità e lavoro: gli usi «ad alto rischio» dell’**AI Act**
  europeo del 2024.
- Diverse definizioni di equità sono **matematicamente incompatibili** quando i
  **tassi di base** (la frequenza reale dell'esito in ciascun gruppo)
  differiscono: parità del valore predittivo, dei falsi positivi e dei falsi
  negativi valgono due alla volta {cite}`chouldechova2017fair`. E quei tassi
  di base sono a loro volta grandezze *misurate*, non date di natura.
- Perciò l'AI responsabile è anche una **scelta sociale e politica**, non solo
  tecnica: la matematica mostra i compromessi, sceglierli spetta a noi. La
  domanda a monte, *se* quel compito vada affidato a un modello, non è tecnica
  affatto.
```

`````
