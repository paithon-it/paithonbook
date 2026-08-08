# AI responsabile: la tecnologia non è neutra

Nel maggio del 2016 la redazione investigativa di ProPublica pubblica
un'inchiesta destinata a diventare un caso di scuola
{cite}`angwin2016machine`. Al centro c'è **COMPAS**, un software venduto ai
tribunali statunitensi che assegna a ogni imputato un punteggio di rischio: la
probabilità, stimata da un algoritmo, che quella persona torni a delinquere.
Non è un dettaglio burocratico: quei punteggi finiscono sotto gli occhi dei
giudici quando decidono su cauzione, libertà vigilata, entità della pena.
Julia Angwin e i suoi colleghi ricostruiscono i punteggi di oltre settemila
imputati della contea di Broward, in Florida, e li confrontano con ciò che è
successo davvero nei due anni successivi. Il risultato è netto: tra gli
imputati che *non* avrebbero commesso nuovi reati, quelli neri venivano
etichettati «ad alto rischio» quasi il doppio delle volte rispetto ai bianchi.
La macchina, pensata per essere più imparziale di un giudice in carne e ossa,
aveva ereditato un pregiudizio.

Due anni dopo, in un laboratorio del MIT, la ricercatrice Joy Buolamwini si
imbatte in un problema più intimo: i sistemi commerciali di analisi del volto
non riconoscono la sua faccia. Funzionano, ma solo se indossa una maschera
bianca. Con Timnit Gebru misura il fenomeno in modo sistematico su tre prodotti
in commercio, nello studio *Gender Shades* {cite}`buolamwini2018gender`: gli
stessi sistemi che sbagliano a classificare il genere in meno di un caso su
cento per gli uomini dalla pelle chiara arrivano a sbagliare in oltre un terzo
dei casi per le donne dalla pelle scura. Non un errore casuale, distribuito a
caso: un errore che colpisce sempre gli stessi.

Questi due episodi dicono la stessa cosa, ed è la tesi di apertura di questo
capitolo: **un modello non è uno strumento neutro**. Impara dai dati, e i dati
portano dentro di sé la storia, le disuguaglianze e i punti ciechi di chi li ha
prodotti e raccolti. Un algoritmo può essere impeccabile nel codice e ingiusto
nell'effetto.

`````{tab} Elementare

Immagina una bilancia tarata usando soltanto uomini adulti. Non è «cattiva»,
non ha nulla di rotto: fa esattamente il suo mestiere. Ma se ci sale sopra un
bambino segna un peso sbagliato, perché è stata calibrata su un mondo che il
bambino non lo prevedeva. Un modello di intelligenza artificiale funziona così:
impara da esempi, e se gli esempi ritraggono soprattutto un certo tipo di
persone, funzionerà bene su quelle e peggio su tutte le altre. Il
riconoscimento facciale allenato per lo più su volti chiari sbaglia di più sui
volti scuri; l'algoritmo dei tribunali allenato su una storia giudiziaria piena
di disparità le ripropone. Non serve un programmatore in malafede: basta uno
specchio. Il modello riflette il mondo che gli abbiamo dato da guardare, difetti
compresi.

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
capitolo. L'azienda che produceva il software rispose che il suo sistema era
*calibrato*: a parità di punteggio, la probabilità reale di recidiva era la
stessa per neri e bianchi. Ed era vero. Il paradosso è che entrambe le parti
avevano ragione: quando i tassi di base delle due popolazioni differiscono,
calibrazione e parità di *entrambi* i tassi di errore (falsi positivi e falsi
negativi) *non possono valere insieme*. Non è un difetto risolvibile con
codice migliore: è un vincolo matematico. Torneremo
su questo punto: qui basti notare che «equo» non è una parola con un'unica
definizione tecnica.

`````

In Europa la reazione a questi problemi è stata anche normativa, e ci riguarda
da vicino. Il **Garante per la protezione dei dati personali** italiano nel 2023
è stata la prima autorità occidentale a sospendere temporaneamente ChatGPT,
imponendone lo stop finché OpenAI non ebbe chiarito basi giuridiche e tutele per
gli utenti. E soprattutto l'Unione Europea ha approvato nel 2024 il primo
regolamento orizzontale al mondo sull'intelligenza artificiale, l'**AI Act**
{cite}`euaiact2024`: una legge che non guarda alla tecnologia in astratto ma
al *rischio* che ciascun impiego comporta per le persone. Ci torneremo; per ora
ci dice che l'AI responsabile non è più solo un tema da conferenza accademica,
ma materia di diritto.

## Che cosa vuol dire «responsabile»

«AI responsabile» è un ombrello che copre diverse preoccupazioni, distinte ma
intrecciate. Conviene nominarle subito, perché sono l'ossatura del capitolo.

- **Equità (*fairness*)**: il modello non deve sistematicamente svantaggiare
  gruppi di persone in base a genere, etnia, età o altre caratteristiche
  protette. È il filo che lega COMPAS e *Gender Shades*.
- **Privacy**: i modelli si nutrono di dati, spesso personali. Proteggerli
  significa impedire che un sistema riveli (direttamente o per inferenza) chi
  c'era nei dati di addestramento.
- **Robustezza e sicurezza**: un modello dovrebbe comportarsi in modo
  prevedibile anche di fronte a input insoliti o *deliberatamente* costruiti per
  ingannarlo. È il territorio degli **attacchi avversari**.
- **Trasparenza**: poter spiegare *perché* un modello ha deciso così. È lo scopo
  dell'**interpretabilità**, di cui il libro parla in un capitolo dedicato: qui
  la trattiamo come uno strumento al servizio della responsabilità, non come fine
  a sé.
- **Allineamento (*alignment*)**: fare in modo che il comportamento di un
  sistema (specialmente di uno potente e generalista) corrisponda davvero alle
  intenzioni e ai valori di chi lo usa, e non a una loro caricatura letterale.
- **Governance**: le regole, gli audit, le responsabilità legali. Chi risponde
  quando un modello sbaglia? L'AI Act è un primo tentativo di risposta.

Nessuna di queste dimensioni si compra con una singola metrica o una singola
libreria. Sono proprietà del *sistema nel suo contesto d'uso*, non del solo
codice.

## Perché il tema esplode adesso

```{figure} ../figures/rischio-esistenziale-dibattito.svg
:name: fig-spettro-rischio
:alt: "Uno spettro orizzontale delle posizioni sul rischio esistenziale posto dall'AI: a un estremo chi lo considera la minaccia prioritaria, all'altro chi lo giudica una distrazione dai danni già in corso, e in mezzo le posizioni intermedie che assegnano peso diverso ai due tipi di danno."
:width: 100%

Il dibattito non è fra allarmisti e negazionisti. Le posizioni si distribuiscono
con continuità, e la differenza vera è quanto peso si dà ai danni futuri
rispetto a quelli già misurabili oggi.
```

{numref}`fig-spettro-rischio` serve a inquadrare cosa questo capitolo fa e
cosa non fa. Di tutto lo spettro, qui si trattano i danni che si possono
misurare adesso, perché sono quelli su cui esistono metodi, metriche e
correzioni; le posizioni agli estremi sono legittime e restano fuori, non
perché irrilevanti ma perché non sono materia tecnica.

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

1. **Potenza**. I modelli moderni hanno da centinaia di milioni a centinaia di
   miliardi di parametri: catturano regolarità sottili nei dati, comprese
   quelle che *vorremmo* non imparassero (le correlazioni spurie tra
   caratteristiche protette ed esito).
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

## Come è organizzato il capitolo

Il resto del capitolo procede lungo tre assi, dagli effetti più visibili a
quelli più strutturali.

Prima l'**equità e i *bias***: da dove nasce un pregiudizio (dai dati, dalle
etichette, dalla scelta stessa dell'obiettivo), come lo si *misura* (riusando
la matrice di confusione e le metriche di errore già viste nel capitolo di
Machine Learning, applicate però *separatamente a ogni gruppo*) e quali
tecniche provano a mitigarlo, senza illudersi che esista una cura definitiva.

Poi **privacy, robustezza e sicurezza**: come un modello può, senza volerlo,
lasciar trapelare i dati su cui è stato addestrato; e come un avversario possa
manipolarne l'input con perturbazioni impercettibili (gli **esempi
avversari**) per fargli sbagliare a comando. Due facce della stessa domanda:
quanto è fragile, e quanto discreto, un modello messo davvero nel mondo.

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

`````{tab} Elementare

Pensa a dividere una torta «in modo giusto». Fette identiche per tutti? Fette
proporzionali a quanto ciascuno ha contribuito a comprarla? Di più a chi ha
più fame? Sono tre idee di giustizia ragionevoli, e portano a tagli diversi:
nessun coltello, per quanto affilato, sceglie da solo quale sia quella
corretta. Con i modelli è identico. Possiamo chiedere che sbaglino ugualmente
poco su ogni gruppo, oppure che uno stesso punteggio significhi la stessa cosa
per tutti, ma spesso non possiamo avere entrambe le cose insieme. Quale
garanzia pretendere è una decisione che spetta alle persone, non alla
matematica. La tecnica ci dice quali sono i compromessi possibili; sceglierli
resta un atto di responsabilità umana.

`````

`````{tab} Superiore

La torta ha una controparte formale, ed è il risultato di impossibilità che
già affiorava nel caso COMPAS. Fissato un classificatore con punteggi di
rischio, tre criteri di equità ragionevoli, **calibrazione** (a parità di
punteggio, stessa probabilità reale di esito tra i gruppi), **bilanciamento
dei falsi positivi** e **bilanciamento dei falsi negativi**, non possono in
generale essere soddisfatti tutti insieme, se non nei casi degeneri, quando i
tassi di base dei gruppi differiscono. In termini della matrice di confusione
del capitolo di Machine Learning: non si può appiattire il divario di **tasso
di falsi positivi** tra i gruppi e allo stesso tempo mantenere il medesimo
valore predittivo dei punteggi. La scelta di *quale* criterio privilegiare non
discende dai dati: è normativa. La statistica delimita lo spazio delle opzioni
e ne espone i costi; qual è il compromesso «giusto» è una domanda di valori,
che va posta esplicitamente e non nascosta dentro una funzione obiettivo.

`````

Con questo spirito (tecnico dove la tecnica basta, esplicito dove non basta),
entriamo nel merito, cominciando da dove tutto è iniziato in apertura:
l'equità e i pregiudizi che si annidano nei dati.

```{admonition} Da ricordare
:class: important
- Un modello **non è neutro**: impara dai dati e ne eredita storia, punti ciechi
  e disuguaglianze. COMPAS (falsi positivi quasi doppi per gli imputati neri) e
  *Gender Shades* (errore fino al $34{,}7\%$ sulle donne dalla pelle scura contro
  meno dell'$1\%$ sugli uomini dalla pelle chiara) sono i casi-simbolo.
- **AI responsabile** è un ombrello: equità, privacy, robustezza/sicurezza,
  trasparenza (l'interpretabilità come strumento), allineamento, governance.
- Il tema è urgente **adesso** per tre spostamenti insieme (modelli potenti,
  diffusi a scala di popolazione, opachi) con impatto reale su credito,
  giustizia, sanità e lavoro: gli usi «ad alto rischio» dell'**AI Act**
  europeo del 2024.
- Molte definizioni di equità sono **matematicamente incompatibili** quando i
  tassi di base differiscono: non si può avere tutto insieme.
- Perciò l'AI responsabile è anche una **scelta sociale e politica**, non solo
  tecnica: la matematica mostra i compromessi, sceglierli spetta a noi.
```
