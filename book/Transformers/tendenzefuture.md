# Tendenze e limiti

Chiudere un capitolo sui Transformer con le previsioni è un esercizio
rischioso: questo campo brucia le profezie in fretta. Più utile fissare le
direzioni di lavoro visibili oggi, e i problemi aperti che le motivano. Perché
il paradosso è proprio questo: mai un'architettura ha funzionato così bene, e
mai è stato così chiaro quanto costa farla funzionare.

## Dove punta la ricerca

Tre cantieri, su tutti, e conviene nominarli prima di scendere in uno.

Il primo è **fare di più con meno**: i grandi modelli sono motori potentissimi
che consumano moltissimo, e buona parte della ricerca è una gara di efficienza,
per farli stare in un telefono invece che in un centro di calcolo. Il secondo è
**unire i sensi**: modelli che leggono, guardano e ascoltano insieme, come
l'assistente a cui mostri una foto e fai una domanda a voce. Il terzo è
**superare i limiti dell'architettura stessa**: l'assemblea in cui ogni parola
parla con ogni altra costa troppo sui testi lunghi, perché raddoppiando le
parole le conversazioni quadruplicano, e questo spinge a cercare modi più
economici di far comunicare le parti di un testo.

Del primo cantiere conviene vedere da vicino il metodo più elegante, che si
chiama **distillazione**: si prende un modello grande e bravo, lo si mette a
fare il maestro, e se ne addestra uno piccolo a imitarlo.

```{figure} ../figures/distillazione-insegnante-allievo.svg
:name: fig-distillazione
:alt: "Un modello maestro, grande, riceve un input e produce non una sola risposta ma una distribuzione di probabilità su tutte le risposte possibili. Un modello allievo, molto più piccolo, viene addestrato a riprodurre quella distribuzione intera invece della sola risposta corretta."
:width: 92%

Perché imparare dal maestro batta imparare dalla risposta giusta. La risposta
giusta dice solo qual è; il maestro dice anche quali errori erano quasi
ragionevoli, e quella è informazione in più.
```

Il dettaglio di {numref}`fig-distillazione` che fa funzionare la distillazione
è **che cosa** passa dal maestro all'allievo. Non la risposta, ma l'intera
graduatoria: davanti a una foto di gatto il maestro non dice «gatto», dice
«gatto quasi certamente, lince un pochino, camion per niente». È
un'informazione che nessun elenco di risposte giuste conterrebbe, perché quella
esitazione fra gatto e lince dice all'allievo che i due si somigliano, e
imparare quali cose si somigliano è metà del mestiere.

`````{tab} Elementare
Gli altri modi di rimpicciolire un modello sono due, e li costruisce il
{doc}`capitolo sull’efficienza </Efficienza/overview>`. Il primo è scrivere
ogni numero con meno cifre; il secondo è togliere di mezzo i numeri che contano
poco. Detti così sembrano
gratis, e non lo sono. Là il prezzo è contato: arrotondare a quattro bit, senza
altri accorgimenti, sposta di quasi un quinto quello che esce da uno strato, e
una rete a cui si tolgono nove pesi su dieci smette di funzionare finché non la
si riaddestra.

C'è poi una strada che non rimpicciolisce niente. Il modello resta grosso, ma
per ogni parola ne accende un pezzo solo, come una redazione che manda
l'articolo di sport a chi si occupa di sport invece di farlo rileggere a tutti
quanti. Sa quello che sanno tutti i suoi redattori, e su ogni articolo spende
quanto ne spende uno.

Sul contesto lungo la ricerca prova invece a far comunicare le parole senza
convocarle tutte insieme, e le due strade più promettenti sono l'attenzione
lineare e i modelli a spazio di stato, che hanno un capitolo ciascuna. Una
terza non tocca il meccanismo e cambia il modo di eseguirlo, tenendo in memoria
meno roba per volta. Sul fronte dei sensi si costruiscono mappe del significato
condivise, dove una foto di gatto e la parola «gatto» cadono nello stesso
punto. Ci si arriva mostrando al modello milioni di immagini con la didascalia
che le accompagna, e chiedendogli di tenere vicine le coppie giuste e lontane
quelle sbagliate. E c'è un cantiere in più, che dieci anni fa nessuno avrebbe
messo in un elenco di ricerca: rendere questi modelli utili e non dannosi, cioè
il post-training, che nel frattempo è diventato una disciplina a sé.
`````

`````{tab} Superiore
Sul fronte dell'efficienza, e per il meccanismo si rimanda al capitolo che gli
è dedicato: **distillazione** (un modello piccolo addestrato a imitare le
uscite di uno grande), **quantizzazione** (pesi a 8 o 4 bit invece che a 32,
dove a quattro bit la perdita smette di essere trascurabile e servono metodi
che facciano più che arrotondare), **pruning**, e architetture
*mixture-of-experts* che attivano solo una frazione dei parametri per ogni
token. Sul fronte del contesto lungo: attenzioni sparse e lineari,
ottimizzazioni di memoria come FlashAttention, e gli *state space model*
(Mamba); a questi ultimi, e alle attenzioni lineari, sono dedicati i due
capitoli che seguono. Sul fronte multimodale: spazi di rappresentazione
condivisi tra testo, immagini e audio, con l'addestramento contrastivo su
coppie immagine-didascalia alla CLIP come collante. A cui si aggiunge il filone
dell’**allineamento**: tecniche (come il fine-tuning con feedback umano) per
rendere i modelli più
utili e meno dannosi, che è oggi un'area di ricerca a pieno titolo, non un
ritocco finale.
`````

## Pensare più a lungo sulle cose difficili

C'è un filone da isolare, perché nasce da un'osservazione così semplice da
sembrare ingenua e perché la sua storia insegna qualcosa su come procede
questo campo.

`````{tab} Elementare

Sessanta piani, e l'ascensore ferma a tutti. La domanda sale in un vassoio, un
bigliettino per parola, a ogni piano qualcuno ci mette mano, e dal tetto esce
la risposta. «Quanto fa due più due» fa il viaggio intero, e un rompicapo da
dieci mosse pure. Noi no: sulle cose facili rispondiamo di getto, sulle
difficili ci fermiamo a pensare.

Nel 2018 qualcuno provò a rifare il palazzo. Le sessanta stanze sono tutte
diverse, e un paio in più non si aggiungono: quel numero è murato. E se la
stanza fosse una sola, e ci si rientrasse? I giri li deciderebbe la domanda:
due per «due più due», venti per il rompicapo. A ogni giro un bigliettino può
dire «io ho finito», e resta lì com'è mentre gli altri continuano.

Nei muri della stanza non è scritto nessun massimo, e sulle frasi lunghe si
vede: davanti a una più lunga di tutte quelle viste in addestramento si fanno
due giri in più, invece di fermarsi in cima perché il palazzo finisce.

Manca un pezzo. Un bigliettino che può fermarsi non ha motivo di farlo, girare
è gratis e un giro in più non fa danno. Perciò all'ingresso si paga un
pedaggio, piccolo, a ogni giro: si paga finché la domanda lo merita, poi si
smette.

Sulla porta c'è una promessa più grossa: con questa stanza si calcola tutto ciò
che è calcolabile. È vera, e sotto ha una riga che quasi nessuno legge: giri
quanti ne servono, senza tetto. A quella condizione ce la fa anche il palazzo
di sessanta piani, se la risposta che esce dal tetto la si rimette al
pianterreno quante volte si vuole, e si conta senza mai arrotondare. A decidere
è quanti giri si concedono, più della forma dell'edificio.

La stanza sola non prese piede: farci sessanta giri costa le stesse ore di
sessanta stanze, perché il lavoro è lo stesso, ma i mobili da scegliere sono
quelli di una stanza sola, e con la stessa spesa si impara meno. Intanto
tirare su palazzi più alti funzionava benissimo.

È tornata da una strada inattesa. Anche i modelli che «ragionano» prima di
rispondere spendono di più sulle domande difficili, solo che invece di girare
in silenzio a porta chiusa scrivono i passaggi su un foglio, una riga alla
volta. Ogni riga va prodotta e poi riletta, quindi costa; in compenso si
insegna meglio, di quaderni coi passaggi svolti in mezzo ne esistono a
montagne. Quale strada convenga resta aperto: la porta chiusa costa meno, il
foglio lascia una traccia da leggere. Che la traccia racconti quello che è
successo davvero nella stanza, però, non è detto.

`````

`````{tab} Superiore

L’**Universal Transformer** {cite}`dehghani2019universal` (l'articolo è del
luglio 2018, presentato a ICLR l'anno successivo, che è la data della voce in
bibliografia) sostituisce gli $L$
strati distinti con **un solo blocco applicato ricorrentemente in profondità**,
cioè con i pesi legati fra le iterazioni. La motivazione dichiarata è
recuperare il *bias induttivo* ricorrente che il Transformer aveva buttato via
insieme alla ricorrenza temporale, e che serve sui compiti a struttura
gerarchica e sulla generalizzazione a lunghezze non viste in addestramento.

Sopra ci mettono l’**Adaptive Computation Time** di Graves
{cite}`graves2016adaptive`: a ogni iterazione,
per **ogni posizione**, una piccola unità emette una probabilità di
arresto; le posizioni che si fermano vengono copiate invariate mentre le altre
continuano a essere aggiornate, e una penalità sul numero di passi (il *ponder
cost*) impedisce di pensare all'infinito. Il calcolo diventa così
**condizionato all'ingresso** invece che fissato dall'architettura.

C'è un punto che il titolo lascia intuire, e va detto con precisione, insieme a
quanto sia solido. Gli autori dimostrano che
legare i pesi e iterare rende il modello **Turing-completo**, ma la loro stessa
formulazione lo dice **sotto certe ipotesi**, e l'ipotesi che porta il peso è
la solita: un numero di passi non limitato a priori. Il ragionamento è che un
Transformer standard esegue un numero di passi sequenziali fissato
dall'architettura, mentre una ricorrenza in profondità lo fa dipendere dai dati.

Sarebbe però scorretto presentare la Turing-incompletezza del Transformer
standard come un fatto assodato, perché la letteratura contiene anche il
risultato opposto: Pérez, Barceló e Marinković {cite}`perez2021attention`
dimostrano che il Transformer
encoder-decoder è Turing-completo, con precisione aritmetica arbitraria e un
numero **illimitato di passi di decodifica**. E quell'ultima ipotesi è
esattamente la generazione autoregressiva, cioè la strada che percorrono oggi i
modelli lasciati scrivere finché serve. Le due prove non si contraddicono,
cambiano le ipotesi;
ma la morale da portarsi via è che «quanti passi può fare» conta più di «come
sono legati i pesi», e che una frase secca sull'espressività di
un'architettura, senza le sue ipotesi accanto, è quasi sempre una frase
sbagliata.

L'idea è rimasta a lungo marginale e oggi è di nuovo centrale, arrivata però
dall'altra parte. I modelli che **ragionano** allocando più calcolo in
inferenza fanno la stessa cosa nello **spazio dei token** invece che nello
spazio latente: generano una catena di passi intermedi, e più il problema è
difficile più ne generano. Le due vie hanno un compromesso opposto e non
risolto. Il calcolo latente è più economico (nessun token da produrre e
rileggere) e **non è ispezionabile**; quello in token costa di più, è più
facile da addestrare con la supervisione esistente, e lascia una traccia che si
può leggere, il che nel {doc}`capitolo sull'interpretabilità
</Interpretabilita/overview>` è tutt'altro che un dettaglio. Che la traccia sia
poi una descrizione *fedele* del calcolo svolto è una domanda a sé, e la
risposta corrente è: non necessariamente.

`````

## I limiti che restano

Un elenco onesto, da tenere accanto agli entusiasmi:

- **Costo**: addestramento e inferenza dei modelli maggiori richiedono risorse
  (economiche, energetiche, di hardware) concentrate in poche aziende; la
  ricerca indipendente lavora per necessità su scala ridotta.
- **Dati**: le grandi raccolte di testo prese dal web (i *corpora*) si stanno
  esaurendo come fonte gratuita di materiale di qualità, e portano con sé le
  distorsioni sistematiche di ciò che è stato scritto online, i **bias**, che i
  modelli assorbono insieme al resto.
- **Affidabilità**: le allucinazioni (risposte fluenti ma false) derivano dal
  mestiere stesso di questi modelli, che è scrivere una parola alla volta
  scegliendo ogni volta la continuazione più probabile; probabile non vuol dire
  vero, e nulla nel meccanismo distingue le due cose. Mitigarle (con il
  recupero di fonti esterne, la verifica, la calibrazione) è un problema aperto.
- **Comprensione**: su cosa i modelli *capiscano* davvero il dibattito
  scientifico è tutt'altro che chiuso, e attribuire loro intenzioni o
  ragionamento senza prove è un errore prima ancora che una scortesia verso i
  fatti. Prudenza, qui, è il modo in cui si tratta
  un'affermazione che non si sa ancora come verificare.

## Niente di nuovo, tutto in un ordine nuovo

Con questo capitolo si chiude un tratto del percorso tecnico del libro: dai
neuroni del percettrone all'attenzione, ogni pezzo dei Transformer è un
concetto che hai già incontrato, montato in una configurazione nuova.
L'evidenziatore che pesa le parole, il posto numerato che dice l'ordine, la
riunione e il lavoro individuale che si alternano piano dopo piano, la mappa
del significato dove le cose simili stanno vicine, il provare-e-correggere che
sistema i numeri un'inezia alla volta: se hai seguito queste immagini hai
seguito tutto, e questi sono i loro nomi propri, quelli che troverai scritti
altrove (attenzione, positional encoding, feed-forward, embedding, discesa del
gradiente). È la lezione migliore di questa storia: le "rivoluzioni" dell'AI, viste da vicino,
sono quasi sempre ricombinazioni ingegnose di idee semplici, rese possibili da
più dati e più calcolo. Chi conosce le idee semplici non insegue le mode: le
legge.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- La ricerca punta su **efficienza** (la *distillazione*, cioè l'apprendista
  che impara dal maestro; la *quantizzazione*, cioè scrivere ogni numero con
  meno cifre per farlo stare in un telefono; e i modelli che tengono acceso
  solo un pezzo di sé per ogni parola), **contesto lungo** (modi più economici
  di far parlare fra loro le parti di un testo, fino ai modelli a spazio di
  stato)
  e **multimodalità** (leggere, guardare e ascoltare con lo stesso meccanismo).
- Un Transformer spende **lo stesso calcolo** su ogni ingresso, facile o
  difficile che sia. Nel 2018 si provò a togliere quel vincolo con un piano
  solo riapplicato più volte, lasciando a ogni parola il diritto di dire «io ho
  finito» e fermarsi. Allora non prese piede, e l'idea è tornata attuale dalla
  parte opposta: i modelli che ragionano spendono più calcolo sulle difficili
  **scrivendo** i passi invece di girare in silenzio. La prima strada costa
  meno, la seconda lascia una traccia da leggere, che però non è detto racconti
  quello che è successo davvero.
- I limiti sono strutturali: costi concentrati, bias dei dati, e il fatto che
  un modello che sceglie ogni volta la continuazione più probabile non ha modo
  di distinguere il probabile dal vero. Su che cosa questi modelli
  «capiscano» davvero il dibattito è tutt'altro che chiuso.
- Tutti gli ingredienti dei Transformer li hai già studiati in questo libro:
  ciò che è nuovo è la composizione, non i mattoni.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Le direzioni di frontiera: **efficienza** (distillazione, quantizzazione a 8
  o 4 bit, pruning, architetture *mixture-of-experts* che attivano solo una
  frazione dei parametri per token), **contesto lungo** (attenzioni sparse e
  lineari, ottimizzazioni di memoria come FlashAttention, *state space model*)
  e **multimodalità** (spazi di rappresentazione condivisi fra testo, immagini
  e audio). A cui si aggiunge l’**allineamento**, che è oggi un'area di ricerca
  a pieno titolo e non un ritocco finale.
- Il calcolo **condizionato all'ingresso**: l’*Universal Transformer*
  {cite}`dehghani2019universal` lega i pesi fra le iterazioni, e sopra ci mette
  l’*Adaptive Computation Time* {cite}`graves2016adaptive`, che a ogni giro dà
  a ogni posizione una probabilità di arresto. La Turing-completezza che ne
  segue vale **sotto ipotesi**, e quella che pesa è il numero di passi non
  limitato a priori.
- Lo stesso filone è tornato dalla porta opposta: invece di iterare in
  silenzio dentro la pila, i modelli che ragionano allungano la **generazione**
  e scrivono i passi. Si paga in token prodotti, si guadagna una traccia
  leggibile, che però non è necessariamente fedele al calcolo svolto.
- Limiti aperti: il costo quadratico $O(n^2)$ dell'attenzione piena e
  l'archivio degli appunti che cresce a ogni token generato (la *KV cache*);
  costi concentrati e bias dei dati; e lo scarto fra massimizzare la
  verosimiglianza di una continuazione e stabilire che sia vera.
```

`````

C'è però un conto che questo capitolo nomina e non salda. Se ogni parola guarda
tutte le altre, il lavoro cresce col quadrato della lunghezza, e mentre il
modello scrive l'archivio dei suoi appunti si allunga a ogni parola prodotta:
sono i due prezzi dell'attenzione, e li paga chi vuole leggere lungo. Da lì
riparte **Attenzione lineare**, che per abbassarli mette le mani sull'unico
pezzo che qui non abbiamo mai discusso, quello che decide come l'attenzione si
spartisce fra le parole.
