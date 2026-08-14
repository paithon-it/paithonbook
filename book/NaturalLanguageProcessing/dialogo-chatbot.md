# Parlare con le macchine: dialogo e chatbot

Questo capitolo si chiude dove il libro era cominciato. Nell'Introduzione
abbiamo raccontato di Joseph Weizenbaum e di ELIZA
{cite}`weizenbaum1966eliza`, il programma che a metà degli anni Sessanta
conversava per iscritto con gli esseri umani, e della sorpresa del suo autore
nello scoprire quante persone attribuivano sentimenti a poche pagine di
codice. C'è un episodio, che Weizenbaum avrebbe raccontato per il resto della
vita, in cui quella sorpresa si condensa tutta: la sua segretaria, che l'aveva
visto costruire il programma per mesi e sapeva benissimo di avere davanti una
macchina, dopo pochi scambi con ELIZA gli chiese di uscire dalla stanza.
Voleva restare sola con il programma. E quando, in un'altra occasione,
Weizenbaum propose di registrare le conversazioni per poterle esaminare, fu
travolto dalle proteste di chi usava il programma: quello era spiare, quelle
erano cose private.

Sessant'anni dopo, milioni di persone conversano ogni giorno con delle
macchine. In quest'ultima sezione mettiamo insieme gli attrezzi del capitolo e
li puntiamo sul compito più antico e ambizioso dell'NLP, il **dialogo**.
Vedremo perché una conversazione è più di una fila di frasi, come sono fatte
le tre famiglie di sistemi di dialogo, come si dà loro un voto, e perché la
storia della segretaria non è un aneddoto d'epoca ma una questione ancora
aperta.

## Che cosa c'è in una conversazione

La tentazione è pensare che dialogare sia facile, una volta che si sa generare
una frase: il modello risponde all'ultima battuta, e via così. Ma una
conversazione non è una collana di frasi indipendenti: è un'attività
*congiunta*, con regole sue che rispettiamo senza accorgercene. Per vederle
basta origliare una telefonata qualsiasi:

> - Ristorante Da Mario, buonasera.
> - Buonasera. Vorrei prenotare un tavolo per domani.
> - Certo. Quanti siete?
> - Quattro. Se possibile all'aperto.
> - Mh-mh… all'aperto ne è rimasto uno solo, alle otto. Va bene?
> - Perfetto, lo prendiamo.
> - Un tavolo per quattro, domani alle otto, all'aperto. A che nome?
> - Rossi.

Otto battute banali. Eppure, sotto la superficie, succedono almeno quattro
cose che nessuna delle frasi, presa da sola, contiene.

`````{tab} Elementare

Rileggiamo la telefonata pezzo per pezzo. Primo: i **turni**. I due parlano
uno alla volta, senza pestarsi la voce, e il cambio avviene in una frazione di
secondo: nessuno dice «passo» come alla radio, eppure entrambi sanno sempre di
chi è il turno.

Secondo: **dire è fare**. «Vorrei prenotare un tavolo» non descrive il mondo:
lo *cambia*; dopo quella frase esiste una richiesta che prima non c'era. Allo
stesso modo, «Sai l'ora?» non è una domanda sul tuo orologio: chi risponde
solo «sì» e tira dritto ha capito le parole ma non la mossa.

Terzo: le **conferme**. Quel «Mh-mh» non aggiunge informazione: dice «ti sto
seguendo, continua». E la penultima battuta («Un tavolo per quattro, domani
alle otto, all'aperto») è una ricevuta letta ad alta voce: il ristoratore
rigioca ciò che ha capito, e il silenzio-assenso del cliente la controfirma.
Senza questi segnali la conversazione si sfalda.

Quarto: i **sottintesi**. «All'aperto *ne* è rimasto uno»: ne... di che? Di
tavoli all'aperto, ma nessuno lo ripete, il filo del discorso lo regge. «*Lo*
prendiamo»: lo, quel tavolo lì. È l'**anafora** di cui si diceva all'inizio del
capitolo: le parole piccole pescano il significato in quello che è già stato
detto. (Nulla a che vedere con l'anafora delle poesie, quella figura retorica
che ripete la stessa parola in testa a più versi: stesso nome, due mestieri
lontanissimi.) Qui però c'è un salto in più rispetto all'inizio del capitolo:
là il significato stava nella stessa frase, qui sta in una **battuta
precedente**, pronunciata da un'altra persona. Morale: per dialogare non basta
capire l'ultima frase, bisogna ricordare la partita.

`````

`````{tab} Superiore

I quattro fenomeni hanno una letteratura precisa.

**Presa del turno** (*turn-taking*). Sacks, Schegloff e Jefferson (1974) ne
descrissero la sistematica: i parlanti proiettano il punto di completamento
del turno altrui e si avvicendano con pause tipiche di circa 0,2 secondi in
lingue e culture diversissime (Stivers et al., 2009), troppo brevi per
pianificare da zero: pianifichiamo *mentre* l'altro parla. Per un sistema
vocale è un vincolo concreto: l'*endpointing* deve distinguere una pausa da
una cessione del turno.

**Atti linguistici** (*speech acts*). Austin (*How to Do Things with Words*,
1962) osservò che gli enunciati *eseguono* azioni; una tassonomia corrente
(Bach e Harnish, 1979) distingue **constativi** (affermare), **direttivi**
(chiedere), **commissivi**
(promettere, accettare) e **riconoscimenti** (ringraziare). «Sai l'ora?» è un
atto *indiretto*: forma interrogativa sì/no, funzione direttiva. Un sistema
di dialogo deve classificare l'atto, non solo la forma.

**Grounding**. Herbert Clark (1996) descrive la conversazione come
costruzione incrementale di un *common ground*, l'insieme delle conoscenze
mutuamente accettate. Ogni contributo va *radicato*: segnali di continuità
(«mh-mh», i *backchannel*), ripetizioni e riformulazioni. Un sistema che non
conferma mai risulta inaffidabile; uno che conferma ogni dato, esasperante:
la politica di conferma è una scelta di progetto.

**Implicature e anafora**. Grice (1975) formulò le massime conversazionali
(quantità, qualità, relazione, modo) da cui i parlanti derivano i sottintesi:
«Quanti siete?» «Quattro» funziona perché la risposta dà esattamente
l'informazione richiesta. E i legami anaforici («ne», «lo») attraversano i
turni: risolverli richiede uno **stato del dialogo** (entità menzionate,
impegni presi, dati accettati). Conclusione operativa: un interlocutore non è
una funzione ultima-frase → risposta, ma una macchina con memoria.

`````

## Tre famiglie di interlocutori artificiali

In sessant'anni i sistemi di dialogo si sono organizzati in tre famiglie:
quelli che rispondono **per regole**, quelli che compilano **moduli**, quelli
che **generano** la risposta parola per parola. Non è solo una successione
storica: tutte e tre sono vive, spesso dentro lo stesso prodotto.

### Lo specchio di regole: dentro ELIZA

Della prima famiglia conosciamo già il capostipite. Nella sezione sulla
cassetta degli attrezzi abbiamo aperto il cofano di ELIZA e trovato **pattern
matching**: lo schema «mi sento X» agganciato e rigirato in «Da quanto tempo
ti senti X?». ELIZA era in realtà un impianto a copioni intercambiabili, e il
copione celebre (DOCTOR) imitava uno psicoterapeuta di scuola rogersiana,
quella che invece di dare consigli rimanda al paziente le sue stesse parole.
Scelta astutissima: è l'unico interlocutore che può rispondere a tutto con una
domanda, rimandare ogni affermazione al mittente e non sapere *niente* del
mondo senza destare sospetti. (Nei primi anni Settanta lo psichiatra Kenneth
Colby costruì il contrario, PARRY, che simulava un paziente paranoide: nel 1972
i due programmi furono perfino messi a conversare fra loro attraverso ARPANET,
la rete di calcolatori americana da cui sarebbe nata internet.)

Vale la pena ricostruirla, perché è il programma più istruttivo del capitolo:
bastano le espressioni regolari della prima sezione e una tabellina di
sostituzioni. Ecco una mini-ELIZA italiana:

```python
import re

# pronomi e verbi da "riflettere": la prospettiva passa da io a tu
RIFLESSI = {"mio": "tuo", "mia": "tua", "miei": "tuoi", "mie": "tue",
            "mi": "ti", "me": "te", "io": "tu",
            "sono": "sei", "ho": "hai", "posso": "puoi", "voglio": "vuoi"}

# coppie (schema, risposta): {0} e' il pezzo di frase catturato dallo schema
REGOLE = [
    (r"mi sento (.+)",            "Da quanto tempo ti senti {0}?"),
    (r"vorrei (.+)",              "Perché vorresti {0}?"),
    (r"(?:penso|credo) che (.+)", "E cosa ti fa credere che {0}?"),
    (r"tutti (.+)",               "Proprio tutti {0}? Nessuna eccezione?"),
    (r"(.+)\?",                   "Perché me lo chiedi?"),
]

def rifletti(testo):
    """Riscrive un frammento dal punto di vista dell'interlocutore."""
    return " ".join(RIFLESSI.get(parola, parola) for parola in testo.split())

def eliza(frase):
    frase = frase.lower().strip(" .!")
    for schema, risposta in REGOLE:       # la prima regola che aggancia vince
        match = re.match(schema, frase)
        if match:
            return risposta.format(*map(rifletti, match.groups()))
    return "Capisco. Vai avanti."         # ripiego: nessuno schema ha agganciato

for battuta in ["Mi sento in colpa verso mia sorella",
                "Penso che nessuno mi ascolti",
                "Il gatto nero salta sul muro"]:
    print("TU:   ", battuta)
    print("ELIZA:", eliza(battuta))

# TU:    Mi sento in colpa verso mia sorella
# ELIZA: Da quanto tempo ti senti in colpa verso tua sorella?
# TU:    Penso che nessuno mi ascolti
# ELIZA: E cosa ti fa credere che nessuno ti ascolti?
# TU:    Il gatto nero salta sul muro
# ELIZA: Capisco. Vai avanti.
```

Trenta righe scarse, eppure l'effetto c'è. Il cuore è la funzione `rifletti`,
che ribalta la prospettiva del frammento catturato («mia sorella» diventa «tua
sorella») ed è questo riuso delle *tue* parole, restituite al posto
grammaticalmente giusto, a creare l'illusione di essere ascoltati. L'ordine
delle regole conta (la prima che aggancia vince), e il ripiego finale copre
tutto il resto del mondo, gatto sul muro compreso. L'ELIZA vera aveva più
copioni, più varianti di risposta e un rudimento di memoria, ma il principio è
tutto qui.

E qui è anche il limite. ELIZA **non tiene niente da parte**: finita una
battuta, la dimentica, e alla battuta dopo riparte da zero come se fosse la
prima. Quello che un interlocutore dovrebbe tenere da parte (chi sono io, di
che cosa stiamo parlando, che cosa ci siamo già detti) ha un nome che tornerà
fra poco: si chiama **stato del dialogo**, ed è la memoria della partita.
ELIZA non ne ha, ed è il contrario esatto di ciò che la telefonata al
ristorante ci ha mostrato essere una conversazione.

### Il modulo da riempire: i sistemi a frame

La seconda famiglia nasce nel 1977 allo Xerox PARC, il laboratorio californiano
in cui in quegli anni si inventava metà dell'informatica che usiamo oggi (il
mouse con le finestre sullo schermo, per dire, viene di lì). L'articolo lo
firmano in sei, e sono sei fra i nomi più noti dell'intelligenza artificiale di
allora: Bobrow, Kaplan, Kay, Norman, Thompson e Winograd
{cite}`bobrow1977gus`. Il loro sistema si chiamava GUS, faceva l'agente di
viaggio e prenotava voli per San Diego.

L'idea che introduce si chiama **frame**, cioè «modulo» nel senso del foglio da
compilare, ed è di quelle che sopravvivono ai decenni: quasi cinquant'anni
dopo è ancora l'ossatura degli assistenti vocali.

`````{tab} Elementare

Pensate a uno sportello pubblico. L'impiegato ha davanti un modulo (in inglese
*frame*, ed è la parola che dà il nome a tutta la famiglia) con delle caselle:
*dove*, *quando*, *quanti*, *a che nome*. La conversazione gli serve a
riempirle, e questo spiega tutto il suo comportamento: fa domande solo per le
caselle ancora vuote, e se in una battuta sola dite «un tavolo per quattro
domani alle otto» ne riempie tre in un colpo senza richiederle. Alla fine
rilegge il modulo per conferma, e la pratica parte.

Un assistente vocale fa la stessa identica cosa, con moduli invisibili. «Che
tempo fa domani a Roma?»: l'assistente riconosce *quale* modulo tirare fuori
(previsioni meteo, non sveglie né musica) e compila le caselle, luogo: Roma;
giorno: domani. Se dite solo «che tempo fa?», la casella del luogo la riempie
con la vostra posizione o ve la chiede: «Per quale città?». Per questo gli
assistenti vocali sono bravissimi dentro i loro moduli (meteo, timer,
chiamate) e cadono nel vuoto un millimetro fuori: là non c'è nessuna casella
da riempire.

`````

`````{tab} Superiore

Un **frame** è una struttura dati con **slot** tipizzati; a ogni slot sono
associati una domanda per elicitarlo e vincoli sul valore (LUOGO: città; DATA:
anche relativa, «domani»; PERSONE: intero). Il dialogo è governato dal
riempimento: domande per gli slot vuoti, riempimenti multipli in un turno,
anche fuori ordine (*iniziativa mista*), azione a modulo completo. GUS gestiva
già date relative, ellissi e semplici anafore: nel 1977.

L'erede moderno è l'architettura a **stato di dialogo**, una catena di
componenti in gran parte già costruita in questo capitolo:

1. **NLU**: classificazione dell'**intento** (un classificatore di testo come
   quelli visti per il sentiment) e **slot filling**, cioè etichettatura di
   sequenze con schema BIO, identica al NER: in «che tempo fa domani a Roma»,
   *domani* → `B-DATA`, *Roma* → `B-LUOGO`, intento = `previsioni_meteo`.
2. **Tracker dello stato**: accumula gli slot riempiti e confermati (la
   memoria della partita).
3. **Policy**: decide la mossa (chiedere, confermare, eseguire).
4. **NLG**: genera la risposta, spesso ancora per *template* riempiti con i
   valori degli slot.

I punti di forza spiegano la longevità: comportamento controllabile (nessuna
risposta inventata), errori localizzabili, successo del compito misurabile.
Il limite è speculare: fuori dai frame previsti il sistema non ha nulla da
dire.

`````

### Prevedere la risposta: i sistemi generativi

La terza famiglia rovescia l'approccio: perché scrivere regole e moduli, se
abbiamo le trascrizioni di milioni di conversazioni? Trattiamo la risposta come
una *traduzione* del turno precedente, e usiamo la stessa identica macchina
della sezione sulla traduzione: una rete che legge la battuta ricevuta e una
che scrive la risposta. Cambiano solo i dati: al posto delle frasi in italiano
e in inglese, botte e risposte.

`````{tab} Elementare

Ricordate la tastiera del telefono che indovina la parola dopo? Immaginatela
ingrandita fino a scrivere non la prossima parola ma l'intera prossima
*battuta*, dopo aver letto milioni di dialoghi veri, per esempio i sottotitoli
dei film. Funziona, ed è già sorprendente. Ma se insegni a un modello a dare
*la risposta più probabile*, impara le risposte che vanno bene ovunque («Non
lo so», «Va bene», «Dipende»), le più probabili proprio perché compatibili con
tutto: l'interlocutore più evasivo del mondo.

E c'è un problema più profondo: un modello che ha imparato a *continuare* i
testi non è ancora un buon interlocutore. Se scrivo «come si toglie una
macchia di vino?», una continuazione plausibilissima trovata su internet è
un'altra domanda sulle macchie, non la soluzione. Per trasformare un ottimo
completatore di frasi in un assistente che risponde serve una seconda scuola,
fatta di esempi di buone risposte e di giudizi umani. Come funzioni questa
scuola (si chiama *post-training*) lo vedremo nel capitolo sui Transformer.

`````

`````{tab} Superiore

Il modello è un seq2seq: si massimizza $P(y \mid x)$, dove $x$ è la storia del
dialogo e $y$ la risposta, con la stessa fattorizzazione autoregressiva della
traduzione {cite}`sutskever2014sequence`. Uno dei primi chatbot neurali
end-to-end è quello di Vinyals e Le (2015), addestrato su log di assistenza
tecnica e sottotitoli di film. Il difetto emerse subito: massima
verosimiglianza e beam search privilegiano risposte ad alta probabilità
*marginale*, generiche per costruzione («I don't know»: Li et al., 2016, che
proposero di riordinare le ipotesi con la mutua informazione tra $x$ e $y$).

Il salto di qualità arriva con la scala (modelli decoder-only pre-addestrati
su corpora web {cite}`brown2020language`) ma soprattutto con il
**post-training**, che risolve il disallineamento di fondo: un language model
modella $P(\text{continuazione} \mid \text{prefisso})$, non «rispondi in modo
utile e onesto». La ricetta è in due tempi {cite}`ouyang2022training`:
**instruction tuning** (fine-tuning supervisionato su coppie richiesta → buona
risposta scritte da persone) e **RLHF** (*reinforcement learning from human
feedback*), dove un modello di ricompensa addestrato sulle preferenze umane
guida l'ottimizzazione della generazione. È questo passaggio a trasformare un
modello di linguaggio in un interlocutore; i dettagli (e i limiti, a partire
dalle risposte fluenti ma false) li rimandiamo al capitolo sui Transformer,
dove seguiremo il percorso dal pre-addestramento ai modelli con cui oggi si
conversa.

`````

## Dare un voto a una chiacchierata

Come si stabilisce se un sistema di dialogo è *buono*? Per la traduzione
avevamo almeno un riferimento con cui confrontarsi. Qui no, ed è il cuore
della difficoltà: a «Che si fa stasera?» esistono mille risposte eccellenti
che non condividono una parola.

`````{tab} Elementare

È la differenza tra correggere un dettato e correggere un tema. Il dettato ha
il testo giusto: conti gli errori ed è fatta. Il tema no: due temi da dieci
possono essere completamente diversi, e nessun righello li misura; serve un
giudice che li legga.

Con i sistemi a modulo si può ancora correggere come un dettato: la
prenotazione è andata a buon fine, sì o no? Quante battute ci sono volute?
Sono numeri, e si contano. Con i chatbot aperti resta il tema: si fanno
conversare con persone in carne e ossa, e si chiede ai giudici di votare: le
risposte filavano? stavano in piedi da un capo all'altro? veniva voglia di
continuare? Costoso, lento, un po' soggettivo: ma a oggi non c'è metro
migliore del giudizio umano su una cosa fatta per gli umani.

`````

`````{tab} Superiore

Per i sistemi *task-oriented* la valutazione è ancorata al compito: tasso di
successo, accuratezza degli slot, efficienza (numero di turni, correzioni).
Il framework PARADISE (Walker et al., 1997) combina successo del compito e
costi del dialogo in un'unica funzione di qualità, stimata sui giudizi di
soddisfazione degli utenti.

Per i chatbot aperti le metriche ereditate dalla traduzione falliscono: la
sovrapposizione con una risposta di riferimento (BLEU {cite}`papineni2002bleu`
e simili, definito nella sezione sulla traduzione) correla pochissimo con i
giudizi umani, proprio per la molteplicità delle risposte valide (Liu et al.,
2016, dal titolo eloquente: *How NOT to evaluate your dialogue system*). Il
motivo si legge nella definizione stessa: BLEU conta $n$-grammi condivisi con
il riferimento, e due risposte ottime alla stessa domanda possono non
condividerne nemmeno uno. La perplessità misura la fluidità, non la qualità del
dialogo. Restano i **giudizi umani**, per dimensioni separate (coerenza,
specificità, correttezza fattuale, interesse) o per confronto a coppie; di
recente si usano anche LLM come giudici, pratica economica ma con bias
documentati (come la preferenza per le risposte lunghe) che impone cautela.

`````

Su tutto questo aleggia la cornice storica che conosciamo dall'Introduzione:
il gioco dell'imitazione di Turing {cite}`turing1950computing`, la
conversazione come banco di prova dell'intelligenza. Va maneggiata con
prudenza, in entrambe le direzioni: ELIZA ha mostrato già nel 1966 che
ingannare un interlocutore per qualche minuto è un'asticella bassa (il test
misura anche la nostra propensione a farci ingannare) e, al contrario,
superare una conversazione non certifica nessuna comprensione. Per questo oggi
il test di Turing è un esperimento mentale fondativo, e non uno strumento con
cui misurare davvero qualcosa.

## La lezione di Weizenbaum

Resta l'ultima domanda, quella da cui siamo partiti: perché la segretaria
chiese di restare sola col programma? Il fenomeno ha oggi un nome, **effetto
ELIZA**: la tendenza, robusta e quasi automatica, ad attribuire comprensione e
intenzioni a qualunque cosa parli la nostra lingua. Non è ingenuità: è il
riflesso di una specie che per tutta la sua storia ha avuto un'unica sorgente
di linguaggio fluente, gli altri esseri umani.

Weizenbaum ne fu così turbato da cambiare mestiere: da costruttore di
programmi a loro critico. Nel libro che scrisse dieci anni dopo, *Computer
Power and Human Reason* {cite}`weizenbaum1976computer`, raccontò di non aver
mai immaginato «che esposizioni brevissime a un programma per calcolatore
relativamente semplice potessero indurre un pensiero delirante così potente in
persone del tutto normali»; lo scandalizzò, più di tutto, la proposta di
psichiatri veri di usare programmi come il suo per la psicoterapia su larga
scala. La sua tesi, spesso semplificata in generico allarme, era invece
precisa, e sta in due verbi: c'è differenza fra *decidere* e *giudicare*.

Decidere è trovare la risposta giusta secondo un criterio che qualcuno ha già
fissato: qual è la strada più corta, se questo conto torna, quale di questi
mille documenti contiene una certa parola. È un'operazione di calcolo, e a una
macchina si delega volentieri. Giudicare è un'altra cosa: stabilire se un
ragazzo va bocciato, se una persona è pericolosa, se questa cura vale la pena
per questo paziente. Lì non c'è nessun criterio scritto da nessuna parte: ci
sono un'esperienza di vita e una responsabilità di cui qualcuno risponde. La
domanda giusta, diceva Weizenbaum, non è cosa le macchine *possono* fare, ma
cosa *dobbiamo* affidare loro.

Cinquant'anni dopo, la questione si è fatta concreta su tre fronti, e merita
un tono sobrio: né allarme, né alzata di spalle.

- **Antropomorfizzazione**, cioè la nostra abitudine ad attribuire qualità
  umane a ciò che umano non è, dal cane di casa alla macchina che «non vuole
  partire». I sistemi attuali sono incomparabilmente più fluenti di ELIZA, e la
  fluidità amplifica l'effetto: scambiare la forma del linguaggio per
  comprensione è l'errore contro cui mette in guardia la critica dei
  «pappagalli stocastici» {cite}`bender2021dangers`, dove *stocastico* vuol
  dire «governato dal caso»: un pappagallo che ripete quello che ha sentito,
  scegliendo di volta in volta secondo le probabilità che ha imparato, senza
  sapere che cosa sta dicendo. E non è solo un equivoco filosofico. Quando un
  chatbot dice «io», ha un nome proprio e fa finta di esitare come farebbe una
  persona, quel riflesso non lo subisce: lo asseconda, per scelta di chi
  l'ha costruito.
- **Privacy.** Chi protestò per i registri di ELIZA era in anticipo di
  sessant'anni. A un interlocutore che non giudica si confidano cose che non
  si scrivono in un modulo: oggi quelle conversazioni sono dati, che possono
  essere conservati, riletti, usati per addestrare altri modelli. Riservatezza
  percepita e riservatezza effettiva raramente coincidono.
- **Dipendenza emotiva.** Esistono applicazioni progettate per la compagnia,
  con milioni di utenti. Un interlocutore disponibile a ogni ora, mai stanco,
  mai in disaccordo, può essere un sollievo reale nella solitudine, e insieme
  un allenamento ingannevole, perché una relazione senza attriti né bisogni
  altrui non somiglia a nessuna relazione umana. Gli effetti, specie sulle
  persone fragili, sono materia di studio ancora aperta.

Niente di tutto questo rende i sistemi di dialogo cattivi in sé: rendono
accessibili servizi a chi non sa usare un'interfaccia, assistono senza orari,
aiutano a studiare. Il punto di Weizenbaum non era vietarli, ma ricordare che
la scelta di cosa affidare a una macchina che parla è una scelta nostra, da
fare a occhi aperti: sapendo che dall'altra parte non c'è nessuno, per quanto
forte sia l'impressione contraria.

Con questo il cerchio del capitolo si chiude. Siamo partiti dalle espressioni
regolari e siamo arrivati a modelli che conversano, e per strada abbiamo visto
il gatto nero saltare sul muro in tutti i modi in cui una macchina lo può
scrivere: come conteggi in un sacchetto, come punti su una mappa di
significati, come riassunto che scorre dentro una rete che legge in fila.
L'ultimo passo, l'architettura che ha mandato in pensione quella lettura in
fila e ha reso possibili gli interlocutori artificiali di oggi, merita un
capitolo intero: i **Transformer** ci aspettano alla prossima pagina.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Una conversazione non è una fila di frasi. Ci si dà il turno senza
  accavallarsi; si **fanno cose** con le parole («sai l'ora?» non è una domanda
  a cui rispondere sì); ci si conferma di continuo che si sta seguendo (il
  «mh-mh» del ristoratore, e la ricevuta riletta ad alta voce); e le parole
  piccole («ne», «lo») pescano il senso nelle battute già dette. Per dialogare
  non basta capire l'ultima frase: bisogna ricordare la partita.
- Tre famiglie di interlocutori artificiali, tutte e tre ancora vive. **A
  regole**, come ELIZA: aggancia uno schema nelle tue parole e te le rigira,
  nessuna memoria, nessun mondo. **A moduli**, come lo sportello che riempie le
  caselle di un formulario facendo domande solo per quelle vuote: è ancora
  l'ossatura degli assistenti vocali, bravissimi dentro il modulo e nel vuoto
  un millimetro fuori. **Generativi**, che scrivono la risposta parola per
  parola dopo aver letto milioni di dialoghi.
- Chi impara a *continuare* i testi non sa ancora *rispondere*: serve una
  seconda scuola, fatta di esempi di buone risposte e di giudizi umani, ed è la
  storia del capitolo sui Transformer.
- **Il dettato e il tema**: un sistema a moduli si corregge come un dettato (la
  prenotazione è andata a buon fine, sì o no?), un chatbot aperto come un tema,
  e per il tema serve un giudice che legga. Costoso e un po' soggettivo, ma non
  c'è di meglio per una cosa fatta per gli umani.
- L'**effetto ELIZA**: attribuiamo comprensione a qualunque cosa parli la
  nostra lingua, ed è un riflesso, non ingenuità. È il monito di Weizenbaum, ed
  è più attuale del suo programma: sapere che dall'altra parte non c'è nessuno,
  per quanto forte sia l'impressione contraria.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Una conversazione non è una fila di frasi: **turni**, **atti linguistici**
  (dire è fare: «sai l'ora?» non chiede un sì/no), **grounding** (i segnali
  di conferma reciproca) e **sottintesi** anaforici richiedono uno *stato del
  dialogo*, una memoria della partita.
- Tre famiglie: sistemi **a regole** (ELIZA {cite}`weizenbaum1966eliza`:
  espressioni regolari e riflessione dei pronomi, nessuno stato), **a frame**
  (GUS {cite}`bobrow1977gus`: slot da riempire con domande mirate; ancora
  l'ossatura degli assistenti vocali), **generativi** (dal seq2seq agli LLM).
- Un modello di linguaggio impara a *continuare* i testi; a *rispondere* lo
  si insegna col **post-training** (instruction tuning e RLHF
  {cite}`ouyang2022training`), sviluppato nel capitolo sui Transformer.
- Valutare il dialogo è difficile perché non esiste *la* risposta giusta:
  successo del compito per i sistemi a frame, **giudizi umani** per i chatbot
  aperti. Le metriche di sovrapposizione (BLEU {cite}`papineni2002bleu`) non
  reggono, perché due risposte ottime possono non condividere un $n$-gramma;
  il test di Turing {cite}`turing1950computing` è una cornice
  storica, non una metrica.
- L'**effetto ELIZA** (attribuire comprensione a ciò che parla) è il riflesso
  su cui i chatbot fanno leva, volenti o nolenti: il monito di Weizenbaum
  {cite}`weizenbaum1976computer` su antropomorfizzazione, privacy e deleghe da
  non dare è più attuale del suo programma.
```
`````
