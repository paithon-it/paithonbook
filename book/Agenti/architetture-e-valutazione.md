# Architetture di agenti e come valutarli

Nella sezione precedente abbiamo costruito un agente e l'abbiamo fatto girare:
pensa, agisce, osserva, ripete. Su un compito ben delimitato (trova un dato,
fai un conto, rispondi) quel ciclo va sorprendentemente lontano. Ma provate a
chiedergli qualcosa di grosso: «prendi questa segnalazione di errore, trova il
file giusto in un progetto da centomila righe, scrivi la correzione e verifica
che i test passino». Il loop ReAct, da solo, si smarrisce: troppe mosse,
troppe strade, troppe occasioni per perdere il filo. Un singolo agente in loop
non basta più.

Da qui due mosse, ed è di queste che parla la sezione. La prima è
**pianificare in anticipo** invece di reagire un passo alla volta. La seconda
è **comporre più agenti**, ciascuno con un mestiere, come si mette insieme una
squadra. E dietro entrambe si nasconde la domanda più scomoda del campo,
quella che nessuno ha davvero chiuso: come si fa a sapere se un agente
**funziona davvero**? Valutare un modello che risponde è già difficile: lo
vedremo nel capitolo conclusivo su LLMOps, con il testo aperto. Valutare un
agente che *agisce*, in più passi, in un ambiente che cambia sotto i suoi
piedi, lo è molto di più.

## Pianificare: scomporre il problema

ReAct ragiona e agisce *un passo alla volta*: decide la mossa, la esegue, guarda
com'è andata, decide la prossima. È flessibile, ma su un compito lungo rischia
di procedere a naso, senza una visione d'insieme, e di infilarsi in vicoli
ciechi. L'alternativa è ribaltare l'ordine: prima **scomporre** il problema in
un piano di sotto-compiti, poi eseguirli. È il pattern **plan-and-execute**,
«pianifica ed esegui».

`````{tab} Elementare

Pensa a come si fa la spesa. Un modo è entrare al supermercato e decidere
scaffale per scaffale, guardandoti intorno: prendo la pasta, ah già mi serviva
anche il latte, torno indietro… Funziona per due o tre cose, ma per una spesa
grossa giri a vuoto, dimentichi metà roba e ti ritrovi tre volte davanti al
banco frigo. L'altro modo è scrivere **la lista prima di entrare**: la dividi
per reparto, e dentro segui l'ordine senza pensarci. Fai meno strada, non
dimentichi niente.

Un agente che pianifica fa la seconda cosa. Prima di toccare qualunque
strumento, si ferma e scrive il piano: «per sistemare questo bug devo (1),
trovare dove nasce l'errore, (2) capire la causa, (3) scrivere la correzione,
(4) far girare i test». Poi esegue i punti uno per uno. Il vantaggio è la
visione d'insieme; il rischio è che la lista, scritta al buio prima di
entrare, non tenga conto di una sorpresa (lo scaffale vuoto, il reparto
spostato) e vada rifatta a metà strada.

`````

`````{tab} Superiore

**ReAct** intreccia ragionamento e azione a ogni passo: la policy sceglie
$a_t$ guardando solo lo stato corrente $s_t$, senza un piano globale
esplicito. È reattivo (si adatta bene alle sorprese) ma su orizzonti lunghi
tende a perdere coerenza, ripetere azioni o divagare. Il pattern
**plan-and-execute** separa due ruoli: un *pianificatore* produce in un colpo
solo una sequenza di sotto-obiettivi $g_1, \dots, g_k$ che decompongono il
compito, e un *esecutore* li affronta uno per uno (spesso con un mini-loop
ReAct dentro ciascuno). Il piano dà struttura, coerenza globale e spesso
**meno chiamate al modello** per il ragionamento di alto livello.

Il compromesso è netto e va dichiarato. Pianificare in anticipo conviene
quando il compito è *decomponibile* e l'ambiente *prevedibile*: il piano regge
fino in fondo. Ma un piano rigido non sa reagire a ciò che non aveva previsto
(un test che rivela un secondo bug, un file che non esiste) e allora serve una
fase di **re-planning**: quando un sotto-obiettivo fallisce, si torna dal
pianificatore e si aggiorna la lista. È lo stesso
spendere-calcolo-per-ragionare del *test-time compute* visto nel capitolo sui
Transformer, ma speso *prima* di agire anziché durante: la pianificazione è
ragionamento su come muoversi, scritto in anticipo. Nessuno dei due estremi
vince sempre; i sistemi robusti mescolano: un piano di massima, rivisto quando
la realtà lo smentisce.

`````

## Più agenti che collaborano

Finora un agente, un modello. Ma se un compito ha nature diverse (pianificare,
scrivere codice, criticarlo), perché affidarlo a un solo generalista? Nasce
l'idea dei **sistemi multi-agente**: più agenti, ciascuno con un **ruolo
specializzato**, che si passano il lavoro e conversano tra loro. Un
*pianificatore* scompone il compito, un *esecutore* lo svolge, un *critico*
rilegge il risultato e segnala gli errori, e il giro ricomincia finché il
critico è soddisfatto.

`````{tab} Elementare

È la differenza tra un artigiano solitario e una piccola bottega. Da solo, fai
tutto tu: progetti, costruisci, controlli il tuo stesso lavoro, e proprio
perché è il *tuo*, i difetti tendi a non vederli. In una bottega ci sono
ruoli: uno disegna il progetto, uno lo realizza, un terzo (il collaudatore) lo
prova e dice cosa non va. Il collaudatore non ha costruito niente, e proprio
per questo nota lo scricchiolio che l'artigiano innamorato del proprio lavoro
ignorava.

Con gli agenti funziona uguale. Invece di un modello che fa e si giudica da
solo, se ne mettono in fila alcuni con compiti diversi, che si scrivono l'un
l'altro come colleghi in chat: «ecco il piano», «ecco il codice», «ho provato,
il test 3 non passa, correggi qui». La specializzazione aiuta: un critico
dedicato pesca errori che l'esecutore non vedeva. Ma attenzione: più teste
significano anche più stipendi da pagare (ogni agente è chiamate al modello
che costano) e più modi di litigare o fraintendersi. Non sempre la bottega
batte il buon artigiano.

`````

`````{tab} Superiore

Un framework che ha reso concreto questo schema è **AutoGen**
{cite}`wu2024autogen`, che modella un sistema multi-agente come una
**conversazione** tra agenti *conversabili*: ognuno ha un ruolo e un prompt di
sistema che lo definisce (assistente, esecutore di codice, revisore, proxy
umano), e l'orchestrazione è il protocollo con cui si scambiano messaggi
(sequenziale, a turni, o con un agente «manager» che decide chi parla dopo).
Lo stesso motore linguistico, istanziato con istruzioni diverse, diventa una
squadra.

L'onestà, qui, è d'obbligo, perché è un terreno dove l'entusiasmo corre più dei
risultati. Aggiungere agenti **non è gratis** e **non è sempre meglio**. Ogni
agente in più è contesto in più da riempire e generazioni in più da pagare: il
costo cresce con il numero di partecipanti e con i giri di conversazione. E
moltiplicare gli agenti moltiplica i **modi di sbagliare**: un fraintendimento
che si propaga, due agenti che entrano in un ping-pong senza convergere,
l'errore di uno che diventa la premessa dell'altro. La letteratura è ancora
lontana da un verdetto netto su *quando* il multi-agente batta un singolo
agente ben progettato {cite}`xi2023rise`; la regola prudente è: si aggiunge un
ruolo solo quando risolve un problema che un agente solo non risolveva, non per
il gusto della squadra.

`````

Questa è l'idea; i meccanismi con cui una squadra di agenti si organizza
davvero (quanto costa il coordinamento e quando lo ripaga, quale grafo di
comunicazione conviene, come ci si mette d'accordo quando i partecipanti non
sono affidabili, e come si può *imparare* a coordinarsi invece di essere
programmati per farlo) hanno un capitolo dedicato più avanti.

## La memoria che dura

C'è un ingrediente che finora abbiamo lasciato ai margini: la **memoria a
lungo termine**. La finestra di contesto è memoria di lavoro: capiente ma
effimera, si svuota a fine sessione. Un agente che debba *ricordare*
attraverso giorni e migliaia di eventi ha bisogno di un archivio esterno e di
un modo intelligente per pescarci dentro. Il caso di studio più istruttivo è
un piccolo esperimento del 2023 che sembra un videogioco: i **generative
agents** di Park e colleghi {cite}`park2023generative`.

Venticinque agenti abitano un paesino simulato, Smallville: si svegliano,
fanno colazione, vanno al lavoro, si incontrano, chiacchierano. Nessuno ha
scritto la loro giornata a mano: ciascuno è un LLM che decide cosa fare in
base a ciò che ricorda. Il risultato più citato è un comportamento *emerso* da
solo: un agente decide di dare una festa di San Valentino, ne parla a
qualcuno, l'invito si propaga di bocca in bocca per il paese, e la sera
diversi agenti si presentano (coordinandosi senza che nessuno abbia
programmato la cosa). La domanda interessante non è «è vivo?» (non lo è), ma
*come* fa un LLM a comportarsi in modo coerente su una scala temporale così
lunga.

`````{tab} Elementare

Il segreto è un **diario**. Ogni agente annota in un quaderno, in frasi
normali, tutto ciò che gli capita: «ho fatto colazione al bar», «Maria mi ha
detto che organizza una festa». Il quaderno cresce a dismisura (migliaia di
righe) e rileggerlo tutto ogni volta è impossibile. Serve quindi un
bibliotecario che, quando l'agente deve decidere qualcosa, gli tiri fuori dal
quaderno *solo le pagine che contano adesso*.

E come sceglie quali pagine? Con tre criteri di buon senso. Quanto è
**recente** il ricordo (ciò che è successo un'ora fa pesa più di ieri); quanto
è **importante** (una festa conta più di una colazione qualsiasi); e quanto
**c'entra** con la situazione di adesso (se sto pensando alla festa, ripesco i
ricordi sulla festa). Ogni tanto, poi, l'agente si ferma e **riflette**:
rilegge gli ultimi appunti e ne ricava una conclusione più alta («a Maria
piace organizzare eventi») che riscrive nel quaderno come un nuovo ricordo. Da
questi pensieri più maturi nascono i suoi piani. Ricordare, ripescare,
riflettere, pianificare: è così che un mucchio di appunti diventa un
comportamento coerente.

`````

`````{tab} Superiore

L'architettura ha tre pezzi. Il **memory stream** è un registro append-only di
osservazioni in linguaggio naturale, ciascuna con un timestamp. Il **recupero**
seleziona, a ogni decisione, le memorie rilevanti con un punteggio che combina
tre segnali normalizzati:

$$
\text{punteggio}(m) = \alpha_{\text{rec}}\,\text{recency}(m)
+ \alpha_{\text{imp}}\,\text{importance}(m)
+ \alpha_{\text{rel}}\,\text{relevance}(m, q),
$$

dove $m$ è una memoria e $q$ la situazione corrente: $\text{recency}(m)$ decade
esponenzialmente con il tempo trascorso dall'ultimo accesso,
$\text{importance}(m)$ è un voto di salienza (da 1 a 10) che il modello stesso
assegna alla memoria quando la scrive, e $\text{relevance}(m, q)$ è la
similarità tra gli embedding della memoria e della query. Nel lavoro originale i
pesi $\alpha$ valgono tutti 1: tre criteri sommati, non uno solo.

Il terzo pezzo è la **riflessione**: periodicamente l'agente sintetizza dalle
memorie recenti alcune inferenze di livello più alto (proposizioni astratte
come «Klaus è appassionato di ricerca») e le riscrive *nel* memory stream come
nuove memorie, recuperabili a loro volta. Si forma così un albero:
osservazioni grezze in basso, riflessioni via via più astratte in alto. La
**pianificazione** traduce infine queste sintesi in piani giornalieri,
decomposti dal grossolano al fine. Il punto architetturale generale, oltre
l'esperimento: la memoria a lungo termine di un agente non è «tenere tutto nel
contesto», ma **memorizzare fuori, recuperare il pertinente, e ogni tanto
ricomprimere in astrazioni** (lo stesso schema recupera-e-condensa che governa
il RAG e il context engineering).

`````

## Valutare un agente: il problema difficile

Arriviamo alla domanda scomoda. Valutare un classificatore è facile: c'è
un'etichetta giusta, confronti, conti gli errori. Valutare un agente è
tutt'altra faccenda, per tre ragioni che si sommano. Primo: spesso non esiste
**una** sola risposta giusta; a un compito «sistema questo bug» corrispondono
molte soluzioni valide. Secondo: il compito è **multi-passo**, e un agente può
arrivare al risultato giusto per la strada sbagliata, o fallire dopo aver
fatto quasi tutto bene. Terzo: l'**ambiente cambia** (una ricerca web dà
risultati diversi oggi e domani) e la stessa prova non è mai identica a se
stessa.

Servono quindi metriche di natura diversa da quelle di un modello che risponde e
basta. Tre, soprattutto: il **tasso di successo** sul compito (l'ha portato a
termine, sì o no?), la valutazione della **traiettoria** (*come* ci è arrivato:
con quali mosse, quanto pulite, quante inutili?) e il **costo** con la
**latenza** (quanti token, quanti secondi, quante chiamate a strumenti?). Un
agente che risolve il compito ma consuma diecimila token e trenta passi non è
«riuscito» allo stesso modo di uno che lo chiude in quattro.

`````{tab} Elementare

Come giudichi uno chef? Non dal singolo piatto assaggiato di sfuggita. Lo
giudichi dal **servizio di un'intera serata**: gli ordini sono usciti giusti?
quanti sono tornati indietro? il tavolo otto ha aspettato un'ora? La domanda
non è «questo piatto è buono» ma «quante volte, su tutti gli ordini della sera,
ha portato in tavola esattamente ciò che era stato chiesto».

Con un agente è lo stesso. Non basta guardare la risposta finale di *una*
prova: gli si danno tanti compiti e si conta la frazione portata a termine
davvero (il **tasso di successo**). Ma un buon capocuoco guarda anche la
cucina, non solo i piatti in uscita: se un piatto è venuto bene per puro caso,
in mezzo a un caos di padelle bruciate, non è un successo su cui contare
domani. Per questo si ispeziona anche **come** l'agente ci è arrivato (la
traiettoria) e quanto è costato in tempo e fatica. Risultato giusto, strada
pulita, conto ragionevole: tre cose diverse, tutte da misurare.

`````

`````{tab} Superiore

Il **tasso di successo** (*success rate*) è la frazione di compiti risolti su un
insieme di prove: la metrica principe, ma grossolana, perché è un sì/no che
ignora *come* si è arrivati e nasconde i successi fortunati. La **valutazione
della traiettoria** (*trajectory evaluation*) guarda la sequenza di azioni:
erano quelle giuste? ce n'erano di inutili o ridondanti? l'agente si è
avvicinato all'obiettivo a ogni passo? Un agente può azzeccare la risposta per
la strada sbagliata (giusto per caso) o sbagliarla dopo una traiettoria
impeccabile (l'ultimo passo va storto): guardare solo il risultato finale
confonde questi casi, e per capire davvero *dove* un agente rompe serve la
traccia.

Sotto tutto c'è la fragilità dei **compiti lunghi**, già incontrata:
l'accumulo di errori. Se a ogni passo la probabilità di sbagliare la mossa è
$p$, la probabilità di attraversare $n$ passi senza incidenti è

$$
P(\text{successo}) \le (1 - p)^n,
$$

che precipita al crescere di $n$: con $p = 0{,}1$, dieci passi danno già
$(0{,}9)^{10} \approx 0{,}35$. Non basta essere bravi a un passo, bisogna
esserlo per molti di fila, ed è la ragione strutturale per cui i tassi di
successo sui compiti lunghi restano modesti. Alla misura del *cosa* si
affianca poi quella del *quanto*: token consumati, latenza, numero di chiamate
a strumenti; perché un agente sostenibile non è solo quello che riesce, ma
quello che riesce a un costo accettabile.

`````

Un frammento di codice rende concreto perché il solo tasso di successo non
basta. Immaginiamo di aver fatto girare un agente su un pugno di compiti e di
aver registrato, per ciascuno, l'esito, i passi, i token e se la traiettoria era
«pulita»:

```python
# ogni episodio: esito, passi, token consumati, traiettoria valida?
episodi = [
    {"successo": True,  "passi": 4,  "token": 2100, "traiettoria_ok": True},
    {"successo": True,  "passi": 9,  "token": 5400, "traiettoria_ok": False},
    {"successo": False, "passi": 12, "token": 8000, "traiettoria_ok": False},
    {"successo": True,  "passi": 5,  "token": 2600, "traiettoria_ok": True},
    {"successo": False, "passi": 6,  "token": 3100, "traiettoria_ok": True},
]

n = len(episodi)
successi = [e for e in episodi if e["successo"]]
tasso_successo = len(successi) / n
# fra i compiti riusciti, quanti per una strada "pulita"?
traiettorie_ok = sum(e["traiettoria_ok"] for e in successi) / len(successi)
token_medi = sum(e["token"] for e in episodi) / n

print(f"episodi: {n}")
print(f"tasso di successo: {tasso_successo:.0%}")
print(f"successi con traiettoria valida: {traiettorie_ok:.0%}")
print(f"token medi per episodio: {token_medi:.0f}")
```

```text
episodi: 5
tasso di successo: 60%
successi con traiettoria valida: 67%
token medi per episodio: 4240
```

I numeri raccontano più del solo «60%». Un compito è riuscito *per caso*, con
una traiettoria sporca (nove passi, cammino non valido): conta come successo,
ma non è un comportamento su cui fare affidamento. E un fallimento è arrivato
dopo una traiettoria **valida**: l'agente ha fatto le mosse giuste ed è
inciampato all'ultimo; un caso ben diverso da chi ha sbagliato tutto. Il tasso
di successo da solo appiattisce queste differenze; costo e traiettoria le
fanno riemergere.

I benchmark seri costruiscono proprio su queste idee. **AgentBench**
{cite}`liu2023agentbench` mette gli LLM alla prova come agenti in **otto
ambienti diversi** (un sistema operativo da manovrare, un database da
interrogare, una casa simulata, un negozio online da navigare, e altri) e
misura quanti compiti ciascun modello porta a termine. Il verdetto è un utile
bagno di umiltà: anche i modelli migliori restano lontani dal risolverli
tutti, e il divario con i modelli a pesi aperti è ampio. **SWE-bench**
{cite}`jimenez2024swebench` alza ancora l'asticella con 2.294 segnalazioni di
errore prese da progetti reali su GitHub: risolvere l'issue significa produrre
una modifica al codice che fa passare i test del progetto. Al momento della
pubblicazione i sistemi migliori ne chiudevano **pochi punti percentuali**, un
numero che da allora è salito parecchio, ma la lezione resta: i compiti lunghi
e realistici sono duri, e un benchmark onesto lo mostra in cifre invece di
nasconderlo.

C'è infine una faccia della valutazione che non è una metrica ma una rete di
sicurezza. Un agente non solo *risponde*: *agisce*, e un'azione può fare danni
(eseguire codice, spendere soldi, scrivere su un database). Valgono qui,
rafforzati, gli stessi presidi che vedremo in LLMOps: i **guardrail** su
ingresso e uscita che filtrano input malevoli e azioni pericolose, e
l'**LLM-as-a-judge** come metro automatico; utile per punteggiare traiettorie
su larga scala, ma con gli stessi bias (posizione, verbosità, auto-preferenza)
di cui diffidare. La valutazione di un agente, come quella dell'output aperto,
non è mai *solo* un numero: è un numero più un sistema di controlli attorno.

## Uno sguardo onesto

Chiudiamo dove il libro chiude sempre: sull'onestà. Gli agenti sono
promettenti: l'idea di un modello che pianifica, usa strumenti, collabora e
ricorda è potente, e i primi risultati su compiti reali, per quanto modesti,
erano impensabili pochi anni fa. Ma sono anche **fragili**, e i loro difetti
non sono dettagli da limare: sono strutturali. Gli errori si **accumulano**
lungo la catena, e un compito lungo li amplifica. Il **costo** cresce con i
passi, con gli agenti, con i giri di conversazione. E l'**imprevedibilità**
che rende versatile un motore linguistico è la stessa che rende difficile
garantire cosa farà: più libertà d'azione, meno controllo.

È, soprattutto, un'area **giovane**: più euristiche che teoria, benchmark
ancora in costruzione, poche certezze su cosa funzioni e perché
{cite}`xi2023rise`. Chi lavora con gli agenti oggi costruisce su terreno che
si muove. Questo non è un motivo per starne alla larga, è un motivo per starci
con lucidità: misurare più che sperare, aggiungere complessità solo quando
paga, e diffidare di ogni numero troppo bello. La distanza tra un agente che
*sembra* funzionare in una demo e uno di cui *fidarsi* in produzione si misura
esattamente con gli strumenti di questa sezione.

```{admonition} Da ricordare
:class: important
- Oltre il ReAct passo-passo, due mosse per i compiti complessi:
  **plan-and-execute** (prima un piano di sotto-compiti, poi l'esecuzione) e i
  sistemi **multi-agente** con ruoli specializzati (pianificatore, esecutore,
  critico). Pianificare dà struttura ma un piano rigido va rifatto quando la
  realtà lo smentisce.
- I sistemi **multi-agente** (come nel framework **AutoGen**
  {cite}`wu2024autogen`, che li modella come una conversazione tra agenti) non
  sono gratis né sempre migliori: più agenti significano più costo e più modi
  di sbagliare. Si aggiunge un ruolo solo quando risolve un problema reale.
- La **memoria a lungo termine** non è tenere tutto nel contesto: i
  **generative agents** {cite}`park2023generative` memorizzano fuori,
  **recuperano** il pertinente (recency + importance + relevance) e
  **riflettono** condensando le memorie in astrazioni (lo stesso schema
  recupera-e-condensa del RAG).
- **Valutare** un agente è più duro che valutare un classificatore: nessuna
  risposta unica, compito multi-passo, ambiente che cambia. Servono **tasso di
  successo**, valutazione della **traiettoria** (non solo del risultato) e
  **costo/latenza**.
- I benchmark **AgentBench** {cite}`liu2023agentbench` (otto ambienti) e
  **SWE-bench** {cite}`jimenez2024swebench` (issue reali di GitHub) mostrano
  tassi di successo inizialmente modesti: un promemoria di onestà. In
  produzione valgono i **guardrail** e l'**LLM-as-a-judge** di LLMOps.
- Gli errori si **accumulano** sui compiti lunghi ($P(\text{successo}) \le
  (1-p)^n$): gli agenti sono promettenti ma fragili, e restano un'area
  **giovane** {cite}`xi2023rise`. Misurare più che sperare.
```
