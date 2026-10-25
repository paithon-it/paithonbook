---
titolo_ricerca: "Harness: il primo turno di un agente di codice"
descrizione: "Che cosa preparare prima di costruire con un agente di
  programmazione: due compiti diversi, l'elenco delle funzionalità come
  struttura e un cancello."
---
# Il primo turno: preparare prima di costruire

Che cosa fa un agente di programmazione nella prima ora di un progetto vuoto,
se gli si chiede il prodotto finito? Scrive il prodotto: pagine, pulsanti, la
logica del carrello. Nessuno gli ha chiesto di rendere il progetto avviabile
con un comando, controllabile con un altro, descritto in un elenco di cose da
fare, e lui non lo fa, o lo fa di sfuggita. È il primo dei guasti che
Young aveva visto: l'agente che prova a fare tutto in un colpo, esaurisce la
finestra a metà, e lascia alla sessione dopo un prodotto iniziato su fondamenta
che nessuno ha verificato.

Nell'harness di Young [il compito della prima sessione non è scrivere il
prodotto]{.pt-matita} {cite}`young2025harnesses`. È una sessione dello stesso
agente, con le stesse istruzioni di sistema (quelle fisse che il modello riceve
in testa a ogni sessione), gli stessi strumenti e lo stesso harness di tutte le
altre; cambia soltanto il primo messaggio, che le affida un altro compito:
preparare il terreno.

Ne escono quattro oggetti. Il primo è uno script, `init.sh`, che avvia
l'applicazione in prova, sulla macchina di lavoro. Il secondo è un file di
avanzamento, in cui la prima sessione scrive che cosa ha preparato e ogni
sessione dopo annoterà che cosa ha fatto. Il terzo è un **elenco delle
funzionalità**, nel caso del clone di `claude.ai` più di duecento, ciascuna
segnata come non ancora riuscita. Il quarto è un primo salvataggio del progetto
nella storia di git, con i file creati fin lì. Nel codice d'esempio che
Anthropic ha pubblicato, la prima sessione, se le avanza tempo, può anche
cominciare la prima funzionalità {cite}`anthropic2025autonomouscoding`; ma il
suo compito è un altro, e da quel momento in poi le sessioni sono tutte uguali
fra loro, e diverse dalla prima.

## Due compiti diversi

Preparare e costruire hanno obiettivi diversi. Chi prepara deve arrivare a un
ambiente che parte, a dei controlli che girano, a una descrizione completa di
che cosa resta da fare; chi costruisce deve far avanzare una funzionalità alla
volta. Young documenta che cosa succede quando nessuno prepara: il lavoro
tentato in un colpo, e le sessioni successive che spendono il loro tempo a
capire come rimettere in piedi l'applicazione. È ragionevole aspettarsi anche
un guasto più sottile, quello delle scelte implicite: se nessuno ha fissato
quale strumento si usa per i test, una sessione ne sceglie uno e la successiva,
non trovandolo scritto, può sceglierne un altro. In altri termini, è la
decisione non scritta che ogni sessione indovina di nuovo, la stessa del
{doc}`repository come memoria
<il-repository-come-memoria>`, applicata alle fondamenta.

Separare le due fasi ha un costo, e conviene guardarlo in faccia: la prima
sessione non produce niente che un utente possa vedere. [Il costo si ripaga,
però, se il lavoro dura abbastanza]{.pt-matita}.

`````{tab} Elementare

In bottega arriva un ordine grosso: l'arredamento di un'intera biblioteca,
duecento pezzi fra scaffali, tavoli e sedie. Il primo giorno nessuno taglia una
tavola per i mobili. Al banco siede un artigiano come gli altri, ma il
capobottega gli ha dato un compito diverso: non fare mobili, preparare. Si
affilano gli attrezzi, si collega la macchina nuova e si prova che parte. Si fa
un pezzo di prova, piccolo, e lo si appoggia sulla dima, per essere sicuri che
la dima stessa funzioni. Si scrive il foglio dell'ordine, con tutti i duecento
pezzi uno sotto l'altro, ciascuno con il modo in cui verrà controllato e
accanto la casella «fatto» vuota. Si scrive sul registro la prima pagina: che
cosa si è preparato e dove sta. E la sera si scatta una fotografia della
bottega in ordine, con tutto al suo posto.

Il giorno dopo arriva un artigiano che di tutto questo non sa niente, e non ha
bisogno di saperlo: accende la macchina, guarda il foglio, sceglie il primo
pezzo non spuntato e comincia. Se qualcosa va storto, la fotografia dice com'era
la bottega quando tutto funzionava, e la si può rimettere esattamente così.

Senza quel primo giorno, il primo artigiano avrebbe cominciato subito uno
scaffale, perché uno scaffale è quello che gli è stato chiesto, e la sera ci
sarebbe stato mezzo scaffale e nessun foglio. Il secondo avrebbe dovuto
indovinare quanti pezzi mancano. E se nessuno aveva deciso quale colla si usa,
lui ne avrebbe presa una diversa da quella del primo.

Il primo giorno costa una giornata in cui non esce nessun mobile. Ogni mattina
dopo, però, risparmia all'artigiano il tempo per capire come si accende la
macchina e dove si era arrivati, diciamo mezz'ora: con giornate di otto ore,
la giornata spesa si ripaga in sedici mattine. Su duecento pezzi e molte
settimane conviene di sicuro; per uno sgabello solo no, e lo si fa e basta. Il
conto, poi, è il meno: il vantaggio vero è che gli artigiani di tutte le
mattine lavorano con la stessa macchina, la stessa colla e lo stesso foglio.

`````

`````{tab} Superiore

Fissiamo una convenzione sugli indici: $\mathbf{p}_k$, $R_k$ e $c_k$ sono
l'elenco degli esiti, il registro e il commit *alla fine* della sessione $k$.
La sessione d'inizializzazione $k = 0$ ha le stesse istruzioni $I$, lo stesso
catalogo $\mathcal{T}$ e lo stesso codice del ciclo delle altre; cambiano il
compito, $x_0$ invece di $x$, e lo stato di partenza, il repository vuoto. Il
suo prodotto non è codice del prodotto ma lo stato da cui parte la sessione
successiva, che nella notazione dell’{doc}`anatomia dell'harness <overview>`
(la sessione $k$ legge $M_k$ e lascia $M_{k+1}$) è

$$
M_1 = \big(F,\; \mathbf{p}_0 = \mathbf{0},\; R_0,\; c_0\big),
$$

dove $F = (f_1, \dots, f_N)$ è l'elenco delle funzionalità, una sequenza
ordinata per priorità, $\mathbf{p}_k \in \{0, 1\}^N$ il vettore dei loro esiti,
$R_0$ la prima nota del registro e $c_0$ il primo commit; più la ricetta
dell'ambiente, `init.sh`. Ricetta e passi di prova sono *uscite* della sessione
$0$, quindi vivono nello stato persistente $M$, non in ciò che è fissato prima
di cominciare. Separare $k = 0$ toglie alle sessioni successive due compiti,
ricostruire come si avvia e come si controlla il progetto e decidere che cosa
resta da fare, che altrimenti ciascuna rifarebbe da capo, con esiti diversi.

Il costo si ammortizza. Senza una fase separata, ogni sessione paga
un costo di ricostruzione $\rho_k$ (in token o in tempo) prima di poter
lavorare; con la fase separata si paga una volta $\kappa_0$, e poi a ogni
sessione soltanto la lettura dell'elenco e del registro. Se $\bar{\rho}$ è il
risparmio netto medio per sessione (la ricostruzione evitata, meno la lettura
che si fa comunque), la fase separata conviene quando il compito dura più di

$$
K^{*} = \frac{\kappa_0}{\bar{\rho}}
$$

sessioni. Per un compito di una sola sessione la fase non conviene mai: la
sessione $0$ ricostruisce l'ambiente almeno una volta, come qualunque sessione,
e in più scrive l'elenco e il registro, quindi $\kappa_0 > \bar{\rho}$ e
$K^{*} > 1$. È un modello, non una misura, con due riserve. La prima è che il
confronto è fatto a parità di esito, mentre senza la fase separata l'esito
cambia: una sessione che ricostruisce l'ambiente può ricostruirlo diverso, e
verosimilmente il beneficio principale sta lì, nella coerenza, più che nei
token. La seconda riguarda $\bar{\rho}$, che non cresce per forza con il
progetto: la ricostruzione evitata cresce, ma cresce anche la lettura di un
registro che si allunga a ogni sessione e di un elenco di duecento voci, che
vale migliaia di token.

`````

## L'elenco delle funzionalità come struttura dati

L'oggetto più importante che esce dal primo turno è l'elenco delle
funzionalità, e conviene guardarlo come una struttura dati, non come un
promemoria. Nell'harness di Young è un file in formato JSON, un testo
organizzato in voci e campi con un nome fisso, e ogni voce ha quattro campi:
una categoria, che raggruppa le voci (funzionali, di interfaccia); una
descrizione di che cosa l'utente deve poter fare; la lista dei passi con cui lo
si prova, dal punto di vista di chi usa l'applicazione e non di chi legge il
codice; e un campo `passes` («passa»), che vale vero o falso a seconda che la
funzionalità funzioni. Per il negozio online del {doc}`repository come memoria
<il-repository-come-memoria>`, dove l'agente lavora a un carrello, una voce
potrebbe essere questa.

```json
{
  "category": "funzionale",
  "description": "Un utente aggiunge un prodotto al carrello e vede il totale aggiornato",
  "steps": [
    "Apri la pagina di un prodotto",
    "Premi «Aggiungi al carrello»",
    "Apri il carrello e controlla che il prodotto ci sia",
    "Controlla che il totale sia il prezzo del prodotto"
  ],
  "passes": false
}
```

Ciascuna delle scelte di quel formato ha la sua ragione.

Anzitutto, le voci nascono tutte false. [L'elenco è così la definizione di
«finito» per l'intero progetto, scritta prima di cominciare]{.pt-matita}: il
lavoro è concluso quando tutte le voci passano, e una sessione che si guarda
intorno e vede molto codice scritto ha davanti il conto delle voci ancora
false, cioè molte meno ragioni per concludere che il progetto sia fatto. È la
difesa diretta contro la vittoria dichiarata.

In secondo luogo, le sessioni successive possono cambiare un campo solo,
`passes`. Le istruzioni di Young lo dicono senza mezzi termini: togliere o
modificare i test, cioè le voci e i loro passi di prova, è inaccettabile, perché
potrebbe portare a funzionalità mancanti o difettose. Nel codice d'esempio la
lista dei divieti è esplicita: non togliere voci, non riscrivere descrizioni o
passi, non unirle, non riordinarle {cite}`anthropic2025autonomouscoding`. Senza
questo vincolo, un modo rapido di far sembrare completo un elenco sarebbe
accorciarlo.

In terzo luogo, il formato è JSON e non un testo libero, e la ragione dichiarata
è empirica: Young osserva che il modello tende meno a modificare o a riscrivere
in modo inappropriato un file JSON che uno in Markdown. È un'osservazione su un
comportamento, non una legge; ma un formato con campi fissi ha anche una
conseguenza che un testo libero non ha: si confronta a macchina, voce per voce e
campo per campo.

La **granularità**, infine, conta quanto il formato. Una voce deve stare in una
sessione e deve potersi provare da utente. «Realizzare il carrello» è troppo
grande: non sta in una sessione e non dice quando è finita. «Aggiungere il campo
quantità alla tabella del carrello» è troppo piccola, e non si prova da utente.
«Un utente aggiunge un prodotto e vede il totale aggiornato» è della misura
giusta.

`````{tab} Elementare

Il foglio dell'ordine della biblioteca ha duecento righe, una per pezzo, e
sempre le stesse colonne: il tipo di pezzo, che cos'è («scaffale basso, tre
ripiani»), come lo si controlla («ci si appoggiano trenta libri e non flette;
i ripiani entrano senza forzare») e la casella «fatto». La sera del primo giorno
tutte le caselle sono vuote. Un mese dopo, chi entra e vede un terzo della sala
già arredato non può dire che il lavoro è finito: il foglio gli mostra
centoquaranta caselle ancora vuote.

Sul foglio c'è una regola scritta in rosso: si può toccare soltanto la casella.
Non si cancellano righe, non si spostano, non si riscrive il modo di controllare
un pezzo perché il pezzo è venuto male. La regola sembra pignola finché non
immagini l'artigiano stanco davanti a uno scaffale che flette con trenta libri:
è molto più facile correggere «trenta» in «dieci» che rifare lo scaffale. La
casella invece si può spuntare e si può anche svuotare: se uno scaffale già
fatto si rompe, il foglio deve dire com'è la bottega oggi, non com'era ieri.

Il primo giorno, del foglio si è fatta una copia, chiusa nell'armadio. La sera
il proprietario mette il foglio accanto alla copia e li confronta riga per riga,
colonna per colonna: se è cambiato qualcosa oltre alle caselle, lo vede subito.
Il confronto è facile proprio perché il foglio ha sempre le stesse colonne; su
un quaderno a pagine libere, scritto ogni giorno in un modo diverso, non si
saprebbe nemmeno da dove cominciare, e lo potrebbe fare una macchina solo con
il foglio a colonne.

Conta anche come sono scritte le righe. «Fare la biblioteca» è l'ordine intero,
troppo per una riga. «Piantare il terzo chiodo» è troppo poco: nessun cliente lo
controlla. Una riga buona è un pezzo che si finisce in un giorno e che il
cliente può provare con le sue mani.

`````

`````{tab} Superiore

L'elenco fissa un vincolo di ammissibilità strutturale. Sia $F$, con le sue
descrizioni, i suoi passi di prova e il suo ordine, fissato dalla sessione $0$.
Una transizione fra due sessioni è ammessa se non tocca $F$ e se ogni esito
positivo viene da una verifica:

$$
F_{k} = F_{k-1} \ \wedge\ \mathbf{p}_k \in \{0,1\}^N \ \wedge\
\big(p_{k,i} = 1 \Rightarrow V_i = 1\big),
$$

dove $V_i$ è il collaudo della voce $f_i$, cioè i suoi passi di prova eseguiti
sull'applicazione; il registro e il commit cambiano, ovviamente, a ogni
sessione. Il vincolo non impedisce che $\lVert \mathbf{p}_k \rVert_1 / N$ sia
l'obiettivo, perché lo è per costruzione: impedisce che lo si possa migliorare
cambiando il metro invece del prodotto. Togliere le voci difficili, indebolirne
i passi o spezzarne una in dieci voci facili sono forme di manomissione dello
strumento di misura, quella che nell'apprendimento per rinforzo si chiama
*reward tampering* {cite}`everitt2021reward`. Il fenomeno vicino, ma diverso, è
quello della legge di Goodhart nella formulazione di Strathern, una misura che
diventa un obiettivo cessa di essere una buona misura
{cite}`strathern1997improving`, di cui la {doc}`sezione su esplorazione e
ricompensa </DeepReinforcementLearning/esplorazione-e-ricompensa>` discute la
forma generale: quello resta anche con $F$ fisso, per esempio quando si
ottimizzano i passi scritti invece della funzionalità che descrivono, o quando
si scrive $p_{k,i} = 1$ senza aver collaudato. Il vincolo non impone neppure la
monotonia di $\mathbf{p}$: una funzionalità che passava può regredire, e deve
poter tornare a zero, ma la non monotonia permette soltanto di *registrare* una
regressione, e scoprirla tocca a qualcuno che riprova le voci già riuscite (nel
codice d'esempio di Anthropic l'agente ne riprova una o due, le più centrali, a
ogni inizio di sessione).

Il formato decide se il vincolo si può controllare. Su un documento in prosa
«$F_k = F_{k-1}$» non ha una verifica meccanica; su un formato con schema fisso
(un JSON, o anche una tabella in Markdown con colonne fisse) diventa un
confronto voce per voce, eseguibile come un cancello a ogni fine sessione.
L'osservazione di Young sul JSON, che il modello lo riscrive meno volentieri,
riduce la frequenza delle violazioni; il confronto le rende rilevabili, ed è la
garanzia che conta. La granularità ha due vincoli, uno per lato: ogni $f_i$
deve stare in una sessione, altrimenti nessuna sessione può chiudere la voce e
la misura non si muove; e deve avere un predicato di successo osservabile
dall'esterno del codice, i passi di prova, che è la condizione perché `passes`
possa essere una verifica e non un'opinione.

`````

## Un cancello per l'elenco

Nell'harness di Young il vincolo sull'elenco è affidato alle istruzioni: il
modello è pregato, con parole forti, di toccare soltanto `passes`. Un harness
può fare di più, e farlo rispettare con un controllo. In concreto, il programma
confronta l'elenco com'era nell'ultimo salvataggio con quello che la sessione
lascia, e lo rifiuta se è cambiato qualcosa oltre ai valori vero o falso dei
campi `passes`: una voce tolta o aggiunta, due voci con la stessa descrizione,
una voce spostata, un campo aggiunto o tolto, una descrizione o un passo di
prova riscritti. Lo si prova su due sessioni finte, una onesta e una che,
davanti al carrello che non funziona, toglie la voce del pagamento e
ammorbidisce la prova del carrello.

```python
import copy

# L'elenco com'era nell'ultimo salvataggio: il ciclo lo rilegge dalla storia
# di git, non dal file che la sessione ha avuto in mano.
elenco = [
    {"category": "funzionale",
     "description": "Un utente aggiunge un prodotto e vede il totale",
     "steps": ["Apri un prodotto", "Premi «Aggiungi»", "Controlla il totale"],
     "passes": False},
    {"category": "funzionale",
     "description": "Un utente paga con la carta",
     "steps": ["Apri il carrello", "Inserisci la carta", "Ricevi la conferma"],
     "passes": False},
    {"category": "interfaccia",
     "description": "La pagina del carrello si legge su un telefono",
     "steps": ["Apri il carrello a 390 pixel", "Nessun testo esce dallo schermo"],
     "passes": False},
]

def controlla(prima, dopo):
    """Rifiuta ogni cambiamento che non sia il valore vero/falso di `passes`."""
    errori = []
    nomi_prima = [v["description"] for v in prima]
    nomi_dopo = [v["description"] for v in dopo]
    if len(set(nomi_dopo)) != len(nomi_dopo):
        errori.append("due voci con la stessa descrizione")
    for nome in sorted(set(nomi_prima) - set(nomi_dopo)):
        errori.append(f"voce tolta: {nome}")
    for nome in sorted(set(nomi_dopo) - set(nomi_prima)):
        errori.append(f"voce nuova o descrizione riscritta: {nome}")
    comuni = set(nomi_prima) & set(nomi_dopo)
    ordine_prima = [n for n in nomi_prima if n in comuni]
    ordine_dopo = [n for n in nomi_dopo if n in comuni]
    if ordine_prima != ordine_dopo:
        errori.append("voci riordinate")
    vecchie = {v["description"]: v for v in prima}
    nuove = {v["description"]: v for v in dopo}
    for nome in sorted(comuni):
        a, b = vecchie[nome], nuove[nome]
        if a.keys() != b.keys():
            errori.append(f"campi aggiunti o tolti in: {nome}")
            continue
        for campo in sorted(a):
            if campo != "passes" and a[campo] != b[campo]:
                errori.append(f"campo «{campo}» cambiato in: {nome}")
        if not isinstance(b["passes"], bool):
            errori.append(f"`passes` non è vero o falso in: {nome}")
    return errori

def avanzamento(voci):
    """Il contatore ingenuo: conta sull'elenco com'è adesso."""
    return f"{sum(v['passes'] is True for v in voci)}/{len(voci)}"

# Sessione onesta: il carrello ora funziona, e si spunta soltanto quello.
onesta = copy.deepcopy(elenco)
onesta[0]["passes"] = True

# Sessione furba: il pagamento è difficile e sparisce; la prova del carrello
# perde il controllo del totale, e il carrello risulta fatto.
furba = copy.deepcopy(elenco)
del furba[1]
furba[0]["steps"] = ["Apri un prodotto", "Premi «Aggiungi»"]
furba[0]["passes"] = True

for nome, dopo in [("onesta", onesta), ("furba", furba)]:
    errori = controlla(elenco, dopo)
    esito = "accettata" if not errori else "RIFIUTATA"
    print(f"sessione {nome}: avanzamento {avanzamento(dopo)}, {esito}")
    for e in errori:
        print("   ", e)
```

```text
sessione onesta: avanzamento 1/3, accettata
sessione furba: avanzamento 1/2, RIFIUTATA
    voce tolta: Un utente paga con la carta
    campo «steps» cambiato in: Un utente aggiunge un prodotto e vede il totale
```

La sessione furba, letta dal solo contatore, sembra andata meglio: una voce su
due invece di una su tre. È il **contatore ingenuo**, quello che conta
sull'elenco com'è adesso, e il codice d'esempio di Anthropic misura
l'avanzamento proprio così; per questo il confronto con l'elenco salvato serve.
Il cancello ferma la sessione furba per la ragione giusta, nominando la voce
sparita e il passo di prova indebolito. Due condizioni lo rendono una garanzia
e non un controllo di cortesia: l'elenco di partenza va letto dalla storia di
git, dove la sessione non può riscriverlo, e il controllo lo lancia il ciclo a
fine sessione, non l'agente quando gli pare. E il controllo
non sa se il carrello funziona davvero: a deciderlo, cioè a scrivere il valore
di `passes`, è il collaudo da utente. [Il controllo sa soltanto che nessuno ha
spostato il metro mentre misurava]{.pt-matita}.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il primo turno non costruisce il prodotto: prepara la bottega. Ne escono il
  modo di accendere le macchine (uno script), la prima pagina del registro, il
  foglio dell'ordine con tutti i pezzi e la fotografia della bottega in ordine
  (il primo salvataggio, a cui si può sempre tornare).
- Il primo giorno senza mobili si ripaga se il lavoro dura molti giorni; per
  un lavoro piccolo no. E il vantaggio più grande è che tutti lavorano con la
  stessa macchina, la stessa colla e lo stesso foglio.
- Il foglio dell'ordine nasce con tutte le caselle vuote, e così nessuno può
  dire «è finito» guardando la sala piena di mobili.
- Sul foglio si può toccare soltanto la casella (spuntarla, o svuotarla se un
  pezzo si rompe): non si cancellano righe, non si spostano e non si
  ammorbidiscono i controlli. Il foglio ha colonne fisse, e così la sera lo si
  confronta con la copia chiusa nell'armadio, anche a macchina.
- Il confronto dice se qualcuno ha toccato il foglio, non se i mobili sono
  buoni: quello lo dice il collaudo.
- Una riga buona è un pezzo che si finisce in un giorno e che il cliente può
  provare con le sue mani.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- La sessione $0$ ha le stesse istruzioni, lo stesso catalogo e lo stesso
  ciclo, un compito diverso e il repository vuoto come partenza; produce
  $M_1 = (F, \mathbf{0}, R_0, c_0)$ e la ricetta dell'ambiente
  {cite}`young2025harnesses`.
- La fase separata si ammortizza dopo $K^{*} = \kappa_0/\bar{\rho}$ sessioni,
  con $K^{*} > 1$ per costruzione; il confronto è a parità di esito, e il
  beneficio principale è verosimilmente la coerenza.
- Ammissibilità: $F_k = F_{k-1}$ e $p_{k,i} = 1 \Rightarrow V_i = 1$. Il
  vincolo impedisce di migliorare la misura cambiando il metro (*reward
  tampering*), non il Goodhart che resta con $F$ fisso; $\mathbf{p}$ non è
  monotono, e le regressioni le trova solo chi riprova le voci riuscite.
- Schema fisso: il vincolo diventa un confronto voce per voce, eseguito dal
  ciclo su un elenco di partenza letto dalla storia, dove l'agente non può
  scrivere.
- Ogni voce sta in una sessione e ha un predicato di successo osservabile da
  utente, i passi di prova.
```

`````
