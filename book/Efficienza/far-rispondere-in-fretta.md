# Starci non è rispondere: la mappa dell’altra metà

Le tre leve del capitolo hanno stretto il modello. Un modello stretto ci sta in
memoria, e questo era il problema del conto in apertura. Ma «ci sta» e
«risponde in fretta» sono due domande diverse, e la seconda non si risolve
rimpicciolendo.

Questa sezione non spiega come si risolve: lo spiegano altri tre punti del
libro, ciascuno per una ragione sua. Qui si dice **perché siano due domande
diverse**, e dove stiano le risposte, perché è materia che si va a cercare nel
posto sbagliato.

## Perché rispondere è un problema di traffico

Il conto che segue non misura niente e non dipende da nessuna macchina: è
aritmetica sul lavoro che c’è da fare.

`````{tab} Elementare

Hai la cantina piena di casse e la cucina al piano di sopra. Per apparecchiare
devi scendere, caricarti una cassa e risalire. La cassa pesa uguale
comunque, e le scale sono sempre quelle: **il viaggio costa lo stesso** che tu
serva una persona o venti.

Se apparecchi per venti, scendi una volta e la fatica si spalma su venti
coperti. Se apparecchi per uno, scendi una volta lo stesso, e quella fatica se
la prende un coperto solo. E se ti tocca scendere una volta per ogni coperto,
passi la giornata sulle scale e in cucina non fai niente.

Sulle scale o ai fornelli, dove ti si ferma la giornata lo decide il paragone
fra due tempi tuoi: quanti coperti apparecchieresti nel tempo di un viaggio, e
quanti coperti ti dà davvero un viaggio. Se nel tempo di un viaggio ne
apparecchi cento e da ogni viaggio ne esce uno, comandano le scale. Se da un
viaggio ne escono duecento, mentre apparecchi c’è tutto il tempo per il viaggio
dopo, e a comandare tornano le mani. Quel confine dipende dalla casa, e con
scale più corte, o mani più lente, si sposta.

Un calcolatore fa esattamente questo. I pesi del modello stanno «in cantina»,
cioè nella memoria, e per farci un conto qualunque bisogna portarli su. Sono
tanti (in un modello vero, gigabyte) e il viaggio costa uguale che li si usi
per una cosa sola o per duecento insieme.

Le casse però non pesano per forza così. Scrivere ogni peso con metà delle
cifre è una cantina in cui ogni cassa pesa la metà. In braccio ne stanno due, i
viaggi si dimezzano, e gli stessi coperti escono da metà scale, quindi il
rapporto fra coperti e viaggi raddoppia. Ecco perché togliere bit ai pesi non
serve soltanto a farli entrare in cantina. Finché a tenerti fermo sono le
scale, fa anche arrivare la cena prima.

Dalla cantina, poi, non salgono solo le casse. Ogni coperto ha la sua roba
(piatto, bicchiere, posate), e quella cresce col numero dei coperti. Per un
coperto o quattro non si sente. Per duecentocinquantasei aggiunge un ottavo al
carico, e da ogni viaggio escono meno coperti di quanti il conto pulito ne
prometta. Il conto tiene finché i coperti sono pochi rispetto a quanto pesano
le casse.

Quindi la domanda che decide tutto non è quanti conti ci sono da fare, è
**quanti conti si riescono a fare per ogni viaggio in cantina**. La tabella qui
sotto è quella domanda, messa in numeri.

`````

`````{tab} Superiore

Uno strato moltiplica una matrice di pesi $n \times n$ per un blocco di
ingressi $n \times k$, dove $k$ è quante cose si elaborano insieme. Le
operazioni in virgola mobile sono $2 n^2 k$, e il due non è un dettaglio: per
ogni casella del risultato si fanno $n$ moltiplicazioni **e altrettante
somme**, ed è la convenzione con cui il capitolo sulla GPU conta i FLOP. I byte
letti dalla memoria sono invece $n^2 b / 8$, con $b$ i bit per peso come nel
resto del capitolo, e **non dipendono da $k$**, perché i pesi sono gli stessi.

Il rapporto fra le due quantità,

$$
I = \frac{2 n^2 k}{n^2 b / 8} = \frac{16\,k}{b},
$$

è l’**intensità aritmetica**, la grandezza in ascissa del modello roofline del
capitolo sulla GPU, e il regime in cui si cade (legati alla banda o legati al
calcolo) lo decide il confronto fra $I$ e il rapporto fra prestazione di picco
e banda della macchina. Con pesi in sedici bit si semplifica in $I = k$, che è
la colonna di destra della tabella qui sotto.

Due cose che $n$ semplificandosi nasconde, e che vale la pena dire. La prima:
$I$ non dipende dalla larghezza dello strato, ma solo perché si stanno contando
i byte dei **pesi** e non quelli di ingressi e uscite, che sono $n k b / 8$
ciascuno; l’approssimazione vale per $k \ll n$ e all’ultima riga della tabella
($k = 256$ contro $n = 4096$) sbaglia già del dodici per cento. La seconda, che
è la conseguenza più utile di tutto il conto: dimezzare $b$ **raddoppia** $I$,
quindi in regime legato alla banda quantizzare non fa solo stare il modello in
memoria, lo fa anche rispondere il doppio più in fretta.

`````

```python
N = 4096                       # la larghezza di uno strato
BIT = 16                       # ogni peso in sedici bit

print(f"{'cose insieme':>13} {'conti':>16} {'byte letti':>12} {'conti per byte':>16}")
for k in (1, 4, 16, 64, 256):
    conti = 2 * N * N * k      # per ogni casella, n moltiplicazioni e n somme
    byte = N * N * BIT / 8     # i pesi si leggono una volta sola
    print(f"{k:>13} {conti/1e6:>11.0f} milioni {byte/1e6:>9.0f} MB "
          f"{conti/byte:>16.1f}")
```

```text
 cose insieme            conti   byte letti   conti per byte
            1          34 milioni        34 MB              1.0
            4         134 milioni        34 MB              4.0
           16         537 milioni        34 MB             16.0
           64        2147 milioni        34 MB             64.0
          256        8590 milioni        34 MB            256.0
```

La colonna dei byte letti non si muove mai: sono sempre gli stessi
trentaquattro megabyte di pesi, che è la cassa da portare su dalla cantina.
Quella dei conti si moltiplica. L’ultima colonna è il rapporto fra le due,
cioè quanti conti si fanno per ogni byte che si va a prendere, e nel capitolo
sulla GPU ha un nome, **intensità aritmetica**. Quando è bassa il processore
sta fermo ad aspettare i dati, e avere un processore veloce non serve a niente;
quando è alta i dati fanno in tempo ad arrivare e il processore lavora.

E adesso il punto. Quando un modello **legge** una domanda, la legge tutta
insieme: se la domanda è di duecento parole, $k$ vale duecento, e l’ultima riga
della tabella è il regime in cui ci si trova. Quando **scrive** la risposta, la
scrive una parola alla volta, perché per scegliere la parola dopo deve aver
scelto quella prima: $k$ vale **uno**, ed è la prima riga.

Sono la stessa moltiplicazione con lo stesso modello, e stanno ai due estremi
opposti della tabella. Il rapporto fra le due efficienze è, a meno dei byte
degli ingressi che il conto trascura, la lunghezza della domanda: con duecento
parole, scrivere è quasi duecento volte meno efficiente che leggere. E non perché l’operazione sia più difficile: perché è
la stessa lettura di trentaquattro megabyte di pesi, spesa per una parola sola
invece che per duecento.

Questo è il motivo per cui rimpicciolire il modello aiuta anche il tempo di
risposta (meno pesi da leggere è meno traffico), ma non basta: finché si scrive
una parola alla volta, si è nella prima riga della tabella qualunque sia la
dimensione del modello.

## Le tre risposte, e dove stanno

Da qui nascono tre idee, tutte e tre nel libro, e nessuna delle tre tocca il
modello.

**Non rifare due volte lo stesso lavoro.** I modelli che scrivono testo, per
scegliere la parola numero cinquecento, rimettono in conto tutte le
quattrocentonovantanove di prima. Ma di ciascuna di quelle parole serve un
riassunto che era già stato calcolato quando è stata scritta, quindi invece di
rifarlo lo si tiene da parte, come si tiene un segnalibro invece di rileggere
il libro da capo a ogni pagina. Quel deposito di riassunti si chiama **cache
delle chiavi e dei valori**, e il libro lo costruisce nel capitolo sui
Transformer (l’architettura di cui quei modelli sono fatti), perché è lì che si
capisce che cosa siano chiavi e valori. Non risolve il problema della tabella
qui sopra: sposta il traffico dai pesi al deposito, che cresce a ogni parola
scritta.

**Riempire la riga.** Se scrivere una parola per un solo utente sta nella prima
riga, scriverla per centoventotto utenti insieme sta nell’ultima: i pesi si
leggono una volta e servono a tutti. È il motivo per cui un servizio che
risponde a molti costa, a testa, molto meno di uno che risponde a uno, e il
{doc}`capitolo su MLOps </MLOps/overview>` lo tratta parlando di come si gestiscono le richieste che
arrivano insieme e la memoria che ciascuna si porta dietro.

**Indovinare avanti e farsi correggere.** Un modello piccolo butta giù qualche
parola di seguito, tirando a indovinare; il modello grande le controlla **tutte
in una passata sola**, che è una passata da $k$ uguale a quanto è lunga la
bozza invece che a uno. Se la bozza era giusta si sono scritte più parole al
prezzo di una; se era sbagliata si è buttato via il tempo del modello piccolo,
che è poco. È la **decodifica speculativa**, e sta nel {doc}`capitolo su MLOps </MLOps/overview>`, nella
sezione sugli LLM in produzione, con la figura che mostra la bozza accettata e
il punto in cui il modello grande la taglia.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **Starci in memoria e rispondere in fretta sono due problemi diversi.** Le
  tre leve del capitolo risolvono il primo, e aiutano il secondo solo di
  riflesso.
- La ragione è che leggere i pesi dalla memoria costa **sempre uguale**, che
  li si usi per una cosa sola o per duecento insieme. Un modello che **legge**
  una domanda di duecento parole le elabora tutte insieme; quando **scrive** la
  risposta va una parola alla volta, e paga la stessa lettura per una parola
  sola.
- Da qui le tre idee che non toccano il modello: **tenersi gli appunti** invece
  di rifarli (la cache delle chiavi e dei valori, nel capitolo sui
  Transformer), **servire molti insieme** perché i pesi letti una volta valgono
  per tutti (nel capitolo su MLOps), e **far scrivere una bozza a un modello
  piccolo** che il grande controlla tutta in una volta (la decodifica
  speculativa, ancora in MLOps).
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Per uno strato $n \times n$ applicato a un blocco $n \times k$ i FLOP sono
  $2 n^2 k$ (moltiplicazioni **e** somme, come conta il capitolo sulla GPU) e i
  byte dei pesi $n^2 b/8$: l’**intensità aritmetica** vale $16k/b$, cioè $k$ a
  sedici bit. Non dipende da $n$ solo finché si trascurano i byte di ingressi e
  uscite, il che vale per $k \ll n$.
- La generazione autoregressiva impone $k = 1$ per passo, quindi è
  strutturalmente **legata alla banda** (a sedici bit $I = 1$, contro un
  ginocchio che sugli acceleratori sta nell’ordine delle centinaia); l’elaborazione del testo in ingresso
  ha $k$ pari alla lunghezza della sequenza ed è legata al calcolo. Sono lo
  stesso modello nei due regimi opposti, ed è la ragione per cui le due fasi si
  misurano con due grandezze separate.
- Le tre leve a modello invariato agiscono tutte sullo stesso denominatore: la
  **cache delle chiavi e dei valori** elimina il ricalcolo (e sposta il costo
  sulla memoria della cache, che cresce con il contesto), il **raggruppamento
  delle richieste** alza $k$ ammortizzando la lettura dei pesi, la **decodifica
  speculativa** alza $k$ verificando in parallelo una bozza prodotta da un
  modello più economico, senza cambiare la distribuzione di uscita.
- Nessuna delle tre appartiene a questo capitolo, perché nessuna cambia il
  modello. La cache la costruisce il {doc}`capitolo sui Transformer </Transformers/overview>`, nella sezione
  sui grandi modelli linguistici; il raggruppamento delle richieste e la
  decodifica speculativa il capitolo su MLOps, nella sezione sugli LLM in
  produzione; e il modo di misurare separatamente i due regimi è ancora in
  MLOps, nella sezione sulle metriche di servizio.
```

`````

Il capitolo si chiude qui, e la cosa da portarsi dietro non è l’elenco delle
tecniche: è che ognuna delle tre leve **si paga**, e che il prezzo si conosce
solo misurandolo sul proprio modello e sui propri dati. Arrotondare a otto bit
costa l’uno per cento e a quattro molto di più; potare novanta pesi su cento
costa un punto di accuratezza e non regala un millisecondo; imitare un maestro
costa tutti gli errori del maestro. Nessuna delle tre è gratis, e il prezzo cambia da un modello all'altro: chi le
adotta senza misurarlo sul proprio sta scegliendo alla cieca.
