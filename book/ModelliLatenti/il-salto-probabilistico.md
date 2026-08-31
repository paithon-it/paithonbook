# Il salto probabilistico: l’ELBO e la riparametrizzazione

La sezione precedente si è chiusa con una diagnosi: la pagella era sbagliata.
Chiedere «la copia somiglia all’originale?» produce un archivio che si rilegge
benissimo e da cui non si può pescare, perché sull’ordine del cassetto quella
domanda non dice niente.

Cambiamola, allora, e cambiamola in modo radicale. La domanda nuova è: **quanto
era probabile che uscisse proprio questa cifra?** A prima vista sembra un
peggioramento, perché è più astratta e più difficile da calcolare. Il resto
della sezione racconta come mai è invece esattamente la domanda giusta, e come
mai la regola che alla sezione precedente mancava, quella su dove vanno messe
le schede, non bisogna aggiungerla: **cade fuori da sola** dal tentativo di
rispondere.

Il percorso è in quattro passi, e conviene averli in testa: il conto che non
si può fare; il modo di aggirarlo chiedendo aiuto a chi sa dove guardare; la
formula che ne esce, che si chiama **ELBO** (dall’inglese *evidence lower
bound*, cioè «limite inferiore» di quel numero che non si sa calcolare) e che
è appunto quella stima prudente, con due termini dai significati netti; e il
trucco tecnico senza il quale niente di tutto questo si potrebbe addestrare.

## Il conto che non si può fare

L’apertura del capitolo lo ha già detto a parole, con i sacchetti di biglie: la
probabilità di un dato è la somma, su tutte le cause nascoste possibili, di
quanto ciascuna lo spiega, contata per quanto quella causa stessa era
probabile. Con due sacchetti sono due addendi. Con una scheda di otto numeri
sono infiniti.

`````{tab} Elementare

L’istinto dice: se sommare tutto non si può, si tira a sorte. Pesco mille
schede a caso, guardo quanto ciascuna spiega bene la cifra che ho in mano,
faccio la media, e ho una stima. È un metodo onesto e in tanti problemi
funziona.

Qui non funziona, e la ragione è semplice da dire e sorprendente da vedere: fra
tutte le schede possibili, quelle che spiegano *questa* cifra sono
pochissime. Pescandone mille a caso, quasi tutte descrivono qualcosa che con
la nostra cifra non c’entra niente, e valgono zero. Il valore vero sta tutto
dentro le pochissime che hanno avuto fortuna, e se quelle non capita di
pescarle, la media viene fuori troppo bassa e nessuno se ne accorge.

Il guaio peggiora in fretta man mano che la scheda si allunga. Con una scheda
da un numero solo la fortuna capita quasi sempre; con una da quaranta non
capita mai. È lo stesso motivo per cui indovinare una parola di quattro lettere
tirando a caso si può fare e indovinarne una di quaranta no: con quattro
lettere dell’alfabeto italiano le parole possibili sono circa duecentomila, con
quaranta il loro numero ha cinquantatré cifre. Le possibilità non crescono, si
moltiplicano.

`````

`````{tab} Superiore

La stima Monte Carlo dal prior,

$$
\hat{p}(\mathbf{x}) = \frac{1}{S} \sum_{s=1}^{S}
p_\theta(\mathbf{x} \mid \mathbf{z}^{(s)}),
\qquad \mathbf{z}^{(s)} \sim p(\mathbf{z}),
$$

dove $S$ è il numero di campioni e $\mathbf{z}^{(s)}$ l’$s$-esimo, è **non
distorta** per $p_\theta(\mathbf{x})$ ma inutile in pratica. Il motivo è che
$p_\theta(\mathbf{x} \mid \mathbf{z})$, come funzione di $\mathbf{z}$, è
concentrata in una regione la cui massa sotto il prior decade
esponenzialmente con $L$: la somma è dominata da pochissimi termini, e la sua
varianza relativa cresce con la stessa legge.

Due conseguenze si misurano su un caso in cui $p_\theta(\mathbf{x})$ si
conosce in forma chiusa. La prima: in scala logaritmica la stima è **distorta
verso il basso**, perché $\log$ è concava e la
disuguaglianza di Jensen impone
$\mathbb{E}[\log \hat{p}] \le \log \mathbb{E}[\hat{p}] = \log p_\theta(\mathbf{x})$.
La seconda: la quota che il campione più grosso si prende sul totale è la
misura diretta del guasto, e passa da una briciola a un terzo del totale
mentre $L$ va da 1 a 40.

`````

Il modello del blocco che segue è un giocattolo, scelto apposta perché la
risposta giusta si conosce in anticipo e ci si può confrontare invece di
fidarsi. La causa nascosta è un numero sorteggiato attorno allo zero, il dato è
quella causa più un po’ di scarto sorteggiato anche lui, e in un caso così
semplice la probabilità del dato si sa scrivere con carta e penna.

Nella tabella «dimensioni» vuol dire quanti numeri ha la causa nascosta, ed è
il conto che si allunga da uno a quaranta. Le colonne «vero» e «stimato» sono
scritte **in scala logaritmica**, cioè schiacciate col logaritmo dei richiami
di matematica: senza, sarebbero numeri con decine di zeri dopo la virgola e non
si guarderebbero. La colonna che conta è quella dell’errore, che è la loro
differenza.

```python
import math
import torch

torch.manual_seed(0)
torch.set_num_threads(1)      # numeri riproducibili su qualunque macchina

# modello giocattolo: z ~ N(0, I), x|z ~ N(z, sigma^2 I). Qui p(x) si sa:
# marginalizzando due gaussiane ne esce una sola, N(0, (1 + sigma^2) I).
SIGMA, CAMPIONI = 0.5, 100_000


def log_p_vero(x):
    var = 1 + SIGMA ** 2
    return (-0.5 * (x ** 2).sum() / var
            - 0.5 * len(x) * math.log(2 * math.pi * var)).item()


def log_p_stimato(x, campioni=CAMPIONI):
    """La media di p(x|z) su z sorteggiati dal prior, in scala logaritmica."""
    z = torch.randn(campioni, len(x))
    log_p_x_dato_z = (-0.5 * ((x - z) ** 2).sum(1) / SIGMA ** 2
                      - 0.5 * len(x) * math.log(2 * math.pi * SIGMA ** 2))
    return torch.logsumexp(log_p_x_dato_z, 0).item() - math.log(campioni), log_p_x_dato_z


print(f"{'dimensioni':>10} {'log p(x) vero':>14} {'stimato':>10} "
      f"{'errore':>8} {'peso del piu grosso':>21}")
for L in (1, 2, 5, 10, 20, 40):
    x = torch.full((L,), 0.6)          # un dato qualunque, lo stesso in ogni dimensione
    stima, pesi = log_p_stimato(x)
    quota = (pesi.max() - torch.logsumexp(pesi, 0)).exp().item()
    print(f"{L:>10} {log_p_vero(x):>14.2f} {stima:>10.2f} "
          f"{stima - log_p_vero(x):>8.2f} {quota:>20.1%}")
```

```text
dimensioni  log p(x) vero    stimato   errore   peso del piu grosso
         1          -1.17      -1.17     0.00                 0.0%
         2          -2.35      -2.35     0.00                 0.0%
         5          -5.87      -5.86     0.01                 0.1%
        10         -11.75     -11.83    -0.08                 2.1%
        20         -23.49     -24.27    -0.78                20.6%
        40         -46.98     -57.04   -10.06                35.1%
```

Con una causa nascosta da un numero solo, centomila sorteggi danno la risposta
esatta a due cifre decimali. Con quaranta numeri sbagliano di dieci nat, e
dieci nat non vogliono dire «un po’»: i nat si sommano dove le probabilità si
moltiplicano, quindi dieci nat di scarto sono una probabilità stimata
**ventiduemila volte più piccola** di quella vera. E l’errore, salendo di
dimensione, è tutto dalla stessa parte: **per difetto**. (Nelle poche
dimensioni la stima balla in tutti e due i versi, e infatti a cinque il segno
è positivo per un centesimo: la spinta verso il basso è una tendenza, e diventa
schiacciante quando le dimensioni crescono.) La colonna a
destra dice perché: su centomila sorteggi, uno solo si prende il trentacinque
per cento del totale. Non stiamo facendo una media, stiamo aspettando un colpo
di fortuna.

E la scheda di questa sezione di numeri ne ha otto soltanto. Quella che permette a
**Stable Diffusion**, il generatore di immagini che il libro racconta più
avanti, di girare su un computer di casa, ne ha sedicimila.

## Chiedere a chi sa dove guardare

Se il problema è che si pesca nel posto sbagliato, la soluzione è pescare nel
posto giusto. E chi sa dov’è il posto giusto? Chi ha la cifra sotto gli occhi:
l’archivista.

Ecco allora la mossa, e conviene dirla prima in italiano che in formule.
Invece di sorteggiare schede alla cieca, chiediamo all’archivista di
**proporre** lui le poche schede che valga la pena guardare per *questa*
cifra. Poi correggiamo il conto per tenere conto del fatto che le schede non
le abbiamo pescate a caso, ma ce le siamo fatte suggerire. È lo stesso
mestiere che fa l’encoder della sezione precedente, con una differenza sola:
non propone una scheda, propone **una zona**.

`````{tab} Elementare

Devi trovare una persona in una città che non conosci. Il metodo a sorte è
aprire l’elenco del telefono a caso e chiamare: in una città grande non la
trovi mai. Il metodo sensato è chiedere a qualcuno che la conosce «in che
quartiere abita?», andare lì, e cercare in quel quartiere.

Cercando solo dove ha detto lui non si perde niente: il conto si corregge
apposta per il fatto che si è guardato in una fetta sola, e resta giusto. (Un
quartiere che in quella città non esiste manderebbe all’aria tutto, e per
fortuna non capita.) Quello che ne esce, però, è una **stima prudente** e non
la probabilità vera, cioè un numero che sta sicuramente sotto a quello giusto.

Perché sotto e non sopra? Non per via della fetta, ma per l’ordine di due
operazioni. I numeri in gioco sono minuscoli, e per maneggiarli si
**schiacciano**, cioè di ciascuno si tiene solo l’ordine di grandezza. Prendi 1
e 100: la media è 50,5, ma schiacciati diventano 0 e 2, la cui media è 1, cioè
10. Il 100, che nella media si prendeva quasi tutto, schiacciato non pesa quasi
niente. Il conto che sappiamo fare è quello schiacciato; il valore vero è
l’altro, e sta sempre più in alto.

Il divario fra la stima e il vero dipende da una cosa sola, da quanto il
consiglio era buono: se il conoscente sapeva davvero il quartiere, il divario
è quasi zero; se ha tirato a indovinare, è grande.

Qui c’è il regalo, ed è la ragione per cui tutto questo funziona. Noi vorremmo
due cose: un modello che spieghi bene i dati, e un archivista che sappia dire
dove guardare. Spingendo in alto la stima prudente si lavora su tutte e due
insieme, perché quel numero sale sia quando il modello migliora, sia quando il
consiglio dell’archivista si fa più preciso. Una sola cosa da spingere in alto,
due mestieri che imparano. (Che salgano davvero tutti e due, e non uno a spese
dell’altro, è quasi sempre vero e non sempre.)

`````

`````{tab} Superiore

Si introduce una distribuzione ausiliaria $q_\phi(\mathbf{z} \mid \mathbf{x})$,
detta **modello di inferenza** o posterior approssimata, con parametri $\phi$,
e si scrive un’identità esatta. Poiché $p_\theta(\mathbf{x})$ non dipende da
$\mathbf{z}$, la si può mettere dentro un valore atteso rispetto a
$q_\phi$ senza cambiarla:

$$
\log p_\theta(\mathbf{x})
= \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}
\!\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})}\right]
+ \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}
\!\left[\log \frac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p_\theta(\mathbf{z} \mid \mathbf{x})}\right],
$$

dove i passaggi sono tre e conviene contarli: la log-verosimiglianza si può
mettere dentro un valore atteso rispetto a $q_\phi$ perché non dipende da
$\mathbf{z}$ e perché $q_\phi$ è normalizzata; poi si applica la definizione di
probabilità condizionata, $p_\theta(\mathbf{z} \mid \mathbf{x}) =
p_\theta(\mathbf{x}, \mathbf{z}) / p_\theta(\mathbf{x})$; e infine si
moltiplica e si divide per $q_\phi(\mathbf{z} \mid \mathbf{x})$ dentro il
logaritmo, spezzandolo poi in due. È quest’ultimo passaggio, non i primi due,
a far comparire l’ELBO. (Perché il secondo dei due addendi sia finito serve che
$p_\theta(\mathbf{z} \mid \mathbf{x})$ sia positiva ovunque lo sia
$q_\phi(\mathbf{z} \mid \mathbf{x})$, cioè che l’archivista non proponga zone
che il modello dichiara impossibili. Qui la condizione è soddisfatta sempre,
ma per due ragioni e non per una: la posterior vera è proporzionale a
$p_\theta(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})$, e con un prior
gaussiano e una verosimiglianza positiva ovunque nessuno dei due fattori si
annulla.) I due addendi hanno un
nome: il primo è l’**ELBO** (*evidence lower bound*, limite inferiore
dell’evidenza), il secondo è la divergenza di Kullback–Leibler fra la posterior
approssimata e quella vera. Quindi

$$
\log p_\theta(\mathbf{x}) = \mathcal{E}_{\theta,\phi}(\mathbf{x})
+ D_{\mathrm{KL}}\!\big(q_\phi(\mathbf{z} \mid \mathbf{x})
\,\|\, p_\theta(\mathbf{z} \mid \mathbf{x})\big)
\;\ge\; \mathcal{E}_{\theta,\phi}(\mathbf{x}),
$$

dove $\mathcal{E}_{\theta,\phi}(\mathbf{x})$ è l’ELBO, e la disuguaglianza vale
perché una divergenza di Kullback–Leibler non è mai negativa. Ne discendono i
due fatti che reggono tutto il metodo {cite}`kingma2019introduction`:

- il **divario** fra l’ELBO e la log-verosimiglianza vera *è* la distanza fra
  la posterior approssimata e quella vera. Non la limita, la eguaglia. Un
  encoder perfetto rende l’ELBO esatto;
- massimizzando l’ELBO rispetto a $\theta$ e $\phi$ insieme si ottengono due
  cose con un’azione sola: si spinge in alto (approssimativamente)
  $\log p_\theta(\mathbf{x})$, cioè si migliora il modello generativo, e si
  stringe il divario, cioè si migliora l’encoder. È il «due al prezzo di uno»
  di Kingma e Welling.

Il capitolo sui modelli di diffusione, più avanti, userà una versione ripesata
di questo stesso limite, e chi ci arriverà riconoscerà l’oggetto. E chi arriva
dal capitolo sul machine learning riconosce la struttura dell’algoritmo EM,
che alterna il miglioramento del bound rispetto a $q$ e rispetto a $\theta$; la
differenza è che qui $q$ non si calcola in forma chiusa, si **apprende**.

`````

```{figure} ../figures/elbo-il-divario.svg
:name: fig-elbo-divario
:alt: "Un grafico con i nat sull’asse verticale e l’addestramento su quello orizzontale. Due curve salgono verso destra. Quella in alto è una riga spessa che sale piano. Quella sotto parte molto più in basso e sale più in fretta, avvicinandosi alla prima senza mai raggiungerla. Due doppie frecce verticali misurano lo spazio fra le due curve, una nella prima metà e una più a destra, e la seconda è molto più corta della prima. Sotto il grafico una legenda in tre righe: la riga spessa è «quanto era probabile il dato, per davvero», che non si sa calcolare e «sale anche lui mentre il modello migliora»; la curva è «la stima prudente, che spingiamo in su», che sale per tutte e due le ragioni, modello migliore e divario più stretto; la doppia freccia è «il divario», quanto l’archivista sbaglia a dire dove guardare, e si stringe da sé."
:width: 82%

Il limite e il divario. La riga in alto è il valore che vorremmo e non sappiamo
calcolare; la curva è quello che calcoliamo e spingiamo in su. La distanza fra
le due misura esattamente quanto la zona proposta dall’archivista differisce
da quella giusta. Salgono tutte e due, ed è il
punto: la curva guadagna sia perché il tetto si alza, sia perché lo raggiunge
meglio.
```

Della {numref}`fig-elbo-divario` conviene fissare una cosa sola, perché è la
sola che serve: **spingendo in su la curva si guadagna quasi sempre da tutte e
due le parti**. O sale perché il modello descrive meglio i dati, o sale perché
il divario si stringe. Che salga soltanto per il secondo motivo, mentre il
modello peggiora, è possibile e ogni tanto succede; ma è l’eccezione, e in
cambio si ottiene una cosa che si addestra come qualunque altra rete.

## I due termini, e il costo di descrizione

Quella stima prudente ha un nome, ed è il nome che dà il titolo alla sezione:
si chiama **ELBO**. Da qui in avanti «ELBO» e «stima prudente» vogliono dire la
stessa identica cosa, e conviene tenerselo perché la parola compare dappertutto,
nei programmi come nei paper.

Scritta tutta insieme, quella stima è compatta e opaca. Spezzata in due pezzi
diventa la cosa che si programma, e quei due pezzi hanno un significato da
prendere sul serio.

`````{tab} Elementare

Il conto si spezza in due voci, e sono le due voci di una spesa.

**Prima voce: quanto male ridipinge il copista.** È la stessa della sezione
precedente, nient’altro che il vecchio «la copia somiglia all’originale?».

**Seconda voce: quanto costa scrivere la scheda.** Qui c’è la novità, ed è la
regola che mancava. Archivista e copista si sono messi d’accordo in anticipo su
un **vocabolario comune**: un modo standard di descrivere un quadro, che vale
per tutti i quadri e non è stato adattato a nessuno. Quando
l’archivista scrive una scheda, paga solo per quello che si discosta da quel
vocabolario. Descrivere un quadro come «uno dei soliti» non costa niente;
descriverlo nel dettaglio, con precisione al millimetro, costa molto.

L’archivista si trova quindi stretto fra due spinte opposte, e questo è il
cuore di tutto. Se resta sul vago, la scheda costa poco e il copista dipinge
male. Se è precisissimo, il copista dipinge benissimo e la scheda costa
un’esagerazione. Il punto di equilibrio è la scheda **più vaga che ancora
basta**: gli si chiede di essere impreciso quanto può permettersi.

Ed è quella imprecisione voluta a riempire i buchi della sezione precedente. Se
ogni quadro non è descritto da un punto ma da un alone, gli aloni di quadri
diversi si toccano, sulla mappa non restano zone vuote, e una scheda inventata
cade dentro l’alone di qualcuno.

C’è una seconda conseguenza, meno ovvia. Il vocabolario comune è uno solo e sta
in un posto solo: quindi pagare poco non vuol dire soltanto essere vaghi, vuol
dire anche **stare lì attorno**. Le schede si raccolgono tutte nella stessa
zona, che è poi la zona in cui si andrà a pescare, ed è per questo che pescare
funziona.

Il punto di rottura c’è, e va detto: se il vocabolario comune è troppo povero,
o se il copista è troppo bravo a cavarsela da solo, all’archivista conviene non
scrivere niente. Costo zero, e il copista dipinge sempre lo stesso quadro
medio. Succede davvero, si chiama **collasso della posterior** (cioè: la zona
proposta dall’archivista è collassata sul vocabolario comune, e non dice più
niente sul singolo quadro).

`````

`````{tab} Superiore

Spezzando $p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} \mid
\mathbf{z})\, p(\mathbf{z})$ dentro il logaritmo, l’ELBO si riscrive

$$
\mathcal{E}_{\theta,\phi}(\mathbf{x}) =
\underbrace{\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}
\big[\log p_\theta(\mathbf{x} \mid \mathbf{z})\big]}_{\text{ricostruzione}}
\;-\;
\underbrace{D_{\mathrm{KL}}\!\big(q_\phi(\mathbf{z} \mid \mathbf{x})
\,\|\, p(\mathbf{z})\big)}_{\text{costo di descrizione}},
$$

dove il primo termine premia i codici da cui il dato si ricostruisce bene e il
secondo penalizza gli encoder che si allontanano dal prior. Il secondo è
esattamente la regolarizzazione che la sezione precedente cercava, e il punto è
che **non è stata aggiunta**: è comparsa spezzando in due un’identità.

La lettura come costo di codifica è precisa e non è una metafora. La
divergenza di Kullback–Leibler dei richiami di matematica misura quanto si paga
in più codificando con la distribuzione sbagliata (là il conto è in bit, qui in
nat: cambia solo la base del logaritmo); qui è il sovrapprezzo
di descrivere $\mathbf{z}$ con la posterior specifica di quel dato invece che
con il codice comune $p(\mathbf{z})$. Il **negativo** dell’ELBO è quindi, alla
lettera, un costo di descrizione totale: i nat spesi per la scheda più i nat
spesi per rifare il dato a partire dalla scheda. Massimizzare l’ELBO è
minimizzare quel costo, che è la formulazione a *minimum description length*
del metodo.

Con prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$ e posterior gaussiana a
covarianza diagonale, il secondo termine si scrive in forma chiusa e non serve
stimarlo:

$$
D_{\mathrm{KL}} = \frac{1}{2} \sum_{j=1}^{L}
\Big( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \Big),
$$

dove $\mu_j$ e $\sigma_j^2$ sono media e varianza della $j$-esima componente
prodotte dall’encoder, e $L$ è la dimensione del latente. Si annulla, come si
vede, quando $\mu_j = 0$ e $\sigma_j^2 = 1$ per ogni $j$, cioè quando l’encoder
ignora il dato e restituisce il prior: è il minimo di quel termine, ed è anche
il modo in cui il metodo può fallire.

`````

## Il trucco della riparametrizzazione

Manca un pezzo, tecnico e decisivo: senza, niente di tutto
questo si potrebbe addestrare in un tempo ragionevole.

Il problema è che nel mezzo del conto c’è un **sorteggio**. L’archivista non
consegna una scheda: consegna una zona, e da quella zona si pesca. La
correzione che deve tornargli indietro riguarda la zona, non il singolo punto
pescato; ma quello che il copista ha visto è il punto.

Ed è qui che la macchina si inceppa. Una rete impara perché la correzione
risale all’indietro, dal voto finale fino a ciascuno dei suoi numeri interni,
un pezzo alla volta: è la procedura che il capitolo sulle reti neurali chiama
*backpropagation*. Ma quella risalita ha bisogno, a ogni pezzo, di una domanda
a cui si sappia rispondere: «se sposto un pochino questo, di quanto cambia
quello?». Davanti a un sorteggio la domanda non ha risposta, perché il numero
uscito è uscito a caso, e la risalita si ferma lì.

`````{tab} Elementare

Giochi a freccette, e vuoi sapere una cosa precisa: se sposto la mira di un
centimetro a destra, il punteggio medio sale o scende? Il modo diretto è
provare: sposti la mira e tiri altre dieci freccette. Ma quelle dieci sono
andate dove sono andate anche per conto loro, e il punteggio è cambiato per due
motivi mescolati, lo spostamento e la fortuna. Per districarli servono migliaia
di tiri.

Il trucco è decidere gli **scarti prima**, e tenerli. Stabilisci in anticipo:
questa freccetta cade tre centimetri sopra il punto di mira, la seconda uno a
destra, la terza due sotto. Adesso sposti la mira, e tutte e dieci le freccette
si spostano insieme a lei, rigidamente, perché il loro scarto dal punto di mira
è inchiodato. Il punteggio cambia solo per lo spostamento, e con dieci tiri lo
vedi.

La formula è tutta qui: invece di dire «pesco un punto dalla zona», si dice
«prendo uno scarto a caso, e poi lo appoggio dove sta la zona, allargandolo
quanto è larga la zona». Il caso è finito fuori, in un pezzo che non dipende da
niente di ciò che vogliamo aggiustare, e la strada per le correzioni resta
aperta.

C’è anche un altro modo di rispondere alla domanda: invece di seguire dove va
la freccetta, si tiene conto di **quanto era probabile che finisse proprio
lì**. Funziona, non imbroglia, e si usa quando gli scarti non si possono
decidere prima. Ma la mano trema molto di più, e la differenza si misura.

Il punto di rottura, che serve alla sezione seguente: il trucco degli scarti
decisi prima si può fare **solo** se la zona è una di quelle che si spostano e
si allargano, come una nuvola su un piano. Se la scelta nascosta fosse «quale dei dieci cassetti», non
esisterebbe nessuno scarto da decidere in anticipo, perché fra il cassetto tre
e il cassetto quattro non c’è niente in mezzo. E lì il trucco non si applica.

`````

`````{tab} Superiore

Il gradiente rispetto a $\theta$ non pone problemi, perché $\theta$ non compare
nella distribuzione su cui si prende il valore atteso e l’operatore di derivata
entra dentro. Rispetto a $\phi$ sì:

$$
\nabla_\phi\, \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[f(\mathbf{z})]
\;\ne\;
\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[\nabla_\phi f(\mathbf{z})],
$$

perché è la misura stessa a dipendere da $\phi$. Il **trucco della
riparametrizzazione**, proposto indipendentemente da Kingma e Welling
{cite}`kingma2014auto` e da Rezende, Mohamed e Wierstra
{cite}`rezende2014stochastic`, riscrive la variabile aleatoria come funzione
derivabile di una sorgente di rumore che di $\phi$ non sa niente (e non furono
i primi: la monografia di Kingma e Welling {cite}`kingma2019introduction`
segnala un lavoro precedente che aveva usato la stessa riscrittura per
apprendere i parametri di una famiglia esponenziale invece che il latente):

$$
\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x})
+ \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),
$$

dove $\boldsymbol{\mu}_\phi$ e $\boldsymbol{\sigma}_\phi$ sono le uscite
dell’encoder, $\boldsymbol{\epsilon}$ è la sorgente di rumore e $\odot$ è il
prodotto componente per componente ($f$, nella disuguaglianza, è la funzione di
cui si prende il valore atteso). Adesso il valore atteso è rispetto a
$p(\boldsymbol{\epsilon})$, che di $\phi$ non
dipende, l’operatore di derivata entra, e un solo campione basta a dare uno
stimatore non distorto del gradiente.

Quella disuguaglianza è scritta per un $f$ che di $\phi$ non dipende, mentre
nell’ELBO l’integrando contiene $-\log q_\phi(\mathbf{z} \mid \mathbf{x})$, che
da $\phi$ dipende eccome. Il conto completo ha allora un addendo in più, e
quell’addendo ha **media nulla**, perché è
$-\mathbb{E}_{q_\phi}[\nabla_\phi \log q_\phi(\mathbf{z} \mid \mathbf{x})]$.
Lasciarlo cadere dà quindi un secondo stimatore, anch’esso non distorto, e con
una proprietà notevole: la sua varianza tende a zero man mano che la posterior
approssimata si avvicina a quella vera. È lo stimatore detto *sticking the
landing* {cite}`roeder2017sticking`.

L’alternativa esiste ed è lo **stimatore a punteggio**,
$\nabla_\phi \mathbb{E}_{q_\phi}[f] = \mathbb{E}_{q_\phi}[f(\mathbf{z})\,
\nabla_\phi \log q_\phi(\mathbf{z} \mid \mathbf{x})]$, che è il gradiente di
policy di REINFORCE incontrato nel capitolo sul deep reinforcement learning.
Anche quello è non distorto, e ha il vantaggio decisivo di funzionare su
variabili discrete, dove la riparametrizzazione non si applica. Paga in
varianza, e quel prezzo si misura.

`````

I due metodi si mettono alla prova su un caso minuscolo in cui la risposta
giusta si conosce in anticipo, così non si tratta di fidarsi. Il gioco è
questo: si sorteggia un numero attorno a un centro (qui il centro vale 2), si
guarda il suo quadrato, e ci si chiede di quanto cambierebbe la media di quel
quadrato se il centro si spostasse. La risposta esatta si sa: la media del
quadrato vale il quadrato del centro più uno (l’uno è quanto balla il
sorteggio), quindi spostando il centro di un pochino la media cambia del doppio
del centro. Col centro a 2, la risposta è **4**. Vediamo quanto ci si
avvicinano i due metodi, e soprattutto con quanta mano ferma.

```python
import torch

torch.manual_seed(0)

# Vogliamo la derivata rispetto a mu di E[z^2] con z ~ N(mu, 1).
# Il valore vero si sa: E[z^2] = mu^2 + 1, quindi la derivata e' 2*mu.
MU, PROVE = 2.0, 200_000

epsilon = torch.randn(PROVE)
z = MU + epsilon                      # gli stessi sorteggi per i due metodi

# 1. si deriva attraverso il sorteggio: z e' mu piu' rumore, quindi d(z^2)/dmu = 2z
riparametrizzato = 2 * z

# 2. non si deriva il sorteggio, si deriva la probabilita' di averlo pescato:
#    per una gaussiana, d(log q)/dmu = z - mu
punteggio = z ** 2 * (z - MU)

for nome, stima in (("riparametrizzazione", riparametrizzato), ("punteggio", punteggio)):
    print(f"{nome:>20}: media {stima.mean():6.3f}   "
          f"deviazione standard {stima.std():7.3f}")
print(f"{'valore vero':>20}: {2 * MU:6.3f}")
print(f"\nla varianza del secondo e' {(punteggio.var() / riparametrizzato.var()):.0f} "
      f"volte quella del primo")
```

```text
 riparametrizzazione: media  3.993   deviazione standard   2.000
           punteggio: media  3.974   deviazione standard   9.306
         valore vero:  4.000

la varianza del secondo e' 22 volte quella del primo
```

Tutti e due i metodi puntano al valore giusto, 4: nessuno dei due imbroglia. La
differenza è la mano, che nel secondo trema molto di più, e a dirlo è la
**deviazione standard**, cioè di quanto una singola risposta si scosta in media
dal valore giusto: 2,0 per il primo metodo, 9,3 per il secondo. Il numero in
fondo eleva al quadrato quelle due, che è il passaggio con cui si arriva alla
**varianza**: 9,3 al quadrato contro 2 al quadrato, cioè ventidue volte tanto.

Ventidue volte di varianza vuol dire, a parità di precisione, ventidue volte i
campioni. In un addestramento che di campioni ne tira uno per esempio, è la
differenza fra un metodo che si usa e uno che non si usa.

## Tutto insieme

I pezzi ci sono tutti. L’encoder, invece di un codice, produce **due** file di
numeri, una media e una larghezza; da lì si pesca con lo scarto deciso prima;
il decoder ricostruisce; e la perdita è la somma delle due voci di spesa.

```python
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits

torch.manual_seed(0)
torch.set_num_threads(1)
X = torch.tensor(load_digits().data / 16.0, dtype=torch.float32)
LATENTE = 8


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.tronco = nn.Sequential(nn.Linear(64, 48), nn.ReLU())
        self.testa = nn.Linear(48, 2 * LATENTE)   # media e log-varianza insieme
        self.decoder = nn.Sequential(nn.Linear(LATENTE, 48), nn.ReLU(),
                                     nn.Linear(48, 64))

    def codifica(self, x):
        return self.testa(self.tronco(x)).chunk(2, dim=1)


vae = VAE()
opt = torch.optim.Adam(vae.parameters(), lr=3e-3)
for passo in range(4000):
    media, log_var = vae.codifica(X)
    # il sorteggio scritto in modo derivabile: rumore fisso, media e scala apprese
    z = media + torch.exp(0.5 * log_var) * torch.randn_like(media)

    ricostruzione = F.binary_cross_entropy_with_logits(
        vae.decoder(z), X, reduction="sum") / len(X)
    costo_descrizione = (-0.5 * (1 + log_var - media ** 2
                                 - log_var.exp()).sum(1)).mean()
    perdita = ricostruzione + costo_descrizione     # cioe' -ELBO

    opt.zero_grad()
    perdita.backward()
    opt.step()

print(f"ricostruzione      {ricostruzione.item():6.1f} nat")
print(f"costo descrizione  {costo_descrizione.item():6.1f} nat")
print(f"ELBO              {-perdita.item():7.1f} nat  "
      f"(log p(x) sta piu' in alto di qui)")
```

```text
ricostruzione        20.2 nat
costo descrizione     3.6 nat
ELBO                -23.8 nat  (log p(x) sta piu' in alto di qui)
```

La prima cosa da notare è che la ricostruzione è **peggiorata**: 20,2 nat
contro i 16,3 della clessidra semplice, sulle stesse cifre e con la stessa
architettura, a parte la testa dell’encoder che qui deve produrre anche la
larghezza. È il prezzo: quei quasi quattro nat sono la vaghezza che abbiamo
comprato.

La macchina che abbiamo appena montato ha un nome, ed è quello del capitolo:
**autoencoder variazionale**, in sigla **VAE**. «Variazionale» è la parola
dell’apertura: alla risposta esatta si è rinunciato, e si è cercata la migliore
dentro una famiglia di risposte semplici, che qui sono le zone gaussiane che
l’archivista propone. Vale adesso la pena vedere che cosa abbiamo preso in
cambio dei quattro nat.

Il vocabolario comune, nel gergo di questa materia e nel codice, si chiama
**prior**, che in inglese vuol dire «ciò che viene prima»: prima di guardare il
dato, è quello che ci si aspetta dalla scheda. Compare nell’uscita qui sotto.

```python
LIVELLI = " .:-=+*#%"


def affianca(*immagini):
    griglie = [(im.reshape(8, 8) * 8).round().long().clamp(0, 8) for im in immagini]
    return "\n".join("   ".join("".join(LIVELLI[i] for i in g[r]) for g in griglie)
                     for r in range(8))


with torch.no_grad():
    # si pesca dal vocabolario comune, che nel codice si chiama prior
    nuove = torch.sigmoid(vae.decoder(torch.randn(500, LATENTE)))

print("quattro cifre pescate dal prior e decodificate")
print(affianca(*nuove[:4]))
```

```text
quattro cifre pescate dal prior e decodificate
   -#=       -*#+.     .+#**.      *%#=.
  :***.     .##--.     -%#=:      :*=*=
  =*=*.     :%:  .     +%:        .. *-
 .*##*      =%-:-.     +%-         -+#*+
 .*##+.     -#-=*.     :*+=       .*%#+:
 .+=**.     .-.+*.      .+*.       .#:
  --**.      -=#-       :#*        -#
   -#*.      -#=       .*%:        #-
```

Una precisazione prima di guardarle, perché cambia come si leggono: quello che
il blocco stampa è il **grigio medio** che il copista dichiara per ciascun
pixel, non un sorteggio. Sorteggiando davvero uscirebbe sale e pepe, e
una parte della morbidezza che si vede è quindi una scelta di come disegnare,
non solo del modello.

Detto questo, non sono capolavori, e non conviene venderle per più di quello
che sono: grosse, un po’ molli, e su qualcuna si esita fra due cifre. Quello
che conta è un’altra cosa: **non è stato dato in pasto niente**. Quei quattro
disegni vengono da quattro file di otto numeri sorteggiate da una gaussiana, e
da
nient’altro, e quella gaussiana era **dichiarata in partenza**. Alla clessidra
della sezione precedente una gaussiana si era dovuta adattare ai codici a cose
fatte, sperando che ci somigliassero: è lì che si era aperto il buco.

Il metro della sezione precedente lo dice senza aggettivi. Rimettiamo in piedi
anche la clessidra semplice, così i due numeri li stampa la stessa macchina.

```python
class Clessidra(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(64, 48), nn.ReLU(),
                                     nn.Linear(48, LATENTE))
        self.decoder = nn.Sequential(nn.Linear(LATENTE, 48), nn.ReLU(),
                                     nn.Linear(48, 64))


torch.manual_seed(0)
ae = Clessidra()
opt = torch.optim.Adam(ae.parameters(), lr=3e-3)
for passo in range(4000):
    perdita_ae = F.binary_cross_entropy_with_logits(
        ae.decoder(ae.encoder(X)), X, reduction="sum") / len(X)
    opt.zero_grad()
    perdita_ae.backward()
    opt.step()


def quanto_e_vuoto(visti, sorteggiati):
    """Di quante spaziature tipiche un codice sorteggiato manca il bersaglio."""
    fra = torch.cdist(visti, visti)
    fra.fill_diagonal_(float("inf"))
    return (torch.cdist(sorteggiati, visti).min(1).values.median()
            / fra.min(1).values.median()).item()


def quanto_somiglia(immagini, veri):
    return torch.cdist(immagini, veri).min(1).values.median().item()


with torch.no_grad():
    codici_ae = ae.encoder(X)
    sorteggiati_ae = codici_ae.mean(0) + codici_ae.std(0) * torch.randn(500, LATENTE)
    media, log_var = vae.codifica(X)
    codici_vae = media + torch.exp(0.5 * log_var) * torch.randn_like(media)
    sorteggiati_vae = torch.randn(500, LATENTE)
    nuove_ae = torch.sigmoid(ae.decoder(sorteggiati_ae))

fra_veri = torch.cdist(X, X)
fra_veri.fill_diagonal_(float("inf"))
print(f"{'':<26}{'buchi nel latente':>18}{'distanza dalle vere':>22}")
print(f"{'cifra vera':<26}{'':>18}{fra_veri.min(1).values.median():>22.2f}")
print(f"{'autoencoder':<26}{quanto_e_vuoto(codici_ae, sorteggiati_ae):>17.1f}x"
      f"{quanto_somiglia(nuove_ae, X):>22.2f}")
print(f"{'autoencoder variazionale':<26}{quanto_e_vuoto(codici_vae, sorteggiati_vae):>17.1f}x"
      f"{quanto_somiglia(nuove, X):>22.2f}")
```

```text
                           buchi nel latente   distanza dalle vere
cifra vera                                                    1.01
autoencoder                             2.2x                  1.62
autoencoder variazionale                1.0x                  1.09
```

La prima colonna è la geometria: un codice sorteggiato dal prior cade, per il
VAE, **praticamente alla distanza tipica** fra i codici che il decoder ha visto
in addestramento. È casa, non terra sconosciuta. Per la clessidra semplice
distava più del doppio.

Prima di leggerla, una precisazione onesta: le due righe non pescano allo
stesso modo. Per il VAE si pesca dal vocabolario comune, che è dichiarato in
partenza; per la clessidra un vocabolario non c’è, e bisogna adattarne uno ai
codici a cose fatte. Non è un favore fatto a nessuno dei due: **è esattamente
la differenza fra le due macchine**, e nasconderla renderebbe il confronto
inutile invece che equo.

La seconda colonna è la conseguenza: una cifra vera dista dalla sua vicina
1,01; una cifra inventata dal VAE dista 1,09, cioè l’otto per cento in più;
una inventata dalla clessidra dista 1,62, cioè più del sessanta per cento in
più.
Un dato inventato dal VAE è quasi indistinguibile, con questo metro, da uno
vero; uno inventato dalla clessidra no.

La stessa differenza, guardata mentre avviene invece che a conti fatti, è
quella di {numref}`fig-cammino-latente`.

```{figure} ../figures/cammino-latente.svg
:name: fig-cammino-latente
:alt: "Due riquadri affiancati, ciascuno un piano dei codici con sedici codici disposti in due gruppi. A sinistra, «senza l’alone», i codici sono punti isolati e i due gruppi sono lontani, con un largo vuoto in mezzo; un segnalino cammina in linea retta da un codice del gruppo di sinistra a uno del gruppo di destra e attraversa quel vuoto. A destra, «con l’alone», i gruppi sono più vicini e ogni codice occupa un alone che si sovrappone a quelli dei vicini del suo gruppo, cosicché il segnalino percorre un cammino altrettanto dritto senza mai uscire davvero allo scoperto. Sotto ogni riquadro un profilo misura quanto il punto in cui si trova il segnalino sia terra già battuta: a sinistra la curva crolla a zero a metà strada, a destra scende fino a 0,89 e non tocca mai il fondo. In basso è stampato il minimo dei due cammini: 0,00 a sinistra, 0,89 a destra."
:width: 100%

Lo stesso cammino, nei due archivi. La curva sotto ciascun riquadro misura
quanto il punto in cui il segnalino si trova sia terra già battuta: a sinistra
crolla a zero a metà strada, a destra scende appena e non tocca mai il fondo, e
i due minimi (0,00 e 0,89) sono stampati in basso a sinistra. Fra i pannelli
cambiano due cose, e sono quelle che cambiano davvero: a destra i due gruppi
stanno più vicini, perché il costo di descrizione li tira verso il centro, e
ciascun codice occupa un alone invece che un punto. (Sono sedici
codici su un piano perché così si guardano, e la soglia che decide che cosa sia
«battuto» è scelta a mano: è un’illustrazione del meccanismo, non una misura.
Nell’esperimento qui sopra i codici sono milleottocento in otto dimensioni, e a
dirlo resta solo la tabella.)
```

## Che cosa il VAE non fa bene

Sarebbe scorretto chiudere qui. Questa famiglia ha tre difetti noti, tutti e
tre strutturali, e conoscerli serve a capire perché il libro, per generare
immagini, alla fine racconta altro.

`````{tab} Elementare

**Le immagini vengono morbide.** E non si risolve allenando di più. La pagella
con cui il copista è giudicato lo punisce moltissimo se dichiara
quasi impossibile un quadro che invece esiste, e quasi per niente se dichiara
possibile un quadro che non esisterebbe mai. Le due pene non sono pari, e
allora conviene **abbondare**: dichiarare possibile più di quel che serve, e in
dubbio coprire. Un archivio che copre più di quello che c’è produce quadri che
somigliano un po’ a tutto e precisamente a niente, ed è quello che sullo
schermo si legge come sfocatura.

**L’archivista può decidere di non scrivere niente.** Se il copista se la cava
già bene da solo, o se all’inizio dell’addestramento la ricostruzione conta
poco, la strada più conveniente è la scheda vuota: costo di descrizione zero, e
il copista dipinge il quadro medio. Da quello stato è difficile uscire, perché
qualunque ritocco alla scheda costa subito e rende solo dopo. Si rimedia con
mestiere (far pesare poco la seconda voce all’inizio, oppure garantire un
minimo di informazione per riga della scheda), ma è una toppa, non una
soluzione.

**L’archivio, guardato tutto insieme, non è proprio quello promesso.** La
regola tiene ogni singola scheda vicina al vocabolario comune, una alla volta.
Che poi *l’insieme* di tutte le schede assomigli al vocabolario comune non è
garantito da nessuno, e infatti non succede del tutto: restano zone che il
vocabolario dichiara plausibili e in cui il copista è stato poco. Sono le
schede da cui escono i disegni deboli.

`````

`````{tab} Superiore

**Sfocatura.** Massimizzare l’ELBO su tutto l’insieme di dati equivale a
minimizzare $D_{\mathrm{KL}}(q_{\mathcal{D},\phi}(\mathbf{x}, \mathbf{z})
\,\|\, p_\theta(\mathbf{x}, \mathbf{z}))$, dove
$q_{\mathcal{D},\phi}(\mathbf{x}, \mathbf{z}) = p_{\text{dati}}(\mathbf{x})\,
q_\phi(\mathbf{z} \mid \mathbf{x})$ è la congiunta che si ottiene pescando un
dato vero e poi codificandolo: è una KL in cui la distribuzione dei dati sta a
**sinistra**. In quella direzione il costo di mettere probabilità
quasi nulla dove i dati ci sono diverge, mentre il costo di metterne dove i
dati non ci sono è mite: il modello ottimale è quindi più disperso dei dati, e
su immagini «più disperso» si legge come sfocato. È la spiegazione che danno
Kingma e Welling {cite}`kingma2019introduction`, i quali osservano anche che il
rimedio non è cambiare obiettivo ma rendere più flessibili la posterior o il
decoder.

**Collasso della posterior.** All’inizio dell’addestramento il termine di
ricostruzione è debole, e $q_\phi(\mathbf{z} \mid \mathbf{x}) \approx
p(\mathbf{z})$ è un equilibrio stabile da cui è difficile uscire: il costo di
descrizione va a zero e il latente smette di portare informazione. Il fenomeno
è documentato su testo da Bowman e colleghi {cite}`bowman2016generating`, che
propongono di far salire lentamente il peso del termine KL; l’alternativa dei
*free bits* {cite}`kingma2016improved` impone invece un minimo di nat per
gruppo di componenti latenti. Il caso peggiore è un decoder molto espressivo
(autoregressivo, per dire), che può modellare i dati da solo e rende il latente
superfluo.

**Scarto fra prior e posterior aggregata.** Il termine KL agisce **su un
esempio alla volta**, quindi vincola ciascuna $q_\phi(\mathbf{z} \mid
\mathbf{x})$ e non l’aggregato $q_\phi(\mathbf{z}) =
\mathbb{E}_{p_{\text{dati}}}[q_\phi(\mathbf{z} \mid \mathbf{x})]$. I due non
coincidono {cite}`hoffman2016elbo,rosca2018distribution`, e nello scarto
restano regioni con massa apprezzabile sotto il prior che il decoder ha
visitato poco: campionarle dà i risultati deboli. È un limite del metodo, non
un difetto di implementazione, e la sezione su Stable Diffusion, nel capitolo
sui modelli di diffusione, mostra come lo si aggiri in pratica invece di
risolverlo.

Una semplificazione va dichiarata. La verosimiglianza usata è una Bernoulli per
pixel applicata a livelli di grigio continui, che è la ricetta consueta su
questi dati e **non è una densità normalizzata** su $[0,1]$: il numero stampato
come ELBO è quindi un ELBO rispetto a quel modello, non rispetto a una densità
propria. La correzione esiste, si chiama Bernoulli continua
{cite}`loaizaganem2019continuous`: i suoi autori misurano che applicarla cambia
i punteggi **e** rende i campioni più nitidi, cioè tocca proprio la sfocatura.
Il confronto fra clessidra e VAE regge lo stesso, perché i due sono addestrati
con la medesima verosimiglianza; il valore assoluto dei nat, no.

`````

Tre difetti veri, quindi, e nessuno dei tre si corregge allenando di più: sono
conseguenze della pagella, non della fatica. Sono il motivo per cui il libro,
per generare immagini, dedica altri capitoli ad altre due famiglie. E sono
anche il motivo per cui questa macchina, dentro quelle due famiglie, continua a
lavorare: le si affida un mestiere diverso, in cui i tre difetti non mordono.
Qual è, lo dice la sezione seguente.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Si cambia domanda: non «la copia somiglia all’originale?» ma «quanto era
  probabile che uscisse proprio questo dato?». La regola che mancava non si
  aggiunge: **esce da sola** provando a rispondere.
- Quel conto non si può fare, e **non basta tirare a sorte**: quasi tutte le
  cause sorteggiate a caso spiegano il dato malissimo, e con quaranta numeri
  nascosti uno solo su centomila sorteggi si prende un terzo del totale.
- Si chiede allora all’archivista, che il dato ce l’ha sotto gli occhi, dove
  conviene guardare. Ne esce una **stima prudente**, sicuramente più bassa del
  vero, e il divario è esattamente quanto il consiglio è impreciso. Spingerla
  in alto migliora, quasi sempre, il modello e il consiglio insieme.
- La stima ha due voci: quanto male si ricostruisce, e **quanto costa scrivere
  la scheda** rispetto a un vocabolario comune deciso prima. La seconda voce è
  ciò che riempie i buchi.
- Per addestrare serve **decidere gli errori prima**: si sorteggia uno scarto,
  e lo si appoggia sulla zona proposta. Così le correzioni tornano indietro.
  L’altro modo (tenere conto di quanto era probabile pescare proprio quel
  punto) funziona e non imbroglia, ma ha la mano molto meno ferma: in questo
  esempio, ventidue volte.
- Il risultato: le cifre pescate dal nulla sono grosse e molli, e su qualcuna
  si esita fra due cifre, ma vengono da niente; e un codice sorteggiato cade
  dove il copista è già stato. Il prezzo è una
  ricostruzione un po’ peggiore, e restano tre difetti veri: le immagini
  vengono morbide, l’archivista può decidere di non scrivere niente, e
  l’archivio nell’insieme non è proprio quello promesso.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Identità esatta:
  $\log p_\theta(\mathbf{x}) = \mathcal{E}_{\theta,\phi}(\mathbf{x}) +
  D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\,
  p_\theta(\mathbf{z} \mid \mathbf{x}))$. L’ELBO è un limite inferiore e il
  divario **è** l’errore della posterior approssimata: massimizzarlo migliora
  modello ed encoder insieme.
- Forma operativa:
  $\mathcal{E} = \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]
  - D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z}))$.
  Con prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$ e posterior gaussiana
  diagonale il secondo termine è in forma chiusa e si annulla se e solo se
  l’encoder restituisce il prior.
- La stima Monte Carlo dal prior è non distorta e inservibile: la sua versione
  logaritmica è distorta verso il basso per Jensen, e in 40 dimensioni sbaglia
  di 10 nat con $10^5$ campioni.
- **Riparametrizzazione** {cite}`kingma2014auto,rezende2014stochastic`:
  $\mathbf{z} = \boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot
  \boldsymbol{\epsilon}$ con $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},
  \mathbf{I})$. Sposta il caso fuori dal grafo delle derivate; misurato, ha
  varianza 22 volte minore dello stimatore a punteggio (REINFORCE), che però è
  l’unico applicabile a latenti discreti.
- Limiti strutturali: **sfocatura** (direzione della KL, quindi copertura),
  **collasso della posterior** {cite}`bowman2016generating,kingma2016improved`
  e **scarto fra prior e posterior aggregata**
  {cite}`hoffman2016elbo,rosca2018distribution`.
```

`````

Quello che abbiamo in mano, alla fine di questa sezione, è una macchina che
comprime e che sa anche pescare. La sezione seguente smette di guardarla da
dentro e la guarda da fuori: che cosa si può chiedere a quel latente una volta
che c’è, che cosa succede se il vocabolario comune si fa di simboli invece che
di numeri, e in quanti posti del libro questa macchina stesse già lavorando
senza essere stata presentata.
