# Imparare guardando: imitazione e clonazione comportamentale

Nel 1989, alla Carnegie Mellon, un furgone attrezzato percorreva le strade
attorno al campus guidato da una rete neurale con un solo strato nascosto. Si
chiamava **ALVINN**, riceveva in ingresso l'immagine di una telecamera e la
lettura di un telemetro laser, e restituiva l'angolo di sterzata
{cite}`pomerleau1989alvinn`. Non aveva imparato per tentativi ed errori, il che
sarebbe stato imprudente su una strada vera: era stato addestrato a
**riprodurre, per ogni immagine di strada, l'angolo di sterzata corretto**.
Apprendimento supervisionato, non prove ed errori.

C'è un dettaglio di quella storia che sembra un'inezia ed è invece la tesi di
questa sezione, arrivata con trent'anni di anticipo. Le immagini su cui ALVINN
imparò, nel lavoro del 1989, erano **simulate**. Quando Pomerleau passò alle
registrazioni di un guidatore vero, due anni dopo, si trovò davanti a un
ostacolo che dovette aggirare a mano: un guidatore bravo non esce mai dalla
corsia, quindi nelle sue registrazioni non c'è un solo fotogramma che mostri
come si rimedia a un'auto storta. Dovette fabbricarseli, deformando le immagini
per ricavarne viste spostate rispetto al centro della corsia
{cite}`pomerleau1991efficient`. Perché fosse necessario è esattamente ciò che
segue.

L'imitazione è, in un certo senso, l'idea più ovvia di tutte, e per questo vale
la pena capire bene perché non basta. Nei capitoli precedenti l'agente impara
da una ricompensa, e la ricompensa è la parte difficile: scriverla per una
guida sicura o per una risposta utile è un problema aperto. Se però qualcuno sa
già fare il compito, quel problema si può aggirare. Non gli si chiede di
scrivere una funzione di ricompensa: gli si chiede di **fare vedere**.

## Clonare un comportamento è apprendimento supervisionato

`````{tab} Elementare

Si registra un esperto mentre lavora e si annota, momento per momento, che cosa
vedeva e che cosa ha fatto. Ne esce un elenco di coppie: «in questa situazione,
questa mossa». A quel punto il problema di reinforcement learning è sparito, e
al suo posto c'è un normalissimo problema di apprendimento supervisionato:
l'ingresso è la situazione, l'uscita da indovinare è la mossa dell'esperto.

Si chiama **clonazione comportamentale**, ed è tanto semplice che il libro l'ha
già usata due volte senza chiamarla così. Il primo AlphaGo, prima di giocare
contro sé stesso, aveva imparato a proporre mosse guardando
centosessantamila partite di giocatori forti. E la prima fase dell'addestramento di un assistente
conversazionale è esattamente questa: si raccolgono risposte scritte da persone
e si insegna al modello a scriverne di simili.

I pregi sono seri e vanno detti. È **stabile**, perché è addestramento
supervisionato ordinario, senza nessuna delle instabilità dei metodi a valore.
È **efficiente**, perché ogni dimostrazione insegna qualcosa subito, mentre un
agente che esplora butta via migliaia di tentativi. Ed è **sicura**, perché non
serve far provare al sistema mosse a caso nel mondo vero.

`````

`````{tab} Superiore

Dato un insieme di dimostrazioni $\mathcal{D} = \{(s_i, a_i)\}$ prodotte da una
politica esperta $\pi^\star$, la **clonazione comportamentale** stima

$$
\hat\pi = \arg\min_{\pi} \; \mathbb{E}_{(s,a)\sim\mathcal{D}}
\big[\, \ell\big(\pi(s),\, a\big) \,\big],
$$

con $\ell$ la cross-entropia per azioni discrete o l'errore quadratico per
azioni continue. Non compaiono ricompense, non compare l'equazione di Bellman,
non compare il bootstrapping: è **regressione o classificazione**, con tutto
ciò che ne consegue in termini di stabilità e di strumenti già noti.

Il libro l'ha già incontrata in tre punti, e conviene riconoscerla. Nel
capitolo sui metodi a gradiente di policy, la rete di policy di AlphaGo è
pre-addestrata in modo supervisionato su partite umane prima del *self-play*.
Nel post-addestramento dei modelli linguistici, la fase di *supervised
fine-tuning* che precede l'RLHF è clonazione comportamentale su dimostrazioni
scritte da persone. E nella prossima sezione, la componente supervisionata del
Decision Transformer è la stessa cosa, condizionata sul ritorno desiderato.

L'assunzione nascosta, che è tutto il problema, è quella di ogni apprendimento
supervisionato: **dati indipendenti e identicamente distribuiti**. Qui non lo
sono, perché gli stati che l'agente incontrerà dipendono dalle azioni che
avrà scelto. La distribuzione degli stati non è fissata dal mondo, è **indotta
dalla politica**, e quella dell'allievo non è quella del maestro.

`````

## Il problema: gli errori si compongono

Qui sta la difficoltà, ed è una di quelle che si capiscono meglio con un
esempio che con una formula.

`````{tab} Elementare

Immagina di imparare a guidare guardando un pilota bravissimo. Bravissimo vuol
dire che non esce mai dalla corsia. Quindi in tutte le ore di registrazione che
hai, l'auto è **sempre al centro della strada**: non c'è un solo fotogramma in
cui sia storta e vada raddrizzata, perché a lui non è mai successo.

Poi guidi tu. Al primo dosso ti sposti di venti centimetri, ed è pochissimo. Ma
quella situazione, «auto leggermente storta», tu non l'hai mai vista, e non hai
idea di cosa si faccia. Fai qualcosa di plausibile e sbagliato, e ti sposti di
quaranta. Adesso sei ancora più lontano da tutto ciò che conosci, e sbagli di
più. Dopo dieci secondi sei nel fosso.

Il punto che vale la pena assorbire è **quanto sia controintuitivo**: il tuo
errore per singola decisione era minuscolo, e misurato sulle registrazioni del
pilota risulterebbe quasi zero. Il guaio non è la grandezza dell'errore, è che
ogni errore ti porta in un posto dove sei più ignorante, e quindi sbagli di
più. **L'errore non si somma, si compone.**

Ed ecco il paradosso: più l'esperto è bravo, peggio è. Un maestro perfetto non
sbaglia mai, quindi non si trova mai nella condizione di dover rimediare,
quindi non ti insegna mai a rimediare. È la sola cosa che ti servirà davvero.

Il rimedio, una volta capito il problema, si scrive da sé: non basta far
vedere. Bisogna **lasciar provare l'allievo, guardare dove finisce, e chiedere
al maestro cosa avrebbe fatto lì**. Le situazioni che contano sono quelle in
cui va a cacciarsi l'allievo, non quelle in cui passava il maestro.

Quel rimedio ha un nome, **DAgger**, e lo si incontrerà da qui in poi con quello.
Viene da *Dataset Aggregation*, cioè «accumulare dati», perché a ogni giro
l'archivio si allarga con le situazioni nuove in cui l'allievo è andato a
finire, etichettate dal maestro. Ha un costo, ed è chiaro anche detto così: il
maestro deve essere lì, disponibile a rispondere, e non basta più una scatola di
vecchie registrazioni.

`````

`````{tab} Superiore

Formalmente, la clonazione minimizza l'errore sotto la distribuzione di stati
$d_{\pi^\star}$ indotta dall'esperto, mentre al momento dell'uso l'agente vive
sotto $d_{\hat\pi}$, indotta da sé stesso. È uno **spostamento di
distribuzione**, con la particolarità di essere **causato dal modello stesso**:
non è il mondo che cambia, è la politica che si porta in una regione che non
conosce, e più sbaglia più ci si porta.

Il risultato di riferimento è di Ross, Gordon e Bagnell
{cite}`ross2011reduction`. Se la politica appresa ha un tasso d'errore $\epsilon$
sotto la distribuzione dell'esperto, su un orizzonte $T$ il costo aggiuntivo
della clonazione comportamentale cresce in generale come $O(\epsilon T^2)$: il
fattore $T$ in più rispetto all'ideale $O(\epsilon T)$ è esattamente la
composizione degli errori. Su orizzonti lunghi la differenza fra $T$ e $T^2$ è
tutta la differenza fra un sistema che funziona e uno che no.

**DAgger** (*Dataset Aggregation*) rimuove il termine in più con un'idea
semplice: iterare. Si addestra una politica sulle dimostrazioni, la si
**esegue** per raccogliere gli stati che *lei* visita, si chiede all'esperto
l'azione corretta **su quegli stati**, si aggiunge tutto al dataset e si
riaddestra. Ripetendo, la distribuzione di addestramento converge a quella
d'uso, e la garanzia torna lineare in $T$ **a una condizione**, che è nascosta
in una costante e va tirata fuori. Il bound è

$$
J(\hat\pi) \;\le\; J(\pi^\star) + u\,T\,\epsilon_N + O(1),
$$

dove $u$ misura di quanto un singolo errore può peggiorare il costo-per-andare
dell'esperto. Nei compiti **recuperabili** $u$ è $O(1)$ e la garanzia è
effettivamente lineare; ma se un errore porta in uno stato da cui non si torna
(il fosso della metafora di due paragrafi fa) $u$ può crescere come $T$, e si
torna al quadrato. È un'ipotesi che vale la pena tenere presente perché
l'esperimento di questa sezione la **viola**: il sistema del codice è instabile,
cioè per costruzione non recuperabile senza correzione. DAgger raccoglie gli
stati giusti; non promette che da quegli stati si possa tornare.

Il prezzo, poi, è che serve un esperto **interrogabile durante
l'addestramento**, non solo un archivio di registrazioni: e nella maggior parte
dei casi pratici quell'esperto è una persona, il che sposta il costo dai dati al
tempo umano.

Vale la pena distinguere la clonazione da un parente che risolve lo stesso
problema in un altro modo. Nell'**apprendimento per rinforzo inverso** non si
impara la politica, si impara la **ricompensa** che rende ottimo il
comportamento osservato, e poi la si ottimizza con i metodi dei capitoli
precedenti. È più costoso, ed è più robusto per una ragione precisa: una
ricompensa è una descrizione **compatta e trasferibile** dell'obiettivo, mentre
una politica è una tabella di reazioni valida solo dove è stata vista. Se
l'ambiente cambia un po', la ricompensa regge e la politica no. È lo stesso
motivo per cui l'RLHF non si ferma alla fase supervisionata: il modello di
ricompensa addestrato sulle preferenze è, di fatto, una ricompensa inferita da
comportamento umano.

Il difetto strutturale, però, gli sta accanto fin dal primo lavoro che definisce
il problema {cite}`ng2000algorithms`, e va detto: l'RL inverso, nella sua forma
nuda, è **mal posto**. Infinite funzioni di ricompensa rendono ottimo lo stesso
comportamento osservato, a cominciare da quella identicamente nulla, sotto la
quale ogni politica è ottima. In generale la ricompensa si recupera solo a meno
di una scala e di un termine di *shaping potential-based*. È un oggetto che il
capitolo rincontrerà: nella sezione sull'esplorazione si dimostra che aggiungere
alla ricompensa un termine della forma $\gamma\Phi(s')-\Phi(s)$ lascia
invariata la policy ottima, *per qualunque* $\Phi$. Là quella proprietà è una
garanzia (si possono dare aiuti senza spostare l'obiettivo); qui, letta al
rovescio, è esattamente l'ambiguità che l'RL inverso non può sciogliere. Stessa
proprietà, due lati.

`````

## In pratica: basta uscire dalla fascia dimostrata

L'affermazione centrale di questa sezione si può toccare con mano in mezza
pagina. Prendiamo un **sistema instabile**, cioè uno che lasciato a sé peggiora
da solo invece di rimettersi a posto: è l'auto storta del racconto qui sopra,
scritta in numeri. Lo «stato» è un numero solo, e dice di quanto siamo fuori
posto: zero vuol dire dritti in mezzo alla corsia, e più il numero cresce (in
positivo o in negativo) più siamo storti. Instabile significa che senza
correzione quel numero, a ogni passo, si moltiplica per $1{,}25$: da solo non
torna a zero, esplode.

L'esperto è un controllore che sa cosa fare ovunque, anche lontanissimo da
zero, e siccome è bravo dallo zero non si allontana mai. Poi, a metà episodio,
diamo una folata di vento che l'esperto nelle sue registrazioni non ha mai
preso.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

INSTABILE, RUMORE, ORIZZONTE, RAFFICA = 1.25, 0.02, 40, 4.0

def esperto(s):
    """Un controllore vero: sa cosa fare ovunque, anche lontano da zero."""
    return -(INSTABILE - 0.85) * s

def episodio(politica, s0, g, raffica=False):
    s, stati, azioni = s0.clone(), [], []
    for t in range(ORIZZONTE):
        if raffica and t == 15:
            s = s + RAFFICA                 # una folata che l'esperto non ha mai preso
        with torch.no_grad():
            a = politica(s)
        stati.append(s.clone()); azioni.append(esperto(s))   # l'esperto etichetta
        s = INSTABILE * s + a + RUMORE * torch.randn(s.shape, generator=g)
    return torch.cat(stati), torch.cat(azioni), s

def rete():
    return nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64),
                         nn.Tanh(), nn.Linear(64, 1))

def allena(politica, S, A, passi=2000):
    ott = torch.optim.Adam(politica.parameters(), lr=3e-3)
    for _ in range(passi):
        ott.zero_grad(); F.mse_loss(politica(S), A).backward(); ott.step()

torch.manual_seed(0)
g = torch.Generator().manual_seed(1)
avvio = lambda n: torch.randn(n, 1, generator=g) * 0.3

# le dimostrazioni: l'esperto è bravo, quindi non si allontana mai da zero
S_dim, A_dim, _ = episodio(esperto, avvio(64), g)
print(f"stati dimostrati: fra {S_dim.min():+.2f} e {S_dim.max():+.2f}")

politica = rete()
allena(politica, S_dim, A_dim)
with torch.no_grad():
    print(f"errore per singolo passo sugli stati dimostrati: "
          f"{F.mse_loss(politica(S_dim), A_dim).item():.6f}  (praticamente perfetto)")
    fuori = torch.tensor([[3.0]])
    print(f"ma a s=3,0 l'esperto direbbe {esperto(fuori).item():+.3f} "
          f"e la clonazione dice {politica(fuori).item():+.3f}")

_, _, fe = episodio(esperto, avvio(64), g, raffica=True)
_, _, fc = episodio(politica, avvio(64), g, raffica=True)
print(f"\ndopo la folata: |stato| finale esperto {fe.abs().mean():.3f}, "
      f"clonazione {fc.abs().mean():.1f}")

# DAgger: si aggiungono gli stati in cui è finita LA POLITICA, etichettati
# dall'esperto. Sono proprio quelli che le dimostrazioni non contenevano.
S_tot, A_tot = S_dim, A_dim
for giro in (1, 2, 3):
    S_v, A_v, _ = episodio(politica, avvio(64), g, raffica=True)
    S_tot, A_tot = torch.cat([S_tot, S_v]), torch.cat([A_tot, A_v])
    allena(politica, S_tot, A_tot, passi=1500)
    _, _, f = episodio(politica, avvio(64), g, raffica=True)
    print(f"dopo il giro {giro} di DAgger: |stato| finale {f.abs().mean():.3f}")

# Controllo, e vale quanto l'esperimento: quel 74,6 è UN seme, e la folata
# gliela diamo noi. Che cosa succede su otto semi, e senza la folata?
def prova(seme):
    torch.manual_seed(seme)
    g = torch.Generator().manual_seed(seme + 1)
    avvio = lambda n: torch.randn(n, 1, generator=g) * 0.3
    S, A, _ = episodio(esperto, avvio(64), g)
    pol = rete(); allena(pol, S, A)
    _, _, e_senza = episodio(esperto, avvio(64), g)          # esperto, nessuna folata
    _, _, c_senza = episodio(pol, avvio(64), g)              # clonazione, nessuna folata
    _, _, c_con = episodio(pol, avvio(64), g, raffica=True)  # clonazione, con folata
    return (e_senza.abs().mean().item(), c_senza.abs().mean().item(),
            c_con.abs().mean().item())

esiti = [prova(s) for s in range(8)]
for nome, colonna in (("esperto, senza folata   ", 0),
                      ("clonazione, senza folata", 1),
                      ("clonazione, con folata  ", 2)):
    v = sorted(e[colonna] for e in esiti)
    print(f"{nome}: mediana {(v[3] + v[4]) / 2:.4g}, da {v[0]:.4g} a {v[-1]:.4g}")
```

I numeri raccontano la storia meglio di qualunque spiegazione.

L'esperto vive in una fascia strettissima, fra $-1{,}00$ e $+0{,}71$. Su quella
fascia la clonazione impara **perfettamente**: l'errore per singolo passo è
$0{,}000000$, e qualunque valutazione fatta sui dati di addestramento direbbe
che il modello è impeccabile.

A $s = 3{,}0$, però, dove non è mai stata, l'esperto correggerebbe di
$-1{,}200$ e la clonazione propone $-0{,}824$. Non è un errore assurdo, è una
correzione **troppo debole di un terzo**, ed è tutto ciò che serve: su un
sistema instabile una correzione insufficiente lascia crescere lo stato, che al
passo dopo è ancora più lontano da ciò che si conosce, dove la correzione è
ancora peggiore.

Il risultato: dopo la folata l'esperto torna a $0{,}071$ e **la clonazione
finisce a $74{,}6$**. Tre ordini di grandezza, partendo da un errore per passo
pari a zero.

Poi i tre giri di DAgger: $0{,}102$, $0{,}077$, $0{,}082$. Riportare
l'esperto a etichettare gli stati in cui era finito **l'allievo** basta a
tornare al livello del maestro. E si noti cosa è cambiato: non il modello, non
il modo di misurare l'errore, non il procedimento con cui lo si riduce. Sono
cambiati **quali situazioni stanno nel mucchio degli esempi**.

### Che cosa questo esperimento dimostra, e che cosa no

Le ultime righe del codice servono a non prendere lucciole per lanterne, e sono
la parte più importante da leggere.

Primo, quel $74{,}6$ è **un seme**. Ripetendo l'esperimento su otto semi
indipendenti, lo stato finale della clonazione dopo la folata ha mediana
$322$ e va da $77$ a $472$: il numero del racconto è l'estremo basso di una
distribuzione la cui mediana è più di quattro volte più grande. La conclusione
qualitativa non cambia di una virgola (la clonazione finisce fuori strada in
tutti e otto i casi, di due o tre ordini di grandezza), ma la cifra precisa non
è una proprietà dell'algoritmo: è una proprietà di quella ripetizione.

Secondo, e conta di più: **senza la folata non succede niente**. Sugli stessi
otto semi, lasciata a sé, la clonazione chiude con mediana $0{,}0285$ (fra
$0{,}025$ e $0{,}031$) e l'esperto con mediana $0{,}0318$: non solo sono dello
stesso ordine, ma su questi otto semi l'allievo sta perfino un filo meglio del
maestro, che è quanto dire che la differenza è rumore. Il fosso arriva solo
quando qualcosa porta l'allievo fuori dalla fascia dimostrata, e in questo
esperimento a portarcelo è una perturbazione **esterna**, che gli diamo noi, e
che in un colpo solo lo scaraventa a quattro volte il bordo della fascia.

Nel racconto della guida, invece, fuori dalla fascia l'allievo ci arriva **da
solo**, un errore alla volta, ed è proprio quell'accumulo il $T^2$ di Ross,
Gordon e Bagnell. Qui la spinta iniziale è un espediente: il modo più rapido di
mettere l'allievo dove non è mai stato, per mostrare cosa succede una volta che
ci si trova. Quello che l'esperimento dimostra, e lo dimostra bene, è la seconda
metà del ragionamento: **fuori dalla fascia dimostrata una politica clonata
sbaglia in modo sistematico, e su un sistema instabile sbagliare in modo
sistematico è irrecuperabile.** La prima metà, cioè che a portarla fuori bastino
i suoi stessi errori, resta un risultato teorico, e questo codice non la prova.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La **clonazione comportamentale** è la scorciatoia più ovvia: si registra
  qualcuno che il lavoro lo sa fare, si annota «in questa situazione, questa
  mossa», e da lì in poi il problema non è più imparare per tentativi, è
  indovinare la risposta giusta. Stabile, sicura, parca di dati, e già usata due
  volte nel libro senza chiamarla per nome.
- La crepa sta in un'assunzione che nessuno dichiara: che le situazioni siano
  sempre le stesse, decise dal mondo. Non è così, perché **le situazioni che
  incontri dipendono dalle mosse che hai fatto**, e le mosse dell'allievo non
  sono quelle del maestro.
- Da qui la **composizione degli errori**: un errore piccolo ti porta in un
  posto che conosci meno, dove sbagli di più, che ti porta ancora più lontano.
  L'errore non si somma, si compone, e su un percorso lungo è la differenza fra
  un sistema che funziona e uno che no.
- Il paradosso da ricordare: **più l'esperto è bravo, meno insegna a
  rimediare**, perché non si trova mai nella condizione di doverlo fare. Il
  pilota che non esce mai di corsia non ti fa mai vedere come si raddrizza
  l'auto.
- Il rimedio si chiama **DAgger**: far provare l'allievo, guardare dove va a
  finire, e chiedere al maestro cosa avrebbe fatto *lì*. Non cambia il modello e
  non cambia il modo di allenarlo: cambia quali situazioni finiscono nel mucchio
  degli esempi. Il prezzo è che serve un maestro disponibile a rispondere, non
  soltanto un archivio di registrazioni.
- C'è un'altra strada: invece della strategia, imparare la **ricompensa**, cioè
  che cosa l'esperto stesse cercando di ottenere. Costa di più e regge meglio se
  il mondo cambia un po', perché descrive l'obiettivo e non solo le reazioni. Ha
  un difetto suo, però: di obiettivi che spiegano lo stesso comportamento ce n'è
  un'infinità, e distinguerli guardando solo il comportamento non si può.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- La **clonazione comportamentale** trasforma il controllo in apprendimento
  supervisionato: coppie (situazione, azione dell'esperto), e si minimizza
  l'errore. Stabile, efficiente nei dati, sicura, e già usata due volte nel
  libro senza il suo nome (la policy iniziale di AlphaGo, la fase supervisionata
  che precede l'RLHF).
- L'assunzione che salta è quella di dati **i.i.d.**: la distribuzione degli
  stati non è fissata dal mondo, è **indotta dalla politica**, e quella
  dell'allievo non è quella del maestro.
- Da qui la **composizione degli errori**: un errore piccolo porta in uno stato
  poco familiare, dove l'errore è più grande, e così via. Il costo cresce come
  $O(\epsilon T^2)$ invece di $O(\epsilon T)$.
- Il paradosso da ricordare: **più l'esperto è bravo, meno insegna a
  rimediare**, perché non si trova mai nella condizione di doverlo fare.
- **DAgger** rimuove il termine in più iterando: esegui la politica, chiedi
  all'esperto cosa fare **negli stati che lei visita**, aggiungi, riaddestra.
  Serve un esperto interrogabile, non solo un archivio; e la garanzia lineare
  vale a patto che il compito sia **recuperabile**, altrimenti la costante $u$
  del bound cresce con $T$ e si torna al quadrato.
- L'**RL inverso** risolve lo stesso problema per un'altra strada: impara la
  **ricompensa** invece della politica. Costa di più ed è più trasferibile,
  perché una ricompensa descrive l'obiettivo mentre una politica descrive solo
  delle reazioni. È la ragione per cui l'RLHF non si ferma alla fase
  supervisionata. È però **mal posto**: la ricompensa si recupera a meno di una
  scala e di un termine di shaping potential-based.
```
`````
