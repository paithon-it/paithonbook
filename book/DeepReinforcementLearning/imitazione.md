# Imparare guardando: imitazione e clonazione comportamentale

Nel 1989, alla Carnegie Mellon, un furgone attrezzato percorreva le strade
attorno al campus guidato da una rete neurale con un solo strato nascosto. Si
chiamava **ALVINN**, prendeva in ingresso l'immagine di una telecamera e
restituiva l'angolo di sterzata {cite}`pomerleau1989alvinn`. Non aveva imparato
per tentativi ed errori, il che sarebbe stato imprudente su una strada vera:
aveva guardato un guidatore umano e aveva imparato a **fare la stessa cosa
nelle stesse situazioni**. Trent'anni prima che l'espressione diventasse di
moda, era già un modello di guida addestrato per imitazione.

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
contro sé stesso, aveva imparato a proporre mosse guardando centomila partite
di giocatori forti. E la prima fase dell'addestramento di un assistente
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
scritte da persone. E nel prossimo capitolo, la componente supervisionata del
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
d'uso, e la garanzia torna lineare in $T$. Il prezzo è che serve un esperto
**interrogabile durante l'addestramento**, non solo un archivio di
registrazioni: e nella maggior parte dei casi pratici quell'esperto è una
persona, il che sposta il costo dai dati al tempo umano.

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

`````

## In pratica: l'errore per passo è zero, e il sistema finisce nel fosso

L'affermazione centrale di questa sezione è quantitativa, e si può verificare
in mezza pagina. Prendiamo un sistema **instabile** (senza controllo lo stato
esplode), un esperto che lo governa perfettamente e quindi non si allontana
mai da zero, e poi diamo una folata di vento che l'esperto non ha mai preso.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
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

g = torch.Generator().manual_seed(1)
avvio = lambda n: torch.randn(n, 1, generator=g) * 0.3

# le dimostrazioni: l'esperto è bravo, quindi non si allontana mai da zero
S_dim, A_dim, _ = episodio(esperto, avvio(64), g)
print(f"stati dimostrati: fra {S_dim.min():+.2f} e {S_dim.max():+.2f}")

politica = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64),
                         nn.Tanh(), nn.Linear(64, 1))

def allena(S, A, passi=2000):
    ott = torch.optim.Adam(politica.parameters(), lr=3e-3)
    for _ in range(passi):
        ott.zero_grad(); F.mse_loss(politica(S), A).backward(); ott.step()

allena(S_dim, A_dim)
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
    allena(S_tot, A_tot, passi=1500)
    _, _, f = episodio(politica, avvio(64), g, raffica=True)
    print(f"dopo il giro {giro} di DAgger: |stato| finale {f.abs().mean():.3f}")
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
la loss, non l'ottimizzatore. Sono cambiati **quali stati stanno nel dataset**.

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
  Serve un esperto interrogabile, non solo un archivio.
- L'**RL inverso** risolve lo stesso problema per un'altra strada: impara la
  **ricompensa** invece della politica. Costa di più ed è più trasferibile,
  perché una ricompensa descrive l'obiettivo mentre una politica descrive solo
  delle reazioni. È la ragione per cui l'RLHF non si ferma alla fase
  supervisionata.
```
