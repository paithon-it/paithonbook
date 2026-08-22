#!/usr/bin/env python3
"""I numeri stampati nel libro sono ancora quelli che il codice produce?

    python3 scripts/verifica-uscite.py              # tutto il libro
    python3 scripts/verifica-uscite.py Efficienza   # un capitolo solo
    python3 scripts/verifica-uscite.py --elenco     # che cosa e' coperto
    python3 scripts/verifica-uscite.py --anche-lenti Transformers   # anche i `pt-lento`

Il libro ha una regola: **se il testo commenta un numero, quel numero lo stampa
la macchina**. In pratica ogni blocco ` ```python ` e' seguito da un blocco
` ```text ` che riporta la sua uscita, e quel blocco e' scritto a mano,
copiandolo da un terminale.

Fino a oggi nessuno lo ricontrollava. `genera-notebook.py --verifica` **esegue**
i notebook e si accorge se il codice si rompe, ma non guarda che cosa stampa:
un ritocco a un blocco, una versione di libreria che cambia una cifra, e i
numeri della pagina si scollano in silenzio da quelli veri. E' il difetto 12
applicato al libro invece che a una frase.

Questo strumento lavora sul **sorgente**, non sui notebook, e la ragione e'
precisa: un notebook nasce solo se il capitolo arriva a tre celle, e i blocchi
marcati `pt-lento` non si eseguono mai. Cioe' la rete del notebook lascia
scoperti proprio i capitoli piccoli e i blocchi lenti, che sono i due posti in
cui il libro ha gia' trovato codice rotto. Qui si controlla tutto quello che il
libro dichiara eseguibile.

I blocchi di un capitolo si eseguono **in fila e nello stesso spazio dei nomi**,
nell'ordine del `_toc.yml`, che e' come li esegue chi legge: una pagina puo'
usare quello che la pagina prima ha definito.

Uscita: 0 se ogni blocco stampa quello che il libro dice, 1 altrimenti.
"""

import argparse
import importlib.util
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile

RADICE = pathlib.Path(__file__).resolve().parents[1]
LIBRO = RADICE / "book"
SEGNALE = "\x00PT\x00"
ATTESA = 900          # secondi per capitolo: qui dentro ci sono addestramenti

# Due macchine sommano in ordine diverso, e i numeri del libro se ne accorgono.
# Le librerie di calcolo portano piu' implementazioni della stessa routine, una
# per insieme di istruzioni vettoriali, e all'avvio scelgono quella adatta alla
# CPU che trovano: registri piu' larghi vogliono dire piu' accumulatori
# parziali, cioe' un albero di somme diverso e arrotondamenti in punti diversi.
# Misurato: su questa macchina (kernel SkylakeX, AVX-512) un residuo dava
# 5,551e-16 e su un runner di GitHub 4,441e-16, e tre accuratezze di Efficienza
# ballavano di mezzo punto, perche' quelle ultime cifre entrano in un
# addestramento che le amplifica. Stesse versioni delle librerie, stesso seme.
#
# Fissarle al minimo comune denominatore (AVX2, che qualunque x86-64 dal 2013
# ha) rimette d'accordo numpy, e su quello si regge il residuo del polyfit di
# `MachineLearning`. NON basta per torch, e la prima stesura di questa riga
# diceva il contrario: `BLAS_INFO=mkl`, cioe' le moltiplicazioni fra matrici di
# torch passano da MKL, che `OPENBLAS_CORETYPE` non tocca. Provate e scartate
# anche `MKL_CBWR` (la riproducibilita' condizionale di MKL),
# `DNNL_MAX_CPU_ISA` e il numero di thread: nessuna sposta i numeri qui.
# Da qui `INSTABILI` piu' sotto. Il racconto per il lettore sta in
# `book/Matematica/analisi-numerica.md`.
#
# Solo su x86-64: `Haswell` su ARM non esiste, e chi verifica da un Mac Apple
# Silicon si troverebbe OpenBLAS a lamentarsi di un nome che non conosce.
KERNEL_FISSI = {
    "OPENBLAS_CORETYPE": "Haswell",
    "ATEN_CPU_CAPABILITY": "avx2",
} if platform.machine() in ("x86_64", "AMD64") else {}

# I capitoli le cui uscite NON sono riproducibili fra macchine diverse, e il
# perche' di ciascuno. Non e' una scusa per non controllarli: qui sono
# controllati sempre, e il cancello di `strumenti/pubblica.py` li esegue prima
# di ogni pubblicazione, sulla macchina che quei numeri li ha prodotti. E'
# `--salta-instabili` (che passa solo la CI pubblica) a lasciarli fuori, perche'
# li' l'hardware ruota.
#
# Misurato sul commit 33cbf33: lo stesso identico codice e' andato rosso 2 volte
# su 6, sempre negli stessi undici punti e sempre con gli stessi valori. I
# runner di GitHub sono di due tipi, ciascuno deterministico, e l'ultima cifra
# di un addestramento in virgola mobile dipende da quale ti tocca.
INSTABILI = {
    "Efficienza":
        "pota e riaddestra una rete, e quantizza: l'accuratezza dopo la"
        " potatura cambia di un punto fra un runner e l'altro",
    "ModelliLatenti":
        "addestra autoencoder e li disegna in ASCII, quindi la differenza"
        " nell'ultima cifra diventa un carattere diverso nel disegno",
    "VerosimiglianzaEsatta":
        "la log-verosimiglianza media si sposta di un millesimo di nat",
}


def avvisa_se_carica() -> None:
    """Dice quanto e' carica la macchina, prima di misurare qualsiasi cosa.

    CLAUDE.md lo raccomanda da un pezzo («`uptime` prima di aprire
    un'indagine») e non basta averlo scritto: la regola si legge come un gesto
    da fare all'inizio, mentre il carico che falsa una misura e' quasi sempre
    quello che si e' lanciato **da soli** un momento prima. E' successo il 18
    agosto 2026: lo stesso notebook ha dato 223 secondi con altri controlli in
    parallelo e 28 con la macchina libera, cioe' un fattore otto, e per un
    momento quel 223 e' stato preso per il costo vero di un blocco.

    Non ferma niente: stampa, perche' un numero misurato sotto carico non e'
    sbagliato, e' solo un numero di cui non ci si puo' fidare.
    """
    try:
        carico = os.getloadavg()[0]
        cpu = os.cpu_count() or 1
    except (OSError, AttributeError):
        return
    if carico > cpu * 0.4:
        print(f"  ATTENZIONE: carico {carico:.1f} su {cpu} CPU. I tempi che "
              f"leggerai qui sotto non valgono; il TIMEOUT nemmeno.\n")


def _genera_notebook():
    """Carica `genera-notebook.py`, che ha gia' il parser dei blocchi e
    l'ordine del toc. Il nome col trattino non e' importabile: si va dal file."""
    percorso = RADICE / "scripts" / "genera-notebook.py"
    spec = importlib.util.spec_from_file_location("gn", percorso)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GN = _genera_notebook()


def riga_di(testo: str, posizione: int) -> int:
    return testo.count("\n", 0, posizione) + 1


def attesa_dopo(testo: str, fine_blocco: int) -> tuple[str, int] | None:
    """Il blocco ` ```text ` che segue subito il codice, se c'e'.

    «Subito» vuol dire con solo righe vuote in mezzo: un blocco di testo che
    arriva dopo un paragrafo non e' l'uscita di quel codice, e' un'altra cosa.
    """
    resto = testo[fine_blocco:]
    m = re.match(r"\n*```text\n(.*?)```", resto, re.S)
    if not m:
        return None
    return m.group(1), fine_blocco + m.start(1)


def blocchi_di(pagina: pathlib.Path, anche_lenti: bool = False):
    """[(riga, codice, atteso_o_None, riga_attesa)] dei blocchi eseguibili.

    Con `anche_lenti` entrano anche i blocchi marcati `pt-lento`, che di norma
    restano fuori da tutto: dal notebook, dalla CI e da qui. Quella marcatura
    e' una decisione (un modello da scaricare, un addestramento lungo), non un
    permesso di non funzionare, e CLAUDE.md dice che quei blocchi vanno
    eseguiti a mano. Questo e' il comando per farlo, invece di un'esortazione.
    """
    testo = pagina.read_text(encoding="utf-8")
    fuori = []
    ammessi = {"cella", "lento"} if anche_lenti else {"cella"}
    for posizione, codice, stato in GN.blocchi(testo):
        if stato not in ammessi:              # «testo» resta comunque fuori
            continue
        # la fine del recinto: da qui in poi si cerca l'uscita attesa
        fine = testo.index("```", testo.index(codice, posizione) + len(codice)) + 3
        att = attesa_dopo(testo, fine)
        # le magie IPython (`%timeit`, `!pip`) sono legittime in un notebook
        # ma non in Python puro: si commentano, come fa gia' la verifica dei
        # notebook, invece di lasciare che facciano rosso l'intero capitolo
        # (la ricerca della fine del blocco usa il testo originale, quindi la
        # trasformazione arriva DOPO)
        codice = "".join(("# " + r) if re.match(r"^\s*[%!]", r) else r
                         for r in codice.splitlines(keepends=True))
        fuori.append((riga_di(testo, posizione), codice,
                      att[0] if att else None,
                      riga_di(testo, att[1]) if att else None))
    return fuori


def preludio_di(capitolo: str) -> str | None:
    """L'impalcatura che il capitolo da' per esistente, se ce n'e'.

    `notebooks/_preludi/<Capitolo>.py` crea i dati e i nomi che il testo
    presuppone (il file audio di Audio, per dire). Non e' contenuto del libro,
    e' il modo in cui il libro dichiara che quel blocco parte da qualcosa: non
    prependerlo qui vorrebbe dire bocciare una pagina per una dipendenza che il
    repository documenta e vuole. Il notebook lo ri-esegue prima di ogni pagina;
    qui una volta sola in testa basta, perche' si esegue tutto in un colpo.
    """
    f = RADICE / "notebooks" / "_preludi" / f"{capitolo}.py"
    return f.read_text(encoding="utf-8") if f.exists() else None


def esegui(codici: list[str]) -> tuple[list[str], str]:
    """Esegue i blocchi in fila nello stesso spazio dei nomi.

    Restituisce (uscita di ciascun blocco, guasto). Il separatore e' un byte
    nullo, che nessun blocco del libro stampa: cosi' l'uscita si spezza per
    blocco senza doverla indovinare.
    """
    # Ogni blocco si scrive in un file VERO e si compila con quel percorso,
    # invece di passare la stringa a `compile` con un nome finto. Costa una
    # scrittura su disco e serve a chi risale al proprio sorgente: il
    # `@triton.jit` di `GPU/kernel-e-cuda.md` chiama `inspect.getsource`, che da
    # una stringa non ha niente da leggere, e falliva con «@jit functions should
    # be defined in a Python file» pur girando benissimo da file. Il guasto era
    # dello strumento, non della pagina, e si vedeva solo dichiarando un'uscita
    # in quel capitolo. In piu' i traceback adesso puntano a righe leggibili.
    driver = (
        "import json, sys, pathlib\n"
        f"SEG = {SEGNALE!r}\n"
        "blocchi = json.load(open(sys.argv[1]))\n"
        "spazio = {'__name__': '__main__'}\n"
        "for i, codice in enumerate(blocchi, 1):\n"
        "    f = pathlib.Path(f'blocco_{i}.py')\n"
        "    f.write_text(codice)\n"
        "    print(SEG, end='')\n"
        "    try:\n"
        "        exec(compile(codice, str(f.resolve()), 'exec'), spazio)\n"
        "    except BaseException as e:\n"
        "        sys.stdout.flush()\n"
        "        print(f'\\nROTTO {i} {type(e).__name__}: {e}', file=sys.stderr)\n"
        "        sys.exit(3)\n"
    )
    with tempfile.TemporaryDirectory() as cartella:
        d = pathlib.Path(cartella)
        (d / "b.json").write_text(json.dumps(codici))
        (d / "driver.py").write_text(driver)
        try:
            esito = subprocess.run(
                [sys.executable, str(d / "driver.py"), str(d / "b.json")],
                capture_output=True, text=True, timeout=ATTESA,
                env={**os.environ, **KERNEL_FISSI}, cwd=cartella)
        except subprocess.TimeoutExpired:
            return [], f"TIMEOUT: piu' di {ATTESA} secondi"

    pezzi = esito.stdout.split(SEGNALE)[1:]
    if esito.returncode != 0:
        guasto = next((r for r in esito.stderr.split("\n") if r.startswith("ROTTO")),
                      (esito.stderr.strip().split("\n") or ["?"])[-1])
        if "ModuleNotFoundError" in guasto:
            return pezzi, "NON VERIFICABILE QUI: " + guasto[:90]
        return pezzi, guasto[:140]
    return pezzi, ""


def normalizza(testo: str) -> list[str]:
    """Le righe, senza le differenze che la pagina non mostra.

    Si toglie lo spazio **in coda a ogni riga** e le righe vuote **agli
    estremi** del blocco, e non e' una scorciatoia: sono le due sole differenze
    che un lettore non puo' vedere, e sono esattamente quelle che la copiatura a
    mano introduce sempre (un editor che salva toglie gli spazi in coda; un
    `print("\\n...")` mette una riga vuota davanti che nessuno ricopia). Alla
    prima corsa di questo strumento erano sei disallineamenti su undici in un
    capitolo solo, tutti e sei di questo tipo: un controllo che non li perdona
    diventa sempre rosso, e un controllo sempre rosso non avvisa di niente.

    Tutto il resto resta significativo, compresi gli spazi **dentro** la riga e
    le righe vuote **in mezzo**, che nelle tabelle del libro separano due
    uscite."""
    return "\n".join(r.rstrip() for r in testo.split("\n")).strip("\n").split("\n")


def confronta(atteso: str, avuto: str) -> str:
    """Vuoto se combaciano, altrimenti la prima riga che differisce."""
    a, b = normalizza(atteso), normalizza(avuto)
    if a == b:
        return ""
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else "(non c'e' piu' niente)"
        y = b[i] if i < len(b) else "(non c'e' piu' niente)"
        if x != y:
            return f"riga {i + 1}\n        libro:   {x!r}\n        codice:  {y!r}"
    return "lunghezze diverse"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capitoli", nargs="*", help="quali capitoli (tutti se vuoto)")
    ap.add_argument("--elenco", action="store_true",
                    help="dice che cosa e' coperto e che cosa no, senza eseguire")
    ap.add_argument("--salta-instabili", action="store_true",
                    help="lascia fuori i capitoli di `INSTABILI`, le cui uscite"
                         " non sono riproducibili fra macchine diverse (lo usa"
                         " la CI pubblica, dove l'hardware ruota)")
    ap.add_argument("--anche-lenti", action="store_true",
                    help="esegue anche i blocchi `pt-lento`, che la CI salta "
                         "(scaricano modelli: serve la rete, e del tempo)")
    args = ap.parse_args()

    tutti = GN.capitoli()
    scelti = {k: v for k, v in tutti.items()
              if not args.capitoli or k in args.capitoli}
    saltati = {}
    if args.salta_instabili:
        saltati = {k: INSTABILI[k] for k in list(scelti) if k in INSTABILI}
        for k in saltati:
            del scelti[k]
    if args.capitoli and not scelti:
        print(f"nessun capitolo fra {args.capitoli}; ci sono: "
              f"{', '.join(sorted(tutti))}")
        sys.exit(2)

    if args.elenco:
        con, senza = 0, []
        for nome, pagine in sorted(scelti.items()):
            b = [x for p in pagine for x in blocchi_di(p)]
            quanti = sum(1 for _, _, a, _ in b if a is not None)
            con += quanti
            if b and not quanti:
                senza.append(nome)
            print(f"  {nome:28} {len(b):3} blocchi, {quanti:3} con un'uscita da controllare")
        print(f"\n{con} blocchi con un'uscita dichiarata.")
        if senza:
            print("capitoli che eseguono codice senza mai dichiarare che cosa "
                  "stampa: " + ", ".join(senza))
        return

    avvisa_se_carica()
    # Un'esclusione taciuta e' un buco; detta, e' una decisione. La si legge nel
    # log della CI, accanto ai capitoli che invece sono stati controllati.
    for nome, perche in sorted(saltati.items()):
        print(f"  ~  {nome:28} fuori da qui: {perche}")
    problemi = 0
    for nome, pagine in sorted(scelti.items()):
        raccolti = [(p, *x) for p in pagine
                    for x in blocchi_di(p, args.anche_lenti)]
        if not raccolti:
            continue
        da_controllare = sum(1 for r in raccolti if r[3] is not None)
        if not da_controllare:
            print(f"  ·  {nome:28} {len(raccolti)} blocchi, nessuna uscita dichiarata")
            continue

        # il preludio si esegue prima di tutto e la sua uscita si butta: e'
        # impalcatura, non una delle uscite che il libro dichiara
        pre = preludio_di(nome)
        codici = [r[2] for r in raccolti]
        uscite, guasto = esegui(([pre] if pre else []) + codici)
        if pre:
            uscite = uscite[1:]
        if guasto:
            print(f"  ✗  {nome:28} {guasto}")
            problemi += 1
            continue

        rotti = []
        for i, (pagina, riga, _, atteso, riga_att) in enumerate(raccolti):
            if atteso is None:
                continue
            diverso = confronta(atteso, uscite[i] if i < len(uscite) else "")
            if diverso:
                rotti.append((pagina.relative_to(LIBRO), riga_att, diverso))
        if rotti:
            problemi += len(rotti)
            print(f"  ✗  {nome:28} {len(rotti)} uscite su {da_controllare} non combaciano")
            for pagina, riga, diverso in rotti:
                print(f"       {pagina}:{riga}  {diverso}")
        else:
            nota = "  (con il preludio del notebook)" if pre else ""
            print(f"  ✓  {nome:28} {da_controllare} uscite su "
                  f"{da_controllare} combaciano{nota}")

    print()
    if problemi:
        print(f"{problemi} uscite non combaciano: o il codice e' cambiato, o il "
              f"blocco ```text e' stato scritto a mano e non ricontrollato.")
        sys.exit(1)
    print("ogni numero stampato nel libro e' quello che il codice produce.")


if __name__ == "__main__":
    main()
