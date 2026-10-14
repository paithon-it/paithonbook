#!/usr/bin/env python3
"""Rimandi fra capitoli: li trova nel testo e li rende link.

Quando una pagina nomina un altro capitolo («nel capitolo sui Transformer»)
il lettore deve poterci andare con un dito. Qui il rimando si trova, si
risolve sul `_toc.yml` e diventa un `{doc}` MyST.

    python3 scripts/rimandi.py              # elenca i candidati
    python3 scripts/rimandi.py --scrivi     # li trasforma in link
    python3 scripts/rimandi.py --verifica   # esce 1 se ne restano di scoperti

Si linka la PRIMA menzione di ogni capitolo in ogni pagina, non tutte: cinque
link allo stesso posto nella stessa pagina sono rumore. I rimandi alla propria
cartella non si toccano, e nemmeno quelli gia' dentro un link.
"""
import bisect
import re
import sys
from pathlib import Path

import yaml

RADICE = Path(__file__).resolve().parent.parent / "book"

# Le parole con cui il testo chiama un capitolo. La chiave e' la cartella.
ALIAS = {
    "Introduzione": ["introduzione"],
    "Python": ["python"],
    "Matematica": ["matematica", "algebra lineare", "analisi numerica",
                   "teoria dell'informazione", "probabilita", "probabilità"],
    "MachineLearning": ["machine learning", "apprendimento automatico"],
    "RetiNeurali": ["reti neurali", "rete neurale", "percettrone",
                    "backpropagation"],
    "PyTorch": ["pytorch"],
    "GPU": ["gpu", "calcolo parallelo", "cuda"],
    "Efficienza": ["efficienza", "quantizzazione", "distillazione"],
    "DeepLearning": ["deep learning", "reti convoluzionali", "cnn",
                     "apprendimento profondo"],
    "VisioneArtificiale": ["visione artificiale", "computer vision",
                           "visione"],
    "Ricerca": ["ricerca e pianificazione", "ricerca", "pianificazione"],
    "ReinforcementLearning": ["reinforcement learning",
                              "apprendimento per rinforzo", "rinforzo"],
    "DeepReinforcementLearning": ["deep reinforcement learning",
                                  "deep rl", "rinforzo profondo"],
    "NaturalLanguageProcessing": ["natural language processing", "nlp",
                                  "elaborazione del linguaggio"],
    "Transformers": ["transformer", "attenzione", "llm",
                     "grandi modelli linguistici"],
    "AttenzioneLineare": ["attenzione lineare"],
    "StateSpaceModel": ["state space model", "spazio degli stati", "mamba"],
    "VisioneLinguaggio": ["visione e linguaggio", "modelli multimodali",
                          "multimodalita", "multimodalità"],
    "Agenti": ["agenti", "agente", "tool use"],
    "IngegneriaLLM": ["prompt, contesto e loop", "prompt engineering",
                      "context engineering", "prompt"],
    "SistemiMultiAgente": ["sistemi multi-agente", "multi-agente"],
    "Audio": ["audio"],
    "SpeechRecognition": ["speech recognition", "riconoscimento vocale",
                          "sintesi vocale"],
    "ModelliLatenti": ["modelli latenti", "autoencoder", "vae"],
    "GAN": ["gan", "reti avversarie"],
    "ModelliDiffusione": ["modelli di diffusione", "diffusione"],
    "VerosimiglianzaEsatta": ["verosimiglianza esatta", "flussi normalizzanti",
                              "verosimiglianza"],
    "ModelliEnergia": ["modelli a energia", "energia", "boltzmann",
                       "hopfield"],
    "AutoSupervisione": ["auto-supervisione", "autosupervisione",
                         "apprendimento auto-supervisionato"],
    "WorldModels": ["world model", "modelli del mondo", "jepa"],
    # «reti neurali su grafo» va prima di «reti neurali»: vince l'alias
    # piu' lungo, e senza questo il nome del capitolo sui grafi portava
    # a quello sulle reti neurali.
    "GraphNeuralNetwork": ["graph neural network", "reti neurali su grafo",
                           "reti neurali sui grafi", "grafi", "gnn"],
    "SistemiRaccomandazione": ["sistemi di raccomandazione",
                               "raccomandazione"],
    "SerieTemporali": ["serie temporali"],
    "PINN": ["pinn", "fisica"],
    "MLOps": ["mlops", "produzione", "messa in produzione"],
    "Interpretabilita": ["interpretabilita", "interpretabilità",
                         "spiegabilita", "spiegabilità"],
    "AIResponsabile": ["ai responsabile", "equita", "equità", "bias",
                       "allineamento", "sicurezza"],
    "Conclusioni": ["conclusioni"],
}

# «nel capitolo sui Transformer», «il capitolo dedicato alla diffusione»,
# «il capitolo precedente sui grafi», «il capitolo di Machine Learning».
RIMANDO = re.compile(
    r"\b(capitol[oi])\s+"
    r"(?:(?:precedente|seguente|successiv[oi]|nuovo|scors[oi])\s+)?"
    # Le preposizioni articolate vanno dalla piu' lunga alla piu' corta.
    # Con `su[ila]?\s*` in testa, «capitolo sulle reti neurali» faceva match
    # su «sul» e lasciava «le reti neurali», che nessun alias riconosce: il
    # rimando spariva senza dire niente, e con esso ogni «sulle», «sull'»,
    # «sugli», «sulla». L'ordine qui non e' estetico, e' il controllo.
    r"(?:sugli\s+|sulle\s+|sulla\s+|sull'|sull’|sui\s+|sul\s+|su\s+|"
    r"degli\s+|delle\s+|della\s+|dell'|dell’|dei\s+|del\s+|di\s+|"
    r"dedicat[oi]\s+agli\s+|dedicat[oi]\s+alle\s+|dedicat[oi]\s+alla\s+|"
    r"dedicat[oi]\s+all'|dedicat[oi]\s+all’|dedicat[oi]\s+ai\s+|"
    r"dedicat[oi]\s+al\s+|dedicat[oi]\s+a\s+|"
    r"che\s+parla\s+d[iael]+\s*|intitolato\s+)?"
    r"([A-Za-zÀ-ù0-9'’\- ]{2,40})",
    re.IGNORECASE,
)


def documenti():
    """Ogni cartella-capitolo con il suo overview, dal toc."""
    toc = yaml.safe_load((RADICE / "_toc.yml").read_text())
    trovati = {}
    for parte in toc.get("parts", []):
        for cap in parte.get("chapters", []):
            f = cap["file"]
            if "/" in f:
                trovati[f.split("/")[0]] = "/" + f.rsplit(".", 1)[0]
    return trovati


def maschera(testo):
    """Le righe che non sono prosa: codice, formule di blocco, direttive.

    Non tutte le recinzioni sono codice, ed e' l'errore che questa funzione
    faceva: `{tab}` e `{admonition}` contengono PROSA, e in questo libro dentro
    le schede sta piu' della meta' delle parole. Trattandole come codice il
    rilevatore leggeva la sola spina dorsale e tornava zero, che e' il numero
    piu' rassicurante che esista.

    E serve una pila, non un interruttore: dentro una scheda a cinque apici sta
    quasi sempre un ```python a tre, e con l'interruttore quel blocco *spegneva*
    la maschera invece di accenderla, cioe' da li' in poi contava a rovescio.
    Si maschera una riga quando almeno una delle recinzioni che la contengono e'
    di codice (recinzione senza `{direttiva}` subito dopo gli apici).
    """
    fuori = set()
    pila = []           # (numero di apici, e' una recinzione di codice)
    dentro_math = False
    for i, riga in enumerate(testo.splitlines()):
        s = riga.strip()
        apici = len(s) - len(s.lstrip("`"))
        if apici >= 3 or s.startswith(":::"):
            fuori.add(i)
            if s.startswith(":::"):
                continue
            coda = s[apici:].strip()
            if pila and not coda and apici >= pila[-1][0]:
                pila.pop()
            else:
                pila.append((apici, not coda.startswith("{")))
            continue
        if s.startswith("$$"):
            dentro_math = not dentro_math
            fuori.add(i)
            continue
        if (dentro_math or any(codice for _, codice in pila)
                or s.startswith(":") or s.startswith("#")):
            fuori.add(i)
    return fuori


# Le spallette di un link (un ruolo come `{doc}` o `{ref}`, un link Markdown)
# e il codice in linea si coprono, non si saltano: una riga che ha gia' un
# link puo' avere accanto un rimando in prosa. Si coprono con un segno che non
# e' ne' spazio ne' lettera: coperta di spazi, «il capitolo {doc}`precedente
# <x>` sulla visione» si leggerebbe «il capitolo ... sulla visione», cioe' un
# rimando che nel testo non c'e'.
COPERTO = "\x00"
SPALLETTE = re.compile(r"\{[a-z-]+\}`[^`]*`|`[^`]*`|\[[^\]]*\]\([^)]*\)")

# Una voce d'elenco o una riga di tabella comincia un pezzo nuovo: un rimando
# non scavalca il confine fra due voci.
NUOVO_PEZZO = re.compile(r"\s*(?:[-*+]\s|\d+\.\s|\|)")


def capoversi(righe, saltate):
    """I pezzi di prosa di una pagina, come (indice della prima riga, righe).

    Il libro va a capo a ottanta colonne, quindi «nel capitolo» in fondo a
    una riga e «su Python» all'inizio della successiva sono lo stesso
    rimando, e nessuna delle due righe lo contiene: si cerca nel capoverso.
    Lo chiudono la riga vuota, una riga mascherata, una voce d'elenco e una
    riga di tabella, perche' due capoversi sono due frasi diverse.
    """
    pezzo, inizio = [], 0
    for n, riga in enumerate(righe):
        if n in saltate or not riga.strip():
            if pezzo:
                yield inizio, pezzo
                pezzo = []
            continue
        if pezzo and NUOVO_PEZZO.match(riga):
            yield inizio, pezzo
            pezzo = []
        if not pezzo:
            inizio = n
        pezzo.append(riga)
    if pezzo:
        yield inizio, pezzo


def candidati(percorso, overview):
    """I rimandi da linkare di una pagina, come (riga, colonna d'inizio,
    colonna di fine, testo, documento, avanzo).

    Riga e colonne sono quelle della riga in cui il rimando comincia; se il
    nome scavalca l'a capo, la colonna di fine e' la fine della riga e
    `avanzo` dice quanti caratteri del nome stanno sulla riga dopo, piu' uno
    per lo spazio che l'a capo diventa. `avanzo` vale -1 quando il rimando sta
    su piu' di due righe: si conta e si mostra, ma il link si scrive a mano.
    """
    testo = percorso.read_text()
    saltate = maschera(testo)
    mio = percorso.relative_to(RADICE).parts[0]
    # Un capitolo gia' linkato in questa pagina conta come visto: se no la
    # SECONDA menzione prende il posto della prima e --verifica non passa mai.
    # Vale QUALUNQUE pagina di quel capitolo, non il solo overview:
    # CONTRIBUTING.md dice di puntare alla SEZIONE che contiene la cosa
    # promessa, quindi una pagina che linka `</Agenti/context-engineering>` ha
    # gia' fatto il suo lavoro. Cercando il solo overview lo strumento chiede
    # un secondo link allo stesso capitolo, cioe' proprio il rumore che la
    # regola vieta.
    # E un link conta in tutte le sue forme: con il titolo (`</C/`), senza
    # (`` `/C/ ``) e in Markdown (`](../C/`).
    visti = {c for c in overview
             if any(s + c + "/" in testo for s in ("</", "`/", "](../"))}
    fuori = []
    for inizio, righe in capoversi(testo.splitlines(), saltate):
        # Le posizioni restano quelle del capoverso: si copre, non si toglie,
        # e l'a capo diventa uno spazio solo nel testo in cui si cerca.
        piatto = SPALLETTE.sub(lambda m: COPERTO * len(m.group(0)),
                               "\n".join(righe)).replace("\n", " ")
        partenze = [0]
        for r in righe[:-1]:
            partenze.append(partenze[-1] + len(r) + 1)
        for m in RIMANDO.finditer(piatto):
            coda = piatto[m.start(2):].lower().replace("’", "'")
            bersaglio = None
            for cartella, parole in ALIAS.items():
                for p in parole:
                    if coda.startswith(p):
                        if bersaglio is None or len(p) > len(bersaglio[1]):
                            bersaglio = (cartella, p)
            if not bersaglio or bersaglio[0] == mio or bersaglio[0] in visti:
                continue
            if bersaglio[0] not in overview:
                continue
            visti.add(bersaglio[0])
            k = bisect.bisect_right(partenze, m.start()) - 1
            fine = m.start(2) + len(bersaglio[1])
            dove = bisect.bisect_right(partenze, fine) - 1
            riga, col = righe[k], m.start() - partenze[k]
            if dove == k:
                stop, avanzo = fine - partenze[k], 0
            elif dove == k + 1:
                # Il nome scavalca l'a capo: il link se lo tiene dentro, e
                # l'a capo dentro le spallette diventa uno spazio in resa.
                stop, avanzo = len(riga), fine - partenze[dove] + 1
            else:
                # Due a capo dentro un rimando, a ottanta colonne, non ci
                # stanno, ma in un testo non ancora riavvolto, a righe corte,
                # si': il rimando si conta, e chi scrive i link lo lascia a mano.
                stop, avanzo = len(riga), -1
            fuori.append((inizio + k + 1, col, stop, riga[col:stop],
                          overview[bersaglio[0]], avanzo))
    return fuori


def main():
    scrivi = "--scrivi" in sys.argv
    verifica = "--verifica" in sys.argv
    # `--solo <Cartella>`: serve quando si lavora a un capitolo solo e non si
    # devono toccare i file degli altri. Senza, `--scrivi` riscrive tutto il
    # libro, che con altri lavori in corso e' un conflitto.
    solo = None
    if "--solo" in sys.argv:
        solo = sys.argv[sys.argv.index("--solo") + 1]
    overview = documenti()
    totale = 0
    letti = 0
    for percorso in sorted(RADICE.rglob("*.md")):
        if "_build" in percorso.parts or "_static" in percorso.parts:
            continue
        if percorso.name in ("aggiornamenti.md", "references.md"):
            continue
        if solo and percorso.relative_to(RADICE).parts[0] != solo:
            continue
        letti += 1
        trovati = candidati(percorso, overview)
        if not trovati:
            continue
        rel = percorso.relative_to(RADICE)
        if scrivi:
            a_mano = [t for t in trovati if t[5] < 0]
            totale += len(trovati) - len(a_mano)
            righe = percorso.read_text().splitlines(keepends=True)
            for n, col, stop, testo, doc, avanzo in sorted(trovati,
                                                            reverse=True):
                if avanzo < 0:
                    continue
                riga = righe[n - 1].rstrip("\n")
                if avanzo:
                    taglio = avanzo - 1  # lo spazio finto fra le due righe
                    seg, resto = righe[n][:taglio], righe[n][taglio:]
                    righe[n - 1] = riga[:col] + "{doc}`" + testo + "\n"
                    righe[n] = seg + " <" + doc + ">`" + resto
                else:
                    righe[n - 1] = (riga[:col] + "{doc}`" + testo + " <" + doc
                                    + ">`" + riga[stop:] + "\n")
            percorso.write_text("".join(righe))
            print(f"  {rel}: {len(trovati) - len(a_mano)} link")
            for n, _, _, testo, doc, _a in a_mano:
                print(f"  {rel}:{n}  «{testo}» -> {doc}: sta su piu' di due"
                      " righe, il link va scritto a mano")
        else:
            totale += len(trovati)
            for n, _, _, testo, doc, _a in trovati:
                print(f"{rel}:{n}  «{testo}» -> {doc}")
    # Il denominatore accanto al numero: uno zero senza «su quanti file» non
    # e' una misura, e con `--solo` sbagliato la glob e' vuota e torna zero.
    assert letti, "nessun file letto: perimetro sbagliato?"
    print(f"\n{totale} rimandi" + (" scritti" if scrivi else " da linkare")
          + f", su {letti} file letti"
          + (f" (solo {solo})" if solo else ""))
    if verifica and totale:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
