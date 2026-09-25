"""La macchina di Turing che somma uno, passo per passo.

È la macchina di `Introduzione/calcolabile.md`: sei regole, gli stati «vai»,
«riporto» e «fine». Qui non si disegnano i fotogrammi a mano: la macchina gira
davvero (`esegui`), e gli assert controllano quello che la didascalia promette,
cioè che da 1011 si arrivi a 1100 in otto passi, fermandosi nello stato «fine».

Il disegno fermo è l'ultimo fotogramma: il nastro con 1100 e la testina sulla
seconda casella, dove la macchina si è fermata.
"""

from paithon_svg import *

NOME = "nastro-e-testina"
TITOLO = "una macchina di Turing somma uno"

REGOLE = {
    ("vai", "0"): ("0", +1, "vai"),
    ("vai", "1"): ("1", +1, "vai"),
    ("vai", " "): (" ", -1, "riporto"),
    ("riporto", "1"): ("0", -1, "riporto"),
    ("riporto", "0"): ("1", 0, "fine"),
    ("riporto", " "): ("1", 0, "fine"),
}
INGRESSO = "1011"
CELLE = range(-1, 6)            # le caselle disegnate, bianche comprese
LATO = 60
X0, Y0 = 110, 70                # l'angolo della casella di indice -1


def esegui():
    """I fotogrammi: (nastro, posizione della testina, stato), dal primo all'ultimo."""
    nastro = dict(enumerate(INGRESSO))
    testina, stato = 0, "vai"
    fotogrammi = [(dict(nastro), testina, stato)]
    while (stato, nastro.get(testina, " ")) in REGOLE:
        scrivi, sposta, stato = REGOLE[(stato, nastro.get(testina, " "))]
        nastro[testina] = scrivi
        testina += sposta
        fotogrammi.append((dict(nastro), testina, stato))
    return fotogrammi


def cx(i: int) -> float:
    return X0 + (i - CELLE[0]) * LATO + LATO / 2


def costruisci() -> Figura:
    foto = esegui()
    finale = "".join(foto[-1][0].get(i, " ") for i in range(0, 4))
    assert finale == "1100", f"la macchina scrive {finale!r}, la didascalia dice 1100"
    assert len(foto) - 1 == 8, f"passi {len(foto) - 1}, la didascalia dice otto"
    assert foto[-1][2] == "fine" and foto[-1][1] == 1
    assert all(p in CELLE for _, p, _ in foto), "la testina esce dal disegno"

    n = len(foto)
    corpo, anim = [], []

    def acceso(nome, quando):
        """Visibile nei fotogrammi `quando` (lista di bool), a riposo sull'ultimo."""
        tappe = [(0.0, f"opacity:{1 if quando[0] else 0}")]
        for k in range(1, n):
            t0, _ = sosta(k, n)
            if quando[k] != quando[k - 1]:
                tappe += [(t0 - 0.6, f"opacity:{1 if quando[k - 1] else 0}"),
                          (t0, f"opacity:{1 if quando[k] else 0}")]
        tappe.append((100.0, f"opacity:{1 if quando[-1] else 0}"))
        anim.append(keyframes(nome, tappe))
        return f"animation:{nome} var(--d) infinite;opacity:{1 if quando[-1] else 0}"

    # le caselle, e per ogni casella un testo per ciascun simbolo che ci passa
    for i in CELLE:
        x = X0 + (i - CELLE[0]) * LATO
        corpo.append(f'<rect class="cella" x="{x}" y="{Y0}" width="{LATO}" height="{LATO}"/>')
        simboli = sorted({f[0].get(i, " ") for f in foto} - {" "})
        for s in simboli:
            quando = [f[0].get(i, " ") == s for f in foto]
            stile = acceso(f"s{i + 1}{s}", quando) if not all(quando) else ""
            corpo.append(f'<text class="sim" x="{cx(i):.0f}" y="{Y0 + 41}" '
                         f'text-anchor="middle" style="{stile}">{s}</text>')

    # la testina: il riposo è la posizione finale, le altre sono spostamenti
    finale_x = cx(foto[-1][1])
    tappe = [(0.0, f"transform:translateX({cx(foto[0][1]) - finale_x:.0f}px)")]
    for k in range(1, n):
        t0, _ = sosta(k, n)
        prima = f"transform:translateX({cx(foto[k - 1][1]) - finale_x:.0f}px)"
        dopo = f"transform:translateX({cx(foto[k][1]) - finale_x:.0f}px)"
        tappe += [(t0 - 1.5, prima), (t0 + 1.5, dopo)]
    tappe.append((100.0, "transform:translateX(0px)"))
    anim.append(keyframes("testina", tappe))
    ty = Y0 + LATO + 10
    corpo.append(f'<path class="testina" d="M{finale_x:.0f},{ty} L{finale_x - 13:.0f},{ty + 20} '
                 f'L{finale_x + 13:.0f},{ty + 20} Z" style="animation:testina var(--d) infinite"/>')

    # lo stato e il contatore dei passi
    for st in ("vai", "riporto", "fine"):
        quando = [f[2] == st for f in foto]
        corpo.append(f'<text class="stato" x="{X0}" y="{Y0 + LATO + 70}" '
                     f'style="{acceso("st" + st, quando)}">stato: {st}</text>')
    for k in range(n):
        quando = [j == k for j in range(n)]
        corpo.append(f'<text class="passo" x="{X0}" y="{Y0 - 22}" '
                     f'style="{acceso(f"p{k}", quando)}">'
                     f'{"all" + chr(8217) + "inizio" if k == 0 else f"dopo il passo {k}"}</text>')

    return Figura(
        larghezza=640, altezza=250,
        alt="Animazione: un nastro di caselle con scritto 1011; una testina a triangolo "
            "si sposta di casella in casella verso destra fino alla prima casella "
            "bianca, poi torna indietro trasformando i due 1 finali in 0 e lo 0 in 1. "
            "Sotto il nastro il nome dello stato cambia da vai a riporto a fine. Alla "
            "fine sul nastro c'è 1100 e la testina è ferma sulla seconda casella.",
        corpo="".join(corpo),
        stile=f"""    .cella   {{ fill:none; stroke:{INK}; stroke-width:1.6; }}
    .sim     {{ font-family:{SANS}; font-size:30px; font-weight:700; fill:{TEAL}; }}
    .testina {{ fill:{TERRACOTTA}; }}
    .stato   {{ font-family:{SANS}; font-size:17px; font-weight:700; fill:{TERRACOTTA}; }}
    .passo   {{ font-family:{SANS}; font-size:15px; fill:{FG_MUTED}; }}""",
        animazioni=anim,
        durata=n * 1.1,
        fermi=".sim, .testina, .stato, .passo",
    )
