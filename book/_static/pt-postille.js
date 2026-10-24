/**
 * PAITHON BOOK: LE POSTILLE
 *
 * Una postilla e' una riga accanto a un capoverso del testo comune che dice
 * di che cosa parla il capoverso. La build la mette in pagina come un
 * `<aside class="pt-postilla" hidden>` subito dopo il suo capoverso
 * (`_ext/pt_postille.py`); qui le si da' il segno che la apre.
 *
 * Il segno compare sul capoverso su cui si e': col mouse quando ci si passa
 * sopra (lo fa il CSS), con la tastiera quando prende il fuoco, con le dita
 * sul capoverso che attraversa la riga di lettura, a un terzo dell'altezza
 * dello schermo, uno alla volta.
 *
 * Premendo il segno il capoverso sfuma e al suo posto la postilla si scrive,
 * una lettera dopo l'altra, in Fraunces corsivo. Tenendolo premuto si
 * sbircia: al rilascio torna il capoverso. Un tocco breve, un clic o Invio
 * la lasciano scritta finche' non si preme di nuovo, o finche' non si preme
 * Esc. La scritta e' un disegno (`aria-hidden`): per chi legge con un lettore
 * di schermo la postilla resta l'`<aside>`, che si apre come prima.
 *
 * Senza questo script le postille restano nascoste e la pagina e' quella di
 * sempre.
 */
(function () {
  'use strict';

  // Sphinx emette due volte i file di `html_js_files`: senza la guardia ogni
  // capoverso avrebbe due segni.
  if (window.__ptPostilleCaricate) return;
  window.__ptPostilleCaricate = true;

  const ETICHETTA = 'Postilla: di che cosa parla questo capoverso';
  // Lo spazio che il segno vuole fuori dalla colonna: 5px di stacco e 24 di
  // segno, come in `pt-postille.css`.
  const SPAZIO = 29;
  const DITA = !!(window.matchMedia &&
                  window.matchMedia('(pointer: coarse)').matches);
  // Il segno e' il marchio servito da `_static`, all'indirizzo relativo che
  // scrive la build: un'aside che chiede un'immagine da un altro indirizzo non
  // viene dalla build, e non prende il segno.
  const MARCHIO = /^(\.\.\/)*_static\/brand\/logos\/paithon-mark\.svg$/;

  // Sotto questa durata la pressione e' un tocco, e la postilla resta
  // scritta; sopra e' uno sguardo, e al rilascio il capoverso torna.
  const TOCCO = 280;

  // La scritta: una lettera per `<span>`, ciascuna col suo indice (`--i`) per
  // il ritardo dell'animazione; le lettere di una parola stanno in una
  // parola che non si spezza, e si va a capo solo fra le parole.
  function scritta(testo) {
    const scritta = document.createElement('span');
    scritta.className = 'pt-postilla__scritta';
    scritta.setAttribute('aria-hidden', 'true');
    const dentro = document.createElement('span');
    let i = 0;
    testo.trim().split(/(\s+)/).forEach(function (pezzo) {
      if (!pezzo) return;
      if (/^\s+$/.test(pezzo)) {
        dentro.appendChild(document.createTextNode(' '));
        i += 1;
        return;
      }
      const parola = document.createElement('span');
      parola.className = 'pt-postilla__parola';
      Array.from(pezzo).forEach(function (lettera) {
        const l = document.createElement('span');
        l.textContent = lettera;
        l.style.setProperty('--i', i);
        i += 1;
        parola.appendChild(l);
      });
      dentro.appendChild(parola);
    });
    scritta.appendChild(dentro);
    return scritta;
  }

  function alterna(coppia, apri) {
    coppia.postilla.hidden = !apri;
    coppia.segno.setAttribute('aria-expanded', apri ? 'true' : 'false');
    coppia.capoverso.classList.toggle('pt-postilla-attiva', apri);
    if (apri) {
      // L'animazione riparte da capo a ogni apertura.
      coppia.scritta.classList.remove('pt-scrive');
      void coppia.scritta.offsetWidth;
      coppia.scritta.classList.add('pt-scrive');
    } else {
      coppia.fissa = false;
    }
  }

  function prepara(postilla) {
    const capoverso = postilla.previousElementSibling;
    if (!capoverso || capoverso.tagName !== 'P') return null;
    if (!MARCHIO.test(postilla.dataset.segno || '')) return null;
    capoverso.classList.add('pt-con-postilla');

    const segno = document.createElement('button');
    segno.type = 'button';
    segno.className = 'pt-postilla__segno';
    segno.setAttribute('aria-expanded', 'false');
    segno.setAttribute('aria-controls', postilla.id);
    segno.setAttribute('aria-label', ETICHETTA);
    segno.title = ETICHETTA;

    const marchio = document.createElement('img');
    // `dark-light`: senza, nel tema scuro il tema da' a ogni immagine del
    // contenuto uno sfondo bianco e un filtro.
    marchio.className = 'dark-light';
    marchio.src = postilla.dataset.segno;
    marchio.alt = '';
    marchio.width = 14;
    marchio.height = 14;
    segno.appendChild(marchio);

    // Il segno sta a meta' della prima riga: il CSS la vuole in pixel.
    const riga = parseFloat(getComputedStyle(capoverso).lineHeight);
    if (riga) capoverso.style.setProperty('--pt-postilla-riga', riga + 'px');

    const coppia = {
      capoverso: capoverso, segno: segno, postilla: postilla,
      scritta: scritta(postilla.textContent), fissa: false,
      inizio: 0, chiudeAlRilascio: false, premuto: false,
    };
    capoverso.appendChild(coppia.scritta);

    // Il dito o il mouse sul segno: la postilla si scrive subito, e il
    // rilascio decide se era un tocco (resta) o uno sguardo (si richiude).
    segno.addEventListener('pointerdown', function (evento) {
      if (evento.button !== 0) return;
      evento.preventDefault();
      // Col puntatore catturato il rilascio arriva al segno anche se il dito
      // o il mouse ne sono usciti; un puntatore non attivo la rifiuta.
      try { segno.setPointerCapture(evento.pointerId); } catch (e) { /* niente */ }
      coppia.premuto = true;
      coppia.inizio = performance.now();
      coppia.chiudeAlRilascio = !postilla.hidden && coppia.fissa;
      if (postilla.hidden) alterna(coppia, true);
    });
    function rilascio(annullato) {
      if (!coppia.premuto) return;
      coppia.premuto = false;
      if (coppia.chiudeAlRilascio) {
        alterna(coppia, false);
      } else if (!annullato && performance.now() - coppia.inizio < TOCCO) {
        coppia.fissa = true;
      } else {
        alterna(coppia, false);
      }
    }
    segno.addEventListener('pointerup', function () { rilascio(false); });
    segno.addEventListener('pointercancel', function () { rilascio(true); });
    // Una pressione lunga col dito non deve aprire il menu del telefono.
    segno.addEventListener('contextmenu', function (evento) {
      evento.preventDefault();
    });
    // Il clic della tastiera (Invio, spazio) non passa dal puntatore:
    // `detail` vale 0, e il clic fissa o richiude. Quello del puntatore l'ha
    // gia' gestito il rilascio.
    segno.addEventListener('click', function (evento) {
      if (evento.detail !== 0) return;
      const apri = postilla.hidden;
      alterna(coppia, apri);
      if (apri) coppia.fissa = true;
    });
    capoverso.appendChild(segno);
    return coppia;
  }

  function seguiLaLettura(coppie) {
    let attivo = null;
    const osservatore = new IntersectionObserver(function (voci) {
      voci.forEach(function (voce) {
        if (!voce.isIntersecting) {
          // Uscito dalla riga di lettura, il capoverso spegne il suo segno
          // anche se nessun altro capoverso con postilla ci e' arrivato.
          if (voce.target === attivo) {
            attivo.classList.remove('pt-in-lettura');
            attivo = null;
          }
          return;
        }
        if (attivo && attivo !== voce.target) {
          attivo.classList.remove('pt-in-lettura');
        }
        attivo = voce.target;
        attivo.classList.add('pt-in-lettura');
      });
    }, { rootMargin: '-33% 0px -66% 0px', threshold: 0 });
    coppie.forEach(function (coppia) { osservatore.observe(coppia.capoverso); });
  }

  function avvia() {
    const coppie = Array.prototype.map
      .call(document.querySelectorAll('aside.pt-postilla'), prepara)
      .filter(Boolean);
    if (!coppie.length) return;

    // Dove fra la colonna e il bordo dello schermo non c'e' lo SPAZIO, il
    // capoverso si stringe di quanto serve e il segno gli sta dentro: niente
    // deve uscire dallo schermo, ne' coprire il testo.
    function misura() {
      const larghezza = document.documentElement.clientWidth;
      coppie.forEach(function (coppia) {
        const libero = larghezza - coppia.capoverso.getBoundingClientRect().right;
        coppia.capoverso.classList.toggle('pt-con-postilla--stretto', libero < SPAZIO);
      });
    }
    misura();
    let attesa = 0;
    window.addEventListener('resize', function () {
      cancelAnimationFrame(attesa);
      attesa = requestAnimationFrame(misura);
    });

    document.addEventListener('keydown', function (evento) {
      if (evento.key !== 'Escape') return;
      coppie.forEach(function (coppia) {
        if (!coppia.postilla.hidden) alterna(coppia, false);
      });
    });

    if (DITA && 'IntersectionObserver' in window) seguiLaLettura(coppie);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', avvia);
  } else {
    avvia();
  }
})();
