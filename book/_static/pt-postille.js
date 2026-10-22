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
 * dello schermo, uno alla volta. La postilla si apre sotto il capoverso e
 * spinge giu' il testo: di lato non c'e' posto, a nessuna larghezza, e niente
 * deve coprire il testo che si sta leggendo.
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

  function alterna(segno, postilla, apri) {
    postilla.hidden = !apri;
    segno.setAttribute('aria-expanded', apri ? 'true' : 'false');
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

    segno.addEventListener('click', function () {
      alterna(segno, postilla, postilla.hidden);
    });
    capoverso.appendChild(segno);
    return { capoverso: capoverso, segno: segno, postilla: postilla };
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
        if (!coppia.postilla.hidden) alterna(coppia.segno, coppia.postilla, false);
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
