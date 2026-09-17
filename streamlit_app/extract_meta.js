/* =====================================================================
   index.html icindeki JS meta verisini (sozluk + yontem katalogu) JSON'a
   cikarir. TEK KAYNAK index.html'dir: Streamlit portu bu JSON'u okur,
   boylece 56 yontem etiketi / 475 ceviri anahtari elle KOPYALANMAZ ve
   panel guncellenince tek komutla yeniden uretilir:
       node streamlit_app/extract_meta.js
   ===================================================================== */
const fs = require("fs");
const path = require("path");
const ROOT = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(ROOT, "dashboard", "index.html"), "utf8");

/* `const NAME = <literal>;` blogunu parantez dengeleyerek kesip getirir.
   Kaba bir regex yetmez: etiketlerin icinde suslu/koseli parantez ve
   tirnak var (ornegin "b∈{1..32}", "k∈{2,4,8}"). */
function literalOf(name) {
  const re = new RegExp(String.raw`(?:const|let|var)\s+${name}\s*=\s*`, "g");
  const m = re.exec(html);
  if (!m) throw new Error("bulunamadi: " + name);
  const i = m.index + m[0].length;
  const open = html[i];
  const close = open === "{" ? "}" : open === "[" ? "]" : null;
  if (!close) throw new Error(name + " bir nesne/dizi degil: " + open);
  let depth = 0, q = null, esc = false;
  for (let j = i; j < html.length; j++) {
    const c = html[j];
    if (esc) { esc = false; continue; }
    if (q) {
      if (c === "\\") esc = true;
      else if (c === q) q = null;
      continue;
    }
    if (c === '"' || c === "'" || c === "`") { q = c; continue; }
    if (c === "/" && html[j + 1] === "*") { j = html.indexOf("*/", j + 2) + 1; continue; }
    if (c === "/" && html[j + 1] === "/") { j = html.indexOf("\n", j); continue; }
    if (c === open) depth++;
    else if (c === close) { depth--; if (depth === 0) return html.slice(i, j + 1); }
  }
  throw new Error(name + ": kapanis bulunamadi");
}

/* Yalnizca TR/EN sozlugu cikarilir. Yontem katalogu (METHODS, MINFO,
   TABLE_GROUPS, on-ayarlar...) tek gorunumlu Streamlit sayfasinda
   kullanilmiyor; onlarin tuketicisi yerel paneldeki tablo ve istatistik
   sekmeleridir ve orada zaten kaynagindan okunuyorlar. */
const NAMES = ["I18N"];

const out = {};
for (const n of NAMES) {
  /* Set -> dizi: TABLE_GROUPS.keys `new Set([...])`, JSON'a dogrudan gecmez. */
  const val = eval("(" + literalOf(n) + ")");
  out[n] = JSON.parse(JSON.stringify(val, (k, v) => (v instanceof Set) ? [...v] : v));
}

const dst = path.join(__dirname, "sp", "meta.json");
fs.writeFileSync(dst, JSON.stringify(out, null, 1), "utf8");
const n = o => Array.isArray(o) ? o.length : Object.keys(o).length;
for (const lang of Object.keys(out.I18N))
  console.log(String(n(out.I18N[lang])).padStart(5) + "  I18N." + lang);
console.log("-> " + path.relative(ROOT, dst));
