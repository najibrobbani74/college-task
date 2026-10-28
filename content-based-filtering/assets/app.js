/* Demo interaktif content-based filtering.
 * Data berasal dari assets/demo-data.js yang dihasilkan oleh
 * program/cbf_embeddings.py.
 */
(function () {
  "use strict";

  const data = window.DEMO_DATA;
  if (!data) {
    document.body.innerHTML =
      '<main class="container section"><div class="card card-pad">Data demo tidak ditemukan. Jalankan <code>program/cbf_embeddings.py</code> terlebih dahulu.</div></main>';
    return;
  }

  const STOPWORDS = new Set(
    (
      "a an the and or but if while when where who whom which what why how " +
      "in on at to of for from with by as is are was were be been being " +
      "that this these those it its their his her they he she we you i " +
      "not no do does did have has had will would can could may might must " +
      "into over after before under above about than then there here"
    ).split(/\s+/)
  );

  const state = {
    word: "astronaut",
    queryId: "interstellar",
    method: "word_mean",
    customText: "",
  };

  const $ = (selector, root) => (root || document).querySelector(selector);
  const $$ = (selector, root) => Array.from((root || document).querySelectorAll(selector));

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function normalizeToken(word) {
    if (["mars", "news", "series", "species", "debris", "angeles"].includes(word)) {
      return word;
    }
    if (word.length > 4 && word.endsWith("ies")) return word.slice(0, -3) + "y";
    if (word.length > 4 && word.endsWith("ses")) return word.slice(0, -2);
    if (word.length > 3 && word.endsWith("s") && !word.endsWith("ss")) {
      return word.slice(0, -1);
    }
    return word;
  }

  function tokenize(text) {
    const words = String(text).toLowerCase().match(/[a-z]+/g) || [];
    return words
      .filter((word) => !STOPWORDS.has(word) && word.length > 2)
      .map(normalizeToken);
  }

  function dot(a, b) {
    let total = 0;
    for (let i = 0; i < a.length; i += 1) total += a[i] * b[i];
    return total;
  }

  function norm(a) {
    return Math.sqrt(dot(a, a));
  }

  function normalizeDense(a) {
    const length = norm(a);
    return length > 1e-12 ? a.map((value) => value / length) : a.slice();
  }

  function cosineDense(a, b) {
    const denominator = norm(a) * norm(b);
    return denominator > 1e-12 ? dot(a, b) / denominator : 0;
  }

  function cosineSparse(a, b) {
    let product = 0;
    for (const key of Object.keys(a)) {
      if (b[key]) product += a[key] * b[key];
    }
    let normA = 0;
    let normB = 0;
    for (const value of Object.values(a)) normA += value * value;
    for (const value of Object.values(b)) normB += value * value;
    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator > 1e-12 ? product / denominator : 0;
  }

  function sparseNorm(a) {
    return Math.sqrt(Object.values(a).reduce((sum, value) => sum + value * value, 0));
  }

  function sparseDot(a, b) {
    let product = 0;
    for (const key of Object.keys(a)) {
      if (b[key]) product += a[key] * b[key];
    }
    return product;
  }

  function sparseToDense(a, keys) {
    return keys.map((key) => a[key] || 0);
  }

  // -----------------------------------------------------------------------
  // TF-IDF (baseline leksikal)
  // -----------------------------------------------------------------------

  const idf = data.tfidf && data.tfidf.idf ? data.tfidf.idf : {};
  const tfidfItemVectors = new Map();

  function tfidfVector(text, shouldNormalize) {
    const tokens = tokenize(text);
    const counts = new Map();
    for (const token of tokens) counts.set(token, (counts.get(token) || 0) + 1);
    const vector = {};
    for (const [token, count] of counts) {
      if (idf[token] !== undefined) vector[token] = count * idf[token];
    }
    if (shouldNormalize === false) return vector;
    const length = sparseNorm(vector);
    if (length > 1e-12) {
      for (const key of Object.keys(vector)) vector[key] /= length;
    }
    return vector;
  }

  function itemTfidf(movie) {
    if (!tfidfItemVectors.has(movie.id)) {
      tfidfItemVectors.set(movie.id, tfidfVector(movie.description, true));
    }
    return tfidfItemVectors.get(movie.id);
  }

  // -----------------------------------------------------------------------
  // Word embedding + mean pooling
  // -----------------------------------------------------------------------

  function meanVectorRaw(text) {
    const dim = data.meta.word_dim;
    const words = tokenize(text).filter((word) => data.wordVectors[word]);
    const vector = new Array(dim).fill(0);
    if (!words.length) return null;
    for (const word of words) {
      const wordVector = data.wordVectors[word];
      for (let i = 0; i < dim; i += 1) vector[i] += wordVector[i];
    }
    return vector.map((value) => value / words.length);
  }

  function meanVector(text) {
    const vector = meanVectorRaw(text);
    return vector ? normalizeDense(vector) : null;
  }

  // -----------------------------------------------------------------------
  // Sentence-BERT (vektor sudah dinormalisasi oleh model)
  // -----------------------------------------------------------------------

  function sbertItem(movie) {
    return data.sbert && data.sbert.itemVectors ? data.sbert.itemVectors[movie.id] : null;
  }

  function sbertPreset(preset) {
    return data.sbert && data.sbert.presetVectors
      ? data.sbert.presetVectors[preset.id]
      : null;
  }

  // -----------------------------------------------------------------------
  // Query helpers
  // -----------------------------------------------------------------------

  function movieById(id) {
    return data.movies.find((movie) => movie.id === id) || null;
  }

  function presetById(id) {
    return data.queryPresets.find((preset) => preset.id === id) || null;
  }

  function currentQuery() {
    if (state.queryId === "__custom__") {
      return {
        type: "custom",
        text: state.customText,
        label: "Teks bebas",
      };
    }
    const preset = presetById(state.queryId);
    if (preset) {
      return {
        type: "preset",
        id: preset.id,
        text: preset.text,
        label: preset.label,
      };
    }
    const movie = movieById(state.queryId);
    if (movie) {
      return {
        type: "movie",
        id: movie.id,
        text: movie.description,
        label: movie.title,
        movie,
      };
    }
    return { type: "custom", text: "", label: "Teks bebas" };
  }

  function queryVector(method, shouldNormalize) {
    const query = currentQuery();
    if (method === "tfidf") {
      return tfidfVector(query.text, shouldNormalize);
    }
    if (method === "word_mean") {
      if (query.type === "movie" && shouldNormalize !== false) {
        return data.wordMean.itemVectors[query.id];
      }
      const raw = meanVectorRaw(query.text);
      if (!raw) return null;
      return shouldNormalize === false ? raw : normalizeDense(raw);
    }
    if (method === "sbert") {
      if (!data.sbert) return null;
      if (query.type === "movie") return data.sbert.itemVectors[query.id];
      if (query.type === "preset") return sbertPreset(query);
      return null;
    }
    return null;
  }

  function candidateVector(method, movie, shouldNormalize) {
    if (method === "tfidf") {
      if (shouldNormalize === false) return tfidfVector(movie.description, false);
      return itemTfidf(movie);
    }
    if (method === "word_mean") {
      const normalized = data.wordMean.itemVectors[movie.id];
      if (shouldNormalize !== false) return normalized;
      const raw = meanVectorRaw(movie.description);
      return raw ? raw : normalized;
    }
    if (method === "sbert") return sbertItem(movie);
    return null;
  }

  function similarity(method, a, b) {
    if (!a || !b) return 0;
    if (method === "tfidf") return cosineSparse(a, b);
    return cosineDense(a, b);
  }

  function rank(method, topN) {
    const query = currentQuery();
    const q = queryVector(method, true);
    if (!q) return [];
    const rows = [];
    for (const movie of data.movies) {
      if (query.type === "movie" && movie.id === query.id) continue;
      const d = candidateVector(method, movie, true);
      if (!d) continue;
      rows.push({ movie, score: similarity(method, q, d) });
    }
    rows.sort((a, b) => b.score - a.score);
    return rows.slice(0, topN || 5);
  }

  function sharedGenres(query, candidate) {
    if (!query || !query.genres || !candidate.genres) return [];
    return query.genres.filter((genre) => candidate.genres.includes(genre));
  }

  function overlappingTokens(queryText, candidateText, knownOnly) {
    const queryTokens = new Set(
      tokenize(queryText).filter((token) => !knownOnly || data.wordVectors[token])
    );
    const candidateTokens = new Set(
      tokenize(candidateText).filter((token) => !knownOnly || data.wordVectors[token])
    );
    return Array.from(queryTokens)
      .filter((token) => candidateTokens.has(token))
      .slice(0, 4);
  }

  // -----------------------------------------------------------------------
  // Rendering: word explorer
  // -----------------------------------------------------------------------

  function wordNeighbors(word) {
    const normalized = word.trim().toLowerCase();
    if (!normalized) return { word: "", rows: [], known: false };

    if (data.wordNeighbors[normalized]) {
      return {
        word: normalized,
        known: true,
        rows: data.wordNeighbors[normalized],
      };
    }

    const vector = data.wordVectors[normalized];
    if (!vector) return { word: normalized, known: false, rows: [] };

    const rows = Object.entries(data.wordVectors)
      .filter(([candidate]) => candidate !== normalized)
      .map(([candidate, candidateVector]) => ({
        word: candidate,
        score: cosineDense(vector, candidateVector),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 8);

    return { word: normalized, known: true, rows };
  }

  function renderWordExplorer() {
    const result = wordNeighbors(state.word);
    const list = $("#neighbor-list");
    const preview = $("#word-vector-preview");
    const status = $("#word-status");

    $$(".pill[data-word]").forEach((pill) => {
      pill.classList.toggle("active", pill.dataset.word === result.word);
    });

    if (!result.known) {
      const suggestions = Object.keys(data.wordVectors).slice(0, 10).join(", ");
      status.textContent = `Kata "${result.word || "-"}" tidak ada di kosakata demo. Coba salah satu: ${suggestions}.`;
      status.className = "note warm";
      list.innerHTML = "";
      preview.textContent = "";
      return;
    }

    status.className = "hint";
    status.textContent = `Vektor ${result.word} (${data.meta.word_dim} dimensi) dibandingkan dengan ${Object.keys(data.wordVectors).length} kata lain menggunakan cosine similarity.`;
    list.innerHTML = result.rows
      .map(
        (row) => `
        <div class="neighbor">
          <span class="neighbor-word">${escapeHtml(row.word)}</span>
          <span class="neighbor-track"><span class="neighbor-fill" style="--score:${Math.max(0, Math.min(1, row.score)).toFixed(4)}"></span></span>
          <span class="neighbor-score">${row.score.toFixed(3)}</span>
        </div>`
      )
      .join("");

    const vector = data.wordVectors[result.word].slice(0, 10);
    preview.innerHTML = `<b>${escapeHtml(result.word)}</b> = [ ${vector
      .map((value) => (value >= 0 ? "+" : "") + value.toFixed(3))
      .join(", ")} , ... ]<br><span>10 dari ${data.meta.word_dim} dimensi pertama</span>`;
  }

  // -----------------------------------------------------------------------
  // Rendering: recommender
  // -----------------------------------------------------------------------

  function methodLabel(method) {
    return {
      tfidf: "TF-IDF",
      word_mean: "Word Embedding (mean pooling)",
      sbert: "Sentence-BERT",
    }[method] || method;
  }

  function reasonFor(method, query, movie) {
    if (method === "sbert") {
      const genres = sharedGenres(query.movie, movie);
      return genres.length
        ? `Kemiripan makna; genre sama: ${genres.join(", ")}`
        : "Kemiripan makna dari vektor kalimat";
    }
    const genres = sharedGenres(query.movie, movie);
    if (genres.length) return `Genre sama: ${genres.join(", ")}`;
    const overlap = overlappingTokens(
      query.text,
      movie.description,
      method === "word_mean"
    );
    if (overlap.length) return `Kata beririsan: ${overlap.join(", ")}`;
    return "Tidak ada kata/genre yang sama; skor dari kedekatan vektor";
  }

  function renderResults(rows) {
    const container = $("#recommendation-list");
    if (!rows.length) {
      container.innerHTML =
        '<div class="note warm">Query tidak memiliki kata yang dikenal. Coba teks lain atau pilih judul film.</div>';
      return;
    }
    container.innerHTML = rows
      .map(
        (row, index) => `
        <div class="result-card">
          <span class="rank">${index + 1}</span>
          <div>
            <div class="result-title">${escapeHtml(row.movie.title)} <span class="small">(${row.movie.year})</span></div>
            <div class="result-meta">${escapeHtml(row.movie.genres.join(" · "))}</div>
            <div class="result-reason">${escapeHtml(reasonFor(state.method, currentQuery(), row.movie))}</div>
          </div>
          <div class="score-wrap">
            <div class="score-number">${row.score.toFixed(3)}</div>
            <div class="score-track"><div class="score-fill" style="--score:${Math.max(0, Math.min(1, row.score)).toFixed(4)}"></div></div>
          </div>
        </div>`
      )
      .join("");
  }

  function renderComparison() {
    const query = currentQuery();
    const methods = ["tfidf", "word_mean", "sbert"];
    const rankings = {};
    for (const method of methods) {
      if (method === "sbert" && (!data.sbert || (query.type === "custom" && !query.text))) {
        rankings[method] = [];
      } else if (method === "sbert" && query.type === "custom") {
        rankings[method] = [];
      } else {
        rankings[method] = rank(method, 3);
      }
    }

    const body = $("#comparison-body");
    const rows = [];
    for (let index = 0; index < 3; index += 1) {
      const cells = methods.map((method) => {
        const row = rankings[method][index];
        return row
          ? `<td><strong>${escapeHtml(row.movie.title)}</strong><br><span class="small">${row.score.toFixed(3)}</span></td>`
          : '<td class="small">—</td>';
      });
      rows.push(`<tr><td>${index + 1}</td>${cells.join("")}</tr>`);
    }
    body.innerHTML = rows.join("");

    const note = $("#comparison-note");
    if (query.type === "custom") {
      note.className = "note";
      note.textContent =
        "Untuk teks bebas, Sentence-BERT tidak dihitung di browser (model 384 dimensi tidak dimuat ke halaman). Kolom SBERT ditampilkan kosong; jalankan program Python untuk menguji teks tersebut dengan SBERT.";
    } else if (!data.sbert) {
      note.className = "note warm";
      note.textContent =
        "Sentence-BERT tidak tersedia pada data demo ini. Jalankan program Python dengan paket sentence-transformers untuk mengaktifkannya.";
    } else {
      note.className = "note good";
      note.textContent =
        "TF-IDF hanya melihat kata yang sama; Word Embedding memakai kedekatan makna kata; Sentence-BERT memakai makna seluruh kalimat. Perbedaan peringkat menunjukkan trade-off masing-masing representasi.";
    }
  }

  function renderCosineExample(rows) {
    const query = currentQuery();
    const panel = $("#cosine-panel");
    if (!rows.length) {
      panel.innerHTML = '<p class="small">Tidak ada pasangan vektor untuk dihitung.</p>';
      return;
    }

    const candidate = rows[0].movie;
    const method = state.method;
    if (method === "sbert") {
      const q = queryVector("sbert", true);
      const d = candidateVector("sbert", candidate, true);
      if (!q || !d) {
        panel.innerHTML = '<p class="small">Vektor SBERT tidak tersedia.</p>';
        return;
      }
      const dotValue = dot(q, d);
      const normA = norm(q);
      const normB = norm(d);
      panel.innerHTML = cosineMarkup(query.label, candidate.title, dotValue, normA, normB, dotValue / (normA * normB), "SBERT (sudah L2-normalized)");
      return;
    }

    if (method === "tfidf") {
      const q = queryVector("tfidf", false);
      const d = candidateVector("tfidf", candidate, false);
      const dotValue = sparseDot(q, d);
      const normA = sparseNorm(q);
      const normB = sparseNorm(d);
      panel.innerHTML = cosineMarkup(query.label, candidate.title, dotValue, normA, normB, dotValue / (normA * normB), "TF-IDF (sebelum normalisasi)");
      return;
    }

    const q = queryVector("word_mean", false);
    const d = candidateVector("word_mean", candidate, false);
    if (!q || !d) {
      panel.innerHTML = '<p class="small">Query tidak memiliki kata yang dikenal.</p>';
      return;
    }
    const dotValue = dot(q, d);
    const normA = norm(q);
    const normB = norm(d);
    panel.innerHTML = cosineMarkup(query.label, candidate.title, dotValue, normA, normB, dotValue / (normA * normB), "Word Embedding (sebelum normalisasi)");
  }

  function cosineMarkup(queryLabel, candidateTitle, dotValue, normA, normB, cosine, methodNote) {
    return `
      <p class="small">Pasangan: <strong>${escapeHtml(queryLabel)}</strong> vs <strong>${escapeHtml(candidateTitle)}</strong> · ${escapeHtml(methodNote)}</p>
      <div class="formula">cos(q, d) = (q · d) / (||q|| × ||d||)</div>
      <div class="calc-grid">
        <div class="calc-item"><span>q · d</span><b>${dotValue.toFixed(4)}</b></div>
        <div class="calc-item"><span>||q||</span><b>${normA.toFixed(4)}</b></div>
        <div class="calc-item"><span>||d||</span><b>${normB.toFixed(4)}</b></div>
        <div class="calc-item"><span>cosine</span><b>${cosine.toFixed(4)}</b></div>
      </div>
      <p class="hint">Nilai cosine mendekati 1 berarti vektor query dan item makin searah. Untuk peringkat, vektor dinormalisasi L2 terlebih dahulu; contoh di atas memakai vektor sebelum normalisasi agar angka lebih mudah dibaca.</p>`;
  }

  function renderVectorPreview(rows) {
    const panel = $("#vector-preview");
    const query = currentQuery();
    if (!rows.length) {
      panel.textContent = "";
      return;
    }
    const candidate = rows[0].movie;
    const method = state.method;

    if (method === "tfidf") {
      const q = queryVector("tfidf", false) || {};
      const d = candidateVector("tfidf", candidate, false) || {};
      const terms = Object.entries(q)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([term, weight]) => {
          const inCandidate = d[term] ? " · ada di kandidat" : "";
          return `${escapeHtml(term)} <b>${weight.toFixed(3)}</b>${inCandidate}`;
        });
      panel.innerHTML = `<b>Term dengan bobot TF-IDF tertinggi pada query</b><br>${terms.join("<br>")}`;
      return;
    }

    const q = queryVector(method, false);
    const d = candidateVector(method, candidate, false);
    if (!q || !d) {
      panel.textContent = "";
      return;
    }
    const dims = Math.min(8, q.length);
    const rowsMarkup = [];
    for (let i = 0; i < dims; i += 1) {
      const value = q[i];
      rowsMarkup.push(`d${i + 1} <b>${value >= 0 ? "+" : ""}${value.toFixed(3)}</b>`);
    }
    panel.innerHTML = `<b>${method === "sbert" ? "Vektor SBERT" : "Vektor sentence embedding"} query</b> (${dims} dimensi pertama)<br>${rowsMarkup.join(" · ")}<br><span>${escapeHtml(query.label)}</span>`;
  }

  function render() {
    renderWordExplorer();
    const rows = rank(state.method, 5);
    renderResults(rows);
    renderComparison();
    renderCosineExample(rows);
    renderVectorPreview(rows);

    const methodNames = {
      tfidf: "TF-IDF",
      word_mean: "Word Embedding + mean pooling",
      sbert: "Sentence-BERT",
    };
    $("#result-heading").textContent = `${methodNames[state.method]} · ${currentQuery().label}`;

    $$('input[name="method"]').forEach((input) => {
      input.checked = input.value === state.method;
    });
    $$(".method-option").forEach((label) => {
      const input = $("input", label);
      const disabled = input.value === "sbert" && (!data.sbert || state.queryId === "__custom__");
      input.disabled = disabled;
      label.classList.toggle("disabled", disabled);
    });

    const customBox = $("#custom-query-box");
    customBox.hidden = state.queryId !== "__custom__";
    const sbertDisabled = $("#sbert-disabled-note");
    sbertDisabled.hidden = !(state.queryId === "__custom__" || !data.sbert);
  }

  // -----------------------------------------------------------------------
  // Events
  // -----------------------------------------------------------------------

  function init() {
    const wordPills = $("#word-pills");
    const words = Object.keys(data.wordNeighbors);
    wordPills.innerHTML = words
      .map(
        (word) =>
          `<button class="pill${word === state.word ? " active" : ""}" type="button" data-word="${escapeHtml(word)}">${escapeHtml(word)}</button>`
      )
      .join("");

    wordPills.addEventListener("click", (event) => {
      const pill = event.target.closest("[data-word]");
      if (!pill) return;
      state.word = pill.dataset.word;
      $("#word-input").value = state.word;
      renderWordExplorer();
    });

    $("#word-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const value = $("#word-input").value.trim().toLowerCase();
      if (!value) return;
      state.word = value;
      renderWordExplorer();
    });

    const querySelect = $("#query-select");
    const movieOptions = data.movies
      .map((movie) => `<option value="${movie.id}">${escapeHtml(movie.title)} (${movie.year})</option>`)
      .join("");
    const presetOptions = data.queryPresets
      .map((preset) => `<option value="${preset.id}">Teks: ${escapeHtml(preset.label)}</option>`)
      .join("");
    querySelect.innerHTML =
      `<optgroup label="Deskripsi film">${movieOptions}</optgroup>` +
      `<optgroup label="Contoh teks bebas">${presetOptions}</optgroup>` +
      '<option value="__custom__">Teks bebas (ketik sendiri)</option>';
    querySelect.value = state.queryId;

    querySelect.addEventListener("change", () => {
      state.queryId = querySelect.value;
      if (state.queryId === "__custom__" && state.method === "sbert") {
        state.method = "word_mean";
      }
      render();
    });

    $("#custom-query").addEventListener("input", (event) => {
      state.customText = event.target.value;
      if (state.queryId === "__custom__") render();
    });

    $$('input[name="method"]').forEach((input) => {
      input.addEventListener("change", () => {
        if (input.checked) {
          state.method = input.value;
          render();
        }
      });
    });

    $("#reset-demo").addEventListener("click", () => {
      state.word = "astronaut";
      state.queryId = "interstellar";
      state.method = "word_mean";
      state.customText = "";
      $("#word-input").value = "";
      $("#custom-query").value = "";
      querySelect.value = state.queryId;
      render();
    });

    render();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
