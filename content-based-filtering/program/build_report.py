#!/usr/bin/env python3
"""
Membangun report.html dari report_source.txt.

Markdown dipakai sebagai sumber tunggal agar laporan mudah dirawat dan
tidak bergantung pada penulisan tag HTML secara manual.

Jalankan dari folder program/:
    python3 build_report.py

Lalu buat PDF dengan Chrome/Chromium:
    google-chrome --headless=new --no-sandbox --disable-gpu \
      --no-pdf-header-footer --print-to-pdf=../report.pdf ../report.html
"""

from __future__ import annotations

from pathlib import Path

import markdown

COVER = """
<section class="cover">
  <div class="cover-top">
    <div class="cover-rule"></div>
    <div class="cover-kicker">Tugas Mandiri 3 - Kajian Content-based Filtering</div>
    <h1 class="cover-title">
      Content-Based Filtering dengan Word Embedding dan Sentence Embedding
    </h1>
    <p class="cover-subtitle">
      Laporan tugas mandiri
    </p>
  </div>
  <div class="cover-meta">
    <div><span>Disusun oleh</span><b>Najib Robbani</b></div>
    <div><span>NIM</span><b>103042580006</b></div>
    <div><span>Tanggal</span><b>Oktober 2026</b></div>
  </div>
</section>
"""

ABSTRACT_AND_TOC = """
<section>
  <div class="abstract">
    <h2>Abstrak</h2>
    <p>
      Content-based filtering (CBF) merekomendasikan item berdasarkan kemiripan
      antara profil pengguna dan fitur item. Pada domain teks, kualitas
      rekomendasi sangat ditentukan oleh cara deskripsi item direpresentasikan.
      Kajian ini membahas dua pendekatan representasi yang banyak digunakan:
      <strong>word embedding</strong>, yang memetakan setiap kata ke vektor padat
      berdasarkan konteks pemakaiannya, dan <strong>sentence embedding</strong>,
      yang memetakan satu kalimat utuh ke vektor yang menangkap makna dan
      komposisinya. Kajian dilengkapi implementasi sederhana pada 12 deskripsi
      film. Word embedding dibangun dari matriks co-occurrence, PPMI, dan SVD;
      sentence embedding dihitung dengan mean pooling dan dibandingkan dengan
      Sentence-BERT. Hasil demonstrasi menunjukkan bahwa ketiga metode
      (TF-IDF, word embedding, dan Sentence-BERT) sama-sama menempatkan
      <em>The Martian</em> sebagai rekomendasi teratas untuk query
      <em>Interstellar</em>, tetapi urutan berikutnya berbeda. TF-IDF dan mean
      pooling masih dapat tertipu oleh kata yang sama pada konteks berbeda,
      sedangkan Sentence-BERT menghasilkan peringkat yang lebih koheren secara
      semantik. Kajian ini menyimpulkan bahwa pemilihan representasi adalah
      trade-off antara kualitas semantik, biaya komputasi, dan kemudahan
      interpretasi.
    </p>
    <p class="keywords">
      <strong>Kata kunci:</strong> content-based filtering, word embedding,
      sentence embedding, Word2Vec, Sentence-BERT, cosine similarity
    </p>
  </div>

  <h2>Daftar Isi</h2>
  <div class="toc">
    <ol>
      <li><span class="toc-num">1</span> Pendahuluan</li>
      <li><span class="toc-num">2</span> Content-Based Filtering</li>
      <li><span class="toc-num">3</span> Word Embedding</li>
      <li><span class="toc-num">4</span> Sentence Embedding</li>
      <li><span class="toc-num">5</span> Perbandingan dan Pembahasan</li>
      <li><span class="toc-num">6</span> Kesimpulan</li>
      <li><span class="toc-num">—</span> Referensi</li>
      <li><span class="toc-num">A</span> Lampiran A: Struktur dan Cara Menjalankan Program</li>
      <li><span class="toc-num">B</span> Lampiran B: Ringkasan Output Program</li>
    </ol>
  </div>

  <div class="note blue" style="margin-top: 1.2rem">
    <strong>Cara membaca laporan ini.</strong> Bagian 2 menjelaskan kerangka CBF
    dan baseline TF-IDF. Bagian 3 dan 4 membahas word embedding dan sentence
    embedding, masing-masing dengan contoh sederhana. Bagian 5 membandingkan
    hasil, dan Bagian 6 merangkum. Program pendukung serta demo interaktif dapat
    diakses melalui repositori yang tercantum pada halaman sampul.
  </div>
</section>
"""

TEMPLATE = """<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Content-Based Filtering dengan Word Embedding dan Sentence Embedding</title>
  <link rel="stylesheet" href="assets/report.css">
</head>
<body>
{cover}
{front_matter}
{body}
</body>
</html>
"""


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    source = here / "report_source.txt"
    if not source.exists():
        raise SystemExit(f"Sumber tidak ditemukan: {source}")

    text = source.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["extra", "sane_lists"],
        output_format="html5",
    )

    # Hanya lampiran yang dimulai di halaman baru. Bagian utama dibuat mengalir
    # agar laporan ringkas dan tidak penuh halaman kosong.
    body = body.replace(
        '<h2>Lampiran',
        '<div class="page-break"></div>\n<h2>Lampiran',
        1,
    )

    html = TEMPLATE.format(
        cover=COVER,
        front_matter="",
        body=body,
    )
    output = root / "report.html"
    output.write_text(html, encoding="utf-8")
    print(f"Ditulis: {output}")
    print(f"Ukuran : {len(html):,} karakter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
