CONTENT-BASED FILTERING: WORD EMBEDDING & SENTENCE EMBEDDING
============================================================

PROGRAM
-------
cbf_embeddings.py
  Program demonstrasi content-based filtering pada 12 deskripsi film.
  Program menghasilkan tiga representasi teks:
    1. TF-IDF (baseline leksikal)
    2. Word embedding statis: co-occurrence -> PPMI -> SVD -> mean pooling
    3. Sentence-BERT: all-MiniLM-L6-v2 (opsional)

  Output:
    - ringkasan di terminal dan output.txt
    - assets/demo-data.json dan assets/demo-data.js untuk halaman demo

build_report.py
  Membangun report.html dari report_source.txt (Markdown).
  Laporan PDF dibuat dari report.html menggunakan Chrome/Chromium.

CARA MENJALANKAN
----------------
1. Instal dependensi dasar:
     pip install numpy scikit-learn

2. (Opsional) Instal Sentence-BERT:
     pip install sentence-transformers

3. Jalankan program:
     python3 cbf_embeddings.py

4. Untuk membangun ulang laporan HTML:
     python3 build_report.py

5. Untuk membuat PDF (dari folder program/):
     google-chrome --headless=new --no-sandbox --disable-gpu \
       --no-pdf-header-footer --print-to-pdf=../report.pdf ../report.html

CATATAN
-------
- Word embedding pada demo dilatih dari korpus sangat kecil. Ia bukan
  Word2Vec produksi; tujuannya adalah memperlihatkan prinsip co-occurrence,
  PPMI, dan SVD secara transparan.
- Sentence-BERT memakai model pretrained. Pada eksekusi pertama, model
  diunduh dari Hugging Face dan disimpan di cache lokal.
- Hasil dapat berbeda sedikit jika versi pustaka atau model berubah.
- Halaman demo: https://najibrobbani74.github.io/college-task/content-based-filtering/
- Repositori  : https://github.com/najibrobbani74/college-task
