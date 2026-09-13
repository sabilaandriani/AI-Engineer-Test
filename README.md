# Technical Test - Junior AI Engineer (PEFT/QLoRA)

Repository ini berisi simulasi proses fine-tuning Large Language Model (LLM) menggunakan teknik PEFT dan QLoRA. Model dilatih menggunakan dataset regulasi kesehatan, spesifiknya **Peraturan Menteri Kesehatan Nomor 10 Tahun 2024 tentang JDIH**

## Persyaratan Sistem
Script ini dirancang untuk berjalan di lingkungan dengan VRAM terbatas (dioptimalkan untuk **Google Colab dengan GPU T4**).

## Struktur File
- `technical_test_ai_engineer.ipynb` : Notebook utama yang berisi seluruh script (Preprocessing, Fine-Tuning, dan Inference).
- `permenkes-no-10-tahun-2024.pdf` : Raw data dokumen regulasi kesehatan (opsional, jika ingin mencoba ekstraksi ulang).
- `dataset_permenkes.jsonl` : Hasil ekstraksi PDF menjadi format instruction-tuning.
- `requirements.txt` : Daftar library dependencies

## Cara Menjalankan Script
1. Buka [Google Colab](https://colab.research.google.com/).
2. Upload file `technical_test_ai_engineer.ipynb` ke Colab.
3. Pastikan runtime sudah di-set ke GPU: **Runtime > Change runtime type > Hardware accelerator: T4 GPU**.
4. Upload file `permenkes-no-10-tahun-2024.pdf` ke session storage Colab (jika ingin menjalankan preprocessing dari awal).
5. Jalankan cell secara berurutan dari atas ke bawah:
   - **Tahap 1:** Ekstraksi PDF ke JSONL
   - **Tahap 2:** Setup QLoRA & Proses Fine-Tuning (Pastikan menggunakan `fp16=False` pada `SFTConfig` untuk kompatibilitas GPU T4).
   - **Tahap 3:** Pengujian Inference Model

## Catatan Teknis
Dikarenakan keterbatasan arsitektur GPU T4 pada Colab versi gratis yang tidak mendukung `BFloat16` secara native, script telah disesuaikan dengan menggunakan `torch.float16` dan mematikan `fp16` pada `SFTConfig` untuk menghindari error `GradScaler`
