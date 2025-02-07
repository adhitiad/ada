# Dokumentasi API Pipeline RAG

API ini menyediakan akses ke pipeline Retrieval Augmented Generation (RAG), memungkinkan Anda untuk memuat data, menjalankan query, mengoptimalkan pipeline, dan mengukur kinerjanya.

## URL Dasar

`http://127.0.0.1:8000`

## Endpoint

### 1. Memuat Data

- **Endpoint:** `/load`
- **Metode:** `GET`
- **Deskripsi:** Memuat data dari file CSV yang dikonfigurasi ke dalam pipeline RAG. Ini menginisialisasi model bahasa, database vektor, dan rantai QA. Jika data sudah dimuat, akan mengembalikan pesan yang menunjukkan hal tersebut.
- **Body Permintaan:** Tidak ada
- **Respons:**
  - **Kode Status:** `201` (Created) - Data berhasil dimuat.
  - **Kode Status:** `200` (OK) - Data sudah dimuat sebelumnya.
  - **Kode Status:** `500` (Internal Server Error) - Terjadi kesalahan saat memuat data. Respons akan menyertakan pesan kesalahan.
  - **Contoh:**

    {
    "message": "Data berhasil dimuat"
    }

### 2. Menjalankan Query

- **Endpoint:** `/query`
- **Metode:** `POST`
- **Deskripsi:** Menjalankan query melalui pipeline RAG.
- **Body Permintaan:**

  {
  "query": "Query Anda di sini"
  }

- **Respons:**
  - **Kode Status:** `200` (OK) - Query berhasil. Respons menyertakan hasil query.
  - **Kode Status:** `400` (Bad Request) - Parameter "query" tidak ada.
  - **Kode Status:** `500` (Internal Server Error) - Terjadi kesalahan saat menjalankan query.
  - **Contoh:**

    {
    "result": "Jawaban untuk query Anda"
    }

### 3. Mengoptimalkan Pipeline

- **Endpoint:** `/optimize`
- **Metode:** `POST`
- **Deskripsi:** Mengoptimalkan pipeline RAG (detail implementasi tidak ditentukan dalam kode yang disediakan). Secara opsional menerima daftar query untuk pengujian setelah optimasi.
- **Body Permintaan:**

  {
  "queries": ["query1", "query2", "query3"] // Opsional
  }

- **Respons:**
  - **Kode Status:** `200` (OK) - Optimasi berhasil.
  - **Kode Status:** `500` (Internal Server Error) - Terjadi kesalahan saat optimasi.
  - **Contoh:**

    {
    "message": "Pipeline berhasil dioptimalkan"
    }

### 4. Benchmark Pipeline

- **Endpoint:** `/benchmark`
- **Metode:** `POST`
- **Deskripsi:** Melakukan benchmark waktu eksekusi query pipeline RAG menggunakan daftar query yang disediakan.
- **Body Permintaan:**

  {
  "queries": ["query1", "query2", "query3"]
  }

- **Respons:**
  - **Kode Status:** `200` (OK) - Benchmark berhasil. Respons menyertakan hasil benchmark.
  - **Kode Status:** `400` (Bad Request) - Parameter "queries" tidak ada atau daftar kosong.
  - **Kode Status:** `500` (Internal Server Error) - Terjadi kesalahan saat benchmark.
  - **Contoh:**

    {
    "results": [
    { "query": "query1", "time": 0.123 },
    { "query": "query2", "time": 0.456 },
    { "query": "query3", "time": 0.789 }
    ]
    }

### 5. Mematikan Server

- **Endpoint:** `/shutdown`
- **Metode:** `POST`
- **Deskripsi:** Mematikan server dengan baik, menutup semua koneksi yang terbuka (seperti Redis).
- **Body Permintaan:** Tidak ada
- **Respons:**
  - **Kode Status:** `200` (OK) - Server sedang dimatikan.

## Penanganan Kesalahan

Sebagian besar endpoint mengembalikan objek JSON dengan kunci "error" yang berisi pesan kesalahan jika terjadi kegagalan.

## Konfigurasi

Path file CSV dan kolom konten dikonfigurasi dalam file `app.py`. Pastikan untuk memperbarui nilai-nilai ini sebelum menjalankan aplikasi.
