# SmartSplit Bill AI: Prototype Pembagi Bill Cerdas

## 1. Deskripsi Proyek
[cite_start]Proyek ini adalah prototipe aplikasi web sederhana berbasis Streamlit yang berfungsi untuk membaca gambar nota/bill, mengekstrak detail transaksi menggunakan Model AI (KIE), dan memungkinkan pengguna untuk membagi item serta menghitung total tagihan per orang. [cite: 18, 20]

## 2. Cara Install/Menjalankan Code Streamlit

### A. Persyaratan (Prerequisites)
* Python 3.8+
* Git
* Akun GitHub

### B. Persiapan Lingkungan
1.  **Clone Repository:**
    ```bash
    git clone [https://github.com/USERNAME/SmartSplit-Bill-AI.git](https://github.com/USERNAME/SmartSplit-Bill-AI.git)
    cd SmartSplit-Bill-AI
    ```
2.  **Buat Virtual Environment (Opsional, Disarankan):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # atau
    .\venv\Scripts\activate   # Windows
    ```
3.  **Install Dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

### C. Menjalankan Aplikasi
Jalankan aplikasi Streamlit dari terminal:
```bash
streamlit run app.py
