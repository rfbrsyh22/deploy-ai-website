FROM python:3.9-slim

# Install dependensi sistem (untuk OpenCV dan Tesseract)
RUN apt-get update && apt-get install -y \    
    tesseract-ocr \
    tesseract-ocr-ind \
    && rm -rf /var/lib/apt/lists/*

# Set direktori kerja di dalam container
WORKDIR /app

# Copy file requirements terlebih dahulu untuk caching
COPY requirements.txt requirements.txt

# Install dependensi
RUN pip install --default-timeout=100000 --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy sisa kode aplikasi ke dalam container
COPY . .

# Buat folder uploads jika belum ada
RUN mkdir -p /app/uploads

# Expose port yang akan digunakan oleh Gunicorn
EXPOSE 8000

# Perintah untuk menjalankan aplikasi saat container start
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:8000", "index:app"]