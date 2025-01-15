NOTE DEVELOP OBJECT DETECTION, SPEED DAN JARAK ANTAR TRUCK

history :

1. mark.py = model basic dengan yolov11 untuk detect orang dan sebagainya, basic
2. mark1.py = model basic dengan yolov11 tidak dapat mendetect unit yang kita inginkan (belum custom data set) di tes dengan model video yang di berikan. GAGAL (X)
3. mark2.py = model sudah menggunakan costum model dari dataset custom yang sudah di training untuk object detection dan melakukan pengetasan pada model video yang diberikan. BERHASIL (V)
4. mark3.py = model sudah menggunakan custom model dari dataset custom yang sudah di training serta penambahan penghitungan jarak antar object berdasarkan size object actual dan jarak antar frame. BERHASIL (V)
5. mark4.py = model menggunakan versi yang kedua yang akurasi sedikit lebih tinggi serta sedikit melakukan tunning agar terdeteksi walaupun dengan tuning persentase sedikit rendah dalam machine untuk detection nya. Penambahan perhitungan jarak dan perhitungan speed. BERHASIL (V)
6. mark5.py = model menggunakan v2, akurasi sudah di tunning agar percentase unit rendah dan tetap terdeteksi, speed lebih smooth. BERHASIL (V)
7. mark6.py = model sudah di update dengan hanya menampilkan jarak jika dibawah 50 meter
8. mark7.py = menghilangkan track history agar tidak ada garis-garis jika sudah selesai terdetect

== NOTE ==
semua model sekarang namanya mark.py hanya update per commit

Note :

- pastikan sudah terinstall python pada machine yang ingin dijalankan
- install dependecy yang dibutuhkan sesuai pada requirements.txt
- jangan lupa rubah directory untuk model biar sesuai dan directory video agar sesuai juga

jalankan program :

- lakukan git clone untuk mengcopy project ke local
- pindah ke directory program local
- buat virtual environtment (python -m venv nama_venv_bebas)
- run di terminal (pip install -r requirements.txt)
- running tiap program berdasarkan kebutuhan masing-masing seperti mark - mark5


15-01-2025
- ada 3 file update baru
    :
        - check_gpu.py untuk check gpu yang tersedia
        - cpu.py computer vision jalan di cpu
        - gpu.py computer vision jalan di gpu