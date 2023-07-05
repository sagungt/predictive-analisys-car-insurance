# Laporan Proyek *Machine Learning* - Sri Agung Tirtayasa

## Domain Proyek

Pada bagian ini, kamu perlu menuliskan latar belakang yang relevan dengan proyek yang diangkat.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
  
  Format Referensi: [Judul Referensi](https://scholar.google.com/) 

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Data asuransi mobil yang digunakan berasal dari Kaggle, data ini merupakan riwayat pengajuan asuransi ke sebuah perusahaan di suatu daerah, data ini berisi 18 fitur tentang informasi *client* asuransi dari perusahaan tersebut. Link Dataset: [Car Insurance Data](https://www.kaggle.com/datasets/sagnik1511/car-insurance-data).

### Variabel-variabel pada Car Insurance dataset adalah sebagai berikut:
- ID : merupakan ID dari riwayat permohonan asuransi.
- AGE : merupakan umur dari pelanggan.
- GENDER: merupakan jenis kelamin pelanggan.
- RACE: merupakan ras pelanggan. Hanya dibagi menjadi mayoritas dan minoritas.
- DRIVING_EXPERIENCE: merupakan jumlah pengalaman berkendara.
- EDUCATION: merupakan pendidikan terakhir pelanggan.
- INCOME: merupakan pendapatan pelanggan. Dibagi menjadi 4 kategori yaitu *upper class*, *middle class*, *poverty* dan *working class*.
- CREDIT_SCORE: merupakan seberapa besar kemungkinan Anda mengajukan klaim. Biasanya dihitung dari kelengkapan dokumen-dokumen yang diperlukan perusahaan.
- VEHICLE_YEAR: merupakan tahun keluaran mobil. Dibedakan hanya sebelum dan setelah 2015.
- MARRIED: merupakan status apakah pelanggan sudah menikah.
- CHILDREN: merupakan apakah pelanggan sudah memiliki anak.
- POSTAL_CODE: merupakan kode pos daerah.
- ANNUAL_MILEAGE: merupakan jarak tempuh tahunan.
- VEHICLE_TYPE: merupakan jenis mobil. Hanya ada 2 kategori yaitu *sports car* dan *sedan*.
- SPEEDING_VIOLATIONS: merupakan jumlah pelanggaran melewati kecepatan batas maksimal.
- DUIS: merupakan jumlah pelanggaran berkendara dibawah pengaruh.
- PAST_ACCIDENTS: merupakan jumlah kecelakaan yang telah dialami pelanggan.
- OUTCOME: merupakan hasil dari pengajuan asuransi mobil.

Jumlah baris data yaitu 10000 dengan ada beberapa *missing value* pada fitur 'CREDIT_SCORE' dan 'ANNUAL_MILEAGE' dengan total sebanyak 1851 baris. Teknik yang digunakan untuk mengatasi *missing value* sekarang adalah dengan cara termudah yaitu menghapus data yang kosong.

Setalah mengatasi *missing value* data target cukup tidak seimbang (*imbalance*) dengan sebaran data:
- Asuransi yang disetujui sebesar 68.9%
- Asuransi yang tidak disetujui sebesar 31.1%

Sebagian besar fitur merupakan data kategorikal setelah diketahui jumlah *unique value* pada setiap fitur.
Tabel 1. Unique Value setiap fitur
|Feature|Unique Value|
|-------|------------|
|AGE|4|
|GENDER|2|
|RACE|2|
|DRIVING_EXPERIENCE|4|
|EDUCATION|3|
|INCOME|4|
|CREDIT_SCORE|8149|
|VEHICLE_OWNERSHIP|2|
|VEHICLE_YEAR|2|
|MARRIED|2|
|CHILDREN|2|
|POSTAL_CODE|4|
|ANNUAL_MILEAGE|21|
|VEHICLE_TYPE|2|
|SPEEDING_VIOLATIONS|21|
|DUIS|7|
|PAST_ACCIDENTS|15|
|OUTCOME|2|

Pembagian numerikal fitur yaitu *nunique >= 15*
- Numerical: 'SPEEDING_VIOLATIONS', 'ANNUAL_MILEAGE', 'PAST_ACCIDENTS', 'CREDIT_SCORE'
- Categorical: 'AGE', 'GENDER', 'DRIVING_EXPERIENCE', 'EDUCATION', 'INCOME', 'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR', 'MARRIED', 'CHILDREN', 'POSTAL_CODE', 'VEHICLE_TYPE', 'DUIS', 'RACE'

Data ini memiliki outliers seperti yang ditunjukan pada box plot di Gambar 1, tetapi outliers tidak akan dihilangkan karena akan membuat beberapa fitur kehilangan value yang mengakibatkan beberapa fitur menjadi tidak berguna yang akan mempengaruhi tingkat akurasi model.

Gambar 1. Violin Plot & Box Plot

Gambar 2. Correlation Matrix

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
