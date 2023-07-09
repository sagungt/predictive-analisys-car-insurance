# Laporan Proyek *Machine Learning* - Sri Agung Tirtayasa

## Domain Proyek

Asuransi atau yang lebih dikenal dengan pertanggungan adalah sesuatu yang tidak asing lagi bagi masyarakat, sebagian besar masyarakat telah membuat perjanjian atau polis asuransi dengan perusahaan asuransi, baik milik Negara ataupun milik swasta. Namun masih banyak kesadaran untuk bebarapa masyarakat tentang asuransi terbilang masih rendah, karena kurang pahamnya masyarakat berkaitan dengan asuransi yang dianggap hanya membuang - buang uang dan tidak ada fungsinya.

Jika dilihat untuk masa depan peran asuransi sangatlah penting bagi masyarakat diantaranya seperti antisipasi untuk kejadian yang tidak terduga, untuk menyusun rencana masa depan, keamanan finansial dan melindungi keluarga serta orang tercinta. Dengan semua manfaat yang diberikan asuransi masyarakat akan merasa tenang pada masa depan keluarganya. [1]

Tidak selamanya praktek asuransi berjalan dengan baik. Dalam praktek ditemukan ditolaknya klaim asuransi tertanggung oleh penanggung setelah risiko, kerugian atau peristiwa yang tidak diinginkan terjadi. Tulisan ini bertujuan untuk mengetahui perlindungan hukum bagi tertanggung yang menghadapi penolakan klaim asuransidan mengetahui akibat hukum apabila pihak penanggung menolak klaim dari pihak tertanggung. [2]

Oleh sebab itu diperlukan sebuah solusi untuk bisa memprediksi permohonan asuransi agar sebagai tolak ukur awal untuk menentukan apakah pelanggan layak mendapatkan asuransi.

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
- Bagaimana cara memahami data ? 
- Bagaimana cara mengolah data asuransi agar bisa menghasilkan model yang baik ?
- Bagaimana membuat atau menentukan model yang terbaik untuk bisa memprediksi dengan akurasi tertinggi ?
- Bagaimana cara meningkatkan akurasi model ?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Untuk memahami data dengan cara melakukan *Exploratory Data Analisys*:
  - Memahami data setiap fitur, tipe data serta mengkategorikan apakah numerikal atau kategorikal.
  - Melakukan visualisasi data agar lebih mudah dipahami. Kali ini kita akan menggunakan beberapa visualisasi data yaitu *pie chart*, *bar chart*, *violin plot* dan *box plot*.
  - Melakukan analisis hubungan antara dua atau lebih fitur dalam dataset. Bisa dilakukan dengan *pair plot*, *correlation matrix*, dll.
- Untuk mengolah data atau preprocessing dengan cara melakukan:
  - Melakukan *handling null value*, dapat dilakukan dengan cara mengisi dengan nilai *mean*, *median* dll atau menghapusnya. Pada kali ini kita lakukan cara termudah dengan menghapus data yang null.
  - Melakukan *encoding* terhadap fitur kategorikal, bisa menggunakan *One Hot Encoder* atau *Label Encoder*. Pada kali ini kita menggunakan keduanya.
  - Mengatasi data tidak seimbang (*imbalance data*). Pada kali ini kita melakukan teknik *oversampling* dengan metode SMOTE (*Synthetic Minority Oversampling Technique*).
  - Melakukan pembagian dataset untuk *train* dan *test* dengan rasio 7:3.
  - Malakukan *encoding* agar semua data dalam skala yang sama dengan menggunakan *Standard Scaler*.
- Untuk menentukan model kita akan membandingkan menggukanan 4 algoritma klaasifikasi:
  - Melakukan prediksi dengan *base model* menggunakan 4 algoritmat yaitu *Logistic Regression*, *Random Forest Classifier*, *XGBoost* dan *Support Vector Classifier*.
  - Menggunakan beberapa metrik untuk tolak ukur model yaitu *accuracy*, *precission*, *recall*, *confusion matrix*, *F1 Score* dan *ROC AUC*.
  - Formula untuk *accuracy*
    ```math
    \begin{array}{rcl}
    accuracy & = & \dfrac{TP + TN}{TP + FP + TN + FN}
    \end{array}
    ```
  - Formula untuk *precission*
    ```math
    \begin{array}{rcl}
    precission & = & \dfrac{TP}{TP + FP}
    \end{array}
    ```
  - Formula untuk *recall*
    ```math
    \begin{array}{rcl}
    recall & = & \dfrac{TP}{TP + FN}
    \end{array}
    ```
  - Formula untuk *F1 score*
    ```math
    \begin{array}{rcl}
    F1 & = & \dfrac{2 * precission * recall}{precission + recall}
    \end{array}
    ```
- Untuk metode peningkatam model, kita akan melakukan:
  - *Hyperparameter tuning* untuk model, teknik yang dipakai menggunakan *GridSearchSV*.
  - Menganalisis fitur-fitur yang paling berdampak pada model, teknik yang dipakai menggunakan *RFECV* (*Recursive Feature Elimination with Cross Validation*).
 
Setalah model yang dibuat sudah cukup optimal, kita sudah bisa melanjutkannya ke tahap implementasi. Perusahaan asuransi yang ingin menerapkan model prediksi ini untuk tahap awal atau tolak ukur awal ketika pelanggan melakukan pengajuan asuransi dan dapat melihat hasilnya. Tetapi model ini tidak harus dipakai untuk *tools* dalam *decision maker* di tahap akhir, melainkan digunakan untuk tahap awal sebagai *insight*. Beberapa perusahaan mungkin memiliki ketentuan-ketentuan tambahan dalam tahap persetujuan pengajuan asuransi.

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

Pada diagram violin plot bagian fitur ANNUAL_MILEAGE dan CREDIT_SCORE ada sedikit korelasi terhadap OUTCOME, terlihat dari perbedaan posisi cembungan. 

Gambar 1. Violin Plot & Box Plot

Gambar 2. Correlation Matrix setelah preprocessing data

Terlihat diagram kategorikal di Gambar 3 pada fitur AGE umur 16-25, DRIVING_EXPERIENCE 0-9 tahun pengalaman dan INCOME kategori poverty berpeluang tinggi untuk dapat mengeklaim asuransi.

Gambar 3. Diagram Categorical Features

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Teknik-teknik yang dilakukan untuk Data Preparation untuk model yang optimal.

- Pada dataset ini terdapat nilai-nilai yang kosong sebanyak 1851 baris. Untuk mengatasi ini kita lakukan cara termudah yaitu dengan menghapus baris-baris yang memiliki nilai kosong.
- LabelEncoding
- OneHotEncoding
- TrainTestSplit
- Oversampling
- StandardScaler

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

Tabel 2. Hasil akurasi base model
|Model|Train Accuracy|Test Accuracy|
|-----|--------------|-------------|
|Logistic Regression|0.894933|0.848671|
|Random Forest Classifier|0.979552|0.833538|
|XGBoost|0.972351|0.832311|
|Support Vector Classifier|0.91088|0.847444|

Pada Tabel 2 menunjukan algoritma Logistic Regression dan SVC menghasilkan tingkat accuracy 84%. Algoritma SVC menghasilkan accuracy yang lebih tinggi karena outliers berpengaruh sedikit pada model ini.

Gambar 4. Perbandingan hasil accuracy setiap model

Model yang digunakan

Parameters
- n_estimators: 
- max_depth: 
- random_state: 
- n_jobs: 
- gamma: 
- criterion: 
- learning_rate: 
- C: 
- kernel: 
- max_iter: 

Models
- Logistic Regression
  - Kelebihan
    - Simple untuk diimplementasikan.
    - Terbukti sangat efisien ketika dataset memiliki fitur yang dapat dipisahkan secara linear.
    - Memungkinkan model diperbarui dengan mudah untuk merefleksikan data baru.
  - Kekurangan
    - Performa kurang baik untuk non-linear data.
    - Performa buruk untuk fitur yang tidak relevan dan berkorelasi tinggi.
    - Ketergantungan tinggi pada penyajian data yang tepat.
- Random Forest Classifier
  - Kelebihan
    - Fleksibel untuk masalah klasifikasi dan regresi.
    - Mengurangi overfitting di pohon keputusan dan membantu meningkatkan akurasi.
    - bekerja dengan baik dengan nilai kategoris dan berkelanjutan
  - Kekurangan
    - Membutuhkan banyak daya komputasi serta sumber daya karena membangun banyak pohon untuk menggabungkan keluarannya.
    - Membutuhkan banyak waktu untuk pelatihan karena menggabungkan banyak pohon keputusan untuk menentukan kelas.
    - Interpretasi yang sulit dan membutuhkan mode penyetelan yang tepat untuk data.
- XGBoost
  - Kelebihan
    - Dirancang untuk pelatihan model yang efisien dan dapat *scalable*, sehingga cocok untuk kumpulan data besar.
    - Memiliki berbagai hyperparameter yang dapat disesuaikan untuk mengoptimalkan kinerja, membuatnya sangat mudah disesuaikan.
    - Memiliki dukungan bawaan untuk menangani nilai yang hilang, membuatnya mudah untuk bekerja dengan data dunia nyata yang sering kali memiliki nilai yang hilang.
  - Kekurangan
    - XGBoost bisa intensif secara komputasi, terutama saat melatih model besar, membuatnya kurang cocok untuk sistem dengan sumber daya terbatas.
    - Dapat rentan terhadap overfitting, terutama saat dilatih pada kumpulan data kecil atau saat terlalu banyak pohon yang digunakan dalam model.
    - Memiliki banyak hyperparameter yang dapat disesuaikan, sehingga penting untuk menyetel parameter dengan benar guna mengoptimalkan kinerja.
    - Dapat memakan banyak memori, terutama saat bekerja dengan kumpulan data besar, membuatnya kurang cocok untuk sistem dengan sumber daya memori terbatas.
- Support Vector Classifier (SVC)
  - Kelebihan
    - Bekerja relatif baik ketika ada batas pemisahan yang jelas antara kelas.
    - Lebih efektif dalam ruang dimensi tinggi dan relatif hemat memori.
    - Efektif dalam kasus di mana dimensi lebih besar dari jumlah sampel.
  - Kekurangan
    - Tidak cocok untuk dataset besar.
    - Tidak bekerja dengan baik ketika kumpulan data memiliki lebih banyak noise.
    - Ketika kelas dalam data adalah titik yang tidak dipisahkan dengan baik, yang berarti ada kelas yang tumpang tindih, SVM tidak bekerja dengan baik.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

- Accuracy
- Precission
- Recall
- ROC AUC
- Confussion Matrix
- F1 Score

### Accuracy

### Precission

### Recall

### ROC AUC

### Confussion Matrix

### F1 Score

Tabel 3. Classification report untuk model Logistic Regression

| |Precission|Recall|F1 Score|Support|
|-|----------|------|--------|-------|
|0.0|0.89|0.89|0.89|1725|
|1.0|0.74|0.74|0.74|720|
|Accuracy|||0.85|2445|
|Marcro Avg|0.82|0.82|0.82|2445|
|Weighted Avg|0.85|0.85|0.85|2445|

Tabel 4. Classification report untuk model Random Forest Classifier

| |Precission|Recall|F1 Score|Support|
|-|----------|------|--------|-------|
|0.0|0.90|0.87|0.89|1725|
|1.0|0.74|0.77|0.74|720|
|Accuracy|||0.84|2445|
|Marcro Avg|0.81|0.82|0.81|2445|
|Weighted Avg|0.85|0.84|0.84|2445|

Tabel 5. Classification report untuk model XGBoost

| |Precission|Recall|F1 Score|Support|
|-|----------|------|--------|-------|
|0.0|0.90|0.88|0.89|1725|
|1.0|0.72|0.76|0.74|720|
|Accuracy|||0.84|2445|
|Marcro Avg|0.81|0.82|0.82|2445|
|Weighted Avg|0.85|0.84|0.85|2445|

Tabel 6. Classification report untuk model Support Vector Classifier

| |Precission|Recall|F1 Score|Support|
|-|----------|------|--------|-------|
|0.0|0.89|0.89|0.89|1725|
|1.0|0.74|0.74|0.74|720|
|Accuracy|||0.85|2445|
|Marcro Avg|0.82|0.82|0.82|2445|
|Weighted Avg|0.85|0.85|0.85|2445|

Tabel 7. Hasil akhir model setelah hyperparameter tuning

|Model|Accuracy|ROC AUC|
|-----|--------|-------|
|Logistic Regression|84.826176|0.918|
|Random Forest Classifier|84.130879|0.908|
|XGBoost|84.417178|0.915|
|Support Vector Classifier|84.826176|0.916|

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
