# Laporan Proyek *Machine Learning* - Sri Agung Tirtayasa

## Domain Proyek

Asuransi atau yang lebih dikenal dengan pertanggungan adalah sesuatu yang tidak asing lagi bagi masyarakat, sebagian besar masyarakat telah membuat perjanjian atau polis asuransi dengan perusahaan asuransi, baik milik Negara ataupun milik swasta. Namun masih banyak kesadaran untuk bebarapa masyarakat tentang asuransi terbilang masih rendah, karena kurang pahamnya masyarakat berkaitan dengan asuransi yang dianggap hanya membuang - buang uang dan tidak ada fungsinya.

Jika dilihat untuk masa depan peran asuransi sangatlah penting bagi masyarakat diantaranya seperti antisipasi untuk kejadian yang tidak terduga, untuk menyusun rencana masa depan, keamanan finansial dan melindungi keluarga serta orang tercinta. Dengan semua manfaat yang diberikan asuransi masyarakat akan merasa tenang pada masa depan keluarganya. [1]

Perusahaan asuransi pada tingkat yang berbeda mengadopsi pemodelan prediktif ke dalam praktik standar mereka, menjadikannya saat yang tepat untuk menyatukan pengalaman beberapa orang. Penting juga untuk memberikan pelajaran yang dipetik di industri dan aplikasi lain dan untuk mengidentifikasi area di mana para aktuaris dapat meningkatkan metode mereka. Karena ilmu data dan analitik prediktif berkembang pesat, tidak diragukan lagi akan ada peluang berkelanjutan untuk meningkatkan metodologi terkemuka saat ini, jadi salah satu tujuan diskusi ini untuk mengatasi masalah agar tetap tidak tertinggal. [2]

Oleh sebab itu diperlukan sebuah solusi untuk bisa memprediksi permohonan asuransi agar sebagai tolak ukur awal untuk menentukan apakah pelanggan layak mendapatkan asuransi. Untuk mencapai tingkat keberhasilan yang tinggi diperlukan data yang lengkap dan model yang optimal. Tentu saja penentuan persetujuan permohonan asuransi merupakan hal yang krusial bagi perusahaan. Model ini tidak sepenuhnya ditunjukan untuk _decision maker_ pada tahap akhir, tetapi bisa juga berguna sebagai _tools_ untuk *insight* awal yang memberikan hasil yang cepat.

## Business Understanding

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
  - Melakukan visualisasi data agar lebih mudah dipahami. Kali ini beberapa teknik visualisasi data yang akan digunakan yaitu *pie chart*, *bar chart*, *violin plot* dan *box plot*.
  - Melakukan analisis hubungan antara dua atau lebih fitur dalam dataset. Bisa dilakukan dengan *pair plot*, *correlation matrix*, dll.
- Untuk mengolah data atau preprocessing dengan cara melakukan:
  - Melakukan *handling null value*, dapat dilakukan dengan cara mengisi dengan nilai *mean*, *median* dll atau menghapusnya. Pendekatan yang akan dignakan sekarang yaitu menghapus data yang kosong/_null_.
  - Melakukan *encoding* terhadap fitur kategorikal, bisa menggunakan *One Hot Encoder* atau *Label Encoder*. Pendekatan yang akan digunakan sekarang yaitu menggunakan keduanya _One Hot Encoder_ dan _Label Encoder_ mengikuti struktur fitur yang ada pada dataset.
  - Mengatasi data tidak seimbang (*imbalance data*). Pendekatan yang akan dilakaukan yaitu dengan menggunakan teknik *oversampling* dengan metode SMOTE (*Synthetic Minority Oversampling Technique*).
  - Melakukan pembagian dataset untuk *train* dan *test* dengan rasio 7:3.
  - Malakukan *encoding* agar semua data dalam skala yang sama dengan menggunakan *Standard Scaler*.
- Pemilihan algoritma dilakukan dengan membandingkan menggukanan 4 algoritma klasifikasi:
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
- Untuk metode peningkatam model, teknik yang akan dilakukan:
  - *Hyperparameter tuning* untuk model, teknik yang dipakai menggunakan *GridSearchCV*.
  - Menganalisis fitur-fitur yang paling berdampak pada model, teknik yang dipakai menggunakan *RFECV* (*Recursive Feature Elimination with Cross Validation*).
 
Setalah model yang dibuat sudah cukup optimal, model sudah bisa melanjutkannya ke tahap implementasi. Perusahaan asuransi yang ingin menerapkan model prediksi ini untuk tahap awal atau tolak ukur awal ketika pelanggan melakukan pengajuan asuransi dan dapat melihat hasilnya. Tetapi model ini tidak harus dipakai untuk *tools* dalam *decision maker* di tahap akhir, melainkan digunakan untuk tahap awal sebagai *insight*. Beberapa perusahaan mungkin memiliki ketentuan-ketentuan tambahan dalam tahap persetujuan pengajuan asuransi.

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
- CREDIT_SCORE: merupakan seberapa besar kemungkinan pelanggan mengajukan klaim. Biasanya dihitung dari kelengkapan dokumen-dokumen yang diperlukan perusahaan.
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
Tabel 1. _Unique Value_ setiap fitur
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

Data ini memiliki outliers seperti yang ditunjukan pada _box plot_ di Gambar 1, tetapi _outliers_ tidak akan dihilangkan karena akan membuat beberapa fitur kehilangan value yang mengakibatkan beberapa fitur menjadi tidak berguna yang akan mempengaruhi tingkat akurasi model.

Pada diagram _violin plot_ bagian fitur ANNUAL_MILEAGE dan CREDIT_SCORE ada sedikit korelasi terhadap OUTCOME, terlihat dari perbedaan posisi cembungan. 

![Violin & Box Plot](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/violin%26box.png?raw=true)

Gambar 1. _Violin Plot_ & _Box Plot_

![Correlation Matrix](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/corrmatrix.png?raw=true)

Gambar 2. _Correlation Matrix_ setelah preprocessing data

Terlihat diagram kategorikal di Gambar 3 pada fitur AGE umur 16-25, DRIVING_EXPERIENCE 0-9 tahun pengalaman dan INCOME kategori _poverty_ berpeluang tinggi untuk dapat mengeklaim asuransi.

![Categorical Features](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/categorical.png?raw=true)

Gambar 3. Diagram Categorical Features

## Data Preparation
Teknik-teknik yang dilakukan untuk Data Preparation untuk model yang optimal.

- Pada dataset ini terdapat nilai-nilai yang kosong sebanyak 1851 baris. Untuk mengatasi ini, teknik yang akan digunakan yaitu dengan menghapus baris-baris yang memiliki nilai kosong karena simpel dan cepat.
- _LabelEncoding_: _Encode_ label target dengan nilai antara 0 dan _n_classes-1_. [3]
- _OneHotEncoding_: _Encode_ fitur kategorikal sebagai _array_ numerik _one-hot_. [3]
- _TrainTestSplit_: Membagi _array_ atau _matrix_ menjadi _subset train_ dan _test_ secara acak. [3]
- _Oversampling_: Salah satu pendekatan untuk mengatasi masalah ketidakseimbangan kelas adalah dengan sampel ulang / _resampling_ dataset pelatihan secara acak. Dua pendekatan utama untuk melakukan _resampling_ secara acak pada kumpulan data yang tidak seimbang adalah dengan menghapus data dari kelas mayoritas, yang disebut _undersampling_, dan menduplikasi data dari kelas minoritas, yang disebut _oversampling_. [4]
- _StandardScaler_: Standarisasi fitur dengan mengatur ulang _mean_ dan penskalaan varian satuan. [3]

## Modeling

Tabel 2. Hasil akurasi base model
|Model|Train Accuracy|Test Accuracy|
|-----|--------------|-------------|
|Logistic Regression|0.894933|0.848671|
|Random Forest Classifier|0.979552|0.833538|
|XGBoost|0.972351|0.832311|
|Support Vector Classifier|0.91088|0.847444|

Pada Tabel 2 menunjukan algoritma _Logistic Regression_ dan _SVC_ menghasilkan tingkat accuracy 84%. Algoritma _SVC_ menghasilkan accuracy yang lebih tinggi karena outliers berpengaruh sedikit pada model ini.

![Base model accuracies](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/basemodel.png?raw=true)

Gambar 4. Perbandingan hasil accuracy setiap model

_Model_ yang digunakan

_Models_
- _Logistic Regression_
  adalah teknik analisis data yang menggunakan matematika untuk menemukan hubungan antara dua faktor data. Kemudian menggunakan hubungan ini untuk memprediksi nilai salah satu faktor berdasarkan yang lain. Prediksi biasanya memiliki jumlah hasil yang terbatas, seperti ya atau tidak.

  _Logistic regression_ adalah model statistik yang menggunakan fungsi logistik, atau fungsi logit, dalam matematika sebagai persamaan antara x dan y. Fungsi logit memetakan y sebagai fungsi sigmoid dari x. $f(x)$ = $1 \over 1 + e^{-x}$ [5]
  - Kelebihan
    - Simple untuk diimplementasikan.
    - Terbukti sangat efisien ketika dataset memiliki fitur yang dapat dipisahkan secara linear.
    - Memungkinkan model diperbarui dengan mudah untuk merefleksikan data baru.
  - Kekurangan
    - Performa kurang baik untuk _non-linear_ data.
    - Performa buruk untuk fitur yang tidak relevan dan berkorelasi tinggi.
    - Ketergantungan tinggi pada penyajian data yang tepat.
  - Parameter
    - _random_state_: 200
- _Random Forest Classifier_
  adalah algoritma _machine learning_ yang fleksibel dan mudah digunakan yang menghasilkan, bahkan tanpa penyetelan _hyperparameter_, hasil yang bagus di sebagian besar waktu. Ini juga merupakan salah satu algoritma yang paling banyak digunakan, karena kesederhanaan dan keragamannya (dapat digunakan untuk tugas klasifikasi dan regresi). _Random Forest_ merupakan algoritma _supervised learning_. "Hutan/_Forest_" yang dibangun adalah _ensemble_/rombongan dari _decision tree_, biasanya dilatih dengan metode _bagging_. Gagasan utama metode _bagging_ adalah bahwa kombinasi model pembelajaran meningkatkan hasil keseluruhan. [6]
  - Kelebihan
    - Fleksibel untuk masalah klasifikasi dan regresi.
    - Mengurangi _overfitting_ di pohon keputusan dan membantu meningkatkan akurasi.
    - Bekerja dengan baik dengan nilai kategoris dan berkelanjutan
  - Kekurangan
    - Membutuhkan banyak daya komputasi serta sumber daya karena membangun banyak pohon untuk menggabungkan keluarannya.
    - Membutuhkan banyak waktu untuk pelatihan karena menggabungkan banyak pohon keputusan untuk menentukan kelas.
    - Interpretasi yang sulit dan membutuhkan mode penyetelan yang tepat untuk data.
  - Parameter
    - _n_estimators_: 50
    - _max_depth_: 15
    - _random_state_: 123
    - _n_jobs_: -1
- _XGBoost_
  (_Extreme Gradient Boosting_) adalah implementasi dari algoritma _gradient boosted trees_ yang _open-source_ yang populer dan efisien. _Gradient boosting_ adalah algoritma _supervised learning_, yang mencoba untuk memprediksi variabel target secara akurat dengan menggabungkan estimasi dari sekumpulan model yang lebih sederhana dan lebih lemah. [7]

  ![XGBoost explanation](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/xbg.png?raw=true)

  Gambar 5. Ilustrasi _XGBoost_

  - Kelebihan
    - Dirancang untuk pelatihan model yang efisien dan dapat *scalable*, sehingga cocok untuk kumpulan data besar.
    - Memiliki berbagai _hyperparameter_ yang dapat disesuaikan untuk mengoptimalkan kinerja, membuatnya sangat mudah disesuaikan.
    - Memiliki dukungan bawaan untuk menangani nilai yang hilang, membuatnya mudah untuk bekerja dengan data dunia nyata yang sering kali memiliki nilai yang hilang.
  - Kekurangan
    - _XGBoost_ bisa intensif secara komputasi, terutama saat melatih model besar, membuatnya kurang cocok untuk sistem dengan sumber daya terbatas.
    - Dapat rentan terhadap _overfitting_, terutama saat dilatih pada kumpulan data kecil atau saat terlalu banyak pohon yang digunakan dalam model.
    - Memiliki banyak hyperparameter yang dapat disesuaikan, sehingga penting untuk menyetel parameter dengan benar guna mengoptimalkan kinerja.
    - Dapat memakan banyak memori, terutama saat bekerja dengan kumpulan data besar, membuatnya kurang cocok untuk sistem dengan sumber daya memori terbatas.
- _Support Vector Classifier (SVC/SVM)_ adalah algoritmat _supervised learning_ yang bisa digunakan untuk regresi dan klasifikasi. Karena ketangguhannya, umumnya diterapkan untuk menyelesaikan tugas klasifikasi. Dalam algoritma ini, titik data pertama kali direpresentasikan dalam ruang n-dimensi. Algoritma kemudian menggunakan pendekatan statistik untuk menemukan garis terbaik yang memisahkan berbagai kelas yang ada dalam data.

    Jika titik data diplot dalam grafik 2 dimensi, maka batas keputusan disebut sebagai garis lurus. Namun, jika ada lebih dari dua dimensi, ini disebut sebagai _hyperplanes_. Meskipun mungkin ada beberapa _hyperplane_ yang memisahkan kelas, SVM memilih satu dengan jarak maksimum antar kelas. Scikit-Learn menyediakan dua pengklasifikasi lainnya — SVC() dan NuSVC() yang digunakan untuk tujuan klasifikasi. [8]
  - Kelebihan
    - Bekerja relatif baik ketika ada batas pemisahan yang jelas antara kelas.
    - Lebih efektif dalam ruang dimensi tinggi dan relatif hemat memori.
    - Efektif dalam kasus di mana dimensi lebih besar dari jumlah sampel.
  - Kekurangan
    - Tidak cocok untuk dataset besar.
    - Tidak bekerja dengan baik ketika kumpulan data memiliki lebih banyak noise.
    - Ketika kelas dalam data adalah titik yang tidak dipisahkan dengan baik, yang berarti ada kelas yang tumpang tindih, _SVC_ tidak bekerja dengan baik.
  - Parameter
    - _gamma_: 'auto'

_Parameters_
- _n_estimators_: Jumlah pohon/_tree_ di dalam hutan/_forest_. [3]
- _max_depth_: Kedalaman maksimum dari pohon/_tree_. [3]
- _random_state_: Mengontrol keacakan sampel. [3]
- _n_jobs_: Jumlah _jobs_ yang akan dijalankan secara paralel. Jika parameter bernilai ```-1``` maka semua proses berjalan secara paralel. [3]
- gamma: Yang menentukan jumlah kelengkungan dalam batas keputusan. Ini menentukan seberapa jauh pengaruh dari satu contoh pelatihan mencapai, dengan nilai rendah yang berarti 'jauh' dan nilai tinggi yang berarti 'dekat'. Jika ```gamma='scale'``` (_default_) maka akan menggunakan $1 \over (n_features * X.var())$ sebagai nilai gamma, jika ```gamma='auto'``` menggunakan value $1 \over n_features$ sebagai nilai gamma dan jika menginputkan nilai value _float_ harus nilai non negatif. [3]
- _criterion_: Fungsi untuk mengukur kualitas split. Kriteria yang didukung adalah ```'gini'```, ```'log_loss'``` dan ```'entropy'```. [3]
- _learning_rate_: Jumlah pengurangan error untuk mencegah _overfitting_ [4]
- _C_: Parameter regularisasi yang mengontrol _trade-off_ antara batas keputusan dan istilah misklasifikasi. Semakin tinggi nilai C, semakin sulit marginnya, dan semakin banyak titik data yang cenderung diklasifikasikan dengan benar. [3]
- _kernel_: Menentukan jenis kernel yang akan digunakan dalam algoritma. Jika tidak ada yang diberikan, ```'rbf'``` akan digunakan. Ada beberapa kernel standar, contohnya kernel linier, kernel polinomial, dan kernel radial. Pilihan kernel dan _hyperparameter_-nya sangat memengaruhi keterpisahan kelas (dalam klasifikasi) dan kinerja algoritma. [3]
- _max_iter_: Jumlah iterasi maksimal. [3]
- _cv_: Menentukan strategi _cross-validation splitting_. Jika inputan ```None``` maka akan menggunakan _5-fold_ _cross-validation_, jika inputan nilai integer untuk menentukan jumlah _fold_ dalam bentuk ```(Stratified)KFold```. [3]

_Hypermarameter tuning_ menggunakan _GridSearchCV_ dengan menginputkan daftar parameter dalam bentuk _array/list_. _GridSearchCV_ adalah fungsi yang hadir dalam paket _model_selection_ Scikit-learn (atau SK-learn).

- _Parameter tuning_ untuk model _Logistic Regression_:
  ```
  parameters = {
    'C': [0.25, 0.5, 0.75, 1],
    'random_state': [0],
    'max_iter': [200, 250, 300],
  }
  ```
  Hasil:
  ```
  ({'C': 0.5, 'max_iter': 200, 'random_state': 0}, 0.8807963365542779)
  ```
- _Parameter tuning_ untuk model _Random Forest Classifier_:
  ```
  parameters = {
    'max_depth': [10, 12, 13],
    'n_estimators': [100, 150, 200],
    'criterion': ['gini','entropy'],
    'random_state': [0]
  }
  ```
  Hasil:
  ```
  ({
  'criterion': 'gini',
  'max_depth': 10,
  'n_estimators': 100,
  'random_state': 0
  }, 0.8671553497942387)
   ```
- _Parameter tuning_ untuk model _XGBoost_:
  ```
  parameters = {
    "max_depth": [ 3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'eval_metric': ['error'],
  }
  ```
  Hasil:
  ```
  ({'eval_metric': 'error', 'learning_rate': 0.1, 'max_depth': 4}, 0.8806666446242737)
  ```
- _Parameter tuning_ untuk model _SVC_:
  ```
  parameters = {
    'C': [0.25, 0.5, 0.75, 1],
    'gamma': ['auto'],
    'kernel': ['linear','rbf']
  }
  ```
  Hasil:
  ```
  ({'C': 0.25, 'gamma': 'auto', 'kernel': 'linear'}, 0.8796390282611032)
  ```

Analisis fitur-fitur paling berdapkan pada model menggunakan _RFECV_.

_RFECV_ (_Recursive Feature Elimination with Cross Validation_) merupakan metode feature elimination yang bekerja secara rekursif mengeliminasi fitur dengan menggunakan Cross Validation juga untuk mencari fitur yang paling optimal. [3]

```
clf_rf_rfecv = RandomForestClassifier()
rfecv = RFECV(estimator=clf_rf_rfecv, step=1, cv=5, scoring='accuracy', verbose=2)
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_copy.columns[rfecv.support_])
```
Hasil:
```
Optimal number of features : 37
Best features : Index(['CREDIT_SCORE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS',
       'PAST_ACCIDENTS', 'AGE_16-25', 'AGE_26-39', 'AGE_40-64', 'AGE_65+',
       'GENDER_female', 'GENDER_male', 'DRIVING_EXPERIENCE_0-9y',
       'DRIVING_EXPERIENCE_10-19y', 'DRIVING_EXPERIENCE_20-29y',
       'DRIVING_EXPERIENCE_30y+', 'EDUCATION_high school', 'EDUCATION_none',
       'EDUCATION_university', 'INCOME_middle class', 'INCOME_poverty',
       'INCOME_upper class', 'INCOME_working class', 'VEHICLE_OWNERSHIP_0.0',
       'VEHICLE_OWNERSHIP_1.0', 'VEHICLE_YEAR_after 2015',
       'VEHICLE_YEAR_before 2015', 'MARRIED_0.0', 'MARRIED_1.0',
       'CHILDREN_0.0', 'CHILDREN_1.0', 'POSTAL_CODE_10238',
       'POSTAL_CODE_21217', 'POSTAL_CODE_32765', 'POSTAL_CODE_92101', 'DUIS_0',
       'DUIS_1', 'RACE_majority', 'RACE_minority'],
      dtype='object')
```

## Evaluation

- _Accuracy_
- _Precission_
- _Recall_
- _ROC AUC_
- _Confusion Matrix_
- _F1 Score_

_**Note**_:
- TP: *True Positive* adalah nilai positif yang diprediksi dengan benar
- TN: *True Negative* adalah nilai negatif yang diprediksi dengan benar
- FP: *False Positive* adalah nilai positif yang diprediksi dengan salah
- FN: *False Negative* adalah nilai negatif yang diprediksi dengan salah

### _Confusion Matrix_
_Confusion Matrix_ adalah petak informasi yang menunjukkan jumlah _True Positives_ [TP], _False Positives_ [FP], _True Negatives_ [TN], dan _False Negatives_ [FN] yang dikembalikan saat menerapkan kumpulan uji data ke algoritma klasifikasi. Dengan menggunakan _Confusion Matrix_ akan terlihat berapa kali model membuat prediksi yang benar dan salah. [9]

![Confision Matrix](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/cf.png?raw=true)

Gambar 6. _Confusion Matrix_
### _Accuracy_
_Accuracy_ adalah pengukuran yang digunakan untuk menentukan model mana yang paling baik dalam mengidentifikasi hubungan dan pola antarvariabel dalam set data berdasarkan input, atau data pelatihan. _Accuracy_ memberi tahu berapa persentase prediksi yang benar dari model tersebut. Akurasi tinggi bisa menyesatkan karena tidak menggambarkan bagaimana _True Positive_ dan _True Negative_ menyebar. [9]
```math
\begin{array}{rcl}
accuracy & = & \dfrac{TP + TN}{TP + FP + TN + FN}
\end{array}
```
### _Precission_
_Precission_ adalah persentase dari identifikasi positif yang dibuat oleh model yang benar. Dengan menggunakan _precission_, akan lebih mudah memahami berapa banyak gambar yang dikatakan berisi objek yang benar-benar berisi objek yang diidentifikasi oleh model. [9]
```math
\begin{array}{rcl}
precission & = & \dfrac{TP}{TP + FP}
\end{array}
```
### _Recall_
_Recall_ merupakan metrik yang umum digunakan untuk model klasifikasi, adalah pecahan positif yang diklasifikasikan dengan benar. _Recall_ juga disebut sebagai _"true positive rate"_, _"sensitivity"_ dan _"hit rate"_. [9]
```math
\begin{array}{rcl}
recall & = & \dfrac{TP}{TP + FN}
\end{array}
```
### _F1 Score_
_F1 score_ (juga dikenal sebagai _F-measure_, atau _balanced F-score_) adalah metrik yang digunakan untuk mengukur performa model _machine learning_ klasifikasi. Ini adalah metrik populer yang digunakan untuk model klasifikasi karena memberikan hasil yang kuat untuk kumpulan data seimbang dan tidak seimbang, tidak seperti akurasi. _F1 score_ adalah metrik _error_ yang mengukur kinerja model dengan menghitung rata-rata harmonik _precission_ dan _recall_ untuk kelas positif minoritas. [9]

![Acc, precission, recall, f1 score](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/cf2.png?raw=true)

Gambar 7. Ilustrasi _Confusion Matrix, Accuracy, Precission, Recall_ dan _F1 Score_

```math
\begin{array}{rcl}
F1 & = & \dfrac{2 * precission * recall}{precission + recall}
\end{array}
```
### _ROC AUC_
Kurva ROC-AUC adalah pengukuran kinerja untuk masalah klasifikasi pada berbagai pengaturan ambang batas. ROC adalah kurva probabilitas dan AUC mewakili tingkat atau ukuran keterpisahan. Ini memberi tahu seberapa banyak model mampu membedakan antar kelas. Semakin tinggi AUC, semakin baik model dalam memprediksi 0 kelas sebagai 0 dan 1 kelas sebagai 1. [9]

![AUC](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/auc.png?raw=true)

Gambar 8. Grafix AUC

Tabel 3. _Classification report_ untuk model _Logistic Regression_

| |Precission|Recall|F1 Score|Support|
|-|----------|------|--------|-------|
|0.0|0.89|0.89|0.89|1725|
|1.0|0.74|0.74|0.74|720|
|Accuracy|||0.85|2445|
|Marcro Avg|0.82|0.82|0.82|2445|
|Weighted Avg|0.85|0.85|0.85|2445|

![Confusion Matrix Logistic Regression](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/evallr.png?raw=true)
![AUC Logistic Regression](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/evallrauc.png?raw=true)

Gambar 9. _Confission Matrix_ dan Grafil _AUC_ untuk model _Logistic Regression_

Skor _AUC_ mencapai 0.918 dengan _False Positive_ 189 dan _False Negative_ 182.

Tabel 4. _Classification report_ untuk model _Random Forest Classifier_

| |Precission|Recall|F1 Score|Support|
|-|----------|------|--------|-------|
|0.0|0.90|0.87|0.89|1725|
|1.0|0.74|0.77|0.74|720|
|Accuracy|||0.84|2445|
|Marcro Avg|0.81|0.82|0.81|2445|
|Weighted Avg|0.85|0.84|0.84|2445|

![Confusion Matrix Random Forest Classifier](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/evalrfcm.png?raw=true)
![AUC Random Forest Classifier](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/evalrfauc.png?raw=true)

Gambar 10. _Confission Matrix_ dan Grafil _AUC_ untuk model _Random Forest Classifier_

Skor _AUC_ mencapai 0.908 dengan _False Positive_ 169 dan _False Negative_ 219.

Tabel 5. _Classification report_ untuk model _XGBoost_

| |Precission|Recall|F1 Score|Support|
|-|----------|------|--------|-------|
|0.0|0.90|0.88|0.89|1725|
|1.0|0.72|0.76|0.74|720|
|Accuracy|||0.84|2445|
|Marcro Avg|0.81|0.82|0.82|2445|
|Weighted Avg|0.85|0.84|0.85|2445|

![Confusion Matrix XGBoost](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/evalxgbcm.png?raw=true)
![AUC XGBoost](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/evalxgbauc.png?raw=true)

Gambar 11. _Confission Matrix_ dan Grafil _AUC_ untuk model _XGBoost_

Skor _AUC_ mencapai 0.915 dengan _False Positive_ 172 dan _False Negative_ 209.

Tabel 6. _Classification report_ untuk model _Support Vector Classifier_

| |Precission|Recall|F1 Score|Support|
|-|----------|------|--------|-------|
|0.0|0.89|0.89|0.89|1725|
|1.0|0.74|0.74|0.74|720|
|Accuracy|||0.85|2445|
|Marcro Avg|0.82|0.82|0.82|2445|
|Weighted Avg|0.85|0.85|0.85|2445|

![Confusion Matrix Support Vector Classifier](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/evalsvccm.png?raw=true)
![AUC Logistic Regression](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/evalsvcauc.png?raw=true)

Gambar 12. _Confission Matrix_ dan Grafil _AUC_ untuk model _Support Vector Classifier_

Skor _AUC_ mencapai 0.916 dengan _False Positive_ 187 dan _False Negative_ 184.

Tabel 7. Hasil akhir model setelah _hyperparameter tuning_

|Model|Accuracy|ROC AUC|
|-----|--------|-------|
|Logistic Regression|84.826176|0.918|
|Random Forest Classifier|84.130879|0.908|
|XGBoost|84.417178|0.915|
|Support Vector Classifier|84.826176|0.916|

![Final Accuracy](https://github.com/sagungt/predictive-analisys-car-insurance/blob/323d871e5ef9bfefd7769fa4a838bc0a1a1dda97/img/evalacc.png?raw=true)

Gambar 13. Perbandingan skor _Accuracy_ setelah _hyperparameter tuning_.

Setelah _hyperparameter tuning_, tetap model _Logistic Regression_ yang menghasilkan akurasi tertinggi dengan nilai 84% dan model yang lain mendapat peningkatan akurasi. Selain akurasi tinggi model ini menghasilkan nilai _ROC AUC_ tertinggi yaitu dengan nilai 0.918. Model ini sudah cukup bagus untuk bisa memprediksi pengajuan asuransi.

## Daftar Referensi

[1] I Wayan Agus Satriya Wedhana Putra. Ida Ayu Sukihana. "KEDUDUKAN AGEN ASURANSI DI ERA DIGITAL DALAM MENAWARKAN PRODUK ASURANSI". Jurnal Kertha Semaya, Vol. 8 No. 3 Tahun 2020, hlm. 350-367. [https://ojs.unud.ac.id/index.php/kerthasemaya/article/download/57420/33615](https://ojs.unud.ac.id/index.php/kerthasemaya/article/download/57420/33615) [Accessed July 8 2023]

[2] Eileen Burns, FSA, MAAA et al. "Considerations for Predictive Modeling in Insurance Applications". 2019 Society of Actuaries [https://www.soa.org/globalassets/assets/files/resources/research-report/2019/considerations-predictive-modeling.pdf](https://www.soa.org/globalassets/assets/files/resources/research-report/2019/considerations-predictive-modeling.pdf) [Accessed July 9 2023]

[3] Boisberranger. J. D, et al.. "Scikit Learn Documentations". [https://scikit-learn.org/stable](https://scikit-learn.org/stable) [Accessed July 10 2023]

[4] xgboost developer. "XGBoost Parameters". [https://xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html) [Accessed July 10 2023]

[5] Amazon Web Service. "What Is Logistic Regression?". [https://aws.amazon.com/what-is/logistic-regression/](https://aws.amazon.com/what-is/logistic-regression/) [Accessed July 10 2023]

[6] Mathew Urwin. "Random Forest: A Complete Guide for Machine Learning". [https://builtin.com/data-science/random-forest-algorithm](https://builtin.com/data-science/random-forest-algorithm) [Accessed July 9 2023]

[7] Amazon Web Service. "How XGBoost Works". [https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-HowItWorks.html](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-HowItWorks.html) [Accessed July 9 2023]

[8] Ashwin Raj. "Everything About Support Vector Classification — Above and Beyond". [https://towardsdatascience.com/everything-about-svm-classification-above-and-beyond-cc665bfd993e](https://towardsdatascience.com/everything-about-svm-classification-above-and-beyond-cc665bfd993e) [Accessed July 9 2023]

[9] Skyre Tran. "6 Useful Metrics to Evaluate Binary Classification Models". [https://towardsdatascience.com/6-useful-metrics-to-evaluate-binary-classification-models-55fd1fed6a20](https://towardsdatascience.com/6-useful-metrics-to-evaluate-binary-classification-models-55fd1fed6a20) [Accessed July 9 2023]
