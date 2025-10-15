---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Profil

Halo! 👋

Saya **Willy Chairullah Fauzi Putra**, mahasiswa aktif di **Teknik Informatika, Universitas Trunojoyo Madura**. Saya memiliki ketertarikan besar pada dunia teknologi, khususnya di bidang **pengembangan web** dan **analisis data** . Selain belajar di kampus, saya juga aktif mengembangkan skill melalui berbagai proyek dan komunitas.

> _"Belajar bukan hanya tentang mendapatkan nilai, tapi juga tentang membangun pemahaman dan pengalaman yang bermanfaat untuk masa depan."_

## Informasi & Minat

```{admonition} Informasi Pribadi
:class: info

**Nama**  : Willy Chairullah Fauzi Putra<br>
**NPM**   : 220411100045<br>
**Email** : willy.chairullah@gmail.com
```

::::{grid}
:gutter: 2
:class-container: text-center

:::{grid-item-card} Web Dev
{bdg-success}`Next Js`
:::

:::{grid-item-card} Analisis data 
{bdg-secondary}`NLP`
:::
::::

<br>

# Web Mining

## Pengantar Web Mining

Web mining adalah proses mengekstrak informasi menggunakan teknik data mining untuk mengekstraksi pola berharga dari sumber yang berhabitat di internet. Disiplin ini terbagi ke dalam tiga lapangan utama: penggalian konten (web content mining), penggalian struktur (web structure mining), dan penggalian penggunaan (web usage mining). 

```{figure} ./images/Web-mining-taxonomy.png
---
scale: 60%
align: center
---
Web mining taxonomy
```

### Web Crawling:

Langkah awal adalah Web Crawling, yaitu proses pengambilan data dari sebuah halaman web secara otomatis. ini biasanya menggunakan alat yang dibuat dengan bahasa pemrograman seperti Python. Web crawler perlu mengikuti tautan dari satu halaman ke halaman lain supaya informasi yang diperoleh lebih banyak dan saling terhubung, membentuk data yang lebih terstruktur. 

### Web Data Preprocessing:

Setelah data dikumpulkan, data itu masih mentah alias berantakan. Oleh karena itu, kita perlu melakukan preprocessing. Tujuannya adalah untuk membersihkan dan merapikan data agar siap diproses oleh mesin, misalnya dengan menghapus tag HTML yang tidak relevan, menangani data yang hilang, atau menyamakan format data.

### Web Structure Mining:

Fokus pada bagaimana halaman-halaman web saling terhubung melalui link. ini sangat berguna karena situs yang banyak ditautkan menjadi referensi dan memiliki urutan teratas dalam hasil SEO.

### Web Content Mining:

Menambang konten dari web, yaitu teks, gambar, video, dan audio. Dalam teks, kita bisa mengekstrak informasi spesifik seperti tempat kejadian, tanggal, dan orang yang terlibat menggunakan teknik Named-Entity Recognition (NER).

### Web Usage Mining:

Fokus pada analisis perilaku pengguna, bukan kontennya manfaatnya sangat banyak, seperti untuk memberikan rekomendasi produk, menempatkan konten populer di halaman utama, dan membuat kategori berita yang sering dibaca. Selain itu, teknik ini juga bisa digunakan untuk optimasi desain situs dan personalisasi konten.

### Pembelajaran Terawasi (Supervised Learning):

Membutuhkan data yang sudah dilabeli. Pelabelan ini harus dilakukan secara manual terlebih dahulu, misalnya untuk mengklasifikasikan sentimen sebuah komentar.

### Pembelajaran Tak Terawasi (Unsupervised Learning):

Tidak memerlukan data berlabel. Clustering adalah salah satu metodenya untuk mengelompokkan data yang mirip. Metode lain termasuk association rule mining dan dimensionality reduction untuk menemukan pola tersembunyi.

### Deployment System:

Tahap terakhir adalah Deployment System. Ini adalah proses di mana model yang sudah dilatih diubah menjadi sistem yang bisa digunakan secara nyata oleh pengguna atau perusahaan. Seperti yang sudah kita diskusikan, ini mengubah model yang hanya berupa kode menjadi sistem fungsional yang bisa diakses dan memberikan manfaat secara langsung.
