# ======= IMPORT LIBRARY =======
import pymupdf
import re
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import streamlit as st
from functools import lru_cache
import pandas as pd
import matplotlib.patches as mpatches
# import spacy
import nltk
from nltk.corpus import stopwords

# ======= FUNGSI DASAR =======

def extract_text_from_pdf(pdf_file):
    """Ekstrak teks dari PDF baik dari file path atau file upload Streamlit"""
    if hasattr(pdf_file, "read"):  # Streamlit upload (BytesIO)
        file_bytes = pdf_file.read()
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    else:  # Path string
        doc = pymupdf.open(pdf_file)

    text = ""
    for page in doc:
        text += page.get_text()
    return text


def filter_paragraphs(text):
    """Filter hanya paragraf dari teks, buang header/footer/metadata/tabel/daftar pustaka"""
    lines = text.split('\n')
    filtered_lines = []
    
    # Flag untuk menandai apakah sudah menemukan abstract atau pendahuluan
    content_started = False
    
    # Deteksi pola abstract atau pendahuluan (mulai konten)
    start_patterns = [
        r'abstract\s*:?',
        r'abstrak\s*:?',
        r'ringkasan\s*:?',
        r'pendahuluan',
        r'introduction',
        r'latar\s+belakang',
        r'hasil\s+dan\s+pembahasan',
        r'metod(e|ologi)',
    ]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()
        
        # Cek apakah konten sudah dimulai
        if not content_started:
            for pattern in start_patterns:
                if re.search(pattern, line_stripped.lower()):
                    content_started = True
                    break
            
            # Skip semua baris sebelum konten dimulai
            if not content_started:
                i += 1
                continue
        
        # Deteksi daftar pustaka - STOP memproses setelah ini
        bibliography_patterns = [
            r'^daftar\s+pustaka\s*$',
            r'^references\s*$',
            r'^bibliography\s*$',
            r'^rujukan\s*$',
            r'^referensi\s*$',
        ]
        
        for pattern in bibliography_patterns:
            if re.search(pattern, line_stripped.lower()):
                # Jika ditemukan daftar pustaka, langsung keluar dari loop
                return '\n'.join(filtered_lines)
        
        # Skip baris kosong tapi simpan untuk whitespace
        if not line_stripped:
            filtered_lines.append('')
            i += 1
            continue
        
        # Hitung rasio whitespace dalam baris
        if len(line_stripped) > 0:
            whitespace_ratio = (len(line) - len(line_stripped)) / len(line)
        else:
            whitespace_ratio = 0
        
        # Kriteria BUKAN paragraf (skip):
        # 1. Baris terlalu pendek (< 40 karakter)
        if len(line_stripped) < 40:
            i += 1
            continue
            
        # 2. Baris dengan banyak angka (kemungkinan tabel/metadata)
        digit_count = sum(c.isdigit() for c in line_stripped)
        if digit_count / len(line_stripped) > 0.3:
            i += 1
            continue
        
        # 3. Baris dengan pola metadata jurnal
        metadata_patterns = [
            r'^(abstract|key\s*words?|artikel|received|revised|published|doi|issn|volume|issue)',
            r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}',  # tanggal
            r'^hal(aman)?\s*[:.]?\s*\d+',  # halaman
            r'^vol(ume)?\s*\.?\s*\d+',  # volume
            r'^no\s*\.?\s*\d+',  # nomor
            r'^(tabel|table|grafik|gambar|figure)\s+\d+',  # caption tabel/gambar
            r'@[a-z]+\.(com|ac\.id)',  # email
            r'^sumber\s*:',  # sumber tabel
        ]
        
        is_metadata = False
        for pattern in metadata_patterns:
            if re.search(pattern, line_stripped.lower()):
                is_metadata = True
                break
        
        if is_metadata:
            i += 1
            continue
        
        # 4. Baris dengan terlalu banyak huruf kapital (kemungkinan judul/header)
        upper_count = sum(c.isupper() for c in line_stripped)
        if len(line_stripped) > 0 and upper_count / len(line_stripped) > 0.5:
            i += 1
            continue
        
        # 5. Baris yang memiliki whitespace berlebihan di tengah
        # (kemungkinan header dengan spacing aneh)
        words = line_stripped.split()
        if len(words) > 2:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len < 4 and len(words) < 10:  # kata-kata pendek tersebar
                i += 1
                continue
        
        # Kriteria ADALAH paragraf:
        # 1. Mengandung tanda baca kalimat lengkap
        sentence_markers = ['.', ',', ';', ':', '(', ')']
        has_punctuation = any(marker in line_stripped for marker in sentence_markers)
        
        # 2. Panjang cukup dan mengandung kata-kata
        word_count = len(words)
        
        # Jika memenuhi kriteria paragraf
        if has_punctuation and word_count >= 5:
            # Coba gabung dengan baris berikutnya jika masih satu paragraf
            paragraph = line_stripped
            j = i + 1
            
            while j < len(lines):
                next_line = lines[j].strip()
                
                # Berhenti jika baris kosong (akhir paragraf)
                if not next_line:
                    break
                
                # Berhenti jika baris berikutnya seperti header
                if len(next_line) < 40:
                    break
                
                # Gabungkan jika masih satu paragraf
                if has_punctuation or len(next_line) > 40:
                    paragraph += " " + next_line
                    j += 1
                else:
                    break
            
            filtered_lines.append(paragraph)
            filtered_lines.append('')  # tambah baris kosong untuk spacing
            i = j
        else:
            i += 1
    
    return '\n'.join(filtered_lines)


def clean_text(text):
    """Bersihkan teks: lowercase, hapus simbol non-alphabet, hapus spasi ganda"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


@st.cache_resource
def get_nlp_tools():
    """Inisialisasi stopword dan stemmer hanya sekali"""
    stop_factory = StopWordRemoverFactory()
    stop_words = set(stop_factory.get_stop_words())
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    return stop_words, stemmer


@lru_cache(maxsize=None)
def cached_stem(word):
    """Stemming cepat dengan cache"""
    stop_words, stemmer = get_nlp_tools()
    return stemmer.stem(word)

# Muat model spaCy global (biar tidak re-load tiap kali)
# nlp = spacy.load("xx_ent_wiki_sm")

# Download stopwords NLTK (hanya sekali)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Dapatkan stopwords bahasa Inggris dari NLTK
ENGLISH_STOPWORDS = set(stopwords.words('english'))

@st.cache_data(show_spinner=False)
def preprocess_text(text, use_stemming=False, user_stopwords=None):
    stop_factory = StopWordRemoverFactory()
    stop_words = set(stop_factory.get_stop_words())
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()

    # Tokenisasi sederhana (AMAN untuk Bahasa Indonesia)
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())

    # Gabungkan semua stopwords
    all_stopwords = stop_words.union(ENGLISH_STOPWORDS)
    if user_stopwords:
        all_stopwords = all_stopwords.union(user_stopwords)

    # Stopword filter (Indonesian + English + User Custom)
    words = [
        w for w in words
        if w not in all_stopwords
    ]

    # Stemming opsional
    if use_stemming:
        words = [stemmer.stem(w) for w in words]

    # Frekuensi minimal
    freq = Counter(words)
    words = [w for w in words if freq[w] >= 2]

    return words


def build_word_graph_fast(words, window_size=2, max_words=5000, min_edge_weight=2):
    """Bangun graph cepat dengan Counter"""
    if len(words) > max_words:
        words = words[:max_words]

    edges = Counter()
    for i in range(len(words) - window_size):
        window = words[i:i + window_size + 1]
        for w1 in window:
            for w2 in window:
                if w1 != w2:
                    edges[(w1, w2)] += 1

    G = nx.Graph()
    for (w1, w2), count in edges.items():
        if count >= min_edge_weight:
            G.add_edge(w1, w2, weight=count)
    return G


def extract_keywords_fast(G, top_n=20, use_centrality=False):
    """Hitung PageRank atau Degree Centrality"""
    if use_centrality:
        scores = nx.degree_centrality(G)
    else:
        try:
            scores = nx.pagerank_numpy(G, alpha=0.85, weight="weight")
        except Exception:
            scores = nx.pagerank(G, alpha=0.85, weight="weight")

    sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords[:top_n], scores
# import matplotlib.patches as mpatches


def plot_word_graph(G, scores, top_n):
    """
    Visualisasi word graph dengan:
    - Node berbentuk oval adaptif (menyesuaikan panjang kata)
    - Garis antar node normal (lembut, seragam)
    - Tema lembut dan nyaman dibaca
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Ambil node teratas
    top_nodes = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n])
    subgraph = G.subgraph(top_nodes.keys())

    # Layout posisi node
    pos = nx.spring_layout(subgraph, k=0.5, seed=42)

    # Warna & colormap
    cmap = plt.cm.Spectral_r
    node_values = [scores[n] for n in subgraph.nodes()]
    norm = plt.Normalize(vmin=min(node_values), vmax=max(node_values))

    # 🎨 Gambar edge normal (lembut dan tidak terlalu menonjol)
    nx.draw_networkx_edges(
        subgraph, pos,
        width=1.0,
        alpha=0.3,
        edge_color="#000",
        ax=ax
    )

    # 🔵 Gambar node berbentuk oval (bukan lingkaran)
    for node in subgraph.nodes():
        x, y = pos[node]
        val = scores[node]
        color = cmap(norm(val))
        # Ukuran oval berdasarkan panjang kata & skor PageRank
        width = 0.12 + len(node) * 0.012  # panjang horizontal menyesuaikan kata
        height = 0.07 + val * 0.25        # sedikit proporsional dengan skor
        ellipse = mpatches.Ellipse(
            (x, y), width=width, height=height,
            color=color, ec="#555", lw=0.6, alpha=0.95
        )
        ax.add_patch(ellipse)

    # ✨ Label di tengah node
    for node, (x, y) in pos.items():
        ax.text(
            x, y, node,
            fontsize=9,
            color='#FFF000', 
            ha='center', va='center',
            fontweight='bold'
        )

    # Tambahkan colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Skor PageRank / Frekuensi Kata", fontsize=10)

    # Judul & tema
    ax.set_title(
        f"Word Graph - {top_n} Kata Kunci Teratas\n"
        f"(Node oval menyesuaikan panjang kata, warna menunjukkan kepentingan)",
        fontsize=12, color='#333', pad=20
    )

    # 🎨 Latar belakang lembut
    fig.patch.set_facecolor("#f7f6f3")
    ax.set_facecolor("#f7f6f3")
    ax.axis("off")

    return fig

def split_into_sentences(text):
    """Pisahkan teks menjadi kalimat menggunakan regex sederhana"""
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]

# ======= STREAMLIT UI =======
st.title("⚡ Ekstraktor Kata Kunci PDF")

uploaded_file = st.file_uploader("📄 Upload file PDF", type=["pdf"])

# ====== Opsi di bagian atas (bersebelahan) ======
col1, col2, col3 = st.columns(3)
with col1:
    use_stemming = st.checkbox(
        "Gunakan stemming", 
        value=False
    )
with col2:
    use_centrality = st.checkbox(
        "Gunakan Degree Centrality", 
        value=False
    )
with col3:
    filter_only_paragraphs = st.checkbox(
        "Filter hanya paragraf",
        value=True,
        help="Hapus header, footer, tabel, dan metadata. Hanya ambil paragraf teks."
    )
# ===============================================

# ====== Input Stopwords Manual ======
st.subheader("⚙️ Stopwords Kustom")
user_stopwords_input = st.text_area(
    "Masukkan stopwords tambahan (pisahkan dengan koma)",
    placeholder="contoh: data, sistem, aplikasi, teknologi",
    help="Kata-kata ini akan diabaikan dalam analisis. Pisahkan dengan koma."
)

# Parse stopwords dari user
user_stopwords = set()
if user_stopwords_input.strip():
    user_stopwords = set([word.strip().lower() for word in user_stopwords_input.split(',') if word.strip()])
    # st.info(f"✅ {len(user_stopwords)} stopwords kustom ditambahkan: {', '.join(sorted(user_stopwords))}")
# =====================================

if uploaded_file:
    with st.spinner("📖 Membaca dan memproses file PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        
        # Filter paragraf jika diaktifkan
        if filter_only_paragraphs:
            filtered_text = filter_paragraphs(raw_text)
            st.success(f"✅ Filter paragraf aktif - Teks hasil filter: {len(filtered_text)} karakter")
        else:
            filtered_text = raw_text
            st.info(f"ℹ️ Filter paragraf nonaktif - Teks lengkap: {len(raw_text)} karakter")
        
        clean = clean_text(filtered_text)

    # ====== 🆕 Tampilkan isi teks dari PDF ======
    st.subheader("📘 Cuplikan Teks dari PDF")
    
    # Tampilkan preview dengan teks yang sudah difilter
    sentences = split_into_sentences(filtered_text)
    preview_text = "\n\n".join(sentences[:5])
    st.text_area("Preview teks (5 kalimat pertama):", preview_text, height=200)

    with st.expander("📄 Lihat seluruh teks hasil ekstraksi"):
        st.text_area("Teks lengkap:", filtered_text, height=400, key="full_text")
    
    # Tampilkan perbandingan jika filter aktif
    if filter_only_paragraphs:
        with st.expander("🔍 Lihat teks asli (sebelum filter)"):
            st.text_area("Teks asli dari PDF:", raw_text, height=400, key="raw_text")
    # ============================================

    # ✅ Proses teks
    with st.spinner("🧠 Memproses teks..."):
        words = preprocess_text(clean, use_stemming=use_stemming, user_stopwords=user_stopwords)

    st.success(f"✅ Preprocessing selesai: {len(words)} kata ({len(set(words))} unik)")

    # ===== Graph dan Ekstraksi =====
    window_size = st.slider("Pilih window size (co-occurrence)", 2, 8, 3)
    top_n = st.slider("Jumlah kata kunci ditampilkan", 5, 80, 20)

    with st.spinner("🔗 Membuat word graph dan menghitung skor..."):
        G = build_word_graph_fast(words, window_size=window_size)
        keywords, scores = extract_keywords_fast(G, top_n=top_n, use_centrality=use_centrality)

    # ===== Output =====
    st.subheader("🔑 Top Kata Kunci")

    # Buat DataFrame dan tambahkan nomor urut
    df_keywords = pd.DataFrame(keywords, columns=["Kata", "Skor"])
    df_keywords.index = range(1, len(df_keywords) + 1)  # mulai dari 1
    df_keywords.index.name = "No"  # beri nama kolom index

    st.dataframe(df_keywords, width='stretch')


    # ===== Graph sesuai top_n =====
    st.subheader("🌐 Visualisasi Word Graph")
    fig = plot_word_graph(G, scores, top_n=top_n)
    st.pyplot(fig)

# else:
#     st.info("⬆️ Silakan upload file PDF terlebih dahulu untuk memulai analisis.")