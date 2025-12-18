import re
from collections import Counter

import fitz  # PyMuPDF
import networkx as nx
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ---------- setup ----------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


# ---------- utils ----------
def extract_text_from_pdf(uploaded_file) -> str:
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    parts = []
    for p in doc:
        parts.append(p.get_text("text"))
    return "\n".join(parts)


def filter_relevant_sentences(raw: str, min_words: int = 9):
    # Menambahkan regex untuk mendeteksi metadata yang lebih spesifik seperti DOI, ISSN, URL, dll.
    stop_at_refs = re.compile(r"^\s*(daftar\s+pustaka|references|bibliography)\s*$", re.I)
    junk_line = re.compile(
        r"^\s*(\d{1,6}|tabel|table|gambar|figure|grafik|chart|sumber|doi|issn|vol|volume|no|issue|e-issn|https?://|mailto:)\b", 
        re.I
    )
    allcaps_title = re.compile(r"^[A-Z\s]{6,}$")

    kept_lines = []
    for ln in raw.splitlines():
        s = ln.strip()
        if stop_at_refs.match(s):
            break
        if not s:
            kept_lines.append("")
            continue
        if junk_line.match(s) or allcaps_title.match(s):
            continue
        digit_ratio = sum(c.isdigit() for c in s) / max(len(s), 1)
        if digit_ratio > 0.35:
            continue
        kept_lines.append(s)

    text = "\n".join(kept_lines)
    text = re.sub(r"-\n(?=[A-Za-z])", "", text)        # sambung kata ter-hyphen
    text = re.sub(r"\n(?!\n)", " ", text)              # newline tunggal jadi spasi
    text = re.sub(r"\s+", " ", text).strip()

    sents = re.split(r"(?<=[.!?])\s+", text)
    marker = re.compile(
        r"\b(merupakan|adalah|yaitu|bertujuan|digunakan|berdasarkan|sehingga|namun|selain itu|melalui|diperkirakan)\b",
        re.I,
    )

    out = []
    for s in sents:
        w = s.split()
        if len(w) < min_words:
            continue
        if sum(c.isdigit() for c in s) / max(len(s), 1) > 0.25:
            continue
        if re.search(r"\b(tabel|gambar|grafik|sumber)\b", s, re.I):
            continue
        if not marker.search(s) and len(w) < 14:
            continue
        out.append(s.strip())

    return out, text  # (list_kalimat, teks_gabungan)


def make_stopwords(user_stopwords: set[str]):
    stop_id = set(StopWordRemoverFactory().get_stop_words())
    stop_en = set(stopwords.words("english"))
    return stop_id | stop_en | (user_stopwords or set())


def tokenize(text: str, stop_set: set[str], use_stemming: bool, min_freq: int):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = re.findall(r"[a-z][a-z0-9\-]{2,}", text)
    tokens = [t for t in tokens if t not in stop_set]

    if use_stemming:
        stemmer = StemmerFactory().create_stemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    if min_freq > 1:
        freq = Counter(tokens)
        tokens = [t for t in tokens if freq[t] >= min_freq]

    # SAFETY: pastikan semua token string (anti bug list jadi node)
    tokens = [str(t) for t in tokens]
    return tokens


def build_word_graph(tokens, window_size: int, min_edge_weight: int):
    edges = Counter()
    n = len(tokens)
    for i in range(n):
        w = tokens[i : i + window_size + 1]
        for a in w:
            for b in w:
                if a != b:
                    edges[(a, b)] += 1

    G = nx.Graph()
    for (a, b), c in edges.items():
        if c >= min_edge_weight:
            G.add_edge(a, b, weight=c)
    return G


def extract_keywords(G, top_n: int, use_degree: bool):
    if G.number_of_nodes() == 0:
        return [], {}

    scores = nx.degree_centrality(G) if use_degree else nx.pagerank(G, alpha=0.85, weight="weight")
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top, scores


def plot_word_graph(G, scores, top_n: int):
    fig, ax = plt.subplots(figsize=(12, 8))

    top_nodes = [k for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    H = G.subgraph(top_nodes).copy()
    pos = nx.spring_layout(H, k=0.6, seed=42)

    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.25, width=1.0)

    node_sizes = [max(scores[n] * 6000, 200) for n in H.nodes()]
    node_colors = [scores[n] for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.Spectral_r)

    # IMPORTANT: label harus string
    labels = {n: str(n) for n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=9, ax=ax)

    ax.axis("off")
    ax.set_title(f"Word Graph (Top {top_n})")
    return fig


# ---------- UI ----------
st.title("📄 PDF → Filter Kalimat → Keyword Graph")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

colA, colB, colC = st.columns(3)
with colA:
    use_stemming = st.checkbox("Stemming", value=False)
with colB:
    use_degree = st.checkbox("Degree Centrality (bukan PageRank)", value=False)
with colC:
    min_words = st.slider("Min kata per kalimat (filter)", 6, 20, 9)

st.subheader("⚙️ Stopwords kustom")
user_stopwords_input = st.text_area("Pisahkan dengan koma", placeholder="contoh: data, sistem, aplikasi")
user_stopwords = {w.strip().lower() for w in user_stopwords_input.split(",") if w.strip()}

if uploaded:
    raw_text = extract_text_from_pdf(uploaded)

    sents, filtered_text = filter_relevant_sentences(raw_text, min_words=min_words)

    st.subheader("🧾 Teks hasil filter (cek relevansi)")
    st.text_area("Filtered text", filtered_text, height=300)

    with st.expander("✅ Kalimat yang dipakai (setelah filter)"):
        st.write(sents)

    stop_set = make_stopwords(user_stopwords)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        window_size = st.slider("Window co-occurrence", 2, 10, 3)
    with col2:
        min_edge_weight = st.slider("Min edge weight", 1, 10, 2)
    with col3:
        top_n = st.slider("Top N keywords", 5, 80, 20)
    with col4:
        min_freq = st.slider("Min frekuensi token", 1, 5, 2)

    tokens = tokenize(filtered_text, stop_set, use_stemming=use_stemming, min_freq=min_freq)
    st.caption(f"Token: {len(tokens)} | unik: {len(set(tokens))}")

    G = build_word_graph(tokens, window_size=window_size, min_edge_weight=min_edge_weight)
    keywords, scores = extract_keywords(G, top_n=top_n, use_degree=use_degree)

    st.subheader("🔑 Kata kunci")
    df = pd.DataFrame(keywords, columns=["Kata", "Skor"])
    st.dataframe(df, use_container_width=True)

    st.subheader("🌐 Visualisasi Word Graph")
    if G.number_of_nodes() == 0:
        st.warning("Graph kosong. Coba turunkan min_edge_weight / min_freq atau longgarkan filter kalimat.")
    else:
        fig = plot_word_graph(G, scores, top_n=top_n)
        st.pyplot(fig)
else:
    st.info("Upload PDF dulu.")
