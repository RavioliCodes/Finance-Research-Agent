"""
document_processor.py
Handles ingestion, chunking, and preprocessing of company documents.
Supports .txt, .md, .csv, and .pdf files (including tables, charts, images).
"""

import os
import re
import hashlib
import base64
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


#  Optional dependencies – graceful fallbacks

try:
    from sentence_transformers import SentenceTransformer
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # pymupdf
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False



@dataclass
class DocumentChunk:
    """A single chunk of text from a document with metadata."""
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    section_header: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return f"Chunk({self.source_file}#{self.chunk_index}: '{preview}...')"


class DocumentProcessor:
    """
    Ingests documents and splits them into semantically coherent chunks.

    PDF support
    -----------
    PDFs are processed page-by-page using pdfplumber (text + tables) and
    optionally pymupdf (image/chart extraction).  Tables are converted to
    Markdown so the table-detection logic keeps them as atomic chunks.
    Charts and images are represented as [FIGURE …] text blocks; if a
    `vision_model` is supplied (e.g. "claude-opus-4-6") each image is
    described by that model via the Anthropic API.

    Semantic chunking
    -----------------
    Within each section, sentences are embedded with sentence-transformers
    and split at positions where cosine distance between consecutive
    context-window embeddings exceeds `breakpoint_percentile`.
    Falls back to a sliding-window approach when sentence-transformers is
    not installed.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf"}

    # Minimum image dimensions to keep (filters out tiny decorative assets).
    _MIN_IMAGE_PX = 80

    def __init__(
        self,
        # Semantic chunking
        model_name: str = "all-MiniLM-L6-v2",
        breakpoint_percentile: int = 85,
        buffer_size: int = 1,
        min_chunk_words: int = 20,
        # Structural fallback
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        # PDF vision (optional)
        vision_model: Optional[str] = None,
    ):
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size
        self.min_chunk_words = min_chunk_words
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vision_model = vision_model
        self.documents: list[DocumentChunk] = []

        if _SEMANTIC_AVAILABLE:
            self._model = SentenceTransformer(model_name)
            print(f"[semantic chunker] loaded model: {model_name}")
        else:
            self._model = None
            print("[semantic chunker] sentence-transformers not found — using structural fallback")

        if vision_model:
            if not _ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "anthropic package required for vision_model: pip install anthropic"
                )
            self._vision_client = anthropic.Anthropic()
        else:
            self._vision_client = None


    #  Public APIs

    def ingest_folder(self, folder_path: str) -> list[DocumentChunk]:
        """Ingest all supported documents from a folder."""
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Document folder not found: {folder_path}")

        all_chunks: list[DocumentChunk] = []
        for fname in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                print(f"  [skip] Unsupported file type: {fname}")
                continue

            fpath = os.path.join(folder_path, fname)
            text = self._read_file(fpath)
            if not text.strip():
                print(f"  [skip] Empty file: {fname}")
                continue

            chunks = self._split_into_chunks(text, source_file=fname)
            all_chunks.extend(chunks)
            print(f"  [ok]   {fname} → {len(chunks)} chunks")

        self.documents = all_chunks
        print(f"\nTotal chunks ingested: {len(self.documents)}")
        return self.documents


    def ingest_text(self, text: str, source_name: str = "inline") -> list[DocumentChunk]:
        """Ingest raw text directly."""
        chunks = self._split_into_chunks(text, source_file=source_name)
        self.documents.extend(chunks)
        return chunks


    def _read_file(self, path: str) -> str:
        """Read any supported file and return its content as plain text."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return self._read_pdf(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
        

    def _read_pdf(self, path: str) -> str:
        """
        Extract structured text from a PDF.

        Per-page pipeline:
          1. Tables  → title (text above bbox) + Markdown rows
          2. Prose   → plain text, with table and image regions excluded
          3. Figures → caption text found adjacent to the image bbox;
                       falls back to vision model only when no caption exists
                       and vision_model is configured.

        Caption/title extraction works by cropping pdfplumber's page to a
        narrow strip above or below each element and reading whatever text
        sits there — no vision model needed for labelled financial content.
        """
        if not _PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "pdfplumber is required for PDF support: pip install pdfplumber"
            )

        # Image bytes are only needed when vision fallback is active.
        page_image_bytes: dict[int, list[tuple[bytes, str]]] = {}
        if self.vision_model and _PYMUPDF_AVAILABLE:
            page_image_bytes = self._extract_pdf_images(path)

        page_texts: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                blocks: list[tuple[float, str]] = []  # (y_top, text)

                # ---- Tables ----
                table_bboxes: list[tuple[float, float, float, float]] = []
                for tbl in page.find_tables():
                    bbox = tbl.bbox  # (x0, top, x1, bottom)
                    table_bboxes.append(bbox)
                    data = tbl.extract()
                    if not data:
                        continue
                    md = self._table_to_markdown(data)
                    title = self._find_caption(page, bbox, above=True, below=False)
                    content = f"{title}\n{md}" if title else md
                    blocks.append((bbox[1], f"\n{content}\n"))

                # ---- Figures / charts ----
                img_bboxes: list[tuple[float, float, float, float]] = []
                for img_idx, img in enumerate(page.images or []):
                    w = img.get("width", 0) or 0
                    h = img.get("height", 0) or 0
                    if w < self._MIN_IMAGE_PX or h < self._MIN_IMAGE_PX:
                        continue

                    bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                    img_bboxes.append(bbox)

                    # 1. Caption text adjacent to the figure (preferred).
                    caption = self._find_caption(page, bbox, above=True, below=True)

                    # 2. Vision model fallback only when no caption found.
                    if not caption and self._vision_client:
                        img_list = page_image_bytes.get(page_num, [])
                        if img_idx < len(img_list):
                            caption = self._describe_image(*img_list[img_idx])

                    desc = caption or f"page {page_num + 1}, figure {img_idx + 1}"
                    blocks.append((img["top"], f"[FIGURE: {desc}]"))

                # ---- Prose (exclude table and image regions) ----
                excluded = table_bboxes + img_bboxes
                words = page.extract_words(x_tolerance=3, y_tolerance=3) or []
                prose_words = [
                    w for w in words
                    if not self._in_any_bbox(w, excluded)
                ]
                if prose_words:
                    blocks.append((prose_words[0]["top"], self._words_to_text(prose_words)))

                blocks.sort(key=lambda b: b[0])
                page_content = "\n".join(b[1] for b in blocks).strip()
                if page_content:
                    page_texts.append(f"--- Page {page_num + 1} ---\n{page_content}")

        return "\n\n".join(page_texts)

    # ---- Caption / title extraction ----

    # Matches common financial caption/title keywords.
    _CAPTION_RE = re.compile(
        r"\b(figure|fig\.?|chart|exhibit|graph|panel|table|note|source)\b",
        re.IGNORECASE,
    )

    def _find_caption(
        self,
        page,
        bbox: tuple[float, float, float, float],
        above: bool = True,
        below: bool = True,
        search_px: float = 40.0,
    ) -> Optional[str]:
        """
        Return text found in a narrow strip directly above and/or below `bbox`.

        Preference order:
          1. Text containing a formal caption keyword (Figure, Table, Note, …)
          2. Any short text (≤ 30 words) — likely a title or label
          3. None
        """
        x0, top, x1, bottom = bbox
        pw, ph = page.width, page.height
        # Add small horizontal margin so we catch slightly wider captions.
        margin = min(20.0, (x1 - x0) * 0.1)

        candidates: list[str] = []

        search_areas: list[tuple[float, float, float, float]] = []
        if below:
            search_areas.append((
                max(0.0, x0 - margin), bottom,
                min(pw, x1 + margin), min(ph, bottom + search_px),
            ))
        if above:
            search_areas.append((
                max(0.0, x0 - margin), max(0.0, top - search_px),
                min(pw, x1 + margin), top,
            ))

        for area in search_areas:
            # Skip degenerate areas (zero height).
            if area[3] <= area[1]:
                continue
            try:
                text = (page.within_bbox(area).extract_text() or "").strip()
            except Exception:
                continue
            if not text:
                continue
            # Formal caption keyword → highest priority, return immediately.
            if self._CAPTION_RE.search(text):
                return text
            # Short unlabelled text → keep as a lower-priority candidate.
            if len(text.split()) <= 30:
                candidates.append(text)

        return candidates[0] if candidates else None

    # ---- Image bytes (vision fallback only) ----

    def _extract_pdf_images(
        self, pdf_path: str
    ) -> dict[int, list[tuple[bytes, str]]]:
        """Return {page_index: [(image_bytes, ext), …]} — used only for vision fallback."""
        doc = fitz.open(pdf_path)
        result: dict[int, list[tuple[bytes, str]]] = {}
        for page_num in range(len(doc)):
            images: list[tuple[bytes, str]] = []
            for img_ref in doc[page_num].get_images(full=True):
                xref = img_ref[0]
                try:
                    img_data = doc.extract_image(xref)
                    if (
                        img_data["width"] >= self._MIN_IMAGE_PX
                        and img_data["height"] >= self._MIN_IMAGE_PX
                    ):
                        images.append((img_data["image"], img_data.get("ext", "png")))
                except Exception:
                    pass
            if images:
                result[page_num] = images
        doc.close()
        return result

    def _describe_image(self, img_bytes: bytes, img_ext: str = "png") -> str:
        """Vision model fallback — only called when no caption was found."""
        media_type_map = {
            "png": "image/png", "jpg": "image/jpeg",
            "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp",
        }
        media_type = media_type_map.get(img_ext.lower(), "image/png")
        b64 = base64.standard_b64encode(img_bytes).decode()

        response = self._vision_client.messages.create(
            model=self.vision_model,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {
                        "type": "text",
                        "text": (
                            "This image is from a financial document. "
                            "If it is a chart, graph, or data visualization, describe the "
                            "key metrics, trends, values, and title visible. "
                            "If it is a logo or purely decorative, reply with 'Decorative image'."
                        ),
                    },
                ],
            }],
        )
        return response.content[0].text.strip()

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        """Convert a pdfplumber table (list of rows) to a Markdown table string."""
        if not table:
            return ""
        cleaned = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            for row in table
        ]
        col_count = max(len(row) for row in cleaned)
        # Pad rows to equal width.
        for row in cleaned:
            row.extend([""] * (col_count - len(row)))

        col_widths = [
            max(len(row[i]) for row in cleaned) for i in range(col_count)
        ]

        def fmt_row(row: list[str]) -> str:
            cells = [row[i].ljust(col_widths[i]) for i in range(col_count)]
            return "| " + " | ".join(cells) + " |"

        separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
        lines = [fmt_row(cleaned[0]), separator]
        lines += [fmt_row(row) for row in cleaned[1:]]
        return "\n".join(lines)

    @staticmethod
    def _in_any_bbox(
        word: dict,
        bboxes: list[tuple[float, float, float, float]],
    ) -> bool:
        """Return True if a pdfplumber word dict falls inside any of the given bboxes."""
        return any(
            bx[0] <= word["x0"] and word["x1"] <= bx[2]
            and bx[1] <= word["top"] and word["bottom"] <= bx[3]
            for bx in bboxes
        )

    @staticmethod
    def _words_to_text(words: list[dict]) -> str:
        """Reconstruct readable text from pdfplumber word dicts, preserving line breaks."""
        lines: dict[int, list[dict]] = {}
        for w in words:
            # Bin words into lines by rounding top-coordinate to nearest 3px.
            y = round(w["top"] / 3) * 3
            lines.setdefault(y, []).append(w)
        result = []
        for y in sorted(lines):
            line_words = sorted(lines[y], key=lambda w: w["x0"])
            result.append(" ".join(w["text"] for w in line_words))
        return "\n".join(result)



    #  Top-level chunking dispatcher

    def _split_into_chunks(self, text: str, source_file: str) -> list[DocumentChunk]:
        sections = self._detect_sections(text)
        all_chunks: list[DocumentChunk] = []
        chunk_idx = 0

        for section_header, section_text in sections:
            if not section_text.strip():
                continue

            if self._model is not None:
                new_chunks = self._semantic_chunks(
                    section_text, source_file, section_header, chunk_idx
                )
            else:
                new_chunks = self._structural_chunks(
                    section_text, source_file, section_header, chunk_idx
                )

            all_chunks.extend(new_chunks)
            chunk_idx += len(new_chunks)

        return all_chunks

    #  Semantic chunking

    def _semantic_chunks(
        self,
        text: str,
        source_file: str,
        section_header: Optional[str],
        start_idx: int,
    ) -> list[DocumentChunk]:
        """Split text semantically using embedding cosine distances."""
        blocks = self._extract_table_blocks(text)
        chunks: list[DocumentChunk] = []
        idx = start_idx

        for is_table, block in blocks:
            block = block.strip()
            if not block:
                continue

            if is_table:
                chunks.append(self._make_chunk(block, source_file, idx, section_header))
                idx += 1
                continue

            sentences = self._split_sentences(block)
            if len(sentences) <= 2:
                if self._word_count(block) >= self.min_chunk_words:
                    chunks.append(self._make_chunk(block, source_file, idx, section_header))
                    idx += 1
                continue

            # Context-window embeddings.
            windows = [
                " ".join(sentences[max(0, i - self.buffer_size): i + self.buffer_size + 1])
                for i in range(len(sentences))
            ]
            embeddings = self._model.encode(windows, show_progress_bar=False)

            distances = self._cosine_distances(embeddings)
            threshold = float(np.percentile(distances, self.breakpoint_percentile))
            breakpoints = {i + 1 for i, d in enumerate(distances) if d > threshold}

            raw_groups: list[list[str]] = []
            current: list[str] = []
            for i, sent in enumerate(sentences):
                if i in breakpoints and current:
                    raw_groups.append(current)
                    current = []
                current.append(sent)
            if current:
                raw_groups.append(current)

            for group in self._merge_small_groups(raw_groups):
                chunk_text = " ".join(group).strip()
                if self._word_count(chunk_text) >= self.min_chunk_words:
                    chunks.append(self._make_chunk(chunk_text, source_file, idx, section_header))
                    idx += 1

        return chunks


    #  Structural fallback

    def _structural_chunks(
        self,
        text: str,
        source_file: str,
        section_header: Optional[str],
        start_idx: int,
    ) -> list[DocumentChunk]:
        words = text.split()
        chunks: list[DocumentChunk] = []
        idx = start_idx

        if len(words) < self.min_chunk_words:
            chunks.append(self._make_chunk(text.strip(), source_file, idx, section_header))
            return chunks

        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            if self._word_count(chunk_text) >= self.min_chunk_words:
                chunks.append(self._make_chunk(chunk_text, source_file, idx, section_header))
                idx += 1
            start += self.chunk_size - self.chunk_overlap
            if end == len(words):
                break

        return chunks


    #  Table block extraction

    def _extract_table_blocks(self, text: str) -> list[tuple[bool, str]]:
        """Partition text into (is_table, block) segments so tables are never split."""
        lines = text.split("\n")
        if not lines:
            return []

        blocks: list[tuple[bool, str]] = []
        current_lines: list[str] = [lines[0]]
        current_is_table = self._is_table_line(lines[0])

        for line in lines[1:]:
            line_is_table = self._is_table_line(line)
            if line_is_table != current_is_table:
                blocks.append((current_is_table, "\n".join(current_lines)))
                current_lines = [line]
                current_is_table = line_is_table
            else:
                current_lines.append(line)

        if current_lines:
            blocks.append((current_is_table, "\n".join(current_lines)))

        return blocks

    @staticmethod
    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.count("|") >= 2:
            return True
        if "\t" in stripped and re.search(r"\d", stripped):
            return True
        return False


    #  Sentence splitting

    _SENT_RE = re.compile(
        r"(?<!\w\.\w)"
        r"(?<![A-Z][a-z]\.)"
        r"(?<!\d\.\d)"
        r"(?<=\.|\?|!)\s+(?=[A-Z])"
    )

    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in self._SENT_RE.split(text.strip()) if s.strip()]


    #  Embedding helpers

    @staticmethod
    def _cosine_distances(embeddings: np.ndarray) -> list[float]:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.maximum(norms, 1e-10)
        sims = (normed[:-1] * normed[1:]).sum(axis=1)
        return (1.0 - sims).tolist()

    def _merge_small_groups(self, groups: list[list[str]]) -> list[list[str]]:
        merged: list[list[str]] = []
        for group in groups:
            if merged and self._word_count(" ".join(merged[-1])) < self.min_chunk_words:
                merged[-1].extend(group)
            else:
                merged.append(list(group))
        return merged


    #  Section detection

    # Common 10-K / financial document section titles (case-insensitive match).
    _FINANCIAL_HEADINGS = re.compile(
        r"^(?:"
        r"results?\s+of\s+operations"
        r"|revenue|net\s+(?:income|revenue|loss)"
        r"|cost\s+of\s+(?:revenue|goods\s+sold)"
        r"|operating\s+(?:expenses?|income|results)"
        r"|gross\s+(?:profit|margin)"
        r"|financial\s+(?:highlights?|summary|results|overview|condition|statements?)"
        r"|balance\s+sheet|cash\s+flow|liquidity"
        r"|segment\s+(?:information|results|reporting)"
        r"|risk\s+factors?"
        r"|(?:selected|consolidated)\s+financial\s+data"
        r"|management.s?\s+discussion"
        r"|quarterly\s+(?:results|financial\s+data)"
        r"|stockholders.\s+equity"
        r"|income\s+(?:taxes|from\s+operations)"
        r"|earnings\s+per\s+share"
        r"|subscriptions?"
        r"|digital\s+(?:media|experience)"
        r"|(?:item|part)\s+\d"
        r")\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _detect_sections(text: str) -> list[tuple[Optional[str], str]]:
        """Split text into (header, body) pairs.

        Recognises:
        - ALL-CAPS headings (original behaviour)
        - Numbered headings like '1. Overview'
        - PDF page markers ('--- Page N ---')
        - Common financial / 10-K section titles (title-case or mixed)
        """
        lines = text.split("\n")
        sections: list[tuple[Optional[str], str]] = []
        current_header: Optional[str] = None
        current_body: list[str] = []

        allcaps_re = re.compile(
            r"^(?:[A-Z][A-Z &/\-–—:,.'()0-9]{4,}|PILLAR \d|"
            r"\d{1,2}[.)]\s+[A-Z])"
        )
        page_re = re.compile(r"^---\s*Page\s+\d+\s*---$")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_body.append(line)
                continue

            is_header = (
                allcaps_re.match(stripped)
                or page_re.match(stripped)
                or (
                    len(stripped.split()) <= 10
                    and DocumentProcessor._FINANCIAL_HEADINGS.match(stripped)
                )
            )

            if is_header:
                if current_body:
                    sections.append((current_header, "\n".join(current_body)))
                current_header = stripped
                current_body = []
            else:
                current_body.append(line)

        if current_body:
            sections.append((current_header, "\n".join(current_body)))

        return sections if sections else [(None, text)]


    #  Utilities

    _PAGE_NUM_RE = re.compile(r"Page\s+(\d+)", re.IGNORECASE)

    def _make_chunk(
        self,
        text: str,
        source_file: str,
        idx: int,
        section_header: Optional[str],
    ) -> DocumentChunk:
        metadata: dict = {}
        for haystack in (section_header or "", text):
            m = self._PAGE_NUM_RE.search(haystack)
            if m:
                metadata["page"] = int(m.group(1))
                break
        return DocumentChunk(
            chunk_id=self._make_id(source_file, idx),
            text=text,
            source_file=source_file,
            chunk_index=idx,
            section_header=section_header,
            metadata=metadata,
        )

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _make_id(source: str, idx: int) -> str:
        raw = f"{source}::{idx}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
