from pathlib import Path
from src.features.document_processing.loaders.docxloader import DocxLoader
from src.features.chunking.strategies.chunking_strategies import chunk_document_with_strategy

docx = Path(r"c:\Users\Neevan\Downloads\SOP-CL-TM-001_Study_Startup.docx")
loader = DocxLoader()
blocks = loader.load(docx, citation_retainment=False)
chunks = chunk_document_with_strategy(
    blocks,
    docx.name,
    strategy="section-based",
    chunk_size=80,
    overlap_sentences=1,
    min_content_words=6,
    citation_retainment=False,
)
print(f"Blocks: {len(blocks)}, Chunks: {len(chunks)}")
for c in chunks:
    chunk_type = getattr(c, "chunk_type", "?")
    print(f"  sec={c.section_name!r} page={c.page} type={chunk_type} | {c.text[:70]}...")
