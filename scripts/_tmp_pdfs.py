from pathlib import Path
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

out = Path(r"c:\Users\DELL\Downloads\Rag_int\rag-builder\data\documents")
out.mkdir(parents=True, exist_ok=True)

def make(label: str) -> str:
    path = out / f"pdf_{label}_{uuid.uuid4().hex[:8]}.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 72
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, f"SOP-{label}: Document Processing Verification {uuid.uuid4()}")
    y -= 28
    c.setFont("Helvetica", 11)
    paragraphs = [
        f"1. Purpose. This standard operating procedure {label} defines intake, chunking, embedding, and retrieval for controlled documents in the DMS platform.",
        "2. Scope. Applies to SOPs, annexes, work instructions, and validation reports stored in Weaviate repositories with hybrid search enabled.",
        "3. Responsibilities. Document owners supply content. Quality assurance reviews metadata fields. Administrators maintain embedding models and processor capacity.",
        "4. Procedure. Upload through the consumer API. Security validates content. Scheduler assigns processors. Section-based chunking creates overlapping segments. Embeddings persist for retrieval.",
        "5. Records. Retain content hashes, repository identifiers, batch identifiers, queue submission timestamps, and processing completion timestamps for auditability.",
        "6. Additional narrative. " + ("quality system validation evidence " * 40),
    ]
    for p in paragraphs:
        for line in p.split(". "):
            if y < 72:
                c.showPage(); y = height - 72; c.setFont("Helvetica", 11)
            c.drawString(72, y, (line[:100]))
            y -= 16
        y -= 8
    c.save()
    return str(path)

for label in ("A", "B", "C"):
    print(make(label))
