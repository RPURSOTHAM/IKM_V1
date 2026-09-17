import pytest
pytest.skip("legacy docx_template/pdf_template removed; replaced by template_compliance", allow_module_level=True)

from pathlib import Path
import pytest
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from src.features.document_processing.extractors.docx_template.extractor import extract_template

def test_docx_run_formatting_extraction(tmp_path: Path) -> None:
    docx_path = tmp_path / "test_runs.docx"
    doc = Document()
    
    # Add a paragraph with mixed formatting runs
    p = doc.add_paragraph()
    
    # 1. Plain text run (inherits all)
    r1 = p.add_run("Plain ")
    
    # 2. Bold, italic, underline, strike run
    r2 = p.add_run("BoldItalicUnderlineStrike ")
    r2.bold = True
    r2.italic = True
    r2.underline = True
    r2.font.strike = True
    
    # 3. Font name and size run
    r3 = p.add_run("Arial14pt ")
    r3.font.name = "Arial"
    r3.font.size = Pt(14)
    
    # 4. Color, superscript, subscript run
    r4 = p.add_run("RedSuper ")
    r4.font.color.rgb = RGBColor(255, 0, 0)
    r4.font.superscript = True
    
    # 5. Caps and hidden run
    r5 = p.add_run("SmallCapsHidden")
    r5.font.small_caps = True
    r5.font.all_caps = False
    
    doc.save(str(docx_path))
    
    # Run extractor
    template = extract_template(docx_path)
    
    blocks = template["content_blocks"]
    assert len(blocks) >= 1
    p_block = blocks[0]
    assert "runs" in p_block
    runs = p_block["runs"]
    assert len(runs) == 5
    
    # 1. Plain
    assert runs[0]["text"] == "Plain "
    assert runs[0]["bold"] is None # Inherited
    assert runs[0]["italic"] is None
    assert runs[0]["underline"] is None
    assert runs[0]["font_name"] is None
    
    # 2. Bold, italic, underline, strike
    assert runs[1]["text"] == "BoldItalicUnderlineStrike "
    assert runs[1]["bold"] is True
    assert runs[1]["italic"] is True
    assert runs[1]["underline"] is True
    assert runs[1]["strike"] is True
    
    # 3. Font name and size
    assert runs[2]["text"] == "Arial14pt "
    assert runs[2]["font_name"] == "Arial"
    assert pytest.approx(runs[2]["font_size"]) == 14.0
    
    # 4. Color and superscript
    assert runs[3]["text"] == "RedSuper "
    assert runs[3]["font_color"] == "FF0000"
    assert runs[3]["superscript"] is True
    assert not runs[3]["subscript"]
    
    # 5. Caps and hidden
    assert runs[4]["text"] == "SmallCapsHidden"
    assert runs[4]["small_caps"] is True
    assert runs[4]["all_caps"] is False

