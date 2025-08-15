
import zipfile, xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import pathlib

def read_docx_paragraphs(path: str) -> List[Tuple[str, str]]:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    root = ET.fromstring(xml)
    paras = []
    for p in root.findall(".//w:p", ns):
        pstyle = ""
        pPr = p.find("w:pPr", ns)
        if pPr is not None:
            ps = pPr.find("w:pStyle", ns)
            if ps is not None:
                pstyle = ps.attrib.get(f'{{{ns["w"]}}}val', "")
        texts = []
        for t in p.findall(".//w:t", ns):
            texts.append(t.text or "")
        txt = "".join(texts).strip()
        if txt:
            paras.append((pstyle, txt))
    return paras

def docx_to_single_guide(docx_path: str) -> Dict:
    paras = read_docx_paragraphs(docx_path)
    context = "\n".join([t for _, t in paras])
    guide = {
        "id": "lifecycle-reference",
        "title": "Delivery of Digital Products",
        "phase": "",
        "url": "",
        "tags": ["#lifecycle", "#reference"],
        "metadata": {"source": pathlib.Path(docx_path).name},
        "context": context
    }
    return {"guide": guide, "hints": {}}
