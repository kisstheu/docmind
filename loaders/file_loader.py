from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import docx2txt
import pymupdf
from rapidocr_onnxruntime import RapidOCR


def robust_read_text(filepath: Path, logger=None) -> str:
    try:
        return filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = filepath.read_text(encoding="gb18030")
            if logger:
                logger.info(f"         💡 成功使用 GB18030(ANSI) 抢救出中文 -> {filepath.name}")
            return text
        except Exception:
            return filepath.read_text(encoding="utf-8", errors="ignore")


def read_file(file: Path, logger=None) -> tuple[Optional[str], bool]:
    content = ""
    used_sidecar = False
    try:
        suffix = file.suffix.lower()
        if suffix in {".txt", ".md"}:
            content = robust_read_text(file, logger=logger)
        elif suffix == ".docx":
            try:
                content = docx2txt.process(file)
            except Exception:
                if logger:
                    logger.info(f"      🕵️ 发现“伪装者”文件 {file.name}，正在尝试暴力读取...")
                content = robust_read_text(file, logger=logger)
        elif suffix == ".pdf":
            ocr_sidecar_path = file.with_name(file.name + ".ocr.txt")
            if ocr_sidecar_path.exists():
                used_sidecar = True
                content = robust_read_text(ocr_sidecar_path, logger=logger)
            else:
                ocr_engine = None
                with pymupdf.open(file) as pdf_doc:
                    for page_num, page in enumerate(pdf_doc):
                        page_text = page.get_text().strip()
                        if len(page_text) < 15:
                            if ocr_engine is None:
                                ocr_engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
                            if logger:
                                logger.info(f"      👁️ 发现“金身”页面 (页码 {page_num + 1})，启动 OCR...")
                            pix = page.get_pixmap(dpi=150)
                            result, _ = ocr_engine(pix.tobytes("png"))
                            if result:
                                page_text = "\n".join([line[1] for line in result])
                        content += page_text + "\n"
                if content.strip():
                    ocr_sidecar_path.write_text(content, encoding="utf-8")
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ 解析文件失败 {file.name}: {e}")
        return None

    if content:
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = content.replace("\u200b", "").replace("\ufeff", "").strip()
    return (content if content else None), used_sidecar
