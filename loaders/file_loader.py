from __future__ import annotations

import io
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import docx2txt
import pymupdf
from PIL import Image
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


def _normalize_content(content: str) -> str:
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = content.replace("\u200b", "").replace("\ufeff", "").strip()
    return content


def _create_ocr_engine() -> RapidOCR:
    return RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)


def _ocr_image_bytes(
    image_bytes: bytes,
    ocr_engine: RapidOCR | None = None,
    logger=None,
    source_name: str = "",
) -> str:
    try:
        if ocr_engine is None:
            ocr_engine = _create_ocr_engine()

        result, _ = ocr_engine(image_bytes)
        if not result:
            return ""

        lines: list[str] = []
        for item in result:
            if len(item) >= 2 and item[1]:
                lines.append(str(item[1]).strip())

        text = "\n".join(x for x in lines if x).strip()
        if text and logger and source_name:
            logger.info(f"      🔎 图片OCR提取成功 -> {source_name}")
        return text
    except Exception as e:
        if logger and source_name:
            logger.warning(f"⚠️ 图片OCR失败 {source_name}: {e}")
        return ""


def _extract_docx_media_ocr_text(docx_file: Path, logger=None) -> str:
    ocr_texts: list[str] = []
    ocr_engine: RapidOCR | None = None

    try:
        with zipfile.ZipFile(docx_file, "r") as zf:
            media_files = [
                name
                for name in zf.namelist()
                if name.startswith("word/media/")
            ]

            for media_name in media_files:
                try:
                    image_bytes = zf.read(media_name)

                    # 验证是否真的是图片，避免对奇怪资源做 OCR
                    with Image.open(io.BytesIO(image_bytes)) as img:
                        img.verify()

                    if ocr_engine is None:
                        ocr_engine = _create_ocr_engine()

                    text = _ocr_image_bytes(
                        image_bytes,
                        ocr_engine=ocr_engine,
                        logger=logger,
                        source_name=f"{docx_file.name}:{media_name}",
                    )
                    if text:
                        ocr_texts.append(f"[图片OCR:{media_name}]\n{text}")
                except Exception as e:
                    if logger:
                        logger.warning(f"⚠️ 处理 DOCX 内图片失败 {docx_file.name}:{media_name}: {e}")
                    continue
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ 提取 DOCX 图片失败 {docx_file.name}: {e}")

    return "\n\n".join(ocr_texts).strip()


def _extract_docx_text_with_embedded_ocr(docx_file: Path, logger=None) -> str:
    body_text = docx2txt.process(docx_file) or ""
    image_ocr_text = _extract_docx_media_ocr_text(docx_file, logger=logger)

    if body_text.strip() and image_ocr_text:
        return f"{body_text.strip()}\n\n{image_ocr_text}".strip()
    if image_ocr_text:
        return image_ocr_text.strip()
    return body_text.strip()


def _convert_doc_to_docx_temp(file: Path, logger=None) -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        cmd = [
            "soffice",
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            str(tmpdir_path),
            str(file),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"LibreOffice 转换失败: {result.stderr.strip()}")

        converted = tmpdir_path / f"{file.stem}.docx"
        if not converted.exists():
            raise RuntimeError("转换后的 docx 不存在")

        # 临时目录会在函数返回后销毁，所以拷贝到一个真实的临时文件
        final_tmp = Path(tempfile.mktemp(suffix=".docx"))
        final_tmp.write_bytes(converted.read_bytes())

        if logger:
            logger.info(f"      🔄 已将 DOC 转为临时 DOCX -> {file.name}")

        return final_tmp


def _read_doc_with_sidecar(file: Path, logger=None):
    sidecar_path = file.with_name(file.name + ".converted.txt")

    if logger:
        logger.info(f"      📄 正在解析 DOC 文件 -> {file.name}")

    # sidecar
    if sidecar_path.exists():
        if sidecar_path.stat().st_mtime >= file.stat().st_mtime:
            if logger:
                logger.info(f"      ♻️ 复用 DOC sidecar -> {sidecar_path.name}")
            return robust_read_text(sidecar_path, logger=logger), True
        else:
            if logger:
                logger.info(f"      🔄 DOC sidecar 已过期，重新解析 -> {file.name}")

    tmp_docx = None
    try:
        if logger:
            logger.info(f"      🔄 正在将 DOC 转换为 DOCX -> {file.name}")

        tmp_docx = _convert_doc_to_docx_temp(file, logger=logger)

        if logger:
            logger.info(f"      🧠 正在提取 DOCX 正文 -> {file.name}")

        content = _extract_docx_text_with_embedded_ocr(tmp_docx, logger=logger)

        if not content.strip():
            if logger:
                logger.warning(f"⚠️ DOCX 正文为空 -> {file.name}")

        content = _normalize_content(content)

        if content:
            sidecar_path.write_text(content, encoding="utf-8")
            if logger:
                logger.info(f"      💾 已写入 DOC sidecar -> {sidecar_path.name}")
            return content, False

        if logger:
            logger.error(f"❌ DOC 解析失败（无内容） -> {file.name}")
        return None, False

    except Exception as e:
        if logger:
            logger.error(f"❌ DOC 解析异常 {file.name}: {e}")
        return None, False

    finally:
        if tmp_docx and tmp_docx.exists():
            try:
                tmp_docx.unlink()
            except Exception:
                pass

def _read_pdf_with_sidecar(file: Path, logger=None) -> tuple[Optional[str], bool]:
    ocr_sidecar_path = file.with_name(file.name + ".ocr.txt")

    # 若 sidecar 存在且不早于原文件，直接复用
    if ocr_sidecar_path.exists() and ocr_sidecar_path.stat().st_mtime >= file.stat().st_mtime:
        if logger:
            logger.info(f"      ♻️ 复用 PDF OCR sidecar -> {ocr_sidecar_path.name}")
        return robust_read_text(ocr_sidecar_path, logger=logger), True

    content = ""
    ocr_engine: RapidOCR | None = None

    with pymupdf.open(file) as pdf_doc:
        for page_num, page in enumerate(pdf_doc):
            page_text = page.get_text().strip()

            # 文字太少，认为可能是扫描页/图片页，触发 OCR
            if len(page_text) < 15:
                if ocr_engine is None:
                    ocr_engine = _create_ocr_engine()
                if logger:
                    logger.info(f"      👁️ 发现“金身”页面 (页码 {page_num + 1})，启动 OCR...")

                pix = page.get_pixmap(dpi=150)
                result, _ = ocr_engine(pix.tobytes("png"))
                if result:
                    page_text = "\n".join([line[1] for line in result if len(line) >= 2 and line[1]])

            content += page_text + "\n"

    content = _normalize_content(content)
    if content:
        ocr_sidecar_path.write_text(content, encoding="utf-8")
        if logger:
            logger.info(f"      💾 已写入 PDF OCR sidecar -> {ocr_sidecar_path.name}")
        return content, False

    return None, False


def read_file(file: Path, logger=None) -> tuple[Optional[str], bool]:
    content = ""
    used_sidecar = False

    try:
        suffix = file.suffix.lower()

        if suffix in {".txt", ".md", ".csv"}:
            content = robust_read_text(file, logger=logger)

        elif suffix == ".docx":
            try:
                content = _extract_docx_text_with_embedded_ocr(file, logger=logger)
            except Exception:
                if logger:
                    logger.info(f"      🕵️ 发现“伪装者”文件 {file.name}，正在尝试暴力读取...")
                content = robust_read_text(file, logger=logger)

        elif suffix == ".doc":
            content, used_sidecar = _read_doc_with_sidecar(file, logger=logger)
            return (content if content else None), used_sidecar

        elif suffix == ".pdf":
            content, used_sidecar = _read_pdf_with_sidecar(file, logger=logger)
            return (content if content else None), used_sidecar

    except Exception as e:
        if logger:
            logger.warning(f"⚠️ 解析文件失败 {file.name}: {e}")
        return None, False

    if content:
        content = _normalize_content(content)

    return (content if content else None), used_sidecar
