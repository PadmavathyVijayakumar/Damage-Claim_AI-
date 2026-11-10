"""
Label & OCR module — Layer 3 of the 4-layer dock camera AI stack.

Reads shipping labels, expiry dates, lot numbers, and SSCC barcodes from
Zone C camera feeds. Outputs are matched against the inbound ASN / PO in
real time to flag SKU mismatches, expired product, or label damage before
putaway.

Engines:
  - PaddleOCR  — multi-orientation dock labels (primary, free/open source)
  - AWS Textract — structured document fields (secondary, cloud)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LabelData:
    """Extracted label fields from one label scan."""

    sscc: Optional[str] = None           # GS1 SSCC-18 pallet barcode
    sku: Optional[str] = None
    lot_number: Optional[str] = None
    expiry_date: Optional[str] = None    # ISO 8601
    po_number: Optional[str] = None
    carrier_name: Optional[str] = None
    seal_number: Optional[str] = None    # trailer seal for Chain-of-Custody
    raw_text: str = ""
    confidence: float = 0.0
    ocr_engine: str = "paddle"
    warnings: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True when the minimum required fields for WMS matching are present."""
        return bool(self.sscc and self.sku)


class PaddleOCRReader:
    """
    PaddleOCR wrapper for dock label reading.

    PaddleOCR handles:
      - Multi-orientation text (rotated labels are common at dock)
      - Multi-language (some carrier labels include non-Latin characters)
      - Mixed structured + unstructured text on the same label

    Model: PP-OCRv4 (default paddle model, strong on warehouse labels).
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self._reader = None

    def _load(self):
        if self._reader is not None:
            return
        try:
            from paddleocr import PaddleOCR

            self._reader = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
            )
            logger.info("PaddleOCR loaded (lang=%s, gpu=%s)", self.lang, self.use_gpu)
        except ImportError:
            logger.warning("PaddleOCR not installed — OCR will return empty results")

    def read(self, image: np.ndarray) -> list[tuple[list, tuple[str, float]]]:
        """Return raw PaddleOCR result list."""
        self._load()
        if self._reader is None:
            return []
        result = self._reader.ocr(image, cls=True)
        return result[0] if result else []


class AWSTextractReader:
    """
    AWS Textract wrapper for structured document fields.

    Used as a secondary engine when PaddleOCR confidence is low or when
    reading printed manifests / BOL documents attached to shipments.
    """

    def __init__(self, region: str = "us-east-1") -> None:
        self.region = region
        self._client = None

    def _load(self):
        if self._client is not None:
            return
        import boto3

        self._client = boto3.client("textract", region_name=self.region)

    def read_bytes(self, image_bytes: bytes) -> list[dict]:
        """Call Textract DetectDocumentText and return block list."""
        self._load()
        response = self._client.detect_document_text(
            Document={"Bytes": image_bytes}
        )
        return response.get("Blocks", [])


class LabelOCRPipeline:
    """
    Orchestrates PaddleOCR + optional Textract and parses extracted text
    into structured LabelData fields.
    """

    # GS1 SSCC prefix pattern — 18 digits starting with (00)
    SSCC_PREFIX = "(00)"

    def __init__(
        self,
        use_gpu: bool = False,
        textract_region: str = "us-east-1",
        textract_fallback: bool = False,
    ) -> None:
        self.paddle = PaddleOCRReader(use_gpu=use_gpu)
        self.textract = AWSTextractReader(region=textract_region) if textract_fallback else None
        self.textract_fallback = textract_fallback

    def read_label(self, image: np.ndarray) -> LabelData:
        """
        Run OCR on a cropped label image and return structured LabelData.

        Args:
            image: HxWx3 numpy array of the label crop from Zone C camera.

        Returns:
            LabelData with parsed fields and confidence score.
        """
        lines: list[str] = []
        confidences: list[float] = []

        raw_results = self.paddle.read(image)
        for item in raw_results:
            if item and len(item) >= 2:
                text, conf = item[1]
                lines.append(text.strip())
                confidences.append(float(conf))

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        raw_text = " | ".join(lines)

        label = LabelData(
            raw_text=raw_text,
            confidence=avg_conf,
            ocr_engine="paddle",
        )
        self._parse_fields(lines, label)

        # Fall back to Textract if confidence is low
        if avg_conf < 0.75 and self.textract_fallback and self.textract:
            logger.debug("Low PaddleOCR confidence (%.2f) — invoking Textract fallback", avg_conf)
            import cv2

            _, buf = cv2.imencode(".jpg", image)
            blocks = self.textract.read_bytes(buf.tobytes())
            textract_lines = [
                b["Text"]
                for b in blocks
                if b.get("BlockType") == "LINE" and "Text" in b
            ]
            self._parse_fields(textract_lines, label)
            label.ocr_engine = "paddle+textract"

        return label

    # ── Parsing helpers ───────────────────────────────────────────────────

    def _parse_fields(self, lines: list[str], label: LabelData) -> None:
        import re

        for line in lines:
            upper = line.upper().strip()

            # SSCC-18 — (00) + 17 digits
            if self.SSCC_PREFIX in upper:
                digits = re.sub(r"\D", "", line)
                if len(digits) == 18:
                    label.sscc = digits

            # SKU / item number
            if re.match(r"(ITEM|SKU|PART)[:\s#]+", upper):
                label.sku = re.sub(r"[^A-Z0-9\-]", "", upper.split()[-1])

            # Lot number
            if re.match(r"LOT[:\s#]", upper):
                label.lot_number = line.split()[-1].strip()

            # Expiry date — various formats
            date_match = re.search(
                r"\b(EXP|EXPIRY|BEST BY|BBD|USE BY)[:\s]*([\d]{2}[/\-][\d]{2}[/\-][\d]{2,4})\b",
                upper,
            )
            if date_match:
                label.expiry_date = date_match.group(2)

            # PO number
            if re.match(r"P\.?O\.?[:\s#]", upper):
                label.po_number = line.split()[-1].strip()

            # Seal number (trailer)
            if re.match(r"SEAL[:\s#]", upper):
                label.seal_number = line.split()[-1].strip()
