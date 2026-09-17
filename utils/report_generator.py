"""
DeepShield — PDF Report Generator (ReportLab)
"""

import io
import base64
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ── Brand Colors ────────────────────────────────────────────────────────────
DARK_BG     = colors.HexColor("#0a0a1a")
CYAN        = colors.HexColor("#00d4ff")
PURPLE      = colors.HexColor("#7c3aed")
RED         = colors.HexColor("#ff4757")
GREEN       = colors.HexColor("#2ed573")
ORANGE      = colors.HexColor("#ffa502")
LIGHT_GRAY  = colors.HexColor("#e0e0e0")
MID_GRAY    = colors.HexColor("#888888")
WHITE       = colors.white


def _prediction_color(prediction: str) -> colors.Color:
    mapping = {"FAKE": RED, "REAL": GREEN, "SUSPICIOUS": ORANGE}
    return mapping.get(prediction.upper(), LIGHT_GRAY)


def _risk_color(risk: str) -> colors.Color:
    mapping = {"CRITICAL": RED, "HIGH": RED, "MEDIUM": ORANGE, "LOW": GREEN}
    return mapping.get(risk.upper(), LIGHT_GRAY)


def generate_pdf_report(doc: dict, user: dict) -> bytes:
    """
    Generate a professional PDF report for a detection result.
    Returns raw PDF bytes.
    """
    buffer = io.BytesIO()
    result = doc.get("result", {})
    prediction = doc.get("prediction", "UNKNOWN")
    confidence = doc.get("confidence", 0)
    risk_level = doc.get("risk_level", "UNKNOWN")
    media_type = doc.get("media_type", "unknown").upper()
    filename = doc.get("filename", "unknown")
    created_at = doc.get("created_at", datetime.utcnow())
    detection_id = doc.get("id", "N/A")

    # ── Document Setup ───────────────────────────────────────────────────────
    pdf = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
    )
    styles = getSampleStyleSheet()
    story = []

    # ── Header ───────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title", parent=styles["Normal"],
        fontSize=26, textColor=CYAN, spaceAfter=4,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=11, textColor=MID_GRAY, spaceAfter=2,
        fontName="Helvetica", alignment=TA_CENTER,
    )
    story.append(Paragraph("🛡️  DeepShield", title_style))
    story.append(Paragraph("Intelligent Synthetic Media Detection Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=CYAN, spaceAfter=12))

    # ── Meta Table ───────────────────────────────────────────────────────────
    meta_label = ParagraphStyle("MetaLabel", parent=styles["Normal"],
                                fontSize=9, textColor=MID_GRAY, fontName="Helvetica")
    meta_value = ParagraphStyle("MetaValue", parent=styles["Normal"],
                                fontSize=10, textColor=WHITE, fontName="Helvetica-Bold")

    if isinstance(created_at, datetime):
        date_str = created_at.strftime("%B %d, %Y at %H:%M UTC")
    else:
        date_str = str(created_at)

    meta_data = [
        [Paragraph("Detection ID", meta_label), Paragraph(str(detection_id), meta_value)],
        [Paragraph("Analyst", meta_label), Paragraph(user.get("name", "N/A"), meta_value)],
        [Paragraph("Date & Time", meta_label), Paragraph(date_str, meta_value)],
        [Paragraph("Filename", meta_label), Paragraph(filename, meta_value)],
        [Paragraph("Media Type", meta_label), Paragraph(media_type, meta_value)],
        [Paragraph("File Size", meta_label), Paragraph(f"{doc.get('file_size', 0):,} bytes", meta_value)],
    ]

    meta_table = Table(meta_data, colWidths=[4.5*cm, 13*cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#111128")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#222244")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#111128"), colors.HexColor("#0d0d20")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Verdict Banner ───────────────────────────────────────────────────────
    pred_color = _prediction_color(prediction)
    verdict_style = ParagraphStyle(
        "Verdict", fontSize=22, textColor=pred_color,
        fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4,
    )
    conf_style = ParagraphStyle(
        "Conf", fontSize=13, textColor=WHITE,
        fontName="Helvetica", alignment=TA_CENTER, spaceAfter=2,
    )
    story.append(Paragraph(f"VERDICT: {prediction}", verdict_style))
    story.append(Paragraph(f"Confidence Score: {confidence:.1f}%", conf_style))
    story.append(HRFlowable(width="100%", thickness=1, color=pred_color, spaceAfter=12))

    # ── Result Details Table ─────────────────────────────────────────────────
    section_style = ParagraphStyle(
        "Section", fontSize=13, textColor=CYAN,
        fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=6,
    )
    cell_l = ParagraphStyle("CL", fontSize=10, textColor=MID_GRAY, fontName="Helvetica")
    cell_r = ParagraphStyle("CR", fontSize=10, textColor=WHITE, fontName="Helvetica-Bold")

    story.append(Paragraph("Detection Results", section_style))

    fake_prob = result.get("fake_probability", 0)
    real_prob = result.get("real_probability", 0)
    proc_time = result.get("processing_time", 0)
    model_used = result.get("model_used", "EfficientNet-B4")

    result_rows = [
        [Paragraph("Prediction", cell_l), Paragraph(prediction, ParagraphStyle("P", fontSize=10, textColor=pred_color, fontName="Helvetica-Bold"))],
        [Paragraph("Risk Level", cell_l), Paragraph(risk_level, ParagraphStyle("R", fontSize=10, textColor=_risk_color(risk_level), fontName="Helvetica-Bold"))],
        [Paragraph("Fake Probability", cell_l), Paragraph(f"{fake_prob * 100:.2f}%", cell_r)],
        [Paragraph("Real Probability", cell_l), Paragraph(f"{real_prob * 100:.2f}%", cell_r)],
        [Paragraph("Confidence Score", cell_l), Paragraph(f"{confidence:.2f}%", cell_r)],
        [Paragraph("Processing Time", cell_l), Paragraph(f"{proc_time:.3f} seconds", cell_r)],
        [Paragraph("AI Model", cell_l), Paragraph(model_used, cell_r)],
    ]

    # Media-type specific rows
    if media_type == "IMAGE":
        face = result.get("face_detected", False)
        result_rows.append([Paragraph("Face Detected", cell_l), Paragraph("Yes" if face else "No", cell_r)])
    elif media_type == "VIDEO":
        frames = result.get("frames_analyzed", 0)
        result_rows.append([Paragraph("Frames Analyzed", cell_l), Paragraph(str(frames), cell_r)])
    elif media_type == "AUDIO":
        dur = result.get("duration_seconds", 0)
        result_rows.append([Paragraph("Audio Duration", cell_l), Paragraph(f"{dur:.2f} seconds", cell_r)])

    result_table = Table(result_rows, colWidths=[6*cm, 11.5*cm])
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#111128")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#111128"), colors.HexColor("#0d0d20")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#222244")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.4*cm))

    # ── AI Explanation ───────────────────────────────────────────────────────
    story.append(Paragraph("AI Explanation", section_style))
    explanation = result.get("ai_explanation", "No explanation available.")
    exp_style = ParagraphStyle(
        "Exp", fontSize=10, textColor=LIGHT_GRAY,
        fontName="Helvetica", leading=16, spaceAfter=8,
    )
    exp_table = Table(
        [[Paragraph(explanation, exp_style)]],
        colWidths=[17.5*cm],
    )
    exp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0d1127")),
        ("BOX", (0, 0), (-1, -1), 1, CYAN),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(exp_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Heatmap / Spectrogram ────────────────────────────────────────────────
    heatmap_b64 = result.get("heatmap_b64") or result.get("spectrogram_b64")
    if heatmap_b64:
        story.append(Paragraph("Visual Analysis", section_style))
        try:
            img_data = base64.b64decode(heatmap_b64)
            img_io = io.BytesIO(img_data)
            rl_img = RLImage(img_io, width=14*cm, height=6*cm)
            story.append(rl_img)
        except Exception:
            pass
        story.append(Spacer(1, 0.4*cm))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=PURPLE, spaceBefore=12, spaceAfter=6))
    footer_style = ParagraphStyle(
        "Footer", fontSize=8, textColor=MID_GRAY,
        fontName="Helvetica", alignment=TA_CENTER,
    )
    story.append(Paragraph(
        "This report was generated by DeepShield — AI-Powered Synthetic Media Detection Platform. "
        "For informational purposes only. Always combine AI analysis with expert human review.",
        footer_style,
    ))
    story.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | "
        f"DeepShield v1.0.0 | B.Tech CSE Final Year Project",
        footer_style,
    ))

    pdf.build(story)
    return buffer.getvalue()
