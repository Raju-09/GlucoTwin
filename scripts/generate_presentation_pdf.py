"""Generate 10-slide master pitch deck presentation PDF for Happiest Health Digital Twin Challenge 2026 using ReportLab."""

from pathlib import Path
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def build_presentation_pdf(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(letter),
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0D47A1")     # Deep Blue
    SECONDARY = colors.HexColor("#1976D2")   # Medium Blue
    ACCENT = colors.HexColor("#2E7D32")      # Emerald Green
    TEXT_DARK = colors.HexColor("#212121")   # Charcoal
    TEXT_MUTED = colors.HexColor("#616161")  # Slate Gray
    BG_LIGHT = colors.HexColor("#F5F7FA")    # Off-white

    title_slide_style = ParagraphStyle(
        "TitleSlideHeader",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=34,
        textColor=PRIMARY,
        alignment=1,
    )

    subtitle_style = ParagraphStyle(
        "TitleSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=14,
        leading=18,
        textColor=TEXT_MUTED,
        alignment=1,
    )

    slide_header_style = ParagraphStyle(
        "SlideHeader",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
    )

    subhead_style = ParagraphStyle(
        "SubHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=SECONDARY,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=TEXT_DARK,
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_DARK,
        leftIndent=15,
    )

    callout_style = ParagraphStyle(
        "CalloutBox",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=14,
        textColor=PRIMARY,
        backColor=colors.HexColor("#E3F2FD"),
        borderPadding=8,
    )

    story = []

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("GlucoTwin", title_slide_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("A Reliability-Aware Digital Twin for 120-Minute Glucose Forecasting & Adverse Event Mitigation", subtitle_style))
    story.append(Spacer(1, 0.4 * inch))
    
    meta_data = [
        [Paragraph("<b>Competition:</b> Happiest Health — Digital Twin Challenge 2026", body_style)],
        [Paragraph("<b>Summit:</b> Reimagining & Reforming Healthcare in India Summit 2026, Bengaluru", body_style)],
        [Paragraph("<b>Team:</b> Team Synapse  •  <b>Team Leader:</b> Sammeta Siva Prasad Raju  •  <b>Member:</b> Bedadakota Devi Chandrika", body_style)],
        [Paragraph("<b>Repository:</b> https://github.com/Raju-09/GlucoTwin  •  <b>License:</b> MIT Open Source", body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[9.5 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#BBDEFB")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 2: Healthcare Challenge in India
    # -------------------------------------------------------------
    story.append(Paragraph("1. The Healthcare Challenge in India", slide_header_style))
    story.append(Paragraph("101 Million Patients: The Critical Need for Proactive Glycemic Management", subhead_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("• <b>Global Diabetes Epicenter:</b> India has over 101 million diagnosed individuals with diabetes (IDF Diabetes Atlas, 2023), with millions undiagnosed in tier-2/3 cities and rural hubs.", bullet_style))
    story.append(Paragraph("• <b>The Fatal Flaw of Modern CGMs:</b> Continuous Glucose Monitors generate readings every 5 minutes, but current alert algorithms are entirely <i>reactive</i>. They fire alarms only after blood glucose breaches dangerous thresholds (&lt;70 mg/dL for hypoglycemia, &gt;180 mg/dL for hyperglycemia), when metabolic stress has already commenced.", bullet_style))
    story.append(Paragraph("• <b>The Proactive 2-Hour Window:</b> A validated 120-minute forward forecast provides actionable lead time for dietary adjustment, physical activity, or preemptive clinical triage before acute excursions occur.", bullet_style))
    story.append(Paragraph("• <b>Data Privacy Sandbox Compliance:</b> Built and evaluated using in-silico synthetic cohorts and open-source formats in strict compliance with India's DPDP Act and global HIPAA frameworks.", bullet_style))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("<i>'Reactive alarms notify patients of emergencies; proactive digital twins prevent them.'</i>", callout_style))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 3: Twin State Mathematical Formulation
    # -------------------------------------------------------------
    story.append(Paragraph("2. Mathematical Formulation: Patient Digital State", slide_header_style))
    story.append(Paragraph("Decoupling Static Phenotype from Dynamic Causal History", subhead_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("GlucoTwin formally represents the virtual patient at any discrete time step <i>t</i> as a triad:", body_style))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("<b>S(t) = ( θ_p ,  h_t ,  δ_t )</b>", title_slide_style))
    story.append(Spacer(1, 0.15 * inch))

    state_desc = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Physiological & Algorithmic Meaning</b>", body_style), Paragraph("<b>Features Included</b>", body_style)],
        [Paragraph("<b>θ_p<br/>Static Phenotype</b>", body_style), Paragraph("Longitudinal EHR context, clinical intake, demographic & baseline metabolic indicators.", body_style), Paragraph("Age, Sex, BMI, Baseline HbA1c, Diabetes Duration, Historical Fasting Glucose, Metformin/SGLT2i/Dawn Flags (9 feats)", body_style)],
        [Paragraph("<b>h_t<br/>Dynamic History</b>", body_style), Paragraph("Causal sliding window of sensor observations (120 min lookback = 24 steps at 5-min cadence).", body_style), Paragraph("CGM Lags 0..23, Rolling Mean/Std (1h, 2h), Discrete Velocity dG/dt, Acceleration, Circadian Sin/Cos, Wearable Telemetry", body_style)],
        [Paragraph("<b>δ_t<br/>Twin Integrity</b>", body_style), Paragraph("Real-time stream quality, latency, telemetry jitter, and fail-closed forecast gating state.", body_style), Paragraph("Integrity State (AVAILABLE, DEGRADED, WITHHELD), Reason Codes, Telemetry Freshness, Quality Score %", body_style)],
    ]
    st_table = Table(state_desc, colWidths=[1.8 * inch, 4.2 * inch, 3.8 * inch])
    st_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E3F2FD")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#90CAF9")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(st_table)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 4: End-to-End System Architecture
    # -------------------------------------------------------------
    story.append(Paragraph("3. End-to-End System Architecture", slide_header_style))
    story.append(Paragraph("Three-Tier Architecture: Ingestion -> Verification -> Serving", subhead_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("• <b>Tier 1 — Preprocessing & Sensor Physics:</b> Continuous CGM readings pass through an interstitial physiological delay model (τ=10 min), AR(1) autocorrelated sensor noise, and rate clamping (|dG/dt| ≤ 3.8 mg/dL/min) to reflect true subcutaneous biology.", bullet_style))
    story.append(Paragraph("• <b>Tier 2 — Causal Multimodal Feature Pipeline:</b> Combines 24 CGM lags, rolling kinematics, circadian encodings, and EHR intake into a 56-feature vector. Supervised training windows use strict chronological boundaries with a 120-minute gap buffer to guarantee zero future data leaks.", bullet_style))
    story.append(Paragraph("• <b>Tier 3 — HistGradientBoosting Core & Reliability Gating:</b> High-efficiency tree regressor trained on 224,681 windows, coupled to a finite-state machine that evaluates input admissibility before returning a prediction.", bullet_style))
    story.append(Paragraph("• <b>Tier 4 — REST API & Clinician Dashboard:</b> Sub-10ms FastAPI backend powering a Streamlit UI with 14-day timeline scrubbing, state inspector, and one-click failure injection.", bullet_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("<i>Full vector diagram available at: docs/architecture_diagram.pdf</i>", callout_style))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 5: The Core Differentiator — Reliability Gate
    # -------------------------------------------------------------
    story.append(Paragraph("4. Core Differentiator: Fail-Closed Reliability Gating", slide_header_style))
    story.append(Paragraph("The Digital Twin Knows When NOT to Trust Its Predictions", subhead_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("In clinical machine learning, an overconfident wrong prediction during hardware failure is catastrophic. GlucoTwin enforces fail-closed behavior across three formal states:", body_style))
    story.append(Spacer(1, 0.1 * inch))

    gate_data = [
        [Paragraph("<b>Status Level</b>", body_style), Paragraph("<b>Trigger Conditions</b>", body_style), Paragraph("<b>System Behavior</b>", body_style)],
        [Paragraph("<b>🟢 AVAILABLE</b>", body_style), Paragraph("Nominal stream, freshness &lt; 15 min, complete 24-step history, values in [40, 400] mg/dL.", body_style), Paragraph("120-min point forecast displayed with high-confidence green badge.", body_style)],
        [Paragraph("<b>🟡 DEGRADED</b>", body_style), Paragraph("Mild telemetry latency (15–20 min) or minor packet loss; historical context intact.", body_style), Paragraph("Forecast displayed alongside advisory warning flag & degraded reason code.", body_style)],
        [Paragraph("<b>🔴 WITHHELD</b>", body_style), Paragraph("Stale stream (&gt;20 min), severe gap (&gt;30 min), sensor dropout, or jump &gt; 4.0 mg/dL/min.", body_style), Paragraph("<b>Output suppressed. No number returned.</b> Explicit error code issued (e.g. ERR_SENSOR_DROPOUT).", body_style)],
    ]
    gt_table = Table(gate_data, colWidths=[2.0 * inch, 4.3 * inch, 3.5 * inch])
    gt_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E3F2FD")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#90CAF9")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(gt_table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("<b>Utility Analysis:</b> Our risk-coverage audit proves that withholding 12% of corrupted inputs suppresses severe prediction errors (>30 mg/dL) and protects clinical fidelity.", body_style))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 6: Multimodal Ablation Study Results
    # -------------------------------------------------------------
    story.append(Paragraph("5. Multimodal Ablation Study (N=60 Patients)", slide_header_style))
    story.append(Paragraph("Evaluated on 43,888 Held-Out Test Windows Under Strict Chronological Hold-Out", subhead_style))
    story.append(Spacer(1, 0.1 * inch))

    ablation_data = [
        [Paragraph("<b>Configuration</b>", body_style), Paragraph("<b>Input Streams & Features</b>", body_style), Paragraph("<b>Test MAE</b>", body_style), Paragraph("<b>Δ MAE vs Ref</b>", body_style), Paragraph("<b>95% Bootstrap CI</b>", body_style)],
        [Paragraph("<b>Persistence Baseline</b>", body_style), Paragraph("Naive y(t+120) = y(t)", body_style), Paragraph("53.25 mg/dL", body_style), Paragraph("+34.94 mg/dL", body_style), Paragraph("—", body_style)],
        [Paragraph("<b>Config A (CGM Only)</b>", body_style), Paragraph("24 Lags + Rolling Kinematics (41 feats)", body_style), Paragraph("18.31 mg/dL", body_style), Paragraph("Reference", body_style), Paragraph("—", body_style)],
        [Paragraph("<b>Config B (CGM + Static EHR)</b>", body_style), Paragraph("Demographics + Labs + Fasting Lab (50 feats)", body_style), Paragraph("<b>17.61 mg/dL</b>", body_style), Paragraph("<b>−0.70 mg/dL</b>", body_style), Paragraph("<b>[+0.49, +0.96]</b> (p &lt; 10⁻⁵)", body_style)],
        [Paragraph("<b>Config C (CGM + Wearables)</b>", body_style), Paragraph("Decoupled HR, HRV, Steps, Sleep (47 feats)", body_style), Paragraph("18.32 mg/dL", body_style), Paragraph("+0.01 mg/dL", body_style), Paragraph("[−0.08, +0.07] (p = 0.95)", body_style)],
        [Paragraph("<b>Config D (Full Two-Stream)</b>", body_style), Paragraph("EHR + CGM + Wearables (56 feats)", body_style), Paragraph("<b>17.67 mg/dL</b>", body_style), Paragraph("<b>−0.64 mg/dL</b>", body_style), Paragraph("[+0.44, +0.88]", body_style)],
    ]
    ab_table = Table(ablation_data, colWidths=[2.2 * inch, 3.2 * inch, 1.3 * inch, 1.4 * inch, 1.7 * inch])
    ab_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E3F2FD")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#90CAF9")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ALIGN', (2,1), (-1,-1), 'CENTER'),
    ]))
    story.append(ab_table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("• <b>EHR Gain:</b> Static EHR context provides a statistically validated error reduction (−0.70 mg/dL) across same-patient trajectories.<br/>• <b>Wearable Negative Finding:</b> Activity telemetry provided negligible incremental gain under this simulator; reported honestly rather than fabricating artificial gains.", body_style))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 7: Falsification & The LOPO Benchmark
    # -------------------------------------------------------------
    story.append(Paragraph("6. Breaking Our Own Claims: Leave-One-Patient-Out (LOPO)", slide_header_style))
    story.append(Paragraph("Cross-Validation Across 60 Patients (224,681 Windows)", subhead_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("To test whether the model learned generalizable human physiology or merely memorized synthetic patient identity, we executed a complete Leave-One-Patient-Out (LOPO) cross-validation and a permuted-EHR control:", body_style))
    story.append(Spacer(1, 0.1 * inch))

    lopo_data = [
        [Paragraph("<b>Evaluation Mode</b>", body_style), Paragraph("<b>Mean LOPO MAE</b>", body_style), Paragraph("<b>Delta vs CGM-Only</b>", body_style), Paragraph("<b>Scientific Verdict</b>", body_style)],
        [Paragraph("<b>CGM Only</b>", body_style), Paragraph("17.65 mg/dL", body_style), Paragraph("Reference", body_style), Paragraph("Solid kinematic time-series tracking.", body_style)],
        [Paragraph("<b>CGM + Real EHR</b>", body_style), Paragraph("<b>17.65 mg/dL</b>", body_style), Paragraph("+0.01 mg/dL (p = 0.956)", body_style), Paragraph("<b>No transfer benefit on unseen patients.</b> EHR acts as same-patient anchor.", body_style)],
        [Paragraph("<b>CGM + Permuted EHR</b>", body_style), Paragraph("<b>20.47 mg/dL</b>", body_style), Paragraph("<b>+2.82 mg/dL degradation</b>", body_style), Paragraph("<b>Catastrophic failure.</b> Confirms model relies on true patient identity.", body_style)],
    ]
    lp_table = Table(lopo_data, colWidths=[2.2 * inch, 1.8 * inch, 2.2 * inch, 3.6 * inch])
    lp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E3F2FD")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#90CAF9")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(lp_table)
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<i>'A system that knows the boundary of its own evidence is far more trustworthy in healthcare than one claiming 99% universal accuracy.'</i>", callout_style))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 8: Clinical Adverse Event Prediction
    # -------------------------------------------------------------
    story.append(Paragraph("7. Clinical Adverse Event Classification (t+120 min)", slide_header_style))
    story.append(Paragraph("Catching Postprandial Spikes 2 Hours Ahead & Suppressing Alarm Fatigue", subhead_style))
    story.append(Spacer(1, 0.1 * inch))

    ev_data = [
        [Paragraph("<b>Adverse Event</b>", body_style), Paragraph("<b>Model</b>", body_style), Paragraph("<b>Events (N)</b>", body_style), Paragraph("<b>Sensitivity</b>", body_style), Paragraph("<b>Specificity</b>", body_style), Paragraph("<b>Precision</b>", body_style), Paragraph("<b>F1-Score</b>", body_style), Paragraph("<b>False Alarms/Day</b>", body_style)],
        [Paragraph("Hyperglycemia (>180 mg/dL)", body_style), Paragraph("Persistence Baseline", body_style), Paragraph("12,246", body_style), Paragraph("26.6%", body_style), Paragraph("70.4%", body_style), Paragraph("25.8%", body_style), Paragraph("0.262", body_style), Paragraph("61.4", body_style)],
        [Paragraph("Hyperglycemia (>180 mg/dL)", body_style), Paragraph("<b>GlucoTwin (Two-Stream)</b>", body_style), Paragraph("12,246", body_style), Paragraph("<b>76.2%</b>", body_style), Paragraph("<b>90.6%</b>", body_style), Paragraph("<b>75.8%</b>", body_style), Paragraph("<b>0.760</b>", body_style), Paragraph("<b>19.6 (−68%)</b>", body_style)],
        [Paragraph("Hypoglycemia (<70 mg/dL)", body_style), Paragraph("Persistence Baseline", body_style), Paragraph("71", body_style), Paragraph("12.7%", body_style), Paragraph("99.9%", body_style), Paragraph("12.7%", body_style), Paragraph("0.127", body_style), Paragraph("0.4", body_style)],
        [Paragraph("Hypoglycemia (<70 mg/dL)", body_style), Paragraph("<b>GlucoTwin (Two-Stream)</b>", body_style), Paragraph("71", body_style), Paragraph("0.0%", body_style), Paragraph("100.0%", body_style), Paragraph("0.0%", body_style), Paragraph("0.000", body_style), Paragraph("0.0", body_style)],
    ]
    ev_table = Table(ev_data, colWidths=[2.0 * inch, 1.8 * inch, 0.9 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 0.9 * inch, 1.2 * inch])
    ev_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E3F2FD")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#90CAF9")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ALIGN', (2,1), (-1,-1), 'CENTER'),
    ]))
    story.append(ev_table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("• <b>Postprandial Excursions:</b> Catches nearly 3× more hyperglycemic events 2 hours in advance while reducing false alarms from 61.4 to 19.6/day (a <b>68% reduction</b>).<br/>• <b>Hypoglycemia Tail Limitation:</b> In adult T2D, hypoglycemia is a rare tail event (&lt;0.2% prevalence). Standard MSE regression regresses towards the mean (~150 mg/dL) and misses rare nadirs below 70 mg/dL; documented as a future milestone for asymmetric loss weighting.", body_style))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 9: Software Engineering & Dashboard Maturity
    # -------------------------------------------------------------
    story.append(Paragraph("8. Clinician Dashboard & Engineering Maturity", slide_header_style))
    story.append(Paragraph("Production-Quality Engineering Ready for Triage", subhead_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("• <b>Doctor's Interactive Dashboard:</b> Streamlit-based UI featuring 14-day continuous time scrubbing across 60 patients, clear data provenance badges ([Observed EHR], [Sensor], [Kinematic], [Model]), and unclamped discrete velocity tracking.", bullet_style))
    story.append(Paragraph("• <b>Live Telemetry Stress Testing:</b> Real-time sidebar allows clinicians and evaluators to inject sensor dropouts, 20-minute latency delays, or non-physiological spikes (400 mg/dL) to verify fail-closed gating behavior live.", bullet_style))
    story.append(Paragraph("• <b>Sub-10ms Serving Latency:</b> Measured live over 100 consecutive API inferences on commodity hardware: <b>Median: 7.79 ms, Mean: 8.11 ms, p95: 11.14 ms, p99: 13.26 ms</b>.", bullet_style))
    story.append(Paragraph("• <b>Containerized Deployment:</b> Fully reproducible via <code>docker-compose up</code> in under 60 seconds with 69 unit tests passing with 100% test coverage.", bullet_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<i>Live demo available locally via: streamlit run frontend/streamlit_app.py</i>", callout_style))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 10: Responsible AI, Limitations & Roadmap
    # -------------------------------------------------------------
    story.append(Paragraph("9. Responsible AI, Limitations & Clinical Translation", slide_header_style))
    story.append(Paragraph("Transparent Boundaries and the Path to Indian Healthcare Impact", subhead_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("• <b>Explicit Non-Goals:</b> Research prototype; NOT an FDA/CDSCO-cleared medical device. Evaluated on in-silico synthetic cohorts and must not be used to direct autonomous insulin dosing or clinical therapy.", bullet_style))
    story.append(Paragraph("• <b>Phase 1 (Completed):</b> In-silico digital twin architecture, multi-stream feature engineering, fail-closed reliability FSM, and LOPO cross-validation.", bullet_style))
    story.append(Paragraph("• <b>Phase 2 (Next Milestone):</b> Prospective transfer validation on clinical research datasets (DiaTrend, OhioT1DM) under data transfer agreements.", bullet_style))
    story.append(Paragraph("• <b>Phase 3 (Long-Term Vision):</b> Low-cost edge deployment in Indian primary healthcare centers (PHCs) to assist community physicians in proactive lifestyle counseling.", bullet_style))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("<b>GlucoTwin: Forecast Earlier. Intervene Smarter. Fail Closed When Uncertain.</b>", title_slide_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Thank you. Team Synapse — Sammeta Siva Prasad Raju & Bedadakota Devi Chandrika", subtitle_style))

    doc.build(story)
    print(f"Presentation PDF generated at: {output_path}")

if __name__ == "__main__":
    out_pdf = Path("docs/glucotwin_presentation.pdf")
    build_presentation_pdf(out_pdf)
