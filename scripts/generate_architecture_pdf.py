"""Generate high-resolution vector PDF of GlucoTwin System Architecture."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_architecture_diagram(output_path: Path):
    fig, ax = plt.subplots(figsize=(14, 8.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Background canvas
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Title & Subtitle
    ax.text(50, 95, "GlucoTwin: System Architecture & Dual-Stream Pipeline", 
            ha="center", va="center", fontsize=18, fontweight="bold", color="#0D47A1")
    ax.text(50, 91.5, "A Reliability-Aware Digital Twin for 120-Minute Glucose Forecasting & Adverse Event Mitigation", 
            ha="center", va="center", fontsize=11, color="#424242", style="italic")

    # Box styles
    def draw_box(x, y, w, h, title, subtitle, bg_color, border_color, title_color="#0D47A1"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.8,rounding_size=1.5",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=1.8,
            zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.5, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=title_color, zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h/2 - 1.2, subtitle, ha="center", va="center", fontsize=8.5, color="#37474F", zorder=3, linespacing=1.3)

    # 1. Input Layer: Two Streams
    draw_box(4, 62, 26, 22, "STATIC STREAM (EHR / θ_p)", 
             "• Demographics (Age, Sex, BMI)\n• Clinical Labs (HbA1c, Fasting Glucose)\n• Disease Duration & Medications (Metformin, SGLT2i)\n• Dawn Phenomenon Flag (9 Phenotype features)", 
             "#E3F2FD", "#1976D2")

    draw_box(4, 30, 26, 24, "DYNAMIC STREAM (CGM & Telemetry / h_t)", 
             "• 5-min Continuous Glucose Monitor (CGM)\n• Sensor Physics Layer (Interstitial Lag τ=10m)\n• Rate Clamp (|dG/dt| ≤ 3.8 mg/dL/min)\n• Wearable Context (Heart Rate, HRV, Steps, Sleep)\n• Lookback Horizon: 24 readings (120 mins)", 
             "#E8F5E9", "#388E3C", title_color="#1B5E20")

    # 2. Ingestion & Causal Window Assembly
    draw_box(36, 42, 25, 36, "DIGITAL TWIN CAUSAL STATE ENGINE\nS(t) = (θ_p, h_t, δ_t)", 
             "• Strict Chronological Hold-Out Alignment\n• 120-Min Future Horizon Boundary Buffer\n• Kinematic Feature Extraction (56 features):\n  - Lags 0..23, Rolling Mean/Std (1h, 2h)\n  - Discrete Velocity dG/dt & Acceleration\n  - Circadian Sin/Cos Encoding\n  - In-Target Range Metric (70-180 mg/dL)", 
             "#FFF8E1", "#FFA000", title_color="#E65100")

    # 3. Model Inference & Gating FSM
    draw_box(68, 56, 28, 25, "PREDICTIVE MODEL CORE", 
             "• HistGradientBoostingRegressor (Two-Stream)\n• 120-Minute Point Forecast at t+120 min\n• Multimodal Feature Fusion (EHR + CGM)\n• Test MAE: 17.61 mg/dL (vs 53.25 Persistence)\n• Adverse Event Engine: 76.2% Sensitivity on >180 mg/dL", 
             "#EDE7F6", "#512DA8", title_color="#311B92")

    draw_box(68, 22, 28, 25, "RELIABILITY & SAFETY GATE (δ_t)", 
             "• Real-Time Finite State Machine (FSM):\n  - AVAILABLE: Nominal telemetry & complete context\n  - DEGRADED: Jitter or latency (15-20m delay)\n  - WITHHELD: Stale (>20m), Gap (>30m), Spike (>4 mg/dL/m)\n• Fail-Closed: Suppresses ungrounded output", 
             "#FFEBEE", "#D32F2F", title_color="#B71C1C")

    # 4. Serving & Dashboard Layer
    draw_box(20, 4, 60, 12, "CLINICIAN INTERFACE & SERVING INFRASTRUCTURE", 
             "FastAPI REST Endpoints (/forecast, /twin/{patient_id}/state, /health)  •  Median Latency: 7.8 ms (p95: 11.1 ms)\nStreamlit Interactive Dashboard  •  14-Day Timeline Scrubbing  •  One-Click Stress Testing  •  Docker Compose", 
             "#F3E5F5", "#7B1FA2", title_color="#4A148C")

    # Arrows (Connectors)
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            label,
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor="#455A64", edgecolor="#455A64", width=1.5, headwidth=6, shrink=0.02),
            ha="center", va="center", fontsize=7.5, color="#263238", fontweight="semibold", zorder=4
        )

    # Connections
    draw_arrow(30, 73, 36, 65, "Static EHR")
    draw_arrow(30, 42, 36, 50, "Causal CGM")
    draw_arrow(61, 68, 68, 68, "56 Tabular Feats")
    draw_arrow(61, 50, 68, 35, "Stream Integrity")
    draw_arrow(82, 56, 82, 47, "Gating Feedback")
    draw_arrow(82, 22, 70, 16, "Status / Alert")
    draw_arrow(75, 56, 60, 16, "120-min Forecast")

    # Footer note
    ax.text(50, 1.2, "Team Synapse (Mohan Babu University)  •  Happiest Health Digital Twin Challenge 2026  •  MIT License", 
            ha="center", va="center", fontsize=8.5, color="#78909C")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Architecture diagram PDF generated at: {output_path}")

if __name__ == "__main__":
    out_pdf = Path("docs/architecture_diagram.pdf")
    create_architecture_diagram(out_pdf)
