import cv2
import json
import tempfile
import gradio as gr
import numpy as np
from pathlib import Path
from fpdf import FPDF
from src.inference import DamageDetectionEngine

PROJECT_ROOT = Path(__file__).resolve().parent

print(f"{'-'*50}\nInitializing Gradio Application\n{'-'*50}")

try:
    engine = DamageDetectionEngine()
except Exception as e:
    print(f"Failed to initialize DamageDetectionEngine: {e}")
    engine = None


def generate_pdf_report(image_bgr: np.ndarray, report_dict: dict) -> str:
    """
    Generates a PDF document containing the annotated image and a structured text summary of the damages.

    Args:
        image_bgr (np.ndarray): The annotated image in BGR format directly from the inference engine.
        report_dict (dict): The raw JSON output from the DamageDetectionEngine.

    Returns:
        str: The absolute file path to the generated temporary PDF file, or None if generation fails.
    """
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(
        0, 10, "AI Car Damage Detection System - Assessment Report", ln=True, align="C"
    )
    pdf.ln(5)

    if image_bgr is not None:
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_img.name, image_bgr)
        pdf.image(temp_img.name, x=15, w=180)
        pdf.ln(10)

    pdf.set_font("Helvetica", size=12)

    if not report_dict:
        pdf.cell(0, 10, "No damages detected or image could not be analyzed.", ln=True)
    else:
        total_damages = sum(len(damages) for damages in report_dict.values())
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, f"Total Damages Detected: {total_damages}", ln=True)
        pdf.ln(5)

        for part, damages in report_dict.items():
            clean_part_name = part.replace("_", " ").title()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, f"Part: {clean_part_name}", ln=True)

            pdf.set_font("Helvetica", size=12)
            for d in damages:
                sev = d["severity"]
                part_cov = (
                    f"{d['part_area_percentage']}%"
                    if d["part_area_percentage"] != "N/A"
                    else "N/A"
                )
                car_cov = f"{d['car_area_percentage']}%"

                line = f"- {d['type'].title()} [{sev}] | Part Coverage: {part_cov} | Total Car Coverage: {car_cov}"
                pdf.cell(0, 8, line, ln=True)
            pdf.ln(5)

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name


def format_report_to_text(report_dict: dict) -> str:
    """
    Converts the raw JSON report into a highly readable, professional Markdown table layout.

    Args:
        report_dict (dict): The raw JSON output from the DamageDetectionEngine containing
                            part names as keys and lists of damage assessments as values.

    Returns:
        str: A formatted Markdown string containing a summary and detailed tables for each part.
    """
    if not report_dict:
        return "✅ **No damages detected** or image could not be analyzed."

    total_damages = sum(len(damages) for damages in report_dict.values())

    summary = f"### 📋 Damage Assessment Summary\n"
    summary += f"**Total Damages Detected:** {total_damages}\n\n---\n\n"

    for part, damages in report_dict.items():
        clean_part_name = part.replace("_", " ").title()
        summary += f"#### 🚘 {clean_part_name}\n\n"

        summary += "| Damage Type | Severity | Part Coverage | Total Car Coverage |\n"
        summary += "| :--- | :--- | :--- | :--- |\n"

        for d in damages:
            sev = d["severity"]
            if "Critical" in sev:
                icon = "🔴"
            elif "Heavy" in sev:
                icon = "🟠"
            elif "Medium" in sev:
                icon = "🟡"
            else:
                icon = "🟢"

            part_cov = (
                f"{d['part_area_percentage']}%"
                if d["part_area_percentage"] != "N/A"
                else "N/A"
            )
            car_cov = f"{d['car_area_percentage']}%"

            summary += (
                f"| **{d['type'].title()}** | {sev} {icon} | {part_cov} | {car_cov} |\n"
            )

        summary += "\n<br>\n\n"

    return summary


def analyze(image_path: str, progress=gr.Progress()) -> tuple:
    """
    Processes the input image through the inference engine, formats text, and prepares downloadable files.

    Args:
        image_path (str): The absolute or relative path to the uploaded or selected image.
        progress (gr.Progress, optional): Gradio progress tracking object. Defaults to gr.Progress().

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray or None: The annotated RGB image.
            - str: The formatted Markdown report.
            - str or None: The file path to the temporary PDF file for download.
            - dict or None: The raw JSON dictionary report.
            - str or None: The file path to the temporary JSON file for download.
    """
    progress(0.1, desc="Analyzing car body and damages...")

    if not image_path or engine is None:
        return None, "No image provided.", None, None, None

    annotated_img_bgr, report = engine.predict(image_path)

    annotated_img_rgb = None
    if annotated_img_bgr is not None:
        annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)

    progress(0.6, desc="Formatting professional report...")
    text_report = format_report_to_text(report)

    progress(0.8, desc="Generating PDF and JSON files...")
    pdf_path = generate_pdf_report(annotated_img_bgr, report)

    temp_json = tempfile.NamedTemporaryFile(
        delete=False, suffix=".json", mode="w", encoding="utf-8"
    )
    json.dump(report, temp_json, indent=4, ensure_ascii=False)
    temp_json.close()

    progress(1.0, desc="Done!")

    return annotated_img_rgb, text_report, pdf_path, report, temp_json.name


demo_paths = [
    PROJECT_ROOT / "assets" / "demo-images" / "kia-picanto" / "kia_picanto_2.png",
    PROJECT_ROOT / "assets" / "demo-images" / "vw-golf" / "vw_golf_1.jpg",
    PROJECT_ROOT / "assets" / "demo-images" / "suzuki-swift" / "suzuki_swift_2.png",
]

demo_images = [str(p) for p in demo_paths if p.exists()]


with gr.Blocks() as app:
    gr.Markdown("# AI Car Damage Detection System 🚗")
    gr.Markdown(
        "Upload a car photo or select a demo image below for automatic damage analysis and severity assessment."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="filepath", label="Upload Photo", sources=["upload", "clipboard"]
            )

            if demo_images:
                gr.Examples(
                    examples=demo_images,
                    inputs=input_image,
                    label="Quick Test (Demo Images)",
                )

            with gr.Row():
                clear_btn = gr.ClearButton(value="🗑️ Clear", components=[input_image])
                submit_btn = gr.Button("🔍 Analyze", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Annotated Result", interactive=False)

            with gr.Tabs():
                with gr.TabItem("📄 Summary Report"):
                    output_text = gr.Markdown(label="Readable Report")
                    download_pdf_btn = gr.DownloadButton("📥 Download PDF Report")

                with gr.TabItem("⚙️ Raw JSON"):
                    output_json = gr.JSON(label="System Data")
                    download_json_btn = gr.DownloadButton("📥 Download JSON Report")

    clear_btn.add(
        [output_image, output_text, download_pdf_btn, output_json, download_json_btn]
    )

    submit_btn.click(
        fn=analyze,
        inputs=input_image,
        outputs=[
            output_image,
            output_text,
            download_pdf_btn,
            output_json,
            download_json_btn,
        ],
    )

if __name__ == "__main__":
    print("\nStarting Gradio web server...")
    app.launch(theme=gr.themes.Soft())
