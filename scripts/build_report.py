"""Build the public-data project writeup from evaluated metrics and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BLUE = "24476B"
PALE_BLUE = "EAF1F7"
GRAY = "555555"


def set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def set_cell_text(cell, text: str, *, bold: bool = False, color: str = "222222") -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.name = "Arial"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = True
    for cell, header in zip(table.rows[0].cells, headers):
        set_cell_text(cell, header, bold=True, color="FFFFFF")
        set_cell_shading(cell, BLUE)
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            set_cell_text(cell, value)
            if row_index % 2 == 1:
                set_cell_shading(cell, "F4F7FA")
    document.add_paragraph().paragraph_format.space_after = Pt(0)


def add_body(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="Normal")
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.06
    paragraph.add_run(text)


def add_heading(document: Document, text: str, level: int = 1) -> None:
    paragraph = document.add_paragraph(text, style=f"Heading {level}")
    paragraph.paragraph_format.keep_with_next = True


def add_figure(document: Document, path: Path, caption: str, width: float = 6.0) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.add_run().add_picture(str(path), width=Inches(width))
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.space_after = Pt(8)
    run = caption_paragraph.add_run(caption)
    run.italic = True
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor.from_string(GRAY)


def style_document(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)

    normal = document.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor(34, 34, 34)
    for name, size in [("Heading 1", 15), ("Heading 2", 12)]:
        style = document.styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(BLUE)
        style.paragraph_format.space_before = Pt(9)
        style.paragraph_format.space_after = Pt(4)
    title = document.styles["Title"]
    title_properties = title._element.get_or_add_pPr()
    title_border = title_properties.find(qn("w:pBdr"))
    if title_border is not None:
        title_properties.remove(title_border)
    title.font.name = "Arial"
    title.font.size = Pt(24)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)
    title.paragraph_format.space_after = Pt(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "public_shot_model_report.docx"
    )
    args = parser.parse_args()

    for required in [
        REPORTS / "metrics_2023.json",
        REPORTS / "metrics_2024.json",
        REPORTS / "make_rate_by_distance.png",
        REPORTS / "make_rate_by_shot_type.png",
        REPORTS / "feature_importance.png",
        REPORTS / "calibration_2024.png",
    ]:
        if not required.exists():
            raise SystemExit(f"Missing analysis output: {required}")

    document = Document()
    style_document(document)

    title = document.add_paragraph(style="Title")
    title.add_run("NBA Shot Outcome Modeling")
    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(2)
    run = subtitle.add_run("Independent analysis using public shot chart records")
    run.italic = True
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor.from_string(GRAY)
    byline = document.add_paragraph()
    byline.paragraph_format.space_after = Pt(8)
    byline.add_run("Gyanesh Nayak  |  November 2026").font.size = Pt(9)

    add_body(
        document,
        "I expected distance and shot type to explain most of the story, and they did matter. But this public dataset also forced me to be clear about what I could not measure: it records the shot and its location, not defender pressure or player tracking. I kept the question focused on the information that is actually in the data and checked whether probability models carried across seasons.",
    )

    add_heading(document, "Data and the first patterns")
    add_body(
        document,
        "I analyzed 872,168 regular-season shots across four source season labels, 2021 through 2024. The overall make rate was 46.95%. The modeled shot-chart fields had no missing values in these files. Make rates varied by season, from 46.11% in 2021 to 47.54% in 2022, so I kept season-level validation instead of randomly splitting individual shots.",
    )
    add_table(
        document,
        ["Source season label", "Shots", "Make rate"],
        [
            ["2021", "216,722", "46.11%"],
            ["2022", "217,218", "47.54%"],
            ["2023", "218,701", "47.43%"],
            ["2024", "219,527", "46.72%"],
        ],
    )
    add_body(
        document,
        "The clearest pattern was distance. Shots inside five feet went in 61.97% of the time, compared with 34.13% from 25 feet or farther. The relationship was not perfectly smooth: the 10-15 foot group had a slightly higher make rate than the 5-10 foot group. That was a reminder to treat distance as useful context, not a complete explanation by itself.",
    )
    add_figure(
        document,
        REPORTS / "make_rate_by_distance.png",
        "Figure 1. Observed make rate by shot distance across the four source seasons.",
        6.1,
    )

    add_heading(document, "Shot type and court location")
    add_body(
        document,
        "Two-point shots were made 54.27% of the time and three-point shots 36.00%. Those are make rates, not points-per-attempt or a measure of shot value. By court zone, restricted-area attempts were made 66.16% of the time, while above-the-break threes were made 35.33%. The most common recorded action was a jump shot, with a 36.77% make rate across 261,256 attempts.",
    )
    add_figure(
        document,
        REPORTS / "make_rate_by_shot_type.png",
        "Figure 2. Observed make rates for two-point and three-point attempts.",
        5.7,
    )

    document.add_page_break()
    add_heading(document, "Features and model comparison")
    add_body(
        document,
        "I used shot distance, court coordinates, period and seconds remaining, shot type, action type, and the recorded shot-zone categories. Numeric features were imputed and scaled in a pipeline; categories were one-hot encoded. I compared logistic regression with a random forest and used a constant training make rate as a baseline. The model outputs probabilities, so I used log loss as the primary score and Brier score as a second probability metric.",
    )
    add_body(
        document,
        "For the 2023 holdout, I trained on 2021-2022. For the 2024 holdout, I trained on 2021-2023. Each validation season was left out of fitting. The same modeling settings were used in both runs.",
    )
    add_table(
        document,
        ["Validation season", "Model", "Train / validation shots", "Log loss", "Brier score"],
        [
            ["2023", "Training-rate baseline", "433,940 / 218,701", "0.69190", "0.24938"],
            ["2023", "Logistic regression", "433,940 / 218,701", "0.64275", "0.22673"],
            ["2023", "Random forest", "433,940 / 218,701", "0.63702", "0.22420"],
            ["2024", "Training-rate baseline", "652,641 / 219,527", "0.69101", "0.24893"],
            ["2024", "Logistic regression", "652,641 / 219,527", "0.64188", "0.22622"],
            ["2024", "Random forest", "652,641 / 219,527", "0.63661", "0.22394"],
        ],
    )
    add_body(
        document,
        "The random forest beat logistic regression in both holdout seasons, although the difference was modest: about 0.0057 log-loss points in 2023 and 0.0053 in 2024. Both models improved on the constant-rate baseline. Seeing similar scores across two seasons is encouraging, but two holdouts are not enough to prove future stability or show that the small gap between the models is statistically meaningful.",
    )
    add_figure(
        document,
        REPORTS / "calibration_2024.png",
        "Figure 3. Binned predictions and observed make rates on the 2024 holdout. This is a visual calibration check, not a guarantee for every subgroup.",
        4.8,
    )

    document.add_page_break()
    add_heading(document, "What the random forest used")
    add_body(
        document,
        "Shot distance had the largest impurity-based importance (18.32%), followed by court Y location (14.37%), the restricted-area zone indicator (11.48%), court X location (8.78%), and seconds remaining (7.55%). These are model-use summaries, not causal effects. Distance, coordinates, and zone labels overlap in the information they carry, so importance can be divided across related features.",
    )
    add_figure(
        document,
        REPORTS / "feature_importance.png",
        "Figure 4. Top 15 random forest impurity importances on the 2024 training fit.",
        6.1,
    )

    add_heading(document, "What I would improve next")
    add_body(
        document,
        "I kept this public-data version to two model families and a straightforward season holdout. I did not tune a large model search or add player-specific effects. With more time, I would test additional held-out seasons, compare the model against a smooth location-based baseline, and study calibration by shot distance and shot type. I would also want better defensive context before making claims about how pressure changes shot probability.",
    )
    add_body(
        document,
        "The main thing I took from this version is that a probability score is only as useful as the validation behind it. The forest improved on the simple baseline in both held-out seasons, but the dataset only answers a narrower question than a tracking dataset could. Being clear about that limit is part of the result, too.",
    )

    add_heading(document, "Data source and scope", level=1)
    add_body(
        document,
        "Source: shufinskiy/nba_data, regular-season shotdetail archives and field descriptions, https://github.com/shufinskiy/nba_data and https://github.com/shufinskiy/nba_data/blob/main/description_fields.md. The source repository describes these shot records as coming from NBA Stats and lists an Apache-2.0 license for the repository. That does not establish separate rights to the underlying NBA records. The raw files are not included with this report. This is an independent public-data analysis and contains no internship assessment data, results, code, or materials.",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
