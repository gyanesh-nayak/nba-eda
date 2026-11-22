"""Build the public-data project writeup from evaluated metrics and figures."""

from __future__ import annotations

import argparse
import json
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


def add_table(
    document: Document,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float] | None = None,
) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = widths is None
    header_properties = table.rows[0]._tr.get_or_add_trPr()
    repeat_header = OxmlElement("w:tblHeader")
    repeat_header.set(qn("w:val"), "true")
    header_properties.append(repeat_header)
    for cell, header in zip(table.rows[0].cells, headers):
        set_cell_text(cell, header, bold=True, color="FFFFFF")
        set_cell_shading(cell, BLUE)
    if widths:
        for cell, width in zip(table.rows[0].cells, widths):
            cell.width = Inches(width)
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        row_properties = table.rows[-1]._tr.get_or_add_trPr()
        row_properties.append(OxmlElement("w:cantSplit"))
        for cell, value in zip(cells, row):
            set_cell_text(cell, value)
            if row_index % 2 == 1:
                set_cell_shading(cell, "F4F7FA")
        if widths:
            for cell, width in zip(cells, widths):
                cell.width = Inches(width)
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

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.paragraph_format.space_before = Pt(4)
    run = footer.add_run("Independent public-data analysis  |  Page ")
    run.font.name = "Arial"
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string(GRAY)
    page_number = OxmlElement("w:fldSimple")
    page_number.set(qn("w:instr"), "PAGE")
    footer._p.append(page_number)


def load_metrics(season: int) -> dict:
    with (REPORTS / f"metrics_{season}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def find_rate(records: list[dict], key: str, contains: str) -> dict:
    match = next(
        (record for record in records if contains.casefold() in record[key].casefold()),
        None,
    )
    if match is None:
        raise SystemExit(f"Could not find {contains!r} in {key} summary")
    return match


def percent(value: float) -> str:
    return f"{value:.2%}"


def score_with_interval(metrics: dict, model: str, metric: str) -> str:
    score = metrics["models"][model][metric]
    interval = metrics["game_cluster_bootstrap_95_ci"][model][metric]
    return f"{score:.4f} ({interval['lower_95']:.4f}–{interval['upper_95']:.4f})"


def readable_feature(name: str) -> str:
    name = name.removeprefix("numeric__").removeprefix("categorical__")
    for prefix, label in [
        ("SHOT_ZONE_BASIC_", "Zone: "),
        ("SHOT_ZONE_AREA_", "Area: "),
        ("SHOT_TYPE_", "Shot type: "),
        ("ACTION_TYPE_", "Action: "),
    ]:
        if name.startswith(prefix):
            return label + name.removeprefix(prefix)
    return {
        "SHOT_DISTANCE": "Shot distance",
        "LOC_X": "Court X location",
        "LOC_Y": "Court Y location",
        "PERIOD": "Quarter",
        "SECONDS_REMAINING_IN_PERIOD": "Seconds left in period",
    }.get(name, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "NBA_Shot_Outcome_Analysis.docx"
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

    metrics_2023 = load_metrics(2023)
    metrics_2024 = load_metrics(2024)
    summary = metrics_2024["data_summary"]
    distance = summary["make_rate_by_distance"]
    close = find_rate(distance, "DISTANCE_BAND", "0-5")
    long = find_rate(distance, "DISTANCE_BAND", "25+")
    shot_type = summary["make_rate_by_shot_type"]
    two_point = find_rate(shot_type, "SHOT_TYPE", "2pt")
    three_point = find_rate(shot_type, "SHOT_TYPE", "3pt")
    zones = summary["make_rate_by_zone"]
    restricted = find_rate(zones, "SHOT_ZONE_BASIC", "restricted")
    above_break = find_rate(zones, "SHOT_ZONE_BASIC", "above the break")
    actions = summary["make_rate_by_common_action"]
    jump_shot = find_rate(actions, "ACTION_TYPE", "jump shot")
    top_features = metrics_2024["random_forest_top_features"]
    top_feature_summary = ", ".join(
        f"{readable_feature(item['feature'])} ({item['importance']:.1%})"
        for item in top_features[:5]
    )

    document = Document()
    style_document(document)
    document.core_properties.title = "NBA Shot Outcome Modeling"
    document.core_properties.subject = "Independent analysis using public NBA shot chart records"
    document.core_properties.author = "Gyanesh Nayak"
    document.core_properties.keywords = "NBA, shot outcomes, calibration, public data"

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
        f"I analyzed {summary['rows']:,} regular-season shots across four source season labels, 2021 through 2024. The overall make rate was {percent(summary['overall_make_rate'])}. The modeled shot-chart fields had no missing values in these files. Make rates varied by season, from {percent(min(item['make_rate'] for item in summary['season_summary']))} to {percent(max(item['make_rate'] for item in summary['season_summary']))}, so I kept season-level validation instead of randomly splitting individual shots.",
    )
    add_table(
        document,
        ["Source season label", "Shots", "Make rate"],
        [
            [str(item["season"]), f"{item['shots']:,}", percent(item["make_rate"])]
            for item in summary["season_summary"]
        ],
    )
    add_body(
        document,
        f"The clearest pattern was distance. Shots inside five feet went in {percent(close['make_rate'])} of the time, compared with {percent(long['make_rate'])} from 25 feet or farther. The relationship was not perfectly smooth: the 10–15 foot group had a slightly higher make rate than the 5–10 foot group. That was a reminder to treat distance as useful context, not a complete explanation by itself.",
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
        f"Two-point shots were made {percent(two_point['make_rate'])} of the time and three-point shots {percent(three_point['make_rate'])}. Those are make rates, not points-per-attempt or a measure of shot value. By court zone, restricted-area attempts were made {percent(restricted['make_rate'])} of the time, while above-the-break threes were made {percent(above_break['make_rate'])}. The most common recorded action was a jump shot, with a {percent(jump_shot['make_rate'])} make rate across {jump_shot['shots']:,} attempts.",
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
    add_body(
        document,
        "Logistic regression used C=0.5 and a 1,000-iteration limit. The random forest used 300 trees, a 30-row minimum leaf size, and square-root feature sampling. Both were run with seed 17. I kept the settings fixed rather than tuning against the held-out seasons.",
    )
    add_table(
        document,
        ["Holdout", "Model", "Train / validation n", "Log loss (95% CI)", "Brier (95% CI)"],
        [
            [str(season), label, f"{metrics['train_rows']:,} / {metrics['validation_rows']:,}",
             score_with_interval(metrics, model, "log_loss"),
             score_with_interval(metrics, model, "brier_score")]
            for season, metrics in [(2023, metrics_2023), (2024, metrics_2024)]
            for model, label in [
                ("training_prevalence_baseline", "Training-rate baseline"),
                ("logistic_regression", "Logistic regression"),
                ("random_forest", "Random forest"),
            ]
        ],
        widths=[0.58, 1.28, 1.18, 2.10, 1.72],
    )
    add_body(
        document,
        "Point scores are followed by 95% percentile intervals from a game-cluster bootstrap. Each resample draws whole validation games with replacement, preserving within-game dependence. The intervals describe uncertainty in each season's evaluation set; they are not a test that the small difference between models is statistically meaningful.",
    )
    add_body(
        document,
        "Both models improved on the constant-rate baseline in both seasons. The paired game-cluster interval for the random-forest versus logistic-regression log-loss difference is reported alongside the scores in the JSON results. Two held-out seasons are useful evidence, but they do not establish future-season performance.",
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
        f"The top five impurity-based importance values were {top_feature_summary}. These are model-use summaries, not causal effects. Distance, coordinates, and zone labels overlap in the information they carry, so importance can be divided across related features.",
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
