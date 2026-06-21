from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SKIP_DIRS = {"__pycache__", ".git", ".idea", ".vscode"}
OUTPUT_HTML = "dashboard.html"
OUTPUT_CSV = "dashboard_data.csv"
QUESTION_TEMPLATE_CSV = "question_text_template.csv"
OCR_INPUT_CSV = "ocr_input.csv"
OCR_RESULTS_CSV = "ocr_results.csv"
FONT = "Microsoft YaHei, SimHei, Arial, sans-serif"


@dataclass
class ImageRecord:
    member_folder: str
    member_name: str
    student_id: str
    path: str
    grade: str
    subject: str
    publisher: str
    unit: str
    section: str
    knowledge_point: str
    question_code: str
    question_text: str
    knowledge_source: str
    content: str


def clean_member_name(folder_name: str) -> tuple[str, str]:
    student_id_match = re.search(r"\d{12}", folder_name)
    student_id = student_id_match.group(0) if student_id_match else ""
    name = re.sub(r"\d{12}", "", folder_name)
    for token in ["截图作业", "截图文件", "截图", "作业", "文件", "的"]:
        name = name.replace(token, "")
    name = re.sub(r"\s+", "", name).strip()
    return name or folder_name.strip(), student_id


def normalize_grade(parts: list[str]) -> str:
    text = " / ".join(parts)
    grade_patterns = [
        ("七年级上", r"七年级上|七上"),
        ("七年级下", r"七年级下|七下"),
        ("八年级上", r"八年级上|八上"),
        ("八年级下", r"八年级下|八下"),
        ("九年级全册", r"九年级全册|九年级|九上|九下"),
        ("七年级", r"七年级"),
        ("八年级", r"八年级"),
    ]
    for label, pattern in grade_patterns:
        if re.search(pattern, text):
            return label
    return "未识别"


def first_matching(parts: list[str], patterns: list[str], default: str) -> str:
    for part in parts:
        if any(re.search(pattern, part, flags=re.IGNORECASE) for pattern in patterns):
            return part.strip()
    return default


def normalize_unit(parts: list[str]) -> str:
    unit, _, _ = normalize_unit_with_index(parts)
    return unit


def normalize_unit_with_index(parts: list[str]) -> tuple[str, int, int]:
    for index, part in enumerate(parts):
        value = part.strip()
        if re.match(r"^Module\s*\d+", value, flags=re.IGNORECASE):
            next_part = parts[index + 1].strip() if index + 1 < len(parts) else ""
            if re.match(r"^Unit\s*\d+", next_part, flags=re.IGNORECASE):
                return f"{value} {next_part}", index, index + 1
            return value, index, index
        if re.match(r"^(Starter\s+)?Unit\s*\d+", value, flags=re.IGNORECASE):
            next_part = parts[index + 1].strip() if index + 1 < len(parts) else ""
            unit_number_only = re.fullmatch(r"(Starter\s+)?Unit\s*\d+", value, flags=re.IGNORECASE)
            next_is_detail = next_part and not re.match(
                r"^(Section|Module|Unit|P\d+|Q\d+|[上下]册|七年级|八年级|九年级|初中英语|英语|人教版|外研版)",
                next_part,
                flags=re.IGNORECASE,
            )
            if unit_number_only and next_is_detail:
                return f"{value} {next_part}", index, index + 1
            return value, index, index
    return "未识别Unit", -1, -1


def normalize_section(parts: list[str], start_index: int = 0) -> str:
    for index, part in enumerate(parts[start_index:], start=start_index):
        value = part.strip()
        if re.match(r"^Section\s+", value, flags=re.IGNORECASE):
            next_part = parts[index + 1].strip() if index + 1 < len(parts) else ""
            if is_lesson_detail(next_part):
                return f"{value} {next_part}"
            return value
    return "未细分Section"


def is_lesson_detail(value: str) -> bool:
    return bool(
        value
        and re.match(
            r"^(\d+[a-z]?\s*[-,]\s*\d+[a-z]?|Grammar|Vocabulary|Project|Pronunciation|Self Check|Reading|Writing|Listening|Speaking)",
            value,
            flags=re.IGNORECASE,
        )
    )


def is_generic_part(value: str) -> bool:
    return bool(
        not value
        or value in {"screenshots", "新建文件夹", "全国", "初中英语", "英语", "人教版", "外研版", "上册", "下册"}
        or re.fullmatch(r"七年级[上下]?册?|八年级[上下]?册?|九年级全册", value)
    )


def normalize_question_code(filename: str) -> str:
    match = re.search(r"(P\d+_Q\d+)", Path(filename).stem, flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def normalize_knowledge_point(parts: list[str], unit: str, unit_end_index: int, section: str, question_code: str) -> str:
    if section != "未细分Section":
        return f"{unit} / {section}" if unit != "未识别Unit" else section

    tail = parts[unit_end_index + 1 :] if unit_end_index >= 0 else parts
    tail = [part.strip() for part in tail if not is_generic_part(part.strip())]
    tail = [part for part in tail if part not in {unit, "前一千张截图"}]
    if tail:
        return f"{unit} / {' / '.join(tail)}" if unit != "未识别Unit" else " / ".join(tail)
    if question_code:
        return f"{unit} / {question_code}" if unit != "未识别Unit" else question_code
    return unit


def parse_path(member_folder: str, relative_path: Path | PurePosixPath) -> ImageRecord:
    member_name, student_id = clean_member_name(member_folder)
    parts = [part for part in relative_path.parts[:-1] if part and part not in SKIP_DIRS]
    subject = first_matching(parts, [r"初中英语", r"英语"], "英语")
    publisher = first_matching(parts, [r"人教版", r"外研版", r"译林版", r"北师大版"], "北师大版")
    grade = normalize_grade(parts)
    unit, _, unit_end_index = normalize_unit_with_index(parts)
    section = normalize_section(parts, unit_end_index + 1 if unit_end_index >= 0 else 0)
    question_code = normalize_question_code(relative_path.name)
    knowledge_point = normalize_knowledge_point(parts, unit, unit_end_index, section, question_code)
    content = knowledge_point
    return ImageRecord(
        member_folder=member_folder,
        member_name=member_name,
        student_id=student_id,
        path=str(relative_path),
        grade=grade,
        subject=subject,
        publisher=publisher,
        unit=unit,
        section=section,
        knowledge_point=knowledge_point,
        question_code=question_code,
        question_text="",
        knowledge_source="目录路径",
        content=content,
    )


def collect_records(root: Path, include_zip: bool = False) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for member_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if member_dir.name in SKIP_DIRS:
            continue
        for file_path in member_dir.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = file_path.relative_to(member_dir)
            records.append(parse_path(member_dir.name, rel))

        if not include_zip:
            continue
        for zip_path in member_dir.rglob("*.zip"):
            with zipfile.ZipFile(zip_path) as archive:
                for name in archive.namelist():
                    virtual_path = PurePosixPath(name)
                    if virtual_path.suffix.lower() not in IMAGE_EXTS:
                        continue
                    records.append(parse_path(member_dir.name, virtual_path))
    return records


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def normalize_key(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\\", "/").strip()


def clean_ocr_text(text: object) -> str:
    if pd.isna(text):
        return ""
    value = str(text)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", value)
    value = re.sub(r"\s+([，。！？；：,.!?;:])", r"\1", value)
    value = re.sub(r"([（〔【])\s+", r"\1", value)
    value = re.sub(r"\s+([）〕】])", r"\1", value)
    return value


def trim_text(text: str, max_len: int = 86) -> str:
    text = clean_ocr_text(text)
    return text if len(text) <= max_len else f"{text[:max_len]}..."


def split_ocr_question_and_answer(text: str) -> tuple[str, str]:
    cleaned = clean_ocr_text(text)
    if not cleaned:
        return "", ""

    answer_patterns = [
        r"[【\[\(（〔〖]?\s*答\s*案\s*[】\]\)）〕〗]?",
        r"\bAnswer\b\s*[:：]?",
    ]
    marker_match = None
    for pattern in answer_patterns:
        marker_match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if marker_match:
            break

    if marker_match:
        question = cleaned[: marker_match.start()]
        after_answer = cleaned[marker_match.end() :]
        explain_match = re.search(r"[【\[\(（〔〖]?\s*详\s*解\s*[】\]\)）〕〗]?|解析|Explanation", after_answer, flags=re.IGNORECASE)
        answer = after_answer[: explain_match.start()] if explain_match else after_answer
    else:
        question = cleaned
        answer = ""

    question = re.sub(r"^\s*\d+\s*[\.、．]?\s*", "", question).strip()
    answer = re.sub(r"^\s*[:：]?\s*", "", answer).strip()
    return question, answer


def make_question_knowledge_label(text: object) -> str:
    question, answer = split_ocr_question_and_answer(str(text))
    if not question:
        return ""

    label = trim_text(question, 78)
    answer = trim_text(answer, 32)
    if answer:
        label = f"{label} | 答案：{answer}"
    return label


def first_existing_column(columns: set[str], candidates: list[str]) -> str | None:
    for column in candidates:
        if column in columns:
            return column
    return None


def load_question_text_mapping(root: Path) -> dict[tuple[str, str], str]:
    candidates = [
        root / "question_text.csv",
        root / "question_texts.csv",
        root / "题目内容.csv",
        root / "ocr_results.csv",
    ]
    mapping: dict[tuple[str, str], str] = {}
    for csv_path in candidates:
        if not csv_path.exists():
            continue
        text_df = pd.read_csv(csv_path, dtype=str).fillna("")
        columns = set(text_df.columns)
        text_col = first_existing_column(columns, ["question_text", "题目内容", "题干", "题目", "知识点", "text", "ocr_text"])
        if not text_col:
            continue

        for _, row in text_df.iterrows():
            question_text = str(row[text_col]).strip()
            if not question_text:
                continue

            member = str(row["member_folder"]).strip() if "member_folder" in columns else ""
            path = ""
            if "source_path" in columns:
                path = normalize_key(row["source_path"])
            elif "full_path" in columns:
                path = normalize_key(row["full_path"])
            elif "path" in columns:
                path = normalize_key(row["path"])
            if path:
                mapping[("path", f"{member}/{path}" if member else path)] = question_text

            grade = str(row["grade"]).strip() if "grade" in columns else ""
            unit = str(row["unit"]).strip() if "unit" in columns else ""
            question_code = str(row["question_code"]).strip().upper() if "question_code" in columns else ""
            if grade and unit and question_code:
                mapping[("grade_unit_question", f"{grade}|{unit}|{question_code}")] = question_text
    return mapping


def apply_question_texts(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    mapping = load_question_text_mapping(root)
    if not mapping:
        write_question_text_template(df, root / QUESTION_TEMPLATE_CSV)
        return df

    def lookup(row: pd.Series) -> str:
        source_path = f"{normalize_key(row['member_folder'])}/{normalize_key(row['path'])}"
        path = normalize_key(row["path"])
        grade_unit_question = f"{row['grade']}|{row['unit']}|{row['question_code']}"
        for key in [
            ("path", source_path),
            ("path", path),
            ("grade_unit_question", grade_unit_question),
        ]:
            if key in mapping:
                return mapping[key]
        return ""

    question_texts = df.apply(lookup, axis=1)
    has_text = question_texts.str.len() > 0
    df.loc[has_text, "question_text"] = question_texts[has_text]
    df.loc[has_text, "knowledge_point"] = question_texts[has_text]
    df.loc[has_text, "content"] = question_texts[has_text]
    df.loc[has_text, "knowledge_source"] = "题目内容"
    return df


def write_question_text_template(df: pd.DataFrame, output_path: Path, limit: int = 5000) -> None:
    if output_path.exists():
        return
    template = (
        df[["member_folder", "member_name", "student_id", "grade", "unit", "section", "question_code", "path"]]
        .drop_duplicates()
        .head(limit)
        .copy()
    )
    template["question_text"] = ""
    template.to_csv(output_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def run_windows_ocr(root: Path, df: pd.DataFrame, limit: int) -> None:
    if limit <= 0:
        return
    script_path = root / "ocr_windows.ps1"
    if not script_path.exists():
        raise SystemExit("未找到 ocr_windows.ps1，无法执行 OCR。")

    ocr_rows = df.head(limit).copy()
    ocr_rows["source_path"] = ocr_rows["member_folder"].map(normalize_key) + "/" + ocr_rows["path"].map(normalize_key)
    ocr_rows["image_path"] = ocr_rows.apply(
        lambda row: str(root / str(row["member_folder"]) / str(row["path"])),
        axis=1,
    )
    input_cols = [
        "source_path",
        "image_path",
        "member_folder",
        "member_name",
        "student_id",
        "grade",
        "unit",
        "section",
        "question_code",
        "path",
    ]
    input_path = root / OCR_INPUT_CSV
    output_path = root / OCR_RESULTS_CSV
    ocr_rows[input_cols].to_csv(input_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-InputCsv",
        str(input_path),
        "-OutputCsv",
        str(output_path),
    ]
    subprocess.run(command, cwd=root, check=True)


def style_figure(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"} if title else None,
        height=height,
        margin=dict(l=28, r=24, t=52 if title else 24, b=32),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color="#dce8ff"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        colorway=["#3cc7ff", "#38e8b0", "#ffd166", "#ff7f91", "#a78bfa", "#f59e0b", "#60a5fa"],
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.16)", zerolinecolor="rgba(148,163,184,0.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.16)", zerolinecolor="rgba(148,163,184,0.2)")
    return fig


def make_member_bar(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("member_name", as_index=False).size().rename(columns={"size": "count"})
    data = data.sort_values("count", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=data["count"],
            y=data["member_name"],
            orientation="h",
            text=data["count"],
            textposition="outside",
            marker=dict(
                color=data["count"],
                colorscale=[[0, "#2563eb"], [0.5, "#14b8a6"], [1, "#f59e0b"]],
                line=dict(color="rgba(255,255,255,0.2)", width=1),
            ),
            hovertemplate="%{y}<br>截图数：%{x}<extra></extra>",
        )
    )
    fig.update_xaxes(title="截图数量")
    return style_figure(fig, "成员采集量排行", 440)


def make_grade_donut(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("grade", as_index=False).size().rename(columns={"size": "count"})
    fig = go.Figure(
        go.Pie(
            labels=data["grade"],
            values=data["count"],
            hole=0.58,
            textinfo="label+percent",
            hovertemplate="%{label}<br>截图数：%{value}<extra></extra>",
        )
    )
    fig.update_traces(marker=dict(line=dict(color="rgba(15,23,42,0.9)", width=2)))
    return style_figure(fig, "年级分布环形图", 360)


def make_stacked_grade_bar(df: pd.DataFrame) -> go.Figure:
    data = df.groupby(["member_name", "grade"], as_index=False).size().rename(columns={"size": "count"})
    members = df.groupby("member_name").size().sort_values(ascending=False).index.tolist()
    grades = data.groupby("grade")["count"].sum().sort_values(ascending=False).index.tolist()
    fig = go.Figure()
    for grade in grades:
        sub = data[data["grade"] == grade].set_index("member_name").reindex(members, fill_value=0)
        fig.add_trace(go.Bar(name=grade, x=members, y=sub["count"], hovertemplate="%{x}<br>%{y}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickangle=-35)
    return style_figure(fig, "成员-年级采集结构", 420)


def make_unit_heatmap(df: pd.DataFrame) -> go.Figure:
    top_points = df.groupby("knowledge_point").size().sort_values(ascending=False).head(14).index.tolist()
    top_members = df.groupby("member_name").size().sort_values(ascending=False).index.tolist()
    pivot = (
        df[df["knowledge_point"].isin(top_points)]
        .pivot_table(index="member_name", columns="knowledge_point", values="path", aggfunc="count", fill_value=0)
        .reindex(index=top_members, columns=top_points, fill_value=0)
    )
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Tealgrn",
            hovertemplate="成员：%{y}<br>知识点：%{x}<br>截图数：%{z}<extra></extra>",
            colorbar=dict(title="截图数"),
        )
    )
    fig.update_xaxes(tickangle=-32)
    return style_figure(fig, "成员与具体知识点热力图", 560)


def with_top_knowledge(df: pd.DataFrame, top_n: int = 80) -> pd.DataFrame:
    top_points = set(df.groupby("knowledge_point").size().sort_values(ascending=False).head(top_n).index)
    view = df.copy()
    view["knowledge_display"] = view["knowledge_point"].where(
        view["knowledge_point"].isin(top_points), "其他具体知识点"
    )
    return view


def make_sunburst(df: pd.DataFrame) -> go.Figure:
    view = with_top_knowledge(df)
    grouped = (
        view.groupby(["member_name", "grade", "unit", "knowledge_display"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    fig = px.sunburst(
        grouped,
        path=["member_name", "grade", "unit", "knowledge_display"],
        values="count",
        color="count",
        color_continuous_scale="Blues",
    )
    return style_figure(fig, "成员-年级-Unit-具体知识点旭日图", 560)


def make_treemap(df: pd.DataFrame) -> go.Figure:
    view = with_top_knowledge(df)
    grouped = (
        view.groupby(["grade", "unit", "knowledge_display"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    fig = px.treemap(
        grouped,
        path=["grade", "unit", "knowledge_display"],
        values="count",
        color="count",
        color_continuous_scale="Viridis",
    )
    return style_figure(fig, "小组具体知识点分布树图", 560)


def make_section_bar(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("knowledge_point", as_index=False).size().rename(columns={"size": "count"})
    data = data.sort_values("count", ascending=False).head(14)
    fig = go.Figure(
        go.Bar(
            x=data["knowledge_point"],
            y=data["count"],
            text=data["count"],
            textposition="outside",
            marker=dict(color=data["count"], colorscale="Sunset"),
            hovertemplate="%{x}<br>截图数：%{y}<extra></extra>",
        )
    )
    fig.update_xaxes(tickangle=-30)
    return style_figure(fig, "具体知识点采集排行", 460)


def make_radar(df: pd.DataFrame) -> go.Figure:
    rows = []
    for member, sub in df.groupby("member_name"):
        rows.append(
            {
                "member_name": member,
                "截图总量": len(sub),
                "覆盖Unit": sub["unit"].nunique(),
                "覆盖知识点": sub["knowledge_point"].nunique(),
                "覆盖年级": sub["grade"].nunique(),
                "教材类型": sub["publisher"].nunique(),
            }
        )
    data = pd.DataFrame(rows).sort_values("截图总量", ascending=False).head(6)
    metrics = ["截图总量", "覆盖Unit", "覆盖知识点", "覆盖年级", "教材类型"]
    max_values = {metric: max(data[metric].max(), 1) for metric in metrics}
    fig = go.Figure()
    for _, row in data.iterrows():
        values = [round(row[metric] / max_values[metric] * 100, 1) for metric in metrics]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=metrics + [metrics[0]],
                fill="toself",
                name=row["member_name"],
                hovertemplate="%{theta}<br>归一化得分：%{r}<extra></extra>",
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(148,163,184,0.2)"),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
        )
    )
    return style_figure(fig, "成员采集覆盖能力雷达图", 460)


def make_publisher_subject(df: pd.DataFrame) -> go.Figure:
    data = df.groupby(["publisher", "grade"], as_index=False).size().rename(columns={"size": "count"})
    fig = px.bar(data, x="publisher", y="count", color="grade", text="count")
    fig.update_traces(textposition="outside")
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="截图数量")
    return style_figure(fig, "教材版本与年级覆盖", 380)


def make_unit_rank(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("unit", as_index=False).size().rename(columns={"size": "count"})
    data = data[data["unit"] != "未识别Unit"].sort_values("count", ascending=True).tail(12)
    fig = go.Figure(
        go.Bar(
            x=data["count"],
            y=data["unit"],
            orientation="h",
            text=data["count"],
            textposition="outside",
            marker=dict(
                color=data["count"],
                colorscale="Agsunset",
                line=dict(color="rgba(255,255,255,0.18)", width=1),
            ),
            hovertemplate="%{y}<br>截图数：%{x}<extra></extra>",
        )
    )
    fig.update_xaxes(title="截图数量")
    return style_figure(fig, "高频Unit覆盖排行", 380)


def make_content_table(df: pd.DataFrame) -> str:
    rows = (
        df.groupby(["grade", "unit", "knowledge_point"], as_index=False)
        .agg(
            截图数=("path", "count"),
            题号示例=("question_code", lambda s: "、".join([v for v in sorted(set(s))[:4] if v]) or "-"),
            参与成员=("member_name", lambda s: "、".join(sorted(set(s)))),
        )
        .sort_values("截图数", ascending=False)
        .head(22)
    )
    tr_items = []
    for _, row in rows.iterrows():
        tr_items.append(
            "<tr>"
            f"<td>{row['grade']}</td>"
            f"<td>{row['unit']}</td>"
            f"<td>{row['knowledge_point']}</td>"
            f"<td class='num'>{int(row['截图数'])}</td>"
            f"<td>{row['题号示例']}</td>"
            f"<td>{row['参与成员']}</td>"
            "</tr>"
        )
    return "\n".join(tr_items)


def make_kpis(df: pd.DataFrame) -> dict[str, str]:
    top_member = df.groupby("member_name").size().sort_values(ascending=False)
    top_point = df.groupby("knowledge_point").size().sort_values(ascending=False)
    return {
        "总截图数": f"{len(df):,}",
        "成员数": f"{df['member_name'].nunique()}",
        "覆盖Unit": f"{df['unit'].nunique()}",
        "覆盖知识点": f"{df['knowledge_point'].nunique()}",
        "覆盖年级": f"{df['grade'].nunique()}",
        "采集最多成员": f"{top_member.index[0]} / {top_member.iloc[0]:,}",
        "最高频知识点": f"{top_point.index[0]} / {top_point.iloc[0]:,}",
    }


def make_member_table(df: pd.DataFrame) -> str:
    member_stats = (
        df.groupby(["member_name", "student_id"], as_index=False)
        .agg(
            截图数=("path", "count"),
            覆盖Unit=("unit", "nunique"),
            覆盖知识点=("knowledge_point", "nunique"),
            覆盖年级=("grade", "nunique"),
        )
        .sort_values("截图数", ascending=False)
    )
    rows = []
    for _, row in member_stats.iterrows():
        student_id = row["student_id"] if row["student_id"] else "-"
        rows.append(
            "<tr>"
            f"<td>{row['member_name']}</td>"
            f"<td>{student_id}</td>"
            f"<td class='num'>{int(row['截图数'])}</td>"
            f"<td class='num'>{int(row['覆盖Unit'])}</td>"
            f"<td class='num'>{int(row['覆盖知识点'])}</td>"
            f"<td class='num'>{int(row['覆盖年级'])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def figure_divs(figures: list[go.Figure]) -> list[str]:
    divs = []
    for index, fig in enumerate(figures):
        divs.append(
            pio.to_html(
                fig,
                include_plotlyjs=True if index == 0 else False,
                full_html=False,
                config={"displayModeBar": False, "responsive": True},
            )
        )
    return divs


def build_html(df: pd.DataFrame, output_path: Path) -> None:
    figures = [
        make_member_bar(df),
        make_grade_donut(df),
        make_stacked_grade_bar(df),
        make_unit_heatmap(df),
        make_treemap(df),
        make_sunburst(df),
        make_section_bar(df),
        make_radar(df),
        make_publisher_subject(df),
        make_unit_rank(df),
    ]
    divs = figure_divs(figures)
    kpis = make_kpis(df)
    kpi_cards = "\n".join(
        f"<div class='kpi'><span>{label}</span><strong>{value}</strong></div>" for label, value in kpis.items()
    )
    content_rows = make_content_table(df)
    member_rows = make_member_table(df)
    coverage_text = "、".join(df.groupby("grade").size().sort_values(ascending=False).index.tolist())

    cards = "\n".join(
        f"<section class='panel'>{div}</section>"
        for div in divs[:2]
    )
    wide_cards = "\n".join(
        f"<section class='panel wide'>{div}</section>"
        for div in divs[2:6]
    )
    lower_cards = "\n".join(
        f"<section class='panel'>{div}</section>"
        for div in divs[6:]
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>软件三班第二组可视化大屏</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-width: 1200px;
      background:
        radial-gradient(circle at 20% 10%, rgba(56, 189, 248, .16), transparent 26rem),
        radial-gradient(circle at 82% 0%, rgba(52, 211, 153, .12), transparent 28rem),
        #07111f;
      color: #dce8ff;
      font-family: {FONT};
    }}
    .screen {{ padding: 24px 28px 34px; }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-end;
      margin-bottom: 18px;
      border-bottom: 1px solid rgba(148, 163, 184, .22);
      padding-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      color: #f8fbff;
      font-size: 34px;
      letter-spacing: 0;
    }}
    .subtitle {{ color: #90a8c8; font-size: 15px; }}
    .stamp {{
      text-align: right;
      color: #8da3c2;
      line-height: 1.8;
      white-space: nowrap;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(7, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .kpi {{
      border: 1px solid rgba(56, 189, 248, .24);
      background: linear-gradient(180deg, rgba(15, 36, 63, .86), rgba(10, 25, 44, .78));
      border-radius: 8px;
      padding: 13px 14px;
      min-height: 82px;
      box-shadow: 0 16px 45px rgba(0, 0, 0, .20);
    }}
    .kpi span {{
      display: block;
      color: #91a7c4;
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .kpi strong {{
      display: block;
      color: #f8fbff;
      font-size: 22px;
      line-height: 1.18;
      word-break: break-word;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .panel {{
      min-width: 0;
      border: 1px solid rgba(148, 163, 184, .2);
      border-radius: 8px;
      background: rgba(9, 22, 39, .82);
      box-shadow: 0 18px 46px rgba(0, 0, 0, .25);
      overflow: hidden;
    }}
    .wide {{ grid-column: span 2; }}
    .summary {{
      margin: 16px 0;
      padding: 16px 18px;
      border: 1px solid rgba(52, 211, 153, .22);
      border-radius: 8px;
      background: rgba(8, 35, 43, .72);
      color: #cfe3f7;
      line-height: 1.8;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid rgba(148, 163, 184, .16);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: #8fd7ff; background: rgba(15, 36, 63, .66); }}
    td {{ color: #d9e7fb; }}
    .num {{ color: #ffd166; text-align: right; font-variant-numeric: tabular-nums; }}
    .table-title {{
      padding: 15px 16px 0;
      color: #f8fbff;
      font-size: 18px;
      font-weight: 700;
    }}
    @media (max-width: 1280px) {{
      body {{ min-width: 1000px; }}
      .kpis {{ grid-template-columns: repeat(4, 1fr); }}
    }}
  </style>
</head>
<body>
  <main class="screen">
    <header>
      <div>
        <h1>软件三班第二组可视化大屏</h1>
        <div class="subtitle">基于各成员截图作业自动统计：成员采集量、题目知识点分布、年级与内容覆盖</div>
      </div>
      <div class="stamp">
        <div>覆盖范围：{coverage_text}</div>
      </div>
    </header>

    <section class="kpis">
      {kpi_cards}
    </section>

    <section class="grid">
      {cards}
      {wide_cards}
      {lower_cards}
      <section class="panel wide">
        <div class="table-title">成员采集明细</div>
        <table>
          <thead><tr><th>成员</th><th>学号</th><th class="num">截图数</th><th class="num">覆盖Unit</th><th class="num">覆盖知识点</th><th class="num">覆盖年级</th></tr></thead>
          <tbody>{member_rows}</tbody>
        </table>
      </section>
      <section class="panel wide">
        <div class="table-title">高频内容与知识点</div>
        <table>
          <thead><tr><th>年级</th><th>Unit</th><th>具体知识点</th><th class="num">截图数</th><th>题号示例</th><th>参与成员</th></tr></thead>
          <tbody>{content_rows}</tbody>
        </table>
      </section>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def print_summary(df: pd.DataFrame, html_path: Path, csv_path: Path) -> None:
    print(f"已生成：{html_path}")
    print(f"明细数据：{csv_path}")
    print(f"总截图数：{len(df):,}")
    print(f"成员数：{df['member_name'].nunique()}")
    print("成员采集量：")
    for member, count in df.groupby("member_name").size().sort_values(ascending=False).items():
        print(f"  {member}: {count:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成小组截图采集情况可视化大屏")
    parser.add_argument("--root", default=".", help="截图作业根目录，默认当前目录")
    parser.add_argument("--include-zip", action="store_true", help="同时统计 ZIP 压缩包内部图片")
    parser.add_argument("--output", default=OUTPUT_HTML, help="输出 HTML 文件名")
    parser.add_argument("--ocr-limit", type=int, default=0, help="先使用 Windows OCR 识别前 N 张截图，再用题目内容作为知识点")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    records = collect_records(root, include_zip=args.include_zip)
    if not records:
        raise SystemExit("没有找到图片文件，请检查目录或图片扩展名。")

    df = pd.DataFrame([record.__dict__ for record in records])
    run_windows_ocr(root, df, args.ocr_limit)
    df = apply_question_texts(df, root)
    csv_path = root / OUTPUT_CSV
    html_path = root / args.output
    save_csv(df, csv_path)
    build_html(df, html_path)
    print_summary(df, html_path, csv_path)


if __name__ == "__main__":
    main()
