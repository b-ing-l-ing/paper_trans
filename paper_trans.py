import os
import fitz
from pydantic import BaseModel, Field
from typing import Optional, List
from openai import OpenAI
from dotenv import load_dotenv
import instructor
import json
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from PIL import Image
from io import BytesIO

load_dotenv()
client = instructor.patch(
    OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    ),
    mode=instructor.Mode.MD_JSON,   
)

# 定义你的 schema 
class MaterialProperty(BaseModel):
    name: str = Field(description="属性名称，如 'band gap', 'R²'")
    value: str = Field(description="数值+单位，如 '1.8 eV', '0.9964'")
    measurement_method: Optional[str] = Field(default=None, description="测量方法")

class Experiment(BaseModel):
    description: str = Field(description="实验简述")
    synthesis_conditions: dict[str, str] = Field(default_factory=dict, description="合成参数键值对")
    characterization: list[str] = Field(default_factory=list, description="表征手段列表")

class PaperData(BaseModel):
    paper_title: Optional[str] = Field(default=None)
    authors: list[str] = Field(default_factory=list)
    doi: Optional[str] = Field(default=None)
    journal: Optional[str] = Field(default=None)
    year: Optional[str] = Field(default=None)
    abstract: Optional[str] = Field(default=None)
    research_objective: Optional[str] = Field(default=None)
    experiments: list[Experiment] = Field(default_factory=list)
    material_properties: list[MaterialProperty] = Field(default_factory=list)
    performance_metrics: list[MaterialProperty] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    conclusions: Optional[str] = Field(default=None)

# 提取
def extract_data(text: str) -> PaperData:
    return client.chat.completions.create(
        model="deepseek-v4-flash",
        response_model=PaperData,          # instructor 保证输出这个类型
        max_retries=3,                     # 格式不对自动重试
        messages=[
            {
                "role": "system",
                "content": """你是化学/材料学专家。从论文中**全面、详尽**地提取所有结构化数据。

                关键要求：
                1. **切勿遗漏**：论文中所有表格、所有实验组的数据都要提取，不只是最优结果
                2. **完整组成**：XRF/XRD 等成分分析要列出所有组分，不是只列主要元素
                3. **全组对比**：不同配比、不同龄期的测试结果全部提取
                4. 只提取论文明确出现的数据，绝不编造。不确定的字段留空。"""
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    
def render_markdown(data: PaperData, source: str) -> str:
    """PaperData 对象 → markdown 字符串"""
    d = data.model_dump()  # pydantic → dict
    md = []

    # 标题 + 元信息
    md.append(f"# {d.get('paper_title') or 'Untitled'}\n")
    if d.get("authors"):
        md.append(f"**作者:** {', '.join(d['authors'])}")
    if d.get("journal"):
        md.append(f"**期刊:** {d['journal']}")
    if d.get("year"):
        md.append(f"**年份:** {d['year']}")
    if d.get("doi"):
        md.append(f"**DOI:** [{d['doi']}](https://doi.org/{d['doi']})")
    md.append(f"**来源:** {source}")
    md.append(f"**提取时间:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # 摘要
    if d.get("abstract"):
        md.append("---\n## 摘要\n")
        md.append(d["abstract"] + "\n")

    # 研究目标
    if d.get("research_objective"):
        md.append("## 研究目标\n")
        md.append(d["research_objective"] + "\n")

    # 实验方法
    for i, exp in enumerate(d.get("experiments", []), 1):
        md.append(f"---\n## 实验 {i}\n")
        md.append(f"**{exp['description']}**\n")
        if exp.get("synthesis_conditions"):
            md.append("| 参数 | 值 |")
            md.append("|------|-----|")
            for k, v in exp["synthesis_conditions"].items():
                md.append(f"| {k} | {v} |")
            md.append("")
        if exp.get("characterization"):
            md.append("**表征手段:**")
            for c in exp["characterization"]:
                md.append(f"- {c}")
            md.append("")

    # 材料属性
    if d.get("material_properties"):
        md.append("---\n## 材料属性\n")
        md.append("| 属性 | 值 | 测量方法 |")
        md.append("|------|-----|----------|")
        for p in d["material_properties"]:
            md.append(f"| {p['name']} | {p['value']} | {p.get('measurement_method','-')} |")
        md.append("")

    # 性能指标
    if d.get("performance_metrics"):
        md.append("---\n## 性能指标\n")
        md.append("| 指标 | 数值 |")
        md.append("|------|------|")
        for m in d["performance_metrics"]:
            md.append(f"| {m['name']} | {m['value']} |")
        md.append("")

    # 关键发现
    if d.get("key_findings"):
        md.append("---\n## 关键发现\n")
        for f in d["key_findings"]:
            md.append(f"- {f}")
        md.append("")

    # 结论
    if d.get("conclusions"):
        md.append("---\n## 结论\n")
        md.append(d["conclusions"] + "\n")

    md.append("---\n*数据由 DeepSeek 提取，请核对原文。*\n")
    return "\n".join(md)

# 保存 markdown 文件
def save_markdown(md_text: str, data: PaperData, output_dir: str = "./output") -> str:
    """保存 markdown 文件"""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    title = data.paper_title or "paper"
    filename = re.sub(r'[^\w\s-]', '', title)[:80]
    filename = re.sub(r'[-\s]+', '_', filename)
    filepath = out / f"{filename}.md"
    filepath.write_text(md_text, encoding="utf-8")
    return str(filepath)

# 保存JSON
def save_json(data: PaperData, output_dir: str = "./output") -> str:
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    title = data.paper_title or "paper"
    filename = re.sub(r'[^\w\s-]', '', title)[:80]
    filename = re.sub(r'[-\s]+', '_', filename)
    filepath = out / f"{filename}.json"
    filepath.write_text(json.dumps(data.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(filepath)

# 解析pdf
def parse_pdf_with_images(pdf_path, image_dir="./output/images"):
    out = Path(image_dir)
    out.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages_text = {}
    pages_images = defaultdict(list)

    for page_num, page in enumerate(doc, 1):
        pages_text[page_num] = page.get_text("text")

        for img_idx, img in enumerate(page.get_images()):
            xref = img[0]
            base_image = doc.extract_image(xref)
            ext = base_image["ext"]
            img_bytes = base_image["image"]

            if ext in ("jpx", "jp2"):
                pil_img = Image.open(BytesIO(img_bytes))
                ext = "png"
                filename = f"page{page_num}_img{img_idx+1}.{ext}"
                if pil_img.mode == "CMYK":
                    pil_img = pil_img.convert("RGB")
                pil_img.save(out / filename, "PNG")
            else:
                filename = f"page{page_num}_img{img_idx+1}.{ext}"
                (out / filename).write_bytes(img_bytes)

            pages_images[page_num].append(filename)

    doc.close()
    full_text = "\n\n".join(pages_text.values())
    return full_text, pages_text, dict(pages_images)

# 插图(放末尾)
def insert_images_by_page(markdown: str, pages_text: dict, pages_images: dict) -> str:
    """在 markdown 中按页码插入图片引用

    策略：找到每个 section 对应的页码范围，插入该范围的图片
    """
    import re

    # 1. 标记每页在全文中的位置
    offset = 0
    page_boundaries = {}  # {页码: 字符位置}
    for pn, text in sorted(pages_text.items()):
        page_boundaries[pn] = offset
        offset += len(text) + 2  # +2 for \n\n

    # 2. 收集所有图片引用，按页分组
    all_img_refs = []
    for pn, imgs in sorted(pages_images.items()):
        for img in imgs:
            all_img_refs.append(f"![图](images/{img})  *(第{pn}页)*")
    img_section = "\n## 图表\n\n" + "\n\n".join(all_img_refs) + "\n"

    # 3. 追加到 markdown 末尾（在脚注之前）
    markdown = markdown.replace(
        "\n---\n*数据由 DeepSeek 提取",
        f"\n{img_section}\n---\n*数据由 DeepSeek 提取",
    )
    return markdown

# 复查
def review_extraction(data: PaperData, text: str) -> PaperData | None:
    """Agent 自查：审核第一次提取结果，返回修正版或 None"""
    review_prompt = f"""请审核以下从论文中提取的数据是否有遗漏或错误。

论文片段：
{text[:15000]}

已提取的数据：
{data.model_dump_json(indent=2)}

审核要点：
1. 有遗漏的实验或参数吗？
2. 有未提取的材料属性或性能指标吗？
3. 有提取错误的值吗？

如果有问题，输出修正后的完整 JSON。如果没问题，只输出 "OK"。"""

    result = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": review_prompt}],
        temperature=0,
    )
    raw = result.choices[0].message.content

    if raw.strip().startswith("OK"):
        return None  # 没问题

    # 解析修正后的 JSON
    import json, re
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end+1]

    revised = json.loads(raw)
    return PaperData(**revised)

def extract_with_review(text: str) -> PaperData:
    """提取 + Agent 自审，返回更完整的结果"""
    data = extract_data(text)
    improved = review_extraction(data, text)
    return improved if improved else data

if __name__ == "__main__":
    
    pdf_path = "./PDFs/固化剂对碱激发固废胶凝材料稳定不锈钢渣的性能影响_付剑东.pdf"

    text, pages_text, pages_images = parse_pdf_with_images(pdf_path, image_dir="./output/images")
    # text = text[:30000]  # 截断
    data = extract_with_review(text)

    md = render_markdown(data, pdf_path)              
    md = insert_images_by_page(md, pages_text, pages_images)
    md_path = save_markdown(md,data, "./output")
    json_path = save_json(data)       # 同时存 JSON，将来塞数据库用
    print(f"Markdown: {md_path}\nJSON: {json_path}")