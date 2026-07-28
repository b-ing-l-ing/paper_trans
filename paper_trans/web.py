"""Web UI —— 上传 PDF → 提取 → 查看结果 + 下载。"""

import os
import time
import zipfile
from pathlib import Path

import markdown_it
from flask import Flask, render_template_string, request, send_from_directory
from urllib.parse import quote

from .parser import parse_pdf
from .pdf_parser import detect_tables
from .extractor import extract_data, extract_with_review, extract_multimodal
from .renderer import render_markdown, insert_images_by_page, save_markdown, save_json

app = Flask(__name__)

OUTPUT_ROOT = Path.cwd() / "output_web"

md_renderer = markdown_it.MarkdownIt("commonmark").enable("table")


@app.route("/download/<path:filename>")
def download(filename):
    from flask import send_file
    filepath = OUTPUT_ROOT / filename
    if not filepath.exists():
        return "File not found", 404
    return send_file(filepath, as_attachment=True)


@app.route("/static/<path:filename>")
def static_file(filename):
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    return send_from_directory(static_dir, filename)


INDEX = r"""\
<!doctype html>
<html lang="zh">
<head>
    <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Paper Trans</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: system-ui; max-width: 900px; margin: 24px auto; padding: 0 12px;
               background: #948CFA; }
        h1 { font-size: 1.3rem; color: #fff; }
        .zone { margin: 16px 0; padding: 28px 20px; border: 2px dashed #fff; border-radius: 8px;
               text-align: center; color: #fff; cursor: pointer; transition: .2s; }
        .zone.drag { border-color: #fff; background: rgba(255,255,255,0.15); }
        .zone:hover { background: rgba(255,255,255,0.08); }
        .zone input { display: none; }
        .opts { margin: 10px 0; }
        .opts { color: #fff; }
        .opts label { margin-right: 16px; font-size: 0.9rem; }
        button { margin-top: 10px; padding: 8px 24px; font-size: 1rem; cursor: pointer;
                 background: #fff; color: #948CFA; border: none; border-radius: 4px; font-weight: bold; }
        button:hover { background: #f0eeff; }
        #filelist { margin: 8px 0; font-size: 0.9rem; color: #fff; }
        #loading { display: none; text-align: center; padding: 24px; }
        #loading img { width: 180px; }
        #loading p { color: #fff; }
        .result-block { margin-top: 24px; border: 1px solid #fff; border-radius: 6px; overflow: hidden;
                         background: #fff; }
        .result-title { background: #fff; color: #948CFA; padding: 8px 12px; font-weight: bold;
                        font-size: 0.9rem; border-bottom: 1px solid #948CFA; }
        .result-body { padding: 12px; font-size: 0.85rem; line-height: 1.6; background: #fff; }
        .result-body table { border-collapse: collapse; margin: 6px 0; font-size: 0.8rem; }
        .result-body th, .result-body td { border: 1px solid #948CFA; padding: 3px 6px; }
        .result-body th { background: #f4f2ff; }
        .result-body h2 { font-size: 1.1rem; border-bottom: 2px solid #948CFA; padding-bottom: 3px; margin-top: 18px; }
        .result-body h3 { font-size: 0.95rem; margin-top: 12px; }
        .result-body img { max-width: 100%; }
        .dl { display: flex; gap: 8px; padding: 8px 12px; flex-wrap: wrap;
              border-bottom: 1px solid #948CFA; background: #fff; }
        .dl a { padding: 4px 12px; background: #948CFA; color: #fff; text-decoration: none;
                border-radius: 3px; font-size: 0.8rem; }
        .dl a:hover { background: #7d74e0; }
        .raw { margin: 0 12px 12px; background: #fff; padding: 8px; }
        .raw summary { cursor: pointer; color: #948CFA; font-size: 0.8rem; }
        .raw pre { background: #f8f8f8; padding: 8px; border: 1px solid #948CFA; border-radius: 4px;
                   overflow-x: auto; font-size: 0.75rem; white-space: pre-wrap; }
        .error { color: #c00; padding: 8px; background: #fff; border: 1px solid #fff; border-radius: 4px; font-size: 0.9rem; }
        .info { color: rgba(255,255,255,0.8); font-size: 0.8rem; margin-top: 6px; }
    </style>
</head>
<body>
    <h1>Paper Trans — 论文数据提取</h1>
    <form method="post" enctype="multipart/form-data" onsubmit="return onSubmit()">
        <div class="zone" id="zone" onclick="document.getElementById('fileinput').click()">
            <p>拖拽 PDF 到此处，或点击选择文件</p>
            <input type="file" id="fileinput" name="pdfs" accept=".pdf" multiple
                   onchange="showFiles(this.files)">
        </div>
        <div id="filelist"></div>
        <div class="opts">
            <label><input type="checkbox" name="multimodal" value="1"> 多模态（Gemini 看图）</label>
            <label><input type="checkbox" name="review" value="1"> 启用审查（纯文本二次校验）</label>
        </div>
        <button type="submit">提取数据</button>
        <p class="info">纯文本约 30 秒，含审查约 60 秒，多模态约 60 秒。</p>
    </form>
    <div id="loading">
        <img src="/static/jindutiao.gif" alt="处理中...">
        <p>正在提取数据，请稍候…</p>
    </div>
    {% if history %}
    <details style="margin-top:20px">
        <summary style="cursor:pointer;color:#fff;font-size:0.9rem">历史记录 ({{ history|length }})</summary>
        <div style="margin-top:8px">
        {% for h in history %}
            <div style="padding:6px 8px;background:rgba(255,255,255,0.1);margin:4px 0;border-radius:4px;
                        display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:0.85rem;color:#fff">
                <span>{{ h.time }}</span>
                <b>{{ h.name }}</b>
                <a href="/download/{{ h.id }}/{{ h.md }}" style="color:#fff">MD</a>
                <a href="/download/{{ h.id }}/{{ h.json }}" style="color:#fff">JSON</a>
                {% if h.zip %}<a href="/download/{{ h.id }}/{{ h.zip }}" style="color:#fff">ZIP</a>{% endif %}
                <a href="/view/{{ h.id }}" style="color:#fff">查看</a>
            </div>
        {% endfor %}
        </div>
    </details>
    {% endif %}
    {% if error %}<div class="error" style="margin-top:12px">{{ error }}</div>{% endif %}
    {% for r in results %}
    <div class="result-block">
        <div class="result-title">{{ r.name }}</div>
        <div class="dl">
            <a href="/download/{{ r.run_id }}/{{ r.md_name }}">下载 MD</a>
            <a href="/download/{{ r.run_id }}/{{ r.json_name }}">下载 JSON</a>
            {% if r.has_images %}<a href="/download/{{ r.run_id }}/{{ r.zip_name }}">下载图片 ZIP</a>{% endif %}
        </div>
        <div class="result-body">{{ r.html | safe }}</div>
        <details class="raw"><summary>原始 Markdown</summary><pre>{{ r.md }}</pre></details>
    </div>
    {% endfor %}
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <script>
        var zone = document.getElementById('zone');
        ['dragenter','dragover'].forEach(e => zone.addEventListener(e, function(ev){
            ev.preventDefault(); zone.classList.add('drag');
        }));
        ['dragleave','drop'].forEach(e => zone.addEventListener(e, function(ev){
            ev.preventDefault(); zone.classList.remove('drag');
        }));
        zone.addEventListener('drop', function(ev){
            document.getElementById('fileinput').files = ev.dataTransfer.files;
            showFiles(ev.dataTransfer.files);
        });
        function showFiles(files){
            var names = []; for(var i=0;i<files.length;i++) names.push(files[i].name);
            document.getElementById('filelist').innerHTML = '<b>已选择:</b> ' + names.join(', ');
        }
        function onSubmit(){
            if(document.getElementById('fileinput').files.length === 0) return false;
            document.getElementById('loading').style.display = 'block';
            return true;
        }
    </script>
</body>
</html>
"""


def _list_history():
    """扫描 output_web/ 目录，返回历史记录列表。"""
    items = []
    if not OUTPUT_ROOT.exists():
        return items
    for d in sorted(OUTPUT_ROOT.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        md_files = list(d.glob("*.md"))
        json_files = list(d.glob("*.json"))
        zip_files = list(d.glob("*.zip"))
        if not md_files:
            continue
        items.append({
            "id": d.name,
            "time": d.name[:14],
            "name": md_files[0].stem,
            "md": quote(md_files[0].name),
            "json": quote(json_files[0].name) if json_files else "",
            "zip": quote(zip_files[0].name) if zip_files else "",
        })
    return items


@app.route("/view/<run_id>")
def view_result(run_id):
    """查看历史结果。"""
    run_dir = OUTPUT_ROOT / run_id
    if not run_dir.exists():
        return "Not found", 404
    md_files = list(run_dir.glob("*.md"))
    json_files = list(run_dir.glob("*.json"))
    zip_files = list(run_dir.glob("*.zip"))
    has_images = bool(list((run_dir / "images").glob("*"))) if (run_dir / "images").exists() else False
    if not md_files:
        return "No result", 404
    import re
    md = md_files[0].read_text(encoding="utf-8")
    md_display = re.sub(r"!\[图\]\(images/", f"![图](/download/{run_id}/images/", md)
    html = md_renderer.render(md_display)
    result = {
        "name": md_files[0].stem,
        "run_id": run_id,
        "md_name": quote(md_files[0].name),
        "json_name": quote(json_files[0].name) if json_files else "",
        "zip_name": quote(zip_files[0].name) if zip_files else "",
        "has_images": has_images,
        "md": md,
        "html": html,
    }
    return render_template_string(
        INDEX, results=[result], history=_list_history(),
    )


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX, results=[], history=_list_history())


@app.route("/", methods=["POST"])
def process():
    from flask import redirect, url_for

    files = request.files.getlist("pdfs")
    valid = [f for f in files if f and f.filename]
    if not valid:
        return render_template_string(INDEX, error="请选择 PDF 文件", results=[], history=_list_history())

    use_multimodal = request.form.get("multimodal") == "1"
    use_review = request.form.get("review") == "1"

    results = []
    for file in valid:
        try:
            run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + str(valid.index(file))
            out_dir = OUTPUT_ROOT / run_id
            img_dir = out_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)

            from werkzeug.utils import secure_filename
            pdf_path = out_dir / secure_filename(file.filename)
            file.save(str(pdf_path))

            text, pages_text, pages_images = parse_pdf(pdf_path, image_dir=str(img_dir))
            tables = detect_tables(str(pdf_path))

            if use_multimodal:
                from .config import settings
                img_list = [str(img_dir / f) for files in pages_images.values() for f in files]
                data = extract_multimodal(text, img_list, tables=tables, model=settings.multimodal_model)
            elif use_review:
                data = extract_with_review(text, tables=tables)
            else:
                data = extract_data(text, tables=tables)

            md = render_markdown(data, str(pdf_path))
            if not md:
                raise RuntimeError("渲染结果为空")
            if pages_images:
                md = insert_images_by_page(md, pages_images)
            md_path = save_markdown(md, data, str(out_dir))
            json_path = save_json(data, str(out_dir))

            has_images = bool(pages_images)
            zip_name = ""
            if has_images:
                zip_name = "images.zip"
                with zipfile.ZipFile(str(out_dir / zip_name), "w") as zf:
                    for f in img_dir.iterdir():
                        zf.write(f, f.name)

            # 修正图片路径为绝对下载路径
            import re
            md_display = re.sub(r"!\[图\]\(images/", f"![图](/download/{run_id}/images/", md)
            results.append({
                "name": file.filename,
                "run_id": run_id,
                "md_name": quote(Path(md_path).name),
                "json_name": quote(Path(json_path).name),
                "zip_name": quote(zip_name) if zip_name else "",
                "has_images": has_images,
                "md": md,
                "html": md_renderer.render(md_display),
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                "name": file.filename,
                "error": str(e),
            })

    from flask import redirect
    # 单文件成功 → 重定向到结果页，刷新不会重复提交
    if len(results) == 1 and "error" not in results[0]:
        return redirect(f"/view/{results[0]['run_id']}")
    # 多文件或失败 → 直接显示
    has_error = any("error" in r for r in results)
    return render_template_string(
        INDEX,
        results=results,
        history=_list_history(),
        error="部分文件处理失败" if has_error else None,
    )


def main():
    import typer
    typer.echo("启动 http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
