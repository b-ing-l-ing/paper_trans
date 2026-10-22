"""端到端测试 —— 走完整链路:解析 → 抽取 → 归一化 → 输出。"""

import json

from paper2data.parser import parse_pdf
from paper2data.table_detector import detect_tables
from paper2data.schema import Schema
from paper2data.extract import extract_dataset
from paper2data.compute import normalize_record

pdf = 'PDFs/固化剂对碱激发固废胶凝材料稳定不锈钢渣的性能影响_付剑东.pdf'
schema = Schema.load('paper2data/schema.json')

# 1. 解析
text, pages_text, pages_images = parse_pdf(pdf, image_dir='./output_p2d_test/images')
tables = detect_tables(pdf)
print(f'[1] 解析: {len(text):,} 字符, {len(tables)} 个表格, {len(pages_images)} 页含图')

# 2. 抽取(调 LLM)
result = extract_dataset(text, schema)
print(f'[2] 抽取: {len(result.records)} 条试验, {len(result.new_fields)} 个新字段')
if result.notes:
    print(f'    备注: {result.notes}')

# 3. 归一化
normalized = [normalize_record(r, schema) for r in result.records]

# 4. 打印
print('\n=== 归一化后 records ===')
for i, rec in enumerate(normalized):
    # 只打印非 null 的字段,避免刷屏
    filled = {k: v for k, v in rec.items() if v is not None}
    print(f'[{i}] {filled}')

print('\n=== new_fields(LLM 补的字段) ===')
for f in result.new_fields:
    print(f'  {f.name} ({f.label}) 单位={f.unit} 类型={f.type}')
    print(f'     {f.desc}')

# 5. 保存
out = {
    'records': normalized,
    'new_fields': [f.model_dump() for f in result.new_fields],
    'notes': result.notes,
}
with open('output_p2d_test/result.json', 'w', encoding='utf-8') as fp:
    json.dump(out, fp, ensure_ascii=False, indent=2)
print('\n[5] 已保存 output_p2d_test/result.json')
