"""Readable source-context report for the actually applied strict quality policy."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import zipfile

from corpus_pipeline.context_review import report as base
from corpus_pipeline.context_review.apply_quality import verify_version
from corpus_pipeline.context_review.strict_policy import LABELS, bound, require
from corpus_pipeline.codex_review.farm import canonical, file_sha


def build(root):
    root=Path(root).resolve(strict=True)
    inputs, initial_bindings=base.load_completed(root)
    sources={r['candidate']['item_id']:r for r in inputs}
    pointer_path=root.parent/'QUALITY_POLICY_CURRENT.json'
    pointer=base.strict_json(pointer_path.read_bytes())
    application,verified=verify_version(Path(pointer['directory']),pointer['revision_sha256'])
    require(pointer==verified,'quality_current_changed')
    artifact=pointer['quality_decisions'];require(file_sha(artifact['path'])==artifact['sha256'],'applied_decisions_changed')
    with gzip.open(artifact['path'],'rt') as stream:rows=[base.strict_json(line) for line in stream]
    require(len(rows)==688 and {r['item_id'] for r in rows}==set(sources),'applied_coverage')
    rows.sort(key=lambda r:r['review_number'])
    destination=root/'strict_blur_20261025'/'report'
    require(not destination.exists(),'strict_report_already_exists');destination.mkdir()
    counts=Counter(r['report_category'] for r in rows)
    blur=Counter(r['blur_status'] for r in rows)
    tasks,sources_counts,cross=defaultdict(Counter),defaultdict(Counter),defaultdict(Counter)
    for row in rows:
        c=sources[row['item_id']]['candidate'];cat=row['report_category']
        tasks[c['task_type']][cat]+=1;sources_counts[c['source_name']][cat]+=1;cross[c['human_review']['decision']][cat]+=1
    fields=['review_number','item_id','source_name','task_type','decision','report_category','rejection_reason','blur_status',
        'original_human_decision','context_action','image_origin_followup','reason','upstream_topic','source_tar','source_member']
    with (destination/'per_item.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            c=sources[row['item_id']]['candidate'];p=row['policy_input']
            out={k:row[k] for k in ('review_number','item_id','decision','report_category','rejection_reason','blur_status','image_origin_followup','reason')}
            out.update(source_name=c['source_name'],task_type=c['task_type'],original_human_decision=c['human_review']['decision'],
                context_action=p['context_action'],upstream_topic=p['context_assessment']['context_interpretation'],
                source_tar=c['source_ref']['path'],source_member=c['source_ref']['member'])
            writer.writerow({k:base.csv_safe(v) for k,v in out.items()})
    # Render each card once; all source text is escaped by the shared renderer.
    with (destination/'review.html').open('w',encoding='utf-8') as stream:
        stream.write('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DCVLM严格模糊规则：688项处理结果</title><style>'+base.STYLE+'</style><main><h1>DCVLM：688项已应用严格模糊规则</h1>')
        stream.write('<div class="intro"><p>最新规则：带模糊的直接不要。页面首先显示实际处理结果，随后保留原图、完整源文本和此前语义质量意见。语义可用不能覆盖模糊排除；旧人工判定仅作为历史证据。</p><p>'+
            ' · '.join(LABELS.get(k,k)+'：'+str(v) for k,v in counts.items())+'</p><p>只重新检查这688个目标；同记录其他图和全DCVLM没有因此自动通过。处理已写入筛选索引，原始TAR和历史判定未改。</p></div>')
        stream.write('<div class="filters"><input id="q" placeholder="编号、来源或原文搜索" aria-label="搜索">')
        options=[('source','来源',sorted(sources_counts)),('task','类型',sorted(tasks)),
            ('human','旧人工结果',sorted(cross)),('rec','实际结果',sorted(counts))]
        for name,title,values in options:
            stream.write('<select id="'+name+'" aria-label="'+title+'"><option value="">全部'+title+'</option>')
            for value in values:stream.write('<option value="'+base.esc(value)+'">'+base.esc(LABELS.get(value,base.LABELS.get(value,value)))+'</option>')
            stream.write('</select>')
        stream.write('<button id="reset">重置筛选</button><button id="collapse">全部折叠</button><span id="count" class="count"></span></div>')
        for row in rows:
            original=sources[row['item_id']];candidate=original['candidate'];p=row['policy_input']
            context_row={**original,'result':p['context_assessment']}
            base.LABELS['retain_caption_prune_invalid_qa']='原图/主caption可留，局部QA需修剪'
            card=base.render_card(context_row,base.original_image(candidate))
            card=card.replace('data-rec="'+p['context_assessment']['recommendation']+'"','data-rec="'+row['report_category']+'"',1)
            card=card.replace('<summary><b>','<summary><b>'+base.esc('实际结果：'+LABELS[row['report_category']])+'</b> · <b>',1)
            panel='<section class="body"><p class="note"><strong>已应用：'+base.esc(LABELS[row['report_category']])+'</strong> · blur_status='+base.esc(row['blur_status'])+' · '+base.esc(row['rejection_reason'])+'<br>'+base.esc(row['reason'])+'</p>'
            card=card.replace('<section class="body">',panel,1)
            details={'blur_assessment':p['blur_assessment'],'independent_blur_adjudication':p.get('blur_adjudication'),
                'image_origin_followup':p['image_origin_followup'],'image_origin_note':p.get('image_origin_note'),
                'authority':row['decision_authority'],'quality_policy_manifest':row['quality_policy_manifest']}
            marker='<p class=id>图像 SHA256：'+base.esc(row['item_id'])+'</p>'
            card=card.replace(marker,'<details><summary>严格模糊检查与来源规则证据</summary>'+base._pre(json.dumps(details,ensure_ascii=False,indent=2))+'</details>'+marker,1)
            stream.write(card)
        stream.write('</main><script>'+base.SCRIPT+'</script></html>')
    categories=['admitted','quality_excluded','ai_excluded','visual_pending']
    text=['# DCVLM 688项：严格模糊筛选已应用','',
        '已执行最新规则“带模糊的直接不要”：明显局部脸部/物体模糊、整体失焦、运动模糊及明确背景虚化均排除，不再以剩余场景有用为理由保留。',
        '真正的材料纹理、绘画笔触、平滑渐变、掩膜不因图像形式而被直接当作模糊；疑难边界另行原图复核。','',
        '| 实际处理结果 | 数量 |','|---|---:|']
    text += ['| '+LABELS[k]+' | '+str(counts.get(k,0))+' |' for k in categories]
    text += ['', '模糊轴独立统计：'+ '；'.join(k+' '+str(v) for k,v in blur.items())+'。主分类有优先级，既模糊又已人工AI排除的图片仍计在AI排除项。','',
        '## 各任务结果','', '| 原任务 | '+' | '.join(LABELS[k] for k in categories)+' |','|---|'+'---:|'*4]
    for task,values in tasks.items():text.append('| '+base.LABELS.get(task,task)+' | '+' | '.join(str(values.get(k,0)) for k in categories)+' |')
    text += ['', '## 上游实际内容和训练价值','',
        '- 图文对583项主要来自DataComp/ReLAION的网页图文。清晰的混凝土、水彩底纹、钢材、布料与壁纸样图可用于材料/颜色/纹理学习；短商品标题只提供弱监督，不等于高质量精细caption。#191/#247/#309就是原文明确描述纹理的例子。',
        '- 交错目标96个，来自80条源记录。OmniCC本批65个目标里49个URL带/t_lazy/，其中47个是NFL官方域名；内容多为体育新闻和相册，图片清晰度与独立图注覆盖是主要问题。有明显模糊的相册图按新规则排除，不能再因为仍看得出运动员而保留。',
        '- MINT PDF/HTML含论文实验、产品资料、报告、教程与网页。#98是前景检测掩模，#281是宝石显微图，#522是游戏桌面材质试作。这些来源用途比孤立图片外观更能说明训练价值。',
        '- #597/#598本来配的是人物画/婴儿文本，图已成为大块模糊色块，直接排除。#679虽有可用主caption，但佛像脸部模糊，最新规则下图像也不再保留，原QA错误分析仅留作证据。',
        '- #548的原文和大学项目资料给出GAN实验来源线索；#478/#489涉及数据集蒸馏图。三项保留独立来源待核标记：语义对应不能解除图像AI来源限制，明确模糊时先按模糊规则排除。','',
        '## 已经写入哪里','',
        '新规则已应用到原筛选结果的688项子集，生成独立机器规则版本，并原子更新QUALITY_POLICY_CURRENT.json。70492项完整筛选索引已重建；其余69804项保持原逻辑内容。此前人工AI排除继续生效，未把模型判断伪装成人工新决定。',
        '生产入口已增加独立质量门控，默认作用于本次已复核的688张：明确排除阻断当前训练view，5张待核不放行。其他图片明确标为本次范围外、未做模糊复核，继续原有AI及数据筛选，不能被称为已经通过模糊检查。交错数据保留源记录和坏图槽的处理动作，删除图槽后还需维护文字引用，不把整个正文一起物理删除。',
        '**本次实际重查范围是688张，不是全DCVLM。** 842426项总体数字只是此前筛选结果的算术汇总，不表示全库刚做了模糊检查。通用预训练可用也不等于高分辨率recaption阶段合格。','',
        '## 文件','',
        '- [逐项原图、完整原文与实际处理结果]('+str(destination/'review.html')+')',
        '- [CSV索引]('+str(destination/'per_item.csv')+')',
        '- [离线页面ZIP]('+str(destination/'review.zip')+')',
        '- [当前筛选版本指针]('+str(pointer_path)+')',
        '- [688项实际处理决策]('+artifact['path']+')',
        '- [交错序列详细来源分析]('+str(root/'INTERLEAVED_ASSESSMENT.md')+')','']
    (destination/'REPORT.md').write_text('\n'.join(text))
    with zipfile.ZipFile(destination/'review.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:archive.write(destination/'review.html','review.html')
    stats={'records':688,'applied_counts':dict(counts),'blur_counts':dict(blur),'by_task':{k:dict(v) for k,v in tasks.items()},
        'by_source':{k:dict(v) for k,v in sources_counts.items()},'previous_human_cross':{k:dict(v) for k,v in cross.items()}}
    report={'schema_version':'corpus.strict_quality.report.v1','complete':True,'summary':stats,'policy_pointer':bound(pointer_path),
        'applied_quality_decisions':artifact,'initial_source_bindings':initial_bindings,'generator':bound(__file__),
        'artifacts':{name:bound(destination/name) for name in ['REPORT.md','review.html','per_item.csv','review.zip']}}
    (destination/'REPORT.json').write_text(canonical(report)+'\n')
    (destination/'COMPLETE.json').write_text(canonical({'complete':True,'records':688,'report':bound(destination/'REPORT.json')})+'\n')
    return stats


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();print(canonical(build(args.root)))
