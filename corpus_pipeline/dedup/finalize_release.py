"""Publish a local, hash-bound DCVLM deduplication candidate index bundle."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
import time
from .asset_release import artifact, require, verify
from corpus_pipeline.postprocess.reconcile import atomic_json, file_sha, digest


def publish_pointer(root):
    manifest=json.loads((root/'manifest.json').read_text())
    atomic_json(root.parent/'DEDUP_CURRENT.json',{'root':str(root),'complete':artifact(root/'COMPLETE.json'),
                'manifest':artifact(root/'manifest.json'),'source_view':manifest['source_view']})


def verified_selection_overlay(root,candidate,base_view,base):
    """Bind the effective selection to its immutable base and independent mask audit."""
    output=root/'quality_overlay'
    complete=json.loads((output/'COMPLETE.json').read_text())
    require(complete.get('complete') is True and
            complete['manifest_sha256']==file_sha(output/'manifest.json'),'Quality overlay incomplete or changed')
    overlay=json.loads((output/'manifest.json').read_text())
    require(overlay.get('complete') is True and overlay['scope']==base['scope']=='full_run',
            'Quality overlay coverage incomplete')
    for key,path in [('base_candidate',candidate),('base_source_view',base_view)]:
        binding=overlay[key]
        require(Path(binding['path']).absolute()==path and
                binding['complete_sha256']==file_sha(path/'COMPLETE.json') and
                binding['manifest_sha256']==file_sha(path/'manifest.json'),'Quality overlay base changed')
    require(overlay['base_counts']==base['counts'],'Quality overlay base counts changed')
    changed=overlay['changed_records'];counts=overlay['final_counts']
    require(isinstance(changed,int) and 0<=changed<=base['counts']['keep'],'Invalid overlay change count')
    expected={k:base['counts'].get(k,0) for k in ['keep','exclude','quarantine']}
    expected['keep']-=changed;expected['exclude']+=changed
    require(counts==expected and complete['counts']==counts and
            complete['changed_records']==changed and
            overlay['records']==complete['records']==base['records'],'Invalid overlay count transition')
    for item in overlay['evidence']:verify(item)
    for item in overlay['files']:
        path=output/item['path']
        require(not Path(item['path']).is_absolute() and path.resolve().is_relative_to(output),
                'Overlay artifact escapes release')
        verify({**item,'path':str(path)})
    rule=json.loads((output/'asset_rule.json').read_text())
    require(digest(rule)==overlay['rule_sha256'] and
            rule['match']=='exact_original_image_ids' and rule['action']=='keep_to_exclude' and
            not rule['propagate_canonical_identity'] and not rule['propagate_sscd_similarity'],
            'Quality rule must use exact original asset IDs only')
    view=Path(overlay['source_view']).absolute()
    require(view==output/'source_view','Unexpected effective source view')
    vc=json.loads((view/'COMPLETE.json').read_text());vm=json.loads((view/'manifest.json').read_text())
    require(overlay['source_view_complete_sha256']==file_sha(view/'COMPLETE.json') and
            overlay['source_view_manifest_sha256']==file_sha(view/'manifest.json')==vc['manifest_sha256'],
            'Effective source view changed')
    require(vc.get('complete') is True and vm.get('complete') is True and vm.get('selection_overlay') is True and
            vm.get('input_candidate_counts_are_base') is True and
            vm['selection_overlay_rule_sha256']==overlay['rule_sha256'] and
            vm['records']==vc['records']==base['records'] and
            all(vm['counts'].get(k,0)==vc['counts'].get(k,0)==counts[k] for k in counts) and
            vm['counts'].get('not_indexed',0)==0,'Effective source view coverage differs')
    categories=Counter()
    for row in overlay['source_category_counts']:categories[row['decision']]+=row['records']
    require(all(categories[k]==counts[k] for k in counts) and not (set(categories)-set(counts)) and
            sum(overlay['reason_counts'].values())==base['records'],
            'Quality overlay summary differs')
    audit_path=root/'audits/quality_overlay.json';audit=json.loads(audit_path.read_text())
    require(audit.get('passed') is True,'Independent quality overlay verification incomplete')
    required=[output/'COMPLETE.json',output/'manifest.json',view/'COMPLETE.json',view/'manifest.json']
    for path in required:
        require(audit['inputs'].get(str(path))==file_sha(path),'Quality overlay audit belongs to another stage')
    for path,sha in audit['inputs'].items():
        require(file_sha(path)==sha,'Quality overlay audited input changed: '+path)
    if (root/'PUBLISH_HOLD.json').exists():
        hold=json.loads((root/'PUBLISH_HOLD.json').read_text())
        require(hold['required_stage']=='quality_overlay' and hold['required_audit']=='audits/quality_overlay.json'
                and set(hold['asset_ids']).issubset(rule['asset_ids']),'Publication hold is not satisfied')
    return overlay,artifact(audit_path)


def verify_audit_binding(root,name,value):
    if name in ('assets','records'):
        for path,expected in value['inputs'].items():
            require(file_sha(path)==expected,'Audited input changed: '+path)
        required=([root/'assets/COMPLETE.json',root/'decisions/COMPLETE.json'] if name=='assets' else
                  [root/'training_candidate_index/COMPLETE.json',root/'training_candidate_index/manifest.json'])
        for path in required:
            require(value['inputs'].get(str(path))==file_sha(path),'Audit belongs to another stage')
        if name=='records':
            view=root/'training_candidate_source_view';v=value['source_view']
            require(Path(v['path']).absolute()==view and v['complete_sha256']==file_sha(view/'COMPLETE.json')
                    and v['manifest_sha256']==file_sha(view/'manifest.json'),'Source view audit is stale')
    else:
        required=({'input_manifest':root/'training_candidate_index/manifest.json',
                   'source_view_manifest':root/'training_candidate_source_view/manifest.json'} if name=='reader' else
                  {'input':root/'asset_similarity_index/COMPLETE.json','graph':root/'graph/COMPLETE.json'})
        for key,path in required.items():
            item=value[key]
            require(Path(item['path']).absolute()==path and item['sha256']==file_sha(path),'Audit evidence is stale')


def finalize(root):
    root=Path(root).absolute()
    if (root/'COMPLETE.json').exists():
        complete=json.loads((root/'COMPLETE.json').read_text())
        require(complete.get('complete') and complete['manifest_sha256']==file_sha(root/'manifest.json'),
                'Existing bundle manifest changed')
        manifest=json.loads((root/'manifest.json').read_text())
        for item in [*manifest['stages'].values(),*manifest['audits'].values(),
                     manifest['code_snapshot'],manifest['report'],*manifest.get('upstream_history',[]),
                     *manifest.get('selection_overlay_artifacts',[])]:
            require(file_sha(item['path'])==item['sha256'],'Existing release evidence changed')
        if (root/'PUBLISH_HOLD.json').exists():
            require(manifest.get('selection_overlay_applied') is True,'Publication hold requires quality overlay')
            candidate=root/'training_candidate_index';base_view=root/'training_candidate_source_view'
            base=json.loads((candidate/'manifest.json').read_text())
            overlay,_=verified_selection_overlay(root,candidate,base_view,base)
            require(manifest['source_view']==overlay['source_view'] and manifest['counts']==overlay['final_counts'],
                    'Published view differs from required quality overlay')
        publish_pointer(root)
        return complete
    names=['decisions','assets','graph_policy','graph','asset_similarity_index',
           'training_candidate_index','training_candidate_source_view']
    stages={name:json.loads((root/name/'COMPLETE.json').read_text()) for name in names}
    require(all(v.get('complete') for v in stages.values()),'A pipeline stage is incomplete')
    audits={}
    for name in ['assets','records','reader','combined_index']:
        value=json.loads((root/'audits'/f'{name}.json').read_text())
        require(value.get('passed') is True,'Independent verification incomplete: '+name)
        if name=='records':
            require(value.get('source_view_audited') is True,'Full source selection mask audit incomplete')
        verify_audit_binding(root,name,value)
        audits[name]=artifact(root/'audits'/f'{name}.json')
    candidate=root/'training_candidate_index';view=root/'training_candidate_source_view'
    r=json.loads((candidate/'manifest.json').read_text())
    v=json.loads((view/'manifest.json').read_text())
    require(file_sha(candidate/'manifest.json')==stages['training_candidate_index']['manifest_sha256'],
            'Record manifest changed')
    require(file_sha(view/'manifest.json')==stages['training_candidate_source_view']['manifest_sha256'],
            'Source view changed')
    require(v['input_candidate_manifest_sha256']==file_sha(candidate/'manifest.json'),'Source view parent changed')
    require(r['scope']=='full_run' and r['records']==v['records'],'Incomplete record coverage')
    require(all(r['counts'].get(k,0)==v['counts'].get(k,0) for k in ['keep','exclude','quarantine']),
            'Selection view disagrees with record decisions')
    a=stages['assets']['counts'];g=stages['graph']
    require(a['assets']==stages['asset_similarity_index']['assets'],'Asset lookup coverage differs')
    require(g['vectors']==stages['asset_similarity_index']['sscd_vectors']==stages['graph_policy']['vectors'],
            'SSCD lookup coverage differs')
    require(g['advisory'] and not g['automatic_drop'] and not r['sscd_candidates_used_for_deletion'],
            'Unreviewed near similarity must not delete supervision')
    base_counts=dict(r['counts']);base_view=view;overlay=None;overlay_artifacts=[]
    if (root/'quality_overlay/COMPLETE.json').exists() or (root/'PUBLISH_HOLD.json').exists():
        overlay,audits['quality_overlay']=verified_selection_overlay(root,candidate,base_view,r)
        names.append('quality_overlay')
        view=Path(overlay['source_view'])
        r={**r,'counts':overlay['final_counts'],'source_category_counts':overlay['source_category_counts'],
           'reasons':overlay['reason_counts']}
        overlay_artifacts=[artifact(root/'quality_overlay/manifest.json'),artifact(view/'COMPLETE.json'),
                           artifact(view/'manifest.json'),artifact(root/'quality_overlay/asset_rule.json')]
        reader_path=root/'audits/quality_overlay_reader.json'
        if reader_path.exists():
            reader=json.loads(reader_path.read_text())
            require(reader.get('passed') is True and
                    reader['source_view_manifest_sha256']==file_sha(view/'manifest.json') and
                    reader['overlay_manifest_sha256']==file_sha(root/'quality_overlay/manifest.json'),
                    'Effective selection reader audit is stale')
            audits['quality_overlay_reader']=artifact(reader_path)
    bycategory=Counter()
    for row in r['source_category_counts']:bycategory[(row['category'],row['decision'])]+=row['records']
    with (root/'source_category_counts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['source','category','decision','records'])
        writer.writeheader();writer.writerows(r['source_category_counts'])
    with (root/'exclusion_reasons.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['reason','records'])
        writer.writerows(sorted(r['reasons'].items()))
    code=root/'code_snapshot';code.mkdir(exist_ok=True)
    modules=['decisions.py','asset_release.py','graph_policy.py','sscd_graph.py','_sscd_graph.cpp',
             'asset_similarity_index.py','training_release.py','candidate_reader.py','source_view.py',
             'release_audit.py','finalize_release.py']
    if overlay:modules.append('selection_overlay.py')
    for name in modules:shutil.copy2(Path(__file__).parent/name,code/name)
    graph_runtime=[]
    for path in sorted((root/'graph').glob('*driver*.py')):
        graph_runtime.append(artifact(path))
    if (root/'graph/RUNTIME.json').exists():graph_runtime.append(artifact(root/'graph/RUNTIME.json'))
    if (root/'graph/PRODUCTION_PROVENANCE.json').exists():
        graph_runtime.append(artifact(root/'graph/PRODUCTION_PROVENANCE.json'))
    code_manifest={'files':[artifact(code/name) for name in modules],
                   'graph_production_runtime':graph_runtime,
                   'note':'Snapshot contains reusable workspace modules. Actual graph production uses the frozen graph/runtime plus hash-bound RAM storage drivers in phase receipts.'}
    atomic_json(root/'CODE.json',code_manifest)
    date=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    duplicate_count=r['reasons'].get('identical_ordered_supervision_and_confirmed_images',0)
    history_files=[];history_text=''
    history_path=root/'audits/upstream_dedup.json'
    if history_path.exists():
        history=json.loads(history_path.read_text())
        require(history.get('passed') is True,'Upstream history verification failed')
        for name,path in [('complete',candidate/'COMPLETE.json'),('manifest',candidate/'manifest.json')]:
            item=history['current_full_run'][name]
            require(Path(item['path'])==path,'Upstream history belongs to another candidate')
            verify(item)
        for item in history['upstream_evidence'].values():verify(item)
        for item in history['upstream_executed_code_verified'].values():verify(item)
        h=history['counts']
        require(h['final_records']==r['records'] and
                history['current_full_run']['new_identical_record_exclusions']==duplicate_count,
                'Upstream history counts differ')
        history_files=[artifact(history_path),artifact(root/'audits/upstream_dedup.md')]
        history_text=('## 输入版本已有的去重\n\n本轮输入是已完成上游去重的 DCVLM 预训练版本。'
            f"上游从 {h['raw_input']:,} 条物理样本开始，经过先行检查后，"
            f"已排除 {h['exact_removed']:,} 条精确重复和 {h['near_removed']:,} 条近重复，"
            f"得到本轮输入的 {h['final_records']:,} 条记录。"
            f'这些是历史处理；本轮新增的精确重复排除为 {duplicate_count:,} 条，'
            '不能把历史去重或 SSCD 候选数计作本轮新增删除。'
            '来源路径、历史运行代码与核查范围见 `audits/upstream_dedup.md` 和对应 JSON。\n\n')
    lines=[f'# DCVLM 全量去重与训练候选索引\n\n完成核验时间：{date}（北京时间）。\n',
        '本版本生成可直接按原始 Parquet 行读取的筛选索引，原始图像和监督内容均保留。'
        '确认重复或已有明确排除结论的记录从候选视图排除；SSCD 相似候选不会单独触发删除。'
        '这是当前保守筛选阶段的完整索引，不代表全库图像已通过人工真伪、模糊或精细质量认证。\n',
        '## 全量结果\n',
        f"- 原训练记录：{r['records']:,}。\n- 保留候选：{r['counts'].get('keep',0):,}。\n"
        f"- 排除：{r['counts'].get('exclude',0):,}，其中确认重复监督：{duplicate_count:,}。\n"
        f"- 完整性隔离：{r['counts'].get('quarantine',0):,}。\n"
        f"- 图像资产：{a['assets']:,}；精确视觉身份：{a['exact_visual_identities']:,}；"
        f"共享相同 RGBA 像素的额外编码：{a['encoded_assets_sharing_exact_pixels']:,}。\n"
        f"- SSCD 向量：{g['vectors']:,}；完整相似边：{g['edges']:,}；"
        f"相似连通组（含孤立图）：{g['components']:,}；最大连通组：{g['largest_component']:,}。\n",
        history_text,
        (f"## 已确认占位图的补充排除\n\n原图与原始商品描述实物核验确认了一张通用商品缺图占位图。"
         f"本轮精确匹配该原始图像 SHA，新增排除 {overlay['changed_records']:,} 条原 keep 记录。"
         "原已排除和隔离记录保持不变；不向相同像素身份或 SSCD 相似图传播。"
         "基础索引与审计不可变，最终决定由 `quality_overlay/changed_records.parquet` 叠加；"
         "训练读取使用下列最终 source_view。资产查询和相似图仍是叠加前快照，使用时须同时应用 "
         "`quality_overlay/asset_rule.json`，不能把该占位图作为可训练候选代表。\n\n" if overlay else ''),
        '## 任务类型\n\n| 类型 | 保留 | 排除 | 隔离 |\n|---|---:|---:|---:|\n']
    for category in sorted({k[0] for k in bycategory}):
        lines.append('| '+category+' | '+' | '.join(f'{bycategory[(category,d)]:,}' for d in ['keep','exclude','quarantine'])+' |\n')
    lines += ['\n每个来源、任务类型和决定的完整统计见 `source_category_counts.csv`；排除原因见 `exclusion_reasons.csv`。\n',
        '## 规则\n\n1. 复用已有尺寸、AI、人工及质量结论，以原始 asset SHA 为准，不通过相似关系传播。'
        '\n2. 只有无冲突的 EXIF 校正尺寸和全部 RGBA 像素一致，才共享精确视觉身份；透明度保留。'
        '\n3. 训练记录仅在任务类型、原格式、包含角色/轮次/图文顺序的文本哈希、顺序精确图像身份全部相同时排重。'
        '同图不同 caption、QA、多轮或交错监督保留。'
        '\n4. SSCD 0.75 是候选阈值。连通组用于审计，候选代表与成员之间必须有直接边；不沿传递链删图。'
        '图构建时已有的 AI/质量排除资产不能成为候选代表；后续质量叠加规则须额外应用。'
        '\n5. 未决筛选保留显式标注；缺失或完整性错误隔离。保留行不重写 caption、不反转 DCVLM 任务、不替换图槽。\n',
        '## 读取\n\n```python\nfrom corpus_pipeline.dedup.source_view import iter_native_batches\n\n'
        f'view = {str(view)!r}\n'
        'for batch in iter_native_batches(view, batch_size=8192, epoch_seed=epoch,\n'
        '                                 worker_id=rank, num_workers=world_size):\n'
        '    # Arrow RecordBatch，原始 segments/images/format 等字段保持不变。\n'
        '    consume(batch)\n```\n\n所有 worker 使用相同 epoch_seed。此接口按原文件顺序批读，每个源文件每轮只校验一次完整 SHA；'
        '图像仍按原 TAR 引用读取，不复制图像。源文件选择掩码与上游索引均绑定 SHA。\n',
        '## 产物入口\n\n- `manifest.json` / `COMPLETE.json`：整体发布绑定。'
        f'\n- `{view.relative_to(root)}/`：最终高效原记录读取视图。'
        '\n- `training_candidate_index/buckets/`：基础逐记录 keep/exclude/quarantine、原因和重复代表；最终决定需应用发布清单中的质量叠加。'
        '\n- `asset_similarity_index/`：按图像 SHA 查询精确身份、筛选结论、SSCD 候选组和候选代表。'
        '\n- `graph/`：完整相似图和直接候选代表；不是已确认的视觉等价类。'
        '\n- `audits/`：独立覆盖、哈希、引用和真实读取检查。\n',
        '## 覆盖限制\n\n上游已有质量评估覆盖 70,492 张；严格模糊复核覆盖 688 张。未做新的全库模糊检测、'
        '全量 recaption 或 layout 生成。未确证的 SSCD 候选仍保留，不能把候选成员数称为删除图数。'
        '原数据原文件未物理删除；未宣称训练效果已经通过模型实验验证。\n']
    (root/'REPORT.md').write_text(''.join(lines))
    manifest={'schema':'corpus.dcvlm_dedup_candidate_release.v1','complete':True,
              'created_at':time.time(),'records':r['records'],'counts':r['counts'],
              'confirmed_duplicate_records_excluded':duplicate_count,'assets':a,
              'stages':{name:artifact(root/name/'COMPLETE.json') for name in names},'audits':audits,
              'source_view':str(view),'asset_lookup':str(root/'asset_similarity_index/COMPLETE.json'),
              'code_snapshot':artifact(root/'CODE.json'),'report':artifact(root/'REPORT.md'),
              'upstream_history':history_files,
              'base_counts':base_counts,'base_source_view':str(base_view),
              'selection_overlay_applied':overlay is not None,'selection_overlay_artifacts':overlay_artifacts,
              'selection_overlay':str(root/'quality_overlay/manifest.json') if overlay else None,
              'asset_lookup_exclusion_overlays':[artifact(root/'quality_overlay/asset_rule.json')] if overlay else [],
              'asset_screening_counts_scope':'base_before_selection_overlay' if overlay else 'current',
              'candidate_index_published':True,'published_training':False,'source_certification':False,
              'near_similarity_candidates_automatically_deleted':False,'source_files_modified':False}
    atomic_json(root/'manifest.json',manifest)
    complete={'complete':True,'schema':manifest['schema'],'manifest_sha256':file_sha(root/'manifest.json'),
              'records':r['records'],'counts':r['counts'],'candidate_index_published':True,
              'published_training':False,'completed_at':time.time()}
    atomic_json(root/'COMPLETE.json',complete)
    publish_pointer(root)
    return complete


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);a=parser.parse_args()
    print(json.dumps(finalize(a.root),ensure_ascii=False))

if __name__=='__main__':main()
