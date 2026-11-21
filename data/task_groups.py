"""One family per draw; derived views and instruction variants share its weight."""
import copy
import hashlib
import json
from data import length_choices
from data.loss_permissions import image_loss_allowed

CONTRACT = 'family_uniform_view_and_text_seed_epoch_v1'
STAGE1_TRAJECTORIES = 'full_edit_gen80_und20_v1'


def choose(count, family_id, seed, epoch, slot):
    key=f'{CONTRACT}:{seed}:{epoch}:{family_id}:{slot}'
    index=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')%count
    return length_choices.index(count,index,('task',family_id,slot,count))


def select(row, seed, epoch, task=None, view_index=None):
    if (row.get('_trajectory_sampling') == STAGE1_TRAJECTORIES
            and view_index is None and task is None):
        view_index = trajectory_view(row, seed, epoch)
    if 'stage1_views' in row:
        return length_choices.task_draw(row,seed,epoch,task,view_index,_select)
    return _select(row,seed,epoch,task,view_index)


def trajectory_view(row, seed, epoch):
    source = row.get('source')
    if source not in ('imgedit', 'pico', 'seed', 'magicbrush'):
        return None
    if source != 'imgedit' and not row['metadata'].get('handoff_item_id'):
        return None
    views = row['views']
    full = [i for i,v in enumerate(views) if v['task'] == 'multi_edit']
    if not full:
        return None
    gate = row.get('stage1_views', [1] * len(views))
    key = f'phase1-trajectory:{seed}:{epoch}:{row["family_id"]}'
    gen = [i for i,v in enumerate(views) if gate[i] and v.get('positive_events')]
    understand = [i for i,v in enumerate(views) if gate[i] and not v.get('positive_events')]
    if gen and understand and source != 'imgedit':
        pool = (gen, understand)[length_choices.pick([4, 1], key + ':direction')]
    else:
        pool = [i for i,g in enumerate(gate) if g]
    weights = length_choices.branch([gate[i] for i in pool])
    weights = weights['weights'] if isinstance(weights, dict) else [1] * len(pool)
    i = pool[length_choices.pick(weights, key + ':view')]
    v = views[i]
    if v['task'] == 'edit_transition':
        for j in full:
            main = views[j]
            if (gate[j] != 1 or main['sequence'] != v['sequence']
                    or main.get('condition_variant') != v.get('condition_variant')):
                continue
            seq = row['sequences'][main['sequence']][:main['stop']]
            results = [k for k,e in enumerate(seq)
                       if e['kind'] == 'image' and e['role'] in ('assistant','tool')]
            if len(results) > 1 and set(results) == set(main['positive_events']):
                return j
    return i


def _select(row, seed, epoch, task=None, view_index=None):
    result = _render(row, seed, epoch, task, view_index)
    if row.get('_upstream_sampling') and row.get('format') == 'task_group':
        from data.upstream import task_record
        return task_record(row, result, seed, epoch)
    return result


def _render(row, seed, epoch, task=None, view_index=None):
    if row.get('format')!='task_group':return row
    views=row['views']
    indices=[i for i,v in enumerate(views) if task is None or v['task']==task]
    if not indices:raise ValueError('Family has no requested training view')
    index=indices[choose(len(indices),row['family_id'],seed,epoch,'view')] if view_index is None else view_index
    if index not in indices:raise ValueError('View does not belong to this family/task')
    v=views[index];kind=v['task'];segments=[];images=list(row['images'])
    selected_condition=(row['condition_variants'][v['condition_variant']] if 'condition_variant' in v
                        else dict(conditions=row.get('conditions'),layout=row.get('layout')))
    def t(value,role,supervised=True):
        segments.append(dict(kind='text',text=value,role=role,turn=len(segments),supervised=supervised))
    def im(i,loss=False):
        segments.append(dict(kind='image',image_index=i,role='assistant' if loss else 'user',
                             turn=len(segments),image_loss_allowed=loss))
    if kind=='description':
        candidates=v['text_candidates']
        caption=candidates[choose(len(candidates),row['family_id'],seed,epoch,'description')]['text']
        im(v['image_index']);t('Describe this image.','user',False);t(caption,'assistant')
    elif kind=='tagging':
        im(v['image_index']);t('List the annotated scene tags.','user',False)
        t(json.dumps(selected_condition['conditions']['tags'],ensure_ascii=False),'assistant')
    elif kind=='region_label':
        condition=selected_condition['conditions'];resolution=condition['resolution']
        objects=[o for k in ('object','background') for o in condition['segmentation'].get(k,[])
                 if o.get('mask') and o.get('class_name')]
        obj=objects[choose(len(objects),row['family_id'],seed,epoch,'region')]
        rle=dict(size=[resolution['height'],resolution['width']],counts=obj['mask'])
        payload=json.dumps(rle,sort_keys=True,separators=(',',':')).encode()
        images.append(dict(kind='rle_mask',rle=rle,sha256=hashlib.sha256(payload).hexdigest(),image_loss_allowed=False))
        im(v['image_index']);im(len(images)-1)
        t(json.dumps(dict(bbox=obj.get('bbox'),coordinate_system=condition['coordinate_system'],resolution=resolution)), 'user',False)
        t(obj['class_name'],'assistant')
    elif kind in ('layout_generation','region_localization'):
        layout=json.dumps(selected_condition['layout'],ensure_ascii=False,separators=(',',':'))
        if kind=='region_localization':
            im(v['image_index']);t('Locate the annotated objects.','user',False);t(layout,'assistant')
        else:
            t(layout,'user',False)
            captions=[c for c in row['descriptions'] if c['image_index']==v['image_index']]
            cap=captions[choose(len(captions),row['family_id'],seed,epoch,'layout_caption')]
            t(cap['text'],'user');im(v['image_index'],True)
    else:
        sequence=row.get('sequences',[row['events']])[v.get('sequence',0)]
        segments=copy.deepcopy(sequence[:v['stop']])
        positives=set(v['positive_events'])
        text_event=v.get('text_event')
        for i,e in enumerate(segments):
            if e['kind']=='image':e['image_loss_allowed']=bool(e['image_loss_allowed'] and i in positives)
            elif kind=='post_result_control':e['supervised']=False
            elif text_event is not None:e['supervised']=bool(e.get('supervised',True) and i==text_event)
        if kind=='post_result_control':
            t(v['text'],'assistant');segments[-1]['observes']=[v['image_index']]
        if kind=='region_edit':
            mask=dict(kind='image',image_index=v['mask_index'],role='user',turn=-1,image_loss_allowed=False,conditioning_mask=True)
            source=next(i for i,e in enumerate(segments) if e['kind']=='image' and e['image_index']==v['mask_source_index'])
            segments.insert(source+1,mask)
    choices=[]
    for i,event in enumerate(segments):
        candidates=event.get('text_candidates')
        if candidates:
            n=choose(len(candidates),row['family_id'],seed,epoch,f'{index}:{i}')
            event['text']=candidates[n]['text'];choices.append(dict(event=i,choice=n,provenance=candidates[n]['provenance']))
    allowed=any(s['kind']=='image' and s['image_loss_allowed'] for s in segments)
    images=[dict(ref,preserve_full_frame=True) for ref in images]
    return dict(row,format='causal_sequence',segments=segments,images=images,direction='T2I' if allowed else 'I2T',
        source_shard='',text_tokens=None,image_loss_allowed=image_loss_allowed(row) and allowed,
        training_view=dict(task=kind,index=index,family_id=row['family_id'],text_choices=choices),
        typed_conditions={**{k:v for k,v in selected_condition.items() if v is not None},
                          **{k:row[k] for k in ('mask_bindings',) if k in row}})


def preference(row, index=0):
    """Typed pair for a preference objective, never flattened into positive SFT."""
    p=(row['preferences'] if 'preferences' in row else [row['preference']])[index]
    if row.get('_upstream_sampling') and row.get('_upstream_binding'):
        from data.upstream import with_roots
        pairs=[(branch, item) for branch in row['_upstream_binding']['branches'] for item in branch.get('preferences',())]
        branch, item=pairs[index]
        def signature(record, pref):
            indices=[pref['source_index'],pref['chosen_index'],*pref['rejected_indices']]
            return pref['source_family_id'],pref['instruction'],[record['images'][i]['exact_identity'] for i in indices]
        if signature(row,p)!=signature(branch,item):
            raise ValueError('Preference no longer matches its original upstream pair')
        result=preference(branch,branch['preferences'].index(item))
        result['input_images']=with_roots(result['input_images'],row)
        result['chosen']=with_roots([result['chosen']],row)[0]
        result['rejected']=with_roots(result['rejected'],row)
        return result
    return dict(family_id=row['family_id'],source=p.get('source',row['source']),source_family_id=p.get('source_family_id'),instruction=p['instruction'],
        input_images=[dict(row['images'][p['source_index']],image_loss_allowed=False)],
        chosen=dict(row['images'][p['chosen_index']],preference_role='chosen'),
        rejected=[dict(row['images'][i],preference_role='rejected',image_loss_allowed=False) for i in p['rejected_indices']],
        objective='paired_preference',positive_sft_rejected=False)
