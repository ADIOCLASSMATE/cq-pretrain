"""Run complete OpenUni metadata planning and bounded real-record adapter canaries."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from corpus_pipeline.production import genund_collections as g
from corpus_pipeline.production.common import atomic_json


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); out=Path(args.output); binding=json.loads((out/'DATASET_BINDING.json').read_text())
    spec={'adapter':'genund_openuni','collection':'genund','root':args.root,'dataset_id':binding['dataset_id']}
    reports=[]
    examples=[('cc12m','data.json'),('laion6m','data.json'),('megalith-10m','megalith10m_all.json'),
              ('redcaps5m_resized','redcaps5m_data.json'),('text-to-image-2M','data/data_1024_10K.json')]
    for subset,name in examples:
        prefix='openuni/data/'+subset; path=Path(args.root)/prefix/name
        image_root=prefix+'/raw'+('/'+Path(name).stem if subset=='text-to-image-2M' else '')
        unit={'source_shard':prefix+'/'+name,'subset':subset,'image_root':image_root,
              'annotation_root':prefix+'/captions' if subset in {'cc12m','laion6m','megalith-10m'} else image_root}
        try:
            index,row=next(g.iter_json_records(path))
            with g.Storage(args.root,binding['dataset_id']) as store: values=list(g.openuni_record(unit,row,index,store))
            result={'subset':subset,'records':len(values),'issues':[r['metadata']['issues'] for r in values],
                    'image_refs':[r['source_image_refs'] for r in values],'caption_lengths':[len(r['raw']['segments'][0]['text']) for r in values],
                    'directions':[r['raw']['direction'] for r in values], 'image_origin':[r['raw'].get('image_origin') for r in values]}
        except Exception as exc: result={'subset':subset,'error_type':type(exc).__name__,'error':str(exc)}
        reports.append(result); atomic_json(out/'openuni_adapter_canary.json',reports)
    units=g.enumerate_units(spec,out/'named_collection_planning')
    atomic_json(out/'openuni_units.json',{'complete':True,'spec':spec,'units':units,'unit_count':len(units),
                'source_modified':False,'annotation_records':sum(u.get('row_end',0)-u.get('row_start',0) for u in units)})
    print(json.dumps({'complete':True,'units':len(units)}))


if __name__=='__main__': main()
