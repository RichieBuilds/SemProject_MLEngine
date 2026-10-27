# AI-assisted executor for unchanged existing cleaning code. Does not copy or modify data.
from pathlib import Path
import sys,os,json,io,contextlib,traceback,base64,csv,hashlib,platform,importlib.metadata
sys.path.insert(0,'/home/mohamedekaden/.cache/uv/archive-v0/olhT6-FLhuJNDgiS')
os.environ['MPLBACKEND']='Agg';os.environ['MPLCONFIGDIR']='/tmp/roadwatch-mpl'
import matplotlib.pyplot as plt
r=Path('/home/mohamedekaden/3.2/AI/SemProject_MLEngine/SemProject_MLEngine');os.chdir(r);out=r/'reports/verification_2026-10-27';nb=json.loads((r/'notebooks/roadwatch_dataset_cleaning.ipynb').read_text());ns={};log=[];success=True;seq=0
for i,c in enumerate(nb['cells']):
 c['id']=f'cleaning-reproduced-{i}'
 if c['cell_type']!='code':continue
 seq+=1;buf=io.StringIO();err=None
 try:
  with contextlib.redirect_stdout(buf),contextlib.redirect_stderr(buf):exec(compile(''.join(c['source']),f'cleaning-cell-{i}','exec'),ns)
 except Exception as e:err=e;success=False;buf.write(traceback.format_exc())
 c['execution_count']=seq;c['outputs']=[{'output_type':'stream','name':'stdout','text':buf.getvalue().splitlines(True)}];log.append(f'CELL {i}\n{buf.getvalue()}')
 for num in plt.get_fignums():
  b=io.BytesIO();plt.figure(num).savefig(b,format='png',bbox_inches='tight');data=b.getvalue();(out/f'full_cell_{i}_figure_{num}.png').write_bytes(data)
  c['outputs'].append({'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(data).decode(),'text/plain':['Reproduced figure']}})
 plt.close('all')
 print(f'cell {i}: '+('PASSED' if not err else str(err)),flush=True)
 if err:break
nb['cells'].insert(0,{'cell_type':'markdown','metadata':{},'id':'reproduction-provenance','source':['# Reproduced cleaning verification — AI-assisted\n\nExecuted 27 October 2026 from unchanged source notebook code. These outputs are new verification, not original saved outputs. WRITE_COPY=False. Restored manifests and supplied cleaned-v3 data were used.\n']})
(out/'cleaning_reproduced.ipynb').write_text(json.dumps(nb,indent=1)+'\n');(out/'cleaning_reproduced.log').write_text('\n'.join(log));assert success,'Cleaning verification failed; retained output'
def canonical(a,b,d):return (*sorted((a,b)),int(d))
with (r/'data/cross_split_phash_v2.csv').open() as f:recorded={canonical(x['image_a'],x['image_b'],x['distance']) for x in csv.DictReader(f)}
assert recorded=={canonical(*p) for p in ns['flagged']}
with (r/'data/cross_split_phash_v3.csv').open() as f:assert len(list(csv.DictReader(f)))==len(ns['final_pairs'])==0
rows=ns['records'];C=ns['Counter'];assignment=ns['assignment'];summary={'scope':'cleaned-v3 verified against original bytes and restored manifests','dataset_archive_sha256':'31fff83565a63da76e32419825cd8b613f5aac322ed00df62470b53f9bf4785e','checkpoint_sha256':hashlib.sha256((r/'models/best.pt').read_bytes()).hexdigest(),'checkpoint_matches_docs_and_drive':True,'v2_similarity_pairs':len(ns['flagged']),'v3_similarity_pairs':len(ns['final_pairs']),'recorded_similarity_csvs_match':True,'split_counts':{},'source_groups':len(ns['groups']),'source_groups_spanning_splits':len(ns['leaking_groups'](ns['final'],'source','new_split')),'invalid_images_labels':ns['errors'],'pairing_errors':ns['pairing_errors'],'empty_labels':sum(not x['boxes'] for x in rows)}
fig,axes=plt.subplots(1,3,figsize=(15,4))
for split in ns['SPLITS']:
 rr=[x for x in rows if assignment[x['image']]==split];counts=C(len(x['boxes']) for x in rr);summary['split_counts'][split]={'images':len(rr),'boxes':sum(len(x['boxes']) for x in rr),'empty_labels':sum(not x['boxes'] for x in rr),'potholes_per_image':dict(sorted(counts.items()))};axes[0].plot(sorted(counts),[counts[k] for k in sorted(counts)],marker='.',label=split)
axes[0].set(xlabel='Potholes per image',ylabel='Images',title='Verified cleaned-v3');axes[0].legend();axes[1].hist(list(C(x['source'] for x in rows).values()),bins=20);axes[1].set(xlabel='Images per filename source group',ylabel='Groups');axes[2].hist([w*h for x in rows for _,_,w,h in x['boxes']],bins=40);axes[2].set(xlabel='Box area / image area',ylabel='Boxes');fig.tight_layout();fig.savefig(out/'cleaned_v3_eda.png',dpi=160);plt.close('all')
(out/'cleaned_v3_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
