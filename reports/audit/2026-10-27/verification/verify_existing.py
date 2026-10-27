# AI-assisted verification runner, not assessed model development.
import json, pathlib, os, sys, io, contextlib, traceback, hashlib, csv
os.environ['MPLBACKEND']='Agg'
sys.path.insert(0,'/home/mohamedekaden/.cache/uv/archive-v0/olhT6-FLhuJNDgiS')
ROOT=pathlib.Path('/home/mohamedekaden/3.2/AI/SemProject_MLEngine/SemProject_MLEngine')
OUT=ROOT/'reports/verification_2026-10-27';OUT.mkdir(exist_ok=True)
os.chdir(ROOT)
import matplotlib.pyplot as plt
nb=json.loads((ROOT/'notebooks/roadwatch_dataset_cleaning.ipynb').read_text())
ns={}; log=[]; cells=[]
for i in [1,3,5,6,8,10]:
 c=json.loads(json.dumps(nb['cells'][i])); buf=io.StringIO();error=None
 try:
  with contextlib.redirect_stdout(buf):exec(compile(''.join(c['source']),f'cleaning-cell-{i}','exec'),ns)
 except Exception as e:error=e;buf.write(traceback.format_exc())
 c['execution_count']=len(cells)+1;c['outputs']=[{'output_type':'stream','name':'stdout','text':buf.getvalue().splitlines(True)}];cells.append(c)
 log.append(f'CELL {i}\n{buf.getvalue()}')
 for num in plt.get_fignums():plt.figure(num).savefig(OUT/f'cell_{i}_figure_{num}.png',bbox_inches='tight')
 plt.close('all')
 if error:break
# Execute the existing pHash algorithm for the original split, without pretending manifests exist.
s=''.join(nb['cells'][12]['source']).split('v2_assignment =')[0]
exec(s,ns)
pairs=ns['cross_split_pairs']({r['image']:r['original_split'] for r in ns['records']})
with (OUT/'original_cross_split_phash_pairs.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['left','right','distance']);w.writerows(pairs)
rows=ns['records'];C=ns['Counter'];D=ns['defaultdict']
summary={'scope':'Original data/1 only; AI-assisted verification, not original training evidence','checkpoint_sha256':hashlib.sha256((ROOT/'models/best.pt').read_bytes()).hexdigest(),'checkpoint_verified_against_expected':False,'splits':{},'source_groups':len({r['source'] for r in rows}),'source_groups_spanning_original_splits':len(ns['leaking_groups'](rows,'source','original_split')),'exact_duplicate_sets':len(ns['duplicate_sets']),'exact_hashes_spanning_original_splits':len(ns['leaking_groups'](rows,'sha256','original_split')),'cross_split_phash_pairs_distance_le_5':len(pairs),'invalid_images_or_labels':ns['errors'],'pairing_errors':ns['pairing_errors']}
for split in ns['SPLITS']:
 rr=[r for r in rows if r['original_split']==split];summary['splits'][split]={'images':len(rr),'boxes':sum(len(r['boxes']) for r in rr),'empty_labels':sum(not r['boxes'] for r in rr),'potholes_per_image':dict(sorted(C(len(r['boxes']) for r in rr).items()))}
with (OUT/'image_audit.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['image','original_split','source','boxes','sha256','phash']);w.writerows((r['image'],r['original_split'],r['source'],len(r['boxes']),r['sha256'],f"{r['phash']:016x}") for r in rows)
fig,axes=plt.subplots(1,3,figsize=(15,4))
for split in ns['SPLITS']:
 rr=[r for r in rows if r['original_split']==split];counts=C(len(r['boxes']) for r in rr);axes[0].plot(sorted(counts),[counts[k] for k in sorted(counts)],marker='.',label=split)
axes[0].set(xlabel='Potholes per image',ylabel='Images',title='Original split');axes[0].legend()
groups=C(r['source'] for r in rows);axes[1].hist(list(groups.values()),bins=20);axes[1].set(xlabel='Images per filename source group',ylabel='Groups')
areas=[w*h for r in rows for x,y,w,h in r['boxes']];axes[2].hist(areas,bins=40);axes[2].set(xlabel='Box area / image area',ylabel='Boxes')
fig.tight_layout();fig.savefig(OUT/'original_eda.png',dpi=160);plt.close('all')
summary['box_area_fraction_quantiles']=dict(zip(['min','25%','median','75%','max'],map(float,ns['np'].quantile(areas,[0,.25,.5,.75,1]))))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'execution.log').write_text('\n'.join(log)+'\nOriginal-split pHash pairs <=5: '+str(len(pairs))+'\n')
nb['cells']=[{'cell_type':'markdown','metadata':{},'source':['# AI-assisted partial verification\n','Executed selected existing cleaning cells on 27 October 2026. Original data only. Stops at missing v2 manifest. This is not original run evidence or a completed cleaning notebook.\n']}]+cells
(OUT/'cleaning_partial_execution.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
import platform,importlib.metadata
(OUT/'environment.txt').write_text(platform.python_version()+'\n'+'\n'.join(f'{m}=={importlib.metadata.version(m)}' for m in ['Pillow','numpy','matplotlib','scipy','ImageHash']))
print(json.dumps(summary,indent=2))
