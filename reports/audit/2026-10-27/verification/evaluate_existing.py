# AI-assisted evaluation runner: no training or parameter search.
from pathlib import Path
import os,json,hashlib,platform,subprocess,time
os.environ['YOLO_CONFIG_DIR']='/tmp/roadwatch-yolo';os.environ['MPLCONFIGDIR']='/tmp/roadwatch-mpl'
import torch,ultralytics,yaml
from ultralytics import YOLO
r=Path('/home/mohamedekaden/3.2/AI/SemProject_MLEngine/SemProject_MLEngine');out=r/'reports/verification_2026-10-27';data=r/'data/cleaned';model_path=r/'models/best.pt'
verified=json.loads((out/'cleaned_v3_summary.json').read_text());assert verified['recorded_similarity_csvs_match'] and verified['v3_similarity_pairs']==0
assert hashlib.sha256(model_path.read_bytes()).hexdigest()=='9dad6abd86b08e02d76a02f5e0190ddf373b39de9b8487646f01a541e8810291'
cfg=out/'evaluation_data.yaml';cfg.write_text(yaml.safe_dump({'path':str(data),'train':'train/images','val':'valid/images','test':'test/images','nc':1,'names':{0:'pothole'}}))
torch.set_num_threads(4)
conditions={'provenance':'AI-assisted reproduced evaluation, not original run output','python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'device':'cpu','threads':4,'imgsz':640,'batch':16,'workers':0,'half':False,'augment':False,'conf':0.001,'iou':0.7,'max_det':300,'original_recorded_environment':{'python':'3.13.16','torch':'2.11.0+cu130','ultralytics':'8.4.189','device':'Tesla T4'},'checkpoint_sha256':verified['checkpoint_sha256']}
(out/'evaluation_conditions.json').write_text(json.dumps(conditions,indent=2)+'\n')
(out/'evaluation_packages.txt').write_text(subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True) if __import__('importlib').util.find_spec('pip') else '\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in __import__('importlib.metadata').metadata.distributions()))+'\n')
model=YOLO(str(model_path));(out/'loaded_architecture.json').write_text(json.dumps(model.model.yaml,indent=2)+'\n')
results={}
for split in ['val','test']:
 start=time.time();m=model.val(data=str(cfg),split=split,imgsz=640,batch=16,device='cpu',workers=0,half=False,augment=False,conf=0.001,iou=0.7,max_det=300,plots=True,save_json=True,project=str(out),name='reproduced_'+split,exist_ok=False)
 results[split]={'precision':float(m.box.mp),'recall':float(m.box.mr),'mAP50':float(m.box.map50),'mAP50_95':float(m.box.map),'speed_ms':m.speed,'elapsed_seconds':time.time()-start,'output_directory':str(m.save_dir)}
 (out/'evaluation_metrics.json').write_text(json.dumps(results,indent=2)+'\n');print('REPRODUCED',split,json.dumps(results[split]),flush=True)
