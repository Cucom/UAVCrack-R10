"""Export six fixed-shape FP32 models, without a training framework."""
import argparse,json,sys
from pathlib import Path
import torch,onnx
from models import build
from predict_onnx import sha
sys.dont_write_bytecode=True

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    rows=json.loads((a.weights_dir/'WEIGHTS.json').read_text())['files'];records=[]
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for e in rows:
        f=a.weights_dir/e['file'];assert sha(f)==e['sha256'];m=build(e['family']);m.load_state_dict(torch.load(f,map_location='cpu',weights_only=True),strict=True);m.eval();dest=a.output_dir/f"{e['family']}_seed{e['seed']}.onnx"
        if dest.exists():raise FileExistsError(dest)
        x=torch.zeros(1,3,378,672)
        torch.onnx.export(m,(x,),str(dest),input_names=['image'],output_names=['logits'],opset_version=17,dynamo=False,external_data=False,do_constant_folding=True)
        g=onnx.load(dest);shape=[1,2 if e['family']=='mit_b2' else 1,378,672]
        for dim,size in zip(g.graph.output[0].type.tensor_type.shape.dim,shape):dim.ClearField('dim_param');dim.dim_value=size
        # Source paths/call stacks are not part of the graph's computations.
        g.doc_string='';g.graph.doc_string=''
        for node in g.graph.node:node.doc_string=''
        for k,v in {'family':e['family'],'seed':str(e['seed']),'source_weight_sha256':e['sha256'],'input_rgb_range':'0..255' if e['family']=='mit_b2' else '0..1','output':'native-resolution logits; family-specific sigmoid/softmax and fusion outside graph'}.items():
            entry=g.metadata_props.add();entry.key=k;entry.value=v
        assert not any(t.data_location==onnx.TensorProto.EXTERNAL for t in g.graph.initializer)
        onnx.checker.check_model(g,full_check=True);onnx.save(g,dest)
        records.append({'family':e['family'],'seed':e['seed'],'file':dest.name,'sha256':sha(dest),'bytes':dest.stat().st_size,'source_weight_sha256':e['sha256'],'output_shape':shape,'input_range':[0,255] if e['family']=='mit_b2' else [0,1]})
        print(f'Exported {dest.name} {dest.stat().st_size} bytes',flush=True);del m,g
    manifest={'official_method':'UAVCrack-R10','format':'ONNX FP32','opset':17,'input_shape':[1,3,378,672],'files':records,'fusion_weights':[.65,.25,.10],'threshold':.4,'torch':torch.__version__,'onnx':onnx.__version__}
    (a.output_dir/'ONNX_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
