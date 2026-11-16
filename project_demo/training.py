"""Small single-GPU NeMo AutoModel LoRA loop with actual save/reload verification.

Uses the NVIDIA AutoModel loader and NVIDIA's LoRA patcher, not HF PEFT.
This teaching loop computes standard next-token loss directly on logits.
"""
from pathlib import Path
import json
from .common import read_jsonl, save_json, fingerprint, edit_distance, words

def assert_prompt_prefix(prompt_ids, full_ids):
    """Do not silently mask answer tokens when a model changes chat boundaries."""
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError('Assistant example does not preserve the exact generation-prompt token prefix')
    if len(full_ids) <= len(prompt_ids):
        raise ValueError('Assistant example has no answer tokens')
    return len(prompt_ids)

def audio_fingerprint(waveform):
    import hashlib
    import numpy as np
    return hashlib.sha256(np.asarray(waveform,dtype='<f4').tobytes()).hexdigest()

def train_lora(config, output, modality, train_rows, heldout_rows):
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer, AutoProcessor
    from nemo_automodel import (NeMoAutoModelForCausalLM,
        NeMoAutoModelForImageTextToText, NeMoAutoModelForMultimodalLM)
    from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules, PeftConfig
    if not torch.cuda.is_available():
        raise RuntimeError("AutoModel LoRA requires a CUDA GPU; no CPU/fake training fallback")
    if not train_rows or not heldout_rows:
        raise ValueError("Training and held-out examples must both be nonempty")
    train_ids={r['id'] for r in train_rows}
    if train_ids.intersection(r['id'] for r in heldout_rows):
        raise ValueError("Training and held-out IDs overlap")
    torch.manual_seed(int(config.get('seed',42)))
    name=config['hf_model']
    revision=config.get('model_revision') or HfApi().model_info(name).sha
    snapshot_download(name,revision=revision)
    loaders={'text':NeMoAutoModelForCausalLM,'image':NeMoAutoModelForImageTextToText,
             'audio':NeMoAutoModelForMultimodalLM}
    processor=(AutoTokenizer if modality=='text' else AutoProcessor).from_pretrained(name,revision=revision)
    tokenizer=getattr(processor,'tokenizer',processor)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model=loaders[modality].from_pretrained(name,revision=revision,torch_dtype=torch.bfloat16,
                                          force_hf=(modality in ('text','audio')),attn_implementation='sdpa').to('cuda')
    if hasattr(model,'disable_talker'): model.disable_talker()
    if modality=='audio' and hasattr(model,'thinker'): model=model.thinker
    peft=PeftConfig(target_modules=['*.q_proj','*.v_proj'],dim=8,alpha=16,use_triton=False)
    patched=apply_lora_to_linear_modules(model,peft)
    save_json(output/'lora-targets.json',{'matched_modules':patched,'adapter_parameter_names':[n for n,p in model.named_parameters() if p.requires_grad]})
    if patched==0: raise RuntimeError('No modules matched the NVIDIA LoRA configuration')
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if not parameters: raise RuntimeError('No trainable adapter parameters')
    initial={n:p.detach().cpu().clone() for n,p in parameters.items()}
    optimizer=torch.optim.AdamW(parameters.values(),lr=float(config.get('learning_rate',1e-4)))

    def encode(row, answer=False):
        contents=[{'type':'text','text':row['prompt']}]
        if modality=='image':
            from PIL import Image
            media=[Image.open(p).convert('RGB') for p in row['images']]
            contents += [{'type':'image','image':im} for im in media]
        elif modality=='audio':
            import soundfile as sf
            import numpy as np
            from scipy.signal import resample_poly
            from math import gcd
            media=[]
            for p in row['audio']:
                wave,rate=sf.read(p,dtype='float32',always_2d=True)
                wave=wave.mean(axis=1)
                d=gcd(rate,16000)
                if rate!=16000: wave=resample_poly(wave,16000//d,rate//d).astype(np.float32)
                if len(wave)<16000: raise ValueError('Training audio must be at least one second')
                media.append(wave)
            contents += [{'type':'audio','audio':a} for a in media]
        messages=[{'role':'user','content':row['prompt'] if modality=='text' else contents}]
        if answer: messages.append({'role':'assistant','content':row['answer']})
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=not answer)
        kwargs={'text':text,'return_tensors':'pt','add_special_tokens':False}
        if modality=='image': kwargs['images']=media
        if modality=='audio': kwargs['audio']=media
        batch=processor(**kwargs)
        return {k:v.to(device='cuda',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                for k,v in batch.items() if isinstance(v,torch.Tensor)}

    def predict(row):
        batch=encode(row)
        extra={'return_audio':False} if hasattr(model,'disable_talker') else {}
        with torch.inference_mode():
            ids=model.generate(**batch,max_new_tokens=96,do_sample=False,**extra)
        if isinstance(ids,tuple): ids=ids[0]
        return tokenizer.decode(ids[0,batch['input_ids'].shape[1]:],skip_special_tokens=True).strip()

    model.eval()
    before=[predict(r) for r in heldout_rows]
    model.train()
    losses=[]
    evaluation_adapter=config.get('evaluate_adapter')
    max_steps=0 if evaluation_adapter else int(config.get('steps',20))
    if not evaluation_adapter and not 1<=max_steps<=1000: raise ValueError('steps must be between 1 and 1000')
    for step in range(max_steps):
        row=train_rows[step % len(train_rows)]
        batch=encode(row,answer=True)
        prompt_ids=encode(row,answer=False)['input_ids'][0].tolist()
        prompt_length=assert_prompt_prefix(prompt_ids,batch['input_ids'][0].tolist())
        labels=batch['input_ids'].clone()
        labels[:,:prompt_length]=-100
        if not torch.any(labels!=-100): raise ValueError('No assistant tokens remain for training')
        optimizer.zero_grad(set_to_none=True)
        result=model(**batch,use_cache=False)
        logits=result.logits if hasattr(result,'logits') else result[0]
        loss=torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),
                                                labels[:,1:].reshape(-1),ignore_index=-100)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.0)
        if not torch.isfinite(grad_norm): raise RuntimeError('Nonfinite adapter gradient')
        optimizer.step()
        losses.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(grad_norm)})
        save_json(output/'training-loss.json',losses)
    changed=any(not torch.equal(initial[n],p.detach().cpu()) for n,p in parameters.items())
    if evaluation_adapter:
        checkpoint=Path(evaluation_adapter)
        source_config=json.loads((checkpoint.parent/'adapter-config.json').read_text(encoding='utf-8'))
        if source_config['base_model']!=name or source_config['revision']!=revision:
            raise ValueError('Adapter does not match the configured base model and pinned revision')
    else:
        if not changed: raise RuntimeError('Training completed without updating any adapter weight')
        adapter={n:p.detach().cpu() for n,p in parameters.items()}
        checkpoint=output/'adapter.pt'
        torch.save(adapter,checkpoint)
    # Deliberately clear adapters before reloading; equality tests the persisted bytes.
    with torch.no_grad():
        for p in parameters.values(): p.zero_()
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if set(loaded)!=set(parameters): raise ValueError('Adapter parameter names do not match this model')
    model.load_state_dict(loaded,strict=False)
    if any(not torch.equal(p.detach().cpu(),loaded[n]) for n,p in parameters.items()):
        raise RuntimeError('Adapter save/reload equality check failed')
    model.eval()
    after=[predict(r) for r in heldout_rows]
    comparison=[{'id':r['id'],'reference':r['answer'],'before':b,'after':a}
                for r,b,a in zip(heldout_rows,before,after)]
    save_json(output/'heldout-comparison.json',comparison)
    result={'backend':'NVIDIA NeMo AutoModel + NVIDIA LoRA','model':name,'revision':revision,
            'modality':modality,'steps':max_steps,'patched_modules':patched,
            'trainable_parameters':sum(p.numel() for p in parameters.values()),
            'actual_weight_update':changed,'adapter_reload_exact':True,
            'train_ids':[r['id'] for r in train_rows],'heldout_ids':[r['id'] for r in heldout_rows],
            'heldout_exact_match_before':sum(b==r['answer'] for b,r in zip(before,heldout_rows))/len(heldout_rows),
            'heldout_exact_match_after':sum(a==r['answer'] for a,r in zip(after,heldout_rows))/len(heldout_rows),
            'losses':losses,'checkpoint':str(checkpoint),'evidence_kind':'real_gpu_adapter_evaluation' if evaluation_adapter else 'real_gpu_lora',
            'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated()}
    save_json(output/'adapter-config.json',{'base_model':name,'revision':revision,
             'peft':{'target_modules':['*.q_proj','*.v_proj'],'dim':8,'alpha':16,'use_triton':False}})
    return result

def train_text_lora(config,output):
    from .pipeline import load_splits
    splits=load_splits(config,'live')
    def convert(rows):
        return [{'id':r['id'],'prompt':'Correct this ordered ASR N-best list. Return only the transcript.\n'+json.dumps(r['hypotheses']),
                 'answer':r['reference']} for r in rows]
    return train_lora(config,output,'text',convert(splits['train']),convert(splits['test'][:int(config.get('limit',3))]))

def train_audio_lora(config,output):
    rows=read_jsonl(config['training_manifest'])
    warmup=config.get('learning_task')=='asr_warmup'
    if warmup:
        if any(r.get('source')!='authored_synthetic_asr' or r.get('learning_task')!='asr_warmup' for r in rows):
            raise ValueError('ASR warm-up only accepts explicitly authored synthetic transcription data')
    elif any(r.get('review_status')!='approved' or r.get('source')!='independent_recording' for r in rows):
        raise ValueError('EQ adaptation requires human-approved independent recordings; SpeechEQ test data cannot be trained on')
    benchmark_path=Path(config['data_dir'])/'benchmark.jsonl'
    benchmark_rows=read_jsonl(benchmark_path) if benchmark_path.exists() else []
    benchmark_ids={r['dataset_id'] for r in benchmark_rows}
    from .pipeline import decode_audio
    benchmark_audio={audio_fingerprint(decode_audio(t[k])) for r in benchmark_rows
                     for t in r['conversation'] for k in ('audio_high','audio_low') if t.get(k)}
    if any(r['id'] in benchmark_ids for r in rows): raise ValueError('Benchmark ID found in training manifest')
    import hashlib
    base=Path(config['training_manifest']).resolve().parent
    seen={}
    for row in rows:
        paths=[str((base/p).resolve()) for p in row['audio']]
        for p in paths:
            import base64
            digest=audio_fingerprint(decode_audio(base64.b64encode(Path(p).read_bytes()).decode()))
            if digest in benchmark_audio:
                raise ValueError('Benchmark audio content found in training data, even under a renamed ID')
            if digest in seen and seen[digest]!=row['split']:
                raise ValueError('Identical audio content leaks across training and validation')
            seen[digest]=row['split']
        row['audio']=paths
    train=[r for r in rows if r['split']=='train']
    heldout=[r for r in rows if r['split']=='validation']
    result=train_lora(config,output,'audio',train,heldout)
    result['learning_task']='synthetic ASR warm-up, not validated EQ training' if warmup else 'independently reviewed EQ adaptation'
    return result

def train_image_lora(config,output):
    from .pipeline import load_real_splits
    splits=load_real_splits(config)
    def convert(rows):
        return [{'id':r['id'],'prompt':'Transcribe the Suzhou numeral image to Arabic digits. Output digits only. If unreadable, output unreadable.',
                 'images':[r['image_path']],'answer':r['label']} for r in rows]
    n=int(config.get('train_limit',16))
    return train_lora(config,output,'image',convert(splits['train'][:n]),convert(splits['test'][:int(config.get('limit',3))]))
