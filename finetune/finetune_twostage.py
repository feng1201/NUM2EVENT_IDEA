



import os ,json ,math ,argparse ,random
from dataclasses import dataclass
import sys ,os
sys .path .append (os .path .dirname (os .path .dirname (__file__ )))
from typing import List ,Dict ,Any
from safetensors .torch import save_file
import torch
from datasets import load_dataset ,Dataset ,DatasetDict
from transformers import (AutoTokenizer ,AutoModelForCausalLM ,
Trainer ,TrainingArguments ,set_seed )
from transformers .trainer_callback import TrainerCallback ,TrainerControl ,TrainerState
from peft import LoraConfig ,get_peft_model
from transformers import TrainerCallback

from finetune .ts_encoder_ex import DualTSMLP
from finetune .qwen_with_ts_ex4sft import QwenWithTSEmbed
from peft import PeftModel
MARK_TS3M ="<ts3m><ts3m/>"
MARK_TSD ="<tsdot><tsdot/>"

import os
import math
import json
from typing import List ,Dict ,Any
from dataclasses import dataclass
from transformers import TrainerCallback

def _ts_config_from (model ):
    ts =model .tsenc
    return {
    "d_model":getattr (ts ,"d_model",None ),
    "patch_size_ot":getattr (ts ,"patch_size_ot",None ),
    "patch_size_dot":getattr (ts ,"patch_size_dot",None ),
    "hidden":getattr (ts ,"hidden",None ),
    "layers":getattr (ts ,"layers",None ),
    "concat_posidx":getattr (ts ,"concat_posidx",False ),
    }

class SaveEveryNEpochsCallback (TrainerCallback ):
    def __init__ (self ,n =1 ,out_dir ="epoch_ckpts",stage =None ):
        self .n =max (1 ,int (n ))
        self .out_dir =out_dir
        self .stage =stage
        os .makedirs (self .out_dir ,exist_ok =True )

    def on_epoch_end (self ,args ,state ,control ,**kwargs ):
        trainer_obj =getattr (self ,"trainer",None )
        if trainer_obj is None :
            print ("trainer_obj")
            return control

        if not trainer_obj .args .local_rank ==0 :
            print ("rank0")
            return control

        if state .epoch is None :
            print ("state.epoch")
            return control
        ep =int (math .floor (state .epoch ))
        if ep %self .n !=0 or ep <=0 :
            print ("ep")
            return control

        ckpt_dir =os .path .join (self .out_dir ,f"epoch_{ep :02d}")
        os .makedirs (ckpt_dir ,exist_ok =True )

        if self .stage ==1 :

            ts_cfg =_ts_config_from (trainer_obj .model )
            with open (os .path .join (ckpt_dir ,"ts_config.json"),"w",encoding ="utf-8")as f :
                json .dump (ts_cfg ,f ,ensure_ascii =False ,indent =2 )
            save_file (trainer_obj .model .tsenc .state_dict (),
            os .path .join (ckpt_dir ,"ts_encoder.safetensors"))
            if trainer_obj .tokenizer is not None :
                trainer_obj .tokenizer .save_pretrained (ckpt_dir )
            print (f"[ckpt][stage1] saved TS encoder to {ckpt_dir }")
        else :
            qwen =trainer_obj .model .qwen
            if isinstance (qwen ,PeftModel ):
                qwen .save_pretrained (ckpt_dir )
                print (f"[ckpt][stage2] saved LoRA adapter to {ckpt_dir }")
            else :
                trainer_obj .model .save_pretrained (ckpt_dir )
                print (f"[ckpt][stage2][WARN] model.qwen is not PeftModel; saved full model instead.")
            ts_cfg =_ts_config_from (trainer_obj .model )
            with open (os .path .join (ckpt_dir ,"ts_config.json"),"w",encoding ="utf-8")as f :
                json .dump (ts_cfg ,f ,ensure_ascii =False ,indent =2 )
            save_file (trainer_obj .model .tsenc .state_dict (),
            os .path .join (ckpt_dir ,"ts_encoder.safetensors"))
            if trainer_obj .tokenizer is not None :
                trainer_obj .tokenizer .save_pretrained (ckpt_dir )
        return control

class LogEvalLossCallback (TrainerCallback ):
    def on_evaluate (self ,args ,state ,control ,metrics =None ,**kwargs ):
        if not metrics :
            print ("[eval] metrics is None/empty")
            return
        ep =f"{state .epoch :.2f}"if getattr (state ,"epoch",None )is not None else "?"
        if "eval_loss"in metrics :
            print (f"[eval] epoch={ep } eval_loss={metrics ['eval_loss']:.6f}")
        else :
            print (f"[eval] epoch={ep } keys={list (metrics .keys ())} (no eval_loss)")

def read_jsonl (path )->List [Dict [str ,Any ]]:

    rows =[]
    with open (path ,"r",encoding ="utf-8")as f :
        for line in f :
            line =line .strip ()
            if not line :
                continue
            try :

                rows .append (json .loads (line ))
            except Exception :

                pass
    return rows

def derive_prompt_answer_from_messages (tokenizer ,messages :List [Dict [str ,str ]]):
    ctx =[]
    for msg in messages :
        role ,content =msg .get ("role",""),msg .get ("content","")
        if role =="assistant":
            prompt =tokenizer .apply_chat_template (ctx ,tokenize =False ,add_generation_prompt =True ,enable_thinking =True )
            answer =content
            return prompt ,answer
        ctx .append ({"role":role ,"content":content })
    raise ValueError ("No assistant message found in messages")

@dataclass
class Collator :
    def __call__ (self ,features :List [Dict [str ,Any ]]):
        batch ={
        "prompt":[f ["prompt"]for f in features ],
        "answer":[f ["answer"]for f in features ],
        "ts3m":[f ["ts3m"]for f in features ],
        "tsdot":[f ["tsdot"]for f in features ],
        }
        return batch
def build_args ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--base_model",type =str ,required =True ,help ="HF")
    ap .add_argument ("--train_jsonl",type =str ,required =True ,help ="JSONL")
    ap .add_argument ("--valid_jsonl",type =str ,required =True ,help ="JSONL")
    ap .add_argument ("--out_dir",type =str ,required =True ,help ="")
    ap .add_argument ("--stage",type =int ,default =1 ,choices =[1 ,2 ],
    help ="：1=MLP；2=，QwenLoRA")
    ap .add_argument ("--use_ts",action ="store_true",help ="（）")
    ap .add_argument ("--no_use_ts",dest ="use_ts",action ="store_false",help ="")
    ap .set_defaults (use_ts =True )
    ap .add_argument ("--patch_size_ot",type =int ,default =8 ,help ="patch")
    ap .add_argument ("--patch_size_dot",type =int ,default =4 ,help ="patch")
    ap .add_argument ("--ts_hidden",type =int ,default =256 ,help ="MLP")
    ap .add_argument ("--ts_layers",type =int ,default =3 ,help ="MLP")
    ap .add_argument ("--ts_concat_posidx",action ="store_true",help ="")
    ap .add_argument ("--lora_r",type =int ,default =16 ,help ="LoRA（）")
    ap .add_argument ("--lora_alpha",type =int ,default =32 ,help ="LoRA")
    ap .add_argument ("--lora_dropout",type =float ,default =0.05 ,help ="LoRAdropout")
    ap .add_argument ("--epochs",type =int ,default =3 ,help ="")
    ap .add_argument ("--batch_size",type =int ,default =4 ,help ="")
    ap .add_argument ("--grad_accum",type =int ,default =1 ,help ="")
    ap .add_argument ("--lr",type =float ,default =1e-4 ,help ="")
    ap .add_argument ("--weight_decay",type =float ,default =0.0 ,help ="")
    ap .add_argument ("--warmup_ratio",type =float ,default =0.03 ,help ="")
    ap .add_argument ("--seed",type =int ,default =42 ,help ="，")
    ap .add_argument ("--save_every_epoch",type =int ,default =1 ,help ="")
    ap .add_argument ("--bf16",action ="store_true",help ="bfloat16")
    ap .add_argument ("--fp16",action ="store_true",help ="float16")
    ap .add_argument ("--gradient_checkpointing",action ="store_true",help ="")
    ap .add_argument ("--logging_steps",type =int ,default =10 ,help ="")
    ap .add_argument ("--eval_steps",type =int ,default =10 ,help ="")
    ap .add_argument ("--save_steps",type =int ,default =0 ,help ="（0）")
    ap .add_argument ("--stage1_dir",type =str ,default =None ,
    help ="（ts_encoder.pt）")
    return ap .parse_args ()


def count_trainable (module ):
    t =sum (p .numel ()for p in module .parameters ()if p .requires_grad )
    a =sum (p .numel ()for p in module .parameters ())
    return t ,a

def main ():
    args =build_args ()
    os .makedirs (args .out_dir ,exist_ok =True )
    set_seed (args .seed )
    tok =AutoTokenizer .from_pretrained (args .base_model ,use_fast =True )

    if tok .pad_token is None :
        tok .pad_token =tok .eos_token
    base_llm =AutoModelForCausalLM .from_pretrained (
    args .base_model ,
    torch_dtype =torch .bfloat16 if args .bf16 else None ,
    low_cpu_mem_usage =True ,
    device_map =None ,
    )
    if hasattr (base_llm ,"config"):
        base_llm .config .use_cache =False
    if args .gradient_checkpointing and hasattr (base_llm ,"gradient_checkpointing_enable"):
        base_llm .gradient_checkpointing_enable ()
    print ("LLM")
    print (base_llm .config .hidden_size )
    tsenc =DualTSMLP (
    d_model =base_llm .config .hidden_size ,
    patch_size_ot =args .patch_size_ot ,
    patch_size_dot =args .patch_size_dot ,
    hidden =args .ts_hidden ,
    layers =args .ts_layers ,
    concat_posidx =args .ts_concat_posidx ,
    )
    model =QwenWithTSEmbed (base_llm ,tsenc ,tok )
    if args .stage ==1 and args .use_ts :
        for p in model .qwen .parameters ():
            p .requires_grad =False
        for p in model .tsenc .parameters ():
            p .requires_grad =True
    elif args .stage ==2 and args .use_ts :
        src_dir =args .stage1_dir if args .stage1_dir else args .out_dir
        path_st =os .path .join (src_dir ,"ts_encoder.safetensors")
        path_pt =os .path .join (src_dir ,"ts_encoder.pt")
        if os .path .exists (path_st ):
            from safetensors .torch import load_file
            state =load_file (path_st ,device ="cpu")
            model .tsenc .load_state_dict (state ,strict =True )
            print (f"[Stage-2] Loaded TS encoder from {path_st }")
        elif os .path .exists (path_pt ):
            state =torch .load (path_pt ,map_location ="cpu")
            model .tsenc .load_state_dict (state ,strict =True )
            print (f"[Stage-2] Loaded TS encoder from {path_pt }")
        else :
            raise FileNotFoundError (f"TS encoder not found in {src_dir }")
        for p in model .tsenc .parameters ():
            p .requires_grad =False
        peft_cfg =LoraConfig (
        r =args .lora_r ,lora_alpha =args .lora_alpha ,lora_dropout =args .lora_dropout ,
        task_type ="CAUSAL_LM",target_modules =["q_proj","k_proj","v_proj","o_proj"],
        )
        model .qwen =get_peft_model (model .qwen ,peft_cfg )
    t_ts ,a_ts =count_trainable (model .tsenc )
    t_llm ,a_llm =count_trainable (model .qwen )
    print (f"[stage={args .stage }] TS trainable {t_ts }/{a_ts }, LLM trainable {t_llm }/{a_llm }")
    def _ensure_pa (ex ):
        if "prompt"in ex and "answer"in ex :
            return ex
        prompt ,answer =derive_prompt_answer_from_messages (tok ,ex ["messages"])
        ex ["prompt"]=prompt
        ex ["answer"]=answer
        return ex

    train_rows =read_jsonl (args .train_jsonl )
    valid_rows =read_jsonl (args .valid_jsonl )

    if args .use_ts :
        for r in train_rows +valid_rows :
            _ensure_pa (r )
            if MARK_TS3M not in r ["prompt"]or MARK_TSD not in r ["prompt"]:
                raise ValueError ("Prompt is missing TS markers. Make sure you've run the converter and are using the new dataset format.")

    train_ds =Dataset .from_list ([_ensure_pa (r )for r in train_rows ])
    valid_ds =Dataset .from_list ([_ensure_pa (r )for r in valid_rows ])
    collate =Collator ()
    first_batch =[dict (train_ds [0 ])]
    targs =TrainingArguments (
    output_dir =args .out_dir ,
    num_train_epochs =args .epochs ,
    per_device_train_batch_size =args .batch_size ,
    per_device_eval_batch_size =max (1 ,args .batch_size //2 ),
    gradient_accumulation_steps =args .grad_accum ,
    learning_rate =args .lr ,
    weight_decay =args .weight_decay ,
    warmup_ratio =args .warmup_ratio ,
    logging_steps =args .logging_steps ,
    eval_strategy ="epoch",
    save_steps =args .save_steps if args .save_steps >0 else 1_000_000_000 ,
    save_total_limit =2 ,
    bf16 =args .bf16 ,
    fp16 =args .fp16 ,
    lr_scheduler_type ="cosine",
    gradient_checkpointing =args .gradient_checkpointing ,
    remove_unused_columns =False ,
    report_to =[],
    ddp_find_unused_parameters =False ,
    ddp_broadcast_buffers =False ,
    group_by_length =False ,

    )

    log_cb =LogEvalLossCallback ()
    save_cb =SaveEveryNEpochsCallback (
    n =args .save_every_epoch ,
    out_dir =os .path .join (args .out_dir ,"epoch_ckpts"),
    stage =args .stage ,
    )

    first_batch =[dict (train_ds [0 ])]
    collated_batch =collate (first_batch )
    trainer =Trainer (
    model =model ,
    args =targs ,
    train_dataset =train_ds ,
    eval_dataset =valid_ds ,
    tokenizer =tok ,
    data_collator =collate ,
    callbacks =[log_cb ,save_cb ]
    )
    trainer .label_names =["answer"]
    save_cb .trainer =trainer
    trainer .train ()
    if args .stage ==1 :
        final_dir =args .out_dir
        os .makedirs (final_dir ,exist_ok =True )
        ts_cfg ={
        "d_model":4096 ,
        "patch_size_ot":2 ,
        "patch_size_dot":2 ,
        "hidden":128 ,
        "layers":4 ,
        }
        with open (os .path .join (final_dir ,"ts_config.json"),"w",encoding ="utf-8")as f :
            json .dump (ts_cfg ,f ,ensure_ascii =False ,indent =2 )
        from safetensors .torch import save_file
        save_file (model .tsenc .state_dict (),os .path .join (final_dir ,"ts_encoder.safetensors"))
        tok .save_pretrained (final_dir )
    else :
        trainer .save_model (args .out_dir )
        tok .save_pretrained (args .out_dir )
    final_dir =args .out_dir
    ts_cfg =_ts_config_from (model )








if __name__ =="__main__":
    main ()
