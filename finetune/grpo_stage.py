


import os
import argparse
import json
from functools import lru_cache
from collections import defaultdict
from typing import Any ,Dict ,List ,Optional ,Union

import torch
from datasets import Dataset
from transformers import (
AutoTokenizer ,
AutoModelForCausalLM ,
)
from ts_encoder_ex import DualTSMLP
from qwen_with_ts_ex4sft import QwenWithTSEmbed ,load_ts_encoder_from_dir
import trl
from trl .trainer .grpo_trainer import GRPOTrainer ,GRPOConfig ,unwrap_model_for_generation ,is_peft_model
from trl .data_utils import maybe_apply_chat_template
from trl .trainer .utils import pad ,selective_log_softmax ,entropy_from_logits
class TSGRPOTrainer (GRPOTrainer ):
    def _prepare_inputs (self ,generation_batch ):
        if isinstance (generation_batch ,dict )and ("prompt"in generation_batch )and len (generation_batch ["prompt"])>0 :
            rows =[]
            B =len (generation_batch ["prompt"])
            for i in range (B ):
                rows .append ({k :generation_batch [k ][i ]for k in generation_batch .keys ()})
            self ._ts_current_batch =rows
        return super ()._prepare_inputs (generation_batch )

    def _generate_single_turn (self ,prompts :List [Union [str ,list ]],images :Optional [List [Any ]]):

        tokenizer =self .processing_class
        prompts_texts ,images ,forward_kwargs =maybe_apply_chat_template (
        prompts ,
        tokenizer ,
        add_generation_prompt =True ,
        images =images ,
        image_token =self .image_token ,
        image_token_id =self .image_token_id ,
        enable_thinking =False
        )

        prompt2rows =defaultdict (list )
        for d in getattr (self ,"_ts_current_batch",[]):
            prompt2rows [d ["prompt"]].append (d )
        cached =[]
        for p in prompts_texts :
            lst =prompt2rows .get (p ,[])
            cached .append (lst .pop (0 )if lst else {"prompt":p ,"ts3m":None ,"tsdot":None })
        with unwrap_model_for_generation (self .model ,silent =True )as unwrapped_model :

            if is_peft_model (unwrapped_model ):
                base =unwrapped_model .base_model .model
            else :
                base =unwrapped_model
            setattr (base ,"_cached_batch_for_generate",cached )

            outputs =base .generate (
            prompts_texts ,
            generation_config =self .generation_config ,
            )
        prompt_ids_list =outputs .get ("prompt_input_ids",[])
        completion_ids_list =outputs .get ("generated_ids",[])
        sampling_per_token_logps =outputs .get ("per_token_logps",None )
        if sampling_per_token_logps is not None :
            sampling_per_token_logps =[x .tolist ()for x in sampling_per_token_logps ]
        num_items_in_batch =len (completion_ids_list )
        print_prompt_completions_sample (
        self .args ,tokenizer ,prompts_texts ,completion_ids_list ,
        num_items_in_batch =num_items_in_batch ,
        )
        return (
        prompt_ids_list ,
        completion_ids_list ,
        num_items_in_batch ,
        sampling_per_token_logps ,
        forward_kwargs ,
        )

def _trailing_after_final_json_len (t :str )->int :
    if not isinstance (t ,str ):
        return 0
    if "⟪FINAL⟫"in t :
        t =t .split ("⟪FINAL⟫",1 )[1 ]
    s =t .lstrip ()
    i =s .find ("{")
    if i <0 :
        return len (s .strip ())
    s2 ,stack ,end =s [i :],0 ,-1
    for j ,ch in enumerate (s2 ):
        if ch =="{":
            stack +=1
        elif ch =="}":
            stack -=1
            if stack ==0 :
                end =j +1
                break
    if end ==-1 :
        return len (s .strip ())
    trailing =s2 [end :].strip ()
    return len (trailing )

LOG_EVERY_N =10
_seen_cnt =0
def _maybe_log_sample (i ,text ,obj ,ts3m ,tsdot ,trainer_state =None ):
    global _seen_cnt
    _seen_cnt +=1
    if os .environ .get ("RANK","0")!="0":
        return
    if _seen_cnt %LOG_EVERY_N !=0 :
        return
    step =getattr (trainer_state ,"global_step",None )
    keys =[]
    if isinstance (obj ,dict ):
        hyps =obj .get ("hypotheses",[])
        for h in hyps [:5 ]:
            if isinstance (h ,dict )and isinstance (h .get ("key"),str ):
                keys .append (h ["key"])

    snippet =text [:4000 ].replace ("\n"," ")
    if len (text )>4000 :
        snippet +="..."
    print (f"[reward][rank0][#{_seen_cnt }] step={step } local_i={i }")
    print (f"  keys={keys }")
    print (f"  tsdot={tsdot }")
    print (f"  ts3m_len={len (ts3m ['vals'])if isinstance (ts3m ,dict )and 'vals'in ts3m else 'NA'}")
    print (f"  sample_snippet: {snippet }")

def load_jsonl (path :str )->List [Dict [str ,Any ]]:
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

def build_hf_dataset (jsonl_path :str )->Dataset :
    rows =load_jsonl (jsonl_path )

    keep =[]
    for r in rows :
        keep .append ({
        "prompt":r .get ("prompt")or r .get ("messages"),
        "ts3m":r .get ("ts3m"),
        "tsdot":r .get ("tsdot"),
        })
    return Dataset .from_list (keep )

ALLOWED_NAME ={"market","eia","opec","opec_plus","united_states"}
ALLOWED_ACTION ={"cut","price_change","raise","report_release","allocate","approve","cap","close"}
ALLOWED_OBJECT ={"crude_oil","gasoline","price","production","natural_gas","diesel"}
ALLOWED_DIR ={"ambiguous","down","up"}
def _extract_json_after_final (text :str ):
    if not isinstance (text ,str ):
        return None
    if "⟪FINAL⟫"in text :
        text =text .split ("⟪FINAL⟫",1 )[1 ]
    text =text .strip ()

    start =text .find ("{")
    if start ==-1 :
        return None
    stack =0
    end =-1
    for i ,ch in enumerate (text [start :],start ):
        if ch =="{":
            stack +=1
        elif ch =="}":
            stack -=1
            if stack ==0 :
                end =i +1
                break
    if end ==-1 :
        return None
    json_str =text [start :end ]

    try :
        return json .loads (json_str )
    except json .JSONDecodeError :
        return None

def _extract_json_after_final (t :str ):
    if not isinstance (t ,str ):return None
    if "⟪FINAL⟫"in t :t =t .split ("⟪FINAL⟫",1 )[1 ]
    t =t .strip ()
    i =t .find ("{")
    if i <0 :return None
    s ,stack ,end =t [i :],0 ,-1
    for j ,ch in enumerate (s ):
        if ch =="{":stack +=1
        elif ch =="}":
            stack -=1
            if stack ==0 :end =j +1 ;break
    if end ==-1 :return None
    try :
        return json .loads (s [:end ])
    except Exception :
        return None
import json
from typing import List ,Optional ,Dict ,Any

def _keys_from_obj (obj ):
    if not isinstance (obj ,dict ):
        return []
    hyps =obj .get ("hypotheses",[])
    out =[]
    for h in hyps [:5 ]:
        if isinstance (h ,dict )and isinstance (h .get ("key"),str ):
            out .append (h ["key"])
    return out
from functools import lru_cache
def _sample_key (ts3m ,tsdot )->str :
    return json .dumps ({"ts3m":ts3m ,"tsdot":tsdot },sort_keys =True ,separators =(",",":"))

@lru_cache (maxsize =1 )
def _load_answer_index_by_sample (jsonl_path :str ):
    idx ={}
    with open (jsonl_path ,"r",encoding ="utf-8")as f :
        for line in f :
            line =line .strip ()
            if not line :
                continue
            try :
                rec =json .loads (line )
            except Exception :
                continue
            ts3m =rec .get ("ts3m")
            tsdot =rec .get ("tsdot")
            a =rec .get ("answer")
            if a is None :
                continue
            gold =_extract_json_after_final (a )or (json .loads (a )if isinstance (a ,str )and a .strip ().startswith ("{")else None )
            if isinstance (gold ,dict ):
                idx [_sample_key (ts3m ,tsdot )]=gold
    return idx

def _score_by_answer_from_jsonl_by_sample (
cand_obj ,
ts3m ,
tsdot ,
jsonl_path :str ,
weight :float =2.0 ,
dir_bonus :float =0.5 ,
subj_bonus :float =0.5 ,
)->float :
    key =_sample_key (ts3m ,tsdot )
    gold_obj =_load_answer_index_by_sample (jsonl_path ).get (key )
    if not gold_obj :
        return 0.0
    def _keys_from_obj (obj ):
        if not isinstance (obj ,dict ):
            return []
        hyps =obj .get ("hypotheses",[])
        out =[]
        for h in hyps [:5 ]:
            if isinstance (h ,dict )and isinstance (h .get ("key"),str ):
                out .append (h ["key"])
        return out
    def _parse_aaod (k :str ):
        parts =k .split ("|")
        if len (parts )<4 :
            parts +=[""]*(4 -len (parts ))
        return tuple (parts [:4 ])
    cand_keys =_keys_from_obj (cand_obj )
    gold_keys =_keys_from_obj (gold_obj )
    if not cand_keys or not gold_keys :
        return 0.0
    cand_aaod_list =[_parse_aaod (k )for k in cand_keys ]
    gold_aaod_list =[_parse_aaod (k )for k in gold_keys ]
    cand_set =set (cand_aaod_list )
    gold_set =set (gold_aaod_list )
    def _proj_name (t ):
        return (t [0 ],)

    def _proj_na (t ):
        return (t [0 ],t [1 ])

    def _proj_nao (t ):
        return (t [0 ],t [1 ],t [2 ])
    cand_name ={_proj_name (t )for t in cand_set }
    gold_name ={_proj_name (t )for t in gold_set }
    inter_name =len (cand_name &gold_name )
    name_rec =inter_name /max (1 ,len (gold_name ))
    cand_na ={_proj_na (t )for t in cand_set }
    gold_na ={_proj_na (t )for t in gold_set }
    inter_na =len (cand_na &gold_na )
    na_rec =inter_na /max (1 ,len (gold_na ))
    cand_nao ={_proj_nao (t )for t in cand_set }
    gold_nao ={_proj_nao (t )for t in gold_set }
    inter_nao =len (cand_nao &gold_nao )
    nao_rec =inter_nao /max (1 ,len (gold_nao ))
    inter_naod =len (cand_set &gold_set )
    naod_rec =inter_naod /max (1 ,len (gold_set ))
    w_name =0.5
    w_na =0.5
    w_nao =1.0
    w_naod =1.0 +dir_bonus
    score =(
    w_name *name_rec +
    w_na *na_rec +
    w_nao *nao_rec +
    w_naod *naod_rec
    )
    print (
    f"[reward-hier] name_rec={name_rec :.3f}  na_rec={na_rec :.3f}  "
    f"nao_rec={nao_rec :.3f}  naod_rec={naod_rec :.3f}  score={score :.3f}"
    )
    return float (weight *score )

def _triplet_enum_penalty (hyps ,per_dup :float =0.1 )->float :

    seen =set ()
    dup =0
    for h in hyps [:5 ]:
        if not isinstance (h ,dict ):
            continue
        key =h .get ("key")
        if not (isinstance (key ,str )and key .count ("|")==3 ):
            continue
        a ,b ,c ,d =key .split ("|")
        trip =(a ,b ,c )
        if trip in seen :
            dup +=1
        else :
            seen .add (trip )
    return -per_dup *dup


def simple_reward_func (completions :List [str ],trainer_state =None ,**cols )->List [float ]:
    rewards :List [float ]=[]
    ts3ms =cols .get ("ts3m",[None ]*len (completions ))
    tsdots =cols .get ("tsdot",[None ]*len (completions ))
    prompts =cols .get ("prompts",[""]*len (completions ))
    for i ,text in enumerate (completions ):
        base =0.0
        obj =None
        obj =_extract_json_after_final (text )
        obj =obj or {}
        hyps =obj .get ("hypotheses",[])

        _maybe_log_sample (i ,text ,obj ,ts3ms [i ]if i <len (ts3ms )else None ,
        tsdots [i ]if i <len (tsdots )else None ,trainer_state )
        if isinstance (hyps ,list )and hyps :
            base +=0.00

            ok_vocab =0
            for h in hyps [:5 ]:
                key =h .get ("key")if isinstance (h ,dict )else None
                if isinstance (key ,str )and key .count ("|")==3 :
                    a ,b ,c ,d =key .split ("|")
                    if (a in ALLOWED_NAME )and (b in ALLOWED_ACTION )and (c in ALLOWED_OBJECT )and (d in ALLOWED_DIR ):
                        ok_vocab +=0.00
            base =base +ok_vocab

            keys_seen =[]
            for h in hyps [:5 ]:
                k =h .get ("key")if isinstance (h ,dict )else None
                if isinstance (k ,str ):
                    keys_seen .append (k .strip ())

            if len (keys_seen )!=len (set (keys_seen )):
                base -=0.0
            base +=_score_by_answer_from_jsonl_by_sample (
            obj ,
            ts3ms [i ],
            tsdots [i ],
            jsonl_path ="num2event/dataset4sft_RB_robust_scaled/wo_huji_v1.jsonl",# Enter the training set path here.
            weight =1.0
            )
            base +=_triplet_enum_penalty (hyps ,per_dup =0.1 )

        if _trailing_after_final_json_len (text )>0 :
            base -=0.05


        rewards .append (float (base ))
    return rewards

def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--base_model",type =str ,default ="Num2Text/QWen3_8B")
    ap .add_argument ("--stage1_dir",type =str ,default ="TS2EVENTS_v2/Qwen_weight_en/stage1_syn_10epochs/epoch_ckpts/epoch_10")
    ap .add_argument ("--train_jsonl",type =str ,default ="TS2EVENTS_v2/dataset4sft_en/rl_dataset_demo.jsonl")
    ap .add_argument ("--out_dir",type =str ,default ="./out_grpo_ts_demo")
    ap .add_argument ("--bf16",action ="store_true",default =" ")
    ap .add_argument ("--sft_lora_dir",type =str ,default =None )
    ap .add_argument ("--use_lora",action ="store_true",default =" ")
    args =ap .parse_args ()
    os .makedirs (args .out_dir ,exist_ok =True )
    tok =AutoTokenizer .from_pretrained (args .base_model ,use_fast =True )
    if tok .pad_token is None :
        tok .pad_token =tok .eos_token
    qwen =AutoModelForCausalLM .from_pretrained (
    args .base_model ,
    torch_dtype =torch .bfloat16 if args .bf16 else None ,
    low_cpu_mem_usage =True ,
    trust_remote_code =True ,
    )
    tsenc =DualTSMLP (d_model =qwen .config .hidden_size ,hidden =128 ,layers =4 ,patch_size_ot =1 ,patch_size_dot =1 ,concat_posidx =False )
    model =QwenWithTSEmbed (qwen =qwen ,ts_encoder =tsenc ,tokenizer =tok )
    load_ts_encoder_from_dir (model ,args .stage1_dir ,strict =False )
    train_ds =build_hf_dataset (args .train_jsonl )
    from peft import LoraConfig ,get_peft_model ,PeftModel
    if args .sft_lora_dir :
        print (f" SFT LoRA：{args .sft_lora_dir }")
        model .qwen =PeftModel .from_pretrained (model .qwen ,args .sft_lora_dir ,is_trainable =True )
        model .qwen .print_trainable_parameters ()
    for p in model .tsenc .parameters ():
        p .requires_grad =False
    grpo_args =GRPOConfig (
    output_dir =args .out_dir ,
    num_iterations =1 ,
    steps_per_generation =8 ,
    num_train_epochs =1 ,
    per_device_train_batch_size =1 ,
    num_generations =8 ,
    max_completion_length =1024 ,
    temperature =0.7 ,
    top_p =0.95 ,
    use_vllm =False ,
    bf16 =args .bf16 ,
    logging_steps =5 ,
    save_steps =3000 ,
    learning_rate =1e-5 ,
    gradient_accumulation_steps =1 ,
    remove_unused_columns =False ,
    mask_truncated_completions =False ,
    report_to =[],
    )
    trainer =TSGRPOTrainer (
    model =model ,
    reward_funcs =simple_reward_func ,
    args =grpo_args ,
    train_dataset =train_ds ,
    processing_class =tok ,
    )
    trainer .train ()
    model .qwen .save_pretrained (args .out_dir )
    tok .save_pretrained (args .out_dir )


if __name__ =="__main__":
    main ()
