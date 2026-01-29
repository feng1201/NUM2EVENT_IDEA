from __future__ import annotations
from typing import List ,Dict ,Any ,Optional
import json
import re
import sys
import torch
import os
from transformers import (
AutoTokenizer ,
AutoModelForCausalLM ,
GenerationConfig ,
)
PLACEHOLDER_KEY ="NAME|ACTION|OBJECT|DIRECTION"

def _strip_code_fence (s :str )->str :
    s =s .strip ()

    if s .startswith ("```"):
        s =re .sub (r"^```(?:json)?\s*","",s ,flags =re .I )
        s =re .sub (r"\s*```$","",s )
    return s .strip ()

def _balanced_json_from_text (s :str ):

    start =s .find ("{")
    if start <0 :
        return None
    depth =0
    for i in range (start ,len (s )):
        if s [i ]=="{":
            depth +=1
        elif s [i ]=="}":
            depth -=1
            if depth ==0 :
                try :
                    return json .loads (s [start :i +1 ])
                except Exception :
                    return None
    return None

def extract_hypotheses_from_text (text :str ,k_max :int =5 ):

    t =_strip_code_fence (text )

    obj =None

    try :
        obj =json .loads (t )
    except Exception :
        pass


    if obj is None :
        obj =_balanced_json_from_text (t )

    hyps =[]
    if isinstance (obj ,dict )and isinstance (obj .get ("hypotheses"),list ):
        for h in obj ["hypotheses"][:k_max ]:
            key =str (h .get ("key","")).strip ().lower ()
            if key and key !=PLACEHOLDER_KEY .lower ():
                hyps .append ({"key":key ,"conf":float (h .get ("conf",0.5 ))})


    if not hyps :
        keys =re .findall (r"\b([a-z0-9_]+)\|([a-z0-9_]+)\|([a-z0-9_]+)\|([a-z0-9_]+)\b",t )
        for k in keys [:k_max ]:
            key ="|".join (k )
            if key .lower ()!=PLACEHOLDER_KEY .lower ():
                hyps .append ({"key":key ,"conf":0.3 })

    return hyps
try :
    from peft import PeftModel
    _HAS_PEFT =True
except Exception :
    _HAS_PEFT =False

from aaodt_schema import AAOD
def build_messages_three_months (
dates_3m :List [str ],ot_3m :List [float ],
dates_curr :List [str ],dot_curr :List [float ],
k_max :int =5 ,
as_markers :bool =False ,
)->List [Dict [str ,str ]]:

    tab_ot ="\n".join ([f"{d }: P={p :.3f}"for d ,p in zip (dates_3m ,ot_3m )])
    tab_dot ="\n".join ([f"{d }: dP={dp :+.3f}"for d ,dp in zip (dates_curr ,dot_curr )])
    input_block =(
    "- Last 3 months (weekly OT): <ts3m><ts3m/>\n"
    "- Current month only (weekly dOT): <tsdot><tsdot/>\n"
    "The weekly OT and weekly dOT are embedding tokens from a learned time-series encoder, not readable numbers."
    )
    user =(
    "You are given weekly U.S. gasoline prices (OT):"
    "- Context window (last 3 months, weekly OT):"
    "<ts3m><ts3m/>"
    "- Current month only (weekly dOT):"
    "<tsdot><tsdot/>"

    "Task:"
    "1. Carefully analyze the time-series trends and reason step-by-step about plausible real-world causes."
    "2. Write a *reasoning paragraph* summarizing your interpretation of the data trend and potential driving factors."
    "   - This paragraph should sound like an analytical narrative (e.g., “The observed trend in the data indicates…”)."
    "   - It should include your reasoning path that connects the observed pattern to possible events."
    "3. Then, hypothesize up to 5 **REAL-TIME events** that occurred WITHIN THE CURRENT MONTH ONLY."

    "Output format:"
    "- First, write your reasoning paragraph."
    "- After that, write the final structured results after ⟪FINAL⟫ in strict JSON format:"

    "⟪FINAL⟫"
    "{"
    "  \"hypotheses\": ["
    "    {"
    "      \"key\": \"NAME|ACTION|OBJECT|DIRECTION\","
    "    }"
    "  ]"
    "}"

    "Allowed token sets:"
    "('NAME ∈ {market, eia, opec, opec_plus, united_states}',"
    "'ACTION ∈ {report_release, price_change, production_cut, forecast_lower, forecast_raise, production_raise, extend_cut, sanction_impose, purchase, release, export_ban_lift}',"
    "'OBJECT ∈ {inventory, crude_oil, gasoline, price, production, natural_gas, inventory, diesel, export, jet_fuel, rig_count, brent, forecast}',"
    "'DIRECTION ∈ {ambiguous, up, down}')"

    "Rules:"
    "- Do NOT fabricate tokens outside these sets."
    "- Use \"|\" (vertical bar), **not underscores** to separate the fields."
    "- Each \"key\" must strictly follow this pipe-separated pattern: NAME|ACTION|OBJECT|DIRECTION ."
    "- The reasoning paragraph should be fluent and analytic, not a list of events."
    "- If the same subject has multiple events, output each event separately; do not merge them."
    ).replace ("{K}",str (k_max ))
    return [
    {"role":"user","content":user }
    ]


def _safe_json_extract (text :str )->Dict [str ,Any ]:
    s =text .strip ()
    try :
        obj =json .loads (s )
        if isinstance (obj ,dict ):
            return obj
    except Exception :
        pass
    m =re .search (r"```json\s*(\{[\s\S]*?\})\s*```",s ,flags =re .I )
    if m :
        try :
            return json .loads (m .group (1 ))
        except Exception :
            pass
    start =s .find ("{")
    if start !=-1 :
        depth =0
        for i in range (start ,len (s )):
            if s [i ]=="{":
                depth +=1
            elif s [i ]=="}":
                depth -=1
                if depth ==0 :
                    chunk =s [start :i +1 ]
                    try :
                        return json .loads (chunk )
                    except Exception :
                        break
    keys =re .findall (r"([A-Za-z0-9_]+)\|([A-Za-z0-9_]+)\|([A-Za-z0-9_]+)\|([A-Za-z0-9_]+)",s )
    if keys :
        hyps =[{"key":"|".join (k ),"conf":0.3 }for k in keys [:5 ]]
        return {"hypotheses":hyps ,"summary":""}
    return {"hypotheses":[],"summary":""}

class QwenReasoner :
    def __init__ (
    self ,
    base :str ,
    adapter :Optional [str ]=None ,
    torch_dtype =torch .bfloat16 ,
    device_map :str ="auto",
    max_new_tokens :int =512 ,
    debug_print_limit :int =3 ,
    use_tsenc :bool =True ,
    ts_ckpt :Optional [str ]=None ,
    ):

        self .tok =AutoTokenizer .from_pretrained (base ,use_fast =True ,trust_remote_code =True )
        base_model =AutoModelForCausalLM .from_pretrained (
        base ,torch_dtype =torch_dtype ,device_map =device_map ,trust_remote_code =True
        )
        self .use_tsenc =bool (use_tsenc )
        self ._has_build_prompt =False
        if self .use_tsenc :
            try :
                self .model =base_model
                self ._has_build_prompt =False
                ROOT =os .path .abspath (os .path .join (os .path .dirname (__file__ ),"num2event"))
                sys .path .insert (0 ,ROOT )
                from newfinetune .qwen_with_ts_ex4sft import QwenWithTSEmbed
                from newfinetune .ts_encoder_ex import DualTSMLP
                d_model =base_model .config .hidden_size
                ts_default =dict (patch_size_ot =2 ,patch_size_dot =2 ,hidden =128 ,layers =4 ,concat_posidx =False )

                cfg =None
                if ts_ckpt and os .path .isdir (ts_ckpt ):
                    cfg_path =os .path .join (ts_ckpt ,"ts_encoder_config.json")
                    if os .path .exists (cfg_path ):
                        with open (cfg_path ,"r",encoding ="utf-8")as f :
                            cfg =json .load (f )
                ts_cfg =dict (
                d_model =d_model ,
                patch_size_ot =(cfg .get ("patch_size_ot")if cfg else ts_default ["patch_size_ot"]),
                patch_size_dot =(cfg .get ("patch_size_dot")if cfg else ts_default ["patch_size_dot"]),
                hidden =(cfg .get ("hidden")if cfg else ts_default ["hidden"]),
                layers =(cfg .get ("layers")if cfg else ts_default ["layers"]),
                concat_posidx =ts_default ["concat_posidx"],
                )
                ts_encoder =DualTSMLP (**ts_cfg )
                self .model =QwenWithTSEmbed (base_model ,ts_encoder ,self .tok )
                if ts_ckpt :
                    if os .path .isdir (ts_ckpt ):

                        st_path =os .path .join (ts_ckpt ,"ts_encoder.safetensors")
                        pt_path =os .path .join (ts_ckpt ,"ts_encoder.pt")
                        if os .path .exists (st_path ):
                            from safetensors .torch import load_file as _load_file
                            state =_load_file (st_path ,device ="cpu")
                            self .model .tsenc .load_state_dict (state ,strict =False )

                        elif os .path .exists (pt_path ):
                            state =torch .load (pt_path ,map_location ="cpu")
                            self .model .tsenc .load_state_dict (state ,strict =False )

                        else :
                            print (f"[TS][WARN] no ts weights found in dir")
                    else :

                        if ts_ckpt .endswith (".safetensors"):
                            from safetensors .torch import load_file as _load_file
                            state =_load_file (ts_ckpt ,device ="cpu")
                        else :
                            state =torch .load (ts_ckpt ,map_location ="cpu")
                        self .model .tsenc .load_state_dict (state ,strict =False )

                self ._has_build_prompt =hasattr (self .model ,"_build_prompt_inputs")
                if not self ._has_build_prompt :
                    print ("[TS][WARN] error in _build_prompt_inputs")
            except Exception as e :
                print (f"[WARN] ts encoder error")
                self .model =base_model
                self .use_tsenc =False
        else :
            self .model =base_model
        self .max_new_tokens =max_new_tokens
        self .debug_print_limit =max (0 ,int (debug_print_limit ))

        gen_cfg =GenerationConfig .from_model_config (self .model .config )

        eos_id =self .tok .eos_token_id
        if eos_id is None and hasattr (self .tok ,"convert_tokens_to_ids"):
            for tok in ["<|im_end|>","</s>","<|endoftext|>"]:
                tid =self .tok .convert_tokens_to_ids (tok )
                if isinstance (tid ,int )and tid >0 :
                    eos_id =tid
                    break
        if eos_id is not None :
            gen_cfg .eos_token_id =eos_id

        if self .tok .pad_token_id is None and eos_id is not None :
            self .tok .pad_token_id =eos_id
        gen_cfg .pad_token_id =self .tok .pad_token_id

        gen_cfg .do_sample =False
        gen_cfg .temperature =0.0
        gen_cfg .top_p =0.95
        gen_cfg .top_k =50
        gen_cfg .max_new_tokens =self .max_new_tokens

        self .model .generation_config =gen_cfg

    def _apply_sampling (self ,temperature :float ):
        self .model .generation_config .do_sample =True
        self .model .generation_config .temperature =float (temperature )
        self .model .generation_config .top_p =0.5









    def infer_keys_three_months (
    self ,
    dates_3m :List [str ],ot_3m :List [float ],
    dates_curr :List [str ],dot_curr :List [float ],
    k_max :int =5 ,
    temperature :float =0.0 ,
    )->List [AAOD ]:




        messages =build_messages_three_months (
        dates_3m ,ot_3m ,dates_curr ,dot_curr ,
        k_max =k_max ,as_markers =self .use_tsenc
        )



        prompt =self .tok .apply_chat_template (messages ,tokenize =False ,add_generation_prompt =True ,enable_thinking =False )





        self ._apply_sampling (temperature )
        ts3m_pack ={"vals":ot_3m ,"mask":[1 ]*len (ot_3m )}
        tsdot_pack ={"vals":dot_curr ,"mask":[1 ]*len (dot_curr )}
        with torch .no_grad ():
            prompt_embeds ,prompt_attn =self .model ._build_prompt_inputs (
            [prompt ],[ts3m_pack ],[tsdot_pack ]
            )
            out =self .model .qwen .generate (
            inputs_embeds =prompt_embeds ,
            attention_mask =prompt_attn ,
            max_new_tokens =self .max_new_tokens ,
            )
        gen_ids =out [0 ]


        text =self .tok .decode (gen_ids ,skip_special_tokens =True )

        obj =_safe_json_extract (text )
        if self .debug_print_limit >0 :
            print ("\n===== RAW GENERATION (decoded) =====",flush =True )
            print (text ,flush =True )
            print ("===== END RAW GENERATION =====",flush =True )







        preds :List [AAOD ]=[]
        hyps =obj .get ("hypotheses",[])

        for i ,h in enumerate (hyps [:k_max ]):
            try :

                key =h .get ("key","")
            except Exception :

                if isinstance (h ,str )and h .strip ():
                    key =h .strip ()
                else :

                    key ="name|action|object|direction"





            if not key :
                continue
            try :
                preds .append (AAOD .parse_key (key ))
            except Exception :

                continue

        seen =set ()
        preds_unique =[]
        for a in preds :
            kk =a .key ().lower ().strip ()
            if not kk or kk =="name|action|object|direction":
                continue
            if kk not in seen :
                seen .add (kk )
                preds_unique .append (a )
            if len (preds_unique )>=k_max :
                break

        return preds_unique


