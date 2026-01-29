import argparse
import json
import re
import atexit
import os
import sys
from typing import List
import pandas as pd

from aaodt_schema import AAOD
from eval_metrics import evaluate_rte
from llm_infer_ts import QwenReasoner
def load_jsonl (path :str )->List [dict ]:
    items :List [dict ]=[]
    with open (path ,"r",encoding ="utf-8")as f :
        for line in f :
            line =line .strip ()
            if not line :
                continue
            items .append (json .loads (line ))
    return items
import re
import json
from typing import List

def parse_gold_aaod_from_answer (answer_field )->List [AAOD ]:
    if isinstance (answer_field ,dict ):
        ans_obj =answer_field
    elif isinstance (answer_field ,str ):
        text =answer_field

        marker ='⟪FINAL⟫'
        idx =text .find (marker )
        if idx !=-1 :
            text =text [idx +len (marker ):]

        m =re .search (r'(\{.*\})\s*$',text ,flags =re .DOTALL )
        if not m :
            print ("No JSON found in answer:",repr (answer_field ))
            return []

        json_str =m .group (1 )
        try :
            ans_obj =json .loads (json_str )
        except Exception as e :
            print ("JSON parse error:",e ,"raw:",json_str )
            return []
    else :
        print (f"[WARN] answer error：{type (answer_field )}")
        return []

    hyps =ans_obj .get ("hypotheses")or []
    aaods :List [AAOD ]=[]
    print ("Labels:")
    for h in hyps :
        key =h .get ("key")if isinstance (h ,dict )else str (h )
        key =(key or "").strip ()
        if not key :
            continue
        try :
            aaods .append (AAOD .parse_key (key ))
        except Exception as e :
            print (f"[WARN] decoding gold key error：{key !r }, err={e }")
    return aaods

def _aaod_events_to_str (aaods :List [AAOD ])->str :
    events =[
    [a .name or "UNK",a .action or "UNK",a .obj or "UNK",a .direction or "UNK"]
    for a in (aaods or [])
    ]
    return repr (events )


class _TeeIO :
    def __init__ (self ,*streams ):
        self ._streams =[s for s in streams if s is not None ]
    def write (self ,data ):
        for s in self ._streams :
            s .write (data )
    def flush (self ):
        for s in self ._streams :
            try :
                s .flush ()
            except Exception :
                pass

    @property
    def encoding (self ):
        for s in self ._streams :
            enc =getattr (s ,"encoding",None )
            if enc :
                return enc
        return "utf-8"


def _setup_tee_logging (log_path :str ):

    if not log_path or not str (log_path ).strip ():
        return
    log_path =os .path .abspath (str (log_path ).strip ())
    log_dir =os .path .dirname (log_path )
    if log_dir :
        os .makedirs (log_dir ,exist_ok =True )

    f =open (log_path ,"a",encoding ="utf-8")
    orig_out ,orig_err =sys .stdout ,sys .stderr
    sys .stdout =_TeeIO (orig_out ,f )
    sys .stderr =_TeeIO (orig_err ,f )

    def _cleanup ():
        try :
            sys .stdout =orig_out
            sys .stderr =orig_err
        finally :
            try :
                f .flush ()
            except Exception :
                pass
            f .close ()

    atexit .register (_cleanup )

def main ():
    parser =argparse .ArgumentParser ()
    parser .add_argument (
    "--data",
    type =str ,
    required =True ,
    help =" prompt/answer/ts3m/tsdot  JSONL ",
    )
    parser .add_argument (
    "--base",
    type =str ,
    default ="Num2Text/QWen3_8B",
    help ="Qwen （）",
    )
    parser .add_argument (
    "--adapter",
    type =str ,
    default ="TS2EVENTS_v2/out_grpo_ts_rec",
    help =" LoRA ；= base",
    )
    parser .add_argument (
    "--skip-llm",
    action ="store_true",
    help =" LLM ，（）",
    )
    parser .add_argument (
    "--k-pred",
    type =int ,
    default =5 ,
    help ="（≤5）",
    )
    parser .add_argument (
    "--out",
    type =str ,
    default ="TS2EVENTS_v2/eval_result_rl/rte_eval_from_jsonl.csv",
    help =" CSV ",
    )
    parser .add_argument (
    "--use-tsenc",
    action ="store_true",
    default =True ,
    help ="（ ts ）",
    )
    parser .add_argument (
    "--ts-ckpt",
    type =str ,
    default ="TS2EVENTS_v2/Qwen_weight_en_RL/stage1_true_ts20epochs_v2/epoch_ckpts/epoch_20",
    help ="Stage-1  ts_encoder /（）",
    )
    parser .add_argument (
    "--log_dir",
    type =str ,
    default ="",
    help ="log （ /path/to/xxx.log  .txt）；",
    )
    args =parser .parse_args ()

    _setup_tee_logging (args .log_dir )

    items =load_jsonl (args .data )
    if not items :
        raise RuntimeError ("No items loaded from JSONL.  --data /。")
    print (f"[INFO] Loaded {len (items )} items from JSONL.")

    reasoner =None
    if not args .skip_llm :
        print ("[INFO] Initializing Qwen reasoner (base-only or base+TSenc)...")
        adapter_path =args .adapter if args .adapter and args .adapter .strip ()else None
        reasoner =QwenReasoner (
        base =args .base ,
        adapter =adapter_path ,
        use_tsenc =args .use_tsenc ,
        ts_ckpt =(args .ts_ckpt if args .ts_ckpt and args .ts_ckpt .strip ()else None ),
        )
        print ("[INFO] QwenReasoner ready!")
    rows_out =[]
    recall_all =0
    pre_all =0
    for idx ,it in enumerate (items ):

        ts3m =it .get ("ts3m")or {}
        tsdot =it .get ("tsdot")or {}
        ts3m_vals =ts3m .get ("vals")or []
        tsdot_vals =tsdot .get ("vals")or []
        dates_3m =[f"t{i }"for i in range (len (ts3m_vals ))]
        dates_curr =[f"c{i }"for i in range (len (tsdot_vals ))]
        gold_primary_aaodt :List [AAOD ]=parse_gold_aaod_from_answer (it .get ("answer"))
        gold_all_aaodt =gold_primary_aaodt

        if not gold_primary_aaodt :
            print (f"[WARN]  {idx }  gold hypotheses。")
        pred_keys :List [AAOD ]=[]
        if reasoner is not None :
            pred_keys =reasoner .infer_keys_three_months (
            dates_3m =dates_3m ,
            ot_3m =ts3m_vals ,
            dates_curr =dates_curr ,
            dot_curr =tsdot_vals ,
            k_max =args .k_pred ,
            )
            if pred_keys :
                print (f"[DEBUG][{idx }] got {len (pred_keys )} preds. First key: {pred_keys [0 ].key ()}")
            else :
                print (f"[DEBUG][{idx }] got 0 preds from LLM (JSON may be empty or unparsable).")
        else :

            pred_keys =[]
        metrics =evaluate_rte (
        preds =pred_keys ,
        gold_primary =gold_primary_aaodt ,
        gold_all =gold_all_aaodt ,
        k_pred_max =args .k_pred ,
        thr =0.6 ,
        )
        rows_out .append (
        {
        "idx":idx ,
        "gold_count":len (gold_primary_aaodt ),
        "pred_count":len (pred_keys ),
        "ground_truth_events":_aaod_events_to_str (gold_primary_aaodt ),
        "predicted_events":_aaod_events_to_str (pred_keys ),
        **metrics ,
        }
        )
        recall_all =recall_all +metrics ['recall_primary@5']
        recall_all_average =recall_all /(idx +1 )

        pre_all =pre_all +metrics ['precision_all']
        pre_all_average =pre_all /(idx +1 )
        print ("precision_all:",pre_all_average ,"recall_all:",recall_all_average )
    if not rows_out :
        raise RuntimeError ("rows_out error.")
    out_df =pd .DataFrame (rows_out )
    out_df .to_csv (args .out ,index =False )

if __name__ =="__main__":
    main ()

