
import json
import re
import argparse
import datetime as dt
import atexit
import os
import sys
from typing import List
import pandas as pd
from aaodt_schema import AAOD
from events_gold import (
EventRow ,
build_gold_rte_for_window ,
load_events_jsonl as load_events_jsonl_strict ,
)
try :
    from events_gold import load_events_jsonl_safe as load_events_jsonl
except Exception :
    load_events_jsonl =load_events_jsonl_strict

from eval_metrics import evaluate_rte
from llm_infer_api import QwenReasoner
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

def load_jsonl (path :str )->List [dict ]:
    items :List [dict ]=[]
    with open (path ,"r",encoding ="utf-8")as f :
        for line in f :
            line =line .strip ()
            if not line :
                continue
            items .append (json .loads (line ))
    return items
def parse_gold_aaod_from_answer (answer_field )->List [AAOD ]:
    if isinstance (answer_field ,dict ):
        ans_obj =answer_field
    elif isinstance (answer_field ,str ):
        text =answer_field
        marker ="⟪FINAL⟫"
        idx =text .find (marker )
        if idx !=-1 :
            text =text [idx +len (marker ):]
        m =re .search (r"(\{.*\})\s*$",text ,flags =re .DOTALL )
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
        print (f"[WARN] answer ：{type (answer_field )}，。")
        return []

    hyps =ans_obj .get ("hypotheses")or []
    aaods :List [AAOD ]=[]
    print ("Labels:")
    for h in hyps :
        key =h .get ("key")if isinstance (h ,dict )else str (h )
        key =(key or "").strip ()
        if not key :
            continue
        aaods .append (AAOD .parse_key (key ))
    return aaods

def read_energy_csv (path :str )->pd .DataFrame :
    try :
        return pd .read_csv (path ,engine ="python")
    except Exception :
        pass
    for enc in ["utf-8","utf-8-sig","latin1","cp1252","mac_roman"]:
        try :
            return pd .read_csv (path ,engine ="python",encoding =enc )
        except Exception :
            continue
    raise RuntimeError (f"Failed to read CSV with multiple encodings: {path }")

def end_of_month (d :dt .date )->dt .date :
    if d .month ==12 :
        return dt .date (d .year ,12 ,31 )
    return dt .date (d .year ,d .month +1 ,1 )-dt .timedelta (days =1 )

def month_range (start :dt .date ,end :dt .date )->List [dt .date ]:
    cur =dt .date (start .year ,start .month ,1 )
    res =[]
    while cur <=end :
        res .append (cur )
        if cur .month ==12 :
            cur =dt .date (cur .year +1 ,1 ,1 )
        else :
            cur =dt .date (cur .year ,cur .month +1 ,1 )
    return res

def detect_date_col (df :pd .DataFrame )->str :

    for c in df .columns :
        if str (c ).lower ()in {"date","time","ds"}:
            return c
    return df .columns [0 ]
def main ():
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--events",type =str ,default ="events_canon_dedup.jsonl",
    help =" JSONL ")
    parser .add_argument ("--energy",type =str ,default ="energy.csv",
    help ="OT  CSV ")
    parser .add_argument ("--base",type =str ,
    default ="Num2Text/weight/base_qwen1p5",
    help ="Qwen1.5B （）")

    parser .add_argument ("--data",type =str ,default ="",
    help =" prompt/answer/ts3m/tsdot  JSONL； --events/--energy")

    parser .add_argument ("--adapter",type =str ,default ="",
    help =" LoRA ；= base")
    parser .add_argument ("--skip-llm",action ="store_true",
    help =" LLM ，（）")
    parser .add_argument ("--k-pred",type =int ,default =5 ,
    help ="（≤5）")
    parser .add_argument ("--k-gold",type =int ,default =20 ,
    help ="（≤5）")
    parser .add_argument ("--out",type =str ,default ="rte_eval_2023_2024.csv",
    help =" CSV ")
    parser .add_argument (
    "--log_dir",
    type =str ,
    default ="",
    help ="log （ /path/to/xxx.log  .txt）；",
    )
    args =parser .parse_args ()
    _setup_tee_logging (args .log_dir )

    if args .data and args .data .strip ():

        items =load_jsonl (args .data )
        if not items :
            raise RuntimeError ("No items loaded from JSONL. ")
        reasoner =None
        if not args .skip_llm :
            adapter_path =args .adapter if args .adapter and args .adapter .strip ()else None
            reasoner =QwenReasoner (base =args .base ,adapter =adapter_path )
        rows_out =[]
        recall_all =0
        pre_all =0
        for idx ,it in enumerate (items ):
            ts3m_vals =(it .get ("ts3m")or {}).get ("vals")or []
            tsdot_vals =(it .get ("tsdot")or {}).get ("vals")or []

            dates_3m =[f"t{i }"for i in range (len (ts3m_vals ))]
            dates_curr =[f"c{i }"for i in range (len (tsdot_vals ))]
            gold_primary_aaodt =parse_gold_aaod_from_answer (it .get ("answer"))
            gold_all_aaodt =gold_primary_aaodt
            pred_keys =[]
            if reasoner is not None :
                pred_keys =reasoner .infer_keys_three_months (
                dates_3m =dates_3m ,ot_3m =ts3m_vals ,
                dates_curr =dates_curr ,dot_curr =tsdot_vals ,
                k_max =args .k_pred ,
                )

            metrics =evaluate_rte (
            preds =pred_keys ,
            gold_primary =gold_primary_aaodt ,
            gold_all =gold_all_aaodt ,
            k_pred_max =args .k_pred ,
            thr =0.6 ,
            )
            recall_all =recall_all +metrics ['recall_primary@5']
            recall_all_average =recall_all /(idx +1 )
            pre_all =pre_all +metrics ['precision_all']
            pre_all_average =pre_all /(idx +1 )
            print ("recall_all:",recall_all_average ,"precision_all:",pre_all_average )
            rows_out .append ({
            "idx":idx ,
            "gold_count":len (gold_primary_aaodt ),
            "pred_count":len (pred_keys ),
            "ground_truth_events":_aaod_events_to_str (gold_primary_aaodt ),
            "predicted_events":_aaod_events_to_str (pred_keys ),
            **metrics ,
            })

        out_df =pd .DataFrame (rows_out )
        out_df .to_csv (args .out ,index =False )


        return














if __name__ =="__main__":
    main ()

