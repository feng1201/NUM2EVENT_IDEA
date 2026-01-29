


from __future__ import annotations 
import os 
import re 
import json 
import orjson 
import time 
import math 
import argparse 
import asyncio 
import dataclasses 
from dataclasses import dataclass 
from typing import Any ,Dict ,List ,Optional ,Tuple 
from tenacity import retry ,wait_exponential ,stop_after_attempt ,retry_if_exception_type 
from jsonschema import validate ,Draft202012Validator 
import pandas as pd 
from tqdm import tqdm 
from dateutil import parser as dtparser 
from openai import AsyncOpenAI 

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





DEFAULT_MODEL ="gemini-2.0-flash-lite"


MAX_INPUT_CHARS =4000 
DEFAULT_CONCURRENCY =8 
REQUEST_TIMEOUT =60 
TEMPERATURE =0.0 

OUTPUT_RAW_JSONL ="eventsdata_en/events_raw.jsonl"
OUTPUT_EVENTS_JSONL ="eventsdata_en/events_flat.jsonl"
OUTPUT_DEDUP_JSONL ="eventsdata_en/events_dedup_filtered.jsonl"
OUTPUT_SUGG_JSONL ="eventsdata_en/vocab_suggestions.jsonl"
OUTPUT_COUNTS_CSV ="eventsdata_en/aaod_counts.csv"





ALLOWED_VALUES :Dict [str ,Any ]={
"names":[

"united_states","saudi_arabia","russia","iran","iraq","united_arab_emirates",
"china","india","european_union",

"opec","opec_plus","eia","iea","api",

"saudi_aramco","exxonmobil","chevron","shell","bp","totalenergies"
"market"
],
"actions":[

"price_change",

"production_raise","production_cut","extend_cut",

"sanction_impose","sanction_lift","embargo_impose","embargo_lift",
"quota_set","quota_raise","quota_cut",
"tax_impose","tax_reduce","subsidy_grant","subsidy_remove",
"price_cap_set","price_cap_adjust","export_ban","export_ban_lift","tariff_impose","tariff_reduce",

"shutdown","restart","maintenance_start","maintenance_end","force_majeure","reopen","blockade",
"outage","strike",

"release","purchase","refill","allocate","ration","auction","tender","waive",

"report_release","forecast_raise","forecast_lower","guidance_raise","guidance_lower",

"approve","revoke","deregulate","mandate","open","close","scale_down","ramp_up","other"
],
"objects":[

"crude_oil","natural_gas","lng","lpg","brent","wti","urals","dubai",
"gasoline","diesel","jet_fuel","fuel_oil","naphtha","gasoil","kerosene","electricity",

"price","production","demand","inventory","export","import",
"rig_count","capacity","throughput","spread","crack","futures","spr",

"refinery","pipeline","terminal",

"forecast","guidance","policy_rule","interest_rate","macro_index"
],
"directions":["up","down","flat","ambiguous"],
}






EXAMPLES_JSON ={
"examples":[
{
"events":[
{
"name":"market",
"action":"price_change",
"object":"price",
"direction":"up",
"date":"1973-10-01",
"summary":"Oil prices increased."
}
],
"vocab_suggestion":{"term":None ,"slot":None ,"score":0.0 }
}
]
}

SIMPLE_EVENT_SCHEMA ={
"type":"object",
"properties":{
"events":{
"type":"array",
"items":{
"type":"object",
"properties":{
"name":{"type":"string"},
"action":{"type":"string"},
"object":{"type":"string"},
"direction":{"type":"string"},
"date":{"type":["string","null"]},
"summary":{"type":"string"}
},
"required":["name","action","object","direction","summary"],
"additionalProperties":False 
}
},
"vocab_suggestion":{
"type":"object",
"properties":{
"term":{"type":["string","null"]},

"slot":{"type":["string","null"],"enum":["names","actions","objects","directions",None ]},
"score":{"type":"number"}
},
"required":["term","slot","score"],
"additionalProperties":False 
}
},
"required":["events"],
"additionalProperties":False 
}

REQUIRED_EVENT_FIELDS ={"name","action","object","direction","summary"}

client =AsyncOpenAI (
api_key =os .getenv ("OPENAI_API_KEY"),
base_url =os .getenv ("OPENAI_BASE_URL"),
timeout =REQUEST_TIMEOUT ,
)
SYSTEM_PROMPT ="""
You are an energy event extraction assistant. Output strictly JSON.

Goal: Return the simplest possible events. For each distinct event in the text,
extract ONLY these fields:
- name (one of allowed_values.names)
- action (one of allowed_values.actions)
- object (one of allowed_values.objects)
- direction (one of allowed_values.directions; if unsure use "ambiguous")
- date (YYYY-MM-DD if explicitly evident; otherwise return null or omit the field; do not make up dates)
- summary (one concise sentence capturing the most important part about this event)
- Do NOT set name to a country just because the text mentions a geography (e.g., "U.S. inventories").
  If it is a report or data, set name to the agency (eia/iea/api). Treat geographies as scope, not actor.
- For price-only market color ("Brent up 1%", "Oil prices fell ..."), set name="market".
- For OPEC+ quota/cut/extension, set name to "opec_plus" (or "opec" if OPEC+ is not explicit).
- Use "united_states" only when the U.S. government is the agent (sanction/tariff/price cap/SPR ops).
- If no clear agent, please reply with the "name" you think should be selected in the "vocab_suggestion" section.

Vocabulary suggestion rules:
- If a clearly relevant term is NOT in allowed_values (e.g., "bivalent booster", "wastewater surveillance"), return it in "vocab_suggestion" with:
  - "term": raw phrase (lowercased),
  - "slot": one of ("names","actions","objects","directions"),
  - "score": confidence 0.0–1.0.
- Return only ONE best candidate; if no candidate, return {"term": none, "slot": none, "score": 0.0}.

Hard rules:
1) Output JSON only, no extra text.
2) Use only allowed_values for name/action/object/direction.
3) One input may yield multiple events; avoid duplicates with same
   (name, action, object, direction, date, summary).
Return shape:
{
  "events": [
    {
      "name": "united states",
      "action": "report_release",
      "object": "price",
      "direction": "up",
      "date": "1998-01-02",
      "summary": "United States released a report indicating gradual increases in retail gasoline prices."
    }
  ],
  "vocab_suggestion": {"term": , "slot": , "score": }
}
"""

def build_assistant_examples ()->Dict [str ,Any ]:
    return EXAMPLES_JSON 

def build_allowed_values ()->Dict [str ,Any ]:
    return ALLOWED_VALUES 

def truncate_text (text :str ,limit :int =MAX_INPUT_CHARS )->str :
    text =str (text )
    if len (text )<=limit :
        return text 
    return text [:limit ]+" ..."

def build_user_prompt (row_id :str ,raw_text :str ,text_field :str ="Final_Search_6")->str :
    raw_text =truncate_text (raw_text )
    return f"""Row ID: {row_id }
Text ({text_field }):
{raw_text }

Extract 0 or more events from the text above. Follow strictly the rules and allowed_values provided earlier.
Output JSON only with shape:
{{
  "events": [ ... zero or more event objects ... ],
  "vocab_suggestion": {{"term": "... or null", "slot": "names|actions|objects|directions or null", "score": 0.0}}
}}"""





from datetime import date as _date 

def _is_nan_like (x )->bool :
    try :

        return pd .isna (x )
    except Exception :
        return False 

def coerce_to_iso_date (val )->str :

    if val is None or _is_nan_like (val ):
        return ""
    s =str (val ).strip ()
    if not s or s .lower ()in {"null","none","na","n/a"}:
        return ""
    try :
        dt =dtparser .parse (s ,fuzzy =True )

        return dt .date ().isoformat ()
    except Exception :
        return s 

def json_dumps (obj :Any )->str :
    return orjson .dumps (obj ,option =orjson .OPT_SERIALIZE_NUMPY ).decode ("utf-8")

def is_enum_ok (val :Optional [str ],enum_list :List [str ])->bool :
    if val is None :
        return True 
    return val in enum_list 

def validate_event_shape (evt :Dict [str ,Any ])->Tuple [bool ,List [str ]]:
    errs =[]
    missing =REQUIRED_EVENT_FIELDS -set (evt .keys ())
    if missing :
        errs .append (f"missing fields: {sorted (missing )}")


    for k in ["name","action","object","direction","summary"]:
        if k in evt and not isinstance (evt [k ],str ):
            errs .append (f"{k } must be string")


    if "date"in evt and not (evt ["date"]is None or isinstance (evt ["date"],str )):
        errs .append ("date must be string or null if present")

    return (len (errs )==0 ),errs 

def enforce_min_allowed (evt :Dict [str ,Any ],allowed :Dict [str ,Any ])->Tuple [bool ,List [str ]]:
    errs =[]
    if evt .get ("action")not in allowed ["actions"]:
        errs .append (f"action '{evt .get ('action')}' not in allowed actions")
    if evt .get ("object")not in allowed ["objects"]:
        errs .append (f"object '{evt .get ('object')}' not in allowed objects")
    if evt .get ("direction")not in allowed ["directions"]:
        errs .append (f"direction '{evt .get ('direction')}' not in allowed directions")
    return (len (errs )==0 ),errs 

import unicodedata 

def _canon_token (s :str )->str :
    if s is None :
        return ""
    t =str (s ).lower ().strip ()

    t =unicodedata .normalize ("NFKD",t )
    t ="".join (ch for ch in t if not unicodedata .combining (ch ))
    t =re .sub (r"[^a-z0-9]+","_",t )
    t =re .sub (r"_+","_",t ).strip ("_")
    return t 

def build_keyword (name :str ,action :str ,obj :str ,direction :str )->str :
    n =_canon_token (name )[:60 ]
    return f"{n }|{action }|{obj }|{direction }"

def aaod_key (evt :Dict [str ,Any ])->str :
    actor_type =(evt .get ("actor")or {}).get ("type")or "other"
    return f"{actor_type }|{evt .get ('action')}|{evt .get ('object')}|{evt .get ('direction')}"

def dedup_key (e :dict )->str :
    name =e .get ("name",{})
    actortype =actor .get ("type","unknown")
    action =e .get ("action","unknown")
    obj =e .get ("object","unknown")
    direction =e .get ("direction","ambiguous")

    loc =e .get ("location",{})
    region ="unknown"

    if isinstance (loc ,dict ):
        region =loc .get ("primary_iso3")or "|".join (sorted (loc .get ("areas")or []))or "unknown"
    elif isinstance (loc ,list )and loc :
        first =loc [0 ]
        if isinstance (first ,dict ):
            region =first .get ("primary_iso3")or "|".join (sorted (first .get ("areas")or []))or "unknown"

    return f"{actortype }|{action }|{obj }|{direction }|{region }"



@retry (wait =wait_exponential (multiplier =1 ,min =2 ,max =20 ),
stop =stop_after_attempt (5 ),
retry =retry_if_exception_type (Exception ))
async def call_openai (model :str ,system_prompt :str ,examples :dict ,
allowed_values :dict ,user_prompt :str )->dict :
    messages =[
    {"role":"system","content":system_prompt .strip ()},
    {"role":"assistant","content":json .dumps ({"allowed_values":allowed_values },ensure_ascii =False )},
    {"role":"assistant","content":json .dumps (examples ,ensure_ascii =False )},
    {"role":"user","content":user_prompt .strip ()}
    ]

    resp =await client .chat .completions .create (
    model =model ,
    messages =messages ,
    temperature =TEMPERATURE ,
    max_tokens =2000 ,
    response_format ={"type":"json_object"}
    )
    content =resp .choices [0 ].message .content .strip ()

    if "{"in content :
        content =content [content .index ("{"):]

    try :
        return json .loads (content )
    except json .JSONDecodeError :

        return {
        "ok":False ,
        "error":"JSONDecodeError",
        "raw":content [:200 ]
        }

async def process_row (sema :asyncio .Semaphore ,row :Dict [str ,Any ],model :str )->Dict [str ,Any ]:
    async with sema :
        row_id =str (row .get ("date")or row .get ("start_date")or row .get ("end_date")or "").strip ()
        text =str (row .get ("Final_Search_6")or "").strip ()
        if not text :
            return {"row_id":row_id ,"ok":True ,"events":[],"vocab_suggestion":{"term":None ,"slot":None ,"score":0.0 }}

        user_prompt =build_user_prompt (row_id =row_id ,raw_text =text )
        try :
            out =await call_openai (
            model =model ,
            system_prompt =SYSTEM_PROMPT ,
            examples =build_assistant_examples (),
            allowed_values =build_allowed_values (),
            user_prompt =user_prompt 
            )
        except Exception as e :
            return {"row_id":row_id ,"ok":False ,
            "vocab_suggestion":{"term":None ,"slot":None ,"score":0.0 },
            "error":f"LLM call failed: {e }"}
        try :
            validate (instance =out ,schema =SIMPLE_EVENT_SCHEMA ,cls =Draft202012Validator )
        except Exception as e :
            vs =out .get ("vocab_suggestion")if isinstance (out ,dict )else None 
            if not isinstance (vs ,dict ):
                vs ={"term":None ,"slot":None ,"score":0.0 }
            return {"row_id":row_id ,"ok":False ,
            "vocab_suggestion":vs ,
            "error":f"schema validation failed: {e }",
            "raw":out }
        events =out .get ("events")or []
        flat_rows =[]
        errs_total =[]

        csv_date_fallback =coerce_to_iso_date (row .get ("date"))
        for evt in events :
            for k in ["action","object","direction"]:
                if k in evt and isinstance (evt [k ],str ):
                    evt [k ]=evt [k ].strip ().lower ()

            ok1 ,errs1 =validate_event_shape (evt )
            ok2 ,errs2 =enforce_min_allowed (evt ,ALLOWED_VALUES )
            errs =errs1 +errs2 
            if ok1 and ok2 :
                name =(evt ["name"]or "").strip ()
                action =evt ["action"]
                obj =evt ["object"]
                direction =evt ["direction"]
                val_date =evt .get ("date")

                if isinstance (val_date ,str ):
                    cand =val_date .strip ()

                    cand =""if cand .lower ()in {"","null","none","na","n/a"}else cand 
                    date_str =coerce_to_iso_date (cand )if cand else ""
                elif val_date is None :
                    date_str =""
                else :
                    date_str =""
                if not date_str :
                    date_str =csv_date_fallback 
                summary =(evt ["summary"]or "").strip ()

                keyword =build_keyword (name ,action ,obj ,direction )
                flat_rows .append ({
                "date":date_str ,
                "name":name ,
                "action":action ,
                "object":obj ,
                "direction":direction ,
                "summary":summary ,
                "keyword":keyword 
                })
            else :
                errs_total .append ({"errors":errs ,"raw":evt })

        vs =out .get ("vocab_suggestion")or {"term":None ,"slot":None ,"score":0.0 }
        return {
        "row_id":row_id ,
        "ok":(len (flat_rows )>0 ),
        "simple_rows":flat_rows ,
        "vocab_suggestion":vs ,
        "errors":errs_total 
        }

def filter_rare_aaod (events :List [Dict [str ,Any ]],min_count :int =5 )->Tuple [List [Dict [str ,Any ]],pd .DataFrame ]:
    aaod_counts :Dict [str ,int ]={}
    for e in events :
        k =aaod_key (e )
        aaod_counts [k ]=aaod_counts .get (k ,0 )+1 
    keep =set ([k for k ,c in aaod_counts .items ()if c >=min_count ])

    filtered =[e for e in events if aaod_key (e )in keep ]

    rows =[{"aaod":k ,"count":c }for k ,c in sorted (aaod_counts .items (),key =lambda x :-x [1 ])]
    report =pd .DataFrame (rows )
    return filtered ,report 

def write_jsonl (path :str ,records :List [Dict [str ,Any ]])->None :
    parent =os .path .dirname (path )
    if parent :
        os .makedirs (parent ,exist_ok =True )
    with open (path ,"w",encoding ="utf-8")as f :
        for r in records :
            f .write (json_dumps (r )+"\n")

async def main_async (args ):
    df =read_energy_csv (args .input )
    if "Final_Search_6"not in df .columns :
        raise RuntimeError ("Column 'Final_Search_6' not found in the input CSV.")
    if "date"not in df .columns :

        if "start_date"in df .columns :
            df ["date"]=df ["start_date"]
        elif "end_date"in df .columns :
            df ["date"]=df ["end_date"]
        else :

            df ["date"]=[f"ROW-{i }"for i in range (len (df ))]

    rows =df [["date","Final_Search_6"]].to_dict ("records")
    if args .limit and args .limit >0 :
        rows =rows [:args .limit ]

    sema =asyncio .Semaphore (args .max_concurrency )
    tasks =[process_row (sema ,row ,args .model )for row in rows ]
    results :List [Dict [str ,Any ]]=[]
    for fut in tqdm (asyncio .as_completed (tasks ),total =len (tasks ),desc ="Extracting"):
        results .append (await fut )
    write_jsonl (OUTPUT_RAW_JSONL ,results )
    simple_rows :List [Dict [str ,Any ]]=[]
    errors :List [Dict [str ,Any ]]=[]
    for r in results :
        if not r .get ("ok"):
            errors .append (r )
        for e in (r .get ("simple_rows")or []):
            simple_rows .append (e )

    out_simple_csv =os .path .splitext (OUTPUT_EVENTS_JSONL )[0 ]+"_simple.csv"
    pd .DataFrame (simple_rows ).to_csv (out_simple_csv ,index =False )


    out_simple_jsonl =os .path .splitext (OUTPUT_EVENTS_JSONL )[0 ]+"_simple.jsonl"
    write_jsonl (out_simple_jsonl ,simple_rows )
    n_rows =len (rows )
    n_ok =sum (1 for r in results if r .get ("ok"))
    print ("\n=== Summary ===")
    print (f"Rows processed: {n_rows }")
    print (f"Row-level OK:   {n_ok }/{n_rows }")
    print (f"Simplified events: {len (simple_rows )}")
    print (f"Saved: {out_simple_csv }, {out_simple_jsonl }, {OUTPUT_RAW_JSONL }")
    if errors :
        print (f"Rows with errors: {len (errors )} (see {OUTPUT_RAW_JSONL })")

    suggestions :List [Dict [str ,Any ]]=[]
    for r in results :
        vs =r .get ("vocab_suggestion")
        if isinstance (vs ,dict ):
            rec =dict (vs )
            rec ["_row_id"]=r .get ("row_id")
            suggestions .append (rec )

    if suggestions :
        write_jsonl (OUTPUT_SUGG_JSONL ,suggestions )
        print (f"Vocab suggestions: {len (suggestions )} → {OUTPUT_SUGG_JSONL }")
def parse_args ():
    p =argparse .ArgumentParser (description ="Energy event extraction with LLM (parallel).")
    p .add_argument ("--input",type =str ,default ="dataset/ene.csv",help ="Path to input CSV")
    p .add_argument ("--model",type =str ,default ="gpt-4o-mini",help ="OpenAI model (e.g., gpt-4o-mini, gpt-4.1)")
    p .add_argument ("--max-concurrency",type =int ,default =5 ,help ="Max parallel requests")
    p .add_argument ("--limit",type =int ,default =0 ,help ="Process only first N rows (0 = all)")
    p .add_argument ("--min-count",type =int ,default =1 ,help ="AAOD min count to keep")
    return p .parse_args ()

def main ():
    os .chdir (os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
    args =parse_args ()
    asyncio .run (main_async (args ))

if __name__ =="__main__":
    main ()
