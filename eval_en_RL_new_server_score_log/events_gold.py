from __future__ import annotations
import json
import math
import re
import datetime as dt
from dataclasses import dataclass
from typing import List ,Dict ,Optional ,Tuple
from aaodt_schema import AAOD ,normalize_slot
STOPWORDS ={
"the","a","an","of","and","to","in","on","for","with","by",
"at","as","is","are","was","were","be","been","or","from"
}
def tokenize_quote (q :Optional [str ])->List [str ]:

    if not q :
        return []
    toks =re .findall (r"[A-Za-z0-9]+",q .lower ())
    return [t for t in toks if t not in STOPWORDS ]

def jaccard (a_tokens :List [str ],b_tokens :List [str ])->float :
    A ,B =set (a_tokens ),set (b_tokens )
    if not A and not B :
        return 1.0
    return len (A &B )/max (1 ,len (A |B ))

@dataclass
class EventRow :
    date :dt .date
    name :Optional [str ]
    action :Optional [str ]
    obj :Optional [str ]
    direction :Optional [str ]
    mechanism :Optional [str ]
    quote :Optional [str ]
    confidence :Optional [float ]
    magnitude :Optional [str ]
    def aaodt (self )->AAOD :
        return AAOD .from_strings (self .name ,self .action ,self .obj ,self .direction )

def _to_date_strict (x :str )->dt .date :

    return dt .date .fromisoformat (str (x )[:10 ])

def _to_date_flexible (x :str )->dt .date :
    import pandas as pd
    s =str (x ).strip ()
    if re .fullmatch (r"\d{4}",s ):
        return dt .date (int (s ),1 ,1 )
    m =re .fullmatch (r"(\d{4})[-/](\d{1,2})",s )
    if m :
        y ,mo =int (m .group (1 )),int (m .group (2 ))
        return dt .date (y ,mo ,1 )
    d =pd .to_datetime (s ,errors ="coerce")
    if pd .isna (d ):

        m2 =re .search (r"\d{4}-\d{2}-\d{2}",s )
        if m2 :
            return dt .date .fromisoformat (m2 .group (0 ))

        m3 =re .search (r"\d{4}",s )
        if m3 :
            return dt .date (int (m3 .group (0 )),1 ,1 )
        raise ValueError (f"Unparseable date: {x }")
    return d .date ()
def load_events_jsonl (path :str )->List [EventRow ]:
    rows :List [EventRow ]=[]
    with open (path ,"r",encoding ="utf-8")as f :
        for line in f :
            if not line .strip ():
                continue
            obj =json .loads (line )
            date_str =(
            (obj .get ("timing")or {}).get ("mentioned_date")
            or obj .get ("date")
            or obj .get ("published_date")
            )
            if not date_str :
                continue
            rows .append (
            EventRow (
            date =_to_date_strict (date_str ),
            name =(obj .get ("actor")or {}).get ("name")or obj .get ("name")or obj .get ("actor_canonical"),
            action =obj .get ("action"),
            obj =obj .get ("object"),
            direction =obj .get ("direction"),
            mechanism =obj .get ("mechanism"),
            quote =(obj .get ("source")or {}).get ("quote")or obj .get ("quote"),
            confidence =float ((obj .get ("confidence")or {}).get ("extraction",0.7 ))
            if isinstance (obj .get ("confidence"),dict )
            else float (obj .get ("confidence",0.7 )),
            magnitude =(obj .get ("magnitude")or {}).get ("comparator")
            if isinstance (obj .get ("magnitude"),dict )
            else obj .get ("magnitude"),
            )
            )
    return rows

def load_events_jsonl_safe (path :str )->List [EventRow ]:
    rows :List [EventRow ]=[]
    with open (path ,"r",encoding ="utf-8")as f :
        for line in f :
            if not line .strip ():
                continue
            obj =json .loads (line )
            date_str =(
            (obj .get ("timing")or {}).get ("mentioned_date")
            or obj .get ("date")
            or obj .get ("published_date")
            )
            if not date_str :
                continue
            try :
                d =_to_date_flexible (date_str )
            except Exception :
                continue
            rows .append (
            EventRow (
            date =d ,
            name =(obj .get ("actor")or {}).get ("name")or obj .get ("name")or obj .get ("actor_canonical"),
            action =obj .get ("action"),
            obj =obj .get ("object"),
            direction =obj .get ("direction"),
            mechanism =obj .get ("mechanism"),
            quote =(obj .get ("source")or {}).get ("quote")or obj .get ("quote"),
            confidence =float ((obj .get ("confidence")or {}).get ("extraction",0.7 ))
            if isinstance (obj .get ("confidence"),dict )
            else float (obj .get ("confidence",0.7 )),
            magnitude =(obj .get ("magnitude")or {}).get ("comparator")
            if isinstance (obj .get ("magnitude"),dict )
            else obj .get ("magnitude"),
            )
            )
    return rows

def time_decay (weeks_ago :int ,tau :float =2.5 )->float :
    return math .exp (-max (0 ,weeks_ago )/tau )

def dir_score (direction :Optional [str ],price_delta_sign :Optional [int ])->float :
    if price_delta_sign is None or direction is None :
        return 0.7
    direction =normalize_slot ("DIRECTION",direction )
    if price_delta_sign ==0 :
        return 0.7
    if (price_delta_sign >0 and direction =="up")or (price_delta_sign <0 and direction =="down"):
        return 1.0
    return 0.4

def magnitude_score (mag :Optional [str ])->float :
    if not mag :
        return 0.6
    m =str (mag ).lower ()
    if any (k in m for k in ["sharply","deep","major","significant","severe"]):
        return 1.0
    if any (k in m for k in ["slight","mild","small"]):
        return 0.7
    return 0.8
@dataclass
class GoldCause :

    aaodt :AAOD
    rep_event_idx :int
    support_size :int
    score :float
    date :dt .date
    mechanism :Optional [str ]
    quote :Optional [str ]
def build_gold_rte_for_window (
window_start_date :dt .date ,
window_end_date :dt .date ,
all_events_in_month :List [EventRow ],
price_delta_sign :Optional [int ],
k_primary :int =10 ,
neardup_date_days :int =14 ,
neardup_jaccard :float =0.85 ,
mmr_lambda :float =0.7 ,
)->Tuple [List [GoldCause ],List [GoldCause ]]:
    buckets :Dict [str ,List [int ]]={}
    for i ,ev in enumerate (all_events_in_month ):
        key =ev .aaodt ().key ()
        buckets .setdefault (key ,[]).append (i )
    if not buckets :
        return [],[]
    rep_indices :List [int ]=[]
    support_sizes :Dict [int ,int ]={}
    used =set ()
    for _ ,idxs in buckets .items ():
        idxs_sorted =sorted (idxs ,key =lambda i :all_events_in_month [i ].date )
        for i in idxs_sorted :
            if i in used :
                continue
            group =[i ]
            qi =tokenize_quote (all_events_in_month [i ].quote )
            for j in idxs_sorted :
                if j ==i or j in used :
                    continue
                date_close =abs ((all_events_in_month [j ].date -all_events_in_month [i ].date ).days )<=neardup_date_days
                qj =tokenize_quote (all_events_in_month [j ].quote )
                if date_close and jaccard (qi ,qj )>=neardup_jaccard :
                    group .append (j )
                    used .add (j )
            used .add (i )
            rep_indices .append (i )
            support_sizes [i ]=len (group )
    items :List [GoldCause ]=[]
    for i in rep_indices :
        ev =all_events_in_month [i ]
        weeks_ago =max (0 ,(window_end_date -ev .date ).days )//7
        score =(
        0.35 *time_decay (weeks_ago )+
        0.20 *dir_score (ev .direction ,price_delta_sign )+
        0.15 *float (ev .confidence or 0.7 )+
        0.10 *min (1.0 ,math .log1p (support_sizes .get (i ,1 ))/math .log (5 ))
        )

        items .append (GoldCause (
        aaodt =ev .aaodt (),
        rep_event_idx =i ,
        support_size =support_sizes .get (i ,1 ),
        score =score ,
        date =ev .date ,
        mechanism =ev .mechanism ,
        quote =ev .quote ,
        ))

    def sim (a :GoldCause ,b :GoldCause )->float :
        objdir =int (a .aaodt .obj ==b .aaodt .obj )+int (a .aaodt .direction ==b .aaodt .direction )+int (a .aaodt .name ==b .aaodt .name )
        qsim =jaccard (tokenize_quote (a .quote ),tokenize_quote (b .quote ))
        return 0.7 *objdir +0.3 *qsim
    selected :List [GoldCause ]=[]
    candidates =items [:]
    while candidates and len (selected )<k_primary :
        best ,best_val =None ,-1e9
        for c in candidates :
            redundancy =0.0 if not selected else max (sim (c ,s )for s in selected )
            val =mmr_lambda *c .score -(1 -mmr_lambda )*redundancy
            if val >best_val :
                best_val ,best =val ,c
        selected .append (best )
        candidates =[c for c in candidates if c is not best ]
    gold_primary =selected
    primary_ids =set (id (x )for x in gold_primary )
    gold_support =[x for x in items if id (x )not in primary_ids ]
    return gold_primary ,gold_support

