import json
import pandas as pd
from collections import Counter

def load_jsonl (path ):
    records =[]
    with open (path ,"r",encoding ="utf-8")as f :
        for line in f :
            line =line .strip ()
            if line :
                records .append (json .loads (line ))
    return records

def save_jsonl (records ,path ,safe_default_str =False ):
    with open (path ,"w",encoding ="utf-8")as f :
        for r in records :
            if safe_default_str :
                f .write (json .dumps (r ,ensure_ascii =False ,default =str )+"\n")
            else :
                f .write (json .dumps (r ,ensure_ascii =False )+"\n")

def dedup_then_filter (records ,min_keyword_count =2 ):
    df =pd .DataFrame (records )
    df ["date"]=pd .to_datetime (df ["date"],errors ="coerce")
    df =df .dropna (subset =["date"])
    df =df .sort_values ("date",kind ="stable")
    df =df .drop_duplicates (subset =["summary"],keep ="first")

    df =df .sort_values (["date"],kind ="stable")
    df =df .drop_duplicates (subset =["date","keyword"],keep ="first")
    counts_all =Counter (df ["keyword"])
    df =df [df ["keyword"].map (lambda k :counts_all [k ]>=min_keyword_count )]
    counts_kept =Counter (df ["keyword"])


    df ["date"]=df ["date"].dt .strftime ("%Y-%m-%d")
    return df .to_dict (orient ="records"),counts_all ,counts_kept
if __name__ =="__main__":
    input_file ="Num2Text/New_num2text/event_extrect/events_flat_simple.jsonl"
    output_file ="Num2Text/New_num2text/event_extrect/events_flat_simple_filtered.jsonl"
    counts_all_file ="Num2Text/New_num2text/event_extrect/keyword_counts.json"
    counts_kept_file ="Num2Text/New_num2text/event_extrect/keyword_counts_filtered.json"
    MIN_COUNT =10
    records =load_jsonl (input_file )
    filtered_records ,counts_all ,counts_kept =dedup_then_filter (records ,min_keyword_count =MIN_COUNT )
    save_jsonl (filtered_records ,output_file )
    with open (counts_all_file ,"w",encoding ="utf-8")as f :
        json .dump (dict (counts_all ),f ,ensure_ascii =False ,indent =2 )
    with open (counts_kept_file ,"w",encoding ="utf-8")as f :
        json .dump (dict (counts_kept ),f ,ensure_ascii =False ,indent =2 )

    print (f": {len (records )}")
    print (f"（）: {sum (counts_all .values ())}  # ")
    print (f": {len (filtered_records )}")
    print (f": >= {MIN_COUNT }")
    print (f": {len (counts_all )}，: {len (counts_kept )}")

