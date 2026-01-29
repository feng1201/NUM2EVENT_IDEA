import argparse
import pandas as pd


def compute_params (train_series :pd .Series ,
method :str ,
log_eps :float =1e-6 ):

    method =method .lower ()
    extra ={}

    if method =="robust":
        median =train_series .median ()
        q1 =train_series .quantile (0.25 )
        q3 =train_series .quantile (0.75 )
        iqr =q3 -q1
        if iqr ==0 :
            raise ValueError ("IQR is 0; cannot apply robust scaling. Check the training data distribution.")
        params ={"median":median ,"iqr":iqr ,"q1":q1 ,"q3":q3 }

    elif method =="zscore":
        mean =train_series .mean ()
        std =train_series .std ()
        if std ==0 :
            raise ValueError ("Std is 0; cannot apply z-score normalization. Check the training data distribution.")
        params ={"mean":mean ,"std":std }

    elif method =="minmax":
        vmin =train_series .min ()
        vmax =train_series .max ()
        if vmax ==vmin :
            raise ValueError ("Min equals max; cannot apply min-max normalization. Check the training data distribution.")
        params ={"min":vmin ,"max":vmax }

    elif method =="logz":


        min_val =train_series .min ()
        if min_val <=0 :
            offset =-min_val +log_eps
        else :
            offset =0.0
        shifted =train_series +offset
        log_vals =(shifted +log_eps ).apply (lambda x :pd .np .log (x ))

        mean =log_vals .mean ()
        std =log_vals .std ()
        if std ==0 :
            raise ValueError ("Std after log-transform is 0; cannot apply z-score normalization. Check the training data distribution.")

        params ={"mean":mean ,"std":std ,"offset":offset }
        extra ["log_min_val"]=float (min_val )

    else :
        raise ValueError (f"Unknown normalization method: {method }")

    return params ,extra


def apply_transform (series :pd .Series ,
method :str ,
params :dict ,
log_eps :float =1e-6 )->pd .Series :

    method =method .lower ()

    if method =="robust":
        median =params ["median"]
        iqr =params ["iqr"]
        return (series -median )/iqr

    elif method =="zscore":
        mean =params ["mean"]
        std =params ["std"]
        return (series -mean )/std

    elif method =="minmax":
        vmin =params ["min"]
        vmax =params ["max"]
        return (series -vmin )/(vmax -vmin )

    elif method =="logz":
        mean =params ["mean"]
        std =params ["std"]
        offset =params ["offset"]
        shifted =series +offset
        log_vals =(shifted +log_eps ).apply (lambda x :pd .np .log (x ))
        return (log_vals -mean )/std

    else :
        raise ValueError (f"Unknown normalization method: {method }")


def normalize_file (
data_path :str ,
save_path :str ,
split_date :str ="2022-01-01",
date_col :str ="Date",
ot_col :str ="OT",
norm_method :str ="robust",
log_eps :float =1e-6 ,
):
    print (f"🔍 Loading data from: {data_path }")
    df =pd .read_csv (data_path )


    df [date_col ]=pd .to_datetime (df [date_col ])


    split_timestamp =pd .Timestamp (split_date )
    train_df =df [df [date_col ]<split_timestamp ]

    if train_df .empty :
        raise ValueError (f"Training split is empty: no data earlier than {split_date }")

    print (f"📆 Using data before {split_date } as training for normalization params.")
    train_series =train_df [ot_col ]


    params ,extra =compute_params (train_series ,norm_method ,log_eps )

    print (f"📊 Normalization method: {norm_method }")
    if norm_method =="robust":
        print (f"   median = {params ['median']:.6f}")
        print (f"   q1     = {params ['q1']:.6f}")
        print (f"   q3     = {params ['q3']:.6f}")
        print (f"   iqr    = {params ['iqr']:.6f}")
    elif norm_method =="zscore":
        print (f"   mean   = {params ['mean']:.6f}")
        print (f"   std    = {params ['std']:.6f}")
    elif norm_method =="minmax":
        print (f"   min    = {params ['min']:.6f}")
        print (f"   max    = {params ['max']:.6f}")
    elif norm_method =="logz":
        print (f"   offset = {params ['offset']:.6f}")
        print (f"   mean(log) = {params ['mean']:.6f}")
        print (f"   std(log)  = {params ['std']:.6f}")
        if "log_min_val"in extra :
            print (f"   original min value = {extra ['log_min_val']:.6f}")


    df [ot_col ]=apply_transform (df [ot_col ],norm_method ,params ,log_eps )


    df .to_csv (save_path ,index =False )
    print (f"✅ Saved normalized CSV to: {save_path }")


if __name__ =="__main__":
    parser =argparse .ArgumentParser (description ="Normalize time-series CSV with different methods.")

    parser .add_argument ("--data_path",type =str ,required =True ,
    help ="Path to input CSV file.")
    parser .add_argument ("--save_path",type =str ,required =True ,
    help ="Path to save normalized CSV.")
    parser .add_argument ("--split_date",type =str ,default ="2022-01-01",
    help ="Use data before this date to compute normalization params (default=2022-01-01).")
    parser .add_argument ("--date_col",type =str ,default ="Date",
    help ="Name of the date column (default='Date').")
    parser .add_argument ("--ot_col",type =str ,default ="OT",
    help ="Name of the numerical column to normalize (default='OT').")
    parser .add_argument ("--norm_method",type =str ,default ="robust",
    choices =["robust","zscore","minmax","logz"],
    help ="Normalization method: robust | zscore | minmax | logz (default='robust').")
    parser .add_argument ("--log_eps",type =float ,default =1e-6 ,
    help ="Small epsilon for log-transform to avoid log(0) (default=1e-6).")

    args =parser .parse_args ()

    normalize_file (
    data_path =args .data_path ,
    save_path =args .save_path ,
    split_date =args .split_date ,
    date_col =args .date_col ,
    ot_col =args .ot_col ,
    norm_method =args .norm_method ,
    log_eps =args .log_eps ,
    )

