from __future__ import annotations 
import re ,math ,argparse ,warnings 

from typing import List ,Dict ,Any ,Tuple ,Set ,Optional ,Sequence 

from dataclasses import dataclass 
import textwrap ,json ,os ,pandas as pd ,numpy as np 

import statsmodels .api as sm 
from statsmodels .regression .linear_model import OLS 
from statsmodels .stats .sandwich_covariance import cov_hac 

script_path ="json_event_irf_and_synthesis_v4.py"

def _lp_prepare_reg_panel (df :pd .DataFrame ,
event_cols :list [str ],
add_week_dummies :bool =True )->tuple [pd .DataFrame ,list [str ],list [str ]]:
    out =df .copy ()
    out ["date"]=pd .to_datetime (out ["date"],errors ="coerce")
    out =out .dropna (subset =["date","dprice"]).reset_index (drop =True )
    lag_cols =[]
    for k in [1 ,2 ,3 ,4 ]:
        col =f"dprice_lag{k }"
        out [col ]=out ["dprice"].shift (k )
        lag_cols .append (col )
    woy_cols =[]
    if add_week_dummies :
        woy =out ["date"].dt .isocalendar ().week .astype (int )
        dummies =pd .get_dummies (woy ,prefix ="w",drop_first =True )
        out =pd .concat ([out ,dummies ],axis =1 )
        woy_cols =list (dummies .columns )
    keep =["date","dprice"]+event_cols +lag_cols +woy_cols 
    out =out [keep ]
    return out ,lag_cols ,woy_cols 


def local_projection_irf_multi_lp (df :pd .DataFrame ,
event_cols :list [str ],
H :int =8 ,
add_week_dummies :bool =True ,
nw_lags :int =4 ,
min_pos :int =0 )->dict [str ,pd .DataFrame ]:
    panel ,lag_cols ,woy_cols =_lp_prepare_reg_panel (df ,event_cols ,add_week_dummies )
    results :dict [str ,list [dict ]]={c :[]for c in event_cols }
    ybase =panel ["dprice"].astype (float )
    pos_counts ={c :int ((panel [c ].fillna (0 )>0 ).sum ())for c in event_cols }
    active_cols =[c for c in event_cols if pos_counts .get (c ,0 )>=int (min_pos )]
    if not active_cols :
        return {c :pd .DataFrame (columns =["h","beta","se","t","n"])for c in event_cols }

    for h in range (H +1 ):
        tmp =panel .copy ()
        tmp [f"lead{h }"]=ybase .shift (-h )
        keep =[f"lead{h }"]+active_cols +lag_cols +woy_cols 
        tmp =tmp .dropna (subset =keep )
        if tmp .empty :
            print ("Empty regression panel")
            for c in event_cols :
                results [c ].append ({"h":h ,"beta":np .nan ,"se":np .nan ,"t":np .nan ,"n":0 })
            continue 
        Y =tmp [f"lead{h }"].values 
        X =tmp [active_cols +lag_cols +woy_cols ].values 

        X =sm .add_constant (X ,has_constant ="add")
        X =X .astype (float )

        m =OLS (Y ,X ).fit ()
        try :
            cov =cov_hac (m ,nlags =int (nw_lags ))
            se_all =np .sqrt (np .diag (cov ))
        except Exception :
            se_all =np .sqrt (np .diag (m .cov_params ()))
        for j ,c in enumerate (active_cols ,start =1 ):
            beta_j =float (m .params [j ])
            se_j =float (se_all [j ])if j <len (se_all )else np .nan 

            t_j =float (beta_j /se_j )if (se_j is not None and se_j >0 )else np .nan 

            results [c ].append ({"h":h ,"beta":beta_j ,"se":se_j ,"t":t_j ,"n":int (len (tmp ))})
        for c in set (event_cols )-set (active_cols ):
            results [c ].append ({"h":h ,"beta":np .nan ,"se":np .nan ,"t":np .nan ,"n":0 })
    return {c :pd .DataFrame (rows )for c ,rows in results .items ()}
def _smooth_vec_ma (x :np .ndarray ,k :int =3 )->np .ndarray :
    if k <=1 or len (x )<3 :
        return x 
    pad =np .r_ [x [0 ],x ,x [-1 ]]
    out =np .convolve (pad ,np .ones (3 )/3.0 ,mode ="same")[1 :-1 ]
    return out 

def build_irf_dict_from_lp (lp_results :dict [str ,pd .DataFrame ],
remove_prefix :str ="evt_",
smooth_k :int =3 ,
shrink_small_t :float =1.0 )->dict [str ,pd .DataFrame ]:
    irf_dict :dict [str ,pd .DataFrame ]={}
    for col ,dfc in lp_results .items ():
        if dfc is None or dfc .empty :
            continue 
        h =dfc ["h"].astype (int ).to_numpy ()
        beta =dfc ["beta"].astype (float ).fillna (0.0 ).to_numpy ()
        if "t"in dfc .columns and shrink_small_t is not None and shrink_small_t >0 :
            tval =np .abs (dfc ["t"].fillna (0.0 ).to_numpy ())
            w =np .minimum (1.0 ,tval /float (shrink_small_t ))
            beta =beta *w 
        beta =_smooth_vec_ma (beta ,k =int (smooth_k ))
        key =col [len (remove_prefix ):]if remove_prefix and col .startswith (remove_prefix )else col 
        irf_dict [key ]=pd .DataFrame ({"h":h ,"irf":beta })
    return irf_dict 


def coerce_numeric_series (s :pd .Series )->pd .Series :
    if s .dtype .kind in "biufc":
        return s .astype (float )
    s2 =s .astype (str ).str .replace (",","",regex =False )
    s2 =s2 .str .replace ("$","",regex =False )
    s2 =s2 .str .strip ()
    return pd .to_numeric (s2 ,errors ="coerce")


def choose_date_column (df :pd .DataFrame )->str :
    cols =list (df .columns )
    lower ={c .lower ():c for c in cols }
    for cand in ["date","week","start_date","end_date","Date","DATE","DateTime"]:
        if cand .lower ()in lower :
            return lower [cand .lower ()]
    for c in cols :
        try :
            pd .to_datetime (df [c ],errors ="raise")
            return c 
        except Exception :
            pass 
    raise ValueError ("No date-like column found; please specify explicitly.")


def robust_read_csv (path :str )->pd .DataFrame :
    encodings =["utf-8","utf-8-sig","gb18030","latin-1"]
    for enc in encodings :
        try :
            df =pd .read_csv (path ,encoding =enc ,low_memory =False )
            return df 
        except Exception :
            continue 
    return pd .read_csv (path ,encoding_errors ="ignore",engine ="python")


def robust_read_jsonl (path :str )->List [dict ]:
    out =[]
    with open (path ,"r",encoding ="utf-8")as f :
        for line in f :
            line =line .strip ()
            if not line :
                continue 
            try :
                out .append (json .loads (line ))
            except Exception :

                try :
                    import orjson 
                    out .append (orjson .loads (line ))
                except Exception :
                    continue 
    return out 

def _is_energy_aaod (aaod :str )->bool :
    aaod =(aaod or "").lower ()
    parts =aaod .split ("|")
    obj =parts [2 ]if len (parts )>2 else ""
    act =parts [1 ]if len (parts )>1 else ""
    energy_objs =["gasoline","diesel","crude_oil","oil","petroleum","price"]
    if any (k in obj for k in energy_objs ):
        return True 
    if "price_change"in act :
        return True 
    return False 

def extract_aaod_key_from_record (rec :dict )->str |None :
    for k in ("keyword","category_key"):
        if k in rec and rec [k ]:
            aaod =normalize_aaod (rec [k ])
            return aaod if aaod else None 
    name =str (rec .get ("name")or (rec .get ("actor")or {}).get ("name","")).strip ()
    act =str (rec .get ("action","")).strip ()
    obj =str (rec .get ("object","")).strip ()
    direc =str (rec .get ("direction","")).strip ()
    if name and act and obj and direc :
        return normalize_aaod (f"{name }|{act }|{obj }|{direc }")
    return None 

def normalize_aaod (key :str )->str :
    if not key :
        return ""
    parts =[p .strip ().lower ().replace (" ","_")for p in str (key ).split ("|")]
    if len (parts )<4 :
        parts =(parts +["ambiguous"]*4 )[:4 ]
    alias ={
    "u_s_energy_information_administration":"eia",
    "united_states_of_america":"united_states",
    "u_s":"united_states",
    "us":"united_states",
    }
    parts [0 ]=alias .get (parts [0 ],parts [0 ])
    return "|".join (parts [:4 ])

def build_seasonal_poisson_intensity (events :pd .DataFrame )->Dict [str ,Dict [str ,Any ]]:
    events =events .copy ()
    events ["date"]=pd .to_datetime (events ["date"],errors ="coerce")
    events =events .dropna (subset =["date"])
    events ["woy"]=events ["date"].dt .isocalendar ().week .astype (int )
    out :Dict [str ,Dict [str ,Any ]]={}
    for aaod ,g in events .groupby ("aaod"):
        cnt =g .groupby ("woy").size ().reindex (range (1 ,54 ),fill_value =0 )
        y =cnt .values .astype (float )
        y =0.9 *y +0.1 *y .mean ()
        mu =y .mean ()+1e-6 
        alpha =math .log (mu )
        gammas ={}
        for i ,v in enumerate (y ):
            safe_value =max (v ,1e-6 )
            log_value =math .log (safe_value )
            gamma_value =log_value -alpha 
            gamma_float =float (gamma_value )
            week_number =int (i +1 )
            gammas [week_number ]=gamma_float 
        out [aaod ]={"alpha":float (alpha ),"gammas":gammas }
    return out 

def _lp_design_matrix (price_deltas :np .ndarray ,H :int )->Tuple [np .ndarray ,List [str ]]:
    cols =[]
    X =[]
    for h in range (1 ,H +1 ):
        cols .append (f"dP_t+{h }")
        X .append (np .roll (price_deltas ,-h ))
    X =np .vstack (X ).T 
    valid =np .arange (len (price_deltas )-H )
    return X [valid ],cols 


def estimate_irf_lp (df :pd .DataFrame ,events_df :pd .DataFrame ,aaod_keys :List [str ],H :int =8 )->Dict [str ,pd .DataFrame ]:
    out :Dict [str ,pd .DataFrame ]={}
    for key in aaod_keys :
        mask =(events_df ["aaod"]==key ).astype (int ).values 
        y =df ["dprice"].values .astype (float )
        X ,cols =_lp_design_matrix (y ,H )

        mask_aligned =mask [:len (X )]
        beta =np .linalg .lstsq (mask_aligned .reshape (-1 ,1 ),X [:,0 ],rcond =None )[0 ]

        h =np .arange (1 ,H +1 )
        irf =float (beta [0 ])*np .exp (-0.5 *(h -1 ))
        out [key ]=pd .DataFrame ({"h":h ,"irf":irf })
    return out 

def fit_ar4_dprice (price_deltas :np .ndarray )->Dict [str ,Any ]:
    y =price_deltas [4 :]
    X =np .column_stack ([price_deltas [3 :-1 ],price_deltas [2 :-2 ],price_deltas [1 :-3 ],price_deltas [0 :-4 ]])
    beta =np .linalg .lstsq (X ,y ,rcond =None )[0 ]
    return {"phi_dprice":beta .tolist ()}

def fit_ar4_price (price_deltas :np .ndarray )->Dict [str ,Any ]:
    price_deltas =np .asarray (price_deltas ,dtype =float )

    if len (price_deltas )<=4 :
        return {"phi":[0.0 ,0.0 ,0.0 ,0.0 ,0.0 ]}

    y =price_deltas [4 :]
    X_lags =np .column_stack ([
    price_deltas [3 :-1 ],
    price_deltas [2 :-2 ],
    price_deltas [1 :-3 ],
    price_deltas [0 :-4 ]
    ])
    X =np .column_stack ([np .ones (X_lags .shape [0 ]),X_lags ])
    beta ,residuals ,rank ,s =np .linalg .lstsq (X ,y ,rcond =None )
    return {"phi_price":beta .tolist ()}

def ar4_predict (next_last4 :List [float ],phi :List [float ])->float :
    return float (sum (p *l for p ,l in zip (phi ,next_last4 )))


def season_week_of_year (dt :pd .Timestamp )->int :
    return int (dt .isocalendar ().week )

def _clip_irf (v :float ,clip :float )->float :
    return max (min (v ,clip ),-clip )

def _canonical_freq (agg :str )->str :
    a =str (agg or "").upper ()
    if a in ("D",):
        return "D"
    if a in ("W","W-MON"):
        return "W-MON"
    if a in ("2W","2W-MON"):
        return "2W-MON"
    if a in ("M","MS"):
        return "MS"
    return "W-MON"

def _events_to_timeseries (ev_df :pd .DataFrame ,agg :str ="W-MON")->Tuple [pd .DatetimeIndex ,pd .DataFrame ]:
    if ev_df is None or ev_df .empty :
        return pd .DatetimeIndex ([]),pd .DataFrame ()

    freq =_canonical_freq (agg )
    dd =ev_df .copy ()
    dd ["date"]=pd .to_datetime (dd ["date"],errors ="coerce")
    dd =dd .dropna (subset =["date"]).reset_index (drop =True )
    dd ["aaod"]=dd ["aaod"].astype (str )
    dd ["date"]=dd ["date"].dt .floor ("D")
    by =dd .groupby ([pd .Grouper (key ="date",freq =freq ),"aaod"]).size ().reset_index (name ="cnt")
    if by .empty :
        first =dd ["date"].min ().floor (freq if freq !="2W-MON"else "W-MON")
        last =dd ["date"].max ().floor (freq if freq !="2W-MON"else "W-MON")
        all_t =pd .date_range (first ,last ,freq =freq )
        return all_t ,pd .DataFrame (index =all_t )
    first =by ["date"].min ()
    last =by ["date"].max ()
    all_t =pd .date_range (first ,last ,freq =freq )
    M_df =(by .pivot (index ="date",columns ="aaod",values ="cnt")
    .reindex (all_t ,fill_value =0.0 )
    .fillna (0.0 )
    .astype (float ))
    return all_t ,M_df 
def _seasonal_mu_aligned (all_t :pd .DatetimeIndex ,
cats :List [str ],
seasonal_alpha_gamma :Dict [str ,Dict [str ,Any ]])->np .ndarray :
    if len (all_t )==0 or not cats :
        return np .zeros ((len (all_t ),len (cats )),dtype =float )

    woy =pd .Index (all_t ).isocalendar ().week .to_numpy ()

    K =len (cats )
    MU =np .zeros ((len (all_t ),K ),dtype =float )
    idx ={c :j for j ,c in enumerate (cats )}
    for a ,j in idx .items ():
        meta =seasonal_alpha_gamma .get (a ,{})or {}
        alpha =float (meta .get ("alpha",0.0 ))
        gammas =meta .get ("gammas",{})or {}

        MU [:,j ]=np .exp (alpha +np .array ([float (gammas .get (int (w ),0.0 ))for w in woy ]))
    return MU 

def _build_multilag_design (M :np .ndarray ,MU :np .ndarray ,L :int )->Tuple [np .ndarray ,np .ndarray ,np .ndarray ]:
    T ,K =M .shape 

    if L <1 :
        raise ValueError ("L must be >=1")
    if T <=L :
        return np .zeros ((0 ,K *L )),np .zeros ((0 ,K )),np .zeros ((0 ,K ))

    Z_list =[]
    for lag in range (1 ,L +1 ):
        start_idx =L -lag 
        end_idx =T -lag 
        lag_matrix =M [start_idx :end_idx ,:]
        Z_list .append (lag_matrix )
    Z =np .concatenate (Z_list ,axis =1 )
    Y =M [L :,:]
    MU_all =MU [L :,:]
    return Z ,Y ,MU_all 


def _ridge_nnls (A :np .ndarray ,r :np .ndarray ,l2 :float =1e-3 ,nonneg :bool =True )->np .ndarray :
    if A .size ==0 :
        return np .zeros ((0 ,),dtype =float )

    lhs =A .T @A +l2 *np .eye (A .shape [1 ])
    rhs =A .T @r 

    try :
        b =np .linalg .solve (lhs ,rhs )
    except np .linalg .LinAlgError :
        b =np .linalg .lstsq (lhs ,rhs ,rcond =None )[0 ]
    if nonneg :
        b =np .maximum (b ,0.0 )
    return b 

def _fit_multilag_hawkes_glm (M :np .ndarray ,
MU :np .ndarray ,
L :int ,
rho_bounds :Tuple [float ,float ]=(0.6 ,0.98 ),
l2 :float =1e-3 ,
nonneg :bool =True ,
var_eps :float =1e-12 )->Tuple [float ,np .ndarray ]:
    T ,K =M .shape 
    if T ==0 or K ==0 or T <=L :
        return float (rho_bounds [0 ]),np .zeros ((L ,K ,K ),dtype =float )
    Z ,Y ,MU_all =_build_multilag_design (M ,MU ,L )
    if Z .shape [0 ]==0 :
        return float (rho_bounds [0 ]),np .zeros ((L ,K ,K ),dtype =float )
    var_mask =(Z .var (axis =0 )>var_eps )
    if not np .any (var_mask ):
        return float (rho_bounds [0 ]),np .zeros ((L ,K ,K ),dtype =float )
    rho_grid =np .linspace (1 ,1.001 ,1 )
    best_loss ,best_rho ,best_Blags =float ("inf"),rho_grid [0 ],None 
    eps =1e-6 
    for rho in rho_grid :
        B_full =np .zeros ((K ,Z .shape [1 ]),dtype =float )
        tot =0.0 
        A_all =rho *Z [:,var_mask ]
        for k in range (K ):
            r_k =np .log (Y [:,k ]+eps )-np .log (MU_all [:,k ]+eps )

            if A_all .size ==0 or np .allclose (A_all .var (axis =0 ),0.0 ):
                b_sub =np .zeros ((np .count_nonzero (var_mask ),),dtype =float )
            else :
                b_sub =_ridge_nnls (A_all ,r_k ,l2 =l2 ,nonneg =nonneg )
            b =np .zeros ((Z .shape [1 ],),dtype =float )
            b [var_mask ]=b_sub 
            B_full [k ,:]=b 
            lambda_rho =20.0 
            rho0 =0.7 
            pen =l2 *float (np .sum (b **2 ))
            resid =r_k -(rho *(Z @b ))
            tot +=float (np .mean (resid **2 ))
            xishu =(lambda_rho *(rho -rho0 )**2 )
            tot =tot +xishu 

        if tot <best_loss :
            best_loss ,best_rho ,best_Bfull =tot ,rho ,B_full 
    best_rho =1 
    B_lags =np .zeros ((L ,K ,K ),dtype =float )
    for ell in range (L ):
        B_lags [ell ,:,:]=best_Bfull [:,ell *K :(ell +1 )*K ]

    B_lags [np .abs (B_lags )<1e-8 ]=0.0 
    return float (best_rho ),B_lags 

def estimate_hawkes_glm_with_offset (
ev_df :pd .DataFrame ,
seasonal_alpha_gamma :Dict [str ,Dict [str ,Any ]],
l1 :float =0.0 ,
rho_bounds :Tuple [float ,float ]=(0.6 ,0.98 ),
agg :str ="W",
L :int =12 ,
)->Dict [str ,Any ]:
    ev =ev_df .copy ()
    ev ["date"]=pd .to_datetime (ev ["date"],errors ="coerce")
    ev =ev .dropna (subset =["date"]).reset_index (drop =True )

    ev ["aaod"]=ev ["aaod"].astype (str )
    ev =ev [ev ["aaod"].notna ()]
    if ev .empty :
        return {"rho":0.7 ,"B_lags":[],"L":L ,"columns":[],"link":"log+offset+L","agg":_canonical_freq (agg )}
    freq =_canonical_freq (agg )
    all_t ,M_df =_events_to_timeseries (ev ,agg =freq )
    cats =M_df .columns .tolist ()
    if len (all_t )==0 or not cats :
        return {"rho":0.7 ,"B_lags":[],"L":L ,"columns":[],"link":"log+offset+L","agg":freq }
    M =M_df .values 

    MU =_seasonal_mu_aligned (all_t ,cats ,seasonal_alpha_gamma )
    nz_rate =float ((M >0 ).mean ())
    print (f"[DBG] Hawkes TS: T={len (all_t )}, K={len (cats )}, nz_rate={nz_rate :.4f}, freq={freq }, L={L }")
    print ("[DBG] M_total:",int (M .sum ()))
    rho_star ,B_lags =_fit_multilag_hawkes_glm (M ,MU ,L =L ,rho_bounds =rho_bounds ,l2 =1e-3 ,nonneg =True )
    print ("[DBG] rho* =",rho_star )

    try :
        flat =B_lags .reshape (L ,-1 )
        top_idx =np .argsort (-flat .max (axis =0 ))[:min (10 ,flat .shape [1 ])]
        print ("[DBG] sample B entries (ell, k, j, val):")
        for idx_flat in top_idx :
            ell =idx_flat //(len (cats ))
            j =idx_flat %(len (cats ))

            k_star =int (np .argmax (B_lags [ell ,:,j ]))
            print ((ell +1 ,cats [k_star ],cats [j ],float (B_lags [ell ,k_star ,j ])))
    except Exception :
        pass 
    print (B_lags )
    return {
    "rho":float (rho_star ),
    "B_lags":B_lags .tolist (),
    "L":int (L ),
    "columns":cats ,
    "link":"log+offset+L",
    "agg":freq ,
    "l1":float (l1 ),
    }

def build_init_hist_from_events (ev_all :pd .DataFrame ,cols :List [str ],
start_date :pd .Timestamp ,L :int ,
agg :str ="W-MON")->List [np .ndarray ]:
    if L <=0 or not cols :
        return []
    dd =ev_all .copy ()
    dd ["date"]=pd .to_datetime (dd ["date"],errors ="coerce")
    dd =dd .dropna (subset =["date"])
    dd ["aaod"]=dd ["aaod"].astype (str )
    if agg .upper ().startswith ("W"):
        dd ["date"]=dd ["date"].dt .to_period ("W-MON").dt .start_time .dt .normalize ()
        step =pd .DateOffset (weeks =1 )
        start_tick =pd .to_datetime (start_date ).to_period ("W-MON").start_time .normalize ()
    else :
        dd ["date"]=dd ["date"].dt .normalize ()
        step =pd .DateOffset (days =1 )
        start_tick =pd .to_datetime (start_date ).normalize ()
    ticks =[start_tick -step *i for i in range (1 ,L +1 )]

    idx ={c :i for i ,c in enumerate (cols )}
    K =len (cols )
    grp =dd .groupby (["date","aaod"]).size ().reset_index (name ="cnt")
    by_date ={d :g for d ,g in grp .groupby ("date")}
    hist =[]
    for d in ticks :
        vec =np .zeros (K ,dtype =float )
        g =by_date .get (d )
        if g is not None :
            for _ ,row in g .iterrows ():
                j =idx .get (row ["aaod"])
                if j is not None :
                    vec [j ]+=float (row ["cnt"])

        hist .append (vec )

    return hist 





def handle_conflicting_events (arrivals ,rng ):
    all_events =list (arrivals .keys ())
    prefix_groups ={}
    for event_name in all_events :
        parts =event_name .split ('|')
        if len (parts )>=4 and parts [-1 ]in ['up','down']:
            prefix ='|'.join (parts [:-1 ])
            if prefix not in prefix_groups :
                prefix_groups [prefix ]=[]
            prefix_groups [prefix ].append (event_name )
    for prefix ,events in prefix_groups .items ():
        if len (events )<2 :
            continue 
        up_events =[e for e in events if e .endswith ('|up')]
        down_events =[e for e in events if e .endswith ('|down')]

        if up_events and down_events :

            for t in range (len (next (iter (arrivals .values ())))):
                up_total =sum (arrivals [up_event ][t ]for up_event in up_events )
                down_total =sum (arrivals [down_event ][t ]for down_event in down_events )

                if up_total >0 and down_total >0 :

                    if rng .random ()<0.7 :

                        for down_event in down_events :
                            arrivals [down_event ][t ]=0 
                    else :
                        for up_event in up_events :
                            arrivals [up_event ][t ]=0 


def soft_cap_exp (x ,start ,cap ,tau ):
    s =np .sign (x );a =abs (float (x ))
    if a <=start :
        return x 
    tau =max (float (tau ),1e-12 )
    y =start +(cap -start )*(1.0 -np .exp (-(a -start )/tau ))
    return s *y 

def synthesize_path_with_full_irf (
dates :pd .DatetimeIndex ,
base_price :float ,
ar_model :Dict [str ,Any ],
irf_dict :Dict [str ,pd .DataFrame ],
event_intensity :Dict [str ,Dict [str ,Any ]],
sample_events :bool =True ,
rng_seed :int =7 ,
clip :float =0.5 ,
drop_descriptive :bool =False ,
hawkes_params :Dict [str ,Any ]|None =None ,
hawkes_cap_mult :float =30.0 ,
lambda_scale :float =1.0 ,
eta :float =1.0 ,
init_last4 :list [float ]|None =None ,
init_prev :np .ndarray |List [float ]|None =None ,
init_hist :List [np .ndarray ]|None =None ,
event_randomness :float =1.0 ,
)->Tuple [pd .DataFrame ,pd .DataFrame ]:

    rng =np .random .default_rng (rng_seed )
    if isinstance (dates ,pd .Series ):
        dates =pd .to_datetime (dates ).reset_index (drop =True )
        dates =pd .DatetimeIndex (dates )
    elif isinstance (dates ,(pd .Index ,list ,np .ndarray )):
        dates =pd .DatetimeIndex (pd .to_datetime (dates ))
    else :
        dates =pd .DatetimeIndex (pd .to_datetime ([dates ]))
    n =len (dates )

    woys =[int (pd .Timestamp (d ).isocalendar ().week )for d in dates ]
    price =np .zeros (n ,dtype =float )
    deltas =np .zeros (n ,dtype =float )
    price [0 ]=base_price 
    last =list (init_last4 )
    def _keep_key (k :str )->bool :
        if not drop_descriptive :
            return True 
        lowers =k .lower ()
        return not ("price_change|"in lowers or "report_release|"in lowers )
    arrivals :Dict [str ,np .ndarray ]={k :np .zeros (n ,dtype =int )for k in irf_dict .keys ()}

    for k in event_intensity .keys ():
        if k not in arrivals :
            arrivals [k ]=np .zeros (n ,dtype =int )

    if hawkes_params :
        rho =float (hawkes_params .get ("rho",1.0 ))
        cols =hawkes_params .get ("columns",[])or []
        K =len (cols )
        if "B_lags"in hawkes_params and K >0 :
            from collections import deque 
            B_lags =np .array (hawkes_params .get ("B_lags",[]),dtype =float )

            if B_lags .ndim !=3 or B_lags .shape [1 ]!=K or B_lags .shape [2 ]!=K :
                B_lags =np .zeros ((0 ,K ,K ),dtype =float )

            L =int (hawkes_params .get ("L",B_lags .shape [0 ]))
            L =max (L ,0 )
            idx ={c :i for i ,c in enumerate (cols )}
            hist =deque ([np .zeros (K ,dtype =float )for _ in range (max (L ,1 ))],maxlen =max (L ,1 ))

            if init_hist is not None and L >0 :
                seed =[np .asarray (v ,dtype =float )for v in list (init_hist )[:L ]]
                hist .clear ()
                for v in reversed (seed ):
                    vv =np .zeros (K ,dtype =float )
                    vv [:min (K ,v .shape [0 ])]=v [:min (K ,v .shape [0 ])]
                    hist .append (vv )
                print (f"[DBG] Hawkes seeds injected: L={L }, K={K }")
                for ell in range (min (L ,3 )):
                    print (f"[DBG] seed lag {ell +1 } (most recent first) sum={hist [ell ].sum ():.3f}")

            alpha_values =[float (meta .get ("alpha",0.0 ))for meta in event_intensity .values ()]
            avg_alpha =sum (alpha_values )/len (alpha_values )if alpha_values else 0.0 

            uniform_base =math .exp (avg_alpha )*float (lambda_scale )

            for t in range (n ):
                for k ,meta in event_intensity .items ():

                    if not _keep_key (k ):
                        continue 
                    alpha =float (meta .get ("alpha",0.0 ))
                    gammas =meta .get ("gammas",{})or {}
                    base =math .exp (alpha +float (gammas .get (int (woys [t ]),0.0 )))*float (lambda_scale )
                    kidx =idx .get (k ,None )
                    if kidx is None or L ==0 or B_lags .size ==0 :
                        lam =base 
                    else :
                        zsum =0.0 
                        for ell in range (L ):
                            contrib =float (hist [ell ]@np .maximum (B_lags [ell ,kidx ,:],0.0 ))
                            zsum +=contrib 
                        cap =max (float (hawkes_cap_mult ),1.0 )
                        cap_log =math .log (cap )
                        arg =float (rho )*float (zsum )
                        arg =max (min (arg ,cap_log ),-cap_log )
                        mult =math .exp (arg )
                        hawkes_lam =base *min (mult ,cap )
                        lam =event_randomness *hawkes_lam +(1.0 -event_randomness )*uniform_base 
                        lam =float (max (lam ,1e-12 ))

                    arrivals [k ][t ]=rng .poisson (lam )
                cur_vec =np .array ([arrivals .get (c ,np .zeros (n ))[t ]for c in cols ],dtype =float )
                if L >0 :
                    hist .appendleft (cur_vec )


                if t ==0 and L >0 :
                    print ("[DBG] hist[0] after first tick (this is y_{t} to be used as next-step lag):",
                    cur_vec [:min (10 ,K )])
        else :
            B_raw =np .array (hawkes_params .get ("B",[]),dtype =float )

            cols =hawkes_params .get ("columns",[])or []
            K =len (cols )
            idx ={c :i for i ,c in enumerate (cols )}

            if B_raw .ndim ==2 :
                B_arr =B_raw 
            elif B_raw .ndim ==1 :
                if B_raw .size ==K *K :
                    B_arr =B_raw .reshape (K ,K )
                elif B_raw .size ==K :
                    B_arr =np .tile (B_raw .reshape (1 ,-1 ),(K ,1 ))
                else :
                    B_arr =np .zeros ((K ,K ))
            else :
                B_arr =np .zeros ((K ,K ))

            if B_arr .shape !=(K ,K ):
                B_pad =np .zeros ((K ,K ))
                rs =min (B_arr .shape [0 ],K )
                cs =min (B_arr .shape [1 ],K )
                B_pad [:rs ,:cs ]=B_arr [:rs ,:cs ]
                B_arr =B_pad 

            prev =np .array (init_prev ,dtype =float )if init_prev is not None else np .zeros (K )
            print ("prev:")
            print (prev )

            for t in range (n ):
                for k ,meta in event_intensity .items ():
                    if not _keep_key (k ):
                        continue 
                    alpha =float (meta ["alpha"])
                    gammas =meta .get ("gammas",{})or {}
                    base =math .exp (alpha +float (gammas .get (int (woys [t ]),0.0 )))*float (lambda_scale )
                    kidx =idx .get (k ,None )
                    if kidx is None :
                        lam =base 
                    else :
                        z =float (prev @np .maximum (B_arr [kidx ,:],0.0 ))
                        lam =min (base *math .exp (rho *z ),base *float (hawkes_cap_mult ))
                    arrivals [k ][t ]=rng .poisson (lam )
                prev =np .array ([arrivals .get (c ,np .zeros (n ))[t ]for c in cols ],dtype =float )
    else :
        print ("Seasonal poisson")
        for k ,meta in event_intensity .items ():
            if not _keep_key (k ):
                continue 
            alpha =float (meta ["alpha"])
            gammas =meta .get ("gammas",{})or {}
            lam =[math .exp (alpha +float (gammas .get (int (w ),0.0 )))*float (lambda_scale )for w in woys ]
            arrivals [k ]=rng .poisson (lam )

    handle_conflicting_events (arrivals ,rng )
    warmup_weeks =4 
    cap_dpt =0.20 
    target_at =0.30 
    mapped_to =0.075 

    start_abs =None 
    tau =None 
    Hmax =0 
    for k ,irf in irf_dict .items ():
        if irf is None or irf .empty :
            continue 
        Hmax =max (Hmax ,int (np .nanmax (irf ["h"].values )))
    tail ={k :np .zeros (Hmax +1 ,dtype =float )for k in irf_dict .keys ()}

    ev_rows =[]

    for t in range (n ):
        for k ,cnt in arrivals .items ():
            cnum =int (cnt [t ])
            if cnum <=0 :
                continue 
            irf_k =irf_dict .get (k )
            if irf_k is None or irf_k .empty :
                continue 
            vec =irf_k .sort_values ("h")["irf"].to_numpy ().astype (float )
            if clip is not None :
                vec =np .clip (vec ,-abs (float (clip )),abs (float (clip )))

            tail [k ]+=cnum *vec 
            ev_rows .append ({"date":dates [t ],"event_col":f"evt_{k }","impact":float (vec [0 ])*cnum })

        shock =0.0 
        for k in tail .keys ():
            shock +=float (tail [k ][0 ])
        dpt_raw =ar4_predict (last ,ar_model .get ("phi",[0 ,0 ,0 ,0 ]))+shock 
        if t <warmup_weeks :
            dpt =dpt_raw 
        else :
            if start_abs is None :

                base =np .quantile (np .abs (deltas [:warmup_weeks ]),0.90 )if warmup_weeks >0 else 0.0 
                start_abs =float (min (base ,2.00 *cap_dpt ))
                T =max (target_at ,start_abs +1e-6 )
                M =float (np .clip (mapped_to ,start_abs +1e-6 ,cap_dpt -1e-6 ))
                denom =1.0 -(M -start_abs )/(cap_dpt -start_abs )
                denom =max (denom ,1e-12 )
                tau =(T -start_abs )/(-np .log (denom ))
            dpt =soft_cap_exp (dpt_raw ,start =start_abs ,cap =cap_dpt ,tau =tau )

        deltas [t ]=dpt 
        price [t ]=price [t -1 ]+dpt if t >0 else base_price 
        last =[dpt ]+last [:3 ]

        if Hmax >0 :
            for k in tail .keys ():
                tail [k ][:-1 ]=tail [k ][1 :]
                tail [k ][-1 ]=0.0 

    price_df =pd .DataFrame ({"date":dates ,"us_price_synth":price ,"dprice_synth":deltas })
    events_df =pd .DataFrame (ev_rows )
    print (price_df )
    return price_df ,events_df 

def _glm_build_design_matrix (ev_mat :np .ndarray ,lags :int =1 )->np .ndarray :
    if lags <=0 :
        return np .zeros ((ev_mat .shape [0 ],0 ))
    Z =[]
    for t in range (ev_mat .shape [0 ]):
        zt =ev_mat [t -1 ,:]if t -1 >=0 else np .zeros (ev_mat .shape [1 ])
        Z .append (zt )
    return np .vstack (Z )

def _glm_fit_B_for_fixed_rho (M :np .ndarray ,MU :np .ndarray ,rho :float ,l1 :float =0.0 ,nonneg :bool =True )->np .ndarray :
    T ,K =M .shape 
    Z =M [:,:K ]
    w =np .ones (T )
    lhs =Z .T @(w [:,None ]*Z )+1e-6 *np .eye (K )
    rhs =Z .T @(w *(np .log (np .clip (MU ,1e-6 ,None ))-np .log (np .clip (MU ,1e-6 ,None ))))
    B =np .linalg .solve (lhs ,rhs )
    if nonneg :
        B =np .maximum (B ,0.0 )
    return B 


def synthesize_window_anchored (
dates_win :pd .DatetimeIndex ,
base_price_real :pd .Series ,
ar_model :Dict [str ,Any ],
irf_dict :Dict [str ,pd .DataFrame ],
event_intensity :Dict [str ,Dict [str ,Any ]],
rng_seed :int ,
clip :float ,
drop_descriptive :bool ,
hawkes_params :Dict [str ,Any ]|None ,
hawkes_cap_mult :float ,
lambda_scale :float ,
anchor_blend :float =0.7 ,
max_abs_weekly_change :float =0.25 ,
init_last4 :Optional [Sequence [float ]]=None ,
init_prev :List [float ]|np .ndarray |None =None ,
init_hist :List [np .ndarray ]|None =None ,
event_randomness :float =1.0 ,
):

    price_syn ,ev_syn =synthesize_path_with_full_irf (
    dates =dates_win ,
    base_price =float (base_price_real .iloc [0 ]),
    ar_model =ar_model ,
    irf_dict =irf_dict ,
    event_intensity =event_intensity ,
    sample_events =True ,
    rng_seed =rng_seed ,
    clip =clip ,
    drop_descriptive =drop_descriptive ,
    hawkes_params =hawkes_params if 'hawkes_params'in locals ()else None ,
    hawkes_cap_mult =hawkes_cap_mult if 'hawkes_cap_mult'in locals ()else 30.0 ,
    lambda_scale =lambda_scale ,
    eta =1.0 ,
    init_last4 =init_last4 ,
    init_prev =init_prev ,
    init_hist =init_hist ,
    event_randomness =event_randomness ,
    )
    df =price_syn .copy ()
    df =df .rename (columns ={"us_price_synth":"P","dprice_synth":"dP"})
    df ["dP"]=df ["dP"].clip (-abs (float (max_abs_weekly_change )),abs (float (max_abs_weekly_change )))

    P_synthetic =[float (df ["P"].iloc [0 ])]
    for i in range (1 ,len (df )):
        P_synthetic .append (P_synthetic [-1 ]+float (df .loc [i ,"dP"]))
    df ["P"]=P_synthetic 

    df =df [["date","P","dP"]]
    return df 

def _glm_fit_B_for_fixed_rho (Z ,Y ,MU ,rho :float ,l2 :float =1e-6 ,nonneg :bool =True ):

    eps =1e-6 
    T =min (len (Y ),len (MU ),Z .shape [0 ])
    Z =Z [:T ,:].astype (float )
    Y =Y [:T ].astype (float )
    MU =MU [:T ].astype (float )

    r =np .log (Y +eps )-np .log (MU +eps )
    A =rho *Z 
    lhs =A .T @A +l2 *np .eye (A .shape [1 ])
    rhs =A .T @r 
    b =np .linalg .solve (lhs ,rhs )
    if nonneg :
        b =np .maximum (b ,0.0 )
    return b 





def _format_context_block (df_ctx :pd .DataFrame )->str :
    lines =[]
    for _ ,r in df_ctx .iterrows ():
        lines .append (f"{r ['date'].date ()}  P={r ['P']:.3f}  dP={r ['dP']:.3f}")
    return "\n".join (lines )

def _format_target_block (df_tar :pd .DataFrame )->str :
    lines =[]
    for _ ,r in df_tar .iterrows ():
        lines .append (f"{r ['date'].date ()}  P={r ['P']:.3f}  dP={r ['dP']:.3f}")
    return "\n".join (lines )

def write_yearly_windows_jsonl (
energy_df :pd .DataFrame ,
ar_model :Dict [str ,Any ],
irf_dict :Dict [str ,pd .DataFrame ],
event_intensity :Dict [str ,Dict [str ,Any ]],
out_path :str ,
window_months :int =3 ,
context_weeks :int =12 ,
year_start :int |None =None ,
year_end :int |None =None ,
rng_seed :int =42 ,
clip :float =0.5 ,
drop_descriptive :bool =False ,
hawkes_params :Dict [str ,Any ]|None =None ,
hawkes_cap_mult :float =30.0 ,
lambda_scale :float =2.0 ,
anchor_blend :float =0.7 ,
max_abs_weekly_change :float =0.25 ,
event_randomness :float =1.0 ,
):

    import orjson 

    dirn =os .path .dirname (out_path )
    if dirn :
        os .makedirs (dirn ,exist_ok =True )

    df =energy_df .copy ().sort_values ("date").reset_index (drop =True )
    df ["date"]=pd .to_datetime (df ["date"],errors ="coerce")
    df =df .dropna (subset =["date"]).reset_index (drop =True )


    yrs =sorted (df ["date"].dt .year .unique ().tolist ())
    if year_start is not None :
        yrs =[y for y in yrs if y >=year_start ]
    if year_end is not None :
        yrs =[y for y in yrs if y <=year_end ]

    with open (out_path ,"wb")as f :
        for y in yrs :

            starts =pd .date_range (f"{y }-01-01",f"{y }-12-01",freq ="MS")
            for st in starts :
                ed =st +pd .DateOffset (months =window_months )-pd .DateOffset (days =1 )

                ctx_start =st -pd .DateOffset (weeks =context_weeks )
                df_ctx =df [(df ["date"]>=ctx_start )&(df ["date"]<st )][["date","us_price","dprice"]].copy ()
                if df_ctx .empty :
                    continue 
                df_ctx =df_ctx .rename (columns ={"us_price":"P","dprice":"dP"})


                df_tar_real =df [(df ["date"]>=st )&(df ["date"]<=ed )][["date","us_price","dprice"]].copy ()
                if df_tar_real .empty :
                    continue 
                df_tar =synthesize_window_anchored (
                dates_win =df_tar_real ["date"],
                base_price_real =df_tar_real ["us_price"],
                ar_model =ar_model ,irf_dict =irf_dict ,
                event_intensity =event_intensity ,
                rng_seed =rng_seed ,clip =clip ,drop_descriptive =drop_descriptive ,
                hawkes_params =hawkes_params ,hawkes_cap_mult =hawkes_cap_mult ,
                lambda_scale =lambda_scale ,anchor_blend =anchor_blend ,
                max_abs_weekly_change =max_abs_weekly_change ,
                event_randomness =event_randomness 
                )
                df_tar =df_tar .rename (columns ={"dP_synth":"dP"})
                ctx_block =_format_context_block (df_ctx )
                tar_block =_format_target_block (df_tar )

                system_msg ={
                "role":"system",
                "content":"You are a careful energy-market analyst. Always produce STRICT, VALID JSON. Do not include any extra commentary outside JSON."
                }
                user_msg ={
                "role":"user",
                "content":(
                "You are given weekly U.S. gasoline prices (OT):\n"
                f"- Context window (last {context_weeks } weeks, weekly OT):\n{ctx_block }\n"
                f"- Target window ({window_months } months, weekly OT):\n{tar_block }\n\n"
                "Task: Hypothesize up to 5 plausible REAL-TIME causes that occurred WITHIN THE CURRENT MONTH ONLY.\n"
                "Each cause must be an AAODT key string: \"NAME|ACTION|OBJECT|DIRECTION\".\n"
                "Return STRICT JSON ONLY with schema:\n"
                "{ \"hypotheses\": [{\"key\": \"NAME|ACTION|OBJECT|DIRECTION\"}]}\n\n"
                "Example format (mimic this exactly, including field names):\n"
                "{\n"
                "  \"hypotheses\": [\n"
                "    {\"key\": \"imf|approve|price|up\"}\n"
                "  ],\n"
                "  \"summary\": \"The IMF has approved a three-year credit for the Philippines, which includes adjustments to oil, gasoline, and electricity prices.\"\n"
                "}\n\n"
                "If fewer than 5 valid causes exist, output fewer (do not fabricate). "
                )
                }

                _ ,events_df =synthesize_path_with_full_irf (
                dates =df_tar_real ["date"],
                base_price =float (df_tar_real ["us_price"].iloc [0 ]),
                ar_model =ar_model ,irf_dict =irf_dict ,event_intensity =event_intensity ,
                sample_events =True ,rng_seed =rng_seed ,clip =clip ,drop_descriptive =drop_descriptive ,
                hawkes_params =hawkes_params ,hawkes_cap_mult =hawkes_cap_mult ,lambda_scale =lambda_scale ,event_randomness =event_randomness ,
                )

                last_month_start =ed -pd .DateOffset (months =1 )+pd .DateOffset (days =1 )
                last_month_end =ed 
                keys =[]
                if events_df is not None and not events_df .empty :
                    tmp =events_df .copy ()
                    tmp ["date"]=pd .to_datetime (tmp ["date"],errors ="coerce")
                    tmp =tmp [(tmp ["date"]>=last_month_start )&(tmp ["date"]<=last_month_end )]
                    seen =set ()
                    for _ ,rr in tmp .iterrows ():
                        k =str (rr .get ("event_col",""))
                        if k .startswith ("evt_"):
                            k =k [len ("evt_"):]
                        parts =k .split ("|")
                        if len (parts )>=4 :
                            key ="|".join (parts [:4 ])
                            if key not in seen :
                                keys .append (key );seen .add (key )
                        if len (keys )>=5 :
                            break 
                sft_assistant ={"hypotheses":[{"key":k ,"summary":""}for k in keys ],"summary":""}
                assistant_msg ={"role":"assistant","content":json .dumps (sft_assistant )}
                row ={"messages":[system_msg ,user_msg ,assistant_msg ]}
                f .write (orjson .dumps (row )+b"\n")

def _month_floor (dt :pd .Timestamp )->pd .Timestamp :
    return pd .Timestamp (dt .year ,dt .month ,1 )

def _month_add (dt :pd .Timestamp ,months :int )->pd .Timestamp :
    return (_month_floor (dt )+pd .DateOffset (months =months ))

def _format_price_lines (df :pd .DataFrame ,date_col ="date",price_col ="P")->str :
    lines =[]
    for _ ,r in df .iterrows ():
        lines .append (f"{pd .Timestamp (r [date_col ]).date ()}: P={float (r [price_col ]):.3f}")
    return "\n".join (lines )

def _format_diff_lines (df :pd .DataFrame ,date_col ="date",diff_col ="dP")->str :
    lines =[]
    for _ ,r in df .iterrows ():
        lines .append (f"{pd .Timestamp (r [date_col ]).date ()}: dP={float (r [diff_col ]):.3f}")
    return "\n".join (lines )

def _extract_gt_keywords_from_events (ev_df :pd .DataFrame ,start :pd .Timestamp ,end :pd .Timestamp ,topk :int =5 ):
    if ev_df is None or ev_df .empty :
        return []
    dd =ev_df .copy ()
    dd ["date"]=pd .to_datetime (dd ["date"],errors ="coerce")
    dd =dd [(dd ["date"]>=start )&(dd ["date"]<=end )]
    if dd .empty :
        return []
    keys =[]
    for _ ,r in dd .iterrows ():
        col =str (r .get ("event_col",""))
        if col .startswith ("evt_"):
            col =col [len ("evt_"):]
        parts =col .split ("|")
        if len (parts )>=4 :
            keys .append ("|".join (parts [:4 ]))
    out ,seen =[],set ()
    for k in keys :
        if k not in seen :
            seen .add (k );out .append (k )
        if len (out )>=topk :
            break 
    return out 

def write_sft_rl_jsonl (
energy_df :pd .DataFrame ,
ar_model :Dict [str ,Any ],
irf_dict :Dict [str ,pd .DataFrame ],
event_intensity :Dict [str ,Dict [str ,Any ]],
sft_out_path :str ,
rl_out_path :str ,
lookback_months :int =12 ,
gen_months :int =3 ,
monthly_stride :int =1 ,
rng_seed :int =42 ,
clip :float =0.5 ,
drop_descriptive :bool =False ,
hawkes_params :Dict [str ,Any ]|None =None ,
hawkes_cap_mult :float =30.0 ,
lambda_scale :float =2.0 ,
anchor_blend :float =0.7 ,
max_abs_weekly_change :float =0.25 ,
context_weeks :int =12 ,
cutoff_date :str |None =None ,
events_df :pd .DataFrame |None =None ,
events_agg :str ="W-MON",
event_randomness :float =1.0 ,
):
    import orjson 
    win_seed =0 
    df =energy_df [["date","us_price","dprice"]].copy ().sort_values ("date").reset_index (drop =True )
    df ["date"]=pd .to_datetime (df ["date"],errors ="coerce")
    df =df .dropna (subset =["date"]).reset_index (drop =True )

    first_date =df ["date"].min ().normalize ()
    last_date =df ["date"].max ().normalize ()

    os .makedirs (os .path .dirname (sft_out_path )or ".",exist_ok =True )
    os .makedirs (os .path .dirname (rl_out_path )or ".",exist_ok =True )
    f_sft =open (sft_out_path ,"wb")
    f_rl =open (rl_out_path ,"wb")
    cur =_month_floor (first_date )+pd .DateOffset (months =lookback_months )
    end_cap =_month_floor (last_date )-pd .DateOffset (months =gen_months -1 )
    if cutoff_date :
        cut_ts =pd .to_datetime (cutoff_date ).normalize ()

        end_cap_by_cut =_month_floor (cut_ts )-pd .DateOffset (months =gen_months -1 )

        end_cap =min (end_cap ,end_cap_by_cut )
    while cur <=end_cap :

        win_start =_month_floor (cur )

        win_seed =int (np .random .randint (100 )+rng_seed +int (win_start .strftime ("%Y%m")))
        win_end =_month_add (win_start ,gen_months )-pd .DateOffset (days =1 )

        ctx_for_last4 =df [df ["date"]<win_start ].tail (5 ).copy ()
        prev4 =ctx_for_last4 ["us_price"].diff ().dropna ().tail (4 ).tolist ()
        if len (prev4 )<4 :
            prev4 =[0.0 ]*(4 -len (prev4 ))+prev4 

        init_hist =None 
        if hawkes_params is not None and events_df is not None :
            cols_hp =hawkes_params .get ("columns",[])
            L_hp =int (hawkes_params .get ("L",0 ))
            if cols_hp and L_hp >0 :
                init_hist =build_init_hist_from_events (events_df ,cols_hp ,win_start ,L_hp ,agg =events_agg )
        tar_real =df [(df ["date"]>=win_start )&(df ["date"]<=win_end )].copy ()
        if tar_real .empty :
            cur =_month_add (cur ,monthly_stride )
            continue 

        syn =synthesize_window_anchored (
        dates_win =tar_real ["date"],
        base_price_real =tar_real ["us_price"],
        ar_model =ar_model ,
        irf_dict =irf_dict ,
        event_intensity =event_intensity ,
        rng_seed =win_seed ,
        clip =clip ,
        drop_descriptive =drop_descriptive ,
        hawkes_params =hawkes_params ,
        hawkes_cap_mult =hawkes_cap_mult ,
        lambda_scale =lambda_scale ,
        anchor_blend =anchor_blend ,
        max_abs_weekly_change =max_abs_weekly_change ,
        init_last4 =prev4 ,
        init_prev =None ,
        init_hist =init_hist ,
        event_randomness =event_randomness 
        )

        df_tar_print =syn [["date","P"]].copy ()
        df_tar_print ["dP"]=df_tar_print ["P"].diff ().fillna (0.0 )

        last_month_start =_month_add (win_start ,gen_months -1 )
        last_month_end =_month_add (win_start ,gen_months )-pd .DateOffset (days =1 )
        df_last_m =df_tar_print [(df_tar_print ["date"]>=last_month_start )&(df_tar_print ["date"]<=last_month_end )].copy ()

        ctx_start =last_month_start -pd .DateOffset (weeks =context_weeks )
        df_ctx =df [(df ["date"]>=ctx_start )&(df ["date"]<last_month_start )].copy ()
        df_ctx_print =df_ctx .rename (columns ={"us_price":"P","dprice":"dP"})[["date","P","dP"]]

        price_tmp ,ev_syn =synthesize_path_with_full_irf (
        dates =tar_real ["date"],
        base_price =float (tar_real ["us_price"].iloc [0 ]),
        ar_model =ar_model ,
        irf_dict =irf_dict ,
        event_intensity =event_intensity ,
        sample_events =True ,
        rng_seed =win_seed ,
        clip =clip ,
        drop_descriptive =drop_descriptive ,
        hawkes_params =hawkes_params ,
        hawkes_cap_mult =hawkes_cap_mult ,
        lambda_scale =lambda_scale ,
        eta =1.0 ,
        init_last4 =prev4 ,
        event_randomness =event_randomness 
        )
        gt_keys =_extract_gt_keywords_from_events (ev_df =ev_syn ,start =last_month_start ,end =last_month_end ,topk =5 )

        ctx_block =_format_price_lines (df_tar_print )

        df_tar_print =syn [["date","dP"]].rename (columns ={"dP_synth":"dP"})

        df_last_m =df_tar_print [(df_tar_print ["date"]>=last_month_start )&
        (df_tar_print ["date"]<=last_month_end )].copy ()


        lastm_dP =_format_diff_lines (df_last_m )

        system_msg ={
        "role":"system",
        "content":"You are a careful energy-market analyst. Always produce STRICT, VALID JSON. Do not include any extra commentary outside JSON."
        }

        user_template_sft =(
        "You are given weekly U.S. gasoline prices (OT):\n"
        f"- Context window (last 3 months, weekly OT):\n{ctx_block }\n"
        f"- Current month only (weekly dOT):\n{lastm_dP }\n\n"
        "Task: Hypothesize up to 5 plausible REAL-TIME causes that occurred WITHIN THE CURRENT MONTH ONLY.\n"
        "Each cause must be an AAODT key string: \"NAME|ACTION|OBJECT|DIRECTION\".\n"
        "Return STRICT JSON ONLY with schema:\n"
        "{ \"hypotheses\": [{\"key\": \"NAME|ACTION|OBJECT|DIRECTION\"}]}\n\n"
        "Example format (mimic this exactly, including field names):\n"
        "{\n"
        "  \"hypotheses\": [\n"
        "    {\"key\": \"imf|approve|price|up\"}\n"
        "  ],\n"
        "}\n\n"
        "If fewer than 5 valid causes exist, output fewer (do not fabricate). "
        )

        sft_hypos =[{"key":k }for k in gt_keys ]
        sft_assistant_obj ={"hypotheses":sft_hypos ,}
        sft_obj ={
        "messages":[
        system_msg ,
        {"role":"user","content":user_template_sft },
        {"role":"assistant","content":orjson .dumps (sft_assistant_obj ).decode ("utf-8")}
        ]
        }
        f_sft .write (orjson .dumps (sft_obj )+b"\n")
        cur =_month_add (cur ,monthly_stride )

    f_sft .close ()
    f_rl .close ()
def main ():
    os .chdir (os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--energy_csv",type =str ,default ="dataset/energy.csv")
    ap .add_argument ("--events_jsonl",type =str ,default ="eventsdata_en/events_flat_simple_filtered.jsonl")
    ap .add_argument ("--outdir",type =str ,default ="syndata_v7")
    ap .add_argument ("--irf_h",type =int ,default =8 )
    ap .add_argument ("--min_count",type =int ,default =1 ,help ="Minimum occurrences for a category to estimate IRF")
    ap .add_argument ("--max_categories",type =int ,default =30 ,help ="Limit categories for LP to keep runtime reasonable")
    ap .add_argument ("--spillover_mode",type =str ,choices =["none","heuristic","usa_only"],default ="none",
    help ="none: weight=1 for all; heuristic: soft weights; usa_only: keep only explicit U.S.")
    ap .add_argument ("--no_energy_filter",action ="store_true",
    help ="If set, do NOT restrict events to energy-related targets.")
    ap .add_argument ("--price_col",type =str ,default ="OT",
    help ="Which column to use as price. 'auto' uses regional mean, "
    "or falls back to 'OT' if regional mean fails. "
    "Otherwise pass an exact column name from your CSV.")
    ap .add_argument ("--drop_descriptive",action ="store_true",
    help ="Drop descriptive categories (price_change/report_release) from synthesis triggers.")
    ap .add_argument ("--irf_clip",type =float ,default =0.5 ,
    help ="Clip magnitude for each IRF shock draw to avoid explosions (default 0.5 in price-delta units).")
    ap .add_argument ("--arrivals_model",choices =["poisson","hawkes"],default ="hawkes",
    help ="poisson: seasonal Poisson only; hawkes: seasonal baseline + cross/self excitation (GLM+offset)")
    ap .add_argument ("--hawkes_l1",type =float ,default =0.01 ,help ="L1 regularization on B (sparsity), e.g., 0.01")
    ap .add_argument ("--hawkes_rho_bounds",type =str ,default ="0.6,0.98",
    help ="Continuous search range for rho, e.g., '0.6,0.98'")
    ap .add_argument ("--hawkes_cap_mult",type =float ,default =30.0 ,
    help ="Cap multiplier for exp(sum B z) to avoid explosion (effective with log link)")
    ap .add_argument ("--by_year_windows",action ="store_true",
    help ="Generate yearly window JSONL as an additional artifact (does not replace CSV outputs).")
    ap .add_argument ("--window_months",type =int ,default =3 ,
    help ="Number of months per target window (default: 3)")
    ap .add_argument ("--context_weeks",type =int ,default =12 ,
    help ="Number of context weeks shown in the prompt (default: 12)")
    ap .add_argument ("--year_start",type =int ,default =None ,
    help ="Earliest year (default: inferred from data)")
    ap .add_argument ("--year_end",type =int ,default =None ,
    help ="Latest year (default: inferred from data)")
    ap .add_argument ("--anchor_blend",type =float ,default =0.7 ,
    help ="Anchor blend: p_t = anchor*real_t + (1-anchor)*model_t (default: 0.7)")
    ap .add_argument ("--max_abs_weekly_change",type =float ,default =0.25 ,
    help ="Clip absolute weekly price change within window ($/gal, default: 0.25)")
    ap .add_argument ("--jsonl_out",type =str ,default =None ,
    help ="Write messages to this JSONL file; if None, write to outdir/prompts_yearly.jsonl")
    ap .add_argument ("--make_sft_rl",action ="store_true",
    help ="Export two JSONL files for SFT and RL using monthly sliding windows (strict format).")
    ap .add_argument ("--lookback_months",type =int ,default =12 ,
    help ="Lookback window in months (default: 12)")
    ap .add_argument ("--gen_months",type =int ,default =3 ,
    help ="Generation window in months (default: 3)")
    ap .add_argument ("--monthly_stride",type =int ,default =1 ,
    help ="Monthly stride in months (default: 1)")
    ap .add_argument ("--sft_out",type =str ,default =None ,
    help ="SFT dataset JSONL output path (default: outdir/sft_windows.jsonl)")
    ap .add_argument ("--rl_out",type =str ,default =None ,
    help ="RL dataset JSONL output path (default: outdir/rl_windows.jsonl)")
    ap .add_argument ("--train_cutoff_date",type =str ,default ="2022-12-31",
    help ="Only generate monthly-window synthetic data on/before this date to avoid test contamination. Format: YYYY-MM-DD.")
    ap .add_argument ("--fit_cutoff_date",type =str ,default ="2022-12-31",
    help ="Fit arrival model / select categories using data on/before this date to avoid leakage. Format: YYYY-MM-DD.")
    ap .add_argument ("--lambda_scale",type =float ,default =1.0 ,
    help ="Event density scaling for Poisson sampling")
    ap .add_argument ("--event_randomness",type =float ,default =1.0 ,
    help ="Randomness strength for event generation (0=random, 1=no change)")
    ap .add_argument ("--rng_seed",type =int ,default =42 ,
    help ="Random seed")
    ap .add_argument ("--hawkes_lags",type =int ,default =12 ,
    help ="Multi-lag Hawkes GLM lag order (e.g., 12 weeks).")
    ap .add_argument ("--hawkes_agg",type =str ,default ="W",
    help ="Aggregation bucket for Hawkes fitting. One of: D, W, 2W, M.")
    args =ap .parse_args ()
    outdir =os .path .abspath (args .outdir )
    os .makedirs (outdir ,exist_ok =True )
    energy =robust_read_csv (args .energy_csv )
    if energy is None or energy .empty :
        raise SystemExit ("Energy CSV empty or unreadable. Check path/encoding.")
    date_col =choose_date_column (energy )
    energy [date_col ]=pd .to_datetime (energy [date_col ],errors ="coerce")
    energy =energy .dropna (subset =[date_col ]).sort_values (date_col ).reset_index (drop =True )
    energy =energy .rename (columns ={date_col :"date"})
    def _lc (s ):return s .lower ()if isinstance (s ,str )else ""
    region_cols =[c for c in energy .columns 
    if ("weekly"in _lc (c )and "all grades"in _lc (c )
    and "gasoline prices"in _lc (c ))]
    price_series =None 
    if getattr (args ,"price_col",None )and args .price_col !="auto":

        if args .price_col not in energy .columns :
            preview_cols =", ".join (list (energy .columns )[:12 ])
            raise SystemExit (f"price_col='{args .price_col }' not found. Try one of: {preview_cols }")
        price_series =coerce_numeric_series (energy [args .price_col ])

    energy ["us_price"]=price_series 
    energy ["dprice"]=energy ["us_price"].astype (float ).diff ().fillna (0.0 )
    ev_list =robust_read_jsonl (args .events_jsonl )
    rows =[]
    for r in ev_list :
        d =pd .to_datetime (r .get ("date"),errors ="coerce")
        k =extract_aaod_key_from_record (r )
        k =normalize_aaod (k )
        rows .append ({"date":d ,"aaod":k })
    ev =pd .DataFrame (rows ).dropna (subset =["date"]).reset_index (drop =True )
    fit_cut =pd .to_datetime (args .fit_cutoff_date ).normalize ()
    ev_fit =ev [ev ["date"]<=fit_cut ].copy ()
    energy_fit =energy [energy ["date"]<=fit_cut ].copy ()
    def _spill_weight (row ):
        if args .spillover_mode =="none":
            return 1.0 
        if args .spillover_mode =="usa_only":

            return 1.0 if (isinstance (row ["aaod"],str )and row ["aaod"].startswith ("united_states|"))else 0.0 

        return 1.0 if (isinstance (row ["aaod"],str )and row ["aaod"].startswith ("united_states|"))else 0.5 

    ev ["spillover_w"]=ev .apply (_spill_weight ,axis =1 )
    df =energy .merge (ev ,on ="date",how ="left")

    if df .empty :
        raise SystemExit (
        "Aligned dataframe is empty after merge. "
        "Please check: (1) energy dates parsed ok; (2) energy 'us_price' not all NaN; "
        "(3) the merge is left-join, so this usually means ENERGY was empty."
        )
    Xev =pd .get_dummies (ev .set_index ("date")["aaod"],prefix ="evt").groupby (level =0 ).sum ()

    df =energy .merge (Xev ,left_on ="date",right_index =True ,how ="left").fillna (0.0 )
    evt_cols =[c for c in df .columns if c .startswith ("evt_")]
    freq =ev_fit ["aaod"].value_counts ().reset_index ()
    freq .columns =["aaod","count"]
    cand_all =[f"evt_{k }"for k in freq .query ("count >= @args.min_count")["aaod"].tolist ()]
    cand =[c for c in cand_all if c in set (evt_cols )][:args .max_categories ]
    event_cols =cand 
    df_train =energy_fit [["date","dprice"]].copy ()
    for c in event_cols :

        if c in df .columns :
            df_train [c ]=df [c ].astype (float )
        else :
            df_train [c ]=0.0 

    lp_out =local_projection_irf_multi_lp (
    df =df_train ,
    event_cols =event_cols ,
    H =int (args .irf_h ),
    add_week_dummies =True ,
    nw_lags =4 ,
    min_pos =0 
    )

    irf_dict =build_irf_dict_from_lp (
    lp_results =lp_out ,
    remove_prefix ="evt_",
    smooth_k =3 ,
    shrink_small_t =1.0 
    )

    ar_model =fit_ar4_price (energy ["dprice"].values .astype (float ))
    ar_model_price =fit_ar4_price (energy ["dprice"].values .astype (float ))

    ev_eff =ev_fit .copy ()
    ev_eff ["aaod"]=ev_eff ["aaod"].astype (str )
    ev_eff =ev_eff [ev_eff ["aaod"].notna ()]
    season_intensity =build_seasonal_poisson_intensity (ev_eff )
    hawkes_params =None 
    hawkes_params =estimate_hawkes_glm_with_offset (
    ev_df =ev_eff .rename (columns ={"aaod":"aaod"}),
    seasonal_alpha_gamma =season_intensity ,
    l1 =float (args .hawkes_l1 ),
    rho_bounds =tuple (float (x )for x in str (args .hawkes_rho_bounds ).split (",")),
    agg =str (args .hawkes_agg ),
    L =int (args .hawkes_lags ),
    )

    energy .to_csv (os .path .join (outdir ,"energy_parsed.csv"),index =False )
    pd .DataFrame ({"key":list (irf_dict .keys ())}).to_csv (os .path .join (outdir ,"irf_keys.csv"),index =False )

    if 1 >=0 :
        try :
            from pathlib import Path as _P 
            sft_path =args .sft_out or str (_P (outdir )/"sft_windows.jsonl")
            rl_path =args .rl_out or str (_P (outdir )/"rl_windows.jsonl")
        except Exception :
            sft_path =args .sft_out or os .path .join (outdir ,"sft_windows.jsonl")
            rl_path =args .rl_out or os .path .join (outdir ,"rl_windows.jsonl")

        hawkes_params_local =None 
        if args .arrivals_model =="hawkes":
            hp_file =os .path .join (outdir ,"hawkes_params.json")
            if os .path .exists (hp_file ):
                try :
                    with open (hp_file ,"r",encoding ="utf-8")as f :
                        hawkes_params_local =json .load (f )
                except Exception :
                    hawkes_params_local =None 
        hp_effective =hawkes_params if (args .arrivals_model =="hawkes")else None 
        print (hp_effective )
        write_sft_rl_jsonl (
        energy_df =energy ,
        ar_model =ar_model ,
        irf_dict =irf_dict ,
        event_intensity =season_intensity ,
        sft_out_path =sft_path ,
        rl_out_path =rl_path ,
        lookback_months =int (args .lookback_months ),
        gen_months =int (args .gen_months ),
        monthly_stride =int (args .monthly_stride ),
        rng_seed =args .rng_seed ,
        clip =args .irf_clip ,
        drop_descriptive =args .drop_descriptive ,
        hawkes_params =hp_effective ,
        hawkes_cap_mult =getattr (args ,"hawkes_cap_mult",30.0 ),
        lambda_scale =getattr (args ,"lambda_scale",2.0 ),
        anchor_blend =float (args .anchor_blend ),
        max_abs_weekly_change =float (args .max_abs_weekly_change ),
        context_weeks =args .context_weeks ,
        cutoff_date =args .train_cutoff_date ,
        events_df =ev ,
        events_agg ="W-MON",
        event_randomness =args .event_randomness 
        )
        print ("[INFO] SFT/RL monthly-sliding JSONL written to:",sft_path ,"and",rl_path )
    if args .arrivals_model =="hawkes"and hawkes_params and hawkes_params .get ("B_lags"):
        with open (os .path .join (outdir ,"hawkes_params.json"),"w",encoding ="utf-8")as f :
            json .dump (hawkes_params ,f ,ensure_ascii =False ,indent =2 )

    print ("[DONE] Outputs saved to",str (outdir ))

if __name__ =="__main__":
    main ()