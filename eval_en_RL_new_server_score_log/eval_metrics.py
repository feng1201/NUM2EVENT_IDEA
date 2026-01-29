from __future__ import annotations
from typing import List ,Tuple ,Dict
import math
from aaodt_schema import AAOD

def aaodt_similarity (pred :AAOD ,gold :AAOD )->float :
    if (pred .name ==gold .name and pred .action ==gold .action and
    pred .obj ==gold .obj and pred .direction ==gold .direction ):
        return 1.0
    if (pred .action ==gold .action and pred .obj ==gold .obj and pred .name ==gold .name ):
        return 0.70
    if (pred .obj ==gold .obj and pred .direction ==gold .direction ):
        return 0.50
    return 0

def greedy_match_scores (preds :List [AAOD ],golds :List [AAOD ])->List [float ]:
    used =set ()
    scores :List [float ]=[]
    for p in preds :
        best =0.0
        best_j =None
        for j ,g in enumerate (golds ):
            if j in used :
                continue
            s =aaodt_similarity (p ,g )
            if s >best :
                best =s
                best_j =j
        if best_j is not None :
            used .add (best_j )
        scores .append (best )
    return scores
def soft_prf1_gold_covered (
preds :List [AAOD ],
golds :List [AAOD ],
thr :float =0.6 ,
)->Tuple [float ,float ,float ]:
    if not preds or not golds :
        return 0.0 ,0.0 ,0.0
    sim_mat :List [List [float ]]=[]
    for p in preds :
        row =[aaodt_similarity (p ,g )for g in golds ]
        sim_mat .append (row )
    tp_pred =0
    for row in sim_mat :
        if row and max (row )>=thr :
            tp_pred +=1
    fp =len (preds )-tp_pred
    tp_gold =0
    for j in range (len (golds )):
        col_max =0.0
        for i in range (len (preds )):
            col_max =max (col_max ,sim_mat [i ][j ])
        if col_max >=thr :
            tp_gold +=1
    fn =len (golds )-tp_gold

    prec =tp_pred /max (1 ,tp_pred +fp )
    rec =tp_gold /max (1 ,tp_gold +fn )
    f1 =0.0 if (prec +rec ==0 )else 2 *prec *rec /(prec +rec )

    return prec ,rec ,f1

def soft_prf1 (preds :List [AAOD ],golds :List [AAOD ],thr :float =0.6 )->Tuple [float ,float ,float ]:
    scores =greedy_match_scores (preds ,golds )
    tp =sum (1 for s in scores if s >=thr )
    fp =len (preds )-tp
    fn =max (0 ,len (golds )-tp )
    prec =tp /max (1 ,tp +fp )
    rec =tp /max (1 ,tp +fn )
    f1 =0.0 if (prec +rec ==0 )else 2 *prec *rec /(prec +rec )
    return prec ,rec ,f1

def ndcg_at_k (preds :List [AAOD ],golds :List [AAOD ],k :int =5 )->float :
    dcg =0.0
    for i ,p in enumerate (preds [:k ],start =1 ):
        gain =max ((aaodt_similarity (p ,g )for g in golds ),default =0.0 )
        dcg +=gain /math .log2 (i +1 )

    ideal =0.0
    for i in range (1 ,min (k ,len (golds ))+1 ):
        ideal +=1.0 /math .log2 (i +1 )
    return (dcg /ideal )if ideal >0 else 0.0
def parsimony_penalty (num_preds :int ,k_max :int =5 )->float :

    if num_preds <=k_max :
        return 1.0
    over =num_preds -k_max
    return max (0.5 ,1.0 -0.10 *over )
def evaluate_rte (
preds :List [AAOD ],
gold_primary :List [AAOD ],
gold_all :List [AAOD ],
k_pred_max :int =20 ,
thr :float =0.1 ,
)->Dict [str ,float ]:
    prec_all ,_ ,_ =soft_prf1_gold_covered (preds ,gold_all ,thr =thr )
    _ ,rec_p ,f1_p =soft_prf1_gold_covered (preds [:k_pred_max ],gold_primary ,thr =thr )
    ndcg_p =ndcg_at_k (preds [:k_pred_max ],gold_primary ,k =k_pred_max )
    pars =parsimony_penalty (len (preds ),k_max =k_pred_max )
    overall =0.45 *f1_p +0.20 *prec_all
    overall *=pars

    return {
    "precision_all":prec_all ,
    "recall_primary@5":rec_p ,
    "f1_primary@5":f1_p ,
    "ndcg_primary@5":ndcg_p ,
    "parsimony":pars ,
    "overall":overall ,
    }


