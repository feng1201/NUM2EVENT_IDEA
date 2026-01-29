from __future__ import annotations
import os
import json
from typing import List ,Tuple
from safetensors .torch import load_file
import torch
import torch .nn as nn
from transformers import AutoModelForCausalLM ,PreTrainedTokenizerBase

MARK_TS3M ="<ts3m><ts3m/>"
MARK_TSD ="<tsdot><tsdot/>"

def load_ts_encoder_from_dir (model ,ckpt_dir :str ,strict :bool =True ):

    cfg_path =os .path .join (ckpt_dir ,"ts_config.json")
    w_path =os .path .join (ckpt_dir ,"ts_encoder.safetensors")
    if not os .path .exists (cfg_path )or not os .path .exists (w_path ):
        raise FileNotFoundError (f"Missing ts files in {ckpt_dir }: ts_config.json / ts_encoder.safetensors")

    with open (cfg_path ,"r",encoding ="utf-8")as f :
        cfg =json .load (f )


    cur ={
    "d_model":getattr (model .tsenc ,"d_model",None ),
    "patch_size_ot":getattr (model .tsenc ,"patch_size_ot",None ),
    "patch_size_dot":getattr (model .tsenc ,"patch_size_dot",None ),
    "hidden":getattr (model .tsenc ,"hidden",None ),
    "layers":getattr (model .tsenc ,"layers",None ),
    "concat_posidx":getattr (model .tsenc ,"concat_posidx",False ),
    }
    if cfg !=cur :
        print ("[TS][WARN] ts_config mismatched. saved:",cfg ," current:",cur )

    state =load_file (w_path ,device ="cpu")
    missing ,unexpected =model .tsenc .load_state_dict (state ,strict =strict )
    if not strict :
        print ("[TS] load_state_dict(strict=False) missing:",missing ,"unexpected:",unexpected )
    print (f"[TS] loaded from {w_path }")


def _pack_ts_batch (ts_list ,device ):


    vals =[torch .tensor (x ["vals"],device =device ,dtype =torch .bfloat16 )for x in ts_list ]
    msks =[torch .tensor (x ["mask"],device =device ,dtype =torch .bfloat16 )for x in ts_list ]

    L =max (v .numel ()for v in vals )if vals else 1

    vx ,mx =[],[]

    for v ,m in zip (vals ,msks ):
        if v .numel ()<L :

            v =torch .cat ([v ,v [-1 :].repeat (L -v .numel ())])
            m =torch .cat ([m ,m [-1 :].repeat (L -m .numel ())])

        vx .append (v .unsqueeze (-1 ))
        mx .append (m .unsqueeze (-1 ))


    x =torch .stack ([torch .cat ([v ,m ],dim =-1 )for v ,m in zip (vx ,mx )],0 )
    return x .to (device )


class QwenWithTSEmbed (nn .Module ):


    def __init__ (
    self ,
    qwen :AutoModelForCausalLM ,
    ts_encoder :nn .Module ,
    tokenizer :PreTrainedTokenizerBase ,
    ctx_cap :int |None =None ,
    ):
        super ().__init__ ()
        self .qwen =qwen
        self .tsenc =ts_encoder
        self .tok =tokenizer
        self .hidden =qwen .config .hidden_size

        self .ctx_cap =ctx_cap

        dev =self ._emb_weight ().device
        dtype =self ._emb_weight ().dtype
        self .tsenc .to (device =dev ,dtype =dtype )
        self .config =qwen .config
        if hasattr (qwen ,"generation_config"):
            self .generation_config =qwen .generation_config
        self .warnings_issued =getattr (qwen ,"warnings_issued",{})
        setattr (qwen ,"warnings_issued",self .warnings_issued )
        if hasattr (qwen ,"add_model_tags"):
            self .add_model_tags =qwen .add_model_tags
        else :
            def _noop_add_model_tags (*args ,**kwargs ):
                return None
            self .add_model_tags =_noop_add_model_tags

        self .warnings_issued =getattr (qwen ,"warnings_issued",{})
        setattr (qwen ,"warnings_issued",self .warnings_issued )

        self ._debug_batches =int (os .getenv ("TS_DEBUG_IO","0"))
        self ._debug_seen =0
        self ._debug_rank0_only =True


        if hasattr (self .qwen .config ,"use_cache"):
            self .qwen .config .use_cache =False

    @staticmethod
    def _split2 (text :str )->Tuple [str ,str ,str ]:

        if MARK_TS3M not in text :
            raise ValueError (f"'{MARK_TS3M }' not found in prompt.")
        if MARK_TSD not in text :
            raise ValueError (f"'{MARK_TSD }' not found in prompt.")

        left ,rest =text .split (MARK_TS3M ,1 )

        mid ,right =rest .split (MARK_TSD ,1 )
        return left ,mid ,right

    @property
    def is_gradient_checkpointing (self )->bool :

        if hasattr (self .qwen ,"is_gradient_checkpointing"):
            return bool (getattr (self .qwen ,"is_gradient_checkpointing"))
        return bool (getattr (self ,"_is_gradient_checkpointing",False ))

    @is_gradient_checkpointing .setter
    def is_gradient_checkpointing (self ,value :bool ):
        if hasattr (self .qwen ,"is_gradient_checkpointing"):
            setattr (self .qwen ,"is_gradient_checkpointing",bool (value ))

        self ._is_gradient_checkpointing =bool (value )

    def enable_input_require_grads (self ):


        if hasattr (self .qwen ,"enable_input_require_grads"):
            return self .qwen .enable_input_require_grads ()


        emb =self .qwen .get_input_embeddings ()

        def _make_inputs_require_grad (module ,inp ,out ):

            if isinstance (out ,torch .Tensor ):
                out .requires_grad_ (True )
            elif isinstance (out ,(list ,tuple )):
                for x in out :
                    if isinstance (x ,torch .Tensor ):
                        x .requires_grad_ (True )


        if not hasattr (self ,"_input_grad_hook"):
            self ._input_grad_hook =emb .register_forward_hook (_make_inputs_require_grad )


    def _emb_module (self )->nn .Module :

        return self .qwen .get_input_embeddings ()

    def _emb_weight (self )->torch .Tensor :

        return self ._emb_module ().weight

    def _ids2emb (self ,ids :torch .Tensor )->torch .Tensor :

        return self ._emb_module ()(ids )


    def gradient_checkpointing_enable (self ,**kwargs ):

        if hasattr (self .qwen ,"gradient_checkpointing_enable"):
            return self .qwen .gradient_checkpointing_enable (**kwargs )

    def gradient_checkpointing_disable (self ):

        if hasattr (self .qwen ,"gradient_checkpointing_disable"):
            return self .qwen .gradient_checkpointing_disable ()

    def get_input_embeddings (self ):

        return self .qwen .get_input_embeddings ()

    def set_input_embeddings (self ,new_emb ):

        self .qwen .set_input_embeddings (new_emb )


    def save_pretrained (self ,save_directory :str ,**kwargs ):

        os .makedirs (save_directory ,exist_ok =True )

        self .qwen .save_pretrained (save_directory ,**kwargs )

        torch .save (self .tsenc .state_dict (),os .path .join (save_directory ,"ts_encoder.pt"))

        ts_cfg ={
        "patch_size_ot":getattr (self .tsenc .enc_ot ,"patch_size",8 ),
        "patch_size_dot":getattr (self .tsenc .enc_dot ,"patch_size",4 ),
        "hidden":getattr (self .tsenc ,"hidden",None )
        or (self .tsenc .enc_ot .mlp [0 ].out_features if hasattr (self .tsenc .enc_ot .mlp [0 ],"out_features")else self .hidden ),
        "layers":sum (1 for m in self .tsenc .enc_ot .mlp if isinstance (m ,nn .Linear )),
        }

        with open (os .path .join (save_directory ,"ts_encoder_config.json"),"w",encoding ="utf-8")as f :
            json .dump (ts_cfg ,f ,ensure_ascii =False ,indent =2 )


    def forward (
    self ,
    prompt :List [str ],
    answer :List [str ],
    ts3m :List [dict ],
    tsdot :List [dict ],
    return_dict :bool =True ,
    input_ids =None ,
    attention_mask =None ,
    use_cache =None ,
    ):

        rank_env =os .getenv ("RANK")or os .getenv ("LOCAL_RANK")or "0"
        try :
            rank =int (rank_env )
        except Exception :
            rank =0
        allow_print =(self ._debug_batches >0 )and (not self ._debug_rank0_only or rank ==0 )
        w =self ._emb_weight ()
        device ,dtype =w .device ,w .dtype
        chunks =[]
        for p in prompt :
            left ,mid ,right =self ._split2 (p )

            right =self ._ensure_assistant_prefix (right )
            ids_left =self .tok (left ,add_special_tokens =False ,return_tensors ="pt").input_ids .to (device )
            ids_mid =self .tok (mid ,add_special_tokens =False ,return_tensors ="pt").input_ids .to (device )
            ids_right =self .tok (right ,add_special_tokens =False ,return_tensors ="pt").input_ids .to (device )

            e_left ,e_mid ,e_right =self ._ids2emb (ids_left ),self ._ids2emb (ids_mid ),self ._ids2emb (ids_right )

            if e_left .dim ()==2 :e_left =e_left .unsqueeze (0 )
            if e_mid .dim ()==2 :e_mid =e_mid .unsqueeze (0 )
            if e_right .dim ()==2 :e_right =e_right .unsqueeze (0 )
            chunks .append ((e_left ,e_mid ,e_right ))

        x3m =_pack_ts_batch (ts3m ,device )
        xdot =_pack_ts_batch (tsdot ,device )
        z3m ,zdot =self .tsenc (x3m ,xdot )

        if allow_print and self ._debug_seen <self ._debug_batches :
            B =len (prompt )
            print (f"[TS-DEBUG] batch={self ._debug_seen } B={B }")
            for i ,p in enumerate (prompt ):
                left ,mid ,right =self ._split2 (p )
                def short (s ,n =120 ):
                    return (s [:n ]+("…"if len (s )>n else ""))
                print (f"  ├─sample#{i }")
                print (f"  │  left:  {short (left )}")
                print (f"  │  mid:   {short (mid )}")
                print (f"  │  right: {short (right )}")


            ids_left0 =self .tok (self ._split2 (prompt [0 ])[0 ],add_special_tokens =False ,return_tensors ="pt").input_ids
            ids_mid0 =self .tok (self ._split2 (prompt [0 ])[1 ],add_special_tokens =False ,return_tensors ="pt").input_ids
            ids_right0 =self .tok (self ._split2 (prompt [0 ])[2 ],add_special_tokens =False ,return_tensors ="pt").input_ids
            print (f"  │  text-token lens (sample#0): left={ids_left0 .size (1 )} mid={ids_mid0 .size (1 )} right={ids_right0 .size (1 )}")
            print (f"  │  ts-pseudo-token lens (all): L3m={z3m .size (1 )} Ldot={zdot .size (1 )}  (H={z3m .size (-1 )})")

            ans_ids0 =self .tok (answer [0 ],add_special_tokens =False ,return_tensors ='pt').input_ids
            print (f"  │  answer tokens (sample#0): {ans_ids0 .size (1 )}  text='{short (answer [0 ],120 )}'")

            self ._debug_seen +=1

        z3m =z3m .to (device =device ,dtype =dtype )
        zdot =zdot .to (device =device ,dtype =dtype )
        rows ,masks ,labs =[],[],[]
        max_len =0
        for i ,(e_left ,e_mid ,e_right )in enumerate (chunks ):

            z3m_i =z3m [i ].unsqueeze (0 )
            zdot_i =zdot [i ].unsqueeze (0 )
            row =torch .cat ([e_left ,z3m_i ,e_mid ,zdot_i ,e_right ],dim =1 )
            ans_ids =self .tok (answer [i ],add_special_tokens =False ,return_tensors ="pt").input_ids .to (device )
            e_ans =self ._ids2emb (ans_ids )
            if e_ans .dim ()==2 :
                e_ans =e_ans .unsqueeze (0 )
            row =torch .cat ([row ,e_ans ],dim =1 )
            L_prompt_i =e_left .size (1 )+z3m_i .size (1 )+e_mid .size (1 )+zdot_i .size (1 )+e_right .size (1 )
            mask_i =torch .ones ((1 ,row .size (1 )),dtype =torch .long ,device =device )
            lab_i =torch .full ((L_prompt_i ,),-100 ,dtype =torch .long ,device =device )
            lab_i =torch .cat ([lab_i ,ans_ids [0 ]],dim =0 )
            rows .append (row );masks .append (mask_i );labs .append (lab_i )
            max_len =max (max_len ,row .size (1 ))
        H =rows [0 ].size (-1 )
        padded_rows ,padded_masks ,padded_labs =[],[],[]
        for row ,mask_i ,lab_i in zip (rows ,masks ,labs ):
            pad =max_len -row .size (1 )
            if pad >0 :
                row =torch .cat ([row ,torch .zeros ((1 ,pad ,H ),dtype =dtype ,device =device )],dim =1 )
                mask_i =torch .cat ([mask_i ,torch .zeros ((1 ,pad ),dtype =torch .long ,device =device )],dim =1 )
                lab_i =torch .cat ([lab_i ,torch .full ((pad ,),-100 ,dtype =torch .long ,device =device )],dim =0 )
            padded_rows .append (row );padded_masks .append (mask_i );padded_labs .append (lab_i )

        inputs_embeds =torch .cat (padded_rows ,dim =0 )
        attention_mask =torch .cat (padded_masks ,dim =0 )
        labels =torch .stack (padded_labs ,dim =0 )
        cap =self .ctx_cap
        if cap is None :
            cap =getattr (self .qwen .config ,"max_position_embeddings",inputs_embeds .size (1 ))

        cap =min (cap ,inputs_embeds .size (1 ))
        if cap <inputs_embeds .size (1 ):

            inputs_embeds =inputs_embeds [:,-cap :,:]
            attention_mask =attention_mask [:,-cap :]
            labels =labels [:,-cap :]
        out =self .qwen (
        inputs_embeds =inputs_embeds ,
        attention_mask =attention_mask ,
        labels =labels ,
        return_dict =True ,
        )

        if self .training :
            try :

                rank_env =os .getenv ("RANK")or os .getenv ("LOCAL_RANK")or "0"
                rank =int (rank_env )if rank_env is not None else 0
            except Exception :
                rank =0
            allow_rank =(rank ==0 )
            if allow_rank and hasattr (out ,"logits"):
                i =0
                lab_i =labels [i ]
                mask =(lab_i !=-100 )

                pred_ids =out .logits [i ,mask ].argmax (dim =-1 )
                tgt_ids =lab_i [mask ]
                pred_text =self .tok .decode (pred_ids .detach ().to ("cpu").tolist (),skip_special_tokens =True )
                tgt_text =self .tok .decode (tgt_ids .detach ().to ("cpu").tolist (),skip_special_tokens =True )

                def short (s ,n =3000 ):return s [:n ]+("…"if len (s )>n else "")
                print ("\n"+"="*88 )
                print ("PROMPT :",short (prompt [i ]))
                print ("TARGET :",short (answer [i ]))
                print ("PRED   :",short (pred_text ))
                print ("="*88 +"\n",flush =True )
        return out if return_dict else (out .loss ,out .logits )

    def _ensure_assistant_prefix (self ,s :str )->str :

        tag ="<|im_start|>assistant\n"
        return s if s .lstrip ().startswith (tag )else (tag +s )

    def _build_prompt_inputs (self ,prompt :list [str ],ts3m :list [dict ],tsdot :list [dict ]):
        w =self ._emb_weight ()
        device ,dtype =w .device ,w .dtype

        chunks =[]
        for p in prompt :
            left ,mid ,right =self ._split2 (p )
            right =self ._ensure_assistant_prefix (right )
            ids_left =self .tok (left ,add_special_tokens =False ,return_tensors ="pt").input_ids .to (device )
            ids_mid =self .tok (mid ,add_special_tokens =False ,return_tensors ="pt").input_ids .to (device )
            ids_right =self .tok (right ,add_special_tokens =False ,return_tensors ="pt").input_ids .to (device )
            e_left ,e_mid ,e_right =self ._ids2emb (ids_left ),self ._ids2emb (ids_mid ),self ._ids2emb (ids_right )
            if e_left .dim ()==2 :e_left =e_left .unsqueeze (0 )
            if e_mid .dim ()==2 :e_mid =e_mid .unsqueeze (0 )
            if e_right .dim ()==2 :e_right =e_right .unsqueeze (0 )
            chunks .append ((e_left ,e_mid ,e_right ))
        x3m =_pack_ts_batch (ts3m ,device )
        xdot =_pack_ts_batch (tsdot ,device )
        z3m ,zdot =self .tsenc (x3m ,xdot )
        z3m ,zdot =z3m .to (device =device ,dtype =dtype ),zdot .to (device =device ,dtype =dtype )


        rows ,masks ,max_len =[],[],0
        for (e_left ,e_mid ,e_right ),i in zip (chunks ,range (len (chunks ))):
            row =torch .cat ([e_left ,z3m [i ].unsqueeze (0 ),e_mid ,zdot [i ].unsqueeze (0 ),e_right ],dim =1 )
            mask =torch .ones ((1 ,row .size (1 )),dtype =torch .long ,device =device )
            rows .append (row );masks .append (mask );max_len =max (max_len ,row .size (1 ))


        H =rows [0 ].size (-1 )
        padded_rows ,padded_masks =[],[]
        for row ,mask in zip (rows ,masks ):
            pad =max_len -row .size (1 )
            if pad >0 :
                row =torch .cat ([row ,torch .zeros ((1 ,pad ,H ),dtype =dtype ,device =device )],dim =1 )
                mask =torch .cat ([mask ,torch .zeros ((1 ,pad ),dtype =torch .long ,device =device )],dim =1 )
            padded_rows .append (row );padded_masks .append (mask )

        inputs_embeds =torch .cat (padded_rows ,dim =0 )
        attention_mask =torch .cat (padded_masks ,dim =0 )
        return inputs_embeds ,attention_mask


    @torch .no_grad ()
    def generate (self ,*args ,**kwargs ):

        batch =getattr (self ,"_cached_batch_for_generate",None )

        if batch is None :
            kwargs ["return_dict_in_generate"]=False
            return self .qwen .generate (*args ,**kwargs )
        prompts =[ex ["prompt"]for ex in batch ]
        ts3m =[ex .get ("ts3m")for ex in batch ]
        tsdot =[ex .get ("tsdot")for ex in batch ]
        prompt_embeds ,attn_mask =self ._build_prompt_inputs (prompts ,ts3m ,tsdot )
        if prompt_embeds .size (0 )==0 :
            print ("Error: inputs_embeds is empty!")
            return None
        kwargs .pop ("input_ids",None )
        kwargs ["inputs_embeds"]=prompt_embeds
        kwargs ["attention_mask"]=attn_mask

        kwargs ["return_dict_in_generate"]=False
        output =self .qwen .generate (**kwargs )
        decoded_output =self .tok .decode (output [0 ],skip_special_tokens =True )

        return output


