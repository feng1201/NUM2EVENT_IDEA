from __future__ import annotations
import torch
import torch .nn as nn
import torch
import torch .nn as nn
import warnings
warnings .filterwarnings ("ignore",message =".*tokenizer is now deprecated.*")

class PatchMLP (nn .Module ):
    def __init__ (
    self ,
    in_ch :int =2 ,
    patch_size :int =8 ,
    hidden :int =256 ,
    out_dim :int =1024 ,
    layers :int =3 ,
    concat_posidx :bool =False ,
    ):
        super ().__init__ ()
        self .patch_size =patch_size

        in_dim =in_ch *patch_size +(patch_size if concat_posidx else 0 )
        seq =[]

        for i in range (max (1 ,layers -1 )):
            seq +=[nn .Linear (in_dim if i ==0 else hidden ,hidden ),nn .GELU ()]

        seq +=[nn .Linear (hidden if layers >1 else in_dim ,out_dim )]
        self .mlp =nn .Sequential (*seq )
        self .concat_posidx =concat_posidx

    def forward (self ,x :torch .Tensor )->torch .Tensor :

        B ,T ,C =x .shape
        assert C >=1 ,"x must have at least one channel (value)."
        val =x [...,0 :1 ]
        has_mask =C >=2
        msk =x [...,1 :2 ]if has_mask else None


        if has_mask :
            valid =msk .squeeze (-1 ).sum (dim =1 ).long ()
        else :
            valid =torch .full ((B ,),T ,dtype =torch .long ,device =x .device )

        outs ,counts =[],[]
        device =x .device

        for i in range (B ):
            vl =max (1 ,int (valid [i ].item ()))
            P =(vl +self .patch_size -1 )//self .patch_size


            v_i =val [i ,:vl ,:]
            if P *self .patch_size >vl :

                padv =v_i [-1 :].repeat (P *self .patch_size -vl ,1 )
                v_i =torch .cat ([v_i ,padv ],dim =0 )
            v_i =v_i .reshape (P ,self .patch_size )
            feats =[v_i ]


            if has_mask :
                m_i =msk [i ,:vl ,:]
                if P *self .patch_size >vl :
                    padm =m_i [-1 :].repeat (P *self .patch_size -vl ,1 )
                    m_i =torch .cat ([m_i ,padm ],dim =0 )
                m_i =m_i .reshape (P ,self .patch_size )
                feats .append (m_i )


            if self .concat_posidx :
                pos =torch .arange (vl ,device =device ).float ()
                if P *self .patch_size >vl :
                    pos =torch .cat ([pos ,pos [-1 :].repeat (P *self .patch_size -vl )],0 )
                pos =(pos /max (1.0 ,vl -1 )).reshape (P ,self .patch_size )
                feats .append (pos )

            xi =torch .cat (feats ,dim =-1 )


            expected_in =self .mlp [0 ].in_features if hasattr (self .mlp [0 ],"in_features")else xi .shape [-1 ]
            assert xi .shape [-1 ]==expected_in ,f"PatchMLP in_features mismatch: got {xi .shape [-1 ]}, expected {expected_in }"

            zi =self .mlp (xi )
            outs .append (zi )
            counts .append (P )


        L =max (counts )
        outs =[torch .cat ([o ,o [-1 :].repeat (L -o .size (0 ),1 )],0 )for o in outs ]
        return torch .stack (outs ,0 )


class DualTSMLP (nn .Module ):
    def __init__ (
    self ,
    d_model :int ,
    patch_size_ot :int =8 ,
    patch_size_dot :int =4 ,
    hidden :int =256 ,
    layers :int =3 ,
    concat_posidx :bool =False ,
    ):
        super ().__init__ ()

        self .enc_ot =PatchMLP (
        in_ch =2 ,
        patch_size =patch_size_ot ,
        hidden =hidden ,
        out_dim =d_model ,
        layers =layers ,
        concat_posidx =concat_posidx
        )
        self .enc_dot =PatchMLP (
        in_ch =2 ,
        patch_size =patch_size_dot ,
        hidden =hidden ,
        out_dim =d_model ,
        layers =layers ,
        concat_posidx =concat_posidx
        )

    @torch .no_grad ()
    def max_tokens (self ,x_ot :torch .Tensor ,x_dot :torch .Tensor )->int :

        return self .enc_ot (x_ot ).size (1 )+self .enc_dot (x_dot ).size (1 )

    def forward (self ,x_ot :torch .Tensor ,x_dot :torch .Tensor ):
        z_ot =self .enc_ot (x_ot )
        z_dot =self .enc_dot (x_dot )
        return z_ot ,z_dot
