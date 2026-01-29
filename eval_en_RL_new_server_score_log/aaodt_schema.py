from __future__ import annotations
from dataclasses import dataclass
from typing import Dict ,List ,Optional ,Tuple

CONTROLLED ={
"NAME":{"eia","market","opec","opec_plus","eia","united_states",
},
"ACTION":{"cut","price_change","raise","report_release","allocate","approve","cap","close",
},
"OBJECT":{
"price","production","crude_oil","import","gasoline",
"tax_rate","subsidy","capacity","refinery","pipeline",
"shipping","reserve","quota","policy",
},
"DIRECTION":{"up","down","ambiguous"},
}

SYNONYMS ={
"u.s.":"united_states","us":"united_states","u.s":"united_states","u_s":"united_states",
"opec":"opec_plus","opec+":"opec_plus",
"cut output":"production_cut","output cut":"production_cut",
"curb output":"production_cut","reduce production":"production_cut",
"boost output":"production_increase","increase production":"production_increase",
"price cap":"cap_impose","windfall tax":"tax_increase",
"ban export":"export_ban","export curb":"export_ban",
"embargo":"embargo_impose","sanction":"sanction_impose",
"spr draw":"spr_release","spr release":"spr_release",
"spr refill":"spr_refill",
"stocks":"inventory","stockpile":"inventory",
"gas":"gasoline","crude":"crude_oil",
"rise":"up","surge":"up","increase":"up","higher":"up",
"fall":"down","drop":"down","decline":"down","lower":"down",
}
def _norm_token (x :Optional [str ])->Optional [str ]:
    if x is None :
        return None
    y =x .strip ().lower ()
    y =SYNONYMS .get (y ,y )

    y =y .replace (" ","_").replace ("-","_").replace ("+","plus").replace ("/","_")
    return y

def normalize_slot (slot :str ,value :Optional [str ])->Optional [str ]:
    if value is None or value =="":
        return None
    v =_norm_token (value )
    if slot =="NAME":
        v =SYNONYMS .get (v ,v )
    if v in CONTROLLED [slot ]:
        return v
    return v

@dataclass (frozen =True )
class AAOD :
    name :Optional [str ]
    action :Optional [str ]
    obj :Optional [str ]
    direction :Optional [str ]
    def key (self )->str :

        n =self .name or "UNK"
        a =self .action or "UNK"
        o =self .obj or "UNK"
        d =self .direction or "UNK"
        return f"{n }|{a }|{o }|{d }"

    @staticmethod
    def from_strings (name :Optional [str ],action :Optional [str ],
    obj :Optional [str ],direction :Optional [str ]
    )->"AAOD":
        return AAOD (
        name =normalize_slot ("NAME",name ),
        action =normalize_slot ("ACTION",action ),
        obj =normalize_slot ("OBJECT",obj ),
        direction =normalize_slot ("DIRECTION",direction ),
        )

    @staticmethod
    def parse_key (key :str )->"AAOD":
        parts =[p .strip ()for p in key .split ("|")]
        parts +=[""]*max (0 ,4 -len (parts ))
        print (parts )
        return AAOD .from_strings (*parts [:4 ])
