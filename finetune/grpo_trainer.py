













import copy
import inspect
import os
import re
import textwrap
import warnings
from collections import defaultdict ,deque
from collections .abc import Sequence ,Sized
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any ,Callable ,Optional ,Union
import datasets
import torch
import torch .utils .data
import transformers
from accelerate .utils import broadcast_object_list ,gather ,gather_object ,is_peft_model ,set_seed
from datasets import Dataset ,IterableDataset
from torch import nn
from torch .distributed .fsdp import FullyShardedDataParallel as FSDP
from torch .utils .data import DataLoader ,Sampler
from transformers import (
AutoConfig ,
AutoModelForSequenceClassification ,
AutoProcessor ,
AutoTokenizer ,
GenerationConfig ,
PreTrainedModel ,
PreTrainedTokenizerBase ,
ProcessorMixin ,
Trainer ,
TrainerCallback ,
is_wandb_available ,
)
from transformers .trainer_utils import seed_worker
from transformers .utils import is_datasets_available ,is_flash_attn_2_available ,is_peft_available ,is_rich_available

from ..data_utils import apply_chat_template ,is_conversational ,maybe_apply_chat_template
from ..extras .profiling import profiling_context ,profiling_decorator
from ..extras .vllm_client import VLLMClient
from ..import_utils import is_liger_kernel_available ,is_vllm_available
from ..models import prepare_deepspeed ,prepare_fsdp ,unwrap_model_for_generation
from ..models .utils import _ForwardRedirection
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .utils import (
disable_dropout_in_model ,
entropy_from_logits ,
generate_model_card ,
get_comet_experiment_url ,
pad ,
print_prompt_completions_sample ,
selective_log_softmax ,
)


if is_peft_available ():
    from peft import PeftConfig ,get_peft_model

if is_liger_kernel_available ():
    from liger_kernel .chunked_loss import LigerFusedLinearGRPOLoss

if is_vllm_available ():
    from vllm import LLM ,SamplingParams
    from vllm .sampling_params import GuidedDecodingParams

if is_wandb_available ():
    import wandb

RewardFunc =Union [str ,PreTrainedModel ,Callable [[list ,list ],list [float ]]]

class RepeatSampler (Sampler ):


    def __init__ (
    self ,
    data_source :Sized ,
    mini_repeat_count :int ,
    batch_size :int =1 ,
    repeat_count :int =1 ,
    shuffle :bool =True ,
    seed :Optional [int ]=None ,
    ):
        self .data_source =data_source
        self .mini_repeat_count =mini_repeat_count
        self .batch_size =batch_size
        self .repeat_count =repeat_count
        self .num_samples =len (data_source )
        self .shuffle =shuffle
        self .seed =seed

        if shuffle :
            self .generator =torch .Generator ()
            if seed is not None :
                self .generator .manual_seed (seed )

    def __iter__ (self ):
        if self .shuffle :
            indexes =torch .randperm (self .num_samples ,generator =self .generator ).tolist ()
        else :
            indexes =list (range (self .num_samples ))

        indexes =[indexes [i :i +self .batch_size ]for i in range (0 ,len (indexes ),self .batch_size )]

        indexes =[chunk for chunk in indexes if len (chunk )==self .batch_size ]

        for chunk in indexes :
            for _ in range (self .repeat_count ):
                for index in chunk :
                    for _ in range (self .mini_repeat_count ):
                        yield index

    def __len__ (self )->int :
        return (self .num_samples //self .batch_size )*self .batch_size *self .mini_repeat_count *self .repeat_count



def nanstd (tensor :torch .Tensor )->torch .Tensor :
    variance =torch .nanmean ((tensor -torch .nanmean (tensor ,keepdim =True ))**2 )
    count =torch .sum (~torch .isnan (tensor ))
    variance *=count /(count -1 )
    return torch .sqrt (variance )


def split_tensor_dict (
tensor_dict :dict [str ,Optional [torch .Tensor ]],num_chunks :int
)->list [dict [str ,Optional [torch .Tensor ]]]:

    first_tensor =next (tensor for tensor in tensor_dict .values ()if tensor is not None )
    chunk_size =first_tensor .shape [0 ]//num_chunks
    return [
    {
    key :tensor [i *chunk_size :(i +1 )*chunk_size ]if tensor is not None else None
    for key ,tensor in tensor_dict .items ()
    }
    for i in range (num_chunks )
    ]


def shuffle_sequence_dict (seq_dict :dict [str ,Optional [Sequence ]])->dict [str ,Optional [Sequence ]]:
    batch_size =len (next (v for v in seq_dict .values ()if v is not None ))
    permutation =torch .randperm (batch_size )

    def permute (v :Optional [Sequence ])->Optional [Sequence ]:
        if v is None :
            return None
        if isinstance (v ,torch .Tensor ):
            return v [permutation ]
        return [v [i ]for i in permutation ]
    return {key :permute (val )for key ,val in seq_dict .items ()}


def nanmin (tensor :torch .Tensor )->torch .Tensor :

    if torch .isnan (tensor ).all ():
        return torch .tensor (float ("nan"),dtype =tensor .dtype ,device =tensor .device )
    return torch .min (tensor [~torch .isnan (tensor )])


def nanmax (tensor :torch .Tensor )->torch .Tensor :
    if torch .isnan (tensor ).all ():
        return torch .tensor (float ("nan"),dtype =tensor .dtype ,device =tensor .device )
    return torch .max (tensor [~torch .isnan (tensor )])

def identity (x ):
    return x
def split_pixel_values_by_grid (batch :dict [str ,torch .Tensor ])->dict [str ,Union [torch .Tensor ,list [torch .Tensor ]]]:

    if "image_grid_thw"not in batch or "pixel_values"not in batch :
        return batch

    lengths =batch ["image_grid_thw"].prod (dim =1 ).tolist ()
    pixel_values =batch ["pixel_values"]

    if sum (lengths )!=pixel_values .size (0 ):
        raise ValueError (f"Mismatch: sum(lengths) = {sum (lengths )} != pixel_values.size(0) = {pixel_values .size (0 )}")

    split_values =list (torch .split (batch ["pixel_values"],lengths ,dim =0 ))
    return {**batch ,"pixel_values":split_values }


def unsplit_pixel_values_by_grid (batch :dict [str ,Union [torch .Tensor ,list [torch .Tensor ]]])->dict [str ,torch .Tensor ]:

    pixel_values =batch .get ("pixel_values")

    if isinstance (pixel_values ,list ):
        merged =torch .cat (pixel_values ,dim =0 )
        return {**batch ,"pixel_values":merged }
    else :
        return batch
def truncate_with_protected_tokens (
ids :torch .Tensor ,mask :torch .Tensor ,target_length :int ,protected_tokens :list [int ]
)->tuple [torch .Tensor ,torch .Tensor ]:

    protected_set =set (protected_tokens )

    def process_sequence (ids ,mask ):

        is_protected =torch .tensor ([x .item ()in protected_set for x in ids ])
        is_non_protected =~is_protected
        num_protected =is_protected .sum ().item ()
        num_non_protected_needed =target_length -num_protected

        if num_non_protected_needed <0 :
            raise ValueError (
            f"target_length ({target_length }) is too small for the protected tokens ({num_protected } tokens). "
            f"Please increase target length to at least {num_protected } or disable truncation."
            )
        non_protected_indices =torch .where (is_non_protected )[0 ]
        keep_non_protected =torch .zeros_like (is_non_protected )
        if num_non_protected_needed >0 :
            keep_indices =non_protected_indices [-num_non_protected_needed :]
            keep_non_protected [keep_indices ]=True
        keep_mask =is_protected |keep_non_protected

        return ids [keep_mask ],mask [keep_mask ]
    truncated_seq =[]
    truncated_mask =[]

    for i in range (ids .shape [0 ]):
        new_ids ,new_mask =process_sequence (ids [i ],mask [i ])
        truncated_seq .append (new_ids )
        truncated_mask .append (new_mask )

    return torch .stack (truncated_seq ),torch .stack (truncated_mask )


class GRPOTrainer (Trainer ):
    _tag_names =["trl","grpo"]

    def __init__ (
    self ,
    model :Union [str ,PreTrainedModel ],
    reward_funcs :Union [RewardFunc ,list [RewardFunc ]],
    args :Optional [GRPOConfig ]=None ,
    train_dataset :Optional [Union [Dataset ,IterableDataset ]]=None ,
    eval_dataset :Optional [Union [Dataset ,IterableDataset ,dict [str ,Union [Dataset ,IterableDataset ]]]]=None ,
    processing_class :Optional [Union [PreTrainedTokenizerBase ,ProcessorMixin ]]=None ,
    reward_processing_classes :Optional [Union [PreTrainedTokenizerBase ,list [PreTrainedTokenizerBase ]]]=None ,
    callbacks :Optional [list [TrainerCallback ]]=None ,
    optimizers :tuple [Optional [torch .optim .Optimizer ],Optional [torch .optim .lr_scheduler .LambdaLR ]]=(None ,None ),
    peft_config :Optional ["PeftConfig"]=None ,
    ):

        if args is None :
            model_name =model if isinstance (model ,str )else model .config ._name_or_path
            model_name =model_name .split ("/")[-1 ]
            args =GRPOConfig (f"{model_name }-GRPO")
        model_init_kwargs =args .model_init_kwargs or {}
        if isinstance (model ,str ):
            model_id =model
            torch_dtype =model_init_kwargs .get ("torch_dtype")
            if isinstance (torch_dtype ,torch .dtype )or torch_dtype =="auto"or torch_dtype is None :
                pass
            elif isinstance (torch_dtype ,str ):
                torch_dtype =getattr (torch ,torch_dtype )
                model_init_kwargs ["torch_dtype"]=torch_dtype
            else :
                raise ValueError (
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype }."
                )
            config =AutoConfig .from_pretrained (model_id )
            architecture =getattr (transformers ,config .architectures [0 ])
            model =architecture .from_pretrained (model_id ,**model_init_kwargs )
        else :
            model_id =model .config ._name_or_path
            if args .model_init_kwargs is not None :
                raise ValueError (
                "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                "This argument can only be used when the `model` argument is a string."
                )

        self .model_kwarg_keys =(
        inspect .signature (model .forward ).parameters .keys ()
        if not hasattr (model ,"get_base_model")
        else inspect .signature (model .get_base_model ().forward ).parameters .keys ()
        )

        if peft_config is not None :
            if not is_peft_available ():
                raise ImportError ("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model =get_peft_model (model ,peft_config )
        if args .gradient_checkpointing :
            model =self ._enable_gradient_checkpointing (model ,args )
        if processing_class is None :
            processing_class =AutoProcessor .from_pretrained (model .config ._name_or_path )
        if isinstance (processing_class ,ProcessorMixin ):
            tokenizer =processing_class .tokenizer
        elif isinstance (processing_class ,PreTrainedTokenizerBase ):
            tokenizer =processing_class
        else :
            raise TypeError ("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer .pad_token is None :
            tokenizer .pad_token =tokenizer .eos_token

        self .pad_token =tokenizer .pad_token
        self .pad_token_id =tokenizer .pad_token_id
        self .eos_token_id =tokenizer .eos_token_id
        self .image_token =getattr (processing_class ,"image_token",None )
        self .image_token_id =getattr (processing_class ,"image_token_id",None )
        self .vision_start_token_id =getattr (model .config ,"vision_start_token_id",None )
        self .vision_end_token_id =getattr (model .config ,"vision_end_token_id",None )
        if not isinstance (reward_funcs ,list ):
            reward_funcs =[reward_funcs ]
        self .reward_func_names =[]
        for i ,reward_func in enumerate (reward_funcs ):
            if isinstance (reward_func ,str ):
                reward_funcs [i ]=AutoModelForSequenceClassification .from_pretrained (
                reward_func ,num_labels =1 ,**model_init_kwargs
                )
            if isinstance (reward_funcs [i ],nn .Module ):
                self .reward_func_names .append (reward_funcs [i ].config ._name_or_path .split ("/")[-1 ])
            else :
                self .reward_func_names .append (reward_funcs [i ].__name__ )
        self .reward_funcs =reward_funcs
        if args .reward_weights is not None :
            if len (args .reward_weights )!=len (reward_funcs ):
                raise ValueError (
                f"Number of reward weights ({len (args .reward_weights )}) must match number of reward "
                f"functions ({len (reward_funcs )})"
                )
            self .reward_weights =torch .tensor (args .reward_weights ,dtype =torch .float32 )
        else :
            self .reward_weights =torch .ones (len (reward_funcs ),dtype =torch .float32 )
        if reward_processing_classes is None :
            reward_processing_classes =[None ]*len (reward_funcs )
        elif not isinstance (reward_processing_classes ,list ):
            reward_processing_classes =[reward_processing_classes ]
        else :
            if len (reward_processing_classes )!=len (reward_funcs ):
                raise ValueError ("The number of reward processing classes must match the number of reward functions.")
        for i ,(reward_processing_class ,reward_func )in enumerate (zip (reward_processing_classes ,reward_funcs )):
            if isinstance (reward_func ,PreTrainedModel ):
                if reward_processing_class is None :
                    reward_processing_class =AutoTokenizer .from_pretrained (reward_func .config ._name_or_path )
                if reward_processing_class .pad_token_id is None :
                    reward_processing_class .pad_token =reward_processing_class .eos_token
                reward_func .config .pad_token_id =reward_processing_class .pad_token_id
                reward_processing_classes [i ]=reward_processing_class
        self .reward_processing_classes =reward_processing_classes

        self .max_prompt_length =args .max_prompt_length
        self .max_completion_length =args .max_completion_length
        self .num_generations =args .num_generations
        self .temperature =args .temperature
        self .top_p =args .top_p
        self .top_k =args .top_k
        self .min_p =args .min_p
        self .repetition_penalty =args .repetition_penalty
        self .use_transformers_paged =args .use_transformers_paged
        self .use_vllm =args .use_vllm
        self .vllm_mode =args .vllm_mode
        self .vllm_gpu_memory_utilization =args .vllm_gpu_memory_utilization
        self .vllm_tensor_parallel_size =args .vllm_tensor_parallel_size
        self .use_liger_loss =args .use_liger_loss
        self .loss_type =args .loss_type
        self .scale_rewards =args .scale_rewards
        self .importance_sampling_level =args .importance_sampling_level
        self .mask_truncated_completions =args .mask_truncated_completions
        self .top_entropy_quantile =args .top_entropy_quantile
        if self .use_liger_loss and self .top_entropy_quantile <1.0 :
            raise NotImplementedError (
            "Liger Kernels don't currently support masking token positions based on entropy."
            )
        if self .use_liger_loss and not self .importance_sampling_level =="token":
            raise NotImplementedError (
            "Liger Kernels currently only support token-level importance sampling. Please set"
            "`importance_sampling_level` to 'token'."
            )

        self .shuffle_dataset =args .shuffle_dataset

        if (
        isinstance (train_dataset ,IterableDataset )
        or isinstance (eval_dataset ,IterableDataset )
        or (
        isinstance (eval_dataset ,dict )and any (isinstance (ds ,IterableDataset )for ds in eval_dataset .values ())
        )
        ):

            raise NotImplementedError (
            "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )
        self .num_iterations =args .num_iterations
        self .epsilon_low =args .epsilon
        self .epsilon_high =args .epsilon_high if args .epsilon_high is not None else args .epsilon
        self ._step =0
        self ._buffered_inputs =None

        model .warnings_issued ["estimate_tokens"]=True

        super ().__init__ (
        model =model ,
        args =args ,
        data_collator =identity ,
        train_dataset =train_dataset ,
        eval_dataset =eval_dataset ,
        processing_class =processing_class ,
        callbacks =callbacks ,
        optimizers =optimizers ,
        )
        self .beta =args .beta
        if self .beta ==0.0 :

            self .ref_model =None
        elif is_peft_model (model ):


            self .ref_model =None
        else :

            config =AutoConfig .from_pretrained (model_id )
            architecture =getattr (transformers ,config .architectures [0 ])
            self .ref_model =architecture .from_pretrained (model_id ,**model_init_kwargs )

        if args .disable_dropout :
            disable_dropout_in_model (model )
            if self .ref_model is not None :
                disable_dropout_in_model (self .ref_model )


        if self .use_liger_loss :
            if not is_liger_kernel_available ():
                raise ImportError (
                "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
                )
            self ._forward_redirection =_ForwardRedirection ()
            self .liger_grpo_loss =LigerFusedLinearGRPOLoss (
            beta =self .beta ,
            epsilon_low =self .epsilon_low ,
            epsilon_high =self .epsilon_high ,
            temperature =self .temperature ,
            use_ref_model =self .beta !=0.0 ,
            loss_type =self .loss_type ,
            max_completion_length =self .max_completion_length ,
            )
        self ._metrics ={"train":defaultdict (list ),"eval":defaultdict (list )}
        self ._total_train_tokens =0
        self .log_completions =args .log_completions
        self .wandb_log_unique_prompts =args .wandb_log_unique_prompts
        self .num_completions_to_print =args .num_completions_to_print

        self ._logs ={
        "image":deque (maxlen =args .generation_batch_size ),
        "prompt":deque (maxlen =args .generation_batch_size ),
        "completion":deque (maxlen =args .generation_batch_size ),
        "rewards":defaultdict (lambda :deque (maxlen =args .generation_batch_size )),
        "advantages":deque (maxlen =args .generation_batch_size ),
        }
        set_seed (args .seed ,device_specific =True )
        if self .use_vllm :
            if not is_vllm_available ():
                raise ImportError (
                "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                "`pip install vllm` to use it."
                )
            if self .vllm_mode =="server"and self .accelerator .is_main_process :
                if args .vllm_server_base_url is not None :
                    base_url =args .vllm_server_base_url
                else :
                    base_url =f"http://{args .vllm_server_host }:{args .vllm_server_port }"
                self .vllm_client =VLLMClient (base_url =base_url ,connection_timeout =args .vllm_server_timeout )
                self .vllm_client .init_communicator (device =torch .cuda .current_device ())

            elif self .vllm_mode =="colocate":


                if not self .accelerator .num_processes %self .vllm_tensor_parallel_size ==0 :
                    raise ValueError (
                    f"vllm_tensor_parallel_size ({self .vllm_tensor_parallel_size }) must divide world size "
                    f"({self .accelerator .num_processes }) evenly."
                    )

                if self .vllm_tensor_parallel_size >1 :
                    self .tp_group ,_ =torch .distributed .new_subgroups_by_enumeration (
                    [
                    list (range (i *self .vllm_tensor_parallel_size ,(i +1 )*self .vllm_tensor_parallel_size ))
                    for i in range (self .accelerator .num_processes //self .vllm_tensor_parallel_size )
                    ]
                    )
                os .environ ["RANK"]=str (self .accelerator .process_index )
                os .environ ["LOCAL_RANK"]=str (self .accelerator .local_process_index )
                os .environ ["WORLD_SIZE"]=str (self .accelerator .num_processes )
                os .environ ["MASTER_ADDR"]=os .environ .get ("MASTER_ADDR","localhost")
                os .environ ["MASTER_PORT"]=os .environ .get ("MASTER_PORT","12345")

                if self .max_prompt_length is not None and self .max_completion_length is not None :
                    max_model_len =self .max_prompt_length +self .max_completion_length
                else :
                    max_model_len =None
                self .llm =LLM (
                model =model .name_or_path ,
                tensor_parallel_size =args .vllm_tensor_parallel_size ,
                gpu_memory_utilization =self .vllm_gpu_memory_utilization ,
                max_num_seqs =self .args .per_device_train_batch_size
                *self .vllm_tensor_parallel_size
                *self .args .steps_per_generation ,
                max_model_len =max_model_len ,
                distributed_executor_backend ="external_launcher",
                seed =self .accelerator .process_index //self .vllm_tensor_parallel_size ,
                max_num_batched_tokens =4096 ,
                model_impl =self .args .vllm_model_impl ,
                )

            self .guided_decoding_regex =args .vllm_guided_decoding_regex
            self ._last_loaded_step =-1
            self .accelerator .wait_for_everyone ()
        else :
            generation_kwargs ={
            "max_new_tokens":self .max_completion_length ,
            "do_sample":True ,
            "pad_token_id":tokenizer .pad_token_id ,
            "bos_token_id":tokenizer .bos_token_id ,
            "eos_token_id":tokenizer .eos_token_id ,
            "temperature":self .temperature ,
            "top_p":self .top_p ,
            "top_k":self .top_k ,
            "min_p":self .min_p ,
            "repetition_penalty":self .repetition_penalty ,
            "cache_implementation":args .cache_implementation ,
            }
            if args .use_transformers_paged :
                generation_kwargs ["max_batch_tokens"]=512
                generation_kwargs ["num_blocks"]=1024
                generation_kwargs ["block_size"]=128
            if args .generation_kwargs is not None :
                generation_kwargs .update (args .generation_kwargs )
            self .generation_config =GenerationConfig (**generation_kwargs )
        self .model_accepts_loss_kwargs =False
        self .model .add_model_tags (self ._tag_names )

        if self .ref_model is not None :
            if self .is_deepspeed_enabled :
                self .ref_model =prepare_deepspeed (self .ref_model ,self .accelerator )
            elif self .is_fsdp_enabled :
                self .ref_model =prepare_fsdp (self .ref_model ,self .accelerator )
            else :
                self .ref_model =self .accelerator .prepare_model (self .ref_model ,evaluation_mode =True )

        if args .sync_ref_model :
            self .add_callback (SyncRefModelCallback (ref_model =self .ref_model ,accelerator =self .accelerator ))

        for i ,reward_func in enumerate (self .reward_funcs ):
            if isinstance (reward_func ,PreTrainedModel ):
                if self .is_deepspeed_enabled :
                    self .reward_funcs [i ]=prepare_deepspeed (reward_func ,self .accelerator )
                else :

                    self .reward_funcs [i ]=self .accelerator .prepare_model (
                    reward_func ,evaluation_mode =True ,device_placement =True
                    )

    def _set_signature_columns_if_needed (self ):
        if self ._signature_columns is None :
            self ._signature_columns =["prompt","image"]
    def get_train_dataloader (self ):
        if self .train_dataset is None :
            raise ValueError ("Trainer: training requires a train_dataset.")

        train_dataset =self .train_dataset
        data_collator =self .data_collator
        if is_datasets_available ()and isinstance (train_dataset ,datasets .Dataset ):
            train_dataset =self ._remove_unused_columns (train_dataset ,description ="training")
        else :
            data_collator =self ._get_collator_with_removed_columns (data_collator ,description ="training")

        dataloader_params ={
        "batch_size":self ._train_batch_size *self .args .steps_per_generation ,
        "collate_fn":data_collator ,
        "num_workers":self .args .dataloader_num_workers ,
        "pin_memory":self .args .dataloader_pin_memory ,
        "persistent_workers":self .args .dataloader_persistent_workers ,
        }

        if not isinstance (train_dataset ,torch .utils .data .IterableDataset ):
            dataloader_params ["sampler"]=self ._get_train_sampler ()
            dataloader_params ["drop_last"]=self .args .dataloader_drop_last
            dataloader_params ["worker_init_fn"]=partial (
            seed_worker ,num_workers =self .args .dataloader_num_workers ,rank =self .args .process_index
            )

            dataloader_params ["prefetch_factor"]=self .args .dataloader_prefetch_factor

        return self .accelerator .prepare (DataLoader (train_dataset ,**dataloader_params ))

    def _get_train_sampler (self ,dataset :Optional [Dataset ]=None )->Sampler :
        if dataset is None :
            dataset =self .train_dataset
        return RepeatSampler (
        data_source =dataset ,
        mini_repeat_count =self .num_generations ,
        batch_size =self .args .generation_batch_size //self .num_generations ,
        repeat_count =self .num_iterations *self .args .steps_per_generation ,
        shuffle =self .shuffle_dataset ,
        seed =self .args .seed ,
        )

    def _get_eval_sampler (self ,eval_dataset )->Sampler :
        return RepeatSampler (
        data_source =eval_dataset ,
        mini_repeat_count =self .num_generations ,
        seed =self .args .seed ,
        )

    def _enable_gradient_checkpointing (self ,model :PreTrainedModel ,args :GRPOConfig )->PreTrainedModel :
        model .config .use_cache =False
        if is_peft_model (model ):
            model .base_model .gradient_checkpointing_enable ()
        else :
            model .gradient_checkpointing_enable ()
        gradient_checkpointing_kwargs =args .gradient_checkpointing_kwargs or {}
        use_reentrant =(
        "use_reentrant"not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs ["use_reentrant"]
        )

        if use_reentrant :
            model .enable_input_require_grads ()

        return model

    @profiling_decorator
    def _get_last_hidden_state (
    self ,
    unwrapped_model ,
    input_ids ,
    attention_mask ,
    logits_to_keep ,
    pixel_values =None ,
    image_grid_thw =None ,
    pixel_attention_mask =None ,
    image_sizes =None ,
    ):
        if is_peft_model (unwrapped_model ):
            unwrapped_model =unwrapped_model .base_model .model
        model_inputs ={"input_ids":input_ids ,"attention_mask":attention_mask }
        if image_grid_thw is not None and pixel_values is not None :
            model_inputs ["image_grid_thw"]=image_grid_thw

        if pixel_values is not None :
            model_inputs ["pixel_values"]=pixel_values

        if pixel_attention_mask is not None :
            model_inputs ["pixel_attention_mask"]=pixel_attention_mask

        if image_sizes is not None :
            model_inputs ["image_sizes"]=image_sizes
        if "logits_to_keep"in self .model_kwarg_keys :

            model_inputs ["logits_to_keep"]=logits_to_keep +1

        last_hidden_state =unwrapped_model .model (**model_inputs ).last_hidden_state

        last_hidden_state =last_hidden_state [:,:-1 ,:]

        last_hidden_state =last_hidden_state [:,-logits_to_keep :,:]
        return last_hidden_state

    def get_high_entropy_mask (
    self ,entropies :torch .Tensor ,mask :torch .Tensor ,threshold :float ,accelerator =None
    )->torch .Tensor :

        non_pad_entropies =entropies [mask .bool ()].float ()
        if non_pad_entropies .numel ()==0 :
            return torch .zeros_like (entropies ,dtype =torch .bool )
        all_non_pad_entropies =self .accelerator .gather (non_pad_entropies )

        entropy_threshold =torch .quantile (all_non_pad_entropies ,threshold )
        masked_entropies =entropies *mask .float ()
        entropy_mask =masked_entropies >=entropy_threshold
        return entropy_mask &mask .bool ()

    @profiling_decorator
    def _get_per_token_logps_and_entropies (
    self ,
    model ,
    input_ids ,
    attention_mask ,
    logits_to_keep ,
    batch_size =None ,
    compute_entropy =False ,
    pixel_values =None ,
    image_grid_thw =None ,
    pixel_attention_mask =None ,
    image_sizes =None ,
    ):

        try :
            unwrapped =self .accelerator .unwrap_model (model )
        except Exception :
            unwrapped =model

        is_ts_model =all (hasattr (unwrapped ,n )for n in ("_split2","_ids2emb","tsenc","qwen","tok"))
        if not is_ts_model :
            batch_size =batch_size or input_ids .size (0 )
            all_logps ,all_entropies =[],[]
            for start in range (0 ,input_ids .size (0 ),batch_size ):
                input_ids_batch =input_ids [start :start +batch_size ]
                attention_mask_batch =attention_mask [start :start +batch_size ]
                model_inputs ={"input_ids":input_ids_batch ,"attention_mask":attention_mask_batch }

                if image_grid_thw is not None and pixel_values is not None :
                    model_inputs ["image_grid_thw"]=image_grid_thw [start :start +batch_size ]
                    start_pixel_idx =image_grid_thw [:start ].prod (-1 ).sum ().item ()
                    end_pixel_idx =image_grid_thw [:start +batch_size ].prod (-1 ).sum ().item ()
                    model_inputs ["pixel_values"]=pixel_values [start_pixel_idx :end_pixel_idx ]
                elif pixel_values is not None :
                    model_inputs ["pixel_values"]=pixel_values [start :start +batch_size ]
                if pixel_attention_mask is not None :
                    model_inputs ["pixel_attention_mask"]=pixel_attention_mask [start :start +batch_size ]
                if image_sizes is not None :
                    model_inputs ["image_sizes"]=image_sizes [start :start +batch_size ]

                if "logits_to_keep"in self .model_kwarg_keys :
                    model_inputs ["logits_to_keep"]=logits_to_keep +1

                logits =model (**model_inputs ).logits
                logits =logits [:,:-1 ,:]
                logits =logits [:,-logits_to_keep :,:]
                logits =logits /self .temperature

                completion_ids =input_ids_batch [:,-logits_to_keep :]
                logps =selective_log_softmax (logits ,completion_ids )
                all_logps .append (logps )
                if compute_entropy :
                    with torch .no_grad ():
                        all_entropies .append (entropy_from_logits (logits ))

            logps =torch .cat (all_logps ,dim =0 )
            entropies =torch .cat (all_entropies ,dim =0 )if compute_entropy else None
            return logps ,entropies

        rows =getattr (self ,"_ts_current_batch",None )
        if not rows :
            raise RuntimeError (
            "TS sidecar cache is empty. Ensure _prepare_inputs cached the generation batch "
            "and set GRPOConfig(remove_unused_columns=False)."
            )
        batch_size =batch_size or input_ids .size (0 )
        all_logps ,all_entropies =[],[]
        emb_dtype =unwrapped ._emb_weight ().dtype
        device =unwrapped ._emb_weight ().device
        from qwen_with_ts_ex4sft import _pack_ts_batch
        for start in range (0 ,input_ids .size (0 ),batch_size ):
            end =start +batch_size
            seg_rows =rows [start :end ]
            per_logps ,per_ents =[],[]
            for i ,row in enumerate (seg_rows ):
                prompt =row ["prompt"]
                ts3m =row .get ("ts3m")
                tsdot =row .get ("tsdot")
                left ,mid ,right =unwrapped ._split2 (prompt )
                if hasattr (unwrapped ,"_ensure_assistant_prefix"):
                    right =unwrapped ._ensure_assistant_prefix (right )

                ids_left =unwrapped .tok (left ,add_special_tokens =False ,return_tensors ="pt").input_ids .to (device )
                ids_mid =unwrapped .tok (mid ,add_special_tokens =False ,return_tensors ="pt").input_ids .to (device )
                ids_right =unwrapped .tok (right ,add_special_tokens =False ,return_tensors ="pt").input_ids .to (device )

                e_left =unwrapped ._ids2emb (ids_left )
                e_mid =unwrapped ._ids2emb (ids_mid )
                e_right =unwrapped ._ids2emb (ids_right )

                x_ot =_pack_ts_batch ([ts3m ],device =device )
                x_dot =_pack_ts_batch ([tsdot ],device =device )
                z_ot ,z_dot =unwrapped .tsenc (x_ot .to (dtype =emb_dtype ),x_dot .to (dtype =emb_dtype ))

                e_prompt =torch .cat ([e_left ,z_ot ,e_mid ,z_dot ,e_right ],dim =1 )
                m_prompt =torch .ones (e_prompt .size (1 ),device =device ,dtype =torch .long ).unsqueeze (0 )


                completion_ids =input_ids [start +i :start +i +1 ][:,-logits_to_keep :]
                comp_embeds =unwrapped ._ids2emb (completion_ids )

                inputs_embeds =torch .cat ([e_prompt ,comp_embeds ],dim =1 ).to (dtype =emb_dtype )
                attn_mask =torch .cat ([m_prompt ,
                torch .ones_like (completion_ids ,dtype =m_prompt .dtype )],dim =1 )
                fwd_kwargs ={"use_cache":False }
                if "logits_to_keep"in self .model_kwarg_keys :
                    fwd_kwargs ["logits_to_keep"]=logits_to_keep +1
                logits =unwrapped .qwen (inputs_embeds =inputs_embeds ,attention_mask =attn_mask ,**fwd_kwargs ).logits
                logits =logits [:,:-1 ,:]
                logits =logits [:,-logits_to_keep :,:]
                logits =logits /self .temperature
                logps =selective_log_softmax (logits ,completion_ids )
                per_logps .append (logps )
                if compute_entropy :
                    with torch .no_grad ():
                        per_ents .append (entropy_from_logits (logits ))

            all_logps .append (torch .cat (per_logps ,dim =0 ))
            if compute_entropy :
                all_entropies .append (torch .cat (per_ents ,dim =0 ))

        logps =torch .cat (all_logps ,dim =0 )
        entropies =torch .cat (all_entropies ,dim =0 )if compute_entropy else None
        return logps ,entropies

    def _fix_param_name_to_vllm (self ,name ,extra_prefixes :Optional [list [str ]]=None ):
        extra_prefixes =extra_prefixes or []
        prefixes =["_checkpoint_wrapped_module."]+extra_prefixes
        for prefix in prefixes :
            name =name .replace (prefix ,"")
        return name

    def _sync_fsdp1_params_to_vllm (self ,module :nn .Module ,prefix :str ="",visited =None ):
        if visited is None :
            visited =set ()
        for child_name ,child_module in module .named_children ():
            child_prefix =f"{prefix }.{child_name }"if prefix else child_name
            self ._sync_fsdp1_params_to_vllm (
            child_module ,prefix =child_prefix ,visited =visited
            )

        if isinstance (module ,FSDP ):
            with FSDP .summon_full_params (module ,recurse =False ,writeback =False ):
                for param_name ,param in module .named_parameters ():
                    full_name =f"{prefix }.{param_name }"if prefix else param_name
                    full_name =self ._fix_param_name_to_vllm (full_name ,extra_prefixes =["_fsdp_wrapped_module."])

                    if full_name in visited :
                        continue
                    visited .add (full_name )

                    if self .vllm_mode =="server"and self .accelerator .is_main_process :
                        self .vllm_client .update_named_param (full_name ,param .data )
                    elif self .vllm_mode =="colocate":
                        llm_model =self .llm .llm_engine .model_executor .driver_worker .model_runner .model
                        llm_model .load_weights ([(full_name ,param .data )])

    def _sync_fsdp2_params_to_vllm (self ,module :nn .Module ):
        for name ,param in module .state_dict ().items ():
            if param .is_cpu :
                param =param .to (torch .device ("cuda"))
            param =param .full_tensor ()

            if self .vllm_mode =="server"and self .accelerator .is_main_process :
                self .vllm_client .update_named_param (name ,param )
            elif self .vllm_mode =="colocate":
                llm_model =self .llm .llm_engine .model_executor .driver_worker .model_runner .model
                llm_model .load_weights ([(name ,param )])

    @profiling_decorator
    def _move_model_to_vllm (self ):

        deepspeed_plugin =self .accelerator .state .deepspeed_plugin
        zero_stage_3 =deepspeed_plugin is not None and deepspeed_plugin .zero_stage ==3
        if zero_stage_3 :
            import deepspeed

            gather_if_zero3 =deepspeed .zero .GatheredParameters
        else :
            gather_if_zero3 =nullcontext

        if is_peft_model (self .model ):
            with gather_if_zero3 (list (self .model .parameters ())):
                self .model .merge_adapter ()
                if self .is_fsdp_enabled :
                    fsdp_plugin =getattr (self .accelerator .state ,"fsdp_plugin",None )
                    fsdp_version =getattr (fsdp_plugin ,"fsdp_version",1 )if fsdp_plugin else 1
                    if fsdp_version ==1 :
                        self ._sync_fsdp1_params_to_vllm (
                        self .model
                        )
                    elif fsdp_version ==2 :
                        self ._sync_fsdp2_params_to_vllm (self .model )
                else :
                    for name ,param in self .model .named_parameters ():
                        name =name .removeprefix ("base_model.model.").replace (".base_layer","")
                        if self .model .prefix in name :
                            continue

                        if "original_module"in name :
                            continue
                        name =self ._fix_param_name_to_vllm (name ,extra_prefixes =["modules_to_save.default."])

                        if self .vllm_mode =="server"and self .accelerator .is_main_process :
                            self .vllm_client .update_named_param (name ,param .data )
                        elif self .vllm_mode =="colocate":
                            llm_model =self .llm .llm_engine .model_executor .driver_worker .model_runner .model
                            llm_model .load_weights ([(name ,param .data )])

                self .model .unmerge_adapter ()
        else :
            if self .is_fsdp_enabled :
                fsdp_plugin =getattr (self .accelerator .state ,"fsdp_plugin",None )
                fsdp_version =getattr (fsdp_plugin ,"fsdp_version",1 )if fsdp_plugin else 1
                if fsdp_version ==1 :
                    self ._sync_fsdp1_params_to_vllm (self .model )
                elif fsdp_version ==2 :
                    self ._sync_fsdp2_params_to_vllm (self .model )
            else :
                for name ,param in self .model .named_parameters ():
                    name =self ._fix_param_name_to_vllm (name )
                    with gather_if_zero3 ([param ]):
                        if self .vllm_mode =="server"and self .accelerator .is_main_process :
                            self .vllm_client .update_named_param (name ,param .data )
                        elif self .vllm_mode =="colocate":
                            llm_model =self .llm .llm_engine .model_executor .driver_worker .model_runner .model
                            llm_model .load_weights ([(name ,param .data )])


        if self .vllm_mode =="server"and self .accelerator .is_main_process :
            self .vllm_client .reset_prefix_cache ()
        elif self .vllm_mode =="colocate":
            self .llm .reset_prefix_cache ()

    @profiling_decorator
    def _prepare_inputs (self ,generation_batch ):
        if isinstance (generation_batch ,list )and generation_batch and isinstance (generation_batch [0 ],dict )and ("prompt"in generation_batch [0 ]):
            self ._ts_current_batch =generation_batch
        elif isinstance (generation_batch ,dict )and ("prompt"in generation_batch )and isinstance (generation_batch ["prompt"],list )and len (generation_batch ["prompt"])>0 :
            rows =[]
            B =len (generation_batch ["prompt"])
            for i in range (B ):
                rows .append ({k :generation_batch [k ][i ]for k in generation_batch .keys ()})
            self ._ts_current_batch =rows

        mode ="train"if self .model .training else "eval"
        if mode =="train":
            generate_every =self .args .steps_per_generation *self .num_iterations
            if self ._step %generate_every ==0 or self ._buffered_inputs is None :
                generation_batch =self ._generate_and_score_completions (generation_batch )
                generation_batch =split_pixel_values_by_grid (generation_batch )
                generation_batch =shuffle_sequence_dict (generation_batch )
                generation_batches =split_tensor_dict (generation_batch ,self .args .steps_per_generation )
                self ._buffered_inputs =[unsplit_pixel_values_by_grid (batch )for batch in generation_batches ]
            inputs =self ._buffered_inputs [self ._step %self .args .steps_per_generation ]
            self ._step +=1
        else :
            inputs =self ._generate_and_score_completions (generation_batch )

        return inputs

    @profiling_decorator
    def _calculate_rewards (self ,inputs ,prompts ,completions ,completion_ids_list ):
        device =self .accelerator .device
        rewards_per_func =torch .zeros (len (prompts ),len (self .reward_funcs ),device =device )


        keys =[key for key in inputs [0 ]if key not in ["prompt","completion","completion_ids"]]
        reward_kwargs ={key :[example [key ]for example in inputs ]for key in keys }


        reward_kwargs ["trainer_state"]=self .state

        for i ,(reward_func ,reward_processing_class ,reward_func_name )in enumerate (
        zip (self .reward_funcs ,self .reward_processing_classes ,self .reward_func_names )
        ):
            with profiling_context (self ,reward_func_name ):
                if isinstance (reward_func ,nn .Module ):
                    if is_conversational (inputs [0 ]):
                        messages =[{"messages":p +c }for p ,c in zip (prompts ,completions )]
                        texts =[apply_chat_template (x ,reward_processing_class )["text"]for x in messages ]
                    else :
                        texts =[p +c for p ,c in zip (prompts ,completions )]
                    reward_inputs =reward_processing_class (
                    text =texts ,return_tensors ="pt",padding =True ,padding_side ="right",add_special_tokens =False
                    )
                    reward_inputs =super ()._prepare_inputs (reward_inputs )
                    with torch .inference_mode ():
                        rewards_per_func [:,i ]=reward_func (**reward_inputs ).logits [:,0 ]
                else :
                    output_reward_func =reward_func (
                    prompts =prompts ,completions =completions ,completion_ids =completion_ids_list ,**reward_kwargs
                    )

                    output_reward_func =[reward if reward is not None else torch .nan for reward in output_reward_func ]

                    rewards_per_func [:,i ]=torch .tensor (output_reward_func ,dtype =torch .float32 ,device =device )

        if torch .isnan (rewards_per_func ).all (dim =1 ).any ():
            nan_row_idx =torch .isnan (rewards_per_func ).all (dim =1 ).nonzero (as_tuple =True )[0 ][0 ]
            row_reward_kwargs ={key :value [nan_row_idx ]for key ,value in reward_kwargs .items ()}
            row_reward_kwargs ["prompt"]=prompts [nan_row_idx ]
            row_reward_kwargs ["completion"]=completions [nan_row_idx ]
            warnings .warn (
            f"All reward functions returned None for the following kwargs: {row_reward_kwargs }. "
            "Please ensure that at least one reward function returns a valid reward."
            )
        rewards_per_func =gather (rewards_per_func )
        return rewards_per_func

    def _generate_and_score_completions (
    self ,inputs :list [dict [str ,Union [torch .Tensor ,Any ]]]
    )->dict [str ,Union [torch .Tensor ,Any ]]:
        device =self .accelerator .device
        mode ="train"if self .model .training else "eval"

        prompts =[x ["prompt"]for x in inputs ]
        original_prompts =copy .deepcopy (prompts )
        kwargs ={}
        has_images ="image"in inputs [0 ]
        if has_images :
            images =[example .get ("image")for example in inputs ]
            kwargs ={"images":[[img ]for img in images ]}
            for prompt in prompts :
                if isinstance (prompt ,list ):
                    for message in prompt :
                        if not isinstance (message ,dict ):
                            continue
                        content =message .get ("content")
                        role =message .get ("role")
                        if isinstance (content ,str ):
                            if role =="user":
                                message ["content"]=[{"type":"image"},{"type":"text","text":content }]
                            elif role =="system":
                                message ["content"]=[{"type":"text","text":content }]

        prompts_text =[maybe_apply_chat_template (example ,self .processing_class )["prompt"]for example in inputs ]

        prompt_inputs =self .processing_class (
        text =prompts_text ,
        return_tensors ="pt",
        padding =True ,
        padding_side ="left",
        add_special_tokens =False ,
        **kwargs ,
        )
        prompt_inputs =super ()._prepare_inputs (prompt_inputs )
        prompt_ids ,prompt_mask =prompt_inputs ["input_ids"],prompt_inputs ["attention_mask"]

        if self .max_prompt_length is not None :
            protected =[self .image_token_id ,self .vision_start_token_id ,self .vision_end_token_id ]
            protected =[token for token in protected if token is not None ]
            prompt_ids ,prompt_mask =truncate_with_protected_tokens (
            prompt_ids ,prompt_mask ,self .max_prompt_length ,protected
            )

            prompts_text =self .processing_class .batch_decode (
            prompt_ids ,skip_special_tokens =False ,clean_up_tokenization_spaces =False
            )
            prompts_text =[re .sub (rf"^({re .escape (self .pad_token )})+","",text )for text in prompts_text ]
            if self .image_token is not None :
                prompts_text =[
                re .sub (rf"({re .escape (self .image_token )})+",self .image_token ,text )for text in prompts_text
                ]
        if self .use_vllm :
            if self .state .global_step !=self ._last_loaded_step :
                self ._move_model_to_vllm ()
                self ._last_loaded_step =self .state .global_step
            if self .vllm_mode =="server":
                all_prompts_text =gather_object (prompts_text )
                if has_images :
                    all_images =gather_object (images )
                if self .accelerator .is_main_process :
                    ordered_set_of_prompts =all_prompts_text [::self .num_generations ]

                    if has_images :
                        ordered_set_of_images =all_images [::self .num_generations ]
                    else :
                        ordered_set_of_images =None

                    with profiling_context (self ,"vLLM.generate"):
                        completion_ids =self .vllm_client .generate (
                        prompts =ordered_set_of_prompts ,
                        images =ordered_set_of_images ,
                        n =self .num_generations ,
                        repetition_penalty =self .repetition_penalty ,
                        temperature =self .temperature ,
                        top_p =self .top_p ,
                        top_k =-1 if self .top_k is None else self .top_k ,
                        min_p =0.0 if self .min_p is None else self .min_p ,
                        max_tokens =self .max_completion_length ,
                        guided_decoding_regex =self .guided_decoding_regex ,
                        generation_kwargs =self .args .generation_kwargs ,
                        )
                else :
                    completion_ids =[None ]*len (all_prompts_text )


                completion_ids =broadcast_object_list (completion_ids ,from_process =0 )
                process_slice =slice (
                self .accelerator .process_index *len (prompts ),
                (self .accelerator .process_index +1 )*len (prompts ),
                )
                completion_ids =completion_ids [process_slice ]
            elif self .vllm_mode =="colocate":
                if self .guided_decoding_regex :
                    guided_decoding =GuidedDecodingParams (regex =self .guided_decoding_regex )
                else :
                    guided_decoding =None

                generation_kwargs ={
                "n":1 ,
                "repetition_penalty":self .repetition_penalty ,
                "temperature":self .temperature ,
                "top_p":self .top_p ,
                "top_k":-1 if self .top_k is None else self .top_k ,
                "min_p":0.0 if self .min_p is None else self .min_p ,
                "max_tokens":self .max_completion_length ,
                "guided_decoding":guided_decoding ,
                }
                if self .args .generation_kwargs is not None :
                    generation_kwargs .update (self .args .generation_kwargs )
                sampling_params =SamplingParams (**generation_kwargs )

                if self .vllm_tensor_parallel_size >1 :


                    orig_size =len (prompts_text )
                    gathered_prompts =[None for _ in range (self .vllm_tensor_parallel_size )]
                    torch .distributed .all_gather_object (gathered_prompts ,prompts_text ,group =self .tp_group )
                    all_prompts_text =[p for sublist in gathered_prompts for p in sublist ]

                    if has_images :
                        gathered_images =[None for _ in range (self .vllm_tensor_parallel_size )]
                        torch .distributed .all_gather_object (gathered_images ,images ,group =self .tp_group )
                        all_images =[img for sublist in gathered_images for img in sublist ]
                    else :
                        all_images =None
                else :
                    all_prompts_text =prompts_text
                    all_images =images if has_images else None

                if has_images and all_images :
                    vllm_inputs =[]
                    for prompt ,image in zip (all_prompts_text ,all_images ):
                        if image is not None :
                            vllm_inputs .append ({"prompt":prompt ,"multi_modal_data":{"image":image }})
                        else :
                            vllm_inputs .append (prompt )
                else :
                    vllm_inputs =all_prompts_text

                with profiling_context (self ,"vLLM.generate"):
                    all_outputs =self .llm .generate (vllm_inputs ,sampling_params =sampling_params ,use_tqdm =False )

                completion_ids =[output .token_ids for outputs in all_outputs for output in outputs .outputs ]

                if self .vllm_tensor_parallel_size >1 :
                    local_rank_in_group =torch .distributed .get_rank (group =self .tp_group )
                    tp_slice =slice (local_rank_in_group *orig_size ,(local_rank_in_group +1 )*orig_size )
                    completion_ids =completion_ids [tp_slice ]


            completion_ids =[torch .tensor (ids ,device =device )for ids in completion_ids ]
            completion_ids =pad (completion_ids ,padding_value =self .pad_token_id )
            prompt_completion_ids =torch .cat ([prompt_ids ,completion_ids ],dim =1 )

        elif self .use_transformers_paged :
            paged_prompt_inputs =self .processing_class (text =prompts_text ,**kwargs )
            previous_attn =self .model_wrapped .config ._attn_implementation

            if is_flash_attn_2_available ():
                self .model_wrapped .config ._attn_implementation ="paged_attention"
            else :
                self .model_wrapped .config ._attn_implementation ="sdpa_paged"
            with (
            profiling_context (self ,"transformers.generate_batch"),
            unwrap_model_for_generation (
            self .model_wrapped ,self .accelerator ,gather_deepspeed3_params =self .args .ds3_gather_for_generation
            )as unwrapped_model ,
            torch .no_grad (),
            FSDP .summon_full_params (self .model_wrapped ,recurse =False )if self .is_fsdp_enabled else nullcontext (),
            ):

                if self .args .bf16 :
                    unwrapped_model .to (torch .bfloat16 )
                elif self .args .fp16 :
                    unwrapped_model .to (torch .float16 )
                with torch .inference_mode ():
                    all_outputs =unwrapped_model .generate_batch (
                    paged_prompt_inputs .input_ids ,generation_config =self .generation_config ,progress_bar =False
                    )
            completion_ids =[output .generated_tokens for output in all_outputs .values ()]
            completion_ids =[torch .tensor (ids ,device =device )for ids in completion_ids ]
            completion_ids =pad (completion_ids ,padding_value =self .pad_token_id ,padding_side ="right")
            prompt_ids =[torch .tensor (ids ,device =device )for ids in paged_prompt_inputs .input_ids ]
            prompt_ids =pad (prompt_ids ,padding_value =self .pad_token_id ,padding_side ="left")
            prompt_completion_ids =torch .cat ([prompt_ids ,completion_ids ],dim =1 )

            self .model_wrapped .config ._attn_implementation =previous_attn
        else :

            with (
            profiling_context (self ,"transformers.generate"),
            unwrap_model_for_generation (
            self .model_wrapped ,self .accelerator ,gather_deepspeed3_params =self .args .ds3_gather_for_generation
            ),
            torch .no_grad (),
            FSDP .summon_full_params (self .model_wrapped ,recurse =False )if self .is_fsdp_enabled else nullcontext (),
            ):
                wrapper =self .accelerator .unwrap_model (self .model )
                ts_mode =hasattr (wrapper ,"_build_prompt_inputs")and hasattr (wrapper ,"qwen")
                if ts_mode :
                    wrapper ._cached_batch_for_generate =inputs

                prompt_inputs ["input_ids"],prompt_inputs ["attention_mask"]=prompt_ids ,prompt_mask
                prompt_completion_ids =wrapper .generate (
                **prompt_inputs ,generation_config =self .generation_config ,return_dict_in_generate =False
                )
            if ts_mode :
                wrapper ._cached_batch_for_generate =None

            if ts_mode :
                completion_ids =prompt_completion_ids
            else :
                prompt_length =prompt_ids .size (1 )
                prompt_ids =prompt_completion_ids [:,:prompt_length ]
                completion_ids =prompt_completion_ids [:,prompt_length :]

        is_eos =completion_ids ==self .eos_token_id
        eos_idx =torch .full ((is_eos .size (0 ),),is_eos .size (1 ),dtype =torch .long ,device =device )
        eos_idx [is_eos .any (dim =1 )]=is_eos .int ().argmax (dim =1 )[is_eos .any (dim =1 )]
        sequence_indices =torch .arange (is_eos .size (1 ),device =device ).expand (is_eos .size (0 ),-1 )
        completion_mask =(sequence_indices <=eos_idx .unsqueeze (1 )).int ()
        completion_ids_list =[
        [id .item ()for id ,m in zip (row ,mask_row )if m ]for row ,mask_row in zip (completion_ids ,completion_mask )
        ]
        completion_lengths =completion_mask .sum (1 )
        if self .mask_truncated_completions :
            truncated_completions =~is_eos .any (dim =1 )
            completion_mask =completion_mask *(~truncated_completions ).unsqueeze (1 ).int ()


        attention_mask =torch .cat ([prompt_mask ,completion_mask ],dim =1 )

        logits_to_keep =completion_ids .size (1 )
        batch_size =self .args .per_device_train_batch_size if mode =="train"else self .args .per_device_eval_batch_size

        with torch .no_grad ():
            generate_every =self .args .steps_per_generation *self .num_iterations
            if self .args .gradient_accumulation_steps %generate_every !=0 :
                old_per_token_logps ,_ =self ._get_per_token_logps_and_entropies (
                self .model ,
                prompt_completion_ids ,
                attention_mask ,
                logits_to_keep ,
                batch_size ,
                pixel_values =prompt_inputs .get ("pixel_values"),
                image_grid_thw =prompt_inputs .get ("image_grid_thw"),
                pixel_attention_mask =prompt_inputs .get ("pixel_attention_mask"),
                image_sizes =prompt_inputs .get ("image_sizes"),
                )
            else :
                old_per_token_logps =None


            if self .beta !=0.0 :
                if self .ref_model is not None :
                    ref_per_token_logps ,_ =self ._get_per_token_logps_and_entropies (
                    self .ref_model ,
                    prompt_completion_ids ,
                    attention_mask ,
                    logits_to_keep ,
                    batch_size =batch_size ,
                    pixel_values =prompt_inputs .get ("pixel_values"),
                    image_grid_thw =prompt_inputs .get ("image_grid_thw"),
                    pixel_attention_mask =prompt_inputs .get ("pixel_attention_mask"),
                    image_sizes =prompt_inputs .get ("image_sizes"),
                    )
                else :
                    with self .accelerator .unwrap_model (self .model ).disable_adapter ():
                        ref_per_token_logps ,_ =self ._get_per_token_logps_and_entropies (
                        self .model ,
                        prompt_completion_ids ,
                        attention_mask ,
                        logits_to_keep ,
                        batch_size =batch_size ,
                        pixel_values =prompt_inputs .get ("pixel_values"),
                        image_grid_thw =prompt_inputs .get ("image_grid_thw"),
                        pixel_attention_mask =prompt_inputs .get ("pixel_attention_mask"),
                        image_sizes =prompt_inputs .get ("image_sizes"),
                        )
            else :
                ref_per_token_logps =None


        completions_text =self .processing_class .batch_decode (completion_ids ,skip_special_tokens =True )
        if is_conversational (inputs [0 ]):
            completions =[]
            for prompt ,completion in zip (prompts ,completions_text ):
                bootstrap =prompt .pop ()["content"]if prompt [-1 ]["role"]=="assistant"else ""
                completions .append ([{"role":"assistant","content":bootstrap +completion }])
        else :
            completions =completions_text

        rewards_per_func =self ._calculate_rewards (inputs ,original_prompts ,completions ,completion_ids_list )
        rewards =(rewards_per_func *self .reward_weights .to (device ).unsqueeze (0 )).nansum (dim =1 )


        mean_grouped_rewards =rewards .view (-1 ,self .num_generations ).mean (dim =1 )
        std_grouped_rewards =rewards .view (-1 ,self .num_generations ).std (dim =1 )
        is_std_zero =torch .isclose (std_grouped_rewards ,torch .zeros_like (std_grouped_rewards ))


        mean_grouped_rewards =mean_grouped_rewards .repeat_interleave (self .num_generations ,dim =0 )
        std_grouped_rewards =std_grouped_rewards .repeat_interleave (self .num_generations ,dim =0 )
        advantages =rewards -mean_grouped_rewards
        if self .scale_rewards :
            advantages =advantages /(std_grouped_rewards +1e-4 )


        process_slice =slice (
        self .accelerator .process_index *len (prompts ),
        (self .accelerator .process_index +1 )*len (prompts ),
        )
        all_process_advantages =advantages .clone ()
        advantages =advantages [process_slice ]


        if mode =="train":
            self .state .num_input_tokens_seen +=self .accelerator .gather (attention_mask .sum ()).sum ().item ()
        self ._metrics [mode ]["num_tokens"]=[self .state .num_input_tokens_seen ]


        agg_completion_lengths =self .accelerator .gather (completion_lengths )
        self ._metrics [mode ]["completions/mean_length"].append (agg_completion_lengths .float ().mean ().item ())
        self ._metrics [mode ]["completions/min_length"].append (agg_completion_lengths .float ().min ().item ())
        self ._metrics [mode ]["completions/max_length"].append (agg_completion_lengths .float ().max ().item ())


        agg_terminated_with_eos =self .accelerator .gather (is_eos .any (dim =1 ))
        term_completion_lengths =agg_completion_lengths [agg_terminated_with_eos ]
        clipped_completions_ratio =1 -len (term_completion_lengths )/len (agg_completion_lengths )
        self ._metrics [mode ]["completions/clipped_ratio"].append (clipped_completions_ratio )
        if len (term_completion_lengths )==0 :
            term_completion_lengths =torch .zeros (1 ,device =device )
        self ._metrics [mode ]["completions/mean_terminated_length"].append (term_completion_lengths .float ().mean ().item ())
        self ._metrics [mode ]["completions/min_terminated_length"].append (term_completion_lengths .float ().min ().item ())
        self ._metrics [mode ]["completions/max_terminated_length"].append (term_completion_lengths .float ().max ().item ())


        for i ,reward_func_name in enumerate (self .reward_func_names ):
            mean_rewards =torch .nanmean (rewards_per_func [:,i ]).item ()
            self ._metrics [mode ][f"rewards/{reward_func_name }/mean"].append (mean_rewards )
            std_rewards =nanstd (rewards_per_func [:,i ]).item ()
            self ._metrics [mode ][f"rewards/{reward_func_name }/std"].append (std_rewards )
        self ._metrics [mode ]["reward"].append (mean_grouped_rewards .mean ().item ())
        self ._metrics [mode ]["reward_std"].append (std_grouped_rewards .mean ().item ())
        self ._metrics [mode ]["frac_reward_zero_std"].append (is_std_zero .float ().mean ().item ())


        self ._logs ["prompt"].extend (gather_object (prompts_text ))
        self ._logs ["completion"].extend (gather_object (completions_text ))
        for i ,name in enumerate (self .reward_func_names ):
            self ._logs ["rewards"][name ].extend (rewards_per_func [:,i ].tolist ())
        self ._logs ["advantages"].extend (all_process_advantages .tolist ())

        if has_images :
            self ._logs ["image"].extend (gather_object (images ))

        output ={
        "prompt_ids":prompt_ids ,
        "prompt_mask":prompt_mask ,
        "completion_ids":completion_ids ,
        "completion_mask":completion_mask ,
        "advantages":advantages ,
        }
        if old_per_token_logps is not None :
            output ["old_per_token_logps"]=old_per_token_logps
        if ref_per_token_logps is not None :
            output ["ref_per_token_logps"]=ref_per_token_logps
        if "pixel_values"in prompt_inputs :
            output ["pixel_values"]=prompt_inputs ["pixel_values"]
        if "image_grid_thw"in prompt_inputs :
            output ["image_grid_thw"]=prompt_inputs ["image_grid_thw"]
        if "pixel_attention_mask"in prompt_inputs :
            output ["pixel_attention_mask"]=prompt_inputs ["pixel_attention_mask"]
        if "image_sizes"in prompt_inputs :
            output ["image_sizes"]=prompt_inputs ["image_sizes"]
        return output

    def compute_liger_loss (self ,unwrapped_model ,inputs ):

        prompt_ids ,prompt_mask =inputs ["prompt_ids"],inputs ["prompt_mask"]
        completion_ids ,completion_mask =inputs ["completion_ids"],inputs ["completion_mask"]
        input_ids =torch .cat ([prompt_ids ,completion_ids ],dim =1 )
        attention_mask =torch .cat ([prompt_mask ,completion_mask ],dim =1 )
        logits_to_keep =completion_ids .size (1 )


        last_hidden_state =self ._get_last_hidden_state (
        unwrapped_model ,
        input_ids ,
        attention_mask ,
        logits_to_keep ,
        inputs .get ("pixel_values"),
        inputs .get ("image_grid_thw"),
        inputs .get ("pixel_attention_mask"),
        inputs .get ("image_sizes"),
        )


        loss ,metrics =self .liger_grpo_loss (
        _input =last_hidden_state ,
        lin_weight =unwrapped_model .lm_head .weight ,
        selected_token_ids =completion_ids ,
        attention_mask =completion_mask ,
        advantages =inputs ["advantages"],
        bias =unwrapped_model .lm_head .bias ,
        old_per_token_logps =inputs .get ("old_per_token_logps"),
        ref_per_token_logps =inputs .get ("ref_per_token_logps"),
        )


        mean_kl =metrics [0 ]if self .beta !=0.0 else None
        print ("kl:",mean_kl )
        clip_ratio =metrics [-1 ]

        mode ="train"if self .model .training else "eval"
        if self .beta !=0.0 :
            self ._metrics [mode ]["kl"].append (self .accelerator .gather (mean_kl ).mean ().item ())
        self ._metrics [mode ]["clip_ratio"].append (self .accelerator .gather (clip_ratio ).mean ().item ())
        return loss

    @profiling_decorator
    def compute_loss (self ,model ,inputs ,return_outputs =False ,num_items_in_batch =None ):
        if return_outputs :
            raise ValueError ("The GRPOTrainer does not support returning outputs")

        if (not isinstance (inputs ,dict ))or ("prompt_ids"not in inputs ):
            inputs =self ._prepare_inputs (inputs )

        if self .use_liger_loss :
            unwrapped_model =self .accelerator .unwrap_model (model )
            return self ._forward_redirection (model ,unwrapped_model ,self .compute_liger_loss ,unwrapped_model ,inputs )
        else :
            return self ._compute_loss (model ,inputs )

    def _compute_loss (self ,model ,inputs ):

        if (not isinstance (inputs ,dict ))or ("prompt_ids"not in inputs ):
            inputs =self ._prepare_inputs (inputs )
        prompt_ids ,prompt_mask =inputs ["prompt_ids"],inputs ["prompt_mask"]
        completion_ids ,completion_mask =inputs ["completion_ids"],inputs ["completion_mask"]
        input_ids =torch .cat ([prompt_ids ,completion_ids ],dim =1 )
        attention_mask =torch .cat ([prompt_mask ,completion_mask ],dim =1 )
        logits_to_keep =completion_ids .size (1 )
        per_token_logps ,entropies =self ._get_per_token_logps_and_entropies (
        model ,
        input_ids ,
        attention_mask ,
        logits_to_keep ,
        compute_entropy =True ,
        pixel_values =inputs .get ("pixel_values"),
        image_grid_thw =inputs .get ("image_grid_thw"),
        pixel_attention_mask =inputs .get ("pixel_attention_mask"),
        image_sizes =inputs .get ("image_sizes"),

        )

        if self .top_entropy_quantile <1.0 :
            entropy_mask =self .get_high_entropy_mask (entropies ,completion_mask ,1 -self .top_entropy_quantile )
        else :
            entropy_mask =None


        if self .beta !=0.0 :
            ref_per_token_logps =inputs ["ref_per_token_logps"]
            per_token_kl =(
            torch .exp (ref_per_token_logps -per_token_logps )-(ref_per_token_logps -per_token_logps )-1
            )
        advantages =inputs ["advantages"]

        old_per_token_logps =inputs .get ("old_per_token_logps")
        old_per_token_logps =per_token_logps .detach ()if old_per_token_logps is None else old_per_token_logps

        log_ratio =per_token_logps -old_per_token_logps
        if self .importance_sampling_level =="token":
            log_importance_weights =log_ratio
        elif self .importance_sampling_level =="sequence":
            log_importance_weights =(log_ratio *completion_mask ).sum (-1 )/completion_mask .sum (-1 ).clamp (min =1.0 )
            log_importance_weights =log_importance_weights .unsqueeze (-1 )
        else :
            raise ValueError (
            f"Unknown importance sampling level: {self .importance_sampling_level }. Possible values are 'token' "
            "and 'sequence'."
            )

        coef_1 =torch .exp (log_importance_weights )
        coef_2 =torch .clamp (coef_1 ,1 -self .epsilon_low ,1 +self .epsilon_high )


        if self .args .delta is not None :
            coef_1 =torch .clamp (coef_1 ,max =self .args .delta )

        per_token_loss1 =coef_1 *advantages .unsqueeze (1 )
        per_token_loss2 =coef_2 *advantages .unsqueeze (1 )
        per_token_loss =-torch .min (per_token_loss1 ,per_token_loss2 )
        if entropy_mask is not None :
            per_token_loss =per_token_loss *entropy_mask
        if self .beta !=0.0 :
            per_token_loss =per_token_loss +self .beta *per_token_kl

        if self .loss_type =="dapo":
            loss =((per_token_loss *completion_mask ).sum (-1 )/completion_mask .sum (-1 ).clamp (min =1.0 )).mean ()
            print ("loss:",loss )

        elif self .loss_type =="bnpo":
            loss =(per_token_loss *completion_mask ).sum ()/completion_mask .sum ().clamp (min =1.0 )
        elif self .loss_type =="dr_grpo":
            loss =(per_token_loss *completion_mask ).sum ()/(per_token_loss .size (0 )*self .max_completion_length )
        else :
            raise ValueError (f"Unknown loss type: {self .loss_type }")


        mode ="train"if self .model .training else "eval"

        completion_token_count =completion_mask .sum ().clamp (min =1.0 )

        def masked_batch_mean (x ):
            if x .shape [1 ]==1 :
                return x .mean ()
            else :
                return (x *completion_mask ).sum ()/completion_token_count

        if self .beta !=0.0 :
            mean_kl =masked_batch_mean (per_token_kl )
            self ._metrics [mode ]["kl"].append (self .accelerator .gather (mean_kl ).nanmean ().item ())
            print ("mean_kl:",float (mean_kl ))

        mean_entropy =masked_batch_mean (entropies )
        self ._metrics [mode ]["entropy"].append (self .accelerator .gather (mean_entropy ).nanmean ().item ())


        is_low_clipped =(coef_1 <1 -self .epsilon_low )&(advantages .unsqueeze (1 )<0 )
        is_high_clipped =(coef_1 >1 +self .epsilon_high )&(advantages .unsqueeze (1 )>0 )
        is_region_clipped =is_low_clipped |is_high_clipped

        low_clip =masked_batch_mean (is_low_clipped .float ())
        high_clip =masked_batch_mean (is_high_clipped .float ())
        clip_ratio =masked_batch_mean (is_region_clipped .float ())

        gathered_low_clip =self .accelerator .gather (low_clip )
        self ._metrics [mode ]["clip_ratio/low_mean"].append (gathered_low_clip .nanmean ().item ())
        self ._metrics [mode ]["clip_ratio/low_min"].append (nanmin (gathered_low_clip ).item ())
        gathered_high_clip =self .accelerator .gather (high_clip )
        self ._metrics [mode ]["clip_ratio/high_mean"].append (gathered_high_clip .nanmean ().item ())
        self ._metrics [mode ]["clip_ratio/high_max"].append (nanmax (gathered_high_clip ).item ())
        gathered_clip_ratio =self .accelerator .gather (clip_ratio )
        self ._metrics [mode ]["clip_ratio/region_mean"].append (gathered_clip_ratio .nanmean ().item ())
        return loss

    def prediction_step (self ,model ,inputs ,prediction_loss_only ,ignore_keys :Optional [list [str ]]=None ):
        print ("input")
        print (inputs )
        inputs =self ._prepare_inputs (inputs )
        with torch .no_grad ():
            with self .compute_loss_context_manager ():
                loss =self .compute_loss (model ,inputs )
            loss =loss .mean ().detach ()
        return loss ,None ,None

    def log (self ,logs :dict [str ,float ],start_time :Optional [float ]=None )->None :
        mode ="train"if self .model .training else "eval"
        metrics ={key :sum (val )/len (val )for key ,val in self ._metrics [mode ].items ()}



        if mode =="eval":
            metrics ={f"eval_{key }":val for key ,val in metrics .items ()}

        logs ={**logs ,**metrics }
        super ().log (logs ,start_time )
        self ._metrics [mode ].clear ()

        if self .accelerator .is_main_process and self .log_completions :
            if is_rich_available ():
                print_prompt_completions_sample (
                self ._logs ["prompt"],
                self ._logs ["completion"],
                self ._logs ["rewards"],
                self ._logs ["advantages"],
                self .state .global_step ,
                self .num_completions_to_print ,
                )

            if self .args .report_to and "wandb"in self .args .report_to and wandb .run is not None :
                import pandas as pd

                table ={
                "step":[str (self .state .global_step )]*len (self ._logs ["prompt"]),
                "prompt":self ._logs ["prompt"],
                "completion":self ._logs ["completion"],
                **self ._logs ["rewards"],
                "advantage":self ._logs ["advantages"],
                }

                if self ._logs ["image"]:
                    table ["image"]=[]
                    for img in self ._logs ["image"]:
                        if img is not None :

                            table ["image"].append (wandb .Image (img ))
                        else :
                            table ["image"].append (None )

                df =pd .DataFrame (table )
                if self .wandb_log_unique_prompts :
                    df =df .drop_duplicates (subset =["prompt"])
                wandb .log ({"completions":wandb .Table (dataframe =df )})


    def _save_checkpoint (self ,model ,trial ):
        if self .args .hub_model_id is None :
            model_name =Path (self .args .output_dir ).name
        else :
            model_name =self .args .hub_model_id .split ("/")[-1 ]
        self .create_model_card (model_name =model_name )
        super ()._save_checkpoint (model ,trial )

    def create_model_card (
    self ,
    model_name :Optional [str ]=None ,
    dataset_name :Optional [str ]=None ,
    tags :Union [str ,list [str ],None ]=None ,
    ):

        if not self .is_world_process_zero ():
            return

        if hasattr (self .model .config ,"_name_or_path")and not os .path .isdir (self .model .config ._name_or_path ):
            base_model =self .model .config ._name_or_path
        else :
            base_model =None


        if tags is None :
            tags =set ()
        elif isinstance (tags ,str ):
            tags ={tags }
        else :
            tags =set (tags )

        if hasattr (self .model .config ,"unsloth_version"):
            tags .add ("unsloth")

        tags .update (self ._tag_names )

        citation =textwrap .dedent (
        """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card =generate_model_card (
        base_model =base_model ,
        model_name =model_name ,
        hub_model_id =self .hub_model_id ,
        dataset_name =dataset_name ,
        tags =tags ,
        wandb_url =wandb .run .url if is_wandb_available ()and wandb .run is not None else None ,
        comet_url =get_comet_experiment_url (),
        trainer_name ="GRPO",
        trainer_citation =citation ,
        paper_title ="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        paper_id ="2402.03300",
        )

        model_card .save (os .path .join (self .args .output_dir ,"README.md"))