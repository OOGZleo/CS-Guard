# Copyright (c) 2024 Zou Group
# Licensed under the MIT License
# https://opensource.org/licenses/MIT
from .engine.guardrails import *
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind
from .engine.base import *
from .engine.attacks import *
from .engine.judger import *



__BASE_NAME_SHORTCUTS__ = {
    "codellama",
    "deepseekcoder",
    "gptoss",
    "qwen",
    "qwencoder",
    "devstral",
    "gpt5",
    "venice"
}

__ATTACK_NAME_SHORTCUTS__ = {
    "cipherchat",
    "codeattack",
    "emoji",
    "gcg",
    "tap",
    "ica",
    "cou",
    "translator_bot",
    "aim",
    "devmoderanti",
    "flipattack",
    "ucar",
    "evilconfidant"
}

__DEFENSE_NAME_SHORTCUTS__ = {
    "MD-Judge-v0.2-internlm2_7",
    "Llama-3.1-nemoguard-8b-content-safety",
    "GuardReasoner",
    "WildGuard",
    "Qwen3GuardGen",
    "Qwen3GuardStream"
    "Llama-Guard-4-12B",
    "ShieldGemma",
    "SelfReminder",
    "PAT",
    "RPO",
    "icd",
    "ib-protector",
    "BackTranslation",
    "self-Defense",
    "Parden",
    "LlamaGuard4",
    "NemoGuard",
    "MDJudge",
    "SmoothLLM"
}


def get_base_engine(engine_name: str="none"):
    if engine_name == "none":
        raise TypeError("engine not set.")
    # check if engine_name starts with "experimental:"
    if engine_name not in __BASE_NAME_SHORTCUTS__:
        raise TypeError("unknown engine, please setup your own engine")
    
    if engine_name == "codellama":
        llm_engine = CodeLlama
    elif engine_name == "deepseekcoder":
        llm_engine = DeepseekCoder
    elif engine_name == "gptoss":
        llm_engine = Gptoss
    elif engine_name == "qwen":
        llm_engine = Qwen
    elif engine_name == "qwencoder":
        llm_engine = QwenCoder
    elif engine_name == "devstral":
        llm_engine = devstral
    elif engine_name == "venice":
        llm_engine = venice
    elif engine_name == "gpt5":
        llm_engine = Gpt5

    return llm_engine

def get_attack_engine(engine_name: str="none"):
    if engine_name == "none":
        raise TypeError("engine not set.")
    # check if engine_name starts with "experimental:"
    if engine_name not in __ATTACK_NAME_SHORTCUTS__:
        raise TypeError("unknown engine, please setup your own engine")
    
    if engine_name == "cipherchat":
        llm_engine = CipherChat
    elif engine_name == "aim":
        llm_engine = AIM
    elif engine_name == "emoji":
        llm_engine = EmojiAttack 
    elif engine_name == "devmoderanti":
        llm_engine = DevmodeRanti
    elif engine_name == "flipattack":
        llm_engine = FlipAttack
    elif engine_name == "ucar":
        llm_engine = UCAR
    elif engine_name == "evilconfidant":
        llm_engine = EvilConfidant

    return llm_engine

def get_defense_engine(engine_name: str="none"):
    if engine_name == "none":
        raise TypeError("engine not set.")
    # check if engine_name starts with "experimental:"
    if engine_name not in __DEFENSE_NAME_SHORTCUTS__:
        raise TypeError("unknown engine, please setup your own engine")
    
    if engine_name == "Qwen3GuardGen":
        llm_engine = Qwen3GuardGen
    elif engine_name == "WildGuard":
        llm_engine = wildguard
    elif engine_name == "LlamaGuard4":
        llm_engine = LlamaGuard4
    elif engine_name == "NemoGuard":
        llm_engine = NemoGuard
    elif engine_name == "MDJudge":
        llm_engine = MDJudge
    elif engine_name == "SelfReminder":
        llm_engine = SelfReminder
    elif engine_name == "Parden":
        llm_engine = Parden
    elif engine_name == "SmoothLLM":
        llm_engine = SmoothLLM
    elif engine_name == "RPO":
        llm_engine = RPO
    elif engine_name == "PAT":
        llm_engine = PAT
    
    
    return llm_engine

def get_judger_engine():
    return Judger