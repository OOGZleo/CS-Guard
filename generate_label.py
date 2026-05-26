import json
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, cast, Dict, List, Optional, Union
from fire import Fire
from tqdm import tqdm
from omegaconf import OmegaConf
from dataset.create_dataset import *
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind
from collections import Counter
import re
import bench_engine as be
from abc import ABC, abstractmethod
import bench_engine
import argparse


class BenchManager():
    def __init__(self, mode:str = "io", attack_engine:str="none", defense_engine:str="none", base_engine:str="none"):
        self.mode = mode
        self.attack_engine = attack_engine
        self.defense_engine = defense_engine
        self.base_engine = base_engine
        self.judger = "none"
        self.eval_dataloader = "none"
        self.system_mode = "default"
        self.dataset_name = "MetaTTP"
    
    def get_final_label_per_prompt(self, judgement_list):
        final_label = [res['judgement_result']['label'][0] for res in judgement_list if res['judgement_result']['status']=='passed']
        
        if len(final_label) == 0:
            label = 'human_review'
        elif 'bad' in final_label:
            label = 'bad'
        elif 'good' in final_label:
            label = 'good'
        else:
            label = 'unclear'
        
        return label

    def split_chunks(self, lst, n):
        return [lst[i:i+n] for i in range(0, len(lst), n)]

    def collect_results(self, batch_input, outputs, engine):
        label_storage = engine.get_storage() 
        with open(label_storage, "a", encoding="utf-8") as file:
            for i, output in enumerate(outputs):
                output_list=[]
                for output_value in output:
                        output_list.append(output_value)
                if self.dataset_name == "MetaTTP":
                    response = {
                        "prompt": batch_input["prompt"][i],
                        "response": output_list,
                        "mitre_category": batch_input["mitre_category"][i],
                        "ttp_id": batch_input["ttp_id_name_mapping"]['TTP_ID'][i],
                        "ttp_name": batch_input["ttp_id_name_mapping"]['TTP_Name'][i]
                    }
                    file.write(json.dumps(response) + "\n")
                
                elif self.dataset_name == "infill" or self.dataset_name == "translate":
                    response = {
                        "prompt": batch_input["prompt"][i],
                        "response": output_list,
                        "category": batch_input["category"][i],
                        "language": batch_input["language"][i]
                    }
                    file.write(json.dumps(response) + "\n")

                elif self.dataset_name == "complete":
                    response = {
                        "prompt": batch_input["prompt"][i],
                        "response": output_list,
                        "category": batch_input["category"][i]
                    }
                    file.write(json.dumps(response) + "\n")

    def collect_labels(self, batch_input, outputs, engine):
        collecting_labels = engine.get_collect_label()
        label_storage = engine.get_storage() 
        if self.mode == "i":
            print("collecting result for prompt detector")
            with open(label_storage, "a", encoding="utf-8") as file:
                for i, output in enumerate(outputs):
                    if self.dataset_name == "MetaTTP":
                        response = {
                            "prompt": batch_input["prompt"][i],
                            "label": collecting_labels(('', output))['label'],
                            "mitre_category": batch_input["mitre_category"][i],
                            "ttp_id": batch_input["ttp_id_name_mapping"]['TTP_ID'][i],
                            "ttp_name": batch_input["ttp_id_name_mapping"]['TTP_Name'][i]
                        }
                        file.write(json.dumps(response) + "\n")
                    
                    elif self.dataset_name == "infill" or self.dataset_name == "translate":
                        response = {
                            "prompt": batch_input["prompt"][i],
                            "label": collecting_labels(('', output))['label'],
                            "category": batch_input["category"][i],
                            "language": batch_input["language"][i]
                        }
                        file.write(json.dumps(response) + "\n")
                    
                    elif self.dataset_name == "complete":
                        response = {
                            "prompt": batch_input["prompt"][i],
                            "label": collecting_labels(('', output))['label'],
                            "category": batch_input["category"][i]
                        }
                        file.write(json.dumps(response) + "\n")

        elif self.mode == "io":
            print("collecting labels")
            sub_lists = self.split_chunks(outputs, len(batch_input["response"]))
            with open(label_storage, "a", encoding="utf-8") as file:
                for idx, lis in tqdm(enumerate(sub_lists), total=len(sub_lists), desc="processing prompts"):
                    judgement_list=[]
                    info = {}
                    if self.dataset_name == "MetaTTP":
                        info = {"prompt": batch_input["prompt"][idx], 
                                'label': "none"}
                                # "mitre_category": batch_input["mitre_category"][idx],
                                # "ttp_id": batch_input["ttp_id"][idx],
                                # "ttp_name": batch_input['ttp_name'][idx]}
                        
                    elif self.dataset_name == "infill" or self.dataset_name == "translate":
                        info = {"prompt": batch_input["prompt"][idx], 
                                'label': "none"}
                                # "category": batch_input["category"][idx],
                                # "language": batch_input["language"][idx]}
                    
                    elif self.dataset_name == "complete":
                        info = {"prompt": batch_input["prompt"][idx], 
                                'label': "none"}
                                # "category": batch_input["category"][idx]}
                    # print("current processing prompt: ", sample["prompt"][idx])
                    # iterate through j llm response to a single prompt
            
                    for jdx, opt in tqdm(enumerate(lis), total=len(lis), desc="processing response candidates"):
                        # info = {"response": sample["response"][jdx][idx]}
                        # print('collecting labels for qwen_guard')
                        print("-"*100+f"prompt {idx}"+"-"*100)
                        print('response: ', batch_input["response"][jdx][idx])
                        judgement_list.append({"response": batch_input["response"][jdx][idx],
                                                "id": jdx,
                                                "judgement_result": collecting_labels((batch_input["response"][jdx][idx], opt))})

                    
                    info['response_candidate'] = judgement_list
                    label_decided = self.get_final_label_per_prompt(judgement_list)
                    info['label'] = label_decided

                    print("final_label: ", info['label'])

                    print("passed")
                
                    file.write(json.dumps(info) + "\n")
    
    def set_base_engine(self, engine):
        self.base_engine = engine

    def set_attack_engine(self, engine):
        self.attack_engine = engine

    def set_defense_engine(self, engine):
        self.defense_engine = engine
    
    def set_judger_engine(self, engine):
        self.judger = engine
    
    def set_data_loader(self, dataset_name, dataset_dirctory, batch_size, num_worker, stage, append, output_storage):

        datamodule = LitDataModule(
            batch_size=batch_size,
            num_worker=num_worker
        )
        start_point = 0
        print("append: ", append)
        # exit(0)
        if append == True:
            print("append is true, checking existing output file for resuming point")
            
            try:
                eval_data = []
                with open(output_storage, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        eval_data.append(obj)
                start_point = len(eval_data)
            except FileNotFoundError:
                start_point = 0
            print(f"resuming from point: {start_point}")
            # exit(0)

        datamodule.setup(dataset=dataset_name, dirctory=dataset_dirctory, stage=stage, start_point=start_point)
        if stage == "eval":
            self.eval_dataloader = datamodule.eval_dataloader()
        elif stage == "test":
            self.eval_dataloader = datamodule.test_dataloader()

    def set_metric(self, name):
        pass

    def set_dataset(self, dataset_name):
        self.dataset_name = dataset_name

    def set_mode(self, mode):
        self.mode = mode

    def get_prompt(self, batch_input, u_mode = None):

        if self.mode == "i":
            return batch_input["prompt"]
        else:
            return batch_input
    
    def set_system_mode(self, system_mode):
        self.system_mode = system_mode
    
    def get_system_mode(self):
        return self.system_mode
        
    def run(self):

        if self.eval_dataloader == "none":
            raise TypeError("dataloader unset")
        # if self.mode == "io" and self.judger=="none":
        #     raise TypeError("Mode is set to io, but no judger model set for evaluating output")

        for sample in tqdm(self.eval_dataloader,desc="running test sample"):
            if self.attack_engine != "none":
                batch_input = self.get_prompt(sample)
                print(sample == batch_input)
                
                a_output = self.attack_engine.generate(batch_input)
                # print('batch input prompt: ', batch_input['prompt'][0])
                # print('sample prompt: ', batch_input['prompt'][0])
                # exit(0)
                print("collecting results")
                if self.attack_engine.get_procedure() == "generate":
                    self.collect_results(sample, a_output, self.attack_engine)
                else:
                    self.collect_labels(sample, a_output, self.attack_engine)

            if self.defense_engine != "none":
                batch_input = self.get_prompt(sample)
                d_output = self.defense_engine.generate(batch_input)
                
                if self.defense_engine.get_procedure() == "generate":
                    print("collecting results")
                    self.collect_results(sample, d_output, self.defense_engine)
                else:
                    self.collect_labels(sample, d_output, self.defense_engine)

            if self.judger != "none":
                print("judger labeling")
                batch_input = self.get_prompt(sample)
                judge_output = self.judger.generate(batch_input)
                # print(sample['prompt'])
                # exit(0)
                self.collect_labels(sample, judge_output, self.judger)

            if self.base_engine != "none":
                batch_input = self.get_prompt(sample)
                base_output = self.base_engine.generate(batch_input)
                # print(sample['prompt'])
                # exit(0)
                self.collect_results(sample, base_output, self.base_engine)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--output",
        help="Path to output file",
        required=True
    )

    parser.add_argument(
        "-d", "--defense",
        help="defense model",
        default=None
    )

    parser.add_argument(
        "-a", "--attack",
        help="attack model",
        default=None
    )

    parser.add_argument(
        "-j", "--judge",
        help="judge model",
        default=None
    )

    parser.add_argument(
        "-dt", "--dataset",
        help="benchmarking dataset for evaluation",
        required=True
    )

    parser.add_argument(
        "-dir", "--dirctory",
        help="directory to input file",
        required=True
    )

    parser.add_argument(
        "-bs", "--batchsize",
        default=20,
        help="batch size for dataloader"
    )

    parser.add_argument(
        "-nw", "--numworker",
        default=14,
        help="num worker for dataloader"
    )

    parser.add_argument(
        "-st", "--stage",
        default="eval",
        help="set evaluation dataset or test dataset",
    )

    parser.add_argument(
        "-m", "--mode",
        default="io",
        help="set mode for evaluation",
    )

    parser.add_argument(
        "-b", "--base",
        default=None,
        help="set base model",
    )

    parser.add_argument(
        "-sd", "--standard",
        default="t2c",
        help="set base model",
    )

    parser.add_argument(
        "-ap", "--append",
        default=False,
        type=bool,
        help="whether to append to existing file",
    )

    args = parser.parse_args()

    manager = BenchManager(mode=args.mode)
    if args.attack != None and args.attack.strip() != "None":
        attack_engine = bench_engine.get_attack_engine(args.attack)(output_storage=args.output, append=args.append, mode=args.mode)
        
        if args.defense != None and args.defense.strip() != "None":
            defense_engine = bench_engine.get_defense_engine(args.defense)(output_storage=attack_engine.get_storage(), append=args.append, mode=args.mode)
            
        # if attack_engine.get_type() == "prompt":
            if defense_engine.requre_base() == True:
                if args.base != None and args.base.strip() != "None":
                    base_engine = bench_engine.get_base_engine(args.base)(output_storage=attack_engine.get_storage(), append=args.append, gpu_memory_utilization=0.92)
                    defense_engine.set_llm(base_engine)
                else:
                    raise TypeError("defense engine type generate but base engine unset")
            attack_engine.set_procedure(defense_engine.get_procedure())
            attack_engine.set_llm(defense_engine)

        elif args.base != None and args.base.strip() != "None":
            base_engine = bench_engine.get_base_engine(args.base)(output_storage=attack_engine.get_storage(), append=args.append, gpu_memory_utilization=0.92)
        # if attack_engine.get_type() == "prompt":
            attack_engine.set_llm(base_engine)
            
        attack_engine.clean_storage()
        attack_engine.set_generation_config()
        manager.set_attack_engine(attack_engine)

    elif args.defense != None and args.defense.strip() != "None":
        defense_engine = bench_engine.get_defense_engine(args.defense)(output_storage=args.output, append=args.append, mode=args.mode)
        if defense_engine.requre_base() == True:
            if args.base != None and args.base.strip() != "None":  
                base_engine = bench_engine.get_base_engine(args.base)(output_storage=defense_engine.get_storage(), append=args.append, gpu_memory_utilization=0.92)
                defense_engine.set_llm(base_engine)
            else:
                raise TypeError("defense engine type strategy but base engine unset")
        defense_engine.clean_storage()
        defense_engine.set_generation_config()
        manager.set_defense_engine(defense_engine)

    elif args.judge != None and args.judge.strip() != "None":
        base_engine = bench_engine.get_base_engine(args.judge)(output_storage=args.output, append=args.append, gpu_memory_utilization=0.92, reasoning_effort="high", max_model_len=20000)
        # base_engine.set_generation_config()
        # self.system_mode = "default"
        if "emoji" in args.output or "Emoji" in args.output:
            print("emoji mode on") 
            manager.set_system_mode("emoji")
        if "dual" in args.output or "Dual" in args.output:
            print("dual mode on") 
            manager.set_system_mode("dual")
        if "cipher" in args.output or "Cipher" in args.output:
            print("cipherchat mode on") 
            manager.set_system_mode("cipherchat")

        judge_engine = bench_engine.get_judger_engine()(output_storage=args.output, append=args.append, llm=base_engine, standard=args.standard, system_mode=manager.get_system_mode())
        judge_engine.clean_storage()
        manager.set_judger_engine(judge_engine)
    
    elif args.base != None and args.base.strip() != "None":
        base_engine = bench_engine.get_base_engine(args.base)(output_storage=args.output, append=args.append, gpu_memory_utilization=0.92)
        base_engine.clean_storage()
        manager.set_base_engine(base_engine)
    

    manager.set_data_loader(args.dataset, args.dirctory, args.batchsize, args.numworker, args.stage, append=args.append, output_storage=args.output)
    manager.set_dataset(args.dataset)
    manager.run()



    
 
