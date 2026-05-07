from .template import engine
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind
import re
import json
import tqdm
from .utilities import split_chunks, get_final_label_per_prompt
from openai import OpenAI

class CodeLlama(engine):
    def __init__(self,
                 model="TheBloke/CodeLlama-13B-Instruct-AWQ",
                 tokenizer="meta-llama/CodeLlama-13b-Instruct-hf",
                 append: bool = False,
                 dtype = "auto",
                 enable_chunked_prefill = True,
                 max_model_len = 15000,
                 max_num_seqs=20,
                 gpu_memory_utilization=0.9,
                 swap_space=13.0,
                 trust_remote_code=False,
                 enforce_eager=False,
                 output_storage: str="none"):
        super().__init__()

        self.llm = LLM(model=model,
            tokenizer=tokenizer, 
            dtype=dtype, 
            enable_chunked_prefill=enable_chunked_prefill, 
            max_model_len=max_model_len, 
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager)
        
        self.append = append    
        self.sampling_params = "none"
        self.output_storage = output_storage

    def collecting_labels(self, opt):
        pass

    def construct_prompt(self, batch_input):
        print("system prompt: ", self.system_prompt)
        messages = []
        for prompt in batch_input:
            msg = []
            msg.append({"role": "system", "content": self.system_prompt})
            msg.append({"role": "user", "content": prompt})
            messages.append(msg)
        # print(messages)
        return messages
    
    def set_generation_config(self, 
                            temperature=1,
                            top_p=1, 
                            top_k=0, 
                            repetition_penalty=1,
                            max_tokens=14500,
                            n=5,
                            output_kind=RequestOutputKind.FINAL_ONLY):
        
        self.sampling_params = SamplingParams(temperature=temperature,
                                top_p=top_p, 
                                top_k=top_k, 
                                repetition_penalty=repetition_penalty,
                                max_tokens=max_tokens,
                                n=n,
                                output_kind=output_kind)
        print("sampling params set to: ", self.sampling_params)

    def generate(self, 
                batch_input):

        if self.output_storage=="none":
            raise TypeError("label storage unset")
        
        if self.sampling_params == "none":
            print("no generation config set, use default")
            self.set_generation_config()
        
        
        messages = self.construct_prompt(batch_input)
        outputs = self.llm.chat(
                            messages=messages,
                            sampling_params=self.sampling_params,
                        )
        output_list = []
        for opt in outputs:
            answer_list = []
            for output_value in opt.outputs:
                answer_list.append(output_value.text)
            output_list.append(answer_list)
        return output_list


class DeepseekCoder(engine):
    def __init__(self,
                 model="RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8",
                 tokenizer="RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8",
                 append: bool = False,
                 dtype = "auto",
                 enable_chunked_prefill = True,
                 max_model_len = 15000,
                 max_num_seqs=20,
                 gpu_memory_utilization=0.9,
                 swap_space=13.0,
                 trust_remote_code=True,
                 enforce_eager=False,
                 output_storage: str="none"):
        super().__init__()


        self.llm = LLM(model=model,
                    tokenizer=tokenizer, 
                    dtype=dtype, 
                    enable_chunked_prefill=enable_chunked_prefill, 
                    max_model_len=max_model_len, 
                    max_num_seqs=max_num_seqs,
                    gpu_memory_utilization=gpu_memory_utilization,
                    swap_space=swap_space,
                    trust_remote_code=trust_remote_code,
                    enforce_eager=enforce_eager)

        self.append = append
        self.sampling_params = "none"
        self.output_storage = output_storage

    def construct_prompt(self, batch_input):
        print("system prompt: ", self.system_prompt)
        messages = []
        for prompt in batch_input:
            msg = []
            msg.append({"role": "system", "content": self.system_prompt})
            msg.append({"role": "user", "content": prompt})
            messages.append(msg)
        return messages
    
    def collecting_labels(self, opt):
        pass

    def set_generation_config(self, 
                            temperature=0.3,
                            top_p=0.95, 
                            top_k=0, 
                            repetition_penalty=1,
                            max_tokens=14500,
                            n=5,
                            output_kind=RequestOutputKind.FINAL_ONLY):
        
        self.sampling_params = SamplingParams(temperature=temperature,
                                top_p=top_p, 
                                top_k=top_k, 
                                repetition_penalty=repetition_penalty,
                                max_tokens=max_tokens,
                                n=n,
                                output_kind=output_kind)


    def generate(self, batch_input):

        if self.output_storage=="none":
            raise TypeError("label storage unset")
        
        if self.sampling_params == "none":
            print("no generation config set, use default")
            self.set_generation_config()
            print("sampling parameter set to: ", self.sampling_params)
        
        messages = self.construct_prompt(batch_input)
        outputs = self.llm.chat(
                            messages=messages,
                            sampling_params=self.sampling_params,
                        )
        output_list = []
        for opt in outputs:
            answer_list = []
            for output_value in opt.outputs:
                answer_list.append(output_value.text)
            output_list.append(answer_list)
        return output_list
    
class Gptoss(engine):
    def __init__(self,
                 model="openai/gpt-oss-20b",
                 tokenizer="openai/gpt-oss-20b",
                 append: bool = False,
                 dtype = "auto",
                 enable_chunked_prefill = True,
                 max_model_len = 15000,
                 max_num_seqs=10,
                 gpu_memory_utilization=0.9,
                 swap_space=10.0,
                 trust_remote_code=False,
                 enforce_eager=False,
                 output_storage: str="none",
                 reasoning_effort="medium"):
        super().__init__()


        self.llm = LLM(model=model,
                    tokenizer=tokenizer, 
                    dtype=dtype, 
                    enable_chunked_prefill=enable_chunked_prefill, 
                    max_model_len=max_model_len, 
                    max_num_seqs=max_num_seqs,
                    gpu_memory_utilization=gpu_memory_utilization,
                    swap_space=swap_space,
                    trust_remote_code=trust_remote_code,
                    enforce_eager=enforce_eager)
        
        self.append = append
        self.sampling_params = "none"
        self.output_storage = output_storage
        self.reasoning_effort = reasoning_effort
        print('reasoning effort set to: ', self.reasoning_effort)
    
    def construct_prompt(self, batch_input):
        print('system prompt: ', self.system_prompt)
        messages = []
        for prompt in batch_input:
            msg = []
            msg.append({"role": "system", "content": self.system_prompt})
            msg.append({"role": "user", "content": prompt, "reasoning_effort": self.reasoning_effort})
            messages.append(msg)
        return messages

    def collecting_labels(self, opt):
        pass

    def set_generation_config(self, 
                            temperature=1,
                            top_p=1, 
                            top_k=0, 
                            repetition_penalty=1,
                            max_tokens=14500,
                            n=5,
                            output_kind=RequestOutputKind.FINAL_ONLY):
        
        self.sampling_params = SamplingParams(temperature=temperature,
                                top_p=top_p, 
                                top_k=top_k, 
                                repetition_penalty=repetition_penalty,
                                max_tokens=max_tokens,
                                n=n,
                                output_kind=output_kind)
        print("sampling params set to: ", self.sampling_params)


    def generate(self, batch_input):

        if self.output_storage=="none":
            raise TypeError("label storage unset")
        
        if self.sampling_params == "none":
            print("no generation config set, use default")
            self.set_generation_config()
        
        messages = self.construct_prompt(batch_input)
        outputs = self.llm.chat(
                            messages=messages,
                            sampling_params=self.sampling_params,
                        )
        output_list = []
        for opt in outputs:
            answer_list = []
            for output_value in opt.outputs:
                answer_list.append(output_value.text)
            output_list.append(answer_list)
        return output_list

class Qwen(engine):
    def __init__(self,
                 model="model/Qwen3-30B-A3B-Instruct-2507-UD-Q6_K_XL.gguf",
                 tokenizer="Qwen/Qwen3-30B-A3B-Instruct-2507",
                 append: bool = False,
                 dtype = "float32",
                 enable_chunked_prefill = True,
                 max_model_len = 15000,
                 max_num_seqs=20,
                 gpu_memory_utilization=0.9,
                 swap_space=13.0,
                 trust_remote_code=False,
                 enforce_eager=False,
                 output_storage: str="none"):
        super().__init__()


        self.llm = LLM(model=model,
                    tokenizer=tokenizer, 
                    dtype=dtype, 
                    enable_chunked_prefill=enable_chunked_prefill, 
                    max_model_len=max_model_len, 
                    max_num_seqs=max_num_seqs,
                    gpu_memory_utilization=gpu_memory_utilization,
                    swap_space=swap_space,
                    trust_remote_code=trust_remote_code,
                    enforce_eager=enforce_eager)

        self.append = append
        self.sampling_params = "none"
        self.output_storage = output_storage
    
    def construct_prompt(self, batch_input):
        print("system prompt: ", self.system_prompt)
        messages = []
        for prompt in batch_input:
            msg = []
            msg.append({"role": "system", "content": self.system_prompt})
            msg.append({"role": "user", "content": prompt})
            messages.append(msg)
       
        return messages

    def collecting_labels(self, opt):
        pass

    def set_generation_config(self, 
                            temperature=1,
                            top_p=1, 
                            top_k=0, 
                            repetition_penalty=1,
                            max_tokens=14500,
                            n=5,
                            output_kind=RequestOutputKind.FINAL_ONLY):
        
        self.sampling_params = SamplingParams(temperature=temperature,
                                top_p=top_p, 
                                top_k=top_k, 
                                repetition_penalty=repetition_penalty,
                                max_tokens=max_tokens,
                                n=n,
                                output_kind=output_kind)


    def generate(self, batch_input):

        if self.output_storage=="none":
            raise TypeError("label storage unset")
        
        if self.sampling_params == "none":
            print("no generation config set, use default")
            self.set_generation_config()
        
        messages = self.construct_prompt(batch_input)
        outputs = self.llm.chat(
                            messages=messages,
                            sampling_params=self.sampling_params,
                        )
        
        output_list = []
        for opt in outputs:
            answer_list = []
            for output_value in opt.outputs:
                answer_list.append(output_value.text)
            output_list.append(answer_list)
        return output_list


class QwenCoder(engine):
    def __init__(self,
                 model="model/Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf",
                 tokenizer="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                 append: bool = False,
                 dtype = "float32",
                 enable_chunked_prefill = True,
                 max_model_len = 15000,
                 max_num_seqs=17,
                 gpu_memory_utilization=0.9,
                 swap_space=10.0,
                 trust_remote_code=False,
                 enforce_eager=False,
                 output_storage: str="none"):
        super().__init__()


        self.llm = LLM(model=model,
                    tokenizer=tokenizer, 
                    dtype=dtype, 
                    enable_chunked_prefill=enable_chunked_prefill, 
                    max_model_len=max_model_len, 
                    max_num_seqs=max_num_seqs,
                    gpu_memory_utilization=gpu_memory_utilization,
                    swap_space=swap_space,
                    trust_remote_code=trust_remote_code,
                    enforce_eager=enforce_eager)

        self.append = append    
        self.sampling_params = "none"
        self.output_storage = output_storage
    
    def construct_prompt(self, batch_input):
        print("system prompt: ", self.system_prompt)
        messages = []
        for prompt in batch_input:
            msg = []
            msg.append({"role": "system", "content": self.system_prompt})
            msg.append({"role": "user", "content": prompt})
            messages.append(msg)
        return messages

    def collecting_labels(self, opt):
        pass

    def set_generation_config(self, 
                            temperature=0.7,
                            top_p=0.8, 
                            top_k=20, 
                            repetition_penalty=1.05,
                            max_tokens=14500,
                            n=5,
                            output_kind=RequestOutputKind.FINAL_ONLY):
        
        self.sampling_params = SamplingParams(temperature=temperature,
                                top_p=top_p, 
                                top_k=top_k, 
                                repetition_penalty=repetition_penalty,
                                max_tokens=max_tokens,
                                n=n,
                                output_kind=output_kind)


    def generate(self, batch_input):

        if self.output_storage=="none":
            raise TypeError("label storage unset")
        
        if self.sampling_params == "none":
            print("no generation config set, use default")
            self.set_generation_config()
        
        messages = self.construct_prompt(batch_input)
        outputs = self.llm.chat(
                            messages=messages,
                            sampling_params=self.sampling_params,
                        )
        output_list = []
        for opt in outputs:
            answer_list = []
            for output_value in opt.outputs:
                answer_list.append(output_value.text)
            output_list.append(answer_list)
        return output_list


class devstral(engine):
    def __init__(self,
                 model="mistralai/Devstral-Small-2-24B-Instruct-2512",
                 tokenizer="mistralai/Devstral-Small-2-24B-Instruct-2512",
                 append: bool = False,
                 dtype = "auto",
                 enable_chunked_prefill = True,
                 max_model_len = 15000,
                 max_num_seqs=20,
                 gpu_memory_utilization=0.9,
                 swap_space=10.0,
                 trust_remote_code=False,
                 enforce_eager=False,
                 output_storage: str="none"):
        super().__init__()

        self.llm = LLM(model=model,
                    tokenizer=tokenizer, 
                    dtype=dtype, 
                    enable_chunked_prefill=enable_chunked_prefill, 
                    max_model_len=max_model_len, 
                    max_num_seqs=max_num_seqs,
                    gpu_memory_utilization=gpu_memory_utilization,
                    swap_space=swap_space,
                    trust_remote_code=trust_remote_code,
                    enforce_eager=enforce_eager)

        self.append = append
        self.sampling_params = "none"
        self.output_storage = output_storage
        self.system_prompt = """You are Devstral-Small-2-24B-Instruct-2512, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
                    You power an AI assistant called Le Chat.
                    Your knowledge base was last updated on 2023-10-01.
                    The current date is 2025-29-12.

                    When you're not sure about some information or when the user's request requires up-to-date or specific data, you must use the available tools to fetch the information. Do not hesitate to use tools whenever they can provide a more accurate or complete response. If no relevant tools are available, then clearly state that you don't have the information and avoid making up anything.
                    If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").
                    You are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is 2025-28-12) and when asked about information at specific dates, you discard information that is at another date.
                    You follow these instructions in all languages, and always respond to the user in the language they use or request.
                    Next sections describe the capabilities that you have.

                    # WEB BROWSING INSTRUCTIONS

                    You cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.

                    # MULTI-MODAL INSTRUCTIONS

                    You have the ability to read images, but you cannot generate images. You also cannot read nor transcribe audio files or videos.

                    # TOOL CALLING INSTRUCTIONS

                    You may have access to tools that you can use to fetch information or perform actions. You must use these tools in the following situations:

                    1. When the request requires up-to-date information.
                    2. When the request requires specific data that you do not have in your knowledge base.
                    3. When the request involves actions that you cannot perform without tools.

                    Always prioritize using tools to provide the most accurate and helpful response. If tools are not available, inform the user that you cannot perform the requested action at the moment."""
    
    def construct_prompt(self, batch_input):
        print("system prompt: ", self.system_prompt)
        messages = []
        for prompt in batch_input:
            msg = []
            msg.append({"role": "system", "content": self.system_prompt})
            msg.append({"role": "user", "content": prompt})
            messages.append(msg)
        return messages

    def collecting_labels(self, opt):
        pass

    def set_generation_config(self, 
                            temperature=0.15,
                            top_p=1, 
                            top_k=0, 
                            repetition_penalty=1,
                            max_tokens=14500,
                            n=5,
                            output_kind=RequestOutputKind.FINAL_ONLY):
        
        self.sampling_params = SamplingParams(temperature=temperature,
                                top_p=top_p, 
                                top_k=top_k, 
                                repetition_penalty=repetition_penalty,
                                max_tokens=max_tokens,
                                n=n,
                                output_kind=output_kind)


    def generate(self, batch_input):

        if self.output_storage=="none":
            raise TypeError("label storage unset")
        
        if self.sampling_params == "none":
            print("no generation config set, use default")
            self.set_generation_config()
        
        messages = self.construct_prompt(batch_input)
        outputs = self.llm.chat(
                            messages=messages,
                            sampling_params=self.sampling_params,
                        )
        output_list = []
        for opt in outputs:
            answer_list = []
            for output_value in opt.outputs:
                answer_list.append(output_value.text)
            output_list.append(answer_list)
        return output_list


class venice(engine):
    def __init__(self,
                 append: bool = False,
                 dtype = "float32",
                 enable_chunked_prefill = True,
                 max_model_len = 15000,
                 max_num_seqs=17,
                 gpu_memory_utilization=0.9,
                 swap_space=10.0,
                 trust_remote_code=False,
                 enforce_eager=False,
                 output_storage: str="none"):
        super().__init__()

        self.client = OpenAI(
            api_key="VENICE_INFERENCE_KEY_kGXNxrL5TByjGJ1eXSKiYYRVNx4jliigVQztN0L0xV",
            base_url="https://api.venice.ai/api/v1",
            max_retries=1000,
            timeout=7200
        )

        self.append = append    
        self.sampling_params = "none"
        self.output_storage = output_storage
    
    def construct_prompt(self, batch_input):
        # print("system prompt: ", self.system_prompt)
        # messages = []
        # for prompt in batch_input:
        #     msg = []
        #     msg.append({"role": "system", "content": self.system_prompt})
        #     msg.append({"role": "user", "content": prompt})
        #     messages.append(msg)
        # return messages
        pass

    def collecting_labels(self, opt):
        pass

    def set_generation_config(self, 
                            temperature=0.7,
                            top_p=0.8, 
                            top_k=20, 
                            repetition_penalty=1.05,
                            max_tokens=14500,
                            n=5,
                            output_kind=RequestOutputKind.FINAL_ONLY):
        
        pass


    def generate(self, input, n=5):

        if self.output_storage=="none":
            raise TypeError("label storage unset")
        
        output_list = []
        for prompt in input:
            print("prompt innside venice: ", prompt)
            response = self.client.chat.completions.create(
                model="e2ee-venice-uncensored-24b-p",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                extra_body={"venice_parameters": {"include_venice_system_prompt": False}},
                # frequency_penalty=0,
                # presence_penalty=0,
                # stop=None,
                # temperature=0,
                # max_completion_tokens = 
                # max_tokens=max_tokens,
                # top_p=top_p,
                n = n
            )

        # print(response.choices[0].message.content)

            answer_list = []
            for i, choice in enumerate(response.choices):
                answer_list.append(choice.message.content)

            output_list.append(answer_list)

        # print(response.choices[0].message.content)

        return output_list
    

class Gpt5(engine):
    def __init__(self,
                 append: bool = False,
                 dtype = "float32",
                 enable_chunked_prefill = True,
                 max_model_len = 15000,
                 max_num_seqs=17,
                 gpu_memory_utilization=0.9,
                 swap_space=10.0,
                 trust_remote_code=False,
                 enforce_eager=False,
                 output_storage: str="none"):
        super().__init__()

        
        self.client = OpenAI(max_retries=1000,timeout=7200)
        self.append = append    
        self.sampling_params = "none"
        self.output_storage = output_storage
        self.n = 5
        # self.temperature = temperature
        # self.top_p = top_p
        # self.top_k = top_k
        # self.repetition_penalty = repetition_penalty
        self.max_tokens = 14500
    
    def construct_prompt(self, batch_input):
        # print("system prompt: ", self.system_prompt)
        # messages = []
        # for prompt in batch_input:
        #     msg = []
        #     msg.append({"role": "system", "content": self.system_prompt})
        #     msg.append({"role": "user", "content": prompt})
        #     messages.append(msg)
        # return messages
        pass

    def collecting_labels(self, opt):
        pass

    def set_generation_config(self, 
                            temperature=None,
                            top_p=0.8, 
                            top_k=20, 
                            repetition_penalty=1.05,
                            max_tokens=14500,
                            n=5,
                            output_kind=RequestOutputKind.FINAL_ONLY):
        
        # self.n = n
        # # self.temperature = temperature
        # # self.top_p = top_p
        # # self.top_k = top_k
        # # self.repetition_penalty = repetition_penalty
        # self.max_tokens = max_tokens
        pass

    def generate(self, input, n: int=5, history: list = []):
        print("system prompt: ", self.system_prompt)
        print("n: ", self.n)
        if self.output_storage=="none":
            raise TypeError("label storage unset")
        output_list = []
        
        

        for prompt in input:
            message_list = [{"role": "developer", "content": self.system_prompt}]
            if len(history) > 0:
                for h in history:
                    message_list.append({"role": "user", "content": h["user"]})
                    message_list.append({"role": "assistant", "content": h["assistant"]})
                    
            message_list.append({"role": "user", "content": prompt})
            print("message list: ", message_list)
            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=message_list,
                # frequency_penalty=0,
                # presence_penalty=0,
                # stop=None,
                # temperature=0,
                max_completion_tokens = 14500,
                # max_tokens=max_tokens,
                # top_p=top_p,
                reasoning_effort="low",
                n = n
            )
            answer_list = []
            for i, choice in enumerate(response.choices):
                answer_list.append(choice.message.content)

            output_list.append(answer_list)

        # print(response.choices[0].message.content)

        return output_list