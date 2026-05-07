from abc import ABC, abstractmethod

class engine(ABC):
    def __init__(self):
        super().__init__()
        self.output_storage = "none"
        self.sampling_params = None
        self.type = None
        self.system_prompt = "You are a helpful assistant."
        self.procedure = None
        self.append = False

    @property
    @abstractmethod
    def generate(self, batch_input):
        pass

    def get_type(self):
        return self.type

    def get_procedure(self):
        return self.procedure
    
    def set_procedure(self, procedure):
        self.procedure = procedure
    
    def get_sampling_params(self):
        return self.sampling_params
    
    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
        
    
    @abstractmethod
    def construct_prompt(self, batch_input):
        messages = []
        for prompt in batch_input["prompt"]:
            msg = []
            msg.append({"role": "system", "content": self.system_prompt})
            msg.append({"role": "user", "content": prompt})
            messages.append(msg)
        return messages

    @abstractmethod
    def collecting_labels(self, opt):
        pass

    def set_storage(self, file_path):
        self.output_storage = file_path
    
    def get_storage(self):
        return self.output_storage

    def clean_storage(self):
        if self.output_storage!="none":
            if self.append == False:
                with open(self.output_storage, "w") as f:
                    pass  
            else:
                print("append mode is on, not cleaning the storage file.")
    
    def get_collect_label(self):
        return self.collecting_labels
    