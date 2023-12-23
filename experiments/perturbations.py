from string import ascii_lowercase
from string import ascii_letters
ascii_letters += " "

from abc import ABC, abstractmethod
from random import random, choice

import torch

class Perturbation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def perturb(self):
        return NotImplemented

    def _check_string(self, string):
        is_string = isinstance(string, str)

        if not is_string:
            raise ValueError('string is not type str')


class CasingPerturbation(Perturbation):
    def __init__(self, p_perturbation):
        self.p_perturbation=p_perturbation
    
    def perturb(self, string):
        self._check_string(string)
        
        s = ""
        for sub in string:
            _isupper = sub.isupper()
            if random() < self.p_perturbation:
                if _isupper:
                    s+=sub.lower()
                else:
                    s+=sub.upper()
            else:
                s+=sub

        return s


class TypoPerturbation(Perturbation):
    def __init__(self, p_perturbation):
        self.p_perturbation=p_perturbation
    
    def perturb(self, string):
        self._check_string(string)
        
        s=""
        for sub in string:
            if sub == " ":
                s+=sub
                continue
            
            _isupper = sub.isupper()
            if random() < self.p_perturbation:
                letter = sub.lower()
                while letter == sub.lower():
                    letter = str(choice(ascii_lowercase))

                if _isupper:
                    letter=letter.upper()
                s+=letter
            else:
                s+=sub
                
        return s


class WhitespacePerturbation(Perturbation):
    def __init__(self, min_whitespace=1, max_whitespace=2):
        assert isinstance(min_whitespace, int)
        assert isinstance(max_whitespace, int)
        assert min_whitespace <= max_whitespace
        
        self.min_whitespace=min_whitespace
        self.max_whitespace=max_whitespace

    def perturb(self, string):
        self._check_string(string)

        s=""

        for sub in string:
            if sub == " ":
                n_whitespaces = choice([i for i in range(self.min_whitespace, self.max_whitespace+1)])

                whitespace = ""
                for ws in range(n_whitespaces):
                    whitespace+= " "
                s+=whitespace
            else:
                s+=sub

        return s


class LMPerturbation(Perturbation):
    def __init__(self, p_perturbation, model, tokenizer, tokenizer_kwargs={}, keep_first_n=0, keep_last_n=0):
        self.p_perturbation=p_perturbation
        self.model=model
        self.tokenizer=tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.keep_first_n=keep_first_n
        self.keep_last_n=keep_last_n


    def perturb(self, string):
        self._check_string(string)

        special_tokens = list(self.tokenizer.special_tokens_map.values())

        inputs = self.tokenizer(string, return_tensors="pt")
        input_ids = inputs.input_ids

        max_token_index = len(input_ids[0]) - self.keep_last_n - 1

        s = []
        mask_positions = []
        n_masked=0

        for i, token in enumerate(inputs['input_ids'][0]):
            if i < self.keep_first_n:
                s.append(token.item())
                continue
            elif i > max_token_index:
                s.append(token.item())
                continue
                
            if random() < self.p_perturbation:
                if len(mask_positions) > 1:
                    if mask_positions[-1:] == [i - 1]:
                        s.append(token.item())
                        continue
                
                mask_id = tokenizer.additional_special_tokens_ids[len(mask_positions)]
                s.append(mask_id)
                mask_positions.append(i)
            else:
                s.append(token.item())

        if len(mask_positions) != 0:
            inputs['input_ids'][0] = torch.LongTensor(s)
            
            outputs = self.model.generate(**inputs, 
                                          max_length=1024, do_sample=False, num_return_sequences=1, 
                                          eos_token_id=self.tokenizer.eos_token_id)
            outputs = outputs.detach().cpu().numpy()
        
            for i, mask_position in enumerate(mask_positions):
                mask_id = tokenizer.additional_special_tokens_ids[i]
                output_mask_location = (outputs==mask_id).nonzero()[1][0]
                fill_location = output_mask_location+1

                inputs['input_ids'][0][mask_position] = outputs[0][fill_location]
                
        s = self.tokenizer.decode(inputs.input_ids[0])

        return s.split(self.tokenizer.eos_token)[0]


class PrependPerturbation(Perturbation):
    def __init__(self, length, alphabet=ascii_letters):
        self.alphabet=alphabet
        self.length=length

        self.appendix = " "
        for i in range(self.length):
            self.appendix+=choice(self.alphabet)
        
    def perturb(self, string, new_appendix=True):
        self._check_string(string)

        if self.length== 0:
            return string

        if new_appendix:

            self.appendix = ""
            for i in range(self.length):
                self.appendix+=choice(self.alphabet)
    
        return f'{self.appendix} {string}'


class AppendPerturbation(Perturbation):
    def __init__(self, length, alphabet=ascii_letters):
        self.alphabet=alphabet
        self.length=length

        self.appendix = " "
        for i in range(self.length):
            self.appendix+=choice(self.alphabet)
        

    def perturb(self, string, new_appendix=True):
        self._check_string(string)

        if self.length==0:
            return string

        if new_appendix:
            self.appendix = " "
            for i in range(self.length):
                self.appendix+=choice(self.alphabet)
    
        return f'{string}{self.appendix}'