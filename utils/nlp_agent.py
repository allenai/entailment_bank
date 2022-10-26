import logging
import math
import os
import re
import requests
import time

# Convenience functions for running multi-angle models, either from loaded model or through API

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SLOT_SHORTFORMS_DEFAULT = {"Q": "question", "C": "context", "A": "answer", "E": "explanation",
                   "M": "mcoptions", "R": "rationale", "P": "proof", "H": "hypothesis", "V": "valid"}

GENERATOR_OPTIONS_DEFAULT = {"min_length": 1, "max_length": 128, "num_beams": 1, "num_return_sequences": 1,
                             "do_sample": False, "top_k": 50, "top_p": 1.0, "temperature": 1.0,
                             "length_penalty": 1.0, "repetition_penalty": 1.0}

DEFAULT_SLOT_FORMAT = {"slot": "$SLOT$", "assign": " = ", "separator": " ; ", "missing_value": "N/A"}


def decompose_slots(string, fmt=None):
    fmt = fmt or DEFAULT_SLOT_FORMAT
    string = string.strip()
    no_slot = "PREFIX"
    slot_re = re.compile('(?i)'+re.escape(fmt['slot']).replace("SLOT", "(\\w*?)"))
    assign_re = re.escape(fmt['assign']).replace('\\ ','\\s*')
    separator_re = re.escape(fmt['separator']).replace('\\ ','\\s*')
    strip_re = re.compile(f"^({assign_re})?(.*?)({separator_re})?$")
    slot_pos = []
    for m in slot_re.finditer(string):
        slot_pos.append((m.span(), m.group(1)))
    if len(slot_pos) == 0:
        return {no_slot: string}
    if slot_pos[0][0][0] > 0:
        slot_pos = [((0,-1), no_slot)] + slot_pos
    res = {}
    for idx, (pos, slot_name) in enumerate(slot_pos):
        if idx == len(slot_pos) - 1:
            value = string[pos[1]+1:]
        else:
            value = string[pos[1]+1:slot_pos[idx+1][0][0]-1]
        m = strip_re.match(value)
        if m is not None:
            value = m.group(2)
        value = value.strip()
        if slot_name in res:
            value = res[slot_name] + " ~AND~ " + value
        res[slot_name] = value
    return res


def split_mcoptions(mcoptions):
    first_option = ord(mcoptions.strip()[1])
    labels = "".join([chr(x) for x in range(first_option, first_option+10)])
    choices = re.split("\\s*\\(["+labels+"]\\)\\s*", mcoptions)[1:]
    return (choices, chr(first_option))


def new_dict_update(old_dict, update_dict):
    if update_dict is None:
        return old_dict
    res = old_dict.copy()
    res.update(update_dict)
    return res


def make_input_string(fields, angle, fmt=None):
    fmt = fmt or DEFAULT_SLOT_FORMAT
    res = []
    # output angles
    for slot in angle[1]:
        res.append(fmt['slot'].replace("SLOT", slot))
    # input angles
    for slot in angle[0]:
        slot_name = fmt['slot'].replace("SLOT", slot)
        value = fields.get(slot, fmt['missing_value'])
        res.append(f"{slot_name}{fmt['assign']}{value}")
    return fmt['separator'].join(res)


def make_api_input_string(fields, angle, slot_key_from_lowercase, explicit_outputs=None, output_prefix=None):
    res = []
    for slot in angle[0]:
        slot_key = slot_key_from_lowercase.get(slot, slot[0].upper())
        value = fields[slot]
        res.append(f"{slot_key}: {value}")
    if explicit_outputs:
        res.append("X: " + make_mcoptions(explicit_outputs))
    for slot in angle[1]:
        slot_key = slot_key_from_lowercase.get(slot, slot[0].upper())
        if output_prefix is not None and slot in output_prefix:
            res.append(slot_key +"-prefix: " + output_prefix[slot])
        else:
            res.append(slot_key)
    return "\n".join(res)



# Load model and tokenizer, also return the cuda device used for input to model
def load_model(model_name_or_path, cuda_devices = None):
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    cuda_devices = cuda_devices or []
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    except:
        if os.path.exists("/t5-11b-tokenizer"):
            tokenizer = T5Tokenizer.from_pretrained("/t5-11b-tokenizer")
        else:
            tokenizer = T5Tokenizer.from_pretrained("t5-11b")
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    device_map = None
    if len(cuda_devices) > 1:
        # Split layers across the multiple GPUs, put extras in later devices to leave a bit extra on first one
        num_layers = model.config.num_layers
        n_gpu = len(cuda_devices)
        layers_per_gpu = num_layers // n_gpu
        has_one_extra = n_gpu - (num_layers - layers_per_gpu * n_gpu)
        device_map = {}
        current = 0
        for device in cuda_devices:
            next = current + layers_per_gpu
            if len(device_map) >= has_one_extra:
                next += 1
            device_map[device] = list(range(current, next))
            current = next
    if len(cuda_devices) > 0:
        device = f"cuda:{cuda_devices[0]}"
    else:
        device = "cpu"

    if device_map is not None:
        model.parallelize(device_map)
    else:
        model.to(device)
    return {"model": model, "tokenizer": tokenizer, "cuda_device": device}


# Run model in free generation mode, with optional output_prefix_string
def run_model(model, input_string, generator_options, output_prefix_string=None, output_scores=False):
    import torch
    with torch.no_grad():
        input_ids = model['tokenizer'].encode(input_string, return_tensors="pt").to(model['cuda_device'])
        encoder_outputs = model['model'].encoder(input_ids)
        decoder_input_ids = {}
        if output_prefix_string is not None:
            decoder_start_token_id = model['model'].config.decoder_start_token_id
            output_ids = model['tokenizer'].encode(output_prefix_string, return_tensors="pt", add_special_tokens=False)
            decoder_input_ids = torch.cat((torch.LongTensor([[decoder_start_token_id] * len(output_ids)]), output_ids),
                                          dim=1).to(model['cuda_device'])
            decoder_input_ids = {"decoder_input_ids": decoder_input_ids}

        output = model['model'].generate(encoder_outputs=encoder_outputs, **decoder_input_ids,
                                         output_scores=output_scores, return_dict_in_generate=True, **generator_options)

        output_strings = model['tokenizer'].batch_decode(output.sequences, skip_special_tokens=True)
        res = {"input_raw": input_string, "output_raw_list": output_strings}
        if output_scores:
            # Subtract pad token if output_prefix not given
            num_prefix_tokens = len(decoder_input_ids.get('decoder_input_ids', [0]))
            output_token_probs = []
            for idx in range(len(output.sequences)):
                token_probs = []
                for token, scores in zip(output.sequences[idx][num_prefix_tokens:], output.scores):
                    probs = torch.softmax(scores[idx], dim=0)
                    token_probs.append((model['tokenizer'].convert_ids_to_tokens(token.item()), probs[token].item()))
            output_token_probs.append(token_probs)
            res["output_token_probs_list"] = output_token_probs
    return res


# Run model in forced generation mode, capturing each token probability
def run_model_with_outputs(model, input_string, output_texts, output_angle):
    import torch
    with torch.no_grad():
        input_string = input_string
        input_ids = model['tokenizer'].encode(input_string, return_tensors="pt").to(model['cuda_device'])
        # Compute encoder output once and reuse for each output text
        encoder_outputs = model['model'].encoder(input_ids)
        all_res = []
        for output_text in output_texts:
            output_string = make_input_string({output_angle: output_text}, [[output_angle], []])
            output_ids = model['tokenizer'].encode(output_string, return_tensors="pt").to(model['cuda_device'])
            res = model['model'](encoder_outputs=encoder_outputs, labels=output_ids, return_dict=True)
            res_softmax = torch.softmax(res.logits[0], dim=1)
            raw_probs = [x[y.item()].item() for x,y in list(zip(res_softmax, output_ids[0]))]
            output_prob = 1
            for raw_prob in raw_probs:
                output_prob *= raw_prob
            loss = res.loss.item()
            all_res.append({
                "input_raw": input_string,
                "output_raw": output_string,
                "output_text": output_text,
                "loss": loss,
                "score": math.exp(-loss),
                "output_prob": output_prob,
                "output_token_probs": raw_probs,
                "output_tokens": model['tokenizer'].convert_ids_to_tokens(output_ids[0])
            })
    return all_res


def make_mcoptions(choices, first_label='A'):
    res = []
    for idx, choice in enumerate(choices):
        res.append(f'({chr(idx+ord(first_label))}) {choice}')
    return " ".join(res)


# Interface to a multi-angle generative model, either by loading a model or calling an API
class MultiAngleModel():
    def __init__(self,
                 model_path=None,
                 api_url=None,
                 generator_options=None,
                 slot_ordering_override=None,
                 slot_shortforms=None,
                 cuda_devices=None):
        assert model_path is not None or api_url is not None
        assert not (model_path is not None and api_url is not None)
        self.slot_shortforms = new_dict_update(SLOT_SHORTFORMS_DEFAULT, slot_shortforms)
        self.slot_key_from_lowercase = {v.lower(): k for k, v in self.slot_shortforms.items()}
        self.generator_options = new_dict_update(GENERATOR_OPTIONS_DEFAULT, generator_options)
        self.api_url = api_url
        self.model = None
        if model_path is not None:
            self.model = load_model(model_path, cuda_devices)

    def __call__(self, fields, inputs, outputs, options=None):
        options = options or {}
        generator_options = new_dict_update(self.generator_options, options.get('generator_options'))
        explicit_outputs = options.get('explicit_outputs')
        if explicit_outputs is True:
            # Automatically extract from mcoptions field
            explicit_outputs = split_mcoptions(fields['mcoptions'])[0]
        angle = [inputs, outputs]
        if isinstance(angle[0], str): angle[0] = [angle[0]]
        if isinstance(angle[1], str): angle[1] = [angle[1]]
        output_prefix = options.get('output_prefix')
        output_prefix_string = None
        if output_prefix is not None:
            if explicit_outputs is not None:
                raise ValueError("Cannot specify both 'explicit_outputs' and 'output_prefix'")
            slots_prefix = []
            for slot in angle[1]:
                if slot in output_prefix:
                    slots_prefix.append(slot)
                else:
                    break
            if len(slots_prefix) != len(output_prefix):
                raise ValueError(f"Slots in output_prefix ({output_prefix}) do not match initial slots in output slots ({angle[1]})")
            output_prefix_string = make_input_string(output_prefix, [slots_prefix, []])
        full_res = {}
        if options.get("debug"):
            full_res['debug'] = {}
        if self.model:
            input_string = make_input_string(fields, angle)
            res = run_model(self.model, input_string, generator_options, output_prefix_string=output_prefix_string)
            res_slots = decompose_slots(res['output_raw_list'][0])
            full_res.update(res_slots)
            if explicit_outputs:
                output_slot = angle[1][0]
                res_explicit = run_model_with_outputs(self.model, input_string, explicit_outputs, output_slot)
                res_explicit.sort(key=lambda x:-x['output_prob'])
                if options.get("debug"):
                    full_res['debug'].update({"generated_output": res_slots, "explicit_outputs": res_explicit})
                full_res[output_slot] = res_explicit[0]['output_text']
                full_res['output_prob'] = res_explicit[0]['output_prob']
        else:
            api_generator_options = {}
            if generator_options:
                for k, v in generator_options.items():
                    v_new = v
                    if isinstance(v, bool):
                        v_new = 1 if v else 0
                    api_generator_options[k] = v_new

            input_string = make_api_input_string(fields, angle, self.slot_key_from_lowercase, explicit_outputs, output_prefix)
            try:
                res_raw = requests.get(self.api_url, params={"input": input_string, **api_generator_options})
                res = res_raw.json()
            except:
                logger.warning(f"Failed API call to {self.api_url}")
                return {"error": f"Failed API call to {self.api_url}"}
            res_slots = res['output_slots_list'][0]
            full_res.update(res_slots)
            if explicit_outputs:
                output_slot = angle[1][0]
                res_explicit = res['explicit_outputs']
                if options.get("debug"):
                    full_res['debug'].update({"generated_output": res_slots, "explicit_outputs": res_explicit})
                full_res[output_slot] = res_explicit[0]['output_text']
                full_res['output_prob'] = res_explicit[0]['output_prob']
        if options.get("debug"):
            full_res["debug"].update({"raw_input": input_string, "generator_options": generator_options})
        return full_res


class InformationRetriever():
    def __init__(self, api_url, max_retries=3):
        self.api_url = api_url
        self.max_retries = max_retries

    def __call__(self, fields, inputs=None, outputs=None, options=None):
        if options is not None:
            fields = fields.copy()
            fields.update(options)
        retry = 0
        res = None
        while retry <= self.max_retries and res is None:
            retry += 1
            try:
                res_raw = requests.post(self.api_url, json=fields)
                res = {"retrievals": res_raw.json()}
            except:
                logger.warning(f"Failed retriever API call to {self.api_url}, retry = {retry}")
                time.sleep(2^retry)
        if res is None:
            return {"error": f"Failed retriever API call to {self.api_url}"}
        return res


class NlpAgent():
    def __init__(self,
                 model,
                 default_fields=None,     # These will be used as starting point for any input fields
                 default_outputs=None,
                 default_options=None):
        self.model = model
        self.default_fields = default_fields or {}
        self.default_outputs = default_outputs
        self.default_options = default_options

    def __call__(self, fields, inputs=None, outputs=None, options=None):
        fields_full = self.default_fields.copy()
        fields_full.update(fields)
        if inputs is None:
            inputs = [k for k, v in fields.items() if v]
        outputs = outputs or self.default_outputs
        options_full = options
        if self.default_options is not None:
            options_full = self.default_options.copy()
            if options is not None:
                options_full.update(options)
        res = self.model(fields_full, inputs, outputs, options_full)
        if "error" in res:
            return res
        if isinstance(outputs, str):
            res = res.get(outputs)
        return res
