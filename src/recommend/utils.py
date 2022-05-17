from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent.parent


########## utils for headline generation ###########

import json
import math
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
#from tqdm import tnrange
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def add_special_tokens(tokenizer_path):
    """ Returns GPT2 tokenizer after adding separator and padding tokens """
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path, pad_token='<|endoftext|>')
    special_tokens = {'sep_token':'<|sep|>'}
    num_add_toks = tokenizer.add_special_tokens(special_tokens)
    return tokenizer
    
    
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
    # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits      


def sample_seq_fast(model, context, length, num_sentences, device, temperature=1, top_k=0, top_p=0.0, eos_stopping=False):
    """ Generates a sequence of tokens 
        Args:
            model: gpt/gpt2 model
            context: tokenized text using gpt/gpt2 tokenizer
            length: length of generated sequence.
            device: torch.device object.
            temperature >0: used to control the randomness of predictions by scaling the logits before applying softmax.
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    
    # generates one senence more than wanted
    # looks at the generated token and estimates the num of sentences on the go
    # after n+1 times ".!?" it takes first n sentences by sent_tokenize
    sent_to_gen = num_sentences + 1
    
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():  
        for _ in range(length):
            inputs = {'input_ids': generated}
            assert len(inputs["input_ids"]) <= 1024 #########################
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if not next_token and eos_stopping:
                break
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if not eos_stopping and next_token in [1, 14, 31]:
                sent_to_gen -= 1
                if not sent_to_gen:
                    break
    return generated 


def generate_summary_fast(context_enc, sep_idx, tokenizer, model, num_sentences, temperature=1, top_k=50, top_p=0.5,
                     device=torch.device('cuda'), eos_stopping=False):

    
    # generates one senence more than wanted
    # looks at the generated token and estimates the num of sentences on the go
    # after n+1 times ".!?" it takes first n sentences by sent_tokenize

    generated_text = sample_seq_fast(model, context_enc, 1024-sep_idx, num_sentences, device, temperature, top_k, top_p, eos_stopping=eos_stopping)
    generated_text = generated_text[0, len(context_enc):].tolist()
    gen_summary = tokenizer.convert_ids_to_tokens(generated_text,skip_special_tokens=True)
    gen_summary = tokenizer.convert_tokens_to_string(gen_summary)
    
    # extract <num_sentences> sentences 
    if not eos_stopping:
        gen_summary.replace("...", ".")
        try:
            gen_summary = " ".join(nltk.sent_tokenize(gen_summary)[:num_sentences])
        except:
            pass
    return gen_summary



def generate_eval_file(data, data_type, tokenizer, model, save_dir, field, num_sentences=5,
                       max_summaries=0, temperature=1, top_k=50, top_p=0.5, 
                       device=torch.device('cuda'), eval_step=True, eos_stopping=False, skip=0):
    print(data_type)
    max_summaries = math.inf if max_summaries == "full" else max_summaries   
    len_data = min(max_summaries, len(data))
    disp_len = "full" if max_summaries == math.inf else len_data
    if eos_stopping:
        save_file = save_dir + f"/{data_type}_{disp_len}_sent{num_sentences}_eos_topk{top_k}_topp{top_p}.jsonl"
    else:
        save_file = save_dir + f"/{data_type}_{disp_len}_sent{num_sentences}_topk{top_k}_topp{top_p}.jsonl"
    print(f"saving to: {save_file}")
    
    how_open = ""
    if skip:
        how_open = "a"
    else:
        how_open = "w+"
    with open(save_file, how_open) as output:
        for s in range(skip, len_data): 
            if s%100 == 0:
                print(s)
            sample = data[s]
            sep_idx = sample['sum_idx']
            context = sample['input_ids'][:sep_idx].tolist()
            gold_summary = sample['input_ids'][sep_idx+1:][:100].tolist()
            # generating with the new faster and better method
            gen_summary = generate_summary_fast(context, sep_idx, tokenizer, model, num_sentences, 
                             temperature=temperature, top_k=top_k, top_p=top_p, 
                             device=device, eos_stopping=eos_stopping)
            
            if not eval_step:
                print_summary(tokenizer.decode(context), gen_summary, tokenizer.decode(gold_summary))
            else:
                new_doc = {field: gen_summary}
                line = json
                json.dump(new_doc, output, ensure_ascii=False)
                output.write("\n")


def generate_one_summary_fast(input_text, tokenizer, model, num_sentences=3,
                        temperature=1, top_k=50, top_p=0.5,
                        device=torch.device('cuda'), eos_stopping=False, sep_tok=True):

    context = tokenizer.encode(input_text)
    context += [tokenizer.sep_token_id]

    gen_summary = generate_summary_fast(context, len(context), tokenizer, model, num_sentences, 
                                    temperature=temperature, top_k=top_k, top_p=top_p, device=device,
                                    eos_stopping=eos_stopping)
                
    print_summary(tokenizer.decode(context), gen_summary, "Not Given")
    
    return gen_summary

####################################################