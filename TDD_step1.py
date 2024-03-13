import os
# import os

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTJForCausalLM, AutoTokenizer, AutoModelForCausalLM,BloomTokenizerFast, BloomForCausalLM, GPTNeoXForCausalLM
import torch
import json
import numpy as np
from tqdm import tqdm
from interpret.lm_saliency import *
# from experiment import flip
import random
import torch.nn.functional as F

# input parameters
model_name = 'gpt2-large' # gpt2-large (36 layers), gpt-j-6B (28 layers), llama-7B (32 layers), bloom-7b (30 layerss), Pythia-6.9b 32 layers

method = "ours" # ours # sota #erasure # rollout
if method == 'sota':
    result = {'IG_base': {}, 'IG_con': {}, 'GN_base': {}, 'GN_con': {}}
if method == "ours":
    result = {'ours': {},'ours_back':{},'ours_add':{}}
if method == "rollout":
    result = {'rollout': {}}


# model_name = "Pythia-6.9b"
data_name = ["anaphor_gender_agreement",'anaphor_number_agreement','animate_subject_passive',
             'determiner_noun_agreement_1','determiner_noun_agreement_irregular_1',
             'determiner_noun_agreement_with_adjective_1','determiner_noun_agreement_with_adj_irregular_1',
             'npi_present_1','distractor_agreement_relational_noun','irregular_plural_subject_verb_agreement_1',
             'regular_plural_subject_verb_agreement_1']
data_name = ['anaphor_gender_agreement','anaphor_number_agreement']
save_name_post = "all"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer_norm = None

for nnn in data_name:
    for kkk in result.keys():
        result[kkk][nnn] = []
print('initial result:',result)

# load models
if model_name == "gpt2-large":
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    lm_head = model.lm_head
    layer_norm = model.transformer.ln_f
    save_name_prefix = 'exp2_gpt2large_'

if model_name == "gpt-j-6B":

    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", load_in_8bit=True,resume_download=True)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    lm_head = model.lm_head
    # layer_norm = model.transformer.ln_f
    layer_norm = model.transformer.ln_f
    save_name_prefix = 'exp2_gptj'

if model_name == 'llama-7B':
    model_path = "meta-llama/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=" ")
    model = AutoModelForCausalLM.from_pretrained(model_path, token=" ",
                                                 load_in_8bit=True)  # load_in_4bit=True
    lm_head = model.lm_head
    layer_norm = model.model.norm
    save_name_prefix = 'exp2_llama7B'

if model_name == "bloom-7b":
    model_path = 'bigscience/bloom-7b1'
    model = BloomForCausalLM.from_pretrained(model_path, load_in_8bit=True,resume_download=True)
    tokenizer = BloomTokenizerFast.from_pretrained(model_path)
    lm_head = model.lm_head
    layer_norm = model.transformer.ln_f
    save_name_prefix = 'exp2_bloom7b'

if model_name == "opt-6.7b":
    # no
    model_path = "facebook/opt-6.7b"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,resume_download=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    lm_head = model.lm_head
    # layer_norm = model.decoder.
    save_name_prefix = 'exp2_opt6b'

if model_name == 'Pythia-6.9b':
    model_path = "EleutherAI/pythia-6.9b-deduped-v0"
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, resume_download=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    lm_head = model.embed_out
    layer_norm = model.gpt_neox.final_layer_norm
    save_name_prefix = 'exp2_pythia6b'

# print(list(model.modules()))
if model_name == "gpt2-large":
    model.to(device)




# load dataset

def attention_rollout(attentions):
    # Initialize with identity matrix
    seq_len = attentions[0].size(2)  # Assuming attentions shape is (batch_size, num_heads, seq_len, seq_len)
    rollout_attention = torch.eye(seq_len).unsqueeze(0).unsqueeze(1)  # shape: (batch_size, 1, seq_len, seq_len)
    rollout_attention = rollout_attention.to('cuda')
    rollout_attention = rollout_attention.float()

    # Iterate over attention weights from top to bottom
    for layer_attention in reversed(attentions):
        # Average attention weights across heads
        avg_attention = layer_attention.mean(dim=1, keepdim=True)  # shape: (batch_size, 1, seq_len, seq_len)
        avg_attention = avg_attention.float()

        # Multiply the rolled-out attention so far with the current layer's attention
        # print('avg_attention:',avg_attention)
        # print('rollout_attention:',rollout_attention)
        rollout_attention = torch.bmm(avg_attention.squeeze(1), rollout_attention.squeeze(1)).unsqueeze(1)

    return rollout_attention

for name in data_name:
    dataset = './data/' + name + '.jsonl'
    print('current dataset:',dataset)
    all_sample = []
    with open(dataset, 'r', encoding='utf-8') as f:
        for line in f:
            json_object = json.loads(line)
            all_sample.append(json_object)

    for ss in tqdm(all_sample):
        sen_input = ss["one_prefix_prefix"].strip()

        # calculate length
        sen_len = len('sen_input split')

        if model_name == "llama-7B":
            if method == "sota" or method == "erasure":
                sen_input = sen_input + ' '
            target = ss["one_prefix_word_good"]
            foil = ss['one_prefix_word_bad']
            # CORRECT_ID = tokenizer(target)['input_ids']
            # print('correct id',CORRECT_ID,tokenizer.convert_ids_to_tokens(CORRECT_ID))
            CORRECT_ID = tokenizer(target)['input_ids'][1]
            FOIL_ID = tokenizer(foil)['input_ids'][1]
            # print('target', target, 'foil', foil, 'CORRECT_ID:', CORRECT_ID, 'decoded:', tokenizer.decode(CORRECT_ID),
            #       'FOIL_ID:', FOIL_ID, tokenizer.decode(FOIL_ID))
        else:
            if method == "sota" or method == "erasure":
                sen_input = sen_input + ' '
            target = ss["one_prefix_word_good"]
            foil = ss['one_prefix_word_bad']
            CORRECT_ID = tokenizer(" " + target)['input_ids'][0]
            FOIL_ID = tokenizer(" " + foil)['input_ids'][0]
            # print('target', target, 'foil', foil, 'CORRECT_ID:', CORRECT_ID, 'decoded:', tokenizer.decode(CORRECT_ID),
            #       'FOIL_ID:', FOIL_ID, tokenizer.decode(FOIL_ID))



        if method == "rollout":
            input_ids = tokenizer(sen_input, return_tensors="pt").input_ids
            input_ids = input_ids.to('cuda')
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True, output_attentions=True)
            hidden_states = outputs.hidden_states
            # target_hidden = hidden_states[k+1]

            attentions = outputs.attentions
            rollout = attention_rollout(attentions)
            rollout = rollout.cpu()
            rollout = np.array(rollout)
            final_rollout = rollout[0][0][-1]
            # print('final_rollout:',final_rollout.shape,final_rollout)
            result['rollout'][name].append(final_rollout)

        elif method == "sota":
            input_tokens = tokenizer(sen_input)['input_ids']
            # print('input_tokens:', input_tokens)
            # print(tokenizer.convert_ids_to_tokens(input_tokens))
            attention_ids = tokenizer(sen_input)['attention_mask']
            base_saliency_matrix, base_embd_matrix = saliency(model, model_name,input_tokens, attention_ids)
            # print('base_saliency_matrix:',len(base_saliency_matrix))
            # print('base_embd_matrix:',len(base_embd_matrix))
            saliency_matrix, embd_matrix = saliency(model, model_name,input_tokens, attention_ids, foil=FOIL_ID)
            base_explanation = input_x_gradient(base_saliency_matrix, base_embd_matrix, normalize=True)
            contra_explanation = input_x_gradient(saliency_matrix, embd_matrix, normalize=True)
            # print('base_explanation:', base_explanation)
            (result['IG_base'])[name].append(base_explanation)
            result['IG_con'][name].append(contra_explanation)

            base_explanation = l1_grad_norm(base_saliency_matrix, normalize=True)
            contra_explanation = l1_grad_norm(saliency_matrix, normalize=True)
            result['GN_base'][name].append(base_explanation)
            result['GN_con'][name].append(contra_explanation)

        elif method == 'ours':

            input_ids = tokenizer(sen_input, return_tensors="pt").input_ids
            input_ids = input_ids.to('cuda')
            # print('input_ids:',input_ids.shape,input_ids)
            # print(tokenizer.decode(input_ids[0]))

            original_input = input_ids

            all_exp = []
            repeat_num = 2

            # repeat_num = len(input_ids.shape[-1])

            for rn in range(repeat_num):
                if rn == 0:
                    input_ids = original_input
                    permuted_indices = None

                    input_ids = input_ids.to('cuda')
                    # print('********** rn:',rn)
                    # print('input_ids:', input_ids.shape, input_ids)
                    # print(tokenizer.decode(input_ids[0]))
                    with torch.no_grad():
                        outputs = model(input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    # print('hidden_states:',len(hidden_states))
                    logits = outputs.logits[0]
                    base_hidden = outputs.hidden_states[0]
                    base_logits = lm_head(base_hidden)[0]
                    base_score = []
                    for pp, qq in zip(input_ids[0], base_logits):


                        current_score = [float(qq[CORRECT_ID]), float(qq[FOIL_ID])]
                        current_score = torch.tensor(current_score)
                        current_score = torch.nn.functional.softmax(current_score, dim=-1)
                        # print('current score:',current_score)
                        diff = current_score[0] - current_score[
                            1]  # should be base (taken from layer 1 or embedding layer) + difference (current context, pervious influence)
                        base_score.append(float(diff))

                        # print('token:', tokenizer.decode(pp), 'target:', target, 'prob:', qq[CORRECT_ID], 'foil:', foil, 'prob:',
                        #       qq[FOIL_ID], 'diff:', diff)

                    learn_score = []
                    tokens = []
                    for xx, yy in zip(input_ids[0], logits):
                        # method 1: yy[CORRECT_ID] * yy[CORRECT_ID]/yy[FOIL_ID] after softmax
                        # method 2: should be softmax

                        current_score = [float(yy[CORRECT_ID]), float(yy[FOIL_ID])]
                        current_score = torch.tensor(current_score)
                        current_score = torch.nn.functional.softmax(current_score, dim=-1)
                        # print('current score:',current_score)
                        diff = current_score[0] - current_score[
                            1]  # should be base (taken from layer 1 or embedding layer) + difference (current context, pervious influence)
                        learn_score.append(float(diff))
                        tokens.append(tokenizer.decode(xx))

                        # print('token:', tokenizer.decode(xx), 'target:', target, 'prob:', yy[CORRECT_ID], 'foil:', foil, 'prob:',
                        #       yy[FOIL_ID], 'diff:', diff)

                    # print('target:',target,'prob:',logits[CORRECT_ID])
                    # print('foil:',foil,'prob:',logits[FOIL_ID])

                    base_score = torch.tensor(base_score)
                    learn_score = torch.tensor(learn_score)
                    final_exp = []
                    for ind in range(len(learn_score)):
                        if ind == 0:
                            final_exp.append(float(learn_score[ind]))
                        else:
                            diff = learn_score[ind] - learn_score[ind - 1]
                            final_exp.append(float(diff))

                    # check

                    final_exp = torch.tensor(final_exp).unsqueeze(0)
                    # print('origianl final exp:', final_exp)

                    if permuted_indices is not None:
                        final_exp = final_exp[0, permuted_indices].unsqueeze(0)

                    final_exp = np.array(final_exp[0])

                else:
                    tensor = input_ids
                    learn_score= []

                    for i in range(1, tensor.shape[1] + 1):
                        new_tensor = tensor[:, -i:]  # This gets the first element
                        # new_tensor = torch.cat((new_tensor, tensor[:, i:]),
                        #                        1)  # This concatenates the remaining elements
                        # print(new_tensor)
                        # print('new_tensor:',tokenizer.decode(new_tensor[0]))
                        with torch.no_grad():
                            outputs = model(new_tensor, output_hidden_states=True)
                        hidden_states = outputs.hidden_states
                        # print('hidden_states:',len(hidden_states))
                        logits = outputs.logits[0]
                        final_logit = logits[-1]
                        # print('final_logit:',final_logit.shape,final_logit)

                        current_score = [float(final_logit[CORRECT_ID]), float(final_logit[FOIL_ID])]
                        current_score = torch.tensor(current_score)
                        current_score = torch.nn.functional.softmax(current_score, dim=-1)
                        # print('current score:',current_score)
                        diff = current_score[0] - current_score[
                            1]  # should be base (taken from layer 1 or embedding layer) + difference (current context, pervious influence)
                        learn_score.append(float(diff))

                    learn_score = torch.tensor(learn_score)
                    final_exp = []
                    for ind in range(len(learn_score)):
                        if ind == 0:
                            final_exp.append(float(learn_score[ind]))
                        else:
                            diff = learn_score[ind] - learn_score[ind - 1]
                            final_exp.append(float(diff))
                    final_exp = np.array(final_exp)
                    # print('final_exp:',final_exp)

                    # print('final_exp = np.array(final_exp[0]):',final_exp.shape,final_exp)
                    final_exp = final_exp[::-1]
                    # print('final_exp FF:', final_exp)
                    # print('final final final_exp',final_exp.shape,final_exp)



                # print('recovered final_exp:',final_exp)
                all_exp.append(final_exp)

            ## start save different samples
            result["ours"][name].append(all_exp[0])

            result["ours_back"][name].append(all_exp[1])
            # print('original exp:',all_exp[1])
            # print('original exp 2:', all_exp)
            all_exp = np.array(all_exp)

            # print('all_exp:',all_exp.shape,all_exp)
            ave_exp = np.mean(all_exp,axis = 0 )
            # print('ave_exp:',ave_exp.shape,ave_exp)
            result["ours_add"][name].append(ave_exp)


out_name = './exp_result/' + model_name + '_result_'+method + save_name_post + '.pt'
torch.save(result,out_name)