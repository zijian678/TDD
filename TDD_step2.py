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
flip_case = 'generate' # pruning generate  ---- generate is the activation task
method = "ours" # ours # sota # erasure # rollout
if method == 'sota':
    result = {'IG_base': {}, 'IG_con': {}, 'GN_base': {}, 'GN_con': {}}
if method == "ours":
    result = {'ours': {},'ours_back':{},'ours_add':{}}
if method == "erasure":
    result = {'erasure': {}}
if method == "rollout":
    result = {'rollout': {}}

# model_name = "Pythia-6.9b"
# data_name = ["anaphor_gender_agreement",'anaphor_number_agreement','animate_subject_passive',
#              'determiner_noun_agreement_1','determiner_noun_agreement_irregular_1',
#              'determiner_noun_agreement_with_adjective_1','determiner_noun_agreement_with_adj_irregular_1',
#              'npi_present_1','distractor_agreement_relational_noun','irregular_plural_subject_verb_agreement_1',
#              'regular_plural_subject_verb_agreement_1']
data_name = ['anaphor_gender_agreement','anaphor_number_agreement']
save_name_post = "all"

###############################################

res_path = './exp_result/' + model_name + '_result_'+method + save_name_post + '.pt'
save_path =  './exp_result/' + 'final_' + model_name + '_result_'+method + save_name_post  + '.pt'
print('save_path:',save_path)
exp_all = torch.load(res_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer_norm = None
if model_name == "gpt2-large":
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    lm_head = model.lm_head
    layer_norm = model.transformer.ln_f
    save_name_prefix = 'exp2_gpt2large_'

if model_name == "gpt-j-6B":

    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16,resume_download=True)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    lm_head = model.lm_head
    # layer_norm = model.transformer.ln_f
    layer_norm = model.transformer.ln_f
    save_name_prefix = 'exp2_gptj'

if model_name == 'llama-7B':
    model_path = "meta-llama/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=" ")
    model = AutoModelForCausalLM.from_pretrained(model_path, token=" ",
                                                 torch_dtype=torch.float16)  # load_in_4bit=True
    lm_head = model.lm_head
    layer_norm = model.model.norm
    save_name_prefix = 'exp2_llama7B'

if model_name == "bloom-7b":
    model_path = 'bigscience/bloom-7b1'
    model = BloomForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,resume_download=True)
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
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, resume_download=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    lm_head = model.embed_out
    layer_norm = model.gpt_neox.final_layer_norm
    save_name_prefix = 'exp2_pythia6b'

# print(list(model.modules()))
if model_name == "gpt2-large":
    model.to(device)


def flip(model, x, token_ids, tokens, target_ids,  fracs, flip_case,random_order = False, tokenizer=None, device='cuda',loss_ids = None):

    x = np.array(x)
    # y_true = y_true.squeeze(0)
    # print('x:',x.shape,x)
    # print('token_ids:',token_ids.shape,token_ids)
    # print('tokens input:',tokenizer.convert_ids_to_tokens(token_ids[0]))
    # # print('y_true:',y_true)
    # print('fracs:',fracs)
    # # print('flip_case:',flip_case)

    if model_name == "llama-7B":
        UNK_IDX = tokenizer.encode(' ')[1]
    else:
        UNK_IDX = tokenizer.encode(' ')[0]


    # print('UNK_IDX:',UNK_IDX,'***',tokenizer.encode(' '),'convert:',tokenizer.convert_tokens_to_ids(' '))
    inputs0 = torch.tensor(token_ids).to(device)
    model = model.to(device)

    model_input = {"input_ids":inputs0.long(),"loss_ids":loss_ids,"meta": None}

    # y0 = model.forward(model_input)[0].squeeze().detach().cpu().numpy()
    # with torch.no_grad():
    #     y0 = model(model_input, output_hidden_states=True)
    # # print('y0:',y0)
    # orig_token_ids = np.copy(token_ids.detach().cpu().numpy())

    if random_order==False:
        inds_sorted = np.argsort(x)[::-1]
        # print('inds_sorted:',inds_sorted)
        # if  flip_case=='generate':
        #     inds_sorted = np.argsort(x)[::-1]
        # elif flip_case=='pruning':
        #     inds_sorted =  np.argsort(np.abs(x))
        # else:
        #     raise
    else:

        inds_ = np.array(list(range(x.shape[-1])))
        remain_inds = np.array(inds_)
        np.random.shuffle(remain_inds)

        inds_sorted = remain_inds

    inds_sorted = inds_sorted.copy()
    # print('inds_sorted:',inds_sorted)
    # vals = x[inds_sorted]

    mse = []
    evidence = []
    # model_outs = {'sentence': tokens, 'y_true':y_true.detach().cpu().numpy(), 'y0':y0}

    # print('x shpape:',x.shape)

    N=len(x)

    evolution = {}
    for frac in fracs:
        inds_generator = iter(inds_sorted)
        n_flip=int(np.ceil(frac*N))
        inds_flip = [next(inds_generator) for i in range(n_flip)]

        if flip_case == 'pruning':

            inputs = inputs0
            for i in inds_flip:
                inputs[:,i] = UNK_IDX

        elif flip_case == 'generate':
            inputs = UNK_IDX*torch.ones_like(inputs0)
            # Set pad tokens
            inputs[inputs0==0] = 0

            for i in inds_flip:
                inputs[:,i] = inputs0[:,i]
        # print('original inputs:', inputs)
        # inputs[:,1] = 83
        # inputs[:, 2] = 50264
        # inputs[:, 3] = 340
        # inputs[:, 4] = 4832

        # print('inputs:',inputs)
        # print('tokens:',tokenizer.convert_ids_to_tokens(inputs.squeeze()))
        # print('tokens:',tokenizer.decode(inputs.squeeze()))

        # model_input = {"input_ids":inputs.long(),"loss_ids":loss_ids,"meta": None}
        model_input = inputs.long()

        # y = model(inputs, labels =  torch.tensor([y_true]*len(token_ids)).long().to(device))['logits'].detach().cpu().numpy()
        # y = model.forward(model_input)[0].squeeze().detach().cpu().numpy()
        y = model(model_input, output_hidden_states=True).logits[0]
        y = y[-1]

        if model_name == "llama-7B":
            target = target_ids[0]
            foil = target_ids[1]
            # CORRECT_ID = tokenizer(target)['input_ids']
            # print('correct id',CORRECT_ID,tokenizer.convert_ids_to_tokens(CORRECT_ID))
            CORRECT_ID = tokenizer(target)['input_ids'][1]
            FOIL_ID = tokenizer(foil)['input_ids'][1]
        else:
            CORRECT_ID = tokenizer(" " + target_ids[0])['input_ids'][0]
            FOIL_ID = tokenizer(" " + target_ids[1])['input_ids'][0]




        # print('CORRECT_ID:',CORRECT_ID,tokenizer.decode(CORRECT_ID),'FOIL_ID:',FOIL_ID,tokenizer.decode(FOIL_ID))

        probs = [float(y[CORRECT_ID]),float(y[FOIL_ID])]
        # print('original probs:',probs)
        probs = torch.tensor(probs)
        probs = torch.nn.functional.softmax(probs,dim = -1)
        # print('final probs:',probs)

        y = probs[0]

        # err = np.sum((y0-y)**2)
        # mse.append(err)
        # evidence.append(softmax(y)[int(y_true)])

      #  print('{:0.2f}'.format(frac), ' '.join(tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy().squeeze())))
        evolution[frac] = (inputs.detach().cpu().numpy(), y)

    # if flip_case == 'generate' and frac == 1.:
    #     assert (inputs0 == inputs).all()
    #
    #
    # model_outs['flip_evolution']  = evolution
    return evolution

# method = "sota" # ours # sota
if method == 'sota':
    res = {'IG_base': {}, 'IG_con': {}, 'GN_base': {}, 'GN_con': {}}
if method == "ours":
    res = {'ours': {},'ours_back':{},'ours_add':{}}

if method == "erasure":
    res = {'erasure': {}}
if method == "rollout":
    res = {'rollout': {}}


for nnn in data_name:
    for kkk in result.keys():
        res[kkk][nnn] = []
print('initial result:',result)

for name in data_name:
    dataset = './data/' + name + '.jsonl'
    print('current dataset:',dataset)
    all_sample = []
    with open(dataset, 'r', encoding='utf-8') as f:
        for line in f:
            json_object = json.loads(line)
            all_sample.append(json_object)

    for kkk in exp_all.keys():
        exps = exp_all[kkk][name]

        for idx in tqdm(range(len(exps))):
            sample = all_sample[idx]


            input = sample["one_prefix_prefix"].strip()
            # print('input:', input)
            sample_length = len(input.split(' '))
            if sample_length < 2:
                continue

            target = sample["one_prefix_word_good"]
            foil = sample['one_prefix_word_bad']
            explanation = exps[idx]
            explanation = explanation.squeeze()
            token_ids = tokenizer(input, return_tensors="pt").input_ids
            target_ids = [target, foil]
            fracs = np.linspace(0, 1, 6)
            # print('fracs:', fracs)

            evolution = flip(model,
                             x=explanation,
                             token_ids=token_ids,
                             tokens=input,
                             target_ids=target_ids,
                             fracs=fracs,
                             flip_case=flip_case,
                             random_order=False,
                             tokenizer=tokenizer, )
            res[kkk][name].append(evolution)

torch.save(res,save_path)
for k2 in res.keys():
    print('current method:',k2)

    current_data = res[k2]
    for k5 in current_data.keys():
        print('current dataset:',k5)
        current_ = res[k2][k5]
        all_acc = []
        for ww in current_:
            sample_acc = []
            for lll in ww.keys():
                # print('ww[lll]:',ww[lll])
                sample_acc.append(ww[lll][1])
            all_acc.append(sample_acc)
        all_acc = np.array(all_acc)
        print('accuracy:', np.mean(all_acc))


# out_name = 'acc_' + k2 + '.pt'
# torch.save(res,out_name)