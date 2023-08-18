import re
import json
import math
from collections import Counter
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
from nltk.util import ngrams
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from dataset import SCGPTDatasetForTest


PAD_TOKEN = '<|pad|>'
BOS_TOKEN = '<|startoftext|>'
EOS_TOKEN = '<|endoftext|>'


NONE_CATEGORICAL_SLOTS = [
    'hotel-name',
    'hotel-address',
    'hotel-phone',
    'hotel-postcode',
    'hotel-ref',
    'train-arriveby',
    'train-leaveat',
    'train-trainid',
    'train-ref',
    'train-price',
    'train-duration',
    'attraction-name',
    'attraction-entrancefee',
    'attraction-openhours',
    'attraction-address',
    'attraction-phone',
    'attraction-postcode',
    'restaurant-food',
    'restaurant-name',
    'restaurant-booktime',
    'restaurant-address',
    'restaurant-phone',
    'restaurant-postcode',
    'restaurant-ref',
    'hospital-department',
    'hospital-address',
    'hospital-phone',
    'hospital-postcode',
    'taxi-leaveat',
    'taxi-destination',
    'taxi-departure',
    'taxi-arriveby',
    'taxi-type',
    'taxi-phone',
    'bus-departure',
    'bus-destination',
    'bus-leaveat',
    'police-address',
    'police-phone',
    'police-postcode',
    # Booking-extra-domain-which-spans-domains-hotel-restaurant-taxi-train
    'booking-name',
    'booking-booktime',
    'booking-ref',
    'booking-address',
    'booking-phone',
    'booking-postcode',
    'booking-arriveby',
    'booking-leaveat',
    'booking-trainid',
    'booking-price',
    'booking-duration',
    'booking-food',
    'booking-destination',
    'booking-departure',
    'booking-type'
]

slot_name_correction_mapping = {
    "addr": "address",
    "arrive": "arriveby",
    "dest": "destination",
    "leave": "leaveat",
    "depart": "departure",
    "id": "trainid",
    "fee": "entrancefee",
    "open": "openhours",
    "post": "postcode",
    "car": "type",
    "ticket": "price"
}

domain_slot_correction_mapping = {
    "restaurant-time": "restaurant-booktime",
    "train-time": "train-duration",
    "booking-time": "booking-booktime"
}


def calculate_entity_f1(ref_entities, gen_entities):
    entity_p = 0
    if len(gen_entities) > 0:
        for ent in gen_entities:
            if ent in ref_entities:
                entity_p += 1
        entity_p /= len(gen_entities)

    entity_r = 0
    if len(ref_entities) > 0:
        for ent in ref_entities:
            if ent in gen_entities:
                entity_r += 1
        entity_r /= len(ref_entities)

    entity_f1 = 0
    if entity_p + entity_r:
        entity_f1 = 2 * entity_p * entity_r / (entity_r + entity_p)

    return entity_f1


def avg_f1(f1_scores):
    sum_f1 = 0
    cnt = 0
    for f1 in f1_scores:
        if f1:
            sum_f1 += f1
            cnt += 1
    return sum_f1 / cnt


def calculate_slot_error_rate_wen_definition(tokenizer, sys_act_dict, resp):
    slot_error_rate = {
        "total": 0,
        "miss": 0,
        "duplicate": 0
    }
    for domain_act, slot_values in sys_act_dict.items():
        domain, act = domain_act.split('-')
        for slot_value in slot_values:
            slot, value = slot_value[0], slot_value[1]

            _slot = slot.lower()
            _slot = slot_name_correction_mapping.get(_slot, _slot)
            domain_slot = f"{domain}-{_slot}".lower()
            _domain_slot = domain_slot_correction_mapping.get(domain_slot, domain_slot)

            if (_domain_slot in NONE_CATEGORICAL_SLOTS) and (value not in ['?', 'none']):
                slot_error_rate["total"] += 1
                _value = value.lower()
                _value = tokenizer.decode(tokenizer(_value)['input_ids'])
                matches = re.findall(_value, resp)
                if len(matches) == 0:
                    slot_error_rate["miss"] += 1
                elif len(matches) > 1:
                    slot_error_rate["duplicate"] += 1
    return slot_error_rate


def calculate_slot_error_rate_kale_definition(tokenizer, sys_act_dict, resp):
    slot_error_rate = {
        "contains_non_categorical_slot": False,
        "contains_error_slot": False,
    }
    for domain_act, slot_values in sys_act_dict.items():
        domain, act = domain_act.split('-')
        for slot_value in slot_values:
            slot, value = slot_value[0], slot_value[1]

            _slot = slot.lower()
            _slot = slot_name_correction_mapping.get(_slot, _slot)
            domain_slot = f"{domain}-{_slot}".lower()
            _domain_slot = domain_slot_correction_mapping.get(domain_slot, domain_slot)

            if (_domain_slot in NONE_CATEGORICAL_SLOTS) and (value not in ['?', 'none']):
                slot_error_rate["contains_non_categorical_slot"] = True
                _value = value.lower()
                _value = tokenizer.decode(tokenizer(_value)['input_ids'])
                matches = re.findall(_value, resp)
                if len(matches) == 0:
                    slot_error_rate["contains_error_slot"] = True
    return slot_error_rate


def sentence_bleu_4(parallel_corpus):
    # input : single sentence, multiple references
    count = [0,0,0,0]
    clip_count = [0,0,0,0]
    weights=[0.25,0.25,0.25,0.25]
    r = 0
    c = 0

    # accumulate ngram statistics
    for hyps, refs in parallel_corpus:
        hyps = [hyp.split() for hyp in hyps]
        refs = [ref.split() for ref in refs]
        # compute ngram counts by matching each hypothesis
        for hyp in hyps:
            # for each ngram
            for i in range(4):
                # accumulate hyp ngram counts
                hypcnts = Counter(ngrams(hyp,i+1))
                cnt = sum(hypcnts.values())
                count[i] += cnt

                # compute clipped counts
                max_counts = {}
                # compare to each reference
                for ref in refs:
                    # get reference ngrams
                    refcnts = Counter(ngrams(ref, i+1))
                    # for each ngram
                    for ng in hypcnts:
                        # clipped counts
                        max_counts[ng] = max( max_counts.get(ng,0),refcnts[ng] )
                # compute clipped counts by clipping the hyp count if necessary
                clipcnt = dict( (ng,min(count,max_counts[ng])) \
                        for ng,count in hypcnts.items() )
                clip_count[i] += sum(clipcnt.values())

            # accumulate r & c, find best match among all references
            bestmatch = [1000,1000]
            for ref in refs:
                if bestmatch[0]==0: break
                # length difference
                diff = abs(len(ref)-len(hyp))
                # if the current diff less than stored one, change it
                if diff<bestmatch[0]:
                    bestmatch[0] = diff
                    bestmatch[1] = len(ref)
            # extract the best length match in references
            r += bestmatch[1]
            c += len(hyp)

    # for numerical stability
    p0 = 1e-7
    # modified brevity penality
    bp = math.exp(-abs(1.0-float(r)/float(c+p0)))
    # smoothed version of modified prec.
    p_ns = [0,0,0,0]
    for i in range(4):
        if i<2: # original version n-gram counts
            p_ns[i] = float(clip_count[i])/float(count[i]+p0)+p0
        else: # smoothed version of ngram counts
            smooth_term = 5*p_ns[i-1]*p_ns[i-1]/p_ns[i-2]
            p_ns[i] = float(clip_count[i]+smooth_term)/float(count[i]+5)+p0
    # weighted prec.
    s = math.fsum(w*math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)
    # final sentence bleu score
    bleu_hyp = bp*math.exp(s)
    return bleu_hyp


def average_wen_slot_error_rate(slot_error_rates):
    total = sum([ser['total'] for ser in slot_error_rates])
    miss = sum([ser['miss'] for ser in slot_error_rates])
    duplicates = sum([ser['duplicate'] for ser in slot_error_rates])
    return (miss + duplicates) / total


def average_kale_slot_error_rate(slot_error_rates):
    total = sum([1 if ser['contains_non_categorical_slot'] else 0 for ser in slot_error_rates])
    error = sum([1 if ser['contains_error_slot'] else 0 for ser in slot_error_rates])
    return error / total


def test(
    tokenizer,
    model,
    dataloader,
    device,
    max_new_tokens
):
    model.eval()
    model = model.to(device)

    prompts_list = list()
    labels_list = list()
    predictions_list = list()
    wen_definitions_slot_error_rates = list()
    kale_definitions_slot_error_rates = list()


    with torch.no_grad():
        for data in tqdm(dataloader,
                         desc="Evaluating",
                         bar_format='{l_bar}{bar:4}{r_bar}{bar:-2b}',
                         leave=True):

            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            input_ids = input_ids.squeeze(dim=1)
            attention_mask = attention_mask.squeeze(dim=1)
            output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
            prediction = tokenizer.batch_decode(output_ids)

            prompt = data['prompt']
            label = data['label']
            sys_act_dict = [json.loads(i) for i in data['sys_act_dict']]
            for _prompt, _sys_act_dict, _prediction, _label in zip(prompt, sys_act_dict, prediction, label):
                _prediction = _prediction[_prediction.find(BOS_TOKEN)+len(BOS_TOKEN):].replace(PAD_TOKEN, "").strip()
                _prediction = _prediction[:_prediction.find(EOS_TOKEN)]

                ser_wen_definition = calculate_slot_error_rate_wen_definition(tokenizer, _sys_act_dict, _prediction)
                wen_definitions_slot_error_rates.append(ser_wen_definition)
                ser_kale_definition = calculate_slot_error_rate_kale_definition(tokenizer, _sys_act_dict, _prediction)
                kale_definitions_slot_error_rates.append(ser_kale_definition)

                prompts_list.append(_prompt)
                predictions_list.append(_prediction)
                labels_list.append([_label,])

    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions_list, references=labels_list)
    results['bleu-4'] = sentence_bleu_4([[[gen,], ref] for gen, ref in zip(predictions_list, labels_list)])
    results['wen_ser'] = average_wen_slot_error_rate(wen_definitions_slot_error_rates)
    results['kale_ser'] = average_kale_slot_error_rate(kale_definitions_slot_error_rates)

    error_analysis = [{"instruction": inst, "ref": ref[0], "gen": gen,
                       "ser_wen_definition": wen_ser, "ser_kale_definition": kale_ser} \
                        for inst, ref, gen, wen_ser, kale_ser in \
                            zip(prompts_list, labels_list, predictions_list,
                                wen_definitions_slot_error_rates, kale_definitions_slot_error_rates)]

    return results, error_analysis


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path, url or short name of the model in huggingface")
    parser.add_argument("--test_dataset", type=str, required=True,
                        help="Path of the test dataset")
    parser.add_argument("--dataset_type", type=str, required=True,
                        help="Path of the test dataset")

    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.")
    parser.add_argument("--batch_size", type=int,
                        default=5, help="Batch size for testing")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.padding_side = "left" # In order to be able to use batching while predicting
    tokenizer.add_special_tokens({'pad_token': PAD_TOKEN,
                                  'bos_token': BOS_TOKEN,
                                  'eos_token': EOS_TOKEN})
    model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
    model.resize_token_embeddings(len(tokenizer))

    print("Creating Response Generation Dataset...")
    test_dataset = \
        SCGPTDatasetForTest(tokenizer=tokenizer, dataset_type=args.dataset_type, dataset_path=args.test_dataset)

    testset_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    result, error_analysis = test(tokenizer=tokenizer,
                                  model=model,
                                  dataloader=testset_dataloader,
                                  device=args.device,
                                  max_new_tokens=args.max_new_tokens)

    with open(f'{args.model_checkpoint}/error_analysis.json', 'w') as f:
        json.dump(error_analysis, f, indent=4, ensure_ascii=False)
    with open(f'{args.model_checkpoint}/result.json', 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("Results")
    print(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
