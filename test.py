import re
import json
from argparse import ArgumentParser

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import evaluate
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from dataset import SCGPTDataset


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


def test(
    tokenizer,
    transformer,
    dataloader,
    device,
    max_output_len
):
    transformer.eval()
    transformer = transformer.to(device)

    instructions_list = list()
    reference_responses = list()
    generated_responses = list()
    f1_scores = list()

    with torch.no_grad():
        for data in tqdm(dataloader,
                         desc="Evaluating",
                         bar_format='{l_bar}{bar:4}{r_bar}{bar:-2b}',
                         leave=True):

            ids = data['input_ids'].to(device, dtype=torch.long)
            ref_response_ids = data['labels'].to(device, dtype=torch.long)

            ids = ids.squeeze(dim=1)
            ref_response_ids = ref_response_ids.squeeze(dim=1)
            ref_response_ids[ref_response_ids==-100] = 0

            instructions = tokenizer.batch_decode(ids)
            ref_responses = tokenizer.batch_decode(ref_response_ids)
            gen_response_ids = transformer.generate(ids, max_new_tokens=max_output_len)
            gen_responses = tokenizer.batch_decode(gen_response_ids)

            for inst, gen_resp, ref_resp in zip(instructions, gen_responses, ref_responses):
                _inst = inst.replace("<pad>", "").replace("</s>", "").strip()
                instructions_list.append(_inst)

                _ref_resp = ref_resp.replace("<pad>", "").replace("</s>", "").strip()
                ref_entities = re.findall("\[.+?\]", ref_resp)
                reference_responses.append([_ref_resp,])

                _gen_resp = gen_resp[:gen_resp.find("</s>")].replace("<pad>", "").replace("</s>", "").strip()
                gen_entities = re.findall("\[.+?\]", _gen_resp)
                generated_responses.append(_gen_resp)

                if len(ref_entities) + len(gen_entities) > 0:
                    entity_f1 = calculate_entity_f1(ref_entities, gen_entities)
                    f1_scores.append(entity_f1)
                else:
                    f1_scores.append(None)

    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=generated_responses, references=reference_responses)
    results['entity_f1'] = avg_f1(f1_scores)

    error_analysis = [{"instruction": inst, "ref": ref[0], "gen": gen, "f1_score": f1} \
                        for inst, ref, gen, f1 in \
                            zip(instructions_list, reference_responses, generated_responses, f1_scores)]

    return results, error_analysis


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path, url or short name of the model in huggingface")
    parser.add_argument("--test_dataset", type=str, required=True,
                        help="Path of the test dataset")

    parser.add_argument("--max_in_seq_length", type=int, default=1024,
                        help="Max input sequence which all sequences will be padded")
    parser.add_argument("--max_out_seq_length", type=int, default=256,
                        help="Max output sequence which all sequences will be padded")
    parser.add_argument("--test_batch_size", type=int,
                        default=5, help="Batch size for testing")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint,
                                              model_max_length=max(args.max_in_seq_length, args.max_out_seq_length))
    transformer = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)

    print("Creating Response Generation Dataset...")
    test_dataset = \
        SCGPTDataset(tokenizer=tokenizer,
                     dataset_path=args.test_dataset,
                     max_in_seq_length=args.max_in_seq_length,
                     max_out_seq_length=args.max_out_seq_length)

    testset_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size)

    result, error_analysis = test(tokenizer=tokenizer,
                                  transformer=transformer,
                                  dataloader=testset_dataloader,
                                  device=args.device,
                                  max_output_len=args.max_out_seq_length)

    with open(f'{args.model_checkpoint}/error_analysis.json', 'w') as f:
        json.dump(error_analysis, f, indent=4, ensure_ascii=False)
    with open(f'{args.model_checkpoint}/result.json', 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("Results")
    print(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
