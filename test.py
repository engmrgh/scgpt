import re
import json
from argparse import ArgumentParser

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import evaluate
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from dataset import SCGPTDatasetForTest


PAD_TOKEN = '<|pad|>'
BOS_TOKEN = '<|startoftext|>'
EOS_TOKEN = '<|endoftext|>'


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
    f1_scores = list()


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

            import pdb
            pdb.set_trace()

            prompt = data['prompt']
            label = data['label']
            for _prompt, _prediction, _label in zip(prompt, prediction, label):
                _prediction = _prediction[_prediction.find(BOS_TOKEN)+len(BOS_TOKEN):].replace(PAD_TOKEN, "").strip()
                _prediction = _prediction[:_prediction.find(EOS_TOKEN)]
                _prediction_entities = re.findall("\[.+?\]", _prediction)

                _label_entities = re.findall("\[.+?\]", _label)
                if len(_label_entities) > 0:
                    entity_f1 = calculate_entity_f1(_label_entities, _prediction_entities)
                    f1_scores.append(entity_f1)
                elif len(_prediction_entities) > 0:
                    f1_scores.append(0)
                else:
                    f1_scores.append(None)

                prompts_list.append(_prompt)
                predictions_list.append(_prediction)
                labels_list.append([_label,])

    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions_list, references=labels_list)
    results['entity_f1'] = avg_f1(f1_scores)

    error_analysis = [{"instruction": inst, "ref": ref[0], "gen": gen, "f1_score": f1} \
                        for inst, ref, gen, f1 in \
                            zip(prompts_list, labels_list, predictions_list, f1_scores)]

    return results, error_analysis


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path, url or short name of the model in huggingface")
    parser.add_argument("--test_dataset", type=str, required=True,
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
        SCGPTDatasetForTest(tokenizer=tokenizer, dataset_path=args.test_dataset)

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
