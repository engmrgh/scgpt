import os
from argparse import ArgumentParser

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from dataset import SCGPTDataset


SPECIAL_TOKENS = [
    # DELEXICALIZE SPECIAL TOKENS
    '[value_arrive]', '[value_address]', '[value_pricerange]', '[value_car]', '[value_day]', '[value_time]',
    '[value_leave]', '[value_food]', '[value_area]', '[value_type]', '[value_phone]', '[value_name]',
    '[value_department]',
    # BELIEF STATE SPECIAL TOKENS
    '[greet]', '[police]', '[recommend]', '[reqmore]', '[offerbook]', '[train]', '[nobook]', '[attraction]',
    '[hospital]', '[general]', '[taxi]', '[restaurant]', '[request]', '[inform]', '[select]', '[nooffer]',
    '[offerbooked]', '[hotel]', '[bye]', '[welcome]',
    # SPECIAL INDICATOR TOKENS
    '<sos_u>', '<eos_u>', '<sos_b>', '<eos_b>', '<sos_a>', '<eos_a>', '<sos_db>', '<eos_db>',
    '<sos_r>', '<eos_r>', '<sos_x>', '<eos_x>' # TODO: remove these!
]


def save_model(args, model, tokenizer, optimizer, name):
    # Save model checkpoint (Overwrite)
    output = f'{args.experiment_name}/{name}'
    if not os.path.exists(output):
        os.makedirs(output)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output)
    tokenizer.save_pretrained(output)

    # Save training arguments together with the trained model
    torch.save(args, os.path.join(output, 'training_args.bin'))
    torch.save(optimizer.state_dict(), os.path.join(output, 'optimizer.pt'))


def generative_training_step(
    tokenizer,
    transformer,
    data,
    device
):
    transformer = transformer.to(device)

    ids = data['input_ids'].to(device, dtype=torch.long)
    attention_mask = data['attention_mask'].to(device, dtype=torch.long)
    lm_labels = data['labels'].to(device, dtype=torch.long)

    ids = ids.squeeze(dim=1)
    attention_mask = attention_mask.squeeze(dim=1)
    lm_labels = lm_labels.squeeze(dim=1)

    outputs = transformer(input_ids=ids, attention_mask=attention_mask, labels=lm_labels)
    return outputs.loss


def validation(
    tokenizer,
    transformer,
    device,
    global_steps,
    tb_logger,
    response_generation_validation_dataloader,
):
    transformer.eval()
    with torch.no_grad():
        total_loss = 0
        total_steps = 0

        for response_generation_data in tqdm(response_generation_validation_dataloader,
                                             desc="Evaluating",
                                             bar_format='{l_bar}{bar:4}{r_bar}{bar:-2b}',
                                             leave=True):
            total_steps += 1
            loss = generative_training_step(tokenizer=tokenizer,
                                            transformer=transformer,
                                            data=response_generation_data,
                                            device=device)
            total_loss += loss.item()

        total_loss /= total_steps
        tb_logger.add_scalars("loss", {"validation": total_loss}, global_steps)
    transformer.train()


def train(
    args,
    tokenizer,
    transformer,
    optimizer,
    gradient_accumulation_steps,
    device,
    tb_logger,
    max_epoch,
    validation_step,
    saving_step,
    response_generation_train_dataloader,
    response_generation_validation_dataloader,
):
    global_steps = 0

    for epoch in range(max_epoch):
        for response_generation_data in tqdm(response_generation_train_dataloader,
                                             desc=f"Training #{epoch}",
                                             bar_format='{l_bar}{bar:4}{r_bar}{bar:-2b}'):
            global_steps += 1
            loss = generative_training_step(tokenizer=tokenizer,
                                            transformer=transformer,
                                            data=response_generation_data,
                                            device=device)
            loss.backward()

            if global_steps % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            tb_logger.add_scalars("loss", {"train": loss.item()}, global_steps)

            if validation_step > 0 and global_steps % validation_step == 0:
                validation(tokenizer=tokenizer,
                           transformer=transformer,
                           response_generation_validation_dataloader=response_generation_validation_dataloader,
                           device=device,
                           tb_logger=tb_logger,
                           global_steps=global_steps)

            if saving_step > 0 and global_steps % saving_step == 0:
                save_model(args, transformer, tokenizer, optimizer,
                           name=f"checkpoint-{global_steps}")


def add_special_tokens_to_model(tokenizer, transformer):
    tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    transformer.resize_token_embeddings(len(tokenizer))
    return tokenizer, transformer


def main():
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", required=True,
                        type=str, help="The unique name of the experiment")
    parser.add_argument("--train_dataset", type=str,
                        required=True, help="Path of the training dataset")
    parser.add_argument('--val_dataset', type=str, required=True,
                        help="Path of the validation dataset.")
    parser.add_argument("--model_checkpoint", type=str, default="metehergul/scgpt",
                        help="Path, url or short name of the model in huggingface")

    parser.add_argument("--lr", type=float,
                        default=5e-5, help="Learning rate")
    parser.add_argument("--max_in_seq_length", type=int, default=1024,
                        help="Max input sequence which all sequences will be padded")
    parser.add_argument("--max_out_seq_length", type=int, default=256,
                        help="Max output sequence which all sequences will be padded")
    parser.add_argument("--train_batch_size", type=int,
                        default=8, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int,
                        default=2, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=8, help="Accumulate gradients on several steps")
    parser.add_argument('--validation_steps', type=int,
                        default=500, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    assert not os.path.exists(args.experiment_name), "Experiment name already exists!"
    tb_logger = SummaryWriter(args.experiment_name)
    tb_logger.add_hparams({
        'train_dataset': args.train_dataset,
        'val_dataset': args.val_dataset,
        'model_checkpoint': args.model_checkpoint,
        'train_batch_size': args.train_batch_size,
        'valid_batch_size': args.valid_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'validation_steps': args.validation_steps,
        'save_steps': args.save_steps,
        'lr': args.lr,
        'n_epochs': args.n_epochs,
        'device': args.device,
        'max_in_seq_length': args.max_in_seq_length,
        'max_out_seq_length': args.max_out_seq_length},
        {'dull_metric': 100})

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint,
                                              model_max_length=max(args.max_in_seq_length, args.max_out_seq_length))

    transformer = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)

    # TODO: Do we need this line? tokenizer, transformer = add_special_tokens_to_model(tokenizer, transformer)

    print("Creating Response Generation Dataset for Training...")
    response_generation_train_dataset = \
        SCGPTDataset(tokenizer=tokenizer,
                     dataset_path=args.train_dataset,
                     max_in_seq_length=args.max_in_seq_length,
                     max_out_seq_length=args.max_out_seq_length)

    response_generation_train_dataloader = DataLoader(response_generation_train_dataset,
                                                      batch_size=args.train_batch_size, shuffle=True)

    print("Creating Response Generation Dataset for Validation...")
    response_generation_validation_dataset = \
        SCGPTDataset(tokenizer=tokenizer,
                     dataset_path=args.val_dataset,
                     max_in_seq_length=args.max_in_seq_length,
                     max_out_seq_length=args.max_out_seq_length)

    response_generation_validation_dataloader = DataLoader(response_generation_validation_dataset,
                                                           batch_size=args.valid_batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.lr)

    train(args=args,
          tokenizer=tokenizer,
          transformer=transformer,
          optimizer=optimizer,
          gradient_accumulation_steps=args.gradient_accumulation_steps,
          tb_logger=tb_logger,
          device=args.device,
          max_epoch=args.n_epochs,
          validation_step=args.validation_steps,
          saving_step=args.save_steps,
          response_generation_train_dataloader=response_generation_train_dataloader,
          response_generation_validation_dataloader=response_generation_validation_dataloader)


if __name__ == "__main__":
    main()
