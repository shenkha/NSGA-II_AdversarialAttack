import sys
sys.dont_write_bytecode = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # avoid tensorflow warnings
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
from typing import List
import torch
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)
import random
import numpy as np
import torch
from nltk import wsd
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from spacy.cli import download
from spacy import load
import warnings
from typing import Union, List, Tuple
from datasets import load_dataset, Dataset
import evaluate
from DialogueAPI import dialogue
import nsga2_new
# from attacker.DGSlow import WordAttacker, StructureAttacker
# from attacker.PWWS import PWWSAttacker
# from attacker.SCPN import SCPNAttacker
# from attacker.VIPER import VIPERAttacker
# from attacker.BAE import BAEAttacker
# from attacker.FD import FDAttacker
# from attacker.HotFlip import HotFlipAttacker
# from attacker.TextBugger import TextBuggerAttacker
# from attacker.MAYA import MAYAAttacker
# from attacker.UAT import UATAttacker
from DG_dataset import DGDataset
import re
DATA2NAME = {
    "blended_skill_talk": "BST",
    "conv_ai_2": "ConvAI2",
    "empathetic_dialogues": "ED",
    "AlekseyKorshuk/persona-chat": "PC",
}


from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
softmax = nn.Softmax(dim=1)
bce_loss = nn.BCELoss()

# Initialize logging and downloads
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')



class SentenceEncoder:
    def __init__(self, model_name='paraphrase-distilroberta-base-v1', device='cpu'):
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device

    def encode(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        # Ensure sentences are on the correct device
        return self.model.encode(sentences, convert_to_tensor=True,
                                 show_progress_bar = False,
                                 device=self.device)

    def get_sim(self, sentence1, sentence2):
        embeddings = self.encode([sentence1, sentence2])
        cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return cos_sim.item()

    def find_best_match(self, original_sentence, candidate_sentences, find_min=False):
        original_embedding = self.encode(original_sentence)
        candidate_embeddings = self.encode(candidate_sentences)
        best_candidate = None
        best_index = None
        best_sim = float('inf') if find_min else float('-inf')

        for i, candidate_embedding in enumerate(candidate_embeddings):
            sim = util.pytorch_cos_sim(original_embedding, candidate_embedding).item()
            if find_min:
                if sim < best_sim:
                    best_sim = sim
                    best_candidate = candidate_sentences[i]
                    best_index = i
            else:
                if sim > best_sim:
                    best_sim = sim
                    best_candidate = candidate_sentences[i]
                    best_index = i

        return best_candidate, best_index, best_sim

class DGAttackEval(DGDataset):
    def __init__(self, 
        args: argparse.Namespace = None, 
        tokenizer: AutoTokenizer = None, 
        model: AutoModelForSeq2SeqLM = None, 
        #attacker: WordAttacker = None, 
        device: torch.device('cpu') = None, 
        task: str = 'seq2seq', 
        bleu: evaluate.load("bleu") = None, 
        rouge: evaluate.load("rouge") = None,
        meteor: evaluate.load("meteor") = None,
        ):
            
            super(DGAttackEval, self).__init__(
                dataset=args.dataset,
                task=task,
                tokenizer=tokenizer,
                max_source_length=args.max_len,
                max_target_length=args.max_len,
                padding=None,
                ignore_pad_token_for_loss=True,
                preprocessing_num_workers=None,
                overwrite_cache=True,
            )
            self.args = args
            self.model = model
            self.device = args.device

            self.num_beams = args.num_beams
            self.num_beam_groups = args.num_beam_groups
            self.max_num_samples = args.max_num_samples

            self.bleu = bleu
            self.rouge = rouge
            self.meteor = meteor

            self.sentencoder = SentenceEncoder(device=args.device)

            self.ori_lens, self.adv_lens = [], []
            self.ori_bleus, self.adv_bleus = [], []
            self.ori_rouges, self.adv_rouges = [], []
            self.ori_meteors, self.adv_meteors = [], []
            self.ori_time, self.adv_time = [], []
            self.cos_sims = []
            self.att_success = 0
            self.total_pairs = 0

            # self.record = []
            #att_method = args.attack_strategy
            out_dir = args.out_dir
            model_n = args.model_name_or_path.split("/")[-1]
            dataset_n = DATA2NAME.get(args.dataset, args.dataset.split("/")[-1])
            #combined = "combined" if args.use_combined_loss and att_method == 'structure' else "single"
            #max_per = args.tas
            #fitness = args.fitness if att_method == 'structure' else 'performance'
            select_beams = args.select_beams
            max_num_samples = args.max_num_samples
            num_gen = args.num_gen
            num_ind = args.num_ind
            att_method = "NSGA-II_newObj_newBERT_" +  str(num_gen)  + "gen_"  + str(num_ind) + "ind_" 
            file_path = f"{out_dir}/{att_method}_{select_beams}_{model_n}_{dataset_n}_{max_num_samples}.txt"
            self.write_file_path = file_path


    def log_and_save(self, display: str):
        print(display)
        with open(self.write_file_path, 'a') as f:
            f.write(display + "\n")
        #self.write_file.write(display + "\n")   
    
    def get_prediction(self, text: str):
        if self.task == 'seq2seq':
            effective_text = text
        else:
            effective_text = text + self.tokenizer.eos_token

        inputs = self.tokenizer(
            effective_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length-1,
        )
        input_ids = inputs.input_ids.to(args.device)
        self.model = self.model.to(args.device)
        t1 = time.time()
        with torch.no_grad():
            outputs = dialogue(
                self.model,
                input_ids,
                early_stopping=False,
                num_beams=self.num_beams,
                num_beam_groups=self.num_beam_groups,
                use_cache=True,
                max_length=self.max_target_length,
            )
        if self.task == 'seq2seq':
            output = self.tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)[0]
        else:
            output = self.tokenizer.batch_decode(
                outputs['sequences'][:, input_ids.shape[-1]:],
                skip_special_tokens=True,
            )[0]
        t2 = time.time()
        return output.strip(), t2 - t1


    def eval_metrics(self, output: str, guided_messages: List[str]):
        if not output:
            return

        bleu_res = self.bleu.compute(
            predictions=[output],
            references=[guided_messages],
            smooth=True,
        )
        rouge_res = self.rouge.compute(
            predictions=[output],
            references=[guided_messages],
        )
        meteor_res = self.meteor.compute(
            predictions=[output],
            references=[guided_messages],
        )
        pred_len = bleu_res['translation_length']
        return bleu_res, rouge_res, meteor_res, pred_len


    def generation_step(self, instance: dict):
        # Set up
        num_entries, total_entries, context, prev_utt_pc = self.prepare_context(instance)
        for entry_idx in range(num_entries):
            free_message, guided_message, original_context, references = self.prepare_entry(
                instance,
                entry_idx,
                context,
                prev_utt_pc,
                total_entries,
            )
            if guided_message is None:
                continue

            prev_utt_pc += [
                free_message,
                guided_message,
            ]
            self.log_and_save("\nDialogue history: {}".format(original_context))
            self.log_and_save("U--{} \n(Ref: ['{}', ...])".format(free_message, references[-1]))
            # Original generation
            eos_token = self.tokenizer.eos_token
            text = original_context + eos_token + free_message
            output, time_gap = self.get_prediction(text)
            self.log_and_save("G--{}".format(output))

            if not output:
                continue
            #print("OUTPUT:",output)
            #print("REF:",references)
            bleu_res, rouge_res, meteor_res, pred_len = self.eval_metrics(output, references)
            self.log_and_save("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
                pred_len, time_gap, bleu_res['bleu'], rouge_res['rougeL'], meteor_res['meteor'],
            ))
            self.ori_lens.append(pred_len)
            self.ori_bleus.append(bleu_res['bleu'])
            self.ori_rouges.append(rouge_res['rougeL'])
            self.ori_meteors.append(meteor_res['meteor'])
            self.ori_time.append(time_gap)

            # Attack
            #success, adv_his = self.attacker.run_attack(text, guided_message)
            #new_text = adv_his[-1][0]
            #new_free_message = new_text.split(self.sp_token)[1].strip()
            #cos_sim = self.attacker.sent_encoder.get_sim(new_free_message, free_message)
            self.model = self.model.to(args.device)
            if args.crossover_flag == 1:
                print("BAT DAU NSGA-II VOI CROSSOVER")
            else:
                print("BAT DAU NSGA-II")

            # if args.objective == "cls":
            #     pop, pop_fitness, best_fitness = self.POPOP(original_context, free_message, guided_message, self.objective_cls, self.tournament_selection ,num_individuals= args.num_ind, max_evaluations= 5 * args.num_ind)
            # else:
            #     pop, pop_fitness, best_fitness = self.POPOP(original_context, free_message, guided_message, self.objective_eos, self.tournament_selection ,num_individuals= args.num_ind, max_evaluations= 5 * args.num_ind)

            # pop_with_fitness = list(zip(pop_fitness, pop))
            # sorted_pop_with_fitness = sorted(pop_with_fitness, key=lambda x: x[0])
            # best_individual = sorted_pop_with_fitness[0][1]
            # best_fitness_value = sorted_pop_with_fitness[0][0]
            # print("Pop:", pop)
            # print("Candidate:", best_individual)
            
            problem = nsga2_new.Problem(self.model, self.tokenizer,original_context, free_message, guided_message, self.device)

            evolution = nsga2_new.Evolution(args.crossover_flag, self.write_file_path, problem, num_of_generations=args.num_gen, num_of_individuals=args.num_ind, num_of_tour_particips=2,
                      tournament_prob=0.9, crossover_param=2, mutation_param=5 )

            resulting_front = evolution.evolve()
            result = []
            for individual in resulting_front:
                result.append((individual.sentence,individual.accuracy, individual.length))
                #print(individual.sentence, individual.cls_loss, individual.eos_loss)
            data_with_fitness = [(sentence, accuracy, length, length / accuracy) for sentence, accuracy, length in result]

            # Sort based on the fitness score (fourth tuple element), in descending order
            sorted_data = sorted(data_with_fitness, key=lambda x: x[3], reverse=True)
            #sorted_data = sorted(result, key=lambda x: x[1])
            new_free_message = sorted_data[0][0]

            new_text = original_context + self.tokenizer.eos_token + new_free_message
            cos_sim = self.sentencoder.get_sim(new_free_message, free_message)
            output, time_gap = self.get_prediction(new_text)
            if not output:
                continue

            self.log_and_save("U'--{} (cosine: {:.3f})".format(new_free_message, cos_sim))
            self.log_and_save("G'--{}".format(output))
            adv_bleu_res, adv_rouge_res, adv_meteor_res, adv_pred_len = self.eval_metrics(output, references)

            # ASR
            success = (
                (bleu_res['bleu'] > adv_bleu_res['bleu']) or
                (rouge_res['rougeL'] > adv_rouge_res['rougeL']) or
                (meteor_res['meteor'] > adv_meteor_res['meteor'])
                #) and cos_sim > 0.01
                ) and cos_sim > 0.7
            if success:
                self.att_success += 1
            else:
                self.log_and_save("Attack failed!")

            self.log_and_save("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
                adv_pred_len, time_gap, adv_bleu_res['bleu'], adv_rouge_res['rougeL'], adv_meteor_res['meteor'],
            ))
            self.adv_lens.append(adv_pred_len)
            self.adv_bleus.append(adv_bleu_res['bleu'])
            self.adv_rouges.append(adv_rouge_res['rougeL'])
            self.adv_meteors.append(adv_meteor_res['meteor'])
            self.adv_time.append(time_gap)
            self.cos_sims.append(cos_sim)
            self.total_pairs += 1

    def adv_load_metrics_and_find_last_entry(self,log_file_path):
        metrics = {
            "adv_lens": [],
            "adv_bleus": [],
            "adv_rouges": [],
            "adv_meteors": [],
            "adv_times": [],
            "cos_sims": [],
            "total_pairs": 0,
            "att_success": 0,
        }
        last_dialogue = None
        capture_next_metrics = False  # Flag to determine if next metrics should be captured
        attack_failed_pattern = re.compile(r'Attack failed!')
        attack_failed = 0
        with open(log_file_path, "r") as file:
            for line in file:
                if "Dialogue history:" in line:
                    last_dialogue = line.strip().split(": ")[1]

                if "U'--" in line:
                    # Capture cosine similarity
                    cos_sim_match = re.search(r'cosine: ([0-9.]+)', line)
                    if cos_sim_match:
                        metrics['cos_sims'].append(float(cos_sim_match.group(1)))
                        if float(cos_sim_match.group(1)) < 0.7:
                            attack_failed +=1
                    capture_next_metrics = True  # Set flag to capture next metrics

                elif attack_failed_pattern.search(line):
                    #metrics['att_success'] += 1  # Counting successful attacks inversely by 'Attack failed!'
                    attack_failed += 1

                elif capture_next_metrics:
                    metric_match = re.search(r'\(length: (\d+), latency: ([0-9.]+), BLEU: ([0-9.]+), ROUGE: ([0-9.]+), METEOR: ([0-9.]+)\)', line)
                    if metric_match:
                        metrics['adv_lens'].append(float(metric_match.group(1)))
                        metrics['adv_times'].append(float(metric_match.group(2)))
                        metrics['adv_bleus'].append(float(metric_match.group(3)))
                        metrics['adv_rouges'].append(float(metric_match.group(4)))
                        metrics['adv_meteors'].append(float(metric_match.group(5)))
                        metrics['total_pairs'] += 1  # Increment counter for each adv sample processed
                        capture_next_metrics = False  # Reset flag after capturing
        metrics['att_success'] = metrics['total_pairs'] - attack_failed  # Adjusting successful attack count
        return last_dialogue, metrics

    import re

    def ori_load_metrics_and_find_last_entry(self, log_file_path):
        metrics = {
            "ori_lens": [],
            "ori_bleus": [],
            "ori_rouges": [],
            "ori_meteors": [],
            "ori_times": [],
            "total_pairs": 0
        }
        last_dialogue = None
        capture_next_metrics = False
        with open(log_file_path, "r") as file:
            for line in file:
                if "Dialogue history:" in line:
                    last_dialogue = line.strip().split(": ")[1]

                if "U--" in line:
                    capture_next_metrics = True
                elif capture_next_metrics:
                    metric_match = re.search(r'\(length: (\d+), latency: ([0-9.]+), BLEU: ([0-9.]+), ROUGE: ([0-9.]+), METEOR: ([0-9.]+)\)', line)
                    if metric_match:
                        metrics['ori_lens'].append(int(metric_match.group(1)))
                        metrics['ori_times'].append(float(metric_match.group(2)))
                        metrics['ori_bleus'].append(float(metric_match.group(3)))
                        metrics['ori_rouges'].append(float(metric_match.group(4)))
                        metrics['ori_meteors'].append(float(metric_match.group(5)))
                        metrics['total_pairs'] += 1  # Increment each time an original sentence metrics are captured
                        capture_next_metrics = False

        return last_dialogue, metrics

    def find_start_index(self,test_dataset, last_dialogue):
        for i, instance in tqdm(enumerate(test_dataset)):
            num_entries, total_entries, context, prev_utt_pc = self.prepare_context(instance)
            for entry_idx in range(num_entries):
                free_message, guided_message, original_context, references = self.prepare_entry(
                        instance,
                        entry_idx,
                        context,
                        prev_utt_pc,
                        total_entries,
                )
                if guided_message is None:
                    continue

                prev_utt_pc += [
                    free_message,
                    guided_message,
                ]

                if original_context == last_dialogue:
                    return i, original_context
                    # Log for debugging
                #print(f"Checking context: {original_context} against last_dialogue: {last_dialogue}")
                print("\nDialogue history: {}".format(original_context))
                print("\nLast Dialogue: {}".format(last_dialogue))
        return -1, None  # If no match found
    

    def generation(self, test_dataset: Dataset):
        last_dialogue = None
        if args.resume:
            start_index = 0
            ids = random.sample(range(len(test_dataset)), self.max_num_samples)
            test_dataset = test_dataset.select(ids)
            last_dialogue, adv_metrics = self.adv_load_metrics_and_find_last_entry(self.args.resume_log_dir)
            last_dialogue, ori_metrics = self.ori_load_metrics_and_find_last_entry(self.args.resume_log_dir)
            # Extend current metrics with the loaded ones
            self.ori_lens.extend(adv_metrics['adv_lens'])
            self.adv_time.extend(adv_metrics['adv_times'])
            self.adv_bleus.extend(adv_metrics['adv_bleus'])
            self.adv_rouges.extend(adv_metrics['adv_rouges'])
            self.adv_meteors.extend(adv_metrics['adv_meteors'])
            self.cos_sims.extend(adv_metrics['cos_sims'])
            self.total_pairs += adv_metrics['total_pairs']

            self.ori_lens.append(ori_metrics['ori_lens'])
            self.ori_bleus.append(ori_metrics['ori_bleus'])
            self.ori_rouges.append(ori_metrics['ori_rouges'])
            self.ori_meteors.append(ori_metrics['ori_meteors'])
            self.ori_time.append(ori_metrics['ori_times'])

            # Update log file to a new file to avoid overlap
            #new_log_filename = os.path.splitext(self.args.resume_log_dir)[0] + "_continued.txt"
            #self.write_file = open(new_log_filename, "w")
            print(f"Resuming from: {last_dialogue}, logging to new file")

            # ids = random.sample(range(len(test_dataset)), self.max_num_samples)
            # test_dataset = test_dataset.select(ids)
            print("Test dataset: ", test_dataset)
            print("LENGTH DATASET:", len(test_dataset))

            if last_dialogue:
            # Iterate through dataset to find where this history matches
                index, found_context = self.find_start_index(test_dataset, last_dialogue)
                if index != -1:
                    print(f"Resuming from index: {index}, context: {found_context}")
                    start_index = index
                else:
                    print("No matching context found. Please check the 'last_dialogue' or dataset processing.")

            for i, instance in tqdm(enumerate(test_dataset)):
                if i >= start_index:
                    self.generation_step(instance)

        else:
            ids = random.sample(range(len(test_dataset)), self.max_num_samples)
            test_dataset = test_dataset.select(ids)
            print("Test dataset: ", test_dataset)
            for i, instance in tqdm(enumerate(test_dataset)):
                self.generation_step(instance)

        Ori_len = np.mean(self.ori_lens)
        Adv_len = np.mean(self.adv_lens)
        Ori_bleu = np.mean(self.ori_bleus)
        Adv_bleu = np.mean(self.adv_bleus)
        Ori_rouge = np.mean(self.ori_rouges)
        Adv_rouge = np.mean(self.adv_rouges)
        Ori_meteor = np.mean(self.ori_meteors)
        Adv_meteor = np.mean(self.adv_meteors)
        Cos_sims = np.mean(self.cos_sims)
        Ori_t = np.mean(self.ori_time)
        Adv_t = np.mean(self.adv_time)

        # Summarize eval results
        self.log_and_save("\nOriginal output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
            Ori_len, Ori_t, Ori_bleu, Ori_rouge, Ori_meteor,
        ))
        self.log_and_save("Perturbed [cosine: {:.3f}] output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
            Cos_sims, Adv_len, Adv_t, Adv_bleu, Adv_rouge, Adv_meteor,
        ))
        self.log_and_save("Attack success rate: {:.2f}%".format(100*self.att_success/self.total_pairs))

    
def main(args: argparse.Namespace):
        random.seed(args.seed)
        model_name_or_path = args.model_name_or_path
        dataset = args.dataset
        max_len = args.max_len
        max_per = args.max_per
        num_beams = args.num_beams
        select_beams = args.select_beams
        #fitness = args.fitness
        num_beam_groups = args.num_beam_groups
    #     att_method = args.attack_strategy
    #     cls_weight = args.cls_weight
    #     eos_weight = args.eos_weight
    #     delta = args.delta
    #     use_combined_loss = args.use_combined_loss
        out_dir = args.out_dir

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # device = torch.device('cpu')
        config = AutoConfig.from_pretrained(model_name_or_path, num_beams=num_beams, num_beam_groups=num_beam_groups)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if 'gpt' in model_name_or_path.lower():
            task = 'clm'
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
            if 'results' not in model_name_or_path.lower():
                tokenizer.add_special_tokens({'pad_token': '<PAD>'})
                tokenizer.add_special_tokens({'mask_token': '<MASK>'})
                model.resize_token_embeddings(len(tokenizer))
        else:
            task = 'seq2seq'
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)

        # Load dataset
        all_datasets = load_dataset(dataset)
        if dataset == "conv_ai_2":
            test_dataset = all_datasets['train']
        elif dataset == "AlekseyKorshuk/persona-chat":
            test_dataset = all_datasets['validation']
        else:
            test_dataset = all_datasets['test']

        # Load evaluation metrics
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        meteor = evaluate.load("meteor")

        # Define DG attack
        dg = DGAttackEval(
            args=args,
            tokenizer=tokenizer,
            model=model,
            #attacker=attacker,
            device= args.device,
            task=task,
            bleu=bleu,
            rouge=rouge,
            meteor=meteor,
        )
        dg.generation(test_dataset)

        # # Save generation files
        # model_n = model_name_or_path.split("/")[-1]
        # dataset_n = DATA2NAME.get(dataset, dataset.split("/")[-1])
        # #combined = "combined" if use_combined_loss else "eos"
        # combined = "POPOP"
        # file_path = f"{out_dir}/{combined}_{model_n}_{dataset_n}.txt"
        # with open(file_path, "w") as f:
        #     for line in dg.record:
        #         f.write(str(line) + "\n")
        # f.close()
    
if __name__ == "__main__":
    import ssl
    import argparse
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    import nltk
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    # nltk.download('averaged_perceptron_tagger')
    ssl._create_default_https_context = ssl._create_unverified_context

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num_samples", type=int, default=5, help="Number of samples to attack")
    parser.add_argument("--max_per", type=int, default=5, help="Number of perturbation iterations per sample")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum length of generated sequence")
    parser.add_argument("--select_beams", type=int, default=2, help="Number of sentence beams to keep for each attack iteration")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for decoding in LLMs")
    parser.add_argument("--num_beam_groups", type=int, default=1, help="Number of beam groups for decoding in LLMs")
#     parser.add_argument("--fitness", type=str, default="adaptive",
#                         choices=["performance", "length", "random", "combined", "adaptive"],
#                         help="Fitness function for selecting the best candidate")
    parser.add_argument("--model_name_or_path", "-m", type=str, default="./DGSlow_Bartbase", help="Path to model")
    parser.add_argument("--dataset", "-d", type=str, default="blended_skill_talk",
                        choices=["blended_skill_talk", "conv_ai_2", "empathetic_dialogues", "AlekseyKorshuk/persona-chat"],
                        help="Dataset to attack")
    parser.add_argument("--out_dir", type=str,
                        default="./results/logging",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=2019, help="Random seed")
    parser.add_argument("--objective", type=str, default="cls", choices=["cls", "eos"], help="Objective")
    parser.add_argument("--num_ind", type=int, default=20, help="Number of Individuals")
    parser.add_argument("--num_gen", type=int, default=5, help="Number of Individuals")
    parser.add_argument("--crossover_flag", type=int, default=0, help="Whether to use Crossover or not")
    parser.add_argument("--device", type=str,default="cuda",help="Determine which GPU to use")
    parser.add_argument("--resume", action="store_true", help="Resume from the last processed entry")
    parser.add_argument("--resume_log_dir", type=str,
                        default="/kaggle/working/results/logging",
                        help="Outpu t directory")
#     parser.add_argument("--eos_weight", type=float, default=0.8, help="Weight for EOS gradient")
#     parser.add_argument("--cls_weight", type=float, default=0.2, help="Weight for classification gradient")
#     parser.add_argument("--delta", type=float, default=0.5, help="Threshold for adaptive search strategy")
#     parser.add_argument("--use_combined_loss", action="store_true", help="Use combined loss")
#     parser.add_argument("--attack_strategy", "-a", type=str,
#                         default='structure',
#                         choices=[
#                             'structure',
#                             'word',
#                             # 'pwws',
#                             # 'scpn',
#                             # 'viper',
#                             # 'bae',
#                             'fd', # white-box attack
#                             'hotflip', # white-box attack
#                             'textbugger', # white-box attack
#                             'uat', # white-box attack
#                             # 'maya',
#                             ],
#                         help="Attack strategy")
    args = parser.parse_args()
    main(args)
