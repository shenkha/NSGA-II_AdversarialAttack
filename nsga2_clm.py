from typing import Union, List, Tuple
import random
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)
import torch
from attack_main_clm import DGAttackEval
import nltk


import torch.nn as nn
import time
from DialogueAPI import dialogue
from attack_main import SentenceEncoder
softmax = nn.Softmax(dim=1)
bce_loss = nn.BCELoss()
import numpy as np

def get_front_0(F):
        l = len(F)
        r = np.zeros(l, dtype=np.int8)
        for i in range(l):
            if r[i] == 0:
                for j in range(i + 1, l):
                    better_sol = find_the_better(F[i], F[j])
                    if better_sol == 0:
                        r[j] = 1
                    elif better_sol == 1:
                        r[i] = 1
                        break
        return r == 0

def find_the_better(x, y):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)
        sub_ = x - y
        x_better = np.all(sub_ <= 0)
        y_better = np.all(sub_ >= 0)
        if x_better == y_better:  # True - True
            return -1
        if y_better:  # False - True
            return 1
        return 0  # True - False

immutable_words = {'was', 'were', 'am', 'is', 'are', 'been', 'being', 'be', 'have', 'has', 'had', 'do', 'does', 'did'}

def identify_salient_words(sentence):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    # Get POS tags
    pos_tags = nltk.pos_tag(tokens)
    #print(pos_tags)
    # Define POS tags of interest (e.g., nouns, verbs, adjectives)
    salient_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP','JJ', 'JJR', 'JJS'}

    # Identify salient words based on POS tags
    salient_words = [word for word, tag in pos_tags if tag in salient_pos_tags and word.isalnum() and len(word) > 1  and word.lower() not in immutable_words]
    return salient_words

def constrained_sum_sample_pos(n, total, low=0):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(
        np.random.choice(np.arange(1, total - (low - 1) * n), n - 1, replace=False)
    )

    return [
        a - b + low - 1
        for a, b in zip(dividers + [total - (low - 1) * n], [0] + dividers)
    ]


def identify_salient_words_with_idx(sentence):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    # Get POS tags
    pos_tags = nltk.pos_tag(tokens)
    # print(pos_tags)
    # Define POS tags of interest (e.g., nouns, verbs, adjectives)
    salient_pos_tags = {
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBZ",
        "VBP",
        "JJ",
        "JJR",
        "JJS",
    }

    immutable_words = {'was', 'were', 'am', 'is', 'are', 'been', 'being', 'be', 'have', 'has', 'had', 'do', 'does', 'did'}

    # Identify salient words based on POS tags

    words_count = {}
    salient_words = []

    for word, tag in pos_tags:
        if word not in words_count:
            words_count[word] = 1
        else:
            words_count[word] += 1

        if not (
            tag in salient_pos_tags
            and word.isalnum()
            and len(word) > 1
            and word.lower() not in immutable_words
        ):
            continue

        salient_words.append(
            (
                words_count[word]
                - 1,  # to differentiate the first and second occurrence of the same word
                word,
            )
        )
    return salient_words

def string_replace(text, _from, _to, idx):
    nth = idx + 1
    arr = text.split(_from)
    part1 = _from.join(arr[:nth])
    part2 = _from.join(arr[nth:])
    return part1 + _to + part2






class Individual(object):

    def __init__(self):
        self.rank = None
        self.crowding_distance = None
        self.sentence = None
        self.guided_sentence = None
        self.domination_count = None
        self.dominated_solutions = None
        self.length = None
        self.accuracy = None
        self.cls_loss = None
        self.eos_loss = None
    
#     def dominates(self, other_individual):
# #         and_condition = True
# #         or_condition = False
# #         for first, second in zip(self.objectives, other_individual.objectives):
# #             and_condition = and_condition and first <= second
# #             or_condition = or_condition or first < second
# #         return (and_condition and or_condition)
#         better_in_at_least_one = False
#         not_worse_in_any = True

#         if self.eos_loss < other_individual.eos_loss:
#             better_in_at_least_one = True
#         elif self.eos_loss > other_individual.eos_loss:
#             not_worse_in_any = False

#         if self.cls_loss < other_individual.cls_loss:
#             better_in_at_least_one = True
#         elif self.cls_loss > other_individual.cls_loss:
#             not_worse_in_any = False

#         return better_in_at_least_one and not_worse_in_any


    #This one is for NSGA-II with length and accuracy
    def dominates(self, other_individual):
        better_in_at_least_one = False
        not_worse_in_any = True

        # Maximizing length: this individual should have greater or equal length
        if self.length > other_individual.length:
            better_in_at_least_one = True
        elif self.length < other_individual.length:
            not_worse_in_any = False

        # Minimizing accuracy: this individual should have less or equal accuracy
        if self.accuracy < other_individual.accuracy:
            better_in_at_least_one = True
        elif self.accuracy > other_individual.accuracy:
            not_worse_in_any = False

        return better_in_at_least_one and not_worse_in_any

    
class Population:
    def __init__(self):
        self.population = []
        self.fronts = []

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return self.population.__iter__()

    def extend(self, new_individuals):
        self.population.extend(new_individuals)

    def append(self, new_individual):
        self.population.append(new_individual)
        

class Problem:

    def __init__(self,model, tokenizer, context, original_sentence, guided_sentence, device,max_len,task):
        self.context  = context 
        self.original_sentence = original_sentence
        self.guided_sentence = guided_sentence
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.task = task
        self.max_len = max_len
        self.eos_token = self.tokenizer.eos_token
        self.sentencoder = SentenceEncoder(model_name='paraphrase-distilroberta-base-v1', device = self.device)
        self.berttokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
        self.bertmodel = AutoModelForMaskedLM.from_pretrained('bert-large-uncased').eval().to(self.device)
        

#     def generate_individual(self):
#         #num_masks = self.num_masks_func(self.original_sentence)
#         mutated_sentences  = predict_masked_sentences_for_salient_words(self.original_sentence)
#         chosen_sentence = random.choice(mutated_sentences) if mutated_sentences else None
#         if chosen_sentence:
#             individual = Individual()
#             individual.sentence = chosen_sentence
#             individual.guided_sentence = self.guided_sentence
#             return individual
#         return None

    def mask_words(self,original_sentence, idx_words):

    # berttokenizer.mask_token == [MASK]
        return [
            string_replace(
                text=original_sentence, _from=word, _to=self.berttokenizer.mask_token, idx=idx
            )
            for idx, word in idx_words
        ]
    
    def get_prediction_sen(self, text: str):
        if self.task == 'seq2seq':
            effective_text = text
        else:
            effective_text = text + self.tokenizer.eos_token

        inputs = self.tokenizer(
            effective_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
        )
        input_ids = inputs.input_ids.to(self.device)
        self.model = self.model.to(self.device)
        t1 = time.time()
        with torch.no_grad():
            outputs = dialogue(
                    self.model,
                    input_ids,
                    early_stopping=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    # num_beams=4,  # Increased number of beams for better quality
                    # num_beam_groups=2,
                    # no_repeat_ngram_size=2,  # Added no_repeat_ngram_size to avoid repetition
                    # repetition_penalty=1.5,  # Added repetition penalty to avoid repetition
                    # do_sample=False,  # Enable sampling
                    # top_k=50,  # Use top-k sampling
                    # top_p=0.95,  # Use top-p (nucleus) sampling
                    # temperature=0.8,  # Control randomness
                    num_beams=5,  # Use beam search with 4 beams
                    num_beam_groups=1,  # Enable diverse beam search with 2 groups
                    no_repeat_ngram_size=3,  # Avoid repeating 3-grams
                    repetition_penalty=1.5,  # Use a repetition penalty of 2
                    length_penalty=1.2,
                    use_cache=True,
                    do_sample=False,  # Disable sampling for diverse beam search
                    max_length=self.max_len,
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
    
    def predict_masked_sentences_for_salient_words(self,sentence, num_sentences=20, top_k=5):
        salient_words = identify_salient_words_with_idx(sentence)

        # Check if there are no salient words
        if not salient_words:
            print("No salient words identified in the sentence.")
            return [sentence] * num_sentences  # or return an appropriate message or empty list

        random.shuffle(salient_words)

        min_num_sents = 0
        if len(salient_words) < num_sentences:
            min_num_sents = 1

        try:
            num_sent_per_word = constrained_sum_sample_pos(
                len(salient_words), num_sentences, low=min_num_sents
            )
        except ValueError as e:
            print(f"Error: {e}")
            return []

        # Remove word with 0 sentence
        num_sent_per_word = [n for n in num_sent_per_word if n != 0]
        salient_words = [
            w for idx, w in enumerate(salient_words) if idx < len(num_sent_per_word)
        ]

        masked_sentences = self.mask_words(sentence, salient_words)

        inputs = self.berttokenizer(masked_sentences, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.bertmodel(**inputs)
            logits = outputs.logits

        inputs = inputs.to("cpu")
        logits = logits.to("cpu")

        input_ids = inputs["input_ids"]
        mask_token_indices = input_ids == self.berttokenizer.mask_token_id
        assert (mask_token_indices.sum(-1) == 1).all()  # Only one masked word per sentence

        mask_token_logits = logits[mask_token_indices]
        generated_sentences = set()

        for mt_logit, m_sent, n_sent, (_, original_word) in zip(
            mask_token_logits, masked_sentences, num_sent_per_word, salient_words
        ):
            top_predictions = mt_logit.topk(n_sent * 5 + 1).indices.tolist()
            count = 0
            for predicted_index in top_predictions:
                predicted_token = self.berttokenizer.decode([predicted_index]).strip()
                if not predicted_token.isalnum() or predicted_token.lower().strip() == original_word.lower().strip():
                    continue
                if count >= n_sent:
                    break
                new_sentence = m_sent.replace(self.berttokenizer.mask_token, predicted_token, 1)
                sim = self.sentencoder.get_sim(new_sentence, sentence)
                if sim < 0.80 or sim > 1.0:
                    continue
                generated_sentences.add(new_sentence)
                count += 1

        result_list = list(generated_sentences)
        if len(result_list) < num_sentences:
            last_sentence = result_list[-1] if result_list else sentence
            result_list.extend([last_sentence] * (num_sentences - len(result_list)))

        return result_list

    

    # def predict_masked_sentences_for_salient_words(self, sentence, num_sentences=20, top_k=5):

    #         salient_words = identify_salient_words(sentence)
    #         generated_sentences = set()  # Use a set to avoid duplicates
    

    #         while len(generated_sentences) < num_sentences and salient_words:
    #             word_to_mask = salient_words.pop()
    #             if not word_to_mask:
    #                 break  # Exit if there are no salient words to mask

    #             masked_sentence = sentence.replace(word_to_mask, self.berttokenizer.mask_token, 1)
    #             inputs = self.berttokenizer.encode_plus(masked_sentence, return_tensors="pt")
    #             input_ids = inputs['input_ids'].to(self.device)

    #             with torch.no_grad():
    #                 outputs = self.bertmodel(input_ids)
    #                 predictions = outputs.logits

    #             mask_token_index = (input_ids == self.berttokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    #             top_predictions = predictions[0, mask_token_index, :].topk(top_k).indices.squeeze().tolist()

    #             for predicted_index in top_predictions:
    #                 predicted_token = self.berttokenizer.decode([predicted_index]).strip()
    #                 if predicted_token.isalnum():  # Filter out non-alphanumeric tokens
    #                     new_sentence = masked_sentence.replace(self.berttokenizer.mask_token, predicted_token, 1)
    #                     sim = self.sentencoder.get_sim(new_sentence, sentence)
    #                     if 0.80 <= sim <= 1.0:
    #                         generated_sentences.add(new_sentence)
    #                     if len(generated_sentences) >= num_sentences:
    #                         break

    #             attempts += 1

    #         result_list = list(generated_sentences)
    #         if len(result_list) < num_sentences:
    #             # If not enough sentences, replicate the last one until reaching the desired number
    #             last_sentence = result_list[-1] if result_list else sentence
    #             result_list.extend([last_sentence] * (num_sentences - len(result_list)))

    #         return result_list
      

    def remove_pad(self, s: torch.Tensor):
        return s[torch.nonzero(s != self.tokenizer.pad_token_id)].squeeze(1)

    def compute_seq_len(self, seq: torch.Tensor):
        if seq.shape[0] == 0: # empty sequence
            return 0
        if seq[0].eq(self.tokenizer.pad_token_id):
            return int(len(seq) - sum(seq.eq(self.tokenizer.pad_token_id)))
        else:
            return int(len(seq) - sum(seq.eq(self.tokenizer.pad_token_id))) - 1
        
    def get_average_score(self,generated_sentences: list, reference_sentences: list, acc_metric):
        scores = []

        for gen_sentence, ref_sentence in zip(generated_sentences, reference_sentences):
            # Calculate BLEU score
            bleu_res = self.bleu.compute(
                predictions=[gen_sentence],
                references=[ref_sentence],
                smooth=True,
            )
            #bleu_score = bleu_res['bleu']
            bleu_score = bleu_res['bleu']
            # Calculate ROUGE-L score
            rouge_res = self.rouge.compute(
                predictions=[gen_sentence],
                references=[ref_sentence],
            )
            rouge_score = rouge_res['rougeL']

            # Calculate METEOR score
            meteor_res = self.meteor.compute(
                predictions=[gen_sentence],
                references=[ref_sentence],
            )
            meteor_score_value = meteor_res['meteor']

            # Calculate the average of BLEU, ROUGE, and METEOR
            average_score = (bleu_score + rouge_score + meteor_score_value) / 3.0
            #Scale the score so that it would not be so small
            average_score = average_score + 1

            if acc_metric =="combined":
                scores.append(average_score)
            elif acc_metric == "bleu":
                bleu_score = bleu_score + 1
                scores.append(bleu_score)
            elif acc_metric == "rouge":
                rouge_score = rouge_score + 1
                scores.append(rouge_score)
            elif acc_metric == "meteor":
                meteor_score_value = meteor_score_value + 1
                scores.append(meteor_score_value)

        return scores

    def get_target_p(self, scores: list, pred_len: list, label: list):
        targets = []
        for i, s in enumerate(scores): # s: T X V
            if pred_len[i] == 0:
                targets.append(torch.tensor(0.0).to(self.device))
            else:
                # if self.pad_token_id != self.eos_token_id:
                s[:, self.pad_token_id] = 1e-12
                softmax_v = softmax(s) # T X V
                target_p = torch.stack([softmax_v[idx, v] for idx, v in enumerate(label[:softmax_v.size(0)])])
                target_p = target_p[:pred_len[i]]
                targets.append(torch.sum(target_p))
        return torch.stack(targets).detach().cpu().numpy()

    def get_prediction(self,sentence: Union[str, List[str]]):
            if self.task == 'seq2seq':
                text = sentence
            else:
                if isinstance(sentence, list):
                    text = [s + self.eos_token for s in sentence]
                elif isinstance(sentence, str):
                    text = sentence + self.eos_token
                else:
                    raise ValueError("sentence should be a list of string or a string")
                
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len,
            )
            input_ids = inputs['input_ids'].to(self.device)
            # ['sequences', 'sequences_scores', 'scores', 'beam_indices']
            with torch.no_grad():
                outputs = dialogue(
                        self.model,
                        input_ids,
                        early_stopping=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        # num_beams=4,  # Increased number of beams for better quality
                        # num_beam_groups=2,
                        # no_repeat_ngram_size=2,  # Added no_repeat_ngram_size to avoid repetition
                        # repetition_penalty=1.5,  # Added repetition penalty to avoid repetition
                        # do_sample=False,  # Enable sampling
                        # top_k=50,  # Use top-k sampling
                        # top_p=0.95,  # Use top-p (nucleus) sampling
                        # temperature=0.8,  # Control randomness
                        num_beams=1,  # Use beam search with 4 beams
                        num_beam_groups=1,  # Enable diverse beam search with 2 groups
                        no_repeat_ngram_size=3,  # Avoid repeating 3-grams
                        repetition_penalty=1.5,  # Use a repetition penalty of 2
                        length_penalty=1.2,
                        use_cache=True,
                        do_sample=False,  # Disable sampling for diverse beam search
                        max_length=self.max_len,
                )

            
            if self.task == 'seq2seq':
                seqs = outputs['sequences'].detach()
            else:
                seqs = outputs['sequences'][:, input_ids.shape[-1]:].detach()

            seqs = [self.remove_pad(seq) for seq in seqs]
            out_scores = outputs['scores']
            pred_len = [self.compute_seq_len(seq) for seq in seqs]
            return pred_len, seqs, out_scores

    def compute_batch_score(self, text: List[str]):
            batch_size = len(text)
            num_beams =  1
            batch_size = len(text)
            index_list = [i * 1 for i in range(batch_size + 1)]

            with torch.no_grad():
                pred_len, seqs, out_scores = self.get_prediction(text)

            scores = [[] for _ in range(batch_size)]
            for out_s in out_scores:
                for i in range(batch_size):
                    current_index = index_list[i]
                    scores[i].append(out_s[current_index: current_index + 1])
            scores = [torch.cat(s) for s in scores]
            scores = [s[:pred_len[i]] for i, s in enumerate(scores)]
            return scores, seqs, pred_len


    def compute_score(self, text: Union[str, List[str]], batch_size: int = None):
            total_size = len(text)
            if batch_size is None:
                batch_size = len(text)

            if batch_size < total_size:
                scores, seqs, pred_len = [], [], []
                for start in range(0, total_size, batch_size):
                    end = min(start + batch_size, total_size)
                    with torch.no_grad():
                        score, seq, p_len = self.compute_batch_score(text[start: end])
                    pred_len.extend(p_len)
                    seqs.extend(seq)
                    scores.extend(score)
            else:
                scores, seqs, pred_len = self.compute_batch_score(text)
            return scores, seqs, pred_len

    def generate_unique_sentences(self,num_sentences):
        # Ensure this call generates the required number of unique sentences
        return self.predict_masked_sentences_for_salient_words(self.original_sentence ,num_sentences=num_sentences)
    
    def generate_individual_from_sentence(self, sentence):
        individual = Individual()
        individual.sentence = sentence
        individual.guided_sentence = self.guided_sentence  
        self.calculate_objectives(individual)
        return individual
    


    def calculate_objectives(self, individual):
        if individual and individual.sentence and individual.guided_sentence:
            #individual.cls_loss = self.get_cls_loss([individual.sentence], [individual.guided_sentence]).item()
            #scores, seqs, pred_len = self.compute_score([individual.sentence])
            #individual.eos_loss = self.leave_eos_target_loss(scores, seqs, pred_len)[0].item()
            if self.task == "seq2seq":
                sp_token = self.tokenizer.eos_token
            else:
                sp_token = '<SEP>'
            text = self.context + sp_token + individual.sentence
            scores, seqs, p_len = self.compute_score([text])
            generated_responses = [self.tokenizer.batch_decode([seq], skip_special_tokens=True)[0].strip() for seq in seqs]

            individual.length = p_len[0]
            # label = self.tokenizer(individual.guided_sentence, truncation=True, max_length=self.max_len, return_tensors='pt')
            # label = label['input_ids'][0] # (T, )
            # res = self.get_target_p(scores, p_len, label) # numpy array (N, )
            #pred_acc.extend(res.tolist())
            #accuracy = self.get_average_score(generated_responses, [individual.guided_sentence],self.acc_metric)
            accuracy = self.get_average_score(generated_responses, [self.ori_ref],self.acc_metric)
            individual.accuracy = accuracy[0]

        else:
            individual.accuracy = float('inf')
            individual.length = float('inf')
            individual.cls_loss = float('inf')
            individual.eos_loss = float('inf')

class NSGA2Utils:

    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5, crossover_flag = 0):

        self.problem = problem
        self.num_of_individuals = num_of_individuals
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob
        self.crossover_param = crossover_param
        self.mutation_param = mutation_param
        self.crossover_flag = crossover_flag

#     def create_initial_population(self):
#         population = Population()
#         #print("Creating initial population...")
#         for _ in range(self.num_of_individuals):
#             individual = self.problem.generate_individual()
#             #print(f"Generated sentence: {individual.sentence}")
#             self.problem.calculate_objectives(individual)
#             population.append(individual)
#         return population

    def create_initial_population(self):
        population = Population()
        
        # Adjusted to call the sentence generation method once
        mutated_sentences = self.problem.generate_unique_sentences(self.num_of_individuals)
        
        for sentence in mutated_sentences:
            individual = self.problem.generate_individual_from_sentence(sentence)
            if individual is not None:
                self.problem.calculate_objectives(individual)
                population.append(individual)
                
        return population

    def fast_nondominated_sort(self, population):
        population.fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            population.fronts.append(temp)

    def calculate_crowding_distance(self, front):
        if len(front) > 0:    
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0
            
            objectives = [(lambda x: x.accuracy), (lambda x: x.length)]
            #objectives = [(lambda x: x.cls_loss), (lambda x: x.eos_loss)]
            #print(objectives)
            #print(objectives[0],objectives[1])
            for obj_func in objectives:
                front.sort(key=obj_func)
                front[0].crowding_distance = front[-1].crowding_distance = float('inf')
                for i in range(1, solutions_num - 1):
                    #print(obj_func(front[i+1]))
                    #print(obj_func(front[i+1][0]), obj_func(front[i+1][1]))
                    #print(obj_func(front[i+1]).item())
                    #temp1 = obj_func(front[i+1]).item()
                    #temp2 = obj_func(front[i-1]).item()
                    distance = obj_func(front[i + 1]) - obj_func(front[i - 1])
                    #distance = temp1 - temp2
                    front[i].crowding_distance += distance / (obj_func(front[-1]) - obj_func(front[0]) or 1)


#             for m in range(len(front[0].objectives)):
#                 front.sort(key=lambda individual: individual.objectives[m])
#                 front[0].crowding_distance = 10 ** 9
#                 front[solutions_num - 1].crowding_distance = 10 ** 9
#                 m_values = [individual.objectives[m] for individual in front]
#                 scale = max(m_values) - min(m_values)
#                 if scale == 0: scale = 1
#                 for i in range(1, solutions_num - 1):
#                     front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / scale
            

    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                        individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def create_children(self, population):
        #Only do mutation but not crossover
        #print("Creating children...")
        children = []
        while len(children) < len(population):
            #These are for crossover then mutation
            parent1 = self.__tournament(population)
            parent2 = parent1
            while parent1 == parent2:
                parent2 = self.__tournament(population)
            
            if self.crossover_flag == 1:
                child1, child2 = self.__crossover(parent1, parent2)
            else:
                child1, child2 = self.__noncrossover(parent1, parent2)
            self.__mutate(child1)
            self.__mutate(child2)
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            children.append(child1)
            children.append(child2)
            #This one is for mutation only
#             parent = self.__tournament(population)
#             child = self.problem.generate_individual_from_sentence(parent)
#             self.__mutate(child)
#             self.problem.calculate_objectives(child)
#             children.append(child)
            #print(f"Child1 generated with sentence: {child1.sentence}")
            #print(f"Child2 generated with sentence: {child2.sentence}")

        return children

    def __noncrossover(self, individual1, individual2):
        child1 = self.problem.generate_individual_from_sentence(individual1.sentence)
        child2 = self.problem.generate_individual_from_sentence(individual2.sentence)
        return child1, child2
    
    def __crossover(self, individual1, individual2):
        split_point_1 = len(individual1.sentence.split()) // 2
        split_point_2 = len(individual2.sentence.split()) // 2

        part1_1 = individual1.sentence.split()[:split_point_1]
        part1_2 = individual1.sentence.split()[split_point_2:]
        part2_1 = individual2.sentence.split()[:split_point_2]
        part2_2 = individual2.sentence.split()[split_point_1:]

        child_sentence_1 = ' '.join(part1_1 + part2_2)
        child_sentence_2 = ' '.join(part2_1 + part1_2)

        child1 = self.problem.generate_individual_from_sentence(child_sentence_1)
        child2 = self.problem.generate_individual_from_sentence(child_sentence_2)
        #print(f"Child1 generated with sentence: {child1.sentence}")
        #print(f"Child2 generated with sentence: {child2.sentence}")
        return child1, child2

    def __get_beta(self):
        u = random.random()
        if u <= 0.5:
            return (2 * u) ** (1 / (self.crossover_param + 1))
        return (2 * (1 - u)) ** (-1 / (self.crossover_param + 1))

    def __mutate(self, child):
        mutated_sentences  = self.problem.predict_masked_sentences_for_salient_words(child.sentence,self.num_of_individuals)
        #print(f"Mutated Sentences: {mutated_sentences}")
        #print(f"Mutating child sentence: {child.sentence}")
        if mutated_sentences:
            child.sentence = random.choice(mutated_sentences)
            #child.sentence = mutated_sentences[0]
            #print(f"Mutated: {child.sentence}")
        else:
            child.sentence = child.sentence

    def __get_delta(self):
        u = random.random()
        if u < 0.5:
            return u, (2 * u) ** (1 / (self.mutation_param + 1)) - 1
        return u, 1 - (2 * (1 - u)) ** (1 / (self.mutation_param + 1))

    def __tournament(self, population):
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (
                    self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
                best = participant

        return best

    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False
    

from tqdm import tqdm
import matplotlib.pyplot as plt

class Evolution:

    def __init__(self, crossover_flag, file_path_gen, problem, num_of_generations=1000, num_of_individuals=100, num_of_tour_particips=2,
                 tournament_prob=0.9, crossover_param=2, mutation_param=5,  ):
        self.utils = NSGA2Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, crossover_param,
                                mutation_param, crossover_flag)
        self.population = None
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals
        file_path = file_path_gen.split('.txt')[0]
        self.write_file_path_gen = file_path + "_Gen.txt"
        self.problem = problem
        self.task = problem.task
    
    def log_and_save_gen(self,display: str):
        print(display)
        with open(self.write_file_path_gen, 'a') as f:
            f.write(display + "\n")
        #self.write_file.write(display + "\n")   
        
    def is_front_converged(self, front):
        """
        Check if all individuals in a front have the same features or if their objectives are not improving.
        """
        if len(front)==1:
            return False

        if not front:
            return False

        first_individual = front[0]
        for individual in front:
            if individual.sentence != first_individual.sentence:
                return False
        return True
    

    def evolve(self):
        self.population = self.utils.create_initial_population()
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        children = self.utils.create_children(self.population)
        returned_population = None
        self.log_and_save_gen("\nDialogue history: {}".format(self.problem.context))
        self.log_and_save_gen("U--{} \n(Ref: ['{}', ...])".format(self.problem.original_sentence, self.problem.guided_sentence))
            # Original generation
        if self.task =="seq2seq":
            sp_token = self.problem.tokenizer.eos_token
        else:
            sp_token = "<SEP>"
        text = self.problem.context + sp_token + self.problem.original_sentence
        output, time_gap = self.problem.get_prediction_sen(text)
        self.log_and_save_gen("G--{}".format(output))

        for i in tqdm(range(self.num_of_generations)):
            self.log_and_save_gen("\nGeneration: {}".format(i))
            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)
            if self.is_front_converged(self.population.fronts[0]):
                self.log_and_save_gen(f"Convergence detected at generation {i}. Ending evolution.")
                self.plot_generation(self.population, i)
#                 print("Current front 0 as below:")
#                 for individual in self.population.fronts[0]:
#                     print(individual.sentence, individual.cls_loss, individual.eos_loss)
                break
            new_population = Population()
            front_num = 0
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
            new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals - len(new_population)])
            returned_population = self.population
            for individual in returned_population.fronts[0]:
                log_message = f"Sentence: '{individual.sentence}', Accuracy: {(individual.accuracy - 0.95)}, Length: {individual.length}"
                self.log_and_save_gen(log_message)
            self.population = new_population
            self.plot_generation(self.population, i)
            
            self.utils.fast_nondominated_sort(self.population)
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            children = self.utils.create_children(self.population)
            
        return returned_population.fronts[0] if returned_population else self.population.fronts[0]
    
    def plot_generation(self, population, generation):
        pred_acc = [(individual.accuracy - 0.95) for individual in population.population]
        pred_len = [individual.length for individual in population.population]

        F = np.zeros((len(pred_acc), len(pred_acc)))
        F[:, 0] = pred_acc
        F[:, 1] = pred_len
        F = np.array(F)
        F[:, 1] *= -1
        ids_fr0 = get_front_0(F)
        fr0 = F[ids_fr0]

        plt.figure(figsize=(10, 6))
        plt.scatter(pred_len, pred_acc, label=f'Generation {generation + 1}')
        plt.scatter(-fr0[:, 1], fr0[:, 0], facecolor='none', edgecolor='red', s=40, label=f'Front 0 {generation + 1}')
#         plt.xlim(15, 40)  # Adjust the scale for CLS Loss
#         plt.ylim(0, 5)    # Adjust the scale for EOS Loss
        plt.title('Mutated Sentences Evaluated by Two Objectives')
        plt.xlabel('Length')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    
    
plt.show()