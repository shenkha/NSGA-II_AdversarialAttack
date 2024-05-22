from typing import Union, List, Tuple
import random

import torch
from attack_main import DGAttackEval
from transformers import ( 
    BertTokenizer,
    BertTokenizerFast,
    BartForConditionalGeneration, 
    BartTokenizer,
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

import torch.nn as nn
softmax = nn.Softmax(dim=1)
bce_loss = nn.BCELoss()
class Individual(object):
 
    def __init__(self):
        self.rank = None
        self.crowding_distance = None
        self.sentence = None
        self.guided_sentence = None
        self.domination_count = None
        self.dominated_solutions = None
        self.eos_loss = None
        self.cls_loss = None
    
#     def __eq__(self, other):
#         if isinstance(self, other.__class__):
#             return self.features == other.features
#         return False

    def dominates(self, other_individual):
#         and_condition = True
#         or_condition = False
#         for first, second in zip(self.objectives, other_individual.objectives):
#             and_condition = and_condition and first <= second
#             or_condition = or_condition or first < second
#         return (and_condition and or_condition)
        better_in_at_least_one = False
        not_worse_in_any = True

        if self.eos_loss < other_individual.eos_loss:
            better_in_at_least_one = True
        elif self.eos_loss > other_individual.eos_loss:
            not_worse_in_any = False

        if self.cls_loss < other_individual.cls_loss:
            better_in_at_least_one = True
        elif self.cls_loss > other_individual.cls_loss:
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

    def __init__(self,model, tokenizer, context, original_sentence, guided_sentence, device):
        self.context  = context 
        self.original_sentence = original_sentence
        self.guided_sentence = guided_sentence
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.device = device

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

    def generate_unique_sentences(self,num_sentences):
        # Ensure this call generates the required number of unique sentences
        return DGAttackEval.predict_masked_sentences_for_salient_words(self.original_sentence ,num_sentences=num_sentences)
    
    def generate_individual_from_sentence(self, sentence):
        individual = Individual()
        individual.sentence = sentence
        individual.guided_sentence = self.guided_sentence  
        self.calculate_objectives(individual)
        return individual
    
    # def get_cls_loss(self, sentence: List[str], labels: List[str]):
        
    #     inputs = self.tokenizer(
    #         sentence, 
    #         return_tensors = "pt",
    #         padding = True,
    #         truncation = True,
    #         max_length = 512,
    #     ).to(device)
    
    #     labels = self.tokenizer(
    #         labels,
    #         return_tensors = "pt",
    #         padding = True,
    #         truncation = True,
    #         max_length = 512,
    #     ).to(device)
    
    #     output = self.model(**inputs, labels = labels['input_ids'])
    #     return -output.loss
    
    

#     def remove_pad(self, s: torch.Tensor):
#         return s[torch.nonzero(s != self.pad_token_id)].squeeze(1)

#     def compute_seq_len(self, seq: torch.Tensor):
#         if seq.shape[0] == 0: # empty sequence
#             return 0
#         if seq[0].eq(self.pad_token_id):
#             return int(len(seq) - sum(seq.eq(self.pad_token_id)))
#         else:
#             return int(len(seq) - sum(seq.eq(self.pad_token_id))) - 1


#     def get_prediction(self, sentence: Union[str, List[str]]):
#         text = sentence
#         inputs = self.tokenizer(
#             text, 
#             return_tensors="pt", 
#             padding=True, 
#             truncation=True, 
#             max_length=511,
#         )
#         input_ids = inputs['input_ids'].to(device)
#         # ['sequences', 'sequences_scores', 'scores', 'beam_indices']
#         outputs = dialogue(
#             self.model, 
#             input_ids,
#             early_stopping=False, 
#             pad_token_id=self.pad_token_id,
#             num_beams=1,
#             num_beam_groups=1, 
#             use_cache=True,
#             max_length=512,
#         )
        
#         seqs = outputs['sequences'].detach()
# #         if self.task == 'seq2seq':
# #             seqs = outputs['sequences'].detach()
# #         else:
# #             seqs = outputs['sequences'][:, input_ids.shape[-1]:].detach()

#         seqs = [self.remove_pad(seq) for seq in seqs]
#         out_scores = outputs['scores']
#         pred_len = [self.compute_seq_len(seq) for seq in seqs]
#         return pred_len, seqs, out_scores

#     def compute_batch_score(self,text: List[str]):
#         batch_size = len(text)
#         num_beams =  1
#         batch_size = len(text)
#         index_list = [i * 1 for i in range(batch_size + 1)]
#         pred_len, seqs, out_scores = self.get_prediction(text)

#         scores = [[] for _ in range(batch_size)]
#         for out_s in out_scores:
#             for i in range(batch_size):
#                 current_index = index_list[i]
#                 scores[i].append(out_s[current_index: current_index + 1])
#         scores = [torch.cat(s) for s in scores]
#         scores = [s[:pred_len[i]] for i, s in enumerate(scores)]
#         return scores, seqs, pred_len
    
    
#     def compute_score(self, text: Union[str, List[str]], batch_size: int = None):
#         total_size = len(text)
#         if batch_size is None:
#             batch_size = len(text)

#         if batch_size < total_size:
#             scores, seqs, pred_len = [], [], []
#             for start in range(0, total_size, batch_size):
#                 end = min(start + batch_size, total_size)
#                 score, seq, p_len = self.compute_batch_score(text[start: end])
#                 pred_len.extend(p_len)
#                 seqs.extend(seq)
#                 scores.extend(score)
#         else:
#             scores, seqs, pred_len = self.compute_batch_score(text)
#         return scores, seqs, pred_len
    
#     def leave_eos_target_loss(self, scores: list, seqs: list, pred_len: list):
#         loss = []
#         for i, s in enumerate(scores): # s: T X V
#             if pred_len[i] == 0:
#                 loss.append(torch.tensor(0.0, requires_grad=True).to(device))
#             else:
#                 s[:,self.pad_token_id] = 1e-12
#                 softmax_v = softmax(s)
#                 eos_p = softmax_v[:pred_len[i],self.eos_token_id]
#                 target_p = torch.stack([softmax_v[idx, v] for idx, v in enumerate(seqs[i][1:])])
#                 target_p = target_p[:pred_len[i]]
#                 pred = eos_p + target_p
#                 pred[-1] = pred[-1] / 2
#                 loss.append(bce_loss(pred, torch.zeros_like(pred)))
#         return loss


    def calculate_objectives(self, individual):
        if individual and individual.sentence and individual.guided_sentence:
            individual.cls_loss = DGAttackEval.get_cls_loss([individual.sentence], [individual.guided_sentence]).item()
            scores, seqs, pred_len = DGAttackEval.compute_score([individual.sentence])
            individual.eos_loss = DGAttackEval.leave_eos_target_loss(scores, seqs, pred_len)[0].item()
            #print(f"Calculated cls_loss: {individual.cls_loss}, eos_loss: {individual.eos_loss} for sentence: {individual.sentence}")
        else:
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
            
            objectives = [(lambda x: x.cls_loss), (lambda x: x.eos_loss)]
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
#         child1 = self.problem.generate_individual()
#         child2 = self.problem.generate_individual()
#         num_of_features = len(child1.features)
#         genes_indexes = range(num_of_features)
#         for i in genes_indexes:
#             beta = self.__get_beta()
#             x1 = (individual1.features[i] + individual2.features[i]) / 2
#             x2 = abs((individual1.features[i] - individual2.features[i]) / 2)
#             child1.features[i] = x1 + beta * x2
#             child2.features[i] = x1 - beta * x2
        #print("Creating crossover...")
        split_point_1 = len(individual1.sentence.split()) // 2
        split_point_2 = len(individual2.sentence.split()) // 2

        part1_1 = individual1.sentence.split()[:split_point_1]
        part1_2 = individual1.sentence.split()[split_point_2:]
        part2_1 = individual2.sentence.split()[:split_point_2]
        part2_2 = individual2.sentence.split()[split_point_1:]

        child_sentence_1 = ' '.join(part1_1 + part1_2)
        child_sentence_2 = ' '.join(part2_1 + part2_2)

        child1 = self.problem.generate_individual_from_sentence(child_sentence_1)
        child2 = self.problem.generate_individual_from_sentence(child_sentence_2)
        #print(f"Child1 generated with sentence: {child1.sentence}")
        #print(f"Child2 generated with sentence: {child2.sentence}")
        return child1, child2
    
    def __crossover(self, individual1, individual2):
#         child1 = self.problem.generate_individual()
#         child2 = self.problem.generate_individual()
#         num_of_features = len(child1.features)
#         genes_indexes = range(num_of_features)
#         for i in genes_indexes:
#             beta = self.__get_beta()
#             x1 = (individual1.features[i] + individual2.features[i]) / 2
#             x2 = abs((individual1.features[i] - individual2.features[i]) / 2)
#             child1.features[i] = x1 + beta * x2
#             child2.features[i] = x1 - beta * x2
        #print("Creating crossover...")
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
        #print("Creating mutate...")
#         num_of_features = len(child.features)
#         for gene in range(num_of_features):
#             u, delta = self.__get_delta()
#             if u < 0.5:
#                 child.features[gene] += delta * (child.features[gene] - self.problem.variables_range[gene][0])
#             else:
#                 child.features[gene] += delta * (self.problem.variables_range[gene][1] - child.features[gene])
#             if child.features[gene] < self.problem.variables_range[gene][0]:
#                 child.features[gene] = self.problem.variables_range[gene][0]
#             elif child.features[gene] > self.problem.variables_range[gene][1]:
#                 child.features[gene] = self.problem.variables_range[gene][1]

        #num_masks = self.problem.num_masks_func(child.sentence)
        original = self.problem.original_sentence
        mutated_sentences  = DGAttackEval.predict_masked_sentences_for_salient_words(child.sentence,self.num_of_individuals)
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

    def __init__(self, crossover_flag, problem, num_of_generations=1000, num_of_individuals=100, num_of_tour_particips=2,
                 tournament_prob=0.9, crossover_param=2, mutation_param=5, write_file_path_gen = None, ):
        self.utils = NSGA2Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, crossover_param,
                                mutation_param, crossover_flag)
        self.population = None
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals
        self.write_file_path_gen = write_file_path_gen
        self.problem = problem
    
    def log_and_save_gen(self,display: str, write_file_path_gen):
        print(display)
        with open(self.write_file_path_gen, 'a') as f:
            f.write(display + "\n")
        #self.write_file.write(display + "\n")   
        
    def is_front_converged(self, front):
        """
        Check if all individuals in a front have the same features or if their objectives are not improving.
        """
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
        self.log_and_save("U--{} \n(Ref: ['{}', ...])".format(self.problem.free_message, self.problem.references[-1]))
            # Original generation
        eos_token = DGAttackEval.tokenizer.eos_token
        text = self.problem.context + eos_token + self.problem.free_message
        output, time_gap = DGAttackEval.get_prediction(text)
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
                self.log_and_save_gen(individual.sentence, individual.cls_loss, individual.eos_loss)
            self.population = new_population
            self.plot_generation(self.population, i)
            
            self.utils.fast_nondominated_sort(self.population)
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            children = self.utils.create_children(self.population)
            
        return returned_population.fronts[0] if returned_population else self.population.fronts[0]
    
    def plot_generation(self, population, generation):
        cls_losses = [individual.cls_loss for individual in population.population]
        eos_losses = [individual.eos_loss for individual in population.population]

        plt.figure(figsize=(10, 6))
        plt.scatter(cls_losses, eos_losses, label=f'Generation {generation + 1}')
        plt.xlim(-8, 0)  # Adjust the scale for CLS Loss
        plt.ylim(-0.01, 0.05)    # Adjust the scale for EOS Loss
        plt.xlabel('CLS Loss')
        plt.ylabel('EOS Loss')
        plt.title('Population Evolution Over Generations')
        plt.legend()
        plt.grid(True)
        plt.pause(0.001)
        plt.clf()  # Clear the figure for the next plot

    
    
plt.show()