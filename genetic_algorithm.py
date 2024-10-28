import random
from collections import defaultdict
from typing import Self
import matplotlib.pyplot as plt
import numpy as np

weight_capacity = 1000
item_names = [
    "Diamante",
    "Ouro",
    "Prata",
    "Bronze",
    "Ferro",
    "Cobre",
    "Pedra",
    "Platina",
    "Aço",
    "Carvão",
]
max_number_of_populations = 6
min_item_weight = 1
max_item_weight = 50
min_item_value = 1
max_item_value = 50

class Item:
    item_list: None | list[Self] = None

    def __init__(self, weight, value, name):
        self.weight = weight
        self.value = value
        self.name = name

    def __hash__(self):
        return hash((self.weight, self.value, self.name))

    def __eq__(self, other):
        return (
            self.weight == other.weight
            and self.value == other.value
            and self.name == other.name
        )

    def get_value(self):
        return self.value

    @classmethod
    def create_random(cls, name: str) -> Self:
        random_weight = random.randint(min_item_weight, max_item_weight)
        random_value = random.randint(min_item_value, max_item_value)
        return cls(random_weight, random_value, name)

    @classmethod
    def create_random_list(cls) -> list[Self]:
        item_list = []
        for name in item_names:
            item_list.append(cls.create_random(name))

        return item_list

    @classmethod
    def get_list(cls) -> list["Item"]:
        if cls.item_list is None:
            cls.item_list = cls.create_random_list()
        return cls.item_list

class Backpack:
    def __init__(self: Self):
        self.item_counts: dict[Item, int] = defaultdict(int)
        self.total_items_weight = 0
        self.total_items_value = 0

    @classmethod
    def create_random(cls) -> Self:
        self = cls()
        self.random_fill()
        return self

    def random_fill(self):
        item_list = Item.get_list()
        random_item = random.choice(item_list)

        while self.get_total_weight() + random_item.weight <= weight_capacity:
            self.add_item(random_item)
            random_item = random.choice(item_list)

    def copy(self):
        other = Backpack()
        other.item_counts = self.item_counts.copy()
        other.total_items_weight = self.total_items_weight
        other.total_items_value = self.total_items_value
        return other

    def add_item(self, item: Item):
        self.item_counts[item] += 1
        self.total_items_weight += item.weight
        self.total_items_value += item.value

    def get_unique_item_count(self):
        return len(self.item_counts)

    def remove_item(self, item):
        if item in self.item_counts and self.item_counts[item] > 0:
            self.item_counts[item] -= 1
            self.total_items_weight -= item.weight
            self.total_items_value -= item.value
            if self.item_counts[item] == 0:
                del self.item_counts[item]

    def get_random_item(self):
        if not self.item_counts:
            return None
        
        items = list(self.item_counts.keys())
        item_weights = list(self.item_counts.values())
        return random.choices(items, weights=item_weights)[0]

    def get_total_value(self):
        return self.total_items_value

    def get_total_weight(self):
        return self.total_items_weight

    def get_item_counts(self):
        return self.item_counts

    def get_item_by_name(self, name):
        for item in self.item_counts.keys():
            if item.name == name:
                return item, self.item_counts[item]
        return None, 0

class Population:
    def __init__(self):
        self.backpacks = []
    
    def add_backpack(self, backpack):
        self.backpacks.append(backpack)

    def get_best_backpack(self):
        best_backpack = max(self.backpacks, key=lambda b: b.get_total_value())
        return best_backpack

def plot_histogram(final_values):
    # Definindo o intervalo dos histogramas
    min_value = min(final_values)
    max_value = max(final_values)

    # Configurando os parâmetros do histograma
    bucket_size = round((max_value - min_value) / 10)
    bins = np.arange(min_value, max_value + bucket_size, bucket_size)  # Inclui o max_value

    # Criando a figura
    plt.figure(figsize=(10, 6))  # Dimensões da figura

    # Criando o histograma para final_values
    plt.hist(final_values, bins=bins, alpha=0.7, color="blue", edgecolor="black")
    plt.title("Distribuição de Melhores Mochilas Encontradas")
    plt.xlabel("Valor Total da Mochila")
    plt.ylabel("Frequência")
    plt.xticks(bins)  # Adiciona ticks personalizados no eixo x

    # Calcular o máximo da frequência para final_values
    final_hist, _ = np.histogram(final_values, bins=bins)
    final_max = final_hist.max()

    # Definindo os ticks do eixo y de forma dinâmica
    y_ticks = np.linspace(0, final_max, num=11).astype(int)  # Garante que o máximo esteja incluído
    plt.yticks(y_ticks)  # Adiciona ticks dinâmicos no eixo y
    plt.ylim(0, final_max)  # Garante que o eixo y vai até o máximo
    plt.grid(axis="y", color="black", linestyle="-", linewidth=0.5)  # Grade horizontal preta

    plt.show()

def create_initial_population(population_size):
    population = Population()
    for _ in range(population_size):
        random_backpack = Backpack.create_random()
        population.add_backpack(random_backpack)
    
    return population

def crossover(parents):    
    children_1 = parents[0].copy()
    children_2 = parents[1].copy()
    aux_backpack_1 = Backpack()
    aux_backpack_2 = Backpack()

    half_weight_average = (children_1.get_total_weight() + children_2.get_total_weight()) / 4

    while children_1.get_total_weight() < half_weight_average:
        rand_item = children_1.get_random_item()
        children_1.remove_item(rand_item)
        aux_backpack_1.add_item(rand_item)
    
    while children_2.get_total_weight() < half_weight_average:
        rand_item = children_2.get_random_item()
        children_2.remove_item(rand_item)
        aux_backpack_2.add_item(rand_item)

    while aux_backpack_2.get_unique_item_count() != 0:
        rand_item = aux_backpack_2.get_random_item()
        if children_1.get_total_weight() + rand_item.weight() > weight_capacity:
            break
        
        aux_backpack_2.remove_item(rand_item)
        children_1.add_item(rand_item)
    
    while aux_backpack_1.get_unique_item_count() != 0:
        rand_item = aux_backpack_1.get_random_item()
        if children_2.get_total_weight() + rand_item.weight() > weight_capacity:
            break
        
        aux_backpack_1.remove_item(rand_item)
        children_2.add_item(rand_item)

    while aux_backpack_1.get_unique_item_count() != 0:
        rand_item = aux_backpack_1.get_random_item()
        aux_backpack_1.remove_item(rand_item)
        children_1.add_item(rand_item)
        
    while aux_backpack_2.get_unique_item_count() != 0:
        rand_item = aux_backpack_2.get_random_item()
        aux_backpack_2.remove_item(rand_item)
        children_2.add_item(rand_item)

    return children_1, children_2

def select_parents(population):
    parents = []
    weights = [mochila.get_total_weight() for mochila in population.backpacks]
    parents = random.choices(population.backpacks, weights=weights, k=2)
    return parents

def mutate(children):
    highest_weight = max(i.weight for i in Item.get_list())
    mutated_child_1 = children[0].copy()
    mutated_child_2 = children[1].copy()
    total_weight_removed_child_1 = 0
    total_weight_removed_child_2 = 0

    while total_weight_removed_child_1 < highest_weight:
        item_to_remove = mutated_child_1.get_random_item()
        mutated_child_1.remove_item(item_to_remove)
        total_weight_removed_child_1 += item_to_remove.weight

    while total_weight_removed_child_2 < highest_weight:
        item_to_remove = mutated_child_2.get_random_item()
        mutated_child_2.remove_item(item_to_remove)
        total_weight_removed_child_2 += item_to_remove.weight

    mutated_child_1.random_fill()
    mutated_child_2.random_fill()
    return mutated_child_1, mutated_child_2

def create_new_population(population, mutation_rate, population_size):
    new_population = Population()

    while len(new_population.backpacks) < population_size:
        parents = select_parents(population)
        children = crossover(parents)
        if random.random() < mutation_rate:
            children = mutate(children)
        
        new_population.add_backpack(children[0])
        if len(new_population.backpacks) < population_size:
            new_population.add_backpack(children[1])
    
    return new_population

def genetic_algorithm(starting_population, population_size, mutation_rate):
    populations = []
    for _ in range(max_number_of_populations):
        created_population = create_new_population(starting_population, mutation_rate, population_size)
        populations.append(created_population)
    
    return populations

def main():
    number_of_iterations = 1000
    population_size = 50  # Alterar variável e fazer teste
    mutation_rate = 0.01  # Alterar e fazer teste. 0 < valor < 1
    starting_population = create_initial_population(population_size)
    best_backpacks_values_list = []

    for _ in range(number_of_iterations):
        populations = genetic_algorithm(
            starting_population, population_size, mutation_rate
        )

        best_backpack_value = 0
        for population in populations:
            best_backpack = population.get_best_backpack()
            if best_backpack.get_total_value() > best_backpack_value:
                best_backpack_value = best_backpack.get_total_value()
        
        best_backpacks_values_list.append(best_backpack_value)

    plot_histogram(best_backpacks_values_list)

if __name__ == "__main__":
    main()