import random
from collections import defaultdict
from math import exp
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
        return random.choice(list(self.item_counts.keys()))

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

def calculate_temperature(
    start_temperature, temperature_function, changing_rate, current_step
):
    if temperature_function == "exponential":
        T = start_temperature * (changing_rate**current_step)
    elif temperature_function == "linear":
        T = start_temperature - (current_step * changing_rate)
    else:
        return None

    return T

highest_weight = max(i.weight for i in Item.get_list())

def create_possible_state(current_state):
    possible_state = current_state.copy()
    total_weight_removed = 0
    while total_weight_removed < highest_weight:
        item_to_remove = possible_state.get_random_item()
        possible_state.remove_item(item_to_remove)
        total_weight_removed += item_to_remove.weight

    possible_state.random_fill()
    return possible_state

def simulated_annealing(
    starting_state, start_temperature, temperature_function, changing_rate
):
    current_step = 1
    step_best_state_found = 1
    T = start_temperature
    current_state = starting_state.copy()
    best_state = starting_state.copy()
    while T > 0:
        possible_state = create_possible_state(current_state)
        delta_value = possible_state.get_total_value() - current_state.get_total_value()
        if delta_value > 0:
            current_state = possible_state.copy()
            if possible_state.get_total_value() > best_state.get_total_value():
                best_state = possible_state.copy()
                step_best_state_found = current_step

        elif random.random() < exp(delta_value / T):
            current_state = possible_state.copy()

        T = calculate_temperature(
            start_temperature, temperature_function, changing_rate, current_step
        )
        current_step += 1

    return current_state, best_state, step_best_state_found

def plot_histograms(final_values, best_values, step_best_state_found):
    # Definindo o intervalo dos histogramas
    min_value = min(min(final_values), min(best_values))
    max_value = max(max(best_values), max(final_values))

    # Configurando os parâmetros do histograma
    bucket_size = (max_value - min_value) / 10
    bins = np.arange(min_value, max_value, bucket_size)  # Faixas de 50 em 50

    # Criando a figura e os eixos
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # Alterando para 3 subplots

    # Histograma para final_values
    axs[0].hist(final_values, bins=bins, alpha=0.7, color="blue", edgecolor="black")
    axs[0].set_title("Distribuição de valores dos últimos estados encontrados")
    axs[0].set_xlabel("Valor do último estado")
    axs[0].set_ylabel("Frequência")
    axs[0].set_xticks(bins)  # Adiciona ticks personalizados no eixo x

    # Calcular o máximo da frequência para final_values
    final_hist, _ = np.histogram(final_values, bins=bins)
    final_max = final_hist.max()

    # Definindo os ticks do eixo y de forma dinâmica
    y_ticks = np.linspace(0, final_max, num=11).astype(
        int
    )  # Garante que o máximo esteja incluído
    axs[0].set_yticks(y_ticks)  # Adiciona ticks dinâmicos no eixo y
    axs[0].set_ylim(0, final_max)  # Garante que o eixo y vai até o máximo
    axs[0].grid(
        axis="y", color="black", linestyle="-", linewidth=0.5
    )  # Grade horizontal preta

    # Histograma para best_values
    axs[1].hist(best_values, bins=bins, alpha=0.7, color="green", edgecolor="black")
    axs[1].set_title("Distribuição de valores dos melhores estados encontrados")
    axs[1].set_xlabel("Valor do melhor estado")
    axs[1].set_ylabel("Frequência")
    axs[1].set_xticks(bins)  # Adiciona ticks personalizados no eixo x

    # Calcular o máximo da frequência para best_values
    best_hist, _ = np.histogram(best_values, bins=bins)
    best_max = best_hist.max()
    y_ticks_best = np.linspace(0, best_max, num=11).astype(int)
    axs[1].set_ylim(0, best_max)  # Garante que o eixo y vai até o máximo
    axs[1].set_yticks(y_ticks_best)  # Adiciona ticks dinâmicos no eixo y
    axs[1].grid(
        axis="y", color="black", linestyle="-", linewidth=0.5
    )  # Grade horizontal preta

    # Histograma para step_best_state_found
    min_step = min(step_best_state_found)
    max_step = max(step_best_state_found)
    bucket_size = (max_step - min_step) / 10
    bins_step = np.arange(
        min_step, max_step + 10, bucket_size
    )  # Faixas de 10 em 10 específicas para step_best_state_found

    axs[2].hist(
        step_best_state_found,
        bins=bins_step,
        alpha=0.7,
        color="orange",
        edgecolor="black",
    )
    axs[2].set_title(
        "Distribuição dos passos em que os melhores valores foram encontrados"
    )
    axs[2].set_xlabel("Número do passo")
    axs[2].set_ylabel("Frequência")
    axs[2].set_xticks(bins_step)  # Adiciona ticks personalizados no eixo x

    # Calcular o máximo da frequência para step_best_state_found
    step_hist, _ = np.histogram(step_best_state_found, bins=bins_step)
    step_max = step_hist.max()
    y_ticks_step = np.linspace(0, step_max, num=11).astype(int)
    axs[2].set_yticks(y_ticks_step)  # Adiciona ticks dinâmicos no eixo y
    axs[2].set_ylim(0, step_max)  # Garante que o eixo y vai até o máximo
    axs[2].grid(
        axis="y", color="black", linestyle="-", linewidth=0.5
    )  # Grade horizontal preta

    # Ajustando o layout
    plt.subplots_adjust(hspace=0.5)  # Espaçamento vertical
    plt.show()

def main():
    final_best_states_list = []
    number_of_iterations = 1000
    start_temperature = 10  # Alterar variável e fazer teste
    temperature_function = ("linear")  # Alterar e fazer teste. opção "exponential" ou "linear"
    changing_rate = 0.001  # Alterar e fazer teste. 0 < valor < 1
    starting_state = Backpack.create_random()  # Alterar para fora ou dentro do loop

    for _ in range(number_of_iterations):
        final_state, best_state, step_best_state_found = simulated_annealing(
            starting_state, start_temperature, temperature_function, changing_rate
        )
        final_best_states_list.append([final_state, best_state, step_best_state_found])
        # print_best_state(best_state)

    final_values, best_values, step_best_state_found = zip(
        *[
            (
                final_state.get_total_value(),
                best_state.get_total_value(),
                step_best_state_found,
            )
            for final_state, best_state, step_best_state_found in final_best_states_list
        ]
    )
    print("Best value found on all iterations:", max(best_values), "\nBest value according to greedy approach:", max(best_values))
    plot_histograms(final_values, best_values, step_best_state_found)

if __name__ == "__main__":
    main()  # Chama a função main