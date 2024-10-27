import random
from collections import defaultdict
from math import exp
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

class Item:
    def __init__(self, weight, value, name):
        self.weight = weight
        self.value = value
        self.name = name

    def __hash__(self):
        return hash((self.weight, self.value, self.name))

    def __eq__(self, other):
        return self.weight == other.weight and self.value == other.value and self.name == other.name

    def get_value(self):
        return self.value

class State:
    def __init__(self):
        self.item_list = defaultdict(int)
        self.total_items_weight = 0
        self.total_items_value = 0

    def add_item(self, item):
        if item in self.item_list:
            self.item_list[item] += 1
        else:
            self.item_list[item] = 1
        self.total_items_weight += item.weight
        self.total_items_value += item.value

    def get_unique_item_count(self):
        return len(self.item_list)

    def remove_item(self, item):
        if item in self.item_list and self.item_list[item] > 0:
            self.item_list[item] -= 1
            self.total_items_weight -= item.weight
            self.total_items_value -= item.value
            if self.item_list[item] == 0:
                del self.item_list[item]

    def get_random_item(self):
        if not self.item_list:
            return None
        return random.choice(list(self.item_list.keys()))

    def get_total_value(self):
        return self.total_items_value       

    def get_total_weight(self):
        return self.total_items_weight       

    def get_item_list(self):
        return self.item_list
    
    def get_item_by_name(self, name):
        for item in self.item_list.keys():
            if item.name == name:
                return item, self.item_list[item]
        return None, 0  # Retorna None e 0 se o item não for encontrado

def create_items(item_names):
    items_list = []
    for name in item_names:
        random_weight = random.randint(1, 10)
        random_value = random.randint(1, 20)
        items_list.append(Item(random_weight, random_value, name))

    return items_list

def create_random_state():
    random_state = State()
    random_item = random.choice(items_list)

    while random_state.get_total_weight() + random_item.weight <= knapsack_weight_capacity:
        random_state.add_item(random_item)
        random_item = random.choice(items_list)

    return random_state

def calculate_temperature(start_temperature, temperature_function, changing_rate, current_step):
    if temperature_function == "exponential":
        T = start_temperature * (changing_rate ** current_step)
    elif temperature_function == "linear":
        T = start_temperature - (current_step * changing_rate)
    else:
        return None
    
    return T

def create_possible_state(current_state):
    possible_state = deepcopy(current_state)
    operation = random.choice(["add", "remove", "swap"])

    if operation == "add":
        item_to_add = random.choice(items_list)
        if (possible_state.get_total_weight() + item_to_add.weight <= knapsack_weight_capacity):
            possible_state.add_item(item_to_add)

    if operation == "remove":
        if(possible_state.get_unique_item_count() < 2):
            return possible_state
        
        item_to_remove = possible_state.get_random_item()
        possible_state.remove_item(item_to_remove)
    
    if operation == "swap":
        if(possible_state.get_unique_item_count() < 2):
            return possible_state
        
        item_to_remove = possible_state.get_random_item()
        item_to_add = possible_state.get_random_item()

        while(item_to_add == item_to_remove):
            item_to_add = possible_state.get_random_item()

        if (possible_state.get_total_weight() + item_to_add.weight <= knapsack_weight_capacity):
            possible_state.remove_item(item_to_remove)
            possible_state.add_item(item_to_add)

    return possible_state

def simulated_annealing(starting_state, start_temperature, temperature_function, changing_rate):
    current_step = 1
    step_best_state_found = 1
    T = start_temperature
    current_state = deepcopy(starting_state)
    best_state = deepcopy(starting_state)
    while (T > 0):
        possible_state = create_possible_state(current_state)
        delta_value = possible_state.get_total_value() - current_state.get_total_value() 
        if(delta_value > 0):
            current_state = deepcopy(possible_state)
            if(possible_state.get_total_value() > best_state.get_total_value()):
                best_state = deepcopy(possible_state)
                step_best_state_found = current_step
        
        elif random.random() < exp(delta_value / T):
            current_state = deepcopy(possible_state)

        T = calculate_temperature(start_temperature, temperature_function, changing_rate, current_step)
        current_step += 1
    
    return current_state, best_state, step_best_state_found

def print_best_state(best_state):
    best_state_items = best_state.get_item_list()
    best_state_items_details = [
        {
        'name': item.name,
        'quantity': quantity,
        'value': item.value,
        'weight': item.weight,
        'total_value': item.value * quantity,
        'total_weight': item.weight * quantity,
        'value_weight_ratio': item.value/item.weight,
        }
        for item, quantity in best_state_items.items()
    ]
    print("Itens no Estado:")
    print("-" * 120)
    print(f"{'Nome':<15} {'Quantidade':<10} {'Valor':<10} {'Peso':<10} {'Valor Total':<15} {'Peso Total':<15} {'Valor/Peso':<15}")
    print("-" * 120)
    for item in best_state_items_details:
        print(f"{item['name']:<15} {item['quantity']:<10} {item['value']:<10} {item['weight']:<10} {item['total_value']:<15} {item['total_weight']:<15} {item['value_weight_ratio']:<15.2f}")
    print("-" * 120)
    print(f"{'Valor Total do Estado:':<35} {best_state.get_total_value():<15}")
    print(f"{'Peso Total do Estado:':<35} {best_state.get_total_weight():<15}")

def plot_absolute_values_histogram(final_values, best_values):
    # Definindo o intervalo dos histogramas
    min_value = min(final_values)
    max_value = max(best_values)

    # Configurando os parâmetros do histograma
    bins = np.arange(min_value, max_value + 100, 100)  # Faixas de 100 em 100

    # Criando a figura e os eixos
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Histograma para final_values
    final_hist, _ = np.histogram(final_values, bins=bins)
    axs[0].hist(final_values, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    axs[0].set_title('Histograma de Final Values')
    axs[0].set_xlabel('Valores')
    axs[0].set_ylabel('Frequência')
    axs[0].set_xticks(bins)  # Adiciona ticks personalizados no eixo x

    # Calcular o máximo da frequência para final_values
    final_max = final_hist.max()
    
    # Definindo os ticks do eixo y de forma dinâmica
    y_ticks = np.linspace(0, final_max, num=11).astype(int)  # Garante que o máximo esteja incluído
    axs[0].set_yticks(y_ticks)  # Adiciona ticks dinâmicos no eixo y
    axs[0].grid(axis='y', color='black', linestyle='-', linewidth=0.5)  # Grade horizontal preta

    # Histograma para best_values
    best_hist, _ = np.histogram(best_values, bins=bins)
    axs[1].hist(best_values, bins=bins, alpha=0.7, color='green', edgecolor='black')
    axs[1].set_title('Histograma de Best Values')
    axs[1].set_xlabel('Valores')
    axs[1].set_ylabel('Frequência')
    axs[1].set_xticks(bins)  # Adiciona ticks personalizados no eixo x

    # Calcular o máximo da frequência para best_values
    best_max = best_hist.max()

    # O limite do eixo y deve ser o máximo entre final_max e best_max
    max_y_limit = max(final_max, best_max)
    
    # Definindo os ticks do eixo y de forma dinâmica para ambos os histogramas
    y_ticks_combined = np.linspace(0, max_y_limit, num=11).astype(int)  # Garante que o máximo esteja incluído
    axs[0].set_ylim(0, max_y_limit)  # Limite y para final_values
    axs[1].set_ylim(0, max_y_limit)  # Limite y para best_values
    axs[0].set_yticks(y_ticks_combined)  # Adiciona ticks dinâmicos ao primeiro histograma
    axs[1].set_yticks(y_ticks_combined)  # Adiciona ticks dinâmicos ao segundo histograma

    axs[1].grid(axis='y', color='black', linestyle='-', linewidth=0.5)  # Grade horizontal preta

    # Ajustando o layout
    plt.tight_layout()
    plt.show()

knapsack_weight_capacity = 300
item_names = ["Diamante", "Ouro", "Prata", "Bronze", "Ferro", "Cobre", "Pedra", "Platina", "Aço", "Carvão"]
number_of_items = len(item_names)
items_list = create_items(item_names)

def main():
    final_best_states_list = []

    for _ in range (300):
        starting_state = create_random_state()
        start_temperature = 100
        temperature_function = "linear"
        changing_rate = 0.8
        final_state, best_state, step_best_state_found = simulated_annealing(starting_state, start_temperature, temperature_function, changing_rate)
        final_best_states_list.append([final_state, best_state])
        #print_best_state(best_state)

    final_values, best_values, step_best_state_found = zip(*[(final_state.get_total_value(), best_state.get_total_value(), step_best_state_found) for final_state, best_state, step_best_state_found in final_best_states_list])
    plot_absolute_values_histogram(final_values, best_values, step_best_state_found)

if __name__ == "__main__":
    main()