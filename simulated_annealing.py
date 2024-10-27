import random
from collections import defaultdict
from math import exp
from pprint import pprint

class Item:
    def __init__(self, weight, value, name):
        self.weight = weight
        self.value = value
        self.name = name

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

    def calculate_total_value(self):
        total_items_value = 0
        for item, quantity in self.item_list.items():
            total_items_value += item.value * quantity
        
        return total_items_value
    
    def calculate_total_weight(self):
        total_items_weight = 0
        for item, quantity in self.item_list.items():
            total_items_weight += item.weight * quantity 

        return total_items_weight

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
    return create_random_state()

def simulated_annealing(starting_state, start_temperature, temperature_function, changing_rate):
    current_step = 1
    T = start_temperature
    current_state = starting_state
    best_state = starting_state
    while (T > 0):
        possible_state = create_possible_state(current_state)
        delta_value = possible_state.get_total_value() - current_state.get_total_value() 
        if(delta_value > 0):
            current_state = possible_state
            if(possible_state.get_total_value() > best_state.get_total_value()):
                best_state = possible_state
        
        elif random.random() < exp(delta_value / T):
            current_state = possible_state


        T = calculate_temperature(start_temperature, temperature_function, changing_rate, current_step)
        current_step += 1
    
    return current_state, best_state

knapsack_weight_capacity = 300
item_names = ["Diamante", "Ouro", "Prata", "Bronze", "Ferro", "Cobre", "Pedra", "Platina", "Aço", "Carvão"]
number_of_items = len(item_names)
items_list = create_items(item_names)

def main():
    for _ in range (1):
        starting_state = create_random_state()
        start_temperature = 100
        temperature_function = "exponential"
        changing_rate = 0.9
        final_state, best_state = simulated_annealing(starting_state, start_temperature, temperature_function, changing_rate)
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
    # Exibindo o valor total e o peso total do estado
    print("-" * 120)
    print(f"{'Valor Total do Estado:':<35} {best_state.get_total_value():<15}")
    print(f"{'Peso Total do Estado:':<35} {best_state.get_total_weight():<15}")

    # Acesso ao primeiro item
    # item_iterator = iter(starting_state.get_item_list())  # Cria um iterador sobre os itens
    # first_item = next(item_iterator)  # Obtém o primeiro item
    # quantity_first_item = starting_state.get_item_list()[first_item]
    # i, q = starting_state.get_item_by_name("Carvão")


if __name__ == "__main__":
    main()