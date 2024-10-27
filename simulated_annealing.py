import random
from collections import defaultdict

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

def create_items():
    item_names = ["Diamante", "Ouro", "Prata", "Bronze", "Ferro", "Cobre", "Pedra", "Platina", "Aço", "Carvão"]
    items_list = []
    for name in item_names:
        random_weight = random.randint(1, 10)
        random_value = random.randint(1, 20)
        items_list.append(Item(random_weight, random_value, name))

    return items_list

def create_starting_state(knapsack_weight_capacity, items_list):
    random_state = State()
    random_item = random.choice(items_list)

    while random_state.get_total_weight() + random_item.weight <= knapsack_weight_capacity:
        random_state.add_item(random_item)
        random_item = random.choice(items_list)

    return random_state

def simulated_annealing():
    pass

def main():
    knapsack_weight_capacity = 300
    items_list = create_items()
    starting_state = create_starting_state(knapsack_weight_capacity, items_list)
    simulated_annealing()
    # Acesso ao primeiro item
    # item_iterator = iter(starting_state.get_item_list())  # Cria um iterador sobre os itens
    # first_item = next(item_iterator)  # Obtém o primeiro item
    # quantity_first_item = starting_state.get_item_list()[first_item]
    # i, q = starting_state.get_item_by_name("Carvão")


if __name__ == "__main__":
    main()