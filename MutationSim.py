import numpy as np
class Generation():
    def __init__(self, generation_num):
        # The generation number determines the size of the array since the number of cells in a generation is 2^n
        self.gen_num = generation_num
        self.size = self.get_size(self.gen_num)
        # Depending on the generation size, the number of cells will be 2^n and all the cells will be filled with zeros to start
        self.cells = np.full(shape=self.size, fill_value=0)
        self.non_mutant_count = 0
        self.mutant_count = 0
        self.new_mutant_count = 0
        self.existing_mutant_count = 0

    # Size is determined by the generation number which represents n
    def get_size(self, generation_num):
        return 2 ** generation_num
    # Uses the information of the previous generation to determine if the cell will be a pre-existing mutant or will be affected by the mutation rate
    def fill_array(self, prev_generation_cells):

        parent_index = 0
        # Uses a two step function to fill the next generation
        for i in range(0, self.size, 2):
            if self.get_parent_val(prev_generation_cells, parent_index) == 1:
                # Performs Binomial twice with n set to 1 (a Bernouli distrobution for each cell), once on i and once on i+1
                # Is the same as performing a Binomial of (2, v)
                # Each cell has the chance of mutating via the mutation rate
                self.cells[i] = np.random.binomial(1, .999274)
                self.cells[i+1] = np.random.binomial(1, .999274)

            # The parent index is increased by one every time the
            parent_index += 1

        self.set_non_mutant()
        self.set_mutants()
    # Uses the previous generation at a desired index to determine if the parent cell was a non mutant or a mutant
    def get_parent_val(self, cells, index):
        return cells[index]
    # Uses the sum function to count the amount of 1's (non mutants) i the array
    def set_non_mutant(self):
        self.non_mutant_count = sum(self.cells)

    # The total mutant count can be set by subtracting the size of the generation by the non mutants
    def set_mutants(self):
        self.mutant_count = self.size - self.non_mutant_count
    # New mutants can be determined by subtracting 2 times the previous mutant count from the total mutants
    def set_new_mutants(self, prev_mutants):
        self.new_mutant_count = self.mutant_count - (2 * prev_mutants)
    # The pre-existing mutants can be found by subtracting the new mutants from the total mutants
    def set_existing_mutants(self):
        self.existing_mutant_count = self.mutant_count - self.new_mutant_count

    # This segment prints the output and output redirection can be used to write each trial to a text file
    def print_values(self):
        print(f'beginning of generation {self.gen_num}\n')

        for i in range(0, self.size):
            print(self.cells[i], end=" ")

        print(f'\nend of generation {self.gen_num}\n')

# The generation array include ths zeroth generation which is why the size is 31
Generation_Array = np.empty(shape=31, dtype=Generation)

Generation_Array[0] = Generation(0)
# Hardcodes the original cell as a non mutant
Generation_Array[0].cells[0] = 1
Generation_Array[0].non_mutant_count = 1



for i in range(1, 31):
    Generation_Array[i] = Generation(i)
    Generation_Array[i].fill_array(Generation_Array[i-1].cells)
    Generation_Array[i].set_new_mutants(Generation_Array[i-1].mutant_count)
    Generation_Array[i].set_existing_mutants()
    print(f'\nGeneration {i} with size {Generation_Array[i].size}\n'
          f'Non-mutants: {Generation_Array[i].non_mutant_count}\n'
          f'New mutants: {Generation_Array[i].new_mutant_count}\n'
          f'Pre-existing mutants: {Generation_Array[i].existing_mutant_count}\n'
          # Uses estimator from remark 11.6, 1 - (non mutants of the new generation / 2 * non mutants of the previous generation)
          f'Mutation Estimator: {1 - Generation_Array[i].non_mutant_count / (2 * Generation_Array[i - 1].non_mutant_count)}\n')
    # Was used to print values and to manually check the integrity of the code
    #Generation_Array[i].print_values()
