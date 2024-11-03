import json
import os
import time
import copy
import uuid

from inspect import signature
from itertools import product, combinations
from queue import PriorityQueue

import rules
from priority_item import PriorityItem
from utils import *
from image import Image
from ARCGraph import ARCGraph

tabu_cool_down = 0


class Task:
    all_possible_abstractions = Image.abstractions
    all_possible_transformations = ARCGraph.transformation_ops

    def __init__(self, task_id, train, test):
        """
        contains all information related to an ARC task
        """

        # get task id from filepath
        self.task_id = task_id
        self.train = train
        self.test = test

        # input output images given
        self.train_input = []
        self.train_output = []
        self.test_input = []
        self.test_output = []
        self.save_images = False

        # abstracted graphs from input output images
        self.input_abstracted_graphs = dict()  # a dictionary of ARCGraphs, where the keys are the abstraction name and
        self.output_abstracted_graphs = dict()  # values are lists of ARCGraphs with the abs name for all inputs/outputs
        self.input_abstracted_graphs_original = dict()  # a dictionary of ARCGraphs, where the keys are the abstraction name and
        self.output_abstracted_graphs_original = dict()

        # metadata to be kept track of
        self.top_level_node_count = 0
        self.total_nodes_explored = 0
        self.total_unique_frontier_nodes = 0
        self.frontier_nodes_expanded = 0

        # attributes used for search
        self.shared_frontier = None  # a priority queue of frontier nodes to be expanded
        self.do_constraint_acquisition = None  # whether to do constraint acquisition or not
        self.time_limit = None  # time limit for search
        self.abstraction = None  # which type of abstraction the search is currently working with
        self.static_objects_for_insertion = dict()  # static objects used for the "insert" transformation
        self.object_sizes = dict()  # object sizes to use for filters
        self.object_degrees = dict()  # object degrees to use for filters
        self.skip_abstractions = set()  # a set of abstractions to be skipped in search
        self.transformation_ops = dict()  # a dictionary of transformation operations to be used in search
        self.frontier_hash = dict()  # used for checking if a resulting image is already found by other transformation, one set per abstraction
        self.tabu_list = {}  # used for temporarily disabling expanding frontier for a specific abstraction
        self.tabu_list_waiting = {}  # list of nodes to be added back to frontier once tabu list expired
        self.current_best_scores = {}  # used for tracking the current best score for each abstraction
        self.current_best = None    # best found solution
        self.solution_apply = None    # the apply calls that produces the best solution
        self.solution_apply_calls = []  # all partial solutions that apply to individual training examples
        self.solution_train_error = float("inf")  # the train error of the best solution
        self.current_best_score = float("inf")  # the current best score
        self.current_best_apply_call = None  # the apply call that produces the current best solution
        self.current_best_abstraction = None  # the abstraction that produced the current best solution
        self.solutions = dict()

        self.load_task_(self.train, self.test)
        self.img_dir = "images/" + self.task_id
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

    def load_task_(self, train, test):
        """
        loads the task from a json file
        """
        train_data = train
        test_data = test

        for i, data_pair in enumerate(train_data["train"]):
            task_example = self.task_id + "_" + str(i + 1)
            self.train_input.append(
                Image(self, grid=data_pair["input"],  name=task_example + "_train_in"))
            self.train_output.append(
                Image(self, grid=data_pair["output"], name=task_example + "_train_out"))
            self.solutions[i] = None

        tdata = train_data["test"][0]
        self.test_input.append(Image(self, grid=tdata["input"], name=self.task_id + "_" + "_test_in"))
        self.test_output.append(Image(self, grid=test_data, name=self.task_id + "_" + "_test_out"))

    def solve(self, shared_frontier=True, time_limit=1800, do_constraint_acquisition=True, save_images=False):
        """
        solve for a solution
        :param save_images: whether to save visualization images of the search process.
        :param shared_frontier: whether the search uses a shared frontier between abstractions.
        :param time_limit: maximum time allowed for search in seconds.
        :param do_constraint_acquisition: whether constraint acquisition is used.
        :return:
        """
        self.shared_frontier = shared_frontier
        self.do_constraint_acquisition = do_constraint_acquisition
        self.save_images = save_images
        self.time_limit = time_limit
        if shared_frontier:
            self.frontier = PriorityQueue()  # frontier for search, each item is a PriorityItem object
        else:
            self.frontier = dict()  # maintain a separate frontier for each abstraction
        
        print("Running task.solve() for #{}".format(self.task_id), flush=True)
        self.start_time = time.time()

        # initialize frontier
        stop_search = self.initialize_frontier() # create first frontier_node

        # main loop: search for next frontier_node = self.frontier.get(False)
        while not stop_search:
            if self.shared_frontier:
                solution = self.search_shared_frontier()
            else:
                solution = self.search_separate_frontier()
            if solution == None:
                print("Failed to initialize")
                return
               
            if solution.priority == 0:
                solutions = self.solutions[solution.name] 
                if solutions == None or len(solutions) == 0:
                    self.solutions[solution.name] = [solution.data]
                else: 
                    self.solutions[solution.name].append(solution.data)                              
                    
                # plot reconstructed train images and merge partial solutions
                for i, g in enumerate(self.input_abstracted_graphs_original[self.abstraction]):        
                    if self.solutions[i] == None:
                       break
                    for j, solution in enumerate(self.solutions[i]):
                        for k, call in enumerate(solution):
                            self.solution_apply_calls.append(call)
                            g.apply(**call)
                        if self.save_images: # plot each solution as _xj   
                            g.plot(save_fig=True, file_name=g.name + "_x{}".format(j))
                            reconstructed = self.train_input[i].undo_abstraction(g)
                            reconstructed.plot(save_fig=True)
                    self.train_output[i].arc_graph.plot(save_fig=True)
                       
                    # apply to test image
                    test_input = self.test_input[0]
                    abstracted_graph = getattr(test_input, Image.abstraction_ops[self.abstraction])()
                    abstracted_graph.plot(save_fig=True)
                    
                    self.solution_apply = []
                    for call in self.solution_apply_calls:
                        operation = call.copy() 
                        del operation['transformation']
                        del operation['transformation_params']     
                        if call not in self.solution_apply: #len(self.get_affected_nodes(operation, abstracted_graph)) > 0:
                            self.solution_apply.append(call)
                            
                    for call in self.solution_apply:        
                        abstracted_graph.apply(**call)
                        
                    abstracted_graph.plot(save_fig=True, file_name=abstracted_graph.name)
                    reconstructed = test_input.undo_abstraction(abstracted_graph)
                    reconstructed.plot(save_fig=True)
                    test_input.arc_graph.plot(save_fig=True)
                    self.test_output[0].arc_graph.plot(save_fig=True)

                    # check if the solution found the correct test output
                    error = 0
                    for node, data in self.test_output[0].graph.nodes(data=True):
                        if data["color"] != reconstructed.graph.nodes[node]["color"]:
                            error += 1
                    if error == 0:
                        print("Found solution!")
                        stop_search = True
                    else:
                        print("Predicted {} out of {} pixels incorrectly".format(error, len(
                            self.test_output[0].graph.nodes())))
                        print("===============================================================")
        
        for i, g in enumerate(self.input_abstracted_graphs_original[self.abstraction]):
            for call in self.solution_apply:        
                g.apply(**call)
            g.plot(save_fig=True, file_name=g.name+"_out")
    
        solving_time = time.time() - self.start_time
        nodes_explored = {"total_nodes_explored": self.total_nodes_explored,
                          "total_unique_frontier_nodes": self.total_unique_frontier_nodes,
                          "frontier_nodes_expanded": self.frontier_nodes_expanded}

        return self.abstraction, self.solution_apply, error / len(
            self.test_output[0].graph.nodes()), self.solution_train_error, solving_time, nodes_explored

    def initialize_frontier(self):
        """
        initializes frontier
        :return: True if a solution is found during initialization or time limit has been reached, False otherwise
        """
        assert(len(self.train_input)==len(self.train_output))
        
        print("Initializing Frontiers")
        
        compatible = True
        for i, in_image in enumerate(self.train_input):
            out_image = self.train_output[i]   
            if in_image.image_size != out_image.image_size:
                compatible = False
            #if self.save_images:    
            in_image.arc_graph.plot(save_fig=True)  
            out_image.arc_graph.plot(save_fig=True)       

        in_abstracted_graphs = {}  # keep track of existing abstracted graphs to check for duplication
        out_abstracted_graphs = {}

        for abstraction in self.all_possible_abstractions:
            # specify the abstraction currently working with
            self.abstraction = abstraction

            # initialize individual frontiers if abstractions do not share one
            if not self.shared_frontier:
                self.frontier[abstraction] = PriorityQueue()

            # initialize additional attributes used in search
            self.current_best_scores[abstraction] = float("inf")
            self.tabu_list[abstraction] = 0
            self.tabu_list_waiting[abstraction] = []
            self.frontier_hash[abstraction] = set()

            # first, produce the abstracted graphs for input output images using the current abstraction
            # these are the 'original' abstracted graphs that will not be updated
            self.input_abstracted_graphs_original[abstraction] = \
                [getattr(input, Image.abstraction_ops[abstraction])() for input in self.train_input]
            self.output_abstracted_graphs_original[abstraction] = \
                [getattr(output, Image.abstraction_ops[abstraction])() for output in self.train_output]
            
            self.skip_duplicates(in_abstracted_graphs, abstraction)
            
            in_abstracted_graphs[abstraction] = self.input_abstracted_graphs_original[abstraction]
            out_abstracted_graphs[abstraction] = self.output_abstracted_graphs_original[abstraction]
            if abstraction == "nbccg" and not compatible: # ex. in and out images size is different   
                in_abs_graphs, out_abs_graphs = self.fold(in_abstracted_graphs[abstraction], out_abstracted_graphs[abstraction])
                        
            # get the list of object sizes and degrees in self.input_abstracted_graphs_original[abstraction]
            self.get_static_object_attributes(abstraction)

            # keep a list of transformation ops that we modify based on constraint acquisition results
            self.transformation_ops[abstraction] = self.all_possible_transformations[self.abstraction]

            # constraint acquisition (global) for self.transformation_ops[self.abstraction] (set up right above)
            if self.do_constraint_acquisition:
                self.constraints_acquisition_global()

            # look for static objects to insert if insert transformation is not pruned by constraint acquisition
            self.static_objects_for_insertion[abstraction] = []
            if len(set(self.transformation_ops[abstraction]) & set(ARCGraph.insertion_transformation_ops)) > 0:
                self.get_static_inserted_objects()

            # initiate frontier with dummy node and expand it (representing doing nothing to the input image)
            frontier_node = PriorityItem([], abstraction, float("inf"), float("inf"), abstraction)
            self.expand_frontier(frontier_node) # top of the search tee

            if self.shared_frontier:
                if len(self.frontier.queue) == 0:  # dead end
                    self.skip_abstractions.add(self.abstraction)
                    continue #  go to another possible abstraction
                frontier_score = self.frontier.queue[0].priority # why queue[0] ??
            else:
                if len(self.frontier[self.abstraction].queue) == 0:
                    self.skip_abstractions.add(self.abstraction)
                    continue #  go to another possible abstraction
                frontier_score = self.frontier[self.abstraction].queue[0].priority

            # check if solution exists in the newly expanded frontier
            if frontier_score == 0:  # if priority is 0, the goal is reached
                if self.shared_frontier:
                    self.current_best = self.frontier.get(False)
                else:
                    self.current_best = self.frontier[self.abstraction].get(False)    
                self.solution_apply_call = self.current_best.data
                self.solution_train_error = self.current_best.priority
                print("Partial Solution Found! Abstraction used: {}, Apply Call = ".format(self.current_best.abstraction))
                print(self.current_best.data)
                print("Runtime till solution: {}".format(time.time() - self.start_time))
                print("===============================================================")
                # return True # one training case is solved but may be not all 
            """ # no timeout for debug
            if time.time() - self.start_time > self.time_limit:  # timeout
                self.solution_apply_call = frontier_node.data
                self.solution_train_error = frontier_node.priority
                self.abstraction = frontier_node.abstraction
                print("Solution Not Found! Best effort has cost of {}, Abstraction used: {}, Apply Call = ".format(
                    frontier_node.priority, self.abstraction))
                print(self.solution_apply_call)
                print("Runtime till solution: {}".format(time.time() - self.start_time))
                return True
            """
            self.top_level_node_count += 1 
        print("Found {} applicable abstractions".format(self.top_level_node_count))    
        return False # continue search for best solution

    def skip_duplicates(self, in_abstracted_graphs, abstraction):
        # skip abstraction if it results in the same set of abstracted graphs as a previous abstraction,
        # for example: nbccg and ccgbr result in the same graphs if there are no enclosed black pixels
        found_match = False
        for i, in_abs_graphs in in_abstracted_graphs.items():
            for j, in_abs_graph in enumerate(in_abs_graphs):
                existing_set = set()
                new_set = set()
                for n, subnodes1 in self.input_abstracted_graphs_original[abstraction][j].graph.nodes(
                        data="nodes"):
                    existing_set.add(frozenset(subnodes1))
                for m, subnodes2 in in_abs_graph.graph.nodes(data="nodes"):
                    new_set.add(frozenset(subnodes2))
                if existing_set != new_set:
                    break  # break if did not match for this instance
            else:  # did not break, found match for all instances
                found_match = True
                break
        if found_match:    
            print("Skipping abstraction {} = {}".format(abstraction, i))
            self.skip_abstractions.add(self.abstraction)        
        return found_match
    
    def match_size(self, decomposition, in_abs_graphs, out_abs_graphs):
        matching = {}
        for i, in_abs_graph in enumerate(in_abs_graphs):
            matching[in_abs_graph.name] = []
            for in_graph in decomposition[in_abs_graph.name]:
                out_abs_graph = out_abs_graphs[i]
                for out_graph in decomposition[out_abs_graph.name]:
                    if out_graph.image.image_size == in_graph.image.image_size:
                        matching[in_abs_graph.name].append(in_graph)
                        #if self.save_images:
                        in_graph.plot(save_fig=True)
        return matching                        
    
    def fold(self, in_abstracted_graphs, out_abstracted_graphs):
        match_in = {}
        match_out = {}
        decomposition = self.carve(in_abstracted_graphs, out_abstracted_graphs)     
        for name, components in decomposition.items(): #ex. name '0520fde7_1_train_in_nbccg'
            if len(components) > 1:
                if "_in_" in name:
                    match_in = self.match_size(decomposition, in_abstracted_graphs, out_abstracted_graphs)
                if "_out_" in name:
                    match_out = self.match_size(decomposition, out_abstracted_graphs, in_abstracted_graphs)  
        
        if len(match_in) == 0 and len(match_out) == 0:
            print("Failed to decompose abstraction!") 
        
        overlay = None    
        if len(match_in) > 0:
            for name, components in match_in.items():
                print(name+": {} to one fold".format(len(components)))

                background_color = 0
                for i, component in enumerate(components):
                    if not overlay:
                        background_color = component.image.background_color
                        image = Image(self, overlay, component.width, component.height, overlay, component.name[-2])
                        overlay = image.arc_graph
                        overlay.image.background_color = background_color
                        overlay.name = component.name[0:-2] + "_O"
        
                    if i < len(components):
                        next = components[i]
                        for node, data in component.graph.nodes(data=True):
                            component_color = data["color"]
                            next_color = next.graph.nodes[node]["color"]
                            if component_color != background_color:
                                if next_color != next.image.background_color:
                                    overlay.graph.nodes[node]["color"] = component_color     
                                    overlay.plot(save_fig=True)  
                        
        #print(name+": One to {}".format(len(components)))    

        # return modified copies!
        return in_abstracted_graphs, out_abstracted_graphs
                    
    def carve(self, in_abs_graphs, out_abs_graphs):
        decomposition = {}
        for i, in_abs_graph in enumerate(in_abs_graphs):
            in_graph = in_abs_graph.undo_abstraction()
            out_abs_graph = out_abs_graphs[i]
            out_graph = out_abs_graph.undo_abstraction()
             
            decomposition[in_abs_graph.name] = []
            joints = in_abs_graph.find_common_descendants()
            if len(joints) > 0: 
                components = in_abs_graph.carve_at(joints)
                for j, component in enumerate(components):
                    name = in_abs_graph.name + "_Y_{}".format(j+1)
                    graph = ARCGraph(component,name,Image(self,graph=component,name=name))
                    in_graph.copy_colors_to(graph)
                    in_abs_graph.overlap(graph, out_graph.graph)
                    decomposition[in_abs_graph.name].append(graph)    
            else:
                decomposition[in_abs_graph.name].append(in_graph)    
    
            decomposition[out_abs_graph.name] = []         
            joints = out_abs_graph.find_common_descendants()
            if len(joints) > 0: 
                components = out_abs_graph.carve_at(joints)        
                for j, component in enumerate(components):
                    name = out_abs_graph.name + "_Y_{}".format(j+1)
                    graph = ARCGraph(component,name,Image(self,graph=component,name=name))
                    out_graph.copy_colors_to(graph)
                    out_abs_graph.overlap(graph, in_graph.graph)
                    decomposition[out_abs_graph.name].append(graph)   
            else:
                decomposition[out_abs_graph.name].append(out_graph)         
        return decomposition  
       
    def list_isomorph(self, in_abs_graphs, out_abs_graphs):
        for i, in_abs_graph in enumerate(in_abs_graphs):
            graphs = in_abs_graph.graph.get(out_abs_graphs[i].graph)
        return list(graphs)
      
    def search_shared_frontier(self):
        """
        perform one iteration of search for a solution using a shared frontier
        :return: True if a solution is found or time limit has been reached, False otherwise
        """
 
        if self.frontier.empty():  # if frontier queue is empty(dead branch of search tree) - go to other branches
            self.solution_apply_call = self.current_best_apply_call
            self.solution_train_error = self.current_best_score
            self.abstraction = self.current_best_abstraction
            print("Empty Frontier is reached! Best score: {}, Abstraction used: {}, Apply Call = ".format(self.current_best_score, self.abstraction))
            print(self.current_best_apply_call)
            print("Runtime till solution: {}".format(time.time() - self.start_time))
            return self.current_best # search space is exhosted    

        frontier_node = self.frontier.get() 

        # if this abstraction is on tabu list, explore something else
        if self.tabu_list[frontier_node.abstraction] > 0:
            print("abstraction {} is in the tabu list with cool down = {}".format(frontier_node.abstraction, self.tabu_list[frontier_node.abstraction]))
            self.tabu_list_waiting[frontier_node.abstraction].append(frontier_node)
            return frontier_node
        
        # if this abstraction is not on tabu list, but has a worse score than before,
        """ # explore it and put it on tabu list
        elif frontier_node.priority >= self.current_best_scores[frontier_node.abstraction]:
            self.tabu_list[frontier_node.abstraction] = tabu_cool_down + 1
        else:
            self.current_best_scores[frontier_node.abstraction] = frontier_node.priority
        """
        apply_calls = frontier_node.data # seach path
        self.abstraction = frontier_node.abstraction

        # check for solution
        if frontier_node.priority == 0:  # if priority is 0, the goal is reached
            self.current_best = frontier_node
            self.solution_apply_call = apply_calls
            self.solution_train_error = 0
            print("Patial Solution Found! Abstraction used: {}, Apply Call = ".format(frontier_node.abstraction))
            print(apply_calls)
            print("Runtime till solution: {}".format(time.time() - self.start_time))
            print("===============================================================")
            return frontier_node # one training case is solved but may be not all
        else:
            if frontier_node.priority < self.current_best_score:
                self.current_best = frontier_node 
                self.current_best_score = frontier_node.priority
                self.current_best_apply_call = apply_calls
                self.current_best_abstraction = self.abstraction
        """"
        print("Loss = {} at depth {} with abstraction {} and path: ".format(
            frontier_node.priority, len(apply_calls), self.abstraction))
        print(apply_calls)
        """
        self.expand_frontier(frontier_node)

        all_on_tabu = all(tabu > 0 for tabu in self.tabu_list.values())
        for abs, tabu in self.tabu_list.items():
            if all_on_tabu:
                self.tabu_list[abs] = 0
                for node in self.tabu_list_waiting[abs]:  # put the nodes in waiting list back into frontier
                    self.frontier.put(node)
            elif tabu > 0:
                self.tabu_list[abs] = tabu - 1
                if tabu - 1 == 0:
                    for node in self.tabu_list_waiting[abs]:  # put the nodes in waiting list back into frontier
                        self.frontier.put(node)
        """
        if time.time() - self.start_time > self.time_limit:  # timeout
            self.solution_apply_call = self.current_best_apply_call
            self.solution_train_error = self.current_best_score
            self.abstraction = self.current_best_abstraction
            print("Solution Not Found due to time limit reached! Best effort has cost of {}, "
                  "Abstraction used: {}, Apply Call = ".format(self.current_best_score, self.abstraction))
            print(self.current_best_apply_call)
            print("Runtime till solution: {}".format(time.time() - self.start_time))
            return True
        """    
        return frontier_node

    def search_separate_frontier(self):
        """
        perform one iteration of search for a solution using a multiple frontiers
        :return: True if a solution is found or time limit has been reached, False otherwise
        """

        for abstraction in Image.abstractions:
            self.abstraction = abstraction

            if self.abstraction in self.skip_abstractions:
                continue

            # if this abstraction is on tabu list, explore something else
            if self.tabu_list[self.abstraction] > 0:
                self.tabu_list[self.abstraction] = self.tabu_list[self.abstraction] - 1
                continue

            frontier_node = self.frontier[self.abstraction].get() # block=True to wait if empty
            apply_calls = frontier_node.data

            # if this abstraction is not on tabu list, but has a worse score than before,
            """ # explore it and put it on tabu list
            if frontier_node.priority >= self.current_best_scores[self.abstraction]:
                # print("abstraction {} is put on the tabu list".format(frontier_node.abstraction))
                self.tabu_list[self.abstraction] = tabu_cool_down + 1
            else:
                self.current_best_scores[self.abstraction] = frontier_node.priority
            """    

            # check for solution
            if frontier_node.priority == 0:  # if priority is 0, the goal is reached
                self.current_best = frontier_node
                self.solution_apply_call = apply_calls
                self.solution_train_error = 0
                print("Solution Found! Abstraction used: {}, Apply Call = ".format(self.abstraction))
                print(apply_calls)
                print("Runtime till solution: {}".format(time.time() - self.start_time))
                print("===============================================================")
                return frontier_node # one training case is solved but may be not all
            else:
                if frontier_node.priority < self.current_best_score:
                    self.current_best_score = frontier_node.priority
                    self.current_best_apply_call = apply_calls
                    self.current_best_abstraction = self.abstraction
                
            print("Loss = {} at depth {} with abstraction {} and apply calls:".format(
                    frontier_node.priority, len(apply_calls), self.abstraction))
            print(apply_calls)
            self.expand_frontier(frontier_node)
            """
            if time.time() - self.start_time > self.time_limit:  # timeout
                self.solution_apply_call = self.current_best_apply_call
                self.solution_train_error = self.current_best_score
                self.abstraction = self.current_best_abstraction
                print(
                    "Solution Not Found! Best effort has cost of {}, Abstraction used: {}, Apply Call = ".format(
                        self.current_best_score, self.abstraction))
                print(self.current_best_apply_call)
                print("Runtime till solution: {}".format(time.time() - self.start_time))
                return True
             """   
        return frontier_node

    def expand_frontier(self, frontier_node):
        """
        expand one frontier node
        """
        
        # apply the parent frontier.data to the parent node abstractions to build up new base
        self.input_abstracted_graphs[self.abstraction] = []  # up-to-date abstracted graphs
        for i, input_abstracted_original in enumerate(self.input_abstracted_graphs_original[self.abstraction]):
            self.frontier_nodes_expanded += 1
            print("___________________________________")
            print("Expanding frontier using abstraction {}".format(input_abstracted_original.name))
            
            token_original = ''
            for c in range(self.train_input[i].width):
                for r in range(self.train_input[i].height):
                    token_original = token_original + str(self.train_input[i].graph.nodes[(r, c)]["color"])      
            
            input_abstracted_graph = input_abstracted_original.copy()   # copy original(init.) abstracted graph
            if len(self.input_abstracted_graphs[self.abstraction]) > i: # if working copy exist, use it instead
                input_abstracted_graph = self.input_abstracted_graphs[self.abstraction][i] 
             
            tree_branch = frontier_node.data # all operations that were tried and passed (aka search tree branch)     
            for apply_call in tree_branch:
                input_abstracted_graph.apply(**apply_call)  # apply all tried and passed operations to base
                
            self.input_abstracted_graphs[self.abstraction].append(input_abstracted_graph) # current frontier
            print("Applied {} operations to frontier {}".format(len(tree_branch),input_abstracted_graph.name))

            filters = self.get_candidate_filters(input_abstracted_graph)
            apply_calls = self.get_candidate_transformations(filters, i) # i = index pointing to current example
            print("Found {} applicable operations for {}".format(len(apply_calls),input_abstracted_graph.name))
            
            added_nodes = 0
            # for apply_call in tqdm(apply_calls):
            for apply_call in apply_calls: # try each proposed operation on fresh copy of base example (above)            
                search_path = tree_branch.copy()  # calls already tried by this frontier (aka search tree branch)
                if apply_call not in search_path:
                    search_path.append(apply_call) # append newly proposed operation for this frotier
                
                input_abstracted_copy = input_abstracted_graph.copy() # fresh copy of base graph to apply_call 
                label = self.apply(search_path, input_abstracted_copy) # try new branch of the search tree 
                self.total_nodes_explored += 1    
                        
                #for i, output in enumerate(self.train_output): # superimpose ouptut over transformed input 
                # create ARCGraph using input image.width/height/background_color, and color the graph
                image = self.train_input[i] # image of original input is used to undo abstraction
                reconstructed = image.undo_abstraction(input_abstracted_copy) # working copy after apply_calls

                token_string = ''
                for c in range(self.train_output[i].width):
                    for r in range(self.train_output[i].height):
                        token_string = token_string + str(reconstructed.graph.nodes[(r, c)]["color"])
                        
                if token_original == token_string:
                   continue # skip operation if reconstracted the same as original input
                                 
                token_string = token_string + "_"
                """
                for node, data in input_abstracted_graphs[i].graph.nodes(data=True):
                    # node order does not correspond to graph plot (ignores adjacency)
                    for j, pixel in enumerate(data["nodes"]):
                        if input_abstracted_graphs[i].is_multicolor:
                            token_string = token_string + str(data["color"][j])
                        else:
                            token_string = token_string + str(data["color"])
                token_string = token_string + "_"
                """
                
                # score (different pixels vs output) after applying operations to abstracted graph
                primary_score = 0                
                for node, data in self.train_output[i].graph.nodes(data=True):
                    if data["color"] != reconstructed.graph.nodes[node]["color"]:
                        if data["color"] == self.train_output[i].background_color or reconstructed.graph.nodes[node]["color"] == self.train_output[i].background_color:
                            # incorrectly identified object/background
                            primary_score += 2
                        else:  # correctly identified object/background but got the color wrong
                            primary_score += 1
                            
                # commulative results across all test cases after scoring         
                if token_string in self.frontier_hash[self.abstraction]: # common hash for all frontiers
                    """ # no timeout for frbug
                    if (time.time() - self.start_time) > self.time_limit:
                        break
                    """    
                    #print("Frontier did not produced new abstraction: "+ label+token_string)
                    continue # reject proposed Candidate Node of the search tree (aka dead end)
                else:
                    added_nodes += 1
                    self.frontier_hash[self.abstraction].add(token_string) # one hash for all examples!
                    
                    if self.save_images:
                        reconstructed.plot(save_fig=True, file_name=reconstructed.name + "_("+str(len(search_path))+")_" + label + token_string)
                        try:
                            input_abstracted_copy.plot(save_fig=True, file_name=input_abstracted_copy.name + "_("+str(len(search_path))+")_" + label + token_string)
                        except:
                            print("Failed to plot graph: {}".format(input_abstracted_copy.name + "_"  + label + token_string))

                    secondary_score = len(search_path) # depth of search tree- how many (operations) were applied
                    
                    # create next frontier 
                    priority_item = PriorityItem(search_path.copy(), self.abstraction, primary_score, secondary_score, i)
                    if self.shared_frontier:
                        self.frontier.put(priority_item) # create next frontier node using expanded cumulated_apply_calls
                    else:
                        self.frontier[self.abstraction].put(priority_item) # use separate frontier for each abstraction
                    
                    """
                    # stop if solution is found or time is up
                    if primary_score == 0 or secondary_score == 0:
                        break 
                    # no timeout for frbug
                    if (time.time() - self.start_time) > self.time_limit:
                        break
                    """

            self.total_unique_frontier_nodes += added_nodes
                
        print("Added {} new branches to frontier {}".format(added_nodes, input_abstracted_graph.name))
        print("______{} in total__________________".format(self.total_unique_frontier_nodes))
        return False

    def get_candidate_filters(self, input_abstracted_graph):
        """
        return list of candidate filters
        """
        apply_filter_calls = []  # final list of filter calls
        filtered_nodes_all = []  # use this list to avoid filters that return the same set of nodes

        for filter_op in ARCGraph.filter_ops:
            # first, we generate all possible values for each parameter
            sig = signature(getattr(ARCGraph, filter_op))
            generated_params = []
            for param in sig.parameters:
                param_name = sig.parameters[param].name
                param_type = sig.parameters[param].annotation
                param_default = sig.parameters[param].default
                if param_name == "self" or param_name == "node":
                    continue
                if param_name == "color":
                    generated_params.append([c for c in range(10)] + ["most", "least"])
                elif param_name == "size":
                    generated_params.append([w for w in self.object_sizes[self.abstraction]] + ["min", "max", "odd"])
                elif param_name == "degree":
                    generated_params.append([d for d in self.object_degrees[self.abstraction]] + ["min", "max", "odd"])
                elif param_type == bool:
                    generated_params.append([False, True])
                elif issubclass(param_type, Enum):
                    generated_params.append([value for value in param_type])

            # then, we combine all generated values to get all possible combinations of parameters
            for item in product(*generated_params): # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'most', 'least'], [False, True]]
                # generate dictionary, keys are the parameter names, values are the corresponding values
                param_vals = {}
                for i, param in enumerate(list(sig.parameters)[2:]):  # skip "self", "node"
                    param_vals[sig.parameters[param].name] = item[i]
                    
                candidate_filter = {"filters": [filter_op], "filter_params": [param_vals]}
                if len(self.get_affected_nodes(candidate_filter, input_abstracted_graph)) > 0:
                    apply_filter_calls.append(candidate_filter)
        """        
        # generate filter calls with two filters for each possible combination of two candidate filters
        single_filter_calls = [d.copy() for d in ret_apply_filter_calls] # make a copy of single filters
        for filter_i, (first_filter_call, second_filter_call) in enumerate(combinations(single_filter_calls, 2)):
            candidate_filter = copy.deepcopy(first_filter_call)
            candidate_filter["filters"].extend(second_filter_call["filters"])
            candidate_filter["filter_params"].extend(second_filter_call["filter_params"])
            ret_apply_filter_calls.append(candidate_filter)
        """    
        return apply_filter_calls

    def get_affected_nodes(self, candidate_filter, input_abstracted_graph):
         #  do not include if the filter result in empty set of nodes (this will be the majority of filters)
        filtered_nodes = []
        for node in input_abstracted_graph.graph.nodes():
            if input_abstracted_graph.apply_filters(node, **candidate_filter):
                filtered_nodes.append(node) # allowed by candidate_filter
        return filtered_nodes 

    def get_candidate_transformations(self, apply_filters_calls, index):
        """
        generate candidate transformations, return list of full operations candidates
        """

        apply_calls = []
        for apply_filters_call in apply_filters_calls:
            """
            if time.time() - self.start_time > self.time_limit:
                break
            """    
            if self.do_constraint_acquisition:
                constraints = self.constraints_acquisition_local(apply_filters_call, index)
                transformation_ops = self.prune_transformations(constraints)
            else:
                transformation_ops = self.transformation_ops[self.abstraction]
            
            filter = apply_filters_call['filters'][0]
            filter_op = filter.split('_')[-1] # last part of filter
            filter_params = apply_filters_call['filter_params'][0]
            filter_value = filter_params[filter_op]
        
            for transform_op in transformation_ops:
                sig = signature(getattr(ARCGraph, transform_op))
                generated_params = self.parameters_generation(apply_filters_call, sig)
                for item in product(*generated_params): # ex. 96 filters = 12 colors (10+most+least) x 4 bindings x 2(Include/Exclude)
                    param_vals = {}
                    trans_value = None
                    for i, param in enumerate(list(sig.parameters)[2:]):  # skip "self", "node"
                        param_vals[sig.parameters[param].name] = item[i]
                    try:
                       trans_value = param_vals[filter_op]     
                    except:
                        pass    
                    if trans_value != filter_value: # no need to update to the same value as filter 
                        apply_call = apply_filters_call #.copy()  # dont need deep copy here since we are not modifying existing entries
                        apply_call["transformation"] = [transform_op]
                        apply_call["transformation_params"] = [param_vals]
                        if apply_call not in apply_calls:
                            apply_calls.append(apply_call.copy())
        return apply_calls # ex, 4,608 calls = 96 filters X (12 tranforms for 'nbccg' graph - 8 pruned) x 12 colors for update_color

    def parameters_generation(self, apply_filters_call, transform_sig): # signature of transform function
        """
        given filter nodes and a transformation, generate parameters to be passed to the transformation
        example: given filters for red nodes and move_node_max,
            return [up, down, left, right, get_relative_pos(red nodes, blue neighbors of red nodes), ...]
        :param apply_filters_call: the specific apply filter call to get the nodes to apply transformations to
        :param all_calls: all apply filter calls, this is used to generate the dynamic parameters
        :param transform_sig: signature for a transformation
        :return: parameters to be passed to the transformation
        """
        generated_params = []
        for param in transform_sig.parameters:
            param_name = transform_sig.parameters[param].name
            param_type = transform_sig.parameters[param].annotation
            param_default = transform_sig.parameters[param].default
            if param_name == "self" or param_name == "node":  # nodes are already generated using the filters
                continue

            # first we generate the static values
            if param_name == "color":
                all_possible_values = [c for c in range(10)] + ["most", "least"]
            elif param_name == "fill_color" or param_name == "border_color":
                all_possible_values = [c for c in range(10)]
            elif param_name == "object_id":
                all_possible_values = [id for id in range(len(self.static_objects_for_insertion[self.abstraction]))] + [
                    -1]
            elif param_name == "point":  # for insertion, could be ImagePoints or a coordinate on image (tuple)
                all_possible_values = [value for value in ImagePoints]
            elif issubclass(param_type, Enum):
                all_possible_values = [value for value in param_type]
            elif param_type == bool:
                all_possible_values = [False]#,True]
            elif param_default is None:
                all_possible_values = [None]
            else:
                all_possible_values = []

            # add dynamic values for all parameters with all possible dynamic bindings
            if param_name in ARCGraph.dynamic_parameters:
                filtered_nodes_all = []
                # the filters that defines the dynamic parameter values, has their own parameters generated_filter_params
                for param_binding_op in ARCGraph.param_binding_ops: # ex. 'param_bind_node_by_size', etc...
                    sig = signature(getattr(ARCGraph, param_binding_op))
                    generated_filter_params = []
                    for param in sig.parameters: # is 'mirror_direction' in param_bind_node_by_size(self, node, size, exclude: bool = False):
                        filter_param_name = sig.parameters[param].name
                        filter_param_type = sig.parameters[param].annotation
                        if filter_param_name == "self" or filter_param_name == "node":
                            continue
                        if filter_param_name == "color":
                            # generated_params[param_name] = [c for c in range(10)]
                            generated_filter_params.append([c for c in range(10)] + ["most", "least"])
                        elif filter_param_name == "size":
                            generated_filter_params.append( # ex. possible sizes [9, 'min', 'max'] for 'na'
                                [w for w in self.object_sizes[self.abstraction]] + ["min", "max"])
                        elif filter_param_type == bool:
                            # ex. for each possible size listed above, we can Include [True, False]
                            generated_filter_params.append([False,True])
                        elif issubclass(filter_param_type, Enum):
                            generated_filter_params.append([value for value in filter_param_type])

                    for item in product(*generated_filter_params): # ex. [[9, 'min', 'max'], [True, False]]
                        param_vals = {}
                        for i, param in enumerate(list(sig.parameters)[2:]):  # skip "self", "node"
                            param_vals[sig.parameters[param].name] = item[i]
                            
                        applicable_to_all = True    # assume all abstracted graphs have safisfying nodes 
                        param_bind_nodes = []       # all nodes satisfying the given apply_filters_call
                        for input_abstracted_graph in self.input_abstracted_graphs[self.abstraction]:
                            param_bind_nodes_found = []
                            for filtered_node in input_abstracted_graph.graph.nodes():
                            # if filtered_node satisfies ex. param_bind_neighbor_by_size(self, node, size, exclude: bool = False)    
                                if input_abstracted_graph.apply_filters(filtered_node, **apply_filters_call):
                                    # return first node satisfying the binding (ex. filtered_node is the neighbor matching the size)
                                    param_binded_node = getattr(input_abstracted_graph, param_binding_op)(filtered_node,**param_vals)
                                    if param_binded_node is None:
                                        # if there is no matching neighbor of binding size or color or shape etc...
                                        applicable_to_all = False
                                        break
                                    #else ex. found the neighbor satifying the binding above  
                                    param_bind_nodes_found.append(param_binded_node)

                            if len(param_bind_nodes_found) > 0:
                                param_bind_nodes.append(param_bind_nodes_found)
                            else:    
                                applicable_to_all = False # not all test cases allow binding!
                                
                        if applicable_to_all and param_bind_nodes not in filtered_nodes_all:
                            all_possible_values.append({"filters": [param_binding_op], "filter_params": [param_vals]})
                            filtered_nodes_all.append(param_bind_nodes)
            generated_params.append(all_possible_values)
        return generated_params

    def apply(self, apply_calls, input_abstracted_copy):
        """
        explore current frontier by applying all operations to all abstractions and create unique tree node label 
        """
        label = str()
        try:
            #for input_abstracted_graph in input_abstracted_graphs: # copies of abstracted graphs 
            for apply_call in apply_calls: # apply operation to each abstracted graph copy
                call_copy = apply_call#.copy()
                transformation = call_copy['transformation'] # not origical operations!
                if len(transformation)==0:
                    print("Empty transformation skipped: {}".format(input_abstracted_copy.name + str(call_copy)))
                    return label
                operation = str(transformation[0])
                
                if len(operation)==0:
                    print("Empty operation skipped: {}".format(input_abstracted_copy.name + str(call_copy)))
                    return label
                if operation not in label:
                    label = label + operation + "_"
                                                                        
                param = call_copy['transformation_params'][0]
                sig = signature(getattr(ARCGraph, operation))
                param_type = list(sig.parameters)[2:]
                
                param_object = param.get(param_type[0]) # JSON object
                object_label = "object" # defualt
                if isinstance(param_object,Enum):
                    object_label = param_object.name
                elif isinstance(param_object,dict):  
                    param_filters = param_object.get('filters')     
                else:
                    try:
                        object_label = next(iter(param_object))
                    except:
                        object_label = param_object
                if "_"+str(object_label)+"_" not in label: 
                    label = label + str(object_label) + "_"
                
                filter_params = call_copy['filter_params'] 
                if isinstance(filter_params,list) or isinstance(filter_params,dict):
                    params = iter(filter_params)
                    param_pair = next(params, None)
                    if param_pair != None:
                        if isinstance(param_pair,str):
                            if "_"+param_pair not in label: 
                                label = label + param_pair + "_" 
                        elif isinstance(param_pair,dict):
                            param_values = iter(param_pair) 
                            param_name = next(param_values, None)
                            if param_name != None and isinstance(param_name,str):    
                                if "_"+param_name+"_" not in label: 
                                    label = label + param_name + "_"
                                param_value = param_pair.get(param_name)
                                if "_"+str(param_value)+"_" not in label: 
                                    label = label + str(param_value) + "_"
                            param_name = next(param_values, None)
                            if param_name != None and isinstance(param_name,str):    
                                if "_"+param_name+"_" not in label: 
                                    label = label + param_name + "_"
                                param_value = param_pair.get(param_name)
                                if "_"+str(param_value)+"_" not in label: 
                                    label = label + str(param_value) + "_"    
                                    
                if isinstance(param_object,dict):
                    param_filters = param_object.get('filters')
                    bindings = iter(param_filters)
                    binding = next(bindings, None)
                    if binding != None :
                        if "_"+str(binding)+"_" not in label: 
                            label = label + str(binding) + "_"                                
                    bind_cond = param_object.get('filter_params')        
                    if bind_cond != None and isinstance(bind_cond,list):
                        conditions = iter(bind_cond) # name/vale conditions
                        cond_pair = next(conditions,None)
                        if cond_pair != None:
                            if isinstance(cond_pair,dict):
                                cond_values = iter(cond_pair) 
                                cond_name = next(cond_values, None)
                                if cond_name != None and isinstance(cond_name,str):
                                    if "_"+cond_name+"_" not in label: # ex. color 
                                        label = label + cond_name + "_"
                                    cond_value = cond_pair.get(cond_name)
                                    #if "_"+str(cond_value)+"_" not in label: # ex. 1
                                    label = label + str(cond_value) + "_"                   
                            elif isinstance(cond_pair,list):
                                cond_name = next(conditions,None)
                                if cond_name != None:            
                                    if "_"+str(cond_name)+"_" not in label: 
                                        label = label + str(cond_name) + "_"
                                    cond_value = next(conditions,None)
                                    if cond_value != None and isinstance(cond_value,dict):
                                        values =  iter(cond_value)
                                        bind_value = next(values,None)
                                        if bind_value != None: 
                                            if "_"+str(bind_value)+"_" not in label: 
                                                label = label + str(bind_value) + "_"
                                        bind_value = next(values,None)
                                        if bind_value != None: 
                                            if "_"+str(bind_value)+"_" not in label: 
                                                label = label + str(bind_value) + "_"
                                        
                if len(param_type)==0:
                    print("Invalid transformation parameter type skipped: {}".format(input_abstracted_copy.name+'='+str(param_type)))
                    return label
                        
                # apply all cumulated calls to the original(!!!) abstracted grpah    
                input_abstracted_copy.apply(**apply_call) # original, not copy !! 
        except:     # ValueError:
            print("Aborted transformation {}".format(input_abstracted_copy.name + '-' + label))
            return label

        return label

    # --------------------------------------Constraint Acquisition-----------------------------------
    def constraints_acquisition_global(self):
        """
        find the constraints that all nodes in the instance must follow
        """
        no_movements = True
        for i, input in enumerate(self.train_input):
            for node, data in input.graph.nodes(data=True):
                output_nodes = self.train_output[i].graph.nodes
                if output_nodes == None or node not in output_nodes:
                    continue
                if (data["color"] != input.background_color and output_nodes[node]["color"] == input.background_color) \
                    or (data["color"] == input.background_color and output_nodes[node]["color"] != input.background_color):
                    no_movements = False
        no_new_objects = True
        for i, output_abstracted_graph in enumerate(self.output_abstracted_graphs_original[self.abstraction]):
            input_abstracted_nodes = self.input_abstracted_graphs_original[self.abstraction][i].graph.nodes()
            for abstracted_node, data in output_abstracted_graph.graph.nodes(data=True):
                if abstracted_node not in input_abstracted_nodes:
                    no_new_objects = False
                    break
        if no_movements:
            pruned_transformations = ["move_node", "extend_node", "move_node_max", "fill_rectangle", "add_border",
                                      "insert"]
            self.transformation_ops[self.abstraction] = [t for t in self.transformation_ops[self.abstraction] if
                                                         t not in pruned_transformations]
        elif no_new_objects:
            pruned_transformations = ["insert"]
            self.transformation_ops[self.abstraction] = [t for t in self.transformation_ops[self.abstraction] if
                                                         t not in pruned_transformations]

    def constraints_acquisition_local(self, apply_filter_call, index):
        """
        given an apply_filter_call, find the set of constraints that
        the nodes returned by the apply_filter_call must satisfy.
        these are called local constraints as they apply to only the nodes
        that satisfies the filter.
        """
        found_constraints = []
        for rule in rules.list_of_rules:
            if self.apply_constraint(rule, apply_filter_call, index):
                found_constraints.append(rule)
        return found_constraints

    def apply_constraint(self, rule, apply_filter_call, index):
        """
        check if the given rule holds for all training instances for the given apply_filter_call
        """
        satisfied = True
        #for index in range(len(self.train_input)):
        params = self.constraints_param_generation(apply_filter_call, rule, index)
        satisfied = satisfied and getattr(rules, rule)(*params) # ex. [[1], []] not equal sequences
        return satisfied

    def constraints_param_generation(self, condition, rule, index):
        """
        given condition and rule, first generate the sequence using the condition
        then transform the sequence into the expected format for the constraint
        :param condition: {'filters': ['filter_nodes_by_color'],
          'filter_params': [{'color': 0, 'exclude': True}]}
        :param rule: "rule_name"
        :param training_index: training instance index
        """

        input_abs = self.input_abstracted_graphs[self.abstraction][index]
        output_abs = self.output_abstracted_graphs_original[self.abstraction][index]

        input_nodes = []
        for node in input_abs.graph.nodes():
            if input_abs.apply_filters(node, **condition):
                input_nodes.append(node)

        output_nodes = []
        for node in output_abs.graph.nodes():
            if output_abs.apply_filters(node, **condition):
                output_nodes.append(node)

        if rule == "color_equal":
            input_sequence = [input_abs.graph.nodes[node]["color"] for node in input_nodes]
            output_sequence = [output_abs.graph.nodes[node]["color"] for node in output_nodes]
            input_sequence.sort()
            output_sequence.sort()
            args = [input_sequence, output_sequence]

        elif rule == "position_equal":
            input_sequence = []
            output_sequence = []
            for node in input_nodes:
                input_sequence.extend([subnode for subnode in input_abs.graph.nodes[node]["nodes"]])
            for node in output_nodes:
                output_sequence.extend([subnode for subnode in output_abs.graph.nodes[node]["nodes"]])
            input_sequence.sort()
            output_sequence.sort()
            args = [input_sequence, output_sequence]

        elif rule == "size_equal":
            input_sequence = [input_abs.graph.nodes[node]["size"] for node in input_nodes]
            output_sequence = [output_abs.graph.nodes[node]["size"] for node in output_nodes]
            input_sequence.sort()
            output_sequence.sort()
            args = [input_sequence, output_sequence]
        return args

    def prune_transformations(self, constraints):
        """
        given a set of constraints that must be satisfied, return a set of transformations that do not violate them
        """
        transformations = self.transformation_ops[self.abstraction]
        for constraint in constraints:
            if constraint == "color_equal":
                pruned_transformations = ["update_color"]
            elif constraint == "position_equal":
                pruned_transformations = ["move_node", "extend_node", "move_node_max"]
            elif constraint == "size_equal":
                pruned_transformations = ["extend_node"]
            transformations = [t for t in transformations if t not in pruned_transformations]
        return transformations

    # --------------------------------- Utility Functions ---------------------------------
    def get_static_inserted_objects(self):
        """
        populate self.static_objects_for_insertion, which contains all static objects detected in the images.
        """
        self.static_objects_for_insertion[self.abstraction] = []
        existing_objects = []

        for i, output_abstracted_graph in enumerate(self.output_abstracted_graphs_original[self.abstraction]):
            # difference_image = self.train_output[i].copy()
            input_abstracted_nodes = self.input_abstracted_graphs_original[self.abstraction][i].graph.nodes()
            for abstracted_node, data in output_abstracted_graph.graph.nodes(data=True):
                if abstracted_node not in input_abstracted_nodes:
                    new_object = data.copy()
                    min_x = min([subnode[1] for subnode in new_object["nodes"]])
                    min_y = min([subnode[0] for subnode in new_object["nodes"]])
                    adjusted_subnodes = []
                    for subnode in new_object["nodes"]:
                        adjusted_subnodes.append((subnode[0] - min_y, subnode[1] - min_x))
                    adjusted_subnodes.sort()
                    if adjusted_subnodes not in existing_objects:
                        existing_objects.append(adjusted_subnodes)
                        self.static_objects_for_insertion[self.abstraction].append(new_object)

    def get_static_object_attributes(self, abstraction):
        """
        populate self.object_sizes and self.object_degrees, which contains all sizes and degrees existing objects
        """
        self.object_sizes[abstraction] = set()
        self.object_degrees[abstraction] = set()
        for abs_graph in self.input_abstracted_graphs_original[abstraction]:
            for node, size in abs_graph.graph.nodes(data="size"):
                self.object_sizes[abstraction].add(size)
            for node, degree in abs_graph.graph.degree():
                self.object_degrees[abstraction].add(degree)

    def apply_solution(self, apply_call, abstraction, save_images=False):
        """
        apply solution abstraction and apply_call to test image
        """
        self.abstraction = abstraction
        self.input_abstracted_graphs_original[abstraction] = [getattr(input, Image.abstraction_ops[abstraction])() for
                                                              input in self.train_input]
        self.output_abstracted_graphs_original[abstraction] = [getattr(output, Image.abstraction_ops[abstraction])() for
                                                               output in self.train_output]
        self.get_static_inserted_objects()
        test_input = self.test_input[0]
        abstracted_graph = getattr(test_input, Image.abstraction_ops[abstraction])()
        for call in apply_call:
            abstracted_graph.apply(**call)
        reconstructed = test_input.undo_abstraction(abstracted_graph)
        if save_images:
            test_input.arc_graph.plot(save_fig=True)
            reconstructed.plot(save_fig=True)
            self.test_output[0].arc_graph.plot(save_fig=True)
        return reconstructed
