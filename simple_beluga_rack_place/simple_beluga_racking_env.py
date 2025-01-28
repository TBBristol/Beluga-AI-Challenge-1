import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os
import json
from itertools import pairwise

def pick_random_instance_from_folder(instance_folder_path):
    """
    Picks a random instance from the training set
    """
    json_file = random.choice(os.listdir(instance_folder_path))
    with open(os.path.join(instance_folder_path, json_file), 'r') as f:
        instance = json.load(f)
    return instance


class BelugaRackingEnv(gym.Env):
    """
    A Beluga Racking environment to train a model to place containers in a Beluga Rack. This env will not pay attention
    to the actual flights and will just be trained to place jigs on the racks in the best position.
    """
    def __init__(self, instance_folder_path, seed=None, max_steps=None):
        super().__init__()
        self.instance_folder_path = instance_folder_path

        #Needs to reset to load an instance
        self.reset(seed=seed)
        # Actions: choose which stack to place the next container
        self.action_space = spaces.Discrete(len(self.racks))
        # Observation space: 8 features for each rack, we pad to 20 racks as that is max
        self.observation_space = spaces.Box(
            low=0.0,
            high=float('inf'),  # or refine as needed
            shape=(20, 8),
            dtype=np.float32
        )


    #TODO action space changes with each instance how do we handle this?

    #Functions to get priority data only call these intially and we will make a list and edit it as we go

    def priority_jigs_for_hangars(self) -> dict:
        """
        Returns a dictionary with the priority of each jig that needs to go to a hangar.
        Priority 1 is the highest priority
        There will be a priority 1 for each hangar and 2,3,4 etc

        Returns:
            _type_: Dict with jig(str)as key and priority(int) as value
            Example:
            {
                "jig0001": 1,
                "jig0002": 2,
                "jig0003": 1,
                ...
            }
        """

        priority_hangar_jigs = {}
        hangar_jigs_lists = [i["schedule"] for i in self._production_lines] #list of lists of jigs for each hangar
        for hangar_jigs in hangar_jigs_lists:
            priority = 1
            for jig in hangar_jigs:
                priority_hangar_jigs[jig] = priority
                priority += 1
        return priority_hangar_jigs
    
    def outgoing_flight_jigs(self) -> list:
        """
        Returns a list of jig TYPES that are going to flights

        Example:
        ["typeA", "typeB", "typeA", "typeC", ...]
        """
        outgoing_flight_jigs = []
        for flight in self._flights:
            for jig in flight["outgoing"]:
                outgoing_flight_jigs.append(jig)
        return outgoing_flight_jigs
    

    #Functions to get info about the environment

    def get_jig_current_size(self, jig: str) -> int:
        """
        Returns the current size of a jig as int
        """
        jig_type = self._jigs[jig]["type"]
        empty = self._jigs[jig]["empty"]
        if empty:
            return self._jig_types[jig_type]["size_empty"]
        else:
            return self._jig_types[jig_type]["size_loaded"]
        
    def get_current_rack_space(self, rack: str) -> int:
        """
        Returns the current space of a rack as int
        """
        space = self.racks[rack]["size"]
        for jig in self.racks[rack]["jigs"]:
            space -= self.get_jig_current_size(jig)
        return space
    
    def get_current_us_hangar_side(self, rack: str) -> int:
        """
        Returns the current number of unsorted jigs in a rack from the hangar side
        """
        rack_us = 0
        rack_priorities = [self.jigs_to_hangars_priority.get(jig, 3000) for jig in self.racks[rack]["jigs"]]
        
        for i in range(1, len(rack_priorities)):
            if rack_priorities[i] < rack_priorities[i - 1]:
                rack_us += 1
        return rack_us
    
    
    def get_current_us_plane_side(self, rack: str) -> int:
        """
        Returns the current number of unsorted jigs in a rack from the plane side
        """
        rack_us = 0
        if not any(self._jigs[jig]["empty"] for jig in self.racks[rack]["jigs"]):
            return 0
        else:
            rack_priorities = [self.jigs_to_flights.index(self._jigs[jig]["type"])+1 if self._jigs[jig]["empty"] else 3000 for jig in self.racks[rack]["jigs"]]
            for i in reversed(range(1, len(rack_priorities))):
                if rack_priorities[i] < rack_priorities[i - 1]:
                    rack_us += 1
        return rack_us
    




    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(a = seed)

        #Load instance and import the data. We will use _ to denonte the data from the instance and not alter it
        self.instance = pick_random_instance_from_folder(self.instance_folder_path)
        self._jig_types = self.instance["jig_types"]
        self._trailers_beluga = self.instance["trailers_beluga"]
        self._trailers_factory = self.instance["trailers_factory"]
        self._hangars = self.instance["hangars"]
        self._production_lines = self.instance["production_lines"]
        self._flights = self.instance["flights"]
        self._jigs = self.instance["jigs"]
        self._racks = self.instance["racks"]

        #ordered list of jigs to come off flights
        self.jigs_off_flights = [jig for flight in self._flights for jig in flight["incoming"]]
        #Dict of jigs to hangars priority where some jigs share priority if going to different hangars
        self.jigs_to_hangars_priority = self.priority_jigs_for_hangars()
        #List of jig types that are going to flights in order
        self.jigs_to_flights = self.outgoing_flight_jigs()
        #Dict of racks with name as key and the rest as values
        self.racks = {i['name']: {k:v for k,v in i.items() if k != "name"} for i in self._racks}
        
        self.current_step = 0
        self.current_unsorted = sum([self.get_current_us_hangar_side(rack) + self.get_current_us_plane_side(rack) for rack in self.racks.keys()])
 
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        """
        Return: obs, reward, terminated, truncated, info
        Gymnasium 0.26+ style
        """
        self.current_step += 1
        
        # Default
        terminated = False
        truncated = False
        info = {}
        
     

        old_unsorted = sum([self.get_current_us_hangar_side(rack) + self.get_current_us_plane_side(rack) for rack in self.racks.keys()])
        if type(action) == tuple:
            action = action[0]
        # check if we have placed all jigs, its terminal
        if not self.jigs_off_flights:
            terminated = True
            return self._get_observation(), 0.0, terminated, truncated, info
        # check if we have no space for the next jig, its terminal
        elif not any([self.get_current_rack_space(rack)-self.get_jig_current_size(self.jigs_off_flights[0]) > 0 for rack in self.racks]):
            terminated = True
            return self._get_observation(), 0.0, terminated, truncated, info
        else:
            # Place the next jig
            next_jig = self.jigs_off_flights.pop(0)
            self.racks[action]['jigs'].insert(0, next_jig)
            
            # Recount unsorted
            self.current_unsorted = sum([self.get_current_us_hangar_side(rack) + self.get_current_us_plane_side(rack) for rack in self.racks.keys()])
            
            # Reward = old_unsorted - new_unsorted
            reward = (old_unsorted - self.current_unsorted) 
          
            
            
     
        # If there's a max step limit, check if we exceed it
        if self.max_steps is not None and self.current_step >= self.max_steps:
            truncated = True
            # Typically you'd still produce a "final" reward 
            # or partial completion result.
        
        obs = self._get_observation()
        return obs, float(reward), terminated, truncated, info
    


  

    def _get_observation(self):
        """
        Return shape = (20, 8) becuase 20 is max number of racks and 8 is the number of features we will pad if needed
          (1) space remaining in rack as int
          (2) size of next jig
          (3) next jig priority
          (4) top jig (hangar side) priority
          (5) lowest remaining jig priority
          (6) # of remaining jigs with priority < top
          (7) # of unsorted in this stack from the hangar side
          (8) # of unsorted in this stack from the plane side 

          Convention is 1 is the highest priority
        """
        obs = np.full((20, 8), -1, dtype=np.float32)
        
        # Next jig priority use 3000 if its not going to a hangar 
        priority_of_next_jig = self.jigs_to_hangars_priority.get(self.jigs_off_flights[0], 3000)
        #Size of next jig
        size_of_next_jig = self.get_jig_current_size(self.jigs_off_flights[0])
        #Lowest priority of remaining jigs
        lowest_priority_remaining_jigs = max([self.jigs_to_hangars_priority.get(jig, 3000) for jig in self.jigs_off_flights])
       
        
        for i in range(len(self._racks)): 
            rack = self._racks[i]["name"]
            
            # 1) Space remaining in rack
            space_remaining = self.get_current_rack_space(rack)

            # 2) Size of next jig
            #Above

            #3) Next jig priority
            #Above
            
            #4) Hangar side top jig priority
            if self.racks[rack]["jigs"]:
                hangar_side_top_jig_priority = self.jigs_to_hangars_priority.get(self.racks[rack]["jigs"][-1], 3000)
            else:
                hangar_side_top_jig_priority = 0  #TODO should this be something else?

            #5) Lowest priority of remaining jigs
            #Above
                
            # 6) # of remaining whose priority is lower (higher number) than hangar side top jig priority
            if self.racks[rack]["jigs"]:
                count_lower_priority = sum(self.jigs_to_hangars_priority.get(jig, 3000) > hangar_side_top_jig_priority for jig in self.jigs_off_flights)
            else:
                count_lower_priority = 0
            
            # 7) # of unsorted in this rack from hangar side
            if self.racks[rack]["jigs"]:
                rack_us = self.get_current_us_hangar_side(rack)
            else:
                rack_us = 0
            
            # 8) # of unsorted in this rack from the plane side
            if self.racks[rack]["jigs"]:
                racl_us_plane = self.get_current_us_plane_side(rack)
            else:
                racl_us_plane = 0

            
            obs[i, 0] = space_remaining
            obs[i, 1] = size_of_next_jig
            obs[i, 2] = priority_of_next_jig
            obs[i, 3] = hangar_side_top_jig_priority
            obs[i, 4] = lowest_priority_remaining_jigs
            obs[i, 5] = count_lower_priority
            obs[i, 6] = rack_us
            obs[i, 7] = racl_us_plane
        
        return obs
    
    def render(self):
        print("=== RENDER ===")
        for i, rack in enumerate(self.racks):
            print(f"{rack} (jigs_priorities= {[self.jigs_to_hangars_priority.get(jig, 3000) for jig in self.racks[rack]['jigs']]}), {rack} (H_unsorted= {self.get_current_us_hangar_side(rack)}),{rack} (P_unsorted= {self.get_current_us_plane_side(rack)})")
            print(f"{rack} (space_remaining= {self.get_current_rack_space(rack)})")
        print(f"Next jig priority= {self.jigs_to_hangars_priority.get(self.jigs_off_flights[0], 3000)}")
       
        print(f"Total unsorted = {self.current_unsorted}")
        print(f"Step = {self.current_step}")
        print("==============\n")
        print(f"Jigs off flights = {self.jigs_off_flights}")
        

    


   

if __name__ == "__main__":
    instance_folder_path = "/Users/ha24583/Documents/GitHub/Beluga-AI-Challenge/instances"
    env = BelugaRackingEnv(instance_folder_path)
    print(env.instance)
