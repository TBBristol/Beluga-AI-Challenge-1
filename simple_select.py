import json
import gymnasium as gym
from typing import Optional
import numpy as np
from itertools import pairwise

import torch
from action_dict import action_dict
import pufferlib
import os
import random
import math
from gymnasium import spaces
from pufferlib.pufferlib.spaces import Box, Discrete



  

class Beluga_rack_stack(gym.Env):
    def __init__(self, json_folder):
        """_summary_

        Args:

            json_path (_type_): _description_
            Priority: if p_i < p_j then jig i is to be needed before jig j
        """
        self.json_folder = json_folder
        self.json_path = random.choice(os.listdir(json_folder))
        self.json_data = json.load(open(json_folder+"/"+self.json_path)) #TODO make this  path 

        #DIRECT JSON DATA
        self.max_racks = 20
        self.jig_types = self.json_data["jig_types"]
        self.racks = self.json_data["racks"]
        self.production_lines = self.json_data["production_lines"]
        self.flights = self.json_data["flights"]
        self.jigs = self.json_data["jigs"]    
        self.racks = self.json_data["racks"]
        

        #INITIAL RL VARS
        self.cumulative_reward = 0
        self.cumulative_length = 0
        self.features = 8 #TODO KEEP ME UP TO DATE

        #ordered list of jigs to come off flights
        self.jigs_from_flights = [jig for flight in self.flights for jig in flight["incoming"]]
       
        #priority list of jigs to go to hangars
        self.hangar_priority_jigs = self.get_priority_jigs_for_hangars()

        #ordered list to go to outgoing flights
        self.outgoing_flight_jigs = self.get_outgoing_flight_jigs()


        #ACTION/OBS SPACE

        #OBS SPACE = 8 features for each rack up to 20 racks

        #TODO trying to make this a pufferbox space
        self.observation_space = spaces.Box(low=-1, high=3000, shape=(self.max_racks, self.features))
        
        #Actions: 
        # 1 for each rack
        self.action_space = spaces.Discrete(20)


        
        
        
        print("init check")
       
    #state functions

    def get_type_priorities_for_planes(self):
        type_priorities_for_planes = {}
        for i,t in enumerate(self.outgoing_flight_jigs):
            if t not in type_priorities_for_planes:
                type_priorities_for_planes[t] = i
        return type_priorities_for_planes

    def get_priority_jigs_for_hangars(self):
        priority_hangar_jigs = {}
        hangar_jigs = [i["schedule"] for i in self.production_lines]
        for hangar in hangar_jigs:
            priority = 1
            for jig in hangar:
                priority_hangar_jigs[jig] = priority
                priority += 1
        return priority_hangar_jigs
    
    def get_outgoing_flight_jigs(self):
        outgoing_flight_jigs = []
        for flight in self.flights:
            for jig in flight["outgoing"]:
                outgoing_flight_jigs.append(jig)
        return outgoing_flight_jigs


    def get_jig_current_size(self, jig: str):
        jig_type = self.jigs[jig]["type"]
        empty = self.jigs[jig]["empty"]
        if empty:
            return self.jig_types[jig_type]["size_empty"]
        else:
            return self.jig_types[jig_type]["size_loaded"]

    def get_current_rack_space(self, rack:dict):
        space = rack["size"]
        for jig in rack["jigs"]:
            space -= self.get_jig_current_size(jig)

        return space    
    
    def num_us_to_plane(self, type_priorities_for_planes: dict) -> list:
        rack_us_to_plane = []
        jigs_in_racks = [i["jigs"] for i in self.racks]
        all_racks_priorities = []
        for rack in jigs_in_racks:
            rack_priorities = []
            for jig in rack:
                if self.jigs[jig]["empty"]:
                    priority = type_priorities_for_planes.get(jig, 3000)
                else:
                    priority = 3000
                rack_priorities.append(priority)
            all_racks_priorities.append(rack_priorities)
        for i in all_racks_priorities:
            rack_us_to_plane.append(sum([1 for j in pairwise(i) if j[0] > j[1]]))
        return rack_us_to_plane
        
    def num_us_in_rack_hangar_side(self, rack_priorities: list) -> list:
            rack_us = []
            for i in rack_priorities:
                #j[1] > j[0] because we want to unload hangar side and lower number is higher priority
                rack_us.append(sum([1 for j in pairwise(i) if j[1] > j[0]]))
            return rack_us
    


    def _get_obs(self):

        """REMEBER: lower number priority is needed first
        Current features per rack:
        rack_spaces:
        s_nxt_h: size of next to be stacked
        p_nxt_h: priority of next to be stacked
        p_h: priority of jig nearest to hangars in rack (ie on top)
        low_h: lowest priority still to unload
        num_low_h: number of jigs still to unload with priority lower than stack priority p_h
        rack_us: number of jigs that are out of order from hangar side
        rack_us_to_plane: number of jigs that are out of order from plane side

        len(8)
        max_racks is self.max_racks
        => observation space is (max_racks,8) 
        """


        rack_spaces: list[int] = [self.get_current_rack_space(rack) for rack in self.racks]
        

        #From aircraft to racks -----------
        if self.jigs_from_flights:
            next_to_be_stacked: str = self.jigs_from_flights[0]
            #size of next to be stacked
            s_nxt_h = self.get_jig_current_size(next_to_be_stacked)
        else:
            s_nxt_h = 0
            next_to_be_stacked = None
        #If the jig is not to be sent to hangars make the priority value very high (ie never be retrived)
        p_nxt_h = self.hangar_priority_jigs.get(next_to_be_stacked, 3000)
        #Priority of jig nearest to hangars in rack
        p_h = [self.hangar_priority_jigs.get(i["jigs"][-1] if i["jigs"] else -1, 3000) for i in self.racks]
        #lowest priority still to unload
        if self.jigs_from_flights:
            low_h = max([self.hangar_priority_jigs.get(jig, 3000) for jig in self.jigs_from_flights])
            priorities_to_unload = [self.hangar_priority_jigs.get(jig, 3000) for jig in self.jigs_from_flights]
             #number of jigs still to unload with priority lower than stack priority p_h
            num_low_h = [sum([1 for i in priorities_to_unload if i > j] ) for j in p_h]
        else:
            low_h = 3000
            num_low_h = 0
       
        
      


        #num of jigs stacked on top of lower priority jigs in each rack
        priorities_hangar_each_rack = [[self.hangar_priority_jigs.get(j, 3000) for j in i["jigs"] ] for i in self.racks]
        #List of list each list is [p1,p2,p3,p4] where p1 is priority of rig beluga side 
        rack_us = self.num_us_in_rack_hangar_side(priorities_hangar_each_rack)

        #From racks to aircraft -----------
        

        #We are worried about what is going to be sent to the plane and its accesibility
        #The types to plane occur more than once so we pick the soonest priority to represent the type
        type_priorities_for_planes = self.get_type_priorities_for_planes()
        #It's hard to know what will be stacked from hangars as there is more than once choice so we just
        #look at current rack state and then when an action is taken it changes and this gives information
        rack_us_to_plane = self.num_us_to_plane(type_priorities_for_planes)

        #TODO think about more featues for plane side


        obs = np.full((self.max_racks,self.features),-1)
        obs[0:len(self.racks),0] = np.array(rack_spaces).T
        obs[0:len(self.racks),1] = s_nxt_h
        obs[0:len(self.racks),2] = p_nxt_h
        obs[0:len(self.racks),3] = np.array(p_h).T
        obs[0:len(self.racks),4] = low_h
        obs[0:len(self.racks),5] = np.array(num_low_h).T
        obs[0:len(self.racks),6] = np.array(rack_us).T
        obs[0:len(self.racks),7] = np.array(rack_us_to_plane).T
        
        return torch.from_numpy(obs)
    
    #Action Code:

    def action_mapping(self, action:int):
        #rack_str = f"rack{action + 1:02d}"
        current_jig = self.jigs_from_flights.pop(0)
        self.racks[action]["jigs"].insert(0,current_jig)

    
    def get_possible_actions(self):

        possible_actions = set()
        current_jig_size = self.get_jig_current_size(self.jigs_from_flights[0])
        for i,rack in enumerate(self.racks):
            if self.get_current_rack_space(rack) - current_jig_size >= 0:
                possible_actions.add(i)
                    
        return list(possible_actions)
        

    def get_action_mask(self, possible_actions):
        
        mask = torch.zeros(self.action_space.n)
        mask[possible_actions] = 1
        return mask

        

        
        

        

            


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if seed:
            random.seed(seed)
        self.json_path = random.choice(os.listdir(self.json_folder))
        self.json_data = json.load(open(self.json_folder+"/"+self.json_path)) #TODO make this  path 

        #DIRECT JSON DATA
        self.max_racks = 20
        self.jig_types = self.json_data["jig_types"]
        self.racks = self.json_data["racks"]
        self.flights = self.json_data["flights"]
        self.jigs = self.json_data["jigs"]    #UPDATE EACH TIME
        self.racks = self.json_data["racks"]

        #INITIAL RL VARS
        self.cumulative_reward = 0
        self.cumulative_length = 0
        self.features = 8 #TODO KEEP ME UP TO DATE

        #ordered list of jigs to come off flights
        self.jigs_from_flights = [jig for flight in self.flights for jig in flight["incoming"]]
       
        #priority list of jigs to go to hangars
        self.hangar_priority_jigs = self.get_priority_jigs_for_hangars()

        #ordered list to go to outgoing flights
        self.outgoing_flight_jigs = self.get_outgoing_flight_jigs()

        observation = self._get_obs()
        info = self._get_info
        observation = observation.to(torch.float32)

        #For this problem variant sometimes we start with no actions possible so check this
        if not self.get_possible_actions():
            observation, info = self.reset()


        return observation, info
    
    
    def get_current_sum_us(self):
        
        #num of jigs stacked on top of lower priority jigs in each rack
        priorities_hangar_each_rack = [[self.hangar_priority_jigs.get(j, 3000) for j in i["jigs"] ] for i in self.racks]
        #List of list each list is [p1,p2,p3,p4] where p1 is priority of rig beluga side 
        rack_us = self.num_us_in_rack_hangar_side(priorities_hangar_each_rack)
        type_priorities_for_planes = self.get_type_priorities_for_planes()
        #It's hard to know what will be stacked from hangars as there is more than once choice so we just
        #look at current rack state and then when an action is taken it changes and this gives information
        rack_us_to_plane = self.num_us_to_plane(type_priorities_for_planes)
        return sum(rack_us) + sum(rack_us_to_plane)
    
    def _get_info(self):
        return [{"episode" : {"r": self.cumulative_reward, "l": self.cumulative_length}}]

 
    
    def _get_terminated(self):
        #If all racks are full or cannot take the next jig from the flight we terminate with the reward we have so far
        #IF there are no actions to take next step then we terminate
        if not self.jigs_from_flights:
            return True
        rack_spaces: list[int] = [self.get_current_rack_space(rack) for rack in self.racks]
        current_jig = self.jigs_from_flights[0]
        current_jig_size = self.get_jig_current_size(current_jig)
        free_spaces = [space for space in rack_spaces if space - current_jig_size >= 0]
        if not free_spaces:
            return True
        if not self.get_possible_actions():
            return True
        return False
        


    def _get_truncated(self):
        #TODO what about when we get stuck
        return False
    
    def get_action_mask(self, possible_actions):
        
        mask = torch.zeros(self.action_space.n)
        mask[possible_actions] = 1
        return mask
    

    def step(self, action):

        if isinstance(action, np.ndarray):
            action = action[0]

        #reward logic
        curr_sum_us = self.get_current_sum_us()
    

        poss_actions = self.get_possible_actions()
        

        if action in  poss_actions:
            #Take the action
            self.action_mapping(action)

        #TODO do we need a penalty for incorect actions to prompt good actions>
        
        #before reward as there is a temr condition in there
        terminated = self._get_terminated()


        #Reward is the difference in US before and after the action
        #As there is no NOOP then it will slowly fill racks as these will be the avilable actions
        reward = curr_sum_us - self.get_current_sum_us()

        #Extra reward for completing and set terminated
        if not self.jigs_from_flights:
            reward += 1000
            terminated = True
           
        truncated = self._get_truncated()
        

       
        self.cumulative_reward += reward
        self.cumulative_length += 1
        info = self._get_info()
        observation = self._get_obs()
        
       
      


        if terminated:
            observation, _ = self.reset() #TODO check if this is correct do we sumbit the real observation or the one we have
        observation = observation.to(torch.float32)
        return observation, reward, terminated, truncated, info
    






if __name__ == "__main__":

    env = Beluga_rack_stack("/Users/ha24583/Documents/GitHub/Beluga-AI-Challenge/toys")
    print(env.get_possible_actions())
    env.step(1)
    
    obs,info = env.reset()
    print(obs)
    
    #TODO YOU DIV YOU ARENT PUTTINOG RACKS BACK AFTER PUTTING TO HANGAR


