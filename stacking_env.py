import json
import gymnasium as gym
from typing import Optional
import numpy as np
from itertools import pairwise
from gymnasium.spaces import Box, Discrete
import torch

class StackingEnv(gym.Env):
    def __init__(self, json_path):
        """_summary_

        Args:
            json_path (_type_): _description_
            Priority: if p_i < p_j then jig i is to be needed before jig j
        """
        self.max_racks = 20
        self.json_path = json_path
        self.json_data = json.load(open(json_path))
        self.jig_types = self.json_data["jig_types"]
        self.racks = self.json_data["racks"]
        self.production_lines = self.json_data["production_lines"]
        self.flights = self.json_data["flights"]
        self.jigs = self.json_data["jigs"]    #UPDATE EACH TIME
        self.racks = self.json_data["racks"]


        self.features = 8 #TODO KEEP ME UP TO DATE

        #ordered list of flights
        self.flight_list = [i["name"] for i in self.flights]
        self.current_flight = self.flight_list[0]

        #ordered list of jigs to come off flights
        self.jig_list = [jig for flight in self.flights for jig in flight["incoming"]]
       
        #priority list of jigs to go to hangars
        self.hangar_priority_jigs = self.get_priority_jigs_for_hangars()

        #ordered list to go to outgoing flights
        self.outgoing_flight_jigs = self.get_outgoing_flight_jigs()

        self.observation_space = Box(low=-1, high=3000, shape=(self.max_racks,self.features))
        
        
        #Actions: 
        # 0-max_racks: move next jig to rack
        # max_racks-2*max_racks: move next jig to plane
        # 2*max_racks-max_racks*len(self.production_lines): move next jig to hangar 1,2,3 (if more than one hanagr)
        self.action_space = Discrete(self.max_racks+self.max_racks+self.max_racks*len(self.production_lines))
        
        
        print("init check")

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

    def get_current_rack_space(self, rack: str):
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    
            
        


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
        => observation space is (max_racks,8) flattened
        """


        rack_spaces: list[int] = [self.get_current_rack_space(rack) for rack in self.racks]
        

        #From aircraft to racks -----------
        next_to_be_stacked: str = self.jig_list[0]
        #size of next to be stacked
        s_nxt_h = self.get_jig_current_size(next_to_be_stacked)
        #If the jig is not to be sent to hangars make the priority value very high (ie never be retrived)
        p_nxt_h = self.hangar_priority_jigs.get(next_to_be_stacked, 3000)
        #Priority of jig nearest to hangars in rack
        p_h = [self.hangar_priority_jigs.get(i["jigs"][-1], 3000) for i in self.racks]
        #lowest priority still to unload
        low_h = max([self.hangar_priority_jigs.get(jig, 3000) for jig in self.jig_list])
        #number of jigs still to unload with priority lower than stack priority p_h
        priorities_to_unload = [self.hangar_priority_jigs.get(jig, 3000) for jig in self.jig_list]
        num_low_h = [sum([1 for i in priorities_to_unload if i > j] ) for j in p_h]


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
        obs[:len(self.racks),0] = np.array(rack_spaces).T
        obs[:len(self.racks),1] = s_nxt_h
        obs[:len(self.racks),2] = p_nxt_h
        obs[:len(self.racks),3] = np.array(p_h).T
        obs[:len(self.racks),4] = low_h
        obs[:len(self.racks),5] = np.array(num_low_h).T
        obs[:len(self.racks),6] = np.array(rack_us).T
        obs[:len(self.racks),7] = np.array(rack_us_to_plane).T
        
        return torch.from_numpy(obs.flatten())

    def _get_info(self):
        ...

    def step(self, action):
        ...

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    






if __name__ == "__main__":
    env = StackingEnv("instances/problem_44_s45_j10_r2_oc83_f5.json")
    obs,info = env.reset()
    print(env.json_data)

    #TODO build out step and actions look at what original lists need modifying at each statge
    