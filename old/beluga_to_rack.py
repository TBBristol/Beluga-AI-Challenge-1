import json
import gymnasium as gym
from typing import Optional
import numpy as np
from itertools import pairwise

import torch
from old.action_dict import action_dict
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
        self.trailers = self.json_data["trailers_beluga"]

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

        #Set up trailers to recieve Jigs
        for t in self.trailers:
            t["jig"] = []


       

        #ACTION/OBS SPACE

        #OBS SPACE = 8 features for each rack up to 20 racks, up to 3 trailers empty/full
        #SO BOX space (20,8) plus first row is trailers binary empty/full with -1 padding so (21,8)

        #TODO trying to make this a pufferbox space
        self.observation_space = spaces.Box(low=-1, high=3000, shape=(self.max_racks+1, self.features))
        
        #Actions: 
        # 3 for plane to trailer, 3x 20 for trailer to rack = 63
        self.action_space = spaces.Discrete(63)
        self.action_dict = {0: 'Beluga_to_t1', 1: 'Beluga_to_t2', 2: 'Beluga_to_t3', 3: 'Trailer_1_to_rack01', 4: 'Trailer_1_to_rack02', 5: 'Trailer_1_to_rack03', 6: 'Trailer_1_to_rack04', 7: 'Trailer_1_to_rack05', 8: 'Trailer_1_to_rack06', 9: 'Trailer_1_to_rack07', 10: 'Trailer_1_to_rack08', 11: 'Trailer_1_to_rack09', 12: 'Trailer_1_to_rack10', 13: 'Trailer_1_to_rack11', 14: 'Trailer_1_to_rack12', 15: 'Trailer_1_to_rack13', 16: 'Trailer_1_to_rack14', 17: 'Trailer_1_to_rack15', 18: 'Trailer_1_to_rack16', 19: 'Trailer_1_to_rack17', 20: 'Trailer_1_to_rack18', 21: 'Trailer_1_to_rack19', 22: 'Trailer_1_to_rack20', 23: 'Trailer_2_to_rack01', 24: 'Trailer_2_to_rack02', 25: 'Trailer_2_to_rack03', 26: 'Trailer_2_to_rack04', 27: 'Trailer_2_to_rack05', 28: 'Trailer_2_to_rack06', 29: 'Trailer_2_to_rack07', 30: 'Trailer_2_to_rack08', 31: 'Trailer_2_to_rack09', 32: 'Trailer_2_to_rack10', 33: 'Trailer_2_to_rack11', 34: 'Trailer_2_to_rack12', 35: 'Trailer_2_to_rack13', 36: 'Trailer_2_to_rack14', 37: 'Trailer_2_to_rack15', 38: 'Trailer_2_to_rack16', 39: 'Trailer_2_to_rack17', 40: 'Trailer_2_to_rack18', 41: 'Trailer_2_to_rack19', 42: "Trailer_2_to_rack20", 43: 'Trailer_3_to_rack01', 44: 'Trailer_3_to_rack02', 45: 'Trailer_3_to_rack03', 46: 'Trailer_3_to_rack04', 47: 'Trailer_3_to_rack05', 48: 'Trailer_3_to_rack06', 49: 'Trailer_3_to_rack07', 50: 'Trailer_3_to_rack08', 51: 'Trailer_3_to_rack09', 52: 'Trailer_3_to_rack10', 53: 'Trailer_3_to_rack11', 54: 'Trailer_3_to_rack12', 55: 'Trailer_3_to_rack13', 56: 'Trailer_3_to_rack14', 57: 'Trailer_3_to_rack15', 58: 'Trailer_3_to_rack16', 59: 'Trailer_3_to_rack17', 60: 'Trailer_3_to_rack18', 61: 'Trailer_3_to_rack19', 62: 'Trailer_3_to_rack20'}
        
        
        
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
    
    def get_trailer_fill(self):
        trailers = np.full((self.features),-1)
        for i,t in enumerate(self.trailers):
            if t['jig']:
                trailers[i] = 1
            else:
                trailers[i] = 0
        return trailers


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

        #Trailers:
        trailers = self.get_trailer_fill()
        
        




        obs = np.full((self.max_racks+1,self.features),-1)
        obs[0,:] = trailers
        obs[1:len(self.racks)+1,0] = np.array(rack_spaces).T
        obs[1:len(self.racks)+1,1] = s_nxt_h
        obs[1:len(self.racks)+1,2] = p_nxt_h
        obs[1:len(self.racks)+1,3] = np.array(p_h).T
        obs[1:len(self.racks)+1,4] = low_h
        obs[1:len(self.racks)+1,5] = np.array(num_low_h).T
        obs[1:len(self.racks)+1,6] = np.array(rack_us).T
        obs[1:len(self.racks)+1,7] = np.array(rack_us_to_plane).T
        
        return torch.from_numpy(obs)
    
    #Action Code:

    def trailer_to_rack(self, trailer_name: str, rack_name:str):
        """Takes the jig that is on trailer_name and moves to to Rack_name on the 
        Beluga side. Removes it from the trailer list and adds it to the rack list

        Args:
            trailer_name (str): _description_
            rack_name (str): _description_
        """
        rack = self.racks[int(rack_name.strip("rack"))]
        space = self.get_current_rack_space(rack)
        jig = self.trailers[int(trailer_name.strip("beluga_trailer"))-1]["jig"][0]
        if jig:
            if self.jigs[jig]["empty"]:
                jig_size = self.jig_types[self.jigs[jig]["type"]]["size_empty"]
            else:
                jig_size = self.jig_types[self.jigs[jig]["type"]]["size_loaded"]
        else:
            ValueError
        #If there is sapce on rack and there are jigs to be loaded from flight
        if space-jig_size >= 0 and jig:
            #Add jig to rack at Beluga side and remove from trailer
            self.racks[int(rack_name.strip("rack"))]["jigs"].insert(0,jig)
            self.trailers[int(trailer_name.strip("beluga_trailer"))-1]["jig"] = []
            
    def beluga_to_trailer(self, trailer_name: str):
        """Takes the next jig on the jig list from the plane and puts it on trailer.
        Removes the jig from the list self.jigs_from_flights and adds it to the trailer jigs


        Args:
            trailer (str): the name of the trailer as a str eg beluga_trailer_1
        """
        #check trailer is empty
        if self.trailers[int(trailer_name.strip("beluga_trailer"))-1]["jig"]:
            ValueError
        jig = self.jigs_from_flights.pop(0)
        self.trailers[int(trailer_name.strip("beluga_trailer"))-1]["jig"].append(jig)


   

    def action_mapping(self, action_index):
        #These are the beluga to trailers
        if action_index <= 2:
           trailer_name = self.trailers[action_index]['name']
           self.beluga_to_trailer(trailer_name)

        #And these the trailer to racks 
         #Trailer to rack actions start at 3. Then T2: 23, T3: 43 ends at 62
        else:
           new_index = action_index - 2  #so 3 is first of these and becomes 1
           trailer = math.ceil(new_index/20) #e.g 22 becomes new index 20 which is then trailer 1 but 23 is 21 which is math.ceil = 2
           rack = new_index - (trailer-1) *20 -1 #e.g. new_index 21 is 0 (rack 0)
           rack_name = f"rack{rack}" if rack >=10 else f"rack0{rack}"
           trailer_name = self.trailers[trailer-1]['name']
           self.trailer_to_rack(trailer_name, rack_name)
#TODO this is ballsed after the else  ^^

    
    def get_possible_actions(self):

        #TODO could treat ats a set? Is it more effciient do you need to check if present
        possible_actions = [i for i in range(63)]
        num_racks = len(self.racks)
        num_trailers = len(self.trailers)

        #if there are no jigs on flights cant move them to the trailers
        if not self.jigs_from_flights:
            for a in range(0,3):
                if a in possible_actions:
                    possible_actions.remove(a)

        #Remove all trailer to rack for racks that don't exist. 
        #Trailer to rack actions start at 3. Then T2: 23, T3: 43 ends at 62
        for i in range(num_racks,20):
            possible_actions.remove(i+3) #So if theres one rack start at 4
            possible_actions.remove(i+23)
            possible_actions.remove(i+43)
        
        #Now remove all trailer actions that don't exist thats num_trailers+1 to 3
        #Then each block of 20

        for t in range(num_trailers, 3): #So if one trailer 1 -> 2 inclusive and actions start at 0
            if t in possible_actions:
                possible_actions.remove(t)
            
            #Now remove blocks of 20 if there
            for a in range(20): #0 -> 19
                if a+3 + 20*t in possible_actions: 
                    possible_actions.remove(a+3 + 20*t)

        


        #if trailer is full remove plane to trailer if it isnt remove trailer to rack
        for i,t in enumerate(self.trailers):

            #If there is a jig in the trailer
            if t['jig']:
                if i in possible_actions:
                #Remove the action for plane to that trailer
                    possible_actions.remove(i)

                #Now remove if no space on the rack for the jig on each trailer
                rack_spaces: list[int] = [self.get_current_rack_space(rack) for rack in self.racks]
                jig_size = self.get_jig_current_size(t['jig'][0])
                for idx, r in enumerate(rack_spaces):
                    if r - jig_size < 0: 
                        if 3 + idx + i*20 in possible_actions: #so rack2 trailer 2 is 3 + 1 + 20 = 24
                            possible_actions.remove(3 + idx + i*20 )


                    
            #If there is no jig in trailer take remove all trailer to rack for that trailer
            else:
                for a in range(3+i*20, 3+i*20+20): #trailer 1 is 3+0*20 = 3 -> 3+0*20+20 = 23(22)
                    if a in possible_actions:
                        possible_actions.remove(a)


           

                    
        return possible_actions
        
 


    

    
    def get_action_mask(self, possible_actions):
        
        mask = torch.zeros(self.action_space.n)
        mask[possible_actions] = 1
        return mask

        

        
        

        

            


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

    
        self.json_path = random.choice(os.listdir(self.json_folder))
        self.json_data = json.load(open(self.json_folder+"/"+self.json_path)) #TODO make this  path 

        #DIRECT JSON DATA
        self.max_racks = 20
        self.jig_types = self.json_data["jig_types"]
        self.racks = self.json_data["racks"]
        self.flights = self.json_data["flights"]
        self.jigs = self.json_data["jigs"]    #UPDATE EACH TIME
        self.racks = self.json_data["racks"]
        self.trailers = self.json_data["trailers_beluga"]

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

        #Set up trailers to recieve Jigs
        for t in self.trailers:
            t["jig"] = []


        observation = self._get_obs()
        info = self._get_info
        observation = observation.to(torch.float32)
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
        #Assume next jig is on a trailer which means we cant terminate if flight has a jig and trailer doens't
        rack_spaces: list[int] = [self.get_current_rack_space(rack) for rack in self.racks]
        for t in self.trailers:
            if t['jig']:
                size = self.get_jig_current_size(t['jig'][0])
                for space in rack_spaces:
                    if space - size < 0:
                        return True
        return False


    def _get_truncated(self):
        #TODO what about when we get stuck
        return False
    

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
        if not self.jigs_from_flights and not any(t["jig"] for t in self.trailers):
            reward += 1000
            terminated = True
           
        truncated = self._get_truncated()
        

       
        self.cumulative_reward += reward
        self.cumulative_length += 1
        info = self._get_info()
        observation = self._get_obs()
        
       
            
        observation = observation.to(torch.float32)
        return observation, reward, terminated, truncated, info
    






if __name__ == "__main__":

    env = Beluga_rack_stack("/Users/ha24583/Documents/GitHub/Beluga-AI-Challenge/toys")
    print(env.get_possible_actions())
    print([env.action_dict[i] for i in env.get_possible_actions()])
    env.step(1)
    
    obs,info = env.reset()
    print(obs)
    
    #TODO YOU DIV YOU ARENT PUTTINOG RACKS BACK AFTER PUTTING TO HANGAR


