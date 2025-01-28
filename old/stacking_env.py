import json
import gymnasium as gym
from typing import Optional
import numpy as np
from itertools import pairwise
from gymnasium.spaces import Box, Discrete, Sequence
import torch
from old.action_dict import action_dict
import pufferlib
import os
import random



  

class StackingEnv(gym.Env):
    def __init__(self, json_folder):
        """_summary_

        Args:

            json_path (_type_): _description_
            Priority: if p_i < p_j then jig i is to be needed before jig j
        """
        self.json_folder = json_folder
        self.json_path = random.choice(os.listdir(json_folder))
        self.json_data = json.load(open(json_folder+"/"+self.json_path)) #TODO make this  path 

        self.cumulative_reward = 0
        self.cumulative_length = 0
        self.max_racks = 20
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
        self.jigs_from_flights = [jig for flight in self.flights for jig in flight["incoming"]]
       
        #priority list of jigs to go to hangars
        self.hangar_priority_jigs = self.get_priority_jigs_for_hangars()

        #ordered list to go to outgoing flights
        self.outgoing_flight_jigs = self.get_outgoing_flight_jigs()

        self.observation_space = Box(low=-1, high=3000, shape=(self.max_racks, self.features))


        self.prod_line_jigs = {p["name"]:[] for p in self.production_lines}
        
        
        #Actions: 
        # 0-max_racks: move next jig to rack
        # max_racks-2*max_racks: move next jig to plane
        # 2*max_racks-max_racks*len(self.production_lines): move next jig to hangar 1,2,3 (if more than one hanagr)
        self.action_space = Discrete(self.max_racks+self.max_racks+self.max_racks*len(self.production_lines))
        self.action_dict = action_dict #imported to keep track of actions        print("init check")
        print("init_checkpoint")
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
        obs[:len(self.racks),0] = np.array(rack_spaces).T
        obs[:len(self.racks),1] = s_nxt_h
        obs[:len(self.racks),2] = p_nxt_h
        obs[:len(self.racks),3] = np.array(p_h).T
        obs[:len(self.racks),4] = low_h
        obs[:len(self.racks),5] = np.array(num_low_h).T
        obs[:len(self.racks),6] = np.array(rack_us).T
        obs[:len(self.racks),7] = np.array(rack_us_to_plane).T
        
        return torch.from_numpy(obs)
    
    #Action Code:

    def beluga_to_rack(self, rack_name: str):
        rack = self.racks[int(rack_name.strip("rack"))]
        space = self.get_current_rack_space(rack)
        current_flight_index = [i["name"] for i in self.flights].index(self.current_flight)
        jig = self.flights[current_flight_index]["incoming"][0]
        if jig:
            if self.jigs[jig]["empty"]:
                jig_size = self.jig_types[self.jigs[jig]["type"]]["size_empty"]
            else:
                jig_size = self.jig_types[self.jigs[jig]["type"]]["size_loaded"]
        #If there is sapce on rack and there are jigs to be loaded from flight
        if space-jig_size >= 0 and jig:
            #remove from list of incomings IF it is there
            jig_removed_from_flight_list = self.jigs_from_flights.pop(0)
            jig_removed_from_flight= self.flights[current_flight_index]["incoming"].pop(0)
            assert jig_removed_from_flight == jig == jig_removed_from_flight_list

            #Add jig to rack at Beluga side and remove from flight
            self.racks[int(rack_name.strip("rack"))]["jigs"].insert(0,jig)
            


    def rack_to_beluga(self, rack_name: str):
        rack = self.racks[int(rack_name.strip("rack"))]
        jig = rack["jigs"][0]
        current_flight_index = [i["name"] for i in self.flights].index(self.current_flight)

        #If there is a jig of the correct type on top of the rack and it is empty
        if self.jigs[jig]["type"] == self.flights[current_flight_index]["outgoing"][0] and self.jigs[jig]["empty"]:
            #Add jig to flight at Beluga side and remove from rack
            loaded = self.flights[current_flight_index]["outgoing"].pop(0)
            removed_from_rack = rack["jigs"].pop(0)
            removed_from_outgoing_types_list = self.outgoing_flight_jigs.pop(0)
            assert jig == removed_from_rack
            assert self.jigs[jig]["type"] == removed_from_outgoing_types_list

    def rack_to_production_line(self, rack_name: str, production_line_name: str):
        rack = self.racks[int(rack_name.strip("rack"))]
        jig = rack["jigs"][0]
        production_line = self.production_lines[int(production_line_name.strip("pl"))-1]

        if jig == production_line["schedule"][0]:
            #Add jig to production line and remove from rack
            jig_removed_from_rack = rack["jigs"].pop(-1)
            jig_added_to_production_line = production_line["schedule"].pop(0)
            jig_removed_from_hangar_priority_list = self.hangar_priority_jigs.pop(jig, None)
            assert jig_removed_from_rack == jig_added_to_production_line
            self.prod_line_jigs[production_line_name].append(jig)
            
    def prod_line_to_rack(self, rack_name: str, production_line_name: str):
        rack = self.racks[int(rack_name.strip("rack"))]
        #TODO dont allow it to hangar if there is jig in the hangar
        #add this to alloqable actions method
        #must be emoty
        #must be soemthing in the prod line
        #must be rack space
               

    def action_mapping(self, action_index):
        if action_index < 20:
            return self.beluga_to_rack(f"rack0{action_index}")
        elif action_index< 40:
            return self.rack_to_beluga(f"rack0{action_index-20}")
        elif action_index < 59:
            return self.rack_to_production_line(f"rack0{action_index-40}","pl1")
        elif action_index < 79:
            return self.rack_to_production_line(f"rack0{action_index-60}","pl2")
        elif action_index < 100:
            return self.rack_to_production_line(f"rack0{action_index-80}","pl3")
        else:
            return ValueError
        
    
    def get_possible_actions(self):

        #TODO possible we can keep some of this info centrally and not have to loop

        possible_actions = [i for i in range(100)]

        current_flight_index = [i["name"] for i in self.flights].index(self.current_flight)
        num_jigs = len(self.jigs)
        num_racks = len(self.racks)

        #Remove actions for all racks that don't exist:
        for i in range(num_racks,20):
            possible_actions.remove(i)
            possible_actions.remove(i+20)
            possible_actions.remove(i+40)
            possible_actions.remove(i+60)
            possible_actions.remove(i+80)

        

       
        #If there is a plane to be completed
        if self.current_flight:
            #If there is a jig in the current flight
            if self.flights[current_flight_index]["incoming"]:
                flight_jig = self.flights[current_flight_index]["incoming"][0]
                if self.jigs[flight_jig]["empty"]:
                    jig_size = self.jig_types[self.jigs[flight_jig]["type"]]["size_empty"]
                else:
                    jig_size = self.jig_types[self.jigs[flight_jig]["type"]]["size_loaded"]
                #Now check if the current jig will fit on the rack
                for i, rack in enumerate(self.racks):
                    space = self.get_current_rack_space(rack)
                    if space-jig_size >= 0:
                        continue
                    else:
                        possible_actions.remove(i)
            #Else take all the flight to rack actions out
            else: 
                for i in range(0,num_racks): possible_actions.remove(i) 
        
        #Else take all the flight to rack actions out
        else: 
            for i in range(0,num_racks): possible_actions.remove(i) 

        
            
        #Now check jig to Beluga actions
        #These are actions 20-40
        #If there is a plane to be completed
        if self.current_flight:
            #If there is a jig to be sent to a Beluga
            if self.flights[current_flight_index]["outgoing"]:
                outgoing_type = self.flights[current_flight_index]["outgoing"][0]
                for i, rack in enumerate(self.racks):
                    if rack["jigs"]:
                        if self.jigs[rack["jigs"][0]]["type"] == outgoing_type and self.jigs[rack["jigs"][0]]["empty"]:
                            continue
                        #If the jig is not the right type and not empty remove the action
                        else:
                            possible_actions.remove(i+20)
                    else:
                        possible_actions.remove(i+20)
            else:
                for i in range(20,num_racks+20): possible_actions.remove(i)
        else:
            for i in range(20,num_racks+20): possible_actions.remove(i)

        #Now check jig to hangar actions
        #These are actions 40-99

        #see which jigs we need to send and to what hangar
        possible_hangar_jigs = {p["schedule"][0]:i for i,p in enumerate(self.production_lines) if p["schedule"]}
        #Check the jig on the hanagar side of the rack
        
        #Check each rack
        for i, rack in enumerate(self.racks):
            #list of Hangars
            hangars = [0,1,2]
            #If the jig is to go to one of the hangars and is not empty
            if rack["jigs"]:
                if rack["jigs"][-1] in possible_hangar_jigs.keys() and not self.jigs[rack["jigs"][-1]]["empty"]:
                    #Remove the hangar from the list
                    hangar_index = possible_hangar_jigs[rack["jigs"][-1]]
                    hangars.remove(hangar_index)
                    #Remove all the actions for the other hangars
                    for h in hangars:
                        possible_actions.remove(i+40+(h*20))
                else:
                    possible_actions.remove(i+40)
                    possible_actions.remove(i+40+20)
                    possible_actions.remove(i+40+40)
            else:
                #Remove all the actions for the rack
                possible_actions.remove(i+40)
                possible_actions.remove(i+40+20)
                possible_actions.remove(i+40+40)

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
        self.jig_types = self.json_data["jig_types"]
        self.racks = self.json_data["racks"]
        self.production_lines = self.json_data["production_lines"]
        self.flights = self.json_data["flights"]
        self.jigs = self.json_data["jigs"]    #UPDATE EACH TIME
        self.racks = self.json_data["racks"]

        #ordered list of flights
        self.flight_list = [i["name"] for i in self.flights]
        self.current_flight = self.flight_list[0]

        #ordered list of jigs to come off flights
        self.jigs_from_flights = [jig for flight in self.flights for jig in flight["incoming"]]
       
        #priority list of jigs to go to hangars
        self.hangar_priority_jigs = self.get_priority_jigs_for_hangars()

        #ordered list to go to outgoing flights
        self.outgoing_flight_jigs = self.get_outgoing_flight_jigs()
        self.cumulative_reward = 0
        self.cumulative_length = 0
        observation = self._get_obs()
        info = self._get_info()
        
        
        observation = observation.to(torch.float32)
        return observation, info
    
    
     

        
    def get_complete_hangars(self):
        complete = 0
        hangars = self.production_lines
        for hangar in hangars:
            if hangar["schedule"]:
                continue
            else:
                complete += 1
        return complete


    
    def get_plane_complete(self):
        current_flight_index = [i["name"] for i in self.flights].index(self.current_flight)
        return self.flights[current_flight_index]["outgoing"] == [] and self.flights[current_flight_index]["incoming"] == []
    
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
        #If there are no outgoing jigs and no schedule for the production lines we are done
        if not self.flight_list and not any(l["schedule"] for line in self.production_lines for l in line):
            return True
        else:
            return False

    def _get_truncated(self):
        ...

    def step(self, action):

        if isinstance(action, np.ndarray):
            action = action[0]

        #reward logic
        curr_sum_us = self.get_current_sum_us()
        curr_complete_hangars = self.get_complete_hangars()
        curr_plane_complete = self.get_plane_complete()




        poss_actions = self.get_possible_actions()
        if action in  poss_actions:
            #Take the action
            self.action_mapping(action)
        

        #Check if need to change flight
        observation = self._get_obs()
        
        


        #Reward is the difference in US before and after the action
        #Also reward for finishing a flight completely
        #Also reward for finishing a hangar
        #and a BIG reward for winning
        reward = curr_sum_us - self.get_current_sum_us()
        reward += self.get_complete_hangars() - curr_complete_hangars
        if self.get_plane_complete():
            reward += 5
            if self.flight_list:
                self.flight_list.pop(0)
                self.current_flight = self.flight_list[0]
            else: self.current_flight = None
           
        truncated = False
        terminated = self._get_terminated()
        if terminated:
            reward += 1000
        self.cumulative_reward += reward
        self.cumulative_length += 1
        info = self._get_info()
        
       
            
        observation = observation.to(torch.float32)
        return observation, reward, terminated, truncated, info
    






if __name__ == "__main__":
    #env = StackingEnv("instances/problem_44_s45_j10_r2_oc83_f5.json")
   #env = StackingEnv("instances/problem_139_s140_j49_r7_oc20_f35.json")
    env = StackingEnv("/Users/ha24583/Documents/GitHub/Beluga-AI-Challenge/toys")
    print(env.get_possible_actions())
    print([env.action_dict[i] for i in env.get_possible_actions()])
    
    obs,info = env.reset()
    print(obs)
    
    #TODO YOU DIV YOU ARENT PUTTINOG RACKS BACK AFTER PUTTING TO HANGAR


