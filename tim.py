import subprocess
import random
from random import randint
import types
import numpy as np


or_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
t_list = [0, 1, 2]
unsolvable_list = ["rack_space_general", "outgoing_flight_not_sat", "schedule_clashes", None]
pp = [True, False]
pm = ["arrivals", "ppddl"]
v = [True, False]
pw = [1, 2, 3, 4, 5]



gen_instance_args = {
    "-s" : 1, #seed
    "-or": random.sample(or_list, 1)[0], #inital occupancy rate of racks fraction
    "-t": random.sample(t_list, 1)[0], #distribution of jigs types: (0) uniform, (1) small jigs preferred, (2) large jigs preferred (default: 1)
    "-f": randint(3, 200), #number of incoming and outgoing Beluga flights
    "-us": random.sample(unsolvable_list, 1)[0], #scenario based on which the generator tries to generate an unsolvable instance
    "-v": random.sample(v, 1)[0], #print debug output
    "-o": "output", #output folder for the problem, if no folder is given, the problem is printed onto stdout
    "-on": "test_instance", #name for the problem, if no name is defined, a name based on the number of jigs, jig types, racks, the exact occupancy rate, the number of flights, the seed and potentially the unsolvability scenarios is generated
    "-pp": random.sample(pp,1)[0], #Enables the probabilistic model, triggering the generation of probabilistic instances (default: False)
    "-pm": "arrivals", #Controls the type of probabilistic model, if the -pp option is enabled. If the arrivals options is used (the default), then the instance will contain only arrival times and will be suitable for a more realistic uncertainty semantic where flights are subject to stochastic delay; if the ppddl option is used, the instance will include information on uncertainty, specified in terms of probabilities of transition between abstract states. Every abstract state is identified by the sequence of a configurable number of the last flights (default: arrivals)
    "-pw": random.sample(pw, 1)[0], #Length of the sequence of flights used as the abstract state in the 'ppddl' probabilistic model. This parameter is ignored unless 'ppddl' probabilistic mode is enabled (default: 1)


}

#print(gen_instance_args.values())


def gen_args():
    args = []
    dict_args = gen_instance_args.copy()
    if dict_args["-pp"]:
        dict_args["-pm"] = random.sample(pm, 1)[0]
    if not dict_args["-us"]:
        del dict_args["-us"]
    if dict_args["-pp"]:
        del dict_args["-pp"]
        args.append("-pp")
    else:
        del dict_args["-pp"]
    if not dict_args['-v']:
        del dict_args['-v']
    else:
        del dict_args['-v']
        args.append('-v')
    for k,v in dict_args.items():
        args.append(k)
        args.append(str(v))
    print(args)
    return args

#print(gen_args())

subprocess.run(["python", "tools/generate_instance.py", *gen_args()])

#['-pp', '-s', '1', '-or', '0.5', '-t', '1', '-f', '618', 
# '-us', 'outgoing_flight_not_sat', '-o', 'output', '-on', 'test_instance', '-pm', 'ppddl', '-pw', '5']

