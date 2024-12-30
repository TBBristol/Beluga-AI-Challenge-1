import os
import subprocess

seed = 1
occ_rate = 0.2
num_flights = 3

for i in range(1000):
    subprocess.run(["python", "toolkit/generate_instance.py", f"-s {seed}", f"-or", f"{occ_rate}",
                    f"-f", f"{num_flights}",  "-o",  "/Users/ha24583/Documents/GitHub/Beluga-AI-Challenge/toys" ])


#TODO make a function to generae lots of singles easily and make them easy ones

