d = {i: f"Curr_Belgua_jig_to_rack{i}" for i in range(20)}
c = {i+20: f"Jig_to_curr_beluga_from_rack{i}" for i in range(20)}
e = {i+40: f"Jig on rack{i} to hangar{1}" for i in range(20)}
f = {i+60: f"Jig on rack{i} to hangar{2}" for i in range(20)}
g = {i+80: f"Jig on rack{i} to hangar{3}" for i in range(20)}

d.update(c)
d.update(e)
d.update(f)
d.update(g)
print(d)