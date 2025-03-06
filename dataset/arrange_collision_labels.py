import os 


bp = os.path.dirname(__file__)
names = os.listdir(os.path.join(bp, "collision_label"))

for name in names: 
    original = os.path.join(bp, "collision_label", name, "collision_labels.npz")
    new = os.path.join(bp, "scenes", name, "collision_labels.npz")
    os.rename(new, original)

