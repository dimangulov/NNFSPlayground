import pickle

HEADERSIZE = 16
d = {1:"hi", 2: "there"}
msg = pickle.dumps(d)
msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8')+msg
print(msg)