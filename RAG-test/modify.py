import json

f = open("urlhaus-04-Nov-2024-alpaca-dataset.json", "r")
data = json.load(f)
print(data[0])

f2 = open("modified.json", "w+")
f2.write("[\n")
for i in data:
    input = i["input"]
    md5_hash = input.split(",")[0].split(" ")[2]
    i["instruction"] += f" with the md5 hash value of: {md5_hash}"
    if i != data[:-1]:
        f2.write(json.dumps(i) + ",\n")
        continue
    f2.write(json.dumps(i) + "\n")
f2.write("]\n")
