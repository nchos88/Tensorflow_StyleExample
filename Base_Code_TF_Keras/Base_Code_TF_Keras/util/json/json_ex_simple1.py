import json 

customer ={ 'id' : 1234,
            'name' : 'sjw',
            'history' : [  {'date': '2015-03-11', 'item': 'iPhone'},
                {'date': '2016-02-23', 'item': 'Monitor'},]}


jsonstring = json.dumps(customer, indent = 4)

print(jsonstring)

with open(r'test\json\simpl1.json' , 'w') as f:
    json.dump(customer , f, indent = 4)


## Load
with open(r'test\json\simpl1_1.json' , 'r') as f:
    res1 = json.load(f)
    print(res1)

print("_"*8)

jsonDic = None
with open(r'test\json\simpl1.json' , 'r') as f:
    res1 = json.load(f)
    print(res1)
    jsonDic = res1

print(type(jsonDic))
for k,v in jsonDic.items():
    line = "key = {} , val = {}".format(k,v)
    print(line)


