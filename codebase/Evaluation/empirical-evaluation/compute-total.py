import pandas as pd
name='InfoBio'
# name='Biocoder'
name='Ecology'
name='BioEng'
name='Genetics'
name='Medical'
print(name)
df = pd.read_csv('outputs2/result-{}.csv'.format(name), sep='\t')
scores = list(map(lambda x:eval(x), list(df['rate-total'])))

print(len(scores))
opns = [a for a,b in scores]
confs = [b for a,b in scores]
if 0 in opns+confs:
    print('exist zero')
print((sum(opns)/len(opns), sum(confs)/len(confs)))