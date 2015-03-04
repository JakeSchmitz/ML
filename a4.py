
import arff

train_data = dict()
test_data = dict()

for row in arff.load('hw4train.arff'):
  i = 1
  for v in row:
    if i not in train_data:
      train_data[i] = []
    train_data[i].append(str(int(v)) if i < 7 else str(v))
    i += 1
  
for row in arff.load('hw4test.arff'):
  i = 1
  for v in row:
    if i not in test_data:
      test_data[i] = []
    test_data[i].append(str(int(v)) if i < 7 else str(v))
    i += 1

file_base = 'a'
test_end = 'test.arff'
train_end = 'train.arff'

# deleted 4 from these because it had the best results for 5 attributes
cols = [1,2,3,5,6,7]
attributes = [1,2,3,5,6]

for a in attributes:
  te = open(file_base + str(a) + test_end, 'w')
  tr = open(file_base + str(a) + train_end, 'w')
  te.write('@relation a4' + str(a) + '\n\n') 
  tr.write('@relation a4' + str(a) + '\n\n') 
  for z in attributes:
    if z != a:
      te.write('@attribute a' + str(z) + ' numeric\n')
      tr.write('@attribute a' + str(z) + ' numeric\n')
  te.write('@attribute label {a, b, c}\n\n@data\n')
  tr.write('@attribute label {a, b, c}\n\n@data\n')
  for i, r in enumerate(train_data[a]):
    record = ''
    for c in cols:
      if a != c:
         record += train_data[c][i] + ','
    record = record[:-1] + '\n'
    tr.write(record)
  for i, r in enumerate(test_data[a]):
    record = ''
    for c in cols:
      if a != c:
         record += test_data[c][i] + ','
    record = record[:-1] + '\n'
    te.write(record)
  te.close()
  tr.close()
