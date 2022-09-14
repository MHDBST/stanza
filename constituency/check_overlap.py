path1='/home/mbastan/structuralDecoding/neurologic_decoding/constituency/en_outall_test.mrg'
path2='/home/mbastan/structuralDecoding/neurologic_decoding/constituency/en_outall_train.mrg'
file1=open(path1)
file2=open(path2)

lines1=file1.readlines()
lines2=file2.readlines()
arr1=[]
new_arr= [lines1[1]]
for line in lines1[2:]:
    if line.startswith('('):
        arr1.append(new_arr)
        new_arr=[]
    new_arr.append(line.strip())
arr1.append(new_arr)


arr2=[]
new_arr= [lines2[1]]
for line in lines2[2:]:
    if line.startswith('('):
        arr2.append(new_arr)
        new_arr=[]
    new_arr.append(line.strip())
arr2.append(new_arr)

match_count=0
for arr in arr1:
    if arr in arr2:
        # print('match found')
        match_count +=1
print('match found:',match_count)
print('match percentage:',match_count/len(arr1))
print('path1',path1)
print('path2',path2)
