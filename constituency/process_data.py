def process_file(name='train'):
    input_file=open('en_my_%s.mrg'%name)
    lines=input_file.readlines()
    with open('en_out_%s.mrg'%name,'w') as f:
        f.write('\n')
        new_lines=[]
        par_count=0
        for i,line in enumerate(lines):
            line=line.strip('\n')
            
            if i % 10000 == 0:
                print('processing line:',i)
            if i == 100000:
                break
            if not line.strip():
                continue
            if line.startswith('('):
                new_lines=[]
                par_count=line.count('(') - line.count(')')
                new_lines.append(line.strip())
                continue
            splits_parant=line.split('(')
            if len(splits_parant) < 2: 
                continue
            txt = splits_parant[0] 
            
            if  len(splits_parant[1].strip().split(' '))== 1 :
                pre_txt = txt + '('+splits_parant[1].strip() +' '
                txt += '('+splits_parant[1].strip().split('-')[0] +' '
            else:
                pre_txt = txt + '('+splits_parant[1].strip() +' '
                
            if pre_txt.strip() == line.strip():
                txt = pre_txt
                
                
                splits_spc=line.strip().split(' ')
                if len(splits_spc) > 1:
                    par_count += txt.count('(') - txt.count(')')
                    added_par =0
                    for _ in range(par_count):
                        txt += ')'
                        par_count -= 1
                        added_par += 1
                    new_lines.append(txt)                
                    for nline in new_lines:
                        f.write(nline)
                        f.write('\n')

                    new_lines.remove(txt)
                    par_count -= txt.count('(') - txt.count(')')
                    txt=txt[:-added_par-1]
                    new_lines.append(txt)
                    par_count += txt.count('(') -  txt.count(')')
                    continue
                else:
                    new_lines.append(txt)
                    par_count += txt.count('(') -  txt.count(')')
                    continue
            for item in splits_parant:
                splits_spc=item.strip().split(' ')
                if len(splits_spc) > 1:
                    txt += '(' 
                    
                    for splt_spc in  splits_spc:
                        if len(splt_spc.replace(')','')) > 0:
                            txt += splt_spc + ' '
                        else:
                            txt += splt_spc.rstrip()
                    par_count += txt.count('(') -  txt.count(')')
                    added_par =0
                    for _ in range(par_count):
                        txt += ')'
                        par_count -= 1
                        added_par += 1
                    new_lines.append(txt) 
                    for nline in new_lines:
                            f.write(nline)
                            f.write('\n')

                    new_lines.remove(txt)
                    par_count -= txt.count('(') - txt.count(')') 
                    txt=txt[:-added_par]
                    if txt.count('(') == line.count('('):
                        new_lines.append(txt)
                        par_count += txt.count('(') -  txt.count(')') 
                        
                        
                    
process_file(name='train')
process_file(name='dev')
process_file(name='test')                   
                    
                    
        
        

