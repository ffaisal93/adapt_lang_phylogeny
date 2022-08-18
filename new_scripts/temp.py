import json
import os

with open('../meta_files/lang_meta_ie.json') as json_file:
        lang_data = json.load(json_file)
        all_data={}
        lang_key={}
        reg_key={}
        reg_key_inv={}
        reg_dict={}
        reg_count=0
        fa_key={}
        fa_key_inv={}
        fa_dict={}
        for j,la_f in enumerate(lang_data):
            fa_key[j]=la_f
            fa_key_inv[la_f]=j
            for i,val in enumerate(set(lang_data[la_f].values())):
                    reg_key[reg_count]=la_f+'_'+val
                    reg_key_inv[la_f+'_'+val]=reg_count
                    reg_count+=1
        reg_count=0
        for j,la_f in enumerate(lang_data):
            for i,val in lang_data[la_f].items():
                fa_dict[i]=fa_key_inv[la_f]
                reg_dict[i]=reg_key_inv[la_f+'_'+val]
                reg_count+=1

print('reg_key',reg_key)

print('reg_dict',reg_dict)

print('reg_key_inv',reg_key_inv)

print('fa_key',fa_key)

print('fa_dict',fa_dict)

print('fa_key_inv',fa_key_inv)

count=0
for x1 in os.listdir('../data/ie'):
    if x1!='.DS_Store' and x1!='._.DS_Store' and x1!='readme.md' and x1!='family.txt':
        for x in os.listdir(os.path.join('../data/ie',x1)):
            if x!='.DS_Store' and x!='readme.md' and x!='family.txt' and x.startswith('._')==False:
                fname = x.split('.txt')[0]
                train_file = os.path.join('../data/ie',x,x1)
                # logger.info(fname)
                print(fname)
                data_files = {}
                lang_key[fname]=count
                count+=1
                # if data_args.train_file is not None:
                #     data_files["train"] = data_args.train_file
                #     extension = data_args.train_file.split(".")[-1]
                # if data_args.validation_file is not None:
                #     data_files["validation"] = data_args.validation_file
                #     extension = data_args.validation_file.split(".")[-1]
                # if extension == "txt":
                #             extension = "text"

print(lang_key)