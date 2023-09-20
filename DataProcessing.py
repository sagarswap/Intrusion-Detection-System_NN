import csv
import pandas as pd
import random

fields=[]
rows=[]
important_fields=["ACK Flag Count", "Active Min", "Active Mean", "Average Packet Size", "Bwd IAT Min", "Bwd IAT Mean", 
                  "Bwd Packet Length Min", "Bwd Packet Length Std", "Bwd Packets/s", "Fwd Packet Length Mean", "Fwd IAT Mean", 
                  "Fwd IAT Min", "Fwd PSH Flags", "Fwd Packets/s", "Flow Duration", "Flow IAT Mean", "Flow IAT Std", 
                  "Flow IAT Min", "Init_Win_bytes_forward", "PSH Flag Count", "SYN Flag Count", "Subflow Fwd Bytes", 
                  "Total Length of Fwd Packets", "Label"]
important_fields_index=[]
no_of_fields=[]
no_of_desired_fields=len(important_fields)
no_of_rows=[]

with open("./data/Wednesday_wh.csv", 'r') as csvfile:
    csvreader=csv.reader(csvfile)
    fields = next(csvreader)
    fields=[j.strip() for j in fields]
    for row in csvreader:
        rows.append(row)
    #print("Total no. of rows: %d"%(csvreader.line_num))
    no_of_rows=len(rows)
    no_of_fields=len(rows[0])

indx=0
for i in range(no_of_desired_fields):
    indx=0
    for j in range (no_of_fields):
        if fields[j]==important_fields[i] :
            important_fields_index.append(j)
            indx=1
    if indx==0:
        print(important_fields[i])
print(important_fields_index)

output=[]
for i in range(no_of_rows):
    row=[]
    for x in important_fields_index:
        row.append(rows[i][x])
    output.append(row)
    #print(row)

ans_ind=len(important_fields_index)-1
for row in output:
    if row[ans_ind]=="BENIGN":
        row.append("1.0")
        row.append("0.0")
        row.append("0.0")
        row.append("0.0")
        row.append("0.0")
        row.append("0.0")
    elif row[ans_ind]=="DoS slowloris":
        row.append("0.0")
        row.append("1.0")
        row.append("0.0")
        row.append("0.0")
        row.append("0.0")
        row.append("0.0")
    elif row[ans_ind]=="DoS Slowhttptest":
        row.append("0.0")
        row.append("0.0")
        row.append("1.0")
        row.append("0.0")
        row.append("0.0")
        row.append("0.0")
    elif row[ans_ind]=="DoS Hulk":
        row.append("0.0")
        row.append("0.0")
        row.append("0.0")
        row.append("1.0")
        row.append("0.0")
        row.append("0.0")
    elif row[ans_ind]=="DoS GoldenEye":
        row.append("0.0")
        row.append("0.0")
        row.append("0.0")
        row.append("0.0")
        row.append("1.0")
        row.append("0.0")
    elif row[ans_ind]=="Heartbleed":
        row.append("0.0")
        row.append("0.0")
        row.append("0.0")
        row.append("0.0")
        row.append("0.0")
        row.append("1.0")
    del row[ans_ind]
del important_fields[ans_ind]
important_fields.append("BENIGN")
important_fields.append("DoS slowloris")
important_fields.append("DoS Slowhttptest")
important_fields.append("DoS Hulk")
important_fields.append("DoS GoldenEye")
important_fields.append("Heartbleed")

train_ind_end=0.9*no_of_rows
train_data=[]
test_data=[]
train_data.append(important_fields)
test_data.append(important_fields)
for row in output:
    if random.choice(range(0, 100))>10:
        train_data.append(row)
    else:
        test_data.append(row)

df1=pd.DataFrame(train_data)
df1.to_csv('./data/train_data.csv', header=False, index=False)
df2=pd.DataFrame(test_data)
df2.to_csv('./data/test_data.csv', header=False, index=False)


#print('Field names are:' + ', '.join(field for field in fields))