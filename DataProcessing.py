import csv

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
important_fields_index.sort()
print(important_fields_index)
output=[]
for i in range(no_of_rows):
    row=[]
    for x in important_fields_index:
        row.append(rows[i][x])
    output.append(row)
    #print(row)

#print(output)




#print('Field names are:' + ', '.join(field for field in fields))