import os

root_path = "rt_gene_dataset"
result_path ="./integrated_label.txt"
target_name = "label_combined.txt"

result_file = open(result_path, "w")
for i in range(1,15):
    sub_path = f's{i:0>3}_glasses'
    label_path = os.path.join(root_path,sub_path, target_name)
    f= open(label_path, "r")
    #discard header
    line = f.readline()

    line = f.readline()

    while line != '':
        s_line = line.split(',')
        s_line[0] = f's{i:0>3}_' +s_line[0]

        new_line = ','.join(s_line)
        result_file.write(new_line)

        line = f.readline()