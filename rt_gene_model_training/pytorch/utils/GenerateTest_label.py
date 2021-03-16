import os

script_path = os.path.dirname(os.path.realpath(__file__))
rt_gene_root = "../../rt_gene_dataset"
subject_path = [os.path.join(rt_gene_root, "s{:03d}_glasses/".format(_i)) for _i in range(0, 17)]

for subject in subject_path:
    out_f = open(os.path.join(subject, "label_combined_test.txt"), "w")
    with open(os.path.join(subject, "label_combined.txt"), "r") as f:
        for i in range(5):
            line = f.readline()
            out_f.write(line)


