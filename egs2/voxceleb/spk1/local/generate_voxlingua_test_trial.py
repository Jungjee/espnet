import os,sys
import numpy as np
from glob import glob

np.random.seed(0)

def main(args):
    root_dir = args[0]
    out_dir = args[1]
    n_utt_trg = int(args[2])

    print(root_dir)
    utt_list = glob(root_dir+"/*/*.wav")
    print(utt_list[:10])
    print(len(utt_list))

    cls_dict = dict()
    for utt in utt_list:
        cls = utt.split("/")[-2]
        if cls not in cls_dict:
            cls_dict[cls] = []
        cls_dict[cls].append(utt)

    f_out1 = open(out_dir+"/trial.scp", "w")
    f_out2 = open(out_dir+"/trial2.scp", "w")
    f_label = open(out_dir+"/trial_label", "w")
    utt_id_set = set()
    # generate target trials
    for i in range(n_utt_trg):
        sel_cls = np.random.choice(list(cls_dict.keys()), 1)[0]
        if len(cls_dict[sel_cls]) < 2:
            print(f"{sel_cls} has {len(cls_dict[sel_cls])} samples only..")
            continue
        utt1, utt2 = np.random.choice(cls_dict[sel_cls], 2, replace=False)

        utt1_id = os.path.splitext("/".join(utt1.split("/")[-2:]))[0]
        utt2_id = os.path.splitext("/".join(utt2.split("/")[-2:]))[0]
        utt_id = f"{utt1_id}*{utt2_id}"
        if utt_id in utt_id_set or f"{utt2_id}*{utt1_id}" in utt_id_set:
            print("already existing trial")
            continue
        else:
            utt_id_set.add(utt_id)

        f_out1.write(f"{utt_id} {utt1}\n")
        f_out2.write(f"{utt_id} {utt2}\n")
        f_label.write(f"{utt_id} 1\n")

    # generate non-target trials
    for i in range(n_utt_trg):
        sel_cls1, sel_cls2 = np.random.choice(list(cls_dict.keys()), 2)
        utt1 = np.random.choice(cls_dict[sel_cls1], 1)[0]
        utt2 = np.random.choice(cls_dict[sel_cls2], 1)[0]

        utt1_id = os.path.splitext("/".join(utt1.split("/")[-2:]))[0]
        utt2_id = os.path.splitext("/".join(utt2.split("/")[-2:]))[0]
        utt_id = f"{utt1_id}*{utt2_id}"

        if utt_id in utt_id_set or f"{utt2_id}*{utt1_id}" in utt_id_set:
            print("already existing trial")
            continue
        else:
            utt_id_set.add(utt_id)

        f_out1.write(f"{utt_id} {utt1}\n")
        f_out2.write(f"{utt_id} {utt2}\n")
        f_label.write(f"{utt_id} 0\n")

    print(f"Generated {len(utt_id_set)} trials")
    f_out1.close()
    f_out2.close()
    f_label.close()

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
