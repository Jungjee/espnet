import argparse
import os
import sys


def main(args):
    src = args.src
    dst = args.dst

    spk2utt = {}
    utt2spk = []
    wav_list = []

    for r, ds, fs in os.walk(src):
        for f in fs:
            if os.path.splitext(f)[1] != ".wav":
                continue

            utt_dir = os.path.join(r, f)
            lid, utt = utt_dir.split("/")[-2:]  # language id, utterance
            utt_id = "/".join([lid, utt.split(".")[0]])
            if lid not in spk2utt:
                spk2utt[lid] = []
            spk2utt[lid].append(utt_id)
            utt2spk.append([utt_id, lid])
            wav_list.append([utt_id, utt_dir])

    with open(os.path.join(dst, "spk2utt"), "w") as f_spk2utt, open(
        os.path.join(dst, "utt2spk"), "w"
    ) as f_utt2spk, open(os.path.join(dst, "wav.scp"), "w") as f_wav:
        for lid in spk2utt:
            f_spk2utt.write(f"{lid}")
            for utt in spk2utt[lid]:
                f_spk2utt.write(f" {utt}")
            f_spk2utt.write("\n")

        for utt in utt2spk:
            f_utt2spk.write(f"{utt[0]} {utt[1]}\n")

        for utt in wav_list:
            f_wav.write(f"{utt[0]} {utt[1]}\n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxCeleb 1&2 downloader")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="source directory of voxcelebs",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="destination directory of voxcelebs",
    )
    args = parser.parse_args()

    sys.exit(main(args))
