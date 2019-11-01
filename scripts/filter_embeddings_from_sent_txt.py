import sys
import json
import codecs

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: {} <embeddings> <filtered_embeddings> <sent> ...".format(sys.argv[0]))

    words_to_keep = set()
    for json_filename in sys.argv[3:]:
        with open(json_filename) as txt_file:
            print("Read words from", json_filename)
            for i, line in enumerate(txt_file.readlines()):
                tokens = line.strip().split(' ')
                for word in tokens:
                    words_to_keep.add(word)

    print("Found {} words in {} dataset(s).".format(len(words_to_keep), len(sys.argv) - 3))
    total_lines = 0
    kept_lines = 0
    out_filename = sys.argv[2]
    with open(sys.argv[1]) as in_file:
        with open(out_filename, "w") as out_file:
            print("write emb file into", out_filename)
            for line in in_file.readlines():
                total_lines += 1
                word = line.split()[0]
                if word in words_to_keep:
                    kept_lines += 1
                    out_file.write(line)

    print("Kept {} out of {} lines.".format(kept_lines, total_lines))
    print("Write result to {}.".format(out_filename))
