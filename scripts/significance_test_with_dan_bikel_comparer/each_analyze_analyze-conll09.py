import sys
import copy


"""
Notes: eval (frames_number + semantic dependencies)
where, semantic dependencies is calculated with the gold frames
"""


def read_conll_file(filename, gold_frames=None):
    with open(filename, 'r') as input_file:
        sentence_num = 0
        total_frames, total_semantic_dependencies, total_words = [], [], []
        # for frames structures in a sentence
        frames, frames_structures, words = [], [], []
        sentence_id = 0
        for id, line in enumerate(input_file.readlines()):
            if line.strip() == "":
                sentence_gold_frame = None
                if gold_frames is not None:
                    sentence_gold_frame = gold_frames[sentence_id]
                each_semantic_structure = zip(*frames_structures)
                assert len(each_semantic_structure) == len(frames)
                # sentence storage
                total_words.append(words)
                sentence_frames, sentence_semantic_dependencies = [], []

                for prd, semantic_roles in zip(frames, each_semantic_structure):
                    sentence_frames.append([prd, prd, "V"])
                focused_frames = frames if sentence_gold_frame is None else copy.deepcopy(sentence_gold_frame)
                for prd, semantic_roles in zip(focused_frames, each_semantic_structure):
                    assert len(words) == len(semantic_roles)
                    for i, semantic_dependency in enumerate(semantic_roles):
                        if semantic_dependency == "_":
                            continue
                        sentence_semantic_dependencies.append([prd if sentence_gold_frame is None else prd[0],
                                                               str(i + 1) + words[i], semantic_dependency])
                        # sentence_semantic_dependencies.append([prd, str(i), semantic_dependency])
                # print sentence_semantic_dependencies
                total_frames.append(sentence_frames)
                total_semantic_dependencies.append(sentence_semantic_dependencies)
                # if sentence_num == 4:
                #     exit()
                sentence_num += 1
                # renew
                frames, frames_structures, words = [], [], []
                sentence_id += 1
                continue
            words.append(line.strip().split()[1])
            tokens = line.strip().split()[12:]
            is_predicate = tokens[0] == "Y"
            if is_predicate:
                assert tokens[1] != "_"
                # frames_number = len(tokens)
                frames.append(str(id) + tokens[1])
            frames_structures.append(tokens[2:])
        print("Total {} sentences in {}".format(sentence_num, filename))
        print("Total {} semantic dependencies in {}".format(
            sum([len(sen_semantic_dep) for sen_semantic_dep in total_semantic_dependencies]), filename))
    return total_frames, total_semantic_dependencies, total_words


def get_p_r_f(gold, sys):
    assert len(gold) == len(sys)
    gold_frames, gold_semantic_dependencies, gold_words = gold
    sys_frames, sys_semantic_dependencies , sys_words = sys
    total_correct_frames, total_correct_semantic_dependencies = 0, 0

    assert len(gold_frames) == len(sys_frames)
    for gold_fs, sys_fs in zip(gold_frames, sys_frames):
        for gold_frame in gold_fs:
            str_gold = "=".join(gold_frame)
            for sys_frame in sys_fs:
                str_sys = "=".join(sys_frame)
                if str_gold == str_sys:
                    total_correct_frames += 1

    assert len(gold_semantic_dependencies) == len(sys_semantic_dependencies)
    for gold_semantci_deps, sys_semantic_deps in zip(gold_semantic_dependencies, sys_semantic_dependencies):
        correct = 0
        for gold_dep in gold_semantci_deps:
            str_gold = "=".join(gold_dep)
            # print str_gold
            for sys_dep in sys_semantic_deps:
                # print sys_dep
                str_sys = "=".join(sys_dep)
                # print str_sys
                if str_gold == str_sys:
                    correct += 1
                    total_correct_semantic_dependencies += 1

            # exit()
        # if len(sys_semantic_dependencies) == 0 or len(gold_semantci_deps) == 0 or correct == 0:
        #     continue
        # p = correct * 1.0 / len(sys_semantic_deps)
        # r = correct * 1.0 / len(gold_semantci_deps)
        # f = 2 * p * r / (p + r)
        # print(p, r, f)
    gold_frames_num = sum(len(sen_deps) for sen_deps in gold_frames)
    gold_semantic_dependencies_num = sum(len(sen_deps) for sen_deps in gold_semantic_dependencies)
    sys_frames_num = sum(len(sen_deps) for sen_deps in sys_frames)
    sys_semantic_dependencies_num = sum(len(sen_deps) for sen_deps in sys_semantic_dependencies)

    print("\n" + "="*10 + "frames precision" + "="*10)
    frame_precision = total_correct_frames * 1.0 / gold_frames_num
    print("Frames precision: {} / {} = {}".format(total_correct_frames, gold_frames_num, frame_precision))

    print("\n" + "="*10 + "semantic dependencies precision(%)" + "="*10)
    p = 100.0 * (total_correct_frames + total_correct_semantic_dependencies) * 1.0 / (sys_frames_num + sys_semantic_dependencies_num)
    print("Precision: ({} + {}) / ({} + {}) = {}".format(total_correct_semantic_dependencies, total_correct_frames,
                                                         sys_semantic_dependencies_num, sys_frames_num, round(p, 2)))
    r = 100.0 * (total_correct_frames + total_correct_semantic_dependencies) * 1.0 / (gold_frames_num + gold_semantic_dependencies_num)
    print("Recall: ({} + {}) / ({} + {}) = {}".format(total_correct_semantic_dependencies, total_correct_frames,
                                                      gold_semantic_dependencies_num, gold_frames_num, round(r, 2)))
    f = 2.0 * (p * r) / (p + r)
    print("F1 score: {}\n".format(round(f, 2)))


def get_performance_by_sentence_length(gold, sys):
    """
    0-10, 11-20, 21-30, 31-40, 41-max
    :param gold:
    :param sys:
    :return:
    """
    assert len(gold) == len(sys)
    gold_frames, gold_semantic_dependencies, gold_words = gold
    sys_frames, sys_semantic_dependencies, sys_words = sys
    total_correct_frames, total_correct_semantic_dependencies = 0, 0
    seg_correct_frames, seg_correct_semantic_dependencies = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    seg_gold_frames, seg_gold_semantic_dependencies = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    seg_sys_frames, seg_sys_semantic_dependencies = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    assert len(gold_frames) == len(sys_frames)
    assert len(gold_semantic_dependencies) == len(sys_semantic_dependencies)
    resulting_f = []
    sent_id = 0
    for words, gold_fs, sys_fs, gold_semantci_deps, sys_semantic_deps \
            in zip(gold_words, gold_frames, sys_frames, gold_semantic_dependencies, sys_semantic_dependencies):
        sentence_level_correct_frames, sen_level_correct_semantic_dependencies = 0, 0
        sentence_length = len(words)
        index = -1
        if sentence_length <= 10:
            index = 0
        elif 11 <= sentence_length <=20:
            index = 1
        elif 21 <= sentence_length <= 30:
            index = 2
        elif 31 <= sentence_length <=40:
            index = 3
        else:
            index = 4

        for gold_frame in gold_fs:
            str_gold = "=".join(gold_frame)
            for sys_frame in sys_fs:
                str_sys = "=".join(sys_frame)
                if str_gold == str_sys:
                    sentence_level_correct_frames += 1

        for gold_dep in gold_semantci_deps:
            str_gold = "=".join(gold_dep)
            # print str_gold
            for sys_dep in sys_semantic_deps:
                # print sys_dep
                str_sys = "=".join(sys_dep)
                # print str_sys
                if str_gold == str_sys:
                    sen_level_correct_semantic_dependencies += 1
                if gold_dep[-1] == "DIS" and sys_dep[-1] == "ADV" and (' '.join(gold_dep[:2]) == ' '.join(sys_dep[:2])):
                    print ' '.join(words)
                    print ' '.join(gold_dep)
                    print ' '.join(sys_dep)

        total_correct_frames += sentence_level_correct_frames
        total_correct_semantic_dependencies += sen_level_correct_semantic_dependencies
        if len(sys_semantic_dependencies) == 0 or len(gold_semantci_deps) == 0 \
                or sen_level_correct_semantic_dependencies == 0:
            continue
        p = (sentence_level_correct_frames + sen_level_correct_semantic_dependencies) * 100.0 / \
            (len(sys_fs) + len(sys_semantic_deps))
        r = (sentence_level_correct_frames + sen_level_correct_semantic_dependencies) * 100.0 / \
            (len(gold_fs) + len(gold_semantci_deps))
        f = 2.0 * p * r / (p + r)
        f = round(f, 2)
        # print(p, r, f)
        resulting_f.append((sent_id, f))
        sent_id += 1

        # ==================== store sentence length level ===================#
        seg_correct_frames[index] += sentence_level_correct_frames
        seg_correct_semantic_dependencies[index] += sen_level_correct_semantic_dependencies
        seg_gold_frames[index] += len(gold_fs)
        seg_gold_semantic_dependencies[index] += len(gold_semantci_deps)
        seg_sys_frames[index] += len(sys_fs)
        seg_sys_semantic_dependencies[index] += len(sys_semantic_deps)
    # print(total_correct_frames, total_correct_semantic_dependencies, len(resulting_f))
    infos = ["0-10", "11-20", "21-30", "31-40", "41-max"]
    print("\n" + "="*10 + "performance according to the sentence lengths (F1 score)" + "="*10)
    for i in range(5):
        p = (seg_correct_frames[i] + seg_correct_semantic_dependencies[i]) * 100.0 / \
            (seg_sys_frames[i] + seg_sys_semantic_dependencies[i])
        r = (seg_correct_frames[i] + seg_correct_semantic_dependencies[i]) * 100.0 / \
            (seg_gold_frames[i] + seg_gold_semantic_dependencies[i])
        f = 2.0 * p * r / (p + r)
        f = round(f, 2)
        print(infos[i] + "\tF1 score: {}".format(f))
    return resulting_f


def get_performance_by_semantic_roles(gold, sys):
    assert len(gold) == len(sys)
    gold_frames, gold_semantic_dependencies, gold_words = gold
    sys_frames, sys_semantic_dependencies , sys_words = sys
    total_correct_frames, total_correct_semantic_dependencies = 0, 0

    assert len(gold_frames) == len(sys_frames)
    for gold_fs, sys_fs in zip(gold_frames, sys_frames):
        for gold_frame in gold_fs:
            str_gold = "=".join(gold_frame)
            for sys_frame in sys_fs:
                str_sys = "=".join(sys_frame)
                if str_gold == str_sys:
                    total_correct_frames += 1

    dict_total_correct_semantic_dependencies, dict_total_gold_semantic_dependencies = {}, {}
    dict_total_sys_semantic_dependencies = {}
    assert len(gold_semantic_dependencies) == len(sys_semantic_dependencies)
    for gold_semantci_deps, sys_semantic_deps in zip(gold_semantic_dependencies, sys_semantic_dependencies):
        correct = 0
        for gold_dep in gold_semantci_deps:
            str_gold = "=".join(gold_dep)
            gold_dependency = gold_dep[-1]
            if gold_dependency in dict_total_gold_semantic_dependencies.keys():
                dict_total_gold_semantic_dependencies[gold_dependency] += 1
            else:
                dict_total_gold_semantic_dependencies[gold_dependency] = 1
            # print str_gold
            for sys_dep in sys_semantic_deps:
                # print sys_dep
                str_sys = "=".join(sys_dep)
                # print str_sys
                if str_gold == str_sys:
                    correct += 1
                    total_correct_semantic_dependencies += 1
                    dependency = sys_dep[-1]
                    if dependency in dict_total_correct_semantic_dependencies.keys():
                        dict_total_correct_semantic_dependencies[dependency] += 1
                    else:
                        dict_total_correct_semantic_dependencies[dependency] = 1
        for sys_dep in sys_semantic_deps:
            sys_dependency = sys_dep[-1]
            if sys_dependency in dict_total_sys_semantic_dependencies.keys():
                dict_total_sys_semantic_dependencies[sys_dependency] += 1
            else:
                dict_total_sys_semantic_dependencies[sys_dependency] = 1
            # exit()
        # if len(sys_semantic_dependencies) == 0 or len(gold_semantci_deps) == 0 or correct == 0:
        #     continue
        # p = correct * 1.0 / len(sys_semantic_deps)
        # r = correct * 1.0 / len(gold_semantci_deps)
        # f = 2 * p * r / (p + r)
        # print(p, r, f)
    list_correct_semantic_dep = sorted(dict_total_correct_semantic_dependencies.items(), key=lambda kv: kv[0])
    list_gold_semantic_dep = sorted(dict_total_gold_semantic_dependencies.items(), key=lambda kv: kv[0])
    list_sys_semantic_dep = sorted(dict_total_sys_semantic_dependencies.items(), key=lambda kv: kv[0])
    # assert len(list_correct_semantic_dep) == len(list_gold_semantic_dep) == len(list_sys_semantic_dep)
    print("\n" + "=" * 10 + "performance according to the semantic roles (F1 score)" + "=" * 10)
    print(len(list_correct_semantic_dep), len(list_gold_semantic_dep), len(list_sys_semantic_dep))
    for role, num in list_correct_semantic_dep:
        if role in dict_total_gold_semantic_dependencies and role in dict_total_sys_semantic_dependencies:
            precision = num * 100.0 / dict_total_sys_semantic_dependencies[role]
            recall = num * 100.0 / dict_total_gold_semantic_dependencies[role]
            f = 2.0 * precision * recall / (precision + recall)
            print("Semantic Role: {}, F1 score: {}, \tCorrect Gold Sys: {}, {}, {}".format(role, round(f, 2),
                            num, dict_total_gold_semantic_dependencies[role], dict_total_sys_semantic_dependencies[role]))


def analyze_specific_semantic_roles(gold, sys, role="A0"):
    assert len(gold) == len(sys)
    gold_frames, gold_semantic_dependencies, gold_words = gold
    sys_frames, sys_semantic_dependencies , sys_words = sys

    dict_error_storage = {}
    assert len(gold_semantic_dependencies) == len(sys_semantic_dependencies)
    for gold_semantci_deps, sys_semantic_deps in zip(gold_semantic_dependencies, sys_semantic_dependencies):
        for gold_dep in gold_semantci_deps:
            str_gold = "=".join(gold_dep)
            gold_first_two = "=".join(gold_dep[:2])
            gold_dependency = gold_dep[-1]
            is_correct = False
            if gold_dependency == role:
                # print str_gold
                for sys_dep in sys_semantic_deps:
                    # print sys_dep
                    str_sys = "=".join(sys_dep)
                    sys_first_two = "=".join(sys_dep[:2])
                    sys_dependency = sys_dep[-1]
                    # print str_sys
                    if gold_first_two == sys_first_two:
                        if gold_dependency == sys_dependency:
                            is_correct = True
                        else:
                            is_correct = False
                            if sys_dependency in dict_error_storage.keys():
                                dict_error_storage[sys_dependency] += 1
                            else:
                                dict_error_storage[sys_dependency] = 1
    print("\n" + "=" * 10 + "performance according to the specific semantic roles (didn't count missed)" + role + "=" * 10)
    sorted_by_value = sorted(dict_error_storage.items(), key=lambda kv: kv[1])
    print(sorted_by_value)
    print("Total error num: {}".format(sum([item[1] for item in sorted_by_value])))


def get_each_sentence_p_r_f(gold, sys):
    assert len(gold) == len(sys)
    gold_frames, gold_semantic_dependencies, gold_words = gold
    sys_frames, sys_semantic_dependencies , sys_words = sys
    total_correct_frames, total_correct_semantic_dependencies = 0, 0

    assert len(gold_frames) == len(sys_frames)
    for gold_fs, sys_fs in zip(gold_frames, sys_frames):
        for gold_frame in gold_fs:
            str_gold = "=".join(gold_frame)
            for sys_frame in sys_fs:
                str_sys = "=".join(sys_frame)
                if str_gold == str_sys:
                    total_correct_frames += 1

    assert len(gold_semantic_dependencies) == len(sys_semantic_dependencies)
    sentenc_id = 0
    for gold_semantci_deps, sys_semantic_deps in zip(gold_semantic_dependencies, sys_semantic_dependencies):
        correct = 0
        for gold_dep in gold_semantci_deps:  # sentence level
            str_gold = "=".join(gold_dep)
            # print str_gold
            for sys_dep in sys_semantic_deps:
                # print sys_dep
                str_sys = "=".join(sys_dep)
                # print str_sys
                if str_gold == str_sys:
                    correct += 1
                    total_correct_semantic_dependencies += 1

            # exit()
        if len(gold_semantci_deps) == 0:
            continue
        if len(sys_semantic_dependencies) == 0:
            r = correct * 100.0 / len(gold_semantci_deps)
            print '\t', sentenc_id, '\t', num_gold, '\t', 0, '\t', "%.2f" % 0.0, '\t', "%.2f" % r, '\t', correct, '\t', correct, '\t', num_gold, '\t', "0 0 0 0"
            continue
        p = correct * 100.0 / len(sys_semantic_deps)
        r = correct * 100.0 / len(gold_semantci_deps)
        # f = 2 * p * r / (p + r)
        num_gold = len(gold_semantci_deps)

        print '\t', sentenc_id, '\t', num_gold, '\t', 0, '\t', "%.2f" % p, '\t', "%.2f" % r, '\t', correct, '\t', correct, '\t', num_gold, '\t', "0 0 0 0"
        sentenc_id += 1


if __name__ == "__main__":
    gold_filename = sys.argv[1]
    sys_filename = sys.argv[2]

    # read the conll gold file
    gold_semantic_dependencies = read_conll_file(gold_filename)
    sys_semantic_dependencies = read_conll_file(sys_filename, gold_semantic_dependencies[0])
    get_each_sentence_p_r_f(gold_semantic_dependencies, sys_semantic_dependencies)
    # P R F
    # get_p_r_f(gold_semantic_dependencies, sys_semantic_dependencies)
    # # performance by sentence length
    # get_performance_by_sentence_length(gold_semantic_dependencies, sys_semantic_dependencies)
    # # performance by semantic roles
    # get_performance_by_semantic_roles(gold_semantic_dependencies, sys_semantic_dependencies)
    # # analyze specific semantic role
    # analyze_specific_semantic_roles(gold_semantic_dependencies, sys_semantic_dependencies, "DIS")



