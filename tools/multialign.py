import random
import os
from os import listdir, mkdir
from os.path import join as join_path
from copy import deepcopy
from multiprocessing import Process, Pipe, Queue
from tqdm import tqdm


def load_corpus(data_dir):
    corpus = {}
    for file_name in listdir(data_dir):
        path = join_path(data_dir, file_name)
        if os.path.isfile(path):
            file = open(path, "r+")
            l_pair = file_name[:5]
            if l_pair not in corpus:
                corpus[l_pair] = {}
            corpus[l_pair][file_name[6:-4]] = file.read().split('\n')
            file.close()
    return corpus


def get_aligned_sentences(comm_pipe, idx_queue, p_queue, corpus, ref_corpus, lang_pairs, local_sentence_n, log_rate):
    sentences_found = {'en': [], 'es': [], 'de': [], 'fr': []}
    while len(sentences_found['en']) != local_sentence_n:
        ref_idx = idx_queue.get()
        ref_sentence = ref_corpus[ref_idx]

        candidates = {}
        for pair in lang_pairs[1:]:
            try:
                trans_idx = corpus[pair]['en'].index(ref_sentence)
                lang_code = list(
                    filter(lambda x: x != 'en', corpus[pair].keys()))[0]
                # translation = corpus[pair][lang_code][trans_idx]
                candidates[lang_code] = trans_idx
            except ValueError:
                pass

        for lang_code, _ in corpus[lang_pairs[0]].items():
            candidates[lang_code] = ref_idx

        if len(candidates.keys()) == len(lang_pairs) + 1:
            for key in sentences_found:
                sentences_found[key].append(candidates[key])
            if len(sentences_found['en']) % log_rate == 0:
                p_queue.put(1)

    comm_pipe.send(sentences_found)


def progress_listener(progress_queue, log_rate):
    p_bar = tqdm(total=sentence_num)
    for _ in iter(progress_queue.get, None):
        p_bar.update(log_rate)
    p_bar.close()


def get_lang_corpus(corpus, lang_code):
    for _, pair_data in corpus.items():
        for key, data in pair_data.items():
            if key == lang_code:
                return data
    return None


def align_corpus(corpus, sentence_num, process_num, log_rate):
    lang_pairs = list(corpus.keys())
    ref_corp = corpus[lang_pairs[0]]['en']
    index_queue = Queue()
    index_pool = list(range(0, len(ref_corp)))
    random.shuffle(index_pool)
    [index_queue.put(idx) for idx in index_pool]

    process_list = []
    conn_points = []
    aligned_corpus = {'en': [], 'es': [], 'de': [], 'fr': []}

    progress_queue = Queue()
    listener = Process(target=progress_listener,
                       args=(progress_queue, log_rate))
    listener.start()

    for _ in range(0, process_num):
        parent_conn, child_conn = Pipe()
        p = Process(target=get_aligned_sentences, args=(child_conn, index_queue, progress_queue,
                                                        corpus, ref_corp, lang_pairs, int(sentence_num / process_num), log_rate))
        p.start()
        process_list.append(p)
        conn_points.append(parent_conn)

    for conn in conn_points:
        aligned_sentences = conn.recv()
        for key in aligned_sentences.keys():
            lang_data = get_lang_corpus(corpus, key)
            aligned_corpus[key] += list(map(lambda x: lang_data[x],
                                            aligned_sentences[key]))
        conn.close()

    for p in process_list:
        p.join()

    progress_queue.put(None)
    listener.join()

    # Flush index Queue
    print("Flushing Queue...")
    while not index_queue.empty():
        index_queue.get()

    print("Finished!")

    return aligned_corpus


def write_aligned_corpus(aligned_corpus, data_dir, val_test_size):
    aligned_path = join_path(data_dir, 'aligned')
    train_path = join_path(aligned_path, 'train')
    val_path = join_path(aligned_path, 'validation')
    test_path = join_path(aligned_path, 'test')
    if 'aligned' not in listdir(data_dir):
        mkdir(aligned_path)
        mkdir(train_path)
        mkdir(val_path)
        mkdir(test_path)

    for lang_key, lang_corpus in aligned_corpus.items():
        train_file = open(
            join_path(train_path, "{}.txt".format(lang_key)), "w")
        train_file.write("\n".join(lang_corpus[:-(val_test_size * 2)]))
        train_file.close()
        val_file = open(join_path(val_path, "{}.txt".format(lang_key)), "w")
        val_file.write(
            "\n".join(lang_corpus[-(val_test_size * 2):-val_test_size]))
        val_file.close()
        test_file = open(join_path(test_path, "{}.txt".format(lang_key)), "w")
        test_file.write("\n".join(lang_corpus[-val_test_size:]))
        test_file.close()

    print("New dataset created!")


if __name__ == "__main__":
    data_dir = "./data/"
    sentence_num = 510000
    process_num = 6
    log_rate = 1000

    corpus = align_corpus(load_corpus(data_dir),
                          sentence_num, process_num, log_rate)
    write_aligned_corpus(corpus, data_dir, 5000)
