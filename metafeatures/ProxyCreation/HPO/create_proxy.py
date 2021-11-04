import numpy as np

def read_entropy_file(filename):
  """
  Read an input file, which includes target data information
  - information: (index, entropy, label)
  """
  index = []
  entropy = []
  label = []
  with open(filename, 'r') as f:
    for line in f.readlines():
      tokens = line.split()
      index.append(int(tokens[0]))
      entropy.append(float(tokens[1]))
      label.append(int(tokens[2]))

  entropy = np.array(entropy)
  return index, entropy, label

def subsampling(file_path, sampling_portion):
    index, entropy, label = read_entropy_file(file_path)
    log_entropy = np.log10(entropy)
    min_log_entropy, max_log_entropy = np.min(log_entropy), np.max(log_entropy)

    bin_width = 0.5
    low_bin = np.round(min_log_entropy)
    while min_log_entropy < low_bin:
        low_bin -= bin_width
    high_bin = np.round(max_log_entropy)
    while max_log_entropy > high_bin:
        high_bin += bin_width
    print(low_bin, high_bin)
    bins = np.arange(low_bin, high_bin+bin_width, bin_width)

    def get_bin_idx(ent):
        for i in range(len(bins)-1):
            if (bins[i] <= ent) and (ent < bins[i+1]):
                return i
        return None

    index_histogram = [] 
    for i in range(len(bins)-1):
        index_histogram.append([])

    for index, e in enumerate(log_entropy):
        bin_idx = get_bin_idx(e)
        if bin_idx is None:
            raise ValueError("[Error] histogram bin settings is wrong ... histogram bins: [%f ~ %f], current: %f"%(low_bin, high_bin, e))
        index_histogram[bin_idx].append(index)

    histo = np.array([len(l) for l in index_histogram])
    inv_histo = (max(histo) - histo + 1) * (histo != 0)
    inv_histo_prob = inv_histo / np.sum(inv_histo)
    num_proxy_data = int(np.floor(sampling_portion * len(entropy)))

    indices = []
    total_indices = []
    total_prob = []
    for index_bin, prob in zip(index_histogram, inv_histo_prob):
        if len(index_histogram) == 0:
            continue
        total_indices += index_bin
        temp = np.array([prob for _ in range(len(index_bin))])
        temp = temp/len(index_bin)
        total_prob += temp.tolist()
    total_prob = total_prob / np.sum(total_prob)
    indices = np.random.choice(total_indices, size=num_proxy_data, replace=False, p=total_prob)

    selected_index_num = [0] * len(histo)
    for i in indices:
        selected_index_num[get_bin_idx(log_entropy[i])] += 1

    np.random.shuffle(indices)
    return indices
