
import math


def sample_num_strategy(mean: int, total: int, upper_bound = 1000) -> int :
    ''' Strategy when sampling images from dataset.

    Logic:
        1. expect samples to get 10% of images of each class,

        2. expect 5%mean < samples < 10%mean,
        if less, use lower bound, if more, use upper bound.

        3. expect 10 < samples < 1000.
        Same as 2.

    Args:
        mean: mean of total images
        total: current class image count

    Returns:
        expected: numbers of samples from this class
    '''
    expected = math.ceil(total * 0.1)

    # 0.05mean <= expected  <= 0.1mean
    expected = max(expected, int(0.05 * mean))
    expected = min(expected, int(0.10 * mean))

    # 10 <= expected <= 1000
    expected = min(max(expected, 10), upper_bound)

    # expected <= total
    expected = min(expected, total)
    # print(f'total vs expected: {total}, {expected}')
    return expected