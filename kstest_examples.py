from scipy import stats
from math import *
import numpy as np
from matplotlib import pyplot as plt


def get_number_position(x, n=7):
    return int(floor(log10(abs(x)))) - (n - 1)


def normcdf(x, loc, scale):
    f = lambda a: (1.0 + erf((a - loc) / (scale * sqrt(2.0)))) / 2.0
    return np.array([f(a) for a in x])


def normcdf_nn(x, loc, scale):
    f = lambda a: (1.0 + erf((a - loc) / (scale * sqrt(2.0)))) / 2.0 if a >= 0 else 0
    # f = lambda a: 0 if a < 0 else normcdf(a, loc, scale)
    return np.array([f(a) for a in x])


def nparray_equal(a1, a2):
    if len(a1) != len(a2):
        return False

    for i in range(len(a1)):
        x1 = a1[i]
        x2 = a2[i]
        last_significant_number = get_number_position(x1, 7)
        tolerance = 0.1**-last_significant_number
        if fabs(x1 - x2) > tolerance:
            print(f"BAD: {x1} is not equal to {x2}, difference is {fabs(x1 - x2)},tolerance is {tolerance}.")
            return False

    return True


def calculate_cdf(data):
    data_size = len(data)

    # Set bins edges
    data_set = sorted(set(data))
    bins = np.append(data_set, data_set[-1] + 1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts = counts.astype(float) / data_size

    # Find the cdf
    cdf = np.cumsum(counts)

    return cdf, bin_edges


def plot_cdf(data, name="cdf", cdf_function=None, args=()):

    cdf, bin_edges = calculate_cdf(data)
    data_set = sorted(set(data))

    # Plot the cdf
    fig=plt.figure()
    ax= fig.add_axes([0.12,0.12,0.76,0.76])
    plt.plot(bin_edges[:-1], cdf,linestyle='--', marker="o", color='b', alpha=0.3, label="calculated with data bin",
             markersize=3)
    plt.ylim((-0.1, 1.1))
    plt.ylabel("CDF")
    plt.grid(True)

    if cdf_function is not None:
        cdf_therocal = cdf_function(data_set, *args)
        plt.scatter(data_set, cdf_therocal, color='r', alpha=0.3,
                    label=f"calculated used {cdf_function.__name__} function", s=20)
    ax.set_title("cdf")
    ax.legend(loc=0)

    plt.show()
    fig.savefig(f"{name}.png")


if __name__ == "__main__":

    print("test get_number_position function:")
    test_list = [1234568934603846, 123.34456, 0.123445677, 0.002432596895]
    print(f"list to test is {test_list}")
    print("get_number_position function returns:")
    for n in test_list:
        print(f"\t {get_number_position(n)}")
    print("\n")

    print("test nparray_equal function:")
    test_list = [123456893460384, 123.34456412342, 0.123445675345, 0.00243259685253]
    test_list_2 = [123456893123243, 123.344564234, 0.123445676516, 0.0024325965235]
    result = nparray_equal(test_list, test_list_2)
    print(result)
    test_list_3 = [123456893123244, 123.344564234, 0.123446676546, 0.0024325965235]
    result2 = nparray_equal(test_list, test_list_3)
    print(result2)
    if result and not result2:
        print("GOOD: nparray_equal is good.")
    else:
        print("BAD: nparray_equal is bad.")

    print("\n")

    print("testing self defined function normcdf:")
    loc = 1
    scale = 3
    size = 1000
    # dist_to_test_stats = stats.norm.rvs(loc, scale, size=size, random_state=np.random.RandomState(seed=123456789))
    dist_to_test_stats = stats.norm.rvs(loc, scale, size=size)

    stats_norm_cdf = lambda x: stats.norm.cdf(x, loc=loc, scale=scale)
    args=()
    cdfvals1 = stats_norm_cdf(dist_to_test_stats, *args)
    cdfvals2 = normcdf(dist_to_test_stats, loc, scale, *args)
    if not nparray_equal(cdfvals1, cdfvals2):
        print(f"BAD: cdfvals1 is {cdfvals1} and cdfvals2 is {cdfvals2}.")
    else:
        print("GOOD: cdfvals1 is equal to cdfvals2.")
    print("\n")

    print("kstest result using string name 'norm' for cdf:")
    print(stats.kstest(dist_to_test_stats, 'norm', args=(loc, scale)))
    print("\n")

    print("kstest result using cdf function in scipy.norm:")
    print(stats.kstest(dist_to_test_stats, stats_norm_cdf)) #lambda x: stats.norm.cdf(x, loc=loc, scale=scale)))
    print("\n")

    print("kstest result using self defined cdf function:")
    print(stats.kstest(dist_to_test_stats, lambda x: normcdf(x, loc=loc, scale=scale)))
    print("\n")

    dist_to_test_stats_lognorm = stats.lognorm.rvs(loc, 0, scale, size=size)
    print("kstest result using string name 'lognorm' for cdf:")
    print(stats.kstest(dist_to_test_stats_lognorm, 'lognorm', args=(loc, 0, scale)))
    print("\n")

    print("kstest result using lognorm cdf function in scipy.norm:")
    print(stats.kstest(dist_to_test_stats_lognorm, lambda x: stats.lognorm.cdf(x, loc, 0, scale)))
    print("\n")

    dist_to_test_stats_expon = stats.expon.rvs(loc, scale, size=size)
    print("kstest result using string name 'expon' for cdf:")
    print(stats.kstest(dist_to_test_stats_expon, 'expon', args=(loc, scale)))
    print("\n")

    print("The KS test is only valid for continuous distributions.")
    dist_to_test_stats_poisson = stats.poisson.rvs(scale, 0, size=size)
    print("kstest result using string name 'poisson' for cdf:")
    print(stats.kstest(dist_to_test_stats_poisson, 'poisson', args=(scale, 0)))
    print("\n")


    dist_to_test_np = np.random.normal(loc, scale, size=size)
    #
    # print("kstest result using string name 'norm' for cdf:")
    # print(stats.kstest(dist_to_test_np, 'norm', args=(loc, scale)))
    # print("\n")
    #
    # print("kstest result using cdf function in scipy.norm:")
    # print(stats.kstest(dist_to_test_np, stats_norm_cdf)) #lambda x: stats.norm.cdf(x, loc=loc, scale=scale)))
    # print("\n")
    #
    # print("kstest result using self defined cdf function:")
    # print(stats.kstest(dist_to_test_np, lambda x: normcdf(x, loc=loc, scale=scale)))
    # print("\n")
    #
    # dist_to_test_stats_2 = stats.norm.rvs(loc, scale, size=size, random_state=np.random.RandomState(seed=987654321))
    # print("kstest 2 sample test with two fixed distribution:")
    # print(stats.ks_2samp(dist_to_test_stats, dist_to_test_stats_2))

    dist_to_test_stats_nn = [x if x > 0 else 0 for x in dist_to_test_stats]
    print("kstest result for modified distribution using string name 'norm' for cdf:")
    print(stats.kstest(dist_to_test_stats_nn, 'norm', args=(loc, scale)))
    print("\n")

    ## https: // www.itl.nist.gov / div898 / handbook / eda / section3 / eda35g.htm
    print("kstest result for modified distribution using modifled cdf function:")
    print(stats.kstest(dist_to_test_stats_nn, lambda x: normcdf_nn(x, loc=loc, scale=scale)))
    print("\n")

    # print("kstest result(one side test: greater ) for modified distribution using modifled cdf function:")
    # print(stats.kstest(dist_to_test_stats_nn, lambda x: normcdf_nn(x, loc=loc, scale=scale), alternative='greater'))
    # print("\n")

    fig = plt.figure()
    ax= fig.add_axes([0.12,0.12,0.76,0.76])
    plt.scatter(sorted(dist_to_test_stats), normcdf(sorted(dist_to_test_stats), loc, scale), alpha=0.3,
                label="normal")
    plt.scatter(sorted(dist_to_test_stats_nn), normcdf_nn(sorted(dist_to_test_stats_nn), loc, scale), alpha=0.3,
                label="normal(non negative)", s=10)
    ax.set_title("cdf")
    ax.legend(loc=0)
    plt.show()
    fig.savefig('cdf.png')

    plot_cdf(dist_to_test_stats, "normal_with_normcdf_cdf", normcdf, (loc, scale))
    plot_cdf(dist_to_test_stats_nn, "normal_nn_with_normcdf_nn_cdf", normcdf_nn, (loc, scale))
    plot_cdf(dist_to_test_stats, "norm_with_normcdf_nn_cdf", normcdf_nn, (loc, scale))


    # https: // www.itl.nist.gov / div898 / handbook / eda / section3 / eda35e.htm
    # it has the advantage of allowing a more sensitive test and the disadvantage that
    # critical values must be calculated for each distribution.
    print("Anderson-Darling Test result for normal distribution using 'norm' as cdf:")
    # 'norm','expon','logistic','gumbel','gumbel_l', 'gumbel_r', 'extreme1'
    print(stats.anderson(dist_to_test_stats, 'norm'))
    print("\n")


    print("Anderson-Darling Test result for modified distribution using 'norm' as cdf:")
    print(stats.anderson(dist_to_test_stats_nn, 'norm'))
    print("\n")


    # https://eric-bunch.github.io/static/Szekely_estats.pdf
    print("energy distance:")
    dist_to_test_stats_set = sorted(set(dist_to_test_stats))
    dist_to_test_stats_nn_set = sorted(set(dist_to_test_stats_nn))

    dist_to_test_stats_cdf = calculate_cdf(dist_to_test_stats)[0]
    dist_to_test_stats_nn_cdf = calculate_cdf(dist_to_test_stats_nn)[0]

    print("calculate_cdf(norm data) vs. stats.norm.cdf(norm data):")
    print(stats.energy_distance(dist_to_test_stats_cdf, stats.norm.cdf(dist_to_test_stats_set, loc, scale)))
    print("calculate_cdf(norm data) vs. normcdf(norm data):")
    print(stats.energy_distance(dist_to_test_stats_cdf, normcdf(dist_to_test_stats_set, loc, scale)))
    print("calculate_cdf(norm_nn data) vs. normcdf_nn(norm_nn data):")
    print(stats.energy_distance(dist_to_test_stats_nn_cdf, normcdf_nn(dist_to_test_stats_nn_set, loc, scale)))
    print("calculate_cdf(norm data) vs. normcdf_nn(norm data):")
    print(stats.energy_distance(dist_to_test_stats_cdf, normcdf_nn(dist_to_test_stats_set, loc, scale)))

    # print("calculate_cdf(norm data) vs. calculate_cdf(norm_nn data):")
    # print(stats.energy_distance(dist_to_test_stats_cdf, dist_to_test_stats_nn_cdf))
    #
    # print("calculate_cdf(stats data) vs. calculate_cdf(np data):")
    # print(stats.energy_distance(calculate_cdf(dist_to_test_stats)[0], calculate_cdf(dist_to_test_np)[0]))

    #
    # print("calculate_cdf(stats data) vs. calculate_cdf(stats data 2):")
    # dist_to_test_stats_2 = stats.norm.rvs(loc, scale, size=size, random_state=np.random.RandomState(seed=987654321))
    # dist_to_test_stats_2_cdf = calculate_cdf(dist_to_test_stats_2)[0]
    # print(stats.energy_distance(dist_to_test_stats_cdf, dist_to_test_stats_2_cdf))