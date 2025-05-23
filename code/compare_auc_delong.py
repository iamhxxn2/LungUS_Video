import pandas as pd
import numpy as np
import scipy.stats
from scipy import stats


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
      x - a 1D numpy array
    Returns:
      array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
#     T = np.zeros(N, dtype=np.float)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
#     T2 = np.empty(N, dtype=np.float)
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
      x - a 1D numpy array
    Returns:
      array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
#     T = np.zeros(N, dtype=np.float)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
#     T2 = np.empty(N, dtype=np.float)
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2



def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight=None):

    if sample_weight is None:

        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)

    else:

        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)





def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):

    """

    The fast version of DeLong's method for computing the covariance of

    unadjusted AUC.

    Args:

      predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]

         sorted such as the examples with label "1" are first

    Returns:

      (AUC value, DeLong covariance)

    Reference:

    @article{sun2014fast,

      title={Fast Implementation of DeLong's Algorithm for

             Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},

      author={Xu Sun and Weichao Xu},

      journal={IEEE Signal Processing Letters},

      volume={21},

      number={11},

      pages={1389--1393},

      year={2014},

      publisher={IEEE}

    }

    """

    # Short variables are named as they are in the paper

    m = label_1_count

    n = predictions_sorted_transposed.shape[1] - m

    positive_examples = predictions_sorted_transposed[:, :m]

    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]



#     tx = np.empty([k, m], dtype=np.float)
#     ty = np.empty([k, n], dtype=np.float)
#     tz = np.empty([k, m + n], dtype=np.float)

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):

        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])

        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])

        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)

    total_positive_weights = sample_weight[:m].sum()

    total_negative_weights = sample_weight[m:].sum()

    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])

    total_pair_weights = pair_weights.sum()

    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights

    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights

    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights

    sx = np.cov(v01)

    sy = np.cov(v10)

    delongcov = sx / m + sy / n

    return aucs, delongcov





def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):

    """

    The fast version of DeLong's method for computing the covariance of

    unadjusted AUC.

    Args:

      predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]

         sorted such as the examples with label "1" are first

    Returns:

      (AUC value, DeLong covariance)

    Reference:

    @article{sun2014fast,

      title={Fast Implementation of DeLong's Algorithm for

             Comparing the Areas Under Correlated Receiver Oerating

             Characteristic Curves},

      author={Xu Sun and Weichao Xu},

      journal={IEEE Signal Processing Letters},

      volume={21},

      number={11},

      pages={1389--1393},

      year={2014},

      publisher={IEEE}

    }

    """

    # Short variables are named as they are in the paper

    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]



#     tx = np.empty([k, m], dtype=np.float)
#     ty = np.empty([k, n], dtype=np.float)
#     tz = np.empty([k, m + n], dtype=np.float)
    
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):

        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
      aucs: 1D array of AUCs
      sigma: AUC DeLong covariances
    Returns:
      log10(pvalue)

    """

    l = np.array([[1, -1]])

    z = np.abs(np.diff(aucs)) / (np.sqrt(np.dot(np.dot(l, sigma), l.T)) + 1e-8)
    pvalue = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))
    #  print(10**(np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)))
    return pvalue

def compute_ground_truth_statistics(ground_truth, sample_weight=None):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
      ground_truth: np.array of 0 and 1
      predictions: np.array of floats of the probability of being class 1
    """
    sample_weight = None
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
       ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)

    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
      ground_truth: np.array of 0 and 1
      predictions_one: predictions of the first model,
         np.array of floats of the probability of being class 1
      predictions_two: predictions of the second model,
         np.array of floats of the probability of being class 1
    """
    sample_weight = None
    order, label_1_count,ordered_sample_weight = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count,sample_weight)

    return calc_pvalue(aucs, delongcov)

# def delong_roc_ci(y_true,y_pred):
#     aucs, auc_cov = delong_roc_variance(y_true, y_pred)
#     auc_std = np.sqrt(auc_cov)
#     lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
#     ci = stats.norm.ppf(
#        lower_upper_q,
#        loc=aucs,
#        scale=auc_std)
#     ci[ci > 1] = 1
#     return aucs,ci



def delong_roc_ci(y_true,y_pred):
    aucs, auc_cov = delong_roc_variance(y_true, y_pred)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(
       lower_upper_q,
       loc=aucs,
       scale=auc_std)
    ci[ci > 1] = 1
    return aucs,ci


def get_95CI(y_true,y_pred_1):
    """
    Return the 95% CI and AUC of prediction
    Args:
      labels: array (n,) the ground truth
      scores1: array (n,) the predicted probability
    """
    alpha = .95
    auc_1, auc_cov_1 = delong_roc_variance(y_true, y_pred_1)

    auc_std = np.sqrt(auc_cov_1)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    # 95% CI
    ci = stats.norm.ppf(
       lower_upper_q,
       loc=auc_1,
       scale=auc_std)
    ci[ci > 1] = 1
    return ci,auc_1

# threshold
from sklearn import metrics
def get_optimal_threshold(labels,y_pred_1):
    '''
    get the threshold according to youden index
    
    Args:
    labels:<numpy.ndarray> (n,) groundtruth
    y_pred_1:<numpy.ndarray> (n,) predicted probabilities
    '''
    fpr, tpr, thresholds = metrics.roc_curve(labels,y_pred_1)
    optimal_index = np.argmax(+tpr-fpr)
    optimal_thresholds = thresholds[optimal_index]
#     print(thresholds)
#     print(optimal_index)
#     print(fpr)
    return optimal_thresholds
def get_metric(y_true, y_prob,threshold,verbose = True):
    '''
    Return the commanly used metric value according to the given threshold
    
    Args:
    y_true:<numpy.ndarray> (n,) groundtruth
    y_prob:<numpy.ndarray> (n,) predicted probabilities
    threshold: <float> 
    
    Return:
    scores: <dict> commanly used metric
    '''
    scores = {}
    y_pred = (y_prob>=(threshold-1E-4)).astype(int)
    # print report
    target_names = ['class 0', 'class 1']
    text = metrics.classification_report(y_true, y_pred, target_names=target_names)
    conf_mat=pd.crosstab(y_true, y_pred,rownames=['label'],colnames=['pre'])
    if verbose:
        print(conf_mat)
        print(text)

    # accuracy
    scores['accuracy'] = metrics.accuracy_score(y_true, y_pred)

    # precision
    try:
        scores['PPV'] = metrics.precision_score(y_true, y_pred)
    except:
        scores['PPV'] = None
    #NPV
    try:
        scores['NPV'] = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1])
    except:
        scores['NPV'] = None
    # recall
    scores['recall (sensitivity)'] = metrics.recall_score(y_true, y_pred)

    scores['recall_neg (specificity)'] = metrics.recall_score(y_true==0, y_pred==0)

    # F1-score
    scores['f1_score'] = metrics.f1_score(y_true, y_pred)

    # ROC/AUC
    scores['AUC95%CI'],  scores['AUC'] = get_95CI(y_true,y_pred)
    return scores

if __name__ == '__main__':
    # example
    import sklearn.datasets
    import sklearn.model_selection
    import sklearn.linear_model    
    import numpy
    import scipy.stats

    x_distr = scipy.stats.norm(0.5, 1)
    y_distr = scipy.stats.norm(-0.5, 1)
    sample_size_x = 7
    sample_size_y = 14
    n_trials = 1000
    aucs = numpy.empty(n_trials)
    variances = numpy.empty(n_trials)
    numpy.random.seed(1234235)
    labels = numpy.concatenate([numpy.ones(sample_size_x), numpy.zeros(sample_size_y)])

    scores1 = numpy.concatenate([
            x_distr.rvs(sample_size_x),
            y_distr.rvs(sample_size_y)])
    scores2 = numpy.concatenate([
            x_distr.rvs(sample_size_x),
            y_distr.rvs(sample_size_y)])

    #examples
    y_true= labels
    y_pred_1 = scores1
    y_pred_2 = scores2

    #Delong test p-value
    pvalue = delong_roc_test(y_true,y_pred_1,y_pred_2)
    print('p_value:', pvalue)
    
    # 95CI, AUC
    ci,auc = get_95CI(labels,scores1)
    print('The AUC of score1:',auc,'\tThe 95CI of score1:',ci)
    
    # find threshold with youden index
    thres = get_optimal_threshold(labels,scores1)
    score = get_metric(labels, scores1,thres,verbose = True)
    print('Threshold is:',thres)
    print('The metrics on this threshold are:\n',score)

