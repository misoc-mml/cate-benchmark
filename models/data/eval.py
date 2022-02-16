import numpy as np
from sklearn.metrics import roc_auc_score

def rmse(a, b):
    """"
    Root mean squared error between two arrays
    """
    return np.sqrt(((a - b)**2).mean())

class Evaluator(object):
    """
    Class that provides some functionality to evaluate the results.

    Param:
    ------

    y :     array-like, shape=(num_samples)
            Observed outcome

    t :     array-like, shape=(num_samples)
            Binary array representing the presence (t[i] == 1) or absence (t[i]==0) of treatment

    y_cf :  array-like, shape=(num_samples) or None, optional
            Counterfactual outcome (i.e., what would the outcome be if !t

    mu0 :   array-like, shape=(num_samples) or None, optional
            Outcome if no treatment and in absence of noise

    mu1 :   array-like, shape=(num_samples) or None, optional
            Outcome if treatment and in absence of noise

    """
    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.mu0 = mu0
        self.mu1 = mu1
        if mu0 is not None and mu1 is not None:
            self.true_ite = mu1 - mu0

    def rmse_ite(self, ypred1, ypred0):
        """"
        Root mean squared error of the Individual Treatment Effect (ITE)

        :param ypred1: prediction for treatment case
        :param ypred0: prediction for control case

        :return: the RMSE of the ITE

        """
        pred_ite = np.zeros_like(self.true_ite)
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return rmse(self.true_ite, pred_ite)

    def abs_ate(self, ypred1, ypred0):
        """
        Absolute error for the Average Treatment Effect (ATE)
        :param ypred1: prediction for treatment case
        :param ypred0: prediction for control case
        :return: absolute ATE
        """
        return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite))

    def pehe(self, ypred1, ypred0):
        """
        Precision in Estimating the Heterogeneous Treatment Effect (PEHE)

        :param ypred1: prediction for treatment case
        :param ypred0: prediction for control case

        :return: PEHE
        """
        return rmse(self.mu1 - self.mu0, ypred1 - ypred0)

    def calc_stats(self, ypred1, ypred0):
        """
        Calculate some metrics

        :param ypred1: predicted outcome if treated
        :param ypred0: predicted outcome if not treated

        :return ite: RMSE of ITE
        :return ate: absolute error for the ATE
        :return pehe: PEHE
        """
        ite = self.rmse_ite(ypred1, ypred0)
        ate = self.abs_ate(ypred1, ypred0)
        pehe = self.pehe(ypred1, ypred0)
        return ite, ate, pehe

    def calc_stats_effect(self, pred_ite):
        ite = pehe = rmse(self.true_ite, pred_ite)
        ate = np.abs(np.mean(pred_ite) - np.mean(self.true_ite))
        return ite, ate, pehe

class EvaluatorJobs(object):
    """
    Class that provides some functionality to evaluate the results on the Jobs dataset.

    Param:
    ------

    yf :     array-like, shape=(num_samples)
             Label - Factual observation (binary)

    t :     array-like, shape=(num_samples)
            Binary array representing the presence (t[i] == 1) or absence (t[i]==0) of treatment

    e :     array-like, shape=(num_samples)
            Binary array representing whether the sample comes (e[i] == 1) from experimental data or not

    """
    def __init__(self, yf, t, e):
        self.yf = yf
        self.t = t
        self.e = e

    def calc_stats(self, yf_p, ycf_p):
        """
        Calculates the Average Treatment Effect on Treated (ATT) and the value of the policy

        :param yf_p: predicted factual outcome
        :param ycf_p: predicted counterfactual outcome

        :return: att, policy
        """
        att, policy = self.evaluate_bin_att(yf_p, ycf_p)
        return att, policy

    def calc_r_pol(self, pol_p):

        # Consider only the cases for which we have experimental data (i.e., e > 0)
        policy = pol_p[self.e > 0]
        t = self.t[self.e > 0]
        yf = self.yf[self.e > 0]

        treat_overlap = (policy == t) * (t > 0)
        control_overlap = (policy == t) * (t < 1)

        if np.sum(treat_overlap) == 0:
            treat_value = 0
        else:
            treat_value = np.mean(yf[treat_overlap])

        if np.sum(control_overlap) == 0:
            control_value = 0
        else:
            control_value = np.mean(yf[control_overlap])

        pit = np.mean(policy)
        policy_value = pit * treat_value + (1 - pit) * control_value

        return 1 - policy_value

    def calc_r_pol_all(self, pol_p):

        # Consider all units (both experimental and non-experimental)
        policy = pol_p
        t = self.t
        yf = self.yf

        treat_overlap = (policy == t) * (t > 0)
        control_overlap = (policy == t) * (t < 1)

        if np.sum(treat_overlap) == 0:
            treat_value = 0.0
        else:
            treat_value = np.mean(yf[treat_overlap], dtype=np.float32)

        if np.sum(control_overlap) == 0:
            control_value = 0.0
        else:
            control_value = np.mean(yf[control_overlap], dtype=np.float32)

        pit = np.mean(policy, dtype=np.float32)
        policy_value = pit * treat_value + (1.0 - pit) * control_value

        return 1.0 - policy_value

    def calc_stats_effect(self, eff_pred):
        att, policy = self.evaluate_bin_att_effect(eff_pred)
        return att, policy

    def evaluate_bin_att_effect(self, eff_pred):
        att = np.mean(self.yf[self.t > 0]) - np.mean(self.yf[(1 - self.t + self.e) > 1])

        att_pred = np.mean(eff_pred[(self.t + self.e) > 1])
        bias_att = att_pred - att

        policy_value = self.policy_val(eff_pred[self.e > 0])

        return np.abs(bias_att), 1 - policy_value

    def policy_val(self, eff_pred):
        """
        Computes the value of the policy defined by predicted effect

        :param eff_pred: predicted effect (for the experimental data only)

        :return: policy value

        """
        # Consider only the cases for which we have experimental data (i.e., e > 0)
        t = self.t[self.e > 0]
        yf = self.yf[self.e > 0]

        if np.any(np.isnan(eff_pred)):
            return np.nan, np.nan

        policy = eff_pred > 0.0
        treat_overlap = (policy == t) * (t > 0)
        control_overlap = (policy == t) * (t < 1)

        if np.sum(treat_overlap) == 0:
            treat_value = 0
        else:
            treat_value = np.mean(yf[treat_overlap])

        if np.sum(control_overlap) == 0:
            control_value = 0
        else:
            control_value = np.mean(yf[control_overlap])

        pit = np.mean(policy)
        policy_value = pit * treat_value + (1 - pit) * control_value

        return policy_value

    def evaluate_bin_att(self, yf_p, ycf_p):
        """

        :param yf_p: predicted factual outcome
        :param ycf_p: predicted counterfactual outcome

        :return:
        """
        att = np.mean(self.yf[self.t > 0]) - np.mean(self.yf[(1 - self.t + self.e) > 1])

        eff_pred = ycf_p - yf_p
        att_pred = np.mean(eff_pred[(self.t + self.e) > 1])
        bias_att = att_pred - att

        policy_value = self.policy_val(eff_pred[self.e > 0])

        return np.abs(bias_att), 1 - policy_value

class EvaluatorTwins(object):
    """
        Class that provides some functionality to evaluate the results on the Twins dataset.

        Param:
        ------

        y :     array-like, shape=(num_samples)
                Label - Factual observation (binary)

        t :     array-like, shape=(num_samples)
                Binary array representing the presence (t[i] == 1) or absence (t[i]==0) of treatment

        y_cf :   array-like, shape=(num_samples)
                 Binary array holding the counterfactual outcome

        """
    def __init__(self, y, t, y_cf=None):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.mu0 = self.y * (1 - self.t) + self.y_cf * self.t
        self.mu1 = self.y * self.t + self.y_cf * (1 - self.t)
        self.true_ite = self.mu1 - self.mu0

    def abs_ate(self, ypred1, ypred0):
        """
        Absolute error for the Average Treatment Effect (ATE)
        :param ypred1: prediction for treatment case
        :param ypred0: prediction for control case
        :return: absolute ATE
        """
        return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite))

    def pehe(self, ypred1, ypred0):
        """
        Precision in Estimating the Heterogeneous Treatment Effect (PEHE)

        :param ypred1: prediction for treatment case
        :param ypred0: prediction for control case

        :return: PEHE
        """
        return rmse(self.true_ite, ypred1 - ypred0)

    def calc_stats(self, ypred1, ypred0):
        """
        Calculate some metrics

        :param ypred1: predicted outcome if treated
        :param ypred0: predicted outcome if not treated

        :return ate, pehe, auc, cf_auc
        """
        ate = self.abs_ate(ypred1, ypred0)
        pehe = self.pehe(ypred1, ypred0)

        # Combined AUC (as in Yao et al.)
        # https://github.com/Osier-Yi/SITE/blob/master/simi_ite/evaluation.py
        y_label = np.concatenate((self.mu0, self.mu1), axis=0)
        y_label_pred = np.concatenate((ypred0, ypred1), axis=0)
        auc = roc_auc_score(y_label, y_label_pred)

        # Counterfactual AUC (as in Louizos et al.)
        y_cf = (1 - self.t) * ypred1 + self.t * ypred0
        cf_auc = roc_auc_score(self.y_cf, y_cf)  

        return ate, pehe, auc, cf_auc

    def calc_stats_effect(self, eff_pred):
        ate = np.abs(np.mean(eff_pred) - np.mean(self.true_ite))
        pehe = rmse(self.true_ite, eff_pred)
        return ate, pehe, 0.0, 0.0