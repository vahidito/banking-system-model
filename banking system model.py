import numpy as np
import mesa as mesa
import pandas as pd
import matplotlib as plt
import scipy as spy
from scipy.stats import norm
from scipy.optimize import linprog

##################################################################################

### global variables

ret_sec_bank = 0.3
ret_sec_shbank = 0.35
rfree = 0.1


########################################################################
# defining the agents: banks, shadow banks, savers, loans


class Bank:
    def __init__(self, bank_cash, lend_to_banks, lend_to_loans, bank_sec, deposits, borrow_from_banks, equity, CAR,
                 provision_per, phi, zeta):
        # balance sheet
        self.bank_cash = bank_cash
        self.lend_to_banks = lend_to_banks
        self.lend_to_loans = lend_to_loans
        self.bank_sec = bank_sec
        self.deposits = deposits
        self.borrow_from_banks = borrow_from_banks
        self.equity = equity
        self.CAR = CAR
        self.provision_per = provision_per
        ##### income and expense of bank
        pd = 0.1
        self.net_income = (ret_sec_bank * self.bank_sec) + (rfree * self.lend_to_banks) - (
                    (rfree * self.borrow_from_banks) / (1 - zeta * pd))
        self.sigma = phi * (deposits + equity)
        self.profit = float(np.random.normal(self.net_income, self.sigma, 1))
        self.pd = norm.cdf((-self.net_income - self.equity) / (self.sigma))


class Shadow_Bank:
    def __init__(self, participation, shadow_bank_cash):
        self.participation = participation
        self.shadow_bank_cash = shadow_bank_cash
        self.security = participation - shadow_bank_cash
        ##### income and expense of shadow bank


###############################################################
# introduction of Iranian Banks

bank_melli = Bank(500, 1000, 1000, 200, 900, 1500, 00, 0.1, 0.1, 0.1, 0.1)
# bank_seppah = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_tosesaderat = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_maskan = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_sanatmadan = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_keshavarzi = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_tosetavon = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_post = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_eghtesadnovin = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_parsian = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_karafarin = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_saman = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_sina = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_khavarmiane = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_shahr = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_dey = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_saderat = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_tejarat = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_mellat = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_refah = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_ayandeh = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_gardeshgary = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_iranzamin = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_sarmaye = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_pasargad = Bank(0,0,0,0,0,0,0,0,0,0,0)
# bank_melal = Bank(0,0,0,0,0,0,0,0,0,0,0)


###############################################################
# introduction of Iranian Shadow Banks


shadow1 = Shadow_Bank(np.random.normal(100), np.random.normal(20))




#####################################################
# 1-BL 2-S 3-BB 4-L 5-C 6-D 7-E
# optimixation phase
c = np.array([rfree, ret_sec_bank,(rfree * self.borrow_from_banks) / (1 - zeta * pd)) , 0.0])
A_ub = np.array([[1.0, -1.0, -3.0, 0.0], [-2.0, 3.0, 7.0, -3.0]])
b_ub = np.array([5.0, -10.0])
A_eq = np.array([[2.0, 8.0, 1.0, 0.0], [4.0, 4.0, 0.0, 1.0]])
b_eq = np.array([60.0, 60.0])
x0_bounds = (0, None)
x1_bounds = (0, 5.0)
x2_bounds = (-np.inf, 0.5)  # +/- np.inf can be used instead of None
x3_bounds = (-3.0, None)
bounds = [x0_bounds, x1_bounds, x2_bounds, x3_bounds]
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
print(result.x)





########################################################
# simulations


#######################################################
print(bank_melli.net_income)
print(bank_melli.profit)
print(bank_melli.pd)
print(shadow1.participation)
print(shadow1.shadow_bank_cash)
