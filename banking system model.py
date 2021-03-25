import numpy as np
import mesa as mesa
import pandas as pd
import matplotlib as plt
import scipy as spy
from scipy.stats import norm
from scipy.optimize import linprog

##################################################################################

### global variables

ret_sec_bank = 0.1
ret_sec_bank_sigma = 0.02
ret_sec_shbank = 0.2
rfree = 0.1
rfree_min = 0.05
rfree_max = 0.4
rfree_vector = [f'{rfree}']


########################################################################
# defining the agents: banks, shadow banks, savers, loans


class Bank:
    def __init__(self, bank_cash, lend_to_banks, lend_to_loans, bank_sec, deposits, borrow_from_banks, equity,
                 alpha_min,
                 provision_per, phi, zeta, car, xs, xbl, xl):
        # balance sheet
        self.bank_cash = bank_cash
        self.lend_to_banks = lend_to_banks
        self.lend_to_loans = lend_to_loans
        self.bank_sec = bank_sec
        self.deposits = deposits
        self.borrow_from_banks = borrow_from_banks
        self.equity = equity
        self.alpha_min = alpha_min
        self.provision_per = provision_per
        self.zeta = zeta
        self.total_assets = bank_cash + lend_to_banks + lend_to_loans + bank_sec
        self.xs = xs
        self.xbl = xbl
        self.xl = xl
        self.car = car
        self.ret_on_sec = np.random.normal(ret_sec_bank, ret_sec_bank_sigma)

        ##### income and expense of bank
        pd = 0.1
        self.net_income = (ret_sec_bank * self.bank_sec) + (rfree * self.lend_to_banks) - (
                (rfree * self.borrow_from_banks) / (1 - zeta * pd))
        self.sigma = phi * (deposits + equity)
        self.profit = float(np.random.normal(self.net_income, self.sigma, 1))
        #self.pd = norm.cdf((-self.net_income - self.equity) / (self.sigma))
        self.pd = np.random.beta(1,20)



class Shadow_Bank:
    def __init__(self, participation, shadow_bank_cash, s_alpha, s_provision):
        self.participation = participation
        self.shadow_bank_cash = shadow_bank_cash
        self.security = participation - shadow_bank_cash
        self.s_alpha = s_alpha
        self.s_provision = s_provision
        ##### income and expense of shadow bank


###############################################################
# introduction of Iranian Banks

bank_melli = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.4, 0.07, 0.05, 0.05, 0.05)
bank_seppah = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.4, 0.07, 0.05, 0.05, 0.05)
bank_tosesaderat = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_maskan = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_sanatmadan = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_keshavarzi = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_tosetavon = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_post = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_eghtesadnovin = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_parsian = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_karafarin = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_saman = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_sina = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_khavarmiane = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_shahr = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_dey = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_saderat = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_tejarat = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_mellat = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_refah = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_ayandeh = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_gardeshgary = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_iranzamin = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_sarmaye = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_pasargad = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)
bank_melal = Bank(500, 1000, 1000, 200, 700, 1500, 500, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05)

###############################################################
# introduction of Iranian Shadow Banks


shadow1 = Shadow_Bank(np.random.normal(100), np.random.normal(20), 0.1, 0.1)

#####################################################
# 1-BL 2-S 3-BB 4-L 5-C 6-D 7-E
# optimixation phase
### objective function of banks

c = np.array(
    [-rfree, -bank_melli.ret_on_sec, ((rfree * bank_melli.borrow_from_banks) / (1 - bank_melli.zeta * bank_melli.pd)), 0, 0, 0,
     0])

A_ub = np.array([[(-1 + bank_melli.car * bank_melli.xbl), (-1 + bank_melli.car * bank_melli.xs), 1,
                  (-1 + bank_melli.car * bank_melli.xl), -1, 1, 0],
                 [-1, 0, (bank_melli.alpha_min + bank_melli.provision_per), 0, -1,
                  (bank_melli.alpha_min + bank_melli.provision_per), 0],
                 [0, 0, 0, 0, -1, (bank_melli.alpha_min + bank_melli.provision_per), 0]])
b_ub = np.array([0, 0, 0])

#### the first bound is structural balance sheet equation
#### the second one is based on this assumtion that  half of the bank’s assets is invested in loans

A_eq = np.array([[1, 1, -1, 1, 1, -1, -1], [0, 0, -1, 2, 0, -1, -1], [1, 1, 0, 1, 1, 0, 0]])
b_eq = np.array([0, 0, bank_melli.total_assets])

x0_bounds = (0, None)
x1_bounds = (0, None)
x2_bounds = (0, None)
x3_bounds = (0, None)
x4_bounds = (0, None)
x5_bounds = (0, None)
x6_bounds = (0, None)

bounds = [x0_bounds, x1_bounds, x2_bounds, x3_bounds, x4_bounds, x5_bounds, x6_bounds]
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

aaa = [result.x[2], result.x[5], result.x[6]]
bbb = [result.x[0], result.x[1], result.x[3], result.x[4]]

########################################################
# optimixation phase
### objective function of shadow banks
# 1- security 2-cash 3-participation

c_s = np.array([-ret_sec_shbank, 0, 0])

A_ub_s = np.array([[0, -1, (shadow1.s_alpha + shadow1.s_provision)]])
b_ub_s = np.array([0])

#### the first bound is structural balance sheet equation

A_eq_s = np.array([[1, 1, -1], [0, 0, 1]])
b_eq_s = np.array([0, shadow1.participation])

x0_bounds_s = (0, None)
x1_bounds_s = (0, None)
x2_bounds_s = (0, None)

bounds_s = [x0_bounds_s, x1_bounds_s, x2_bounds_s]
result_s = linprog(c_s, A_ub=A_ub_s, b_ub=b_ub_s, A_eq=A_eq_s, b_eq=b_eq_s, bounds=bounds_s)
print(result_s.x)

#############################################################

# Equilibrium in interBank market

demand_of_banks = bank_melli.borrow_from_banks + bank_seppah.borrow_from_banks + bank_tosesaderat.borrow_from_banks + bank_maskan.borrow_from_banks + bank_sanatmadan.borrow_from_banks + bank_keshavarzi.borrow_from_banks + bank_tosetavon.borrow_from_banks + bank_post.borrow_from_banks + bank_eghtesadnovin.borrow_from_banks +bank_parsian.borrow_from_banks + bank_karafarin.borrow_from_banks + bank_saman.borrow_from_banks + bank_saman.borrow_from_banks + bank_sina.borrow_from_banks + bank_khavarmiane.borrow_from_banks + bank_shahr.borrow_from_banks + bank_dey.borrow_from_banks + bank_saderat.borrow_from_banks + bank_tejarat.borrow_from_banks + bank_mellat.borrow_from_banks + bank_refah.borrow_from_banks + bank_ayandeh.borrow_from_banks + bank_gardeshgary.borrow_from_banks + bank_iranzamin.borrow_from_banks + bank_sarmaye.borrow_from_banks + bank_sarmaye.borrow_from_banks + bank_pasargad.borrow_from_banks +bank_melal.borrow_from_banks
supply_of_banks = bank_melli.lend_to_banks + bank_seppah.lend_to_banks + bank_tosesaderat.lend_to_banks + bank_maskan.lend_to_banks + bank_sanatmadan.lend_to_banks + bank_keshavarzi.lend_to_banks + bank_tosetavon.lend_to_banks + bank_post.lend_to_banks + bank_eghtesadnovin.lend_to_banks +bank_parsian.lend_to_banks + bank_karafarin.lend_to_banks + bank_saman.lend_to_banks + bank_saman.lend_to_banks + bank_sina.lend_to_banks + bank_khavarmiane.lend_to_banks + bank_shahr.lend_to_banks + bank_dey.lend_to_banks + bank_saderat.lend_to_banks + bank_tejarat.lend_to_banks + bank_mellat.lend_to_banks + bank_refah.lend_to_banks + bank_ayandeh.lend_to_banks + bank_gardeshgary.lend_to_banks + bank_iranzamin.lend_to_banks + bank_sarmaye.lend_to_banks + bank_sarmaye.lend_to_banks + bank_pasargad.lend_to_banks +bank_melal.lend_to_banks

print(demand_of_banks)
print(supply_of_banks)

if demand_of_banks > supply_of_banks :
    rfree = (rfree + rfree_max)/2
elif demand_of_banks < supply_of_banks :
    rfree = (rfree + rfree_min) / 2
else:
    rfree = rfree

rfree_vector.append(f'{rfree}')
print(rfree_vector)

