import numpy as np
import mesa as mesa
import pandas as pd
import matplotlib.pyplot as plt
import scipy as spy
from scipy.stats import norm
from scipy.optimize import linprog

##################################################################################

### global variables
n_sim = 30
ret_sec_bank = 0.18
ret_sec_bank_sigma = 0.01
ret_sec_shbank = 0.2
rfree = 0.18
rfree_min = 0.1
rfree_max = 0.25
rfree_vector = [f'{rfree}']


########################################################################
# defining the agents: banks, shadow banks, savers, loans


class Bank:
    def __init__(self, bank_cash, lend_to_banks, lend_to_loans, bank_sec, deposits, borrow_from_banks, equity,
                 alpha_min, provision_per, phi, zeta, car, xs, xbl, xl):
        # balance sheet
        bankrupt = False
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
                (rfree * self.borrow_from_banks) / (1 - self.zeta * pd))
        self.phi = phi
        self.sigma = phi * (self.deposits + self.equity)
        self.profit = float(np.random.normal(self.net_income, self.sigma, 1))
        # self.pd = float(norm.ppf((-self.net_income - self.equity) / (self.sigma)))
        self.pd = np.random.beta(1, 20)


class Shadow_Bank:
    def __init__(self, participation, shadow_bank_cash, security, s_alpha, s_provision):
        self.participation = participation
        self.shadow_bank_cash = shadow_bank_cash
        self.security = security
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


#####################################################
# 1-BL 2-S 3-BB 4-L 5-C
# optimixation phase
### objective function of banks
def optimize_bank(mmm):
    c = np.array([-rfree, -mmm.ret_on_sec, ((rfree * mmm.borrow_from_banks) / (1 - mmm.zeta * mmm.pd)), 0, 0])
    A_ub = np.array([[(-1 + mmm.car * mmm.xbl), (-1 + mmm.car * mmm.xs), 1, (-1 + mmm.car * mmm.xl), -1],
                     [-1, 0, (mmm.alpha_min + mmm.provision_per), 0, -1], [0, 0, 0, 0, -1]])
    b_ub = np.array([-mmm.deposits, -(mmm.alpha_min + mmm.provision_per) * mmm.deposits,
                     -(mmm.alpha_min + mmm.provision_per) * mmm.deposits])
    A_eq = np.array([[1, 1, -1, 1, 1], [0, 0, -1, 2, 0]])
    b_eq = np.array([mmm.deposits + mmm.equity, mmm.deposits + mmm.equity])
    x0_bounds = (0, None)
    x1_bounds = (0, None)
    x2_bounds = (0, None)
    x3_bounds = (0, None)
    x4_bounds = (0, None)
    bounds = [x0_bounds, x1_bounds, x2_bounds, x3_bounds, x4_bounds]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    mmm.bank_cash = result.x[4]
    mmm.lend_to_banks = result.x[0]
    mmm.lend_to_loans = result.x[3]
    mmm.bank_sec = result.x[1]
    mmm.borrow_from_banks = result.x[2]
    mmm.ret_on_sec = float(np.random.normal(ret_sec_bank, ret_sec_bank_sigma))
    mmm.net_income = (mmm.ret_on_sec * mmm.bank_sec) + (rfree * mmm.lend_to_banks) - (
            (rfree * mmm.borrow_from_banks) / (1 - mmm.zeta * mmm.pd))
    mmm.sigma = abs(mmm.phi * (mmm.deposits + mmm.equity))
    mmm.profit = float(np.random.normal(mmm.net_income, mmm.sigma, 1))

    mmm = Bank(result.x[4], result.x[0], result.x[3], result.x[1], result.x[2], mmm.deposits, mmm.equity,
               mmm.alpha_min, mmm.provision_per, mmm.phi, mmm.zeta, mmm.car, mmm.xs, mmm.xbl, mmm.xl)

    ########################################################
    # optimixation phase
    ### objective function of shadow banks
    # 1- security 2-cash 3-participation
    def optimize_shadow_bank(nnn):
        c_s = np.array([-ret_sec_shbank, 0, 0])
        ####  bound is structural balance sheet equation
        A_ub_s = np.array([[0, -1, (nnn.s_alpha + nnn.s_provision)]])
        b_ub_s = np.array([0])
        A_eq_s = np.array([[1, 1, -1], [0, 0, 1]])
        b_eq_s = np.array([0, nnn.participation])
        x0_bounds_s = (0, None)
        x1_bounds_s = (0, None)
        x2_bounds_s = (0, None)
        bounds_s = [x0_bounds_s, x1_bounds_s, x2_bounds_s]
        result_s = linprog(c_s, A_ub=A_ub_s, b_ub=b_ub_s, A_eq=A_eq_s, b_eq=b_eq_s, bounds=bounds_s)
        # print(result_s.x)

    ###############################################################
    # introduction of Iranian Shadow Banks

    shadow1 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.1, 0.1)

    ###############################################################


for i in range(0, 20):
    optimize_bank(bank_melli)
    optimize_bank(bank_seppah)
    optimize_bank(bank_tosesaderat)
    optimize_bank(bank_maskan)
    optimize_bank(bank_sanatmadan)
    optimize_bank(bank_keshavarzi)
    optimize_bank(bank_tosetavon)
    optimize_bank(bank_post)
    optimize_bank(bank_eghtesadnovin)
    optimize_bank(bank_parsian)
    optimize_bank(bank_karafarin)
    optimize_bank(bank_saman)
    optimize_bank(bank_sina)
    optimize_bank(bank_khavarmiane)
    optimize_bank(bank_shahr)
    optimize_bank(bank_dey)
    optimize_bank(bank_saderat)
    optimize_bank(bank_tejarat)
    optimize_bank(bank_mellat)
    optimize_bank(bank_refah)
    optimize_bank(bank_ayandeh)
    optimize_bank(bank_gardeshgary)
    optimize_bank(bank_iranzamin)
    optimize_bank(bank_sarmaye)
    optimize_bank(bank_pasargad)
    optimize_bank(bank_melal)
    ###############################################################
    #### start of simulation
    # bank_melli_equity = [f'{bank_melli.equity}']
    # for i in range(0, 1):
    #     optimize_bank(bank_melli)
    #     bank_melli_equity.append(f'{bank_melli.equity}')
    # print(bank_melli_equity)
    demand_of_banks = bank_melli.borrow_from_banks + bank_seppah.borrow_from_banks + bank_tosesaderat.borrow_from_banks + bank_maskan.borrow_from_banks + bank_sanatmadan.borrow_from_banks + bank_keshavarzi.borrow_from_banks + bank_tosetavon.borrow_from_banks + bank_post.borrow_from_banks + bank_eghtesadnovin.borrow_from_banks + bank_parsian.borrow_from_banks + bank_karafarin.borrow_from_banks + bank_saman.borrow_from_banks + bank_saman.borrow_from_banks + bank_sina.borrow_from_banks + bank_khavarmiane.borrow_from_banks + bank_shahr.borrow_from_banks + bank_dey.borrow_from_banks + bank_saderat.borrow_from_banks + bank_tejarat.borrow_from_banks + bank_mellat.borrow_from_banks + bank_refah.borrow_from_banks + bank_ayandeh.borrow_from_banks + bank_gardeshgary.borrow_from_banks + bank_iranzamin.borrow_from_banks + bank_sarmaye.borrow_from_banks + bank_sarmaye.borrow_from_banks + bank_pasargad.borrow_from_banks + bank_melal.borrow_from_banks
    supply_of_banks = bank_melli.lend_to_banks + bank_seppah.lend_to_banks + bank_tosesaderat.lend_to_banks + bank_maskan.lend_to_banks + bank_sanatmadan.lend_to_banks + bank_keshavarzi.lend_to_banks + bank_tosetavon.lend_to_banks + bank_post.lend_to_banks + bank_eghtesadnovin.lend_to_banks + bank_parsian.lend_to_banks + bank_karafarin.lend_to_banks + bank_saman.lend_to_banks + bank_saman.lend_to_banks + bank_sina.lend_to_banks + bank_khavarmiane.lend_to_banks + bank_shahr.lend_to_banks + bank_dey.lend_to_banks + bank_saderat.lend_to_banks + bank_tejarat.lend_to_banks + bank_mellat.lend_to_banks + bank_refah.lend_to_banks + bank_ayandeh.lend_to_banks + bank_gardeshgary.lend_to_banks + bank_iranzamin.lend_to_banks + bank_sarmaye.lend_to_banks + bank_sarmaye.lend_to_banks + bank_pasargad.lend_to_banks + bank_melal.lend_to_banks
    # print(demand_of_banks)
    # print(supply_of_banks)
    if demand_of_banks > supply_of_banks:
        rfree = (rfree + rfree_max) / 2
    elif demand_of_banks < supply_of_banks:
        rfree = (rfree + rfree_min) / 2
    else:
        rfree = rfree

    rfree_vector.append(f'{rfree}')

# print(rfree_vector)
rfree_plot = []
for i in range(0,len(rfree_vector)):
    rfree_plot.append([float(rfree_vector[i])])
plt.plot(rfree_plot)
plt.show()
## test
 ##### dynamics of model
# the name of bank which is source of the shock
www = bank_melli
sig = 0.01
shock = sig * (www.deposits + www.borrow_from_banks)

if shock <= www.bank_cash:
    landa = sig * (www.deposits + www.borrow_from_banks)
else:
    landa = 0

if landa => 0 :