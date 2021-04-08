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
ret_sec_shbank = 0.18
rfree = 0.185
rfree_min = 0.1
rfree_max = 0.25
rfree_vector = [f'{rfree}']
intrinsic_value = 100
p_market = intrinsic_value + np.random.normal(0)
p_market_max = 110
p_market_min = 90
p_market_vector = [f'{p_market}']
ret_sec_bank_vector = [f'{ret_sec_bank}']


########################################################################
# defining the agents: banks, shadow banks


class Bank:
    def __init__(self, bank_cash, lend_to_banks, lend_to_loans, bank_sec, deposits, borrow_from_banks, equity,
                 alpha_min, provision_per, phi, zeta, car, xs, xbl, xl):
        self.bankrupt = False
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
        self.stock = bank_sec / p_market
        self.security_sale = 0
        self.supply_of_stock_b = 0
        self.demand_of_stock_b = 0

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
        self.exit = False
        self.participation = participation
        self.shadow_bank_cash = shadow_bank_cash
        self.security = security
        self.s_alpha = s_alpha
        self.s_provision = s_provision
        self.int_value = np.random.normal(intrinsic_value)
        self.stock = security / p_market

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

# introduction of Iranian Shadow Banks

shadow1 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.01, 0.01)
shadow2 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.3, 0.3)
shadow3 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.91, 0.01)
shadow4 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.1, 0.1)
shadow5 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.1, 0.1)
shadow6 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.1, 0.1)
shadow7 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.1, 0.1)
shadow8 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.1, 0.1)
shadow9 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.1, 0.1)
shadow10 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.1, 0.1)
shadow11 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.1, 0.1)
shadow12 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.1, 0.1)
shadow13 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.1, 0.1)
shadow14 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.1, 0.1)
shadow15 = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.1, 0.1)


#####################################################

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
    bank_sec_old = mmm.bank_sec
    mmm.bank_sec = result.x[1]
    mmm.borrow_from_banks = result.x[2]
    mmm.ret_on_sec = float(np.random.normal(ret_sec_bank, ret_sec_bank_sigma))
    mmm.net_income = (mmm.ret_on_sec * mmm.bank_sec) + (rfree * mmm.lend_to_banks) - (
            (rfree * mmm.borrow_from_banks) / (1 - mmm.zeta * mmm.pd))
    mmm.sigma = abs(mmm.phi * (mmm.deposits + mmm.equity))
    mmm.profit = float(np.random.normal(mmm.net_income, mmm.sigma, 1))


    mmm.sec_sale = bank_sec_old - mmm.bank_sec
    if mmm.sec_sale > 0:
        mmm.demand_of_stock_b = 0
        mmm.supply_of_stock_b = mmm.security_sale
    elif mmm.security_sale < 0:
        mmm.demand_of_stock_b = mmm.security_sale
        mmm.supply_of_stock_b = 0
    else:
        mmm.demand_of_stock_b = 0
        mmm.supply_of_stock_b = 0

    mmm = Bank(result.x[4], result.x[0], result.x[3], result.x[1], result.x[2], mmm.deposits, mmm.equity,
               mmm.alpha_min, mmm.provision_per, mmm.phi, mmm.zeta, mmm.car, mmm.xs, mmm.xbl, mmm.xl)
    ########################################################
    # optimixation phase
    ### objective function of shadow banks
    # 1- security 2-cash


def optimize_shadow_bank(nnn):
    c_s = np.array([np.random.normal(-ret_sec_shbank, 0.01), 0])
    A_ub_s = np.array([[0, -1]])
    b_ub_s = np.array([-(nnn.s_alpha + nnn.s_provision) * nnn.participation])
    A_eq_s = np.array([[1, 1]])
    b_eq_s = np.array([nnn.participation])
    x0_bounds_s = (0, None)
    x1_bounds_s = (0, None)

    bounds_s = [x0_bounds_s, x1_bounds_s]
    result_s = linprog(c_s, A_ub=A_ub_s, b_ub=b_ub_s, A_eq=A_eq_s, b_eq=b_eq_s, bounds=bounds_s)
    nnn.shadow_bank_cash = result_s.x[1]
    nnn.security_old = nnn.security
    nnn.security = result_s.x[0]
    nnn.shadow_bank_cash = result_s.x[1]

    nnn.security_sale = nnn.security_old - nnn.security
    if nnn.security_sale > 0:
        nnn.demand_of_stock = 0
        nnn.supply_of_stock = nnn.security_sale
    elif nnn.security_sale < 0:
        nnn.demand_of_stock = abs(nnn.security_sale)
        nnn.supply_of_stock = 0
    else:
        nnn.demand_of_stock = 0
        nnn.supply_of_stock = 0

    nnn = Shadow_Bank(np.random.normal(100), np.random.normal(20), np.random.normal(80), 0.01, 0.01)

###############################################################

###############################################################

for ttt in range(n_sim):
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

    optimize_shadow_bank(shadow1)
    optimize_shadow_bank(shadow2)
    optimize_shadow_bank(shadow3)
    optimize_shadow_bank(shadow4)
    optimize_shadow_bank(shadow5)
    optimize_shadow_bank(shadow6)
    optimize_shadow_bank(shadow7)
    optimize_shadow_bank(shadow8)
    optimize_shadow_bank(shadow9)
    optimize_shadow_bank(shadow10)
    optimize_shadow_bank(shadow11)
    optimize_shadow_bank(shadow12)
    optimize_shadow_bank(shadow13)
    optimize_shadow_bank(shadow14)
    optimize_shadow_bank(shadow15)
    demand_of_banks = bank_melli.borrow_from_banks + bank_seppah.borrow_from_banks + bank_tosesaderat.borrow_from_banks + bank_maskan.borrow_from_banks + bank_sanatmadan.borrow_from_banks + bank_keshavarzi.borrow_from_banks + bank_tosetavon.borrow_from_banks + bank_post.borrow_from_banks + bank_eghtesadnovin.borrow_from_banks + bank_parsian.borrow_from_banks + bank_karafarin.borrow_from_banks + bank_saman.borrow_from_banks + bank_saman.borrow_from_banks + bank_sina.borrow_from_banks + bank_khavarmiane.borrow_from_banks + bank_shahr.borrow_from_banks + bank_dey.borrow_from_banks + bank_saderat.borrow_from_banks + bank_tejarat.borrow_from_banks + bank_mellat.borrow_from_banks + bank_refah.borrow_from_banks + bank_ayandeh.borrow_from_banks + bank_gardeshgary.borrow_from_banks + bank_iranzamin.borrow_from_banks + bank_sarmaye.borrow_from_banks + bank_sarmaye.borrow_from_banks + bank_pasargad.borrow_from_banks + bank_melal.borrow_from_banks
    supply_of_banks = bank_melli.lend_to_banks + bank_seppah.lend_to_banks + bank_tosesaderat.lend_to_banks + bank_maskan.lend_to_banks + bank_sanatmadan.lend_to_banks + bank_keshavarzi.lend_to_banks + bank_tosetavon.lend_to_banks + bank_post.lend_to_banks + bank_eghtesadnovin.lend_to_banks + bank_parsian.lend_to_banks + bank_karafarin.lend_to_banks + bank_saman.lend_to_banks + bank_saman.lend_to_banks + bank_sina.lend_to_banks + bank_khavarmiane.lend_to_banks + bank_shahr.lend_to_banks + bank_dey.lend_to_banks + bank_saderat.lend_to_banks + bank_tejarat.lend_to_banks + bank_mellat.lend_to_banks + bank_refah.lend_to_banks + bank_ayandeh.lend_to_banks + bank_gardeshgary.lend_to_banks + bank_iranzamin.lend_to_banks + bank_sarmaye.lend_to_banks + bank_sarmaye.lend_to_banks + bank_pasargad.lend_to_banks + bank_melal.lend_to_banks

    if demand_of_banks > supply_of_banks:
        rfree = (rfree + rfree_max) / 2
    elif demand_of_banks < supply_of_banks:
        rfree = (rfree + rfree_min) / 2
    else:
        rfree = rfree

    rfree_vector.append(rfree)

    stock_supply_of_banks = bank_melli.supply_of_stock_b + bank_seppah.supply_of_stock_b + bank_tosesaderat.supply_of_stock_b + bank_maskan.supply_of_stock_b + bank_sanatmadan.supply_of_stock_b + bank_keshavarzi.supply_of_stock_b + bank_tosetavon.supply_of_stock_b + bank_post.supply_of_stock_b + bank_eghtesadnovin.supply_of_stock_b + bank_parsian.supply_of_stock_b + bank_karafarin.supply_of_stock_b + bank_saman.supply_of_stock_b + bank_saman.supply_of_stock_b + bank_sina.supply_of_stock_b + bank_khavarmiane.supply_of_stock_b + bank_shahr.supply_of_stock_b + bank_dey.supply_of_stock_b + bank_saderat.supply_of_stock_b + bank_tejarat.supply_of_stock_b + bank_mellat.supply_of_stock_b + bank_refah.supply_of_stock_b + bank_ayandeh.supply_of_stock_b + bank_gardeshgary.supply_of_stock_b + bank_iranzamin.supply_of_stock_b + bank_sarmaye.supply_of_stock_b + bank_sarmaye.supply_of_stock_b + bank_pasargad.supply_of_stock_b + bank_melal.supply_of_stock_b
    stock_demand_of_banks = bank_melli.demand_of_stock_b + bank_seppah.demand_of_stock_b + bank_tosesaderat.demand_of_stock_b + bank_maskan.demand_of_stock_b + bank_sanatmadan.demand_of_stock_b + bank_keshavarzi.demand_of_stock_b + bank_tosetavon.demand_of_stock_b + bank_post.demand_of_stock_b + bank_eghtesadnovin.demand_of_stock_b + bank_parsian.demand_of_stock_b + bank_karafarin.demand_of_stock_b + bank_saman.demand_of_stock_b + bank_saman.demand_of_stock_b + bank_sina.demand_of_stock_b + bank_khavarmiane.demand_of_stock_b + bank_shahr.demand_of_stock_b + bank_dey.demand_of_stock_b + bank_saderat.demand_of_stock_b + bank_tejarat.demand_of_stock_b + bank_mellat.demand_of_stock_b + bank_refah.demand_of_stock_b + bank_ayandeh.demand_of_stock_b + bank_gardeshgary.demand_of_stock_b + bank_iranzamin.demand_of_stock_b + bank_sarmaye.demand_of_stock_b + bank_sarmaye.demand_of_stock_b + bank_pasargad.demand_of_stock_b + bank_melal.demand_of_stock_b
    stock_demand_of_shadow_banks = shadow1.demand_of_stock + shadow2.demand_of_stock + shadow3.demand_of_stock + shadow4.demand_of_stock + shadow5.demand_of_stock + shadow6.demand_of_stock + shadow7.demand_of_stock + shadow8.demand_of_stock + shadow9.demand_of_stock + shadow10.demand_of_stock + shadow11.demand_of_stock + shadow12.demand_of_stock + shadow13.demand_of_stock + shadow14.demand_of_stock + shadow15.demand_of_stock
    stock_supply_of_shadow_banks = shadow1.supply_of_stock + shadow2.supply_of_stock + shadow3.supply_of_stock + shadow4.supply_of_stock + shadow5.supply_of_stock + shadow6.supply_of_stock + shadow7.supply_of_stock + shadow8.supply_of_stock + shadow9.supply_of_stock + shadow10.supply_of_stock + shadow11.supply_of_stock + shadow12.supply_of_stock + shadow13.supply_of_stock + shadow14.supply_of_stock + shadow15.supply_of_stock
    total_stock_demand = stock_demand_of_banks + stock_demand_of_shadow_banks
    total_stock_supply = stock_supply_of_banks + stock_supply_of_shadow_banks
    p_market_old = p_market
    if total_stock_demand > total_stock_supply:
        p_market = (p_market + p_market_max) / 2
    elif demand_of_banks < supply_of_banks:
        p_market = (p_market + p_market_min) / 2
    else:
        p_market = p_market

    ret_on_sec = p_market / p_market_old - 1
    p_market_vector.append(p_market)
    ret_sec_bank_vector.append(ret_on_sec)


# print(rfree_vector)
# print(ret_sec_bank_vector)


rfree_plot = []
for i in range(0, len(rfree_vector)):
    rfree_plot.append([float(rfree_vector[i])])



p_market_plot = []
for i in range(0, len(p_market_vector)):
    p_market_plot.append([float(p_market_vector[i])])



ret_on_sec_plot = []
for i in range(0, len(ret_sec_bank_vector)):
    ret_on_sec_plot.append([float(ret_sec_bank_vector[i])])

# print(ret_on_sec_plot)
# print(p_market_plot)
# print(rfree_plot)

# plt.plot(rfree_plot)
# plt.show()
plt.plot(p_market_plot)
plt.show()
# plt.plot(ret_on_sec_plot)
# plt.show()


##### dynamics of model
# the name of bank which is source of the shock
# shock_hit = bank_melli
# sig = 0.01
# shock = sig * (shock_hit.deposits + shock_hit.borrow_from_banks)


# def dynamic_bank(www):
#     if shock <= www.bank_cash:
#         landa = sig * (www.deposits + www.borrow_from_banks)
#         www.equity = www.equity - (www.bank_cash - landa)
#         www.bank_cash = www.bank_cash - landa
#     elif (www.bank_cash + www.lend_to_banks) >= shock >= www.bank_cash and www.lend_to_banks != 0:
#         www.equity = www.equity - www.bank_cash
#         www.bank_cash = 0
#         www.lend_to_banks = www.lend_to_banks - (shock - www.bank_cash)
#     elif (www.bank_cash + www.lend_to_banks + www.bank_sec) >= shock >= (www.bank_cash + www.lend_to_banks):
#         delta = www.bank_cash + www.lend_to_banks + www.bank_sec - shock
#         www.bank_cash = 0
#         www.equity = www.equity - www.bank_cash
#         www.lend_to_banks = 0
#         www.bank_sec = www.bank_sec - delta
#         www.stock = www.bank_sec / p_market
#     elif (www.bank_cash + www.lend_to_banks + www.bank_sec) <= shock:
#         www.bankrupt = True
