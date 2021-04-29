import numpy as np
import matplotlib.pyplot as plt
import scipy as spy
from scipy.stats import norm
from scipy.optimize import linprog
import math

##################################################################################

### global variables
n_sim = 30
etha = 1
ret_sec_bank = 0.18
ret_sec_bank_sigma = 0.03
ret_sec_shbank = 0.18
rfree = 0.18
rfree_min = 0.20
rfree_max = 0.25
rfree_vector = [f'{rfree}']
intrinsic_value = 100
p_market = intrinsic_value + np.random.normal(0)
p_market_old = p_market
p_market_max = 150
p_market_min = 80
p_market_vector = [f'{p_market}']
ret_sec_bank_vector = [f'{ret_sec_bank}']

every = 0
every_thing_vector = [f'{every}']

every1 = 0
every1_thing_vector = [f'{every1}']

every2 = p_market
every2_thing_vector = [f'{every2}']


########################################################################
# defining the agents: banks, shadow banks


class Bank:
    def __init__(self, bank_cash, lend_to_banks, lend_to_loans, bank_sec, deposits, borrow_from_banks, equity,
                 alpha_min, provision_per, phi, zeta, car, xs, xbl, xl, etha_max):
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
        self.phi = phi
        self.sigma = phi * (self.deposits + self.equity)
        # self.profit = float(np.random.normal(self.net_income, self.sigma, 1))
        # self.pd = float(norm.ppf((-self.net_income - self.equity) / (self.sigma)))
        self.pd = np.random.beta(1, 20)
        self.etha_max = etha_max


class Shadow_Bank:
    def __init__(self, participation, shadow_bank_cash, security, s_alpha, s_provision):
        self.exit = False
        self.participation = participation
        self.shadow_bank_cash = shadow_bank_cash
        self.security = security
        self.s_alpha = s_alpha
        self.s_provision = s_provision
        self.int_value = np.random.normal(intrinsic_value)
        self.redemption = 0
        self.stock = security / p_market

        ##### income and expense of shadow bank


###############################################################
# introduction of Iranian Banks

bank_melli = Bank(2.9, 22.9, 66.3, 33.5, 102.3, 20.2, 3.2, 0.1, 0.1, 0.1, 0.4, 0.07, 0.05, 0.05, 0.05, 1)
bank_seppah = Bank(0.8, 6, 17.4, 8.8, 14.7, 6.3, 12.1, 0.1, 0.1, 0.1, 0.4, 0.07, 0.05, 0.05, 0.05, 1)
bank_tosesaderat = Bank(8.3, 3.2, 22.8, 21.5, 5.7, 8.2, 42, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_maskan = Bank(3.4, 0, 54.2, 9.3, 6.9, 9.8, 50.3, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_sanatmadan = Bank(6.7, 0, 106, 18.3, 8.9, 42.7, 79.3, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_keshavarzi = Bank(25.3, 3.2, 88, 47.4, 33.6, 2.1, 128.2, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_tosetavon = Bank(1.3, 0.2, 16.9, 4.4, 5.6, 2.3, 14.8, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_post = Bank(0.6, 4.4, 11.1, 1.3, 13.3, 4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_eghtesadnovin = Bank(4, 6.2, 44, 10.9, 51.1, 7.3, 6.7, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_parsian = Bank(9.5, 27.6, 94.6, 26.4, 33.3, 11.3, 113.5, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_karafarin = Bank(4.8, 7.5, 52.8, 13, 61.3, 8.7, 8.1, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_saman = Bank(0.5, 14.1, 22.6, 18, 46.3, 4.9, 4, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_sina = Bank(2, 2.6, 16, 3.8, 21.6, 0.2, 2.6, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_khavarmiane = Bank(0.2, 5.2, 11.8, 4, 12.7, 5.3, 3.2, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_shahr = Bank(0.8, 9.9, 43.7, 21.4, 85.8, 5.1, -15.1, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_dey = Bank(1, 3.2, 8.2, 15.1, 27.8, 8.4, -8.7, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_saderat = Bank(16.3, 45.5, 204.3, 60.4, 231.3, 41.6, 53.6, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_tejarat = Bank(13.1, 46.4, 133.8, 51.6, 197.6, 22.3, 25, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_mellat = Bank(21.2, 50.8, 311.3, 68.4, 269.6, 127.1, 55, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_refah = Bank(2.4, 3.1, 19.1, 4.5, 25.9, 0.2, 3.1, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_ayandeh = Bank(1.5, 21.8, 97.6, 79.3, 186.1, 22.8, -8.7, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_gardeshgary = Bank(1, 0.2, 19.8, 29.7, 38.4, 8.4, 3.8, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_iranzamin = Bank(0.3, 4.1, 3.5, 29.2, 33.6, 6.1, -2.6, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_sarmaye = Bank(0.5, 2.7, 5.4, 5.4, 18.3, 21.7, -26.1, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 0)
bank_pasargad = Bank(18.7, 21.6, 79.9, 45.8, 116.9, 19.3, 29.9, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)
bank_melal = Bank(0.2, 2.3, 10.8, 21.1, 15.2, 15, 4, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.05, 0.05, 1)

# introduction of Iranian Shadow Banks

shadow1 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.01, 0.01)
shadow2 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.3, 0.3)
shadow3 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.91, 0.01)
shadow4 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow5 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow6 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow7 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow8 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow9 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow10 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow11 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow12 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow13 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow14 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow15 = Shadow_Bank(np.random.normal(45), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)


#####################################################

#####################################################
# 1-BL 2-S 3-BB 4-L 5-C
# optimixation phase
### objective function of banks

def optimize_bank(mmm):
    if mmm.bankrupt == False:
        c = np.array([-rfree, -mmm.ret_on_sec, ((rfree * mmm.borrow_from_banks) / (1 - mmm.zeta * mmm.pd)), 0, 0])
        A_ub = np.array([[(-1 + mmm.car * mmm.xbl), (-1 + mmm.car * mmm.xs), 1, (-1 + mmm.car * mmm.xl), -1],
                         [-1, 0, (mmm.alpha_min + mmm.provision_per), 0, -1], [0, 0, 0, 0, -1], [0, 1, 0, 0, 0]])
        b_ub = np.array([-mmm.deposits, -(mmm.alpha_min + mmm.provision_per) * mmm.deposits,
                         -(mmm.alpha_min + mmm.provision_per) * mmm.deposits, mmm.etha_max * mmm.total_assets])
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
                   mmm.alpha_min, mmm.provision_per, mmm.phi, mmm.zeta, mmm.car, mmm.xs, mmm.xbl, mmm.xl, mmm.etha_max)
    else:
        mmm = Bank(0, 0, 0, 0, 0, 0, 0, mmm.alpha_min, mmm.provision_per, mmm.phi, mmm.zeta, mmm.car, mmm.xs, mmm.xbl,
                   mmm.xl, mmm.etha_max)

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

    nnn = Shadow_Bank((result_s.x[0] + result_s.x[1]), result_s.x[0], result_s.x[1], nnn.s_alpha, nnn.s_provision)


######################################################################################################

def redemption(www):
    p_change = 1 - (p_market / p_market_old)
    if p_change < 0:
        www.redemption = 0
    else:
        www.redemption = www.participation * ((math.exp(etha * p_change) - 1))

    if www.redemption < www.shadow_bank_cash:
        www.shadow_bank_cash = www.shadow_bank_cash - www.redemption
        www.participation = www.participation - www.redemption

    elif www.shadow_bank_cash < www.redemption < www.security + www.shadow_bank_cash:
        www.shadow_bank_cash = 0
        www.security = www.security + www.shadow_bank_cash - www.redemption
        www.participation = www.participation - www.redemption
    elif www.security + www.shadow_bank_cash < www.redemption:
        www.exit = True
        www.participation = 0
        www.security = 0
        www.shadow_bank_cash = 0


###############################################################
##### dynamics of model
# the name of bank which is source of the shock

shock_hit = bank_melli

sig = 0.5
shock = sig * (shock_hit.deposits + shock_hit.borrow_from_banks)


def dynamic_bank(www):
    if shock <= www.bank_cash:
        www.equity = www.equity - shock
        www.bank_cash = www.bank_cash - shock
    elif (www.bank_cash + www.lend_to_banks) >= shock >= www.bank_cash:
        landa = (sig * (www.deposits + www.borrow_from_banks)) - www.bank_cash
        www.equity = www.equity - www.bank_cash
        www.bank_cash = 0
        www.lend_to_banks = www.lend_to_banks - (shock - www.bank_cash)
    elif (www.bank_cash + www.lend_to_banks + www.bank_sec) >= shock >= (www.bank_cash + www.lend_to_banks):
        delta = www.bank_cash + www.lend_to_banks + www.bank_sec - shock
        www.bank_cash = 0
        www.equity = www.equity - www.bank_cash
        www.lend_to_banks = 0
        www.bank_sec = www.bank_sec - delta
        www.stock = www.bank_sec / p_market
    elif (www.bank_cash + www.lend_to_banks + www.bank_sec) <= shock:
        www.bank_cash = 0
        www.equity = 0
        www.lend_to_banks = 0
        www.bank_sec = 0
        www.borrow_from_banks = 0
        www.bankrupt = True


###############################################################
################ start the simulation
################
dynamic_bank(shock_hit)
for ttt in range(n_sim):

    # first banks
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

    # second shadow banks determine the redemptions because it is almost a legal issue the they optimize which is a ecnomic behaviour

    redemption(shadow1)
    redemption(shadow2)
    redemption(shadow3)
    redemption(shadow4)
    redemption(shadow5)
    redemption(shadow6)
    redemption(shadow7)
    redemption(shadow8)
    redemption(shadow9)
    redemption(shadow10)
    redemption(shadow11)
    redemption(shadow12)
    redemption(shadow13)
    redemption(shadow14)
    redemption(shadow15)

    ## optimization of shadow banks

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
    diference = demand_of_banks - supply_of_banks

    if demand_of_banks > supply_of_banks:
        # rrr = (demand_of_banks/supply_of_banks) - 1
        # rfree = (1+rrr) * rfree
        rfree = (rfree + rfree_max) / 2
    elif demand_of_banks < supply_of_banks:
        # rrr = (supply_of_banks / demand_of_banks) - 1
        # rfree = (1 - rrr) * rfree
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
    ret_sec_shbank = ret_on_sec
    p_market_vector.append(p_market)
    ret_sec_bank_vector.append(ret_on_sec)

    every = total_stock_demand
    every_thing_vector.append(every)

    every1 = total_stock_supply
    every1_thing_vector.append(every1)

    every2 = p_market
    every2_thing_vector.append(every2)

# rfree_plot = []
# for i in range(0, len(rfree_vector)):
#     rfree_plot.append([float(rfree_vector[i])])
#
# p_market_plot = []
# for i in range(0, len(p_market_vector)):
#     p_market_plot.append([float(p_market_vector[i])])
#
# ret_on_sec_plot = []
# for i in range(0, len(ret_sec_bank_vector)):
#     ret_on_sec_plot.append([float(ret_sec_bank_vector[i])])

every_thing_plot = []
for i in range(0, len(every_thing_vector)):
    every_thing_plot.append([float(every_thing_vector[i])])

every1_thing_plot = []
for i in range(0, len(every1_thing_vector)):
    every1_thing_plot.append([float(every1_thing_vector[i])])

every2_thing_plot = []
for i in range(0, len(every2_thing_vector)):
    every2_thing_plot.append([float(every2_thing_vector[i])])

# print(ret_on_sec_plot)
# print(p_market_plot)
# print(rfree_plot)
# print(every_thing_vector)


# plt.plot(rfree_plot)
# plt.show()
# plt.plot(p_market_plot)
# plt.show()
# plt.plot(ret_on_sec_plot)
# plt.show()

plt.plot(every_thing_plot)
plt.plot(every1_thing_plot)
plt.plot(every2_thing_plot)

plt.show()
#
# plt.plot(every2_thing_plot)
# plt.show()


# print(every_thing_plot)
# print(every1_thing_plot)
# print(every2_thing_plot)