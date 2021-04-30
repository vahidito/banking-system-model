import numpy as np
import matplotlib.pyplot as plt
import scipy as spy
from scipy.stats import norm
from scipy.optimize import linprog
import math

################################################
# global variables and initialization

n_sim = 25
etha = 0.8
ret_sec_bank = 0.18
ret_sec_bank_sigma = 0.03
ret_sec_shbank = 0.18
rfree = 0.18
rfree_min = 0.15
rfree_max = 0.25

rfree_vector = [f'{rfree}']

intrinsic_value = 1
p_market = 1
p_market_old = p_market
p_market_max = 1.50
p_market_min = 0.80

p_market_vector = [f'{p_market}']

#################################################
# variables for visualization

every = p_market
every_thing_vector = [f'{every}']

every1 = 0
every1_thing_vector = [f'{every1}']

every2 = 0
every2_thing_vector = [f'{every2}']

every3 = intrinsic_value
every3_thing_vector = [f'{every3}']

every4 = rfree
every4_thing_vector = [f'{every4}']


#################################################
# defining the agents: banks, shadow banks
# we made two class of agents


class Bank:
    def __init__(self, bank_cash, lend_to_banks, lend_to_loans, bank_sec, deposits, borrow_from_banks, equity,
                 alpha_min, provision_per, phi, zeta, car, xs, xbl, xl, etha_max):

        self.bank_cash = bank_cash
        self.lend_to_banks = lend_to_banks
        self.lend_to_loans = lend_to_loans
        self.bank_sec = bank_sec
        self.deposits = deposits
        self.borrow_from_banks = borrow_from_banks
        self.equity = equity
        if self.equity < 0:
            self.bankrupt = True
        else:
            self.bankrupt = False

        # setting parameters of the bank

        self.alpha_min = alpha_min
        self.provision_per = provision_per
        self.zeta = zeta
        self.total_assets = bank_cash + lend_to_banks + lend_to_loans + bank_sec
        self.xs = xs
        self.xbl = xbl
        self.xl = xl
        self.car = car
        self.ret_on_sec = np.random.normal(ret_sec_bank, ret_sec_bank_sigma)

        self.security_sale = 0
        self.supply_of_stock_b = 0
        self.demand_of_stock_b = 0
        self.rwa = (xs * self.bank_sec) + (xl * self.lend_to_loans) + (xbl * self.lend_to_banks)
        self.nbl = 0
        self.ns = 0
        self.init_value_stock = np.random.normal(intrinsic_value, 0.05)
        self.nd = 0

        # income and expense of bank

        self.phi = phi
        self.sigma = phi * (self.deposits + self.equity)
        # self.profit = float(np.random.normal(self.net_income, self.sigma, 1))
        # self.pd = float(norm.ppf((-self.net_income - self.equity) / (self.sigma)))
        self.pd = np.random.beta(1, 20)
        self.etha_max = etha_max
        self.sig = 0
        self.bank_type = 1
        self.name = [f'{self}']


class Shadow_Bank:
    def __init__(self, participation, shadow_bank_cash, security, s_alpha, s_provision):

        self.participation = participation
        if self.participation < 0:
            self.exit = True
        else:
            self.exit = False

        self.shadow_bank_cash = shadow_bank_cash
        self.security = security
        self.s_alpha = s_alpha
        self.s_provision = s_provision
        self.int_value = np.random.normal(intrinsic_value, 0.01)
        self.redemption = 0
        self.nd_s = 0
        self.ns_s = 0


#################################################
# making vectors for sensitive analysis

alpha_vector = [0.08, 0.1, 0.12]
car_vector = [0.06, 0.07, 0.08]
ind_alpha = 2
ind_car = 0
alpha = alpha_vector[ind_alpha]
car = car_vector[ind_car]
etha_max_bank = 0.2
###############################################################

# introduction of Iranian Banks
# using financial statements in Codal website to initial the banks

bank_melli = Bank(2.9, 22.9, 66.3, 33.5, 102.3, 20.2, 3.2, alpha, 0.1, 0.1, 0.4, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_seppah = Bank(0.8, 6, 17.4, 8.8, 14.7, 6.3, 12.1, alpha, 0.1, 0.1, 0.4, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_tosesaderat = Bank(8.3, 3.2, 22.8, 21.5, 5.7, 8.2, 42, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_maskan = Bank(3.4, 0, 54.2, 9.3, 6.9, 9.8, 50.3, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_sanatmadan = Bank(6.7, 0, 106, 18.3, 8.9, 42.7, 79.3, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_keshavarzi = Bank(25.3, 3.2, 88, 47.4, 33.6, 2.1, 128.2, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05,
                       etha_max_bank)
bank_tosetavon = Bank(1.3, 0.2, 16.9, 4.4, 5.6, 2.3, 14.8, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_post = Bank(0.6, 4.4, 11.1, 1.3, 13.3, 4, 0.1, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_eghtesadnovin = Bank(4, 6.2, 44, 10.9, 51.1, 7.3, 6.7, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_parsian = Bank(9.5, 27.6, 94.6, 26.4, 33.3, 11.3, 113.5, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05,
                    etha_max_bank)
bank_karafarin = Bank(4.8, 7.5, 52.8, 13, 61.3, 8.7, 8.1, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_saman = Bank(0.5, 14.1, 22.6, 18, 46.3, 4.9, 4, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_sina = Bank(2, 2.6, 16, 3.8, 21.6, 0.2, 2.6, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_khavarmiane = Bank(0.2, 5.2, 11.8, 4, 12.7, 5.3, 3.2, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_shahr = Bank(0.8, 9.9, 43.7, 21.4, 85.8, 5.1, -15.1, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_dey = Bank(1, 3.2, 8.2, 15.1, 27.8, 8.4, -8.7, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_saderat = Bank(16.3, 45.5, 204.3, 60.4, 231.3, 41.6, 53.6, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05,
                    etha_max_bank)
bank_tejarat = Bank(13.1, 46.4, 133.8, 51.6, 197.6, 22.3, 25, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05,
                    etha_max_bank)
bank_mellat = Bank(21.2, 50.8, 311.3, 68.4, 269.6, 127.1, 55, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05,
                   etha_max_bank)
bank_refah = Bank(2.4, 3.1, 19.1, 4.5, 25.9, 0.2, 3.1, 0.1, alpha, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_ayandeh = Bank(1.5, 21.8, 97.6, 79.3, 186.1, 22.8, -8.7, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05,
                    etha_max_bank)
bank_gardeshgary = Bank(1, 0.2, 19.8, 29.7, 38.4, 8.4, 3.8, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_iranzamin = Bank(0.3, 4.1, 3.5, 29.2, 33.6, 6.1, -2.6, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_sarmaye = Bank(0.5, 2.7, 5.4, 5.4, 18.3, 21.7, -26.1, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)
bank_pasargad = Bank(18.7, 21.6, 79.9, 45.8, 116.9, 19.3, 29.9, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05,
                     etha_max_bank)
bank_melal = Bank(0.2, 2.3, 10.8, 21.1, 15.2, 15, 4, alpha, 0.1, 0.1, 0.1, car, 0.05, 0.05, 0.05, etha_max_bank)

all_banks = [bank_melli, bank_seppah, bank_tosesaderat, bank_maskan, bank_sanatmadan, bank_keshavarzi, bank_tosetavon,
             bank_post, bank_eghtesadnovin, bank_parsian, bank_karafarin, bank_saman, bank_sina, bank_khavarmiane,
             bank_shahr, bank_dey, bank_saderat, bank_tejarat, bank_mellat, bank_refah, bank_ayandeh, bank_gardeshgary,
             bank_iranzamin, bank_sarmaye, bank_pasargad, bank_melal]
# introduction of Iranian Shadow Banks
# this introduction is randomly

shadow1 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.01, 0.01)
shadow2 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.3, 0.3)
shadow3 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.91, 0.01)
shadow4 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow5 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow6 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow7 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow8 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow9 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow10 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow11 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow12 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow13 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow14 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)
shadow15 = Shadow_Bank(np.random.normal(45, 1), np.random.normal(3.5), np.random.normal(41.5), 0.1, 0.1)

all_shadow_banks = [shadow1, shadow2, shadow3, shadow4, shadow5, shadow6, shadow7, shadow8, shadow9, shadow10, shadow11,
                    shadow12, shadow13, shadow14, shadow15]


#####################################################
# Optimization phase of model
#####################################################
# 1-BL 2-S 3-BB 4-L 5-C
# optimization phase
# objective function of banks

def optimize_bank(mmm):
    if mmm.bankrupt == False:
        c = np.array([-rfree, -mmm.ret_on_sec, ((rfree * 1) / (1 - mmm.zeta * mmm.pd)), 0, 0])
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
        mmm.bank_sec_old = mmm.bank_sec
        mmm.bank_sec = result.x[1]
        diff = mmm.bank_sec - mmm.bank_sec_old
        if diff > 0 and mmm.sig == 0:
            mmm.nd = diff
        elif diff < 0 and mmm.sig == 0:
            mmm.ns = -diff
        mmm.borrow_from_banks = result.x[2]
        mmm.ret_on_sec = float(np.random.normal(ret_sec_bank, ret_sec_bank_sigma))
        mmm.net_income = (mmm.ret_on_sec * mmm.bank_sec) + (rfree * mmm.lend_to_banks) - (
                (rfree * mmm.borrow_from_banks) / (1 - mmm.zeta * mmm.pd))
        mmm.sigma = abs(mmm.phi * (mmm.deposits + mmm.equity))
        mmm.profit = float(np.random.normal(mmm.net_income, mmm.sigma, 1))

        mmm = Bank(result.x[4], result.x[0], result.x[3], result.x[1], result.x[2], mmm.deposits, mmm.equity,
                   mmm.alpha_min, mmm.provision_per, mmm.phi, mmm.zeta, mmm.car, mmm.xs, mmm.xbl, mmm.xl, mmm.etha_max)
    else:
        mmm = Bank(0, 0, 0, 0, 0, 0, 0, mmm.alpha_min, mmm.provision_per, mmm.phi, mmm.zeta, mmm.car, mmm.xs, mmm.xbl,
                   mmm.xl, mmm.etha_max)

    ########################################################
    # optimization phase
    # objective function of shadow banks
    # 1- security 2-cash


def optimize_shadow_bank(nnn):
    if nnn.exit == False:

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

    else:
        nnn = Shadow_Bank(0, 0, 0, nnn.s_alpha, nnn.s_provision)


################################################
# defining redemption in model
def redemption(www):
    p_change = www.int_value - p_market
    if p_change < 0:
        www.redemption = 0
        www.ns_s = 0

    else:
        www.redemption = www.participation * (math.exp(etha * p_change) - 1)
        www.ns_s = www.redemption

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


###

######################################################################################################
###############################################################
###############################################################
# determine the quantity of loan and security
# which must be released in case of hitting shock
# the name of bank which is source of the shock


def bank_shock(www, sig=0):
    www.sig = sig
    shock = www.sig * (www.deposits + www.borrow_from_banks)
    if shock <= www.bank_cash:
        www.equity = www.equity - shock
        www.bank_cash = www.bank_cash - shock
        www.ns = 0
    elif (www.bank_cash + www.lend_to_banks) >= shock >= www.bank_cash:
        www.equity = www.equity - www.bank_cash
        www.bank_cash = 0
        www.lend_to_banks = www.lend_to_banks - (shock - www.bank_cash)
        www.ns = 0
    elif (www.bank_cash + www.lend_to_banks + www.bank_sec) >= shock >= (www.bank_cash + www.lend_to_banks):
        delta = www.bank_cash + www.lend_to_banks + www.bank_sec - shock
        www.bank_cash = 0
        www.equity = www.equity - www.bank_cash
        www.lend_to_banks = 0
        www.bank_sec = www.bank_sec - delta
        www.ns = delta
    elif (www.bank_cash + www.lend_to_banks + www.bank_sec) <= shock:
        delta = www.bank_cash + www.lend_to_banks + www.bank_sec - shock
        www.ns = delta
        www.bank_cash = 0
        www.equity = 0
        www.lend_to_banks = 0
        www.bank_sec = 0
        www.borrow_from_banks = 0
        www.bankrupt = True


#################################

bank_shock(bank_melli, 0)
bank_shock(bank_seppah)
bank_shock(bank_tosesaderat)
bank_shock(bank_maskan)
bank_shock(bank_sanatmadan)
bank_shock(bank_keshavarzi)
bank_shock(bank_tosetavon)
bank_shock(bank_post)
bank_shock(bank_eghtesadnovin)
bank_shock(bank_parsian)
bank_shock(bank_karafarin)
bank_shock(bank_saman)
bank_shock(bank_sina)
bank_shock(bank_khavarmiane)
bank_shock(bank_shahr)
bank_shock(bank_dey)
bank_shock(bank_saderat)
bank_shock(bank_tejarat)
bank_shock(bank_mellat)
bank_shock(bank_refah)
bank_shock(bank_ayandeh)
bank_shock(bank_gardeshgary)
bank_shock(bank_iranzamin)
bank_shock(bank_sarmaye)
bank_shock(bank_pasargad)
bank_shock(bank_melal)

#################################

for i in range(n_sim):

    #################################
    # supply of stock by shadow banks and banks

    total_supply_of_bank = bank_melli.ns + bank_seppah.ns + bank_tosesaderat.ns + bank_maskan.ns + bank_sanatmadan.ns + bank_keshavarzi.ns + bank_tosetavon.ns + bank_post.ns + bank_eghtesadnovin.ns + bank_parsian.ns + bank_karafarin.ns + bank_saman.ns + bank_saman.ns + bank_sina.ns + bank_khavarmiane.ns + bank_shahr.ns + bank_dey.ns + bank_saderat.ns + bank_tejarat.ns + bank_mellat.ns + bank_refah.ns + bank_ayandeh.ns + bank_gardeshgary.ns + bank_iranzamin.ns + bank_sarmaye.ns + bank_pasargad.ns + bank_melal.ns

    # from source of redemption

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

    total_supply_of_shadow = shadow1.ns_s + shadow2.ns_s + shadow3.ns_s + shadow4.ns_s + shadow5.ns_s + shadow6.ns_s + shadow7.ns_s + shadow8.ns_s + shadow9.ns_s + shadow10.ns_s + shadow11.ns_s + shadow12.ns_s + shadow13.ns_s + shadow14.ns_s + shadow15.ns_s


    ##########################################
    # demand of stock

    def demand_of_stock_shadow_bank(mmm):
        if p_market >= mmm.int_value:
            mmm.nd_s = 0
        elif p_market < mmm.int_value and mmm.shadow_bank_cash > mmm.ns_s:
            mmm.nd_s = mmm.shadow_bank_cash - mmm.ns_s
            mmm.shadow_bank_cash = mmm.shadow_bank_cash - mmm.nd_s
            mmm.security = mmm.security + mmm.nd_s


    demand_of_stock_shadow_bank(shadow1)
    demand_of_stock_shadow_bank(shadow2)
    demand_of_stock_shadow_bank(shadow3)
    demand_of_stock_shadow_bank(shadow4)
    demand_of_stock_shadow_bank(shadow5)
    demand_of_stock_shadow_bank(shadow6)
    demand_of_stock_shadow_bank(shadow7)
    demand_of_stock_shadow_bank(shadow8)
    demand_of_stock_shadow_bank(shadow9)
    demand_of_stock_shadow_bank(shadow10)
    demand_of_stock_shadow_bank(shadow11)
    demand_of_stock_shadow_bank(shadow12)
    demand_of_stock_shadow_bank(shadow13)
    demand_of_stock_shadow_bank(shadow14)
    demand_of_stock_shadow_bank(shadow15)

    total_demand_of_shadow = shadow1.nd_s + shadow2.nd_s + shadow3.nd_s + shadow4.nd_s + shadow5.nd_s + shadow6.nd_s + shadow7.nd_s + shadow8.nd_s + shadow9.nd_s + shadow10.nd_s + shadow11.nd_s + shadow12.nd_s + shadow13.nd_s + shadow14.nd_s + shadow15.nd_s


    def demand_of_stock_bank(mmm):
        if p_market > mmm.init_value_stock:
            mmm.nd = 0
        elif p_market < mmm.init_value_stock and mmm.bank_cash > 0 and mmm.sig == 0:
            mmm.nd = ( (mmm.alpha_min + mmm.car)) * mmm.bank_cash


    demand_of_stock_bank(bank_melli)
    demand_of_stock_bank(bank_seppah)
    demand_of_stock_bank(bank_tosesaderat)
    demand_of_stock_bank(bank_maskan)
    demand_of_stock_bank(bank_sanatmadan)
    demand_of_stock_bank(bank_keshavarzi)
    demand_of_stock_bank(bank_tosetavon)
    demand_of_stock_bank(bank_post)
    demand_of_stock_bank(bank_eghtesadnovin)
    demand_of_stock_bank(bank_parsian)
    demand_of_stock_bank(bank_karafarin)
    demand_of_stock_bank(bank_saman)
    demand_of_stock_bank(bank_sina)
    demand_of_stock_bank(bank_khavarmiane)
    demand_of_stock_bank(bank_shahr)
    demand_of_stock_bank(bank_dey)
    demand_of_stock_bank(bank_saderat)
    demand_of_stock_bank(bank_tejarat)
    demand_of_stock_bank(bank_mellat)
    demand_of_stock_bank(bank_refah)
    demand_of_stock_bank(bank_ayandeh)
    demand_of_stock_bank(bank_gardeshgary)
    demand_of_stock_bank(bank_iranzamin)
    demand_of_stock_bank(bank_sarmaye)
    demand_of_stock_bank(bank_pasargad)
    demand_of_stock_bank(bank_melal)

    total_demand_of_bank = bank_melli.nd + bank_seppah.nd + bank_tosesaderat.nd + bank_maskan.nd + bank_sanatmadan.nd + bank_keshavarzi.nd + bank_tosetavon.nd + bank_post.nd + bank_eghtesadnovin.nd + bank_parsian.nd + bank_karafarin.nd + bank_saman.nd + bank_saman.nd + bank_sina.nd + bank_khavarmiane.nd + bank_shahr.nd + bank_dey.nd + bank_saderat.nd + bank_tejarat.nd + bank_mellat.nd + bank_refah.nd + bank_ayandeh.nd + bank_gardeshgary.nd + bank_iranzamin.nd + bank_sarmaye.nd + bank_pasargad.nd + bank_melal.nd
    # total_demand_of_bank = 0

    ###########################################################################################
    ###########################################################################################
    ###########################################################################################
    # equilibrium in capital market

    total_supply_of_stock = total_supply_of_shadow + total_supply_of_bank
    total_demand_of_stock = total_demand_of_shadow + total_demand_of_bank
    distance = total_supply_of_stock - total_demand_of_stock

    ###########################################################################################
    ###########################################################################################
    ###########################################################################################
    # print(total_demand_of_stock)
    # print(total_supply_of_stock)
    if (total_supply_of_stock + total_demand_of_stock) == 0:
        growth_of_price = 0
    else:
        minus = abs(total_demand_of_stock - total_supply_of_stock) / (total_supply_of_stock + total_demand_of_stock)
        growth_of_price = 0.05 * minus

    if total_demand_of_stock > total_supply_of_stock:

        p_market = p_market * (1 + growth_of_price)
    elif total_demand_of_stock < total_supply_of_stock:

        p_market = p_market * (1 - growth_of_price)

    else:
        p_market = p_market

    # print(p_market)

    ########################################################
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
    ########################################################
    # equilibrium in interbank loan market

    demand_of_banks = bank_melli.borrow_from_banks + bank_seppah.borrow_from_banks + bank_tosesaderat.borrow_from_banks + bank_maskan.borrow_from_banks + bank_sanatmadan.borrow_from_banks + bank_keshavarzi.borrow_from_banks + bank_tosetavon.borrow_from_banks + bank_post.borrow_from_banks + bank_eghtesadnovin.borrow_from_banks + bank_parsian.borrow_from_banks + bank_karafarin.borrow_from_banks + bank_saman.borrow_from_banks + bank_saman.borrow_from_banks + bank_sina.borrow_from_banks + bank_khavarmiane.borrow_from_banks + bank_shahr.borrow_from_banks + bank_dey.borrow_from_banks + bank_saderat.borrow_from_banks + bank_tejarat.borrow_from_banks + bank_mellat.borrow_from_banks + bank_refah.borrow_from_banks + bank_ayandeh.borrow_from_banks + bank_gardeshgary.borrow_from_banks + bank_iranzamin.borrow_from_banks + bank_sarmaye.borrow_from_banks + bank_sarmaye.borrow_from_banks + bank_pasargad.borrow_from_banks + bank_melal.borrow_from_banks
    supply_of_banks = bank_melli.lend_to_banks + bank_seppah.lend_to_banks + bank_tosesaderat.lend_to_banks + bank_maskan.lend_to_banks + bank_sanatmadan.lend_to_banks + bank_keshavarzi.lend_to_banks + bank_tosetavon.lend_to_banks + bank_post.lend_to_banks + bank_eghtesadnovin.lend_to_banks + bank_parsian.lend_to_banks + bank_karafarin.lend_to_banks + bank_saman.lend_to_banks + bank_saman.lend_to_banks + bank_sina.lend_to_banks + bank_khavarmiane.lend_to_banks + bank_shahr.lend_to_banks + bank_dey.lend_to_banks + bank_saderat.lend_to_banks + bank_tejarat.lend_to_banks + bank_mellat.lend_to_banks + bank_refah.lend_to_banks + bank_ayandeh.lend_to_banks + bank_gardeshgary.lend_to_banks + bank_iranzamin.lend_to_banks + bank_sarmaye.lend_to_banks + bank_sarmaye.lend_to_banks + bank_pasargad.lend_to_banks + bank_melal.lend_to_banks
    difference = demand_of_banks - supply_of_banks

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

    ########################################################
    # visualization

    every = p_market
    every_thing_vector.append(every)

    every1 = total_supply_of_stock
    every1_thing_vector.append(every1)

    every2 = total_demand_of_stock
    every2_thing_vector.append(every2)

    every3 = intrinsic_value
    every3_thing_vector.append(every3)

    every4 = rfree
    every4_thing_vector.append(every4)

every_thing_plot = []
for i in range(0, len(every_thing_vector)):
    every_thing_plot.append([float(every_thing_vector[i])])

every1_thing_plot = []
for i in range(0, len(every1_thing_vector)):
    every1_thing_plot.append([float(every1_thing_vector[i])])

every2_thing_plot = []
for i in range(0, len(every2_thing_vector)):
    every2_thing_plot.append([float(every2_thing_vector[i])])

every3_thing_plot = []
for i in range(0, len(every3_thing_vector)):
    every3_thing_plot.append([float(every3_thing_vector[i])])

every4_thing_plot = []
for i in range(0, len(every4_thing_vector)):
    every4_thing_plot.append([float(every4_thing_vector[i])])

##################################################################
##################################################################
##################################################################
# draw some charts


plt.plot(every1_thing_plot, 'b+-')
plt.plot(every2_thing_plot, 'ro-')
plt.legend(labels=('Supply', 'Demand'), loc='upper right')

sringe = f'minimum liquidity ratio = {alpha_vector[ind_alpha]} \n CAR = {car_vector[ind_car]}'
plt.figtext(0.5, 0.9,
            sringe,
            horizontalalignment="center",
            wrap=True, fontsize=10,
            bbox={'facecolor': 'grey',
                  'alpha': 0.3, 'pad': 5})
plt.xlabel("time")
plt.ylabel("Supply and Demand of Securities")
plt.show()

plt.plot(every_thing_plot, 'ro-')
plt.plot(every3_thing_plot, 'b--')
plt.xlabel("time")
plt.ylabel("Price in Capital Market")
plt.legend(labels=('Price', 'Intrinsic Value'), loc='lower left')
plt.show()

plt.plot(every4_thing_plot, 'ro-')
plt.xlabel("time")
plt.ylabel("Interbank Interest Rate")
plt.show()

#######################
# determining of bank type

lender_vector = []
investor_vector = []
high_leveraged_vector = []
all_banks_types = []


def determiner(www):
    a1 = rfree
    a2 = www.ret_on_sec
    a3 = ((rfree * 1) / (1 - www.zeta * www.pd))
    if a2 < a1 < a3:
        www.bank_type = 1
    elif a1 < a2 < a3:
        www.bank_type = 2
    elif a1 < a3 < a2:
        www.bank_type = 3


for i in all_banks:
    determiner(i)

    if i.bank_type == 1:
        lender_vector.append(str(i.name))

    elif i.bank_type == 2:
        investor_vector.append(str(i.name))

    elif i.bank_type == 3:
        high_leveraged_vector.append(str(i.name))

    all_banks_types.append(i.bank_type)

bank_type_zip = list(zip(all_banks, all_banks_types))

##################################################
# proportion of each type of all ---- PIE CHART

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('equal')
types1 = ['Lender', 'Investor', 'High_Leveraged']
banks1 = [len(lender_vector), len(investor_vector), len(high_leveraged_vector)]
ax.pie(banks1, labels=types1, autopct='%1.2f%%', colors=['#ff9999', '#66b3ff', '#99ff99'],
       explode=(0.05, 0.05, 0.05))

plt.show()

#####################################################################
# type of banks
# for i in all_banks:
#     print(i.ns)
for i in all_shadow_banks:
    print(i.ns_s)
    print(i.nd_s)
#####################################################################
