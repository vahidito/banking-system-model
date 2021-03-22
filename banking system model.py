import numpy as np
import mesa as mesa
import pandas as pd
import matplotlib as plt
#import sci
# defining the agents: banks, shadow banks, savers, loans

class Bank:
    def __init__(self, bank_cash, lend_to_banks, lend_to_loans, bank_sec, reserves, borrow_from_banks, CAR, provision_per):
        # balance sheet
        self.bank_cash = bank_cash
        self.lend_to_banks = lend_to_banks
        self.lend_to_loans = lend_to_loans
        self.bank_sec = bank_sec
        self.reserves = reserves
        self.borrow_from_banks = borrow_from_banks
        self.equity = bank_cash + lend_to_banks + lend_to_loans + bank_sec - reserves - borrow_from_banks
        self.CAR = CAR
        self.provision_per = provision_per
        ##### income and expense



class Shadow_Bank:
    def __init__(self, participation, shadow_bank_cash):
        self.participation = participation
        self.shadow_bank_cash = shadow_bank_cash
        self.security = participation - shadow_bank_cash

# introduction of Iranian Banks

bank_melli = Bank(0,0,0,0,0,0,0,0)
bank_seppah = Bank(0,0,0,0,0,0,0,0)
bank_tosesaderat = Bank(0,0,0,0,0,0,0,0)
bank_maskan = Bank(0,0,0,0,0,0,0,0)
bank_sanatmadan = Bank(0,0,0,0,0,0,0,0)
bank_keshavarzi = Bank(0,0,0,0,0,0,0,0)
bank_tosetavon = Bank(0,0,0,0,0,0,0,0)
bank_post = Bank(0,0,0,0,0,0,0,0)
bank_eghtesadnovin = Bank(0,0,0,0,0,0,0,0)
bank_parsian = Bank(0,0,0,0,0,0,0,0)
bank_karafarin = Bank(0,0,0,0,0,0,0,0)
bank_saman = Bank(0,0,0,0,0,0,0,0)
bank_sina = Bank(0,0,0,0,0,0,0,0)
bank_khavarmiane = Bank(0,0,0,0,0,0,0,0)
bank_shahr = Bank(0,0,0,0,0,0,0,0)
bank_dey = Bank(0,0,0,0,0,0,0,0)
bank_saderat = Bank(0,0,0,0,0,0,0,0)
bank_tejarat = Bank(0,0,0,0,0,0,0,0)
bank_mellat = Bank(0,0,0,0,0,0,0,0)
bank_refah = Bank(0,0,0,0,0,0,0,0)
bank_ayandeh = Bank(0,0,0,0,0,0,0,0)
bank_gardeshgary = Bank(0,0,0,0,0,0,0,0)
bank_iranzamin = Bank(0,0,0,0,0,0,0,0)
bank_sarmaye = Bank(0,0,0,0,0,0,0,0)
bank_pasargad = Bank(0,0,0,0,0,0,0,0)
bank_melal = Bank(0,0,0,0,0,0,0,0)


# optimixation phase