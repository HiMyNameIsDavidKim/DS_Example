import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

WELFARE_META = {'h14_g3': 'sex',
                'h14_g4': 'birth',
                'h14_g10': 'marriage_type',
                'h14_g11': 'religion',
                'p1402_8aq1': 'income',
                'h14_eco9': 'code_job',
                'h14_reg7': 'code_region'}


class Welfare(object):
    def __init__(self):
        raw_welfare = pd.read_spss('./data/Koweps_hpwc14_2019_beta2.sav')
        self.welfare = raw_welfare.copy()

    def process(self):
        self.spec()
        self.rename()
        self.plot_sex_income()
        self.plot_age_income()

    def spec(self):
        print('###### SPEC ######')
        print('###### shape ######')
        print(self.welfare.shape)
        print('###### info ######')
        print(self.welfare.info())
        print('###### dtypes ######')
        print(self.welfare.dtypes)
        print('###### describe ######')
        print(self.welfare.describe())

    def rename(self):
        print('###### RENAME ######')
        welfare = self.welfare
        welfare = welfare[[i for i in list(WELFARE_META.keys())]]
        welfare = welfare.rename(columns=WELFARE_META)
        print('###### Rename is completed ######')
        print('###### dtypes ######')
        print(welfare.dtypes)
        self.welfare = welfare

    def plot_sex_income(self):
        welfare = self.welfare
        welfare['sex'] = np.where(welfare['sex'] == 1, 'male', 'female')
        sns.countplot(data=welfare, x='sex').set_title('sex')
        plt.show()

        sns.histplot(data=welfare, x='income').set_title('income')
        plt.show()

        sex_income = welfare.dropna(subset=['income'])\
            .groupby('sex', as_index=False) \
            .agg(mean_income=('income', 'mean'))
        sns.barplot(data=sex_income, x='sex', y='mean_income')\
            .set_title('sex & mean of income')
        plt.show()
        self.welfare = welfare

    def plot_age_income(self):
        welfare = self.welfare
        sns.histplot(data=welfare, x='birth').set_title('birth')
        plt.show()

        welfare = welfare.assign(age=2019 - welfare['birth'] + 1)
        sns.histplot(data=welfare, x='age').set_title('age')
        plt.show()

        age_income = welfare.dropna(subset=['income']) \
            .groupby('age') \
            .agg(mean_income=('income', 'mean'))
        sns.lineplot(data=age_income, x='age', y='mean_income')\
            .set_title('age & mean of income')
        plt.show()
        self.welfare = welfare

    def plot_ages_income(self):
        welfare = self.welfare
        welfare = welfare.assign(ageg=np.where(welfare['age'] < 30, 'young',
                                               np.where(welfare['age'] <= 59, 'middle', 'old')))
        sns.countplot(data=welfare, x='ageg')


welfare_menu = ["Exit", # 0
                "Spec", # 1
                "Rename", # 2
                "plot_1", # 3
                "plot_2", # 4
]
welfare_lambda = {
    "1" : lambda t: t.process(),
    "2" : lambda t: t.rename(),
    "3" : lambda t: t.plot_sex_income(),
    "4" : lambda t: t.plot_age_income(),
    "5" : lambda t: print(" ** No Function ** "),
    "6" : lambda t: print(" ** No Function ** "),
    "7" : lambda t: print(" ** No Function ** "),
    "8" : lambda t: print(" ** No Function ** "),
    "9" : lambda t: print(" ** No Function ** "),
}


if __name__ == '__main__':
    w = Welfare()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(welfare_menu)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                welfare_lambda[menu](w)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")