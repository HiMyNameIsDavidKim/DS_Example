import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

WELFARE_META = {'h14_g3': 'sex',
                'h14_g4': 'birth',
                'h14_g10': 'marriage_type',
                'h14_g11': 'religion',
                'p1402_8aq1': 'income',
                'h14_eco9': 'code_job',
                'h14_reg7': 'code_region'}


class Welfare(object):
    def __init__(self):
        global raw_welfare, list_job
        raw_welfare = pd.read_spss('./data/Koweps_hpwc14_2019_beta2.sav')
        list_job = pd.read_excel('./data/Koweps_Codebook_2019.xlsx', sheet_name='직종코드')
        self.welfare = raw_welfare.copy()
        self.welfare = self.welfare[[i for i in list(WELFARE_META.keys())]]
        self.welfare = self.welfare.rename(columns=WELFARE_META)

    def reset_data(self):
        self.welfare = raw_welfare.copy()
        self.welfare = self.welfare[[i for i in list(WELFARE_META.keys())]]
        self.welfare = self.welfare.rename(columns=WELFARE_META)

    def process(self):
        self.spec()
        self.rename()

    def spec(self):
        print('###### SPEC ######')
        print('###### shape ######')
        print(raw_welfare.shape)
        print('###### info ######')
        print(raw_welfare.info())
        print('###### dtypes ######')
        print(raw_welfare.dtypes)
        print('###### describe ######')
        print(raw_welfare.describe())

    def rename(self):
        print('###### RENAME ######')
        welfare = self.welfare
        self.welfare = self.welfare[[i for i in list(WELFARE_META.keys())]]
        self.welfare = self.welfare.rename(columns=WELFARE_META)
        print('###### Rename is completed ######')
        print('###### dtypes ######')
        print(welfare.dtypes)

    def plot_sex_i(self):
        self.reset_data()
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

    def plot_age_i(self):
        self.reset_data()
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

    def plot_ages_i(self):
        self.reset_data()
        welfare = self.welfare
        welfare = welfare.assign(age=2019 - welfare['birth'] + 1)
        welfare = welfare.assign(ages=np.where(welfare['age'] < 30, 'young',
                                               np.where(welfare['age'] <= 59, 'middle', 'old')))
        sns.countplot(data=welfare, x='ages').set_title('ages')
        plt.show()

        ages_income = welfare.dropna(subset=['income']) \
            .groupby('ages', as_index=False) \
            .agg(mean_income=('income', 'mean'))
        sns.barplot(data=ages_income, x='ages', y='mean_income', order=['young', 'middle', 'old'])\
            .set_title('ages & mean of income')
        plt.show()

    def plot_ages_sex_i(self):
        self.reset_data()
        welfare = self.welfare
        welfare = welfare.assign(age=2019 - welfare['birth'] + 1)
        welfare = welfare.assign(ages=np.where(welfare['age'] < 30, 'young',
                                               np.where(welfare['age'] <= 59, 'middle', 'old')))

        sex_ages = welfare.dropna(subset=['income']) \
            .groupby(['ages', 'sex'], as_index=False) \
            .agg(mean_income=('income', 'mean'))
        sns.barplot(data=sex_ages, x='ages', y='mean_income', hue='sex',
                    order=['young', 'middle', 'old'])\
            .set_title('ages & sex')
        plt.show()

        sex_age = welfare.dropna(subset=['income']) \
            .groupby(['age', 'sex'], as_index=False) \
            .agg(mean_income=('income', 'mean'))
        sns.lineplot(data=sex_age, x='age', y='mean_income', hue='sex')\
            .set_title('age & sex')
        plt.show()

    def plot_job_i(self):
        self.reset_data()
        welfare = self.welfare
        welfare = welfare.merge(list_job, how='left', on='code_job')

        job_income = welfare.dropna(subset=['job', 'income']) \
            .groupby('job', as_index=False) \
            .agg(mean_income=('income', 'mean'))
        top10 = job_income.sort_values('mean_income', ascending=False).head(10)
        sns.barplot(data=top10, y='job', x='mean_income').set_title('top10 jobs')
        plt.show()

    def plot_job_sex_i(self):
        self.reset_data()
        welfare = self.welfare
        welfare['sex'] = np.where(welfare['sex'] == 1, 'male', 'female')
        welfare = welfare.merge(list_job, how='left', on='code_job')

        job_male = welfare.dropna(subset=['job']) \
            .query('sex == "male"') \
            .groupby('job', as_index=False) \
            .agg(n=('job', 'count')) \
            .sort_values('n', ascending=False) \
            .head(10)
        sns.barplot(data=job_male, y='job', x='n').set(xlim=[0, 500], title='male jobs')
        plt.show()

        job_female = welfare.dropna(subset=['job']) \
            .query('sex == "female"') \
            .groupby('job', as_index=False) \
            .agg(n=('job', 'count')) \
            .sort_values('n', ascending=False) \
            .head(10)
        sns.barplot(data=job_female, y='job', x='n').set(xlim=[0, 500], title='female jobs')
        plt.show()

    def plot_religion_d(self):
        self.reset_data()
        welfare = self.welfare
        welfare['religion'] = np.where(welfare['religion'] == 1, 'yes', 'no')
        welfare['marriage'] = np.where(welfare['marriage_type'] == 1, 'marriage',
                                       np.where(welfare['marriage_type'] == 3, 'divorce', 'etc'))

        rel_div = welfare.query('marriage != "etc"') \
            .groupby('religion', as_index=False) \
            ['marriage'] \
            .value_counts(normalize=True)
        rel_div = rel_div.query('marriage == "divorce"') \
            .assign(proportion=rel_div['proportion'] * 100) \
            .round(1)
        sns.barplot(data=rel_div, x='religion', y='proportion').set_title('religion & divorce')
        plt.show()

    def plot_ages_d(self):
        self.reset_data()
        welfare = self.welfare
        welfare = welfare.assign(age=2019 - welfare['birth'] + 1)
        welfare = welfare.assign(ages=np.where(welfare['age'] < 30, 'young',
                                               np.where(welfare['age'] <= 59, 'middle', 'old')))
        welfare['marriage'] = np.where(welfare['marriage_type'] == 1, 'marriage',
                                       np.where(welfare['marriage_type'] == 3, 'divorce', 'etc'))

        age_div = welfare.query('marriage != "etc"') \
            .groupby('ages', as_index=False) \
            ['marriage'] \
            .value_counts(normalize=True)
        age_div = age_div.query('ages != "young" & marriage == "divorce"') \
            .assign(proportion=age_div['proportion'] * 100) \
            .round(1)
        sns.barplot(data=age_div, x='ages', y='proportion').set_title('ages & divorce')
        plt.show()

    def plot_ages_religion_d(self):
        self.reset_data()
        welfare = self.welfare
        welfare = welfare.assign(age=2019 - welfare['birth'] + 1)
        welfare = welfare.assign(ages=np.where(welfare['age'] < 30, 'young',
                                               np.where(welfare['age'] <= 59, 'middle', 'old')))
        welfare['religion'] = np.where(welfare['religion'] == 1, 'yes', 'no')
        welfare['marriage'] = np.where(welfare['marriage_type'] == 1, 'marriage',
                                       np.where(welfare['marriage_type'] == 3, 'divorce', 'etc'))

        age_rel_div = welfare.query('marriage != "etc" & ages != "young"') \
            .groupby(['ages', 'religion'], as_index=False) \
            ['marriage'] \
            .value_counts(normalize=True)
        age_rel_div = age_rel_div.query('marriage == "divorce"') \
            .assign(proportion=age_rel_div['proportion'] * 100) \
            .round(1)
        sns.barplot(data=age_rel_div, x='ages', y='proportion', hue='religion').set_title('ages & religion & divorce')
        plt.show()

    def plot_region_ages(self):
        self.reset_data()
        welfare = self.welfare
        welfare = welfare.assign(age=2019 - welfare['birth'] + 1)
        welfare = welfare.assign(ages=np.where(welfare['age'] < 30, 'young',
                                               np.where(welfare['age'] <= 59, 'middle', 'old')))

        list_region = pd.DataFrame({'code_region': [1, 2, 3, 4, 5, 6, 7],
                                    'region': ['서울',
                                               '수도권(인천/경기)',
                                               '부산/경남/울산',
                                               '대구/경북',
                                               '대전/충남',
                                               '강원/충북',
                                               '광주/전남/전북/제주도']})
        welfare = welfare.merge(list_region, how='left', on='code_region')
        region_ages = welfare.groupby('region', as_index=False) \
            ['ages'] \
            .value_counts(normalize=True)
        region_ages = region_ages.assign(proportion=region_ages['proportion'] * 100) \
            .round(1)
        sns.barplot(data=region_ages, y='region', x='proportion', hue='ages')\
            .set_title('region & ages')
        plt.show()

        pivot_df = region_ages[['region', 'ages', 'proportion']].pivot(index='region',
                                                                       columns='ages',
                                                                       values='proportion')
        reorder_df = pivot_df.sort_values('old')[['young', 'middle', 'old']]
        reorder_df.plot.barh(stacked=True)\
            .set_title('region & ages')
        plt.show()


welfare_menu = ["Exit", # 0
                "Spec", # 1
                "Rename", # 2
                "plot_sex_income", # 3
                "plot_age_income", # 4
                "plot_ages_income",  # 5
                "plot_ages_sex_income",  # 6
                "plot_job_income",  # 7
                "plot_job_sex_income",  # 8
                "plot_religion_divorce",  # 9
                "plot_ages_divorce",  # 10
                "plot_ages_religion_divorce",  # 11
                "plot_region_ages",  # 11
                ]

welfare_lambda = {
    "1" : lambda t: t.spec(),
    "2" : lambda t: t.rename(),
    "3" : lambda t: t.plot_sex_i(),
    "4" : lambda t: t.plot_age_i(),
    "5" : lambda t: t.plot_ages_i(),
    "6" : lambda t: t.plot_ages_sex_i(),
    "7" : lambda t: t.plot_job_i(),
    "8" : lambda t: t.plot_job_sex_i(),
    "9" : lambda t: t.plot_religion_d(),
    "10": lambda t: t.plot_ages_d(),
    "11": lambda t: t.plot_ages_religion_d(),
    "12": lambda t: t.plot_region_ages(),
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