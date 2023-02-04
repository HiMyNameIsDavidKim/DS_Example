import numpy as np
import pandas as pd
from matplotlib import font_manager, rc, pyplot as plt
from scipy import stats
import seaborn as sns

font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


class StatExam(object):
    def __init__(self):
        self.data_mpg = pd.read_csv('./data/mpg.csv')
        self.data_economics = pd.read_csv('./data/economics.csv')
        self.data_mtcars = pd.read_csv('./data/mtcars.csv')

    def process(self):
        self.corr_mtrx()

    def t_test(self):
        mpg = self.data_mpg
        mpg.query('category in ["compact", "suv"]')\
            .groupby('category', as_index=False)\
            .agg(n=('category', 'count'), mean=('cty', 'mean'))

        compact = mpg.query('category == "compact"')['cty']
        suv = mpg.query('category == "suv"')['cty']

        ttest = stats.ttest_ind(compact, suv, equal_var=True)
        print(ttest)
        print('statistic : 검정 통계량, pvalue : p값, p값으로 유의여부 판단.')

    def corr_anal(self):
        economics = self.data_economics
        economics[['unemploy', 'pce']].corr()
        pce = stats.pearsonr(economics['unemploy'], economics['pce'])
        print(pce)
        print('statistic : 검정 통계량, pvalue : p값, p값으로 유의여부 판단.')

    def corr_mtrx(self):
        car_cor = self.data_mtcars.corr()
        car_cor = round(car_cor, 2)

        mask = np.zeros_like(car_cor)
        mask[np.triu_indices_from(mask)] = 1
        mask_new = mask[1:, :-1]
        cor_new = car_cor.iloc[1:, :-1]

        plt.rcParams['axes.unicode_minus'] = False

        sns.heatmap(data=cor_new,
                    annot=True,  # 상관계수 표시
                    cmap='RdBu',  # 컬러맵
                    mask=mask_new,  # mask 적용
                    linewidths=.5,  # 경계 구분선 추가
                    vmax=1,  # 가장 진한 파란색으로 표현할 최대값
                    vmin=-1,  # 가장 진한 빨간색으로 표현할 최소값
                    cbar_kws={'shrink': .5})  # 범례 크기 줄이기
        plt.show()


stat_menu = ["Exit",  # 0
                "T-test",  # 1
                "Correlation Analysis",  # 2
                "Correlation Matrix",  # 3
                ]

stat_lambda = {
    "1": lambda t: t.t_test(),
    "2": lambda t: t.corr_anal(),
    "3": lambda t: t.corr_mtrx(),
    "4": lambda t: print(" ** No Function ** "),
    "5": lambda t: print(" ** No Function ** "),
    "6": lambda t: print(" ** No Function ** "),
    "7": lambda t: print(" ** No Function ** "),
    "8": lambda t: print(" ** No Function ** "),
    "9": lambda t: print(" ** No Function ** "),
}

if __name__ == '__main__':
    sx = StatExam()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(stat_menu)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                stat_lambda[menu](sx)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")
