import re
import PIL
import konlpy.tag
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from wordcloud import WordCloud

font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


class CommentBTS(object):
    def __init__(self):
        self.comment = pd.read_csv('./data/news_comment_BTS.csv', encoding='UTF-8')
        self.df_word = pd.DataFrame([[]])
        self.path_cloud = f'./data/cloud.png'
        self.dic_word = {}

    def process(self):
        self.divide_word()
        self.plot_word_count()

    def divide_word(self):
        self.comment['reply'] = self.comment['reply'].str.replace('[^가-힣]', ' ', regex=True)

        kkma = konlpy.tag.Kkma()
        nouns = self.comment['reply'].apply(kkma.nouns)
        nouns = nouns.explode()

        df_word = pd.DataFrame({'word': nouns})
        print(df_word)
        self.df_word = df_word

    def cal_n(self):
        df_word = self.df_word
        df_word['count'] = df_word['word'].str.len()
        df_word = df_word.query('count >= 2')
        df_word = df_word.groupby('word', as_index=False) \
            .agg(n=('word', 'count')) \
            .sort_values('n', ascending=False)
        self.df_word = df_word
        self.dic_word = df_word.set_index('word').to_dict()['n']

    def plot_word_count(self):
        self.cal_n()

        top20 = self.df_word.head(20)
        sns.barplot(data=top20, y='word', x='n').set_title('word count')
        plt.show()

    def word_cloud(self):
        self.cal_n()

        icon = PIL.Image.open(self.path_cloud)
        img = PIL.Image.new('RGB', icon.size, (255, 255, 255))
        img.paste(icon, icon)
        img = np.array(img)

        wc = WordCloud(random_state=1234,  # 난수 고정
                       font_path=font,  # 폰트 설정
                       width=400,  # 가로 크기
                       height=400,  # 세로 크기
                       background_color='white',  # 배경색
                       mask=img)  # mask 설정
        img_wordcloud = wc.generate_from_frequencies(self.dic_word)
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(img_wordcloud)
        plt.show()


comment_menu = ["Exit",  # 0
                "Divide word(first)",  # 1
                "Plot word count",  # 2
                "Create word cloud",  # 3
                ]

comment_lambda = {
    "1": lambda t: t.divide_word(),
    "2": lambda t: t.plot_word_count(),
    "3": lambda t: t.word_cloud(),
    "4": lambda t: print(" ** No Function ** "),
    "5": lambda t: print(" ** No Function ** "),
    "6": lambda t: print(" ** No Function ** "),
    "7": lambda t: print(" ** No Function ** "),
    "8": lambda t: print(" ** No Function ** "),
    "9": lambda t: print(" ** No Function ** "),
}

if __name__ == '__main__':
    cb = CommentBTS()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(comment_menu)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                comment_lambda[menu](cb)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")
