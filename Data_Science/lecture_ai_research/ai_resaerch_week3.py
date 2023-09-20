from datetime import datetime, timedelta

dt = (datetime.today() - timedelta(1)).strftime('%Y%m%d')
jcode = '081'
headers = {'user-agent' : 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'}

url = f'https://media.naver.com/press/{jcode}/ranking?type=popular%date={dt}'

print(url)


