import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据文件
df = pd.read_csv(r"C:\Users\user_some_one\Downloads\dy_action.csv")

print('\n=====字段信息=====')
print(df.info())

print('\n=====前五行预览=====')
print(df.head())

print('\n查看各列空值情况')
print(df.isna().sum())

print('\n查看重复行数情况',df.duplicated().sum())
df = df.drop_duplicates()

print('\n点赞取值分布情况')
print(df['like_type'].value_counts())

print('\n转发取值分布')
print(df['relay_type'].value_counts())

# 删除多余索引列
df = df.drop(columns='Unnamed: 0')

# 转换时间列数据格式为时间格式
df['time'] = pd.to_datetime(df['time'])

# 衍生时间维度
df['hour'] = df['time'].dt.hour
df['weekday'] = df['time'].dt.weekday
df['is_workday'] =np.where (df['time'].dt.weekday<5,1,0)

# 最终数据确认
print('\n数据预处理完成')
print('清洗结束数据形状：',df.shape)
print('空值',df.isna().sum().sum())
print('重复行',df.duplicated().sum())

# 分析1：全天用户活跃度时段分布
plt.figure(figsize=(12,5))
act_hour = df['hour'].value_counts().sort_index()
plt.plot(act_hour.index,act_hour.values,marker='o',c='#27ae60')
plt.title('24小时短视频用户活跃度分布')
plt.xlabel('时间')
plt.ylabel('用户观看次数')
plt.grid(True,alpha=0.3)
plt.show()


# 分析2：工作日vs周末观看热度差异
work_data = df.groupby(['is_workday','hour']).size().unstack(fill_value=0)
plt.figure(figsize=(12,5))
work_data.loc[0].plot(label='周末',c='#e74c3c')
work_data.loc[1].plot(label='工作日',c='#3498db')
plt.title('周末vs工作日不同时段短视频观看热度')
plt.legend()
plt.grid(True,alpha=0.3)
plt.show()


# 分析3：各类型视频排行
plt.figure(figsize=(12,5))
cate_rank = df['video_category'].value_counts().sort_values().head(10)
print(cate_rank)
cate_rank.plot(kind='barh',color='#9b59b6')
plt.title('短视频播放热度排行前十的类型')
# plt.tight_layout()
plt.show()


# 分析4：各类型视频点赞率、转发率对比
cate_behavior = df.groupby('video_category').agg(总播放=('user_id','count'),
                                                 点赞率=('like_type','mean'),
                                                 转发率=('relay_type','mean')).sort_values('点赞率')
print(cate_behavior)

plt.figure(figsize=(12,5))
cate_behavior[['点赞率','转发率']].plot(kind='barh',color=["#e74c3c", "#3498db"],ax=plt.gca())
plt.title('各类视频点赞转发率')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# 分析5：整体平台互动大盘数据
total_like_rate = df['like_type'].mean()
total_relay_rate = df['relay_type'].mean()

print(f'平台整体点赞率：{total_like_rate:.2%}')
print(f'平台整体转发率：{total_relay_rate:.2%}')

# 分析6：用户活跃分层
user_play_count = df.groupby("user_id").size().sort_values(ascending=False)
user_play_count = user_play_count.rename("观看视频数")

# 分层标准
def classify_user(count):
    if count >= 120:
        return "重度用户"
    elif count >= 60:
        return "中度用户"
    else:
        return "轻度用户"

user_type = user_play_count.apply(classify_user)
user_type_count = user_type.value_counts()

print("\n==== 用户活跃度分层 ====")
print(user_type_count)

# 画图
plt.figure(figsize=(6, 6))
user_type_count.plot(kind='pie', autopct='%1.1f%%', colors=['#ff6b6b','#4ecdc4','#95e1d3'])
plt.title('用户活跃度分层')
plt.ylabel('')
plt.show()
