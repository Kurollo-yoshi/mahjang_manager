## 月一友人戦成績管理
## -------------------------------------------------------------------------------
import sqlite3
import datetime
import hashlib
from PIL import Image

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from deta import Deta

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

## Config
## -------------------------------------------------------------------------------
# ファイルパス
# DB
dbname = "mahjang_manager/01_data/mahjang.db"
# ヘッダー画像
header_image = "mahjang_manager/01_data/head.webp"
# アイコン画像
icon_image = "mahjang_manager/01_data/icon.png"

# アプリ名
app_title = "友人戦成績管理アプリ"

# プレイヤー名
player_1 = "紅花さん"
player_2 = "ルチチ"
player_3 = "Tamasuke"
player_4 = "Kurollo"

# DB上の名前の定義
name_list = ["紅花さん","ルチチ","Tamasuke","Kurollo"]

# アプリ機能
mode_1 = "全期間集計"
mode_2 = "一日集計"
mode_3 = "入力"
mode_4 = "管理"

# グラフサイズ
# 円グラフ
circle_size = 300
# 線グラフ
chart_height = None
chart_width = 700

## Function
## -------------------------------------------------------------------------------
def select2dataframe(start, end):
    """データベースからデータを取得し、dataframeに格納
    """
    conn = sqlite3.connect(dbname)
    dataframe = pd.read_sql(sql_select, conn, params=[str(start), str(end)])
    conn.close()
    return dataframe

def circle_graph(dataframe):
    """順位率の円グラフを作成
    """
    # データの順位を算出
    rank_list = ["一位","二位","三位","四位"]
    rank_df = dataframe[name_list].rank(axis=1, ascending=False).astype("int")
    # グラフの作成
    for i, tmp_columns in enumerate(st.columns(4)):
        # 個人順位の集計
        rank_1st = (rank_df[name_list[i]] == 1).sum()
        rank_2nd = (rank_df[name_list[i]] == 2).sum()
        rank_3rd = (rank_df[name_list[i]] == 3).sum()
        rank_4th = (rank_df[name_list[i]] == 4).sum()
        # グラフの表示
        tmp_columns.subheader(name_list[i])
        fig = go.Figure()
        fig.add_trace(
            go.Pie(
                labels = rank_list,
                values = [rank_1st, rank_2nd, rank_3rd, rank_4th],
                sort=False,
            )
        )
        fig.update_traces(
            textinfo = 'label+percent'
        )
        fig.update_layout(
            height=circle_size,
            width=circle_size,
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
            showlegend=False,
        modebar_remove=[
                'toImage' # 画像ダウンロード
        ]
        )
        tmp_columns.plotly_chart(fig, config=dict({'displaylogo': False}))

def chart_graph(dataframe):
    """折れ線グラフを表示
    """
    col2, col1 = st.columns(2)
    col1.subheader("対戦記録(順位)")
    fig = go.Figure()
    rank_df = dataframe[name_list].rank(axis=1, ascending=False).astype("int")
    for i, tmp_name in enumerate(name_list):
        fig.add_trace(
            go.Scatter(
                x = rank_df.index,
                y = rank_df[tmp_name],
                name = name_list[i],
                marker = dict(
                    line=dict(width=3)
                )
            )
        )
    fig.update_layout(
        height=chart_height,
        width=chart_width,
	autosize=True,
        plot_bgcolor = "#202020",
        xaxis=dict(dtick=5),
        yaxis=dict(title="順位",dtick=1,autorange='reversed'),
        modebar_remove=[
                'toImage',  # 画像ダウンロード
                'zoom2d',  # ズームモード
                'pan2d',  # 移動モード
                'select2d',  # 四角形で選択
                'lasso2d',  # ラッソで選択
                'zoomIn2d',  # 拡大
                'zoomOut2d',  # 縮小
                'autoScale2d',  # 自動範囲設定
                'resetScale2d',  # 元の縮尺
        ]
    )
    # グリッドの調整
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='gray')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='gray')
    col1.plotly_chart(fig, config=dict({'displaylogo': False}))

    col2.subheader("対戦記録(ポイント)")
    fig = go.Figure()
    score_value = dataframe[name_list]
    cumsum_data = score_value.cumsum()
    for i, tmp_name in enumerate(name_list):
        fig.add_trace(
            go.Scatter(
                x = score_value.index,
                y = cumsum_data[tmp_name],
                name = name_list[i],
                marker = dict(
                    line=dict(width=3)
                )
            )
        )
    fig.update_layout(
        height=chart_height,
        width=chart_width,
	autosize=True,
        plot_bgcolor = "#202020",
        xaxis=dict(dtick=5),
        yaxis=dict(title="ポイント"),
        modebar_remove=[
                'toImage',  # 画像ダウンロード
                'zoom2d',  # ズームモード
                'pan2d',  # 移動モード
                'select2d',  # 四角形で選択
                'lasso2d',  # ラッソで選択
                'zoomIn2d',  # 拡大
                'zoomOut2d',  # 縮小
                'autoScale2d',  # 自動範囲設定
                'resetScale2d',  # 元の縮尺
        ]
    )
    # グリッドの調整
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='gray')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='gray')
    col2.plotly_chart(fig, config=dict({'displaylogo': False}))

def display_func(display_dataframe):
        # 順位と総得点を表示
        ranking_df = pd.DataFrame([display_dataframe.sum()[name_list]]).T
        ranking_df.columns = ["総得点"]
        ranking_df = ranking_df.sort_values("総得点", ascending=False)
        st.markdown("## 総合順位(対局数 {})".format(len(display_dataframe)))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("1st",ranking_df.index[0],round(ranking_df.iloc[0].values[0],3))
        col2.metric("2nd",ranking_df.index[1],round(ranking_df.iloc[1].values[0],3))
        col3.metric("3rd",ranking_df.index[2],round(ranking_df.iloc[2].values[0],3))
        col4.metric("4th",ranking_df.index[3],round(ranking_df.iloc[3].values[0],3))
        st.markdown("---")

        # 円グラフを表示
        st.markdown("## 順位分布")
        circle_graph(display_dataframe)
        st.markdown("---")
        # 折れ線グラフを表示
        chart_graph(display_dataframe)
        st.markdown("---")

# パスワードのハッシュ化
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

# パスワードの確認
def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

# ログイン
def login_user(username,password):
    conn = sqlite3.connect(dbname)
    cur = conn.cursor()
    cur.execute('SELECT * FROM USERTABLE WHERE username =? AND password = ?',(username,password))
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

def login_func():
    """ログイン機能
    """
    if "login" not in st.session_state:
        st.session_state.login = False
    username = st.text_input("ユーザー名を入力してください")
    password = st.text_input("パスワードを入力してください", type="password")

    if st.session_state.login:
        return True
    if st.button("ログイン"):
        hashed_pswd = make_hashes(password)
        result = login_user(username,check_hashes(password,hashed_pswd))
        if result:
            st.success("ログインしました")
            st.session_state.login = True
            return st.session_state.login
        else:
            st.success("ユーザー名かパスワードが間違っています")
            st.session_state.login = False
            return st.session_state.login

def get_some_data():
    # 表示に必要な情報を取得(機能ごとに分けたほうがいいかも)
    data_lsit = []
    for tmp_data in db.fetch().items:
        data_lsit.append(tmp_data["data"]+[tmp_data["date"]]+[tmp_data["key"]])

    df_all_data = pd.DataFrame(data_lsit)
    df_all_data.columns = name_list + ["Date","key"]

    # データの最大、最小時間
    raw_min_date = df_all_data["Date"].min()
    raw_max_date = df_all_data["Date"].max()
    return df_all_data, raw_min_date, raw_max_date

## Main
## -------------------------------------------------------------------------------
st.set_page_config(
    page_title=app_title,
    page_icon=icon_image,
    layout="wide"
)

# Title
st.title(app_title)
header_img = Image.open(header_image)
st.image(header_img,use_column_width=True)

# Detaに接続
deta = Deta(st.secrets["deta_key"])
db = deta.Base("mahjang_manager_db")

df_all_data, raw_min_date, raw_max_date = get_some_data()

# Select Mode
mode = st.selectbox("機能選択",[mode_1,mode_2,mode_3, mode_4])
try:
    if mode==mode_1: # 1日集計
        # グラフを表示
        display_func(df_all_data)

    elif mode==mode_2: # 全期間集計
        date_list = df_all_data["Date"].drop_duplicates()
        start_data = st.selectbox("日付を選択",date_list,index=len(date_list)-1)
        # 日時を元にDBからデータを取得
        display_dataframe = df_all_data[df_all_data["Date"]==start_data][name_list+["Date"]]
        # グラフを表示
        display_func(display_dataframe)

    elif mode==mode_3: # 入力
        if login_func():
                st.markdown("## 順位点を入力")
                player_1_value = st.number_input(name_list[0])
                player_2_value = st.number_input(name_list[1])
                player_3_value = st.number_input(name_list[2])
                player_4_value = st.number_input(name_list[3])
                sum_values = sum([player_1_value,player_2_value,player_3_value,player_4_value])
                if st.button("データを登録"):
                    if sum_values==0:
                        conn = sqlite3.connect(dbname)
                        cur = conn.cursor()
                        data = [str(datetime.datetime.now().date()), player_1_value, player_2_value, player_3_value, player_4_value]
                        cur.execute(spl_insert, data)
                        conn.commit()
                        cur.close()
                        conn.close()
                        st.success("データが登録されました")
                    else:
                        st.warning("入力値が不正です")

    elif mode==mode_4: # 入力済みの対局データを取得
        if login_func():
            # 日時を元にDBからデータを取得
            display_dataframe = select2dataframe(raw_start_date, raw_end_date).set_index("id")
            gb = GridOptionsBuilder.from_dataframe(display_dataframe, editable=True)
            gb.configure_selection(selection_mode="multiple", use_checkbox=True)
            gb.configure_pagination()
            gridOptions = gb.build()
            st.write("データを編集する際は変更後にチェックをいれること")	
            data = AgGrid(
                display_dataframe,
                gridOptions=gridOptions,
                enable_enterprise_modules=True,
                allow_unsafe_jscode=True,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                theme="dark",
                data_return_mode=DataReturnMode.AS_INPUT
            )

            selection_data = data["selected_rows"]
            if st.button("更新"):
                conn = sqlite3.connect(dbname)
                cur = conn.cursor()
                for i in range(len(selection_data)):
                    data = [selection_data[i]["ayaka"], selection_data[i]["rutiti"], selection_data[i]["tama"], selection_data[i]["kurollo"], selection_data[i]["rowIndex"]+1]
                    cur.execute(update_sql, data)
                conn.commit()
                cur.close()
                conn.close()
                st.success("情報が更新されました")

            if st.button("削除"):
                conn = sqlite3.connect(dbname)
                cur = conn.cursor()
                for i in range(len(selection_data)):
                    cur.execute(delete_sql, [selection_data[i]["rowIndex"]+1])
                conn.commit()
                cur.close()
                conn.close()
                st.success("データが削除されました")

except Exception as e:
    raise
