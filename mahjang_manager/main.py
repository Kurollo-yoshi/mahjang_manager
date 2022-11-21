## 月一友人戦成績管理
## ------------------------------------------------------------------------------------------------------------
import json
from re import A
import sqlite3
import datetime
import hashlib
from operator import itemgetter
from PIL import Image

import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from deta import Deta

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

from mahjong.shanten import Shanten
#麻雀牌
from mahjong.tile import TilesConverter

## Config
## ------------------------------------------------------------------------------------------------------------
# ファイルパス
# DB
dbname = "mahjang_manager/01_data/mahjang.db"
# ヘッダー画像
header_image = "mahjang_manager/01_data/header.jpg"
# アイコン画像
icon_image = "mahjang_manager/01_data/icon.png"

# アプリ名
app_title = "友人戦成績管理アプリ"

# プレイヤー名
player_1 = "Kurollo"
player_2 = "Tamasuke"
player_3 = "ルチチ"
player_4 = "紅花さん"

# DB上の名前の定義
name_list = ['Kurollo', 'Tamasuke', 'ルチチ', '紅花さん']

# アプリ機能
mode_1 = "全期間集計"
mode_2 = "単月集計"
mode_3 = "入力"
mode_4 = "管理"

# グラフサイズ
# 円グラフ
circle_size = 300
# 線グラフ
chart_height = 300
chart_width = 450

# 詳細データのカラム
detail_columns = [
    'Kurollo_放銃回数', 'Tamasuke_放銃回数', 'ルチチ_放銃回数', '紅花さん_放銃回数',
    'Kurollo_放銃合計', 'Tamasuke_放銃合計', 'ルチチ_放銃合計', '紅花さん_放銃合計',
    '局数',
    'Kurollo_副露回数', 'Tamasuke_副露回数', 'ルチチ_副露回数', '紅花さん_副露回数',
    'Kurollo_配牌', 'Tamasuke_配牌', 'ルチチ_配牌', '紅花さん_配牌',
    'Kurollo_和了回数', 'Tamasuke_和了回数', 'ルチチ_和了回数', '紅花さん_和了回数',
    'Kurollo_和了合計', 'Tamasuke_和了合計', 'ルチチ_和了合計', '紅花さん_和了合計'
]

# 各詳細の目安データ
limit_dict = {
	"和了率(%)":[22,24],
	"副露率(%)":[20,40],
	"平均打点":[5500,6500],
	"放銃率(%)":[10,13],
	"平均放銃":None,
	"配牌向聴":None
}

## Function
## ------------------------------------------------------------------------------------------------------------

# 表示関係
# ------------------------------------------------------------------------------------------------------------
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
                x = [i for i in range(len(rank_df.index))],
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
                x = [i for i in range(len(score_value.index))],
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

def display_deteil(detail_dataframe):
    """詳細結果の表示 & グラフ表示
    """
    sum_dataframe = create_detail(detail_dataframe)
    sum_dataframe["データ項目"] = sum_dataframe.index
    sum_dataframe = sum_dataframe[["データ項目"] + name_list]

    gb = GridOptionsBuilder.from_dataframe(sum_dataframe)
    gb.configure_selection(selection_mode="single", use_checkbox=True,pre_selected_rows=[0])
    gb.configure_default_column(min_column_width=5)
#     gb.configure_column(header_name="データ項目",min_column_width=20)
#     gb.configure_column(header_name="Kurollo",min_column_width=10)
#     gb.configure_column(header_name="Tamasuke",min_column_width=10)
#     gb.configure_column(header_name="ルチチ",min_column_width=10)
#     gb.configure_column(header_name="紅花さん",min_column_width=10)
    gridOptions = gb.build()
    data = AgGrid(
        sum_dataframe,
        gridOptions=gridOptions,
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        theme="dark",
        data_return_mode=DataReturnMode.AS_INPUT,
        height=250
        #fit_columns_on_grid_load=True
    )
    if len(data["selected_rows"])>0:
        sel_data = data["selected_rows"][0]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(name=sel_data["データ項目"], x=name_list, y=[sel_data["Kurollo"],sel_data["Tamasuke"],sel_data["ルチチ"],sel_data["紅花さん"]])
        )
        if limit_dict[sel_data["データ項目"]]:
            fig.add_hrect(
                y0=limit_dict[sel_data["データ項目"]][0],
                y1=limit_dict[sel_data["データ項目"]][1],
                fillcolor = "green",
                opacity=0.5,
                line=dict(width=0,color=None),
                layer="above"
        )
        fig.update_layout(
                yaxis=dict(title=sel_data["データ項目"]),
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
        st.plotly_chart(fig, config=dict({'displaylogo': False}))

def display_func(display_dataframe,detail_dataframe, all=True):
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
    if all:
        # 詳細データを表示
        display_deteil(detail_dataframe)

# データ操作
# ------------------------------------------------------------------------------------------------------------
def create_detail(dataframe):
    """詳細データのdataframeを作成
    """
    # 和了率
    win_num_data = dataframe[['Kurollo_和了回数', 'Tamasuke_和了回数', 'ルチチ_和了回数', '紅花さん_和了回数']]
    win_rate = (win_num_data.sum()*100/dataframe["局数"].sum()).round().values
    # 副露率
    meld_num_data = dataframe[['Kurollo_副露回数', 'Tamasuke_副露回数', 'ルチチ_副露回数', '紅花さん_副露回数']]
    meld_rate = (meld_num_data.sum()*100/dataframe["局数"].sum()).round().values
    # 平均配牌シャンテン数
    start_num_data = dataframe[['Kurollo_配牌', 'Tamasuke_配牌', 'ルチチ_配牌', '紅花さん_配牌']]
    start_num = (start_num_data.sum()/dataframe["局数"].sum()).round(2).values
    # 平均打点
    win_score_data = dataframe[['Kurollo_和了合計', 'Tamasuke_和了合計', 'ルチチ_和了合計', '紅花さん_和了合計']]
    win_score = (win_score_data.sum().values/win_num_data.sum().values).round()
    # 放銃率
    deal_num_data = dataframe[['Kurollo_放銃回数', 'Tamasuke_放銃回数', 'ルチチ_放銃回数', '紅花さん_放銃回数']]
    deal_rate = (deal_num_data.sum()*100/dataframe["局数"].sum()).round().values
    # 平均放銃
    deal_score_data = dataframe[['Kurollo_放銃合計', 'Tamasuke_放銃合計', 'ルチチ_放銃合計', '紅花さん_放銃合計']]
    dael_score = (deal_score_data.sum().abs().values/deal_num_data.sum().values).round()
    detail = pd.DataFrame(
                        [win_rate, meld_rate, win_score, deal_rate, dael_score, start_num],
                        index=["和了率(%)","副露率(%)","平均打点","放銃率(%)","平均放銃","配牌向聴"],
                        columns=name_list
                    )
    detail = detail.loc[["和了率(%)","副露率(%)","平均打点","放銃率(%)","平均放銃","配牌向聴"]]
    return detail

def get_some_data(db):
    # 表示に必要な情報を取得(機能ごとに分けたほうがいいかも)
    data_lsit = []
    detail_list = []
    for tmp_data in db.fetch().items:
        data_lsit.append(tmp_data["result_point"]+[tmp_data["date"]]+[tmp_data["key"]])
        detail_list.append(tmp_data["deal_num"]+tmp_data["deal_sum"]+[tmp_data["game_num"]]+tmp_data["meld_num"]+tmp_data["start_sum"]+tmp_data["win_num"]+tmp_data["win_sum"])

    df_all_data = pd.DataFrame(data_lsit)
    df_all_data.columns = name_list + ["Date","key"]
    df_all_data["Date"] = pd.to_datetime(df_all_data["Date"])
    df_all_data = df_all_data.sort_values("Date",ascending=True)

    df_detail = pd.DataFrame(detail_list)
    df_detail.columns = detail_columns

    # データの最大、最小時間
    raw_min_date = df_all_data["Date"].min()
    raw_max_date = df_all_data["Date"].max()
    return df_all_data, df_detail, raw_min_date, raw_max_date

def pai_num2name(pai_num):
    """牌番号を牌名に変換
    """
    pai_str = str(pai_num)
    reach = False
    if "r" in pai_str: # リーチ?
        reach = True
        pai_num.replace("r","")
    pai_name = pai_dict[pai_str]
    return reach, pai_name

def calc_win_num(data_json):
    """個人和了回数を取得
    """
    log_data = data_json["log"]
    # 初期配列
    win_count_sum = np.array([0,0,0,0])
    for tmp_data in log_data:
        win_data = tmp_data[16]
        if win_data[0] == "和了": # 和了のときのみ
            win_num = [1 if i>0 else 0 for i in win_data[1]] # 点数移動→和了回数のリストに変換
            win_count_sum = win_count_sum + np.array(win_num)
    return win_count_sum.tolist()

def calc_win_sum(data_json):
    """和了合計得点を取得
    """
    log_data = data_json["log"]
    # 初期配列
    win_count_sum = np.array([0,0,0,0])
    for tmp_data in log_data:
        win_data = tmp_data[16]
        if win_data[0] == "和了": # 和了のときのみ
            win_num = [i if i>0 else 0 for i in win_data[1]] # 点数移動→和了回数のリストに変換
            win_count_sum = win_count_sum + np.array(win_num)
    return win_count_sum.tolist()

def calc_meld_num(data_json):
    """副露回数を取得
    """
    master_meld_arr = np.array([0,0,0,0])
    for game_log in data_json["log"]: # 対局ごと
        check_meld_arr = np.array([0,0,0,0])
        for i, player in enumerate([5,8,11,14]): # プレイヤーごと
            tmp_tumo_data = game_log[player]
            for tmp_tumo in tmp_tumo_data: # ツモごと
                if type(tmp_tumo) is str: # ツモに副露がある場合
                    check_meld_arr[i] = 1
        master_meld_arr = master_meld_arr + check_meld_arr
    return master_meld_arr.tolist()

def calc_deal_num(data_json):
    """個人放銃回数を算出
    """
    log_data = data_json["log"]
    # 初期配列
    deal_count_sum = np.array([0,0,0,0])
    for tmp_data in log_data:
        win_data = tmp_data[16]
        if len(win_data)!=1: # 九種九牌
            # 和了のときのみ & ロンは点数変動が二人のとき
            if (win_data[0] == "和了")&((np.array(win_data[1])==0).sum()==2):
                deal_num_list = [1 if i<0 else 0 for i in win_data[1]]
                deal_count_sum = deal_count_sum + deal_num_list
    return deal_count_sum.tolist()

def calc_deal_sum(data_json):
    """個人合計放銃点を算出
    """
    log_data = data_json["log"]
    # 初期配列
    deal_count_sum = np.array([0,0,0,0])
    for tmp_data in log_data:
        win_data = tmp_data[16]
        if len(win_data)!=1: # 九種九牌
            # 和了のときのみ & ロンは点数変動が二人のとき
            if (win_data[0] == "和了")&((np.array(win_data[1])==0).sum()==2):
                deal_num_list = [i if i<0 else 0 for i in win_data[1]]
                deal_count_sum = deal_count_sum + deal_num_list
    return deal_count_sum.tolist()

def calc_start_sum(data_json):
    """配牌シャンテン数の合計を算出
    """
    shanten = Shanten()
    master_shanten_arr = np.array([0,0,0,0])
    for tmp_log in data_json["log"]:
        for i, player in enumerate([4,7,10,13]):
            tmp_tahei = tmp_log[player]
            pai_str_list = [pai_num2name(i)[1] for i in tmp_tahei]
            pai_num_list  = ["","","",""]
            for hai in pai_str_list:
                for j, kind in enumerate(["m", "p", "s", "z"]):
                    if kind in hai:
                        hai_num = hai.replace(kind,"")
                        pai_num_list[j] = pai_num_list[j] + str(hai_num)
            # シャンテン数を算出
            tiles = TilesConverter.string_to_34_array(man=pai_num_list[0], pin=pai_num_list[1], sou=pai_num_list[2],honors=pai_num_list[3])
            result = shanten.calculate_shanten(tiles)
            master_shanten_arr[i] = master_shanten_arr[i] + result
    return master_shanten_arr.tolist()

def sort_match(name_list, data_list):
    """ データの順番を統一化
    """
    arg_index = np.argsort(name_list)
    sorted_data = np.array(data_list)[arg_index]
    return sorted_data.tolist()

def reshape_data(data_json):
    """jsonファイルから情報を取得
    """
    try:
        # 対局者名(席順で東南西北のリスト)
        player_name = data_json["name"]
        # 対局日時(YYYY/MM/DD/ HH:mm:ss)
        date_str = data_json["title"][1]
        # 対局結果(player_nameの順序で[素点,ポイント, .....]のリスト)
        result_data = data_json["sc"]
        result_score = list(itemgetter(0,2,4,6)(result_data))
        result_point = list(itemgetter(1,3,5,7)(result_data))
        # 局数
        game_num = len(data_json["log"])
        # 個人和了回数
        win_num = calc_win_num(data_json)
        # 和了合計得点
        win_sum = calc_win_sum(data_json)
        # 個人副露回数
        meld_num = calc_meld_num(data_json)
        # 個人放銃回数
        deal_num = calc_deal_num(data_json)
        # 個人放銃合計得点
        deal_sum = calc_deal_sum(data_json)
        # 配牌シャンテン数合計
        start_sum = calc_start_sum(data_json)

        # データの並び順を統一させる ['Kurollo', 'Tamasuke', 'ルチチ', '紅花さん']
        result_score = sort_match(player_name, result_score)
        result_point = sort_match(player_name, result_point)
        win_num = sort_match(player_name, win_num)
        win_sum = sort_match(player_name, win_sum)
        meld_num = sort_match(player_name, meld_num)
        deal_num = sort_match(player_name, deal_num)
        deal_sum = sort_match(player_name, deal_sum)
        start_sum = sort_match(player_name, start_sum)

        result_dict = {
            "date"          : date_str,     # 対局日時(YYYY/MM/DD/ HH:mm:ss)
            "game_num"      : game_num,     # 局数
            "result_score"  : result_score, # 対局結果(素点) [player1, player2, player3, player4]
            "result_point"  : result_point, # 対局結果(ポイント) [player1, player2, player3, player4]
            "win_num"       : win_num,      # 個人和了回数
            "win_sum"       : win_sum,      # 和了合計得点
            "meld_num"      : meld_num,     # 個人副露回数(1局の中でないたかどうか)
            "deal_num"      : deal_num,     # 個人放銃回数
            "deal_sum"      : deal_sum,     # 個人放銃合計得点
            "start_sum"     : start_sum,    # 配牌シャンテン数合計
        }
        return result_dict, player_name
    except Exception as e:
        raise

# 通知
# ------------------------------------------------------------------------------------------------------------
def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    if "send" not in st.session_state:
        st.session_state.send = False
    if st.session_state.send:
        return True
    line_notify_token = 'A4aDm0dKREXD9ydI3n6lSCmweLqnilW5qJYCMD1Zctf'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)
    st.session_state.send = True

# 認証
# ------------------------------------------------------------------------------------------------------------
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

## Main
## -------------------------------------------------------------------------------
st.set_page_config(
    page_title=app_title,
    page_icon=icon_image,
    layout="wide"
)
send_line_notify("システムが起動しました")

# pai_dictの生成(pickleで読み込んだほうがいい)
# 麻雀牌と数字の対応辞書を作成
pai_dict = {}
man_list = [str(i)+"m" for i in range(1,10)]
pin_list = [str(i)+"p" for i in range(1,10)]
sou_list = [str(i)+"s" for i in range(1,10)]
zih_list = [str(i)+"z" for i in range(1,8)]
# マンズ
for i, tmp_hai in enumerate(man_list): # マンズ
    pai_dict = {**pai_dict, **{str(i+11):tmp_hai}}
for i, tmp_hai in enumerate(pin_list): # ピンズ
    pai_dict = {**pai_dict, **{str(i+21):tmp_hai}}
for i, tmp_hai in enumerate(sou_list): # ソウズ
    pai_dict = {**pai_dict, **{str(i+31):tmp_hai}}
for i, tmp_hai in enumerate(zih_list): # 字牌
    pai_dict = {**pai_dict, **{str(i+41):tmp_hai}}
for i, tmp_hai in enumerate(["5m","5p","5s"]): # 赤牌
    pai_dict = {**pai_dict, **{str(i+51):tmp_hai}}
#
pai_dict = {**pai_dict, **{str(60):None}}


# Title
st.title(app_title)
header_img = Image.open(header_image)
st.image(header_img,use_column_width=True)

# Detaに接続
deta = Deta(st.secrets["deta_key"])
db = deta.Base("mahjang_manager_db")

df_all_data, df_detail, raw_min_date, raw_max_date = get_some_data(db)

# Select Mode
mode = st.selectbox("機能選択",[mode_1,mode_2,mode_3, mode_4])
try:
    if mode==mode_1: # 1日集計
        # グラフを表示
        display_func(df_all_data, df_detail)

    elif mode==mode_2: # 全期間集計
        df_date = pd.to_datetime(df_all_data["Date"]).dt.strftime("%Y-%m")
        date_list = df_date.drop_duplicates().sort_values(ascending=False)
        start_data = st.selectbox("年月を選択",date_list,index=0)
        # 日時を元にDBからデータを取得
        display_dataframe = df_all_data[df_date==start_data][name_list+["Date"]]
        # グラフを表示
        display_func(display_dataframe, df_detail, False)

    elif mode==mode_3: # 入力
        if login_func():
            load_file = st.file_uploader("ファイルアップロード", type='json')
            if load_file:
                jan_data = json.load(load_file)
                if st.button("アップロード"):
                    try:
                        tmp_dict, player_name = reshape_data(jan_data)
                        assert name_list==sorted(player_name)
                        db.put(tmp_dict)
                        st.success("データが登録されました")
                        send_line_notify("データが登録されました")
                        send_line_notify(tmp_dict)
                    except:
                        st.warning("入力値が不正です")

    elif mode==mode_4: # 入力済みの対局データを取得
        if login_func():
            gb = GridOptionsBuilder.from_dataframe(df_all_data, editable=True)
            gb.configure_selection(selection_mode="multiple", use_checkbox=True)
            gb.configure_pagination()
            gridOptions = gb.build()
            st.write("データを編集する際は変更後にチェックをいれること")
            data = AgGrid(
                df_all_data,
                gridOptions=gridOptions,
                enable_enterprise_modules=True,
                allow_unsafe_jscode=True,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                theme="dark",
                data_return_mode=DataReturnMode.AS_INPUT
            )

            selection_data = data["selected_rows"]

            if st.button("削除"):
                send_line_notify("データが削除されました")
                for i in range(len(selection_data)):
                    db.delete(selection_data[i]["key"])
                    send_line_notify(selection_data[i])
                st.success("データが削除されました")

except Exception as e:
    raise
