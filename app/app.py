import streamlit as st
# ページをワイドレイアウトに設定
st.set_page_config(layout="wide")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import seaborn as sns

from data_access import get_data
from options import get_store_options, get_category_options
from data_merge import merge_data
from data_modify import transform_data
from ml_model import get_correlation_heatmap, get_vif, get_tsne_plot, meanshift_clustering, evaluate_random_forest_classifier,get_age_analysis_plots
from chatgpt import interpret_grouped_data
from sales_call_merge import merge_sales_call_and_reception
from sales_call_analysis import analyze_sales_call_time
from auth_utils import login_page, register_store

mpl.rcParams['font.family'] = 'AppleGothic'
mpl.rcParams['axes.unicode_minus'] = False

# ===== ログインチェック =====
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# サイドバーにログイン/新規登録の選択肢を追加
if not st.session_state.get("logged_in", False):
    auth_option = st.sidebar.radio("認証オプション", ("ログイン", "新規登録"))

if not st.session_state["logged_in"]:
    if auth_option == "ログイン":
        login_page()
    else:
        register_store()
    st.stop()  # ログイン状態になっていない場合はここで処理を終了
# ===== ログイン済みの場合 =====


st.title("接客データ分析アプリ")

# 店舗選択 (名称表示からID取得)
store_dict = get_store_options()  # {name: id}
store_names = list(store_dict.keys())
selected_store_name = st.sidebar.selectbox("店舗を選択", store_names)
store = store_dict.get(selected_store_name)

# カテゴリ選択 (名称表示からID取得)
category_dict = get_category_options()  # {name: id}
category_names = list(category_dict.keys())
selected_category_name = st.sidebar.selectbox("家電カテゴリを選択", category_names)
category = category_dict.get(selected_category_name)

# ログイン済みの場合、サイドバーにログアウトボタンを表示
if st.session_state.get("logged_in", False):
    if st.sidebar.button("ログアウト"):
        st.session_state["logged_in"] = False
        try:
            st.rerun()
        except Exception as e:
            st.error("ページの再実行がサポートされていません。ブラウザの再読み込みをしてください。")

# タブを用意
tabs = st.tabs(["データダッシュボード", "判断軸分析", "店員呼び出し分析", "機械学習", "回答のみクラスタリング"])


# タブ1: データダッシュボード
with tabs[0]:
    st.header("データダッシュボード")
    # 生データ取得
    df_raw = merge_data()
    # グラフを3列に並べて表示
    cols = st.columns(3)
    # 性別比率
    with cols[0]:
        st.subheader("性別比率")
        gender_counts = df_raw['gender'].fillna('Unknown').value_counts()
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        # ColorBrewerのdivergingパレットを適用
        colors_gender = sns.color_palette("RdBu", len(gender_counts))
        ax2.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=colors_gender)
        ax2.axis('equal')
        st.pyplot(fig2, use_container_width=True)
    # 年齢層比率
    with cols[1]:
        st.subheader("年齢層比率")
        bins = [0,20,30,40,50,60,100]
        labels = ['<20','20-29','30-39','40-49','50-59','60+']
        age_group = pd.cut(df_raw['age'].fillna(-1), bins=bins, labels=labels)
        age_counts = age_group.value_counts().loc[labels]
        fig3, ax3 = plt.subplots(figsize=(4, 4))
        # ColorBrewerのdivergingパレットを適用
        colors_age = sns.color_palette("RdBu", len(age_counts))
        ax3.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%', startangle=90, colors=colors_age)
        ax3.axis('equal')
        st.pyplot(fig3, use_container_width=True)
    # 曜日別回答数
    with cols[2]:
        st.subheader("曜日別回答数")
        df_raw['weekday'] = pd.to_datetime(df_raw['reception_time']).dt.day_name()
        wd_counts = df_raw['weekday'].value_counts().reindex(
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        ).fillna(0)
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        # ColorBrewerのdivergingパレットを適用
        colors_week = sns.color_palette("RdBu", len(wd_counts))
        wd_counts.plot(kind='bar', ax=ax4, color=colors_week)
        ax4.set_xlabel('Weekday')
        ax4.set_ylabel('回答数')
        st.pyplot(fig4, use_container_width=True)
    st.success("ダッシュボード表示完了")


# タブ2: 判断軸分析（自動実行＆GPT結果表示）
with tabs[1]:
    # 自動分析を実行し、GPTのレポートのみ表示
    df_enc, df_enc_id = transform_data()
    df_filt = df_enc_id[(df_enc_id["store_id"] == store) & (df_enc_id["reception_category_id"] == category)]
    df_model = df_filt.drop(columns=["user_id","reception_category_id","reception_id"], errors="ignore")
    _, _, _, df_cl = meanshift_clustering(df_model)
    df_cl["cluster_ms"] = "Cluster_" + df_cl["cluster_ms"].astype(str)
    _, _, grouped, _, _, _ = get_age_analysis_plots(df_cl)
    # GPTレポート生成
    try:
        report = interpret_grouped_data(grouped)
        st.write(report)
    except Exception as e:
        st.error(f"GPTレポート生成中にエラーが発生しました: {e}")


# タブ3: 店員呼び出し分析
with tabs[2]:
    if st.button("店員呼び出しのヒートマップ分析"):
        df_merged = merge_sales_call_and_reception(store, category)
        if df_merged.empty:
            st.error("sales_call と reception を JOIN した結果がありません。")
        else:
            fig = analyze_sales_call_time(df_merged)
            st.pyplot(fig)

# タブ4: 機械学習
with tabs[3]:
    st.header("機械学習")
    
    st.write("データのマージ結果")
    df_merged = merge_data()
    st.dataframe(df_merged.head())

    st.write("データのエンコーディング結果")
    df_encoded, df_encoded_with_id = transform_data() # タプルで取得
    st.dataframe(df_encoded.head())


    if st.button("相関ヒートマップの描画を実行"):
        try:
            df_encoded, df_encoded_with_id = transform_data()
            st.write("取得したデータ（先頭5行）:", df_encoded.head())
            if df_encoded is None or df_encoded.empty:
                st.error("機械学習用のデータがありません。")
            else:
                st.session_state.heatmap_fig = get_correlation_heatmap(df_encoded)
        except Exception as e:
            st.error(f"相関ヒートマップの描画中にエラーが発生しました: {e}")
    if "heatmap_fig" in st.session_state:
         st.pyplot(st.session_state.heatmap_fig)
    

    # 多重共線性（VIF）の測定ボタン
    if st.button("多重共線性（VIF）を測定"):
        try:
            df_numeric = df_encoded.select_dtypes(include=[np.number]).dropna().astype(float)
            if df_numeric.empty:
                st.error("数値カラムが存在しないか、欠損値のみのためVIFを計算できません。")
            else:
                st.session_state.vif_df = get_vif(df_numeric)
        except Exception as e:
            st.error(f"多重共線性（VIF）計算中にエラーが発生しました: {e}")
    if "vif_df" in st.session_state:
         st.write("多重共線性（VIF）計算結果:")
         st.dataframe(st.session_state.vif_df)

    # t-SNE の散布図を描画するボタン
    if st.button("t-SNE の散布図を描画"):
        try:
            st.session_state.tsne_fig = get_tsne_plot(df_encoded)
        except Exception as e:
            st.error(f"t-SNE の散布図を描画中にエラーが発生しました: {e}")
    if "tsne_fig" in st.session_state:
         st.pyplot(st.session_state.tsne_fig)

    # MeanShift（教師なし学習）によるクラスタリングを実施し、t-SNEの散布図にクラスタごとに色分け
    # 並行座標プロットを表示する
    if st.button("MeanShiftクラスタリングを実行"):
        try:
            tsne_ms_fig, parallel_fig, n_clusters_ms, df_cluster = meanshift_clustering(df_encoded)
            
            # クラスタラベルに "Cluster_" を付ける
            df_cluster["cluster_ms"] = df_cluster["cluster_ms"].astype(str)
            df_cluster["cluster_ms"] = "Cluster_" + df_cluster["cluster_ms"]
            
            st.session_state.tsne_ms_fig = tsne_ms_fig
            st.session_state.parallel_fig = parallel_fig
            st.session_state.n_clusters_ms = n_clusters_ms
            st.session_state.df_for_cluster = df_cluster
        except Exception as e:
            st.error(f"MeanShiftクラスタリング中にエラーが発生しました: {e}")
    if ("tsne_ms_fig" in st.session_state and 
        "n_clusters_ms" in st.session_state and
        "parallel_fig" in st.session_state):
         st.write(f"MeanShiftの推定クラスタ数: {st.session_state.n_clusters_ms}")
         st.pyplot(st.session_state.tsne_ms_fig)
         st.pyplot(st.session_state.parallel_fig)

    # MeanShiftのクラスタレベルを疑似的な「目的変数」として分類モデルを作成、疑似的どの特徴量がクラスタ分割に寄与しているか」を可視化
    st.write("モデルの作成はMeanShiftクラスタリングの実行後にクリックしてください")
    if  st.button("モデルの作成（ランダムフォレスト）"):
        try:
            if "df_for_cluster" not in st.session_state:
                st.error("MeanShiftクラスタリングが実行されていません。")
            else:
                # MeanShiftクラスタリングで作成した df_for_cluster を利用
                df_cluster = st.session_state.df_for_cluster
                # st.write(df_cluster)

                # クラスタラベル以外の列を特徴量として使用（不要なクラスタ列が存在する場合は除外）
                X = df_cluster.drop(columns=["cluster_ms"], errors="ignore").copy()
                y = df_cluster["cluster_ms"].astype(str)
                
                fig_cm, class_rep, feature_importances = evaluate_random_forest_classifier(X, y)
                # セッションステートに保存
                st.session_state.fig_cm = fig_cm
                st.session_state.class_rep = class_rep
                st.session_state.feature_importances = feature_importances

                # 年齢分析のプロットを取得して表示
                cluster_age_stats, fig_box, grouped, corr_matrix, fig_corr, cluster_age_corr_df = get_age_analysis_plots(df_cluster)
                st.session_state.cluster_age_stats = cluster_age_stats
                st.session_state.fig_box = fig_box
                st.session_state.grouped = grouped
                st.session_state.corr_matrix = corr_matrix
                st.session_state.fig_corr = fig_corr
                st.session_state.cluster_age_corr_df = cluster_age_corr_df

        except Exception as e:
            st.error(f"ランダムフォレスト評価中にエラーが発生しました: {e}")

    # すでにセッションステートに保存されている場合は、以下で表示
    if "fig_cm" in st.session_state:
        st.pyplot(st.session_state.fig_cm)
        st.text(st.session_state.class_rep)
        st.dataframe(st.session_state.feature_importances)

    if "cluster_age_stats" in st.session_state:
        st.write("各クラスタの年齢統計:")
        st.dataframe(st.session_state.cluster_age_stats)
        st.pyplot(st.session_state.fig_box)
        st.write("年齢層ごとのアンケート結果と年齢の平均:")
        st.dataframe(st.session_state.grouped)
        st.write("相関行列:")
        st.dataframe(st.session_state.corr_matrix)
        st.pyplot(st.session_state.fig_corr)
        st.write("各クラスタにおける年齢と他因子の相関:")
        st.dataframe(st.session_state.cluster_age_corr_df)
    

    # # GPTによる相関行列分析の実行ボタン
    # st.write("GPTによる分析はモデルの作成実行後にクリックしてください")
    # if st.button("GPTによる相関行列分析を実行"):
    #     try:
    #         if "corr_matrix" not in st.session_state:
    #             st.error("相関行列が見つかりません。先にモデル作成を実行してください。")
    #         else:
    #             interpretation = interpret_corr_matrix(st.session_state.corr_matrix)
    #             st.write("ChatGPT の解釈:")
    #             st.write(interpretation)
    #     except Exception as e:
    #         st.error(f"GPTによる分析中にエラーが発生しました: {e}")

    # GPTによる "grouped" 分析の実行ボタン
    st.write("GPTによる grouped データ分析はモデルの作成実行後にクリックしてください")
    if st.button("GPTによる grouped データ分析を実行"):
        try:
            if "grouped" not in st.session_state:
                st.error("groupedデータが見つかりません。先にモデル作成を実行してください。")
            else:
                from chatgpt import interpret_grouped_data
                interpretation = interpret_grouped_data(st.session_state.grouped)
                st.write("ChatGPT の解釈:")
                st.write(interpretation)
        except Exception as e:
            st.error(f"GPTによる分析中にエラーが発生しました: {e}")

# ===== タブ5: 回答のみクラスタリング =====
with tabs[4]:
    st.header("回答のみクラスタリング")
    st.write("顧客属性（年齢・世帯人数など）を除き、回答データのみでクラスタリングを行います。")
    if st.button("回答のみクラスタリングを実行"):
        # データ取得
        df_encoded, df_encoded_with_id = transform_data()
        # 属性列を除外して回答データのみ抽出
        df_resp = df_encoded.drop(columns=["store_id", "age", "household"], errors="ignore")
        # クラスタリング実行
        with st.spinner("クラスタリング実行中..."):
            tsne_fig_resp, parallel_fig_resp, n_clusters_resp, df_cluster_resp = meanshift_clustering(df_resp)
        # クラスタラベル整形
        df_cluster_resp["cluster_resp"] = "Cluster_" + df_cluster_resp["cluster_ms"].astype(str)
        # 可視化
        st.subheader(f"推定クラスタ数: {n_clusters_resp}")
        st.pyplot(tsne_fig_resp)
        st.pyplot(parallel_fig_resp)
        # 属性プロファイルとクラスタラベルをインデックスベースで結合
        df_profile = df_encoded_with_id.copy()
        df_profile = df_profile.merge(
            df_cluster_resp[['cluster_resp']],
            left_index=True,
            right_index=True,
            how='left'
        )
        # 属性プロファイル（クラスタ後に年齢・世帯人数を確認）
        st.subheader("クラスタごとの年齢・世帯人数平均")
        df_profile_summary = df_profile.groupby("cluster_resp")[['age','household']].mean()
        st.dataframe(df_profile_summary)
    else:
        st.info("クラスタリングを実行するにはボタンをクリックしてください。")


