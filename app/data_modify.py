# data_modify.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from data_merge import merge_data

def transform_data():
    """
    DBから連結したデータを取得し、以下の処理を実施する：
      1. 回答データをワイドフォーマットに変換（pivot_table）
      2. reception_id ごとに一意なユーザー・接客情報を抽出し、ワイドデータとマージ
      3. カラム名をリネーム（質問番号と回答種別から、意味のある名称へ）
      4. カテゴリカルな列のエンコーディング（ラベルエンコーディングおよびワンホットエンコーディング）
      5. 時刻情報の分解（年、月、日、時間、曜日）および不要な時刻カラムの削除
    """
    # DBから連結済みデータを取得
    df_merged = merge_data()

    # 回答部分のワイドフォーマット作成（answerカラムをピボット対象とする）
    df_wide_format = df_merged.pivot_table(
        index="reception_id",
        columns="question_id",
        values="answer",
        aggfunc="first"
    )
    # カラム名を "q{question_id}_answer" 形式に変換
    df_wide_format.columns = [f"q{question_id}_answer" for question_id in df_wide_format.columns]
    df_wide_format.reset_index(inplace=True)

    # reception_id ごとに一意なユーザー・接客情報を抽出
    df_reception_user = df_merged.drop_duplicates("reception_id")[
        ["reception_id", "store_id", "age", "gender", "household", "user_time", "reception_category_id", "reception_time"]
    ]
    # ワイドフォーマットとマージ（キー：reception_id）
    df_wide_format = pd.merge(df_wide_format, df_reception_user, on="reception_id", how="left")

    # カラム名のリネーム（質問内容に合わせた意味のある名称へ）
    rename_map = {
        "q1_answer":  "room_layout",            # 1. 部屋の間取り
        "q2_answer":  "pet_room",               # 2. ペット同伴
        "q3_answer":  "has_carpet",             # 3. カーペット・ラグ有無
        "q4_answer":  "app_features",           # 4. アプリ機能
        "q5_answer":  "mop_clean",              # 5. 水拭き機能
        "q6_answer":  "dust_station",           # 6. ダストステーション
        "q7_answer":  "prefer_quiet",           # 7. 静音優先
        "q8_answer":  "thorough_cleaning",      # 8. 隅々まで掃除
        "q9_answer":  "fully_automated",        # 9. 完全自動化
        "q10_answer": "clean_when_away",        #10. 外出中清掃
        "q11_answer": "reduce_maintenance"      #11. メンテナンス削減
    }
    # 辞書を用いて列名を置き換え
    df_renamed = df_wide_format.rename(columns=rename_map)

    # エンコーディング処理
    df_encoded = df_renamed.copy()
    # ラベルエンコーディング：間取り（q1_room_layout）
    le = LabelEncoder()
    # 欠損値を 'missing' とし、すべて文字列型に統一してからエンコーディング
    df_encoded["room_layout"] = df_encoded["room_layout"].fillna("missing").astype(str)
    df_encoded["room_layout"] = le.fit_transform(df_encoded["room_layout"])

    # ワンホットエンコーディング：アプリ機能（q4_app_features）
    df_encoded = pd.get_dummies(
        df_encoded,
        columns=["app_features"],
        prefix="app_features",
        dummy_na=True,
        drop_first=True
    )

    #  gender をワンホットエンコーディング
    df_encoded["gender"] = df_encoded["gender"].fillna("missing")
    df_encoded = pd.get_dummies(
        df_encoded,
        columns=["gender"],
        prefix="gender",
        dummy_na=True,    # 欠損値も別カテゴリとして扱う
        drop_first=True   # 多重共線性を避けるため、最初のカテゴリを削除
    )
    df_encoded.rename(
        columns={
            "gender_2.0": "gender_M",
            "gender_3.0": "gender_F",
            "gender_nan": "gender_missing"
        },
        inplace=True
    )


    # 時刻情報の分解
    # timeカラムをdatetime型に変換
    df_encoded["reception_time"] = pd.to_datetime(df_encoded["reception_time"])
    # 年、月、日、時間のカラムを作成
    df_encoded["year"] = df_encoded["reception_time"].dt.year
    df_encoded["month"] = df_encoded["reception_time"].dt.month
    df_encoded["day"] = df_encoded["reception_time"].dt.day
    df_encoded["hour"] = df_encoded["reception_time"].dt.hour
    # 新たに曜日のカラムも追加
    df_encoded["weekday"] = df_encoded["reception_time"].dt.day_name()

    # 曜日をワンホットエンコーディング（drop_firstで多重共線性防止）
    df_encoded = pd.get_dummies(
        df_encoded,
        columns=["weekday"],
        prefix="weekday",
        drop_first=True
    )

    # ワンホットエンコーディングした列を整数型に変換
    for col in df_encoded.columns:
        if col.startswith(("app_features", "weekday_", "gender_")):
            df_encoded[col] = df_encoded[col].astype(int)

    # ブール列を数値に変換.True が 1 に、False が 0 に
    bool_cols = ["pet_room", "has_carpet", "mop_clean", "dust_station", "prefer_quiet"]
    for col in bool_cols:
        df_encoded[col] = df_encoded[col].astype('Int64')


    # 不要な時刻カラムの削除（reception_time と user_time）
    df_encoded.drop(columns=["reception_time", "user_time"], inplace=True)
    # 処理件数をコンソールに出力（データ本体は出力しない）
    print(f"transform_data 実行完了: レコード数={len(df_encoded)} 件、列数={len(df_encoded.columns)}")

    # タブ1の商品名のフィルタリングにはreception_category_idが必要なため、idの削除前にコピーをしておく
    df_encoded_with_id = df_encoded.copy()

    # id類（user_id、category_id、reception_id）はリーケージの原因となりえるため、すべて削除する
    df_encoded = df_encoded.drop(columns=["user_id", "reception_category_id", "reception_id"], errors="ignore")
   
    return df_encoded, df_encoded_with_id


