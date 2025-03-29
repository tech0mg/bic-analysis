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

    # 回答部分のワイドフォーマット作成
    # ※answer_numeric と answer_categorical, answer_boolean をピボット対象とする
    
    df_wide_format = df_merged.pivot_table(
        index="reception_id",
        columns="question_id",
        values=["answer_numeric", "answer_boolean", "answer_categorical"],
        aggfunc="first"
    )
    # カラム名を "q{question_id}_{回答タイプ}" 形式に変換
    df_wide_format.columns = [f"q{col[1]}_{col[0]}" for col in df_wide_format.columns]
    df_wide_format.reset_index(inplace=True)

    # reception_id ごとに一意なユーザー・接客情報を抽出
    df_reception_user = df_merged.drop_duplicates("reception_id")[
        ["reception_id", "store_id", "age", "gender", "household", "user_time", "reception_category_id", "reception_time"]
    ]
    # ワイドフォーマットとマージ（キー：reception_id）
    df_wide_format = pd.merge(df_wide_format, df_reception_user, on="reception_id", how="left")

    # カラム名のリネーム
    rename_map = {
        "q1_answer_categorical":  "room_layout",            # (1) ロボット掃除機を使いたい部屋の間取りを教えてください。
        "q2_answer_boolean":      "pet_room",               # (2) ペットを飼っている部屋でロボット掃除機を使いたいですか？
        "q3_answer_boolean":      "carpet_rug",             # (3) お部屋にはカーペットやラグがありますか？
        "q4_answer_categorical":  "app_features",           # (4) ロボット掃除機のアプリで欲しい機能はどれですか？
        "q5_answer_boolean":      "wipe_cleaning",          # (5) 拭き掃除もロボット掃除機に任せたいですか？
        "q6_answer_boolean":      "auto_dustbin",           # (6) 自動ゴミ収集機能が欲しいですか？
        "q7_answer_boolean":      "quiet_mode",             # (7) 吸引力が落ちても静かに掃除して欲しい
        "q8_answer_numeric":      "less_collision",         # (8) 細かい場所の掃除制度は下がっても衝突回数が減る方がいい
        "q9_answer_numeric":      "fully_automated",        # (9) 掃除はロボット掃除機に全て任せたい
        "q10_answer_numeric":     "detailed_suction",       # (10) 部屋全体・部屋の隅・壁際などの細かい場所でもしっかり吸引してほしい？
        "q11_answer_numeric":     "finish_when_away",       # (11) 家にいる時に掃除してくれるよりも、外出中に掃除が終わっている方がいい
        "q12_answer_numeric":     "maintenance_reduction"   # (12) 多少高くてもメンテナンスの手間が減るほどいい
    }
    # 辞書を用いて列名を置き換え
    df_renamed = df_wide_format.rename(columns=rename_map)

    # エンコーディング処理
    df_encoded = df_renamed.copy()
    # ラベルエンコーディング：間取り（q1_room_layout）
    le = LabelEncoder()
    # 欠損値がある場合は、一旦文字列に変換して埋める
    df_encoded["room_layout"] = df_encoded["room_layout"].fillna("missing")
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
    bool_cols = ["pet_room", "carpet_rug", "wipe_cleaning", "auto_dustbin", "quiet_mode"]
    for col in bool_cols:
        df_encoded[col] = df_encoded[col].astype('Int64')


    # 不要な時刻カラムの削除（reception_time と user_time）
    df_encoded.drop(columns=["reception_time", "user_time"], inplace=True)
    print(df_encoded.head(5))

    # タブ1の商品名のフィルタリングにはreception_category_idが必要なため、idの削除前にコピーをしておく
    df_encoded_with_id = df_encoded.copy()

    # id類（user_id、category_id、reception_id）はリーケージの原因となりえるため、すべて削除する
    df_encoded = df_encoded.drop(columns=["user_id", "reception_category_id", "reception_id"], errors="ignore")
   
    return df_encoded, df_encoded_with_id


