# sales_call_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_sales_call_time(df: pd.DataFrame):
    """
    sales_call と reception を JOIN した DataFrame(df) から
    呼び出し時刻の曜日・時間帯ごとのヒートマップを表示する
    """
    # call_time を datetime に変換
    df["call_time"] = pd.to_datetime(df["call_time"], errors="coerce")
    
    # 曜日名と時間を列に追加
    df["call_weekday"] = df["call_time"].dt.day_name()  # 例: Monday, Tuesday, ...
    df["call_hour"] = df["call_time"].dt.hour          # 0～23
    
    # 集計: 曜日×時間帯 の呼び出し数
    # 例: size() を使って件数をカウント
    pivot_data = df.groupby(["call_weekday", "call_hour"]).size().unstack(fill_value=0)
    
    # 曜日を順序付け (月～日など)
    pivot_data = pivot_data.reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    
    # ヒートマップを描画
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(pivot_data, cmap="Reds", annot=True, fmt="d", ax=ax)
    ax.set_title("店員呼び出し数 (曜日×時間帯)")
    ax.set_xlabel("時間帯 (hour)")
    ax.set_ylabel("曜日 (weekday)")
    
    plt.tight_layout()
    plt.show()
    return fig
