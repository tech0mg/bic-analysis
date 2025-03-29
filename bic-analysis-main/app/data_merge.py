# data_merge.py
import pandas as pd
from db import SessionLocal
from models import User, Reception, Answer_info

def merge_data():
    """
    DBから各テーブル(User, Reception, Answer_info)のデータを取得し、
    answer_info と reception を左外部結合し、さらに reception.user_id と user.id をキーに左外部結合する。
    結果をpandas DataFrameとして返す。
    """
    session = SessionLocal()
    try:
        # Answer_info を基点に、Reception と User を外部結合（LEFT JOIN）する
        query = session.query(Answer_info, Reception, User) \
            .outerjoin(Reception, Answer_info.reception_id == Reception.id) \
            .outerjoin(User, Reception.user_id == User.id) \
            .all()
        
        rows = []
        for answer, reception, user in query:
            row = {}
            # answer_info のデータ
            row["answer_id"] = answer.id
            row["answer_reception_id"] = answer.reception_id
            row["question_id"] = answer.question_id
            row["answer_numeric"] = answer.answer_numeric
            row["answer_boolean"] = answer.answer_boolean
            row["answer_categorical"] = answer.answer_categorical
            
            # reception のデータ（存在すれば）
            if reception:
                row["reception_id"] = reception.id
                row["reception_user_id"] = reception.user_id
                row["reception_category_id"] = reception.category_id
                row["reception_time"] = reception.time
            else:
                row["reception_id"] = None
                row["reception_user_id"] = None
                row["reception_category_id"] = None
                row["reception_time"] = None
            
            # user のデータ（存在すれば）
            if user:
                row["user_id"] = user.id
                row["store_id"] = user.store_id
                row["age"] = user.age
                row["gender"] = user.gender
                row["household"] = user.household
                row["user_time"] = user.time
            else:
                row["user_id"] = None
                row["store_id"] = None
                row["age"] = None
                row["gender"] = None
                row["household"] = None
                row["user_time"] = None

            rows.append(row)
        
        df_merged = pd.DataFrame(rows)
        return df_merged
    except Exception as e:
        print(f"データ連結中にエラーが発生しました: {e}")
        return pd.DataFrame()
    finally:
        session.close()
    