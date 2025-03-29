import pandas as pd
from db import SessionLocal
from models import User, Reception

# 店舗情報を取得して表示させるための関数
def get_data(store, cat_id):
    session = SessionLocal()
    try:
        try:
            results = session.query(User, Reception)\
                .join(Reception, User.id == Reception.user_id)\
                .filter(User.store_id == store, Reception.category_id == cat_id)\
                .all()
        except Exception as query_error:
            print(f"クエリ実行中のエラー: {query_error}")
            return pd.DataFrame()  # クエリエラーの場合は空の DataFrame を返す
        
        rows = []
        for user, reception in results:
            try:
                rows.append({
                    "user_id": user.id,
                    "store_id": user.store_id,
                    "reception_id": reception.id,
                    "category_id": reception.category_id,
                    "time": reception.time
                })
            except Exception as row_error:
                print(f"レコード処理中のエラー: {row_error}")
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            print("指定した条件に該当するデータはありません (store:", store, ", category:", cat_id, ")")
        return df
    except Exception as e:
        print(f"get_data 内で予期せぬエラーが発生しました: {e}")
        return pd.DataFrame(rows)
    finally:
        session.close()
