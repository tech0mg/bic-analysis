# sales_call_merge.py
import pandas as pd
from db import SessionLocal
from models import Sales_call, Reception

def merge_sales_call_and_reception(selected_store, selected_category) -> pd.DataFrame:
    """
    sales_call と reception を JOIN して、店員呼び出し時刻と対応する接客情報をまとめた DataFrame を返す
    store_id と category_idはユーザーが選択した値 (selected_store, selected_category) を使用
    """
    session = SessionLocal()
    try:
        # sales_call.reception_id == reception.id で内部結合
        query = session.query(Sales_call, Reception) \
            .join(Reception, Sales_call.reception_id == Reception.id) \
            .all()

        if not query:
            # クエリ結果が0件の場合、空の DataFrame を返して呼び出し元で対処
            print("DEBUG: sales_call と reception を JOIN した結果がありません。")
            return pd.DataFrame()
        
        rows = []
        for sc, r in query:
            row = {}
            # sales_call の情報
            row["sales_call_id"] = sc.id
            row["call_time"] = sc.time  # 呼び出し時刻
            row["call_reception_id"] = sc.reception_id
            
            # reception の情報
            row["reception_id"] = r.id
            row["reception_time"] = r.time
            row["store_id"] = selected_store
            row["category_id"] = selected_category

            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        print(f"merge_sales_call_and_reception 内でエラーが発生: {e}")
        return pd.DataFrame()
    finally:
        session.close()