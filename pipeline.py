import os
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# AYARLAR
# =========================================================
DATA_DIR = r"C:\Users\onurh\OneDrive\Masaüstü\KKBOX\train"   

TRAIN_FILE = "train.csv"
MEMBERS_FILE = "members_v3.csv"
TRANSACTIONS_FILE = "transactions.csv"
USER_LOGS_FILE = "user_logs.csv"

TRANSACTION_CHUNK_SIZE = 300_000
USERLOG_CHUNK_SIZE = 500_000

RANDOM_STATE = 42
TEST_SIZE = 0.20


# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def fpath(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def log(msg: str):
    print(f"[INFO] {msg}")


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """RAM kullanımını azaltır."""
    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df


# =========================================================
# 1) TRAIN
# =========================================================
def load_train() -> pd.DataFrame:
    log("train.csv yükleniyor...")
    df = pd.read_csv(fpath(TRAIN_FILE))
    df = reduce_memory(df)

    required_cols = {"msno", "is_churn"}
    if not required_cols.issubset(df.columns):
        raise ValueError("train.csv içinde 'msno' ve 'is_churn' sütunları bulunmalıdır.")

    log(f"train boyutu: {df.shape}")
    return df


# =========================================================
# 2) MEMBERS
# =========================================================
def load_members() -> pd.DataFrame:
    log("members_v3.csv yükleniyor...")
    members = pd.read_csv(fpath(MEMBERS_FILE))
    members = members.copy()

    # registration_init_time -> datetime -> yıl/ay
    if "registration_init_time" in members.columns:
        members["registration_init_time"] = pd.to_datetime(
            members["registration_init_time"].astype(str),
            format="%Y%m%d",
            errors="coerce"
        )
        members["registration_year"] = members["registration_init_time"].dt.year
        members["registration_month"] = members["registration_init_time"].dt.month
        members.drop(columns=["registration_init_time"], inplace=True)

    # yaş temizliği
    if "bd" in members.columns:
        members.loc[(members["bd"] < 10) | (members["bd"] > 80), "bd"] = np.nan

    # cinsiyet / şehir / registered_via varsa object bırak
    members = reduce_memory(members)
    log(f"members boyutu: {members.shape}")
    return members


# =========================================================
# 3) TRANSACTIONS - CHUNK ÖZET
# =========================================================
def aggregate_transactions() -> pd.DataFrame:
    log("transactions.csv chunk'lar halinde işleniyor...")

    usecols = [
        "msno",
        "payment_method_id",
        "payment_plan_days",
        "plan_list_price",
        "actual_amount_paid",
        "is_auto_renew",
        "transaction_date",
        "membership_expire_date",
        "is_cancel"
    ]

    user_stats = {}

    for i, chunk in enumerate(
        pd.read_csv(fpath(TRANSACTIONS_FILE), usecols=usecols, chunksize=TRANSACTION_CHUNK_SIZE)
    ):
        log(f"transactions chunk {i + 1} işlendi")

        chunk["transaction_date"] = pd.to_datetime(
            chunk["transaction_date"].astype(str),
            format="%Y%m%d",
            errors="coerce"
        )

        chunk["membership_expire_date"] = pd.to_datetime(
            chunk["membership_expire_date"].astype(str),
            format="%Y%m%d",
            errors="coerce"
        )

        for row in chunk.itertuples(index=False):
            msno = row.msno

            if msno not in user_stats:
                user_stats[msno] = {
                    "tx_count": 0,
                    "payment_method_set": set(),
                    "payment_plan_days_sum": 0.0,
                    "plan_list_price_sum": 0.0,
                    "actual_amount_paid_sum": 0.0,
                    "is_auto_renew_sum": 0.0,
                    "is_cancel_sum": 0.0,
                    "last_transaction_date": pd.NaT,
                    "last_membership_expire_date": pd.NaT,
                    "last_payment_method_id": np.nan,
                    "last_payment_plan_days": np.nan,
                    "last_plan_list_price": np.nan,
                    "last_actual_amount_paid": np.nan,
                    "last_is_auto_renew": np.nan,
                    "last_is_cancel": np.nan,
                }

            s = user_stats[msno]
            s["tx_count"] += 1

            if pd.notna(row.payment_method_id):
                s["payment_method_set"].add(row.payment_method_id)

            for field, value in [
                ("payment_plan_days_sum", row.payment_plan_days),
                ("plan_list_price_sum", row.plan_list_price),
                ("actual_amount_paid_sum", row.actual_amount_paid),
                ("is_auto_renew_sum", row.is_auto_renew),
                ("is_cancel_sum", row.is_cancel),
            ]:
                if pd.notna(value):
                    s[field] += float(value)

            # son işlem
            tx_date = row.transaction_date
            if pd.notna(tx_date):
                if pd.isna(s["last_transaction_date"]) or tx_date > s["last_transaction_date"]:
                    s["last_transaction_date"] = tx_date
                    s["last_membership_expire_date"] = row.membership_expire_date
                    s["last_payment_method_id"] = row.payment_method_id
                    s["last_payment_plan_days"] = row.payment_plan_days
                    s["last_plan_list_price"] = row.plan_list_price
                    s["last_actual_amount_paid"] = row.actual_amount_paid
                    s["last_is_auto_renew"] = row.is_auto_renew
                    s["last_is_cancel"] = row.is_cancel

        del chunk
        gc.collect()

    rows = []
    for msno, s in user_stats.items():
        tx_count = max(s["tx_count"], 1)

        days_to_expire = np.nan
        if pd.notna(s["last_transaction_date"]) and pd.notna(s["last_membership_expire_date"]):
            days_to_expire = (s["last_membership_expire_date"] - s["last_transaction_date"]).days

        rows.append({
            "msno": msno,
            "tx_count": s["tx_count"],
            "payment_method_nunique": len(s["payment_method_set"]),
            "payment_plan_days_mean": s["payment_plan_days_sum"] / tx_count,
            "plan_list_price_mean": s["plan_list_price_sum"] / tx_count,
            "actual_amount_paid_mean": s["actual_amount_paid_sum"] / tx_count,
            "is_auto_renew_mean": s["is_auto_renew_sum"] / tx_count,
            "is_cancel_mean": s["is_cancel_sum"] / tx_count,

            "last_payment_method_id": s["last_payment_method_id"],
            "last_payment_plan_days": s["last_payment_plan_days"],
            "last_plan_list_price": s["last_plan_list_price"],
            "last_actual_amount_paid": s["last_actual_amount_paid"],
            "last_is_auto_renew": s["last_is_auto_renew"],
            "last_is_cancel": s["last_is_cancel"],

            "days_to_expire": days_to_expire,
            "last_transaction_year": s["last_transaction_date"].year if pd.notna(s["last_transaction_date"]) else np.nan,
            "last_transaction_month": s["last_transaction_date"].month if pd.notna(s["last_transaction_date"]) else np.nan,
        })

    tx_df = pd.DataFrame(rows)
    tx_df = reduce_memory(tx_df)

    tx_df.to_csv(fpath("transactions_reduced.csv"), index=False)
    log(f"transactions_reduced.csv kaydedildi: {tx_df.shape}")

    return tx_df


# =========================================================
# 4) USER LOGS - CHUNK ÖZET
# =========================================================
def aggregate_user_logs() -> pd.DataFrame:
    log("user_logs.csv chunk'lar halinde işleniyor...")

    usecols = [
        "msno",
        "num_25",
        "num_50",
        "num_75",
        "num_985",
        "num_100",
        "num_unq",
        "total_secs"
    ]

    user_stats = {}

    for i, chunk in enumerate(
        pd.read_csv(fpath(USER_LOGS_FILE), usecols=usecols, chunksize=USERLOG_CHUNK_SIZE)
    ):
        log(f"user_logs chunk {i + 1} işlendi")

        for row in chunk.itertuples(index=False):
            msno = row.msno

            if msno not in user_stats:
                user_stats[msno] = {
                    "log_count": 0,
                    "num_25_sum": 0.0,
                    "num_50_sum": 0.0,
                    "num_75_sum": 0.0,
                    "num_985_sum": 0.0,
                    "num_100_sum": 0.0,
                    "num_unq_sum": 0.0,
                    "total_secs_sum": 0.0,
                }

            s = user_stats[msno]
            s["log_count"] += 1

            for field, value in [
                ("num_25_sum", row.num_25),
                ("num_50_sum", row.num_50),
                ("num_75_sum", row.num_75),
                ("num_985_sum", row.num_985),
                ("num_100_sum", row.num_100),
                ("num_unq_sum", row.num_unq),
                ("total_secs_sum", row.total_secs),
            ]:
                if pd.notna(value):
                    s[field] += float(value)

        del chunk
        gc.collect()

    rows = []
    for msno, s in user_stats.items():
        total_song_events = (
            s["num_25_sum"] + s["num_50_sum"] + s["num_75_sum"] +
            s["num_985_sum"] + s["num_100_sum"]
        )

        full_play_ratio = np.nan
        if total_song_events > 0:
            full_play_ratio = s["num_100_sum"] / total_song_events

        rows.append({
            "msno": msno,
            "log_count": s["log_count"],
            "num_25_sum": s["num_25_sum"],
            "num_50_sum": s["num_50_sum"],
            "num_75_sum": s["num_75_sum"],
            "num_985_sum": s["num_985_sum"],
            "num_100_sum": s["num_100_sum"],
            "num_unq_sum": s["num_unq_sum"],
            "total_secs_sum": s["total_secs_sum"],
            "avg_secs_per_log": s["total_secs_sum"] / max(s["log_count"], 1),
            "full_play_ratio": full_play_ratio,
            "total_song_events": total_song_events,
        })

    logs_df = pd.DataFrame(rows)
    logs_df = reduce_memory(logs_df)

    logs_df.to_csv(fpath("user_logs_reduced.csv"), index=False)
    log(f"user_logs_reduced.csv kaydedildi: {logs_df.shape}")

    return logs_df


# =========================================================
# 5) MERGE + TEMİZLİK + FEATURE ENGINEERING
# =========================================================
def build_final_dataset() -> pd.DataFrame:
    train = load_train()
    members = load_members()
    transactions = aggregate_transactions()
    user_logs = aggregate_user_logs()

    log("Tablolar birleştiriliyor...")
    df = train.merge(members, on="msno", how="left")
    df = df.merge(transactions, on="msno", how="left")
    df = df.merge(user_logs, on="msno", how="left")

    # -----------------------------------------------------
    # Veri temizleme / temel işlemler
    # -----------------------------------------------------
    log("Eksik veri ve temel temizlik işlemleri yapılıyor...")

    # Kategorik sütunlarda eksik veri
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("unknown")

    # Sayısal sütunlarda eksik veri
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col != "is_churn":
            df[col] = df[col].fillna(df[col].median())

    # city bazen sayısal gelebilir; yine de kategori gibi kullanılabilir
    if "city" in df.columns:
        df["city"] = df["city"].astype(str)

    if "registered_via" in df.columns:
        df["registered_via"] = df["registered_via"].astype(str)

    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str)

    # -----------------------------------------------------
    # Ek özellikler
    # -----------------------------------------------------
    log("Ek özellikler oluşturuluyor...")

    # ödeme farkı
    if {"plan_list_price_mean", "actual_amount_paid_mean"}.issubset(df.columns):
        df["discount_amount"] = df["plan_list_price_mean"] - df["actual_amount_paid_mean"]

    # dinleme yoğunluğu
    if {"total_secs_sum", "num_unq_sum"}.issubset(df.columns):
        df["secs_per_unique_song"] = np.where(
            df["num_unq_sum"] > 0,
            df["total_secs_sum"] / df["num_unq_sum"],
            0
        )

    # tam dinleme eğilimi
    if {"num_100_sum", "total_song_events"}.issubset(df.columns):
        df["perfect_play_rate"] = np.where(
            df["total_song_events"] > 0,
            df["num_100_sum"] / df["total_song_events"],
            0
        )

    # tekrar kayıt kontrolü
    df = df.drop_duplicates(subset=["msno"])

    df = reduce_memory(df)

    # Ön izleme ve kayıt
    df.to_csv(fpath("merged_preprocessed_full.csv"), index=False)
    log(f"merged_preprocessed_full.csv kaydedildi: {df.shape}")

    return df


# =========================================================
# 6) MODELLEMEYE HAZIR DOSYALAR
# =========================================================
def prepare_model_inputs(df: pd.DataFrame):
    log("Modellemeye hazır veri setleri oluşturuluyor...")

    if "is_churn" not in df.columns:
        raise ValueError("Hedef sütun 'is_churn' bulunamadı.")

    # Ana tam veri
    full_df = df.copy()

    # msno'yu ayrıca koru
    ids = full_df["msno"] if "msno" in full_df.columns else None

    # model matrisinden msno çıkar
    model_df = full_df.drop(columns=["msno"], errors="ignore")

    # Eğitim/test ayır
    X = model_df.drop(columns=["is_churn"])
    y = model_df["is_churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Kaydet
    X_train.to_csv(fpath("X_train.csv"), index=False)
    X_test.to_csv(fpath("X_test.csv"), index=False)
    y_train.to_csv(fpath("y_train.csv"), index=False)
    y_test.to_csv(fpath("y_test.csv"), index=False)

    log(f"X_train: {X_train.shape}")
    log(f"X_test : {X_test.shape}")
    log(f"y_train: {y_train.shape}")
    log(f"y_test : {y_test.shape}")

    # İsteğe bağlı tam ön izleme
    preview_cols = ["msno", "is_churn"] if "msno" in full_df.columns else ["is_churn"]
    preview_cols += [c for c in full_df.columns if c not in preview_cols][:20]
    full_df[preview_cols].head(1000).to_csv(fpath("preview_1000_rows.csv"), index=False)

    log("Modellemeye hazır tüm dosyalar kaydedildi.")


# =========================================================
# ANA AKIŞ
# =========================================================
if __name__ == "__main__":
    final_df = build_final_dataset()
    prepare_model_inputs(final_df)

    print("\n=== HAZIR ===")
    print("Üretilen dosyalar:")
    print("- transactions_reduced.csv")
    print("- user_logs_reduced.csv")
    print("- merged_preprocessed_full.csv")
    print("- X_train.csv")
    print("- X_test.csv")
    print("- y_train.csv")
    print("- y_test.csv")
    print("- preview_1000_rows.csv")