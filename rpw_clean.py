"""
rpw_clean.py  -  Clean and analyse the World Bank Remittance Prices Worldwide dataset.

Run:   python rpw_clean.py
Needs: pandas, openpyxl, matplotlib   (pip install pandas openpyxl matplotlib)

Outputs go to ./output/ (CSV files + 2 charts).
Data credit: "The World Bank, Remittance Prices Worldwide, available at
http://remittanceprices.worldbank.org"
"""
from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------
# CONFIG - change these
# ----------------------------------------------------------------------------
XLSX_PATH = "rpw_dataset_2011_2025_q1.xlsx"
OUT = Path("output"); OUT.mkdir(exist_ok=True)
CACHE = OUT / "raw_cache.pkl"      # reading the 49 MB Excel takes ~2 min; cache it
MIN_OBS = 10                       # min price quotes before a corridor/provider is ranked
RECENT_QUARTERS = 4                # "current" prices = last N quarters in the data
MAX_SANE_COST = 50                 # % - quotes above this are flagged as outliers

AFRICA = [  # ISO-3 codes of the 54 African countries (the 'region' columns can't be
    # used: "Middle East & North Africa" also includes non-African countries)
    "DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV",
    "DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR",
    "LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN",
    "SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE",
]


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Read both dataset sheets and stack them into one table."""
    if CACHE.exists():
        return pd.read_pickle(CACHE)

    print("Reading Excel (slow the first time, ~2 min)...")
    old = pd.read_excel(XLSX_PATH, sheet_name="Dataset (up to Q1 2016)")
    new = pd.read_excel(XLSX_PATH, sheet_name="Dataset (from Q2 2016)")

    # the two sheets name a few columns differently
    old = old.rename(columns={"pick-up method": "pickup method"})
    new = new.loc[:, ~new.columns.astype(str).str.startswith("Unnamed")]

    common = [c for c in new.columns if c in old.columns]   # keep columns both share
    raw = pd.concat([old[common], new[common]], ignore_index=True)
    raw.to_pickle(CACHE)
    return raw


# ----------------------------------------------------------------------------
# 2. CLEAN
# ----------------------------------------------------------------------------
def simplify_firm_type(s: str) -> str:
    s = str(s).lower()
    if "mobile" in s:                                  return "Mobile operator"
    if "non-bank" in s:                                return "Other"
    if "bank" in s and "money transfer" in s:          return "Bank / MTO"
    if "bank" in s:                                    return "Bank"
    if "money transfer" in s:                          return "Money transfer operator"
    if "post" in s:                                    return "Post office"
    return "Other"


def simplify_pickup(s: str) -> str:
    s = str(s).lower()
    if "mobile" in s:        return "Mobile wallet"
    if "bank account" in s:  return "Bank account"
    if "cash" in s:          return "Cash"
    return "Other"


SPEED_MAP = {
    "less than one hour": "< 1 hour", "same day": "Same day", "next day": "Next day",
    "1-3 days": "2 days", "2 days": "2 days", "3-5 days": "3-5 days",
    "6 days or more": "6+ days",
}


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # tidy text columns: strip spaces, turn '..' placeholders into real NaN
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in text_cols:
        df[c] = df[c].astype("string").str.strip().replace({"..": pd.NA, "": pd.NA})

    # nicer column names
    df = df.rename(columns={
        "source_code": "src_code", "source_name": "src_country",
        "destination_code": "dst_code", "destination_name": "dst_country",
        "speed actual": "speed", "pickup method": "pickup",
        "cc1 total cost %": "cost_pct_200",      # cost of sending USD 200, as % of 200
        "cc2 total cost %": "cost_pct_500",      # cost of sending USD 500
        "cc1 fx margin": "fx_margin_pct_200",
    })

    # dates / periods:  '2016_2Q' -> year 2016, quarter 2
    df["year"] = df["period"].str[:4].astype(int)
    df["quarter"] = df["period"].str[-2].astype(int)
    df["period_sort"] = df["year"] * 10 + df["quarter"]
    df["date"] = pd.to_datetime(df["date"], format="%d/%b/%Y", errors="coerce")

    # categories
    df["firm_type"] = df["firm_type"].map(simplify_firm_type)
    df["pickup"] = df["pickup"].map(simplify_pickup)
    df["speed"] = df["speed"].str.lower().map(SPEED_MAP)
    df["transparent"] = df["transparent"].str.lower().eq("yes")

    # cost in dollars (the quote is for sending USD 200)
    df["cost_usd_200"] = df["cost_pct_200"] / 100 * 200

    # African flags + readable corridor label
    df["src_africa"] = df["src_code"].isin(AFRICA)
    df["dst_africa"] = df["dst_code"].isin(AFRICA)
    df["route_type"] = np.select(
        [df.src_africa & df.dst_africa, df.dst_africa, df.src_africa],
        ["Africa -> Africa", "World -> Africa", "Africa -> World"],
        default="Other",
    )
    df["corridor_name"] = df["src_country"] + " → " + df["dst_country"]

    # drop rows with no price, drop exact duplicates
    df = df.dropna(subset=["cost_pct_200"]).drop_duplicates(subset=["id"])

    # flag suspicious quotes (kept in the data, excluded from rankings)
    df["outlier"] = (df["cost_pct_200"] < 0) | (df["cost_pct_200"] > MAX_SANE_COST)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. AGGREGATE
# ----------------------------------------------------------------------------
def summarise(g) -> pd.Series:
    """Median is the headline number (robust to odd quotes); mean shown for reference."""
    return pd.Series({
        "n_quotes": len(g),
        # when grouping BY firm, the column is gone from the group, so fall back to 1
        "n_providers": g["firm"].nunique() if "firm" in g.columns else 1,
        "median_cost_pct": g["cost_pct_200"].median(),
        "mean_cost_pct": g["cost_pct_200"].mean(),
        "min_cost_pct": g["cost_pct_200"].min(),
        "max_cost_pct": g["cost_pct_200"].max(),
        "median_cost_usd_on_200": g["cost_usd_200"].median(),
    })


def recent(df: pd.DataFrame) -> pd.DataFrame:
    """Only the last N quarters, so rankings reflect today's prices."""
    periods = sorted(df["period_sort"].unique())[-RECENT_QUARTERS:]
    return df[df["period_sort"].isin(periods)]


def corridor_ranking(df: pd.DataFrame, route_type="Africa -> Africa") -> pd.DataFrame:
    d = recent(df[(df.route_type == route_type) & ~df.outlier])
    out = d.groupby("corridor_name").apply(summarise, include_groups=False)
    out = out[out.n_quotes >= MIN_OBS]
    return out.sort_values("median_cost_pct").round(2)


def by_group(df: pd.DataFrame, col: str, route_type="Africa -> Africa") -> pd.DataFrame:
    d = recent(df[(df.route_type == route_type) & ~df.outlier])
    out = d.groupby(col).apply(summarise, include_groups=False)
    return out[out.n_quotes >= MIN_OBS].sort_values("median_cost_pct").round(2)


def yearly_trend(df: pd.DataFrame) -> pd.DataFrame:
    d = df[~df.outlier]
    t = d.groupby(["year", "route_type"])["cost_pct_200"].median().unstack().round(2)
    return t


def provider_ranking(df: pd.DataFrame) -> pd.DataFrame:
    d = recent(df[(df.route_type == "Africa -> Africa") & ~df.outlier])
    out = d.groupby(["firm", "firm_type"]).apply(summarise, include_groups=False)
    return out[out.n_quotes >= MIN_OBS].sort_values("median_cost_pct").round(2).reset_index()


def lookup_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per corridor + provider + pickup method (latest quotes) -
    this is the table your Streamlit lookup tool reads."""
    d = recent(df[~df.outlier])
    keys = ["src_country", "dst_country", "firm", "firm_type", "pickup", "speed"]
    out = (d.groupby(keys, dropna=False)
             .agg(cost_pct_200=("cost_pct_200", "median"),
                  cost_usd_200=("cost_usd_200", "median"),
                  cost_pct_500=("cost_pct_500", "median"),
                  n_quotes=("id", "count"))
             .round(2).reset_index())
    return out.sort_values(["src_country", "dst_country", "cost_pct_200"])


# ----------------------------------------------------------------------------
# 4. CHARTS
# ----------------------------------------------------------------------------
def make_charts(ranking: pd.DataFrame, trend: pd.DataFrame):
    import matplotlib.pyplot as plt

    # chart 1: 5 cheapest vs 5 most expensive corridors
    sel = pd.concat([ranking.head(5), ranking.tail(5)])["median_cost_pct"]
    colors = ["#2e8b57"] * 5 + ["#c0392b"] * 5
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(sel.index[::-1], sel.values[::-1], color=colors[::-1])
    for i, v in enumerate(sel.values[::-1]):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center")
    ax.set_xlabel("Median cost of sending $200 (% of amount)")
    ax.set_title("Cheapest vs most expensive African corridors")
    fig.text(0.01, 0.01, "Source: World Bank, Remittance Prices Worldwide", fontsize=8)
    plt.tight_layout(); fig.savefig(OUT / "chart_corridors.png", dpi=200); plt.close(fig)

    # chart 2: cost trend over the years
    fig, ax = plt.subplots(figsize=(9, 5))
    trend.plot(ax=ax, marker="o")
    ax.set_ylabel("Median cost of sending $200 (%)"); ax.set_xlabel("")
    ax.set_title("Remittance cost over time")
    plt.tight_layout(); fig.savefig(OUT / "chart_trend.png", dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------
# 5. RUN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    df = clean(load_raw())
    print(f"Clean rows: {len(df):,}  |  periods: {df.period.min()} to {df.period.max()}")
    print(df.route_type.value_counts(), "\n")

    ranking = corridor_ranking(df)
    trend = yearly_trend(df)
    firm_types = by_group(df, "firm_type")
    pickups = by_group(df, "pickup")
    senders = by_group(df, "src_country")
    providers = provider_ranking(df)
    lookup = lookup_table(df[df.route_type == "Africa -> Africa"])

    # save everything
    df[df.route_type == "Africa -> Africa"].to_csv(OUT / "clean_africa_to_africa.csv", index=False)
    ranking.to_csv(OUT / "corridor_ranking.csv")
    trend.to_csv(OUT / "yearly_trend.csv")
    firm_types.to_csv(OUT / "by_firm_type.csv")
    pickups.to_csv(OUT / "by_pickup_method.csv")
    senders.to_csv(OUT / "by_sending_country.csv")
    providers.to_csv(OUT / "provider_ranking.csv", index=False)
    lookup.to_csv(OUT / "lookup_table.csv", index=False)

    # what goes in the thread
    pd.set_option("display.width", 200)
    print("5 CHEAPEST corridors:\n", ranking.head(5)[["n_quotes", "median_cost_pct", "median_cost_usd_on_200"]], "\n")
    print("5 MOST EXPENSIVE corridors:\n", ranking.tail(5)[["n_quotes", "median_cost_pct", "median_cost_usd_on_200"]], "\n")
    print("By provider type:\n", firm_types[["n_quotes", "median_cost_pct"]], "\n")
    print("By pickup method:\n", pickups[["n_quotes", "median_cost_pct"]], "\n")
    print("Trend by year:\n", trend, "\n")

    make_charts(ranking, trend)
    print("Done. Files saved in", OUT.resolve())
