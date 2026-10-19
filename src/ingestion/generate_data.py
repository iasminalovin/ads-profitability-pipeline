"""Generate synthetic ads and sales data (CSV) for the ads profitability project."""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
OUTPUT_DIR = Path("data/raw")

# campaign_id, platform, name, avg daily clicks, cost per click (RON),
# conversion rate (share of clicks that become orders), avg order value (RON),
# cogs_ratio (share of revenue that is the cost of the product itself)
CAMPAIGNS = [
    (1, "google_ads", "Brand Search", 120, 0.90, 0.070, 300, 0.70),
    (2, "google_ads", "Generic Search - Laptops", 200, 3.50, 0.012, 2500, 0.85),
    (3, "google_ads", "Shopping - Phones", 250, 2.00, 0.020, 1800, 0.88),
    (4, "google_ads", "Display Retargeting", 300, 0.60, 0.030, 400, 0.70),
    (5, "google_ads", "Display Prospecting", 500, 1.20, 0.004, 400, 0.70),
    (6, "facebook_ads", "Lookalike Audience", 350, 1.50, 0.015, 350, 0.70),
    (7, "facebook_ads", "Interest - Gadgets", 400, 1.10, 0.006, 250, 0.72),
    (8, "facebook_ads", "Retargeting", 150, 0.80, 0.050, 380, 0.70),
    (9, "facebook_ads", "Instagram Influencer Promo", 600, 1.40, 0.003, 300, 0.70),
    (10, "google_ads", "YouTube Video Ads", 700, 0.50, 0.002, 500, 0.75),
]
CITIES = ["Bucharest", "Cluj-Napoca", "Timisoara",
          "Iasi", "Constanta", "Brasov", "Craiova", "Sibiu"]
CATEGORIES = ["laptops", "phones", "accessories", "home_appliances", "gadgets"]


def seasonality(day: pd.Timestamp) -> float:
    """More traffic during Black Friday season and December."""
    if day.month == 11 and day.day >= 20:
        return 1.8
    if day.month == 12:
        return 1.3
    return 1.0


def make_orders(rng, day, n, campaign_id, avg_order_value, cogs_ratio) -> pd.DataFrame:
    """Create n orders for one campaign (or organic traffic) on one day."""
    revenue = rng.lognormal(mean=np.log(avg_order_value),
                            sigma=0.4, size=n).round(2)
    return pd.DataFrame(
        {
            "order_date": [day.date()] * n,
            "campaign_id": [campaign_id] * n,
            "product_category": rng.choice(CATEGORIES, size=n),
            "revenue": revenue,
            "cogs": (revenue * cogs_ratio).round(2),
            "status": rng.choice(["completed", "returned", "cancelled"], size=n, p=[0.92, 0.06, 0.02]),
        }
    )


def assign_customers(orders: pd.DataFrame, rng) -> pd.DataFrame:
    """65% of orders come from a brand new customer, the rest from an existing one."""
    n_customers = 0
    assigned = []
    for is_new in rng.random(len(orders)) < 0.65:
        if is_new or n_customers == 0:
            n_customers += 1
            assigned.append(n_customers)
        else:
            assigned.append(int(rng.integers(1, n_customers + 1)))
    orders["customer_id"] = assigned
    return orders


def generate(rng):
    days = pd.date_range(START_DATE, END_DATE, freq="D")
    ad_rows = []
    order_frames = []

    for day in days:
        factor = seasonality(day)

        for campaign_id, _, _, base_clicks, cpc, conv, aov, cogs_ratio in CAMPAIGNS:
            clicks = int(rng.poisson(base_clicks * factor))
            impressions = int(clicks / rng.uniform(0.01, 0.03)
                              )  # click-through rate 1-3%
            spend = round(clicks * cpc * rng.uniform(0.9, 1.1), 2)
            ad_rows.append(
                {
                    "date": day.date(),
                    "campaign_id": campaign_id,
                    "impressions": impressions,
                    "clicks": clicks,
                    "spend": spend,
                }
            )

            n_orders = int(rng.binomial(clicks, conv))
            if n_orders > 0:
                order_frames.append(make_orders(
                    rng, day, n_orders, campaign_id, aov, cogs_ratio))

        # organic orders: customers who did not come from any ad
        n_organic = int(rng.poisson(15 * factor))
        if n_organic > 0:
            order_frames.append(make_orders(
                rng, day, n_organic, np.nan, 300, 0.70))

    # the hidden parameters (conversion rate, cpc...) stay secret: the pipeline must discover them
    campaigns = pd.DataFrame(CAMPAIGNS, columns=[
        "campaign_id", "platform", "campaign_name", "_clicks", "_cpc", "_conv", "_aov", "_cogs",
    ]).iloc[:, :3]

    ad_spend = pd.DataFrame(ad_rows)

    orders = pd.concat(order_frames, ignore_index=True)
    orders["campaign_id"] = orders["campaign_id"].astype("Int64")
    orders = assign_customers(orders, rng)
    orders.insert(0, "order_id", range(1, len(orders) + 1))

    customers = orders.groupby("customer_id", as_index=False).agg(
        signup_date=("order_date", "min"))
    customers["city"] = rng.choice(CITIES, size=len(customers))

    return campaigns, ad_spend, orders, customers


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    campaigns, ad_spend, orders, customers = generate(rng)

    tables = {"campaigns": campaigns, "ad_spend": ad_spend,
              "orders": orders, "customers": customers}
    for name, df in tables.items():
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
