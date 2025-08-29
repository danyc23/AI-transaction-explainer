from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile):
    try:
        df = load_transactions(file.file)

        # Add Year-Month column for grouping
        df["YearMonth"] = df["Transaction Date"].dt.to_period("M").astype(str)

        # Split spending vs payments
        spending_df = df[df['CAD'] < 0].copy()
        payments_df = df[df['CAD'] > 0].copy()

        # Categorization
        def categorize(desc: str):
            d = str(desc).lower()
            if any(x in d for x in ['uber', 'evocar', 'compass']):
                return "Transportation"
            if any(x in d for x in ['tim hortons','mcdonald','dominos','pizza','burger','donair', 'chicken']):
                return "Food & Dining"
            if any(x in d for x in ['shoppers','safeway','amazon','lululemon']):
                return "Shopping & Retail"
            if any(x in d for x in ['gym','wodify','pool']):
                return "Fitness & Recreation"
            if any(x in d for x in ['liquor','beer','wine']):
                return "Alcohol & Beverages"
            if any(x in d for x in ['netflix','apple','google','youtube']):
                return "Subscriptions & Tech"
            return "Other"

        if not spending_df.empty:
            spending_df["Category"] = spending_df["Description 1"].apply(categorize)
        else:
            spending_df["Category"] = []

        # Monthly aggregation
        monthly_summary = {}
        for month, group in df.groupby("YearMonth"):
            spend = group[group["CAD"] < 0]
            pay = group[group["CAD"] > 0]

            # Category breakdown for spending in that month
            if not spend.empty:
                cat_spending = spend.assign(Category=spend["Description 1"].apply(categorize))
                cat_spending = cat_spending.groupby("Category")["CAD"].sum().abs().to_dict()
            else:
                cat_spending = {}

            monthly_summary[month] = {
                "total_cad_spending": round(abs(spend["CAD"].sum()), 2),
                "total_usd_spending": round(abs(group[group["USD"] < 0]["USD"].sum()), 2),
                "total_payments": round(abs(pay["CAD"].sum()), 2),
                "category_breakdown": cat_spending,
                "spending_transaction_count": len(spend),
                "payment_transaction_count": len(pay),
            }

        # Overall summary
        overall = {
            "total_cad_spending": round(abs(spending_df["CAD"].sum()), 2),
            "total_usd_spending": round(abs(df[df["USD"] < 0]["USD"].sum()), 2),
            "total_payments": round(abs(payments_df["CAD"].sum()), 2),
            "category_breakdown": spending_df.groupby("Category")["CAD"].sum().abs().to_dict() if not spending_df.empty else {},
            "spending_transaction_count": len(spending_df),
            "payment_transaction_count": len(payments_df),
            "date_range": f"{df['Transaction Date'].min()} to {df['Transaction Date'].max()}"
        }

        return {
            "overall": overall,
            "by_month": monthly_summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Transaction Analyzer API", "endpoints": ["/analyze", "/docs"]}


colnames = [
    "Account Type",
    "Account Number",
    "Transaction Date",
    "Cheque Number",
    "Description 1",
    "Description 2",
    "CAD",
    "USD"
]

def load_transactions(file):
    # Force 8 columns, drop any extra junk from trailing commas
    df = pd.read_csv(
        file,
        names=colnames,
        skiprows=1,        # skip the header line, since we supply colnames
        usecols=colnames,  # ignore any extra columns from extra commas
        engine="python"
    )

    # Parse dates & numbers
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    df["CAD"] = pd.to_numeric(df["CAD"], errors="coerce").fillna(0)
    df["USD"] = pd.to_numeric(df["USD"], errors="coerce").fillna(0)

    return df
