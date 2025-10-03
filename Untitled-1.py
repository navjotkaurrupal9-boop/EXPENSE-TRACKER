import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import base64

# --------- App Setup ----------
st.set_page_config(page_title="Smart Expense Tracker", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid")

# Initialize session state
if 'expenses' not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns=["Date", "Category", "Amount", "Description"])

# ---------- Helper Functions ----------
def add_expense(date, category, amount, description):
    """Add new expense row to session state DataFrame."""
    # Ensure date saved as ISO string for CSV friendliness
    date_str = pd.to_datetime(date).date().isoformat()
    new_row = pd.DataFrame([[date_str, category, float(amount), description]], columns=st.session_state.expenses.columns)
    st.session_state.expenses = pd.concat([st.session_state.expenses, new_row], ignore_index=True)

def save_expenses_to_local(filename="expenses.csv"):
    """Save current expenses to a local CSV file."""
    st.session_state.expenses.to_csv(filename, index=False)
    st.success(f"Saved to {filename}")

def get_csv_download_link(df, filename="expenses.csv"):
    """Return a Streamlit download button for the dataframe as CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some browsers need this
    href = f"data:file/csv;base64,{b64}"
    return href

def load_expenses_from_uploaded(uploaded_file):
    """Load uploaded CSV into session state (replace existing)."""
    try:
        df = pd.read_csv(uploaded_file)
        # Normalize columns and types where possible
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date.astype(str)
        if "Amount" in df.columns:
            df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
        # Keep only expected columns if extras exist
        expected = ["Date", "Category", "Amount", "Description"]
        for col in expected:
            if col not in df.columns:
                df[col] = ""
        st.session_state.expenses = df[expected]
        st.success("Expenses loaded from uploaded file.")
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")

def visualize_bar(df):
    """Bar plot of category totals."""
    df = df.copy()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    cat_totals = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    if cat_totals.empty:
        st.info("No data to plot.")
        return
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=cat_totals.index, y=cat_totals.values, ax=ax)
    ax.set_xlabel("Category")
    ax.set_ylabel("Total Amount")
    ax.set_title("Spending by Category")
    plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=True)

def visualize_pie(df):
    """Pie chart of spending distribution."""
    df = df.copy()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    cat_totals = df.groupby("Category")["Amount"].sum()
    if cat_totals.sum() == 0:
        st.info("No data for pie chart.")
        return
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(cat_totals, labels=cat_totals.index, autopct="%1.1f%%", startangle=140)
    ax.set_title("Spending Distribution")
    ax.axis('equal')
    st.pyplot(fig)

def expense_summary(df):
    """Show smart summary statistics."""
    if df.empty:
        st.info("No expenses yet.")
        return

    df = df.copy()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    total = df["Amount"].sum()
    avg = df["Amount"].mean() if len(df) > 0 else 0.0
    count = len(df)
    # Highest spending category
    cat_totals = df.groupby("Category")["Amount"].sum()
    if not cat_totals.empty:
        highest_cat = cat_totals.idxmax()
        highest_amt = cat_totals.max()
    else:
        highest_cat, highest_amt = None, 0.0

    st.metric("Total Spent", f"₹{total:.2f}")
    cols = st.columns(3)
    cols[0].metric("Average per Expense", f"₹{avg:.2f}")
    cols[1].metric("Number of Entries", f"{count}")
    if highest_cat is not None:
        cols[2].metric("Highest Category", f"{highest_cat} (₹{highest_amt:.2f})")
    else:
        cols[2].metric("Highest Category", "—")

    # Recent month summary (if date column exists)
    if "Date" in df.columns:
        try:
            df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
            if df["Date_dt"].notna().any():
                last_month = df["Date_dt"].dt.to_period("M").max()
                recent = df[df["Date_dt"].dt.to_period("M") == last_month]
                recent_total = recent["Amount"].sum()
                st.write(f"Recent month ({last_month.strftime('%Y-%m')}) total: ₹{recent_total:.2f}")
        except Exception:
            pass

def predict_next_month(df):
    """Predict next month's spending using simple monthly average."""
    if df.empty:
        st.info("No data for prediction.")
        return None
    df = df.copy()
    df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date_dt"])
    if df.empty:
        st.info("No dated entries for prediction.")
        return None
    df["Month"] = df["Date_dt"].dt.to_period("M")
    monthly_totals = df.groupby("Month")["Amount"].sum().sort_index()
    if len(monthly_totals) == 0:
        return None
    predicted = monthly_totals.mean()  # simple average
    trend = None
    if len(monthly_totals) >= 2:
        trend = monthly_totals.pct_change().mean()
    return {"predicted_total": float(predicted), "trend_pct": float(trend) if trend is not None else None, "history": monthly_totals}

def budget_alerts(df, budget):
    """Show budget alerts based on total spent."""
    df = df.copy()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    spent = df["Amount"].sum()
    if budget <= 0:
        return
    if spent > budget:
        st.error(f"🚨 Budget Exceeded! Spent ₹{spent:.2f} / ₹{budget:.2f}")
    elif spent > 0.9 * budget:
        st.warning(f"⚠ You're very close to your budget: ₹{spent:.2f} / ₹{budget:.2f} (>{90}%)")
    elif spent > 0.75 * budget:
        st.info(f"🔔 You are at {spent:.2f} / {budget:.2f} ({(spent/budget)*100:.0f}%) — keep an eye on it.")
    else:
        st.success(f"✅ Under budget: ₹{spent:.2f} / ₹{budget:.2f}")

# ---------- UI ----------
st.title("💡 Smart Expense Tracker (No AI)")

# Sidebar: Add expense + file ops + budget
with st.sidebar:
    st.header("Add Expense")
    date = st.date_input("Date")
    category = st.selectbox("Category", ["Food", "Transportation", "Entertainment", "Utilities", "Others"])
    amount = st.number_input("Amount (₹)", min_value=0.0, format="%.2f")
    description = st.text_input("Description (optional)")

    if st.button("Add Expense"):
        add_expense(date, category, amount, description)
        st.success("Expense added.")
        st.rerun()  # refresh display to show new row immediately

    st.markdown("---")
    st.header("File Operations")
    if st.button("Save to local CSV"):
        save_expenses_to_local()

    uploaded_file = st.file_uploader("Load expenses from CSV", type=["csv"])
    if uploaded_file is not None:
        load_expenses_from_uploaded(uploaded_file)

    st.markdown("*Export / Download*")
    if not st.session_state.expenses.empty:
        csv_href = get_csv_download_link(st.session_state.expenses, "expenses.csv")
        st.markdown(f"[⬇ Download current CSV]({csv_href})", unsafe_allow_html=True)

    st.markdown("---")
    st.header("Budget Alerts")
    budget = st.number_input("Set Monthly Budget (₹)", min_value=0.0, format="%.2f", value=0.0)
    if st.button("Check Budget"):
        budget_alerts(st.session_state.expenses, budget)

# Main area: table, visualizations, summary, prediction
st.header("📊 Current Expenses")
st.write(st.session_state.expenses)

left, right = st.columns([2,1])

with left:
    st.header("Visualizations")
    st.subheader("Bar chart: Spending by Category")
    visualize_bar(st.session_state.expenses)
    st.subheader("Pie chart: Distribution")
    visualize_pie(st.session_state.expenses)

with right:
    st.header("Smart Summary")
    expense_summary(st.session_state.expenses)
    st.markdown("---")
    st.header("Simple Prediction")
    pred = predict_next_month(st.session_state.expenses)
    if pred:
        st.write(f"Estimated next month spending (simple average): *₹{pred['predicted_total']:.2f}*")
        if pred["trend_pct"] is not None:
            trend_pct = pred["trend_pct"] * 100
            st.write(f"Average monthly trend: *{trend_pct:.1f}%* (positive means increasing)")
        # Show small history table
        try:
            history_df = pred["history"].reset_index()
            history_df.columns = ["Month", "Total"]
            history_df["Month"] = history_df["Month"].astype(str)
            st.table(history_df.tail(6))
        except Exception:
            pass

st.markdown("---")
st.header("Quick Insights")
# Additional quick insights based on thresholds
df = st.session_state.expenses.copy()
if df.empty:
    st.info("No data for insights. Add some expenses to get quick insights.")
else:
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    cat_totals = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    if not cat_totals.empty:
        top_cat = cat_totals.index[0]
        top_amt = cat_totals.iloc[0]
        st.write(f"🔝 Top category: *{top_cat}* (₹{top_amt:.2f})")
    # Look for many small expenses
    small_expenses_count = (df["Amount"] < 100).sum()
    if small_expenses_count > 5:
        st.write(f"🔎 You have {small_expenses_count} small expenses (<₹100). These add up — consider reducing them.")

st.caption("Built with Python, Streamlit, pandas, matplotlib & seaborn — no external AI required.")