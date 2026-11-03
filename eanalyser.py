import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Personal Expense Analyser",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)
FILE = "expenses.csv"
COLUMNS = ["Date", "Category", "Amount", "Description"]


st.title("💰 PERSONAL EXPENSE ANALYSER")
st.write("Track, Analyse and Predict Your Expenses")

def load_expenses():

    if os.path.exists(FILE):

        df = pd.read_csv(FILE)

        for col in COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        df["Date"] = pd.to_datetime(
            df["Date"], errors="coerce"
        )

        df["Amount"] = pd.to_numeric(
            df["Amount"], errors="coerce"
        )

        df = df.dropna(
            subset=["Date", "Category", "Amount"]
        )

        df = df[df["Amount"] > 0]

        return df[COLUMNS]

    return pd.DataFrame(columns=COLUMNS)
def save_expenses(df):

    df.to_csv(FILE, index=False)

def add_expenses():

    df = load_expenses()

    st.subheader("➕ Add Expenses")

    date = st.date_input(
        "Enter Date",
        max_value=pd.Timestamp.today().date()
    )

    category = st.selectbox(
        "Enter Category",
        ["Food", "Travel", "Shopping", "Bills", "Others"]
    )

    amount = st.number_input(
        "Enter Amount",
        min_value=0.0,
        step=1.0
    )

    description = st.text_input("Enter Description")

    if st.button("Add Expense"):

        if amount <= 0:

            st.error("Amount must be positive.")

        elif date > pd.Timestamp.today().date():

            st.error("Future transactions not possible")

        else:

            new_expenses = pd.DataFrame([
                {
                    "Date": pd.Timestamp(date),
                    "Category": category,
                    "Amount": amount,
                    "Description": description
                }
            ])

            df = pd.concat(
                [df, new_expenses],
                ignore_index=True
            )

            save_expenses(df)

            st.success("Expense added successfully!")
def view_expenses():

    df = load_expenses()

    st.subheader("📋 View Expenses")

    if df.empty:

        st.warning("No expense history")

    else:

        df = df.sort_values("Date")

        st.dataframe(
            df,
            use_container_width=True
        )
def delete_expenses():

    df = load_expenses()

    st.subheader("🗑️ Delete Expenses")

    if df.empty:

        st.warning("No expenses available")

        return

    df = df.reset_index(drop=True)

    choice = st.selectbox(
        "Select Expense",
        df.index,
        format_func=lambda x: (
            f"{x+1}. {df.loc[x, 'Date'].date()} | "
            f"{df.loc[x, 'Category']} | "
            f"₹{df.loc[x, 'Amount']:.2f}"
        )
    )

    if st.button("Delete Expense"):

        df = df.drop(index=choice)

        save_expenses(df)

        st.success("Expense deleted successfully")

        st.rerun()
def expense_summary():

    df = load_expenses()

    st.subheader("📊 Expense Summary")

    if df.empty:

        st.warning("No expense detected")

        return

    st.metric(
        "Total Spendings",
        f"₹{df['Amount'].sum():.2f}"
    )

    st.metric(
        "Average Spendings",
        f"₹{df['Amount'].mean():.2f}"
    )

    st.metric(
        "Number of Transactions",
        len(df)
    )

    st.write("### Category-wise Amount Spendings")

    category_total = df.groupby(
        "Category"
    )["Amount"].sum()

    st.dataframe(category_total)

    st.write("### Monthly Spending")

    df["Month"] = df["Date"].dt.to_period("M")

    monthly = df.groupby("Month")["Amount"].sum()

    st.dataframe(monthly)

    highest_category = category_total.idxmax()

    st.success(
        f"Highest Spending Category: {highest_category}"
    )
def visualize_expenses():

    df = load_expenses()

    st.subheader("📈 Visualize Expenses")

    if df.empty:

        st.warning("No expenses available")

        return

    category_total = df.groupby(
        "Category"
    )["Amount"].sum()

    df["Month"] = df["Date"].dt.to_period("M")

    monthly = df.groupby(
        "Month"
    )["Amount"].sum()

    monthly = monthly.sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    category_total.plot(
        kind="bar",
        ax=axes[0]
    )

    axes[0].set_title("Category wise expense")

    axes[0].set_xlabel("Category")

    axes[0].set_ylabel("Amount")

    axes[0].tick_params(axis="x", rotation=35)

    category_total.plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=axes[1]
    )

    axes[1].set_title("Expense distribution")

    axes[1].set_xlabel("")

    axes[1].set_ylabel("")

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    monthly.index = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        monthly.index,
        monthly.values,
        marker='o'
    )

    ax.set_title("Monthly expense trend")

    ax.set_xlabel("Month")

    ax.set_ylabel("Amount")

    plt.xticks(rotation=45)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ---------------- BUDGET ANALYSIS ----------------

def budget_analyse():

    df = load_expenses()

    st.subheader("💰 Monthly Budget Analysis")

    budget = st.number_input(
        "Enter your budget",
        min_value=0.0,
        step=100.0
    )

    if st.button("Analyse Budget"):

        if budget <= 0:

            st.error("Budget must be positive")

            return

        current_month = pd.Timestamp.today().to_period('M')

        if df.empty:

            current_expense = 0

        else:

            current_expense = df[
                df["Date"].dt.to_period("M") == current_month
            ]["Amount"].sum()

        remaining = budget - current_expense

        st.write(f"Monthly budget: ₹{budget:.2f}")

        st.write(
            f"Current month spending: ₹{current_expense:.2f}"
        )

        if remaining < 0:

            st.error("Exceeds budget")

        else:

            st.success("Within budget")

        st.write(f"Remaining: ₹{remaining:.2f}")

        st.progress(
            min(current_expense / budget, 1.0)
        )

def predict_expense():

    df = load_expenses()

    st.subheader("🔮 Predict Future Expense")

    if df.empty:

        st.warning("No expense data found")

        return

    df["month"] = df["Date"].dt.to_period('M')

    monthly = df.groupby('month')['Amount'].sum()

    monthly = monthly.sort_index()

    if len(monthly) < 2:

        st.warning("Need at least two months for prediction")

        return

    if st.button("Predict Expense"):

        monthly_df = monthly.reset_index()

        monthly_df.columns = ["Month", "Expense"]

        monthly_df["MonthNumber"] = (
            monthly_df["Month"].dt.year * 12
            + monthly_df["Month"].dt.month
        )

        x = monthly_df["MonthNumber"]

        y = monthly_df["Expense"]

        model = LinearRegression()

        model.fit(
            x.values.reshape(-1, 1),
            y
        )

        last_month = monthly_df["Month"].iloc[-1]

        next_month = last_month + 1

        next_month_number = (
            next_month.year * 12 + next_month.month
        )

        prediction = model.predict(
            [[next_month_number]]
        )

        prediction = max(0, prediction[0])

        st.success(
            f"Predicted expense: ₹{prediction:.2f}"
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            monthly_df["Month"].dt.to_timestamp(),
            monthly_df["Expense"],
            label="Previous expenses",
            marker='o',
            color='green'
        )

        ax.scatter(
            next_month.to_timestamp(),
            prediction,
            label="Predicted expense",
            marker='o',
            color='red'
        )

        ax.set_title("Monthly Expense Prediction")

        ax.set_xlabel("Month")

        ax.set_ylabel("Expense")

        ax.legend()

        plt.xticks(rotation=45)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


def detect_unusual_expense():

    df = load_expenses()

    st.subheader("🚨 Unusual Expense Detection")

    if len(df) < 5:

        st.warning(
            "Record at least 5 expenses to see unusual transactions"
        )

        return

    if st.button("Detect Unusual Expenses"):

        model = IsolationForest(
            contamination=0.05,
            random_state=42
        )

        df["Anomaly"] = model.fit_predict(
            df[["Amount"]]
        )

        unusual = df[df["Anomaly"] == -1]

        if unusual.empty:

            st.success("No unusual transactions")

        else:

            st.error("Unusual data detected!")

            st.dataframe(
                unusual[
                    ["Date", "Category", "Amount", "Description"]
                ],
                use_container_width=True
            )




st.sidebar.title("📌 MENU")

choice = st.sidebar.selectbox(
    "Select Option",
    [
        "Dashboard",
        "Add Expense",
        "View Expense History",
        "Delete Expense",
        "Expense Summary",
        "Visualize Expense",
        "Monthly Budget Analysis",
        "Predict Future Expense",
        "Detect Unusual Expenses"
    ]
)




if choice == "Dashboard":

    df = load_expenses()

    st.header("📊 Dashboard")

    if df.empty:

        st.info("No expenses available. Add your first expense!")

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Spending",
            f"₹{df['Amount'].sum():.2f}"
        )

        col2.metric(
            "Average Spending",
            f"₹{df['Amount'].mean():.2f}"
        )

        col3.metric(
            "Transactions",
            len(df)
        )

        st.subheader("Recent Expenses")

        st.dataframe(
            df.sort_values(
                "Date",
                ascending=False
            ).head(5),
            use_container_width=True
        )


elif choice == "Add Expense":

    add_expenses()


elif choice == "View Expense History":

    view_expenses()


elif choice == "Delete Expense":

    delete_expenses()


elif choice == "Expense Summary":

    expense_summary()


elif choice == "Visualize Expense":

    visualize_expenses()


elif choice == "Monthly Budget Analysis":

    budget_analyse()


elif choice == "Predict Future Expense":

    predict_expense()


elif choice == "Detect Unusual Expenses":

    detect_unusual_expense()


st.sidebar.markdown("---")

st.sidebar.info(
    "Personal Expense Analyser\n\n"
    "Developed using Python, Streamlit, "
    "Pandas, NumPy, Matplotlib and Scikit-learn."
)
