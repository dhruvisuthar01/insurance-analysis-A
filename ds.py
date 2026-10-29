
"""
INSURANCE ANALYTICS USING MACHINE LEARNING
Python for Data Science / PDS-CIPAT Project

Run:
    python insurance_analytics_gui.py

Install:
    pip install pandas numpy matplotlib scikit-learn scipy requests beautifulsoup4
Optional:
    pip install xgboost

Project syllabus coverage:
A  Python Core Concepts       -> functions, classes, loops, conditions, exceptions
B  Data Structures             -> lists, dictionaries, tuples, sets
C  Python Libraries            -> NumPy, Pandas, Matplotlib, SciPy, sklearn
D  Custom Modules              -> insurance_utils.py
E  Exception Handling          -> try/except in file, web and CRUD operations
F  Web Scraping                -> Web Scraping tab
G  File Handling               -> CSV open/save/export
H  CRUD Operations             -> Create/Read/Update/Delete records
I  NumPy                       -> numerical generation/statistics
J  SciPy                       -> t-test, chi-square, confidence interval
K  Pandas                      -> cleaning, grouping, aggregation, CSV
L  Data Preprocessing          -> cleaning + categorical encoding + scaling
M  Data Visualization         -> Matplotlib charts
N  Statistical Analysis       -> descriptive statistics/correlation
O  Probability                 -> probability tab calculations
P  Sampling & Inference       -> random sampling + bootstrap confidence interval
Q  Hypothesis Testing         -> t-test + chi-square
R  Machine Learning            -> regression + classification models
"""
print("Starting... loading libraries...", flush=True)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import io
import threading

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)


# ---------------------------------------------------------------------------
# Embedded custom module: insurance_utils.py
# Kept inside this single file so the project runs with ONE Python file.
# ---------------------------------------------------------------------------

"""
Custom helper module for the Insurance Analytics project.

This module demonstrates:
- custom modules/functions
- Python core concepts
- NumPy/Pandas based preprocessing
- probability/statistics helpers
- reusable ML feature preparation
"""
import numpy as np
import pandas as pd
from scipy import stats

REQUIRED = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
BANDS = [(30, "Low Risk", 0.00), (60, "Medium Risk", 0.25), (100, "High Risk", 0.50)]


def sample_data(n=1000):
    """Create a reproducible sample insurance dataset using NumPy + Pandas."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "age": rng.integers(18, 65, n),
        "sex": rng.choice(["male", "female"], n),
        "bmi": rng.normal(30, 6, n).round(1).clip(16, 50),
        "children": rng.poisson(1.1, n).clip(0, 5),
        "smoker": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
        "region": rng.choice(
            ["northeast", "northwest", "southeast", "southwest"], n
        ),
    })
    df["charges"] = (
        2000
        + 250 * df.age
        + 300 * (df.bmi - 25).clip(lower=0)
        + 450 * df.children
        + 23000 * (df.smoker == "yes")
        + rng.normal(0, 3000, n)
    ).clip(lower=1100).round(2)
    return df


def clean_data(raw):
    """Clean numeric/categorical columns and return a quality report."""
    missing_cols = [c for c in REQUIRED if c not in raw.columns]
    if missing_cols:
        raise ValueError("Missing columns: " + ", ".join(missing_cols))

    raw = raw[REQUIRED].copy()
    for col in ["age", "bmi", "children", "charges"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    for col in ["sex", "smoker", "region"]:
        raw[col] = (
            raw[col].astype(str).str.strip().str.lower().replace("nan", np.nan)
        )

    missing_total = int(raw.isna().sum().sum())
    dups = int(raw.duplicated().sum())
    df = raw.dropna().drop_duplicates().reset_index(drop=True)

    def outliers(col):
        if df.empty:
            return 0
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        return int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())

    cells = max(raw.shape[0] * raw.shape[1], 1)
    lines = [
        "DATA QUALITY REPORT", "=" * 58,
        f"Rows before cleaning     : {len(raw)}",
        f"Rows after cleaning      : {len(df)}",
        f"Missing values (total)   : {missing_total} ({missing_total / cells:.2%} of cells)",
        f"Duplicate rows removed   : {dups}",
        f"Numerical features       : {df.select_dtypes('number').shape[1]}",
        f"Categorical features     : {df.select_dtypes(exclude='number').shape[1]}",
        "", "Missing values per column (before cleaning):",
        raw.isna().sum().to_string(), "",
        "Outliers detected (IQR rule):",
        f"  bmi     : {outliers('bmi')}",
        f"  charges : {outliers('charges')}", "",
        "Data types:", df.dtypes.to_string(),
    ]
    return df, "\n".join(lines)


def make_X(df, regions):
    """Preprocess categorical variables into numeric model features."""
    X = df[["age", "bmi", "children"]].copy()
    X["male"] = (df["sex"] == "male").astype(int)
    X["smoker"] = (df["smoker"] == "yes").astype(int)
    for r in regions:
        X["region_" + str(r)] = (df["region"] == r).astype(int)
    return X


def band_for(score):
    for top, label, loading in BANDS:
        if score <= top:
            return label, loading
    return BANDS[-1][1], BANDS[-1][2]


def probability_summary(df):
    """Basic probability calculations for the project."""
    n = max(len(df), 1)
    high_threshold = df["charges"].quantile(0.75)
    high = df["charges"] >= high_threshold
    smoker = df["smoker"] == "yes"
    smoker_and_high = smoker & high
    return {
        "P(smoker)": float(smoker.mean()),
        "P(high-cost)": float(high.mean()),
        "P(smoker AND high-cost)": float(smoker_and_high.mean()),
        "P(high-cost | smoker)": float(high[smoker].mean()) if smoker.any() else 0.0,
        "P(high-cost | non-smoker)": float(high[~smoker].mean()) if (~smoker).any() else 0.0,
        "Sample size": n,
    }


def statistical_summary(df):
    """Descriptive statistics, correlation and confidence interval."""
    charges = df["charges"].dropna().to_numpy(dtype=float)
    n = len(charges)
    mean = float(np.mean(charges)) if n else 0.0
    median = float(np.median(charges)) if n else 0.0
    std = float(np.std(charges, ddof=1)) if n > 1 else 0.0

    if n > 1:
        ci_low, ci_high = stats.t.interval(
            0.95, df=n - 1, loc=mean, scale=stats.sem(charges)
        )
    else:
        ci_low = ci_high = mean

    corr = df[["age", "bmi", "children", "charges"]].corr()["charges"].drop("charges")
    return {
        "n": n, "mean": mean, "median": median, "std": std,
        "min": float(np.min(charges)) if n else 0.0,
        "max": float(np.max(charges)) if n else 0.0,
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "correlation": corr,
    }


def hypothesis_tests(df):
    """Example hypothesis tests using SciPy."""
    smokers = df.loc[df["smoker"] == "yes", "charges"]
    nonsmokers = df.loc[df["smoker"] == "no", "charges"]
    if len(smokers) >= 2 and len(nonsmokers) >= 2:
        t_stat, p_value = stats.ttest_ind(smokers, nonsmokers, equal_var=False)
    else:
        t_stat, p_value = np.nan, np.nan

    table = pd.crosstab(df["smoker"], df["sex"])
    if table.shape[0] >= 2 and table.shape[1] >= 2:
        chi2, chi_p, dof, _ = stats.chi2_contingency(table)
    else:
        chi2, chi_p, dof = np.nan, np.nan, np.nan

    return {
        "Welch t-test: smoker vs non-smoker": (float(t_stat), float(p_value)),
        "Chi-square: smoker vs sex": (float(chi2), float(chi_p), int(dof)),
    }


def bootstrap_mean_ci(values, iterations=1000, confidence=0.95, seed=42):
    """Bootstrap sampling/inference for the mean."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True)
    means = samples.mean(axis=1)
    alpha = 1 - confidence
    return (
        float(values.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_SCRAPER = True
except Exception:
    HAS_SCRAPER = False


class App(tk.Tk):
    """Main Tkinter application demonstrating the complete syllabus."""

    def __init__(self):
        super().__init__()
        self.title("INSUREX | Pastel Insurance Analytics Studio")
        self.geometry("1500x940")
        self.minsize(1280, 820)

        self.df, self.quality = clean_data(sample_data())
        self.source = "Built-in sample data"
        self.regions = sorted(self.df["region"].unique())
        self.risk = self.df["charges"].rank(pct=True) * 100
        self.reg_model = None
        self.cls_model = None
        self.reg_best_name = ""
        self.cls_best_name = ""

        self.build_style()
        self.build_header()
        self.build_tabs()
        self.refresh_all()

    # ---------------------------- UI
    def build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # PASTEL palette: warm ivory + lavender + sage + blush.
        self.configure(bg="#FFF8F1")

        style.configure("TFrame", background="#FFF8F1")
        style.configure("Card.TFrame", background="#FFFCF8")
        style.configure("Panel.TFrame", background="#F3EAF4")
        style.configure("Sidebar.TFrame", background="#5B5268")
        style.configure("SidebarInner.TFrame", background="#5B5268")

        style.configure(
            "TLabel", background="#FFF8F1", foreground="#4B4455",
            font=("Segoe UI", 10)
        )
        style.configure(
            "Title.TLabel", background="#FFF8F1", foreground="#5B5268",
            font=("Segoe UI", 25, "bold")
        )
        style.configure(
            "Sub.TLabel", background="#FFF8F1", foreground="#8B8193",
            font=("Segoe UI", 10)
        )
        style.configure(
            "Hero.TLabel", background="#FFFCF8", foreground="#5B5268",
            font=("Segoe UI", 18, "bold")
        )
        style.configure(
            "Metric.TLabel", background="#FFFCF8", foreground="#8AA7A0",
            font=("Segoe UI", 24, "bold")
        )
        style.configure(
            "MetricSmall.TLabel", background="#FFFCF8", foreground="#8B8193",
            font=("Segoe UI", 9, "bold")
        )
        style.configure(
            "Signal.TLabel", background="#F3EAF4", foreground="#4B4455",
            font=("Segoe UI", 10)
        )

        style.configure(
            "Card.TLabelframe", background="#FFFCF8",
            foreground="#5B5268", bordercolor="#DED4DF",
            relief="solid", borderwidth=1
        )
        style.configure(
            "Card.TLabelframe.Label", background="#FFFCF8",
            foreground="#5B5268", font=("Segoe UI", 10, "bold")
        )

        style.configure(
            "TNotebook", background="#FFF8F1",
            borderwidth=0, tabmargins=(0, 0, 0, 0)
        )
        style.configure(
            "TNotebook.Tab", background="#EEE5EF", foreground="#756B7C",
            padding=(18, 9), font=("Segoe UI", 9, "bold"), borderwidth=0
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#5B5268")],
            foreground=[("selected", "#FFFCF8")]
        )

        style.configure(
            "Treeview",
            background="#FFFCF8", foreground="#4B4455",
            fieldbackground="#FFFCF8", rowheight=34,
            font=("Segoe UI", 9), borderwidth=0
        )
        style.configure(
            "Treeview.Heading",
            background="#5B5268", foreground="#FFFCF8",
            font=("Segoe UI", 9, "bold"), padding=8
        )
        style.map(
            "Treeview",
            background=[("selected", "#E6F1EE")],
            foreground=[("selected", "#5B5268")]
        )

        style.configure(
            "TEntry", fieldbackground="#FFFCF8", foreground="#4B4455",
            bordercolor="#D9CFD9", lightcolor="#D9CFD9",
            darkcolor="#D9CFD9", padding=7
        )
        style.configure(
            "TCombobox", fieldbackground="#FFFCF8", foreground="#4B4455",
            padding=6
        )

        style.configure(
            "TButton", background="#EEE6E0", foreground="#5B5268",
            font=("Segoe UI", 9, "bold"), padding=(13, 8),
            borderwidth=0
        )
        style.map(
            "TButton",
            background=[("active", "#E3D9D2"), ("pressed", "#D9CEC5")],
            foreground=[("active", "#5B5268")]
        )
        style.configure(
            "Accent.TButton", background="#8AA7A0", foreground="#FFFCF8",
            font=("Segoe UI", 9, "bold"), padding=(14, 9)
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#78968F"), ("pressed", "#66827C")]
        )
        style.configure(
            "Soft.TButton", background="#FFFCF8", foreground="#5B5268",
            font=("Segoe UI", 9, "bold"), padding=(13, 8),
            bordercolor="#DDD2D8"
        )
        style.map(
            "Soft.TButton",
            background=[("active", "#F5EEE9")]
        )
        style.configure(
            "Workspace.TButton", background="#5B5268", foreground="#D8CFDE",
            font=("Segoe UI", 10, "bold"), padding=(14, 11),
            anchor="w", borderwidth=0
        )
        style.map(
            "Workspace.TButton",
            background=[("active", "#74677F")],
            foreground=[("active", "#FFFCF8")]
        )
        style.configure(
            "WorkspaceActive.TButton", background="#8AA7A0",
            foreground="#FFFCF8", font=("Segoe UI", 10, "bold"),
            padding=(14, 11), anchor="w", borderwidth=0
        )

        style.configure(
            "SidebarTitle.TLabel", background="#5B5268", foreground="#FFFCF8",
            font=("Segoe UI", 17, "bold")
        )
        style.configure(
            "SidebarSub.TLabel", background="#5B5268", foreground="#B9AFBE",
            font=("Segoe UI", 8)
        )
        style.configure(
            "SidebarCaption.TLabel", background="#5B5268", foreground="#9A8EA0",
            font=("Segoe UI", 8, "bold")
        )

    def build_header(self):
        header = ttk.Frame(self, padding=(24, 15))
        header.pack(fill="x")

        left = ttk.Frame(header)
        left.pack(side="left", fill="x", expand=True)

        brand = ttk.Frame(left)
        brand.pack(anchor="w")
        ttk.Label(brand, text="INSUREX", style="Title.TLabel").pack(side="left")
        ttk.Label(
            brand, text="  /  RISK INTELLIGENCE",
            style="Sub.TLabel"
        ).pack(side="left", pady=(8, 0))

        ttk.Label(
            left,
            text="INSURANCE ANALYTICS  •  PDS-CIPAT  •  DECISION SUPPORT",
            style="Sub.TLabel"
        ).pack(anchor="w", pady=(2, 0))

        actions = ttk.Frame(header)
        actions.pack(side="right")

        self.status = ttk.Label(
            actions, text="●  LIVE WORKSPACE", style="Sub.TLabel"
        )
        self.status.pack(side="right", padx=(14, 0))

        ttk.Button(
            actions, text="Export CSV", command=self.export_csv,
            style="Soft.TButton"
        ).pack(side="right", padx=4)

        ttk.Button(
            actions, text="Load Dataset", command=self.load_csv,
            style="Accent.TButton"
        ).pack(side="right", padx=4)

    def build_tabs(self):
        # Product-style navigation: a persistent command rail replaces a
        # long row of generic tabs. Analytical modules remain grouped inside
        # focused workspaces.
        shell = ttk.Frame(self)
        shell.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        sidebar = ttk.Frame(shell, style="Sidebar.TFrame", width=220, padding=16)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ttk.Label(sidebar, text="INSUREX", style="SidebarTitle.TLabel").pack(anchor="w")
        ttk.Label(
            sidebar, text="ANALYTICS COMMAND RAIL",
            style="SidebarSub.TLabel"
        ).pack(anchor="w", pady=(0, 20))

        root = ttk.Notebook(shell)
        self.tabs = root
        root.pack(side="left", fill="both", expand=True, padx=(12, 0))

        # Hide the default notebook tab strip; navigation is handled by the
        # command rail so the interface feels like an application, not a form.
        try:
            style = ttk.Style(self)
            style.layout("Hidden.TNotebook.Tab", [])
            style.configure("Hidden.TNotebook", tabmargins=0, borderwidth=0)
            root.configure(style="Hidden.TNotebook")
        except Exception:
            pass

        workspaces = [
            ("01", "Overview"),
            ("02", "Data Lab"),
            ("03", "Statistics Lab"),
            ("04", "Visual Intelligence"),
            ("05", "ML Studio"),
            ("06", "Decision Engine"),
        ]

        nav_buttons = []
        for idx, (num, title) in enumerate(workspaces):
            b = ttk.Button(
                sidebar,
                text=f"{num}   {title}",
                style="WorkspaceActive.TButton" if idx == 0 else "Workspace.TButton",
                command=lambda i=idx: self.select_workspace(i, nav_buttons)
            )
            b.pack(fill="x", pady=3)
            nav_buttons.append(b)

        ttk.Label(
            sidebar, text="PROJECT MODULES",
            style="SidebarCaption.TLabel"
        ).pack(anchor="w", pady=(25, 8))

        ttk.Label(
            sidebar,
            text="Data → Statistics →\nVisuals → ML → Decisions",
            style="SidebarSub.TLabel",
            justify="left"
        ).pack(anchor="w")

        bottom = ttk.Frame(sidebar, style="SidebarInner.TFrame")
        bottom.pack(side="bottom", fill="x")
        ttk.Label(
            bottom, text="PYTHON FOR DATA SCIENCE",
            style="SidebarCaption.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            bottom, text="PDS-CIPAT  •  BE III / SEM V",
            style="SidebarSub.TLabel"
        ).pack(anchor="w", pady=(2, 0))

        def workspace():
            page = ttk.Frame(root)
            root.add(page)
            nb = ttk.Notebook(page)
            nb.pack(fill="both", expand=True, padx=2, pady=2)
            return nb

        # 1 — Overview
        overview = ttk.Frame(root)
        root.add(overview)
        self._dashboard_parent = overview
        self.build_dashboard_tab()

        # 2 — Data Lab
        data_nb = workspace()
        self.tabs = data_nb
        self.build_data_tab()
        self.build_text_tab("Quality", "quality_box")

        # 3 — Statistics Lab
        stats_nb = workspace()
        self.tabs = stats_nb
        self.build_text_tab("Descriptive", "summary_box")
        self.build_probability_tab()
        self.build_sampling_tab()
        self.build_hypothesis_tab()

        # 4 — Visual Intelligence
        visual_nb = workspace()
        self.tabs = visual_nb
        self.build_chart_tab()
        self.build_scraping_tab()

        # 5 — ML Studio
        ml_nb = workspace()
        self.tabs = ml_nb
        self.build_reg_tab()
        self.build_cls_tab()

        # 6 — Decision Engine
        pred_nb = workspace()
        self.tabs = pred_nb
        self.build_predict_tab()

        self.tabs = root
        root.select(0)

    def select_workspace(self, index, buttons):
        self.tabs.select(index)
        for i, button in enumerate(buttons):
            button.configure(
                style="WorkspaceActive.TButton" if i == index else "Workspace.TButton"
            )

    def build_dashboard_tab(self):
        tab = getattr(self, "_dashboard_parent", None)
        if tab is None:
            tab = ttk.Frame(self.tabs)
            self.tabs.add(tab, text="◆ Executive Dashboard")

        hero = ttk.Frame(tab, style="Card.TFrame", padding=20)
        hero.pack(fill="x", padx=14, pady=14)

        hero_left = ttk.Frame(hero, style="Card.TFrame")
        hero_left.pack(side="left", fill="x", expand=True)

        ttk.Label(
            hero_left, text="Insurance Analytics Command Center",
            style="Hero.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            hero_left,
            text="A single workspace for portfolio monitoring, statistical analysis, "
                 "risk discovery and predictive modelling.",
            style="MetricSmall.TLabel"
        ).pack(anchor="w", pady=(5, 12))

        quick = ttk.Frame(hero_left, style="Card.TFrame")
        quick.pack(anchor="w")

        ttk.Button(
            quick, text="Explore Data", style="Accent.TButton",
            command=lambda: self.tabs.select(1)
        ).pack(side="left", padx=(0, 7))

        ttk.Button(
            quick, text="Open Analytics", style="Soft.TButton",
            command=lambda: self.tabs.select(2)
        ).pack(side="left", padx=7)

        ttk.Button(
            quick, text="Compare Models", style="Soft.TButton",
            command=lambda: self.tabs.select(4)
        ).pack(side="left", padx=7)

        ttk.Button(
            quick, text="Predict Customer", style="Soft.TButton",
            command=lambda: self.tabs.select(5)
        ).pack(side="left", padx=7)

        # KPI strip
        cards = ttk.Frame(tab)
        cards.pack(fill="x", padx=14, pady=(0, 10))

        self.metric_vars = {}
        metrics = [
            ("TOTAL RECORDS", "records", "Dataset size"),
            ("AVERAGE CHARGE", "avg", "Portfolio average"),
            ("SMOKER RATE", "smoker", "Population share"),
            ("MAXIMUM CHARGE", "risk", "Observed peak"),
        ]

        for title, key, sub in metrics:
            card = ttk.Frame(cards, style="Card.TFrame", padding=16)
            card.pack(side="left", fill="both", expand=True, padx=5)

            ttk.Label(card, text=title, style="MetricSmall.TLabel").pack(anchor="w")
            var = tk.StringVar(value="—")
            self.metric_vars[key] = var
            ttk.Label(card, textvariable=var, style="Metric.TLabel").pack(
                anchor="w", pady=(5, 0)
            )
            ttk.Label(card, text=sub, style="MetricSmall.TLabel").pack(anchor="w")

        # Main analytics area
        main = ttk.Frame(tab)
        main.pack(fill="both", expand=True, padx=14, pady=4)

        chart_panel = ttk.Frame(main, style="Card.TFrame", padding=14)
        chart_panel.pack(side="left", fill="both", expand=True, padx=(5, 7))

        ttk.Label(
            chart_panel, text="PORTFOLIO CHARGE PROFILE",
            style="Hero.TLabel"
        ).pack(anchor="w")

        self.dashboard_fig = Figure(figsize=(7.0, 3.2), dpi=100)
        self.dashboard_fig.patch.set_facecolor("#FFFCF8")
        self.dashboard_ax = self.dashboard_fig.add_subplot(111)
        self.dashboard_canvas = FigureCanvasTkAgg(
            self.dashboard_fig, master=chart_panel
        )
        self.dashboard_canvas.get_tk_widget().pack(
            fill="both", expand=True, pady=(8, 0)
        )

        signals = ttk.Frame(main, style="Card.TFrame", padding=14)
        signals.pack(side="right", fill="both", expand=True, padx=(7, 5))

        ttk.Label(
            signals, text="RISK SIGNALS", style="Hero.TLabel"
        ).pack(anchor="w", pady=(0, 8))

        self.signal_vars = {}
        for title, key in [
            ("Strongest numeric driver", "driver"),
            ("Smoker / non-smoker gap", "gap"),
            ("Highest-cost customer", "highest"),
            ("Data quality", "quality"),
        ]:
            box = ttk.Frame(signals, style="Panel.TFrame", padding=12)
            box.pack(fill="x", pady=5)
            ttk.Label(box, text=title.upper(), style="MetricSmall.TLabel").pack(anchor="w")
            var = tk.StringVar(value="—")
            self.signal_vars[key] = var
            ttk.Label(box, textvariable=var, style="SignalValue.TLabel").pack(
                anchor="w", pady=(4, 0)
            )

        # Insight footer
        insight = ttk.Frame(tab, style="Card.TFrame", padding=14)
        insight.pack(fill="x", padx=14, pady=(5, 14))

        ttk.Label(
            insight, text="EXECUTIVE INSIGHT", style="MetricSmall.TLabel"
        ).pack(anchor="w")

        self.insight_box = tk.Text(
            insight, height=3, wrap="word", relief="flat",
            bg="#FFFCF8", fg="#4B4455", insertbackground="white",
            font=("Segoe UI", 10), padx=2, pady=6
        )
        self.insight_box.pack(fill="x", pady=(4, 0))

    def update_dashboard(self):
        if not hasattr(self, "metric_vars"):
            return

        n = len(self.df)
        avg = self.df["charges"].mean() if n else 0
        smoker_rate = (
            self.df["smoker"].eq("yes").mean() * 100
            if n else 0
        )
        top = self.df.loc[self.df["charges"].idxmax()] if n else None

        self.metric_vars["records"].set(f"{n:,}")
        self.metric_vars["avg"].set(f"₹{avg:,.0f}")
        self.metric_vars["smoker"].set(f"{smoker_rate:.1f}%")
        self.metric_vars["risk"].set(
            f"₹{top['charges']:,.0f}" if top is not None else "—"
        )

        if not n:
            return

        smoker_avg = self.df.loc[
            self.df["smoker"].eq("yes"), "charges"
        ].mean()
        nonsmoker_avg = self.df.loc[
            self.df["smoker"].eq("no"), "charges"
        ].mean()

        corr = self.df[
            ["age", "bmi", "children", "charges"]
        ].corr()["charges"].drop("charges")

        strongest = corr.abs().idxmax()

        self.signal_vars["driver"].set(
            f"{strongest.title()}  ({corr[strongest]:+.3f})"
        )
        self.signal_vars["gap"].set(
            f"₹{(smoker_avg - nonsmoker_avg):,.0f} higher for smokers"
        )
        self.signal_vars["highest"].set(
            f"₹{top['charges']:,.2f}"
        )
        missing = int(self.df.isna().sum().sum())
        self.signal_vars["quality"].set(
            "Clean • No missing values" if missing == 0
            else f"{missing} missing value(s)"
        )

        self.insight_box.delete("1.0", "end")
        self.insight_box.insert(
            "1.0",
            f"Portfolio average is ₹{avg:,.2f}. "
            f"Smokers represent {smoker_rate:.1f}% of records and have an average "
            f"charge of ₹{smoker_avg:,.0f}, compared with ₹{nonsmoker_avg:,.0f} "
            f"for non-smokers. Among numeric variables, {strongest} has the strongest "
            f"correlation with charges ({corr[strongest]:+.3f}). "
            f"The dashboard is a monitoring layer; final decisions should use the "
            f"statistical and model evaluation tabs."
        )

        # Professional dashboard chart: average charges by smoker status.
        ax = self.dashboard_ax
        ax.clear()
        ax.set_facecolor("#FFFCF8")

        labels = ["Non-smoker", "Smoker"]
        values = [nonsmoker_avg, smoker_avg]
        bars = ax.bar(labels, values, width=0.52, color=["#9B8AA5", "#8AA7A0"])

        ax.set_title(
            "Average Insurance Charge by Smoking Status",
            color="#5B5268", fontsize=11, fontweight="bold", pad=12
        )
        ax.set_ylabel("Average charge (₹)", color="#8E8796")
        ax.tick_params(colors="#8E8796")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", alpha=0.12, color="#B8AEBB")

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"₹{value:,.0f}",
                ha="center", va="bottom",
                color="#5B5268", fontsize=9, fontweight="bold"
            )

        self.dashboard_fig.tight_layout(pad=1.2)
        self.dashboard_canvas.draw_idle()

    def build_data_tab(self):
        f = ttk.Frame(self.tabs)
        self.tabs.add(f, text="Data & CRUD")

        form = ttk.LabelFrame(f, text="Create / Update Record", padding=8)
        form.pack(fill="x", padx=8, pady=8)

        self.crud_vars = {c: tk.StringVar() for c in REQUIRED}
        fields = [
            ("age", "38"), ("sex", "male"), ("bmi", "27.5"),
            ("children", "1"), ("smoker", "no"), ("region", "northeast"),
            ("charges", "10000")
        ]
        for i, (c, default) in enumerate(fields):
            self.crud_vars[c].set(default)
            r, col = divmod(i, 7)
            ttk.Label(form, text=c).grid(row=0, column=col, padx=4)
            w = ttk.Entry(form, textvariable=self.crud_vars[c], width=14)
            w.grid(row=1, column=col, padx=4)

        btns = ttk.Frame(form)
        btns.grid(row=2, column=0, columnspan=7, pady=8)
        ttk.Button(btns, text="Add / Create", command=self.create_record).pack(side="left", padx=4)
        ttk.Button(btns, text="Update Selected", command=self.update_record).pack(side="left", padx=4)
        ttk.Button(btns, text="Delete Selected", command=self.delete_record).pack(side="left", padx=4)
        ttk.Button(btns, text="Clear Form", command=self.clear_crud_form).pack(side="left", padx=4)

        self.tree = ttk.Treeview(f, columns=REQUIRED, show="headings")
        for c in REQUIRED:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=125, anchor="center")
        sb = ttk.Scrollbar(f, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 8))
        sb.pack(side="right", fill="y", pady=(0, 8))
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

    def build_text_tab(self, title, attr):
        f = ttk.Frame(self.tabs)
        self.tabs.add(f, text=title)

        header = ttk.Frame(f, style="Card.TFrame", padding=(16, 12))
        header.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Label(
            header, text=title.upper(), style="Hero.TLabel"
        ).pack(anchor="w")

        descriptions = {
            "Quality": "Data validation, missing values, duplicates, data types and preprocessing checks.",
            "Descriptive": "Distribution, central tendency, spread, group comparisons and correlation.",
        }
        ttk.Label(
            header,
            text=descriptions.get(title, "Analytical output generated from the current dataset."),
            style="MetricSmall.TLabel"
        ).pack(anchor="w", pady=(3, 0))

        box = tk.Text(
            f, font=("Cascadia Mono", 10), wrap="none",
            relief="flat", bd=0,
            bg="#FBF5EF", fg="#4B4455",
            insertbackground="#5B5268",
            padx=16, pady=14
        )
        box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        setattr(self, attr, box)

    def build_probability_tab(self):
        f = ttk.Frame(self.tabs)
        self.tabs.add(f, text="Probability")
        ttk.Label(
            f, text="Probability Analysis", style="Title.TLabel"
        ).pack(anchor="w", padx=10, pady=10)
        self.prob_box = tk.Text(f, font=("Cascadia Mono", 10), bg="#FBF5EF", fg="#4B4455", relief="flat", bd=0, padx=18, pady=18)
        self.prob_box.pack(fill="both", expand=True, padx=10, pady=10)

    def build_sampling_tab(self):
        f = ttk.Frame(self.tabs)
        self.tabs.add(f, text="Sampling & Inference")

        top = ttk.Frame(f, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Sample size:").pack(side="left")
        self.sample_size = tk.IntVar(value=50)
        ttk.Spinbox(top, from_=5, to=1000, textvariable=self.sample_size, width=8).pack(side="left", padx=5)
        ttk.Button(top, text="Take Random Sample", command=self.run_sampling).pack(side="left", padx=5)
        ttk.Button(top, text="Bootstrap 95% CI", command=self.run_bootstrap).pack(side="left", padx=5)

        self.sampling_box = tk.Text(f, font=("Cascadia Mono", 10), bg="#FBF5EF", fg="#4B4455", relief="flat", bd=0, padx=18, pady=18)
        self.sampling_box.pack(fill="both", expand=True, padx=10, pady=10)

    def build_hypothesis_tab(self):
        f = ttk.Frame(self.tabs)
        self.tabs.add(f, text="Hypothesis Testing")
        ttk.Label(
            f, text="H0: Average charges are equal for smokers and non-smokers",
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", padx=10, pady=8)
        self.hyp_box = tk.Text(f, font=("Cascadia Mono", 10), bg="#FBF5EF", fg="#4B4455", relief="flat", bd=0, padx=18, pady=18)
        self.hyp_box.pack(fill="both", expand=True, padx=10, pady=8)

    def build_scraping_tab(self):
        f = ttk.Frame(self.tabs)
        self.tabs.add(f, text="Web Scraping")

        bar = ttk.Frame(f, padding=10)
        bar.pack(fill="x")
        ttk.Label(bar, text="URL:").pack(side="left")
        self.url_var = tk.StringVar(value="https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)")
        ttk.Entry(bar, textvariable=self.url_var, width=75).pack(side="left", padx=6)
        ttk.Button(bar, text="Scrape Tables", command=self.scrape_web).pack(side="left")

        self.scrape_box = tk.Text(f, font=("Cascadia Mono", 9), wrap="none", bg="#FBF5EF", fg="#4B4455", relief="flat", bd=0, padx=14, pady=14)
        self.scrape_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def make_table(self, parent, cols, height=6):
        t = ttk.Treeview(parent, columns=cols, show="headings", height=height)
        for c in cols:
            t.heading(c, text=c)
            t.column(c, width=120, anchor="center")
        t.tag_configure("best", background="#F7E7C6")
        return t

    def build_chart_tab(self):
        f = ttk.Frame(self.tabs)
        self.tabs.add(f, text="Data Visualization")

        bar = ttk.Frame(f, padding=6)
        bar.pack(fill="x")
        self.charts = {
            "Charges distribution": self.c_hist,
            "Smokers vs non-smokers": self.c_smoker,
            "Age vs charges": self.c_age,
            "BMI vs charges": self.c_bmi,
            "Average charges by region": self.c_region,
            "Age group distribution": self.c_agegroup,
            "Risk band distribution": self.c_riskbands,
            "Risk score vs charges": self.c_riskscatter,
            "Correlation heatmap": self.c_corr,
        }
        self.chart_var = tk.StringVar(value=list(self.charts)[0])
        cb = ttk.Combobox(
            bar, textvariable=self.chart_var, values=list(self.charts),
            state="readonly", width=34
        )
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.draw_chart())

        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor="#FFFCF8")
        self.canvas = FigureCanvasTkAgg(self.fig, master=f)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def build_reg_tab(self):
        f = ttk.Frame(self.tabs)
        self.tabs.add(f, text="Regression Models")
        top = ttk.Frame(f)
        top.pack(fill="x", padx=8, pady=(8, 2))
        ttk.Label(top, text="Premium / Charges Prediction",
                  font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Button(top, text="Run Models", command=self.train_regression).pack(side="right")
        self.reg_table = self.make_table(
            f, ["Model", "MAE", "RMSE", "R2", "MAPE %"]
        )
        self.reg_table.pack(fill="x", padx=8)
        self.reg_best = ttk.Label(f, text="")
        self.reg_best.pack(anchor="w", padx=8, pady=4)
        self.reg_fig = Figure(figsize=(7, 3.6), dpi=100, facecolor="#FFFCF8")
        self.reg_canvas = FigureCanvasTkAgg(self.reg_fig, master=f)
        self.reg_canvas.get_tk_widget().pack(fill="both", expand=True)

    def build_cls_tab(self):
        f = ttk.Frame(self.tabs)
        self.tabs.add(f, text="Classification")
        top = ttk.Frame(f)
        top.pack(fill="x", padx=8, pady=(8, 2))
        self.cls_title = ttk.Label(top, text="", font=("Segoe UI", 11, "bold"))
        self.cls_title.pack(side="left")
        ttk.Button(top, text="Run Models", command=self.train_classification).pack(side="right")
        self.cls_table = self.make_table(
            f, ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
        )
        self.cls_table.pack(fill="x", padx=8)
        self.cls_best = ttk.Label(f, text="")
        self.cls_best.pack(anchor="w", padx=8, pady=4)
        self.cls_fig = Figure(figsize=(7, 3.6), dpi=100, facecolor="#FFFCF8")
        self.cls_canvas = FigureCanvasTkAgg(self.cls_fig, master=f)
        self.cls_canvas.get_tk_widget().pack(fill="both", expand=True)

    def build_predict_tab(self):
        f = ttk.Frame(self.tabs, padding=20)
        self.tabs.add(f, text="Predict Premium")

        ttk.Label(
            f, text="Risk Assessment & Premium Calculation",
            font=("Segoe UI", 12, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=(0, 12))

        self.age = tk.StringVar(value="38")
        self.bmi = tk.StringVar(value="27.5")
        self.kids = tk.StringVar(value="1")
        self.sex = tk.StringVar(value="male")
        self.smoker = tk.StringVar(value="no")
        self.region = tk.StringVar()

        self.region_cb = ttk.Combobox(
            f, textvariable=self.region, state="readonly"
        )
        rows = [
            ("Age", ttk.Entry(f, textvariable=self.age)),
            ("BMI", ttk.Entry(f, textvariable=self.bmi)),
            ("Children", ttk.Entry(f, textvariable=self.kids)),
            ("Sex", ttk.Combobox(
                f, textvariable=self.sex, values=["male", "female"], state="readonly"
            )),
            ("Smoker", ttk.Combobox(
                f, textvariable=self.smoker, values=["yes", "no"], state="readonly"
            )),
            ("Region", self.region_cb),
        ]
        for i, (label, w) in enumerate(rows, start=1):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", pady=4)
            w.grid(row=i, column=1, sticky="w", pady=4)

        ttk.Button(f, text="Calculate", command=self.predict).grid(
            row=8, column=0, columnspan=2, pady=12
        )
        self.result = ttk.Label(f, text="", font=("Segoe UI", 12), justify="left")
        self.result.grid(row=9, column=0, columnspan=2, sticky="w")

    # ---------------------------- File handling
    def load_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            raw = pd.read_csv(path)
            self.df, self.quality = clean_data(raw)
            self.source = os.path.basename(path)
            self.refresh_all()
            messagebox.showinfo("Loaded", f"Loaded {len(self.df)} cleaned records.")
        except (FileNotFoundError, PermissionError, pd.errors.ParserError, ValueError) as e:
            messagebox.showerror("Could not load file", str(e))
        except Exception as e:
            messagebox.showerror("Unexpected error", str(e))

    def export_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return
        try:
            self.df.to_csv(path, index=False)
            messagebox.showinfo("Export complete", f"Saved data to:\n{path}")
        except (PermissionError, OSError) as e:
            messagebox.showerror("Export failed", str(e))

    # ---------------------------- CRUD
    def get_crud_record(self):
        try:
            return {
                "age": float(self.crud_vars["age"].get()),
                "sex": self.crud_vars["sex"].get().strip().lower(),
                "bmi": float(self.crud_vars["bmi"].get()),
                "children": int(self.crud_vars["children"].get()),
                "smoker": self.crud_vars["smoker"].get().strip().lower(),
                "region": self.crud_vars["region"].get().strip().lower(),
                "charges": float(self.crud_vars["charges"].get()),
            }
        except (ValueError, TypeError) as e:
            raise ValueError("Age/BMI/Children/Charges must contain valid numbers.") from e

    def create_record(self):
        try:
            record = self.get_crud_record()
            self.df = pd.concat([self.df, pd.DataFrame([record])], ignore_index=True)
            self.df, self.quality = clean_data(self.df)
            self.refresh_all()
            messagebox.showinfo("CRUD", "Record created successfully.")
        except Exception as e:
            messagebox.showerror("Create failed", str(e))

    def update_record(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("CRUD", "Select a record first.")
            return
        try:
            record = self.get_crud_record()
            idx = int(self.tree.item(sel[0], "tags")[0]) if self.tree.item(sel[0], "tags") else None
            if idx is None or idx >= len(self.df):
                # fallback: use the displayed values to locate the row
                old = self.tree.item(sel[0], "values")
                mask = (self.df.astype(str) == pd.Series(old, index=REQUIRED).astype(str)).all(axis=1)
                matches = np.flatnonzero(mask.to_numpy())
                if len(matches) == 0:
                    raise ValueError("Selected record could not be mapped to the dataset.")
                idx = int(matches[0])
            self.df.loc[idx, REQUIRED] = [record[c] for c in REQUIRED]
            self.df, self.quality = clean_data(self.df)
            self.refresh_all()
            messagebox.showinfo("CRUD", "Record updated successfully.")
        except Exception as e:
            messagebox.showerror("Update failed", str(e))

    def delete_record(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("CRUD", "Select a record first.")
            return
        if not messagebox.askyesno("Confirm delete", "Delete selected record?"):
            return
        try:
            idx = self.tree.item(sel[0], "tags")
            if idx:
                idx = int(idx[0])
                self.df = self.df.drop(self.df.index[idx]).reset_index(drop=True)
            else:
                values = self.tree.item(sel[0], "values")
                mask = (self.df.astype(str) == pd.Series(values, index=REQUIRED).astype(str)).all(axis=1)
                matches = np.flatnonzero(mask.to_numpy())
                if len(matches):
                    self.df = self.df.drop(self.df.index[int(matches[0])]).reset_index(drop=True)
            self.df, self.quality = clean_data(self.df)
            self.refresh_all()
        except Exception as e:
            messagebox.showerror("Delete failed", str(e))

    def on_tree_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        values = self.tree.item(sel[0], "values")
        for c, value in zip(REQUIRED, values):
            self.crud_vars[c].set(str(value))

    def clear_crud_form(self):
        defaults = {"age": "38", "sex": "male", "bmi": "27.5", "children": "1",
                    "smoker": "no", "region": self.regions[0] if self.regions else "northeast",
                    "charges": "10000"}
        for c, v in defaults.items():
            self.crud_vars[c].set(v)

    # ---------------------------- Refresh
    def refresh_all(self):
        self.update_dashboard()
        self.status.config(
            text=f"{self.source} | {len(self.df)} rows | 7 fields"
        )
        self.risk = self.df["charges"].rank(pct=True) * 100
        self.regions = sorted(self.df["region"].dropna().unique())

        self.tree.delete(*self.tree.get_children())
        for idx, row in self.df.head(500).iterrows():
            self.tree.insert("", "end", values=list(row), tags=(str(idx),))

        self.quality_box.delete("1.0", "end")
        self.quality_box.insert("1.0", self.quality)

        s = [
            "DESCRIPTIVE STATISTICS", "=" * 65,
            self.df.describe().round(2).to_string(), "",
            "AVERAGE CHARGES BY GROUP", "=" * 65
        ]
        for col in ["smoker", "sex", "region", "children"]:
            s += [self.df.groupby(col)["charges"].mean().round(2).to_string(), ""]
        s += [
            "CORRELATION WITH CHARGES", "=" * 65,
            self.df[["age", "bmi", "children", "charges"]]
            .corr()["charges"].round(3).to_string()
        ]
        st = statistical_summary(self.df)
        s += [
            "",
            f"95% CI for mean charges: {st['ci_low']:,.2f} to {st['ci_high']:,.2f}",
        ]
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("1.0", "\n".join(s))

        self.region_cb["values"] = self.regions
        if self.regions:
            self.region.set(self.regions[0])
        self.result.config(text="")

        self.update_probability()
        self.update_hypothesis()

        self.config(cursor="watch")
        self.update_idletasks()
        try:
            self.train_regression()
            self.train_classification()
        finally:
            self.config(cursor="")
        self.draw_chart()

    # ---------------------------- Probability / statistics
    def update_probability(self):
        p = probability_summary(self.df)
        lines = [
            "PROBABILITY ANALYSIS", "=" * 55,
            f"P(smoker)                    = {p['P(smoker)']:.3f}",
            f"P(high-cost)                 = {p['P(high-cost)']:.3f}",
            f"P(smoker AND high-cost)      = {p['P(smoker AND high-cost)']:.3f}",
            f"P(high-cost | smoker)        = {p['P(high-cost | smoker)']:.3f}",
            f"P(high-cost | non-smoker)    = {p['P(high-cost | non-smoker)']:.3f}",
            f"Sample size                  = {p['Sample size']}",
            "",
            "High-cost is defined as charges >= 75th percentile.",
        ]
        self.prob_box.delete("1.0", "end")
        self.prob_box.insert("1.0", "\n".join(lines))

    def update_hypothesis(self):
        tests = hypothesis_tests(self.df)
        t_stat, t_p = tests["Welch t-test: smoker vs non-smoker"]
        chi2, chi_p, dof = tests["Chi-square: smoker vs sex"]
        lines = [
            "HYPOTHESIS TESTING", "=" * 60,
            "1) Welch independent t-test",
            "H0: Mean charges are equal for smokers and non-smokers.",
            f"t-statistic = {t_stat:.4f}",
            f"p-value    = {t_p:.6g}",
            f"Decision    = {'Reject H0' if t_p < 0.05 else 'Fail to reject H0'} at alpha = 0.05",
            "",
            "2) Chi-square test of independence",
            "H0: Smoking status and sex are independent.",
            f"chi-square  = {chi2:.4f}",
            f"degrees df  = {dof}",
            f"p-value     = {chi_p:.6g}",
            f"Decision    = {'Reject H0' if chi_p < 0.05 else 'Fail to reject H0'} at alpha = 0.05",
        ]
        self.hyp_box.delete("1.0", "end")
        self.hyp_box.insert("1.0", "\n".join(lines))

    def run_sampling(self):
        try:
            n = int(self.sample_size.get())
            n = min(max(n, 5), len(self.df))
            sample = self.df.sample(n=n, random_state=42)
            st = statistical_summary(sample)
            lines = [
                "RANDOM SAMPLING & INFERENCE", "=" * 60,
                f"Population size : {len(self.df)}",
                f"Sample size     : {n}",
                "",
                f"Population mean charges: {self.df.charges.mean():,.2f}",
                f"Sample mean charges    : {sample.charges.mean():,.2f}",
                f"Sample median          : {sample.charges.median():,.2f}",
                f"Sample std. deviation  : {sample.charges.std():,.2f}",
                f"95% CI (sample mean)   : {st['ci_low']:,.2f} to {st['ci_high']:,.2f}",
                "",
                "Sampling method: simple random sampling without replacement.",
            ]
            self.sampling_box.delete("1.0", "end")
            self.sampling_box.insert("1.0", "\n".join(lines))
        except Exception as e:
            messagebox.showerror("Sampling error", str(e))

    def run_bootstrap(self):
        try:
            mean, low, high = bootstrap_mean_ci(self.df["charges"].values)
            lines = [
                "BOOTSTRAP INFERENCE", "=" * 60,
                "Iterations : 1000",
                "Confidence : 95%",
                f"Observed mean : {mean:,.2f}",
                f"Bootstrap CI  : {low:,.2f} to {high:,.2f}",
                "",
                "This demonstrates resampling with replacement and inference.",
            ]
            self.sampling_box.delete("1.0", "end")
            self.sampling_box.insert("1.0", "\n".join(lines))
        except Exception as e:
            messagebox.showerror("Bootstrap error", str(e))

    # ---------------------------- Web scraping
    def scrape_web(self):
        if not HAS_SCRAPER:
            messagebox.showerror(
                "Web scraping unavailable",
                "Install: pip install requests beautifulsoup4"
            )
            return
        url = self.url_var.get().strip()
        if not url.startswith(("http://", "https://")):
            messagebox.showwarning("Invalid URL", "Enter a URL beginning with http:// or https://")
            return
        self.scrape_box.delete("1.0", "end")
        self.scrape_box.insert("1.0", "Downloading page...\n")
        self.update_idletasks()

        try:
            response = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 InsuranceAnalyticsStudentProject"}
            )
            response.raise_for_status()

            # BeautifulSoup is used to demonstrate HTML parsing.
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.get_text(" ", strip=True) if soup.title else "No title"

            # Pandas read_html extracts HTML tables when present.
            tables = pd.read_html(io.StringIO(response.text))
            lines = [
                "WEB SCRAPING RESULT", "=" * 65,
                f"Page title : {title}",
                f"HTTP code  : {response.status_code}",
                f"Tables found: {len(tables)}",
                ""
            ]
            for i, table in enumerate(tables[:3], start=1):
                lines += [
                    f"--- TABLE {i} ---",
                    f"Shape: {table.shape}",
                    table.head(8).to_string(index=False),
                    ""
                ]
            if not tables:
                text = soup.get_text(" ", strip=True)
                lines.append(text[:5000])
            self.scrape_box.insert("end", "\n".join(lines))
        except ValueError:
            self.scrape_box.insert("end", "\nNo HTML tables found on this page.")
        except Exception as e:
            self.scrape_box.insert("end", f"\nScraping failed: {type(e).__name__}: {e}")

    # ---------------------------- Regression
    def train_regression(self):
        X = make_X(self.df, self.regions)
        y = self.df["charges"]
        if len(self.df) < 10:
            return
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=1
        )
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=1),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=1
            ),
            "Gradient Boosting": GradientBoostingRegressor(random_state=1),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.1, verbosity=0
            )

        results = {}
        for name, model in models.items():
            try:
                model.fit(Xtr, ytr)
                pred = model.predict(Xte)
                rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
                denom = yte.replace(0, np.nan)
                mape = float(np.mean(np.abs((yte - pred) / denom).dropna()) * 100)
                results[name] = (
                    model, mean_absolute_error(yte, pred),
                    rmse, r2_score(yte, pred), mape
                )
            except Exception as e:
                print(f"Skipping {name}: {e}")

        if not results:
            return

        best = max(results, key=lambda k: results[k][3])
        self.reg_model = results[best][0]
        self.reg_best_name = best

        self.reg_table.delete(*self.reg_table.get_children())
        for name, (_, mae, rmse, r2, mape) in results.items():
            tag = ("best",) if name == best else ()
            self.reg_table.insert(
                "", "end", tags=tag,
                values=(name, f"{mae:,.0f}", f"{rmse:,.0f}",
                        f"{r2:.3f}", f"{mape:.2f}")
            )
        self.reg_best.config(
            text=f"Best model: {best} (R² = {results[best][3]:.3f})"
            + ("" if HAS_XGB else " | Optional: pip install xgboost")
        )

        trees = {k: v for k, v in results.items()
                 if hasattr(v[0], "feature_importances_")}
        self.reg_fig.clear()
        if trees:
            tb = max(trees, key=lambda k: trees[k][3])
            imp = pd.Series(
                trees[tb][0].feature_importances_, index=X.columns
            ).sort_values().tail(10)
            ax = self.reg_fig.add_subplot(111)
            ax.barh(imp.index, imp.values)
            ax.set(
                title=f"Feature importance (top 10) - {tb}",
                xlabel="Importance score"
            )
        else:
            ax = self.reg_fig.add_subplot(111)
            ax.text(0.5, 0.5, "Feature importance not available",
                    ha="center", va="center")
        self.reg_fig.tight_layout()
        self.reg_canvas.draw()

    # ---------------------------- Classification
    def train_classification(self):
        if len(self.df) < 20:
            return
        threshold = self.df["charges"].quantile(0.75)
        y = (self.df["charges"] >= threshold).astype(int)
        X = make_X(self.df, self.regions)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=1, stratify=y
        )
        models = {
            "Logistic Regression": make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000)
            ),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=1),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=1
            ),
            "Gradient Boosting": GradientBoostingClassifier(random_state=1),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.1, verbosity=0
            )

        results = {}
        for name, model in models.items():
            try:
                model.fit(Xtr, ytr)
                pred = model.predict(Xte)
                results[name] = (
                    model, pred, accuracy_score(yte, pred),
                    precision_score(yte, pred, zero_division=0),
                    recall_score(yte, pred, zero_division=0),
                    f1_score(yte, pred, zero_division=0)
                )
            except Exception as e:
                print(f"Skipping {name}: {e}")

        if not results:
            return

        best = max(results, key=lambda k: results[k][5])
        self.cls_model = results[best][0]
        self.cls_best_name = best

        self.cls_title.config(
            text=f"High-cost claim classification "
                 f"(charges >= {threshold:,.0f}, top 25% = High Risk)"
        )
        self.cls_table.delete(*self.cls_table.get_children())
        for name, (_, _, acc, pr, rc, f1) in results.items():
            tag = ("best",) if name == best else ()
            self.cls_table.insert(
                "", "end", tags=tag,
                values=(name, f"{acc:.3f}", f"{pr:.3f}",
                        f"{rc:.3f}", f"{f1:.3f}")
            )
        self.cls_best.config(
            text=f"Best model: {best} (F1 = {results[best][5]:.3f})"
        )

        cm = confusion_matrix(yte, results[best][1], labels=[0, 1])
        self.cls_fig.clear()
        ax = self.cls_fig.add_subplot(111)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low risk", "High risk"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Low risk", "High risk"])
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black"
                )
        ax.set(title=f"Confusion matrix - {best}",
               xlabel="Predicted", ylabel="Actual")
        self.cls_fig.colorbar(im, ax=ax)
        self.cls_fig.tight_layout()
        self.cls_canvas.draw()

    # ---------------------------- Prediction
    def predict(self):
        if self.reg_model is None or self.cls_model is None:
            messagebox.showwarning("Model not ready", "Train the models first.")
            return
        try:
            row = pd.DataFrame([{
                "age": float(self.age.get()),
                "sex": self.sex.get(),
                "bmi": float(self.bmi.get()),
                "children": int(self.kids.get()),
                "smoker": self.smoker.get(),
                "region": self.region.get()
            }])
            if row.age.iloc[0] < 18 or row.bmi.iloc[0] <= 0:
                raise ValueError("Age must be >= 18 and BMI must be positive.")
        except (ValueError, TypeError) as e:
            messagebox.showwarning("Invalid input", str(e))
            return

        try:
            X = make_X(row, self.regions)
            base = max(float(self.reg_model.predict(X)[0]), 0)
            score = float((self.df["charges"] < base).mean() * 100)
            label, loading = band_for(score)
            final = base * (1 + loading)
            p_high = float(self.cls_model.predict_proba(X)[0][1])

            self.result.config(text=(
                f"Predicted risk score : {score:.1f} / 100\n"
                f"Risk category        : {label}\n"
                f"Base premium         : {base:,.2f}\n"
                f"Risk loading         : +{loading:.0%} ({base * loading:,.2f})\n"
                f"Final premium        : {final:,.2f}\n"
                f"High-cost claim prob.: {p_high:.1%}\n"
                f"Model                : {self.reg_best_name}"
            ))
        except Exception as e:
            messagebox.showerror("Prediction error", str(e))

    # ---------------------------- Charts
    def draw_chart(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        try:
            self.charts[self.chart_var.get()](ax)
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            ax.text(0.5, 0.5, f"Chart error: {e}", ha="center", va="center")
            self.canvas.draw()

    def c_hist(self, ax):
        ax.hist(self.df.charges, bins=30, edgecolor="white")
        ax.set(title="Distribution of charges", xlabel="Charges", ylabel="Count")

    def c_smoker(self, ax):
        groups = [
            self.df[self.df.smoker == s].charges for s in ["no", "yes"]
        ]
        ax.boxplot(groups)
        ax.set_xticklabels(["Non-smoker", "Smoker"])
        ax.set(title="Charges by smoking status", ylabel="Charges")

    def _scatter_by_smoker(self, ax, col, title):
        for s in ["no", "yes"]:
            d = self.df[self.df.smoker == s]
            ax.scatter(d[col], d.charges, s=12, alpha=0.6, label=f"smoker={s}")
        ax.set(title=title, xlabel=col.upper(), ylabel="Charges")
        ax.legend()

    def c_age(self, ax):
        self._scatter_by_smoker(ax, "age", "Age vs charges")

    def c_bmi(self, ax):
        self._scatter_by_smoker(ax, "bmi", "BMI vs charges")

    def c_region(self, ax):
        m = self.df.groupby("region")["charges"].mean()
        ax.bar(m.index, m.values)
        ax.set(title="Average charges by region", ylabel="Mean charges")

    def c_agegroup(self, ax):
        labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
        g = pd.cut(
            self.df.age, [17, 25, 35, 45, 55, 65, 200], labels=labels
        )
        counts = g.value_counts().reindex(labels)
        ax.bar(labels, counts.values)
        ax.set(title="Age distribution", xlabel="Age group", ylabel="Count")

    def c_riskbands(self, ax):
        counts = []
        prev = -1
        for top, _, _ in BANDS:
            counts.append(int(((self.risk > prev) & (self.risk <= top)).sum()))
            prev = top
        ax.pie(
            counts, labels=[b[1] for b in BANDS], autopct="%1.0f%%",
            wedgeprops={"width": 0.45}
        )
        ax.set_title("Risk band distribution (0-30 / 31-60 / 61-100)")

    def c_riskscatter(self, ax):
        ax.scatter(self.risk, self.df.charges, s=10, alpha=0.5)
        z = np.polyfit(self.risk, self.df.charges, 1)
        xs = np.linspace(0, 100, 50)
        ax.plot(xs, np.polyval(z, xs))
        ax.set(title="Risk score vs charges",
               xlabel="Risk score", ylabel="Charges")

    def c_corr(self, ax):
        c = self.df[["age", "bmi", "children", "charges"]].corr()
        im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(c)))
        ax.set_xticklabels(c.columns)
        ax.set_yticks(range(len(c)))
        ax.set_yticklabels(c.columns)
        for i in range(len(c)):
            for j in range(len(c)):
                ax.text(j, i, f"{c.iloc[i, j]:.2f}",
                        ha="center", va="center")
        ax.set_title("Correlation heatmap")
        self.fig.colorbar(im, ax=ax)


if __name__ == "__main__":
    try:
        print("Libraries loaded. Training models, please wait...", flush=True)
        app = App()
        print("Window opened. Close it to stop the program.")
        app.mainloop()
    except Exception:
        import traceback
        traceback.print_exc()
        input("\nAn error occurred (see above). Press Enter to close...")
