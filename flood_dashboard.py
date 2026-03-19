#!/usr/bin/env python
# coding: utf-8
"""
flood_dashboard.py
GUI design layer — imports all logic from flood_model.py
Run: python flood_dashboard.py

Requires:
  pip install customtkinter matplotlib pandas numpy scikit-learn
              requests seaborn python-dotenv pillow
"""

import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import threading
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(".env")

# ── Import your model code ────────────────────────────────────────────────────
from flood_model import (
    cities,
    get_weather_data,
    prepare_data,
    eda_analysis,
    train_model,
    regression_data,
    train_regression_model,
    predict_future,
    predict_flood_risk,
    display_flood_risks,
)

# ── Theme ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

C = {
    "bg":          "#0d1117",   # main background
    "panel":       "#010409",   # deeper background (sidebars / base)
    "card":        "#161b22",   # cards / containers
    "hover":       "#21262d",   # hover states
    
    "orange":      "#f0883e",   # GitHub warning/orange
    "blue":        "#58a6ff",   # primary accent (links/buttons)
    "green":       "#3fb950",   # success/positive
    "yellow":      "#d29922",   # warnings
    "red":         "#f85149",   # errors
    
    "text":        "#e6edf3",   # main text
    "muted":       "#7d8590",   # secondary text
    "border":      "#30363d",   # borders/dividers
}

plt.rcParams.update({
    "figure.facecolor": C["card"], "axes.facecolor": C["card"],
    "axes.edgecolor":   C["border"], "axes.labelcolor": C["muted"],
    "xtick.color":      C["muted"],  "ytick.color":     C["muted"],
    "text.color":       C["text"],   "grid.color":      C["border"],
    "grid.alpha": 0.4,  "lines.linewidth": 2,
})

# ── Helpers ───────────────────────────────────────────────────────────────────

def risk_colour(prob: float) -> str:
    return C["green"] if prob < 0.35 else C["yellow"] if prob < 0.65 else C["red"]

def risk_label(prob: float) -> str:
    return "LOW RISK" if prob < 0.35 else "MODERATE RISK" if prob < 0.65 else "HIGH RISK"

# ── Reusable widgets ──────────────────────────────────────────────────────────

class Card(ctk.CTkFrame):
    def __init__(self, master, **kw):
        super().__init__(master, fg_color=C["card"], corner_radius=10, **kw)

class KPI(ctk.CTkFrame):
    """Compact metric tile."""
    def __init__(self, master, label, value="—", unit="", color=None, **kw):
        super().__init__(master, fg_color=C["card"], corner_radius=10, **kw)
        self._color = color or C["blue"]
        ctk.CTkLabel(self, text=label, font=("SF Pro Text", 9),
                     text_color=C["muted"]).pack(anchor="w", padx=12, pady=(8, 0))
        self._v = ctk.CTkLabel(self, text=value,
                               font=("SF Pro Display", 22, "bold"),
                               text_color=self._color)
        self._v.pack(anchor="w", padx=12)
        if unit:
            ctk.CTkLabel(self, text=unit, font=("SF Pro Text", 9),
                         text_color=C["muted"]).pack(anchor="w", padx=12, pady=(0, 8))

    def set(self, value, color=None):
        self._v.configure(text=value, text_color=color or self._color)

class Divider(ctk.CTkFrame):
    def __init__(self, master, **kw):
        super().__init__(master, height=1, fg_color=C["border"], **kw)


# ── Main App ──────────────────────────────────────────────────────────────────

class FloodDashboard(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("FloodControl  ·  NCR Flood Prediction")
        self.geometry("1300x800")
        self.minsize(1100, 680)
        self.configure(fg_color=C["bg"])

        # State
        self.df:         pd.DataFrame | None = None
        self.model       = None
        self.le          = None
        self.accuracy:   float = 0.0
        self.reg_models: dict  = {}
        self._canvas:    dict  = {}   # keep FigureCanvasTkAgg refs

        self._build_ui()

    # =========================================================  Layout  ======

    def _build_ui(self):
        self._topbar()
        body = ctk.CTkFrame(self, fg_color=C["bg"])
        body.pack(fill="both", expand=True)
        self._sidebar(body)
        self._main_area(body)

    # ── Top bar ───────────────────────────────────────────────────────────────

    def _topbar(self):
        bar = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        ctk.CTkLabel(bar, text="⛈  FloodControl",
                     font=("SF Pro Display", 16, "bold"),
                     text_color=C["text"]).pack(side="left", padx=20)

        self._status = ctk.CTkLabel(bar, text="Ready — load a CSV to begin",
                                    font=("SF Pro Text", 10),
                                    text_color=C["muted"])
        self._status.pack(side="left", padx=16)

        self._clock = ctk.CTkLabel(bar, text="", font=("SF Pro Text", 11),
                                   text_color=C["muted"])
        self._clock.pack(side="right", padx=20)
        self._tick()

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _sidebar(self, parent):
        sb = ctk.CTkFrame(parent, fg_color=C["panel"],
                          corner_radius=0, width=240)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        p = {"padx": 14, "pady": 4}

        # Dataset section
        ctk.CTkLabel(sb, text="Dataset", font=("SF Pro Display", 13, "bold"),
                     text_color=C["text"]).pack(anchor="w", padx=14, pady=(18, 4))

        ctk.CTkButton(sb, text="📂  Load CSV",
                      font=("SF Pro Text", 11),
                      fg_color=C["card"], border_color=C["border"], border_width=1,
                      hover_color=C["hover"],
                      command=self._load_csv).pack(fill="x", **p)

        self._file_lbl = ctk.CTkLabel(sb, text="No file loaded",
                                      font=("SF Pro Text", 9),
                                      text_color=C["muted"], wraplength=200)
        self._file_lbl.pack(anchor="w", padx=14)

        Divider(sb).pack(fill="x", padx=14, pady=10)

        # Actions
        ctk.CTkLabel(sb, text="Actions", font=("SF Pro Text", 9, "bold"),
                     text_color=C["muted"]).pack(anchor="w", **p)

        actions = [
            ("📊  Run EDA",          self._run_eda),
            ("🤖  Train Classifier", self._train_classifier),
            ("📈  Train Regressors", self._train_regressors),
            ("🌧  Live City Risk",   self._live_city_risk),
        ]
        for lbl, cmd in actions:
            ctk.CTkButton(sb, text=lbl, font=("SF Pro Text", 11),
                          fg_color=C["card"], border_color=C["border"],
                          border_width=1, hover_color=C["hover"],
                          anchor="w", command=cmd).pack(fill="x", **p)

        Divider(sb).pack(fill="x", padx=14, pady=10)

        # Model stats
        ctk.CTkLabel(sb, text="Model Stats", font=("SF Pro Text", 9, "bold"),
                     text_color=C["muted"]).pack(anchor="w", **p)

        self._kpi_acc   = KPI(sb, "Accuracy",  color=C["green"])
        self._kpi_acc.pack(fill="x", **p)
        self._kpi_rows  = KPI(sb, "Rows",      color=C["blue"])
        self._kpi_rows.pack(fill="x", **p)
        self._kpi_feats = KPI(sb, "Features",  color=C["orange"])
        self._kpi_feats.pack(fill="x", **p)

        ctk.CTkLabel(sb, text="v2.0  ·  FloodSense  ·  NCR",
                     font=("SF Pro Text", 8), text_color=C["border"]
                     ).pack(side="bottom", pady=8)

    # ── Main area ─────────────────────────────────────────────────────────────

    def _main_area(self, parent):
        main = ctk.CTkFrame(parent, fg_color=C["bg"])
        main.pack(fill="both", expand=True)

        # Header strip (mirrors reference "forecast" banner)
        self._header = ctk.CTkFrame(main, fg_color=C["panel"],
                                    corner_radius=0, height=120)
        self._header.pack(fill="x")
        self._header.pack_propagate(False)

        left = ctk.CTkFrame(self._header, fg_color="transparent")
        left.pack(side="left", padx=28, pady=14)

        ctk.CTkLabel(left, text="FLOOD PREDICTION  ·  METRO MANILA",
                     font=("SF Pro Text", 9, "bold"),
                     text_color=C["muted"]).pack(anchor="w")
        self._h_risk = ctk.CTkLabel(left, text="Load data & train model",
                                    font=("SF Pro Display", 30, "bold"),
                                    text_color=C["text"])
        self._h_risk.pack(anchor="w")
        self._h_sub = ctk.CTkLabel(left, text="",
                                   font=("SF Pro Text", 10),
                                   text_color=C["muted"])
        self._h_sub.pack(anchor="w")

        # Inline weather KPIs (right side of header)
        right = ctk.CTkFrame(self._header, fg_color="transparent")
        right.pack(side="right", padx=28)
        self._h_rain = self._inline_kpi(right, "24h Rain")
        self._h_temp = self._inline_kpi(right, "Avg Temp")
        self._h_hum  = self._inline_kpi(right, "Humidity")

        # City cards strip (like the 16:00–20:00 cards)
        strip = ctk.CTkFrame(main, fg_color=C["panel"],
                             corner_radius=0, height=100)
        strip.pack(fill="x")
        strip.pack_propagate(False)
        self._city_cards = {}
        for city in cities:
            f = ctk.CTkFrame(strip, fg_color="transparent")
            f.pack(side="left", expand=True, padx=10, pady=12)
            ctk.CTkLabel(f, text=city, font=("SF Pro Text", 9),
                         text_color=C["muted"]).pack()
            v = ctk.CTkLabel(f, text="—",
                             font=("SF Pro Display", 18, "bold"),
                             text_color=C["text"])
            v.pack()
            s = ctk.CTkLabel(f, text="not fetched",
                             font=("SF Pro Text", 8),
                             text_color=C["muted"])
            s.pack()
            self._city_cards[city] = (v, s)

        # Orange sparkline canvas
        self._spark = tk.Canvas(strip, height=14, bg=C["panel"],
                                highlightthickness=0)
        self._spark.pack(fill="x", padx=20, pady=(0, 6))

        # Tabs
        tabs = ctk.CTkTabview(
            main, fg_color=C["bg"],
            segmented_button_fg_color=C["panel"],
            segmented_button_selected_color=C["orange"],
            segmented_button_unselected_color=C["panel"],
            segmented_button_selected_hover_color="#c04c10",
            text_color=C["text"],
        )
        tabs.pack(fill="both", expand=True, padx=14, pady=6)
        for t in ["City Risk", "EDA Charts", "Forecast", "Data Table", "Logs"]:
            tabs.add(t)
        self._tabs = tabs

        self._tab_city_risk(tabs.tab("City Risk"))
        self._tab_eda(tabs.tab("EDA Charts"))
        self._tab_forecast(tabs.tab("Forecast"))
        self._tab_table(tabs.tab("Data Table"))
        self._tab_logs(tabs.tab("Logs"))

    def _inline_kpi(self, parent, label):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(side="left", padx=16)
        ctk.CTkLabel(f, text=label, font=("SF Pro Text", 9),
                     text_color=C["muted"]).pack()
        lbl = ctk.CTkLabel(f, text="—",
                           font=("SF Pro Display", 16, "bold"),
                           text_color=C["text"])
        lbl.pack()
        return lbl

    # =========================================================  Tabs  ========

    def _tab_city_risk(self, tab):
        """Grid of city risk cards."""
        scroll = ctk.CTkScrollableFrame(tab, fg_color=C["bg"])
        scroll.pack(fill="both", expand=True)
        self._city_risk_inner = scroll

        ctk.CTkLabel(scroll,
                     text="Click '🌧  Live City Risk' to fetch live predictions.",
                     font=("SF Pro Text", 11), text_color=C["muted"]
                     ).pack(pady=40)

    def _tab_eda(self, tab):
        self._eda_scroll = ctk.CTkScrollableFrame(tab, fg_color=C["bg"])
        self._eda_scroll.pack(fill="both", expand=True)
        ctk.CTkLabel(self._eda_scroll,
                     text="Run EDA from the sidebar to see charts here.",
                     font=("SF Pro Text", 11), text_color=C["muted"]
                     ).pack(pady=40)

    def _tab_forecast(self, tab):
        ctk.CTkLabel(tab, text="5-Step Ahead Forecast",
                     font=("SF Pro Display", 13, "bold"),
                     text_color=C["text"]).pack(anchor="w", padx=4, pady=(6, 2))
        self._forecast_frame = Card(tab)
        self._forecast_frame.pack(fill="both", expand=True, pady=4)
        ctk.CTkLabel(self._forecast_frame,
                     text="Train regressors first.",
                     font=("SF Pro Text", 11), text_color=C["muted"]
                     ).pack(pady=40)

    def _tab_table(self, tab):
        frame = Card(tab)
        frame.pack(fill="both", expand=True)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("F.Treeview",
                        background=C["card"], foreground=C["text"],
                        rowheight=24, fieldbackground=C["card"],
                        borderwidth=0, font=("SF Pro Text", 10))
        style.configure("F.Treeview.Heading",
                        background=C["panel"], foreground=C["muted"],
                        font=("SF Pro Text", 10, "bold"), relief="flat")
        style.map("F.Treeview",
                  background=[("selected", C["blue"])],
                  foreground=[("selected", "white")])
        style.configure(
            "Vertical.TScrollbar",
            background=C["card"],
            troughcolor=C["panel"],
            bordercolor=C["border"],
            arrowcolor=C["muted"],
            gripcount=0
        )

        style.map(
            "Vertical.TScrollbar",
            background=[("active", C["hover"]), ("!active", C["card"])]
        )

        style.configure(
            "Horizontal.TScrollbar",
            background=C["card"],
            troughcolor=C["panel"],
            bordercolor=C["border"],
            arrowcolor=C["muted"],
            gripcount=0
        )

        style.map(
            "Horizontal.TScrollbar",
            background=[("active", C["hover"]), ("!active", C["card"])]
        )

        self._tree = ttk.Treeview(frame, style="F.Treeview", show="headings")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview, style="Vertical.TScrollbar")
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self._tree.xview, style="Horizontal.TScrollbar")
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._tree.pack(fill="both", expand=True)

    def _tab_logs(self, tab):
        self._log_box = ctk.CTkTextbox(tab, font=("Menlo", 10),
                                       fg_color=C["card"],
                                       text_color=C["muted"],
                                       corner_radius=10)
        self._log_box.pack(fill="both", expand=True)
        self._log("FloodControl started.")
        self._log(f"API_KEY: {'✓ loaded' if os.getenv('API_KEY') else '✗ not set (Open-Meteo does not need one)'}")
        self._log(f"Cities: {', '.join(cities.keys())}")

    # =========================================================  Helpers  =====

    def _tick(self):
        self._clock.configure(
            text=datetime.now().strftime("%b %d, %Y  %I:%M:%S %p"))
        self.after(1000, self._tick)

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_box.insert("end", f"[{ts}]  {msg}\n")
        self._log_box.see("end")

    def _set_status(self, msg):
        self._status.configure(text=msg)

    def _draw_sparkline(self, values):
        c = self._spark
        c.update_idletasks()
        w = max(c.winfo_width(), 500)
        h = 10
        c.delete("all")
        if len(values) < 2:
            return
        xs = np.linspace(4, w - 4, len(values))
        mn, mx = min(values), max(values) + 0.001
        ys = [h - (v - mn) / (mx - mn) * (h - 2) - 1 for v in values]
        for i in range(len(xs) - 1):
            c.create_line(xs[i], ys[i], xs[i+1], ys[i+1],
                          fill=C["orange"], width=2)
        for x, y in zip(xs[::max(1, len(xs)//6)], ys[::max(1, len(ys)//6)]):
            c.create_oval(x-3, y-3, x+3, y+3,
                          fill=C["orange"], outline="")

    # =========================================================  Actions  =====

    # ── Load CSV ──────────────────────────────────────────────────────────────
    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Open flood dataset",
            filetypes=[("CSV files", "*.csv"), ("All", "*.*")]
        )
        if not path:
            return
        try:
            self.df = pd.read_csv(path)
            name = os.path.basename(path)
            self._file_lbl.configure(text=name)
            self._kpi_rows.set(str(len(self.df)))
            self._set_status(f"Loaded {name}  ({len(self.df)} rows)")
            self._log(f"Dataset: {path}  |  {len(self.df)} rows, cols: {list(self.df.columns)}")
            self._populate_table(self.df)
            # Sparkline from FloodOccurrence rolling mean
            if "FloodOccurrence" in self.df.columns:
                vals = self.df["FloodOccurrence"].rolling(30).mean().dropna().tolist()
                self.after(200, self._draw_sparkline, vals)
        except Exception as e:
            messagebox.showerror("Error loading CSV", str(e))
            self._log(f"CSV error: {e}")

    def _populate_table(self, df):
        self._tree["columns"] = list(df.columns)
        for col in df.columns:
            self._tree.heading(col, text=col)
            self._tree.column(col, width=110, minwidth=70)
        self._tree.delete(*self._tree.get_children())
        for _, row in df.head(500).iterrows():
            self._tree.insert("", "end", values=list(row))

    # ── EDA ───────────────────────────────────────────────────────────────────
    def _run_eda(self):
        if self.df is None:
            messagebox.showwarning("No data", "Load a CSV first.")
            return
        self._log("Running EDA…")
        self._set_status("Running EDA…")
        for w in self._eda_scroll.winfo_children():
            w.destroy()
        threading.Thread(target=self._eda_worker, daemon=True).start()

    def _eda_worker(self):
        try:
            df = self.df
            num_df = df.select_dtypes(include=[np.number])
            charts = []

            # 1. Flood occurrence count
            fig, ax = plt.subplots(figsize=(5, 3))
            if "FloodOccurrence" in df.columns:
                counts = df["FloodOccurrence"].value_counts()
                ax.bar(counts.index.astype(str), counts.values,
                       color=[C["green"], C["red"]])
                ax.set_title("Flood Occurrence Count")
            charts.append(("Occurrence", fig))

            # 2. Rainfall histogram
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            if "Rainfall_mm" in df.columns:
                ax2.hist(df["Rainfall_mm"].dropna(), bins=30,
                         color=C["blue"], alpha=0.85, edgecolor="none")
                ax2.set_title("Rainfall Distribution (mm)")
            charts.append(("Rainfall", fig2))

            # 3. Water level box
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            if "WaterLevel_m" in df.columns:
                bp = ax3.boxplot(df["WaterLevel_m"].dropna(),
                                 patch_artist=True,
                                 medianprops=dict(color=C["orange"], lw=2))
                bp["boxes"][0].set_facecolor(C["blue"])
                ax3.set_title("Water Level (m)")
            charts.append(("Water Level", fig3))

            # 4. Correlation heatmap
            if num_df.shape[1] >= 2:
                fig4, ax4 = plt.subplots(figsize=(5, 4))
                corr = num_df.corr()
                im = ax4.imshow(corr.values, cmap="coolwarm",
                                vmin=-1, vmax=1, aspect="auto")
                ax4.set_xticks(range(len(corr.columns)))
                ax4.set_yticks(range(len(corr.columns)))
                ax4.set_xticklabels(corr.columns, rotation=45,
                                    ha="right", fontsize=8)
                ax4.set_yticklabels(corr.columns, fontsize=8)
                for i in range(len(corr)):
                    for j in range(len(corr.columns)):
                        ax4.text(j, i, f"{corr.values[i,j]:.1f}",
                                 ha="center", va="center",
                                 fontsize=7, color="white")
                ax4.set_title("Correlation Heatmap")
                fig4.colorbar(im, ax=ax4)
                charts.append(("Correlation", fig4))

            self.after(0, self._render_eda, charts)
            self.after(0, self._log, "EDA complete.")
            self.after(0, self._set_status, "EDA complete")
        except Exception as e:
            self.after(0, self._log, f"EDA error: {e}")
            self.after(0, self._set_status, "EDA error")

    def _render_eda(self, charts):
        for w in self._eda_scroll.winfo_children():
            w.destroy()
        row = None
        for i, (title, fig) in enumerate(charts):
            if i % 2 == 0:
                row = ctk.CTkFrame(self._eda_scroll, fg_color="transparent")
                row.pack(fill="x", pady=4)
            card = Card(row)
            card.pack(side="left", expand=True, fill="both", padx=6)
            ctk.CTkLabel(card, text=title, font=("SF Pro Text", 11, "bold"),
                         text_color=C["text"]).pack(anchor="w", padx=10, pady=(8, 0))
            fig.tight_layout()
            cv = FigureCanvasTkAgg(fig, master=card)
            cv.draw()
            cv.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self._canvas[f"eda_{i}"] = cv
        self._tabs.set("EDA Charts")

    # ── Train classifier ──────────────────────────────────────────────────────
    def _train_classifier(self):
        if self.df is None:
            messagebox.showwarning("No data", "Load a CSV first.")
            return
        self._log("Training classifier…")
        self._set_status("Training…")
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self):
        try:
            X, y, le = prepare_data(self.df.copy())
            self.le = le
            model, acc = train_model(X, y)
            self.model = model
            self.accuracy = acc
            importances = model.feature_importances_
            self.after(0, self._apply_train, model, acc,
                       importances, list(X.columns))
        except Exception as e:
            self.after(0, self._log, f"Train error: {e}")
            self.after(0, self._set_status, "Training failed")

    def _apply_train(self, model, acc, importances, feat_names):
        self._kpi_acc.set(f"{acc*100:.1f}%",
                          color=C["green"] if acc >= 0.8 else C["yellow"])
        self._kpi_feats.set(str(len(feat_names)))
        self._log(f"Classifier trained. Accuracy: {acc*100:.1f}%")
        self._set_status(f"Classifier ready  ·  accuracy {acc*100:.1f}%")
        self._h_risk.configure(text="Model Trained ✓", text_color=C["green"])
        self._h_sub.configure(
            text=f"Accuracy {acc*100:.1f}%  ·  {datetime.now().strftime('%B %d, %Y')}")

    # ── Train regressors ──────────────────────────────────────────────────────
    def _train_regressors(self):
        if self.df is None:
            messagebox.showwarning("No data", "Load a CSV first.")
            return
        self._log("Training regressors…")
        self._set_status("Training regressors…")
        threading.Thread(target=self._reg_worker, daemon=True).start()

    def _reg_worker(self):
        features = ["Rainfall_mm", "WaterLevel_m", "SoilMoisture_pct"]
        reg = {}
        try:
            for feat in features:
                if feat not in self.df.columns:
                    continue
                Xr, yr = regression_data(self.df, feat)
                if len(Xr) < 5:
                    continue
                reg[feat] = train_regression_model(Xr, yr)
                self.after(0, self._log, f"Regressor trained: {feat}")
            self.reg_models = reg
            self.after(0, self._render_forecast)
            self.after(0, self._set_status, "Regressors ready")
        except Exception as e:
            self.after(0, self._log, f"Regressor error: {e}")

    def _render_forecast(self):
        for w in self._forecast_frame.winfo_children():
            w.destroy()
        if not self.reg_models:
            ctk.CTkLabel(self._forecast_frame, text="No regressors available.",
                         font=("SF Pro Text", 11),
                         text_color=C["muted"]).pack(pady=20)
            return

        feats = list(self.reg_models.keys())
        fig, axes = plt.subplots(1, len(feats), figsize=(5 * len(feats), 3.5))
        if len(feats) == 1:
            axes = [axes]

        for ax, feat in zip(axes, feats):
            m  = self.reg_models[feat]
            lv = float(self.df[feat].iloc[-1])
            preds = predict_future(m, lv)
            steps = list(range(1, len(preds) + 1))
            ax.plot(steps, preds, marker="o", color=C["orange"], lw=2.5)
            ax.fill_between(steps, preds, alpha=0.12, color=C["orange"])
            ax.axhline(lv, ls="--", color=C["muted"], lw=1, alpha=0.6)
            ax.set_title(feat, pad=6)
            ax.set_xlabel("Steps ahead")
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=self._forecast_frame)
        cv.draw()
        cv.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)
        self._canvas["forecast"] = cv
        self._tabs.set("Forecast")
        self._log("Forecast rendered.")

    # ── Live city risk ────────────────────────────────────────────────────────
    def _live_city_risk(self):
        if self.model is None or self.le is None:
            messagebox.showwarning("No model",
                                   "Train the classifier first.")
            return
        self._log("Fetching live weather for all cities…")
        self._set_status("Fetching live data…")
        threading.Thread(target=self._city_risk_worker, daemon=True).start()

    def _city_risk_worker(self):
        results = {}
        for city, (lat, lon) in cities.items():
            try:
                pred, prob, rain, temp, hum = predict_flood_risk(
                    self.model, self.le, city, lat, lon
                )
                results[city] = {
                    "pred": int(pred), "prob": float(prob),
                    "rain": rain, "temp": temp, "hum": hum,
                }
                self.after(0, self._log,
                           f"{city}: {'FLOOD' if pred else 'OK'}  "
                           f"prob={prob*100:.1f}%  rain={rain:.1f}mm")
            except Exception as e:
                results[city] = {"error": str(e)}
                self.after(0, self._log, f"{city}: error — {e}")

        self.after(0, self._render_city_risk, results)
        self.after(0, self._set_status, "Live city risk loaded")

    def _render_city_risk(self, results):
        # Update city strip cards
        for city, (v_lbl, s_lbl) in self._city_cards.items():
            r = results.get(city, {})
            if "error" in r:
                v_lbl.configure(text="ERR", text_color=C["muted"])
                s_lbl.configure(text=r["error"][:20])
            else:
                prob = r["prob"]
                rc   = risk_colour(prob)
                v_lbl.configure(text=f"{prob*100:.0f}%", text_color=rc)
                s_lbl.configure(text=risk_label(prob), text_color=rc)

        # Update header
        highest = max(
            (r for r in results.values() if "prob" in r),
            key=lambda r: r["prob"], default=None
        )
        if highest:
            rc = risk_colour(highest["prob"])
            self._h_risk.configure(text=risk_label(highest["prob"]),
                                   text_color=rc)
            self._h_sub.configure(
                text=f"Highest city probability: {highest['prob']*100:.1f}%  "
                     f"·  {datetime.now().strftime('%b %d  %I:%M %p')}"
            )
            self._h_rain.configure(text=f"{highest['rain']:.1f} mm")
            self._h_temp.configure(text=f"{highest['temp']:.1f} °C")
            self._h_hum.configure(text=f"{highest['hum']:.0f} %")

        # Render city risk panel
        for w in self._city_risk_inner.winfo_children():
            w.destroy()

        grid_row = None
        for i, (city, r) in enumerate(results.items()):
            if i % 2 == 0:
                grid_row = ctk.CTkFrame(self._city_risk_inner,
                                        fg_color="transparent")
                grid_row.pack(fill="x", pady=6)

            card = Card(grid_row)
            card.pack(side="left", expand=True, fill="both", padx=6)

            if "error" in r:
                ctk.CTkLabel(card, text=city, font=("SF Pro Display", 14, "bold"),
                             text_color=C["text"]).pack(anchor="w", padx=14, pady=(14, 2))
                ctk.CTkLabel(card, text=f"Error: {r['error']}",
                             font=("SF Pro Text", 10),
                             text_color=C["red"]).pack(anchor="w", padx=14, pady=(0, 14))
                continue

            prob = r["prob"]
            rc   = risk_colour(prob)
            rl   = risk_label(prob)

            # City name + risk badge
            top_row = ctk.CTkFrame(card, fg_color="transparent")
            top_row.pack(fill="x", padx=14, pady=(14, 2))

            ctk.CTkLabel(top_row, text=city,
                         font=("SF Pro Display", 15, "bold"),
                         text_color=C["text"]).pack(side="left")

            badge = ctk.CTkFrame(top_row,
                                 fg_color=rc,
                                 corner_radius=6)
            badge.pack(side="right")
            ctk.CTkLabel(badge, text=rl,
                         font=("SF Pro Text", 9, "bold"),
                         text_color="white").pack(padx=8, pady=2)

            # Probability bar
            bar_bg = ctk.CTkFrame(card, fg_color=C["hover"],
                                  corner_radius=4, height=8)
            bar_bg.pack(fill="x", padx=14, pady=4)
            bar_bg.pack_propagate(False)
            bar_fill = ctk.CTkFrame(bar_bg, fg_color=rc,
                                    corner_radius=4, height=8)
            # approximate width via pack — use place after render
            bar_fill.place(relx=0, rely=0, relwidth=prob, relheight=1)

            # Stats row
            stats = ctk.CTkFrame(card, fg_color="transparent")
            stats.pack(fill="x", padx=14, pady=(2, 14))
            for label, val in [
                ("Probability", f"{prob*100:.1f}%"),
                ("24h Rain",    f"{r['rain']:.1f} mm"),
                ("Temp",        f"{r['temp']:.1f} °C"),
                ("Humidity",    f"{r['hum']:.0f} %"),
            ]:
                col = ctk.CTkFrame(stats, fg_color="transparent")
                col.pack(side="left", expand=True)
                ctk.CTkLabel(col, text=label, font=("SF Pro Text", 8),
                             text_color=C["muted"]).pack()
                ctk.CTkLabel(col, text=val,
                             font=("SF Pro Display", 13, "bold"),
                             text_color=rc if label == "Probability" else C["text"]
                             ).pack()

        self._tabs.set("City Risk")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = FloodDashboard()
    app.mainloop()
