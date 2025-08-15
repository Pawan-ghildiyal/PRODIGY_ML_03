# PRODIGY_ML_03 — Decision Tree Classifier with an embedded Tkinter UI
# Dataset: Social_Network_Ads.csv  (features: Age, EstimatedSalary; label: Purchased)

import os
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CSV_PATH = r"C:\Users\pawan\OneDrive\Desktop\PRODIGY_ML_03\data\Social_Network_Ads.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.25
CRITERION = "entropy"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at:\n{CSV_PATH}")

data = pd.read_csv(CSV_PATH, encoding="utf-8", skip_blank_lines=True)
required_cols = {"Age", "EstimatedSalary", "Purchased"}
if not required_cols.issubset(data.columns):
    raise ValueError(f"CSV must contain columns: {', '.join(sorted(required_cols))}")

X = data[["Age", "EstimatedSalary"]]
y = data["Purchased"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

clf = DecisionTreeClassifier(criterion=CRITERION, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
ACC = accuracy_score(y_test, y_pred)
CONF = confusion_matrix(y_test, y_pred)
CLF_REPORT = classification_report(y_test, y_pred)

class PlotArea(ttk.Frame):
    """Reusable frame that holds a single Matplotlib FigureCanvas."""
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = None
        self.fig = None

    def clear(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        self.fig = None

    def new_canvas(self, width=7.5, height=5.5, dpi=100):
        """Create a new Figure and attach a TkAgg canvas BEFORE plotting."""
        self.clear()
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        # pack now so plot functions that query renderer are safe
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        return self.fig, self.canvas

    def draw(self):
        if self.canvas:
            self.canvas.draw()


def render_tree_into(area: PlotArea):
    """Builds a tree figure and renders it inside the given PlotArea."""
    fig, _ = area.new_canvas()
    ax = fig.add_subplot(111)
    plot_tree(
        clf,
        feature_names=["Age", "EstimatedSalary"],
        class_names=["Not Purchased", "Purchased"],
        filled=True,
        ax=ax
    )
    ax.set_title("Decision Tree Classifier", pad=12)
    fig.tight_layout()
    area.draw()


def render_boundary_into(area: PlotArea):
    """Builds a decision boundary figure and renders it inside the given PlotArea."""
    X_set, y_set = X_train.values, y_train.values
    x1_min, x1_max = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
    x2_min, x2_max = X_set[:, 1].min() - 1000, X_set[:, 1].max() + 1000

    xx1, xx2 = np.meshgrid(
        np.arange(start=x1_min, stop=x1_max, step=1),
        np.arange(start=x2_min, stop=x2_max, step=250)
    )

    grid = pd.DataFrame(np.c_[xx1.ravel(), xx2.ravel()],
                        columns=["Age", "EstimatedSalary"])
    Z = clf.predict(grid).reshape(xx1.shape)

    fig, _ = area.new_canvas()
    ax = fig.add_subplot(111)

    ax.contourf(xx1, xx2, Z, alpha=0.5, cmap=ListedColormap(("red", "green")))
    ax.scatter(
        X_set[:, 0], X_set[:, 1],
        c=y_set, cmap=ListedColormap(("red", "green")),
        edgecolors="black", linewidths=0.5
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Estimated Salary")
    ax.set_title("Decision Boundary (Training Set)")
    fig.tight_layout()
    area.draw()


def predict_from_inputs():
    try:
        age = int(age_var.get().strip())
        salary = int(salary_var.get().strip())
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter numeric Age and Estimated Salary.")
        return

    sample = pd.DataFrame([[age, salary]], columns=["Age", "EstimatedSalary"])
    pred = clf.predict(sample)[0]
    try:
        prob = clf.predict_proba(sample)[0, pred]
    except Exception:
        prob = None

    if pred == 1:
        text = "Prediction: Purchased ✅"
        color = "#188038"
    else:
        text = "Prediction: Not Purchased ❌"
        color = "#B00020"

    if prob is not None:
        text += f"  (confidence ~ {prob:.2f})"
    result_lbl.config(text=text, foreground=color)


root = tk.Tk()
root.title("PRODIGY_ML_03 — Decision Tree Classifier (Embedded UI)")
root.geometry("900x650")

style = ttk.Style(root)
try:
    style.theme_use("clam")
except Exception:
    pass

ttk.Label(root, text="Decision Tree on Social_Network_Ads",
          font=("Segoe UI", 16, "bold")).pack(pady=8)

metrics = ttk.Frame(root)
metrics.pack(fill="x", padx=12, pady=6)

ttk.Label(metrics, text=f"Accuracy: {ACC:.2f}", font=("Segoe UI", 12)).grid(row=0, column=0, sticky="w", padx=(0, 12))
ttk.Label(metrics, text="Confusion Matrix:", font=("Segoe UI", 12, "bold")).grid(row=1, column=0, sticky="nw")
conf_box = ScrolledText(metrics, width=28, height=5, font=("Consolas", 10))
conf_box.grid(row=1, column=1, sticky="w")
conf_box.insert("1.0", str(CONF))
conf_box.configure(state="disabled")

ttk.Label(metrics, text="Classification Report:", font=("Segoe UI", 12, "bold")).grid(row=0, column=2, sticky="nw", padx=(24, 0))
report_box = ScrolledText(metrics, width=60, height=7, font=("Consolas", 10))
report_box.grid(row=1, column=2, sticky="w")
report_box.insert("1.0", CLF_REPORT)
report_box.configure(state="disabled")

for i in range(3):
    metrics.grid_columnconfigure(i, weight=1)

nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True, padx=12, pady=8)

# tab: Predict
predict_tab = ttk.Frame(nb)
nb.add(predict_tab, text="Predict")

form = ttk.Frame(predict_tab)
form.pack(pady=16)

age_var = tk.StringVar()
salary_var = tk.StringVar()

ttk.Label(form, text="Age").grid(row=0, column=0, sticky="e", padx=6, pady=6)
ttk.Entry(form, textvariable=age_var, width=18).grid(row=0, column=1, padx=6, pady=6)

ttk.Label(form, text="Estimated Salary").grid(row=1, column=0, sticky="e", padx=6, pady=6)
ttk.Entry(form, textvariable=salary_var, width=18).grid(row=1, column=1, padx=6, pady=6)

ttk.Button(form, text="Predict", command=predict_from_inputs).grid(row=2, column=0, columnspan=2, pady=10)

result_lbl = ttk.Label(predict_tab, text="", font=("Segoe UI", 12, "bold"))
result_lbl.pack(pady=6)

tree_tab = ttk.Frame(nb)
nb.add(tree_tab, text="Decision Tree")
tree_area = PlotArea(tree_tab)
tree_area.pack(fill="both", expand=True, padx=8, pady=(8, 0))
ttk.Button(tree_tab, text="Render / Refresh Tree", command=lambda: render_tree_into(tree_area)).pack(pady=8)

boundary_tab = ttk.Frame(nb)
nb.add(boundary_tab, text="Decision Boundary")
boundary_area = PlotArea(boundary_tab)
boundary_area.pack(fill="both", expand=True, padx=8, pady=(8, 0))
ttk.Button(boundary_tab, text="Render / Refresh Boundary", command=lambda: render_boundary_into(boundary_area)).pack(pady=8)

render_tree_into(tree_area)

root.mainloop()
