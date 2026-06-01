"""
Risk Scorecard Model v3 — Streamlit App
========================================
Upload CSV → Get scored output, risk band analysis, charts, and downloadable reports.
"""

import io
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Scorecard Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .stMetric label { font-size: 13px !important; color: #6c757d !important; }
    .band-card { border-radius: 12px; padding: 16px 20px; margin: 6px 0; color: white; font-weight: 600; }
    .lr-card  { background: linear-gradient(135deg, #27ae60, #2ecc71); }
    .mr-card  { background: linear-gradient(135deg, #f39c12, #f1c40f); color: #333; }
    .hr-card  { background: linear-gradient(135deg, #e67e22, #e74c3c); }
    .vhr-card { background: linear-gradient(135deg, #c0392b, #922b21); }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    .stDownloadButton > button { background: #2c3e50; color: white; border-radius: 8px; border: none; padding: 8px 20px; }
    .stDownloadButton > button:hover { background: #1a252f; }
    div[data-testid="stSidebar"] { background: #2c3e50; color: white; }
    div[data-testid="stSidebar"] .stMarkdown p { color: #ecf0f1; }
    div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 { color: white; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
SCORE_CARD = {
    ('Name_Validation', 'Yes_Correct Customer'):  20,
    ('Name_Validation', 'No_Incorrect Customer'):  5,
    ('Connected',       'connected'):             10,
    ('Connected',       'Not connected'):          3,
    ('PAN',             'Yes'):                    7,
    ('UID',             'Yes'):                    7,
    ('VI',              'Yes'):                    6,
    ('DL',              'Yes'):                    5,
    ('Mobile_No',       'Yes'):                   10,
    ('CIBIL',           'Yes'):                   20,
    ('Document',        'Documents Collected'):   15,
    ('Document',        'Documents Pending'):      2,
}

PALETTE = {
    'Low Risk':       '#27ae60',
    'Medium Risk':    '#f39c12',
    'High Risk':      '#e67e22',
    'Very High Risk': '#c0392b'
}
BAND_ORDER = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']

COLUMN_MAP = {
    'Document Collection Status (Yes/No)19-11-2025': 'Document',
    'Mobile No. Status (Yes/No)_27-01-2025':         'Mobile_No',
    'Pan Card Status':                                'PAN',
    'Voter ID Number Status':                         'VI',
    'Universal ID Number Status':                     'UID',
    'DL_ Status':                                     'DL',
    'Name Validation Final Remark':                   'Name_Validation',
    'Contactable/Non-Contactable':                    'Connected',
    'CIBIL Data':                                     'CIBIL',
    'Customer ID':                                    'Customer_ID',
    'Bank Lan':                                       'Bank_Lan',
}

FEATURES = ['Name_Validation', 'Connected', 'PAN', 'UID', 'VI', 'DL',
            'Mobile_No', 'CIBIL', 'Document']

# ─────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
def load_and_clean(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1')
    except Exception:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    df.rename(columns=COLUMN_MAP, inplace=True)
    str_cols = df.select_dtypes('object').columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())
    return df


def build_target(row):
    signals = [
        row.get('CIBIL')          == 'No',
        row.get('Connected')      == 'Not connected',
        row.get('Document')       == 'Documents Pending',
        row.get('Name_Validation')== 'No_Incorrect Customer',
        row.get('Mobile_No')      == 'No',
    ]
    return int(sum(signals) >= 3)


def rule_score(row):
    return sum(pts for (col, val), pts in SCORE_CARD.items()
               if row.get(col) == val)


def risk_band(s):
    if s >= 70:   return 'Low Risk'
    elif s >= 45: return 'Medium Risk'
    elif s >= 25: return 'High Risk'
    else:         return 'Very High Risk'


def compute_woe(df, col, target='Target'):
    total_bad  = max(df[target].sum(), 1)
    total_good = max(len(df) - total_bad, 1)
    rows = []
    for val, grp in df.groupby(col):
        bad  = grp[target].sum()
        good = len(grp) - bad
        dist_bad  = bad  / total_bad
        dist_good = good / total_good
        woe = np.log((dist_bad + 1e-9) / (dist_good + 1e-9))
        iv  = (dist_bad - dist_good) * woe
        rows.append({'Feature': col, 'Category': val,
                     'Count': len(grp), 'Bad': bad, 'Good': good,
                     'Bad_Rate': bad / max(len(grp), 1), 'WoE': woe, 'IV': iv})
    return pd.DataFrame(rows)


def run_logistic(df):
    present = [f for f in FEATURES if f in df.columns]
    for col in present:
        woe_map = compute_woe(df, col).set_index('Category')['WoE'].to_dict()
        df[f'WoE_{col}'] = df[col].map(woe_map).fillna(0)

    woe_feats = [f'WoE_{c}' for c in present]
    X = df[woe_feats]
    y = df['Target']

    if y.nunique() < 2 or len(df) < 20:
        return None, None, None, None, None, None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    lr = LogisticRegression(max_iter=500, C=0.5, random_state=42,
                            class_weight='balanced', solver='lbfgs')
    lr.fit(X_tr, y_tr)

    y_pred = lr.predict(X_te)
    y_prob = lr.predict_proba(X_te)[:, 1]

    auc  = roc_auc_score(y_te, y_prob)
    gini = 2 * auc - 1
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    ks = (tpr - fpr).max()
    return auc, gini, ks, fpr, tpr, classification_report(
        y_te, y_pred, target_names=['Good', 'Bad'], output_dict=True)


@st.cache_data(show_spinner=False)
def process_data(file_bytes, file_name):
    df = load_and_clean(io.BytesIO(file_bytes))
    df['Target']    = df.apply(build_target, axis=1)
    df['Rule_Score']= df.apply(rule_score, axis=1)
    df['Risk_Band'] = df['Rule_Score'].apply(risk_band)

    woe_tables  = pd.concat([compute_woe(df, c) for c in FEATURES if c in df.columns], ignore_index=True)
    iv_summary  = woe_tables.groupby('Feature')['IV'].sum().sort_values(ascending=False)

    band_perf = []
    for band in BAND_ORDER:
        subset = df[df['Risk_Band'] == band]
        if len(subset) == 0:
            continue
        band_perf.append({
            'Risk_Band':    band,
            'Count':        len(subset),
            'Pct':          len(subset) / len(df) * 100,
            'Default_Rate': subset['Target'].mean() * 100,
            'Avg_Score':    subset['Rule_Score'].mean(),
            'Min_Score':    subset['Rule_Score'].min(),
            'Max_Score':    subset['Rule_Score'].max(),
        })
    band_perf_df = pd.DataFrame(band_perf)

    auc, gini, ks, fpr, tpr, clf_report = run_logistic(df.copy())

    return df, woe_tables, iv_summary, band_perf_df, auc, gini, ks, fpr, tpr, clf_report


# ─────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    return buf.getvalue()


def plot_pie(df):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    vc = df['Risk_Band'].value_counts().reindex(BAND_ORDER).fillna(0)
    colors = [PALETTE[b] for b in BAND_ORDER]
    wedges, texts, autotexts = ax.pie(
        vc.values, labels=None, autopct='%1.1f%%',
        colors=colors, startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2))
    for at in autotexts:
        at.set_fontsize(10); at.set_fontweight('bold'); at.set_color('white')
    ax.legend(BAND_ORDER, loc='lower center', bbox_to_anchor=(0.5, -0.12),
              ncol=2, fontsize=9, frameon=False)
    ax.set_title('Risk Band Distribution', fontweight='bold', fontsize=12, pad=12)
    fig.tight_layout()
    return fig


def plot_score_hist(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    for band in BAND_ORDER:
        grp = df[df['Risk_Band'] == band]
        ax.hist(grp['Rule_Score'], bins=25, alpha=0.75,
                color=PALETTE[band], label=band, edgecolor='white', linewidth=0.5)
    for thresh, label, color in [(25,'VHR|HR','#e74c3c'), (45,'HR|MR','#e67e22'), (70,'MR|LR','#27ae60')]:
        ax.axvline(thresh, color=color, lw=2, ls='--')
        ax.text(thresh + 0.5, ax.get_ylim()[1] * 0.92, label,
                fontsize=8, color=color, fontweight='bold')
    ax.set_xlabel('Rule Score', fontsize=11); ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Score Distribution by Risk Band', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_default_rate(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    dr = df.groupby('Risk_Band')['Target'].mean() * 100
    dr = dr.reindex(BAND_ORDER).fillna(0)
    bars = ax.bar(range(len(BAND_ORDER)), dr.values,
                  color=[PALETTE[b] for b in BAND_ORDER],
                  edgecolor='white', width=0.6, linewidth=0.5)
    ax.set_xticks(range(len(BAND_ORDER)))
    ax.set_xticklabels(['LR', 'MR', 'HR', 'VHR'], fontsize=10)
    ax.set_ylabel('Default Rate (%)', fontsize=11)
    ax.set_title('Default Rate by Risk Band', fontweight='bold', fontsize=12)
    for bar, val in zip(bars, dr.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_iv(iv_summary):
    fig, ax = plt.subplots(figsize=(5, 4))
    iv_plot = iv_summary.sort_values()
    colors_iv = ['#27ae60' if v >= 0.1 else '#e67e22' if v >= 0.02 else '#e74c3c'
                 for v in iv_plot.values]
    ax.barh(iv_plot.index, iv_plot.values, color=colors_iv, edgecolor='white', alpha=0.9)
    ax.axvline(0.1, color='green', ls='--', lw=1.2, alpha=0.8, label='Medium (0.1)')
    ax.axvline(0.3, color='blue',  ls='--', lw=1.2, alpha=0.8, label='Strong (0.3)')
    ax.set_xlabel('Information Value (IV)', fontsize=11)
    ax.set_title('Feature IV Ranking', fontweight='bold', fontsize=12)
    ax.legend(fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_roc(fpr, tpr, auc, gini, ks):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='#2980b9', lw=2.5,
            label=f'AUC={auc:.3f}  |  Gini={gini:.3f}')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#2980b9')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve  (KS = {ks:.3f})', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_score_decile_default(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = pd.cut(df['Rule_Score'],
                  bins=[-1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                  labels=['0-10','11-20','21-30','31-40','41-50',
                          '51-60','61-70','71-80','81-90','91-100'])
    ks_tbl = df.groupby(bins, observed=True)['Target'].mean() * 100
    colors = ['#27ae60' if i >= 7 else '#f39c12' if i >= 5 else '#e67e22' if i >= 3 else '#c0392b'
              for i in range(len(ks_tbl))]
    ax.bar(range(len(ks_tbl)), ks_tbl.values, color=colors, edgecolor='white', alpha=0.85)
    ax.set_xticks(range(len(ks_tbl)))
    ax.set_xticklabels(ks_tbl.index, rotation=40, fontsize=8)
    ax.set_ylabel('Default Rate (%)', fontsize=11)
    ax.set_title('Default Rate by Score Bucket', fontweight='bold', fontsize=12)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_boxplot(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    data_box = [df[df['Risk_Band'] == b]['Rule_Score'].values for b in BAND_ORDER]
    bp = ax.boxplot(data_box, patch_artist=True, labels=['LR', 'MR', 'HR', 'VHR'],
                    medianprops=dict(color='black', linewidth=2))
    for patch, band in zip(bp['boxes'], BAND_ORDER):
        patch.set_facecolor(PALETTE[band])
        patch.set_alpha(0.85)
    ax.set_ylabel('Risk Score', fontsize=11)
    ax.set_title('Score Distribution by Band', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def build_full_dashboard(df, iv_summary, auc, gini, ks, fpr, tpr):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle('Risk Scorecard Analyser — Full Dashboard',
                 fontsize=20, fontweight='bold', y=0.998)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.35)

    # Row 0
    ax1 = fig.add_subplot(gs[0, 0])
    vc = df['Risk_Band'].value_counts().reindex(BAND_ORDER).fillna(0)
    colors = [PALETTE[b] for b in BAND_ORDER]
    ax1.pie(vc.values, labels=None, autopct='%1.1f%%', colors=colors,
            startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
    ax1.legend(BAND_ORDER, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7)
    ax1.set_title('Band Distribution', fontweight='bold', fontsize=11)

    ax2 = fig.add_subplot(gs[0, 1:3])
    for band in BAND_ORDER:
        grp = df[df['Risk_Band'] == band]
        ax2.hist(grp['Rule_Score'], bins=25, alpha=0.75,
                 color=PALETTE[band], label=band, edgecolor='white')
    for thresh, color in [(25, '#e74c3c'), (45, '#e67e22'), (70, '#27ae60')]:
        ax2.axvline(thresh, color=color, lw=2, ls='--')
    ax2.set_xlabel('Score'); ax2.set_ylabel('Count')
    ax2.set_title('Score Distribution by Risk Band', fontweight='bold', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    ax3 = fig.add_subplot(gs[0, 3])
    dr = df.groupby('Risk_Band')['Target'].mean() * 100
    dr = dr.reindex(BAND_ORDER).fillna(0)
    bars = ax3.bar(range(len(BAND_ORDER)), dr.values,
                   color=colors, edgecolor='white', width=0.6)
    ax3.set_xticks(range(len(BAND_ORDER)))
    ax3.set_xticklabels(['LR', 'MR', 'HR', 'VHR'], fontsize=9)
    ax3.set_ylabel('Default Rate (%)')
    ax3.set_title('Default Rate by Band', fontweight='bold', fontsize=11)
    for bar, val in zip(bars, dr.values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)

    # Row 1
    ax4 = fig.add_subplot(gs[1, 0])
    data_box = [df[df['Risk_Band'] == b]['Rule_Score'].values for b in BAND_ORDER]
    bp = ax4.boxplot(data_box, patch_artist=True, labels=['LR', 'MR', 'HR', 'VHR'],
                     medianprops=dict(color='black', linewidth=2))
    for patch, band in zip(bp['boxes'], BAND_ORDER):
        patch.set_facecolor(PALETTE[band])
    ax4.set_ylabel('Score'); ax4.set_title('Score by Band (Box)', fontweight='bold', fontsize=11)

    ax5 = fig.add_subplot(gs[1, 1])
    iv_plot = iv_summary.sort_values()
    colors_iv = ['#27ae60' if v >= 0.1 else '#e67e22' if v >= 0.02 else '#e74c3c'
                 for v in iv_plot.values]
    ax5.barh(iv_plot.index, iv_plot.values, color=colors_iv, alpha=0.85)
    ax5.axvline(0.1, color='green', ls='--', lw=1)
    ax5.set_xlabel('IV'); ax5.set_title('Feature IV Ranking', fontweight='bold', fontsize=11)

    ax6 = fig.add_subplot(gs[1, 2])
    band_counts = df['Risk_Band'].value_counts().reindex(BAND_ORDER).fillna(0)
    bars = ax6.barh(BAND_ORDER, band_counts.values, color=colors, alpha=0.85)
    ax6.set_xlabel('Customers')
    ax6.set_title('Portfolio Size by Band', fontweight='bold', fontsize=11)
    ax6.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    for bar, count in zip(bars, band_counts.values):
        pct = count / max(len(df), 1) * 100
        ax6.text(count + 100, bar.get_y() + bar.get_height() / 2,
                 f'{pct:.1f}%', va='center', fontsize=9, fontweight='bold')

    ax7 = fig.add_subplot(gs[1, 3])
    if fpr is not None:
        ax7.plot(fpr, tpr, color='#2980b9', lw=2.5,
                 label=f'AUC={auc:.3f}, Gini={gini:.3f}')
        ax7.fill_between(fpr, tpr, alpha=0.08, color='#2980b9')
        ax7.plot([0, 1], [0, 1], 'k--', lw=1.2)
        ax7.legend(fontsize=9)
    ax7.set_xlabel('FPR'); ax7.set_ylabel('TPR')
    ax7.set_title(f'ROC Curve (KS={ks:.3f})' if ks else 'ROC Curve',
                  fontweight='bold', fontsize=11)

    # Row 2 — Decision Matrix
    ax8 = fig.add_subplot(gs[2, :])
    ax8.axis('off')
    headers = ['Risk Band', 'Score Range', 'Count', 'Default Rate', 'Decision', 'Action']
    actions = {
        'Low Risk':       ('Score 70–100', '✅ APPROVE',  'Fast-track, minimal documentation'),
        'Medium Risk':    ('Score 45–69',  '🔍 REVIEW',   'Standard underwriting process'),
        'High Risk':      ('Score 25–44',  '⛔ DECLINE',  'Defer — require guarantor'),
        'Very High Risk': ('Score 0–24',   '⛔ DECLINE',  'Possible fraud — escalate'),
    }
    rows = []
    for band in BAND_ORDER:
        subset = df[df['Risk_Band'] == band]
        sr, decision, action = actions[band]
        rows.append([band, sr, f"{len(subset):,}",
                     f"{subset['Target'].mean() * 100:.1f}%" if len(subset) else "—",
                     decision, action])

    tbl = ax8.table(cellText=rows, colLabels=headers,
                    loc='center', cellLoc='left',
                    colWidths=[0.13, 0.12, 0.12, 0.13, 0.14, 0.26])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.3)
    row_colors = ['#d5f4e6', '#fef5e7', '#fdebd0', '#fadbd8']
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
        elif r <= 4:
            cell.set_facecolor(row_colors[r - 1])
    ax8.set_title('Decision Matrix', fontweight='bold', fontsize=13, pad=20)

    fig.tight_layout(rect=[0, 0, 1, 0.995])
    return fig


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Risk Scorecard\n**Analyser v3**")
    st.markdown("---")
    st.markdown("### 🎯 Score Bands")
    st.markdown("""
| Band | Score | Action |
|------|-------|--------|
| 🟢 Low Risk | 70–100 | Approve |
| 🟡 Medium | 45–69 | Review |
| 🟠 High | 25–44 | Decline |
| 🔴 Very High | 0–24 | Decline |
    """)
    st.markdown("---")
    st.markdown("### 📋 Required Columns")
    st.markdown("""
- Bank Lan
- Customer ID
- Document Collection Status
- Mobile No. Status
- Pan Card Status
- Voter ID Number Status
- Universal ID Number Status
- DL_ Status
- Name Validation Final Remark
- Contactable/Non-Contactable
- CIBIL Data
    """)
    st.markdown("---")
    st.caption("Built by Ansh Kambli · Reliance ARC")

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
st.title("📊 Risk Scorecard Analyser")
st.markdown("Upload your borrower portfolio CSV to get instant risk scoring, band classification, and model validation.")

uploaded = st.file_uploader("Upload Portfolio CSV", type=['csv'], label_visibility='collapsed')

if uploaded is None:
    st.info("👆 Upload a CSV file to begin analysis.", icon="📁")

    with st.expander("📝 Sample Data Format", expanded=False):
        sample = pd.DataFrame({
            'Bank Lan': ['30899513288', '485462032'],
            'Customer ID': ['30899513288', '485462032'],
            'Document Collection Status (Yes/No)19-11-2025': ['Documents Collected', 'Documents Pending'],
            'Mobile No. Status (Yes/No)_27-01-2025': ['Yes', 'No'],
            'Pan Card Status': ['No', 'No'],
            'Voter ID Number Status': ['Yes', 'No'],
            'Universal ID Number Status': ['No', 'No'],
            'DL_ Status': ['No', 'No'],
            'Name Validation Final Remark': ['Yes_Correct Customer', 'No_Incorrect Customer'],
            'Contactable/Non-Contactable': ['connected', 'Not connected'],
            'CIBIL Data': ['Yes', 'Yes'],
        })
        st.dataframe(sample, use_container_width=True)
    st.stop()

# ─── Process ───────────────────────────────────────────────────────
file_bytes = uploaded.read()
with st.spinner("Scoring portfolio..."):
    df, woe_tables, iv_summary, band_perf_df, auc, gini, ks, fpr, tpr, clf_report = \
        process_data(file_bytes, uploaded.name)

st.success(f"✅ Processed **{len(df):,}** records", icon="✅")

# ─── KPI Row ───────────────────────────────────────────────────────
st.markdown("### 📈 Portfolio Overview")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Records",    f"{len(df):,}")
k2.metric("Overall Default Rate", f"{df['Target'].mean():.1%}")
k3.metric("AUC",   f"{auc:.4f}"  if auc else "N/A")
k4.metric("Gini",  f"{gini:.4f}" if gini else "N/A")
k5.metric("KS",    f"{ks:.4f}"   if ks else "N/A")

# ─── Band Cards ────────────────────────────────────────────────────
st.markdown("### 🎯 Risk Band Summary")
bc1, bc2, bc3, bc4 = st.columns(4)
band_css = {'Low Risk': 'lr', 'Medium Risk': 'mr', 'High Risk': 'hr', 'Very High Risk': 'vhr'}
band_icons = {'Low Risk': '🟢', 'Medium Risk': '🟡', 'High Risk': '🟠', 'Very High Risk': '🔴'}
band_actions = {'Low Risk': '✅ APPROVE', 'Medium Risk': '🔍 REVIEW',
                'High Risk': '⛔ DECLINE', 'Very High Risk': '⛔ DECLINE'}
cols = [bc1, bc2, bc3, bc4]
for col, band in zip(cols, BAND_ORDER):
    subset = df[df['Risk_Band'] == band]
    n = len(subset)
    pct = n / max(len(df), 1) * 100
    dr = subset['Target'].mean() * 100 if n > 0 else 0
    css = band_css[band]
    col.markdown(f"""
<div class="band-card {css}-card">
    <div style="font-size:22px">{band_icons[band]} {band}</div>
    <div style="font-size:28px; font-weight:900; margin:4px 0">{n:,}</div>
    <div style="font-size:13px; opacity:0.9">{pct:.1f}% of portfolio</div>
    <div style="font-size:12px; margin-top:6px">Default: {dr:.1f}% · {band_actions[band]}</div>
</div>
""", unsafe_allow_html=True)

# ─── Charts ────────────────────────────────────────────────────────
st.markdown("### 📊 Visual Analysis")

row1 = st.columns([1.2, 1.8, 1.2])
with row1[0]:
    st.pyplot(plot_pie(df), use_container_width=True)
with row1[1]:
    st.pyplot(plot_score_hist(df), use_container_width=True)
with row1[2]:
    st.pyplot(plot_default_rate(df), use_container_width=True)

row2 = st.columns([1.2, 1.2, 1.8, 1.2])
with row2[0]:
    st.pyplot(plot_boxplot(df), use_container_width=True)
with row2[1]:
    st.pyplot(plot_iv(iv_summary), use_container_width=True)
with row2[2]:
    st.pyplot(plot_score_decile_default(df), use_container_width=True)
with row2[3]:
    if auc is not None:
        st.pyplot(plot_roc(fpr, tpr, auc, gini, ks), use_container_width=True)
    else:
        st.info("ROC requires both Good/Bad classes (min 20 records).")

# ─── Band Performance Table ─────────────────────────────────────────
st.markdown("### 📋 Band Performance Detail")
styled_bp = band_perf_df.copy()
styled_bp['Pct'] = styled_bp['Pct'].map(lambda x: f"{x:.1f}%")
styled_bp['Default_Rate'] = styled_bp['Default_Rate'].map(lambda x: f"{x:.2f}%")
styled_bp['Avg_Score'] = styled_bp['Avg_Score'].map(lambda x: f"{x:.1f}")
styled_bp['Count'] = styled_bp['Count'].map(lambda x: f"{x:,}")
st.dataframe(styled_bp, use_container_width=True, hide_index=True)

# ─── WoE / IV Table ─────────────────────────────────────────────────
with st.expander("🔬 WoE & IV Analysis"):
    woe_display = woe_tables.copy()
    woe_display['Bad_Rate'] = woe_display['Bad_Rate'].map(lambda x: f"{x:.2%}")
    woe_display['WoE'] = woe_display['WoE'].map(lambda x: f"{x:.4f}")
    woe_display['IV']  = woe_display['IV'].map(lambda x: f"{x:.4f}")
    st.dataframe(woe_display, use_container_width=True, hide_index=True)

    st.markdown("#### IV Summary by Feature")
    iv_df = iv_summary.reset_index()
    iv_df.columns = ['Feature', 'IV']
    iv_df['Strength'] = iv_df['IV'].apply(
        lambda v: '🟢 Strong' if v >= 0.3 else '🟡 Medium' if v >= 0.1 else '🟠 Weak' if v >= 0.02 else '🔴 Negligible'
    )
    iv_df['IV'] = iv_df['IV'].map(lambda x: f"{x:.4f}")
    st.dataframe(iv_df, use_container_width=True, hide_index=True)

# ─── Scored Data Preview ────────────────────────────────────────────
st.markdown("### 🗃️ Scored Records")
display_cols = ['Bank_Lan', 'Customer_ID', 'Rule_Score', 'Risk_Band', 'Target',
                'Name_Validation', 'Connected', 'PAN', 'UID', 'VI', 'DL',
                'Mobile_No', 'CIBIL', 'Document']
display_cols = [c for c in display_cols if c in df.columns]

search_q = st.text_input("🔍 Filter by Customer ID / Bank Lan", placeholder="Enter ID...")
display_df = df[display_cols]
if search_q:
    mask = (display_df.get('Customer_ID', pd.Series(dtype=str)).astype(str).str.contains(search_q, na=False) |
            display_df.get('Bank_Lan', pd.Series(dtype=str)).astype(str).str.contains(search_q, na=False))
    display_df = display_df[mask]

st.dataframe(display_df, use_container_width=True, height=350, hide_index=True)
st.caption(f"Showing {len(display_df):,} records")

# ─── Downloads ─────────────────────────────────────────────────────
st.markdown("### ⬇️ Download Reports")
dl1, dl2, dl3, dl4 = st.columns(4)

# Scored CSV
scored_csv = df[display_cols].to_csv(index=False).encode('utf-8')
dl1.download_button("📥 Scored Output", scored_csv,
                    file_name="scored_output.csv", mime="text/csv")

# Band Performance
band_csv = band_perf_df.to_csv(index=False).encode('utf-8')
dl2.download_button("📥 Band Performance", band_csv,
                    file_name="band_performance.csv", mime="text/csv")

# WoE/IV
woe_csv = woe_tables.to_csv(index=False).encode('utf-8')
dl3.download_button("📥 WoE / IV Table", woe_csv,
                    file_name="woe_iv_table.csv", mime="text/csv")

# Full Dashboard PNG
dashboard_fig = build_full_dashboard(df, iv_summary, auc, gini, ks, fpr, tpr)
dashboard_bytes = fig_to_bytes(dashboard_fig)
plt.close(dashboard_fig)
dl4.download_button("📥 Dashboard PNG", dashboard_bytes,
                    file_name="risk_dashboard.png", mime="image/png")

st.markdown("---")
st.caption("Risk Scorecard Analyser · Built on Scorecard Model v3 · Four-Tier Classification")