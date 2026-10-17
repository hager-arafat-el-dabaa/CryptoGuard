

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

"""
CryptoGuard - Illicit Bitcoin Transaction Detection (dashboard)


"""
import json
import math
import random

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CryptoGuard", layout="wide", initial_sidebar_state="collapsed")

# ---------- palette ----------------------------------------------------------
NAVY = "#07101E"
NAVY_2 = "#0C1729"
CARD = "#111D31"
CARD_2 = "#17263F"
GOLD = "#D4B06A"
GOLD_HI = "#F2D89B"
BTC = "#F7931A"
OLIVE = "#A9BC85"
STEEL = "#8DB8D6"
TEAL = "#5FB8A6"
SLATE = "#64748B"
ROSE = "#D0857B"
WHITE = "#F5F2EA"
MUTED = "#A9B4C4"
FAMILY_COLORS = {"Supervised": OLIVE, "Semi-supervised": GOLD, "Unsupervised": STEEL}

# ---------- data (from the notebook) ----------------------------------------
RESULTS = json.loads(r"""{"dataset":{"transactions":203769,"raw_feature_columns":167,"edges":234355,"unknown":157205,"licit":42019,"illicit":4545,"labeled":46564,"illicit_pct_of_labeled":9.76,"time_step_first":1,"time_step_last":49,"degree_summary":{"mean":2.300203,"std":4.328377,"min":1.0,"25%":1.0,"50%":2.0,"75%":2.0,"max":473.0},"integrity":{"ids_in_features_not_in_classes":0,"ids_in_classes_not_in_features":0,"edge_ids_not_in_features":0,"missing_values":[0,0,0],"duplicate_rows":[0,0,0]}},"split":{"train_steps":[1,34],"test_steps":[35,49],"train":{"licit":26432,"illicit":3462,"total":29894},"test":{"licit":15587,"illicit":1083,"total":16670}},"smote":{"before":{"licit":26432,"illicit":3462},"after":{"licit":26432,"illicit":26432}},"semi":{"unknown_total":157205,"unknown_used":106371,"high_conf_illicit":2002,"high_conf_licit":77320,"uncertain":27049,"original_train":29894,"pseudo_labeled":4004,"semi_train_total":33898,"semi_train_illicit":5464},"models":[{"name":"Logistic Regression","family":"Supervised","settings":"max_iter=1000","metrics":{"Accuracy":0.763,"Precision":0.1962,"Recall":0.855,"F1-Score":0.3191,"ROC-AUC":0.8843,"PR-AUC":0.2873}},{"name":"Decision Tree","family":"Supervised","settings":"default settings","metrics":{"Accuracy":0.9235,"Precision":0.4435,"Recall":0.6999,"F1-Score":0.543,"ROC-AUC":0.8194,"PR-AUC":0.3299}},{"name":"Random Forest","family":"Supervised","settings":"200 trees","metrics":{"Accuracy":0.9775,"Precision":0.9136,"Recall":0.7221,"F1-Score":0.8066,"ROC-AUC":0.936,"PR-AUC":0.7987}},{"name":"XGBoost","family":"Supervised","settings":"200 trees, max depth 6, learning rate 0.1","metrics":{"Accuracy":0.9779,"Precision":0.9152,"Recall":0.7276,"F1-Score":0.8107,"ROC-AUC":0.9466,"PR-AUC":0.817}},{"name":"LightGBM","family":"Supervised","settings":"200 trees, max depth 6, learning rate 0.1","metrics":{"Accuracy":0.9785,"Precision":0.926,"Recall":0.7276,"F1-Score":0.8149,"ROC-AUC":0.9488,"PR-AUC":0.819}},{"name":"Soft Voting Ensemble","family":"Supervised","settings":"mean of RF, XGBoost and LightGBM probabilities","metrics":{"Accuracy":0.9789,"Precision":0.9368,"Recall":0.7248,"F1-Score":0.8173,"ROC-AUC":0.9397,"PR-AUC":0.8086}},{"name":"Semi Random Forest","family":"Semi-supervised","settings":null,"metrics":{"Accuracy":0.9797,"Precision":0.9559,"Recall":0.7211,"F1-Score":0.8221,"ROC-AUC":0.9297,"PR-AUC":0.7934}},{"name":"Semi XGBoost","family":"Semi-supervised","settings":null,"metrics":{"Accuracy":0.9771,"Precision":0.9033,"Recall":0.7248,"F1-Score":0.8043,"ROC-AUC":0.9414,"PR-AUC":0.805}},{"name":"Semi LightGBM","family":"Semi-supervised","settings":null,"metrics":{"Accuracy":0.9781,"Precision":0.9194,"Recall":0.7267,"F1-Score":0.8118,"ROC-AUC":0.9373,"PR-AUC":0.8043}},{"name":"Semi Soft Voting Ensemble","family":"Semi-supervised","settings":null,"metrics":{"Accuracy":0.9788,"Precision":0.9334,"Recall":0.7248,"F1-Score":0.816,"ROC-AUC":0.9333,"PR-AUC":0.7987}},{"name":"Isolation Forest","family":"Unsupervised","settings":"200 trees, contamination 0.05","metrics":{"Accuracy":0.4643,"Precision":0.0205,"Recall":0.1551,"F1-Score":0.0363,"ROC-AUC":0.1897,"PR-AUC":0.0371}}],"ablation_without_feature_104":{"Accuracy":0.9785,"Precision":0.9269,"Recall":0.7258,"F1-Score":0.8141,"ROC-AUC":0.9433,"PR-AUC":0.8106},"confusion_matrix_soft_voting":{"TN":15534,"FP":53,"FN":298,"TP":785}}""")

MODELS = {m["name"]: m for m in RESULTS["models"]}
DS = RESULTS["dataset"]
SPLIT = RESULTS["split"]
SEMI = RESULTS["semi"]
SMOTE = RESULTS["smote"]
CM = RESULTS["confusion_matrix_soft_voting"]
ABL = RESULTS["ablation_without_feature_104"]
SV = MODELS["Soft Voting Ensemble"]["metrics"]
SEMI_RF = MODELS["Semi Random Forest"]["metrics"]
IFO = MODELS["Isolation Forest"]["metrics"]

ROLES = {
    "Logistic Regression": "خط أساس خطي. استدعاء عالٍ جداً، لكن أغلب التنبيهات إنذارات كاذبة.",
    "Decision Tree": "شجرة واحدة بالإعدادات الافتراضية. تضحّي بالدقة مقابل الاستدعاء.",
    "Random Forest": "مجموعة من 200 شجرة. دقة قوية واستدعاء مستقر.",
    "XGBoost": "تعزيز تدريجي: 200 شجرة، عمق 6، معدل تعلّم 0.1.",
    "LightGBM": "تعزيز تدريجي بنفس إعدادات XGBoost. أعلى ROC-AUC بين النماذج الخاضعة للإشراف.",
    "Soft Voting Ensemble": "يحسب متوسط احتمالات الشبهة من RF وXGBoost وLightGBM. أفضل F1 خاضع للإشراف.",
    "Semi Random Forest": "Random Forest أعيد تدريبه مع 4,004 معاملة بتصنيف مبدئي متوازن. أعلى F1 بشكل عام.",
    "Semi XGBoost": "XGBoost أعيد تدريبه مع البيانات ذات التصنيف المبدئي.",
    "Semi LightGBM": "LightGBM أعيد تدريبه مع البيانات ذات التصنيف المبدئي.",
    "Semi Soft Voting Ensemble": "تصويت مرجّح على النماذج الثلاثة شبه الخاضعة للإشراف.",
    "Isolation Forest": "يتعلم بدون تصنيفات. درجته تشير في الاتجاه المعاكس للمعاملات غير المشروعة (ROC-AUC أقل من 0.5).",
}
METRIC_NOTES = {
    "Accuracy": "مضللة هنا: أغلب المعاملات مشروعة، فالنموذج الذي يعلّم القليل يحصل على دقة عالية أيضاً.",
    "Precision": "من المعاملات التي وُسمت بالشبهة، النسبة التي كانت فعلاً شبهة.",
    "Recall": "من المعاملات الشبهة في مجموعة الاختبار، النسبة التي تم اكتشافها.",
    "F1-Score": "المتوسط التوافقي للدقة والاستدعاء. المعيار الرئيسي لاختيار النموذج في المشروع.",
    "ROC-AUC": "جودة الترتيب عبر كل العتبات. لا يتأثر بعتبة 0.5.",
    "PR-AUC": "متوسط الدقة. أدق من ROC-AUC لما المعاملات الشبهة تكون نادرة.",
}


# ---------- helpers ----------------------------------------------------------
def h(s):
    """Collapse whitespace so Markdown never treats HTML as an indented code block."""
    return " ".join(s.split())


def fmt(n):
    return f"{n:,}"


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def btc_mark(size=44):
    return (f'<svg viewBox="0 0 64 64" width="{size}" height="{size}" class="btc">'
            f'<defs><linearGradient id="btcg" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="#FFB347"/><stop offset="1" stop-color="{BTC}"/></linearGradient></defs>'
            f'<circle cx="32" cy="32" r="30" fill="url(#btcg)"/>'
            f'<text x="32" y="43" text-anchor="middle" font-family="Arial,sans-serif" font-size="34" '
            f'font-weight="700" fill="{NAVY}">B</text></svg>')


def network_svg(width=1000, height=460, n=40, seed=11, cls="cg-net", highlight_every=6):
    """Decorative transaction graph. Positions are generated, not data."""
    rnd = random.Random(seed)
    pts = [(rnd.uniform(30, width - 30), rnd.uniform(24, height - 24)) for _ in range(n)]
    edges = set()
    for i, (x, y) in enumerate(pts):
        near = sorted(range(n), key=lambda j: (pts[j][0] - x) ** 2 + (pts[j][1] - y) ** 2)[1:3]
        for j in near:
            edges.add((min(i, j), max(i, j)))
    out = [f'<svg class="{cls}" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid slice" '
           f'xmlns="http://www.w3.org/2000/svg">']
    for k, (i, j) in enumerate(sorted(edges)):
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        hot = " hot" if k % highlight_every == 0 else ""
        out.append(f'<line class="ln{hot}" pathLength="1" style="--d:{k * 0.035:.2f}s" '
                   f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
    for k, (x, y) in enumerate(pts):
        r = 6 if k % 7 == 0 else 3
        out.append(f'<circle class="nd" style="--d:{k * 0.05:.2f}s" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>')
    out.append("</svg>")
    return "".join(out)


def pipeline_svg(labels):
    """Horizontal flow of the pipeline steps with animated particles."""
    w, height = 1000, 150
    n = len(labels)
    step = (w - 120) / (n - 1)
    parts = [f'<svg class="flow" viewBox="0 0 {w} {height}" xmlns="http://www.w3.org/2000/svg">',
             f'<line class="rail" x1="60" y1="60" x2="{w - 60}" y2="60"/>']
    for i, lab in enumerate(labels):
        x = 60 + i * step
        parts.append(f'<circle class="pn" style="--d:{i * 0.12:.2f}s" cx="{x:.1f}" cy="60" r="16"/>')
        parts.append(f'<text class="pt" x="{x:.1f}" y="62" text-anchor="middle">{i + 1:02d}</text>')
        parts.append(f'<text class="pl" x="{x:.1f}" y="118" text-anchor="middle">{lab}</text>')
    parts.append("</svg>")
    return "".join(parts)


def sec(kicker, title, lead=None):
    lead_html = f'<p class="l">{lead}</p>' if lead else ""
    st.markdown(h(f'<div class="cg-sec"><div class="cg-kicker">{kicker}</div>'
                  f'<div class="t">{title}</div>{lead_html}</div>'), unsafe_allow_html=True)


def metric(label, value, note="", accent=False, i=0):
    cls = "cg-metric accent" if accent else "cg-metric"
    return (f'<div class="{cls}" style="animation-delay:{i * 80}ms"><div class="lab">{label}</div>'
            f'<div class="val">{value}</div><div class="note">{note}</div></div>')


def card(title, body, i=0):
    return (f'<div class="cg-card" style="animation-delay:{i * 80}ms"><div class="ct">{title}</div>'
            f'<div class="cb">{body}</div></div>')


def grid(items, cls="c4"):
    st.markdown(h(f'<div class="cg-grid {cls}">' + "".join(items) + "</div>"), unsafe_allow_html=True)


def callout(title, body, kind=""):
    cls = f"cg-callout {kind}".strip()
    st.markdown(h(f'<div class="{cls}"><div class="ct">{title}</div><div class="cb">{body}</div></div>'),
                unsafe_allow_html=True)


def pill(kind, text):
    return f'<span class="cg-pill p-{kind}">{text}</span>'


def bar_row(label, value, width, cls=""):
    return (f'<div class="cg-barlab"><span>{label}</span><b>{value}</b></div>'
            f'<div class="cg-bar {cls}"><i style="--w:{width:.2f}%"></i></div>')


def plot(fig, height=400, legend=False):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=MUTED, size=12),
        margin=dict(l=10, r=10, t=44, b=10), showlegend=legend,
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0A1322", bordercolor=GOLD, font=dict(color=WHITE)),
        title=dict(font=dict(family="Cormorant Garamond, Georgia, serif", size=20, color=WHITE)),
    )
    fig.update_xaxes(gridcolor="rgba(245,242,234,.06)", zerolinecolor="rgba(245,242,234,.12)")
    fig.update_yaxes(gridcolor="rgba(245,242,234,.06)", zerolinecolor="rgba(245,242,234,.12)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------- style ------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Cormorant+Garamond:wght@500;600;700&display=swap');
:root{--navy:#07101E;--navy2:#0C1729;--card:#111D31;--card2:#17263F;--gold:#D4B06A;--gold-hi:#F2D89B;--btc:#F7931A;
  --olive:#A9BC85;--steel:#8DB8D6;--rose:#D0857B;--white:#F5F2EA;--muted:#A9B4C4;
  --line:rgba(245,242,234,.08);--line-gold:rgba(212,176,106,.36);}
.stApp{background:
  radial-gradient(1100px 620px at 96% -12%, rgba(247,147,26,.10), transparent 60%),
  radial-gradient(900px 560px at -8% 108%, rgba(169,188,133,.09), transparent 60%),
  linear-gradient(180deg,#07101E 0%,#0B1628 100%);
  font-family:"Inter",system-ui,sans-serif;color:var(--white);}
.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.35;
  background-image:linear-gradient(rgba(212,176,106,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(212,176,106,.035) 1px,transparent 1px);
  background-size:56px 56px;mask-image:radial-gradient(ellipse at 50% 30%,#000 30%,transparent 80%);}
footer,#MainMenu,.stDeployButton{visibility:hidden;}
.block-container{max-width:1240px;padding-top:2.2rem;padding-bottom:4.5rem;position:relative;z-index:1;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0A1424 0%,#070E1B 100%);border-right:1px solid var(--line);}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]{border-radius:12px;margin:3px 8px;border:1px solid transparent;transition:background .25s,border-color .25s;}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover{background:rgba(245,242,234,.05);}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"]{background:rgba(212,176,106,.12);border-color:var(--line-gold);}

.cg-brand{display:flex;align-items:center;gap:12px;margin:.2rem 0 1.2rem;}
.cg-brand b{font-family:"Cormorant Garamond",Georgia,serif;font-size:1.35rem;font-weight:700;letter-spacing:.01em;}
.cg-brand small{display:block;color:var(--muted);font-size:.66rem;letter-spacing:.22em;text-transform:uppercase;}
.btc{filter:drop-shadow(0 6px 18px rgba(247,147,26,.35));}

.cg-kicker{font:600 .7rem "Inter",sans-serif;letter-spacing:.26em;text-transform:uppercase;color:var(--gold);}
.cg-h1{font-family:"Cormorant Garamond",Georgia,serif;font-weight:700;font-size:clamp(2.4rem,4.4vw,3.6rem);line-height:1.03;margin:.4rem 0 .8rem;
  background:linear-gradient(120deg,#F5F2EA 40%,#F2D89B 100%);-webkit-background-clip:text;background-clip:text;color:transparent;}
.cg-lead{color:var(--muted);max-width:70ch;line-height:1.75;font-size:1.02rem;}

.cg-hero{display:grid;grid-template-columns:1.1fr 1fr;gap:26px;align-items:center;padding:8px 0 4px;}
.cg-hero-art{position:relative;height:300px;border-radius:24px;border:1px solid var(--line-gold);overflow:hidden;
  background:radial-gradient(circle at 50% 40%,rgba(247,147,26,.14),transparent 62%),linear-gradient(180deg,#0E1B30,#0A1424);
  box-shadow:0 30px 60px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.04);}
.cg-hero-art .cg-net{position:absolute;inset:0;width:100%;height:100%;}
.cg-hero-art .btc-wrap{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);}
@media (max-width:820px){.cg-hero{grid-template-columns:1fr;}}

.cg-sec{margin:2.6rem 0 1rem;padding-top:.6rem;position:relative;}
.cg-sec::before{content:"";position:absolute;left:0;top:0;width:64px;height:2px;background:linear-gradient(90deg,var(--btc),var(--gold));border-radius:2px;}
.cg-sec .t{font-family:"Cormorant Garamond",Georgia,serif;font-weight:700;font-size:1.95rem;color:var(--white);margin:.3rem 0 .35rem;line-height:1.1;}
.cg-sec .l{color:var(--muted);font-size:.95rem;line-height:1.7;max-width:74ch;margin:0 0 .6rem;}

.cg-grid{display:grid;gap:16px;margin:.5rem 0 .9rem;}
.c2{grid-template-columns:repeat(auto-fit,minmax(300px,1fr));}
.c3{grid-template-columns:repeat(auto-fit,minmax(250px,1fr));}
.c4{grid-template-columns:repeat(auto-fit,minmax(190px,1fr));}
.cg-card,.cg-metric{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:18px;padding:18px 20px;position:relative;overflow:hidden;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.03),0 16px 36px rgba(0,0,0,.25);transition:transform .3s ease,border-color .3s ease,box-shadow .3s ease;}
.cg-card:hover,.cg-metric:hover{transform:translateY(-3px);border-color:var(--line-gold);box-shadow:0 0 0 1px rgba(212,176,106,.10),0 24px 48px rgba(0,0,0,.4);}
.cg-card .ct{font-family:"Cormorant Garamond",Georgia,serif;font-size:1.3rem;font-weight:700;color:var(--white);margin-bottom:.35rem;line-height:1.2;}
.cg-card .cb{color:var(--muted);font-size:.9rem;line-height:1.7;}
.cg-card ul{margin:.3rem 0 0;padding-left:1.1rem;color:var(--muted);font-size:.9rem;line-height:1.75;}
.cg-card b{color:var(--white);font-weight:600;}
.cg-metric::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(180deg,var(--btc),transparent);}
.cg-metric .lab{font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);}
.cg-metric .val{font-family:"Cormorant Garamond",Georgia,serif;font-size:2.4rem;font-weight:700;color:var(--white);line-height:1.1;margin:.3rem 0;font-variant-numeric:lining-nums;}
.cg-metric .note{font-size:.8rem;color:var(--muted);line-height:1.5;}
.cg-metric.accent .val{background:linear-gradient(120deg,#FFE3A8,#F7931A);-webkit-background-clip:text;background-clip:text;color:transparent;}

.cg-pill{display:inline-block;font:600 .62rem "Inter",sans-serif;letter-spacing:.14em;text-transform:uppercase;padding:4px 10px;border-radius:999px;border:1px solid;margin-left:6px;vertical-align:middle;}
.p-sup{color:var(--olive);border-color:rgba(169,188,133,.45);background:rgba(169,188,133,.08);}
.p-semi{color:var(--gold);border-color:rgba(212,176,106,.5);background:rgba(212,176,106,.08);}
.p-uns{color:var(--steel);border-color:rgba(141,184,214,.45);background:rgba(141,184,214,.08);}

.cg-callout{border-left:3px solid var(--btc);background:linear-gradient(90deg,rgba(247,147,26,.10),rgba(247,147,26,.02));padding:14px 18px;border-radius:0 14px 14px 0;margin:.8rem 0;}
.cg-callout .ct{font-weight:600;color:var(--white);font-size:.93rem;margin-bottom:.2rem;}
.cg-callout .cb{color:var(--muted);font-size:.9rem;line-height:1.7;}
.cg-callout.note{border-color:var(--olive);background:linear-gradient(90deg,rgba(169,188,133,.10),rgba(169,188,133,.02));}
.cg-callout.caution{border-color:var(--rose);background:linear-gradient(90deg,rgba(208,133,123,.12),rgba(208,133,123,.02));}

.cg-strip{display:grid;grid-template-columns:repeat(49,1fr);gap:3px;margin:.6rem 0 .3rem;}
.cg-strip i{height:26px;border-radius:4px;background:linear-gradient(180deg,#B9CA96,#8FA16C);transition:transform .2s;}
.cg-strip i.te{background:linear-gradient(180deg,#F2D89B,#B8924F);}
.cg-strip i:hover{transform:scaleY(1.18);}
.cg-strip-lab{display:flex;justify-content:space-between;font-size:.75rem;color:var(--muted);margin-bottom:.9rem;}

.cg-bar{height:7px;border-radius:999px;background:rgba(245,242,234,.07);margin:4px 0 10px;overflow:hidden;}
.cg-bar>i{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,#8FA16C,#A9BC85);width:var(--w);}
.cg-bar.gold>i{background:linear-gradient(90deg,#B8924F,#F7931A);}
.cg-bar.steel>i{background:linear-gradient(90deg,#5E8BAA,#8DB8D6);}
.cg-barlab{display:flex;justify-content:space-between;font-size:.78rem;color:var(--muted);}
.cg-barlab b{color:var(--white);font-weight:600;}

.cg-tw{overflow-x:auto;border:1px solid var(--line);border-radius:16px;background:rgba(10,18,32,.6);}
.cg-table{width:100%;border-collapse:collapse;font-size:.87rem;min-width:640px;}
.cg-table th{font:600 .66rem "Inter",sans-serif;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);text-align:right;padding:13px 14px;background:rgba(8,14,26,.75);}
.cg-table td{padding:11px 14px;text-align:right;color:var(--white);border-top:1px solid var(--line);white-space:nowrap;font-variant-numeric:tabular-nums;}
.cg-table tr:hover td{background:rgba(245,242,234,.03);}
.cg-table td.l,.cg-table th:first-child,.cg-table th:nth-child(2){text-align:left;}
.cg-table td.l{color:var(--muted);}
.cg-table td.best{color:var(--btc);font-weight:700;}

.flow{width:100%;height:auto;margin:.4rem 0 .2rem;}
.flow .rail{stroke:rgba(212,176,106,.25);stroke-width:2;stroke-dasharray:6 8;}
.flow .pn{fill:#0E1B30;stroke:var(--gold);stroke-width:1.5;}
.flow .pt{fill:var(--gold);font:700 12px Inter,sans-serif;}
.flow .pl{fill:var(--muted);font:500 13px Inter,sans-serif;}

.cg-intro-wrap{position:relative;min-height:80vh;}
.cg-intro-wrap .cg-net{position:absolute;inset:-40px -20px;width:calc(100% + 40px);height:calc(100% + 80px);opacity:.55;pointer-events:none;}
.cg-intro{position:relative;min-height:80vh;display:flex;flex-direction:column;align-items:center;text-align:center;padding-top:5vh;}
.cg-intro-title{font-family:"Cormorant Garamond",Georgia,serif;font-weight:700;font-size:clamp(3.4rem,9.5vw,7.8rem);line-height:1;
  background:linear-gradient(120deg,#F5F2EA 25%,#F2D89B 60%,#F7931A 100%);-webkit-background-clip:text;background-clip:text;color:transparent;}
.cg-intro-sub{margin-top:1rem;font-weight:500;font-size:clamp(.7rem,1.4vw,.95rem);letter-spacing:.42em;color:var(--muted);text-transform:uppercase;}
.cg-intro-rule{width:110px;height:1px;background:linear-gradient(90deg,transparent,var(--btc),transparent);margin:1.8rem 0 1.6rem;}
.cg-icards{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:18px;width:100%;max-width:1080px;text-align:left;}
.cg-icard{background:linear-gradient(180deg,rgba(23,38,63,.96),rgba(15,26,45,.96));border:1px solid var(--line);border-radius:20px;padding:22px;box-shadow:0 24px 50px rgba(0,0,0,.4);}
.cg-icard .k{font:600 .64rem "Inter",sans-serif;letter-spacing:.22em;text-transform:uppercase;color:var(--gold);}
.cg-icard .t{font-family:"Cormorant Garamond",Georgia,serif;font-size:1.6rem;font-weight:700;color:var(--white);margin:.35rem 0 .55rem;}
.cg-icard ul{margin:0;padding-left:1rem;color:var(--muted);font-size:.88rem;line-height:1.8;}
.cg-icard .n{margin-top:.75rem;font-size:.8rem;color:var(--olive);line-height:1.55;}
.cg-intro-desc{max-width:68ch;color:var(--muted);line-height:1.8;margin:2.2rem 0 .8rem;font-size:1.02rem;}
.cg-intro-tag{font:600 .68rem "Inter",sans-serif;letter-spacing:.32em;color:var(--btc);text-transform:uppercase;}

.cg-net line{stroke:rgba(212,176,106,.22);stroke-width:1;}
.cg-net line.hot{stroke:rgba(247,147,26,.55);stroke-width:1.4;}
.cg-net circle{fill:var(--gold);}
.cg-net circle:nth-of-type(7n){fill:var(--btc);}

@media (prefers-reduced-motion: no-preference){
  @keyframes cgTitle{from{opacity:0;transform:translateY(14vh);filter:blur(12px);}to{opacity:1;transform:translateY(0);filter:blur(0);}}
  @keyframes cgFade{from{opacity:0;}to{opacity:1;}}
  @keyframes cgDrop{from{opacity:0;transform:translateY(-30px);}to{opacity:1;transform:translateY(0);}}
  @keyframes cgRise{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:translateY(0);}}
  @keyframes cgDraw{from{stroke-dashoffset:1;opacity:0;}to{stroke-dashoffset:0;opacity:1;}}
  @keyframes cgPop{from{opacity:0;transform:scale(.2);}to{opacity:1;transform:scale(1);}}
  @keyframes cgFlow{to{stroke-dashoffset:-28;}}
  @keyframes cgGlow{0%,100%{box-shadow:0 0 0 0 rgba(247,147,26,.0);}50%{box-shadow:0 0 0 12px rgba(247,147,26,.10);}}

  .cg-intro-title{animation:cgTitle 3s cubic-bezier(.22,.7,.2,1) backwards;}
  .cg-intro-sub{animation:cgFade 1.4s ease 2s backwards;}
  .cg-intro-rule{animation:cgFade 1s ease 2.7s backwards;}
  .cg-icard{animation:cgDrop 1s cubic-bezier(.2,.8,.2,1) backwards;}
  .cg-icard:nth-child(1){animation-delay:3.1s;} .cg-icard:nth-child(2){animation-delay:3.7s;} .cg-icard:nth-child(3){animation-delay:4.3s;}
  .cg-intro-desc,.cg-intro-tag{animation:cgFade 1.2s ease 4.9s backwards;}
  .cg-card,.cg-metric{animation:cgRise .7s ease backwards;}
  .cg-net line{stroke-dasharray:1;stroke-dashoffset:1;animation:cgDraw 2.2s ease var(--d) forwards;}
  .cg-net circle{opacity:0;animation:cgPop .6s ease var(--d) forwards;transform-origin:center;transform-box:fill-box;}
  .flow .rail{animation:cgFlow 1.4s linear infinite;}
  .flow .pn{opacity:0;animation:cgPop .6s ease var(--d) forwards;transform-origin:center;transform-box:fill-box;}
  .cg-hero-art .btc-wrap{animation:cgGlow 3.2s ease-in-out infinite;border-radius:50%;}
}
@media (max-width:760px){
  .block-container{padding-left:1rem;padding-right:1rem;}
  .cg-metric .val{font-size:2rem;}
  .cg-hero-art{height:240px;}
}
</style>
"""


# ---------- intro ------------------------------------------------------------
def render_intro():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(h(f"""
    <div class="cg-intro-wrap">
      {network_svg(seed=21, n=46)}
      <div class="cg-intro">
        <div class="cg-intro-title">CryptoGuard</div>
        <div class="cg-intro-sub">Illicit Bitcoin Transaction Detection</div>
        <div class="cg-intro-rule"></div>
        <div class="cg-icards">
          <div class="cg-icard">
            <div class="k">Supervised learning</div>
            <div class="t">Labeled transactions</div>
            <ul><li>Logistic Regression</li><li>Decision Tree</li><li>Random Forest</li>
                <li>XGBoost</li><li>LightGBM</li><li>Soft Voting Ensemble</li></ul>
            <div class="n">Trained on time steps 1 to 34, tested on steps 35 to 49.</div>
          </div>
          <div class="cg-icard">
            <div class="k">Semi-supervised learning</div>
            <div class="t">Pseudo-labeled unknowns</div>
            <ul><li>Semi Random Forest</li><li>Semi XGBoost</li><li>Semi LightGBM</li>
                <li>Semi Soft Voting</li></ul>
            <div class="n">Adds 4,004 confident pseudo-labeled transactions from the training period.</div>
          </div>
          <div class="cg-icard">
            <div class="k">Unsupervised learning</div>
            <div class="t">Anomaly detection</div>
            <ul><li>Isolation Forest</li></ul>
            <div class="n">Trained without labels. A complementary check, not the main classifier.</div>
          </div>
        </div>
        <p class="cg-intro-desc">CryptoGuard ranks Bitcoin transactions by how likely they are to be illicit,
        using graph features, a time-based split and classical machine learning. A high score marks a
        transaction for review. It is not proof of illegal activity.</p>
        <div class="cg-intro-tag">Elliptic Bitcoin Transaction Dataset</div>
      </div>
    </div>
    """), unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Explore CryptoGuard", type="primary", width="stretch"):
            st.session_state["entered"] = True
            st.rerun()


# ---------- pages ------------------------------------------------------------
def page_overview():
    st.markdown(h(f"""
    <div class="cg-hero">
      <div>
        <div class="cg-kicker">Machine learning project</div>
        <div class="cg-h1">Illicit Bitcoin transaction detection</div>
        <p class="cg-lead">CryptoGuard studies the Elliptic Bitcoin transaction graph. It builds a temporal
        graph-degree feature, splits the data by time instead of at random, and compares supervised,
        semi-supervised and unsupervised models. The goal is to rank transactions so investigators can focus
        on the most suspicious ones first.</p>
      </div>
      <div class="cg-hero-art">
        {network_svg(seed=5, n=38, height=300)}
        <div class="btc-wrap">{btc_mark(92)}</div>
      </div>
    </div>
    """), unsafe_allow_html=True)

    grid([
        metric("Transactions", fmt(DS["transactions"]), f"{DS['raw_feature_columns'] - 2} features after ID and time step", i=0),
        metric("Graph edges", fmt(DS["edges"]), "Links used to build the degree feature", i=1),
        metric("Labeled", fmt(DS["labeled"]), f"{DS['illicit_pct_of_labeled']:.2f}% illicit. {fmt(DS['unknown'])} unlabeled.", i=2),
        metric("Final test set", fmt(SPLIT["test"]["total"]), f"{fmt(SPLIT['test']['illicit'])} illicit. Steps 35 to 49, used once.", accent=True, i=3),
    ])

    sec("Approaches", "Three learning approaches, one question",
        "All models were scored on the same final test set at a fixed 0.5 threshold.")
    grid([
        card(f"Supervised {pill('sup', 'Implemented')}",
             f"Six models trained on labeled transactions only. Best F1: <b>{pct(SV['F1-Score'])}</b> "
             f"(Soft Voting), precision <b>{pct(SV['Precision'])}</b>, recall <b>{pct(SV['Recall'])}</b>.", 0),
        card(f"Semi-supervised {pill('semi', 'Implemented')}",
             f"Four models retrained with confident pseudo-labels. Best F1: <b>{pct(SEMI_RF['F1-Score'])}</b> "
             "(Semi Random Forest), a small gain over the supervised ensemble.", 1),
        card(f"Unsupervised {pill('uns', 'Implemented')}",
             f"Isolation Forest, no labels. Test F1 is only <b>{pct(IFO['F1-Score'])}</b>, so it is a "
             "complementary analysis, not a classifier.", 2),
    ], cls="c3")

    sec("Reading the results", "What the numbers do and do not say")
    callout("Accuracy is not the headline",
            f"Only {DS['illicit_pct_of_labeled']:.2f}% of labeled transactions are illicit. Accuracy above 97 percent "
            "is easy to reach by flagging few transactions, so models are ranked by F1 with precision and recall alongside.")
    callout("The top models are close",
            f"Semi Random Forest ({pct(SEMI_RF['F1-Score'])}) and Soft Voting ({pct(SV['F1-Score'])}) differ by about half "
            "a point on a single test set. That is too small to rank them with confidence.", "note")
    callout("Scores are not verdicts",
            "The models output a probability. Nothing here connects to a live Bitcoin network, and a flagged "
            "transaction is not shown to be illegal.", "caution")


def page_dataset():
    sec("Dataset", "The Elliptic transaction graph",
        "Three files: transaction features, class labels and an edge list. Class 1 is illicit, class 2 is licit, the rest are unknown.")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        labels = ["Unknown", "Licit", "Illicit"]
        values = [DS["unknown"], DS["licit"], DS["illicit"]]
        fig = go.Figure(go.Bar(x=labels, y=values, marker=dict(color=[SLATE, TEAL, BTC]),
                               text=[fmt(v) for v in values], textposition="outside",
                               textfont=dict(color=WHITE)))
        fig.update_layout(title="Transactions by class")
        fig.update_yaxes(title="Transactions", tickformat=",")
        plot(fig, height=390)
    with right:
        grid([
            metric("Unknown", fmt(DS["unknown"]), f"{DS['unknown'] / DS['transactions'] * 100:.2f}% of all transactions", i=0),
            metric("Labeled", fmt(DS["labeled"]), f"{fmt(DS['licit'])} licit and {fmt(DS['illicit'])} illicit", i=1),
            metric("Illicit share of labeled", f"{DS['illicit_pct_of_labeled']:.2f}%", "The imbalance the models must handle", accent=True, i=2),
            metric("Time steps", f"{DS['time_step_first']} to {DS['time_step_last']}", "Used for a chronological split", i=3),
        ], cls="c2")

    sec("Time", "Forty-nine time steps, split by time",
        "Gold steps are the test period. Time is used for the split, never as a model feature.")
    cells = []
    for t in range(DS["time_step_first"], DS["time_step_last"] + 1):
        test = t >= SPLIT["test_steps"][0]
        cells.append(f'<i class="{"te" if test else ""}" title="Step {t}: {"test" if test else "train"}"></i>')
    st.markdown(h('<div class="cg-strip">' + "".join(cells) + "</div>"), unsafe_allow_html=True)
    st.markdown(h(f"""<div class="cg-strip-lab"><span>Step {DS['time_step_first']}</span>
        <span>Train: steps 1 to 34</span><span>Test: steps 35 to 49</span><span>Step {DS['time_step_last']}</span></div>"""),
        unsafe_allow_html=True)

    sec("Degree", "Graph degree, the one added feature")
    ds = DS["degree_summary"]
    grid([
        metric("Mean degree", f"{ds['mean']:.2f}", "Edges counted at each transaction's own time step", i=0),
        metric("Median degree", f"{ds['50%']:.0f}", f"75th percentile is {ds['75%']:.0f}", i=1),
        metric("Largest degree", f"{ds['max']:.0f}", "A few hub transactions have very many links", i=2),
        metric("Standard deviation", f"{ds['std']:.2f}", "Heavy right tail", i=3),
    ])

    sec("Quality", "Data integrity checks")
    ig = DS["integrity"]
    grid([
        card(f"IDs match {pill('sup', 'Checked')}",
             f"Features and classes share every ID. All {ig['edge_ids_not_in_features']} edge endpoints exist in the features file.", 0),
        card(f"No missing values {pill('sup', 'Checked')}", "Zero missing values in the features, classes and edge files.", 1),
        card(f"No duplicate rows {pill('sup', 'Checked')}", "Zero duplicate rows in any of the three files.", 2),
    ], cls="c3")

    sec("Limitations", "What this dataset cannot tell us")
    st.markdown(h(f"""<div class="cg-card"><div class="cb"><ul>
      <li>Only {pct(DS['labeled'] / DS['transactions'], 1)} of transactions carry a label, so models learn from a small, non-random slice.</li>
      <li>Only {fmt(DS['illicit'])} illicit labels exist, so every result carries wide uncertainty.</li>
      <li>Features are anonymized, so they cannot be explained in domain terms.</li>
      <li>Results come from one dataset snapshot and say nothing about later Bitcoin activity.</li>
    </ul></div></div>"""), unsafe_allow_html=True)


def page_methodology():
    sec("Pipeline", "From raw files to one test score",
        "Every step is in the notebook. The test set is touched only at the very end.")
    st.markdown(pipeline_svg(["Load", "Separate", "Degree", "Split", "Scale", "SMOTE", "Train", "Pseudo", "Isolate", "Test"]),
                unsafe_allow_html=True)
    steps = [
        ("01", "Load and verify", "Load the three files and check IDs, missing values and duplicates."),
        ("02", "Separate labeled and unknown", "Merge labels into features. Unknown transactions stay out of supervised training."),
        ("03", "Temporal degree", "Count each transaction's edges that exist at its own time step, so future edges do not leak in."),
        ("04", "Chronological split", "Train on steps 1 to 34, test on steps 35 to 49. No validation set."),
        ("05", "Scale on training data", "StandardScaler is fitted on the training set only, then applied to the test set."),
        ("06", "Balance with SMOTE", f"Oversample illicit transactions in training only: {fmt(SMOTE['before']['illicit'])} to {fmt(SMOTE['after']['illicit'])} per class."),
        ("07", "Train supervised models", "Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM and a soft voting ensemble."),
        ("08", "Pseudo-label unknowns", "Use the ensemble on unknown transactions from steps 1 to 34. Keep probabilities of 0.90 or more, or 0.10 or less."),
        ("09", "Unsupervised check", "Fit Isolation Forest on unlabeled training transactions and compare its flags with the illicit labels."),
        ("10", "Evaluate once", "Score every model on the untouched test set at threshold 0.5."),
    ]
    grid([card(f"{n}. {t}", b, i) for i, (n, t, b) in enumerate(steps)], cls="c3")

    sec("Balance", "Class balance before and after SMOTE")
    grid([
        metric("Train licit", fmt(SMOTE["before"]["licit"]), "Real labeled transactions", i=0),
        metric("Train illicit, before", fmt(SMOTE["before"]["illicit"]), "Real labeled transactions", i=1),
        metric("Train illicit, after", fmt(SMOTE["after"]["illicit"]), "Synthetic samples added by SMOTE", accent=True, i=2),
        metric("Test illicit", fmt(SPLIT["test"]["illicit"]), f"Out of {fmt(SPLIT['test']['total'])}, natural imbalance kept", i=3),
    ])

    sec("Semi-supervised", "How the pseudo-labels were built")
    grid([
        metric("Unknown, steps 1 to 34", fmt(SEMI["unknown_used"]), f"Out of {fmt(SEMI['unknown_total'])} unknown overall", i=0),
        metric("Confident illicit", fmt(SEMI["high_conf_illicit"]), "Probability of 0.90 or higher", i=1),
        metric("Confident licit", fmt(SEMI["high_conf_licit"]), "Probability of 0.10 or lower, sampled to match", i=2),
        metric("Left uncertain", fmt(SEMI["uncertain"]), "Between 0.10 and 0.90, not used", i=3),
    ])
    callout("Training set size",
            f"Labeled training rows: {fmt(SEMI['original_train'])}. Adding {fmt(SEMI['pseudo_labeled'])} pseudo-labeled rows gives "
            f"{fmt(SEMI['semi_train_total'])}, of which {fmt(SEMI['semi_train_illicit'])} are marked illicit. Some of those labels "
            "are the model's own guesses, so the semi-supervised gain should be read with care.", "note")


def model_card(name, family, i):
    mt = MODELS[name]["metrics"]
    fam_kind = {"Supervised": "sup", "Semi-supervised": "semi", "Unsupervised": "uns"}[family]
    settings = MODELS[name].get("settings") or ""
    bars = (bar_row("Recall", pct(mt["Recall"]), mt["Recall"] * 100)
            + bar_row("Precision", pct(mt["Precision"]), mt["Precision"] * 100, "gold")
            + bar_row("F1-Score", pct(mt["F1-Score"]), mt["F1-Score"] * 100, "steel"))
    body = (f'<div>{pill(fam_kind, family)}</div>'
            f'<div class="cb" style="margin:.5rem 0">{ROLES.get(name, "")}</div>'
            f'<div class="cb" style="font-size:.8rem">Settings: {settings}</div>'
            f'<div style="margin-top:.6rem">{bars}</div>'
            f'<div class="cb" style="font-size:.8rem">ROC-AUC {mt["ROC-AUC"]:.4f} &middot; PR-AUC {mt["PR-AUC"]:.4f}</div>')
    return f'<div class="cg-card" style="animation-delay:{i * 80}ms"><div class="ct">{name}</div>{body}</div>'


def page_approaches():
    sec("Approaches", "Models by learning approach", "Test-set results from the notebook. Bars are on a 0 to 100 percent scale.")
    fams = {"Supervised": [], "Semi-supervised": [], "Unsupervised": []}
    for m in RESULTS["models"]:
        fams[m["family"]].append(m["name"])
    tab1, tab2, tab3 = st.tabs(["Supervised", "Semi-supervised", "Unsupervised"])
    with tab1:
        st.markdown(h("""<p class="cg-lead">Trained on the 29,894 labeled transactions from steps 1 to 34, after scaling and SMOTE.</p>"""),
                    unsafe_allow_html=True)
        grid([model_card(n, "Supervised", i) for i, n in enumerate(fams["Supervised"])], cls="c3")
    with tab2:
        st.markdown(h("""<p class="cg-lead">Each model is retrained with its supervised settings plus 4,004 balanced pseudo-labeled transactions.</p>"""),
                    unsafe_allow_html=True)
        grid([model_card(n, "Semi-supervised", i) for i, n in enumerate(fams["Semi-supervised"])], cls="c3")
        callout("Compared with the supervised version",
                "For the ensemble, pseudo-labeling changes F1 by less than one point. Semi XGBoost is slightly below plain XGBoost. "
                "The gain is not consistent across models.", "note")
    with tab3:
        st.markdown(h("""<p class="cg-lead">Isolation Forest is trained without labels, on unknown transactions from steps 1 to 34. It is not a classifier.</p>"""),
                    unsafe_allow_html=True)
        grid([model_card(n, "Unsupervised", i) for i, n in enumerate(fams["Unsupervised"])], cls="c3")
        callout("Why its ROC-AUC is below 0.5",
                "Higher anomaly scores mean more unusual. Illicit transactions tend to sit at lower scores than licit ones, "
                "so the ranking works against the label. The forest does not detect illicit activity on this data.", "caution")


def page_comparison():
    sec("Comparison", "Final test results across all models",
        "Test set: steps 35 to 49, 16,670 transactions, 1,083 illicit. Threshold 0.5 for every model.")
    choice = st.selectbox("Metric", ["F1-Score", "Recall", "Precision", "PR-AUC", "ROC-AUC", "Accuracy"], index=0)
    st.markdown(h(f'<p class="cg-lead"><b>{choice}.</b> {METRIC_NOTES[choice]}</p>'), unsafe_allow_html=True)

    rows = sorted(((m["name"], m["family"], m["metrics"][choice]) for m in RESULTS["models"]), key=lambda t: t[2])
    fig = go.Figure(go.Bar(
        x=[t[2] for t in rows], y=[t[0] for t in rows], orientation="h",
        marker=dict(color=[FAMILY_COLORS[t[1]] for t in rows]),
        text=[f"{t[2]:.4f}" for t in rows], textposition="outside", textfont=dict(color=WHITE),
    ))
    fig.update_layout(title=f"{choice} by model")
    fig.update_xaxes(range=[0, 1.12], title=choice)
    plot(fig, height=470)
    st.caption("Olive is supervised, gold is semi-supervised, steel is unsupervised.")

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        fig = go.Figure()
        for fam in FAMILY_COLORS:
            pts = [m for m in RESULTS["models"] if m["family"] == fam]
            fig.add_trace(go.Scatter(
                x=[m["metrics"]["Recall"] for m in pts], y=[m["metrics"]["Precision"] for m in pts],
                mode="markers+text", name=fam, text=[m["name"] for m in pts], textposition="top center",
                textfont=dict(size=10, color=MUTED), marker=dict(size=13, color=FAMILY_COLORS[fam],
                                                                line=dict(width=1, color=NAVY)),
                hovertemplate="%{text}<br>Recall %{x:.4f}<br>Precision %{y:.4f}<extra></extra>",
            ))
        fig.update_layout(title="Precision against recall")
        fig.update_xaxes(title="Recall (illicit)", range=[0, 1])
        fig.update_yaxes(title="Precision (illicit)", range=[0, 1.05])
        plot(fig, height=470, legend=True)
    with right:
        z = [[CM["TN"], CM["FP"]], [CM["FN"], CM["TP"]]]
        fig = go.Figure(go.Heatmap(
            z=z, x=["Predicted licit", "Predicted illicit"], y=["Actual licit", "Actual illicit"],
            text=[[f"{v:,}" for v in row] for row in z], texttemplate="%{text}",
            textfont=dict(size=17, color=WHITE), colorscale=[[0, "#14233C"], [1, "#B86E12"]], showscale=False,
        ))
        fig.update_layout(title="Soft Voting: confusion matrix")
        fig.update_yaxes(autorange="reversed")
        plot(fig, height=390)
        st.caption("Read from the notebook's confusion-matrix image. Matches the printed accuracy, precision and recall.")

    sec("Table", "All metrics side by side")
    best = max(m["metrics"]["F1-Score"] for m in RESULTS["models"])
    head = "".join(f"<th>{c}</th>" for c in ["Model", "Family", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
    body = ""
    for m in sorted(RESULTS["models"], key=lambda x: x["metrics"]["F1-Score"], reverse=True):
        mt = m["metrics"]
        cls = ' class="best"' if mt["F1-Score"] == best else ""
        body += (f'<tr><td class="l">{m["name"]}</td><td class="l">{m["family"]}</td>'
                 f'<td>{pct(mt["Accuracy"])}</td><td>{pct(mt["Precision"])}</td><td>{pct(mt["Recall"])}</td>'
                 f'<td{cls}>{pct(mt["F1-Score"])}</td><td>{mt["ROC-AUC"]:.4f}</td><td>{mt["PR-AUC"]:.4f}</td></tr>')
    st.markdown(h(f'<div class="cg-tw"><table class="cg-table"><thead><tr>{head}</tr></thead>'
                  f'<tbody>{body}</tbody></table></div>'), unsafe_allow_html=True)


def page_evaluation():
    sec("Protocol", "How the evaluation was kept clean",
        "The test period was never used for training, scaling, SMOTE or threshold choice.")
    grid([
        card(f"Training {pill('sup', 'Steps 1 to 34')}",
             f"{fmt(SPLIT['train']['total'])} labeled transactions, {fmt(SPLIT['train']['illicit'])} illicit. "
             "Used to fit the scaler, SMOTE and every model.", 0),
        card(f"Validation {pill('semi', 'None')}",
             "No validation set. Settings were fixed in advance and not tuned on held-out data.", 1),
        card(f"Test {pill('sup', 'Steps 35 to 49')}",
             f"{fmt(SPLIT['test']['total'])} labeled transactions, {fmt(SPLIT['test']['illicit'])} illicit. Scored once at threshold 0.5.", 2),
    ], cls="c3")

    sec("Findings", "Key numbers")
    grid([
        metric("Soft Voting recall", pct(SV["Recall"]), f"Caught {fmt(CM['TP'])} of {fmt(SPLIT['test']['illicit'])} illicit test transactions", accent=True, i=0),
        metric("Soft Voting precision", pct(SV["Precision"]), f"{fmt(CM['TP'])} of {fmt(CM['TP'] + CM['FP'])} flags were illicit", i=1),
        metric("F1 without feature 104", pct(ABL["F1-Score"]), f"Full ensemble: {pct(SV['F1-Score'])}", i=2),
        metric("ROC-AUC without feature 104", pct(ABL["ROC-AUC"]), f"Full ensemble: {pct(SV['ROC-AUC'])}", i=3),
    ])
    callout("Recall stays limited on later steps",
            "Precision is high on steps 35 to 42, then falls to near zero for most steps from 43 onward. "
            "The notebook does not print per-step illicit counts, so the cause cannot be checked from these results.", "caution")
    callout("Feature 104 is not the cause",
            f"Removing feature 104 gives F1 {pct(ABL['F1-Score'])} and ROC-AUC {pct(ABL['ROC-AUC'])}, close to the full "
            "ensemble. The feature shows a trend over time, but removing it does not fix recall.", "note")

    sec("Limitations", "What to keep in mind")
    st.markdown(h("""<div class="cg-card"><div class="cb"><ul>
      <li>One fixed test period of 15 steps. Results on other periods are unknown.</li>
      <li>Threshold 0.5 was never tuned. Another cutoff changes precision and recall.</li>
      <li>No validation set, so settings were fixed rather than searched.</li>
      <li>Pseudo-labels come from the ensemble's own confidence, so errors can reinforce themselves.</li>
      <li>Results come from one random seed (42). Variance across seeds is not reported.</li>
    </ul></div></div>"""), unsafe_allow_html=True)


# ---------- navigation -------------------------------------------------------
def main():
    if not st.session_state.get("entered", False):
        render_intro()
        return
    st.markdown(CSS, unsafe_allow_html=True)
    nav = st.navigation([
        st.Page(page_overview, title="Overview", url_path="overview", default=True),
        st.Page(page_dataset, title="Dataset Explorer", url_path="dataset"),
        st.Page(page_methodology, title="Methodology", url_path="methodology"),
        st.Page(page_approaches, title="Learning Approaches", url_path="approaches"),
        st.Page(page_comparison, title="Model Comparison", url_path="comparison"),
        st.Page(page_evaluation, title="Evaluation and Findings", url_path="evaluation"),
    ], position="sidebar")
    with st.sidebar:
        st.markdown(h(f'<div class="cg-brand">{btc_mark(34)}<div><b>CryptoGuard</b>'
                      '<small>Transaction risk</small></div></div>'), unsafe_allow_html=True)
        if st.button("Back to intro", width="stretch"):
            st.session_state["entered"] = False
            st.rerun()
    nav.run()


main()