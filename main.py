import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA = "data/enpscyt_2024.csv"
ITEMS = {
    "P37_2": "Amenaza el desarrollo de la sociedad",
    "P37_5": "Generará más daños que beneficios",
    "P37_9": "Aumentará el dominio de las máquinas sobre las personas",
    "P37_12": "Creará más problemas que soluciones",
    "P37_14": "Creará divisiones sociales",
    "P37_16": "Controlará nuestra vida privada",
}
NUM = [k + "_num" for k in ITEMS]
FEATS = NUM + ["indice_riesgo"]
LABELS = list(ITEMS.values()) + ["Índice de riesgo"]


def parse(v):
    if pd.isna(v) or str(v).strip() in ["No sabe", "No contesta"]:
        return np.nan
    s = str(v).strip()
    if "10" in s:
        return 10.0
    if s.startswith("1"):
        return 1.0
    try:
        return float(s[0])
    except Exception:
        return np.nan


df = pd.read_csv(DATA)
d = df[df["P36"].isin(["Positivos", "Muy positivos", "Negativos", "Muy negativos"])].copy()
d["target"] = d["P36"].map({"Muy negativos": 0, "Negativos": 0, "Positivos": 1, "Muy positivos": 1})
for k in ITEMS:
    d[k + "_num"] = d[k].apply(parse)
d[NUM] = d[NUM].fillna(d[NUM].median())
d["indice_riesgo"] = d[NUM].mean(axis=1)
X, y = d[FEATS], d["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

MODELS = {
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42),
}
RES = {}
for n, m in MODELS.items():
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    rep = classification_report(yte, pred, target_names=["Rechazo", "Aceptación"], output_dict=True)
    RES[n] = dict(f1=round(f1_score(yte, pred, average="macro"), 4), cm=confusion_matrix(yte, pred).tolist(),
                  rep={k: {a: round(b, 3) for a, b in rep[k].items()} for k in ["Rechazo", "Aceptación"]})
BEST = max(RES, key=lambda k: RES[k]["f1"])


def contribs(rows):
    return MODELS["XGBoost"].get_booster().predict(xgb.DMatrix(rows, feature_names=FEATS), pred_contribs=True)


GLOBAL = np.abs(contribs(Xte)[:, :-1]).mean(axis=0).round(4).tolist()
CORR = d[FEATS + ["target"]].corr().round(2)
MEANS = d.groupby("target")[FEATS].mean().round(2)
RAW = {str(k): int(v) for k, v in df["P36"].value_counts().items()}
IDX_SORTED = np.sort(d["indice_riesgo"].values)
BASE = float(y.mean())

LAB = dict(zip(FEATS, LABELS))
N0 = len(df)
g = lambda k: RAW.get(k, 0)
pos, neg_, ext = g("Positivos") + g("Muy positivos"), g("Negativos") + g("Muy negativos"), g("Muy positivos") + g("Muy negativos")
nsnc = g("No sabe") + g("No contesta")
ct = CORR.loc[NUM, "target"]
ir = CORR.loc[NUM, "indice_riesgo"]
ii, jj = np.triu_indices(len(NUM), 1)
inter = CORR.loc[NUM, NUM].values[ii, jj]
kmax = int(np.argmax(inter))
neg_all, weak = bool((ct < 0).all()), bool(np.abs(ct).max() < .3)
diffs = MEANS.loc[0, NUM] - MEANS.loc[1, NUM]
g1, g2 = diffs.abs().idxmax(), diffs.abs().idxmin()
n_hi = int((diffs > 0).sum())
lead = "En las seis afirmaciones" if n_hi == 6 else f"En {n_hi} de 6 afirmaciones"
f1s = {n: r["f1"] for n, r in RES.items()}
worst = min(f1s, key=f1s.get)
gap = f1s[BEST] - f1s[worst]
rp = lambda n, c, k: RES[n]["rep"][c][k]
fa = [rp(n, "Aceptación", "f1-score") for n in RES]
fr = [rp(n, "Rechazo", "f1-score") for n in RES]


def cm_text(n):
    (tn, fp), (fn, tp) = RES[n]["cm"]
    return (f"De {tn+fp} personas que rechazan, el modelo detecta {tn} ({tn/(tn+fp)*100:.0f}%) y confunde {fp} con aceptación. "
            f"De {fn+tp} que aceptan, acierta {tp} ({tp/(fn+tp)*100:.0f}%) y falla {fn}. "
            f"Cuando predice rechazo, acierta el {rp(n, 'Rechazo', 'precision')*100:.0f}% de las veces.")


sv_ = np.array(GLOBAL)
o = np.argsort(-sv_)
sh = sv_ / sv_.sum()
TEXTS = dict(
    eda1=(f"Cada barra es una categoría de P36. Aceptación (azul) = Positivos ({g('Positivos'):,}) + Muy positivos ({g('Muy positivos'):,}) "
          f"= {pos:,} personas ({pos/N0*100:.0f}% de {N0:,}). Rechazo (naranja) = Negativos ({g('Negativos'):,}) + Muy negativos "
          f"({g('Muy negativos'):,}) = {neg_:,} ({neg_/N0*100:.0f}%). Excluidas (gris) = No sabe ({g('No sabe'):,}) + No contesta "
          f"({g('No contesta'):,}) = {nsnc:,} ({nsnc/N0*100:.0f}%). Entre las {len(d):,} respuestas útiles, {BASE*100:.0f}% es aceptación y "
          f"{100-BASE*100:.0f}% rechazo: hay desbalance, lo que justifica usar F1-macro."),
    eda2=(f"Las seis afirmaciones se correlacionan entre sí (r promedio = {inter.mean():.2f}; la mayor, {inter.max():.2f}, entre "
          f"«{LAB[NUM[ii[kmax]]]}» y «{LAB[NUM[jj[kmax]]]}») y el índice de riesgo se correlaciona {ir.min():.2f}–{ir.max():.2f} con cada ítem: "
          f"miden una actitud común, lo que respalda resumirlas en un índice. Con la aceptación, las correlaciones "
          f"{'son negativas' if neg_all else 'tienen signos mixtos'} y {'débiles' if weak else 'moderadas'} (de {ct.min():.2f} a {ct.max():.2f})"
          + (": más riesgo percibido se asocia con menos aceptación, pero ninguna pregunta por sí sola la predice." if neg_all and weak else ".")),
    eda3=(f"{lead} quienes rechazan la IA puntúan {'más alto' if n_hi == 6 else 'distinto'} que quienes la aceptan; la diferencia promedio es de "
          f"{diffs.abs().mean():.1f} puntos (escala 1–10). La mayor brecha está en «{LAB[g1]}» ({MEANS.loc[0, g1]:.1f} vs {MEANS.loc[1, g1]:.1f}) "
          f"y la menor en «{LAB[g2]}» ({MEANS.loc[0, g2]:.1f} vs {MEANS.loc[1, g2]:.1f}). Con diferencias de ese tamaño en una escala de 10 puntos, "
          f"{'los grupos se solapan y la separación es parcial' if diffs.abs().mean() < 2 else 'los grupos se separan con claridad'}."),
    table=(f"{BEST} lidera con F1-macro {f1s[BEST]}; el menor es {worst} ({f1s[worst]}). La diferencia ({gap:.3f}) es "
           f"{'pequeña' if gap < .06 else 'notable'}: con una sola partición 75/25 no puede afirmarse de forma concluyente que un modelo sea superior. "
           f"En los tres, el F1 de Aceptación ({min(fa):.2f}–{max(fa):.2f}) {'supera' if min(fa) > max(fr) else 'difiere del'} "
           f"de Rechazo ({min(fr):.2f}–{max(fr):.2f}), reflejo del desbalance de clases."),
    cm={n: cm_text(n) for n in RES},
    shap=(f"«{LABELS[o[0]]}» es la variable de mayor impacto ({sh[o[0]]*100:.0f}% del total), seguida de «{LABELS[o[1]]}» ({sh[o[1]]*100:.0f}%). "
          f"Como el índice de riesgo es el promedio de los ítems y estos están correlacionados, el crédito se reparte entre variables relacionadas; "
          f"la importancia individual debe leerse con cautela. Describe cómo usa el modelo la información, no demuestra causalidad."),
)

app = FastAPI(title="Umbral")


@app.get("/api/summary")
def summary():
    return dict(n=len(d), n_raw=len(df), acept=int((y == 1).sum()), rechazo=int((y == 0).sum()), best=BEST,
                models=RES, items=list(ITEMS.values()), labels=LABELS, shap=GLOBAL, corr=CORR.values.tolist(),
                means=MEANS.values.tolist(), raw=RAW, texts=TEXTS)


class Req(BaseModel):
    values: list[float]
    model: str = BEST


@app.post("/api/predict")
def predict(r: Req):
    row = pd.DataFrame([r.values], columns=NUM)
    row["indice_riesgo"] = row[NUM].mean(axis=1)
    probs = {n: float(m.predict_proba(row[FEATS])[0, 1]) for n, m in MODELS.items()}
    p = probs[r.model]
    c = contribs(row[FEATS])[0][:-1]
    idx = float(row["indice_riesgo"][0])
    pct = float((IDX_SORTED <= idx).mean() * 100)
    agree = sum((q >= .5) == (p >= .5) for q in probs.values())
    margin = abs(p - .5) * 100
    i = int(np.argmax(np.abs(c)))
    lectura = (
        f"{r.model} estima {p*100:.0f}% de probabilidad de aceptación, por lo que clasifica el perfil como "
        f"{'aceptación' if p >= .5 else 'rechazo'}; {agree} de 3 modelos coinciden. "
        f"El índice de riesgo ({idx:.2f}) supera al {pct:.0f}% de los encuestados. "
        f"La tasa base de aceptación en la muestra es {BASE*100:.0f}%. "
        f"La variable de mayor influencia fue «{LABELS[i]}», que empuja hacia "
        f"{'la aceptación' if c[i] > 0 else 'el rechazo'}. "
        + ("La probabilidad está cerca del umbral de 0,50, así que la clasificación es poco concluyente."
           if margin < 10 else "El margen frente al umbral de 0,50 es amplio.")
    )
    return dict(prob=p, probs=probs, indice=idx, pct=pct, agree=agree, margin=margin,
                contribs=c.round(3).tolist(), lectura=lectura)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")