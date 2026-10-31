"""도전과제: geom_bar position='fill'(비율) vs 'stack'(건수, 기본값) 비교."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from plotnine import ggplot, geom_bar, aes, theme_bw, labs, coord_flip, theme, element_text
OUT = "./images"
df = pd.read_csv("chapter-4/data/income_data.csv", index_col=False)
for col in ["Education", "Race"]:
    for pos in ["fill", "stack"]:
        p = (ggplot(df) + geom_bar(aes(x=col, fill="Target"), position=pos) + theme_bw() + coord_flip()
             + labs(title=f"{col} ~ Target  (position='{pos}')", y="proportion" if pos == "fill" else "count", x="")
             + theme(text=element_text(family="DejaVu Sans"), figure_size=(6.2, 4.6)))
        p.save(f"{OUT}/eda-{col.lower()}-{pos}.png", dpi=130, verbose=False)
# 표로도: 범주별 >50K 비율과 건수
t = df.groupby("Race")["Target"].agg(n="size", rate=lambda s: (s.str.strip() == ">50K").mean()).sort_values("n")
print(t.round(3).to_string())
