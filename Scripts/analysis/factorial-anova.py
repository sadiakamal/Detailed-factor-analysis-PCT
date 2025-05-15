import pandas as pd
from typing import List, Optional
import re
import statsmodels.api as sm
from statsmodels.formula.api import ols

ALL_DATA_GSHEET =  "https://docs.google.com/spreadsheets/d/1rVr6Ly0UShe51Rj6NzE-XSaH6KZtmC9AVrQalmaHCZA/edit?gid=331280590#gid=331280590"

def read_gsheet(url: Optional[str]=None, sheet_id: Optional[str]=None, gid: Optional[str]=None):
    if url is not None:
        match = re.search(r"spreadsheets/d/([^/]+)/.*?[?&]gid=(\d+)", url)
        if match:
            sheet_id = match.group(1)
            gid = match.group(2)
        else:
            print("can't parse url to get sheet id and gid")
    else:
        assert sheet_id is not None and gid is not None, "Sheet id an gid must be not None when url is not None"
    _url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&id={sheet_id}&gid={gid}"
    return pd.read_csv(_url)


df = read_gsheet(ALL_DATA_GSHEET)
df.rename(columns={"Social Libertarian/Authoritarian": "output_x"}, inplace=True)
df.rename(columns={"Economic Left/Right": "output_y"}, inplace=True)
assert set(list(df)) == {"model", "fine_tune_dataset", "n_beams", "tmp", "top_k", "prompt", "output_x", "output_y"}
for k in [x for x in list(df) if not x.startswith("output")]:
    print(k, set(df[k]))


def factorial_anova(_filter_df: pd.DataFrame,
                    factors:  List[str]=["fine_tune_dataset", "n_beams", "tmp", "top_k", "prompt"],
                    predicted_var: str = "output_x"):
    """
    do a factorial anova on the provided dataframe.
    :param _filter_df: output from a model
    :param factors: list of factors (column names in the df)
    :param predicted_var: predicted variable (column name in the df)
    :return:
    """
    formula = f"{predicted_var} ~" + ' + '.join([f'C({col})' for col in factors])
    model = ols(formula, data=_filter_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
    return anova_table

for predicted_var_k, predicted_var_v in {"Social Libertarian/Authoritarian": "output_x",
                                         "Economic Left/Right": "output_y"}.items():
    for model in ["Llama3", "Mistral", "Gemma"]:
        model_df = df[df['model'] == model]
        print(f"Predicted_var: {predicted_var_k}, Model: {model}")
        print("-"*30)
        print(factorial_anova(_filter_df=model_df, predicted_var=predicted_var_v))
        print("="*30)
