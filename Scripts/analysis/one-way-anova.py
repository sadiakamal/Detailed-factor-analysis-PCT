import pandas as pd
from typing import List, Optional
import re
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import levene
from pingouin import welch_anova

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


# def factorial_anova(_filter_df: pd.DataFrame,
#                     factors:  Optional[List[str]]=["fine_tune_dataset", "n_beams", "tmp", "top_k", "prompt"],
#                     dependent_var: Optional[str] = "output_x",
#                     base_v_finetuned: bool=True,
#                     formula = Optional[str],
#                     independent_var: Optional[str] = None,
#                     ):
#     """
#     do a factorial anova on the provided dataframe.
#     :param _filter_df: output from a model
#     :param factors: list of factors (column names in the df)
#     :param dependent_var: predicted variable (column name in the df)
#     :param base_v_finetuned: instead of saying which dataset it is finetuned on, we will just have 2 values
#     for this factor -- whether it is finetuned or not
#     :param formula: if the formula is given we are externally controlling whether we are doing one way/factorial anovas,
#     if we supply this we don't need the factors or predicted_var
#     :param independent_var: if we want to do levene's test
#     :return:
#     """
#     if base_v_finetuned:
#         mapping = {'Base': 'Base'}
#         _filter_df.loc[:, 'fine_tune_dataset'] = _filter_df['fine_tune_dataset'].map(mapping).fillna('fine_tuned')
#     # levene
#     if independent_var: # if we are doing one-factor anova, check for leven's
#         grouped_data = [group[dependent_var].values for name, group in _filter_df.groupby(independent_var)]
#         stat, pval = levene(*grouped_data, center='mean')
#         print("Levene’s Test:")
#         print(f"Statistic = {stat:.4f}, p-value = {pval:.4g}")
#         if pval < 0.05:
#             assert independent_var is not None and dependent_var is not None
#             welch_results = welch_anova(dv=dependent_var, between=independent_var, data=_filter_df)
#             return welch_results
#     if formula is None:  # if formula is not supplied, we need to build it out
#         formula = f"{dependent_var} ~" + ' + '.join([f'C({col})' for col in factors])
#     _model = ols(formula, data=_filter_df).fit()
#     anova_table = sm.stats.anova_lm(_model, typ=2)
#     anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
#     return anova_table


def one_way_anova(_filter_df: pd.DataFrame, independent_var: Optional[str] = "fine_tune_dataset",
                  dependent_var: Optional[str] = "output_x", collapse: bool=True):
    """
    do one way anova on the provided dataframe.
    :param _filter_df: output from a model
    :param dependent_var: predicted variable (column name in the df)
    :param collapse: instead of saying which dataset it is finetuned on, we will just have 2 values
    for this factor -- whether it is finetuned or not
    :param independent_var: if we want to do levene's test
    :return:
    """
    if collapse:
        mapping = {'Base': 'Base'}
        _filter_df.loc[:, 'fine_tune_dataset'] = _filter_df['fine_tune_dataset'].map(mapping).fillna('fine_tuned')
    # levene
    grouped_data = [group[dependent_var].values for name, group in _filter_df.groupby(independent_var)]
    stat, pval = levene(*grouped_data, center='mean')
    print(f"Levene’s Test: Statistic = {stat:.4f}, p-value = {pval:.4g}")
    if pval < 0.05:
        print("Welch")
        assert independent_var is not None and dependent_var is not None
        welch_results = welch_anova(dv=dependent_var, between=independent_var, data=_filter_df)
        return welch_results
    formula=f"{dependent_var} ~ C({independent_var})"
    _model = ols(formula, data=_filter_df).fit()
    anova_table = sm.stats.anova_lm(_model, typ=2)
    anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
    return anova_table

if __name__ == "__main__":
    df = read_gsheet(ALL_DATA_GSHEET)
    df.rename(columns={"Social Libertarian/Authoritarian": "output_x"}, inplace=True)
    df.rename(columns={"Economic Left/Right": "output_y"}, inplace=True)
    assert set(list(df)) == {"model", "fine_tune_dataset", "n_beams", "tmp", "top_k", "prompt", "output_x", "output_y"}
    for k in [x for x in list(df) if not x.startswith("output")]:
        print(k, set(df[k]))

    for predicted_var_k, predicted_var_v in {"Social Libertarian/Authoritarian": "output_x",
                                             "Economic Left/Right": "output_y"}.items():
        for model in ["Llama3", "Mistral", "Gemma"]:
            model_df = df[df['model'] == model]
            print(f"Predicted_var: {predicted_var_k}, Model: {model}")
            print("-"*30)
            print(one_way_anova(_filter_df=model_df, dependent_var=predicted_var_v, independent_var="tmp",
                                collapse=False))
            print("="*30)
