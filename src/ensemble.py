import pandas as pd


def group_result_mean(df):
    group = df.groupby(["id", "answer"]).logit_score.mean().reset_index()
    result = group.griloc[group.groupby("id").logit_score.idxmax()]
    return result


def group_result_sum(df):
    group = df.groupby(["id", "answer"]).logit_score.sum().reset_index()
    result = group.iloc[group.groupby("id").logit_score.idxmax()]
    result.loc[:, "answer"] = result.answer.apply(lambda x: x.strip())
    return result


if __name__ == "__main__":
    csv_list = []
    df = group_result_sum(pd.concat((pd.read_csv(path)) for path in csv_list))
    df[["id", "answer"]].to_csv("result_lamma_conv_original_drop.csv", index=False)
