import re, os
import pandas as pd


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def replace_text(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s{2,1000}", " ", text)
    return text


def main(folder_path, new_path):
    # train test 불러오기
    test = pd.read_csv(os.path.join(folder_path, "test.csv"))
    train = pd.read_csv(os.path.join(folder_path, "train.csv"))
    all_train = set(train.id.values)

    # 데이터 전처리
    train["context"] = train.context.apply(lambda x: replace_text(x))
    test["context"] = test.context.apply(lambda x: replace_text(x))

    # 정답이 없는 context 제거
    answer_in_context = train.apply(lambda x: x.answer in x.context, axis=1)
    train = train[answer_in_context]

    # context별 아이디 설정
    context_id = {context: idx for idx, context in enumerate(train.context.value_counts().keys())}
    train.loc[:, "context_id"] = train.context.apply(lambda x: context_id[x]).values

    # 제거된 train 저장
    remove_train = all_train - set(train.id.values)
    remove_train = pd.DataFrame(remove_train, columns=["id"])

    # 파일 저장
    createDirectory(new_path)
    train.to_csv(os.path.join(new_path, "train.csv"), index=False)
    test.to_csv(os.path.join(new_path, "test.csv"), index=False)
    remove_train.to_csv(os.path.join(new_path, "clean_remove_train.csv"), index=False)


if __name__ == "__main__":
    folder_path = "LightningHydraTemplate/data/raw_data"
    new_path = "LightningHydraTemplate/data/clean_data"
    main(folder_path, new_path)
