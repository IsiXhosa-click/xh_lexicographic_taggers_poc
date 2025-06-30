from sklearn.metrics import f1_score, classification_report

with open("all_res.csv") as data:
    all_lines = data.read().split("\n")
    head, body = all_lines[0], all_lines[1:]
    model_answers = dict()

    for i, model in enumerate(head.split(";")[1:]):
        i = i + 1  # account for skipping the 'word' column

        model_answers[model] = []

        for line in body:
            if not line:
                continue

            ans = line.split(";")[i]

            replacements = {f"N{k}" : f"N0{k}" for k in [*range(1, 10), "1a", "2a"]}

            for from_, to in replacements.items():
                if ans == from_:
                    ans = to
                    break

            model_answers[model].append(ans)

def no_class(preds):
    return ["".join(filter(lambda c: not c.isnumeric(), ans)) for ans in preds]


gold = model_answers["Human with corrections (gold)"]
gold_no_class = no_class(model_answers["Human with corrections (gold)"])

with open("out_analysis.csv", "w") as f:
    model_answers = sorted(model_answers.items(), key=lambda kv: kv[0])

    f.write("Model; Accuracy (including noun classes); Accuracy (not including noun classes);\n")

    for model, answers in model_answers:
        if model == "Human with corrections (gold)":
            continue

        answers_no_class = no_class(answers)

        f1_macro = f1_score(y_true=gold, y_pred=answers, average="macro")
        acc = f1_score(y_true=gold, y_pred=answers, average="micro")
        f1_macro_no_class = f1_score(y_true=gold_no_class, y_pred=answers_no_class, average="macro")
        acc_no_class = f1_score(y_true=gold_no_class, y_pred=answers_no_class, average="micro")

        print(f"{model} - macro (non class / class): {f1_macro_no_class:.2f} / {f1_macro:.2f}, acc non class / class): {acc_no_class:.2f} / {acc:.2f}")


        mat = classification_report(y_true=gold, y_pred=answers, zero_division=0)
        print(mat)

        mat = classification_report(y_true=gold_no_class, y_pred=answers_no_class, zero_division=0)
        print(mat)

        f.write(f"{model}; {acc*100:.4f}%; {acc_no_class*100:.4f}%;\n")
