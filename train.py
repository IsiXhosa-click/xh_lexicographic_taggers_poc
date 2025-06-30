import os
import pickle
import tempfile
from pathlib import Path
import torch
import tqdm
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
import time
from sklearn.metrics import f1_score, classification_report
from ray.tune import get_checkpoint, Checkpoint
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.util.client import ray
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search import ConcurrencyLimiter
from dataset import AnnotatedCorpusDataset, SEQ_PAD_IX, UNK_IX
from lstm import BiLSTMTagger

FINAL_CONFIG = {
    "embed_dim": 285,
    "hidden_dim": 1875,
    "dropout": 0.2084209571463339,
    "char_dropout": 0.02,
    "lstm_layers": 1,
    "batch_size": 4,
    "lr": 8.104087435423319e-05,
    "weight_decay": 0.0002998446875344429,
    "gradient_clip": 6957.25375369464,
}

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train, dev = AnnotatedCorpusDataset.load_data(device)

    model = BiLSTMTagger(FINAL_CONFIG, train)
    train_model(10, train, dev, FINAL_CONFIG, model, name_of_config(FINAL_CONFIG))

def tune_main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train, dev = AnnotatedCorpusDataset.load_data(device)

    config = {
        "embed_dim": tune.uniform(16, 512),
        "hidden_dim": tune.uniform(512, 2048),
        "dropout": tune.uniform(0, 0.4),
        "char_dropout": tune.uniform(0.01, 0.2),
        "lr": tune.uniform(1e-6, 1e-3),
        "lstm_layers": tune.uniform(1, 4),
        "batch_size": 4,
        "weight_decay": tune.uniform(1e-8, 1e-3),
        "gradient_clip": tune.uniform(0.5, 100),
    }

    tune_model(config, 20, train, dev)


def _collate_by_padding(batch):
    """Collate sequences by padding them - used to adapt `AnnotatedCorpusDataset` to torch's `DataLoader`"""

    words = pad_sequence([words for words, tag in batch], batch_first=True, padding_value=SEQ_PAD_IX)
    expected_tags = torch.stack([tag for words, tag in batch])
    return words, expected_tags


def name_of_config(config):
    return "model-bilstm_lang-xh_" + "_".join(f"{k}-{v}" for k, v in sorted(list(config.items())))


def train_model(epochs, trainset, dev, config, model, name, use_ray=False):
    train_loader = DataLoader(
        trainset,
        batch_sampler=BatchSampler(RandomSampler(trainset), config["batch_size"], False),
        collate_fn=_collate_by_padding,
    )

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # When hyperparameter tuning with ray, we do some additional steps such as saving checkpoints
    start_epoch = 0

    if use_ray:
        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "rb") as fp:
                    checkpoint_state = pickle.load(fp)
                start_epoch = checkpoint_state["epoch"]
                model.load_state_dict(checkpoint_state["model_state_dict"])
                optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])


    best_macro = 0.0
    best_macro_epoch = 0
    micro_at_best_macro = 0.0

    # Train the model for as many epochs as the config states
    batches = len(train_loader)
    for epoch in range(start_epoch, epochs):
        # Set model to training mode (affects layers such as BatchNorm)
        model.train()

        train_loss = 0
        start = time.time()
        for word, tag in tqdm.tqdm(iter(train_loader)):
            # Clear gradients
            model.zero_grad()

            # Calculate loss and backprop
            loss = model.loss(word, tag)
            train_loss += loss.item() * word.size(dim=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()

        # Print some output about how the model is doing in this epoch
        elapsed = time.time() - start
        print(f"Eval (elapsed = {elapsed:.2f}s)")
        valid_loss, valid_batches, _report, f1_micro, f1_macro, f1_weighted = analyse_model(model, config, dev)
        elapsed = time.time() - start
        print(f"Epoch {epoch} done in {elapsed:.2f}s. "
              f"Train loss: {train_loss / batches:.3f}. "
              f"Valid loss: {valid_loss / valid_batches:.3f}. "
              f"Micro F1: {f1_micro:.3f}. Macro f1: {f1_macro:.3f}")

        # Save the model if it has done better than the previous best epochs
        if not use_ray and f1_macro > best_macro:
            best_macro = f1_macro
            best_macro_epoch = epoch
            micro_at_best_macro = f1_micro

            out_dir = "out_models/"
            out_dir = os.path.join(out_dir, "checkpoints", name)
            print("saving to", out_dir)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, name) + ".pt", "wb") as f:
                torch.save(model.state_dict(), f)

        # Checkpoint the model if hyperparameter tuning with ray
        if use_ray:
            checkpoint_data = {
                "epoch": epoch,
                "best_epoch": best_macro_epoch,
                "best_macro": best_macro,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                tune.report(
                    {"loss": valid_loss / valid_batches, "f1_macro": f1_macro, "f1_micro": f1_micro},
                    checkpoint=checkpoint,
                )

    print(f"Best Macro f1: {best_macro} in epoch {best_macro_epoch} (micro here was {micro_at_best_macro})")

    return { "micro_at_best_macro": micro_at_best_macro, "best_macro": best_macro, "model": model }

def tune_model(config, epochs, trainset: AnnotatedCorpusDataset, valid: AnnotatedCorpusDataset, cpus=1, hrs=4):
    """Tune the given model with Ray"""

    ray.init(num_cpus=cpus)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    algo = BayesOptSearch(metric="f1_macro", mode="max")
    algo = ConcurrencyLimiter(algo, max_concurrent=1)

    # We just use the basic ASHA schedular
    scheduler = ASHAScheduler(
        metric="f1_macro",
        mode="max",
        max_t=epochs,
        grace_period=10,
        reduction_factor=2,
    )

    # Move the trainset & validset into shared memory (they are very large)
    trainset, valid = ray.put(trainset), ray.put(valid)

    # Do the hyperparameter tuning
    result = tune.run(
        lambda conf: train_model(epochs, ray.get(trainset), ray.get(valid), conf, model_for_config(ray.get(trainset), conf), name_of_config(config), use_ray=True),
        name="train-1",
        resume=True,
        resources_per_trial={"gpu": 1.0 / cpus} if torch.cuda.is_available() else None,
        config=config,
        num_samples=1000,
        time_budget_s=hrs * 60 * 60,
        search_alg=algo,
        scheduler=scheduler,
        storage_path="/mnt/c/Users/caelm/PycharmProjects/xh_lexicographic_taggers/checkpoints/",
    )

    # Print out the epoch with best macro & micro F1 scores
    for metric in ["f1_macro", "f1_micro"]:
        best_trial = result.get_best_trial(metric, "max", "all")
        print(f"Best trial by {metric}:")
        print(f" config: {best_trial.config}")
        print(f" val loss: {best_trial.last_result['loss']}")
        print(f" macro f1 {best_trial.last_result['f1_macro']}")
        print(f" micro {best_trial.last_result['f1_micro']}")

        best_model = model_for_config(ray.get(trainset), best_trial.config)
        best_model.to(device)

        best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric=metric, mode="max")
        with best_checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                best_checkpoint_data = pickle.load(fp)

            best_model.load_state_dict(best_checkpoint_data["model_state_dict"])
            _, _, report, f1_micro, f1_macro, f1_weighted = analyse_model(best_model, best_trial.config, ray.get(valid))
            print(f" Micro F1: {f1_micro}. Macro f1: {f1_macro}. Weighted F1: {f1_weighted}")
            print(
                f" Best macro f1: {best_checkpoint_data['best_macro']} at epoch "
                f"{best_checkpoint_data['best_epoch']}")
            print(report)


def model_for_config(trainset, config):
    """Create a model with the given config"""

    for param in ("embed_dim", "hidden_dim", "lstm_layers"):
        config[param] = int(config[param])

    model = BiLSTMTagger(config, trainset)
    return model


def analyse_model(model, config, valid: AnnotatedCorpusDataset):
    """Analyse the given model on the validation dataset"""

    valid_loader = DataLoader(
        valid,
        batch_sampler=BatchSampler(RandomSampler(valid), config["batch_size"], False),
        collate_fn=_collate_by_padding,
    )

    with torch.no_grad():
        # Set model to evaluation mode (affects layers such as BatchNorm)
        model.eval()

        all_predicted = []
        all_expected = []
        valid_loss = 0.0

        for words, expected_tags in valid_loader:
            loss = model.loss(words, expected_tags)
            valid_loss += loss.item() * words.size(dim=0)

            # Get the model's predicted tags
            predicted_tags = model.forward_tags_only(words)

            # For the F1 score, we essentially concatenate all the predicted tags into a list, and do the same with
            # the gold standard tags. Then we call the f1_score function from sklearn.

            # This loop splits by batch
            for expected, pred in zip(torch.unbind(expected_tags), torch.unbind(predicted_tags)):
                all_predicted.append(valid.ix_to_tag[pred.item()])
                all_expected.append(valid.ix_to_tag[expected.item()])

        # Calculate scores & return
        f1_micro = f1_score(all_expected, all_predicted, average="micro")
        f1_macro = f1_score(all_expected, all_predicted, average="macro")
        f1_weighted = f1_score(all_expected, all_predicted, average="weighted")
        report = classification_report(all_expected, all_predicted, zero_division=0.0)

        return valid_loss, len(valid_loader), report, f1_micro, f1_macro, f1_weighted


def write_out_testset():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train, dev = AnnotatedCorpusDataset.load_data(device)

    model = model_for_config(train, FINAL_CONFIG)
    name = "model-bilstm_lang-xh_batch_size-4_char_dropout-0.02_dropout-0.2084209571463339_embed_dim-285_gradient_clip-6957.25375369464_hidden_dim-1875_lr-8.104087435423319e-05_lstm_layers-1_weight_decay-0.0002998446875344429"
    model.load_state_dict(torch.load("out_models/checkpoints/" + name + "/" + name + ".pt"))

    with open("data/test.txt") as f:
        testset = f.read().split("\n")

    testset_encoded = []
    for word in testset:
        tensor = torch.tensor(
            [
                (train.char_to_ix[letter] if letter in train.char_to_ix else UNK_IX) for letter in word
            ],
            device=device
        )
        tensor = tensor.reshape(1, len(word))
        testset_encoded.append(tensor)


    with torch.no_grad():
        # Set model to evaluation mode (affects layers such as BatchNorm)
        model.eval()

        with open("out.csv", "w") as f:
            for word, word_raw in zip(testset_encoded, testset):
                predicted_tags = model.forward_tags_only(word)
                # This loop splits by batch
                for pred in torch.unbind(predicted_tags):
                    f.write(word_raw + ";" + train.ix_to_tag[pred.item()] + "\n")

if __name__ == "__main__":
    # main()
    write_out_testset()
