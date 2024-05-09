import textattack
import os
import transformers
import sys
from textattack.datasets import HuggingFaceDataset
from textattack import Attacker
from textattack.attack_recipes import BAEGarg2019
from textattack.metrics.attack_metrics import (
    AttackQueries,
    AttackSuccessRate,
    WordsPerturbed,
)
import json


def load_model_and_tokenizer(hugging_face_path):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(hugging_face_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(hugging_face_path)
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    return model_wrapper


def save_to_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def save_perturbed_metrics(run_results, drive_path=None):
    perturbed_examples = []
    attack_success_stats = AttackSuccessRate().calculate(run_results)
    words_perturbed_stats = WordsPerturbed().calculate(run_results)
    words_perturbed_stats.pop('num_words_changed_until_success', None)
    attack_query_stats = AttackQueries().calculate(run_results)
    perturbed_dict = {"attack_success_stats": attack_success_stats, "words_perturbed_stats": words_perturbed_stats,
                      "attack_query_stats": attack_query_stats}
    for result in run_results:
        if isinstance(result, textattack.attack_results.SuccessfulAttackResult):
            perturbed_result = {"original_text": result.original_text(), "perturbed_text": result.perturbed_text(),
                                "gfr": result.goal_function_result_str()}
            perturbed_examples.append(perturbed_result)
    perturbed_dict["perturbed_examples"] = perturbed_examples

    try:
        save_to_file(perturbed_dict, f'perturbed_text/perturbed_{i}.json')
        if drive_path is not None:
            path = os.path.join(drive_path, f'perturbed_{i}.json')
            save_to_file(perturbed_dict, path)
    except:
        print(f"Error while saving perturbed examples on drive at iteration {i}")


if __name__ == "__main__":
    start_index = int(sys.argv[1])
    bert_imdb = load_model_and_tokenizer("textattack/bert-base-uncased-imdb")
    imdb_dataset_train = HuggingFaceDataset("imdb", split="train", shuffle=True)
    attack = BAEGarg2019.build(bert_imdb)

    batch_size = 10
    n_iter = 300
    print('Starting from index {}'.format(start_index))
    for i in range(start_index, n_iter):
        attack_args = textattack.AttackArgs(num_examples=batch_size, num_examples_offset=i * batch_size, parallel=True)
        attacker = Attacker(attack, imdb_dataset_train, attack_args)
        adv_exp_bae_bert_imdb = attacker.attack_dataset()
        save_perturbed_metrics(adv_exp_bae_bert_imdb,
                               '/run/user/1000/gvfs/google-drive:host=studenti.unipd.it,user=damiano.bertoldo/GVfsSharedWithMe/1OKUVPYQzqhGpqt_wXW2NRFoYukW7q2V7/18czngAsdABbSvUzTbIo8SIRi_G-f8Z7G/')
