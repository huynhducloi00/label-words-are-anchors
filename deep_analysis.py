from sklearn.metrics import roc_auc_score
from icl.util_classes.arg_classes import DeepArgs
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

demonstration_shot = 0
model_name = "lmsys/vicuna-13b-v1.5"
task_names = ["obqa"]  # ,'TREC','AGNews','EmoC']


def get_auc_roc_score(
    task_name, seeds=[42], sample_size=1000, model_name="gpt2-xl", demonstration_shot=1
):
    if model_name=='lmsys/vicuna-13b-v1.5':
        num_layer= 40
    elif model_name == "gpt2-xl":
        num_layer = 48
    elif model_name == "gpt-j-6b":
        num_layer = 28
    else:
        raise NotImplementedError
    try:
        args = DeepArgs(
            task_name=task_name,
            seeds=seeds,
            sample_size=sample_size,
            model_name=model_name,
            using_old=False,
            demonstration_shot=demonstration_shot,
        )
        raw_result = args.load_result()
    except:
        args = DeepArgs(
            task_name=task_name,
            seeds=seeds,
            sample_size=sample_size,
            model_name=model_name,
            using_old=True,
            demonstration_shot=demonstration_shot,
        )
        raw_result = args.load_result()
    score_list = []
    for seed in range(len(seeds)):
        y = raw_result[0][seed]
        scores = []
        gold = y.predictions[0].argmax(-1) #using big number token
        num_class = y.predictions[0].shape[-1]
        select_list = []
        for i in range(num_class):
            if (gold == i).sum() > 0:
                select_list.append(i)
        for layer in range(0, num_layer):
            if demonstration_shot == 1:
                # the difference in implementation of demonstration_shot >=2 causes the difference in the order of layer and class,
                # since we have run the experiments, we do not align the order in the implementation and just change the order here
                pred = y.predictions[2].reshape(-1, num_layer, num_class)[:, layer, :]
            else:
                pred = y.predictions[2].reshape(-1, num_class, num_layer)[:, :, layer]
            pred1 = pred[:, select_list].astype(np.float32)
            if len(select_list) == 2:
                pred = pred1[:, 1]
            else:
                pred = pred1 / pred1.sum(-1, keepdims=True)
            scores.append(roc_auc_score(gold, pred, multi_class="ovr" if task_name=='obqa' else 'ovo'))
        score_list.append(scores)
    return score_list


def get_mean_auc_roc_score(
    task_name, seeds=None, sample_size=1000, model_name="gpt2-xl", demonstration_shot=1
):
    if seeds is None:
        seeds = [42]  # ,43,44,45,46]
    score_list = get_auc_roc_score(
        task_name, seeds, sample_size, model_name, demonstration_shot=demonstration_shot
    )
    return np.mean(score_list, 0)


scores_list = [
    get_mean_auc_roc_score(
        task, model_name=model_name, demonstration_shot=demonstration_shot
    )
    for task in ["obqa"]
]  # ,'trec','agnews','emo']]
PARENT='graphing'
pickle.dump(
    scores_list, open(f"{PARENT}/auc_roc_scores_{model_name}_{demonstration_shot}.pkl", "wb")
)


mpl.rcParams["text.usetex"] = False
scores = np.mean(np.array(scores_list), axis=0)

normalized_scores = np.cumsum(scores - 0.5) / np.cumsum(scores - 0.5)[-1]

fig, ax1 = plt.subplots()

ax1.plot(range(1, len(scores) + 1), scores, "b--")
ax1.set_xlabel("Layers")
ax1.set_ylabel(r"$\mathrm{AUCROC_l}$")
ax1.tick_params("y")

ax2 = ax1.twinx()

ax2.plot(range(1, len(scores) + 1), normalized_scores, "r-")
ax2.set_ylabel(r"$R_l$")
ax2.tick_params("y")

ax1.legend([r"$\mathrm{AUCROC_l}$"], loc="upper left")
ax2.legend([r"$R_l$"], loc="upper right")

plt.show()
fig.savefig(
    f"{PARENT}/AUC_ROC_{model_name}_{demonstration_shot}.pdf", dpi=300, bbox_inches="tight"
)


# fig, ax = plt.subplots()
# x = range(len(scores_list[0]))
# for task_name, scores in zip(task_names, scores_list):
#     ax.plot(x, scores, label=task_name)

# ax.legend()

# ax.set_title(f"AUC-ROC Score of {model_name} on Different Tasks")
# ax.set_xlabel("Layer")
# ax.set_ylabel("Score")

# plt.show()

# fig.savefig(
#     f"AUC-ROC_Score_{model_name.replace('/','_')}_{demonstration_shot}.png",
#     dpi=300,
#     bbox_inches="tight",
# )


# fig, ax = plt.subplots()
# x = range(len(scores_list[0]))
# for task_name, scores in zip(task_names, scores_list):
#     n_scores = np.cumsum(scores - 0.5)
#     n_scores = n_scores / n_scores[-1]
#     ax.plot(x, n_scores, label=task_name)

# ax.legend()

# ax.set_title(f"Prediction Ratio of {model_name} on Different Tasks")
# ax.set_xlabel("Layer")
# ax.set_ylabel("Ratio")

# plt.show()

# fig.savefig(
#     f"Prediction_Ratio_of_{model_name.replace('/','_')}_{demonstration_shot}.png"
# )
