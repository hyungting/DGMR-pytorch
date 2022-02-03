import torch

ckpt = torch.load("model_best.pth.tar")
g_weight = ckpt["model"]["G"]

temp = "xxx"
for k in g_weight.keys():
    if temp in k:
        continue

    if len(g_weight[k].shape) == 1:
        text = f"{k}: {g_weight[k][0]}"
    elif len(g_weight[k].shape) == 2:
        text = f"{k}: {g_weight[k][0][0]}"
    elif len(g_weight[k].shape) == 3:
        text = f"{k}: {g_weight[k][0][0][0]}"
    elif len(g_weight[k].shape) == 4:
        text = f"{k}: {g_weight[k][0][0][0][0]}"
    else:
        text = f"{k}: {g_weight[k]}"
    print(text)
    #temp = k[0:17]
#with open("weight.txt", "w") as f:
#    for k in g_weight.keys():
#        text = f"{k}: {g_weight[k].shape}\n"
#        f.write(text)
#print(g_weight["gru_layer3.5.out_gate.weight_u"])
#print(g_weight["g_block3.5.conv_out.0.weight"])

